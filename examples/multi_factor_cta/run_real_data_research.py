"""Reproduce admission, frozen OOS portfolios and broker reconciliation on local real data.

Run after ``python -m examples.multi_factor_cta.real_data_manifest``. Every
candidate is declared before evaluation. Existing champion/zoo files are neither
read nor written. Equity reference prices support admission diagnostics only;
portfolio results contain the fixed commodity universe.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from examples.alpha_mining_agent.forensic_pipeline import IS_END_DT
from examples.alpha_mining_agent.hypothesis_bank import ALL_SIGNAL_SPECS
from examples.alpha_mining_agent.run_evolution_loop import ResearchAuditContext
from examples.multi_factor_cta.data_engine import AssetClass, AssetSpec, CrossAssetDataEngine
from examples.multi_factor_cta.dual_engine_validation import run_dual_engine_validation
from examples.multi_factor_cta.run_framework_demo import load_research_universe, synthesize_admitted_asset
from examples.multi_factor_cta.synthesis_engine import (
    CrossAssetHRPSynthesizer,
    IntraAssetThreePillarSynthesizer,
    account_portfolio_execution,
    build_asset_execution_inputs,
)
from examples.multi_factor_cta.test_all_weather_50commodities_portfolio import evaluate_50_weighting_schemes

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / "examples/results/research_rerun_20261017"
PRIMARY_SCHEME = "4_SecHRP_100%"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execution_prices(grid: pd.DataFrame) -> pd.Series:
    prices = pd.Series(grid.open.to_numpy(dtype=float), index=pd.DatetimeIndex(grid.dt))
    prices.loc[prices.index[-1] + pd.Timedelta(microseconds=1)] = float(grid.close.iloc[-1])
    return prices


def _save_dual(path: Path, prices: pd.DataFrame, weights: pd.DataFrame, fees: dict) -> dict:
    result = run_dual_engine_validation(prices, weights, fees)
    path.mkdir(parents=True, exist_ok=True)
    _write_json(path / "summary.json", result["summary"])
    prices.to_parquet(path / "execution_prices.parquet")
    weights.to_parquet(path / "executed_weights.parquet")
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_parquet(path / f"{key}.parquet")
    return result["summary"]


def run_research(run_dir: Path, phase: str) -> dict:
    """Run the same official admission/synthesis functions against one frozen manifest."""
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest[phase]
    for record in records:
        if _file_hash(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"Manifest data hash mismatch: {record['path']}")
    output = run_dir / phase
    output.mkdir(parents=True, exist_ok=True)
    tfs = ("1d",) if phase == "daily" else ("1d", "60m", "30m", "15m")
    evaluated_tfs = tfs if phase == "daily" else tfs[1:]
    candidates = sorted(
        f"{sid}@{tf}" for sid, factor in ALL_SIGNAL_SPECS.items() for tf in evaluated_tfs if tf in factor.eligible_tfs
    )
    specs = [
        AssetSpec(
            symbol=row["symbol"],
            name=row["id"],
            asset_class=(
                AssetClass.COMMODITY_FUTURES if row["asset_group"] == "commodity" else AssetClass.EQUITY_REFERENCE
            ),
            sector=row["sector"],
            base_tf="1d" if phase == "daily" else "15m",
            eligible_tfs=tfs,
            fee_rate=row["fee_rate"],
            has_night_session=row["has_night_session"],
            data_path=Path(row["path"]),
            adjust_rolls=False,
        )
        for row in records
    ]
    # Fingerprint the actual dirty worktree, not only its last committed revision.
    source_paths = sorted(
        [*ROOT.glob("examples/alpha_mining_agent/*.py"), *ROOT.glob("examples/multi_factor_cta/*.py")]
    )
    import czsc._native as native

    protocol = {
        "phase": phase,
        "manifest_sha256": _file_hash(manifest_path),
        "source_sha256": {str(p.relative_to(ROOT)): _file_hash(p) for p in source_paths},
        "native_sha256": _file_hash(Path(native.__file__)),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("czsc", "wbt", "backtrader", "pandas")},
        "candidate_ids": candidates,
        "specs": [asdict(spec) for spec in specs],
        "selection_cutoff": str(IS_END_DT),
        "primary_scheme": PRIMARY_SCHEME,
        "accounting": "fixed-initial-notional additive returns; close decisions execute at next observed open",
        "rebalance": "restore target notional at every quoted mark; freeze quantity without a quote",
        "cost_model": "shared Rust actual traded quantity times execution price times single-side fee",
        "cost_assumption": "existing per-side base fees plus separately reported 0/2/5/10 bp extra friction",
        "diagnostic_control": "ag/cu/rb/m, repeating 20-bar blocks of 0,+0.12,+0.06,-0.12; final close liquidation",
        "limitations": manifest["limitations"],
    }
    _write_json(output / "protocol.json", protocol)
    print(f"[{phase}] loading {len(specs)} real assets; {len(candidates)} fixed candidates", flush=True)
    frames = load_research_universe(CrossAssetDataEngine(output / "native_cache"), specs)
    for asset_frames in frames.values():
        for frame in asset_frames.values():
            if not frame.empty and frame.dt.min() < pd.Timestamp(manifest["training_start"]):
                raise ValueError("The formal loader retained observations before the declared research start")
    audit_context = ResearchAuditContext(frames)
    synthesizer = IntraAssetThreePillarSynthesizer()
    returns, positions, prices, sectors, fees = {}, {}, {}, {}, {}
    rows, pooled_written = [], set()
    for i, (spec, record) in enumerate(zip(specs, records, strict=True)):
        target, units, _, audits = synthesize_admitted_asset(
            spec, frames, candidates, synthesizer, is_end_dt=IS_END_DT, audit_context=audit_context
        )
        grid = frames[spec.symbol][spec.base_tf]
        compact = {}
        for uid, audit in audits.items():
            pooled = audit["gate1b"]
            pooled_text = json.dumps(pooled, sort_keys=True, default=str, ensure_ascii=False)
            pooled_key = hashlib.sha256(pooled_text.encode()).hexdigest()[:24]
            if pooled_key not in pooled_written:
                _write_json(output / "pooled" / f"{pooled_key}.json", pooled)
                pooled_written.add(pooled_key)
            compact[uid] = {k: v for k, v in audit.items() if k != "gate1b"} | {
                "gate1b_ref": f"pooled/{pooled_key}.json"
            }
            execution = audit["execution"]
            rows.append(
                {
                    "id": record["id"],
                    "symbol": spec.symbol,
                    "asset_group": record["asset_group"],
                    "candidate": uid,
                    "passed": audit["passed"],
                    "failed_gate": audit["failed_gate"],
                    "reason": audit["reason"],
                    "execution_passed": execution["passed"],
                    "pooled_passed": pooled["passed"],
                    "pooled_reason": pooled["reason"],
                    "pool_size": pooled.get("pool_size"),
                    "pooled_t": pooled.get("pooled_t_stat"),
                    **{
                        key: execution.get(key)
                        for key in (
                            "is_n_purged",
                            "is_effective_n",
                            "is_mean_ci_low_pct",
                            "is_win_rate_pct",
                            "is_pl_ratio",
                            "is_safety_margin_pct",
                            "is_net_sharpe",
                            "is_net_cum_ret_pct",
                        )
                    },
                    "smoothness_passed": audit["parameter_smoothness"]["passed"],
                }
            )
        _write_json(output / "admission" / f"{record['id']}.json", compact)
        pd.DataFrame(rows).to_csv(output / "admission_summary.csv", index=False)
        pd.DataFrame({"dt": grid.dt, "target": target}).to_parquet(output / f"target_{record['id']}.parquet")
        if record["asset_group"] == "commodity":
            key = record["id"]
            returns[key], positions[key] = build_asset_execution_inputs(grid, target)
            prices[key] = _execution_prices(grid)
            sectors[key], fees[key] = spec.sector, spec.fee_rate
        print(f"[{phase}] {i + 1}/{len(specs)} {record['id']}: admitted {len(units)}/{len(audits)}", flush=True)
    results = evaluate_50_weighting_schemes(
        returns,
        sectors,
        asset_positions=positions,
        asset_fee_rates=fees,
        periods_per_year=252 if phase == "daily" else 252 * 16,
    )
    raw_returns = pd.DataFrame(returns).sort_index().fillna(0.0)
    portfolio_rows, stress_rows = [], []
    figure = go.Figure()
    for scheme, result in results.items():
        ledger = result["ledger"]
        weights = result["weights_df"]
        weights.to_parquet(output / f"weights_{scheme.replace('/', '_')}.parquet")
        ledger["cost_returns_df"].to_parquet(output / f"costs_{scheme.replace('/', '_')}.parquet")
        for split in ("is", "oos", "full"):
            mask = weights.index <= IS_END_DT if split == "is" else weights.index > IS_END_DT
            if split == "full":
                mask = np.ones(len(weights), dtype=bool)
            portfolio_rows.append(
                {
                    "scheme": scheme,
                    "split": split,
                    **result[split],
                    "gross_return_pct": float(ledger["gross_returns_df"].loc[mask].to_numpy().sum() * 100),
                    "cost_return_pct": float(ledger["cost_returns_df"].loc[mask].to_numpy().sum() * 100),
                    "turnover": float(ledger["turnover_df"].loc[mask].to_numpy().sum()),
                }
            )
        for extra_bp in (0, 2, 5, 10):
            stress = account_portfolio_execution(
                raw_returns,
                weights,
                {s: fee + extra_bp / 10000 for s, fee in fees.items()},
                tradable_mask=pd.DataFrame(returns).reindex_like(raw_returns).notna(),
            )
            daily = stress["daily_returns"]
            oos = daily[daily.index > IS_END_DT.normalize()]
            stress_rows.append(
                {
                    "scheme": scheme,
                    "extra_friction_bp": extra_bp,
                    **CrossAssetHRPSynthesizer._calc_return_series_stats(oos),
                    "cost_return_pct": float(
                        stress["cost_returns_df"].loc[weights.index > IS_END_DT].to_numpy().sum() * 100
                    ),
                }
            )
        figure.add_trace(
            go.Scatter(x=result["daily_returns"].index, y=1 + result["daily_returns"].cumsum(), name=scheme)
        )
    pd.DataFrame(portfolio_rows).to_csv(output / "portfolio_metrics.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(output / "friction_stress.csv", index=False)
    figure.update_layout(
        title=f"{phase}: formally admitted commodity portfolio; fixed notional", yaxis_title="1 + cumulative net return"
    )
    figure.write_html(output / "portfolio.html", include_plotlyjs=True)
    all_prices = pd.DataFrame(prices).sort_index()
    all_prices.to_parquet(output / "execution_prices.parquet")
    return reconcile_research(run_dir, phase)


def reconcile_research(run_dir: Path, phase: str) -> dict:
    """Resume broker validation from saved final exposures without recomputing admission."""
    output = run_dir / phase
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if _file_hash(run_dir / "manifest.json") != protocol["manifest_sha256"]:
        raise ValueError("Research manifest changed after selection")
    for record in manifest[phase]:
        if _file_hash(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"Manifest data hash mismatch: {record['path']}")
    fees = {spec["name"]: spec["fee_rate"] for spec in protocol["specs"] if spec["asset_class"] == "COMMODITY_FUTURES"}
    all_prices = pd.read_parquet(output / "execution_prices.parquet")
    final_weights = pd.read_parquet(output / f"weights_{PRIMARY_SCHEME}.parquet")
    changes = final_weights.diff().fillna(final_weights.iloc[0])
    if (changes.where(all_prices.isna(), 0.0).abs() > 1e-12).any().any():
        raise ValueError("An asset traded while its execution quote was unavailable")
    # Separate per-asset broker runs retain full histories without filling pre-listing prices.
    dual_summaries = {}
    for symbol in all_prices:
        observed = all_prices[symbol].dropna()
        weights = final_weights.loc[observed.index, [symbol]]
        dual_summaries[symbol] = _save_dual(
            output / "dual_formal" / symbol, observed.to_frame(), weights, {symbol: fees[symbol]}
        )
    # A second shared-broker replay covers a synchronized multi-asset price window.
    control_symbols = ["ag", "cu", "rb", "m"]
    control_prices = all_prices[control_symbols].dropna()
    schedule = np.array([0.0, 0.12, 0.06, -0.12])[(np.arange(len(control_prices)) // 20) % 4]
    schedule[-1] = 0.0
    control_weights = pd.DataFrame(dict.fromkeys(control_symbols, schedule), index=control_prices.index)
    control = _save_dual(
        output / "dual_diagnostic_control", control_prices, control_weights, {s: fees[s] for s in control_symbols}
    )
    audit_table = pd.read_csv(output / "admission_summary.csv")
    portfolio_metrics = pd.read_csv(output / "portfolio_metrics.csv")
    primary = portfolio_metrics[portfolio_metrics.scheme == PRIMARY_SCHEME].set_index("split")
    aggregate = {
        key: float(sum(row[key] for row in dual_summaries.values()))
        for key in (
            "formal_gross_return",
            "formal_cost_return",
            "formal_net_return",
            "wbt_raw_gross_return",
            "wbt_standardized_gross_return",
            "wbt_standardized_cost_return",
            "wbt_standardized_net_return",
            "backtrader_fixed_gross_return",
            "backtrader_fixed_cost_return",
            "backtrader_fixed_net_return",
        )
    }
    for key, column in (("formal_gross_return", "gross_return_pct"), ("formal_cost_return", "cost_return_pct")):
        if not np.isclose(aggregate[key], primary.loc["full", column] / 100, atol=1e-10, rtol=1e-9):
            raise ValueError(f"Per-asset broker inputs disagree with the formal portfolio: {key}")
    summary = {
        "phase": phase,
        "assets": len(protocol["specs"]),
        "commodity_assets": len(fees),
        "candidates": len(protocol["candidate_ids"]),
        "admission_checks": len(audit_table),
        "admitted_units": int(audit_table.passed.sum()),
        "commodity_admitted_units": int(audit_table.loc[audit_table.asset_group == "commodity", "passed"].sum()),
        "execution_passed_units": int(audit_table.execution_passed.sum()),
        "pooled_passed_units": int(audit_table.pooled_passed.sum()),
        "formal_primary_oos": primary.loc["oos"].to_dict(),
        "formal_primary_full": primary.loc["full"].to_dict(),
        **aggregate,
        "dual_formal": dual_summaries,
        "dual_diagnostic_control": control,
        "reconciliation_source_sha256": {
            str(p.relative_to(ROOT)): _file_hash(p)
            for p in (
                Path(__file__),
                ROOT / "examples/multi_factor_cta/dual_engine_validation.py",
            )
        },
    }
    _write_json(output / "summary.json", summary)
    print(
        f"[{phase}] complete: {summary['admitted_units']} admitted; primary OOS={summary['formal_primary_oos']}",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--phase", choices=("daily", "intraday"), default="daily")
    parser.add_argument("--reconcile-only", action="store_true", help="Replay saved final exposures without refitting")
    args = parser.parse_args()
    (reconcile_research if args.reconcile_only else run_research)(args.run_dir, args.phase)
