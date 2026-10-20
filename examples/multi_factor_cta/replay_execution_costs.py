"""Replay frozen research exposures under the shared actual-quantity cost contract.

The source run remains untouched. No candidate admission, selection, allocation,
price adjustment or target generation is repeated. WBT independently checks zero-
fee gross returns; standardized WBT net returns use the shared Rust cost ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import pandas as pd

import czsc._native
from examples.multi_factor_cta.dual_engine_validation import run_dual_engine_validation
from examples.multi_factor_cta.synthesis_engine import account_portfolio_execution

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "examples/results/research_rerun_20261017"
DEFAULT_OUTPUT = ROOT / "examples/results/research_cost_fix_20261017"
SCENARIOS = (
    ("zero_fee", None),
    ("base", 0.0),
    ("base_plus_2bp", 2.0),
    ("base_plus_5bp", 5.0),
    ("base_plus_10bp", 10.0),
)
RETURN_FIELDS = ("gross_return", "cost_return", "net_return")
FIXED_ENGINES = ("formal", "wbt_raw", "wbt_standardized", "backtrader_fixed")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _save_result(path: Path, result: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=False)
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_parquet(path / f"{key}.parquet")
    _write_json(path / "summary.json", result["summary"])


def _aggregate_formal(results: dict[str, dict[str, Any]]) -> tuple[dict, pd.DataFrame]:
    """Aggregate already budgeted per-asset ledgers without a second allocation."""
    bars = (
        pd.concat(
            [
                result["bar_ledger"].loc[:, pd.IndexSlice[list(FIXED_ENGINES), list(RETURN_FIELDS)]]
                for result in results.values()
            ]
        )
        .groupby(level=0)
        .sum()
    )
    summaries = [result["summary"] for result in results.values()]
    summary = {
        "assets": len(results),
        "asset_replays": len(results),
        "execution_marks": len(bars),
        "nonzero_weight_marks": sum(row["nonzero_weight_marks"] for row in summaries),
        "backtrader_fixed_orders": sum(row["backtrader_fixed_orders"] for row in summaries),
        "flat_path": all(row["flat_path"] for row in summaries),
        "execution_reconciled": all(row["execution_reconciled"] for row in summaries),
        "capital_basis": "fixed_initial_capital",
        "initial_capital": summaries[0]["initial_capital"],
        "aggregation": "sum of final budgeted asset returns; no shared-broker margin assertion",
        "absolute_bar_tolerance": summaries[0]["absolute_bar_tolerance"],
        "absolute_total_tolerance": summaries[0]["absolute_total_tolerance"],
        **{
            f"{engine}_{metric}": float(bars[engine][metric].sum())
            for engine in FIXED_ENGINES
            for metric in RETURN_FIELDS
        },
        "maximum_asset_bar_absolute_error": max(
            check["max_bar_absolute_error"]
            for row in summaries
            for engine in row["reconciliation_checks"].values()
            for check in engine.values()
        ),
        "maximum_asset_total_absolute_error": max(
            check["total_absolute_error"]
            for row in summaries
            for engine in row["reconciliation_checks"].values()
            for check in engine.values()
        ),
    }
    summary["reconciliation_checks"] = {}
    for engine, metrics in (
        ("backtrader_fixed", RETURN_FIELDS),
        ("wbt_raw", ("gross_return",)),
        ("wbt_standardized", RETURN_FIELDS),
    ):
        summary["reconciliation_checks"][engine] = {}
        for metric in metrics:
            maximum = float((bars[engine][metric] - bars["formal"][metric]).abs().max())
            total = abs(float(bars[engine][metric].sum() - bars["formal"][metric].sum()))
            passed = maximum <= summary["absolute_bar_tolerance"] and total <= summary["absolute_total_tolerance"]
            summary["reconciliation_checks"][engine][metric] = {
                "max_bar_absolute_error": maximum,
                "total_absolute_error": total,
                "passed": passed,
            }
            summary["execution_reconciled"] &= passed
    return summary, bars


def _stress_row(phase: str, group: str, scenario: str, fees: dict[str, float], summary: dict) -> dict:
    checks = [check for engine in summary["reconciliation_checks"].values() for check in engine.values()]
    return {
        "phase": phase,
        "group": group,
        "scenario": scenario,
        "minimum_fee_bp": min(fees.values()) * 1e4,
        "maximum_fee_bp": max(fees.values()) * 1e4,
        "assets": summary["assets"],
        "execution_marks": summary["execution_marks"],
        "nonzero_weight_marks": summary["nonzero_weight_marks"],
        "backtrader_fixed_orders": summary["backtrader_fixed_orders"],
        **{f"{engine}_{metric}": summary[f"{engine}_{metric}"] for engine in FIXED_ENGINES for metric in RETURN_FIELDS},
        "max_bar_absolute_error": max(check["max_bar_absolute_error"] for check in checks),
        "max_total_absolute_error": max(check["total_absolute_error"] for check in checks),
        "execution_reconciled": summary["execution_reconciled"],
        "flat_path": summary["flat_path"],
    }


def _replay_schemes(
    output: Path,
    phase: str,
    scenario: str,
    prices: pd.DataFrame,
    schemes: dict[str, pd.DataFrame],
    fees: dict[str, float],
    cutoff: pd.Timestamp,
) -> list[dict]:
    """Reaccount all frozen schemes with Rust; this does not allocate or fit."""
    tradable = prices.notna()
    returns = prices.ffill().pct_change(fill_method=None).where(tradable, 0.0).fillna(0.0)
    rows = []
    for name, weights in schemes.items():
        ledger = account_portfolio_execution(returns, weights, fees, tradable_mask=tradable)
        bars = pd.DataFrame(
            {
                "gross_return": ledger["gross_returns_df"].sum(axis=1),
                "cost_return": ledger["cost_returns_df"].sum(axis=1),
                "net_return": ledger["bar_net_returns"],
                "turnover": ledger["turnover_df"].sum(axis=1),
            }
        )
        destination = output / phase / scenario / "schemes" / name
        destination.mkdir(parents=True)
        bars.to_parquet(destination / "bar_ledger.parquet")
        bars.groupby(bars.index.normalize()).sum().to_parquet(destination / "daily_ledger.parquet")
        rows.append(
            {
                "phase": phase,
                "scheme": name,
                "scenario": scenario,
                "engine": "shared_rust_formal_ledger_only",
                "capital_basis": "fixed_initial_capital",
                "minimum_fee_bp": min(fees.values()) * 1e4,
                "maximum_fee_bp": max(fees.values()) * 1e4,
                "assets": len(weights.columns),
                "execution_marks": len(weights),
                "nonzero_weight_marks": int((weights.abs() > 1e-12).to_numpy().sum()),
                "flat_path": not bool((weights.abs() > 1e-12).any().any()),
                **{metric: float(bars[metric].sum()) for metric in bars},
                **{f"oos_{metric}": float(bars.loc[bars.index > cutoff, metric].sum()) for metric in bars},
            }
        )
    return rows


def replay_execution_costs(source: Path, output: Path, *, phases: tuple[str, ...] = ("daily", "intraday")) -> dict:
    """Replay saved real-price formal and diagnostic inputs at all five fee levels."""
    source = source.resolve()
    output = output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be independent directories")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists() or any((output / phase).exists() for phase in phases):
        raise FileExistsError("Output already contains a cost replay; choose a new output directory")
    source_files = [source / "manifest.json"]
    all_inputs = {}
    scheme_inputs = {}
    for phase in phases:
        phase_root = source / phase
        protocol_path = phase_root / "protocol.json"
        source_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if source_protocol["manifest_sha256"] != _sha256(source / "manifest.json"):
            raise ValueError(f"Source manifest changed after {phase} admission")
        fees = {
            spec["name"]: spec["fee_rate"]
            for spec in source_protocol["specs"]
            if spec["asset_class"] == "COMMODITY_FUTURES"
        }
        source_files.extend(
            [protocol_path, phase_root / "admission_summary.csv", phase_root / "weights_4_SecHRP_100%.parquet"]
        )
        portfolio_weights = pd.read_parquet(phase_root / "weights_4_SecHRP_100%.parquet")
        portfolio_prices = pd.read_parquet(phase_root / "execution_prices.parquet")
        scheme_paths = sorted(phase_root.glob("weights_*.parquet"))
        if len(scheme_paths) != 10:
            raise ValueError(f"Expected all 10 frozen schemes, found {len(scheme_paths)} in {phase}")
        schemes = {path.stem.removeprefix("weights_"): pd.read_parquet(path) for path in scheme_paths}
        for name, weights in schemes.items():
            if not weights.index.equals(portfolio_prices.index) or not weights.columns.equals(portfolio_prices.columns):
                raise ValueError(f"Frozen scheme is not aligned with its execution prices: {phase}/{name}")
        scheme_inputs[phase] = (portfolio_prices, schemes, pd.Timestamp(source_protocol["selection_cutoff"]))
        source_files.extend(scheme_paths)
        source_files.append(phase_root / "execution_prices.parquet")
        if set(portfolio_prices.columns) != set(fees):
            raise ValueError(f"Saved commodity universe changed: {phase}")
        inputs = {}
        for symbol in fees:
            path = phase_root / "dual_formal" / symbol
            prices_path, weights_path = path / "execution_prices.parquet", path / "executed_weights.parquet"
            prices, weights = pd.read_parquet(prices_path), pd.read_parquet(weights_path)
            if list(prices.columns) != [symbol]:
                raise ValueError(f"Unexpected saved asset input: {prices_path}")
            expected_prices = portfolio_prices[[symbol]].dropna()
            expected_weights = portfolio_weights.loc[expected_prices.index, [symbol]]
            pd.testing.assert_frame_equal(prices, expected_prices, check_freq=False)
            pd.testing.assert_frame_equal(weights, expected_weights, check_freq=False)
            inputs[symbol] = (prices, weights)
            source_files.extend([prices_path, weights_path])
        control = phase_root / "dual_diagnostic_control"
        control_prices = pd.read_parquet(control / "execution_prices.parquet")
        control_weights = pd.read_parquet(control / "executed_weights.parquet")
        source_files.extend([control / "execution_prices.parquet", control / "executed_weights.parquet"])
        all_inputs[phase] = (fees, inputs, control_prices, control_weights)
    native_path = Path(czsc._native.__file__)
    code_paths = [
        Path(__file__),
        ROOT / "examples/multi_factor_cta/dual_engine_validation.py",
        ROOT / "examples/multi_factor_cta/synthesis_engine.py",
        ROOT / "crates/czsc-core/src/research_execution.rs",
        ROOT / "czsc/__init__.py",
        native_path,
    ]
    code_paths.extend(ROOT.glob("crates/czsc-core/src/*execution*.rs"))
    protocol = {
        "source_run": str(source),
        "phases": phases,
        "initial_capital": 1_000_000.0,
        "capital_basis": "fixed initial capital; all per-bar returns expressed over that same capital",
        "rebalance_policy": "target quantity w*N/P at every executable mark",
        "fee_scenarios": dict(SCENARIOS),
        "wbt_contract": "raw WBT zero-fee gross; shared Rust actual-quantity costs and Rust net standardization",
        "selection": "reuse frozen prices and final exposures; no refit, new admission or target selection",
        "formal_aggregation": "sum per-asset final portfolio budgets; control uses one four-asset Backtrader broker",
        "scheme_replays": "all 10 frozen schemes at all 5 fee levels use Rust accounting; actual dual-engine replay covers primary only",
        "diagnostic_only": "nonzero controls are fixed mechanical paths, not admitted strategies or alpha evidence",
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("czsc", "wbt", "backtrader", "pandas")},
        "source_sha256": {str(path): _sha256(path) for path in source_files},
        "implementation_sha256": {str(path): _sha256(path) for path in dict.fromkeys(code_paths)},
    }
    _write_json(output / "protocol.json", protocol)
    stress_rows = []
    scheme_stress_rows = []
    phase_summaries = {}
    for phase, (base_fees, inputs, control_prices, control_weights) in all_inputs.items():
        phase_output = output / phase
        for symbol, (prices, weights) in inputs.items():
            folder = phase_output / "inputs" / symbol
            folder.mkdir(parents=True)
            prices.to_parquet(folder / "execution_prices.parquet")
            weights.to_parquet(folder / "executed_weights.parquet")
        control_folder = phase_output / "inputs" / "diagnostic_control"
        control_folder.mkdir(parents=True)
        control_prices.to_parquet(control_folder / "execution_prices.parquet")
        control_weights.to_parquet(control_folder / "executed_weights.parquet")
        phase_summaries[phase] = {}
        portfolio_prices, schemes, cutoff = scheme_inputs[phase]
        portfolio_prices.to_parquet(phase_output / "inputs" / "execution_prices.parquet")
        for name, weights in schemes.items():
            weights.to_parquet(phase_output / "inputs" / f"weights_{name}.parquet")
        for scenario, extra_bp in SCENARIOS:
            fees = {symbol: 0.0 if extra_bp is None else fee + extra_bp / 1e4 for symbol, fee in base_fees.items()}
            scheme_stress_rows.extend(_replay_schemes(output, phase, scenario, portfolio_prices, schemes, fees, cutoff))
            pd.DataFrame(scheme_stress_rows).to_csv(output / "scheme_friction_stress.csv", index=False)
            print(f"[{phase}/{scenario}] replaying {len(inputs)} frozen formal assets", flush=True)
            results = {}
            for symbol, (prices, weights) in inputs.items():
                result = run_dual_engine_validation(prices, weights, {symbol: fees[symbol]})
                _save_result(phase_output / scenario / "formal" / symbol, result)
                # Keep only the small fixed-basis aggregate inputs in memory.
                results[symbol] = {"bar_ledger": result["bar_ledger"], "summary": result["summary"]}
            formal, formal_bars = _aggregate_formal(results)
            del results
            _write_json(phase_output / scenario / "formal_summary.json", formal)
            formal_bars.to_parquet(phase_output / scenario / "formal_bar_ledger.parquet")
            formal_bars.groupby(formal_bars.index.normalize()).sum().to_parquet(
                phase_output / scenario / "formal_daily_ledger.parquet"
            )
            control_fees = {symbol: fees[symbol] for symbol in control_prices}
            control_result = run_dual_engine_validation(control_prices, control_weights, control_fees)
            _save_result(phase_output / scenario / "diagnostic_control", control_result)
            control_summary = control_result["summary"]
            phase_summaries[phase][scenario] = {"formal": formal, "diagnostic_control": control_summary}
            stress_rows.extend(
                [
                    _stress_row(phase, "formal", scenario, fees, formal),
                    _stress_row(phase, "diagnostic_control", scenario, control_fees, control_summary),
                ]
            )
            pd.DataFrame(stress_rows).to_csv(output / "friction_stress.csv", index=False)
            print(
                f"[{phase}/{scenario}] formal={formal['execution_reconciled']} control={control_summary['execution_reconciled']} "
                f"net={control_summary['formal_net_return']:.12g} cost={control_summary['formal_cost_return']:.12g}",
                flush=True,
            )
            if not formal["execution_reconciled"] or not control_summary["execution_reconciled"]:
                raise AssertionError(f"Absolute execution reconciliation failed: {phase}/{scenario}")
    for category in ("source_sha256", "implementation_sha256"):
        for path, original_hash in protocol[category].items():
            if _sha256(Path(path)) != original_hash:
                raise ValueError(f"{category} changed during replay: {path}")
    summary = {
        "execution_reconciled": True,
        "formal_asset_replays": sum(len(inputs) * len(SCENARIOS) for _, inputs, _, _ in all_inputs.values()),
        "diagnostic_control_replays": len(phases) * len(SCENARIOS),
        "formal_scheme_rust_replays": len(scheme_stress_rows),
        "all_frozen_schemes_flat": all(row["flat_path"] for row in scheme_stress_rows),
        "source_files_unchanged": True,
        "phases": phase_summaries,
        "limitations": [
            "Flat formal portfolios do not establish economic profitability.",
            "Standardized WBT costs share Rust implementation; independent cost evidence comes from actual Backtrader fills.",
            "Fractional normalized units do not model contract lots, real margins, rolls or exchange calendars.",
            "Equity-fraction Backtrader account remains a separate sizing diagnostic.",
            "WBT 0.2.1 high-precision gross mismatch is independently regression-tested to fail strict reconciliation; frozen zero/0.06/0.12 controls do not exercise that limitation.",
        ],
    }
    _write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("daily", "intraday", "both"), default="both")
    args = parser.parse_args()
    selected = ("daily", "intraday") if args.phase == "both" else (args.phase,)
    replay_execution_costs(args.source, args.output, phases=selected)
