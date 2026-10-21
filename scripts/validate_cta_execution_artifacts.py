"""Validate sealed CTA ledgers without selecting or changing any factor.

Usage: uv run --no-sync python -m scripts.validate_cta_execution_artifacts \
    --input path/to/ledger.pkl --output path/to/validation

Each pickle must contain a single ledger dictionary with returns_df,
execution_prices_df, weights_df, tradable_mask and fee_rates. The original gross,
cost and turnover panels are checked when present. Friction scenarios add
0/2/5/10 single-side basis points to the frozen base fees and keep the fills fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import czsc
from examples.multi_factor_cta.audit_protocol import ArtifactFingerprint, canonical_json, read_verified_artifact
from examples.multi_factor_cta.data_engine import assign_trading_date
from examples.multi_factor_cta.dual_engine_validation import _backtrader_replay

EXTRA_FRICTION_BPS = (0.0, 2.0, 5.0, 10.0)
BAR_TOLERANCE = 1e-10
TOTAL_TOLERANCE = 1e-8


def _stats(net: pd.Series, gross: pd.Series, cost: pd.Series) -> dict[str, float | int | str]:
    days = pd.DatetimeIndex(assign_trading_date(pd.Series(net.index), has_night_session=True))
    daily = net.groupby(days).sum()
    nav = 1.0 + daily.cumsum()
    peak = nav.clip(lower=1.0).cummax()
    std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    return {
        "start": str(net.index.min()),
        "end": str(net.index.max()),
        "trading_days": len(daily),
        "gross_return_pct": float(gross.sum() * 100.0),
        "cost_return_pct": float(cost.sum() * 100.0),
        "net_return_pct": float(net.sum() * 100.0),
        "sharpe": float(daily.mean() / std * np.sqrt(252.0)) if std > 0 else 0.0,
        "max_drawdown_pct": float((nav / peak - 1.0).min() * 100.0),
    }


def _check_panels(bundle: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    returns = bundle["returns_df"]
    prices = bundle["execution_prices_df"]
    weights = bundle["weights_df"]
    mask = bundle["tradable_mask"]
    if returns.empty or not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("A nonempty DatetimeIndex execution panel is required")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("Execution timestamps must be unique and chronological")
    for panel in [prices, weights, mask]:
        if not panel.index.equals(returns.index) or not panel.columns.equals(returns.columns):
            raise ValueError("All execution panels must be exactly aligned")
    if mask.isna().any().any() or not all(pd.api.types.is_bool_dtype(dtype) for dtype in mask.dtypes):
        raise ValueError("tradable_mask must contain explicit booleans")
    if not np.isfinite(returns.to_numpy(dtype=float)).all() or not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("Returns and executed weights must be finite")
    observed = prices.where(mask).stack()
    if len(observed) != int(mask.to_numpy().sum()) or not np.isfinite(observed).all() or (observed <= 0).any():
        raise ValueError("Every observed execution requires its real positive price")
    fees = {symbol: float(bundle["fee_rates"][symbol]) for symbol in returns}
    return returns, prices, weights, mask, fees


def _difference(actual: pd.DataFrame, expected: pd.DataFrame) -> dict[str, float | bool]:
    if not actual.index.equals(expected.index) or not actual.columns.equals(expected.columns):
        raise ValueError("Reference accounting panels are not aligned")
    delta = actual - expected
    max_bar = float(np.abs(delta.to_numpy()).max())
    total = abs(float(delta.to_numpy().sum()))
    return {
        "max_bar_asset_abs_error": max_bar,
        "total_abs_error": total,
        "passed": max_bar <= BAR_TOLERANCE and total <= TOTAL_TOLERANCE,
    }


def validate_bundle(input_path: Path, output_dir: Path, *, initial_capital: float = 1_000_000.0) -> dict:
    """Reconcile saved panels, stress fixed orders and independently execute them in Backtrader."""
    metadata = json.loads(input_path.with_name(input_path.name + ".provenance.json").read_bytes())
    fingerprint = ArtifactFingerprint(metadata["fingerprint"], canonical_json(metadata["manifest"]))
    source_bytes = read_verified_artifact(input_path, fingerprint)
    bundle = pickle.loads(source_bytes)
    returns, prices, weights, mask, fees = _check_panels(bundle)
    output_dir.mkdir(parents=True, exist_ok=True)
    field_names = {"gross": "gross_returns", "cost": "cost_returns", "net": "net_returns", "turnover": "turnover"}
    base = {}
    stresses = []
    daily_stresses = []
    for extra_bp in EXTRA_FRICTION_BPS:
        asset_ledgers = {
            symbol: czsc.executed_weight_ledger(
                returns[symbol].tolist(),
                weights[symbol].tolist(),
                fee_rate=fees[symbol] + extra_bp / 10000.0,
                tradable=mask[symbol].tolist(),
            )
            for symbol in returns
        }
        panels = {
            name: pd.DataFrame(
                {symbol: getattr(result, field) for symbol, result in asset_ledgers.items()}, index=returns.index
            )
            for name, field in field_names.items()
        }
        if extra_bp == 0:
            base = panels
        daily = pd.DataFrame({name: panels[name].sum(axis=1) for name in ("gross", "cost", "net")})
        days = pd.DatetimeIndex(assign_trading_date(pd.Series(daily.index), has_night_session=True))
        daily = daily.groupby(days).sum().reset_index(names="trading_day")
        daily["extra_single_side_bp"] = extra_bp
        daily_stresses.append(daily)
        stresses.append(
            {
                "extra_single_side_bp": extra_bp,
                **_stats(panels["net"].sum(axis=1), panels["gross"].sum(axis=1), panels["cost"].sum(axis=1)),
            }
        )
    pd.DataFrame(stresses).to_csv(output_dir / "friction_sensitivity.csv", index=False)
    pd.concat(daily_stresses, ignore_index=True).to_csv(output_dir / "friction_daily_returns.csv", index=False)
    references = {}
    for name, saved in [("gross", "gross_returns_df"), ("cost", "cost_returns_df"), ("turnover", "turnover_df")]:
        if saved in bundle:
            references[saved] = _difference(base[name], bundle[saved])

    actual = {
        name: pd.DataFrame(0.0, index=returns.index, columns=returns.columns) for name in ("gross", "cost", "net")
    }
    orders, per_asset = [], []
    for symbol in returns:
        observed_idx = returns.index[mask[symbol].to_numpy()]
        if len(observed_idx) == 0:
            if weights[symbol].abs().max() > 0:
                raise ValueError(f"{symbol}: nonzero exposure without a single executable quote")
            continue
        px = prices.loc[observed_idx, [symbol]].copy()
        target = weights.loc[observed_idx, [symbol]].copy()
        if abs(float(target.iloc[-1, 0])) > 1e-12:
            raise ValueError(f"{symbol}: saved ledger is missing terminal liquidation at an observed price")
        observed_returns = px[symbol].pct_change(fill_method=None).iloc[1:]
        if not np.allclose(observed_returns, returns.loc[observed_idx[1:], symbol], rtol=1e-10, atol=1e-12):
            raise ValueError(f"{symbol}: saved returns differ from its actual fill-to-fill prices")
        # A flat initialization at the first known quote has zero PnL. The extra
        # bookkeeping mark allows the broker to place a first-quote entry order;
        # it does not fill missing historical quotes or infer an earlier signal.
        if not target.to_numpy().any():
            per_asset.append(
                {"symbol": symbol, "observed_marks": len(observed_idx), "orders": 0, "terminal_quantity": 0.0}
            )
            continue
        initialization = observed_idx[0] - pd.Timedelta(microseconds=1)
        px.loc[initialization] = px.iloc[0]
        target.loc[initialization] = 0.0
        px, target = px.sort_index(), target.sort_index()
        ledger, fills, inventory = _backtrader_replay(
            px, target, {symbol: fees[symbol]}, initial_capital, "fixed_capital"
        )
        for name, column in [("gross", "gross_return"), ("cost", "cost_return"), ("net", "net_return")]:
            actual[name].loc[observed_idx, symbol] = ledger.loc[observed_idx, column].to_numpy()
        orders.append(fills)
        per_asset.append(
            {
                "symbol": symbol,
                "observed_marks": len(observed_idx),
                "orders": len(fills),
                "terminal_quantity": inventory[symbol],
            }
        )

    comparisons = {name: _difference(actual[name], base[name]) for name in actual}
    bar_panels = pd.concat(
        {f"native_{name}": frame for name, frame in base.items() if name != "turnover"}
        | {f"backtrader_{name}": frame for name, frame in actual.items()},
        axis=1,
    )
    bar_panels.to_parquet(output_dir / "bar_reconciliation.parquet")
    (pd.concat(orders, ignore_index=True) if orders else pd.DataFrame()).to_parquet(
        output_dir / "backtrader_orders.parquet"
    )
    pd.DataFrame(per_asset).to_csv(output_dir / "asset_reconciliation.csv", index=False)
    result = {
        "variant": bundle.get("variant", input_path.stem),
        "fold": bundle.get("fold"),
        "input_path": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "input_fingerprint": fingerprint.sha256,
        "validation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "initial_capital": initial_capital,
        "accounting": "fixed_initial_capital",
        "signal_selection": "unchanged",
        "friction_definition": "base fee plus additional single-side basis points; identical saved target weights",
        "saved_panel_checks": references,
        "backtrader_checks": comparisons,
        "passed": all(check["passed"] for check in [*references.values(), *comparisons.values()]),
        "bar_tolerance": BAR_TOLERANCE,
        "total_tolerance": TOTAL_TOLERANCE,
        "friction": stresses,
        "scope": "continuous reference prices; no contract-roll, limit-lock or market-impact model",
    }
    (output_dir / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _validate_job(args: tuple[Path, Path]) -> dict:
    return validate_bundle(*args)


def validate_directory(input_dir: Path, output_dir: Path, *, workers: int = 1) -> list[dict]:
    """Validate every sealed fold, then stitch unchanged folds for comparable variant statistics."""
    files = sorted(input_dir.glob("*.pkl"))
    if not files:
        raise ValueError(f"No execution ledgers in {input_dir}")
    jobs = [(path, output_dir / path.stem) for path in files]
    if workers == 1:
        results = [_validate_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_validate_job, jobs))
    summaries, daily = [], []
    for source, result in zip(files, results, strict=True):
        summaries.append(
            {
                "variant": result["variant"],
                "fold": result["fold"],
                "passed": result["passed"],
                "max_bar_asset_abs_error": max(
                    row["max_bar_asset_abs_error"] for row in result["backtrader_checks"].values()
                ),
                "input_sha256": result["input_sha256"],
            }
        )
        frame = pd.read_csv(output_dir / source.stem / "friction_daily_returns.csv", parse_dates=["trading_day"])
        frame["variant"] = result["variant"]
        daily.append(frame)
    pd.DataFrame(summaries).to_csv(output_dir / "validation_summary.csv", index=False)
    stress_rows = []
    for (variant, bp), frame in pd.concat(daily).groupby(["variant", "extra_single_side_bp"]):
        if frame.trading_day.duplicated().any():
            raise ValueError(f"Overlapping fold trading days for {variant}")
        series = frame.set_index("trading_day").sort_index()
        stress_rows.append(
            {"variant": variant, "extra_single_side_bp": bp, **_stats(series.net, series.gross, series.cost)}
        )
    pd.DataFrame(stress_rows).to_csv(output_dir / "variant_friction_sensitivity.csv", index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.input.is_dir():
        results = validate_directory(args.input, args.output, workers=args.workers)
    else:
        results = [validate_bundle(args.input, args.output)]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if not all(result["passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
