# ruff: noqa: E402
"""Realistic execution stress tests for the cap-1.5 net-symbol candidate.

The current executable candidate is ``cap15_core_threshold_external_raw``. This
script keeps the same target construction, then adds practical execution
frictions:

- per-symbol daily turnover limits;
- tiered slippage by per-symbol turnover size;
- long-only / no-short constraints;
- short carry costs for hedge / borrow usage.

Run:
    uv run --no-sync python examples/signals_dev/validate_daily_cta_realistic_execution_stress.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_daily_cta_cross_section_walk_forward as cs
from validate_daily_cta_core_execution_constraints import load_core_target_weights, load_price_returns
from validate_daily_cta_gross_net_bridge_audit import raw_external_panel
from validate_daily_cta_layered_low_corr_overlay import (
    TARGET_ANNUAL_RETURN,
    TARGET_MAX_DRAWDOWN,
    evaluate_window_stress,
)
from validate_daily_cta_robust_blend_rebalance_compression import load_component_panel
from validate_daily_cta_walk_forward import FEE_RATE, annualized_stats, summarize_yearly

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_realistic_execution_stress"
THRESHOLD = 0.10
SYMBOL_CAP = 1.5
TRADING_DAYS = 252


@dataclass(frozen=True)
class StressScheme:
    """One realistic execution stress scheme."""

    name: str
    max_symbol_turnover: float | None = None
    tiered_slippage: bool = False
    extra_slippage_bps: float = 0.0
    long_only_symbols: tuple[str, ...] = ()
    short_carry_annual: float = 0.0
    external_extra_bps: float = 0.0


SCHEMES = (
    StressScheme("cap15_base"),
    StressScheme("cap15_turnover_cap_1"),
    StressScheme("cap15_turnover_cap_0_5", max_symbol_turnover=0.5),
    StressScheme("cap15_turnover_cap_0_25", max_symbol_turnover=0.25),
    StressScheme("cap15_tiered_slippage", tiered_slippage=True),
    StressScheme("cap15_tc05_tiered", max_symbol_turnover=0.5, tiered_slippage=True),
    StressScheme("cap15_tc025_tiered", max_symbol_turnover=0.25, tiered_slippage=True),
    StressScheme("cap15_short_carry_5pct", short_carry_annual=0.05),
    StressScheme("cap15_short_carry_10pct", short_carry_annual=0.10),
    StressScheme("cap15_etf_no_short", long_only_symbols=("159915.XSHE",)),
    StressScheme("cap15_all_long_only", long_only_symbols=tuple(cs.SYMBOLS)),
    StressScheme(
        "cap15_harsh_realistic",
        max_symbol_turnover=0.5,
        tiered_slippage=True,
        short_carry_annual=0.10,
        external_extra_bps=5.0,
    ),
)


def scheme_with_name(name: str) -> StressScheme:
    """Return a copy of a named scheme with defaults filled."""
    for scheme in SCHEMES:
        if scheme.name == name:
            return scheme
    raise KeyError(name)


def tiered_slippage_cost(turnover_vec: np.ndarray) -> float:
    """Calculate extra slippage cost from per-symbol turnover tiers."""
    turnover = np.abs(turnover_vec)
    rates = np.select(
        [turnover <= 0.25, turnover <= 0.50, turnover <= 1.00],
        [0.0002, 0.0005, 0.0010],
        default=0.0020,
    )
    return float(np.sum(turnover * rates))


def build_core_target(scheme: StressScheme) -> pd.DataFrame:
    """Build the capped target core weights under shorting constraints."""
    target = load_core_target_weights()
    values = target[list(cs.SYMBOLS)].to_numpy(float)
    values = np.clip(values, -SYMBOL_CAP, SYMBOL_CAP)
    if scheme.long_only_symbols:
        for symbol in scheme.long_only_symbols:
            idx = list(cs.SYMBOLS).index(symbol)
            values[:, idx] = np.clip(values[:, idx], 0.0, SYMBOL_CAP)
    target[list(cs.SYMBOLS)] = values
    return target


def execute_core_scheme(scheme: StressScheme) -> pd.DataFrame:
    """Execute one constrained core scheme."""
    target = build_core_target(scheme)
    returns = load_price_returns(target["dt"])
    joined = target.merge(returns, on="dt", suffixes=("_target", "_ret"), how="inner")
    target_values = joined[[f"{symbol}_target" for symbol in cs.SYMBOLS]].to_numpy(float)
    ret_values = joined[[f"{symbol}_ret" for symbol in cs.SYMBOLS]].to_numpy(float)
    dates = pd.to_datetime(joined["dt"]).reset_index(drop=True)
    current = target_values[0].copy()
    rows = []

    for i, dt in enumerate(dates):
        target_today = target_values[i]
        year_changed = i == 0 or dt.year != dates.iloc[i - 1].year
        drift = float(np.abs(target_today - current).sum())
        rebalance = year_changed or drift >= THRESHOLD
        turnover_vec = np.zeros_like(current)
        if rebalance:
            desired_change = target_today - current
            if scheme.max_symbol_turnover is None:
                change = desired_change
            else:
                change = np.clip(desired_change, -scheme.max_symbol_turnover, scheme.max_symbol_turnover)
            current = current + change
            turnover_vec = np.abs(change)

        turnover = float(turnover_vec.sum())
        base_cost = turnover * FEE_RATE
        flat_slippage = turnover * scheme.extra_slippage_bps / 10000
        tier_cost = tiered_slippage_cost(turnover_vec) if scheme.tiered_slippage else 0.0
        short_carry = float(np.abs(np.minimum(current, 0.0)).sum() * scheme.short_carry_annual / TRADING_DAYS)
        core_ret = float(np.dot(current, ret_values[i]) - base_cost - flat_slippage - tier_cost - short_carry)
        rows.append(
            {
                "dt": dt,
                "scheme": scheme.name,
                "core_ret": core_ret,
                "core_weight": float(np.abs(current).sum()),
                "core_turnover": turnover,
                "base_cost": base_cost,
                "flat_slippage": flat_slippage,
                "tiered_slippage": tier_cost,
                "short_carry": short_carry,
                "rebalance": bool(rebalance),
                "unfilled_drift": float(np.abs(target_today - current).sum()),
                **{symbol: current[idx] for idx, symbol in enumerate(cs.SYMBOLS)},
            }
        )
    return pd.DataFrame(rows)


def external_raw_panel() -> pd.DataFrame:
    """Build raw external overlay panel."""
    desired_weights, component_returns = load_component_panel()
    return raw_external_panel(desired_weights, component_returns)


def build_portfolio_paths() -> pd.DataFrame:
    """Build all realistic stress paths."""
    external = external_raw_panel()
    paths = []
    for scheme in SCHEMES:
        core = execute_core_scheme(scheme)
        joined = core.merge(external, on="dt", how="left")
        joined[["external_raw_ret", "external_raw_weight", "external_raw_turnover"]] = joined[
            ["external_raw_ret", "external_raw_weight", "external_raw_turnover"]
        ].fillna(0.0)
        external_cost = joined["external_raw_turnover"] * scheme.external_extra_bps / 10000
        joined["external_cost"] = external_cost
        joined["external_ret"] = joined["external_raw_ret"] - external_cost
        joined["ret"] = joined["core_ret"] + joined["external_ret"]
        joined["weight"] = joined["core_weight"] + joined["external_raw_weight"]
        joined["turnover"] = joined["core_turnover"] + joined["external_raw_turnover"]
        joined["nav"] = (1 + joined["ret"]).cumprod()
        joined["drawdown"] = joined["nav"] / joined["nav"].cummax() - 1
        paths.append(joined)
    return pd.concat(paths, ignore_index=True)


def summarize_paths(paths: pd.DataFrame) -> pd.DataFrame:
    """Summarize performance and realistic execution diagnostics."""
    rows = []
    years = (paths["dt"].max() - paths["dt"].min()).days / 365.25
    for scheme, group in paths.groupby("scheme", sort=False):
        stats = annualized_stats(group["dt"].reset_index(drop=True), group["ret"].to_numpy(), group["weight"].to_numpy())
        yearly = summarize_yearly(group[["dt", "ret", "weight"]])
        rows.append(
            {
                "scheme": scheme,
                **stats,
                "rebalance_days": int(group["rebalance"].sum()),
                "turnover_sum": float(group["turnover"].sum()),
                "annualized_turnover": float(group["turnover"].sum() / years) if years > 0 else float("nan"),
                "max_core_turnover": float(group["core_turnover"].max()),
                "max_symbol_abs_weight": float(group[list(cs.SYMBOLS)].abs().max().max()),
                "avg_unfilled_drift": float(group["unfilled_drift"].mean()),
                "max_unfilled_drift": float(group["unfilled_drift"].max()),
                "base_cost_sum": float(group["base_cost"].sum()),
                "flat_slippage_sum": float(group["flat_slippage"].sum()),
                "tiered_slippage_sum": float(group["tiered_slippage"].sum()),
                "short_carry_sum": float(group["short_carry"].sum()),
                "external_cost_sum": float(group["external_cost"].sum()),
                "loss_years": int((yearly["return"] < 0).sum()),
                "pass_20_10": bool(
                    stats["annual_return"] >= TARGET_ANNUAL_RETURN
                    and stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
                ),
            }
        )
    return pd.DataFrame(rows)


def yearly_stats(paths: pd.DataFrame) -> pd.DataFrame:
    """Calculate yearly stats for each scheme."""
    rows = []
    for scheme, group in paths.groupby("scheme", sort=False):
        yearly = summarize_yearly(group[["dt", "ret", "weight"]])
        yearly["scheme"] = scheme
        rows.append(yearly)
    return pd.concat(rows, ignore_index=True)


def pressure_windows(paths: pd.DataFrame) -> pd.DataFrame:
    """Calculate pressure-window stats for each scheme."""
    rows = []
    for scheme, group in paths.groupby("scheme", sort=False):
        stress = evaluate_window_stress(group[["dt", "ret", "weight"]].copy(), scheme)
        stress["scheme"] = scheme
        rows.append(stress)
    return pd.concat(rows, ignore_index=True)


def symbol_summary(paths: pd.DataFrame) -> pd.DataFrame:
    """Summarize symbol-level exposure and turnover."""
    rows = []
    for scheme, group in paths.groupby("scheme", sort=False):
        for symbol in cs.SYMBOLS:
            weight = group[symbol].astype(float)
            turnover = weight.diff().abs().fillna(0.0)
            rows.append(
                {
                    "scheme": scheme,
                    "symbol": symbol,
                    "avg_abs_weight": float(weight.abs().mean()),
                    "weight_p95": float(weight.abs().quantile(0.95)),
                    "max_abs_weight": float(weight.abs().max()),
                    "turnover_sum": float(turnover.sum()),
                    "max_daily_turnover": float(turnover.max()),
                    "days_short": int((weight < 0).sum()),
                    "days_abs_weight_gt_1": int((weight.abs() > 1.0).sum()),
                    "days_abs_weight_gt_1_5": int((weight.abs() > 1.5).sum()),
                }
            )
    return pd.DataFrame(rows)


def save_nav_plot(paths: pd.DataFrame) -> None:
    """Save net-value plot."""
    import plotly.graph_objects as go

    focus = (
        "cap15_base",
        "cap15_tc05_tiered",
        "cap15_harsh_realistic",
        "cap15_etf_no_short",
        "cap15_all_long_only",
    )
    fig = go.Figure()
    for scheme, group in paths[paths["scheme"].isin(focus)].groupby("scheme", sort=False):
        fig.add_trace(go.Scatter(x=group["dt"], y=group["nav"], mode="lines", name=scheme))
    fig.update_layout(
        title="cap15 realistic execution stress",
        xaxis_title="date",
        yaxis_title="net value",
        template="plotly_white",
    )
    fig.write_html(OUTPUT_DIR / "realistic_execution_stress_nav.html", include_plotlyjs="cdn")


def main() -> None:
    """Run realistic execution stress tests."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = build_portfolio_paths()
    summary = summarize_paths(paths)
    yearly = yearly_stats(paths)
    stress = pressure_windows(paths)
    failed = stress[~stress["pass_20_10"]].sort_values(["scheme", "annual_return"])
    symbols = symbol_summary(paths)

    paths.to_csv(OUTPUT_DIR / "realistic_execution_stress_daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "realistic_execution_stress_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUTPUT_DIR / "realistic_execution_stress_yearly.csv", index=False, encoding="utf-8-sig")
    stress.to_csv(OUTPUT_DIR / "realistic_execution_stress_windows.csv", index=False, encoding="utf-8-sig")
    failed.to_csv(OUTPUT_DIR / "realistic_execution_stress_failed_windows.csv", index=False, encoding="utf-8-sig")
    symbols.to_csv(OUTPUT_DIR / "realistic_execution_stress_symbol_summary.csv", index=False, encoding="utf-8-sig")
    save_nav_plot(paths)

    print("Summary:")
    print(
        summary[
            [
                "scheme",
                "annual_return",
                "max_drawdown",
                "final_nav",
                "annualized_turnover",
                "max_core_turnover",
                "max_symbol_abs_weight",
                "avg_unfilled_drift",
                "pass_20_10",
            ]
        ].to_string(index=False)
    )
    print("\nFailed windows:")
    print(failed[["scheme", "window", "annual_return", "max_drawdown"]].head(30).to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
