# ruff: noqa: E402
"""Core CTA execution constraints for the robust blend.

This is an executable-weight stress test. It replaces the aggregate core CTA
component in ``robust_50rv_blend`` with rebuilt net symbol weights for
000852 / 000905 / 159915, then applies practical constraints:

- per-symbol net exposure caps;
- no-trade band at 10% total symbol-weight drift;
- 2-day / 3-day target smoothing.

External low-correlation overlay component returns are kept unchanged. This
does not modify the research signal; it tests whether the core CTA sleeve can be
executed with less single-symbol leverage and less switching.

Run:
    uv run --no-sync python examples/signals_dev/validate_daily_cta_core_execution_constraints.py
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
from validate_daily_cta_layered_low_corr_overlay import TARGET_ANNUAL_RETURN, TARGET_MAX_DRAWDOWN
from validate_daily_cta_walk_forward import FEE_RATE, annualized_stats, summarize_yearly

ROOT = Path(__file__).resolve().parents[2]
COMPONENT_DAILY = (
    ROOT / "examples" / "results" / "daily_cta_robust_blend_component_audit" / "robust_50rv_component_daily.csv"
)
CORE_SYMBOL_WEIGHTS = (
    ROOT / "examples" / "results" / "daily_cta_core_symbol_weights" / "core_annual_state_symbol_weights.csv"
)
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_core_execution_constraints"


@dataclass(frozen=True)
class ExecutionScheme:
    """One core CTA execution constraint scheme."""

    name: str
    symbol_cap: float | None = None
    smooth_days: int | None = None
    drift_threshold: float | None = 0.10


SCHEMES = (
    ExecutionScheme("net_daily_raw", drift_threshold=None),
    ExecutionScheme("net_threshold_10pct"),
    ExecutionScheme("cap2_threshold_10pct", symbol_cap=2.0),
    ExecutionScheme("cap15_threshold_10pct", symbol_cap=1.5),
    ExecutionScheme("cap10_threshold_10pct", symbol_cap=1.0),
    ExecutionScheme("smooth2_threshold_10pct", smooth_days=2),
    ExecutionScheme("smooth3_threshold_10pct", smooth_days=3),
    ExecutionScheme("cap15_smooth2_threshold_10pct", symbol_cap=1.5, smooth_days=2),
    ExecutionScheme("cap15_smooth3_threshold_10pct", symbol_cap=1.5, smooth_days=3),
)


def load_price_returns(dates: pd.Series) -> pd.DataFrame:
    """Load close-to-close returns for the three core symbols."""
    panel = cs.load_panel()
    panel = panel[panel["dt"].isin(pd.to_datetime(dates))].sort_values("dt").reset_index(drop=True)
    returns = panel[["dt"]].copy()
    returns[list(cs.SYMBOLS)] = panel[list(cs.SYMBOLS)].pct_change(fill_method=None).fillna(0.0)
    return returns


def load_external_components() -> pd.DataFrame:
    """Load exact non-core component returns and weights."""
    data = pd.read_csv(COMPONENT_DAILY, parse_dates=["dt"])
    data["dt"] = pd.to_datetime(data["dt"]).dt.tz_localize(None)
    external = data[~data["component"].str.startswith("core_cn_cta_")].copy()
    out = external.groupby("dt", as_index=False).agg(
        external_ret=("component_ret", "sum"),
        external_weight=("component_weight", "sum"),
    )
    return out.sort_values("dt").reset_index(drop=True)


def load_core_target_weights() -> pd.DataFrame:
    """Build robust-blend core target net symbol weights from component scales."""
    components = pd.read_csv(COMPONENT_DAILY, parse_dates=["dt"])
    components["dt"] = pd.to_datetime(components["dt"]).dt.tz_localize(None)
    core_components = components[components["component"].str.startswith("core_cn_cta_")].copy()

    core_weights = pd.read_csv(CORE_SYMBOL_WEIGHTS, parse_dates=["dt"])
    core_weights["dt"] = pd.to_datetime(core_weights["dt"]).dt.tz_localize(None)
    rows = []
    for component in ("core_cn_cta_stress_balanced", "core_cn_cta_capital_efficient"):
        preset = component.replace("core_cn_cta_", "")
        comp = core_components[core_components["component"].eq(component)][["dt", "component_weight"]].copy()
        full = core_weights[core_weights["preset"].eq(preset)].copy()
        joined = comp.merge(full, on="dt", how="inner")
        scale = np.divide(
            joined["component_weight"].to_numpy(float),
            joined["rebuilt_gross_abs_weight"].to_numpy(float),
            out=np.zeros(len(joined), dtype=float),
            where=joined["rebuilt_gross_abs_weight"].to_numpy(float) > 1e-12,
        )
        part = joined[["dt"]].copy()
        for symbol in cs.SYMBOLS:
            part[symbol] = joined[symbol].to_numpy(float) * scale
        rows.append(part)
    target = pd.concat(rows, ignore_index=True).groupby("dt", as_index=False)[list(cs.SYMBOLS)].sum()
    return target.sort_values("dt").reset_index(drop=True)


def apply_target_constraints(target: pd.DataFrame, scheme: ExecutionScheme) -> pd.DataFrame:
    """Apply target smoothing and per-symbol caps."""
    out = target.copy()
    values = out[list(cs.SYMBOLS)].to_numpy(float)
    if scheme.smooth_days is not None and scheme.smooth_days > 1:
        smoothed = np.zeros_like(values)
        smoothed[0] = values[0]
        for i in range(1, len(values)):
            if out["dt"].iloc[i].year != out["dt"].iloc[i - 1].year:
                smoothed[i] = values[i]
            else:
                smoothed[i] = smoothed[i - 1] + (values[i] - smoothed[i - 1]) / scheme.smooth_days
        values = smoothed
    if scheme.symbol_cap is not None:
        values = np.clip(values, -scheme.symbol_cap, scheme.symbol_cap)
    out[list(cs.SYMBOLS)] = values
    return out


def simulate_core_execution(target: pd.DataFrame, returns: pd.DataFrame, scheme: ExecutionScheme) -> pd.DataFrame:
    """Simulate core symbol execution under one scheme."""
    constrained = apply_target_constraints(target, scheme)
    joined = constrained.merge(returns, on="dt", suffixes=("_target", "_ret"), how="inner")
    current = joined[[f"{symbol}_target" for symbol in cs.SYMBOLS]].iloc[0].to_numpy(float)
    rows = []
    target_values = joined[[f"{symbol}_target" for symbol in cs.SYMBOLS]].to_numpy(float)
    ret_values = joined[[f"{symbol}_ret" for symbol in cs.SYMBOLS]].to_numpy(float)
    dates = pd.to_datetime(joined["dt"]).reset_index(drop=True)
    for i, dt in enumerate(dates):
        target_today = target_values[i]
        symbol_ret = ret_values[i]
        year_changed = i == 0 or dt.year != joined["dt"].iloc[i - 1].year
        if year_changed:
            turnover = float(np.abs(target_today - current).sum())
            current = target_today.copy()
            rebalance = True
        else:
            drift = float(np.abs(target_today - current).sum())
            rebalance = scheme.drift_threshold is None or drift >= scheme.drift_threshold
            turnover = drift if rebalance else 0.0
            if rebalance:
                current = target_today.copy()
        core_ret = float(np.dot(current, symbol_ret) - turnover * FEE_RATE)
        rows.append(
            {
                "dt": dt,
                "scheme": scheme.name,
                "core_ret": core_ret,
                "core_turnover": turnover,
                "core_weight": float(np.abs(current).sum()),
                "rebalance": rebalance,
                **{symbol: current[idx] for idx, symbol in enumerate(cs.SYMBOLS)},
            }
        )
    return pd.DataFrame(rows)


def build_portfolio_paths() -> pd.DataFrame:
    """Build portfolio paths for all execution schemes."""
    target = load_core_target_weights()
    returns = load_price_returns(target["dt"])
    external = load_external_components()
    paths = []
    for scheme in SCHEMES:
        core = simulate_core_execution(target, returns, scheme)
        joined = core.merge(external, on="dt", how="left")
        joined[["external_ret", "external_weight"]] = joined[["external_ret", "external_weight"]].fillna(0.0)
        joined["ret"] = joined["core_ret"] + joined["external_ret"]
        joined["weight"] = joined["core_weight"] + joined["external_weight"]
        joined["turnover"] = joined["core_turnover"]
        joined["nav"] = (1 + joined["ret"]).cumprod()
        joined["drawdown"] = joined["nav"] / joined["nav"].cummax() - 1
        paths.append(joined)
    return pd.concat(paths, ignore_index=True)


def summarize_paths(paths: pd.DataFrame) -> pd.DataFrame:
    """Summarize performance and execution diagnostics."""
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
                "core_turnover_sum": float(group["core_turnover"].sum()),
                "core_annualized_turnover": float(group["core_turnover"].sum() / years) if years > 0 else float("nan"),
                "max_core_weight": float(group["core_weight"].max()),
                "max_symbol_abs_weight": float(group[list(cs.SYMBOLS)].abs().max().max()),
                "loss_years": int((yearly["return"] < 0).sum()),
                "pass_20_10": bool(
                    stats["annual_return"] >= TARGET_ANNUAL_RETURN
                    and stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
                ),
            }
        )
    return pd.DataFrame(rows)


def symbol_execution_summary(paths: pd.DataFrame) -> pd.DataFrame:
    """Summarize symbol-level exposure under each scheme."""
    rows = []
    for scheme, group in paths.groupby("scheme", sort=False):
        for symbol in cs.SYMBOLS:
            weight = group[symbol].astype(float)
            turnover = weight.diff().abs().fillna(weight.abs())
            rows.append(
                {
                    "scheme": scheme,
                    "symbol": symbol,
                    "avg_abs_weight": float(weight.abs().mean()),
                    "weight_p95": float(weight.abs().quantile(0.95)),
                    "max_abs_weight": float(weight.abs().max()),
                    "turnover_sum": float(turnover.sum()),
                    "max_daily_turnover": float(turnover.max()),
                    "days_abs_weight_gt_1": int((weight.abs() > 1.0).sum()),
                    "days_abs_weight_gt_1_5": int((weight.abs() > 1.5).sum()),
                    "days_abs_weight_gt_2": int((weight.abs() > 2.0).sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Run core execution constraint tests."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = build_portfolio_paths()
    summary = summarize_paths(paths)
    yearly = pd.concat(
        [
            summarize_yearly(group[["dt", "ret", "weight"]]).assign(scheme=scheme)
            for scheme, group in paths.groupby("scheme", sort=False)
        ],
        ignore_index=True,
    )
    symbol_summary = symbol_execution_summary(paths)
    top_turnover = paths.sort_values("core_turnover", ascending=False).head(60)

    paths.to_csv(OUTPUT_DIR / "core_execution_constraint_daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "core_execution_constraint_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUTPUT_DIR / "core_execution_constraint_yearly.csv", index=False, encoding="utf-8-sig")
    symbol_summary.to_csv(OUTPUT_DIR / "core_execution_constraint_symbol_summary.csv", index=False, encoding="utf-8-sig")
    top_turnover.to_csv(OUTPUT_DIR / "core_execution_constraint_top_turnover.csv", index=False, encoding="utf-8-sig")

    print("Summary:")
    print(
        summary[
            [
                "scheme",
                "annual_return",
                "max_drawdown",
                "final_nav",
                "rebalance_days",
                "core_turnover_sum",
                "max_symbol_abs_weight",
                "pass_20_10",
            ]
        ].to_string(index=False)
    )
    print("\nSymbol summary:")
    print(symbol_summary.to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
