# ruff: noqa: E402
"""Bridge audit between gross component research and net-symbol execution.

The previous audits showed that the executable net-symbol core path is much
stronger than the gross component research path. This script decomposes that
gap into explicit transitions:

1. gross component threshold execution for all components;
2. keep gross core threshold execution, but use raw external overlay returns;
3. replace gross core with rebuilt net-symbol core execution;
4. add the 1.5x per-symbol cap on the net-symbol core execution.

Run:
    uv run --no-sync python examples/signals_dev/validate_daily_cta_gross_net_bridge_audit.py
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
from validate_daily_cta_core_execution_constraints import (
    ExecutionScheme,
    load_core_target_weights,
    load_price_returns,
    simulate_core_execution,
)
from validate_daily_cta_layered_low_corr_overlay import TARGET_ANNUAL_RETURN, TARGET_MAX_DRAWDOWN
from validate_daily_cta_robust_blend_rebalance_compression import (
    component_unit_returns,
    load_component_panel,
)
from validate_daily_cta_walk_forward import annualized_stats, summarize_yearly

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_gross_net_bridge_audit"
EXTRA_COST_RATES = (0.0, 0.0002, 0.0005, 0.0010, 0.0020)
THRESHOLD = 0.10


@dataclass(frozen=True)
class BridgeLeg:
    """One pairwise bridge transition."""

    from_scheme: str
    to_scheme: str
    label: str


BRIDGE_LEGS = (
    BridgeLeg("gross_threshold_all_components", "gross_core_threshold_external_raw", "external execution effect"),
    BridgeLeg("gross_core_threshold_external_raw", "net_core_threshold_external_raw", "core gross-to-net effect"),
    BridgeLeg("net_core_threshold_external_raw", "cap15_core_threshold_external_raw", "1.5x symbol cap effect"),
    BridgeLeg("gross_threshold_all_components", "cap15_core_threshold_external_raw", "total bridge effect"),
)


def is_core_component(column: str) -> bool:
    """Return whether a component id belongs to the core CTA sleeve."""
    return "core_cn_cta_" in column


def simulate_gross_component_threshold(
    desired_weights: pd.DataFrame, unit_returns: pd.DataFrame
) -> pd.DataFrame:
    """Simulate the 10% component threshold rule and split core/external legs."""
    dates = desired_weights.index
    desired = desired_weights.to_numpy(float)
    unit_ret = unit_returns.to_numpy(float)
    core_mask = np.array([is_core_component(col) for col in desired_weights.columns], dtype=bool)
    current = desired[0].copy()
    rows = []

    for i, dt in enumerate(dates):
        year_changed = i == 0 or dates[i].year != dates[i - 1].year
        turnover_vec = np.zeros(current.shape, dtype=float)
        rebalance = False
        if year_changed:
            turnover_vec += np.abs(desired[i] - current)
            current = desired[i].copy()
            rebalance = True

        component_ret = current * unit_ret[i]
        desired_today = desired[i]
        if not year_changed:
            drift = float(np.abs(desired_today - current).sum())
            rebalance = drift >= THRESHOLD
            if rebalance:
                turnover_vec += np.abs(desired_today - current)
                current = desired_today.copy()

        rows.append(
            {
                "dt": dt,
                "gross_core_ret": float(component_ret[core_mask].sum()),
                "gross_external_threshold_ret": float(component_ret[~core_mask].sum()),
                "gross_core_weight": float(np.abs(current[core_mask]).sum()),
                "gross_external_threshold_weight": float(np.abs(current[~core_mask]).sum()),
                "gross_core_turnover": float(turnover_vec[core_mask].sum()),
                "gross_external_threshold_turnover": float(turnover_vec[~core_mask].sum()),
                "gross_rebalance": bool(rebalance),
            }
        )
    return pd.DataFrame(rows)


def raw_external_panel(desired_weights: pd.DataFrame, component_returns: pd.DataFrame) -> pd.DataFrame:
    """Build raw external overlay return, weight and turnover series."""
    external_cols = [col for col in desired_weights.columns if not is_core_component(col)]
    external_weights = desired_weights[external_cols].copy()
    external_returns = component_returns[external_cols].copy()
    turnover = external_weights.diff().abs()
    turnover.iloc[0] = external_weights.iloc[0].abs()
    return pd.DataFrame(
        {
            "dt": desired_weights.index,
            "external_raw_ret": external_returns.sum(axis=1).to_numpy(float),
            "external_raw_weight": external_weights.abs().sum(axis=1).to_numpy(float),
            "external_raw_turnover": turnover.sum(axis=1).to_numpy(float),
        }
    )


def net_core_panel() -> pd.DataFrame:
    """Simulate uncapped and capped net-symbol core execution."""
    target = load_core_target_weights()
    returns = load_price_returns(target["dt"])
    schemes = (
        ExecutionScheme("net_core_threshold", drift_threshold=THRESHOLD),
        ExecutionScheme("cap15_core_threshold", symbol_cap=1.5, drift_threshold=THRESHOLD),
    )
    frames = []
    for scheme in schemes:
        core = simulate_core_execution(target, returns, scheme)
        frames.append(
            core.rename(
                columns={
                    "scheme": "core_scheme",
                    "core_ret": f"{scheme.name}_ret",
                    "core_weight": f"{scheme.name}_weight",
                    "core_turnover": f"{scheme.name}_turnover",
                    "rebalance": f"{scheme.name}_rebalance",
                }
            )[
                [
                    "dt",
                    f"{scheme.name}_ret",
                    f"{scheme.name}_weight",
                    f"{scheme.name}_turnover",
                    f"{scheme.name}_rebalance",
                    *cs.SYMBOLS,
                ]
            ]
        )
    out = frames[0].merge(frames[1], on="dt", suffixes=("_net", "_cap15"), how="inner")
    return out


def build_bridge_daily() -> pd.DataFrame:
    """Build daily bridge paths from gross research to net execution."""
    desired_weights, component_returns = load_component_panel()
    unit_returns = component_unit_returns(desired_weights, component_returns)
    gross = simulate_gross_component_threshold(desired_weights, unit_returns)
    external = raw_external_panel(desired_weights, component_returns)
    net_core = net_core_panel()

    base = gross.merge(external, on="dt", how="inner").merge(net_core, on="dt", how="inner")
    rows = []
    specs = (
        {
            "scheme": "gross_threshold_all_components",
            "core_ret": base["gross_core_ret"],
            "external_ret": base["gross_external_threshold_ret"],
            "core_weight": base["gross_core_weight"],
            "external_weight": base["gross_external_threshold_weight"],
            "core_turnover": base["gross_core_turnover"],
            "external_turnover": base["gross_external_threshold_turnover"],
            "rebalance": base["gross_rebalance"],
        },
        {
            "scheme": "gross_core_threshold_external_raw",
            "core_ret": base["gross_core_ret"],
            "external_ret": base["external_raw_ret"],
            "core_weight": base["gross_core_weight"],
            "external_weight": base["external_raw_weight"],
            "core_turnover": base["gross_core_turnover"],
            "external_turnover": base["external_raw_turnover"],
            "rebalance": base["gross_rebalance"] | (base["external_raw_turnover"] > 0),
        },
        {
            "scheme": "net_core_threshold_external_raw",
            "core_ret": base["net_core_threshold_ret"],
            "external_ret": base["external_raw_ret"],
            "core_weight": base["net_core_threshold_weight"],
            "external_weight": base["external_raw_weight"],
            "core_turnover": base["net_core_threshold_turnover"],
            "external_turnover": base["external_raw_turnover"],
            "rebalance": base["net_core_threshold_rebalance"] | (base["external_raw_turnover"] > 0),
        },
        {
            "scheme": "cap15_core_threshold_external_raw",
            "core_ret": base["cap15_core_threshold_ret"],
            "external_ret": base["external_raw_ret"],
            "core_weight": base["cap15_core_threshold_weight"],
            "external_weight": base["external_raw_weight"],
            "core_turnover": base["cap15_core_threshold_turnover"],
            "external_turnover": base["external_raw_turnover"],
            "rebalance": base["cap15_core_threshold_rebalance"] | (base["external_raw_turnover"] > 0),
        },
    )
    for spec in specs:
        frame = pd.DataFrame(
            {
                "dt": base["dt"],
                "scheme": spec["scheme"],
                "core_ret": spec["core_ret"],
                "external_ret": spec["external_ret"],
                "ret": spec["core_ret"] + spec["external_ret"],
                "core_weight": spec["core_weight"],
                "external_weight": spec["external_weight"],
                "weight": spec["core_weight"] + spec["external_weight"],
                "core_turnover": spec["core_turnover"],
                "external_turnover": spec["external_turnover"],
                "turnover": spec["core_turnover"] + spec["external_turnover"],
                "rebalance": spec["rebalance"],
            }
        )
        rows.append(frame)
    daily = pd.concat(rows, ignore_index=True)
    daily["nav"] = daily.groupby("scheme", sort=False)["ret"].transform(lambda x: (1 + x).cumprod())
    daily["drawdown"] = daily.groupby("scheme", sort=False)["nav"].transform(lambda x: x / x.cummax() - 1)
    return daily


def summarize_paths(daily: pd.DataFrame) -> pd.DataFrame:
    """Summarize bridge paths."""
    rows = []
    for scheme, group in daily.groupby("scheme", sort=False):
        stats = annualized_stats(group["dt"].reset_index(drop=True), group["ret"].to_numpy(), group["weight"].to_numpy())
        years = (group["dt"].iloc[-1] - group["dt"].iloc[0]).days / 365.25
        rows.append(
            {
                "scheme": scheme,
                **stats,
                "core_ret_sum": float(group["core_ret"].sum()),
                "external_ret_sum": float(group["external_ret"].sum()),
                "turnover_sum": float(group["turnover"].sum()),
                "annualized_turnover": float(group["turnover"].sum() / years) if years > 0 else float("nan"),
                "rebalance_days": int(group["rebalance"].sum()),
                "pass_20_10": bool(
                    stats["annual_return"] >= TARGET_ANNUAL_RETURN
                    and stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
                ),
            }
        )
    return pd.DataFrame(rows)


def cost_sensitivity(daily: pd.DataFrame) -> pd.DataFrame:
    """Apply extra transaction costs to each bridge path."""
    rows = []
    for scheme, group in daily.groupby("scheme", sort=False):
        for cost_rate in EXTRA_COST_RATES:
            adjusted_ret = group["ret"] - group["turnover"] * cost_rate
            stats = annualized_stats(group["dt"].reset_index(drop=True), adjusted_ret.to_numpy(), group["weight"].to_numpy())
            rows.append(
                {
                    "scheme": scheme,
                    "extra_cost_bps": cost_rate * 10000,
                    **stats,
                    "pass_20_10": bool(
                        stats["annual_return"] >= TARGET_ANNUAL_RETURN
                        and stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
                    ),
                }
            )
    return pd.DataFrame(rows)


def yearly_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate yearly bridge stats."""
    rows = []
    for scheme, group in daily.groupby("scheme", sort=False):
        yearly = summarize_yearly(group[["dt", "ret", "weight"]])
        yearly["scheme"] = scheme
        rows.append(yearly)
    return pd.concat(rows, ignore_index=True)


def transition_attribution(daily: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Attribute pairwise bridge transitions by daily return differences."""
    summary_idx = summary.set_index("scheme")
    rows = []
    pivot = daily.pivot(index="dt", columns="scheme", values="ret")
    for leg in BRIDGE_LEGS:
        diff = pivot[leg.to_scheme] - pivot[leg.from_scheme]
        from_stats = summary_idx.loc[leg.from_scheme]
        to_stats = summary_idx.loc[leg.to_scheme]
        rows.append(
            {
                "label": leg.label,
                "from_scheme": leg.from_scheme,
                "to_scheme": leg.to_scheme,
                "annual_return_delta": float(to_stats["annual_return"] - from_stats["annual_return"]),
                "max_drawdown_delta": float(to_stats["max_drawdown"] - from_stats["max_drawdown"]),
                "final_nav_delta": float(to_stats["final_nav"] - from_stats["final_nav"]),
                "daily_ret_diff_sum": float(diff.sum()),
                "daily_ret_diff_mean": float(diff.mean()),
                "daily_ret_diff_abs_sum": float(diff.abs().sum()),
                "positive_diff_days": int((diff > 0).sum()),
                "negative_diff_days": int((diff < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def yearly_transition_attribution(daily: pd.DataFrame) -> pd.DataFrame:
    """Attribute pairwise bridge transitions by calendar year."""
    pivot = daily.pivot(index="dt", columns="scheme", values="ret")
    rows = []
    for year, group in pivot.groupby(pivot.index.year):
        for leg in BRIDGE_LEGS:
            from_ret = group[leg.from_scheme]
            to_ret = group[leg.to_scheme]
            rows.append(
                {
                    "year": int(year),
                    "label": leg.label,
                    "from_scheme": leg.from_scheme,
                    "to_scheme": leg.to_scheme,
                    "from_return": float((1 + from_ret).prod() - 1),
                    "to_return": float((1 + to_ret).prod() - 1),
                    "return_delta": float((1 + to_ret).prod() - (1 + from_ret).prod()),
                    "daily_ret_diff_sum": float((to_ret - from_ret).sum()),
                }
            )
    return pd.DataFrame(rows)


def save_nav_plot(daily: pd.DataFrame) -> None:
    """Save net-value plot for bridge paths."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for scheme, group in daily.groupby("scheme", sort=False):
        fig.add_trace(go.Scatter(x=group["dt"], y=group["nav"], mode="lines", name=scheme))
    fig.update_layout(
        title="Gross-to-net bridge audit",
        xaxis_title="date",
        yaxis_title="net value",
        template="plotly_white",
    )
    fig.write_html(OUTPUT_DIR / "gross_net_bridge_nav.html", include_plotlyjs="cdn")


def main() -> None:
    """Run gross-to-net bridge audit."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = build_bridge_daily()
    summary = summarize_paths(daily)
    costs = cost_sensitivity(daily)
    yearly = yearly_stats(daily)
    attribution = transition_attribution(daily, summary)
    yearly_attribution = yearly_transition_attribution(daily)

    daily.to_csv(OUTPUT_DIR / "gross_net_bridge_daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "gross_net_bridge_summary.csv", index=False, encoding="utf-8-sig")
    costs.to_csv(OUTPUT_DIR / "gross_net_bridge_cost_sensitivity.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUTPUT_DIR / "gross_net_bridge_yearly.csv", index=False, encoding="utf-8-sig")
    attribution.to_csv(OUTPUT_DIR / "gross_net_bridge_attribution.csv", index=False, encoding="utf-8-sig")
    yearly_attribution.to_csv(OUTPUT_DIR / "gross_net_bridge_yearly_attribution.csv", index=False, encoding="utf-8-sig")
    save_nav_plot(daily)

    print("Bridge summary:")
    print(
        summary[
            [
                "scheme",
                "annual_return",
                "max_drawdown",
                "final_nav",
                "core_ret_sum",
                "external_ret_sum",
                "annualized_turnover",
                "pass_20_10",
            ]
        ].to_string(index=False)
    )
    print("\nTransition attribution:")
    print(
        attribution[
            [
                "label",
                "annual_return_delta",
                "max_drawdown_delta",
                "final_nav_delta",
                "daily_ret_diff_sum",
            ]
        ].to_string(index=False)
    )
    print("\nYearly attribution:")
    print(
        yearly_attribution[
            yearly_attribution["label"].isin(["core gross-to-net effect", "1.5x symbol cap effect"])
        ][["year", "label", "from_return", "to_return", "return_delta"]].to_string(index=False)
    )
    print("\nCost sensitivity:")
    print(
        costs[costs["extra_cost_bps"].isin([0.0, 5.0, 10.0, 20.0])][
            ["scheme", "extra_cost_bps", "annual_return", "max_drawdown", "final_nav", "pass_20_10"]
        ].to_string(index=False)
    )
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
