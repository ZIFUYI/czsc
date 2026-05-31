# ruff: noqa: E402
"""Portfolio-level audit for the cap-1.5 executable robust blend.

This follow-up compares the original gross-research execution path with the
net-symbol executable paths created from rebuilt 000852 / 000905 / 159915 core
CTA weights. The focus is whether the practical core constraint
``cap15_threshold_10pct`` remains acceptable after yearly, pressure-window and
incremental-cost checks.

Run:
    uv run --no-sync python examples/signals_dev/validate_daily_cta_cap15_blend_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_daily_cta_cross_section_walk_forward as cs
from validate_daily_cta_layered_low_corr_overlay import (
    TARGET_ANNUAL_RETURN,
    TARGET_MAX_DRAWDOWN,
    evaluate_window_stress,
)
from validate_daily_cta_walk_forward import annualized_stats, summarize_yearly

ROOT = Path(__file__).resolve().parents[2]
COMPRESSION_DAILY = (
    ROOT
    / "examples"
    / "results"
    / "daily_cta_robust_blend_rebalance_compression"
    / "robust_50rv_rebalance_compression_daily.csv"
)
CORE_CONSTRAINT_DAILY = (
    ROOT
    / "examples"
    / "results"
    / "daily_cta_core_execution_constraints"
    / "core_execution_constraint_daily.csv"
)
CORE_CONSTRAINT_SYMBOL = (
    ROOT
    / "examples"
    / "results"
    / "daily_cta_core_execution_constraints"
    / "core_execution_constraint_symbol_summary.csv"
)
COMPONENT_DAILY = (
    ROOT
    / "examples"
    / "results"
    / "daily_cta_robust_blend_component_audit"
    / "robust_50rv_component_daily.csv"
)
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_cap15_blend_audit"

GROSS_SCHEMES = {
    "daily_raw": "gross_daily_raw",
    "threshold_10pct": "gross_threshold_10pct",
}
NET_SCHEMES = (
    "net_threshold_10pct",
    "cap15_threshold_10pct",
    "cap10_threshold_10pct",
)
EXTRA_COST_RATES = (0.0, 0.0002, 0.0005, 0.0010, 0.0020)


def load_external_component_turnover() -> pd.DataFrame:
    """Calculate daily turnover from non-core component weights."""
    data = pd.read_csv(COMPONENT_DAILY, parse_dates=["dt"])
    data["dt"] = pd.to_datetime(data["dt"]).dt.tz_localize(None)
    external = data[~data["component"].str.startswith("core_cn_cta_")].copy()
    wide = external.pivot_table(index="dt", columns="component", values="component_weight", aggfunc="sum").fillna(0.0)
    turnover = wide.diff().abs()
    turnover.iloc[0] = wide.iloc[0].abs()
    out = turnover.sum(axis=1).rename("external_turnover").reset_index()
    return out.sort_values("dt").reset_index(drop=True)


def load_gross_paths() -> pd.DataFrame:
    """Load original gross research paths from the rebalance compression test."""
    daily = pd.read_csv(COMPRESSION_DAILY, parse_dates=["dt"])
    daily["dt"] = pd.to_datetime(daily["dt"]).dt.tz_localize(None)
    daily = daily[daily["scheme"].isin(GROSS_SCHEMES)].copy()
    daily["audit_scheme"] = daily["scheme"].map(GROSS_SCHEMES)
    daily["execution_style"] = "gross_component"
    daily["core_turnover"] = float("nan")
    daily["external_turnover"] = float("nan")
    daily["cost_turnover"] = daily["turnover"].astype(float)
    return daily[
        [
            "dt",
            "audit_scheme",
            "execution_style",
            "ret",
            "weight",
            "turnover",
            "cost_turnover",
            "core_turnover",
            "external_turnover",
            "rebalance",
            "nav",
            "drawdown",
        ]
    ]


def load_net_paths() -> pd.DataFrame:
    """Load net-symbol executable paths and add external turnover proxy."""
    daily = pd.read_csv(CORE_CONSTRAINT_DAILY, parse_dates=["dt"])
    daily["dt"] = pd.to_datetime(daily["dt"]).dt.tz_localize(None)
    daily = daily[daily["scheme"].isin(NET_SCHEMES)].copy()
    external_turnover = load_external_component_turnover()
    daily = daily.merge(external_turnover, on="dt", how="left")
    daily["external_turnover"] = daily["external_turnover"].fillna(0.0)
    daily["audit_scheme"] = daily["scheme"]
    daily["execution_style"] = "net_symbol_core"
    daily["turnover"] = daily["core_turnover"].astype(float) + daily["external_turnover"].astype(float)
    daily["cost_turnover"] = daily["turnover"]
    return daily[
        [
            "dt",
            "audit_scheme",
            "execution_style",
            "ret",
            "weight",
            "turnover",
            "cost_turnover",
            "core_turnover",
            "external_turnover",
            "rebalance",
            "nav",
            "drawdown",
            *cs.SYMBOLS,
        ]
    ]


def load_audit_paths() -> pd.DataFrame:
    """Load all paths under a consistent audit schema."""
    paths = pd.concat([load_gross_paths(), load_net_paths()], ignore_index=True, sort=False)
    paths = paths.sort_values(["audit_scheme", "dt"]).reset_index(drop=True)
    paths["nav"] = paths.groupby("audit_scheme", sort=False)["ret"].transform(lambda x: (1 + x).cumprod())
    paths["drawdown"] = paths.groupby("audit_scheme", sort=False)["nav"].transform(lambda x: x / x.cummax() - 1)
    return paths


def summarize_paths(paths: pd.DataFrame) -> pd.DataFrame:
    """Summarize no-extra-cost performance and execution diagnostics."""
    rows = []
    for scheme, group in paths.groupby("audit_scheme", sort=False):
        stats = annualized_stats(group["dt"].reset_index(drop=True), group["ret"].to_numpy(), group["weight"].to_numpy())
        years = (group["dt"].iloc[-1] - group["dt"].iloc[0]).days / 365.25
        rows.append(
            {
                "scheme": scheme,
                "execution_style": group["execution_style"].iloc[0],
                **stats,
                "rebalance_days": int(group["rebalance"].sum()),
                "turnover_sum": float(group["turnover"].sum()),
                "annualized_turnover": float(group["turnover"].sum() / years) if years > 0 else float("nan"),
                "avg_cost_turnover": float(group["cost_turnover"].mean()),
                "max_daily_turnover": float(group["cost_turnover"].max()),
                "pass_20_10": bool(
                    stats["annual_return"] >= TARGET_ANNUAL_RETURN
                    and stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
                ),
            }
        )
    return pd.DataFrame(rows)


def cost_sensitivity(paths: pd.DataFrame) -> pd.DataFrame:
    """Apply extra transaction costs using the path-specific turnover proxy."""
    rows = []
    for scheme, group in paths.groupby("audit_scheme", sort=False):
        for cost_rate in EXTRA_COST_RATES:
            adjusted_ret = group["ret"] - group["cost_turnover"] * cost_rate
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


def yearly_stats(paths: pd.DataFrame) -> pd.DataFrame:
    """Calculate yearly stats for each audited path."""
    rows = []
    for scheme, group in paths.groupby("audit_scheme", sort=False):
        yearly = summarize_yearly(group[["dt", "ret", "weight"]])
        yearly["scheme"] = scheme
        rows.append(yearly)
    return pd.concat(rows, ignore_index=True)


def pressure_windows(paths: pd.DataFrame) -> pd.DataFrame:
    """Calculate pressure-window performance for each path."""
    rows = []
    for scheme, group in paths.groupby("audit_scheme", sort=False):
        stress = evaluate_window_stress(group[["dt", "ret", "weight"]].copy(), scheme)
        stress["scheme"] = scheme
        rows.append(stress)
    return pd.concat(rows, ignore_index=True)


def save_nav_plot(paths: pd.DataFrame) -> None:
    """Save a comparison net-value plot."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for scheme, group in paths.groupby("audit_scheme", sort=False):
        fig.add_trace(go.Scatter(x=group["dt"], y=group["nav"], mode="lines", name=scheme))
    fig.update_layout(
        title="Robust blend execution audit: gross vs net core cap",
        xaxis_title="date",
        yaxis_title="net value",
        template="plotly_white",
    )
    fig.write_html(OUTPUT_DIR / "cap15_blend_nav.html", include_plotlyjs="cdn")


def save_symbol_summary() -> None:
    """Copy focused symbol-level summaries for executable core paths."""
    symbol = pd.read_csv(CORE_CONSTRAINT_SYMBOL)
    symbol = symbol[symbol["scheme"].isin(NET_SCHEMES)].copy()
    symbol.to_csv(OUTPUT_DIR / "cap15_blend_symbol_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    """Run the cap-1.5 robust-blend audit."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = load_audit_paths()
    summary = summarize_paths(paths)
    costs = cost_sensitivity(paths)
    yearly = yearly_stats(paths)
    stress = pressure_windows(paths)
    failed = stress[~stress["pass_20_10"]].sort_values(["scheme", "annual_return"])

    paths.to_csv(OUTPUT_DIR / "cap15_blend_daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "cap15_blend_summary.csv", index=False, encoding="utf-8-sig")
    costs.to_csv(OUTPUT_DIR / "cap15_blend_cost_sensitivity.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUTPUT_DIR / "cap15_blend_yearly.csv", index=False, encoding="utf-8-sig")
    stress.to_csv(OUTPUT_DIR / "cap15_blend_window_stress.csv", index=False, encoding="utf-8-sig")
    failed.to_csv(OUTPUT_DIR / "cap15_blend_failed_windows.csv", index=False, encoding="utf-8-sig")
    save_nav_plot(paths)
    save_symbol_summary()

    print("Summary:")
    print(
        summary[
            [
                "scheme",
                "execution_style",
                "annual_return",
                "max_drawdown",
                "final_nav",
                "annualized_turnover",
                "max_daily_turnover",
                "pass_20_10",
            ]
        ].to_string(index=False)
    )
    print("\nCost sensitivity:")
    print(
        costs[costs["extra_cost_bps"].isin([0.0, 2.0, 5.0, 10.0])][
            ["scheme", "extra_cost_bps", "annual_return", "max_drawdown", "final_nav", "pass_20_10"]
        ].to_string(index=False)
    )
    print("\nWorst failed windows:")
    print(failed[["scheme", "window", "annual_return", "max_drawdown"]].head(20).to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
