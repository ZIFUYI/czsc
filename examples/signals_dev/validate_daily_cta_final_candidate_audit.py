# ruff: noqa: E402, I001
"""Final candidate audit for the optimized daily CTA research track.

This script compares the current viable candidates under one consistent
reporting schema: annualized return, drawdown, weak-window diagnostics,
yearly performance, pressure-window failures and exposure / turnover proxies.

Run:
    uv run --no-sync python examples/signals_dev/validate_daily_cta_final_candidate_audit.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_daily_cta_layered_low_corr_overlay import (  # noqa: E402
    TARGET_ANNUAL_RETURN,
    TARGET_MAX_DRAWDOWN,
    evaluate_window_stress,
)
from validate_daily_cta_walk_forward import annualized_stats, summarize_yearly  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_final_candidate_audit"
ANNUAL_STATE_DAILY = (
    ROOT
    / "examples"
    / "results"
    / "daily_cta_cross_section_walk_forward"
    / "annual_state_scale"
    / "annual_state_scale_daily.csv"
)
BOND_META_DAILY = (
    ROOT / "examples" / "results" / "daily_cta_layered_overlay_meta_ensemble" / "meta_ensemble_daily_top12.csv"
)
RV_META_DAILY = (
    ROOT / "examples" / "results" / "daily_cta_relative_value_meta_ensemble" / "relative_value_meta_daily_top12.csv"
)
BLEND_DAILY = ROOT / "examples" / "results" / "daily_cta_meta_blend" / "meta_blend_daily_top12.csv"


@dataclass(frozen=True)
class CandidateSpec:
    """One audited candidate path."""

    short_name: str
    source_file: Path
    label_column: str
    label_value: str
    family: str
    implementation_note: str


CANDIDATES = (
    CandidateSpec(
        "annual_state_capital",
        ANNUAL_STATE_DAILY,
        "preset",
        "capital_efficient",
        "baseline",
        "三标的横截面 CTA + 年度训练状态缩放，结构最简单。",
    ),
    CandidateSpec(
        "annual_state_stress",
        ANNUAL_STATE_DAILY,
        "preset",
        "stress_balanced",
        "baseline",
        "三标的横截面 CTA + 年度训练状态缩放，压力窗口略优。",
    ),
    CandidateSpec(
        "highest_return_bond_inflation",
        BOND_META_DAILY,
        "meta_label",
        "stress_balanced_inflation_conservative_min2_rolling3_top3_rank_ensemble",
        "highest_return",
        "当前收益最高候选；依赖 bond_plus / gold / commodity 分层 meta。",
    ),
    CandidateSpec(
        "low_drawdown_bond",
        BOND_META_DAILY,
        "meta_label",
        "capital_efficient_bond_conservative_min2_rolling3_top5_equal_ensemble",
        "low_drawdown",
        "bond_plus 方向中回撤较低的高收益候选。",
    ),
    CandidateSpec(
        "relative_value_only",
        RV_META_DAILY,
        "meta_label",
        "stress_balanced_rv_drop25_long_min1_rolling3_top5_equal_ensemble",
        "relative_value",
        "gold vs commodity long-only 相对价值 meta；年份结构更均衡。",
    ),
    CandidateSpec(
        "robust_50rv_blend",
        BLEND_DAILY,
        "label",
        (
            "blend_RV50_stress_balanced_inflation_conservative_min2_rolling3_top3_rank_ensemble__"
            "capital_efficient_rv_drop25_long_min1_rolling3_top5_equal_ensemble"
        ),
        "blend",
        "50% bond/inflation top3 rank + 50% capital_efficient RV top5 equal；当前压力窗口最高。",
    ),
    CandidateSpec(
        "return_35rv_blend",
        BLEND_DAILY,
        "label",
        (
            "blend_RV35_stress_balanced_inflation_conservative_min2_rolling3_top3_rank_ensemble__"
            "stress_balanced_rv_drop25_long_min1_rolling3_top5_equal_ensemble"
        ),
        "blend",
        "35% RV blend；保留 30%+ 年化，同时略改善弱窗口。",
    ),
)


def load_candidate_daily(spec: CandidateSpec) -> pd.DataFrame:
    """Load and normalize one candidate daily path."""
    daily = pd.read_csv(spec.source_file, parse_dates=["dt"])
    daily["dt"] = pd.to_datetime(daily["dt"]).dt.tz_localize(None)
    daily = daily[daily[spec.label_column].eq(spec.label_value)].copy()
    if daily.empty:
        raise ValueError(f"No rows for {spec.short_name}: {spec.label_column}={spec.label_value}")
    daily = daily[["dt", "ret", "weight", "test_year"]].sort_values("dt").reset_index(drop=True)
    daily["label"] = spec.short_name
    daily["nav"] = (1 + daily["ret"]).cumprod()
    daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1
    return daily


def annual_from_yearly(yearly: pd.DataFrame, years: list[int]) -> tuple[float, float]:
    """Calculate annualized return and min drawdown for selected years."""
    part = yearly[yearly["year"].isin(years)]
    return float((1 + part["return"]).prod() ** (1 / len(part)) - 1), float(part["max_drawdown"].min())


def annual_drop_year(yearly: pd.DataFrame, drop_year: int) -> tuple[float, float]:
    """Calculate annualized return and min drawdown after dropping one year."""
    part = yearly[yearly["year"] != drop_year]
    return float((1 + part["return"]).prod() ** (1 / len(part)) - 1), float(part["max_drawdown"].min())


def summarize_candidate(daily: pd.DataFrame, spec: CandidateSpec) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Build summary, yearly and stress outputs for one candidate."""
    stats = annualized_stats(daily["dt"].reset_index(drop=True), daily["ret"].to_numpy(), daily["weight"].to_numpy())
    yearly = summarize_yearly(daily[["dt", "ret", "weight"]].copy())
    annual_2022_2023, mdd_2022_2023 = annual_from_yearly(yearly, [2022, 2023])
    annual_drop_2025, mdd_drop_2025 = annual_drop_year(yearly, 2025)
    pre = daily[daily["dt"].dt.year <= 2025]
    pre_stats = annualized_stats(pre["dt"].reset_index(drop=True), pre["ret"].to_numpy(), pre["weight"].to_numpy())
    stress = evaluate_window_stress(daily.copy(), spec.short_name)
    stress_agg = {
        "stress_windows": int(len(stress)),
        "stress_pass_count": int(stress["pass_20_10"].sum()),
        "stress_pass_rate": float(stress["pass_20_10"].mean()),
        "stress_annual_median": float(stress["annual_return"].median()),
        "stress_mdd_median": float(stress["max_drawdown"].median()),
    }
    summary = {
        **stats,
        **stress_agg,
        "short_name": spec.short_name,
        "family": spec.family,
        "source_label": spec.label_value,
        "implementation_note": spec.implementation_note,
        "pre2026_annual_return": pre_stats["annual_return"],
        "pre2026_max_drawdown": pre_stats["max_drawdown"],
        "annual_2022_2023": annual_2022_2023,
        "mdd_2022_2023": mdd_2022_2023,
        "annual_drop_2025": annual_drop_2025,
        "mdd_drop_2025": mdd_drop_2025,
        "loss_years": int((yearly["return"] < 0).sum()),
        "weight_p95": float(daily["weight"].abs().quantile(0.95)),
        "turnover_proxy": float(daily["weight"].abs().diff().abs().fillna(daily["weight"].abs()).sum()),
        "pass_full_and_pre2026": bool(
            stats["annual_return"] >= TARGET_ANNUAL_RETURN
            and stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
            and pre_stats["annual_return"] >= TARGET_ANNUAL_RETURN
            and pre_stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
        ),
    }
    yearly["short_name"] = spec.short_name
    stress["short_name"] = spec.short_name
    return summary, yearly, stress


def save_nav_plot(daily: pd.DataFrame) -> None:
    """Save a net-value plot for audited candidates."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for label, group in daily.groupby("label", sort=False):
        fig.add_trace(go.Scatter(x=group["dt"], y=group["nav"], mode="lines", name=label))
    fig.update_layout(
        title="Daily CTA final candidate audit",
        xaxis_title="date",
        yaxis_title="net value",
        template="plotly_white",
    )
    fig.write_html(OUTPUT_DIR / "final_candidate_nav.html", include_plotlyjs="cdn")


def main() -> None:
    """Run final candidate audit."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_rows = []
    summary_rows = []
    yearly_rows = []
    stress_rows = []

    for spec in CANDIDATES:
        daily = load_candidate_daily(spec)
        summary, yearly, stress = summarize_candidate(daily, spec)
        daily_rows.append(daily)
        summary_rows.append(summary)
        yearly_rows.append(yearly)
        stress_rows.append(stress)

    daily_out = pd.concat(daily_rows, ignore_index=True)
    summary_out = pd.DataFrame(summary_rows).sort_values(
        ["pass_full_and_pre2026", "stress_pass_count", "annual_return", "max_drawdown"],
        ascending=[False, False, False, False],
    )
    yearly_out = pd.concat(yearly_rows, ignore_index=True)
    stress_out = pd.concat(stress_rows, ignore_index=True)
    failed_stress = stress_out[~stress_out["pass_20_10"]].copy()

    summary_out.to_csv(OUTPUT_DIR / "final_candidate_summary.csv", index=False, encoding="utf-8-sig")
    yearly_out.to_csv(OUTPUT_DIR / "final_candidate_yearly.csv", index=False, encoding="utf-8-sig")
    stress_out.to_csv(OUTPUT_DIR / "final_candidate_window_stress.csv", index=False, encoding="utf-8-sig")
    failed_stress.to_csv(OUTPUT_DIR / "final_candidate_failed_windows.csv", index=False, encoding="utf-8-sig")
    daily_out.to_csv(OUTPUT_DIR / "final_candidate_daily.csv", index=False, encoding="utf-8-sig")
    save_nav_plot(daily_out)

    print("Final candidate summary:")
    print(
        summary_out[
            [
                "short_name",
                "family",
                "annual_return",
                "max_drawdown",
                "pre2026_annual_return",
                "annual_2022_2023",
                "annual_drop_2025",
                "stress_pass_count",
                "stress_pass_rate",
                "max_abs_weight",
                "turnover_proxy",
            ]
        ].to_string(index=False)
    )
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
