# ruff: noqa: E402
"""Tradeability audit for final daily CTA candidates.

The final candidate audit compares return and drawdown. This follow-up checks
whether the selected candidates are still usable after considering portfolio
level exposure, day-to-day turnover and incremental trading costs.

Run:
    uv run --no-sync python examples/signals_dev/validate_daily_cta_final_candidate_tradeability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_daily_cta_layered_low_corr_overlay import TARGET_ANNUAL_RETURN, TARGET_MAX_DRAWDOWN
from validate_daily_cta_walk_forward import annualized_stats, summarize_yearly

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "examples" / "results" / "daily_cta_final_candidate_audit"
SOURCE_DAILY = SOURCE_DIR / "final_candidate_daily.csv"
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_final_candidate_tradeability"

FOCUS_CANDIDATES = (
    "highest_return_bond_inflation",
    "robust_50rv_blend",
    "annual_state_capital",
)

EXTRA_COST_RATES = (0.0, 0.0001, 0.0002, 0.0005, 0.0010, 0.0020)


def max_consecutive_days(mask: pd.Series) -> int:
    """Return max run length for a boolean mask."""
    if mask.empty:
        return 0
    group_id = mask.ne(mask.shift(fill_value=False)).cumsum()
    runs = mask.groupby(group_id).sum()
    return int(runs.max()) if not runs.empty else 0


def load_focus_daily() -> pd.DataFrame:
    """Load the audited daily paths for the focus candidates."""
    if not SOURCE_DAILY.exists():
        raise FileNotFoundError(f"Missing final candidate daily file: {SOURCE_DAILY}")
    daily = pd.read_csv(SOURCE_DAILY, parse_dates=["dt"])
    daily["dt"] = pd.to_datetime(daily["dt"]).dt.tz_localize(None)
    daily = daily[daily["label"].isin(FOCUS_CANDIDATES)].copy()
    if daily.empty:
        raise ValueError(f"No focus candidates found in {SOURCE_DAILY}")
    daily = daily.sort_values(["label", "dt"]).reset_index(drop=True)
    daily["turnover"] = daily.groupby("label")["weight"].diff().abs()
    first_rows = daily.groupby("label").head(1).index
    daily.loc[first_rows, "turnover"] = daily.loc[first_rows, "weight"].abs()
    daily["turnover"] = daily["turnover"].fillna(0.0)
    return daily


def exposure_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Summarize exposure and turnover diagnostics for each candidate."""
    rows = []
    for label, group in daily.groupby("label", sort=False):
        abs_weight = group["weight"].abs()
        turnover = group["turnover"]
        years = (group["dt"].iloc[-1] - group["dt"].iloc[0]).days / 365.25
        rows.append(
            {
                "label": label,
                "days": int(len(group)),
                "avg_abs_weight": float(abs_weight.mean()),
                "median_abs_weight": float(abs_weight.median()),
                "weight_p95": float(abs_weight.quantile(0.95)),
                "weight_p99": float(abs_weight.quantile(0.99)),
                "max_abs_weight": float(abs_weight.max()),
                "days_abs_weight_gt_1": int((abs_weight > 1.0).sum()),
                "days_abs_weight_gt_2": int((abs_weight > 2.0).sum()),
                "days_abs_weight_gt_2_5": int((abs_weight > 2.5).sum()),
                "days_near_flat": int((abs_weight < 0.10).sum()),
                "max_consecutive_near_flat": max_consecutive_days(abs_weight < 0.10),
                "max_consecutive_gt_2": max_consecutive_days(abs_weight > 2.0),
                "turnover_sum": float(turnover.sum()),
                "annualized_turnover": float(turnover.sum() / years) if years > 0 else float("nan"),
                "avg_daily_turnover": float(turnover.mean()),
                "turnover_p95": float(turnover.quantile(0.95)),
                "turnover_p99": float(turnover.quantile(0.99)),
                "max_daily_turnover": float(turnover.max()),
                "days_turnover_gt_1": int((turnover > 1.0).sum()),
                "days_turnover_gt_2": int((turnover > 2.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def cost_sensitivity(daily: pd.DataFrame) -> pd.DataFrame:
    """Apply incremental transaction costs to each candidate path."""
    rows = []
    for label, group in daily.groupby("label", sort=False):
        group = group.sort_values("dt").copy()
        for cost_rate in EXTRA_COST_RATES:
            adjusted_ret = group["ret"] - group["turnover"] * cost_rate
            stats = annualized_stats(group["dt"].reset_index(drop=True), adjusted_ret.to_numpy(), group["weight"].to_numpy())
            yearly = summarize_yearly(
                pd.DataFrame({"dt": group["dt"].to_numpy(), "ret": adjusted_ret.to_numpy(), "weight": group["weight"].to_numpy()})
            )
            rows.append(
                {
                    "label": label,
                    "extra_cost_bps": cost_rate * 10000,
                    "annual_return": stats["annual_return"],
                    "max_drawdown": stats["max_drawdown"],
                    "final_nav": stats["final_nav"],
                    "calmar": stats["calmar"],
                    "loss_years": int((yearly["return"] < 0).sum()),
                    "pass_20_10": bool(
                        stats["annual_return"] >= TARGET_ANNUAL_RETURN
                        and stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
                    ),
                }
            )
    return pd.DataFrame(rows)


def yearly_cost_sensitivity(daily: pd.DataFrame) -> pd.DataFrame:
    """Save yearly returns under the most practical cost assumptions."""
    rows = []
    for label, group in daily.groupby("label", sort=False):
        group = group.sort_values("dt").copy()
        for cost_rate in (0.0, 0.0002, 0.0005, 0.0010):
            adjusted_ret = group["ret"] - group["turnover"] * cost_rate
            yearly = summarize_yearly(
                pd.DataFrame({"dt": group["dt"].to_numpy(), "ret": adjusted_ret.to_numpy(), "weight": group["weight"].to_numpy()})
            )
            yearly["label"] = label
            yearly["extra_cost_bps"] = cost_rate * 10000
            rows.append(yearly)
    return pd.concat(rows, ignore_index=True)


def save_tradeability_nav(daily: pd.DataFrame, cost_df: pd.DataFrame) -> None:
    """Save net-value plot at selected incremental cost rates."""
    import plotly.graph_objects as go

    fig = go.Figure()
    selected_costs = (0.0, 0.0002, 0.0005, 0.0010)
    for label, group in daily.groupby("label", sort=False):
        group = group.sort_values("dt").copy()
        for cost_rate in selected_costs:
            nav = (1 + group["ret"] - group["turnover"] * cost_rate).cumprod()
            name = f"{label} +{cost_rate * 10000:g}bps"
            fig.add_trace(go.Scatter(x=group["dt"], y=nav, mode="lines", name=name))
    fig.update_layout(
        title="Final candidates: incremental cost sensitivity",
        xaxis_title="date",
        yaxis_title="net value",
        template="plotly_white",
    )
    fig.write_html(OUTPUT_DIR / "final_candidate_tradeability_nav.html", include_plotlyjs="cdn")

    latest = cost_df[cost_df["extra_cost_bps"].isin([0.0, 2.0, 5.0, 10.0])].copy()
    latest.to_csv(OUTPUT_DIR / "final_candidate_cost_selected.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    """Run the tradeability audit."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = load_focus_daily()
    exposure = exposure_summary(daily)
    costs = cost_sensitivity(daily)
    yearly_costs = yearly_cost_sensitivity(daily)

    daily.to_csv(OUTPUT_DIR / "final_candidate_tradeability_daily.csv", index=False, encoding="utf-8-sig")
    exposure.to_csv(OUTPUT_DIR / "final_candidate_exposure_turnover.csv", index=False, encoding="utf-8-sig")
    costs.to_csv(OUTPUT_DIR / "final_candidate_cost_sensitivity.csv", index=False, encoding="utf-8-sig")
    yearly_costs.to_csv(OUTPUT_DIR / "final_candidate_yearly_cost_sensitivity.csv", index=False, encoding="utf-8-sig")
    save_tradeability_nav(daily, costs)

    print("Exposure / turnover:")
    print(
        exposure[
            [
                "label",
                "avg_abs_weight",
                "weight_p95",
                "max_abs_weight",
                "annualized_turnover",
                "turnover_p95",
                "max_daily_turnover",
                "days_turnover_gt_1",
            ]
        ].to_string(index=False)
    )
    print("\nCost sensitivity:")
    print(
        costs[costs["extra_cost_bps"].isin([0.0, 2.0, 5.0, 10.0])][
            ["label", "extra_cost_bps", "annual_return", "max_drawdown", "final_nav", "pass_20_10"]
        ].to_string(index=False)
    )
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
