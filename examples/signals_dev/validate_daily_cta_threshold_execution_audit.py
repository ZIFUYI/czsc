# ruff: noqa: E402
"""Focused audit for the recommended threshold execution rule.

The rebalance compression test selected ``threshold_10pct`` as the practical
execution rule for ``robust_50rv_blend``. This script audits that rule by year,
month, pressure window and rebalance concentration.

Run:
    uv run --no-sync python examples/signals_dev/validate_daily_cta_threshold_execution_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_daily_cta_layered_low_corr_overlay import evaluate_window_stress
from validate_daily_cta_walk_forward import annualized_stats, summarize_yearly

ROOT = Path(__file__).resolve().parents[2]
COMPRESSION_DIR = ROOT / "examples" / "results" / "daily_cta_robust_blend_rebalance_compression"
COMPRESSION_DAILY = COMPRESSION_DIR / "robust_50rv_rebalance_compression_daily.csv"
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_threshold_execution_audit"
SCHEMES = ("daily_raw", "threshold_10pct")
EXTRA_COST_RATES = (0.0, 0.0002, 0.0005, 0.0010)


def load_daily() -> pd.DataFrame:
    """Load daily paths for raw and threshold execution."""
    if not COMPRESSION_DAILY.exists():
        raise FileNotFoundError(f"Missing compression daily file: {COMPRESSION_DAILY}")
    daily = pd.read_csv(COMPRESSION_DAILY, parse_dates=["dt"])
    daily["dt"] = pd.to_datetime(daily["dt"]).dt.tz_localize(None)
    return daily[daily["scheme"].isin(SCHEMES)].sort_values(["scheme", "dt"]).reset_index(drop=True)


def monthly_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly return / drawdown / turnover diagnostics."""
    rows = []
    for scheme, scheme_daily in daily.groupby("scheme", sort=False):
        for month, group in scheme_daily.groupby(scheme_daily["dt"].dt.to_period("M")):
            nav = (1 + group["ret"]).cumprod()
            rows.append(
                {
                    "scheme": scheme,
                    "month": str(month),
                    "return": float(nav.iloc[-1] - 1),
                    "max_drawdown": float((nav / nav.cummax() - 1).min()),
                    "trading_days": int(len(group)),
                    "rebalance_days": int(group["rebalance"].sum()),
                    "turnover": float(group["turnover"].sum()),
                    "max_daily_turnover": float(group["turnover"].max()),
                    "avg_weight": float(group["weight"].abs().mean()),
                    "max_weight": float(group["weight"].abs().max()),
                }
            )
    return pd.DataFrame(rows)


def yearly_cost_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate yearly stats under selected incremental costs."""
    rows = []
    for scheme, scheme_daily in daily.groupby("scheme", sort=False):
        for cost_rate in EXTRA_COST_RATES:
            adjusted = scheme_daily.copy()
            adjusted["ret"] = adjusted["ret"] - adjusted["turnover"] * cost_rate
            yearly = summarize_yearly(adjusted[["dt", "ret", "weight"]])
            yearly["scheme"] = scheme
            yearly["extra_cost_bps"] = cost_rate * 10000
            rows.append(yearly)
    return pd.concat(rows, ignore_index=True)


def summary_cost_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate full-period stats under selected incremental costs."""
    rows = []
    for scheme, scheme_daily in daily.groupby("scheme", sort=False):
        for cost_rate in EXTRA_COST_RATES:
            ret = scheme_daily["ret"] - scheme_daily["turnover"] * cost_rate
            stats = annualized_stats(scheme_daily["dt"].reset_index(drop=True), ret.to_numpy(), scheme_daily["weight"].to_numpy())
            rows.append({"scheme": scheme, "extra_cost_bps": cost_rate * 10000, **stats})
    return pd.DataFrame(rows)


def window_stress(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate pressure-window stats for both schemes."""
    rows = []
    for scheme, scheme_daily in daily.groupby("scheme", sort=False):
        stress = evaluate_window_stress(scheme_daily[["dt", "ret", "weight"]].copy(), scheme)
        stress["scheme"] = scheme
        rows.append(stress)
    return pd.concat(rows, ignore_index=True)


def rebalance_dates(daily: pd.DataFrame) -> pd.DataFrame:
    """List largest rebalance dates for threshold execution."""
    threshold = daily[daily["scheme"].eq("threshold_10pct")].copy()
    threshold = threshold[threshold["rebalance"]].copy()
    threshold["year"] = threshold["dt"].dt.year
    threshold["month"] = threshold["dt"].dt.strftime("%Y-%m")
    return threshold.sort_values("turnover", ascending=False)[
        ["dt", "year", "month", "turnover", "weight", "desired_weight", "ret", "nav", "drawdown"]
    ].reset_index(drop=True)


def main() -> None:
    """Run focused threshold execution audit."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = load_daily()
    yearly = pd.concat(
        [
            summarize_yearly(group[["dt", "ret", "weight"]]).assign(scheme=scheme)
            for scheme, group in daily.groupby("scheme", sort=False)
        ],
        ignore_index=True,
    )
    months = monthly_stats(daily)
    stress = window_stress(daily)
    costs = summary_cost_stats(daily)
    yearly_cost = yearly_cost_stats(daily)
    rebalances = rebalance_dates(daily)
    worst_months = months.sort_values(["scheme", "return"]).groupby("scheme").head(12).reset_index(drop=True)
    failed_stress = stress[~stress["pass_20_10"]].sort_values(["scheme", "annual_return"])

    daily.to_csv(OUTPUT_DIR / "threshold_execution_daily.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUTPUT_DIR / "threshold_execution_yearly.csv", index=False, encoding="utf-8-sig")
    months.to_csv(OUTPUT_DIR / "threshold_execution_monthly.csv", index=False, encoding="utf-8-sig")
    worst_months.to_csv(OUTPUT_DIR / "threshold_execution_worst_months.csv", index=False, encoding="utf-8-sig")
    stress.to_csv(OUTPUT_DIR / "threshold_execution_window_stress.csv", index=False, encoding="utf-8-sig")
    failed_stress.to_csv(OUTPUT_DIR / "threshold_execution_failed_windows.csv", index=False, encoding="utf-8-sig")
    costs.to_csv(OUTPUT_DIR / "threshold_execution_cost_summary.csv", index=False, encoding="utf-8-sig")
    yearly_cost.to_csv(OUTPUT_DIR / "threshold_execution_yearly_cost.csv", index=False, encoding="utf-8-sig")
    rebalances.to_csv(OUTPUT_DIR / "threshold_execution_top_rebalances.csv", index=False, encoding="utf-8-sig")

    print("Yearly:")
    print(yearly[["scheme", "year", "return", "max_drawdown", "annual_return", "max_abs_weight"]].to_string(index=False))
    print("\nWorst months:")
    print(worst_months[["scheme", "month", "return", "max_drawdown", "turnover", "rebalance_days"]].head(20).to_string(index=False))
    print("\nFailed windows:")
    print(failed_stress[["scheme", "window", "annual_return", "max_drawdown"]].head(20).to_string(index=False))
    print("\nTop rebalances:")
    print(rebalances.head(15).to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
