"""Robustness checks for the adaptive cross-sectional daily CTA mix.

This script perturbs the training-only state thresholds used by
``aggressive_state_switch_adaptive_mix``. It does not re-optimize signals or
read future test-year returns when selecting a source mode.

Run:
    .venv/bin/python examples/signals_dev/validate_daily_cta_adaptive_mix_robustness.py
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from validate_daily_cta_walk_forward import annualized_stats, summarize_yearly

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "examples" / "results" / "daily_cta_cross_section_walk_forward"
OUTPUT_DIR = INPUT_DIR / "adaptive_mix_robustness"
DAILY_NAV_FILE = INPUT_DIR / "cross_section_daily_nav.csv"
SELECTED_FILE = INPUT_DIR / "selected_cross_section_by_fold.csv"

BASELINE_MODE = "aggressive_baseline"
LOW_RISK_MODE = "low_risk_top3"
RECENT_GUARD_MODE = "aggressive_state_switch_recent_guard"
TARGET_ANNUAL_RETURN = 0.20
TARGET_MAX_DRAWDOWN = -0.10

OFFICIAL_THRESHOLDS = {
    "low_ann": 0.05,
    "low_mdd": -0.08,
    "weak_ann": 0.17,
    "weak_recent": 0.00,
    "hot_ann": 0.20,
    "hot_recent": 0.45,
}

THRESHOLD_GRID = {
    "low_ann": (0.03, 0.04, 0.05, 0.06, 0.07),
    "low_mdd": (-0.10, -0.09, -0.08, -0.07, -0.06),
    "weak_ann": (0.15, 0.16, 0.17, 0.18, 0.19),
    "weak_recent": (-0.05, 0.00, 0.05),
    "hot_ann": (0.18, 0.19, 0.20, 0.21, 0.22),
    "hot_recent": (0.35, 0.40, 0.45, 0.50, 0.55),
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load stitched daily returns and baseline fold states."""
    daily = pd.read_csv(DAILY_NAV_FILE, parse_dates=["dt"])
    selected = pd.read_csv(SELECTED_FILE)
    baseline = selected[selected["mode"] == BASELINE_MODE].copy()
    baseline["test_year"] = pd.to_datetime(baseline["test_start"]).dt.year.astype(int)
    return daily, baseline


def choose_source(row: pd.Series, thresholds: dict[str, float]) -> str:
    """Choose one source mode from training-only fold statistics."""
    if row["annual_return"] < thresholds["low_ann"] and row["max_drawdown"] > thresholds["low_mdd"]:
        return BASELINE_MODE
    if row["annual_return"] < thresholds["weak_ann"] and row["recent_year_return"] < thresholds["weak_recent"]:
        return LOW_RISK_MODE
    if row["annual_return"] >= thresholds["hot_ann"] and row["recent_year_return"] >= thresholds["hot_recent"]:
        return BASELINE_MODE
    return RECENT_GUARD_MODE


def build_mix_daily(daily: pd.DataFrame, baseline: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """Build the mixed out-of-sample daily return path for one threshold set."""
    parts = []
    for _, row in baseline.sort_values("test_year").iterrows():
        test_year = int(row["test_year"])
        source = choose_source(row, thresholds)
        part = daily[(daily["mode"] == source) & (daily["test_year"] == test_year)].copy()
        if part.empty:
            raise ValueError(f"No daily rows for {source} {test_year}")
        part["source_mode"] = source
        parts.append(part)
    mixed = pd.concat(parts, ignore_index=True).sort_values("dt").reset_index(drop=True)
    mixed["nav"] = (1 + mixed["ret"]).cumprod()
    mixed["drawdown"] = mixed["nav"] / mixed["nav"].cummax() - 1
    return mixed


def source_signature(mixed: pd.DataFrame) -> str:
    """Compact source-mode signature by test year."""
    pairs = []
    for year, group in mixed.groupby("test_year"):
        source = str(group["source_mode"].iloc[0])
        short = {
            BASELINE_MODE: "base",
            LOW_RISK_MODE: "low3",
            RECENT_GUARD_MODE: "guard",
        }[source]
        pairs.append(f"{int(year)}:{short}")
    return "|".join(pairs)


def evaluate_thresholds(
    daily: pd.DataFrame,
    baseline: pd.DataFrame,
    thresholds: dict[str, float],
) -> tuple[dict, pd.DataFrame]:
    """Evaluate one threshold configuration."""
    mixed = build_mix_daily(daily, baseline, thresholds)
    stats = annualized_stats(mixed["dt"], mixed["ret"].to_numpy(), mixed["weight"].to_numpy())
    stats.update(thresholds)
    stats["pass_target"] = (
        stats["annual_return"] >= TARGET_ANNUAL_RETURN and stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
    )
    stats["source_signature"] = source_signature(mixed)
    yearly = summarize_yearly(mixed)
    yearly["source_signature"] = stats["source_signature"]
    for key, value in thresholds.items():
        yearly[key] = value
    return stats, yearly


def iter_thresholds() -> list[dict[str, float]]:
    """Generate the perturbation grid."""
    keys = tuple(THRESHOLD_GRID)
    return [dict(zip(keys, values, strict=True)) for values in product(*(THRESHOLD_GRID[key] for key in keys))]


def main() -> None:
    """Run robustness checks and persist summary files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily, baseline = load_inputs()

    rows = []
    yearly_parts = []
    for thresholds in iter_thresholds():
        stats, yearly = evaluate_thresholds(daily, baseline, thresholds)
        rows.append(stats)
        if stats["pass_target"]:
            yearly_parts.append(yearly)

    summary = pd.DataFrame(rows).sort_values(["pass_target", "annual_return", "calmar"], ascending=False)
    summary.to_csv(OUTPUT_DIR / "adaptive_mix_threshold_perturbation.csv", index=False, encoding="utf-8-sig")
    summary.head(50).to_csv(OUTPUT_DIR / "adaptive_mix_threshold_top50.csv", index=False, encoding="utf-8-sig")

    if yearly_parts:
        pd.concat(yearly_parts, ignore_index=True).to_csv(
            OUTPUT_DIR / "adaptive_mix_threshold_pass_yearly.csv", index=False, encoding="utf-8-sig"
        )

    official_stats, official_yearly = evaluate_thresholds(daily, baseline, OFFICIAL_THRESHOLDS)
    pd.DataFrame([official_stats]).to_csv(
        OUTPUT_DIR / "adaptive_mix_official_thresholds.csv", index=False, encoding="utf-8-sig"
    )
    official_yearly.to_csv(OUTPUT_DIR / "adaptive_mix_official_yearly.csv", index=False, encoding="utf-8-sig")

    pass_rows = summary[summary["pass_target"]].copy()
    aggregate = {
        "total_configs": len(summary),
        "pass_configs": len(pass_rows),
        "pass_rate": len(pass_rows) / len(summary) if len(summary) else 0.0,
        "annual_median": float(summary["annual_return"].median()),
        "max_drawdown_median": float(summary["max_drawdown"].median()),
        "pass_annual_min": float(pass_rows["annual_return"].min()) if len(pass_rows) else np.nan,
        "pass_max_drawdown_min": float(pass_rows["max_drawdown"].min()) if len(pass_rows) else np.nan,
        "official_annual_return": official_stats["annual_return"],
        "official_max_drawdown": official_stats["max_drawdown"],
        "official_pass_target": official_stats["pass_target"],
    }
    pd.DataFrame([aggregate]).to_csv(OUTPUT_DIR / "adaptive_mix_robustness_aggregate.csv", index=False)

    print(pd.DataFrame([aggregate]).to_string(index=False))
    print("\nTop threshold configurations:")
    top_cols = [
        "annual_return",
        "max_drawdown",
        "calmar",
        "final_nav",
        "low_ann",
        "low_mdd",
        "weak_ann",
        "weak_recent",
        "hot_ann",
        "hot_recent",
        "source_signature",
    ]
    print(summary.head(10)[top_cols].to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
