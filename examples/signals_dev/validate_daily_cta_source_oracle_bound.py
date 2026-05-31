"""Oracle upper bound over existing yearly source modes.

This script answers a narrow robustness question: with the existing generated
out-of-sample mode paths, can any per-year source selection make the 2020-2025
window reach 20% annual return while keeping drawdown within 10%?

The search is intentionally an oracle: it chooses source modes with hindsight
at the calendar-year level. If even this upper bound does not pass, further
threshold tuning over the same source modes is unlikely to solve the weakness.

Run:
    .venv/bin/python examples/signals_dev/validate_daily_cta_source_oracle_bound.py
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from validate_daily_cta_walk_forward import annualized_stats

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "examples" / "results" / "daily_cta_cross_section_walk_forward"
OUTPUT_DIR = INPUT_DIR / "source_oracle_bound"
DAILY_NAV_FILE = INPUT_DIR / "cross_section_daily_nav.csv"

SOURCE_MODES = (
    "aggressive_baseline",
    "aggressive_state_switch_recent_guard",
    "low_risk_top3",
    "low_risk_top10",
    "aggressive_state_switch",
    "aggressive_brake_blend_50",
)
PRE_END_YEAR = 2025
TARGET_MAX_DRAWDOWN = -0.10


def load_mode_year_cache() -> tuple[list[int], pd.Series, dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]]:
    """Load daily mode paths and cache returns by year and mode."""
    daily = pd.read_csv(DAILY_NAV_FILE, parse_dates=["dt"])
    years = sorted(int(year) for year in daily["test_year"].unique())
    dates = []
    cache = {}
    for year in years:
        year_dates = None
        for mode in SOURCE_MODES:
            part = daily[(daily["test_year"] == year) & (daily["mode"] == mode)].sort_values("dt")
            if part.empty:
                raise ValueError(f"No rows for {mode} {year}")
            cache[(year, mode)] = (part["ret"].to_numpy(float), part["weight"].to_numpy(float))
            if year_dates is None:
                year_dates = part["dt"].reset_index(drop=True)
        dates.append(year_dates)
    return years, pd.concat(dates, ignore_index=True), cache


def evaluate_combo(
    years: list[int],
    dates: pd.Series,
    cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]],
    combo: tuple[str, ...],
) -> dict:
    """Evaluate one oracle source-mode combination."""
    returns = []
    weights = []
    pre_len = 0
    for year, mode in zip(years, combo, strict=True):
        ret, weight = cache[(year, mode)]
        returns.append(ret)
        weights.append(weight)
        if year <= PRE_END_YEAR:
            pre_len += len(ret)
    ret_all = np.concatenate(returns)
    weight_all = np.concatenate(weights)
    pre_stats = annualized_stats(dates.iloc[:pre_len].reset_index(drop=True), ret_all[:pre_len], weight_all[:pre_len])
    full_stats = annualized_stats(dates, ret_all, weight_all)
    row = {
        "source_signature": "|".join(f"{year}:{mode}" for year, mode in zip(years, combo, strict=True)),
        "pre_annual_return": pre_stats["annual_return"],
        "pre_max_drawdown": pre_stats["max_drawdown"],
        "pre_calmar": pre_stats["calmar"],
        "pre_final_nav": pre_stats["final_nav"],
        "full_annual_return": full_stats["annual_return"],
        "full_max_drawdown": full_stats["max_drawdown"],
        "full_calmar": full_stats["calmar"],
        "full_final_nav": full_stats["final_nav"],
        "full_max_abs_weight": full_stats["max_abs_weight"],
        "pre_drawdown_ok": pre_stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN,
        "full_drawdown_ok": full_stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN,
    }
    for year, mode in zip(years, combo, strict=True):
        row[f"source_{year}"] = mode
    return row


def main() -> None:
    """Run the yearly oracle search and save outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    years, dates, cache = load_mode_year_cache()
    rows = []
    for combo in product(SOURCE_MODES, repeat=len(years)):
        row = evaluate_combo(years, dates, cache, combo)
        if row["pre_drawdown_ok"] and row["full_drawdown_ok"]:
            rows.append(row)

    feasible = pd.DataFrame(rows).sort_values(["pre_annual_return", "full_annual_return"], ascending=False)
    feasible.to_csv(OUTPUT_DIR / "source_oracle_feasible.csv", index=False, encoding="utf-8-sig")
    feasible.head(100).to_csv(OUTPUT_DIR / "source_oracle_top100.csv", index=False, encoding="utf-8-sig")

    best = feasible.iloc[0].to_dict()
    aggregate = {
        "source_modes": len(SOURCE_MODES),
        "years": len(years),
        "feasible_combos": len(feasible),
        "best_pre_annual_return": best["pre_annual_return"],
        "best_pre_max_drawdown": best["pre_max_drawdown"],
        "best_full_annual_return": best["full_annual_return"],
        "best_full_max_drawdown": best["full_max_drawdown"],
        "best_source_signature": best["source_signature"],
    }
    pd.DataFrame([aggregate]).to_csv(OUTPUT_DIR / "source_oracle_aggregate.csv", index=False, encoding="utf-8-sig")

    print(pd.DataFrame([aggregate]).to_string(index=False))
    print("\nTop oracle combinations:")
    print(
        feasible.head(10)[
            [
                "pre_annual_return",
                "pre_max_drawdown",
                "full_annual_return",
                "full_max_drawdown",
                "full_calmar",
                "source_signature",
            ]
        ].to_string(index=False)
    )
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
