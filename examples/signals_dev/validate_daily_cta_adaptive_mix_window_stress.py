"""Window stress tests for the adaptive cross-sectional daily CTA mix.

This script checks whether ``aggressive_state_switch_adaptive_mix`` remains
robust when the out-of-sample period starts later, ends earlier, or excludes one
calendar year. It uses the already generated daily OOS returns and does not
reselect parameters.

Run:
    .venv/bin/python examples/signals_dev/validate_daily_cta_adaptive_mix_window_stress.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from validate_daily_cta_walk_forward import annualized_stats, summarize_yearly

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "examples" / "results" / "daily_cta_cross_section_walk_forward"
OUTPUT_DIR = INPUT_DIR / "adaptive_mix_window_stress"
DAILY_NAV_FILE = INPUT_DIR / "cross_section_daily_nav.csv"
MODE = "aggressive_state_switch_adaptive_mix"
TARGET_ANNUAL_RETURN = 0.20
TARGET_MAX_DRAWDOWN = -0.10


def summarize_window(daily: pd.DataFrame, kind: str, window: str, **meta: int) -> dict:
    """Summarize one stressed date window."""
    stats = annualized_stats(daily["dt"], daily["ret"].to_numpy(), daily["weight"].to_numpy())
    stats["kind"] = kind
    stats["window"] = window
    stats["n_days"] = len(daily)
    stats["pass_target"] = (
        stats["annual_return"] >= TARGET_ANNUAL_RETURN and stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
    )
    stats.update(meta)
    return stats


def build_stress_rows(daily: pd.DataFrame) -> list[dict]:
    """Build moving-start, moving-end and drop-one-year stress rows."""
    rows = []
    years = sorted(int(year) for year in daily["test_year"].unique())
    first_year, last_year = years[0], years[-1]

    rows.append(summarize_window(daily, "full", f"{first_year}-{last_year}", start_year=first_year, end_year=last_year))

    for start_year in years:
        part = daily[daily["test_year"] >= start_year].copy()
        rows.append(
            summarize_window(
                part,
                "moving_start",
                f"{start_year}-{last_year}",
                start_year=start_year,
                end_year=last_year,
            )
        )

    for end_year in years:
        part = daily[daily["test_year"] <= end_year].copy()
        rows.append(
            summarize_window(
                part,
                "moving_end",
                f"{first_year}-{end_year}",
                start_year=first_year,
                end_year=end_year,
            )
        )

    for drop_year in years:
        part = daily[daily["test_year"] != drop_year].copy()
        rows.append(summarize_window(part, "drop_one_year", f"drop_{drop_year}", drop_year=drop_year))

    return rows


def main() -> None:
    """Run window stress tests and persist CSV outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(DAILY_NAV_FILE, parse_dates=["dt"])
    daily = daily[daily["mode"] == MODE].sort_values("dt").reset_index(drop=True)
    if daily.empty:
        raise ValueError(f"No daily rows for mode: {MODE}")

    stress = pd.DataFrame(build_stress_rows(daily))
    stress.to_csv(OUTPUT_DIR / "adaptive_mix_window_stress.csv", index=False, encoding="utf-8-sig")

    yearly = summarize_yearly(daily)
    yearly.to_csv(OUTPUT_DIR / "adaptive_mix_yearly.csv", index=False, encoding="utf-8-sig")

    aggregate = {
        "stress_windows": len(stress),
        "pass_windows": int(stress["pass_target"].sum()),
        "pass_rate": float(stress["pass_target"].mean()),
        "full_annual_return": float(stress.loc[stress["kind"] == "full", "annual_return"].iloc[0]),
        "full_max_drawdown": float(stress.loc[stress["kind"] == "full", "max_drawdown"].iloc[0]),
        "moving_end_2025_annual_return": float(
            stress.loc[(stress["kind"] == "moving_end") & (stress["end_year"] == 2025), "annual_return"].iloc[0]
        ),
        "drop_2026_annual_return": float(
            stress.loc[(stress["kind"] == "drop_one_year") & (stress["drop_year"] == 2026), "annual_return"].iloc[0]
        ),
    }
    pd.DataFrame([aggregate]).to_csv(OUTPUT_DIR / "adaptive_mix_window_stress_aggregate.csv", index=False)

    print(pd.DataFrame([aggregate]).to_string(index=False))
    print("\nWindow stress:")
    print(
        stress[
            [
                "kind",
                "window",
                "annual_return",
                "max_drawdown",
                "calmar",
                "final_nav",
                "pass_target",
                "n_days",
            ]
        ].to_string(index=False)
    )
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
