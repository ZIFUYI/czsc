"""Window stress tests for annual-state scaled daily CTA variants.

The test uses already generated OOS daily returns from
``validate_daily_cta_annual_state_scale.py``. It does not reselect parameters.

Run:
    .venv/bin/python examples/signals_dev/validate_daily_cta_annual_state_window_stress.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from validate_daily_cta_walk_forward import annualized_stats, summarize_yearly

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "examples" / "results" / "daily_cta_cross_section_walk_forward" / "annual_state_scale"
DAILY_FILE = INPUT_DIR / "annual_state_scale_daily.csv"
OUTPUT_DIR = INPUT_DIR / "window_stress"
TARGET_ANNUAL_RETURN = 0.20
TARGET_MAX_DRAWDOWN = -0.10
MIN_CONSECUTIVE_WINDOW_YEARS = 2


def summarize_window(daily: pd.DataFrame, preset: str, kind: str, window: str, **meta: int) -> dict:
    """Summarize one stressed date window."""
    stats = annualized_stats(daily["dt"], daily["ret"].to_numpy(), daily["weight"].to_numpy())
    stats["preset"] = preset
    stats["kind"] = kind
    stats["window"] = window
    stats["n_days"] = len(daily)
    stats["pass_target"] = (
        stats["annual_return"] >= TARGET_ANNUAL_RETURN and stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
    )
    stats.update(meta)
    return stats


def build_stress_rows(daily: pd.DataFrame, preset: str) -> list[dict]:
    """Build moving-start, moving-end, drop-one-year and consecutive-window rows."""
    rows = []
    years = sorted(int(year) for year in daily["test_year"].unique())
    first_year, last_year = years[0], years[-1]

    rows.append(
        summarize_window(daily, preset, "full", f"{first_year}-{last_year}", start_year=first_year, end_year=last_year)
    )

    for start_year in years:
        part = daily[daily["test_year"] >= start_year].copy()
        rows.append(
            summarize_window(
                part,
                preset,
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
                preset,
                "moving_end",
                f"{first_year}-{end_year}",
                start_year=first_year,
                end_year=end_year,
            )
        )

    for drop_year in years:
        part = daily[daily["test_year"] != drop_year].copy()
        rows.append(summarize_window(part, preset, "drop_one_year", f"drop_{drop_year}", drop_year=drop_year))

    for start_idx, start_year in enumerate(years):
        for end_year in years[start_idx + MIN_CONSECUTIVE_WINDOW_YEARS - 1 :]:
            part = daily[(daily["test_year"] >= start_year) & (daily["test_year"] <= end_year)].copy()
            rows.append(
                summarize_window(
                    part,
                    preset,
                    "consecutive_window",
                    f"{start_year}-{end_year}",
                    start_year=start_year,
                    end_year=end_year,
                )
            )

    return rows


def main() -> None:
    """Run window stress tests and persist CSV outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(DAILY_FILE, parse_dates=["dt"])
    if daily.empty:
        raise ValueError(f"No daily rows found in {DAILY_FILE}")

    stress_rows = []
    yearly_rows = []
    for preset, group in daily.groupby("preset"):
        group = group.sort_values("dt").reset_index(drop=True)
        stress_rows.extend(build_stress_rows(group, str(preset)))
        yearly = summarize_yearly(group)
        yearly["preset"] = preset
        yearly_rows.append(yearly)

    stress = pd.DataFrame(stress_rows)
    yearly = pd.concat(yearly_rows, ignore_index=True)
    stress.to_csv(OUTPUT_DIR / "annual_state_scale_window_stress.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUTPUT_DIR / "annual_state_scale_yearly.csv", index=False, encoding="utf-8-sig")

    aggregate_rows = []
    for preset, group in stress.groupby("preset"):
        full = group[group["kind"] == "full"].iloc[0]
        moving_end_2025 = group[(group["kind"] == "moving_end") & (group["end_year"] == 2025)].iloc[0]
        drop_2020 = group[(group["kind"] == "drop_one_year") & (group["drop_year"] == 2020)].iloc[0]
        drop_2025 = group[(group["kind"] == "drop_one_year") & (group["drop_year"] == 2025)].iloc[0]
        consecutive = group[group["kind"] == "consecutive_window"]
        aggregate_rows.append(
            {
                "preset": preset,
                "stress_windows": int(len(group)),
                "pass_windows": int(group["pass_target"].sum()),
                "pass_rate": float(group["pass_target"].mean()),
                "full_annual_return": float(full["annual_return"]),
                "full_max_drawdown": float(full["max_drawdown"]),
                "moving_end_2025_annual_return": float(moving_end_2025["annual_return"]),
                "moving_end_2025_max_drawdown": float(moving_end_2025["max_drawdown"]),
                "drop_2020_annual_return": float(drop_2020["annual_return"]),
                "drop_2020_max_drawdown": float(drop_2020["max_drawdown"]),
                "drop_2025_annual_return": float(drop_2025["annual_return"]),
                "drop_2025_max_drawdown": float(drop_2025["max_drawdown"]),
                "consecutive_pass_rate": float(consecutive["pass_target"].mean()),
                "consecutive_min_annual_return": float(consecutive["annual_return"].min()),
                "consecutive_worst_drawdown": float(consecutive["max_drawdown"].min()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(OUTPUT_DIR / "annual_state_scale_window_stress_aggregate.csv", index=False, encoding="utf-8-sig")

    print(aggregate.to_string(index=False))
    print("\nCapital efficient stress windows:")
    show = stress[stress["preset"] == "capital_efficient"].copy()
    print(
        show[
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
