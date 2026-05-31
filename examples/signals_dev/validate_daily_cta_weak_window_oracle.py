# ruff: noqa: E402, I001
"""Audit weak-window upper bounds in the layered overlay candidate library.

This is not a deployable strategy. It reads the existing fixed-candidate yearly
statistics and asks whether the current candidate library already contains
rules that could fix the weak windows:

- 2022-2023 rolling window;
- drop-2025 stress window.

The audit helps distinguish "meta selector missed a good rule" from "the
candidate library has no strong enough source for that period".

Run:
    uv run --no-sync python examples/signals_dev/validate_daily_cta_weak_window_oracle.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LAYERED_OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_layered_low_corr_overlay"
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_weak_window_oracle"
TARGET_ANNUAL_RETURN = 0.20
TARGET_MAX_DRAWDOWN = -0.10


def annual_from_returns(returns: pd.Series) -> float:
    """Calculate annualized return from calendar-year returns."""
    return float((1 + returns).prod() ** (1 / len(returns)) - 1)


def build_oracle_table() -> pd.DataFrame:
    """Build per-label weak-window diagnostics from yearly candidate stats."""
    summary = pd.read_csv(LAYERED_OUTPUT_DIR / "layered_low_corr_overlay_summary.csv")
    yearly = pd.read_csv(LAYERED_OUTPUT_DIR / "layered_low_corr_overlay_yearly.csv")
    rows = []
    for label, group in yearly.groupby("label"):
        years = group.set_index("year").sort_index()
        if not {2022, 2023, 2025}.issubset(years.index):
            continue
        window_2223 = years.loc[[2022, 2023]]
        drop_2025 = years[years.index != 2025]
        rows.append(
            {
                "label": label,
                "annual_2022_2023": annual_from_returns(window_2223["return"]),
                "mdd_2022_2023": float(window_2223["max_drawdown"].min()),
                "annual_drop_2025": annual_from_returns(drop_2025["return"]),
                "mdd_drop_2025": float(drop_2025["max_drawdown"].min()),
            }
        )
    out = pd.DataFrame(rows).merge(
        summary[
            [
                "label",
                "annual_return",
                "max_drawdown",
                "pre2026_annual_return",
                "pre2026_max_drawdown",
                "base_preset",
                "overlay_mode",
                "overlay_weight",
                "budget",
                "sleeve_config",
            ]
        ],
        on="label",
        how="left",
    )
    out["pass_2022_2023"] = (out["annual_2022_2023"] >= TARGET_ANNUAL_RETURN) & (
        out["mdd_2022_2023"] >= TARGET_MAX_DRAWDOWN
    )
    out["pass_drop_2025"] = (out["annual_drop_2025"] >= TARGET_ANNUAL_RETURN) & (
        out["mdd_drop_2025"] >= TARGET_MAX_DRAWDOWN
    )
    out["pass_both_weak_windows"] = out["pass_2022_2023"] & out["pass_drop_2025"]
    return out


def main() -> None:
    """Run weak-window oracle audit."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_oracle_table()

    best_2022_2023 = table[table["mdd_2022_2023"] >= TARGET_MAX_DRAWDOWN].sort_values(
        ["annual_2022_2023", "annual_drop_2025", "max_drawdown"],
        ascending=[False, False, False],
    )
    best_drop_2025 = table[table["mdd_drop_2025"] >= TARGET_MAX_DRAWDOWN].sort_values(
        ["annual_drop_2025", "annual_2022_2023", "max_drawdown"],
        ascending=[False, False, False],
    )
    aggregate = pd.DataFrame(
        [
            {
                "candidate_count": int(len(table)),
                "pass_2022_2023_count": int(table["pass_2022_2023"].sum()),
                "pass_drop_2025_count": int(table["pass_drop_2025"].sum()),
                "pass_both_weak_windows_count": int(table["pass_both_weak_windows"].sum()),
                "best_2022_2023_annual_under_10dd": float(best_2022_2023["annual_2022_2023"].iloc[0]),
                "best_drop_2025_annual_under_10dd": float(best_drop_2025["annual_drop_2025"].iloc[0]),
            }
        ]
    )

    table.to_csv(OUTPUT_DIR / "weak_window_oracle_all.csv", index=False, encoding="utf-8-sig")
    best_2022_2023.head(50).to_csv(
        OUTPUT_DIR / "weak_window_oracle_best_2022_2023.csv", index=False, encoding="utf-8-sig"
    )
    best_drop_2025.head(50).to_csv(
        OUTPUT_DIR / "weak_window_oracle_best_drop_2025.csv", index=False, encoding="utf-8-sig"
    )
    aggregate.to_csv(OUTPUT_DIR / "weak_window_oracle_aggregate.csv", index=False, encoding="utf-8-sig")

    print("Weak-window oracle aggregate:")
    print(aggregate.to_string(index=False))
    print("\nBest 2022-2023 candidates under 10% yearly drawdown:")
    print(
        best_2022_2023[
            [
                "label",
                "annual_2022_2023",
                "mdd_2022_2023",
                "annual_drop_2025",
                "mdd_drop_2025",
                "annual_return",
                "max_drawdown",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print("\nBest drop-2025 candidates under 10% yearly drawdown:")
    print(
        best_drop_2025[
            [
                "label",
                "annual_drop_2025",
                "mdd_drop_2025",
                "annual_2022_2023",
                "mdd_2022_2023",
                "annual_return",
                "max_drawdown",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
