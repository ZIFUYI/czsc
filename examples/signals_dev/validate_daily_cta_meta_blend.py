# ruff: noqa: E402, I001
"""Blend bond / inflation meta ensembles with relative-value meta ensembles.

The bond / inflation meta candidate has the highest full-sample return, while
the relative-value meta candidate improves weak-window structure. This script
tests simple fixed blends between already validated prior-OOS meta paths.

Run:
    uv run --no-sync python examples/signals_dev/validate_daily_cta_meta_blend.py
"""

from __future__ import annotations

import sys
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
BOND_META_DIR = ROOT / "examples" / "results" / "daily_cta_layered_overlay_meta_ensemble"
RV_META_DIR = ROOT / "examples" / "results" / "daily_cta_relative_value_meta_ensemble"
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_meta_blend"
BOND_LABELS = (
    "stress_balanced_inflation_conservative_min2_rolling3_top3_rank_ensemble",
    "stress_balanced_bond_conservative_min2_rolling3_top5_equal_ensemble",
    "capital_efficient_bond_conservative_min2_rolling3_top5_equal_ensemble",
)
RV_LABELS = (
    "stress_balanced_rv_drop25_long_min1_rolling3_top5_equal_ensemble",
    "capital_efficient_rv_drop25_long_min1_rolling3_top5_equal_ensemble",
    "stress_balanced_rv_drop25_long_min1_rolling3_top3_rank_ensemble",
)
RV_WEIGHTS = tuple(x / 100 for x in range(0, 55, 5))


def load_meta_daily(file_csv: Path, label_col: str = "meta_label") -> pd.DataFrame:
    """Load one meta daily file."""
    daily = pd.read_csv(file_csv, parse_dates=["dt"])
    daily["dt"] = pd.to_datetime(daily["dt"]).dt.tz_localize(None)
    if label_col != "meta_label":
        daily = daily.rename(columns={label_col: "meta_label"})
    return daily[["dt", "ret", "weight", "meta_label", "test_year"]].copy()


def summarize_path(daily: pd.DataFrame, label: str) -> dict:
    """Summarize one blended path."""
    stats = annualized_stats(daily["dt"].reset_index(drop=True), daily["ret"].to_numpy(), daily["weight"].to_numpy())
    pre = daily[daily["dt"].dt.year <= 2025]
    pre_stats = annualized_stats(pre["dt"].reset_index(drop=True), pre["ret"].to_numpy(), pre["weight"].to_numpy())
    yearly = summarize_yearly(daily[["dt", "ret", "weight"]])
    part_2223 = yearly[yearly["year"].isin([2022, 2023])]
    drop_2025 = yearly[yearly["year"] != 2025]
    annual_2022_2023 = float((1 + part_2223["return"]).prod() ** (1 / len(part_2223)) - 1)
    annual_drop_2025 = float((1 + drop_2025["return"]).prod() ** (1 / len(drop_2025)) - 1)
    stats.update(
        {
            "label": label,
            "pre2026_annual_return": pre_stats["annual_return"],
            "pre2026_max_drawdown": pre_stats["max_drawdown"],
            "annual_2022_2023": annual_2022_2023,
            "mdd_2022_2023": float(part_2223["max_drawdown"].min()),
            "annual_drop_2025": annual_drop_2025,
            "mdd_drop_2025": float(drop_2025["max_drawdown"].min()),
            "pass_full_and_pre2026": bool(
                stats["annual_return"] >= TARGET_ANNUAL_RETURN
                and stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
                and pre_stats["annual_return"] >= TARGET_ANNUAL_RETURN
                and pre_stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
            ),
        }
    )
    return stats


def blend_paths(bond: pd.DataFrame, rv: pd.DataFrame, rv_weight: float, label: str) -> pd.DataFrame:
    """Blend two meta daily paths with fixed weights."""
    joined = bond.merge(rv, on="dt", suffixes=("_bond", "_rv"), how="inner")
    bond_weight = 1 - rv_weight
    out = pd.DataFrame(
        {
            "dt": joined["dt"],
            "ret": bond_weight * joined["ret_bond"] + rv_weight * joined["ret_rv"],
            "weight": bond_weight * joined["weight_bond"].abs() + rv_weight * joined["weight_rv"].abs(),
            "test_year": joined["test_year_bond"],
            "label": label,
        }
    )
    out["nav"] = (1 + out["ret"]).cumprod()
    out["drawdown"] = out["nav"] / out["nav"].cummax() - 1
    return out


def main() -> None:
    """Run fixed blend validation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bond_daily = load_meta_daily(BOND_META_DIR / "meta_ensemble_daily_top12.csv")
    rv_daily = load_meta_daily(RV_META_DIR / "relative_value_meta_daily_top12.csv")

    daily_rows = []
    summary_rows = []
    yearly_rows = []
    for bond_label in BOND_LABELS:
        bond = bond_daily[bond_daily["meta_label"].eq(bond_label)].copy()
        for rv_label in RV_LABELS:
            rv = rv_daily[rv_daily["meta_label"].eq(rv_label)].copy()
            for rv_weight in RV_WEIGHTS:
                label = f"blend_RV{int(rv_weight * 100):02d}_{bond_label}__{rv_label}"
                path = blend_paths(bond, rv, rv_weight, label)
                daily_rows.append(path)
                row = summarize_path(path, label)
                row.update(
                    {
                        "bond_label": bond_label,
                        "rv_label": rv_label,
                        "bond_weight": 1 - rv_weight,
                        "rv_weight": rv_weight,
                    }
                )
                summary_rows.append(row)
                yearly = summarize_yearly(path[["dt", "ret", "weight"]])
                yearly["label"] = label
                yearly_rows.append(yearly)

    daily_out = pd.concat(daily_rows, ignore_index=True)
    summary_out = pd.DataFrame(summary_rows)
    yearly_out = pd.concat(yearly_rows, ignore_index=True)

    stress_labels = list(
        summary_out.sort_values(
            [
                "pass_full_and_pre2026",
                "annual_drop_2025",
                "annual_2022_2023",
                "annual_return",
                "max_drawdown",
            ],
            ascending=[False, False, False, False, False],
        )["label"]
        .head(30)
        .drop_duplicates()
    )
    stress_rows = []
    for label in stress_labels:
        stress = evaluate_window_stress(daily_out[daily_out["label"].eq(label)].copy(), label)
        stress["label"] = label
        stress_rows.append(stress)
    stress_out = pd.concat(stress_rows, ignore_index=True)
    stress_agg = (
        stress_out.groupby("label")
        .agg(
            stress_windows=("pass_20_10", "size"),
            stress_pass_rate=("pass_20_10", "mean"),
            stress_annual_median=("annual_return", "median"),
            stress_mdd_median=("max_drawdown", "median"),
        )
        .reset_index()
    )
    summary_out = summary_out.merge(stress_agg, on="label", how="left")
    top = summary_out.sort_values(
        [
            "pass_full_and_pre2026",
            "stress_pass_rate",
            "annual_drop_2025",
            "annual_2022_2023",
            "annual_return",
        ],
        ascending=[False, False, False, False, False],
        na_position="last",
    )

    summary_out.to_csv(OUTPUT_DIR / "meta_blend_summary.csv", index=False, encoding="utf-8-sig")
    top.head(50).to_csv(OUTPUT_DIR / "meta_blend_top50.csv", index=False, encoding="utf-8-sig")
    yearly_out.to_csv(OUTPUT_DIR / "meta_blend_yearly.csv", index=False, encoding="utf-8-sig")
    stress_out.to_csv(OUTPUT_DIR / "meta_blend_window_stress_top.csv", index=False, encoding="utf-8-sig")
    daily_out[daily_out["label"].isin(top["label"].head(12))].to_csv(
        OUTPUT_DIR / "meta_blend_daily_top12.csv", index=False, encoding="utf-8-sig"
    )

    print("Top meta blends:")
    print(
        top[
            [
                "label",
                "rv_weight",
                "annual_return",
                "max_drawdown",
                "pre2026_annual_return",
                "annual_2022_2023",
                "annual_drop_2025",
                "stress_pass_rate",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
