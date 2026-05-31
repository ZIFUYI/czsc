# ruff: noqa: E402, I001
"""Focused meta ensembles for weak-window overlay candidates.

The broad meta-ensemble search is useful but can be slow after the policy
universe grows. This script only tests policies designed to reduce dependence
on 2025 and to improve the weak 2022-2023 window, while keeping the same
prior-OOS-only top-N selection protocol.

Run:
    uv run --no-sync python examples/signals_dev/validate_daily_cta_weak_window_meta_ensemble.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_daily_cta_layered_low_corr_overlay import (  # noqa: E402
    BASE_PRESETS,
    OUTPUT_DIR as LAYERED_OUTPUT_DIR,
    evaluate_window_stress,
)
from validate_daily_cta_layered_overlay_meta_ensemble import (  # noqa: E402
    build_selection_plan,
    render_candidate_daily,
    stitch_from_plan,
    summarize_path,
)
from validate_daily_cta_walk_forward import summarize_yearly  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_weak_window_meta_ensemble"
POLICIES = ("energy_conservative", "weak_window_conservative")


def build_plans(summary: pd.DataFrame, yearly: pd.DataFrame) -> list[pd.DataFrame]:
    """Build focused top-N selection plans."""
    plans = []
    for base_preset in BASE_PRESETS:
        for policy in POLICIES:
            for min_train_years in [1, 2, 3]:
                for train_window in ["expanding", "rolling3"]:
                    for top_n in [3, 5, 10]:
                        for weight_method in ["equal", "rank"]:
                            plans.append(
                                build_selection_plan(
                                    summary,
                                    yearly,
                                    base_preset,
                                    policy,
                                    min_train_years,
                                    train_window,
                                    top_n,
                                    weight_method,
                                )
                            )
    return plans


def needed_candidate_labels(plans: list[pd.DataFrame]) -> set[str]:
    """Collect every selected underlying candidate label."""
    labels: set[str] = set()
    for plan in plans:
        for raw in plan["selected_labels"]:
            labels.update(str(raw).split("|"))
    return labels


def main() -> None:
    """Run the focused weak-window meta ensemble validation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(LAYERED_OUTPUT_DIR / "layered_low_corr_overlay_summary.csv")
    yearly_candidates = pd.read_csv(LAYERED_OUTPUT_DIR / "layered_low_corr_overlay_yearly.csv")

    plans = build_plans(summary, yearly_candidates)
    selected_out = pd.concat(plans, ignore_index=True)
    candidate_daily = render_candidate_daily(summary, needed_candidate_labels(plans))

    daily_rows = []
    summary_rows = []
    yearly_rows = []
    for plan in plans:
        path = stitch_from_plan(candidate_daily, plan)
        meta_label = path["meta_label"].iloc[0]
        stats = summarize_path(path, meta_label)
        first = plan.iloc[0]
        stats.update(
            {
                "base_preset": first["base_preset"],
                "policy": first["policy"],
                "min_train_years": first["min_train_years"],
                "train_window": first["train_window"],
                "top_n": first["top_n"],
                "weight_method": first["weight_method"],
            }
        )
        daily_rows.append(path)
        summary_rows.append(stats)
        yearly = summarize_yearly(path[["dt", "ret", "weight"]])
        yearly["meta_label"] = meta_label
        yearly_rows.append(yearly)

    summary_out = pd.DataFrame(summary_rows)
    daily_out = pd.concat(daily_rows, ignore_index=True)
    yearly_out = pd.concat(yearly_rows, ignore_index=True)

    stress_rows = []
    stress_labels = list(
        summary_out.sort_values(["pass_full_and_pre2026", "annual_return", "max_drawdown"], ascending=False)[
            "meta_label"
        ]
        .head(30)
        .drop_duplicates()
    )
    for meta_label in stress_labels:
        path = daily_out[daily_out["meta_label"] == meta_label].copy()
        stress = evaluate_window_stress(path.rename(columns={"meta_label": "label"}), meta_label)
        stress["meta_label"] = meta_label
        stress_rows.append(stress)
    stress_out = pd.concat(stress_rows, ignore_index=True)
    stress_agg = (
        stress_out.groupby("meta_label")
        .agg(
            stress_windows=("pass_20_10", "size"),
            stress_pass_rate=("pass_20_10", "mean"),
            stress_annual_median=("annual_return", "median"),
            stress_mdd_median=("max_drawdown", "median"),
        )
        .reset_index()
    )
    summary_out = summary_out.merge(stress_agg, on="meta_label", how="left")
    top = summary_out.sort_values(
        ["pass_full_and_pre2026", "stress_pass_rate", "annual_return", "max_drawdown"],
        ascending=[False, False, False, False],
        na_position="last",
    )

    selected_out.to_csv(OUTPUT_DIR / "weak_window_meta_selected_by_year.csv", index=False, encoding="utf-8-sig")
    summary_out.to_csv(OUTPUT_DIR / "weak_window_meta_summary.csv", index=False, encoding="utf-8-sig")
    top.head(30).to_csv(OUTPUT_DIR / "weak_window_meta_top30.csv", index=False, encoding="utf-8-sig")
    yearly_out.to_csv(OUTPUT_DIR / "weak_window_meta_yearly.csv", index=False, encoding="utf-8-sig")
    stress_out.to_csv(OUTPUT_DIR / "weak_window_meta_window_stress_top.csv", index=False, encoding="utf-8-sig")
    daily_out[daily_out["meta_label"].isin(top["meta_label"].head(12))].to_csv(
        OUTPUT_DIR / "weak_window_meta_daily_top12.csv", index=False, encoding="utf-8-sig"
    )

    print("Top focused weak-window meta ensembles:")
    print(
        top[
            [
                "meta_label",
                "annual_return",
                "max_drawdown",
                "pre2026_annual_return",
                "pre2026_max_drawdown",
                "stress_pass_rate",
                "stress_annual_median",
                "stress_mdd_median",
                "pass_full_and_pre2026",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
