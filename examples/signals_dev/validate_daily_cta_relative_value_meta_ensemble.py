# ruff: noqa: E402, I001
"""Prior-OOS meta ensembles for relative-value overlays.

This script tests whether relative-value candidates found by
``validate_daily_cta_relative_value_overlay.py`` survive a stricter
meta-selection protocol. Each test year ranks candidates using only prior
out-of-sample years, blends top-N eligible rules, and freezes the selected
ensemble for that year.

Run:
    uv run --no-sync python examples/signals_dev/validate_daily_cta_relative_value_meta_ensemble.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_daily_cta_layered_low_corr_overlay import (  # noqa: E402
    BASE_PRESETS,
    TARGET_ANNUAL_RETURN,
    TARGET_MAX_DRAWDOWN,
    evaluate_window_stress,
)
from validate_daily_cta_layered_overlay_meta_ensemble import (  # noqa: E402
    ensemble_weights,
)
from validate_daily_cta_relative_value_overlay import (  # noqa: E402
    OUTPUT_DIR as RELATIVE_OUTPUT_DIR,
    RELATIVE_CONFIGS,
    class_return_panel,
    load_base_daily,
    relative_sleeve,
)
from validate_daily_cta_walk_forward import annualized_stats, summarize_yearly  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_relative_value_meta_ensemble"
YEARS = tuple(range(2020, 2027))
POLICIES = (
    "rv_practical_long",
    "rv_drop25_long",
    "rv_weak_long",
    "rv_spread_upper_bound",
)


def candidate_filter(summary: pd.DataFrame, policy: str, base_preset: str) -> pd.Series:
    """Return a boolean mask for one relative-value meta universe."""
    mask = summary["base_preset"].eq(base_preset)
    rotate_long = summary["relative_config"].str.contains("rotate_long", regex=False)
    spread_ls = summary["relative_config"].str.contains("spread_ls", regex=False)

    if policy == "rv_practical_long":
        return mask & rotate_long & summary["left"].isin(["gold", "agri", "india_asia", "us_broad", "energy"])
    if policy == "rv_drop25_long":
        return mask & rotate_long & summary["left"].eq("gold") & summary["right"].eq("commodity")
    if policy == "rv_weak_long":
        return (
            mask
            & rotate_long
            & (
                (summary["left"].eq("agri") & summary["right"].eq("cn_equity"))
                | (summary["left"].eq("india_asia") & summary["right"].eq("hk"))
                | (summary["left"].eq("us_broad") & summary["right"].eq("hk"))
                | (summary["left"].eq("energy") & summary["right"].eq("cn_equity"))
            )
        )
    if policy == "rv_spread_upper_bound":
        return (
            mask
            & spread_ls
            & (
                (summary["left"].eq("agri") & summary["right"].eq("cn_equity"))
                | (summary["left"].eq("india_asia") & summary["right"].eq("hk"))
                | (summary["left"].eq("us_broad") & summary["right"].eq("hk"))
                | (summary["left"].eq("energy") & summary["right"].eq("cn_equity"))
                | (summary["left"].eq("gold") & summary["right"].eq("commodity"))
            )
        )
    raise ValueError(f"Unsupported policy: {policy}")


def train_rank_rows(yearly: pd.DataFrame, labels: list[str], train_years: list[int]) -> pd.DataFrame:
    """Rank candidate labels on prior OOS years with vectorized aggregation."""
    part = yearly[yearly["year"].isin(train_years) & yearly["label"].isin(labels)].copy()
    if part.empty:
        return pd.DataFrame()
    grouped = part.groupby("label", sort=False)
    ranked = grouped.agg(
        cumulative_return=("return", lambda x: float((1 + x).prod() - 1)),
        max_drawdown=("max_drawdown", "min"),
        loss_years=("return", lambda x: int((x < 0).sum())),
        max_abs_weight=("max_abs_weight", "max"),
    ).reset_index()
    ranked["annual_return"] = (1 + ranked["cumulative_return"]) ** (1 / len(train_years)) - 1
    ranked["calmar"] = ranked["annual_return"] / ranked["max_drawdown"].abs()
    ranked["score"] = (
        ranked["calmar"]
        + 0.75 * ranked["annual_return"]
        - 3.0 * (ranked["max_drawdown"].abs() - 0.08).clip(lower=0)
        - 0.05 * ranked["loss_years"]
        - 0.02 * ranked["max_abs_weight"]
    )
    eligible = ranked[
        (ranked["annual_return"] > 0)
        & (ranked["max_drawdown"] >= TARGET_MAX_DRAWDOWN)
        & (ranked["loss_years"] <= max(1, len(train_years) // 2))
    ].copy()
    if eligible.empty:
        eligible = ranked[ranked["max_drawdown"] >= -0.12].copy()
    if eligible.empty:
        eligible = ranked.copy()
    return eligible.sort_values(["score", "calmar", "annual_return"], ascending=False).reset_index(drop=True)


def build_selection_plan(
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    base_preset: str,
    policy: str,
    min_train_years: int,
    train_window: str,
    top_n: int,
    weight_method: str,
) -> pd.DataFrame:
    """Build selected labels and weights without rendering daily paths."""
    universe = summary[candidate_filter(summary, policy, base_preset)].copy()
    labels = universe["label"].tolist()
    base_label = f"{base_preset}_base_only"
    meta_label = f"{base_preset}_{policy}_min{min_train_years}_{train_window}_top{top_n}_{weight_method}_ensemble"
    rows = []

    for year in YEARS:
        if train_window == "expanding":
            train_years = [x for x in YEARS if x < year]
        elif train_window == "rolling3":
            train_years = [x for x in YEARS if year - 3 <= x < year]
        else:
            raise ValueError(f"Unsupported train_window: {train_window}")

        if len(train_years) < min_train_years:
            selected = pd.DataFrame({"label": [base_label], "score": [np.nan]})
            weights = np.array([1.0])
            reason = "base_until_min_train_years"
        else:
            ranked = train_rank_rows(yearly, labels, train_years)
            if ranked.empty:
                selected = pd.DataFrame({"label": [base_label], "score": [np.nan]})
                weights = np.array([1.0])
                reason = "base_no_ranked_candidates"
            else:
                selected = ranked.head(top_n).copy()
                weights = ensemble_weights(selected, weight_method)
                reason = "topn_prior_oos"

        rows.append(
            {
                "meta_label": meta_label,
                "test_year": year,
                "base_preset": base_preset,
                "policy": policy,
                "min_train_years": min_train_years,
                "train_window": train_window,
                "top_n": top_n,
                "weight_method": weight_method,
                "reason": reason,
                "train_years": ",".join(str(x) for x in train_years),
                "selected_labels": "|".join(selected["label"].tolist()),
                "selected_scores": "|".join("" if pd.isna(x) else f"{x:.6f}" for x in selected["score"]),
                "selected_weights": "|".join(f"{x:.6f}" for x in weights),
                "selected_count": int(len(selected)),
            }
        )
    return pd.DataFrame(rows)


def relative_config_by_name(name: str):
    """Find a relative-value config by serialized name."""
    for config in RELATIVE_CONFIGS:
        if config.name == name:
            return config
    raise ValueError(f"Unknown relative config: {name}")


def render_candidate_daily(summary: pd.DataFrame, labels: set[str]) -> pd.DataFrame:
    """Render only the candidate daily paths needed by selected plans."""
    returns = None
    base_cache: dict[str, pd.DataFrame] = {}
    sleeve_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    rows = []
    indexed = summary.drop_duplicates("label").set_index("label")

    for label in sorted(labels):
        if label.endswith("_base_only"):
            base_preset = label.replace("_base_only", "")
            if base_preset not in base_cache:
                base_cache[base_preset] = load_base_daily(base_preset)
            daily = base_cache[base_preset].copy()
            daily["ret"] = daily["base_ret"]
            daily["weight"] = daily["base_weight"].abs()
            daily["label"] = label
            rows.append(daily[["dt", "ret", "weight", "label"]])
            continue

        row = indexed.loc[label]
        base_preset = row["base_preset"]
        if base_preset not in base_cache:
            base_cache[base_preset] = load_base_daily(base_preset)
        base = base_cache[base_preset]

        if returns is None:
            returns = class_return_panel()
        key = (row["left"], row["right"], row["relative_config"])
        if key not in sleeve_cache:
            sleeve_cache[key] = relative_sleeve(
                returns,
                row["left"],
                row["right"],
                relative_config_by_name(row["relative_config"]),
            )
        joined = base.merge(sleeve_cache[key], on="dt", how="left").fillna({"sleeve_ret": 0.0, "sleeve_weight": 0.0})
        overlay_weight = float(row["overlay_weight"])
        if row["overlay_mode"] == "replacement":
            joined["ret"] = (1 - overlay_weight) * joined["base_ret"] + overlay_weight * joined["sleeve_ret"]
            joined["weight"] = (1 - overlay_weight) * joined["base_weight"].abs() + overlay_weight * joined[
                "sleeve_weight"
            ]
        elif row["overlay_mode"] == "additive":
            joined["ret"] = joined["base_ret"] + overlay_weight * joined["sleeve_ret"]
            joined["weight"] = joined["base_weight"].abs() + overlay_weight * joined["sleeve_weight"]
        else:
            raise ValueError(f"Unsupported overlay mode: {row['overlay_mode']}")
        joined["label"] = label
        rows.append(joined[["dt", "ret", "weight", "label"]])

    out = pd.concat(rows, ignore_index=True)
    out["dt"] = pd.to_datetime(out["dt"]).dt.tz_localize(None)
    return out


def stitch_from_plan(candidate_daily: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """Stitch one ensemble path from rendered candidate daily paths."""
    daily_parts = []
    for row in plan.itertuples(index=False):
        labels = row.selected_labels.split("|")
        weights = [float(x) for x in row.selected_weights.split("|")]
        parts = []
        for label, alloc in zip(labels, weights, strict=True):
            part = candidate_daily[
                (candidate_daily["label"] == label) & (candidate_daily["dt"].dt.year == row.test_year)
            ].copy()
            part["ret"] = alloc * part["ret"]
            part["weight"] = alloc * part["weight"].abs()
            parts.append(part[["dt", "ret", "weight"]])
        test = pd.concat(parts, ignore_index=True).groupby("dt", as_index=False).sum()
        test["meta_label"] = row.meta_label
        test["test_year"] = row.test_year
        daily_parts.append(test)
    stitched = pd.concat(daily_parts, ignore_index=True).sort_values("dt").reset_index(drop=True)
    stitched["nav"] = (1 + stitched["ret"]).cumprod()
    stitched["drawdown"] = stitched["nav"] / stitched["nav"].cummax() - 1
    return stitched


def summarize_path(daily: pd.DataFrame, meta_label: str) -> dict:
    """Summarize one stitched ensemble path."""
    stats = annualized_stats(daily["dt"].reset_index(drop=True), daily["ret"].to_numpy(), daily["weight"].to_numpy())
    pre = daily[daily["dt"].dt.year <= 2025]
    pre_stats = annualized_stats(pre["dt"].reset_index(drop=True), pre["ret"].to_numpy(), pre["weight"].to_numpy())
    stats.update(
        {
            "meta_label": meta_label,
            "pre2026_annual_return": pre_stats["annual_return"],
            "pre2026_max_drawdown": pre_stats["max_drawdown"],
            "pass_full_and_pre2026": bool(
                stats["annual_return"] >= TARGET_ANNUAL_RETURN
                and stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
                and pre_stats["annual_return"] >= TARGET_ANNUAL_RETURN
                and pre_stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
            ),
        }
    )
    return stats


def build_plans(summary: pd.DataFrame, yearly: pd.DataFrame) -> list[pd.DataFrame]:
    """Build all focused relative-value meta selection plans."""
    plans = []
    for base_preset in BASE_PRESETS:
        for policy in POLICIES:
            for min_train_years in [1, 2]:
                for train_window in ["rolling3"]:
                    for top_n in [3, 5]:
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
    """Collect selected underlying labels."""
    labels: set[str] = set()
    for plan in plans:
        for raw in plan["selected_labels"]:
            labels.update(str(raw).split("|"))
    return labels


def main() -> None:
    """Run relative-value meta ensemble validation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RELATIVE_OUTPUT_DIR / "relative_value_overlay_summary.csv")
    yearly_candidates = pd.read_csv(RELATIVE_OUTPUT_DIR / "relative_value_overlay_yearly.csv")

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

    stress_labels = list(
        summary_out.sort_values(
            ["pass_full_and_pre2026", "annual_return", "max_drawdown"],
            ascending=[False, False, False],
        )["meta_label"]
        .head(40)
        .drop_duplicates()
    )
    stress_rows = []
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

    selected_out.to_csv(OUTPUT_DIR / "relative_value_meta_selected_by_year.csv", index=False, encoding="utf-8-sig")
    summary_out.to_csv(OUTPUT_DIR / "relative_value_meta_summary.csv", index=False, encoding="utf-8-sig")
    top.head(40).to_csv(OUTPUT_DIR / "relative_value_meta_top40.csv", index=False, encoding="utf-8-sig")
    yearly_out.to_csv(OUTPUT_DIR / "relative_value_meta_yearly.csv", index=False, encoding="utf-8-sig")
    stress_out.to_csv(OUTPUT_DIR / "relative_value_meta_window_stress_top.csv", index=False, encoding="utf-8-sig")
    daily_out[daily_out["meta_label"].isin(top["meta_label"].head(12))].to_csv(
        OUTPUT_DIR / "relative_value_meta_daily_top12.csv", index=False, encoding="utf-8-sig"
    )

    print("Top relative-value meta ensembles:")
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
