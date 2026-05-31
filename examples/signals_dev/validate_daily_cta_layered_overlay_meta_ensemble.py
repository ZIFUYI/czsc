# ruff: noqa: E402, I001
"""Top-N meta walk-forward ensemble for layered low-correlation overlays.

This is stricter than selecting one overlay rule per test year. Each year uses
only prior out-of-sample years to rank candidates, then blends the top-N
eligible candidates for the next test year. A robust edge should survive this
selection-neighborhood averaging better than a single overfit rule.

Run:
    .venv/bin/python examples/signals_dev/validate_daily_cta_layered_overlay_meta_ensemble.py
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
    OUTPUT_DIR as LAYERED_OUTPUT_DIR,
    SLEEVE_CONFIGS,
    TARGET_ANNUAL_RETURN,
    TARGET_MAX_DRAWDOWN,
    build_sleeve_library,
    class_return_panel,
    combine_sleeves,
    evaluate_window_stress,
    load_base_daily,
    load_price_panel,
)
from validate_daily_cta_layered_overlay_meta_walk_forward import candidate_filter  # noqa: E402
from validate_daily_cta_walk_forward import annualized_stats, summarize_yearly  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_layered_overlay_meta_ensemble"
YEARS = tuple(range(2020, 2027))
POLICIES = (
    "gold_conservative",
    "commodity_conservative",
    "energy_conservative",
    "weak_window_conservative",
    "bond_conservative",
    "inflation_conservative",
)


def train_rank_rows(yearly: pd.DataFrame, labels: list[str], train_years: list[int]) -> pd.DataFrame:
    """Rank candidate labels on prior out-of-sample years."""
    rows = []
    for label in labels:
        part = yearly[(yearly["label"] == label) & (yearly["year"].isin(train_years))].copy()
        if part.empty:
            continue
        cumulative = float((1 + part["return"]).prod() - 1)
        annual_return = float((1 + cumulative) ** (1 / len(part)) - 1)
        max_drawdown = float(part["max_drawdown"].min())
        loss_years = int((part["return"] < 0).sum())
        calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else float("nan")
        max_abs_weight = float(part["max_abs_weight"].max())
        score = (
            calmar
            + 0.75 * annual_return
            - 3.0 * max(0.0, abs(max_drawdown) - 0.08)
            - 0.05 * loss_years
            - 0.02 * max_abs_weight
        )
        rows.append(
            {
                "label": label,
                "annual_return": annual_return,
                "cumulative_return": cumulative,
                "max_drawdown": max_drawdown,
                "calmar": calmar,
                "loss_years": loss_years,
                "max_abs_weight": max_abs_weight,
                "score": score,
            }
        )
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
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


def ensemble_weights(selected: pd.DataFrame, method: str) -> np.ndarray:
    """Build top-N ensemble weights."""
    if method == "equal":
        return np.ones(len(selected), dtype=float) / len(selected)
    if method == "rank":
        raw = 1 / (np.arange(len(selected), dtype=float) + 1)
        return raw / raw.sum()
    raise ValueError(f"Unsupported ensemble weight method: {method}")


def stitch_ensemble_path(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    base_preset: str,
    policy: str,
    min_train_years: int,
    train_window: str,
    top_n: int,
    weight_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stitch one meta ensemble path."""
    universe = summary[candidate_filter(summary, policy, base_preset)].copy()
    labels = universe["label"].tolist()
    base_label = f"{base_preset}_base_only"
    meta_label = f"{base_preset}_{policy}_min{min_train_years}_{train_window}_top{top_n}_{weight_method}_ensemble"

    daily_parts = []
    selected_rows = []
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

        test_rows = []
        for alloc, label in zip(weights, selected["label"], strict=True):
            part = daily[(daily["label"] == label) & (daily["dt"].dt.year == year)].copy()
            part["ret"] = alloc * part["ret"]
            part["weight"] = alloc * part["weight"].abs()
            test_rows.append(part[["dt", "ret", "weight"]])
        test = pd.concat(test_rows, ignore_index=True).groupby("dt", as_index=False).sum()
        test["meta_label"] = meta_label
        test["test_year"] = year
        daily_parts.append(test)

        selected_rows.append(
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

    stitched = pd.concat(daily_parts, ignore_index=True).sort_values("dt").reset_index(drop=True)
    stitched["nav"] = (1 + stitched["ret"]).cumprod()
    stitched["drawdown"] = stitched["nav"] / stitched["nav"].cummax() - 1
    return stitched, pd.DataFrame(selected_rows)


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
    """Build selected labels and weights for one ensemble without daily data."""
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


def config_by_name(name: str):
    """Find a sleeve config by serialized name."""
    for config in SLEEVE_CONFIGS:
        if config.name == name:
            return config
    raise ValueError(f"Unknown sleeve config: {name}")


def render_candidate_daily(summary: pd.DataFrame, labels: set[str]) -> pd.DataFrame:
    """Render only the candidate daily paths needed by selection plans."""
    price_panel = None
    class_returns = None
    library = None
    base_cache: dict[str, pd.DataFrame] = {}
    sleeve_cache: dict[tuple[str, str], pd.DataFrame] = {}
    rows = []

    indexed = summary.drop_duplicates("label").set_index("label")
    for label in sorted(labels):
        row = indexed.loc[label]
        base_preset = row["base_preset"]
        if base_preset not in base_cache:
            base_cache[base_preset] = load_base_daily(base_preset)
        base = base_cache[base_preset]

        if row["overlay_mode"] == "base_only":
            daily = base.copy()
            daily["ret"] = daily["base_ret"]
            daily["weight"] = daily["base_weight"].abs()
            daily["label"] = label
            rows.append(daily[["dt", "ret", "weight", "label"]])
            continue

        if price_panel is None:
            price_panel = load_price_panel()
            class_returns = class_return_panel(price_panel)
            library = build_sleeve_library(class_returns)

        key = (row["budget"], row["sleeve_config"])
        if key not in sleeve_cache:
            sleeve_cache[key] = combine_sleeves(
                class_returns, library, row["budget"], config_by_name(row["sleeve_config"])
            )
        joined = base.merge(sleeve_cache[key], on="dt", how="inner")
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


def save_nav_plot(daily: pd.DataFrame, labels: list[str]) -> None:
    """Save net-value plot for top ensemble candidates."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for label in labels:
        part = daily[daily["meta_label"] == label]
        fig.add_trace(go.Scatter(x=part["dt"], y=part["nav"], mode="lines", name=label))
    fig.update_layout(
        title="Layered overlay top-N meta ensemble candidates",
        xaxis_title="date",
        yaxis_title="net value",
        template="plotly_white",
    )
    fig.write_html(OUTPUT_DIR / "meta_ensemble_nav.html", include_plotlyjs="cdn")


def main() -> None:
    """Run top-N meta ensemble validation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(LAYERED_OUTPUT_DIR / "layered_low_corr_overlay_summary.csv")
    yearly_candidates = pd.read_csv(LAYERED_OUTPUT_DIR / "layered_low_corr_overlay_yearly.csv")

    plans = []
    needed_labels: set[str] = set()
    for base_preset in BASE_PRESETS:
        for policy in POLICIES:
            for min_train_years in [1, 2, 3]:
                for train_window in ["expanding", "rolling3"]:
                    for top_n in [3, 5, 10]:
                        for weight_method in ["equal", "rank"]:
                            plan = build_selection_plan(
                                summary,
                                yearly_candidates,
                                base_preset,
                                policy,
                                min_train_years,
                                train_window,
                                top_n,
                                weight_method,
                            )
                            plans.append(plan)
                            for labels in plan["selected_labels"]:
                                needed_labels.update(labels.split("|"))

    selected_out = pd.concat(plans, ignore_index=True)
    candidate_daily = render_candidate_daily(summary, needed_labels)

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
    top = summary_out.sort_values(
        ["pass_full_and_pre2026", "annual_return", "max_drawdown"],
        ascending=[False, False, False],
    )
    daily_out = pd.concat(daily_rows, ignore_index=True)
    yearly_out = pd.concat(yearly_rows, ignore_index=True)

    stress_rows = []
    stress_labels = list(
        pd.concat(
            [
                top.head(30),
                summary_out.sort_values(["max_drawdown", "annual_return"], ascending=[False, False]).head(20),
            ],
            ignore_index=True,
        )["meta_label"].drop_duplicates()
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

    summary_out.to_csv(OUTPUT_DIR / "meta_ensemble_summary.csv", index=False, encoding="utf-8-sig")
    top.head(20).to_csv(OUTPUT_DIR / "meta_ensemble_top20.csv", index=False, encoding="utf-8-sig")
    selected_out.to_csv(OUTPUT_DIR / "meta_ensemble_selected_by_year.csv", index=False, encoding="utf-8-sig")
    yearly_out.to_csv(OUTPUT_DIR / "meta_ensemble_yearly.csv", index=False, encoding="utf-8-sig")
    stress_out.to_csv(OUTPUT_DIR / "meta_ensemble_window_stress_top.csv", index=False, encoding="utf-8-sig")
    daily_top = daily_out[daily_out["meta_label"].isin(top["meta_label"].head(12))].copy()
    daily_top.to_csv(OUTPUT_DIR / "meta_ensemble_daily_top12.csv", index=False, encoding="utf-8-sig")
    save_nav_plot(daily_top, top["meta_label"].head(8).tolist())

    print("Top meta ensembles:")
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
