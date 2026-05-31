# ruff: noqa: E402, I001
"""Bond-plus sleeve parameter sensitivity for layered meta ensembles.

The current best layered overlay candidates use ``bond_plus`` through a fixed
implementation detail in ``combine_sleeves``: the bond-plus sleeve is always
``long_VW60_TV8_C3``. This script keeps the already-generated meta selection
plans frozen, then perturbs only that bond-plus execution sleeve to check
whether the result depends on one leverage / target-volatility point.

Run:
    uv run --no-sync python examples/signals_dev/validate_daily_cta_bond_plus_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_daily_cta_layered_low_corr_overlay import (  # noqa: E402
    BOND_CONFIG,
    CLASS_BUDGETS,
    OUTPUT_DIR as LAYERED_OUTPUT_DIR,
    TARGET_ANNUAL_RETURN,
    TARGET_MAX_DRAWDOWN,
    SleeveConfig,
    build_sleeve_library,
    class_return_panel,
    evaluate_window_stress,
    load_base_daily,
    load_price_panel,
    sleeve_returns,
)
from validate_daily_cta_layered_overlay_meta_ensemble import (  # noqa: E402
    OUTPUT_DIR as META_OUTPUT_DIR,
    config_by_name,
    stitch_from_plan,
    summarize_path,
)
from validate_daily_cta_walk_forward import summarize_yearly  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_bond_plus_sensitivity"
TOP_META_COUNT = 20
BOND_PLUS_VARIANTS = (
    SleeveConfig("long", 60, 0.04, 1.0),
    SleeveConfig("long", 60, 0.04, 2.0),
    SleeveConfig("long", 60, 0.04, 3.0),
    SleeveConfig("long", 60, 0.06, 1.0),
    SleeveConfig("long", 60, 0.06, 2.0),
    SleeveConfig("long", 60, 0.06, 3.0),
    SleeveConfig("long", 60, 0.08, 1.0),
    SleeveConfig("long", 60, 0.08, 2.0),
    SleeveConfig("long", 60, 0.08, 3.0),
    SleeveConfig("long", 60, 0.10, 2.0),
    SleeveConfig("long", 60, 0.10, 3.0),
    SleeveConfig("long", 60, 0.12, 2.0),
    SleeveConfig("long", 60, 0.12, 3.0),
)


def combine_sleeves_with_bond_variant(
    class_returns: pd.DataFrame,
    library: dict[str, pd.DataFrame],
    budget_name: str,
    config: SleeveConfig,
    bond_plus_config: SleeveConfig,
) -> pd.DataFrame:
    """Combine class sleeves while replacing only the bond-plus sleeve."""
    budget = CLASS_BUDGETS[budget_name]
    out = class_returns[["dt"]].copy()
    out["sleeve_ret"] = 0.0
    out["sleeve_weight"] = 0.0
    sleeve_names = []
    for class_name, class_weight in budget.items():
        if class_name == "bond_cash":
            cfg = BOND_CONFIG
        elif class_name == "bond_plus":
            cfg = bond_plus_config
        else:
            cfg = config
        sleeve_name = f"{class_name}_{cfg.name}"
        sleeve = library[sleeve_name]
        out["sleeve_ret"] += class_weight * sleeve["ret"].to_numpy(float)
        out["sleeve_weight"] += class_weight * sleeve["weight"].abs().to_numpy(float)
        sleeve_names.append(sleeve_name)
    out["sleeves"] = "|".join(sleeve_names)
    return out


def render_candidate_daily_with_bond_variant(
    summary: pd.DataFrame,
    labels: set[str],
    bond_plus_config: SleeveConfig,
) -> pd.DataFrame:
    """Render candidate daily paths under one bond-plus parameter variant."""
    price_panel = None
    class_returns = None
    library = None
    base_cache: dict[str, pd.DataFrame] = {}
    sleeve_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
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
            variant_sleeve = sleeve_returns(class_returns, "bond_plus", bond_plus_config)
            library[variant_sleeve["sleeve"].iloc[0]] = variant_sleeve

        key = (row["budget"], row["sleeve_config"], bond_plus_config.name)
        if key not in sleeve_cache:
            sleeve_cache[key] = combine_sleeves_with_bond_variant(
                class_returns,
                library,
                row["budget"],
                config_by_name(row["sleeve_config"]),
                bond_plus_config,
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


def selected_top_plans() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the current top meta labels and their frozen selection plans."""
    top = pd.read_csv(META_OUTPUT_DIR / "meta_ensemble_top20.csv")
    selected = pd.read_csv(META_OUTPUT_DIR / "meta_ensemble_selected_by_year.csv")
    meta_labels = top["meta_label"].head(TOP_META_COUNT).tolist()
    plans = selected[selected["meta_label"].isin(meta_labels)].copy()
    return top[top["meta_label"].isin(meta_labels)].copy(), plans


def needed_candidate_labels(plans: pd.DataFrame) -> set[str]:
    """Collect every underlying candidate label needed by frozen plans."""
    labels: set[str] = set()
    for raw in plans["selected_labels"]:
        labels.update(str(raw).split("|"))
    return labels


def main() -> None:
    """Run bond-plus parameter sensitivity validation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(LAYERED_OUTPUT_DIR / "layered_low_corr_overlay_summary.csv")
    top, plans = selected_top_plans()
    labels = needed_candidate_labels(plans)

    summary_rows = []
    yearly_rows = []
    selected_rows = []
    stress_rows = []
    daily_rows = []

    for variant in BOND_PLUS_VARIANTS:
        candidate_daily = render_candidate_daily_with_bond_variant(summary, labels, variant)
        for meta_label, plan in plans.groupby("meta_label", sort=False):
            path = stitch_from_plan(candidate_daily, plan)
            stats = summarize_path(path, meta_label)
            first = plan.iloc[0]
            stats.update(
                {
                    "bond_plus_config": variant.name,
                    "bond_plus_target_vol": variant.target_vol,
                    "bond_plus_leverage_cap": variant.leverage_cap,
                    "base_preset": first["base_preset"],
                    "policy": first["policy"],
                    "min_train_years": first["min_train_years"],
                    "train_window": first["train_window"],
                    "top_n": first["top_n"],
                    "weight_method": first["weight_method"],
                }
            )
            summary_rows.append(stats)

            yearly = summarize_yearly(path[["dt", "ret", "weight"]])
            yearly["meta_label"] = meta_label
            yearly["bond_plus_config"] = variant.name
            yearly_rows.append(yearly)

            stress = evaluate_window_stress(path.rename(columns={"meta_label": "label"}), meta_label)
            stress["meta_label"] = meta_label
            stress["bond_plus_config"] = variant.name
            stress_rows.append(stress)

            selected = plan.copy()
            selected["bond_plus_config"] = variant.name
            selected_rows.append(selected)

            if meta_label in set(top["meta_label"].head(5)):
                keep = path.copy()
                keep["bond_plus_config"] = variant.name
                daily_rows.append(keep)

    summary_out = pd.DataFrame(summary_rows)
    stress_out = pd.concat(stress_rows, ignore_index=True)
    stress_agg = (
        stress_out.groupby(["bond_plus_config", "meta_label"])
        .agg(
            stress_windows=("pass_20_10", "size"),
            stress_pass_rate=("pass_20_10", "mean"),
            stress_annual_median=("annual_return", "median"),
            stress_mdd_median=("max_drawdown", "median"),
        )
        .reset_index()
    )
    summary_out = summary_out.merge(stress_agg, on=["bond_plus_config", "meta_label"], how="left")
    summary_out["pass_full_pre2026_stress_median"] = (
        summary_out["pass_full_and_pre2026"]
        & (summary_out["stress_annual_median"] >= TARGET_ANNUAL_RETURN)
        & (summary_out["stress_mdd_median"] >= TARGET_MAX_DRAWDOWN)
    )
    top_out = summary_out.sort_values(
        [
            "pass_full_pre2026_stress_median",
            "pass_full_and_pre2026",
            "stress_pass_rate",
            "annual_return",
            "max_drawdown",
        ],
        ascending=[False, False, False, False, False],
    )

    variant_agg = (
        summary_out.groupby("bond_plus_config")
        .agg(
            meta_count=("meta_label", "size"),
            pass_full_pre2026=("pass_full_and_pre2026", "mean"),
            pass_full_pre2026_stress_median=("pass_full_pre2026_stress_median", "mean"),
            best_annual_return=("annual_return", "max"),
            best_max_drawdown=("max_drawdown", "max"),
            median_annual_return=("annual_return", "median"),
            median_max_drawdown=("max_drawdown", "median"),
            median_stress_pass_rate=("stress_pass_rate", "median"),
        )
        .reset_index()
        .sort_values(
            ["pass_full_pre2026_stress_median", "best_annual_return", "best_max_drawdown"],
            ascending=[False, False, False],
        )
    )

    summary_out.to_csv(OUTPUT_DIR / "bond_plus_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    top_out.head(50).to_csv(OUTPUT_DIR / "bond_plus_sensitivity_top50.csv", index=False, encoding="utf-8-sig")
    variant_agg.to_csv(OUTPUT_DIR / "bond_plus_sensitivity_variant_aggregate.csv", index=False, encoding="utf-8-sig")
    pd.concat(yearly_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "bond_plus_sensitivity_yearly.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(selected_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "bond_plus_sensitivity_selected_by_year.csv", index=False, encoding="utf-8-sig"
    )
    stress_out.to_csv(OUTPUT_DIR / "bond_plus_sensitivity_window_stress.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "bond_plus_sensitivity_daily_top5.csv", index=False, encoding="utf-8-sig"
    )

    print("Variant aggregate:")
    print(variant_agg.to_string(index=False))
    print("\nTop sensitivity rows:")
    print(
        top_out[
            [
                "bond_plus_config",
                "meta_label",
                "annual_return",
                "max_drawdown",
                "pre2026_annual_return",
                "pre2026_max_drawdown",
                "stress_pass_rate",
                "pass_full_and_pre2026",
                "pass_full_pre2026_stress_median",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
