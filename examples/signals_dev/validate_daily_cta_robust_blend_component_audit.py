# ruff: noqa: E402, I001
"""Component exposure audit for the robust 50% RV blend candidate.

The tradeability audit shows that ``robust_50rv_blend`` is the most resilient
final candidate under incremental trading costs. This script expands that path
from a single total portfolio weight into component-level exposure:

- annual-state core CTA sleeve;
- layered low-correlation class sleeves;
- relative-value class sleeves;
- a proxy ETF-level split for class sleeves.

The core CTA component is kept as one aggregate sleeve because the saved final
candidate path does not contain its exact three-symbol constituent weights.

Run:
    uv run --no-sync python examples/signals_dev/validate_daily_cta_robust_blend_component_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_daily_cta_layered_low_corr_overlay as layered
import validate_daily_cta_relative_value_overlay as rv

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_robust_blend_component_audit"
FINAL_DAILY = ROOT / "examples" / "results" / "daily_cta_final_candidate_audit" / "final_candidate_daily.csv"
LAYERED_SELECTED = (
    ROOT / "examples" / "results" / "daily_cta_layered_overlay_meta_ensemble" / "meta_ensemble_selected_by_year.csv"
)
RV_SELECTED = (
    ROOT / "examples" / "results" / "daily_cta_relative_value_meta_ensemble" / "relative_value_meta_selected_by_year.csv"
)
LAYERED_SUMMARY = (
    ROOT / "examples" / "results" / "daily_cta_layered_low_corr_overlay" / "layered_low_corr_overlay_summary.csv"
)
RV_SUMMARY = ROOT / "examples" / "results" / "daily_cta_relative_value_overlay" / "relative_value_overlay_summary.csv"

BOND_META_LABEL = "stress_balanced_inflation_conservative_min2_rolling3_top3_rank_ensemble"
RV_META_LABEL = "capital_efficient_rv_drop25_long_min1_rolling3_top5_equal_ensemble"
FINAL_LABEL = "robust_50rv_blend"
BLEND_ALLOCS = {"bond_meta": 0.50, "rv_meta": 0.50}
CORE_SYMBOL_PROXY = ("000852.XSHG", "000905.XSHG", "159915.XSHE")


def split_pipe(raw: str) -> list[str]:
    """Split a pipe-delimited plan cell."""
    if pd.isna(raw) or str(raw) == "":
        return []
    return str(raw).split("|")


def parse_float_pipe(raw: str) -> list[float]:
    """Split a pipe-delimited float plan cell."""
    return [float(x) for x in split_pipe(raw)]


def config_by_name(name: str) -> layered.SleeveConfig:
    """Resolve a layered sleeve config by serialized name."""
    for config in (*layered.SLEEVE_CONFIGS, layered.BOND_CONFIG, *layered.BOND_PLUS_CONFIGS):
        if config.name == name:
            return config
    raise ValueError(f"Unknown layered sleeve config: {name}")


def relative_config_by_name(name: str) -> rv.RelativeConfig:
    """Resolve a relative-value config by serialized name."""
    for config in rv.RELATIVE_CONFIGS:
        if config.name == name:
            return config
    raise ValueError(f"Unknown relative-value config: {name}")


def layered_sleeve_component_weights(
    class_returns: pd.DataFrame,
    library: dict[str, pd.DataFrame],
    budget_name: str,
    config_name: str,
) -> pd.DataFrame:
    """Return class-level absolute sleeve weights for one layered config."""
    budget = layered.CLASS_BUDGETS[budget_name]
    out = class_returns[["dt"]].copy()
    rows = []
    for class_name, class_weight in budget.items():
        if class_name == "bond_cash":
            cfg = layered.BOND_CONFIG
        elif class_name == "bond_plus":
            cfg = layered.BOND_PLUS_CONFIGS[0]
        else:
            cfg = config_by_name(config_name)
        sleeve_name = f"{class_name}_{cfg.name}"
        sleeve = library[sleeve_name][["dt", "ret", "weight"]].copy()
        sleeve["component"] = f"class_{class_name}"
        sleeve["class_name"] = class_name
        sleeve["component_weight"] = class_weight * sleeve["weight"].abs()
        sleeve["component_ret"] = class_weight * sleeve["ret"]
        rows.append(sleeve[["dt", "component", "class_name", "component_weight", "component_ret"]])
    if not rows:
        out["component"] = "none"
        out["class_name"] = "none"
        out["component_weight"] = 0.0
        out["component_ret"] = 0.0
        return out[["dt", "component", "class_name", "component_weight", "component_ret"]]
    return pd.concat(rows, ignore_index=True)


def relative_class_weights(returns: pd.DataFrame, left: str, right: str, config: rv.RelativeConfig) -> pd.DataFrame:
    """Return class-level absolute weights for one relative-value sleeve."""
    left_ret = returns[f"{left}_ret"].astype(float).to_numpy()
    right_ret = returns[f"{right}_ret"].astype(float).to_numpy()
    left_nav = pd.Series(np.cumprod(1 + left_ret), index=returns.index)
    right_nav = pd.Series(np.cumprod(1 + right_ret), index=returns.index)
    rel_score = (left_nav / right_nav) / (left_nav / right_nav).shift(config.lookback) - 1

    left_w = np.zeros(len(returns), dtype=float)
    right_w = np.zeros(len(returns), dtype=float)
    valid = rel_score.notna().to_numpy()
    enough = (
        (returns[f"{left}_constituents"].to_numpy(int) > 0)
        & (returns[f"{right}_constituents"].to_numpy(int) > 0)
        & valid
    )
    positive = rel_score.to_numpy(float) > config.threshold
    negative = rel_score.to_numpy(float) < -config.threshold

    if config.mode == "rotate_long":
        left_w[enough & positive] = 1.0
        right_w[enough & negative] = 1.0
    elif config.mode == "spread_ls":
        left_w[enough & positive] = 0.5
        right_w[enough & positive] = -0.5
        left_w[enough & negative] = -0.5
        right_w[enough & negative] = 0.5
    else:
        raise ValueError(f"Unsupported relative mode: {config.mode}")

    raw_ret = pd.Series(left_w).shift(1).fillna(0).to_numpy() * left_ret
    raw_ret += pd.Series(right_w).shift(1).fillna(0).to_numpy() * right_ret
    realized_vol = pd.Series(raw_ret).rolling(config.vol_window, min_periods=config.vol_window).std().to_numpy()
    realized_vol = realized_vol * np.sqrt(252)
    leverage = np.divide(config.target_vol, realized_vol, out=np.zeros_like(realized_vol), where=realized_vol > 0)
    leverage = np.nan_to_num(np.clip(leverage, 0, config.leverage_cap), nan=0.0, posinf=0.0, neginf=0.0)

    left_final = np.abs(left_w * leverage)
    right_final = np.abs(right_w * leverage)
    prev_left = np.r_[0.0, left_w[:-1] * leverage[:-1]]
    prev_right = np.r_[0.0, right_w[:-1] * leverage[:-1]]
    left_turnover = np.abs(left_w * leverage - prev_left)
    right_turnover = np.abs(right_w * leverage - prev_right)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "dt": returns["dt"],
                    "component": f"class_{left}",
                    "class_name": left,
                    "component_weight": left_final,
                    "component_ret": prev_left * left_ret - left_turnover * rv.FEE_RATE,
                }
            ),
            pd.DataFrame(
                {
                    "dt": returns["dt"],
                    "component": f"class_{right}",
                    "class_name": right,
                    "component_weight": right_final,
                    "component_ret": prev_right * right_ret - right_turnover * rv.FEE_RATE,
                }
            ),
        ],
        ignore_index=True,
    )


def expand_core_rows(base: pd.DataFrame, scale: float, source_side: str, selected_label: str, year: int) -> pd.DataFrame:
    """Build aggregate core CTA rows for one selected candidate."""
    part = base[base["dt"].dt.year == year].copy()
    return pd.DataFrame(
        {
            "dt": part["dt"],
            "source_side": source_side,
            "selected_label": selected_label,
            "component": f"core_cn_cta_{part['preset'].iloc[0]}",
            "class_name": "cn_cta_core",
            "component_weight": scale * part["base_weight"].abs().to_numpy(float),
            "component_ret": scale * part["base_ret"].to_numpy(float),
            "test_year": year,
        }
    )


def expand_layered_candidate(
    label: str,
    year: int,
    scale: float,
    source_side: str,
    summary: pd.DataFrame,
    class_returns: pd.DataFrame,
    library: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Expand one layered candidate label into class-level components."""
    row = summary.loc[label]
    base = layered.load_base_daily(row["base_preset"])
    rows = []
    if row["overlay_mode"] == "base_only":
        rows.append(expand_core_rows(base, scale, source_side, label, year))
    else:
        overlay_weight = float(row["overlay_weight"])
        core_scale = scale * (1 - overlay_weight if row["overlay_mode"] == "replacement" else 1.0)
        rows.append(expand_core_rows(base, core_scale, source_side, label, year))
        sleeve_components = layered_sleeve_component_weights(
            class_returns, library, row["budget"], row["sleeve_config"]
        )
        sleeve_components = sleeve_components[sleeve_components["dt"].dt.year == year].copy()
        sleeve_components["component_weight"] *= scale * overlay_weight
        sleeve_components["component_ret"] *= scale * overlay_weight
        sleeve_components["source_side"] = source_side
        sleeve_components["selected_label"] = label
        sleeve_components["test_year"] = year
        rows.append(
            sleeve_components[
                [
                    "dt",
                    "source_side",
                    "selected_label",
                    "component",
                    "class_name",
                    "component_weight",
                    "component_ret",
                    "test_year",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def expand_rv_candidate(
    label: str,
    year: int,
    scale: float,
    source_side: str,
    summary: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Expand one relative-value candidate label into class-level components."""
    rows = []
    if label.endswith("_base_only"):
        base_preset = label.replace("_base_only", "")
        base = rv.load_base_daily(base_preset)
        rows.append(expand_core_rows(base, scale, source_side, label, year))
    else:
        row = summary.loc[label]
        base = rv.load_base_daily(row["base_preset"])
        overlay_weight = float(row["overlay_weight"])
        core_scale = scale * (1 - overlay_weight if row["overlay_mode"] == "replacement" else 1.0)
        rows.append(expand_core_rows(base, core_scale, source_side, label, year))
        rel_components = relative_class_weights(
            returns, row["left"], row["right"], relative_config_by_name(row["relative_config"])
        )
        rel_components = rel_components[rel_components["dt"].dt.year == year].copy()
        rel_components["component_weight"] *= scale * overlay_weight
        rel_components["component_ret"] *= scale * overlay_weight
        rel_components["source_side"] = source_side
        rel_components["selected_label"] = label
        rel_components["test_year"] = year
        rows.append(
            rel_components[
                [
                    "dt",
                    "source_side",
                    "selected_label",
                    "component",
                    "class_name",
                    "component_weight",
                    "component_ret",
                    "test_year",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def expand_plan(
    plan: pd.DataFrame,
    source_side: str,
    blend_alloc: float,
    layered_summary: pd.DataFrame,
    rv_summary: pd.DataFrame,
    layered_class_returns: pd.DataFrame,
    layered_library: dict[str, pd.DataFrame],
    rv_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Expand one meta plan into daily component exposure rows."""
    rows = []
    for plan_row in plan.itertuples(index=False):
        labels = split_pipe(plan_row.selected_labels)
        weights = parse_float_pipe(plan_row.selected_weights)
        for label, selected_alloc in zip(labels, weights, strict=True):
            scale = blend_alloc * selected_alloc
            if source_side == "bond_meta":
                rows.append(
                    expand_layered_candidate(
                        label,
                        int(plan_row.test_year),
                        scale,
                        source_side,
                        layered_summary,
                        layered_class_returns,
                        layered_library,
                    )
                )
            elif source_side == "rv_meta":
                rows.append(
                    expand_rv_candidate(label, int(plan_row.test_year), scale, source_side, rv_summary, rv_returns)
                )
            else:
                raise ValueError(f"Unsupported source side: {source_side}")
    out = pd.concat(rows, ignore_index=True)
    out["component_weight"] = out["component_weight"].astype(float)
    return out


def proxy_asset_rows(component_daily: pd.DataFrame) -> pd.DataFrame:
    """Split class-level component weights into proxy ETF-level exposures."""
    rows = []
    symbol_map = {**layered.ASSET_CLASSES, **rv.CLASS_SYMBOLS, "cn_cta_core": CORE_SYMBOL_PROXY}
    for row in component_daily.itertuples(index=False):
        symbols = symbol_map.get(row.class_name, (row.class_name,))
        symbols = tuple(x for x in symbols if not str(x).startswith("__"))
        if not symbols:
            continue
        for symbol in symbols:
            rows.append(
                {
                    "dt": row.dt,
                    "proxy_asset": symbol,
                    "class_name": row.class_name,
                    "component": row.component,
                    "component_weight": row.component_weight / len(symbols),
                    "component_ret": row.component_ret / len(symbols),
                    "test_year": row.test_year,
                }
            )
    return pd.DataFrame(rows)


def exposure_summary(data: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Summarize exposure and turnover for components or proxy assets."""
    daily = data.groupby(["dt", group_col], as_index=False)["component_weight"].sum()
    rows = []
    for name, group in daily.groupby(group_col, sort=False):
        group = group.sort_values("dt").copy()
        weight = group["component_weight"].abs()
        turnover = weight.diff().abs().fillna(weight)
        rows.append(
            {
                group_col: name,
                "avg_weight": float(weight.mean()),
                "weight_p95": float(weight.quantile(0.95)),
                "weight_p99": float(weight.quantile(0.99)),
                "max_weight": float(weight.max()),
                "days_gt_0_25": int((weight > 0.25).sum()),
                "days_gt_0_50": int((weight > 0.50).sum()),
                "days_gt_1_00": int((weight > 1.00).sum()),
                "turnover_sum": float(turnover.sum()),
                "turnover_p95": float(turnover.quantile(0.95)),
                "max_daily_turnover": float(turnover.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["max_weight", "avg_weight"], ascending=False).reset_index(drop=True)


def reconcile_total(component_daily: pd.DataFrame) -> pd.DataFrame:
    """Compare expanded total exposure with the saved final candidate weight."""
    expanded = component_daily.groupby("dt", as_index=False).agg(
        component_weight=("component_weight", "sum"),
        component_ret=("component_ret", "sum"),
    )
    final = pd.read_csv(FINAL_DAILY, parse_dates=["dt"])
    final["dt"] = pd.to_datetime(final["dt"]).dt.tz_localize(None)
    final = final[final["label"].eq(FINAL_LABEL)][["dt", "ret", "weight"]].copy()
    check = expanded.merge(final, on="dt", how="inner")
    check["weight_abs_diff"] = (check["component_weight"] - check["weight"].abs()).abs()
    check["ret_abs_diff"] = (check["component_ret"] - check["ret"]).abs()
    return pd.DataFrame(
        [
            {
                "days": int(len(check)),
                "expanded_weight_mean": float(check["component_weight"].mean()),
                "saved_weight_mean": float(check["weight"].abs().mean()),
                "max_weight_abs_diff": float(check["weight_abs_diff"].max()),
                "p99_weight_abs_diff": float(check["weight_abs_diff"].quantile(0.99)),
                "expanded_ret_sum": float(check["component_ret"].sum()),
                "saved_ret_sum": float(check["ret"].sum()),
                "max_ret_abs_diff": float(check["ret_abs_diff"].max()),
                "p99_ret_abs_diff": float(check["ret_abs_diff"].quantile(0.99)),
            }
        ]
    )


def save_component_plot(component_daily: pd.DataFrame) -> None:
    """Save stacked component exposure plot."""
    import plotly.graph_objects as go

    wide = (
        component_daily.groupby(["dt", "component"], as_index=False)["component_weight"].sum()
        .pivot(index="dt", columns="component", values="component_weight")
        .fillna(0.0)
    )
    fig = go.Figure()
    for component in wide.columns:
        fig.add_trace(go.Scatter(x=wide.index, y=wide[component], mode="lines", stackgroup="one", name=component))
    fig.update_layout(
        title="robust_50rv_blend component exposure",
        xaxis_title="date",
        yaxis_title="absolute exposure",
        template="plotly_white",
    )
    fig.write_html(OUTPUT_DIR / "robust_50rv_component_exposure.html", include_plotlyjs="cdn")


def main() -> None:
    """Run component exposure audit."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layered_summary = pd.read_csv(LAYERED_SUMMARY).drop_duplicates("label").set_index("label")
    rv_summary = pd.read_csv(RV_SUMMARY).drop_duplicates("label").set_index("label")
    bond_plan = pd.read_csv(LAYERED_SELECTED)
    bond_plan = bond_plan[bond_plan["meta_label"].eq(BOND_META_LABEL)].copy()
    rv_plan = pd.read_csv(RV_SELECTED)
    rv_plan = rv_plan[rv_plan["meta_label"].eq(RV_META_LABEL)].copy()

    layered_prices = layered.load_price_panel()
    layered_class_returns = layered.class_return_panel(layered_prices)
    layered_library = layered.build_sleeve_library(layered_class_returns)
    rv_returns = rv.class_return_panel()

    bond_components = expand_plan(
        bond_plan,
        "bond_meta",
        BLEND_ALLOCS["bond_meta"],
        layered_summary,
        rv_summary,
        layered_class_returns,
        layered_library,
        rv_returns,
    )
    rv_components = expand_plan(
        rv_plan,
        "rv_meta",
        BLEND_ALLOCS["rv_meta"],
        layered_summary,
        rv_summary,
        layered_class_returns,
        layered_library,
        rv_returns,
    )
    component_daily = pd.concat([bond_components, rv_components], ignore_index=True)
    proxy_daily = proxy_asset_rows(component_daily)

    component_summary = exposure_summary(component_daily, "component")
    proxy_summary = exposure_summary(proxy_daily, "proxy_asset")
    reconciliation = reconcile_total(component_daily)
    selection_plan = pd.concat(
        [
            bond_plan.assign(source_side="bond_meta", blend_alloc=BLEND_ALLOCS["bond_meta"]),
            rv_plan.assign(source_side="rv_meta", blend_alloc=BLEND_ALLOCS["rv_meta"]),
        ],
        ignore_index=True,
    )

    component_daily.to_csv(OUTPUT_DIR / "robust_50rv_component_daily.csv", index=False, encoding="utf-8-sig")
    component_summary.to_csv(OUTPUT_DIR / "robust_50rv_component_summary.csv", index=False, encoding="utf-8-sig")
    proxy_daily.to_csv(OUTPUT_DIR / "robust_50rv_proxy_asset_daily.csv", index=False, encoding="utf-8-sig")
    proxy_summary.to_csv(OUTPUT_DIR / "robust_50rv_proxy_asset_summary.csv", index=False, encoding="utf-8-sig")
    reconciliation.to_csv(OUTPUT_DIR / "robust_50rv_reconciliation.csv", index=False, encoding="utf-8-sig")
    selection_plan.to_csv(OUTPUT_DIR / "robust_50rv_selection_plan.csv", index=False, encoding="utf-8-sig")
    save_component_plot(component_daily)

    print("Reconciliation:")
    print(reconciliation.to_string(index=False))
    print("\nTop components:")
    print(component_summary.head(12).to_string(index=False))
    print("\nTop proxy assets:")
    print(proxy_summary.head(15).to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
