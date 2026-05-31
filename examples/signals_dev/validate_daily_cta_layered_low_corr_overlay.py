"""Layered low-correlation overlay on top of the current daily CTA candidate.

The previous low-correlation-pool experiment put equity indices, gold, overseas
equity and bond / money ETFs into one cross-sectional rank. That failed because
asset-count expansion did not equal risk-source expansion. This script tests a
more conservative structure:

- keep the original three-symbol cross-sectional CTA as the core sleeve;
- build independent class-level sleeves for Hong Kong equity, overseas equity,
  gold and bond / money ETFs;
- combine the class sleeves with fixed asset-class budgets;
- evaluate both replacement and additive overlays;
- report full, pre-2026 and window-stress robustness.

Run:
    .venv/bin/python examples/signals_dev/validate_daily_cta_layered_low_corr_overlay.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_daily_cta_walk_forward import annualized_stats, summarize_yearly  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LOW_CORR_DIR = ROOT / "examples" / "results" / "daily_holding_low_corr_pool"
COMMODITY_DIR = ROOT / "examples" / "results" / "daily_holding_commodity_pool"
BOND_DIR = ROOT / "examples" / "results" / "daily_holding_bond_pool"
BASE_DAILY_FILE = (
    ROOT
    / "examples"
    / "results"
    / "daily_cta_cross_section_walk_forward"
    / "annual_state_scale"
    / "annual_state_scale_daily.csv"
)
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_layered_low_corr_overlay"

FEE_RATE = 0.0002
TARGET_ANNUAL_RETURN = 0.20
TARGET_MAX_DRAWDOWN = -0.10

ASSET_CLASSES = {
    "hk": ("510900.XSHG", "159920.XSHE"),
    "overseas": ("513100.XSHG", "513500.XSHG", "513030.XSHG"),
    "gold": ("518880.XSHG", "159934.XSHE"),
    "bond_cash": ("511010.XSHG", "511220.XSHG", "511880.XSHG"),
    "bond_plus": ("511020.XSHG", "511030.XSHG", "511260.XSHG", "511270.XSHG", "511660.XSHG"),
    "convertible_bond": ("511180.XSHG", "511380.XSHG"),
    "energy": ("162411.XSHE", "163208.XSHE", "159930.XSHE", "159945.XSHE"),
    "commodity": ("160216.XSHE", "161815.XSHE", "165513.XSHE"),
    "resource": ("510410.XSHG", "159944.XSHE", "512400.XSHG", "161226.XSHE"),
    "agri": ("159985.XSHE",),
}
ASSET_FILE_NAMES = {
    symbol: LOW_CORR_DIR / f"{symbol.replace('.', '_')}_daily_20100101_20260529.csv"
    for symbols in {
        key: value for key, value in ASSET_CLASSES.items() if key in {"hk", "overseas", "gold", "bond_cash"}
    }.values()
    for symbol in symbols
}
ASSET_FILE_NAMES.update(
    {
        symbol: COMMODITY_DIR / f"{symbol.replace('.', '_')}_daily_20100101_20260529.csv"
        for symbols in {
            key: value for key, value in ASSET_CLASSES.items() if key in {"energy", "commodity", "resource", "agri"}
        }.values()
        for symbol in symbols
    }
)
ASSET_FILE_NAMES.update(
    {
        symbol: BOND_DIR / f"{symbol.replace('.', '_')}_daily_20100101_20260529.csv"
        for symbols in {
            key: value for key, value in ASSET_CLASSES.items() if key in {"bond_plus", "convertible_bond"}
        }.values()
        for symbol in symbols
    }
)

CLASS_BUDGETS = {
    "gold": {"gold": 1.0},
    "gold_bond": {"gold": 0.75, "bond_cash": 0.25},
    "overseas_gold": {"overseas": 0.5, "gold": 0.5},
    "overseas_gold_bond": {"overseas": 0.35, "gold": 0.45, "bond_cash": 0.20},
    "energy": {"energy": 1.0},
    "commodity": {"commodity": 1.0},
    "resource": {"resource": 1.0},
    "agri": {"agri": 1.0},
    "bond_plus": {"bond_plus": 1.0},
    "bond_plus_gold": {"bond_plus": 0.45, "gold": 0.55},
    "bond_plus_commodity": {"bond_plus": 0.40, "commodity": 0.35, "gold": 0.25},
    "convertible_bond": {"convertible_bond": 1.0},
    "commodity_gold": {"commodity": 0.45, "energy": 0.25, "gold": 0.30},
    "resource_gold": {"resource": 0.45, "energy": 0.25, "gold": 0.30},
    "inflation_pack": {"commodity": 0.30, "energy": 0.25, "resource": 0.25, "gold": 0.20},
    "risk_parity_like": {"hk": 0.2, "overseas": 0.3, "gold": 0.4, "bond_cash": 0.1},
    "defensive": {"overseas": 0.25, "gold": 0.35, "bond_cash": 0.40},
    "growth": {"hk": 0.25, "overseas": 0.45, "gold": 0.30},
}
OVERLAY_WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.30)
ADD_WEIGHTS = (0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)
BASE_PRESETS = ("capital_efficient", "stress_balanced")


@dataclass(frozen=True)
class SleeveConfig:
    """One class-level timing and risk configuration."""

    trend: str
    vol_window: int
    target_vol: float
    leverage_cap: float

    @property
    def name(self) -> str:
        return f"{self.trend}_VW{self.vol_window}_TV{int(self.target_vol * 100)}_C{self.leverage_cap:g}"


SLEEVE_CONFIGS = (
    SleeveConfig("ma60", 40, 0.08, 1.0),
    SleeveConfig("ma120", 40, 0.08, 1.0),
    SleeveConfig("mom60", 40, 0.08, 1.0),
    SleeveConfig("mom120", 40, 0.08, 1.0),
    SleeveConfig("dual_ma60_120", 40, 0.08, 1.0),
    SleeveConfig("ma120", 60, 0.10, 1.2),
    SleeveConfig("mom120", 60, 0.10, 1.2),
)
BOND_CONFIG = SleeveConfig("long", 60, 0.04, 1.0)
BOND_PLUS_CONFIGS = (
    SleeveConfig("long", 60, 0.08, 3.0),
    SleeveConfig("long", 60, 0.12, 3.0),
    SleeveConfig("ma60", 60, 0.08, 3.0),
    SleeveConfig("ma120", 60, 0.08, 3.0),
    SleeveConfig("mom60", 60, 0.08, 3.0),
    SleeveConfig("mom120", 60, 0.08, 3.0),
)


def load_base_daily(preset: str) -> pd.DataFrame:
    """Load the annual-state scaled core CTA daily path."""
    daily = pd.read_csv(BASE_DAILY_FILE, parse_dates=["dt"])
    daily = daily[daily["preset"] == preset].sort_values("dt").reset_index(drop=True)
    if daily.empty:
        raise ValueError(f"No rows found for preset={preset!r} in {BASE_DAILY_FILE}")
    return daily[["dt", "ret", "weight", "test_year", "preset"]].rename(
        columns={"ret": "base_ret", "weight": "base_weight"}
    )


def load_price_panel() -> pd.DataFrame:
    """Load aligned low-correlation ETF close prices."""
    frames = []
    for symbol, file_csv in ASSET_FILE_NAMES.items():
        data = pd.read_csv(file_csv, parse_dates=["dt"])
        data["dt"] = pd.to_datetime(data["dt"]).dt.tz_localize(None)
        frames.append(data[["dt", "close"]].rename(columns={"close": symbol}))
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on="dt", how="inner")
    return panel.sort_values("dt").reset_index(drop=True)


def class_return_panel(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Build equal-weight class returns from constituent ETF returns."""
    returns = price_panel[["dt"]].copy()
    for class_name, symbols in ASSET_CLASSES.items():
        asset_rets = price_panel[list(symbols)].pct_change(fill_method=None).fillna(0.0)
        returns[class_name] = asset_rets.mean(axis=1)
    return returns


def trend_signal(index_nav: pd.Series, config: SleeveConfig) -> pd.Series:
    """Generate a long/flat class-level trend signal."""
    if config.trend == "long":
        return pd.Series(1.0, index=index_nav.index)
    if config.trend.startswith("ma"):
        window = int(config.trend.replace("ma", ""))
        ma = index_nav.rolling(window, min_periods=window).mean()
        return (index_nav > ma).astype(float)
    if config.trend.startswith("mom"):
        window = int(config.trend.replace("mom", ""))
        return (index_nav / index_nav.shift(window) - 1 > 0).astype(float)
    if config.trend == "dual_ma60_120":
        ma60 = index_nav.rolling(60, min_periods=60).mean()
        ma120 = index_nav.rolling(120, min_periods=120).mean()
        return ((index_nav > ma60) & (ma60 > ma120)).astype(float)
    raise ValueError(f"Unsupported trend signal: {config.trend}")


def sleeve_returns(class_returns: pd.DataFrame, class_name: str, config: SleeveConfig) -> pd.DataFrame:
    """Build one class sleeve daily return and risk weight."""
    ret = class_returns[class_name].astype(float).to_numpy()
    index_nav = pd.Series(np.cumprod(1 + ret), index=class_returns.index)
    signal = trend_signal(index_nav, config).to_numpy(float)
    realized_vol = pd.Series(ret).rolling(config.vol_window, min_periods=config.vol_window).std().to_numpy()
    realized_vol = realized_vol * np.sqrt(252)
    leverage = np.divide(config.target_vol, realized_vol, out=np.zeros_like(realized_vol), where=realized_vol > 0)
    leverage = np.nan_to_num(np.clip(leverage, 0, config.leverage_cap), nan=0.0, posinf=0.0, neginf=0.0)
    weight = signal * leverage
    prev_weight = np.r_[0.0, weight[:-1]]
    turnover = np.abs(weight - prev_weight)
    sleeve_ret = prev_weight * ret - turnover * FEE_RATE
    return pd.DataFrame(
        {
            "dt": class_returns["dt"],
            "class_name": class_name,
            "sleeve": f"{class_name}_{config.name}",
            "ret": sleeve_ret,
            "weight": weight,
        }
    )


def build_sleeve_library(class_returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build all class-level sleeve paths."""
    library = {}
    for class_name in ASSET_CLASSES:
        if class_name == "bond_cash":
            configs = (BOND_CONFIG,)
        elif class_name == "bond_plus":
            configs = BOND_PLUS_CONFIGS
        else:
            configs = SLEEVE_CONFIGS
        for config in configs:
            sleeve = sleeve_returns(class_returns, class_name, config)
            library[sleeve["sleeve"].iloc[0]] = sleeve
    return library


def combine_sleeves(
    class_returns: pd.DataFrame,
    library: dict[str, pd.DataFrame],
    budget_name: str,
    config: SleeveConfig,
) -> pd.DataFrame:
    """Combine class sleeves under one fixed class budget."""
    budget = CLASS_BUDGETS[budget_name]
    out = class_returns[["dt"]].copy()
    out["sleeve_ret"] = 0.0
    out["sleeve_weight"] = 0.0
    out["sleeves"] = ""
    sleeve_names = []
    for class_name, class_weight in budget.items():
        if class_name == "bond_cash":
            cfg = BOND_CONFIG
        elif class_name == "bond_plus":
            cfg = BOND_PLUS_CONFIGS[0]
        else:
            cfg = config
        sleeve_name = f"{class_name}_{cfg.name}"
        sleeve = library[sleeve_name]
        out["sleeve_ret"] += class_weight * sleeve["ret"].to_numpy(float)
        out["sleeve_weight"] += class_weight * sleeve["weight"].abs().to_numpy(float)
        sleeve_names.append(sleeve_name)
    out["sleeves"] = "|".join(sleeve_names)
    return out


def summarize_daily(daily: pd.DataFrame, label: str, symbol: str = "portfolio") -> dict:
    """Summarize one daily path."""
    stats = annualized_stats(daily["dt"].reset_index(drop=True), daily["ret"].to_numpy(), daily["weight"].to_numpy())
    stats["label"] = label
    stats["symbol"] = symbol
    stats["pass_20_10"] = bool(
        stats["annual_return"] >= TARGET_ANNUAL_RETURN and stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
    )
    return stats


def evaluate_window_stress(daily: pd.DataFrame, label: str) -> pd.DataFrame:
    """Evaluate moving endpoints and delete-year robustness."""
    rows = []
    years = sorted(int(x) for x in daily["dt"].dt.year.unique())
    rows.append({**summarize_daily(daily, label), "window_type": "full", "window": "full_2020_2026"})
    for year in years[1:]:
        part = daily[daily["dt"].dt.year <= year].copy()
        rows.append({**summarize_daily(part, label), "window_type": "moving_end", "window": f"end_{year}"})
    for year in years[:-1]:
        part = daily[daily["dt"].dt.year >= year].copy()
        rows.append({**summarize_daily(part, label), "window_type": "moving_start", "window": f"start_{year}"})
    for start_idx in range(len(years)):
        for end_idx in range(start_idx + 1, len(years)):
            start_year, end_year = years[start_idx], years[end_idx]
            part = daily[(daily["dt"].dt.year >= start_year) & (daily["dt"].dt.year <= end_year)].copy()
            rows.append(
                {
                    **summarize_daily(part, label),
                    "window_type": "rolling",
                    "window": f"rolling_{start_year}_{end_year}",
                }
            )
    for year in years:
        part = daily[daily["dt"].dt.year != year].copy()
        rows.append({**summarize_daily(part, label), "window_type": "drop_year", "window": f"drop_{year}"})
    return pd.DataFrame(rows)


def build_overlay_paths() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and evaluate all layered overlay paths."""
    price_panel = load_price_panel()
    class_returns = class_return_panel(price_panel)
    library = build_sleeve_library(class_returns)

    daily_rows = []
    summary_rows = []
    yearly_rows = []
    stress_rows = []

    for preset in BASE_PRESETS:
        base = load_base_daily(preset)
        base_daily = base.copy()
        base_daily["ret"] = base_daily["base_ret"]
        base_daily["weight"] = base_daily["base_weight"].abs()
        base_daily["overlay_mode"] = "base_only"
        base_daily["overlay_weight"] = 0.0
        base_daily["budget"] = "base_only"
        base_daily["base_preset"] = preset
        base_daily["sleeve_config"] = "none"
        base_daily["sleeve_ret"] = 0.0
        base_daily["sleeve_weight"] = 0.0
        base_daily["sleeves"] = ""
        base_daily["nav"] = (1 + base_daily["ret"]).cumprod()
        base_daily["drawdown"] = base_daily["nav"] / base_daily["nav"].cummax() - 1
        base_label = f"{preset}_base_only"
        base_daily["label"] = base_label
        daily_rows.append(base_daily)
        pre = summarize_daily(base_daily[base_daily["dt"].dt.year <= 2025], base_label)
        summary = summarize_daily(base_daily, base_label)
        summary.update(
            {
                "base_preset": preset,
                "overlay_mode": "base_only",
                "overlay_weight": 0.0,
                "budget": "base_only",
                "sleeve_config": "none",
                "pre2026_annual_return": pre["annual_return"],
                "pre2026_max_drawdown": pre["max_drawdown"],
            }
        )
        summary_rows.append(summary)
        yearly = summarize_yearly(base_daily[["dt", "ret", "weight"]].copy())
        yearly["label"] = base_label
        yearly["base_preset"] = preset
        yearly["overlay_mode"] = "base_only"
        yearly["overlay_weight"] = 0.0
        yearly["budget"] = "base_only"
        yearly["sleeve_config"] = "none"
        yearly_rows.append(yearly)

        for budget_name in CLASS_BUDGETS:
            for config in SLEEVE_CONFIGS:
                sleeve_port = combine_sleeves(class_returns, library, budget_name, config)
                joined = base.merge(sleeve_port, on="dt", how="inner")
                for overlay_weight in OVERLAY_WEIGHTS:
                    daily = joined.copy()
                    daily["ret"] = (1 - overlay_weight) * daily["base_ret"] + overlay_weight * daily["sleeve_ret"]
                    daily["weight"] = (1 - overlay_weight) * daily["base_weight"].abs() + overlay_weight * daily[
                        "sleeve_weight"
                    ]
                    daily["overlay_mode"] = "replacement"
                    daily["overlay_weight"] = overlay_weight
                    daily["budget"] = budget_name
                    daily["base_preset"] = preset
                    daily["sleeve_config"] = config.name
                    daily["nav"] = (1 + daily["ret"]).cumprod()
                    daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1
                    label = f"{preset}_replace_{overlay_weight:g}_{budget_name}_{config.name}"
                    daily["label"] = label
                    daily_rows.append(daily)
                    summary = summarize_daily(daily, label)
                    summary.update(
                        {
                            "base_preset": preset,
                            "overlay_mode": "replacement",
                            "overlay_weight": overlay_weight,
                            "budget": budget_name,
                            "sleeve_config": config.name,
                            "pre2026_annual_return": summarize_daily(daily[daily["dt"].dt.year <= 2025], label)[
                                "annual_return"
                            ],
                            "pre2026_max_drawdown": summarize_daily(daily[daily["dt"].dt.year <= 2025], label)[
                                "max_drawdown"
                            ],
                        }
                    )
                    summary_rows.append(summary)
                    yearly = summarize_yearly(daily[["dt", "ret", "weight"]].copy())
                    yearly["label"] = label
                    yearly["base_preset"] = preset
                    yearly["overlay_mode"] = "replacement"
                    yearly["overlay_weight"] = overlay_weight
                    yearly["budget"] = budget_name
                    yearly["sleeve_config"] = config.name
                    yearly_rows.append(yearly)

                for add_weight in ADD_WEIGHTS:
                    daily = joined.copy()
                    daily["ret"] = daily["base_ret"] + add_weight * daily["sleeve_ret"]
                    daily["weight"] = daily["base_weight"].abs() + add_weight * daily["sleeve_weight"]
                    daily["overlay_mode"] = "additive"
                    daily["overlay_weight"] = add_weight
                    daily["budget"] = budget_name
                    daily["base_preset"] = preset
                    daily["sleeve_config"] = config.name
                    daily["nav"] = (1 + daily["ret"]).cumprod()
                    daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1
                    label = f"{preset}_add_{add_weight:g}_{budget_name}_{config.name}"
                    daily["label"] = label
                    daily_rows.append(daily)
                    pre = summarize_daily(daily[daily["dt"].dt.year <= 2025], label)
                    summary = summarize_daily(daily, label)
                    summary.update(
                        {
                            "base_preset": preset,
                            "overlay_mode": "additive",
                            "overlay_weight": add_weight,
                            "budget": budget_name,
                            "sleeve_config": config.name,
                            "pre2026_annual_return": pre["annual_return"],
                            "pre2026_max_drawdown": pre["max_drawdown"],
                        }
                    )
                    summary_rows.append(summary)
                    yearly = summarize_yearly(daily[["dt", "ret", "weight"]].copy())
                    yearly["label"] = label
                    yearly["base_preset"] = preset
                    yearly["overlay_mode"] = "additive"
                    yearly["overlay_weight"] = add_weight
                    yearly["budget"] = budget_name
                    yearly["sleeve_config"] = config.name
                    yearly_rows.append(yearly)

    daily_out = pd.concat(daily_rows, ignore_index=True)
    summary_out = pd.DataFrame(summary_rows)
    yearly_out = pd.concat(yearly_rows, ignore_index=True)

    robust = summary_out[
        (summary_out["annual_return"] >= TARGET_ANNUAL_RETURN)
        & (summary_out["max_drawdown"] >= TARGET_MAX_DRAWDOWN)
        & (summary_out["pre2026_annual_return"] >= TARGET_ANNUAL_RETURN)
        & (summary_out["pre2026_max_drawdown"] >= TARGET_MAX_DRAWDOWN)
    ].copy()
    stress_candidates = pd.concat(
        [
            robust.sort_values(
                ["max_drawdown", "pre2026_annual_return", "annual_return"], ascending=[False, False, False]
            ).head(20),
            robust.sort_values(["annual_return", "max_drawdown"], ascending=[False, False]).head(20),
            robust.sort_values(["pre2026_annual_return", "max_drawdown"], ascending=[False, False]).head(20),
        ],
        ignore_index=True,
    ).drop_duplicates("label")
    if stress_candidates.empty:
        stress_candidates = pd.concat(
            [
                summary_out.sort_values(
                    ["max_drawdown", "pre2026_annual_return", "annual_return"], ascending=[False, False, False]
                ).head(20),
                summary_out.sort_values(["annual_return", "max_drawdown"], ascending=[False, False]).head(20),
                summary_out.sort_values(["pre2026_annual_return", "max_drawdown"], ascending=[False, False]).head(20),
            ],
            ignore_index=True,
        ).drop_duplicates("label")
    stress_labels = list(stress_candidates["label"])
    for label in [f"{preset}_base_only" for preset in BASE_PRESETS]:
        if label not in stress_labels:
            stress_labels.append(label)
    for label in stress_labels:
        path = daily_out[daily_out["label"] == label].copy()
        stress_rows.append(evaluate_window_stress(path, label))
    stress_out = pd.concat(stress_rows, ignore_index=True)
    return summary_out, yearly_out, daily_out, stress_out


def save_nav_plot(daily: pd.DataFrame, labels: list[str]) -> None:
    """Save net-value plot for selected overlay candidates."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for label in labels:
        group = daily[daily["label"] == label]
        fig.add_trace(go.Scatter(x=group["dt"], y=group["nav"], mode="lines", name=label))
    fig.update_layout(
        title="Layered low-correlation overlay candidates",
        xaxis_title="date",
        yaxis_title="net value",
        template="plotly_white",
    )
    fig.write_html(OUTPUT_DIR / "layered_low_corr_overlay_nav.html", include_plotlyjs="cdn")


def main() -> None:
    """Run layered low-correlation overlay validation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, yearly, daily, stress = build_overlay_paths()

    summary["pass_full_and_pre2026"] = (
        (summary["annual_return"] >= TARGET_ANNUAL_RETURN)
        & (summary["max_drawdown"] >= TARGET_MAX_DRAWDOWN)
        & (summary["pre2026_annual_return"] >= TARGET_ANNUAL_RETURN)
        & (summary["pre2026_max_drawdown"] >= TARGET_MAX_DRAWDOWN)
    )
    top = summary.sort_values(
        ["pass_full_and_pre2026", "max_drawdown", "pre2026_annual_return", "annual_return"],
        ascending=[False, False, False, False],
    ).head(20)
    top.to_csv(OUTPUT_DIR / "layered_low_corr_overlay_top20.csv", index=False, encoding="utf-8-sig")

    stress_agg = (
        stress.groupby("label")
        .agg(
            stress_windows=("pass_20_10", "size"),
            stress_pass_rate=("pass_20_10", "mean"),
            stress_annual_median=("annual_return", "median"),
            stress_mdd_median=("max_drawdown", "median"),
        )
        .reset_index()
        .sort_values(["stress_pass_rate", "stress_annual_median"], ascending=[False, False])
    )
    stress_agg.to_csv(OUTPUT_DIR / "layered_low_corr_overlay_window_stress_aggregate.csv", index=False)

    selected_daily_labels = set(top["label"].head(20)) | set(stress_agg["label"].head(20))
    selected_daily_labels |= {f"{preset}_base_only" for preset in BASE_PRESETS}
    daily_selected = daily[daily["label"].isin(selected_daily_labels)].copy()

    summary.to_csv(OUTPUT_DIR / "layered_low_corr_overlay_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUTPUT_DIR / "layered_low_corr_overlay_yearly.csv", index=False, encoding="utf-8-sig")
    daily_selected.to_csv(OUTPUT_DIR / "layered_low_corr_overlay_daily.csv", index=False, encoding="utf-8-sig")
    stress.to_csv(OUTPUT_DIR / "layered_low_corr_overlay_window_stress.csv", index=False, encoding="utf-8-sig")
    save_nav_plot(daily_selected, top["label"].head(8).tolist())

    print("Top candidates:")
    print(
        top[
            [
                "label",
                "annual_return",
                "max_drawdown",
                "pre2026_annual_return",
                "pre2026_max_drawdown",
                "final_nav",
                "pass_full_and_pre2026",
            ]
        ].to_string(index=False)
    )
    print("\nStress aggregate:")
    print(stress_agg.head(20).to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
