"""Validate cross-border asset overlays for the daily CTA portfolio.

The pool focuses on overseas / USD-exposed ETFs and LOFs. Assets are grouped by
economic exposure, converted into dynamic equal-weight class returns, then
tested as additive / replacement overlays on top of the current annual-state
core CTA path.

Run:
    uv run --no-sync python examples/signals_dev/validate_daily_cta_cross_border_overlay.py
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

from validate_daily_cta_layered_low_corr_overlay import (  # noqa: E402
    BASE_DAILY_FILE,
    BASE_PRESETS,
    FEE_RATE,
    TARGET_ANNUAL_RETURN,
    TARGET_MAX_DRAWDOWN,
    evaluate_window_stress,
)
from validate_daily_cta_walk_forward import annualized_stats, summarize_yearly  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CROSS_BORDER_DIR = ROOT / "examples" / "results" / "daily_holding_cross_border_pool"
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_cross_border_overlay"
OVERLAY_WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.30)
ADD_WEIGHTS = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)

ASSET_CLASSES = {
    "us_broad": (
        "513100.XSHG",
        "159941.XSHE",
        "161130.XSHE",
        "513300.XSHG",
        "513500.XSHG",
        "161125.XSHE",
        "159612.XSHE",
        "159632.XSHE",
        "159655.XSHE",
    ),
    "us_sector": ("161126.XSHE", "161128.XSHE", "161127.XSHE", "513290.XSHG", "160644.XSHE"),
    "usd_bond": ("501300.XSHG",),
    "japan": ("513880.XSHG", "513520.XSHG", "159866.XSHE"),
    "india_asia": ("164824.XSHE", "159687.XSHE"),
    "europe": ("513030.XSHG",),
}
CLASS_BUDGETS = {
    "usd_bond": {"usd_bond": 1.0},
    "us_broad": {"us_broad": 1.0},
    "us_sector": {"us_sector": 1.0},
    "japan": {"japan": 1.0},
    "india_asia": {"india_asia": 1.0},
    "overseas_balanced": {"us_broad": 0.35, "usd_bond": 0.25, "japan": 0.20, "india_asia": 0.20},
    "usd_defensive": {"usd_bond": 0.55, "us_broad": 0.25, "japan": 0.20},
    "growth_ex_us": {"japan": 0.35, "india_asia": 0.35, "europe": 0.30},
}


@dataclass(frozen=True)
class SleeveConfig:
    """One timing and risk configuration."""

    trend: str
    vol_window: int
    target_vol: float
    leverage_cap: float
    min_constituents: int = 1

    @property
    def name(self) -> str:
        return (
            f"{self.trend}_VW{self.vol_window}_TV{int(self.target_vol * 100)}_"
            f"C{self.leverage_cap:g}_N{self.min_constituents}"
        )


SLEEVE_CONFIGS = (
    SleeveConfig("long", 40, 0.06, 1.0),
    SleeveConfig("long", 40, 0.08, 1.0),
    SleeveConfig("long", 60, 0.10, 1.5),
    SleeveConfig("ma60", 40, 0.08, 1.0),
    SleeveConfig("ma120", 40, 0.08, 1.0),
    SleeveConfig("mom60", 40, 0.08, 1.0),
    SleeveConfig("mom120", 40, 0.08, 1.0),
    SleeveConfig("dual_ma60_120", 40, 0.08, 1.0),
    SleeveConfig("ma120", 60, 0.10, 1.2),
    SleeveConfig("mom120", 60, 0.10, 1.2),
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


def load_close_panel() -> pd.DataFrame:
    """Load cross-border close prices with outer date alignment."""
    frames = []
    for symbols in ASSET_CLASSES.values():
        for symbol in symbols:
            file_csv = CROSS_BORDER_DIR / f"{symbol.replace('.', '_')}_daily_20100101_20260529.csv"
            data = pd.read_csv(file_csv, parse_dates=["dt"])
            frames.append(data[["dt", "close"]].rename(columns={"close": symbol}))
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on="dt", how="outer")
    return panel.sort_values("dt").reset_index(drop=True)


def dynamic_class_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Build dynamic equal-weight class returns."""
    out = price_panel[["dt"]].copy()
    for class_name, symbols in ASSET_CLASSES.items():
        prices = price_panel[list(symbols)].copy()
        returns = prices.pct_change(fill_method=None)
        active = prices.notna() & prices.shift(1).notna()
        out[f"{class_name}_ret"] = returns.where(active).mean(axis=1, skipna=True).fillna(0.0)
        out[f"{class_name}_constituents"] = active.sum(axis=1).astype(int)
    return out


def trend_signal(index_nav: pd.Series, config: SleeveConfig) -> pd.Series:
    """Generate a long/flat trend signal."""
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


def class_sleeve(class_returns: pd.DataFrame, class_name: str, config: SleeveConfig) -> pd.DataFrame:
    """Build one class sleeve."""
    ret = class_returns[f"{class_name}_ret"].astype(float).to_numpy()
    index_nav = pd.Series(np.cumprod(1 + ret), index=class_returns.index)
    signal = trend_signal(index_nav, config).to_numpy(float)
    enough = (class_returns[f"{class_name}_constituents"].to_numpy(int) >= config.min_constituents).astype(float)
    realized_vol = pd.Series(ret).rolling(config.vol_window, min_periods=config.vol_window).std().to_numpy()
    realized_vol = realized_vol * np.sqrt(252)
    leverage = np.divide(config.target_vol, realized_vol, out=np.zeros_like(realized_vol), where=realized_vol > 0)
    leverage = np.nan_to_num(np.clip(leverage, 0, config.leverage_cap), nan=0.0, posinf=0.0, neginf=0.0)
    weight = signal * enough * leverage
    prev_weight = np.r_[0.0, weight[:-1]]
    turnover = np.abs(weight - prev_weight)
    sleeve_ret = prev_weight * ret - turnover * FEE_RATE
    return pd.DataFrame(
        {
            "dt": class_returns["dt"],
            "class_name": class_name,
            "ret": sleeve_ret,
            "weight": weight,
        }
    )


def combine_budget(
    class_returns: pd.DataFrame,
    library: dict[tuple[str, str], pd.DataFrame],
    budget_name: str,
    config: SleeveConfig,
) -> pd.DataFrame:
    """Combine class sleeves under one fixed budget."""
    out = class_returns[["dt"]].copy()
    out["sleeve_ret"] = 0.0
    out["sleeve_weight"] = 0.0
    for class_name, class_weight in CLASS_BUDGETS[budget_name].items():
        sleeve = library[(class_name, config.name)]
        out["sleeve_ret"] += class_weight * sleeve["ret"].to_numpy(float)
        out["sleeve_weight"] += class_weight * sleeve["weight"].abs().to_numpy(float)
    return out


def summarize_daily(daily: pd.DataFrame, label: str) -> dict:
    """Summarize one daily path."""
    stats = annualized_stats(daily["dt"].reset_index(drop=True), daily["ret"].to_numpy(), daily["weight"].to_numpy())
    pre = daily[daily["dt"].dt.year <= 2025]
    pre_stats = annualized_stats(pre["dt"].reset_index(drop=True), pre["ret"].to_numpy(), pre["weight"].to_numpy())
    stats.update(
        {
            "label": label,
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


def add_weak_window_stats(summary: pd.DataFrame, yearly: pd.DataFrame) -> pd.DataFrame:
    """Add 2022-2023 and drop-2025 diagnostics from yearly stats."""
    rows = []
    for label, group in yearly.groupby("label"):
        years = group.set_index("year").sort_index()
        part_2223 = years.loc[[2022, 2023]]
        drop_2025 = years[years.index != 2025]
        rows.append(
            {
                "label": label,
                "annual_2022_2023": float((1 + part_2223["return"]).prod() ** (1 / len(part_2223)) - 1),
                "mdd_2022_2023": float(part_2223["max_drawdown"].min()),
                "annual_drop_2025": float((1 + drop_2025["return"]).prod() ** (1 / len(drop_2025)) - 1),
                "mdd_drop_2025": float(drop_2025["max_drawdown"].min()),
            }
        )
    return summary.merge(pd.DataFrame(rows), on="label", how="left")


def build_overlay_paths() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and evaluate cross-border overlay paths."""
    class_returns = dynamic_class_returns(load_close_panel())
    library = {
        (class_name, config.name): class_sleeve(class_returns, class_name, config)
        for class_name in ASSET_CLASSES
        for config in SLEEVE_CONFIGS
    }

    daily_rows = []
    summary_rows = []
    yearly_rows = []
    for preset in BASE_PRESETS:
        base = load_base_daily(preset)
        for budget_name in CLASS_BUDGETS:
            for config in SLEEVE_CONFIGS:
                sleeve = combine_budget(class_returns, library, budget_name, config)
                joined = base.merge(sleeve, on="dt", how="left").fillna({"sleeve_ret": 0.0, "sleeve_weight": 0.0})
                for overlay_weight in OVERLAY_WEIGHTS:
                    daily = joined.copy()
                    daily["ret"] = (1 - overlay_weight) * daily["base_ret"] + overlay_weight * daily["sleeve_ret"]
                    daily["weight"] = (1 - overlay_weight) * daily["base_weight"].abs() + overlay_weight * daily[
                        "sleeve_weight"
                    ]
                    label = f"{preset}_replace_{overlay_weight:g}_{budget_name}_{config.name}"
                    daily["label"] = label
                    daily["nav"] = (1 + daily["ret"]).cumprod()
                    daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1
                    daily_rows.append(daily)
                    row = summarize_daily(daily, label)
                    row.update(
                        {
                            "base_preset": preset,
                            "overlay_mode": "replacement",
                            "overlay_weight": overlay_weight,
                            "budget": budget_name,
                            "sleeve_config": config.name,
                        }
                    )
                    summary_rows.append(row)
                    yearly = summarize_yearly(daily[["dt", "ret", "weight"]])
                    yearly["label"] = label
                    yearly_rows.append(yearly)

                for add_weight in ADD_WEIGHTS:
                    daily = joined.copy()
                    daily["ret"] = daily["base_ret"] + add_weight * daily["sleeve_ret"]
                    daily["weight"] = daily["base_weight"].abs() + add_weight * daily["sleeve_weight"]
                    label = f"{preset}_add_{add_weight:g}_{budget_name}_{config.name}"
                    daily["label"] = label
                    daily["nav"] = (1 + daily["ret"]).cumprod()
                    daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1
                    daily_rows.append(daily)
                    row = summarize_daily(daily, label)
                    row.update(
                        {
                            "base_preset": preset,
                            "overlay_mode": "additive",
                            "overlay_weight": add_weight,
                            "budget": budget_name,
                            "sleeve_config": config.name,
                        }
                    )
                    summary_rows.append(row)
                    yearly = summarize_yearly(daily[["dt", "ret", "weight"]])
                    yearly["label"] = label
                    yearly_rows.append(yearly)

    daily_out = pd.concat(daily_rows, ignore_index=True)
    summary_out = pd.DataFrame(summary_rows)
    yearly_out = pd.concat(yearly_rows, ignore_index=True)
    summary_out = add_weak_window_stats(summary_out, yearly_out)

    stress_rows = []
    stress_labels = list(
        pd.concat(
            [
                summary_out.sort_values(["annual_2022_2023", "annual_drop_2025"], ascending=False).head(20),
                summary_out.sort_values(["annual_drop_2025", "annual_2022_2023"], ascending=False).head(20),
                summary_out.sort_values(["pass_full_and_pre2026", "annual_return"], ascending=False).head(20),
            ],
            ignore_index=True,
        )["label"].drop_duplicates()
    )
    for label in stress_labels:
        stress_rows.append(evaluate_window_stress(daily_out[daily_out["label"] == label].copy(), label))
    stress_out = pd.concat(stress_rows, ignore_index=True)
    return summary_out, yearly_out, daily_out, stress_out


def main() -> None:
    """Run cross-border overlay validation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, yearly, daily, stress = build_overlay_paths()
    stress_agg = (
        stress.groupby("label")
        .agg(
            stress_windows=("pass_20_10", "size"),
            stress_pass_rate=("pass_20_10", "mean"),
            stress_annual_median=("annual_return", "median"),
            stress_mdd_median=("max_drawdown", "median"),
        )
        .reset_index()
    )
    summary = summary.merge(stress_agg, on="label", how="left")
    top = summary.sort_values(
        ["annual_2022_2023", "annual_drop_2025", "pass_full_and_pre2026", "annual_return"],
        ascending=False,
        na_position="last",
    )

    summary.to_csv(OUTPUT_DIR / "cross_border_overlay_summary.csv", index=False, encoding="utf-8-sig")
    top.head(80).to_csv(OUTPUT_DIR / "cross_border_overlay_top80.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUTPUT_DIR / "cross_border_overlay_yearly.csv", index=False, encoding="utf-8-sig")
    stress.to_csv(OUTPUT_DIR / "cross_border_overlay_window_stress.csv", index=False, encoding="utf-8-sig")
    daily[daily["label"].isin(top["label"].head(20))].to_csv(
        OUTPUT_DIR / "cross_border_overlay_daily_top20.csv", index=False, encoding="utf-8-sig"
    )

    print("Top cross-border overlays by 2022-2023:")
    print(
        top[
            [
                "label",
                "annual_return",
                "max_drawdown",
                "pre2026_annual_return",
                "annual_2022_2023",
                "mdd_2022_2023",
                "annual_drop_2025",
                "mdd_drop_2025",
                "stress_pass_rate",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
