# ruff: noqa: E402, I001
"""Validate relative-value overlays for the daily CTA portfolio.

Previous tests showed that adding more single-direction ETF / REIT / QDII
trend sleeves does not repair the weak 2022-2023 window. This script tests
pairwise relative-strength rules between asset classes:

- ``rotate_long``: long the stronger class only;
- ``spread_ls``: long the stronger class and short the weaker class.

The second mode is less directly implementable with funds, but it estimates
whether the missing edge is in relative value rather than single-asset trend.

Run:
    uv run --no-sync python examples/signals_dev/validate_daily_cta_relative_value_overlay.py
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
from validate_daily_cta_walk_forward import DATA_FILES, annualized_stats, load_daily_data, summarize_yearly  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LOW_CORR_DIR = ROOT / "examples" / "results" / "daily_holding_low_corr_pool"
COMMODITY_DIR = ROOT / "examples" / "results" / "daily_holding_commodity_pool"
BOND_DIR = ROOT / "examples" / "results" / "daily_holding_bond_pool"
CROSS_BORDER_DIR = ROOT / "examples" / "results" / "daily_holding_cross_border_pool"
REIT_DIR = ROOT / "examples" / "results" / "daily_holding_reit_pool"
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_relative_value_overlay"
OVERLAY_WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.30)
ADD_WEIGHTS = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)

CLASS_SYMBOLS = {
    "cn_equity": tuple(DATA_FILES),
    "hk": ("510900.XSHG", "159920.XSHE"),
    "gold": ("518880.XSHG", "159934.XSHE"),
    "bond_plus": ("511020.XSHG", "511030.XSHG", "511260.XSHG", "511270.XSHG", "511660.XSHG"),
    "energy": ("162411.XSHE", "163208.XSHE", "159930.XSHE", "159945.XSHE"),
    "commodity": ("160216.XSHE", "161815.XSHE", "165513.XSHE"),
    "resource": ("510410.XSHG", "159944.XSHE", "512400.XSHG", "161226.XSHE"),
    "agri": ("159985.XSHE",),
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
    "usd_bond": ("501300.XSHG",),
    "japan": ("513880.XSHG", "513520.XSHG", "159866.XSHE"),
    "india_asia": ("164824.XSHE", "159687.XSHE"),
    "reit": ("__reit_dynamic__",),
}
RELATIVE_PAIRS = (
    ("us_broad", "cn_equity"),
    ("us_broad", "hk"),
    ("usd_bond", "bond_plus"),
    ("usd_bond", "cn_equity"),
    ("japan", "hk"),
    ("india_asia", "hk"),
    ("india_asia", "cn_equity"),
    ("gold", "cn_equity"),
    ("gold", "commodity"),
    ("commodity", "cn_equity"),
    ("energy", "cn_equity"),
    ("resource", "cn_equity"),
    ("agri", "cn_equity"),
    ("reit", "cn_equity"),
)


@dataclass(frozen=True)
class RelativeConfig:
    """One pairwise relative-value rule."""

    lookback: int
    mode: str
    vol_window: int
    target_vol: float
    leverage_cap: float
    threshold: float = 0.0

    @property
    def name(self) -> str:
        bp = int(self.threshold * 10000)
        return (
            f"{self.mode}_N{self.lookback}_B{bp}_VW{self.vol_window}_"
            f"TV{int(self.target_vol * 100)}_C{self.leverage_cap:g}"
        )


RELATIVE_CONFIGS = tuple(
    RelativeConfig(lookback, mode, vol_window, target_vol, cap)
    for lookback in (20, 60, 120)
    for mode in ("rotate_long", "spread_ls")
    for vol_window in (40, 60)
    for target_vol in (0.08, 0.12, 0.16)
    for cap in (1.0, 1.5)
)


def file_for_symbol(symbol: str) -> Path:
    """Resolve a cached daily OHLCV file for one fund symbol."""
    if symbol in {"510900.XSHG", "159920.XSHE", "518880.XSHG", "159934.XSHE"}:
        return LOW_CORR_DIR / f"{symbol.replace('.', '_')}_daily_20100101_20260529.csv"
    if symbol in {
        "162411.XSHE",
        "163208.XSHE",
        "159930.XSHE",
        "159945.XSHE",
        "160216.XSHE",
        "161815.XSHE",
        "165513.XSHE",
        "510410.XSHG",
        "159944.XSHE",
        "512400.XSHG",
        "161226.XSHE",
        "159985.XSHE",
    }:
        return COMMODITY_DIR / f"{symbol.replace('.', '_')}_daily_20100101_20260529.csv"
    if symbol in {"511020.XSHG", "511030.XSHG", "511260.XSHG", "511270.XSHG", "511660.XSHG"}:
        return BOND_DIR / f"{symbol.replace('.', '_')}_daily_20100101_20260529.csv"
    return CROSS_BORDER_DIR / f"{symbol.replace('.', '_')}_daily_20100101_20260529.csv"


def load_base_daily(preset: str) -> pd.DataFrame:
    """Load the annual-state scaled core CTA daily path."""
    daily = pd.read_csv(BASE_DAILY_FILE, parse_dates=["dt"])
    daily = daily[daily["preset"] == preset].sort_values("dt").reset_index(drop=True)
    if daily.empty:
        raise ValueError(f"No rows found for preset={preset!r} in {BASE_DAILY_FILE}")
    return daily[["dt", "ret", "weight", "test_year", "preset"]].rename(
        columns={"ret": "base_ret", "weight": "base_weight"}
    )


def dynamic_equal_return(panel: pd.DataFrame, symbols: tuple[str, ...]) -> tuple[pd.Series, pd.Series]:
    """Build dynamic equal-weight return and constituent count."""
    prices = panel[list(symbols)].copy()
    returns = prices.pct_change(fill_method=None)
    active = prices.notna() & prices.shift(1).notna()
    return returns.where(active).mean(axis=1, skipna=True).fillna(0.0), active.sum(axis=1).astype(int)


def load_reit_dynamic_return() -> pd.DataFrame:
    """Build a dynamic equal-weight REIT return series."""
    files = sorted(REIT_DIR.glob("*_daily_20100101_20260529.csv"))
    frames = []
    for file_csv in files:
        data = pd.read_csv(file_csv, parse_dates=["dt"])
        symbol = str(data["symbol"].dropna().iloc[0])
        frames.append(data[["dt", "close"]].rename(columns={"close": symbol}))
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on="dt", how="outer")
    panel = panel.sort_values("dt").reset_index(drop=True)
    ret, count = dynamic_equal_return(panel, tuple(panel.columns.drop("dt")))
    return pd.DataFrame({"dt": panel["dt"], "reit_ret": ret, "reit_constituents": count})


def class_return_panel() -> pd.DataFrame:
    """Build dynamic equal-weight class return panel."""
    frames = []
    for class_name, symbols in CLASS_SYMBOLS.items():
        if class_name in {"cn_equity", "reit"}:
            continue
        for symbol in symbols:
            data = pd.read_csv(file_for_symbol(symbol), parse_dates=["dt"])
            frames.append(data[["dt", "close"]].rename(columns={"close": symbol}))
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on="dt", how="outer")
    for symbol in DATA_FILES:
        data = load_daily_data(symbol)[["dt", "close"]].rename(columns={"close": symbol})
        panel = panel.merge(data, on="dt", how="outer")
    panel = panel.sort_values("dt").reset_index(drop=True)

    out = panel[["dt"]].copy()
    for class_name, symbols in CLASS_SYMBOLS.items():
        if class_name == "reit":
            continue
        ret, count = dynamic_equal_return(panel, symbols)
        out[f"{class_name}_ret"] = ret
        out[f"{class_name}_constituents"] = count
    reit = load_reit_dynamic_return()
    out = out.merge(reit, on="dt", how="left")
    out["reit_ret"] = out["reit_ret"].fillna(0.0)
    out["reit_constituents"] = out["reit_constituents"].fillna(0).astype(int)
    return out.sort_values("dt").reset_index(drop=True)


def relative_sleeve(
    returns: pd.DataFrame,
    left: str,
    right: str,
    config: RelativeConfig,
) -> pd.DataFrame:
    """Build one pairwise relative-value sleeve."""
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
        raise ValueError(f"Unsupported mode: {config.mode}")

    raw_ret = pd.Series(left_w).shift(1).fillna(0).to_numpy() * left_ret
    raw_ret += pd.Series(right_w).shift(1).fillna(0).to_numpy() * right_ret
    realized_vol = pd.Series(raw_ret).rolling(config.vol_window, min_periods=config.vol_window).std().to_numpy()
    realized_vol = realized_vol * np.sqrt(252)
    leverage = np.divide(config.target_vol, realized_vol, out=np.zeros_like(realized_vol), where=realized_vol > 0)
    leverage = np.nan_to_num(np.clip(leverage, 0, config.leverage_cap), nan=0.0, posinf=0.0, neginf=0.0)

    left_final = left_w * leverage
    right_final = right_w * leverage
    prev_left = np.r_[0.0, left_final[:-1]]
    prev_right = np.r_[0.0, right_final[:-1]]
    turnover = np.abs(left_final - prev_left) + np.abs(right_final - prev_right)
    sleeve_ret = prev_left * left_ret + prev_right * right_ret - turnover * FEE_RATE
    sleeve_weight = np.abs(left_final) + np.abs(right_final)
    return pd.DataFrame({"dt": returns["dt"], "sleeve_ret": sleeve_ret, "sleeve_weight": sleeve_weight})


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
    """Build relative-value overlay paths."""
    returns = class_return_panel()
    sleeve_cache = {
        (left, right, config.name): relative_sleeve(returns, left, right, config)
        for left, right in RELATIVE_PAIRS
        for config in RELATIVE_CONFIGS
    }

    daily_rows = []
    summary_rows = []
    yearly_rows = []
    for preset in BASE_PRESETS:
        base = load_base_daily(preset)
        for left, right in RELATIVE_PAIRS:
            for config in RELATIVE_CONFIGS:
                sleeve = sleeve_cache[(left, right, config.name)]
                joined = base.merge(sleeve, on="dt", how="left").fillna({"sleeve_ret": 0.0, "sleeve_weight": 0.0})

                for overlay_weight in OVERLAY_WEIGHTS:
                    daily = joined.copy()
                    daily["ret"] = (1 - overlay_weight) * daily["base_ret"] + overlay_weight * daily["sleeve_ret"]
                    daily["weight"] = (1 - overlay_weight) * daily["base_weight"].abs() + overlay_weight * daily[
                        "sleeve_weight"
                    ]
                    label = f"{preset}_replace_{overlay_weight:g}_{left}_vs_{right}_{config.name}"
                    daily["label"] = label
                    daily_rows.append(daily)
                    row = summarize_daily(daily, label)
                    row.update(
                        {
                            "base_preset": preset,
                            "overlay_mode": "replacement",
                            "overlay_weight": overlay_weight,
                            "left": left,
                            "right": right,
                            "relative_config": config.name,
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
                    label = f"{preset}_add_{add_weight:g}_{left}_vs_{right}_{config.name}"
                    daily["label"] = label
                    daily_rows.append(daily)
                    row = summarize_daily(daily, label)
                    row.update(
                        {
                            "base_preset": preset,
                            "overlay_mode": "additive",
                            "overlay_weight": add_weight,
                            "left": left,
                            "right": right,
                            "relative_config": config.name,
                        }
                    )
                    summary_rows.append(row)
                    yearly = summarize_yearly(daily[["dt", "ret", "weight"]])
                    yearly["label"] = label
                    yearly_rows.append(yearly)

    daily_out = pd.concat(daily_rows, ignore_index=True)
    summary_out = add_weak_window_stats(pd.DataFrame(summary_rows), pd.concat(yearly_rows, ignore_index=True))
    yearly_out = pd.concat(yearly_rows, ignore_index=True)

    stress_labels = list(
        pd.concat(
            [
                summary_out.sort_values(["annual_2022_2023", "annual_drop_2025"], ascending=False).head(30),
                summary_out.sort_values(["annual_drop_2025", "annual_2022_2023"], ascending=False).head(30),
                summary_out.sort_values(["pass_full_and_pre2026", "annual_return"], ascending=False).head(30),
            ],
            ignore_index=True,
        )["label"].drop_duplicates()
    )
    stress_rows = []
    for label in stress_labels:
        stress_rows.append(evaluate_window_stress(daily_out[daily_out["label"] == label].copy(), label))
    return summary_out, yearly_out, daily_out, pd.concat(stress_rows, ignore_index=True)


def main() -> None:
    """Run relative-value overlay validation."""
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

    summary.to_csv(OUTPUT_DIR / "relative_value_overlay_summary.csv", index=False, encoding="utf-8-sig")
    top.head(100).to_csv(OUTPUT_DIR / "relative_value_overlay_top100.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUTPUT_DIR / "relative_value_overlay_yearly.csv", index=False, encoding="utf-8-sig")
    stress.to_csv(OUTPUT_DIR / "relative_value_overlay_window_stress.csv", index=False, encoding="utf-8-sig")
    daily[daily["label"].isin(top["label"].head(20))].to_csv(
        OUTPUT_DIR / "relative_value_overlay_daily_top20.csv", index=False, encoding="utf-8-sig"
    )

    print("Top relative-value overlays by 2022-2023:")
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
