"""Validate dynamic public-REIT overlays for the daily CTA portfolio.

REIT listings are staggered, so this script does not use an inner-joined price
panel. It builds a dynamic equal-weight REIT return series from all currently
listed constituents and tests whether a REIT sleeve can improve the weak
2022-2023 and drop-2025 windows left by the current layered overlays.

Run:
    uv run --no-sync python examples/signals_dev/validate_daily_cta_reit_overlay.py
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
REIT_DIR = ROOT / "examples" / "results" / "daily_holding_reit_pool"
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_reit_overlay"
OVERLAY_WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.30)
ADD_WEIGHTS = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)


@dataclass(frozen=True)
class ReitConfig:
    """One REIT timing and risk configuration."""

    trend: str
    vol_window: int
    target_vol: float
    leverage_cap: float
    min_constituents: int = 5

    @property
    def name(self) -> str:
        return (
            f"{self.trend}_VW{self.vol_window}_TV{int(self.target_vol * 100)}_"
            f"C{self.leverage_cap:g}_N{self.min_constituents}"
        )


REIT_CONFIGS = (
    ReitConfig("long", 40, 0.06, 1.0),
    ReitConfig("long", 40, 0.08, 1.0),
    ReitConfig("long", 60, 0.08, 1.5),
    ReitConfig("ma60", 40, 0.08, 1.0),
    ReitConfig("ma120", 40, 0.08, 1.0),
    ReitConfig("mom60", 40, 0.08, 1.0),
    ReitConfig("mom120", 40, 0.08, 1.0),
    ReitConfig("dual_ma60_120", 40, 0.08, 1.0),
    ReitConfig("ma120", 60, 0.10, 1.2),
    ReitConfig("mom120", 60, 0.10, 1.2),
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


def load_reit_close_panel() -> pd.DataFrame:
    """Load REIT close prices with outer date alignment."""
    files = sorted(REIT_DIR.glob("*_daily_20100101_20260529.csv"))
    if not files:
        raise FileNotFoundError(f"No REIT CSV files found in {REIT_DIR}")
    frames = []
    for file_csv in files:
        data = pd.read_csv(file_csv, parse_dates=["dt"])
        symbol = str(data["symbol"].dropna().iloc[0])
        frame = data[["dt", "close"]].rename(columns={"close": symbol})
        frames.append(frame)
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on="dt", how="outer")
    return panel.sort_values("dt").reset_index(drop=True)


def dynamic_reit_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Build dynamic equal-weight REIT returns from available constituents."""
    prices = price_panel.drop(columns=["dt"]).copy()
    returns = prices.pct_change(fill_method=None)
    active = prices.notna() & prices.shift(1).notna()
    returns = returns.where(active)
    out = price_panel[["dt"]].copy()
    out["reit_ret"] = returns.mean(axis=1, skipna=True).fillna(0.0)
    out["constituents"] = active.sum(axis=1).astype(int)
    return out


def trend_signal(index_nav: pd.Series, config: ReitConfig) -> pd.Series:
    """Generate a long/flat REIT trend signal."""
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


def build_reit_sleeve(reit_returns: pd.DataFrame, config: ReitConfig) -> pd.DataFrame:
    """Build one REIT sleeve daily return and risk weight."""
    ret = reit_returns["reit_ret"].astype(float).to_numpy()
    index_nav = pd.Series(np.cumprod(1 + ret), index=reit_returns.index)
    signal = trend_signal(index_nav, config).to_numpy(float)
    enough = (reit_returns["constituents"].to_numpy(int) >= config.min_constituents).astype(float)
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
            "dt": reit_returns["dt"],
            "reit_ret": sleeve_ret,
            "reit_weight": weight,
            "reit_constituents": reit_returns["constituents"],
            "reit_config": config.name,
        }
    )


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


def build_overlay_paths() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and evaluate REIT overlay paths."""
    reit_returns = dynamic_reit_returns(load_reit_close_panel())
    sleeves = {config.name: build_reit_sleeve(reit_returns, config) for config in REIT_CONFIGS}

    daily_rows = []
    summary_rows = []
    yearly_rows = []

    for preset in BASE_PRESETS:
        base = load_base_daily(preset)
        for config_name, sleeve in sleeves.items():
            joined = base.merge(sleeve, on="dt", how="left")
            joined[["reit_ret", "reit_weight"]] = joined[["reit_ret", "reit_weight"]].fillna(0.0)
            joined["reit_constituents"] = joined["reit_constituents"].fillna(0).astype(int)

            for overlay_weight in OVERLAY_WEIGHTS:
                daily = joined.copy()
                daily["ret"] = (1 - overlay_weight) * daily["base_ret"] + overlay_weight * daily["reit_ret"]
                daily["weight"] = (1 - overlay_weight) * daily["base_weight"].abs() + overlay_weight * daily[
                    "reit_weight"
                ]
                daily["overlay_mode"] = "replacement"
                daily["overlay_weight"] = overlay_weight
                daily["base_preset"] = preset
                daily["nav"] = (1 + daily["ret"]).cumprod()
                daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1
                label = f"{preset}_replace_{overlay_weight:g}_reit_{config_name}"
                daily["label"] = label
                daily_rows.append(daily)

                summary = summarize_daily(daily, label)
                summary.update(
                    {
                        "base_preset": preset,
                        "overlay_mode": "replacement",
                        "overlay_weight": overlay_weight,
                        "reit_config": config_name,
                    }
                )
                summary_rows.append(summary)

                yearly = summarize_yearly(daily[["dt", "ret", "weight"]].copy())
                yearly["label"] = label
                yearly_rows.append(yearly)

            for add_weight in ADD_WEIGHTS:
                daily = joined.copy()
                daily["ret"] = daily["base_ret"] + add_weight * daily["reit_ret"]
                daily["weight"] = daily["base_weight"].abs() + add_weight * daily["reit_weight"]
                daily["overlay_mode"] = "additive"
                daily["overlay_weight"] = add_weight
                daily["base_preset"] = preset
                daily["nav"] = (1 + daily["ret"]).cumprod()
                daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1
                label = f"{preset}_add_{add_weight:g}_reit_{config_name}"
                daily["label"] = label
                daily_rows.append(daily)

                summary = summarize_daily(daily, label)
                summary.update(
                    {
                        "base_preset": preset,
                        "overlay_mode": "additive",
                        "overlay_weight": add_weight,
                        "reit_config": config_name,
                    }
                )
                summary_rows.append(summary)

                yearly = summarize_yearly(daily[["dt", "ret", "weight"]].copy())
                yearly["label"] = label
                yearly_rows.append(yearly)

    daily_out = pd.concat(daily_rows, ignore_index=True)
    summary_out = pd.DataFrame(summary_rows)
    yearly_out = pd.concat(yearly_rows, ignore_index=True)

    stress_rows = []
    stress_labels = list(
        pd.concat(
            [
                summary_out.sort_values(["pass_full_and_pre2026", "annual_return"], ascending=False).head(20),
                summary_out.sort_values(["max_drawdown", "annual_return"], ascending=False).head(20),
            ],
            ignore_index=True,
        )["label"].drop_duplicates()
    )
    for label in stress_labels:
        path = daily_out[daily_out["label"] == label].copy()
        stress_rows.append(evaluate_window_stress(path, label))
    stress_out = pd.concat(stress_rows, ignore_index=True)
    return summary_out, yearly_out, daily_out, stress_out


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


def main() -> None:
    """Run REIT overlay validation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, yearly, daily, stress = build_overlay_paths()
    summary = add_weak_window_stats(summary, yearly)
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
        ["pass_full_and_pre2026", "annual_2022_2023", "annual_drop_2025", "annual_return"],
        ascending=False,
        na_position="last",
    )

    summary.to_csv(OUTPUT_DIR / "reit_overlay_summary.csv", index=False, encoding="utf-8-sig")
    top.head(50).to_csv(OUTPUT_DIR / "reit_overlay_top50.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUTPUT_DIR / "reit_overlay_yearly.csv", index=False, encoding="utf-8-sig")
    stress.to_csv(OUTPUT_DIR / "reit_overlay_window_stress.csv", index=False, encoding="utf-8-sig")
    daily[daily["label"].isin(top["label"].head(20))].to_csv(
        OUTPUT_DIR / "reit_overlay_daily_top20.csv", index=False, encoding="utf-8-sig"
    )

    print("Top REIT overlays:")
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
