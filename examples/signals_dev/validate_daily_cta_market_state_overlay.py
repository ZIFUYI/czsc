"""Market-state risk overlay for adaptive cross-sectional daily CTA.

The overlay scales the already generated ``aggressive_state_switch_adaptive_mix``
OOS returns by a causal market breadth / momentum state. It is an independent
risk-budget module and does not reselect source modes.

Run:
    .venv/bin/python examples/signals_dev/validate_daily_cta_market_state_overlay.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from validate_daily_cta_walk_forward import DATA_FILES, annualized_stats, load_daily_data, summarize_yearly

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "examples" / "results" / "daily_cta_cross_section_walk_forward"
OUTPUT_DIR = INPUT_DIR / "market_state_overlay"
DAILY_NAV_FILE = INPUT_DIR / "cross_section_daily_nav.csv"
MODE = "aggressive_state_switch_adaptive_mix"

OVERLAY_PRESETS = {
    "conservative": {
        "lookback": 20,
        "momentum_threshold": 0.00,
        "trend_fraction_threshold": 0.67,
        "bad_scale": 0.75,
        "good_scale": 1.05,
        "hot_vol_threshold": 0.35,
    },
    "aggressive": {
        "lookback": 20,
        "momentum_threshold": 0.00,
        "trend_fraction_threshold": 0.67,
        "bad_scale": 0.85,
        "good_scale": 1.15,
        "hot_vol_threshold": 0.20,
    },
}


def load_close_panel(dates: pd.Series) -> pd.DataFrame:
    """Load and align close prices for current symbols."""
    frames = []
    for symbol in DATA_FILES:
        data = load_daily_data(symbol)[["dt", "close"]].rename(columns={"close": symbol})
        frames.append(data)
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on="dt", how="inner")
    panel = panel.sort_values("dt").reset_index(drop=True)
    panel = panel[panel["dt"].isin(dates)].reset_index(drop=True)
    return panel[list(DATA_FILES)].astype(float)


def build_overlay_scale(close: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    """Build a causal market-state exposure scale."""
    lookback = int(params["lookback"])
    market_momentum = (close / close.shift(lookback) - 1).mean(axis=1)
    trend_fraction = (close > close.rolling(lookback, min_periods=lookback).mean()).mean(axis=1)
    market_vol = close.pct_change().mean(axis=1).rolling(lookback, min_periods=lookback).std() * np.sqrt(252)

    bad_state = (market_momentum.shift(1) < params["momentum_threshold"]) | (
        trend_fraction.shift(1) < params["trend_fraction_threshold"]
    )
    hot_vol = market_vol.shift(1) > params["hot_vol_threshold"]
    scale = np.where(bad_state, params["bad_scale"], params["good_scale"])
    scale = np.where(hot_vol, np.minimum(scale, 1.0), scale)
    return np.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)


def evaluate_overlay(daily: pd.DataFrame, scale: np.ndarray) -> tuple[pd.DataFrame, dict, dict]:
    """Apply overlay and summarize full / pre-2026 windows."""
    overlaid = daily.copy()
    overlaid["overlay_scale"] = scale
    overlaid["ret"] = overlaid["ret"] * overlaid["overlay_scale"]
    overlaid["weight"] = overlaid["weight"] * overlaid["overlay_scale"]
    overlaid["nav"] = (1 + overlaid["ret"]).cumprod()
    overlaid["drawdown"] = overlaid["nav"] / overlaid["nav"].cummax() - 1
    full_stats = annualized_stats(overlaid["dt"], overlaid["ret"].to_numpy(), overlaid["weight"].to_numpy())
    pre_mask = overlaid["test_year"].to_numpy() <= 2025
    pre_stats = annualized_stats(
        overlaid.loc[pre_mask, "dt"].reset_index(drop=True),
        overlaid.loc[pre_mask, "ret"].to_numpy(),
        overlaid.loc[pre_mask, "weight"].to_numpy(),
    )
    return overlaid, full_stats, pre_stats


def main() -> None:
    """Run market-state overlay validation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(DAILY_NAV_FILE, parse_dates=["dt"])
    daily = daily[daily["mode"] == MODE].sort_values("dt").reset_index(drop=True)
    close = load_close_panel(daily["dt"])
    if len(close) != len(daily):
        raise ValueError(f"Close panel length {len(close)} != daily length {len(daily)}")

    daily_rows = []
    yearly_rows = []
    summary_rows = []
    for preset, params in OVERLAY_PRESETS.items():
        scale = build_overlay_scale(close, params)
        overlaid, full_stats, pre_stats = evaluate_overlay(daily, scale)
        overlaid["preset"] = preset
        overlaid["mode"] = f"{MODE}_{preset}_overlay"
        daily_rows.append(overlaid)

        yearly = summarize_yearly(overlaid)
        yearly["preset"] = preset
        yearly["mode"] = f"{MODE}_{preset}_overlay"
        yearly_rows.append(yearly)

        summary_rows.append(
            {
                "preset": preset,
                **{f"full_{key}": value for key, value in full_stats.items()},
                **{f"pre2026_{key}": value for key, value in pre_stats.items()},
                "avg_overlay_scale": float(overlaid["overlay_scale"].mean()),
                "min_overlay_scale": float(overlaid["overlay_scale"].min()),
                "max_overlay_scale": float(overlaid["overlay_scale"].max()),
                **params,
            }
        )

    daily_out = pd.concat(daily_rows, ignore_index=True)
    yearly_out = pd.concat(yearly_rows, ignore_index=True)
    summary_out = pd.DataFrame(summary_rows)
    daily_out.to_csv(OUTPUT_DIR / "market_state_overlay_daily.csv", index=False, encoding="utf-8-sig")
    yearly_out.to_csv(OUTPUT_DIR / "market_state_overlay_yearly.csv", index=False, encoding="utf-8-sig")
    summary_out.to_csv(OUTPUT_DIR / "market_state_overlay_summary.csv", index=False, encoding="utf-8-sig")

    print(summary_out.to_string(index=False))
    print("\nYearly:")
    print(yearly_out[["preset", "year", "return", "max_drawdown", "annual_return", "calmar"]].to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
