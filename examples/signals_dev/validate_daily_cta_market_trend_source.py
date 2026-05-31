"""Validate market-level daily trend sources against the current best CTA mix.

This script tests whether simple market-level trend / reversal sources can add
out-of-sample return to the aggressive market-state overlay. It is deliberately
small and reproducible: no parameter is selected from the test-period outcome;
the output is a research screen for deciding whether this family deserves more
work.

Run:
    .venv/bin/python examples/signals_dev/validate_daily_cta_market_trend_source.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from validate_daily_cta_walk_forward import DATA_FILES, FEE_RATE, annualized_stats, load_daily_data, summarize_yearly

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "examples" / "results" / "daily_cta_cross_section_walk_forward"
OVERLAY_DIR = INPUT_DIR / "market_state_overlay"
OUTPUT_DIR = INPUT_DIR / "market_trend_source"
BASE_FILE = OVERLAY_DIR / "market_state_overlay_daily.csv"
BASE_PRESET = "aggressive"

LOOKBACKS = (5, 10, 20, 40, 60, 120)
VOL_WINDOWS = (20, 60)
TARGET_VOLS = (0.08, 0.12, 0.16, 0.20)
LEVERAGE_CAPS = (1.0, 1.5, 2.0)
BLEND_WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.30)
SOURCE_MODES = ("market_trend_equal", "market_contra_equal", "breadth_trend_equal", "cross_reversal_ls")


def load_base_daily() -> pd.DataFrame:
    """Load aggressive overlay daily returns as the current base strategy."""
    daily = pd.read_csv(BASE_FILE, parse_dates=["dt"])
    daily = daily[daily["preset"] == BASE_PRESET].sort_values("dt").reset_index(drop=True)
    if daily.empty:
        raise ValueError(f"No rows found for preset={BASE_PRESET!r} in {BASE_FILE}")
    return daily


def load_close_panel(dates: pd.Series) -> pd.DataFrame:
    """Load close price panel aligned to the base strategy dates."""
    frames = []
    for symbol in DATA_FILES:
        data = load_daily_data(symbol)[["dt", "close"]].rename(columns={"close": symbol})
        frames.append(data)
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on="dt", how="inner")
    panel = panel[panel["dt"].isin(dates)].sort_values("dt").reset_index(drop=True)
    return panel[list(DATA_FILES)].astype(float)


def build_raw_weights(close: pd.DataFrame, lookback: int, mode: str) -> pd.DataFrame:
    """Build raw market-level source weights before volatility targeting."""
    momentum = close / close.shift(lookback) - 1
    market_momentum = momentum.mean(axis=1)
    breadth = (momentum > 0).mean(axis=1)
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    if mode == "market_trend_equal":
        signal = np.sign(market_momentum.shift(1)).fillna(0.0).to_numpy(float)
        weights.loc[:, :] = signal[:, None] / len(close.columns)
    elif mode == "market_contra_equal":
        signal = -np.sign(market_momentum.shift(1)).fillna(0.0).to_numpy(float)
        weights.loc[:, :] = signal[:, None] / len(close.columns)
    elif mode == "breadth_trend_equal":
        signal = np.where(
            breadth.shift(1) >= 2 / 3,
            1.0,
            np.where(breadth.shift(1) <= 1 / 3, -1.0, 0.0),
        )
        weights.loc[:, :] = signal[:, None] / len(close.columns)
    elif mode == "cross_reversal_ls":
        prev_momentum = momentum.shift(1)
        for idx, row in prev_momentum.iterrows():
            if row.notna().sum() < 2:
                continue
            top_symbol = row.idxmax()
            bottom_symbol = row.idxmin()
            if row[top_symbol] - row[bottom_symbol] > 0.01:
                weights.at[idx, top_symbol] = -0.5
                weights.at[idx, bottom_symbol] = 0.5
    else:
        raise ValueError(f"Unsupported source mode: {mode}")
    return weights


def volatility_target_returns(
    raw_weights: pd.DataFrame, price_returns: pd.DataFrame, vol_window: int, target_vol: float, leverage_cap: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw weights into fee-adjusted returns with volatility targeting."""
    raw_returns = (raw_weights.shift(1).fillna(0.0) * price_returns).sum(axis=1)
    realized_vol = raw_returns.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)
    leverage = target_vol / realized_vol.replace(0, np.nan)
    leverage = leverage.clip(lower=0, upper=leverage_cap).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = raw_weights.multiply(leverage, axis=0)
    prev_weights = weights.shift(1).fillna(0.0)
    turnover = (weights - prev_weights).abs().sum(axis=1)
    returns = (prev_weights * price_returns).sum(axis=1) - turnover * FEE_RATE
    exposure = weights.abs().sum(axis=1)
    return returns.to_numpy(float), exposure.to_numpy(float)


def summarize_returns(daily: pd.DataFrame, returns: np.ndarray, weight: np.ndarray, name: str) -> dict:
    """Summarize full and 2020-2025 performance."""
    full = annualized_stats(daily["dt"], returns, weight)
    pre_mask = daily["test_year"].to_numpy() <= 2025
    pre = annualized_stats(
        daily.loc[pre_mask, "dt"].reset_index(drop=True),
        returns[pre_mask],
        weight[pre_mask],
    )
    yearly = summarize_yearly(pd.DataFrame({"dt": daily["dt"], "ret": returns, "weight": weight}))
    return {
        "name": name,
        "full_annual_return": full["annual_return"],
        "full_max_drawdown": full["max_drawdown"],
        "full_final_nav": full["final_nav"],
        "pre2026_annual_return": pre["annual_return"],
        "pre2026_max_drawdown": pre["max_drawdown"],
        "return_2022": float(yearly.loc[yearly["year"] == 2022, "return"].iloc[0]),
        "return_2023": float(yearly.loc[yearly["year"] == 2023, "return"].iloc[0]),
        "max_abs_weight": full["max_abs_weight"],
    }


def main() -> None:
    """Run market source and blend validation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_daily()
    close = load_close_panel(base["dt"])
    if len(close) != len(base):
        raise ValueError(f"Close panel length {len(close)} != base length {len(base)}")

    price_returns = close.pct_change().fillna(0.0)
    source_rows = []
    blend_rows = []
    source_cache = []
    for lookback in LOOKBACKS:
        for mode in SOURCE_MODES:
            raw_weights = build_raw_weights(close, lookback, mode)
            for vol_window in VOL_WINDOWS:
                for target_vol in TARGET_VOLS:
                    for leverage_cap in LEVERAGE_CAPS:
                        returns, weight = volatility_target_returns(
                            raw_weights, price_returns, vol_window, target_vol, leverage_cap
                        )
                        name = f"{mode}_N{lookback}_VW{vol_window}_TV{int(target_vol * 100)}_C{leverage_cap:g}"
                        row = summarize_returns(base, returns, weight, name)
                        row["mode"] = mode
                        row["lookback"] = lookback
                        row["vol_window"] = vol_window
                        row["target_vol"] = target_vol
                        row["leverage_cap"] = leverage_cap
                        source_rows.append(row)
                        source_cache.append((name, returns, weight, row))

    base_returns = base["ret"].to_numpy(float)
    base_weight = base["weight"].to_numpy(float)
    for name, returns, weight, _ in source_cache:
        for blend_weight in BLEND_WEIGHTS:
            blended_returns = (1 - blend_weight) * base_returns + blend_weight * returns
            blended_weight = (1 - blend_weight) * base_weight + blend_weight * weight
            row = summarize_returns(base, blended_returns, blended_weight, f"blend_{blend_weight:g}__{name}")
            row["source"] = name
            row["blend_weight"] = blend_weight
            blend_rows.append(row)

    source_summary = pd.DataFrame(source_rows).sort_values(
        ["pre2026_annual_return", "full_annual_return"], ascending=False
    )
    blend_summary = pd.DataFrame(blend_rows).sort_values(
        ["pre2026_annual_return", "full_annual_return"], ascending=False
    )
    source_summary.to_csv(OUTPUT_DIR / "market_trend_source_summary.csv", index=False, encoding="utf-8-sig")
    blend_summary.to_csv(OUTPUT_DIR / "market_trend_source_blend_summary.csv", index=False, encoding="utf-8-sig")

    feasible_blend = blend_summary[blend_summary["pre2026_max_drawdown"] >= -0.10].copy()
    print("Top standalone sources:")
    print(
        source_summary[
            [
                "name",
                "full_annual_return",
                "full_max_drawdown",
                "pre2026_annual_return",
                "pre2026_max_drawdown",
                "return_2022",
                "return_2023",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print("\nTop blends with pre2026 drawdown >= -10%:")
    print(
        feasible_blend[
            [
                "name",
                "full_annual_return",
                "full_max_drawdown",
                "pre2026_annual_return",
                "pre2026_max_drawdown",
                "return_2022",
                "return_2023",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
