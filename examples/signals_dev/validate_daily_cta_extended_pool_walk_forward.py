"""Extended-symbol-pool walk-forward validation for cross-sectional daily CTA.

This script reuses the existing three-symbol cross-sectional WFO implementation
and only replaces the close-price panel and output directory. It keeps the same
candidate library, selection policies, fees, folds and risk controls so the
result is directly comparable with the original three-symbol research.

Run:
    .venv/bin/python examples/signals_dev/validate_daily_cta_extended_pool_walk_forward.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import validate_daily_cta_cross_section_walk_forward as xsec
from validate_daily_cta_walk_forward import DATA_FILES as BASE_DATA_FILES

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_cross_section_walk_forward_extended_pool"
ORIGINAL_RAW_WEIGHTS_FROM_SCORES = xsec.raw_weights_from_scores
EXTENDED_DATA_FILES = {
    **BASE_DATA_FILES,
    "000300.XSHG": ROOT
    / "examples"
    / "results"
    / "daily_holding_extended_pool"
    / "000300_XSHG_daily_20100101_20260529.csv",
    "000016.XSHG": ROOT
    / "examples"
    / "results"
    / "daily_holding_extended_pool"
    / "000016_XSHG_daily_20100101_20260529.csv",
    "399006.XSHE": ROOT
    / "examples"
    / "results"
    / "daily_holding_extended_pool"
    / "399006_XSHE_daily_20100101_20260529.csv",
    "000906.XSHG": ROOT
    / "examples"
    / "results"
    / "daily_holding_extended_pool"
    / "000906_XSHG_daily_20100101_20260529.csv",
    "000985.XSHG": ROOT
    / "examples"
    / "results"
    / "daily_holding_extended_pool"
    / "000985_XSHG_daily_20100101_20260529.csv",
}
EXTENDED_MODES = (
    "rank_ls",
    "rank_long",
    "market_switch",
    "trend_equal",
    "rank_ls_top2",
    "rank_ls_top3",
    "rank_long_top2",
    "rank_short_bottom2",
)


def load_extended_daily_data(symbol: str) -> pd.DataFrame:
    """Load one extended-pool daily OHLCV cache."""
    file_csv = EXTENDED_DATA_FILES[symbol]
    if not file_csv.exists():
        raise FileNotFoundError(f"Daily data file not found: {file_csv}")
    data = pd.read_csv(file_csv, parse_dates=["dt"])
    data["dt"] = pd.to_datetime(data["dt"]).dt.tz_localize(None)
    data["symbol"] = symbol
    return data.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)


def load_extended_panel() -> pd.DataFrame:
    """Load and align close prices for all extended-pool symbols."""
    frames = []
    for symbol in EXTENDED_DATA_FILES:
        data = load_extended_daily_data(symbol)[["dt", "close"]].dropna(subset=["close"])
        data = data.rename(columns={"close": symbol})
        frames.append(data)
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on="dt", how="inner")
    return panel.sort_values("dt").reset_index(drop=True)


def extended_raw_weights_from_scores(scores: pd.DataFrame, mode: str, threshold: float) -> pd.DataFrame:
    """Convert scores into raw weights with extra diversified extended-pool modes."""
    if mode in {"rank_ls", "rank_long", "market_switch", "trend_equal"}:
        return ORIGINAL_RAW_WEIGHTS_FROM_SCORES(scores, mode, threshold)

    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    values = scores.to_numpy(float)
    columns = list(scores.columns)
    for idx, row in enumerate(values):
        valid = pd.notna(row)
        if valid.sum() < 3:
            continue
        valid_values = row[valid]
        valid_columns = [col for col, ok in zip(columns, valid, strict=True) if ok]
        order = valid_values.argsort()

        if mode in {"rank_ls_top2", "rank_ls_top3"}:
            top_n = 2 if mode == "rank_ls_top2" else 3
            if len(order) < top_n * 2:
                continue
            top_idx = order[-top_n:]
            bottom_idx = order[:top_n]
            spread = float(valid_values[top_idx].mean() - valid_values[bottom_idx].mean())
            if spread <= threshold:
                continue
            for j in top_idx:
                weights.at[idx, valid_columns[j]] = 0.5 / top_n
            for j in bottom_idx:
                weights.at[idx, valid_columns[j]] = -0.5 / top_n
        elif mode == "rank_long_top2":
            positive_idx = [j for j in order[::-1][:2] if valid_values[j] > threshold]
            if positive_idx:
                for j in positive_idx:
                    weights.at[idx, valid_columns[j]] = 1.0 / len(positive_idx)
        elif mode == "rank_short_bottom2":
            negative_idx = [j for j in order[:2] if valid_values[j] < -threshold]
            if negative_idx:
                for j in negative_idx:
                    weights.at[idx, valid_columns[j]] = -1.0 / len(negative_idx)
        else:
            raise ValueError(f"Unsupported extended mode: {mode}")
    return weights


def main() -> None:
    """Run the original cross-sectional WFO on the extended symbol pool."""
    xsec.SYMBOLS = tuple(EXTENDED_DATA_FILES)
    xsec.OUTPUT_DIR = OUTPUT_DIR
    xsec.load_panel = load_extended_panel
    xsec.MODES = EXTENDED_MODES
    xsec.raw_weights_from_scores = extended_raw_weights_from_scores
    xsec.main()


if __name__ == "__main__":
    main()
