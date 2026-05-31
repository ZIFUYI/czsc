"""Low-correlation asset-pool WFO validation for cross-sectional daily CTA.

This expands the original three-symbol pool with Hong Kong equity, overseas
equity, gold and bond / money-market ETFs fetched from JQData.

Run:
    .venv/bin/python examples/signals_dev/validate_daily_cta_low_corr_pool_walk_forward.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import validate_daily_cta_cross_section_walk_forward as xsec
from validate_daily_cta_extended_pool_walk_forward import EXTENDED_MODES, extended_raw_weights_from_scores
from validate_daily_cta_walk_forward import DATA_FILES as BASE_DATA_FILES

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_cross_section_walk_forward_low_corr_pool"
LOW_CORR_DATA_FILES = {
    **BASE_DATA_FILES,
    "510900.XSHG": ROOT
    / "examples"
    / "results"
    / "daily_holding_low_corr_pool"
    / "510900_XSHG_daily_20100101_20260529.csv",
    "159920.XSHE": ROOT
    / "examples"
    / "results"
    / "daily_holding_low_corr_pool"
    / "159920_XSHE_daily_20100101_20260529.csv",
    "513100.XSHG": ROOT
    / "examples"
    / "results"
    / "daily_holding_low_corr_pool"
    / "513100_XSHG_daily_20100101_20260529.csv",
    "513500.XSHG": ROOT
    / "examples"
    / "results"
    / "daily_holding_low_corr_pool"
    / "513500_XSHG_daily_20100101_20260529.csv",
    "513030.XSHG": ROOT
    / "examples"
    / "results"
    / "daily_holding_low_corr_pool"
    / "513030_XSHG_daily_20100101_20260529.csv",
    "518880.XSHG": ROOT
    / "examples"
    / "results"
    / "daily_holding_low_corr_pool"
    / "518880_XSHG_daily_20100101_20260529.csv",
    "159934.XSHE": ROOT
    / "examples"
    / "results"
    / "daily_holding_low_corr_pool"
    / "159934_XSHE_daily_20100101_20260529.csv",
    "511010.XSHG": ROOT
    / "examples"
    / "results"
    / "daily_holding_low_corr_pool"
    / "511010_XSHG_daily_20100101_20260529.csv",
    "511220.XSHG": ROOT
    / "examples"
    / "results"
    / "daily_holding_low_corr_pool"
    / "511220_XSHG_daily_20100101_20260529.csv",
    "511880.XSHG": ROOT
    / "examples"
    / "results"
    / "daily_holding_low_corr_pool"
    / "511880_XSHG_daily_20100101_20260529.csv",
}


def load_low_corr_daily_data(symbol: str) -> pd.DataFrame:
    """Load one low-correlation-pool daily OHLCV cache."""
    file_csv = LOW_CORR_DATA_FILES[symbol]
    if not file_csv.exists():
        raise FileNotFoundError(f"Daily data file not found: {file_csv}")
    data = pd.read_csv(file_csv, parse_dates=["dt"])
    data["dt"] = pd.to_datetime(data["dt"]).dt.tz_localize(None)
    data["symbol"] = symbol
    return data.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)


def load_low_corr_panel() -> pd.DataFrame:
    """Load and align close prices for all low-correlation-pool symbols."""
    frames = []
    for symbol in LOW_CORR_DATA_FILES:
        data = load_low_corr_daily_data(symbol)[["dt", "close"]].dropna(subset=["close"])
        data = data.rename(columns={"close": symbol})
        frames.append(data)
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on="dt", how="inner")
    return panel.sort_values("dt").reset_index(drop=True)


def save_correlation_audit(panel: pd.DataFrame) -> None:
    """Save simple daily-return correlation diagnostics for the selected pool."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    returns = panel[list(LOW_CORR_DATA_FILES)].pct_change().dropna(how="all")
    corr = returns.corr()
    corr.to_csv(OUTPUT_DIR / "low_corr_pool_return_correlation.csv", encoding="utf-8-sig")
    summary = pd.DataFrame(
        {
            "symbol": corr.index,
            "avg_abs_corr": [corr.loc[symbol].drop(symbol).abs().mean() for symbol in corr.index],
            "max_abs_corr": [corr.loc[symbol].drop(symbol).abs().max() for symbol in corr.index],
        }
    )
    summary.to_csv(OUTPUT_DIR / "low_corr_pool_correlation_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    """Run the cross-sectional WFO on the low-correlation asset pool."""
    panel = load_low_corr_panel()
    save_correlation_audit(panel)
    xsec.SYMBOLS = tuple(LOW_CORR_DATA_FILES)
    xsec.OUTPUT_DIR = OUTPUT_DIR
    xsec.load_panel = lambda: panel.copy()
    xsec.MODES = EXTENDED_MODES
    xsec.raw_weights_from_scores = extended_raw_weights_from_scores
    xsec.main()


if __name__ == "__main__":
    main()
