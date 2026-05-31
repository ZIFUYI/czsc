"""Fetch low-correlation ETF daily OHLCV bars from JQData.

The symbols are selected to add Hong Kong equity, overseas equity, gold and
bond / money-market exposures to the current equity-index CTA pool.

Run:
    .venv/bin/python examples/signals_dev/fetch_daily_ohlcv_jq_low_corr_pool.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_holding_low_corr_pool"
START_DATE = "2010-01-01"
END_DATE = "2026-05-29"
SYMBOLS = (
    "510900.XSHG",  # H-share ETF
    "159920.XSHE",  # Hang Seng ETF
    "513100.XSHG",  # Nasdaq ETF
    "513500.XSHG",  # S&P 500 ETF
    "513030.XSHG",  # Germany ETF
    "518880.XSHG",  # Gold ETF
    "159934.XSHE",  # Gold ETF
    "511010.XSHG",  # Treasury bond ETF
    "511220.XSHG",  # City bond ETF
    "511880.XSHG",  # Money market ETF
)


def load_jq_credentials() -> None:
    """Load JQData credentials from .env without printing secrets."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def fetch_daily(symbol: str) -> pd.DataFrame:
    """Fetch one symbol's raw daily OHLCV from JQData."""
    import jqdatasdk as jq

    user = os.getenv("JQDATA_USERNAME")
    password = os.getenv("JQDATA_PASSWORD")
    if not user or not password:
        raise RuntimeError("请先设置 JQDATA_USERNAME / JQDATA_PASSWORD")

    data = jq.get_price(
        symbol,
        start_date=START_DATE,
        end_date=END_DATE,
        frequency="daily",
        fields=["open", "close", "high", "low", "volume", "money"],
        fq=None,
    )
    if data is None or data.empty:
        raise RuntimeError(f"JQData returned no daily bars for {symbol}")
    data = data.sort_index().reset_index(names="dt")
    data["dt"] = pd.to_datetime(data["dt"]).dt.strftime("%Y-%m-%d")
    data["symbol"] = symbol
    data = data.rename(columns={"volume": "vol", "money": "amount"})
    return data[["dt", "symbol", "open", "high", "low", "close", "vol", "amount"]]


def main() -> None:
    """Fetch all configured symbols and save CSV caches."""
    import jqdatasdk as jq

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    load_jq_credentials()
    jq.auth(os.environ["JQDATA_USERNAME"], os.environ["JQDATA_PASSWORD"])

    rows = []
    for symbol in SYMBOLS:
        data = fetch_daily(symbol)
        file_name = f"{symbol.replace('.', '_')}_daily_{START_DATE.replace('-', '')}_{END_DATE.replace('-', '')}.csv"
        output_file = OUTPUT_DIR / file_name
        data.to_csv(output_file, index=False, encoding="utf-8-sig")
        valid = data.dropna(subset=["close"])
        rows.append(
            {
                "symbol": symbol,
                "file": str(output_file),
                "start": valid["dt"].iloc[0] if not valid.empty else None,
                "end": valid["dt"].iloc[-1] if not valid.empty else None,
                "rows": len(data),
                "valid_rows": len(valid),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "jq_low_corr_fetch_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
