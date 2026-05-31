"""Fetch raw daily OHLCV bars from JQData for extended daily CTA pool tests.

Run:
    .venv/bin/python examples/signals_dev/fetch_daily_ohlcv_jq_extended_pool.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_holding_extended_pool"
START_DATE = "2010-01-01"
END_DATE = "2026-05-29"
SYMBOLS = (
    "000300.XSHG",  # CSI 300
    "000016.XSHG",  # SSE 50
    "399006.XSHE",  # ChiNext
    "000906.XSHG",  # CSI 800
    "000985.XSHG",  # CSI All Share
    "000688.XSHG",  # STAR 50, shorter history; useful for audit but not 2020 WFO
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

    load_jq_credentials()
    user = os.getenv("JQDATA_USERNAME")
    password = os.getenv("JQDATA_PASSWORD")
    if not user or not password:
        raise RuntimeError("请先设置 JQDATA_USERNAME / JQDATA_PASSWORD")

    jq.auth(user, password)
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for symbol in SYMBOLS:
        data = fetch_daily(symbol)
        file_name = f"{symbol.replace('.', '_')}_daily_{START_DATE.replace('-', '')}_{END_DATE.replace('-', '')}.csv"
        output_file = OUTPUT_DIR / file_name
        data.to_csv(output_file, index=False, encoding="utf-8-sig")
        rows.append(
            {
                "symbol": symbol,
                "file": str(output_file),
                "start": data["dt"].iloc[0],
                "end": data["dt"].iloc[-1],
                "rows": len(data),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "jq_fetch_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
