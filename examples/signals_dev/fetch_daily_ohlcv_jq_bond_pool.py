"""Fetch bond ETF daily OHLCV bars from JQData.

These assets are candidates for improving the weak 2022-2023 windows left by
equity / gold / commodity overlays.

Run:
    .venv/bin/python examples/signals_dev/fetch_daily_ohlcv_jq_bond_pool.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_holding_bond_pool"
START_DATE = "2010-01-01"
END_DATE = "2026-05-29"
SYMBOLS = (
    "511020.XSHG",  # 活跃国债
    "511030.XSHG",  # 公司债
    "511180.XSHG",  # 上证转债
    "511260.XSHG",  # 十年国债
    "511270.XSHG",  # 10年地债
    "511360.XSHG",  # 短融ETF
    "511380.XSHG",  # 转债ETF
    "511660.XSHG",  # 建信添益
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


def fetch_daily(symbol: str) -> tuple[pd.DataFrame, dict]:
    """Fetch one symbol's raw daily OHLCV from JQData."""
    import jqdatasdk as jq

    info = jq.get_security_info(symbol)
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
    data = data[["dt", "symbol", "open", "high", "low", "close", "vol", "amount"]]
    valid = data.dropna(subset=["close"])
    row = {
        "symbol": symbol,
        "display_name": getattr(info, "display_name", None),
        "name": getattr(info, "name", None),
        "security_start_date": str(getattr(info, "start_date", None)),
        "security_end_date": str(getattr(info, "end_date", None)),
        "start": valid["dt"].iloc[0] if not valid.empty else None,
        "end": valid["dt"].iloc[-1] if not valid.empty else None,
        "rows": len(data),
        "valid_rows": len(valid),
    }
    return data, row


def main() -> None:
    """Fetch all configured symbols and save CSV caches."""
    import jqdatasdk as jq

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    load_jq_credentials()
    user = os.getenv("JQDATA_USERNAME")
    password = os.getenv("JQDATA_PASSWORD")
    if not user or not password:
        raise RuntimeError("请先设置 JQDATA_USERNAME / JQDATA_PASSWORD")
    jq.auth(user, password)

    rows = []
    for symbol in SYMBOLS:
        data, row = fetch_daily(symbol)
        file_name = f"{symbol.replace('.', '_')}_daily_{START_DATE.replace('-', '')}_{END_DATE.replace('-', '')}.csv"
        output_file = OUTPUT_DIR / file_name
        data.to_csv(output_file, index=False, encoding="utf-8-sig")
        row["file"] = str(output_file)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "jq_bond_fetch_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
