"""Fetch public REIT daily OHLCV bars from JQData.

REITs are tested as an additional low-correlation source for the weak
2022-2023 windows. The pool is discovered from JQData's ``fund`` securities
where ``type == 'reits'`` and the listing date is no later than the configured
backtest end date.

Run:
    uv run --no-sync python examples/signals_dev/fetch_daily_ohlcv_jq_reit_pool.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_holding_reit_pool"
START_DATE = "2010-01-01"
END_DATE = "2026-05-29"


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


def discover_reit_symbols() -> pd.DataFrame:
    """Discover listed public REIT symbols from JQData metadata."""
    import jqdatasdk as jq

    securities = jq.get_all_securities(["fund"])
    reits = securities[securities["type"].eq("reits")].copy()
    reits = reits[pd.to_datetime(reits["start_date"]) <= pd.Timestamp(END_DATE)].copy()
    return reits.sort_values(["start_date", "display_name"])


def fetch_daily(symbol: str, info: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Fetch one symbol's raw daily OHLCV from JQData."""
    import jqdatasdk as jq

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
        "display_name": info["display_name"],
        "name": info["name"],
        "security_start_date": str(info["start_date"]),
        "security_end_date": str(info["end_date"]),
        "start": valid["dt"].iloc[0] if not valid.empty else None,
        "end": valid["dt"].iloc[-1] if not valid.empty else None,
        "rows": len(data),
        "valid_rows": len(valid),
    }
    return data, row


def main() -> None:
    """Fetch all discovered REIT symbols and save CSV caches."""
    import jqdatasdk as jq

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    load_jq_credentials()
    user = os.getenv("JQDATA_USERNAME")
    password = os.getenv("JQDATA_PASSWORD")
    if not user or not password:
        raise RuntimeError("请先设置 JQDATA_USERNAME / JQDATA_PASSWORD")
    jq.auth(user, password)

    reits = discover_reit_symbols()
    rows = []
    for symbol, info in reits.iterrows():
        data, row = fetch_daily(symbol, info)
        file_name = f"{symbol.replace('.', '_')}_daily_{START_DATE.replace('-', '')}_{END_DATE.replace('-', '')}.csv"
        output_file = OUTPUT_DIR / file_name
        data.to_csv(output_file, index=False, encoding="utf-8-sig")
        row["file"] = str(output_file)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "jq_reit_fetch_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
