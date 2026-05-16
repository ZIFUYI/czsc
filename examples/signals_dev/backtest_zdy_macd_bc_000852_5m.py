# -*- coding: utf-8 -*-
"""Backtest zdy_macd_bc_V230422 on 000852.XSHG 5-minute bars."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Avoid optional ClickHouse/lz4 import issues when importing czsc top-level modules.
for mod in [
    "clickhouse_connect",
    "clickhouse_connect.driver",
    "clickhouse_connect.driver.client",
    "clickhouse_connect.driver.httpclient",
    "clickhouse_connect.driver.compression",
]:
    sys.modules.setdefault(mod, MagicMock())

from czsc.core import Freq
from backtest_zdy_macd_bc_000852_15m import fixed_hold_backtest, scan_signals, summarize_trades
from test_zdy_macd_bc_000852 import read_jq_sdk_bars


def main():
    bars = read_jq_sdk_bars(freq="5m", czsc_freq=Freq.F5, sdt="20240101", edt="20260430")
    signals, signal_key = scan_signals(bars, th=50, init_n=900)
    trades = fixed_hold_backtest(bars, signals, hold_bars=3)
    summary = summarize_trades(trades)

    out_dir = ROOT / "examples" / "results" / "macd_area_bc_000852_jq_sdk"
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_file = out_dir / "zdy_macd_bc_5m_hold3_trades.csv"
    summary_file = out_dir / "zdy_macd_bc_5m_hold3_summary.csv"
    trades.to_csv(trades_file, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")

    print(f"bars: {len(bars)}, range: {bars[0].dt} -> {bars[-1].dt}")
    print(f"signal key: {signal_key}")
    print(f"raw signals: {len(signals)}")
    print("raw signal counts:")
    print(signals["value"].value_counts().to_string())
    print("\nbacktest summary:")
    print(summary.to_string(index=False))
    print(f"\nsaved trades: {trades_file}")
    print(f"saved summary: {summary_file}")


if __name__ == "__main__":
    main()
