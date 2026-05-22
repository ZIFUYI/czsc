# -*- coding: utf-8 -*-
"""RBreaker intraday research on 000852.XSHG with JoinQuant 1-minute bars."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd


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

from czsc import Freq
from backtest_rbreaker_000852_5m import (
    bars_to_frame,
    make_cost_sensitivity,
    make_rbreaker_levels,
    make_yearly_summary,
    simulate_rbreaker,
)
from test_zdy_macd_bc_000852 import read_jq_sdk_bars


def main():
    symbol = "000852.XSHG"
    sdt, edt = "20240101", "20260430"
    daily = bars_to_frame(read_jq_sdk_bars(freq="daily", czsc_freq=Freq.D, sdt=sdt, edt=edt))
    intraday = bars_to_frame(read_jq_sdk_bars(freq="1m", czsc_freq=Freq.F1, sdt=sdt, edt=edt))
    levels = make_rbreaker_levels(daily)

    variants = [
        {"name": "趋势突破_双向", "include_trend": True, "include_reversal": False, "direction": "both"},
        {"name": "趋势突破_只做多", "include_trend": True, "include_reversal": False, "direction": "long"},
        {"name": "趋势突破_只做空", "include_trend": True, "include_reversal": False, "direction": "short"},
        {"name": "趋势加反转_双向", "include_trend": True, "include_reversal": True, "direction": "both"},
        {"name": "趋势加反转_只做多", "include_trend": True, "include_reversal": True, "direction": "long"},
        {"name": "趋势加反转_只做空", "include_trend": True, "include_reversal": True, "direction": "short"},
    ]

    all_trades = []
    all_summaries = []
    for kwargs in variants:
        trades, summary = simulate_rbreaker(intraday, levels, **kwargs)
        all_trades.append(trades)
        all_summaries.append(summary)

    trades_df = pd.concat(all_trades, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    available_days = len(set(intraday["trade_date"]).intersection(set(levels["trade_date"])))
    cost_df = make_cost_sensitivity(trades_df, available_days=available_days)
    yearly_df = make_yearly_summary(trades_df)

    out_dir = ROOT / "examples" / "results" / "rbreaker_000852_jq_sdk"
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_file = out_dir / "rbreaker_1m_trades.csv"
    summary_file = out_dir / "rbreaker_1m_summary.csv"
    cost_file = out_dir / "rbreaker_1m_cost_sensitivity.csv"
    yearly_file = out_dir / "rbreaker_1m_yearly.csv"
    trades_df.to_csv(trades_file, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
    cost_df.to_csv(cost_file, index=False, encoding="utf-8-sig")
    yearly_df.to_csv(yearly_file, index=False, encoding="utf-8-sig")

    print(f"symbol: {symbol}")
    print(f"daily bars: {len(daily)}, range: {daily['dt'].iloc[0]} -> {daily['dt'].iloc[-1]}")
    print(f"1m bars: {len(intraday)}, range: {intraday['dt'].iloc[0]} -> {intraday['dt'].iloc[-1]}")
    print(f"rbreaker level days: {len(levels)}")
    print("\nsummary:")
    print(summary_df[summary_df["分组"] == "全部"].to_string(index=False))
    print("\nby signal:")
    signal_rows = summary_df[summary_df["分组"].str.contains("_", regex=False)]
    print(signal_rows.to_string(index=False))
    print("\ncost sensitivity:")
    print(cost_df[cost_df["分组"] == "全部"].to_string(index=False))
    print("\nyearly:")
    print(yearly_df.to_string(index=False))
    print(f"\nsaved trades: {trades_file}")
    print(f"saved summary: {summary_file}")
    print(f"saved cost sensitivity: {cost_file}")
    print(f"saved yearly: {yearly_file}")


if __name__ == "__main__":
    main()
