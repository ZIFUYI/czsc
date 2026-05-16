# -*- coding: utf-8 -*-
"""RBreaker 5-minute breakout research on 000852.XSHG, scale=0.3, since 2020."""
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

from czsc.core import Freq
from backtest_rbreaker_000852_5m import bars_to_frame, make_rbreaker_levels, make_yearly_summary, simulate_rbreaker
from test_zdy_macd_bc_000852 import read_jq_sdk_bars


def make_cost_equity(trades: pd.DataFrame, costs_bp=(0, 2, 5, 10)) -> pd.DataFrame:
    """Summarize compounded returns after different per-trade costs."""
    rows = []
    for cost_bp in costs_bp:
        df = trades.copy(deep=True)
        df.loc[:, "ret"] = df["ret"] - cost_bp / 10000
        df.loc[:, "ret_bp"] = (df["ret"] * 10000).round(2)
        equity = (1 + df["ret"]).cumprod()
        drawdown = equity / equity.cummax() - 1
        rows.append(
            {
                "单笔成本BP": cost_bp,
                "交易次数": len(df),
                "胜率": round((df["ret"] > 0).mean(), 4),
                "平均收益BP": round(df["ret_bp"].mean(), 2),
                "中位收益BP": round(df["ret_bp"].median(), 2),
                "期末净值": round(equity.iloc[-1], 4),
                "累计收益": round(equity.iloc[-1] - 1, 4),
                "最大回撤": round(drawdown.min(), 4),
            }
        )
    return pd.DataFrame(rows)


def main():
    symbol = "000852.XSHG"
    sdt, edt = "20200101", "20260516"
    level_scale = 0.3

    daily = bars_to_frame(read_jq_sdk_bars(freq="daily", czsc_freq=Freq.D, sdt=sdt, edt=edt))
    intraday = bars_to_frame(read_jq_sdk_bars(freq="5m", czsc_freq=Freq.F5, sdt=sdt, edt=edt))
    levels = make_rbreaker_levels(daily)

    trades, summary = simulate_rbreaker(
        intraday=intraday,
        levels=levels,
        name="趋势突破_双向_S0.3_2020",
        include_trend=True,
        include_reversal=False,
        direction="both",
        level_scale=level_scale,
    )
    yearly = make_yearly_summary(trades)
    cost = make_cost_equity(trades)

    out_dir = ROOT / "examples" / "results" / "rbreaker_000852_jq_sdk"
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_file = out_dir / "rbreaker_5m_s03_2020_trades.csv"
    summary_file = out_dir / "rbreaker_5m_s03_2020_summary.csv"
    yearly_file = out_dir / "rbreaker_5m_s03_2020_yearly.csv"
    cost_file = out_dir / "rbreaker_5m_s03_2020_cost.csv"
    levels_file = out_dir / "rbreaker_5m_s03_2020_levels.csv"

    trades.to_csv(trades_file, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
    yearly.to_csv(yearly_file, index=False, encoding="utf-8-sig")
    cost.to_csv(cost_file, index=False, encoding="utf-8-sig")
    levels.to_csv(levels_file, index=False, encoding="utf-8-sig")

    print(f"symbol: {symbol}")
    print(f"date range requested: {sdt} -> {edt}")
    print(f"daily bars: {len(daily)}, range: {daily['dt'].iloc[0]} -> {daily['dt'].iloc[-1]}")
    print(f"5m bars: {len(intraday)}, range: {intraday['dt'].iloc[0]} -> {intraday['dt'].iloc[-1]}")
    print(f"level scale: {level_scale}")
    print("\nsummary:")
    print(summary.to_string(index=False))
    print("\ncost:")
    print(cost.to_string(index=False))
    print("\nyearly:")
    print(yearly.to_string(index=False))
    print(f"\nsaved trades: {trades_file}")
    print(f"saved summary: {summary_file}")
    print(f"saved yearly: {yearly_file}")
    print(f"saved cost: {cost_file}")
    print(f"saved levels: {levels_file}")


if __name__ == "__main__":
    main()
