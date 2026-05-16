# -*- coding: utf-8 -*-
"""Backtest zdy_macd_bc_V230422 on 000852.XSHG 15-minute bars.

Rules:

1. 下跌背驰 -> 开多；上涨背驰 -> 开空。
2. The signal is confirmed at bar i close; enter at bar i+1 open.
3. Hold exactly 3 bars and exit at bar i+3 close.
4. While a trade is open, later signals are ignored.
"""
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

from czsc.core import CZSC, Freq
from czsc.signals.zdy import zdy_macd_bc_V230422
from test_zdy_macd_bc_000852 import read_jq_sdk_bars


def scan_signals(bars, th: int = 50, init_n: int = 300) -> tuple[pd.DataFrame, str]:
    """Generate non-其他 MACD area divergence signals."""
    if len(bars) <= init_n + 5:
        raise ValueError(f"15分钟K线数量不足：{len(bars)}")

    c = CZSC(bars[:init_n])
    signals = []
    signal_key = ""
    for i, bar in enumerate(bars[init_n:], start=init_n):
        c.update(bar)
        signal = zdy_macd_bc_V230422(c, di=1, th=th)
        if not signal_key:
            signal_key = next(iter(signal.keys()))

        value = next(iter(signal.values()))
        if "其他" in value:
            continue

        direction = "多头" if value.startswith("下跌") else "空头"
        signals.append(
            {
                "bar_index": i,
                "dt": bar.dt,
                "close": bar.close,
                "signal": f"{signal_key}_{value}",
                "value": value,
                "direction": direction,
            }
        )

    return pd.DataFrame(signals), signal_key


def fixed_hold_backtest(bars, signals: pd.DataFrame, hold_bars: int = 3) -> pd.DataFrame:
    """Enter next open after signal and exit after hold_bars closes."""
    trades = []
    next_available_index = 0

    for _, sig in signals.iterrows():
        signal_index = int(sig["bar_index"])
        entry_index = signal_index + 1
        exit_index = signal_index + hold_bars
        if entry_index >= len(bars) or exit_index >= len(bars):
            continue

        if entry_index < next_available_index:
            continue

        direction = sig["direction"]
        entry_bar = bars[entry_index]
        exit_bar = bars[exit_index]
        entry_price = float(entry_bar.open)
        exit_price = float(exit_bar.close)
        ret = exit_price / entry_price - 1 if direction == "多头" else 1 - exit_price / entry_price

        trades.append(
            {
                "symbol": entry_bar.symbol,
                "signal_dt": sig["dt"],
                "signal_value": sig["value"],
                "direction": direction,
                "entry_dt": entry_bar.dt,
                "exit_dt": exit_bar.dt,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "hold_bars": hold_bars,
                "ret": ret,
                "ret_bp": round(ret * 10000, 2),
            }
        )
        next_available_index = exit_index + 1

    return pd.DataFrame(trades)


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Summarize trades by all / direction / signal value."""

    def _summary(df: pd.DataFrame, name: str) -> dict:
        if df.empty:
            return {
                "分组": name,
                "交易次数": 0,
                "胜率": 0,
                "平均收益BP": 0,
                "中位收益BP": 0,
                "累计收益": 0,
                "最大回撤": 0,
                "盈亏比": 0,
            }

        equity = (1 + df["ret"]).cumprod()
        drawdown = equity / equity.cummax() - 1
        win = df[df["ret"] > 0]["ret"]
        loss = df[df["ret"] <= 0]["ret"]
        profit_loss_ratio = abs(win.mean() / loss.mean()) if len(win) and len(loss) and loss.mean() else 0
        return {
            "分组": name,
            "交易次数": len(df),
            "胜率": round((df["ret"] > 0).mean(), 4),
            "平均收益BP": round(df["ret_bp"].mean(), 2),
            "中位收益BP": round(df["ret_bp"].median(), 2),
            "累计收益": round(equity.iloc[-1] - 1, 4),
            "最大回撤": round(drawdown.min(), 4),
            "盈亏比": round(profit_loss_ratio, 4),
        }

    rows = [_summary(trades, "全部")]
    for direction, df in trades.groupby("direction", sort=False):
        rows.append(_summary(df, direction))
    for value, df in trades.groupby("signal_value", sort=False):
        rows.append(_summary(df, value))

    return pd.DataFrame(rows)


def main():
    bars = read_jq_sdk_bars(freq="15m", czsc_freq=Freq.F15, sdt="20240101", edt="20260430")
    signals, signal_key = scan_signals(bars, th=50, init_n=300)
    trades = fixed_hold_backtest(bars, signals, hold_bars=3)
    summary = summarize_trades(trades)

    out_dir = ROOT / "examples" / "results" / "macd_area_bc_000852_jq_sdk"
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_file = out_dir / "zdy_macd_bc_15m_hold3_trades.csv"
    summary_file = out_dir / "zdy_macd_bc_15m_hold3_summary.csv"
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
