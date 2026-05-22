# -*- coding: utf-8 -*-
"""RBreaker intraday research on 000852.XSHG with JoinQuant 5-minute bars.

This script uses the previous trading day's daily HLC to calculate today's
RBreaker levels. Signals are evaluated on 5-minute bar close, entries use the
next 5-minute bar open, and open trades are forced out near the close.
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

from czsc import Freq
from test_zdy_macd_bc_000852 import read_jq_sdk_bars


def bars_to_frame(bars) -> pd.DataFrame:
    """Convert RawBar list to DataFrame."""
    rows = []
    for bar in bars:
        dt = pd.to_datetime(bar.dt)
        rows.append(
            {
                "symbol": bar.symbol,
                "dt": dt,
                "trade_date": dt.date(),
                "open": float(bar.open),
                "close": float(bar.close),
                "high": float(bar.high),
                "low": float(bar.low),
                "vol": float(bar.vol),
                "amount": float(bar.amount),
            }
        )
    return pd.DataFrame(rows)


def make_rbreaker_levels(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate RBreaker levels for each trading day from the previous day."""
    daily = daily.sort_values("dt").reset_index(drop=True).copy()
    rows = []
    for i in range(1, len(daily)):
        prev = daily.iloc[i - 1]
        cur = daily.iloc[i]
        high, close, low = prev["high"], prev["close"], prev["low"]
        pivot = (high + close + low) / 3
        rows.append(
            {
                "trade_date": cur["trade_date"],
                "prev_dt": prev["dt"],
                "prev_high": high,
                "prev_close": close,
                "prev_low": low,
                "pivot": pivot,
                "break_buy": high + 2 * pivot - 2 * low,
                "see_sell": pivot + high - low,
                "verse_sell": 2 * pivot - low,
                "verse_buy": 2 * pivot - high,
                "see_buy": pivot - (high - low),
                "break_sell": low - 2 * (high - pivot),
            }
        )
    return pd.DataFrame(rows)


def summarize_trades(trades: pd.DataFrame, available_days: int) -> pd.DataFrame:
    """Summarize strategy performance."""

    def _summary(df: pd.DataFrame, name: str) -> dict:
        if df.empty:
            return {
                "分组": name,
                "可交易天数": available_days,
                "交易次数": 0,
                "覆盖率": 0,
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
            "可交易天数": available_days,
            "交易次数": len(df),
            "覆盖率": round(len(df) / available_days, 4) if available_days else 0,
            "胜率": round((df["ret"] > 0).mean(), 4),
            "平均收益BP": round(df["ret_bp"].mean(), 2),
            "中位收益BP": round(df["ret_bp"].median(), 2),
            "累计收益": round(equity.iloc[-1] - 1, 4),
            "最大回撤": round(drawdown.min(), 4),
            "盈亏比": round(profit_loss_ratio, 4),
        }

    rows = [_summary(trades, "全部")]
    if not trades.empty:
        for direction, df in trades.groupby("direction", sort=False):
            rows.append(_summary(df, direction))
        for signal, df in trades.groupby("signal", sort=False):
            rows.append(_summary(df, signal))
    return pd.DataFrame(rows)


def make_cost_sensitivity(trades: pd.DataFrame, available_days: int, costs_bp=(0, 2, 5, 10)) -> pd.DataFrame:
    """Create round-trip cost sensitivity summaries."""
    rows = []
    for strategy, df in trades.groupby("strategy", sort=False):
        for cost_bp in costs_bp:
            net = df.copy(deep=True).reset_index(drop=True)
            net.loc[:, "ret"] = net["ret"] - cost_bp / 10000
            net.loc[:, "ret_bp"] = (net["ret"] * 10000).round(2)
            summary = summarize_trades(net, available_days)
            summary = summary[summary["分组"] == "全部"].copy()
            summary.insert(0, "策略", strategy)
            summary.insert(1, "单笔成本BP", cost_bp)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def make_yearly_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Summarize compounded trade returns by calendar year."""
    rows = []
    if trades.empty:
        return pd.DataFrame(rows)

    trades = trades.copy(deep=True).reset_index(drop=True)
    trades.loc[:, "year"] = pd.to_datetime(trades["entry_dt"]).dt.year
    for (strategy, year), df in trades.groupby(["strategy", "year"], sort=False):
        equity = (1 + df["ret"]).cumprod()
        drawdown = equity / equity.cummax() - 1
        rows.append(
            {
                "策略": strategy,
                "年份": year,
                "交易次数": len(df),
                "胜率": round((df["ret"] > 0).mean(), 4),
                "平均收益BP": round(df["ret_bp"].mean(), 2),
                "累计收益": round(equity.iloc[-1] - 1, 4),
                "最大回撤": round(drawdown.min(), 4),
            }
        )
    return pd.DataFrame(rows)


def simulate_rbreaker(
    intraday: pd.DataFrame,
    levels: pd.DataFrame,
    name: str,
    include_trend: bool = True,
    include_reversal: bool = False,
    direction: str = "both",
    exit_time: str = "14:55",
    level_scale: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a one-trade-per-day intraday RBreaker strategy."""
    assert direction in {"both", "long", "short"}
    exit_at = pd.to_datetime(exit_time).time()
    level_map = {row["trade_date"]: row for _, row in levels.iterrows()}
    available_dates = sorted(set(intraday["trade_date"]).intersection(level_map))
    trades = []

    for trade_date in available_dates:
        day = intraday[intraday["trade_date"] == trade_date].sort_values("dt").reset_index(drop=True)
        if len(day) < 5:
            continue

        lv = level_map[trade_date].copy()
        if level_scale != 1.0:
            day_open = float(day.iloc[0]["open"])
            for col in ["break_buy", "see_sell", "verse_sell", "verse_buy", "see_buy", "break_sell"]:
                lv[col] = day_open + level_scale * (lv[col] - day_open)

        day_high = -float("inf")
        day_low = float("inf")

        for i in range(len(day) - 1):
            bar = day.iloc[i]
            day_high = max(day_high, bar["high"])
            day_low = min(day_low, bar["low"])

            signal = None
            trade_dir = None
            if include_trend and bar["close"] > lv["break_buy"]:
                signal, trade_dir = "做多_趋势", "多头"
            elif include_trend and bar["close"] < lv["break_sell"]:
                signal, trade_dir = "做空_趋势", "空头"
            elif include_reversal and day_high > lv["see_sell"] and bar["close"] < lv["verse_sell"]:
                signal, trade_dir = "做空_反转", "空头"
            elif include_reversal and day_low < lv["see_buy"] and bar["close"] > lv["verse_buy"]:
                signal, trade_dir = "做多_反转", "多头"

            if not signal:
                continue
            if direction == "long" and trade_dir != "多头":
                continue
            if direction == "short" and trade_dir != "空头":
                continue

            entry = day.iloc[i + 1]
            if entry["dt"].time() >= exit_at:
                continue

            exit_candidates = day[(day.index >= i + 1) & (day["dt"].dt.time >= exit_at)]
            exit_bar = exit_candidates.iloc[0] if not exit_candidates.empty else day.iloc[-1]
            entry_price = float(entry["open"])
            exit_price = float(exit_bar["close"])
            ret = exit_price / entry_price - 1 if trade_dir == "多头" else 1 - exit_price / entry_price

            trades.append(
                {
                    "strategy": name,
                    "level_scale": level_scale,
                    "symbol": entry["symbol"],
                    "trade_date": trade_date,
                    "prev_dt": lv["prev_dt"],
                    "signal_dt": bar["dt"],
                    "signal": signal,
                    "direction": trade_dir,
                    "entry_dt": entry["dt"],
                    "exit_dt": exit_bar["dt"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "ret": ret,
                    "ret_bp": round(ret * 10000, 2),
                    "break_buy": lv["break_buy"],
                    "break_sell": lv["break_sell"],
                    "see_sell": lv["see_sell"],
                    "verse_sell": lv["verse_sell"],
                    "verse_buy": lv["verse_buy"],
                    "see_buy": lv["see_buy"],
                }
            )
            break

    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades_df, available_days=len(available_dates))
    summary.insert(0, "策略", name)
    summary.insert(1, "价位缩放", level_scale)
    return trades_df, summary


def make_breakout_scale_sweep(intraday: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    """Sweep level_scale for the trend-breakout-only strategy."""
    rows = []
    for scale in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        _, summary = simulate_rbreaker(
            intraday=intraday,
            levels=levels,
            name=f"趋势突破_双向_S{scale}",
            include_trend=True,
            include_reversal=False,
            direction="both",
            level_scale=scale,
        )
        rows.append(summary[summary["分组"] == "全部"])
    return pd.concat(rows, ignore_index=True)


def main():
    symbol = "000852.XSHG"
    sdt, edt = "20240101", "20260430"
    daily = bars_to_frame(read_jq_sdk_bars(freq="daily", czsc_freq=Freq.D, sdt=sdt, edt=edt))
    intraday = bars_to_frame(read_jq_sdk_bars(freq="5m", czsc_freq=Freq.F5, sdt=sdt, edt=edt))
    levels = make_rbreaker_levels(daily)

    variants = [
        {"name": "趋势突破_双向", "include_trend": True, "include_reversal": False, "direction": "both"},
        {"name": "趋势突破_只做多", "include_trend": True, "include_reversal": False, "direction": "long"},
        {"name": "趋势突破_只做空", "include_trend": True, "include_reversal": False, "direction": "short"},
        {"name": "趋势加反转_双向", "include_trend": True, "include_reversal": True, "direction": "both"},
        {"name": "趋势加反转_只做多", "include_trend": True, "include_reversal": True, "direction": "long"},
        {"name": "趋势加反转_只做空", "include_trend": True, "include_reversal": True, "direction": "short"},
        {"name": "只反转_双向", "include_trend": False, "include_reversal": True, "direction": "both"},
        {"name": "只反转_只做多", "include_trend": False, "include_reversal": True, "direction": "long"},
        {"name": "只反转_只做空", "include_trend": False, "include_reversal": True, "direction": "short"},
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
    scale_sweep_df = make_breakout_scale_sweep(intraday, levels)

    out_dir = ROOT / "examples" / "results" / "rbreaker_000852_jq_sdk"
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_file = out_dir / "rbreaker_5m_trades.csv"
    summary_file = out_dir / "rbreaker_5m_summary.csv"
    levels_file = out_dir / "rbreaker_daily_levels.csv"
    cost_file = out_dir / "rbreaker_5m_cost_sensitivity.csv"
    yearly_file = out_dir / "rbreaker_5m_yearly.csv"
    scale_sweep_file = out_dir / "rbreaker_5m_breakout_scale_sweep.csv"
    trades_df.to_csv(trades_file, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
    levels.to_csv(levels_file, index=False, encoding="utf-8-sig")
    cost_df.to_csv(cost_file, index=False, encoding="utf-8-sig")
    yearly_df.to_csv(yearly_file, index=False, encoding="utf-8-sig")
    scale_sweep_df.to_csv(scale_sweep_file, index=False, encoding="utf-8-sig")

    print(f"symbol: {symbol}")
    print(f"daily bars: {len(daily)}, range: {daily['dt'].iloc[0]} -> {daily['dt'].iloc[-1]}")
    print(f"5m bars: {len(intraday)}, range: {intraday['dt'].iloc[0]} -> {intraday['dt'].iloc[-1]}")
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
    print("\nbreakout scale sweep:")
    print(scale_sweep_df.to_string(index=False))
    print(f"\nsaved trades: {trades_file}")
    print(f"saved summary: {summary_file}")
    print(f"saved levels: {levels_file}")
    print(f"saved cost sensitivity: {cost_file}")
    print(f"saved yearly: {yearly_file}")
    print(f"saved breakout scale sweep: {scale_sweep_file}")


if __name__ == "__main__":
    main()
