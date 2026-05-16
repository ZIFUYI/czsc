# -*- coding: utf-8 -*-
"""ATR intraday breakout research on 000852.XSHG with JoinQuant 5-minute bars."""
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
from backtest_rbreaker_000852_5m import bars_to_frame
from test_zdy_macd_bc_000852 import read_jq_sdk_bars


def add_atr(df: pd.DataFrame, n: int) -> pd.Series:
    """Calculate simple rolling ATR."""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (close.shift(1) - low).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def make_atr_break_signals(df: pd.DataFrame, timeperiod: int = 5, th: int = 30) -> pd.DataFrame:
    """Generate signals following tas_atr_break_V230424."""
    out = df.copy(deep=True)
    atr = add_atr(out, timeperiod)
    hh = out["high"].rolling(timeperiod).max()
    ll = out["low"].rolling(timeperiod).min()
    k = th / 10

    upper = hh - k * atr
    lower = ll + k * atr
    central = (upper > out["close"]) & (out["close"] > lower)

    out.loc[:, "strategy"] = f"ATR_BREAK_N{timeperiod}_T{th}"
    out.loc[:, "signal"] = None
    out.loc[(~central) & (out["close"] > lower), "signal"] = "做多_ATR突破"
    out.loc[(~central) & (out["close"] < upper), "signal"] = "做空_ATR突破"
    return out


def make_kcatr_signals(df: pd.DataFrame, n: int = 30, m: int = 16, th: float = 2.0) -> pd.DataFrame:
    """Generate KC-ATR channel breakout signals following kcatr_up_dw_line_V230823."""
    out = df.copy(deep=True)
    atr = add_atr(out, n)
    middle = out["close"].rolling(m).mean()
    upper = middle + atr * th
    lower = middle - atr * th

    out.loc[:, "strategy"] = f"KCATR_N{n}_M{m}_T{th:g}"
    out.loc[:, "signal"] = None
    out.loc[out["close"] > upper, "signal"] = "做多_KCATR"
    out.loc[out["close"] < lower, "signal"] = "做空_KCATR"
    return out


def simulate_intraday(signals: pd.DataFrame, exit_time: str = "14:55") -> pd.DataFrame:
    """Enter next bar open on first daily signal and exit near close."""
    exit_at = pd.to_datetime(exit_time).time()
    trades = []

    for trade_date, day in signals.groupby("trade_date", sort=True):
        day = day.sort_values("dt").reset_index(drop=True)
        for i in range(len(day) - 1):
            bar = day.iloc[i]
            signal = bar["signal"]
            if not isinstance(signal, str) or not signal:
                continue

            entry = day.iloc[i + 1]
            if entry["dt"].time() >= exit_at:
                continue

            exit_candidates = day[(day.index >= i + 1) & (day["dt"].dt.time >= exit_at)]
            exit_bar = exit_candidates.iloc[0] if not exit_candidates.empty else day.iloc[-1]
            direction = "多头" if signal.startswith("做多") else "空头"
            entry_price = float(entry["open"])
            exit_price = float(exit_bar["close"])
            ret = exit_price / entry_price - 1 if direction == "多头" else 1 - exit_price / entry_price
            trades.append(
                {
                    "strategy": bar["strategy"],
                    "symbol": entry["symbol"],
                    "trade_date": trade_date,
                    "signal_dt": bar["dt"],
                    "signal": signal,
                    "direction": direction,
                    "entry_dt": entry["dt"],
                    "exit_dt": exit_bar["dt"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "ret": ret,
                    "ret_bp": round(ret * 10000, 2),
                }
            )
            break

    return pd.DataFrame(trades)


def summarize_trades(trades: pd.DataFrame, available_days: int) -> pd.DataFrame:
    """Summarize trades by strategy / direction."""

    def _summary(df: pd.DataFrame, strategy: str, group: str) -> dict:
        if df.empty:
            return {
                "策略": strategy,
                "分组": group,
                "可交易天数": available_days,
                "交易次数": 0,
                "覆盖率": 0,
                "胜率": 0,
                "平均收益BP": 0,
                "中位收益BP": 0,
                "累计收益": 0,
                "最大回撤": 0,
            }

        equity = (1 + df["ret"]).cumprod()
        drawdown = equity / equity.cummax() - 1
        return {
            "策略": strategy,
            "分组": group,
            "可交易天数": available_days,
            "交易次数": len(df),
            "覆盖率": round(len(df) / available_days, 4),
            "胜率": round((df["ret"] > 0).mean(), 4),
            "平均收益BP": round(df["ret_bp"].mean(), 2),
            "中位收益BP": round(df["ret_bp"].median(), 2),
            "累计收益": round(equity.iloc[-1] - 1, 4),
            "最大回撤": round(drawdown.min(), 4),
        }

    rows = []
    for strategy, df in trades.groupby("strategy", sort=False):
        rows.append(_summary(df, strategy, "全部"))
        for direction, dfg in df.groupby("direction", sort=False):
            rows.append(_summary(dfg, strategy, direction))
        for signal, dfg in df.groupby("signal", sort=False):
            rows.append(_summary(dfg, strategy, signal))
    return pd.DataFrame(rows)


def cost_sensitivity(trades: pd.DataFrame, costs_bp=(0, 2, 5, 10)) -> pd.DataFrame:
    """Summarize cost sensitivity by strategy."""
    rows = []
    for strategy, df in trades.groupby("strategy", sort=False):
        for cost_bp in costs_bp:
            net = df.copy(deep=True)
            net.loc[:, "ret"] = net["ret"] - cost_bp / 10000
            net.loc[:, "ret_bp"] = (net["ret"] * 10000).round(2)
            equity = (1 + net["ret"]).cumprod()
            drawdown = equity / equity.cummax() - 1
            rows.append(
                {
                    "策略": strategy,
                    "单笔成本BP": cost_bp,
                    "交易次数": len(net),
                    "胜率": round((net["ret"] > 0).mean(), 4),
                    "平均收益BP": round(net["ret_bp"].mean(), 2),
                    "中位收益BP": round(net["ret_bp"].median(), 2),
                    "期末净值": round(equity.iloc[-1], 4),
                    "累计收益": round(equity.iloc[-1] - 1, 4),
                    "最大回撤": round(drawdown.min(), 4),
                }
            )
    return pd.DataFrame(rows)


def yearly_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Summarize yearly returns by strategy."""
    rows = []
    trades = trades.copy(deep=True)
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


def main():
    sdt, edt = "20200101", "20260516"
    bars = read_jq_sdk_bars(freq="5m", czsc_freq=Freq.F5, sdt=sdt, edt=edt)
    intraday = bars_to_frame(bars)
    available_days = intraday["trade_date"].nunique()

    signal_frames = []
    for th in [10, 20, 30, 40]:
        signal_frames.append(make_atr_break_signals(intraday, timeperiod=5, th=th))
    for th in [0.8, 1.0, 1.2, 1.5, 2.0]:
        signal_frames.append(make_kcatr_signals(intraday, n=30, m=16, th=th))

    all_trades = []
    for signals in signal_frames:
        trades = simulate_intraday(signals)
        all_trades.append(trades)
    trades = pd.concat(all_trades, ignore_index=True)
    summary = summarize_trades(trades, available_days=available_days)
    cost = cost_sensitivity(trades)
    yearly = yearly_summary(trades)

    out_dir = ROOT / "examples" / "results" / "atr_000852_jq_sdk"
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_file = out_dir / "atr_5m_2020_trades.csv"
    summary_file = out_dir / "atr_5m_2020_summary.csv"
    cost_file = out_dir / "atr_5m_2020_cost.csv"
    yearly_file = out_dir / "atr_5m_2020_yearly.csv"
    trades.to_csv(trades_file, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
    cost.to_csv(cost_file, index=False, encoding="utf-8-sig")
    yearly.to_csv(yearly_file, index=False, encoding="utf-8-sig")

    print(f"5m bars: {len(intraday)}, range: {intraday['dt'].iloc[0]} -> {intraday['dt'].iloc[-1]}")
    print("\nsummary:")
    print(summary[summary["分组"] == "全部"].to_string(index=False))
    print("\ncost:")
    print(cost[cost["单笔成本BP"].isin([0, 5])].to_string(index=False))
    print("\nyearly top candidates:")
    keep = yearly[yearly["策略"].isin(["KCATR_N30_M16_T1", "KCATR_N30_M16_T1.2", "ATR_BREAK_N5_T30"])]
    print(keep.to_string(index=False))
    print(f"\nsaved trades: {trades_file}")
    print(f"saved summary: {summary_file}")
    print(f"saved cost: {cost_file}")
    print(f"saved yearly: {yearly_file}")


if __name__ == "__main__":
    main()
