# -*- coding: utf-8 -*-
"""Dual Thrust intraday research on 000852.XSHG with JoinQuant 5-minute bars."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
from backtest_rbreaker_000852_5m import bars_to_frame
from test_zdy_macd_bc_000852 import read_jq_sdk_bars


def make_dual_thrust_levels(daily: pd.DataFrame, n: int, k1: float, k2: float) -> pd.DataFrame:
    """Calculate classic Dual Thrust levels from previous N daily bars."""
    daily = daily.sort_values("dt").reset_index(drop=True).copy()
    rows = []
    for i in range(n, len(daily)):
        window = daily.iloc[i - n : i]
        cur = daily.iloc[i]
        hh = float(window["high"].max())
        hc = float(window["close"].max())
        lc = float(window["close"].min())
        ll = float(window["low"].min())
        range_ = max(hh - lc, hc - ll)
        rows.append(
            {
                "trade_date": cur["trade_date"],
                "N": n,
                "K1": k1,
                "K2": k2,
                "HH": hh,
                "HC": hc,
                "LC": lc,
                "LL": ll,
                "range": range_,
                "window_start": window.iloc[0]["dt"],
                "window_end": window.iloc[-1]["dt"],
            }
        )
    return pd.DataFrame(rows)


def simulate_dual_thrust(
    intraday: pd.DataFrame,
    levels: pd.DataFrame,
    name: str,
    exit_time: str = "14:55",
) -> tuple[pd.DataFrame, int]:
    """Simulate a one-trade-per-day classic Dual Thrust breakout strategy."""
    exit_at = pd.to_datetime(exit_time).time()
    level_map = {row["trade_date"]: row for _, row in levels.iterrows()}
    available_dates = sorted(set(intraday["trade_date"]).intersection(level_map))
    trades = []

    for trade_date in available_dates:
        day = intraday[intraday["trade_date"] == trade_date].sort_values("dt").reset_index(drop=True)
        if len(day) < 5:
            continue

        lv = level_map[trade_date]
        day_open = float(day.iloc[0]["open"])
        buy_line = day_open + float(lv["range"]) * float(lv["K1"])
        sell_line = day_open - float(lv["range"]) * float(lv["K2"])

        for i in range(len(day) - 1):
            bar = day.iloc[i]
            signal = None
            direction = None
            if float(bar["close"]) > buy_line:
                signal, direction = "做多_DualThrust", "多头"
            elif float(bar["close"]) < sell_line:
                signal, direction = "做空_DualThrust", "空头"

            if not signal:
                continue

            entry = day.iloc[i + 1]
            if entry["dt"].time() >= exit_at:
                continue

            exit_candidates = day[(day.index >= i + 1) & (day["dt"].dt.time >= exit_at)]
            exit_bar = exit_candidates.iloc[0] if not exit_candidates.empty else day.iloc[-1]
            entry_price = float(entry["open"])
            exit_price = float(exit_bar["close"])
            ret = exit_price / entry_price - 1 if direction == "多头" else 1 - exit_price / entry_price

            trades.append(
                {
                    "strategy": name,
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
                    "N": int(lv["N"]),
                    "K1": float(lv["K1"]),
                    "K2": float(lv["K2"]),
                    "day_open": day_open,
                    "range": float(lv["range"]),
                    "buy_line": buy_line,
                    "sell_line": sell_line,
                }
            )
            break

    return pd.DataFrame(trades), len(available_dates)


def summarize_trades(trades: pd.DataFrame, available_days: int) -> dict:
    """Summarize a single strategy's trade list."""
    if trades.empty:
        return {
            "可交易天数": available_days,
            "交易次数": 0,
            "覆盖率": 0,
            "胜率": 0,
            "平均收益BP": 0,
            "中位收益BP": 0,
            "累计收益": 0,
            "最大回撤": 0,
        }

    equity = (1 + trades["ret"]).cumprod()
    drawdown = equity / equity.cummax() - 1
    return {
        "可交易天数": available_days,
        "交易次数": len(trades),
        "覆盖率": round(len(trades) / available_days, 4) if available_days else 0,
        "胜率": round((trades["ret"] > 0).mean(), 4),
        "平均收益BP": round(trades["ret_bp"].mean(), 2),
        "中位收益BP": round(trades["ret_bp"].median(), 2),
        "累计收益": round(equity.iloc[-1] - 1, 4),
        "最大回撤": round(drawdown.min(), 4),
    }


def cost_sensitivity(trades: pd.DataFrame, costs_bp=(0, 2, 5, 10)) -> pd.DataFrame:
    """Summarize compounded returns after different per-trade costs."""
    rows = []
    for cost_bp in costs_bp:
        df = trades.copy(deep=True).reset_index(drop=True)
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


def yearly_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Summarize compounded trade returns by calendar year."""
    rows = []
    trades = trades.copy(deep=True).reset_index(drop=True)
    trades.loc[:, "year"] = pd.to_datetime(trades["entry_dt"]).dt.year
    for year, df in trades.groupby("year", sort=True):
        equity = (1 + df["ret"]).cumprod()
        drawdown = equity / equity.cummax() - 1
        rows.append(
            {
                "年份": year,
                "交易次数": len(df),
                "胜率": round((df["ret"] > 0).mean(), 4),
                "平均收益BP": round(df["ret_bp"].mean(), 2),
                "累计收益": round(equity.iloc[-1] - 1, 4),
                "最大回撤": round(drawdown.min(), 4),
            }
        )
    return pd.DataFrame(rows)


def daily_returns(trades: pd.DataFrame, all_days: list) -> pd.Series:
    """Aggregate trade returns by date and align to all days, no-trade as zero."""
    if trades.empty:
        return pd.Series(0.0, index=pd.Index(all_days, name="trade_date"))
    s = trades.groupby("trade_date")["ret"].sum()
    return s.reindex(all_days, fill_value=0.0)


def direction_series(trades: pd.DataFrame, all_days: list) -> pd.Series:
    """Map daily trade direction to 1 / -1 / 0."""
    if trades.empty:
        return pd.Series(0, index=pd.Index(all_days, name="trade_date"))
    dmap = {"多头": 1, "空头": -1}
    s = trades.groupby("trade_date")["direction"].first().map(dmap)
    return s.reindex(all_days, fill_value=0).astype(int)


def correlation_report(dt_trades: pd.DataFrame, rb_trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare Dual Thrust daily returns with RBreaker S0.3 daily returns."""
    dt_trades = dt_trades.copy(deep=True).assign(trade_date=pd.to_datetime(dt_trades["trade_date"]).dt.normalize())
    rb_trades = rb_trades.copy(deep=True).assign(trade_date=pd.to_datetime(rb_trades["trade_date"]).dt.normalize())

    start = min(dt_trades["trade_date"].min(), rb_trades["trade_date"].min())
    end = max(dt_trades["trade_date"].max(), rb_trades["trade_date"].max())
    all_days = list(pd.date_range(start=start, end=end, freq="D"))
    aligned = pd.DataFrame(
        {
            "trade_date": all_days,
            "dual_thrust": daily_returns(dt_trades, all_days).values,
            "rbreaker_s03": daily_returns(rb_trades, all_days).values,
            "dual_thrust_dir": direction_series(dt_trades, all_days).values,
            "rbreaker_dir": direction_series(rb_trades, all_days).values,
        }
    )
    aligned.loc[:, "dual_active"] = aligned["dual_thrust_dir"] != 0
    aligned.loc[:, "rbreaker_active"] = aligned["rbreaker_dir"] != 0
    aligned.loc[:, "same_direction"] = (
        aligned["dual_active"] & aligned["rbreaker_active"] & (aligned["dual_thrust_dir"] == aligned["rbreaker_dir"])
    )
    aligned.loc[:, "opposite_direction"] = (
        aligned["dual_active"] & aligned["rbreaker_active"] & (aligned["dual_thrust_dir"] == -aligned["rbreaker_dir"])
    )
    aligned.loc[:, "month"] = pd.to_datetime(aligned["trade_date"]).dt.to_period("M").astype(str)

    masks = {
        "全日历日，空仓收益按0": pd.Series(True, index=aligned.index),
        "双方都有交易的日期": aligned["dual_active"] & aligned["rbreaker_active"],
        "Dual有交易的日期": aligned["dual_active"],
        "RBreaker有交易的日期": aligned["rbreaker_active"],
    }
    rows = []
    for label, mask in masks.items():
        df = aligned[mask]
        rows.append(
            {
                "样本": label,
                "天数": len(df),
                "Dual交易天数": int(df["dual_active"].sum()),
                "R交易天数": int(df["rbreaker_active"].sum()),
                "Pearson": round(df["dual_thrust"].corr(df["rbreaker_s03"], method="pearson"), 4),
                "Spearman": round(df["dual_thrust"].corr(df["rbreaker_s03"], method="spearman"), 4),
                "同涨同跌占比": round(((df["dual_thrust"] * df["rbreaker_s03"]) > 0).mean(), 4),
            }
        )

    monthly = aligned.groupby("month", as_index=False)[["dual_thrust", "rbreaker_s03"]].sum()
    rows.append(
        {
            "样本": "月度收益",
            "天数": len(monthly),
            "Dual交易天数": "",
            "R交易天数": "",
            "Pearson": round(monthly["dual_thrust"].corr(monthly["rbreaker_s03"], method="pearson"), 4),
            "Spearman": round(monthly["dual_thrust"].corr(monthly["rbreaker_s03"], method="spearman"), 4),
            "同涨同跌占比": round(((monthly["dual_thrust"] * monthly["rbreaker_s03"]) > 0).mean(), 4),
        }
    )

    common = aligned[aligned["dual_active"] & aligned["rbreaker_active"]]
    direction = pd.DataFrame(
        [
            {"指标": "Dual交易天数", "值": int(aligned["dual_active"].sum())},
            {"指标": "R交易天数", "值": int(aligned["rbreaker_active"].sum())},
            {"指标": "共同交易天数", "值": len(common)},
            {"指标": "共同交易占Dual交易天数", "值": round(len(common) / aligned["dual_active"].sum(), 4)},
            {"指标": "共同交易占R交易天数", "值": round(len(common) / aligned["rbreaker_active"].sum(), 4)},
            {"指标": "方向一致占比", "值": round(aligned["same_direction"].sum() / len(common), 4) if len(common) else 0},
            {"指标": "方向相反占比", "值": round(aligned["opposite_direction"].sum() / len(common), 4) if len(common) else 0},
        ]
    )
    return aligned, pd.DataFrame(rows), direction


def plot_outputs(aligned: pd.DataFrame, dt_trades: pd.DataFrame, rb_trades: pd.DataFrame, out_dir: Path) -> None:
    """Plot equity comparison and correlation diagnostics."""
    plot_df = aligned.copy()
    plot_df.loc[:, "dt"] = pd.to_datetime(plot_df["trade_date"])
    plot_df.loc[:, "dual_nav"] = (1 + plot_df["dual_thrust"]).cumprod()
    plot_df.loc[:, "rbreaker_nav"] = (1 + plot_df["rbreaker_s03"]).cumprod()
    plot_df.loc[:, "rolling_60_corr"] = plot_df["dual_thrust"].rolling(60).corr(plot_df["rbreaker_s03"])
    plot_df.loc[:, "rolling_120_corr"] = plot_df["dual_thrust"].rolling(120).corr(plot_df["rbreaker_s03"])

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["dt"], plot_df["dual_nav"], label="Dual Thrust N5 K0.2")
    plt.plot(plot_df["dt"], plot_df["rbreaker_nav"], label="RBreaker S0.3")
    plt.title("000852.XSHG 5m Intraday Equity Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "dual_thrust_vs_rbreaker_equity.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(plot_df["dual_thrust"], plot_df["rbreaker_s03"], s=8, alpha=0.5)
    axes[0].axhline(0, color="grey", lw=0.8)
    axes[0].axvline(0, color="grey", lw=0.8)
    axes[0].set_title("Daily Return Scatter")
    axes[0].set_xlabel("Dual Thrust")
    axes[0].set_ylabel("RBreaker S0.3")
    axes[0].grid(alpha=0.3)
    axes[1].plot(plot_df["dt"], plot_df["rolling_60_corr"], label="60D")
    axes[1].plot(plot_df["dt"], plot_df["rolling_120_corr"], label="120D")
    axes[1].set_title("Rolling Correlation")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "dual_thrust_rbreaker_correlation.png", dpi=160)
    plt.close()


def main():
    symbol = "000852.XSHG"
    sdt, edt = "20200101", "20260516"
    out_dir = ROOT / "examples" / "results" / "dual_thrust_000852_jq_sdk"
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = bars_to_frame(read_jq_sdk_bars(freq="daily", czsc_freq=Freq.D, sdt=sdt, edt=edt))
    intraday = bars_to_frame(read_jq_sdk_bars(freq="5m", czsc_freq=Freq.F5, sdt=sdt, edt=edt))

    all_trades = []
    summary_rows = []
    for n in [3, 5, 10, 20]:
        for k in [0.1, 0.2, 0.3, 0.4, 0.5]:
            name = f"DualThrust_N{n}_K{int(k * 100):02d}_{int(k * 100):02d}"
            levels = make_dual_thrust_levels(daily, n=n, k1=k, k2=k)
            trades, available_days = simulate_dual_thrust(intraday, levels, name=name)
            if not trades.empty:
                all_trades.append(trades)
            summary = summarize_trades(trades, available_days)
            summary_rows.append({"策略": name, "N": n, "K1": k, "K2": k, **summary})

    trades_all = pd.concat(all_trades, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(["累计收益", "平均收益BP"], ascending=False)

    benchmark = trades_all[trades_all["strategy"] == "DualThrust_N5_K20_20"].copy()
    benchmark_summary = summary[summary["策略"] == "DualThrust_N5_K20_20"].copy()
    benchmark_cost = cost_sensitivity(benchmark)
    benchmark_yearly = yearly_summary(benchmark)

    rbreaker_file = ROOT / "examples" / "results" / "rbreaker_000852_jq_sdk" / "rbreaker_5m_s03_2020_trades.csv"
    rb_trades = pd.read_csv(rbreaker_file, parse_dates=["trade_date", "signal_dt", "entry_dt", "exit_dt"])
    rb_trades = rb_trades.assign(trade_date=pd.to_datetime(rb_trades["trade_date"]).dt.normalize())
    benchmark = benchmark.assign(trade_date=pd.to_datetime(benchmark["trade_date"]).dt.normalize())
    aligned, corr_summary, direction = correlation_report(benchmark, rb_trades)

    trades_all.to_csv(out_dir / "dual_thrust_5m_2020_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "dual_thrust_5m_2020_summary.csv", index=False, encoding="utf-8-sig")
    benchmark.to_csv(out_dir / "dual_thrust_n5_k20_5m_2020_trades.csv", index=False, encoding="utf-8-sig")
    benchmark_summary.to_csv(out_dir / "dual_thrust_n5_k20_5m_2020_summary.csv", index=False, encoding="utf-8-sig")
    benchmark_cost.to_csv(out_dir / "dual_thrust_n5_k20_5m_2020_cost.csv", index=False, encoding="utf-8-sig")
    benchmark_yearly.to_csv(out_dir / "dual_thrust_n5_k20_5m_2020_yearly.csv", index=False, encoding="utf-8-sig")
    aligned.to_csv(out_dir / "dual_thrust_n5_k20_vs_rbreaker_daily_returns.csv", index=False, encoding="utf-8-sig")
    corr_summary.to_csv(out_dir / "dual_thrust_n5_k20_vs_rbreaker_correlation.csv", index=False, encoding="utf-8-sig")
    direction.to_csv(out_dir / "dual_thrust_n5_k20_vs_rbreaker_direction.csv", index=False, encoding="utf-8-sig")
    plot_outputs(aligned, benchmark, rb_trades, out_dir)

    print(f"symbol: {symbol}")
    print(f"date range requested: {sdt} -> {edt}")
    print(f"daily bars: {len(daily)}, range: {daily['dt'].iloc[0]} -> {daily['dt'].iloc[-1]}")
    print(f"5m bars: {len(intraday)}, range: {intraday['dt'].iloc[0]} -> {intraday['dt'].iloc[-1]}")
    print("\nparameter sweep top 10:")
    print(summary.head(10).to_string(index=False))
    print("\nbenchmark summary:")
    print(benchmark_summary.to_string(index=False))
    print("\nbenchmark cost:")
    print(benchmark_cost.to_string(index=False))
    print("\nbenchmark yearly:")
    print(benchmark_yearly.to_string(index=False))
    print("\ncorrelation with RBreaker S0.3:")
    print(corr_summary.to_string(index=False))
    print("\ndirection overlap:")
    print(direction.to_string(index=False))
    print(f"\noutputs: {out_dir}")


if __name__ == "__main__":
    main()
