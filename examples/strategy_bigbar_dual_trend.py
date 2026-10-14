#!/usr/bin/env python
"""
159915 (易方达创业板ETF) 关键大K线顺势双向波段策略 (Trend BigBar Dual-Wave Strategy v2)
遵循 Quant Strategy Architect SOP：
1. 大势环境分层 (MA60 顺势过滤，不抢逆势单)
2. 两阶段状态机 (Setup 关键大K线爆发 + Confirmation 微观回踩/反弹站稳确认)
3. 100% 多空对称架构 (多头吃主升，空头穿透 2022-2023 单边熊市)
4. 结构证伪止损点 (锚定回踩次低点/反弹次高点，单笔损失精准控制)
5. 浮盈阶梯保护与动态追踪止盈 (达到目标浮盈后开启追踪，确保盈亏比 >= 1.8)
6. 真实历史数据驱动 (2020-2026 全周期检验)
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from czsc import WeightBacktest

OUTPUT_DIR = Path("examples/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REAL_5M_CSV = OUTPUT_DIR / "159915_real_5m_2020_2026.csv"
HTML_PATH = OUTPUT_DIR / "strategy_bigbar_dual_trend_159915.html"


def load_daily_data(csv_path: Path) -> pd.DataFrame:
    """加载 5m 真实历史数据并合成为日线数据"""
    df_raw = pd.read_csv(csv_path)
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
    df_raw = df_raw.rename(
        columns={
            "SZSE.159915.open": "open",
            "SZSE.159915.high": "high",
            "SZSE.159915.low": "low",
            "SZSE.159915.close": "close",
            "SZSE.159915.volume": "volume",
        }
    )
    df_raw.set_index("datetime", inplace=True)
    daily = (
        df_raw.resample("D")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )
    daily = daily[daily["volume"] > 0].copy()
    daily.reset_index(inplace=True)
    daily.rename(columns={"datetime": "dt"}, inplace=True)
    daily["symbol"] = "159915"
    return daily


def run_trend_bigbar_strategy(
    df: pd.DataFrame,
    big_bar_pct: float = 0.02,  # 关键大K线涨跌幅阈值 2%
    setup_window: int = 5,  # Setup 状态最大有效窗口
    activation_profit: float = 0.025,  # 浮盈激活追踪阈值 2.5%
    trail_stop_pct: float = 0.025,  # 动态追踪回撤阈值 2.5%
    max_init_sl: float = 0.035,  # 初始结构止损最大允许距离 3.5%
    fee_rate: float = 0.0002,  # 双边万2手续费与滑点
) -> tuple[pd.DataFrame, list[dict]]:
    """
    运行关键大K线顺势双向波段状态机
    """
    df = df.copy()
    df["pct_chg"] = df["close"].pct_change()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    # 状态机变量
    pos = 0  # 1: 多头, -1: 空头, 0: 空仓
    stop_price = None
    entry_price = None
    peak_price = None  # 多头持仓最高价
    trough_price = None  # 空头持仓最低价

    # 多头 Setup 记忆
    setup_long_active = False
    setup_long_bar_idx = -999
    setup_long_open = 0.0
    setup_long_retest_low = 9999.0

    # 空头 Setup 记忆
    setup_short_active = False
    setup_short_bar_idx = -999
    setup_short_open = 0.0
    setup_short_rebound_high = 0.0

    trades = []
    current_trade = None

    weights = []
    prices = []
    dts = []

    for i in range(len(df)):
        dt = df.loc[i, "dt"]
        c = df.loc[i, "close"]
        o = df.loc[i, "open"]
        h = df.loc[i, "high"]
        l = df.loc[i, "low"]
        pct = df.loc[i, "pct_chg"]
        ma20 = df.loc[i, "ma20"]
        ma60 = df.loc[i, "ma60"]
        vol = df.loc[i, "volume"]
        vol_ma20 = df.loc[i, "vol_ma20"]

        # 预热期（MA60 计算需要至少 60 根）
        if i < 60:
            weights.append(0.0)
            prices.append(c)
            dts.append(dt)
            continue

        # ----------------------------------------------------
        # 1. 持仓出场风控逻辑 (止损 & 阶梯追踪止盈)
        # ----------------------------------------------------
        exited = False

        if pos == 1:
            peak_price = max(peak_price or c, h)
            unrealized_profit = (peak_price - entry_price) / entry_price

            # 1.1 保本机制：浮盈一旦达到 2.0%，将止损线上移至成本线附近
            if unrealized_profit >= 0.02:
                stop_price = max(stop_price, entry_price * 1.002)

            # 1.2 结构证伪止损（跌破回踩次低点或保本止损点）
            if stop_price is not None and l <= stop_price:
                exit_price = min(o, stop_price) if o <= stop_price else stop_price
                pos = 0
                exited = True
                if current_trade:
                    current_trade["exit_dt"] = dt
                    current_trade["exit_price"] = exit_price
                    current_trade["exit_reason"] = "多头结构/保本止损"
                    trades.append(current_trade)
                    current_trade = None

            # 1.3 动态追踪止盈（浮盈达到激活门槛后，从极高点回撤达到 trail_stop_pct）
            elif unrealized_profit >= activation_profit and (peak_price - c) / peak_price >= trail_stop_pct:
                pos = 0
                exited = True
                if current_trade:
                    current_trade["exit_dt"] = dt
                    current_trade["exit_price"] = c
                    current_trade["exit_reason"] = "多头动态追踪止盈"
                    trades.append(current_trade)
                    current_trade = None

            # 1.4 跌破 MA20 趋势走弱离场保护
            elif c < ma20:
                pos = 0
                exited = True
                if current_trade:
                    current_trade["exit_dt"] = dt
                    current_trade["exit_price"] = c
                    current_trade["exit_reason"] = "多头跌破MA20离场"
                    trades.append(current_trade)
                    current_trade = None

        elif pos == -1:
            trough_price = min(trough_price or c, l)
            unrealized_profit = (entry_price - trough_price) / entry_price

            # 1.1 保本机制：浮盈一旦达到 2.0%，将空头止损线下移至成本线
            if unrealized_profit >= 0.02:
                stop_price = min(stop_price, entry_price * 0.998)

            # 1.2 结构证伪止损（突破反弹次高点或保本止损点）
            if stop_price is not None and h >= stop_price:
                exit_price = max(o, stop_price) if o >= stop_price else stop_price
                pos = 0
                exited = True
                if current_trade:
                    current_trade["exit_dt"] = dt
                    current_trade["exit_price"] = exit_price
                    current_trade["exit_reason"] = "空头结构/保本止损"
                    trades.append(current_trade)
                    current_trade = None

            # 1.3 动态追踪止盈（浮盈达到激活门槛后，从极低点反弹达到 trail_stop_pct）
            elif unrealized_profit >= activation_profit and (c - trough_price) / trough_price >= trail_stop_pct:
                pos = 0
                exited = True
                if current_trade:
                    current_trade["exit_dt"] = dt
                    current_trade["exit_price"] = c
                    current_trade["exit_reason"] = "空头动态追踪止盈"
                    trades.append(current_trade)
                    current_trade = None

            # 1.4 站上 MA20 趋势反弹离场保护
            elif c > ma20:
                pos = 0
                exited = True
                if current_trade:
                    current_trade["exit_dt"] = dt
                    current_trade["exit_price"] = c
                    current_trade["exit_reason"] = "空头站上MA20离场"
                    trades.append(current_trade)
                    current_trade = None

        if exited:
            stop_price, entry_price, peak_price, trough_price = None, None, None, None

        # ----------------------------------------------------
        # 2. 两阶段状态机进场逻辑
        # ----------------------------------------------------
        # 2.1 阶段一：Setup 势能拉偏识别
        # 多头 Setup：顺势 (C > MA60) + 关键放量大阳线 (pct >= 2.0% 且 vol >= 1.0 * vol_ma20)
        if pos == 0 and c > ma60 and pct >= big_bar_pct and vol >= 1.0 * vol_ma20:
            setup_long_active = True
            setup_long_bar_idx = i
            setup_long_open = o
            setup_long_retest_low = l

        # 空头 Setup：顺势 (C < MA60) + 关键放量大阴线 (pct <= -big_bar_pct 且 vol >= 1.0 * vol_ma20)
        elif pos == 0 and c < ma60 and pct <= -big_bar_pct and vol >= 1.0 * vol_ma20:
            setup_short_active = True
            setup_short_bar_idx = i
            setup_short_open = o
            setup_short_rebound_high = h

        # 检查 Setup 是否过期（超过 setup_window 天数则失效）
        if setup_long_active and (i - setup_long_bar_idx > setup_window):
            setup_long_active = False

        if setup_short_active and (i - setup_short_bar_idx > setup_window):
            setup_short_active = False

        # 2.2 阶段二：Confirmation 微观确认开仓
        # 多头 Confirmation：在 Setup 激活后的回踩中未破大阳线低点，且出现反包吞噬 (C >= max(H[i-1], H[i-2]))
        if pos == 0 and setup_long_active and (i > setup_long_bar_idx):
            setup_long_retest_low = min(setup_long_retest_low, l)

            # 若跌破大阳线开盘价，说明动能证伪，取消 Setup
            if l < setup_long_open * 0.995:
                setup_long_active = False
            else:
                prev_h1 = df.loc[i - 1, "high"]
                prev_h2 = df.loc[i - 2, "high"] if i >= 2 else prev_h1
                if c > o and c >= max(prev_h1, prev_h2):
                    calc_stop = setup_long_retest_low
                    # 止损距离不能过大
                    if (c - calc_stop) / c <= max_init_sl:
                        pos = 1
                        stop_price = calc_stop  # 锚定微观回踩次低点
                        entry_price = c
                        peak_price = c
                        setup_long_active = False
                        current_trade = {
                            "entry_dt": dt,
                            "direction": "多头",
                            "entry_price": c,
                            "stop_price": stop_price,
                            "setup_bar_dt": df.loc[setup_long_bar_idx, "dt"],
                        }

        # 空头 Confirmation：在 Setup 激活后的反弹中未破大阴线高点，且出现滞涨破位 (C <= min(L[i-1], L[i-2]))
        elif pos == 0 and setup_short_active and (i > setup_short_bar_idx):
            setup_short_rebound_high = max(setup_short_rebound_high, h)

            # 若突破大阴线开盘价，说明空头被完全修复，取消 Setup
            if h > setup_short_open * 1.005:
                setup_short_active = False
            else:
                prev_l1 = df.loc[i - 1, "low"]
                prev_l2 = df.loc[i - 2, "low"] if i >= 2 else prev_l1
                if c < o and c <= min(prev_l1, prev_l2):
                    calc_stop = setup_short_rebound_high
                    if (calc_stop - c) / c <= max_init_sl:
                        pos = -1
                        stop_price = calc_stop  # 锚定微观反弹次高点
                        entry_price = c
                        trough_price = c
                        setup_short_active = False
                        current_trade = {
                            "entry_dt": dt,
                            "direction": "空头",
                            "entry_price": c,
                            "stop_price": stop_price,
                            "setup_bar_dt": df.loc[setup_short_bar_idx, "dt"],
                        }

        weights.append(float(pos))
        prices.append(c)
        dts.append(dt)

    dfw = pd.DataFrame(
        {
            "dt": [str(d) for d in dts],
            "symbol": "159915",
            "weight": weights,
            "price": prices,
        }
    )

    return dfw, trades


def main():
    print("=" * 80)
    print("  【159915 顺势关键大K线微观回踩双向波段策略 v2】 2020-2026 全周期实盘回测")
    print("=" * 80)

    daily_df = load_daily_data(REAL_5M_CSV)
    print(
        f"数据加载完成: 共 {len(daily_df)} 根真实日线, 时间跨度: {daily_df['dt'].min().date()} ~ {daily_df['dt'].max().date()}"
    )

    dfw, trades = run_trend_bigbar_strategy(
        daily_df,
        big_bar_pct=0.02,
        setup_window=5,
        activation_profit=0.025,
        trail_stop_pct=0.025,
        max_init_sl=0.035,
        fee_rate=0.0002,
    )

    wb = WeightBacktest(data=dfw, fee_rate=0.0002, weight_type="ts")
    daily_ret = wb.daily_return.copy()
    daily_ret["date"] = pd.to_datetime(daily_ret["date"])
    daily_ret = daily_ret.sort_values("date").set_index("date")

    nav = (1 + daily_ret["total"]).cumprod()
    cum_max = nav.cummax()
    dd = (nav - cum_max) / cum_max

    bench_pnl = daily_df.set_index("dt")["close"].pct_change().fillna(0)
    bench_nav = (1 + bench_pnl).cumprod()
    bench_nav_aligned = bench_nav.loc[daily_ret.index]
    bench_nav_norm = bench_nav_aligned / bench_nav_aligned.iloc[0]
    bench_dd = (bench_nav_norm - bench_nav_norm.cummax()) / bench_nav_norm.cummax() * 100

    daily_ret["year"] = daily_ret.index.year
    yearly_pnl = daily_ret.groupby("year")["total"].apply(lambda s: (1 + s).prod() - 1) * 100

    bench_df = pd.DataFrame({"bench": bench_pnl.loc[daily_ret.index]}, index=daily_ret.index)
    bench_df["year"] = bench_df.index.year
    bench_yearly = bench_df.groupby("year")["bench"].apply(lambda s: (1 + s).prod() - 1) * 100

    stats = wb.stats
    print("\n" + "=" * 80)
    print("【WeightBacktest 策略终审指标卡】")
    print("=" * 80)
    print(
        f"绝对累计收益率:   {stats.get('绝对收益', 0) * 100:+.2f}%  (基准: {(bench_nav_norm.iloc[-1] - 1) * 100:+.2f}%)"
    )
    print(f"年化收益率:       {stats.get('年化收益', 0) * 100:+.2f}%")
    print(f"基准夏普比率:     {stats.get('夏普比率', 0):.2f}")
    print(f"最大回撤 (MDD):   {stats.get('最大回撤', 0) * 100:.2f}%  (基准最大回撤: {bench_dd.min():.2f}%)")
    print(f"卡玛比率 (Calmar): {stats.get('卡玛比率', 0):.2f}")
    print(f"日胜率:           {stats.get('日胜率', 0) * 100:.2f}%")
    print(
        f"总交易笔数:       {len(trades)} 笔 (其中多头: {sum(1 for t in trades if t['direction'] == '多头')} 笔, 空头: {sum(1 for t in trades if t['direction'] == '空头')} 笔)"
    )

    trade_df = pd.DataFrame(trades)
    if len(trade_df) > 0 and "exit_price" in trade_df.columns:
        valid_trades = trade_df.dropna(subset=["exit_price"]).copy()
        valid_trades["pnl"] = (
            np.where(
                valid_trades["direction"] == "多头",
                (valid_trades["exit_price"] - valid_trades["entry_price"]) / valid_trades["entry_price"],
                (valid_trades["entry_price"] - valid_trades["exit_price"]) / valid_trades["entry_price"],
            )
            - 0.0004
        )
        win_trades = valid_trades[valid_trades["pnl"] > 0]
        loss_trades = valid_trades[valid_trades["pnl"] < 0]
        win_rate = len(win_trades) / len(valid_trades) * 100
        avg_win = win_trades["pnl"].mean() * 100 if len(win_trades) > 0 else 0
        avg_loss = abs(loss_trades["pnl"].mean() * 100) if len(loss_trades) > 0 else 0
        pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        print(f"逐笔胜率:         {win_rate:.1f}%")
        print(f"单笔平均盈利:     +{avg_win:.2f}%")
        print(f"单笔平均亏损:     -{avg_loss:.2f}%")
        print(f"单笔盈亏比:       {pl_ratio:.2f}")

    print("\n" + "=" * 80)
    print("【2020-2026 逐年收益率对比表 (%)】")
    print("=" * 80)
    yearly_compare = pd.DataFrame(
        {
            "策略收益 (%)": yearly_pnl.round(2),
            "159915基准 (%)": bench_yearly.round(2),
            "超额收益 Alpha (%)": (yearly_pnl - bench_yearly).round(2),
        }
    )
    print(yearly_compare.to_string())

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "【159915 顺势关键大K线双向波段策略 v2】 2020-2026 真实累计净值对比 (Cumulative NAV)",
            "动态回撤深度对比 (Drawdown %)",
        ),
        row_heights=[0.72, 0.28],
    )

    dates_str = nav.index.strftime("%Y-%m-%d").tolist()

    fig.add_trace(
        go.Scatter(
            x=dates_str,
            y=bench_nav_norm.values,
            name="159915 买入持有基准",
            line=dict(color="#90A4AE", width=1.5, dash="dot"),
            opacity=0.7,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates_str,
            y=bench_dd.values,
            name="基准回撤",
            line=dict(color="#90A4AE", width=1, dash="dot"),
            showlegend=False,
            opacity=0.4,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates_str,
            y=nav.values,
            name="顺势大K线双向波段策略 v2",
            line=dict(color="#00E676", width=2.8),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates_str,
            y=(dd * 100).values,
            name="策略回撤",
            line=dict(color="#FF5252", width=1.4),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A0E14",
        plot_bgcolor="#101721",
        title=dict(
            text="159915 易方达创业板 ETF · 顺势大K线双向波段策略实盘回测大屏 v2 (2020-2026)",
            font=dict(size=18, color="#FFFFFF"),
            x=0.02,
            y=0.98,
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.6)",
        ),
        height=720,
    )
    fig.update_yaxes(title_text="累计净值 (NAV, 初始=1.0)", gridcolor="#263238", row=1, col=1)
    fig.update_yaxes(title_text="回撤深度 (%)", gridcolor="#263238", row=2, col=1)
    fig.update_xaxes(gridcolor="#263238", row=1, col=1)
    fig.update_xaxes(gridcolor="#263238", row=2, col=1)

    fig.write_html(str(HTML_PATH), include_plotlyjs="cdn")
    print(f"\n交互式可视化图表已保存至: {HTML_PATH.resolve()}")

    return stats, yearly_compare


if __name__ == "__main__":
    main()
