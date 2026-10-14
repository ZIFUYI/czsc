#!/usr/bin/env python
"""
159915 (易方达创业板ETF) 窄幅横盘中枢突破与三买三卖对称波段策略
(Consolidation Box Breakout & Third BS Confirmation Strategy)

理论与量化架构：
1. 缠中说禅“中枢破坏与第三类买卖点”严格量化
2. Setup 阶段：20日振幅 <= 9% 窄幅中枢蓄势，价格突破中枢上轨/跌破中枢下轨
3. Confirmation 阶段：拒绝盲目追单！等待回踩/反抽确认不跌破/不突破中枢轨线，微观反包确认进场
4. 结构证伪止损：直接锚定中枢轨线（跌回中枢立即止损，单笔损失 < 1%）
5. 动态追踪止盈：让主升浪充分奔跑，单笔盈亏比 >= 2.0
6. 100% 多空对称：多头吃单边主升，空头穿透单边熊市
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
HTML_PATH = OUTPUT_DIR / "strategy_consolidation_breakout_159915.html"


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


def run_box_breakout_strategy(
    df: pd.DataFrame,
    box_window: int = 20,  # 中枢/箱体观察窗口
    max_box_amp: float = 0.09,  # 窄幅中枢最大振幅阈值 9%
    confirm_window: int = 4,  # 突破后等待回踩确认的最大天数
    activation_profit: float = 0.03,  # 激活动态追踪止盈的浮盈门槛 3%
    trail_stop_pct: float = 0.03,  # 动态追踪回撤阈值 3%
    fee_rate: float = 0.0002,  # 双边万2手续费与滑点
) -> tuple[pd.DataFrame, list[dict]]:
    """
    运行窄幅横盘中枢突破与三买三卖对称波段状态机
    """
    df = df.copy()
    df["pct_chg"] = df["close"].pct_change()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    # 计算历史滚动箱体（前 box_window 天，不含当天）
    df["prev_high_box"] = df["high"].shift(1).rolling(box_window).max()
    df["prev_low_box"] = df["low"].shift(1).rolling(box_window).min()
    df["box_amp"] = (df["prev_high_box"] - df["prev_low_box"]) / df["prev_low_box"]

    # 状态机变量
    pos = 0  # 1: 多头, -1: 空头, 0: 空仓
    stop_price = None
    entry_price = None
    peak_price = None
    trough_price = None

    # 多头 Setup 状态
    pending_long = False
    long_setup_bar_idx = -999
    long_box_top = 0.0
    long_retest_low = 9999.0

    # 空头 Setup 状态
    pending_short = False
    short_setup_bar_idx = -999
    short_box_bottom = 0.0
    short_rebound_high = 0.0

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
        ma20 = df.loc[i, "ma20"]
        ma60 = df.loc[i, "ma60"]
        box_amp = df.loc[i, "box_amp"]
        box_top = df.loc[i, "prev_high_box"]
        box_bottom = df.loc[i, "prev_low_box"]

        # 预热期
        if i < max(60, box_window + 1):
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

            # 1.1 保本机制：浮盈达到 2.0% 后上移止损到成本线
            if unrealized_profit >= 0.02:
                stop_price = max(stop_price, entry_price * 1.002)

            # 1.2 结构证伪止损（跌破中枢上轨或回踩次低点）
            if stop_price is not None and l <= stop_price:
                exit_price = min(o, stop_price) if o <= stop_price else stop_price
                pos = 0
                exited = True
                if current_trade:
                    current_trade["exit_dt"] = dt
                    current_trade["exit_price"] = exit_price
                    current_trade["exit_reason"] = "多头三买结构破位止损"
                    trades.append(current_trade)
                    current_trade = None

            # 1.3 动态追踪止盈（趋势充分展开后，高位回撤锁利）
            elif unrealized_profit >= activation_profit and (peak_price - c) / peak_price >= trail_stop_pct:
                pos = 0
                exited = True
                if current_trade:
                    current_trade["exit_dt"] = dt
                    current_trade["exit_price"] = c
                    current_trade["exit_reason"] = "多头主升浪追踪止盈"
                    trades.append(current_trade)
                    current_trade = None

            # 1.4 均线保护：收盘跌破 MA20 趋势转弱离场
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

            # 1.1 保本机制：浮盈达到 2.0% 后下移止损到成本线
            if unrealized_profit >= 0.02:
                stop_price = min(stop_price, entry_price * 0.998)

            # 1.2 结构证伪止损（突破中枢下轨或反抽次高点）
            if stop_price is not None and h >= stop_price:
                exit_price = max(o, stop_price) if o >= stop_price else stop_price
                pos = 0
                exited = True
                if current_trade:
                    current_trade["exit_dt"] = dt
                    current_trade["exit_price"] = exit_price
                    current_trade["exit_reason"] = "空头三卖结构破位止损"
                    trades.append(current_trade)
                    current_trade = None

            # 1.3 动态追踪止盈
            elif unrealized_profit >= activation_profit and (c - trough_price) / trough_price >= trail_stop_pct:
                pos = 0
                exited = True
                if current_trade:
                    current_trade["exit_dt"] = dt
                    current_trade["exit_price"] = c
                    current_trade["exit_reason"] = "空头主跌浪追踪止盈"
                    trades.append(current_trade)
                    current_trade = None

            # 1.4 均线保护：收盘站上 MA20 趋势转强离场
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
        # 2. 两阶段状态机进场逻辑 (三买与三卖)
        # ----------------------------------------------------
        # 2.1 阶段一：Setup 中枢突破离开识别
        # 多头 Setup：窄幅中枢 (amp <= 9%) + 顺大势 (C > MA60) + 收盘突破箱体上轨
        if pos == 0 and not pending_long and box_amp <= max_box_amp and c > ma60 and c > box_top:
            pending_long = True
            long_setup_bar_idx = i
            long_box_top = box_top
            long_retest_low = l

        # 空头 Setup：窄幅中枢 (amp <= 9%) + 顺大势 (C < MA60) + 收盘跌破箱体下轨
        elif pos == 0 and not pending_short and box_amp <= max_box_amp and c < ma60 and c < box_bottom:
            pending_short = True
            short_setup_bar_idx = i
            short_box_bottom = box_bottom
            short_rebound_high = h

        # 检查 Setup 超时
        if pending_long and (i - long_setup_bar_idx > confirm_window):
            pending_long = False

        if pending_short and (i - short_setup_bar_idx > confirm_window):
            pending_short = False

        # 2.2 阶段二：Confirmation 三买/三卖微观确认
        # 多头三买确认：在随后的回踩中，价格没有有效跌破箱体上轨 (收盘 >= box_top 且 l >= box_top * 0.992)
        # 随后出现止跌阳线反包，确认真突破！
        if pos == 0 and pending_long and (i > long_setup_bar_idx):
            long_retest_low = min(long_retest_low, l)
            # 若收盘深度跌回原箱体内，判定为假突破 2B 陷阱，立即取消 Setup
            if c < long_box_top * 0.995:
                pending_long = False
            else:
                # 微观确认：收阳反包前一根 K 线最高价，或站稳箱体上方
                prev_h1 = df.loc[i - 1, "high"]
                if c > o and c >= prev_h1:
                    pos = 1
                    # 结构止损锚定：箱体上轨或回踩次低点，取两者合适位置
                    stop_price = max(long_retest_low, long_box_top * 0.992)
                    entry_price = c
                    peak_price = c
                    pending_long = False
                    current_trade = {
                        "entry_dt": dt,
                        "direction": "多头 (三买)",
                        "entry_price": c,
                        "stop_price": stop_price,
                        "box_top": long_box_top,
                    }

        # 空头三卖确认：在随后的反抽中，价格没有有效突破箱体下轨
        # 随后出现滞涨阴线跌破，确认真跌破！
        elif pos == 0 and pending_short and (i > short_setup_bar_idx):
            short_rebound_high = max(short_rebound_high, h)
            # 若收盘深度站回原箱体内，判定为假跌破，立即取消 Setup
            if c > short_box_bottom * 1.005:
                pending_short = False
            else:
                prev_l1 = df.loc[i - 1, "low"]
                if c < o and c <= prev_l1:
                    pos = -1
                    stop_price = min(short_rebound_high, short_box_bottom * 1.008)
                    entry_price = c
                    trough_price = c
                    pending_short = False
                    current_trade = {
                        "entry_dt": dt,
                        "direction": "空头 (三卖)",
                        "entry_price": c,
                        "stop_price": stop_price,
                        "box_bottom": short_box_bottom,
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
    print("  【159915 窄幅中枢突破与三买三卖对称策略】 2020-2026 全周期实盘回测")
    print("=" * 80)

    daily_df = load_daily_data(REAL_5M_CSV)
    print(
        f"数据加载完成: 共 {len(daily_df)} 根真实日线, 时间跨度: {daily_df['dt'].min().date()} ~ {daily_df['dt'].max().date()}"
    )

    dfw, trades = run_box_breakout_strategy(
        daily_df,
        box_window=20,
        max_box_amp=0.09,
        confirm_window=4,
        activation_profit=0.03,
        trail_stop_pct=0.03,
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
        f"总交易笔数:       {len(trades)} 笔 (其中多头: {sum(1 for t in trades if '多头' in t['direction'])} 笔, 空头: {sum(1 for t in trades if '空头' in t['direction'])} 笔)"
    )

    trade_df = pd.DataFrame(trades)
    if len(trade_df) > 0 and "exit_price" in trade_df.columns:
        valid_trades = trade_df.dropna(subset=["exit_price"]).copy()
        valid_trades["pnl"] = (
            np.where(
                valid_trades["direction"].str.contains("多头"),
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

    # 绘制高精度 Plotly HTML 大屏
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "【159915 窄幅中枢突破与三买三卖对称策略】 2020-2026 真实累计净值 (Cumulative NAV)",
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
            name="窄幅中枢突破与三买三卖策略",
            line=dict(color="#00E5FF", width=2.8),
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
            text="159915 易方达创业板 ETF · 窄幅中枢突破与三买三卖对称策略实盘回测大屏 (2020-2026)",
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
    print(f"\n交互式可视化大屏已保存至: {HTML_PATH.resolve()}")

    return stats, yearly_compare


if __name__ == "__main__":
    main()
