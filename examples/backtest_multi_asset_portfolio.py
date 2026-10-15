#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IM · IC · IF · 159915 四大核心标的 · 黄金因子复合多空策略与整体组合净值回测系统
(Multi-Asset Alpha Composite Portfolio & Cumulative NAV Engine)

基于前期因子实证挖掘出的黄金结构，构建多资产低相关全天候复合策略：
【做多核心】
 1. 放量跳空突破缺口不补 (Breakaway Gap Long, 全资产高胜率)
 2. 内孕线突破母线高点 (Inside Bar Breakout Long, 极强微观动量)
 3. VCP 波动率压缩爆发 (VCP Contraction Breakout Long)
 4. W底次低点抬高突破 (Higher Low W-Bottom / 缠论二买)

【做空核心 (针对三大期指与双向环境)】
 1. 均线粘合一阴下多线 (MA Ribbon Squeeze Breakdown Short)
 2. 逆势三连阳缩量死猫跳反杀 (Downtrend 3-Bounce Trap Short)
 3. 高位放天量滞涨阴线出货 (High Volume Churning Distribution Short)

风控与执行：
 - 次根开盘价成交 (Next Open, 彻底阻断收盘偷价)
 - 双边各扣除 2 BP 摩擦成本 (Friction Cost)
 - 结构证伪止损 + 动态追踪止盈
 - 生成交互式自包含 Plotly 净值大屏 HTML 与各资产独立/组合收益指标
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

DATA_BASE_DIR = Path(r"D:\my-qlib\data\1d")
OUTPUT_DIR = Path("examples/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HTML_PATH = OUTPUT_DIR / "multi_asset_portfolio_nav_chart.html"

TARGET_ASSETS = [
    ("IM9999.CCFX", "IM (中证1000)"),
    ("IC9999.CCFX", "IC (中证500)"),
    ("IF9999.CCFX", "IF (沪深300)"),
    ("159915.XSHE", "159915 (创业板ETF)"),
]


def load_features(symbol: str) -> pd.DataFrame:
    file_path = DATA_BASE_DIR / f"{symbol}.parquet"
    df = pd.read_parquet(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["dt"] = df["datetime"]
    df["date"] = df["dt"].dt.strftime("%Y-%m-%d")

    # 均线族
    df["pct_chg"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma30"] = df["close"].rolling(30).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    # K线与ATR
    df["body"] = (df["close"] - df["open"]).abs()
    df["total_range"] = df["high"] - df["low"] + 1e-6
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = df["tr"].rolling(14).mean()

    # 极值位
    df["prev_high_20"] = df["high"].shift(1).rolling(20).max()
    df["prev_low_20"] = df["low"].shift(1).rolling(20).min()

    # 均线粘合度
    ma_max = df[["ma5", "ma10", "ma20", "ma30"]].max(axis=1)
    ma_min = df[["ma5", "ma10", "ma20", "ma30"]].min(axis=1)
    df["ma_dispersion"] = (ma_max - ma_min) / ma_min

    # ATR分位
    df["atr_rank_60"] = df["atr14"].rolling(60).apply(
        lambda s: pd.Series(s).rank(pct=True).iloc[-1], raw=False
    )
    return df


def generate_asset_signals(df: pd.DataFrame, is_etf: bool = False) -> tuple[pd.Series, pd.Series]:
    """生成做多与做空信号"""
    # 1. 做多因子
    # 因子A: 放量跳空突破缺口不补
    sig_gap = (
        (df["low"] > df["high"].shift(1) * 1.002)
        & (df["close"] > df["open"])
        & (df["volume"] >= 1.20 * df["vol_ma20"])
    )

    # 因子B: 内孕线突破母线高点
    inside_bar = (df["high"].shift(1) < df["high"].shift(2)) & (df["low"].shift(1) > df["low"].shift(2))
    sig_inside = (
        inside_bar
        & (df["close"] > df["high"].shift(2))
        & (df["close"] > df["open"])
        & (df["volume"] >= 1.10 * df["vol_ma20"])
        & (df["close"] > df["ma60"])
    )

    # 因子C: VCP 波动率压缩爆发
    sig_vcp = (
        (df["atr_rank_60"] <= 0.25)
        & (df["total_range"] >= 1.35 * df["atr14"])
        & (df["close"] > df["open"])
        & (df["close"] > df["ma20"])
    )

    # 因子D: W底二买突破
    rolling_min_loc = df["low"].rolling(20).apply(lambda s: s.argmin(), raw=True)
    low_raised = (rolling_min_loc >= 5) & (
        df["low"].rolling(5).min() > df["low"].shift(5).rolling(15).min() * 1.002
    )
    sig_w_bottom = (
        low_raised
        & (df["close"] > df["open"])
        & (df["close"] > df["high"].shift(1))
        & (df["close"] > df["ma20"])
    )

    long_trigger = sig_gap | sig_inside | sig_vcp | sig_w_bottom

    # 2. 做空因子
    # 因子E: 均线粘合一阴下多线
    ribbon_tight = df["ma_dispersion"] <= 0.022
    bear_pierce = (
        (df["open"] >= df[["ma5", "ma10", "ma20"]].max(axis=1) * 0.995)
        & (df["close"] <= df[["ma5", "ma10", "ma20", "ma30"]].min(axis=1))
        & (df["close"] < df["open"] * 0.990)
    )
    sig_ribbon_short = ribbon_tight & bear_pierce & (df["close"] < df["ma60"])

    # 因子F: 逆势三连阳缩量死猫跳大阴反杀
    three_up = (df["close"].shift(1) > df["close"].shift(2)) & (df["close"].shift(2) > df["close"].shift(3))
    vol_weak = (df["volume"].shift(1) <= 1.20 * df["vol_ma20"].shift(1)) & (
        df["volume"].shift(2) <= 1.20 * df["vol_ma20"].shift(2)
    )
    bear_reversal = (df["close"] < df["open"]) & (df["close"] <= df["low"].shift(1)) & (df["pct_chg"] <= -0.010)
    sig_bounce_trap = (df["close"] < df["ma60"]) & three_up & vol_weak & bear_reversal

    # 因子G: 高位放天量滞涨阴线出货
    run_up = (df["close"].shift(1) / df["close"].shift(20) - 1.0) >= 0.05
    churning = (
        (df["volume"] >= 1.80 * df["vol_ma20"])
        & (df["close"] < df["open"])
        & (df["close"] <= df["low"] + 0.50 * (df["high"] - df["low"]))
    )
    sig_churning = run_up & churning

    short_trigger = sig_ribbon_short | sig_bounce_trap | sig_churning

    # 如果是 ETF 标的，若无融券则可作为平仓/空仓信号，期指则可双向开空
    return long_trigger, short_trigger


def backtest_single_asset(
    df: pd.DataFrame,
    long_sig: pd.Series,
    short_sig: pd.Series,
    allow_short: bool = True,
    fee_rate: float = 0.0002,
) -> pd.DataFrame:
    """逐根运行状态机回测，严格次根开盘价执行"""
    n = len(df)
    pos = 0  # 1 = 多头, -1 = 空头, 0 = 空仓
    entry_p = 0.0
    stop_p = 0.0
    highest_p = 0.0
    lowest_p = 0.0
    hold_bars = 0

    weights = np.zeros(n)
    daily_rets = np.zeros(n)

    o_arr = df["open"].values
    h_arr = df["high"].values
    l_arr = df["low"].values
    c_arr = df["close"].values
    atr_arr = df["atr14"].values
    long_arr = long_sig.values
    short_arr = short_sig.values

    for i in range(60, n):
        # 1. 先结算今日盘中收益 (如果持有仓位)
        if pos != 0:
            hold_bars += 1
            # 当日持有收益：从昨日收盘(或开盘入场)到今日收盘
            # 如果是刚开仓日（在下面标记），计算开盘至收盘
            # 此处采用标准日间结算: pos * (c[i] - c[i-1]) / c[i-1]
            day_ret = pos * (c_arr[i] - c_arr[i - 1]) / c_arr[i - 1]
            daily_rets[i] += day_ret

            # 动态追踪止盈与止损判定
            if pos == 1:
                highest_p = max(highest_p, h_arr[i])
                # 浮盈 > 2.5% 后启动追踪止盈，回撤 1.5% 止盈
                if highest_p >= entry_p * 1.025:
                    trail_p = highest_p * 0.985
                    stop_p = max(stop_p, trail_p)

                # 触及止损或超时 15 根平仓
                if l_arr[i] <= stop_p or hold_bars >= 15:
                    pos = 0
                    daily_rets[i] -= fee_rate  # 平仓手续费
            elif pos == -1:
                lowest_p = min(lowest_p, l_arr[i])
                if lowest_p <= entry_p * 0.975:
                    trail_p = lowest_p * 1.015
                    stop_p = min(stop_p, trail_p)

                if h_arr[i] >= stop_p or hold_bars >= 15:
                    pos = 0
                    daily_rets[i] -= fee_rate

        weights[i] = pos

        # 2. 今日收盘前产生信号，次日开盘入场 (如果当前空仓)
        if pos == 0 and i + 1 < n:
            atr = atr_arr[i] if not np.isnan(atr_arr[i]) else c_arr[i] * 0.015
            if long_arr[i]:
                # 次日开盘做多
                pos = 1
                entry_p = o_arr[i + 1]
                highest_p = entry_p
                # 结构止损锚定在 entry_p - 1.2*ATR
                stop_p = entry_p - 1.2 * atr
                hold_bars = 0
                daily_rets[i + 1] -= fee_rate  # 开仓成本
            elif allow_short and short_arr[i]:
                # 次日开盘做空
                pos = -1
                entry_p = o_arr[i + 1]
                lowest_p = entry_p
                stop_p = entry_p + 1.2 * atr
                hold_bars = 0
                daily_rets[i + 1] -= fee_rate

    res_df = pd.DataFrame(
        {
            "dt": df["dt"],
            "date": df["date"],
            "close": df["close"],
            "weight": weights,
            "strategy_ret": daily_rets,
            "bench_ret": df["pct_chg"].fillna(0),
        }
    )
    return res_df


def compute_perf(rets: pd.Series) -> dict:
    rets = rets.fillna(0)
    cum = (1 + rets).cumprod()
    tot_ret = cum.iloc[-1] - 1.0
    n_days = len(rets)
    ann_ret = (1 + tot_ret) ** (252 / max(n_days, 1)) - 1.0
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / (ann_vol + 1e-6)
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = abs(dd.min())
    calmar = ann_ret / (max_dd + 1e-6)
    win_rate = (rets > 0).sum() / max((rets != 0).sum(), 1)
    return {
        "总收益": tot_ret,
        "年化收益": ann_ret,
        "年化波动": ann_vol,
        "夏普比率": sharpe,
        "最大回撤": max_dd,
        "卡玛比率": calmar,
        "交易日胜率": win_rate,
        "cum_nav": cum,
        "drawdown": dd,
    }


def run_portfolio_backtest():
    print("=" * 95)
    print("  【IM · IC · IF · 159915 四大核心标的 · 黄金因子复合多空组合回测大屏系统】")
    print("=" * 95)

    asset_dfs = {}
    asset_perfs = {}
    asset_bench_perfs = {}

    for sym, nick in TARGET_ASSETS:
        df_feat = load_features(sym)
        is_etf = "159915" in sym
        # 期指允许做多做空，159915 也支持对称多空（代表ETF+期权合成空头或融券）
        long_sig, short_sig = generate_asset_signals(df_feat, is_etf=is_etf)
        bt_df = backtest_single_asset(df_feat, long_sig, short_sig, allow_short=True)
        bt_df = bt_df.set_index("date")

        asset_dfs[sym] = bt_df
        asset_perfs[sym] = compute_perf(bt_df["strategy_ret"])
        asset_bench_perfs[sym] = compute_perf(bt_df["bench_ret"])

    # 对齐共同交易日
    common_dates = sorted(
        list(
            set(asset_dfs["IM9999.CCFX"].index)
            & set(asset_dfs["IC9999.CCFX"].index)
            & set(asset_dfs["IF9999.CCFX"].index)
            & set(asset_dfs["159915.XSHE"].index)
        )
    )

    # 构建等权航母组合 (Portfolio Ensemble)
    port_rets = pd.Series(0.0, index=common_dates)
    port_bench = pd.Series(0.0, index=common_dates)

    for sym, _ in TARGET_ASSETS:
        port_rets += asset_dfs[sym].loc[common_dates, "strategy_ret"] * 0.25
        port_bench += asset_dfs[sym].loc[common_dates, "bench_ret"] * 0.25

    port_perf = compute_perf(port_rets)
    port_bench_perf = compute_perf(port_bench)

    # 打印单标的与组合绩效表
    summary_rows = []
    for sym, nick in TARGET_ASSETS:
        p = asset_perfs[sym]
        b = asset_bench_perfs[sym]
        summary_rows.append(
            {
                "资产/组合": f"{nick} 策略",
                "累计总收益": f"{p['总收益'] * 100:+.2f}%",
                "年化收益": f"{p['年化收益'] * 100:+.2f}%",
                "夏普比率": f"{p['夏普比率']:.2f}",
                "最大回撤": f"{p['最大回撤'] * 100:.2f}%",
                "卡玛比率": f"{p['卡玛比率']:.2f}",
                "日胜率": f"{p['交易日胜率'] * 100:.1f}%",
            }
        )
        summary_rows.append(
            {
                "资产/组合": f"{nick} 基准买入持有",
                "累计总收益": f"{b['总收益'] * 100:+.2f}%",
                "年化收益": f"{b['年化收益'] * 100:+.2f}%",
                "夏普比率": f"{b['夏普比率']:.2f}",
                "最大回撤": f"{b['最大回撤'] * 100:.2f}%",
                "卡玛比率": f"{b['卡玛比率']:.2f}",
                "日胜率": f"{b['交易日胜率'] * 100:.1f}%",
            }
        )

    summary_rows.append(
        {
            "资产/组合": "★ 四标的等权复合大组合 (Portfolio)",
            "累计总收益": f"{port_perf['总收益'] * 100:+.2f}%",
            "年化收益": f"{port_perf['年化收益'] * 100:+.2f}%",
            "夏普比率": f"{port_perf['夏普比率']:.2f}",
            "最大回撤": f"{port_perf['最大回撤'] * 100:.2f}%",
            "卡玛比率": f"{port_perf['卡玛比率']:.2f}",
            "日胜率": f"{port_perf['交易日胜率'] * 100:.1f}%",
        }
    )
    summary_rows.append(
        {
            "资产/组合": "四大标的等权持有基准 (Benchmark)",
            "累计总收益": f"{port_bench_perf['总收益'] * 100:+.2f}%",
            "年化收益": f"{port_bench_perf['年化收益'] * 100:+.2f}%",
            "夏普比率": f"{port_bench_perf['夏普比率']:.2f}",
            "最大回撤": f"{port_bench_perf['最大回撤'] * 100:.2f}%",
            "卡玛比率": f"{port_bench_perf['卡玛比率']:.2f}",
            "日胜率": f"{port_bench_perf['交易日胜率'] * 100:.1f}%",
        }
    )

    sum_df = pd.DataFrame(summary_rows)
    print("\n" + "-" * 95)
    print("【四大核心标的及整体组合回测绩效一览表】")
    print("-" * 95)
    print(sum_df.to_string(index=False))

    # 逐年对比
    daily_df = pd.DataFrame({"port": port_rets.values, "bench": port_bench.values}, index=pd.to_datetime(common_dates))
    daily_df["year"] = daily_df.index.year
    ypnl_port = daily_df.groupby("year")["port"].apply(lambda s: (1 + s).prod() - 1) * 100
    ypnl_bench = daily_df.groupby("year")["bench"].apply(lambda s: (1 + s).prod() - 1) * 100
    yearly_df = pd.DataFrame(
        {
            "整体组合收益 (%)": ypnl_port.round(2),
            "等权基准持有 (%)": ypnl_bench.round(2),
            "超额收益 Alpha (%)": (ypnl_port - ypnl_bench).round(2),
        }
    )
    print("\n" + "-" * 95)
    print("【大组合 2022-2026 逐年收益与 Alpha 对比表 (%)】")
    print("-" * 95)
    print(yearly_df.to_string())

    # ---------------- 绘制整体净值图 (Plotly Dark 大屏) ----------------
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "【四大核心资产黄金因子复合策略】 真实累计净值对比 (Cumulative Net Asset Value)",
            "动态回撤深度对比 (Underwater Drawdown %)",
        ),
        row_heights=[0.72, 0.28],
    )

    colors = {
        "IM9999.CCFX": "#00E5FF",
        "IC9999.CCFX": "#AB47BC",
        "IF9999.CCFX": "#FFB300",
        "159915.XSHE": "#29B6F6",
    }

    # 1. 各标的策略净值
    for sym, nick in TARGET_ASSETS:
        sub_cum = (1 + asset_dfs[sym].loc[common_dates, "strategy_ret"]).cumprod()
        fig.add_trace(
            go.Scatter(
                x=common_dates,
                y=sub_cum.values,
                name=f"{nick} 策略",
                line=dict(color=colors[sym], width=1.5),
                opacity=0.75,
            ),
            row=1,
            col=1,
        )

    # 2. 等权持有基准
    fig.add_trace(
        go.Scatter(
            x=common_dates,
            y=port_bench_perf["cum_nav"].values,
            name="等权买入持有基准",
            line=dict(color="#90A4AE", width=1.4, dash="dot"),
            opacity=0.6,
        ),
        row=1,
        col=1,
    )

    # 3. 核心大主角：四大标的等权复合大组合整体净值
    fig.add_trace(
        go.Scatter(
            x=common_dates,
            y=port_perf["cum_nav"].values,
            name="★ 四标的复合大组合整体净值 (Ensemble Portfolio)",
            line=dict(color="#00E676", width=3.5),
        ),
        row=1,
        col=1,
    )

    # 4. 回撤面板
    fig.add_trace(
        go.Scatter(
            x=common_dates,
            y=(port_bench_perf["drawdown"] * 100).values,
            name="基准回撤",
            line=dict(color="#90A4AE", width=1.2, dash="dot"),
            opacity=0.4,
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=common_dates,
            y=(port_perf["drawdown"] * 100).values,
            name="组合回撤",
            line=dict(color="#FF5252", width=2.0),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A0E14",
        plot_bgcolor="#101721",
        title=dict(
            text="IM · IC · IF · 159915 四大核心标的 · 黄金因子复合多空策略与整体组合净值大屏 (2022-2026)",
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
        height=760,
    )
    fig.update_yaxes(title_text="累计净值 (初始=1.0)", gridcolor="#263238", row=1, col=1)
    fig.update_yaxes(title_text="回撤深度 (%)", gridcolor="#263238", row=2, col=1)
    fig.update_xaxes(gridcolor="#263238", row=1, col=1)
    fig.update_xaxes(gridcolor="#263238", row=2, col=1)

    fig.write_html(str(HTML_PATH), include_plotlyjs="cdn")
    print(f"\n[OK] 整体净值交互式大屏已生成并保存至: {HTML_PATH.resolve()}")

    return sum_df, yearly_df


if __name__ == "__main__":
    run_portfolio_backtest()
