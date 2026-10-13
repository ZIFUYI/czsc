#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用天勤 (TQSdk) 下载的 2020-2026 100% 真实 159915 5分钟数据
执行期权日内抄底策略族全量回测与真实净值图表生成
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

from czsc import (
    WeightBacktest,
    format_standard_kline,
)
from examples.bottom_reversal_159915_strategies import (
    StrategyA_2B_OptionPulse,
    StrategyB_123_OptionTrend,
    StrategyC_Multi_OptionMACD,
    BottomReversalOptionFamily,
)

OUTPUT_DIR = Path("examples/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REAL_CSV = OUTPUT_DIR / "159915_real_5m_2020_2026.csv"
HTML_PATH = OUTPUT_DIR / "bottom_reversal_nav_chart_real_tq.html"


def load_real_tq_bars(csv_path: Path):
    """把天勤 CSV 数据转换成 CZSC 标准 K 线格式"""
    df_raw = pd.read_csv(csv_path)

    # 规范列名
    df = pd.DataFrame()
    df["dt"] = pd.to_datetime(df_raw["datetime"])
    df["symbol"] = "159915"
    df["open"] = df_raw["SZSE.159915.open"].astype(float)
    df["high"] = df_raw["SZSE.159915.high"].astype(float)
    df["low"] = df_raw["SZSE.159915.low"].astype(float)
    df["close"] = df_raw["SZSE.159915.close"].astype(float)
    df["vol"] = df_raw["SZSE.159915.volume"].astype(float)
    df["amount"] = df["vol"] * df["close"]

    # 过滤无效空值
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("dt").reset_index(drop=True)
    return df


def holds_to_weight_df(holds: pd.DataFrame) -> pd.DataFrame:
    """转换 holds 为标准权重格式"""
    df = holds[["dt", "symbol", "pos", "price"]].rename(columns={"pos": "weight"})
    df["dt"] = df["dt"].astype(str)
    df["price"] = df["price"].astype(float)
    df["weight"] = df["weight"].astype(float)
    if df.duplicated(subset=["dt", "symbol"]).any():
        df = df.groupby(["dt", "symbol"], as_index=False).agg(
            weight=("weight", "mean"),
            price=("price", "first"),
        )
    return df[["dt", "symbol", "weight", "price"]]


def run_real_backtest():
    print("=" * 70)
    print("  加载天勤 100% 真实 159915 创业板 ETF (2020-2026) 5分钟历史行情...")
    print("=" * 70)

    df_real = load_real_tq_bars(REAL_CSV)
    print(f"数据总行数: {len(df_real)} 根 5m K线")
    print(f"时间跨度: {df_real['dt'].min()} ~ {df_real['dt'].max()}")
    print(f"历史价格区间: {df_real['low'].min():.3f} ~ {df_real['high'].max():.3f} 元")
    print(f"标的自身 7年总涨跌幅: {(df_real['close'].iloc[-1] / df_real['close'].iloc[0] - 1)*100:+.2f}%\n")

    bars = format_standard_kline(df_real, freq="5分钟")

    # 基准 Benchmark：159915 现货买入持有真实日度净值
    df_bench = df_real.copy()
    df_bench["date"] = df_bench["dt"].dt.strftime("%Y-%m-%d")
    daily_bench = df_bench.groupby("date")["close"].last().pct_change().fillna(0)
    bench_nav = (1 + daily_bench).cumprod()

    # 策略清单
    strats = [
        ("策略 A (2B 极速脉冲)", StrategyA_2B_OptionPulse, "#00E5FF"),      # 青色
        ("策略 B (123 趋势波段)", StrategyB_123_OptionTrend, "#FFB300"),      # 金黄色
        ("策略 C (多周期底背离)", StrategyC_Multi_OptionMACD, "#AB47BC"),     # 紫色
        ("策略族 (三合一全组合)", BottomReversalOptionFamily, "#00E676"),     # 翡翠绿 (主推)
    ]

    results = {}
    stats_table = []
    yearly_breakdowns = {}

    for name, cls, color in strats:
        print(f"运行真实数据回测: {name}...")
        tactic = cls(symbol="159915")
        # 从 2020-03-01 开始回测 (预留 2 个月计算长期均线和指标底仓)
        res = tactic.backtest(bars, sdt="2020-03-01")
        holds = res.holds_df()
        pairs = res.pairs_df()
        dfw = holds_to_weight_df(holds)

        # 扣除双边 2 BP (0.0002) 手续费与冲击滑点
        wb = WeightBacktest(data=dfw, fee_rate=0.0002, weight_type="ts")
        daily_ret = wb.daily_return.copy()
        daily_ret["date"] = pd.to_datetime(daily_ret["date"])
        daily_ret = daily_ret.sort_values("date").set_index("date")

        nav = (1 + daily_ret["total"]).cumprod()
        cum_max = nav.cummax()
        dd = (nav - cum_max) / cum_max

        # 计算分年度收益率
        daily_ret["year"] = daily_ret.index.year
        yearly_pnl = daily_ret.groupby("year")["total"].apply(lambda s: (1 + s).prod() - 1)
        yearly_breakdowns[name] = yearly_pnl

        st = wb.stats
        stats_table.append({
            "策略名称": name,
            "真实绝对收益": f"{st.get('绝对收益', 0)*100:.2f}%",
            "真实年化收益": f"{st.get('年化收益', 0)*100:.2f}%",
            "夏普比率": f"{st.get('夏普比率', 0):.2f}",
            "最大回撤": f"{st.get('最大回撤', 0)*100:.2f}%",
            "卡玛比率": f"{st.get('卡玛比率', 0):.2f}",
            "日胜率": f"{st.get('日胜率', 0)*100:.2f}%",
            "真实交易笔数": len(pairs),
        })

        results[name] = {
            "dates": nav.index.strftime("%Y-%m-%d").tolist(),
            "nav": nav.values.tolist(),
            "dd": (dd * 100).values.tolist(),
            "color": color,
        }

    # 对齐 Benchmark
    dates = results["策略族 (三合一全组合)"]["dates"]
    bench_nav_aligned = bench_nav.loc[dates]
    bench_nav_norm = bench_nav_aligned / bench_nav_aligned.iloc[0]
    bench_dd = (bench_nav_norm - bench_nav_norm.cummax()) / bench_nav_norm.cummax() * 100

    # 基准分年度收益率
    df_bench_aligned = pd.DataFrame({"bench_ret": daily_bench.loc[dates]}, index=pd.to_datetime(dates))
    df_bench_aligned["year"] = df_bench_aligned.index.year
    bench_yearly = df_bench_aligned.groupby("year")["bench_ret"].apply(lambda s: (1 + s).prod() - 1)
    yearly_breakdowns["基准 159915 买入持有"] = bench_yearly

    # 绘制真实数据 Plotly 大屏
    print("\n生成天勤真实数据交互式大屏...")
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "【159915 真实天勤行情 · 期权日内抄底策略族 2020-2026】真实累计净值走势 (Cumulative NAV)",
            "真实动态回撤深度 (Drawdown %)"
        ),
        row_heights=[0.72, 0.28],
    )

    # Benchmark
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=bench_nav_norm.values,
            name="159915 现货买入持有 (真实基准)",
            line=dict(color="#90A4AE", width=1.6, dash="dot"),
            opacity=0.7,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=bench_dd.values,
            name="基准回撤",
            line=dict(color="#90A4AE", width=1, dash="dot"),
            showlegend=False,
            opacity=0.4,
        ),
        row=2,
        col=1,
    )

    # 策略曲线
    for name, data in results.items():
        is_family = "全组合" in name
        width = 3.5 if is_family else 1.8
        fig.add_trace(
            go.Scatter(
                x=data["dates"],
                y=data["nav"],
                name=name,
                line=dict(color=data["color"], width=width),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data["dates"],
                y=data["dd"],
                name=f"{name} 回撤",
                line=dict(color=data["color"], width=1.2),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A0E14",
        plot_bgcolor="#101721",
        title=dict(
            text="159915 创业板 ETF 期权日内抄底策略族 —— 天勤真实数据回测大屏 (2020-2026)",
            font=dict(size=18, color="#FFFFFF", family="Arial, sans-serif"),
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
            bordercolor="#37474F",
            borderwidth=1,
        ),
        margin=dict(l=50, r=30, t=100, b=50),
        height=750,
    )

    fig.update_yaxes(title_text="累计净值 (NAV, 初始=1.0)", gridcolor="#263238", row=1, col=1)
    fig.update_yaxes(title_text="回撤幅度 (%)", gridcolor="#263238", row=2, col=1)
    fig.update_xaxes(gridcolor="#263238", row=1, col=1)
    fig.update_xaxes(gridcolor="#263238", row=2, col=1)

    fig.write_html(str(HTML_PATH), include_plotlyjs="cdn")
    print(f"真实大屏图表已保存至: {HTML_PATH.resolve()}")

    # 打印分年度收益表
    df_yearly = pd.DataFrame(yearly_breakdowns) * 100
    print("\n" + "=" * 70)
    print("【天勤真实数据 · 2020-2026 分年度收益率统计 (%)】")
    print("=" * 70)
    print(df_yearly.round(2).to_string())

    print("\n" + "=" * 70)
    print("【天勤真实数据 · 7年全周期真实绩效汇总表】")
    print("=" * 70)
    df_stats = pd.DataFrame(stats_table)
    print(df_stats.to_string(index=False))

    return results, df_stats, df_yearly


if __name__ == "__main__":
    run_real_backtest()
