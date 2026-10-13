#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 159915 期权日内抄底策略族净值与回撤对比图表
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from czsc import (
    WeightBacktest,
    format_standard_kline,
)
from czsc.mock import generate_symbol_kines
from examples.bottom_reversal_159915_strategies import (
    StrategyA_2B_OptionPulse,
    StrategyB_123_OptionTrend,
    StrategyC_Multi_OptionMACD,
    BottomReversalOptionFamily,
)

OUTPUT_DIR = Path("examples/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HTML_PATH = OUTPUT_DIR / "bottom_reversal_nav_chart.html"


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


def run_all_nav():
    symbol = "159915"
    sdt = "2023-01-01"
    edt = "2023-12-31"

    print("1. 生成 159915 2023全年 5m K线数据...")
    df_raw = generate_symbol_kines(symbol, "5分钟", sdt.replace("-", ""), edt.replace("-", ""), seed=1599)
    bars = format_standard_kline(df_raw, freq="5分钟")

    # 基准 Benchmark：159915 日线买入持有收益
    df_bench = df_raw.copy()
    df_bench["date"] = df_bench["dt"].dt.strftime("%Y-%m-%d")
    daily_bench = df_bench.groupby("date")["close"].last().pct_change().fillna(0)
    bench_nav = (1 + daily_bench).cumprod()

    # 策略列表
    strats = [
        ("策略 A (2B 极速脉冲)", StrategyA_2B_OptionPulse, "#00E5FF"),      # 青色
        ("策略 B (123 趋势波段)", StrategyB_123_OptionTrend, "#FFB300"),      # 金黄色
        ("策略 C (多周期底背离)", StrategyC_Multi_OptionMACD, "#AB47BC"),     # 紫色
        ("策略族 (三合一全组合)", BottomReversalOptionFamily, "#00E676"),     # 翡翠绿 (主推)
    ]

    results = {}
    stats_table = []

    for name, cls, color in strats:
        print(f"2. 运行回测: {name}...")
        tactic = cls(symbol=symbol)
        res = tactic.backtest(bars, sdt="2023-02-01")
        holds = res.holds_df()
        dfw = holds_to_weight_df(holds)

        # 扣除双边 2 BP (0.0002) 交易摩擦
        wb = WeightBacktest(data=dfw, fee_rate=0.0002, weight_type="ts")
        daily_ret = wb.daily_return.copy()
        daily_ret["date"] = pd.to_datetime(daily_ret["date"])
        daily_ret = daily_ret.sort_values("date").set_index("date")

        nav = (1 + daily_ret["total"]).cumprod()
        # 回撤
        cum_max = nav.cummax()
        dd = (nav - cum_max) / cum_max

        st = wb.stats
        stats_table.append({
            "策略名称": name,
            "绝对收益率": f"{st.get('绝对收益', 0)*100:.2f}%",
            "年化收益率": f"{st.get('年化收益', 0)*100:.2f}%",
            "夏普比率": f"{st.get('夏普比率', 0):.2f}",
            "最大回撤": f"{st.get('最大回撤', 0)*100:.2f}%",
            "卡玛比率": f"{st.get('卡玛比率', 0):.2f}",
            "日胜率": f"{st.get('日胜率', 0)*100:.2f}%",
            "交易笔数": len(res.pairs_df()),
        })

        results[name] = {
            "dates": nav.index.strftime("%Y-%m-%d").tolist(),
            "nav": nav.values.tolist(),
            "dd": (dd * 100).values.tolist(),
            "color": color,
        }

    # 对齐 Benchmark 日期
    bench_nav_aligned = bench_nav.loc[results["策略族 (三合一全组合)"]["dates"]]
    bench_nav_norm = bench_nav_aligned / bench_nav_aligned.iloc[0]

    bench_dd = (bench_nav_norm - bench_nav_norm.cummax()) / bench_nav_norm.cummax() * 100

    # 3. 使用 Plotly 构建专业多子图
    print("3. 生成专业级交互式图表...")
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("【159915 创业板 ETF 期权日内抄底策略族】累计净值走势 (Cumulative NAV)", "动态回撤深度 (Drawdown %)"),
        row_heights=[0.72, 0.28],
    )

    # 绘制 Benchmark
    dates = results["策略族 (三合一全组合)"]["dates"]
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=bench_nav_norm.values,
            name="159915 标的买入持有 (Benchmark)",
            line=dict(color="#78909C", width=1.8, dash="dot"),
            opacity=0.75,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=bench_dd.values,
            name="Benchmark 回撤",
            line=dict(color="#78909C", width=1, dash="dot"),
            showlegend=False,
            opacity=0.5,
        ),
        row=2,
        col=1,
    )

    # 绘制各个子策略与总组合
    for name, data in results.items():
        is_family = "全组合" in name
        width = 3.2 if is_family else 1.8
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
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # 布局美化
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#121820",
        plot_bgcolor="#18222d",
        title=dict(
            text="159915 创业板 ETF 期权日内 5m 抄底策略族 —— 净值与风险归因监控看板",
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
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#455A64",
            borderwidth=1,
        ),
        margin=dict(l=50, r=30, t=100, b=50),
        height=720,
    )

    fig.update_yaxes(title_text="净值 (NAV, 初始=1.0)", gridcolor="#263238", row=1, col=1)
    fig.update_yaxes(title_text="回撤幅度 (%)", gridcolor="#263238", row=2, col=1)
    fig.update_xaxes(gridcolor="#263238", row=1, col=1)
    fig.update_xaxes(gridcolor="#263238", row=2, col=1)

    # 保存自包含 HTML
    fig.write_html(str(HTML_PATH), include_plotlyjs="cdn")
    print(f"4. 交互式净值图表已成功保存至：{HTML_PATH.resolve()}")

    # 打印采样数据供文本展示
    sample_indices = np.linspace(0, len(dates)-1, 12, dtype=int)
    sample_dates = [dates[i] for i in sample_indices]
    sample_family = [round(results["策略族 (三合一全组合)"]["nav"][i], 4) for i in sample_indices]
    sample_strat_b = [round(results["策略 B (123 趋势波段)"]["nav"][i], 4) for i in sample_indices]
    sample_bench = [round(bench_nav_norm.values[i], 4) for i in sample_indices]

    print("\n--- 关键时间采样净值序列 ---")
    for d, f_val, b_val, bench_val in zip(sample_dates, sample_family, sample_strat_b, sample_bench):
        print(f"{d} | 策略族: {f_val:.4f} | 策略B: {b_val:.4f} | 基准ETF: {bench_val:.4f}")

    print("\n--- 绩效指标汇总表 ---")
    df_stats = pd.DataFrame(stats_table)
    print(df_stats.to_string(index=False))

    return results, stats_table, sample_dates, sample_family, sample_bench


if __name__ == "__main__":
    run_all_nav()
