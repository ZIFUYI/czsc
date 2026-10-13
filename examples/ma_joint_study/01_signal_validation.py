"""双顺均线粘合发散信号研究 —— 阶段一：单信号特征与统计有效性检验

流程目标（原作者标准方法论）：
1. 信号计算与状态分布验证（完全分类、无未来函数）
2. 状态驻留性分析（平均持续 K 线数）
3. 首发触发事件（First Entry）识别
4. 事件研究法（Event Study）：T+1 到 T+20 根 K 线的未来收益、胜率与 95% 置信区间
5. 生成可视化 HTML 快照与统计汇总

运行：
    uv run python examples/ma_joint_study/01_signal_validation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 确保在 Windows 控制台输出中文不乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy import stats

from czsc import Freq, format_standard_kline, generate_czsc_signals
from czsc.mock import generate_symbol_kines

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """加载研究数据集。优先使用真实 510500 30m 行情，备选 mock 000852。"""
    real_data_path = Path("examples/results/trend_pullback_510500_30m/bars_30m.parquet")
    if real_data_path.exists():
        df = pd.read_parquet(real_data_path)
        print(
            f"[数据] 加载真实数据 {real_data_path}：{len(df)} 根 30m K线，时间跨度 {df['dt'].min()} ~ {df['dt'].max()}"
        )
        return df
    else:
        df = generate_symbol_kines("000852", "30分钟", "20200101", "20260101", seed=42)
        print(f"[数据] 使用 mock 数据：{len(df)} 根 30m K线")
        return df


def calculate_signal(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """计算 tas_ma_joint_V260520 信号。"""
    bars = format_standard_kline(df, freq=Freq.F30)
    signals_config = [{"name": "tas_ma_joint_V260520", "freq": "30分钟"}]

    sigs_df = generate_czsc_signals(bars, signals_config=signals_config, df=True)

    sig_col = [c for c in sigs_df.columns if "均线粘合发散" in c or "joint" in c][0]
    print(f"\n[信号] 成功计算信号列：{sig_col}")
    return sigs_df, sig_col


def analyze_state_distribution(sigs_df: pd.DataFrame, sig_col: str) -> pd.DataFrame:
    """分析状态分布及平均持续时长（驻留性）。"""
    s = sigs_df[sig_col].copy()
    # 提取核心状态：看多 / 看空 / 其他
    state_series = s.apply(lambda x: x.split("_")[0] if isinstance(x, str) else "其他")
    sigs_df["state"] = state_series

    # 1. 频次与占比
    counts = state_series.value_counts()
    pcts = state_series.value_counts(normalize=True) * 100

    # 2. 连续状态长度分析（每次连续维持该状态多少根 K 线）
    change_mask = state_series != state_series.shift(1)
    run_id = change_mask.cumsum()
    run_lengths = sigs_df.groupby(run_id)["state"].agg(state="first", length="count")

    duration_stats = run_lengths.groupby("state")["length"].agg(
        连续次数="count",
        平均持续K线="mean",
        中位数持续K线="median",
        最大持续K线="max",
    )

    summary = pd.DataFrame({"总出现K线数": counts, "时间占比(%)": pcts}).join(duration_stats)
    return summary


def run_event_study(sigs_df: pd.DataFrame, horizons: list[int] | None = None) -> dict:
    """事件研究法（Event Study）：检验首次看多/看空后的未来累计收益率。

    首发触发点定义：
    - 多头首发：前一根状态 != '看多' 且 当前状态 == '看多'
    - 空头首发：前一根状态 != '看空' 且 当前状态 == '看空'
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 8, 10, 15, 20]

    df = sigs_df.copy()
    df["dt"] = pd.to_datetime(df["dt"])
    df["close"] = df["close"].astype(float)
    df = df.sort_values("dt").reset_index(drop=True)

    # 标记首发触发点（First Entry Trigger）
    df["prev_state"] = df["state"].shift(1)
    df["is_long_trigger"] = (df["state"] == "看多") & (df["prev_state"] != "看多")
    df["is_short_trigger"] = (df["state"] == "看空") & (df["prev_state"] != "看空")

    close = df["close"].values.astype(np.float64)
    n = len(df)

    # 计算各周期的未来收益率（基点 BP）
    for h in horizons:
        ret_forward = np.full(n, np.nan)
        ret_forward[:-h] = (close[h:] / close[:-h] - 1.0) * 10000.0
        df[f"fwd_ret_{h}"] = ret_forward

    results = {}
    for event_type, mask in [("多头首发触发", df["is_long_trigger"]), ("空头首发触发", df["is_short_trigger"])]:
        event_df = df[mask].copy()
        event_count = len(event_df)

        stats_list = []
        for h in horizons:
            rets = event_df[f"fwd_ret_{h}"].dropna().values
            if len(rets) == 0:
                continue

            mean_bp = np.mean(rets)
            median_bp = np.median(rets)
            std_bp = np.std(rets, ddof=1)
            se_bp = std_bp / np.sqrt(len(rets))
            ci95_low = mean_bp - 1.96 * se_bp
            ci95_high = mean_bp + 1.96 * se_bp

            win_rate = np.mean(rets > 0) * 100.0 if event_type == "多头首发触发" else np.mean(rets < 0) * 100.0

            t_val, p_val = stats.ttest_1samp(rets, 0.0)

            stats_list.append(
                {
                    "持有K线数": h,
                    "平均收益(BP)": round(mean_bp, 1),
                    "中位数(BP)": round(median_bp, 1),
                    "胜率(%)": round(win_rate, 2),
                    "收益标准差(BP)": round(std_bp, 1),
                    "95%置信区间": f"[{ci95_low:.1f}, {ci95_high:.1f}]",
                    "t统计量": round(t_val, 2),
                    "p值": round(p_val, 4),
                }
            )

        results[event_type] = {
            "触发总次数": event_count,
            "绩效明细": pd.DataFrame(stats_list),
        }

    return results, df


def export_html_visualizer(df: pd.DataFrame, sig_col: str, output_path: Path) -> None:
    """输出阶段一的可视化复盘 HTML（包含价格线与首发买卖触发点标记）。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plot_df = df.tail(1500).copy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
        subplot_titles=["30分钟K线走势与首发信号触发点", "状态时序 (1=看多, -1=看空, 0=其他)"],
    )

    # 主图收盘价曲线
    fig.add_trace(
        go.Scatter(
            x=plot_df["dt"],
            y=plot_df["close"],
            mode="lines",
            name="收盘价",
            line={"color": "#4682B4", "width": 1.5},
        ),
        row=1,
        col=1,
    )

    # 标记多头首发点
    long_triggers = plot_df[plot_df["is_long_trigger"]]
    fig.add_trace(
        go.Scatter(
            x=long_triggers["dt"],
            y=long_triggers["close"],
            mode="markers",
            name="多头首发触发",
            marker={"symbol": "triangle-up", "size": 10, "color": "#FF4500"},
        ),
        row=1,
        col=1,
    )

    # 标记空头首发点
    short_triggers = plot_df[plot_df["is_short_trigger"]]
    fig.add_trace(
        go.Scatter(
            x=short_triggers["dt"],
            y=short_triggers["close"],
            mode="markers",
            name="空头首发触发",
            marker={"symbol": "triangle-down", "size": 10, "color": "#32CD32"},
        ),
        row=1,
        col=1,
    )

    # 副图状态阶梯
    state_numeric = plot_df["state"].map({"看多": 1, "看空": -1, "其他": 0})
    fig.add_trace(
        go.Scatter(
            x=plot_df["dt"],
            y=state_numeric,
            mode="lines",
            name="信号状态",
            line={"color": "#8A2BE2", "width": 1.2, "shape": "hv"},
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="双顺均线粘合发散信号（tas_ma_joint_V260520）有效性复盘",
        hovermode="x unified",
        template="plotly_white",
        height=700,
    )
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"\n[可视化] 交互式复盘图表已保存至：{output_path}")


def main() -> None:
    print("=" * 70)
    print("  CZSC 原作者标准流程 —— 阶段一：单信号特征与有效性检验")
    print("=" * 70)

    # 1. 加载数据
    df = load_dataset()

    # 2. 计算信号
    sigs_df, sig_col = calculate_signal(df)

    # 3. 状态分布与驻留性分析
    dist_df = analyze_state_distribution(sigs_df, sig_col)
    print("\n[一、状态分布与驻留性分析]")
    print(dist_df.to_string())

    # 4. 事件研究法（Event Study）
    (event_results, df_with_events) = run_event_study(sigs_df)
    print("\n" + "=" * 70)
    print("[二、事件研究法收益表现（Event Study）]")
    print("=" * 70)

    for event_name, res in event_results.items():
        print(f"\n>>> 【{event_name}】(全期触发次数 = {res['触发总次数']})")
        print(res["绩效明细"].to_string(index=False))

    # 5. 输出可视化复盘 HTML
    html_path = OUTPUT_DIR / "phase1_signal_validation.html"
    export_html_visualizer(df_with_events, sig_col, html_path)

    # 6. 保存阶段一统计结果 json
    report_dict = {
        "signal": sig_col,
        "sample_size": len(df),
        "date_range": [str(df["dt"].min()), str(df["dt"].max())],
        "state_distribution": dist_df.to_dict(),
        "long_event_triggers": int(event_results["多头首发触发"]["触发总次数"]),
        "short_event_triggers": int(event_results["空头首发触发"]["触发总次数"]),
        "long_performance": event_results["多头首发触发"]["绩效明细"].to_dict(orient="records"),
        "short_performance": event_results["空头首发触发"]["绩效明细"].to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "phase1_results.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    print("\n[产物] 阶段一检验数据已写入 output/phase1_results.json")


if __name__ == "__main__":
    main()
