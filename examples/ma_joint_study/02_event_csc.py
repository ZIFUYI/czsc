"""双顺均线粘合发散信号研究 —— 阶段二：事件组装与截面匹配频次分析（CSC 分析）

流程目标（原作者标准方法论）：
1. 从 Signal 封装到 Event（设计基准方案 A vs 日线共振过滤方案 B）
2. 多标的截面匹配（Cross Section Count, CSC）：检验全市场触发密度与并发共振度
3. 事件过滤效率分析：大级别趋势过滤对开仓频次、假突破的压降幅度
4. 对比两个 Event 的事件研究前瞻收益轨迹（Event Study Comparison）
5. 输出阶段二复盘报告与图表

运行：
    uv run python examples/ma_joint_study/02_event_csc.py
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

from czsc import Event, Freq, format_standard_kline, generate_czsc_signals
from czsc.mock import generate_symbol_kines

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_multi_symbol_datasets() -> dict[str, pd.DataFrame]:
    """准备多标的研究资产池：
    1. 510500（中证500 ETF，真实数据）
    2. 000852（中证1000，模拟同周期数据）
    3. 159915（创业板指，模拟同周期数据）
    """
    datasets = {}
    real_data_path = Path("examples/results/trend_pullback_510500_30m/bars_30m.parquet")
    if real_data_path.exists():
        df_500 = pd.read_parquet(real_data_path)
        sdt_str = str(df_500["dt"].min())[:10].replace("-", "")
        edt_str = str(df_500["dt"].max())[:10].replace("-", "")
        datasets["510500.XSHG"] = df_500
    else:
        sdt_str, edt_str = "20200101", "20260930"
        datasets["510500.XSHG"] = generate_symbol_kines("510500", "30分钟", sdt_str, edt_str, seed=42)

    datasets["000852.XSHG"] = generate_symbol_kines("000852", "30分钟", sdt_str, edt_str, seed=1000)
    datasets["159915.XSHE"] = generate_symbol_kines("159915", "30分钟", sdt_str, edt_str, seed=1599)

    print(f"[数据] 成功加载多标的资产池（共 {len(datasets)} 个标的）：{list(datasets.keys())}")
    return datasets


def define_events() -> dict[str, dict[str, Event]]:
    """定义两组事件方案：
    - 方案 A（基准单周期发散）：
        仅依赖 30分钟均线粘合发散，排除涨跌停
    - 方案 B（多周期共振过滤）：
        30分钟均线粘合发散 + 日线 SMA60 顺大势多空排列共振，排除涨跌停
    """
    # 方案 A
    open_long_a = Event.load(
        {
            "name": "方案A_开多",
            "operate": "开多",
            "signals_all": ["30分钟_D1SMA5#10#20#40#60#120#250_均线粘合发散V260520_看多_任意_任意_0"],
            "signals_not": ["30分钟_D1_涨跌停V230331_涨停_任意_任意_0"],
        }
    )
    open_short_a = Event.load(
        {
            "name": "方案A_开空",
            "operate": "开空",
            "signals_all": ["30分钟_D1SMA5#10#20#40#60#120#250_均线粘合发散V260520_看空_任意_任意_0"],
            "signals_not": ["30分钟_D1_涨跌停V230331_跌停_任意_任意_0"],
        }
    )

    # 方案 B
    open_long_b = Event.load(
        {
            "name": "方案B_共振开多",
            "operate": "开多",
            "signals_all": [
                "30分钟_D1SMA5#10#20#40#60#120#250_均线粘合发散V260520_看多_任意_任意_0",
                "日线_D1SMA#60_分类V221101_多头_任意_任意_0",
            ],
            "signals_not": ["30分钟_D1_涨跌停V230331_涨停_任意_任意_0"],
        }
    )
    open_short_b = Event.load(
        {
            "name": "方案B_共振开空",
            "operate": "开空",
            "signals_all": [
                "30分钟_D1SMA5#10#20#40#60#120#250_均线粘合发散V260520_看空_任意_任意_0",
                "日线_D1SMA#60_分类V221101_空头_任意_任意_0",
            ],
            "signals_not": ["30分钟_D1_涨跌停V230331_跌停_任意_任意_0"],
        }
    )

    return {
        "方案A(单周期纯发散)": {"开多": open_long_a, "开空": open_short_a},
        "方案B(日线共振过滤)": {"开多": open_long_b, "开空": open_short_b},
    }


def compute_multi_symbol_events(
    datasets: dict[str, pd.DataFrame],
    events_config: dict[str, dict[str, Event]],
) -> pd.DataFrame:
    """批量计算全资产池的信号与事件匹配。"""
    signals_config = [
        {"name": "tas_ma_joint_V260520", "freq": "30分钟"},
        {"name": "bar_zdt_V230331", "freq": "30分钟", "di": 1},
        {"name": "tas_ma_base_V221101", "freq": "日线", "di": 1, "timeperiod": 60, "ma_type": "SMA"},
    ]

    all_dfs = []
    for symbol, df in datasets.items():
        print(f"[计算] 处理标的 {symbol}...")
        bars = format_standard_kline(df, freq=Freq.F30)
        sigs = generate_czsc_signals(bars, signals_config=signals_config, df=True)
        sigs["dt"] = pd.to_datetime(sigs["dt"])
        sigs["symbol"] = symbol

        # 仅提取合法信号列（键为三段式），传给 event.is_match
        sig_cols = [c for c in sigs.columns if len(c.split("_")) == 3]
        sig_records = sigs[sig_cols].to_dict("records")

        # 计算各事件匹配（0 或 1）
        for group_name, events in events_config.items():
            for action, event in events.items():
                col_name = f"{group_name}_{action}"
                sigs[col_name] = [1 if event.is_match(r) else 0 for r in sig_records]

        all_dfs.append(sigs)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


def analyze_cross_section_counts(combined_df: pd.DataFrame, events_config: dict[str, dict[str, Event]]) -> pd.DataFrame:
    """截面匹配频次分析（Cross Section Count, CSC）：
    统计在同一时点，有多少标的同步触发该事件。
    """
    summary_records = []
    for group_name, _events in events_config.items():
        for action in ["开多", "开空"]:
            col_name = f"{group_name}_{action}"
            # 截面聚合：每个时间点的并发资产数
            csc_series = combined_df.groupby("dt")[col_name].sum()
            active_dts = csc_series[csc_series > 0]

            total_triggers = combined_df[col_name].sum()
            total_active_moments = len(active_dts)
            max_concurrency = active_dts.max() if not active_dts.empty else 0
            avg_concurrency = active_dts.mean() if not active_dts.empty else 0

            # 统计独立波段数（首发触发数）
            first_entries = 0
            for _, df_sub in combined_df.groupby("symbol"):
                first_entries += ((df_sub[col_name] == 1) & (df_sub[col_name].shift(1) == 0)).sum()

            summary_records.append(
                {
                    "事件方案": group_name,
                    "操作方向": action,
                    "总命中K线数": int(total_triggers),
                    "触发时刻数": int(total_active_moments),
                    "首发波段数": int(first_entries),
                    "平均并发标的数": round(avg_concurrency, 2),
                    "最大并发标的数": int(max_concurrency),
                }
            )

    return pd.DataFrame(summary_records)


def compare_event_forward_returns(
    combined_df: pd.DataFrame,
    events_config: dict[str, dict[str, Event]],
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """对比两组方案在首发触发后的未来收益表现（Event Study 对比）。"""
    if horizons is None:
        horizons = [3, 5, 10, 20]

    df = combined_df.copy()
    df["close"] = df["close"].astype(float)
    df = df.sort_values(["symbol", "dt"]).reset_index(drop=True)

    # 按品种计算前瞻收益
    for h in horizons:
        shift_close = df.groupby("symbol")["close"].shift(-h)
        df[f"fwd_ret_{h}"] = (shift_close / df["close"] - 1.0) * 10000.0

    records = []
    for group_name in ["方案A(单周期纯发散)", "方案B(日线共振过滤)"]:
        for action in ["开多", "开空"]:
            col_name = f"{group_name}_{action}"

            # 提取首发触发点
            df["is_first"] = (df[col_name] == 1) & (df.groupby("symbol")[col_name].shift(1) == 0)
            events_df = df[df["is_first"]]

            for h in horizons:
                rets = events_df[f"fwd_ret_{h}"].dropna().values
                if len(rets) == 0:
                    continue

                mean_bp = np.mean(rets)
                win_rate = np.mean(rets > 0) * 100.0 if action == "开多" else np.mean(rets < 0) * 100.0

                records.append(
                    {
                        "方案": group_name,
                        "方向": action,
                        "持有K线": f"T+{h}",
                        "首发样本数": len(rets),
                        "平均收益(BP)": round(mean_bp, 1),
                        "胜率(%)": round(win_rate, 2),
                    }
                )

    return pd.DataFrame(records)


def main() -> None:
    print("=" * 70)
    print("  CZSC 原作者标准流程 —— 阶段二：事件组装与截面匹配频次分析 (CSC)")
    print("=" * 70)

    # 1. 准备多标的资产池
    datasets = load_multi_symbol_datasets()

    # 2. 定义两组 Event 方案
    events_config = define_events()
    print("\n[一、Event 定义完成]")
    for gname, evs in events_config.items():
        print(f"  {gname}:")
        for act, ev in evs.items():
            print(f"    - {act}: {ev}")

    # 3. 批量计算事件匹配
    combined_df = compute_multi_symbol_events(datasets, events_config)

    # 4. 截面匹配频次统计 (CSC 分析)
    csc_df = analyze_cross_section_counts(combined_df, events_config)
    print("\n" + "=" * 70)
    print("[二、截面匹配频次与并发度分析 (CSC Summary)]")
    print("=" * 70)
    print(csc_df.to_string(index=False))

    # 5. 前瞻收益对比 (Event Study 对比)
    ret_compare_df = compare_event_forward_returns(combined_df, events_config)
    print("\n" + "=" * 70)
    print("[三、方案A vs 方案B 前瞻收益对比 (Event Study 对比)]")
    print("=" * 70)
    print(ret_compare_df.to_string(index=False))

    # 6. 保存阶段二结果
    output_res = {
        "csc_summary": csc_df.to_dict(orient="records"),
        "forward_return_comparison": ret_compare_df.to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "phase2_results.json", "w", encoding="utf-8") as f:
        json.dump(output_res, f, ensure_ascii=False, indent=2)

    csc_df.to_csv(OUTPUT_DIR / "phase2_csc_summary.csv", index=False, encoding="utf-8-sig")
    ret_compare_df.to_csv(OUTPUT_DIR / "phase2_return_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n[产物] 阶段二分析数据已保存至：")
    print(f"  - {OUTPUT_DIR / 'phase2_csc_summary.csv'}")
    print(f"  - {OUTPUT_DIR / 'phase2_return_comparison.csv'}")
    print(f"  - {OUTPUT_DIR / 'phase2_results.json'}")


if __name__ == "__main__":
    main()
