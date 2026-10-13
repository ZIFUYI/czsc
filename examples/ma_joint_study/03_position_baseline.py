"""双顺均线粘合发散信号研究 —— 阶段三：最小持仓策略与交易对基准回测 (Position Baseline)

流程目标（原作者标准方法论）：
1. 将 Event 注入 Position 状态机，配置 interval、timeout、stop_loss 风控参数
2. 对比两组持仓策略：
   - Position A（基准多头：纯 30m 均线发散）
   - Position B（共振多头：30m 发散 + 日线 60MA 顺势共振）
3. 执行原生 Rust CzscStrategyBase 回测，产出精确开平仓对（Pairs）
4. 深度评估交易对指标：
   - 胜率 (Win Rate)、盈亏比 (PL Ratio)、单笔均益 (Mean Trade PnL)
   - 盈亏平衡胜率 (Break-even Win Rate) 与安全边际
   - 出场归因（正常信号离场 vs 300BP止损 vs 超时平仓）
5. 跨资产横向对比（510500、000852、159915）并输出结构化结果

运行：
    uv run python examples/ma_joint_study/03_position_baseline.py
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
import pyarrow.ipc as ipc

from czsc import CzscStrategyBase, Event, Freq, Position, format_standard_kline
from czsc.mock import generate_symbol_kines

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_events() -> dict:
    """定义两组多头进出场事件。"""
    # 方案 A：纯单周期
    open_a = Event.load(
        {
            "name": "开多_单周期发散",
            "operate": "开多",
            "signals_all": ["30分钟_D1SMA5#10#20#40#60#120#250_均线粘合发散V260520_看多_任意_任意_0"],
            "signals_not": ["30分钟_D1_涨跌停V230331_涨停_任意_任意_0"],
        }
    )
    # 方案 B：日线共振
    open_b = Event.load(
        {
            "name": "开多_日线共振发散",
            "operate": "开多",
            "signals_all": [
                "30分钟_D1SMA5#10#20#40#60#120#250_均线粘合发散V260520_看多_任意_任意_0",
                "日线_D1SMA#60_分类V221101_多头_任意_任意_0",
            ],
            "signals_not": ["30分钟_D1_涨跌停V230331_涨停_任意_任意_0"],
        }
    )
    # 通用平多
    exit_common = Event.load(
        {
            "name": "平多_均线向下破位",
            "operate": "平多",
            "signals_all": ["30分钟_D1SMA5#10#20#40#60#120#250_均线粘合发散V260520_看空_任意_任意_0"],
            "signals_not": [],
        }
    )
    return {"open_a": open_a, "open_b": open_b, "exit": exit_common}


def make_strategy_class(name_prefix: str, open_event: Event, exit_event: Event):
    """动态创建策略类。"""

    class CustomStrategy(CzscStrategyBase):
        @property
        def positions(self) -> list[Position]:
            pos = Position(
                symbol=self.symbol,
                name=f"{name_prefix}_{self.symbol}",
                opens=[open_event],
                exits=[exit_event],
                interval=3600 * 4,  # 两次开仓最小间隔 4 小时
                timeout=16 * 25,  # 最长持仓 25 个交易日（200 根 30m K线）
                stop_loss=300,  # 300 BP (3%) 硬止损
                t0=False,
            )
            return [pos]

    return CustomStrategy


def evaluate_pairs_performance(df_pairs: pd.DataFrame, fee_bp: float = 2.0) -> dict:
    """全面评估交易对微观指标。"""
    if df_pairs.empty:
        return {}

    df = df_pairs.copy()
    # 盈亏比例字段在 Rust 中为 BP（基点）
    pnl = df["盈亏比例"].values.astype(float) - fee_bp  # 扣除双边手续费

    total_trades = len(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_count = len(wins)
    len(losses)
    win_rate = win_count / total_trades * 100.0 if total_trades > 0 else 0.0

    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1.0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else np.nan

    # 盈亏平衡胜率：1 / (1 + 盈亏比)
    be_win_rate = (1.0 / (1.0 + pl_ratio)) * 100.0 if not np.isnan(pl_ratio) else np.nan

    mean_pnl = np.mean(pnl)
    total_pnl = np.sum(pnl)
    max_win = np.max(pnl) if len(pnl) > 0 else 0.0
    max_loss = np.min(pnl) if len(pnl) > 0 else 0.0

    # 持仓时间统计
    hold_days = df["持仓天数"].values.astype(float)
    hold_bars = df["持仓K线数"].values.astype(float)

    # 出场类型归因
    exit_events = df["事件序列"].astype(str)
    stop_loss_count = exit_events.str.contains("止损").sum()
    timeout_count = exit_events.str.contains("超时").sum()
    signal_exit_count = total_trades - stop_loss_count - timeout_count

    return {
        "总交易笔数": total_trades,
        "交易胜率(%)": round(win_rate, 2),
        "盈亏比": round(pl_ratio, 2),
        "盈亏平衡胜率(%)": round(be_win_rate, 2),
        "安全胜率边际(%)": round(win_rate - be_win_rate, 2) if not np.isnan(be_win_rate) else np.nan,
        "单笔均益(BP)": round(mean_pnl, 1),
        "累计总收益(BP)": round(total_pnl, 1),
        "最大单笔盈利(BP)": round(max_win, 1),
        "最大单笔亏损(BP)": round(max_loss, 1),
        "平均持仓天数": round(np.mean(hold_days), 1),
        "平均持仓K线数": round(np.mean(hold_bars), 1),
        "止损离场占比(%)": round(stop_loss_count / total_trades * 100.0, 1),
        "正常信号平仓占比(%)": round(signal_exit_count / total_trades * 100.0, 1),
    }


def main() -> None:
    print("=" * 70)
    print("  CZSC 原作者标准流程 —— 阶段三：最小持仓策略与交易对基准回测 (Position)")
    print("=" * 70)

    # 1. 准备多资产数据
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

    # 2. 构建事件与策略类
    evs = build_events()
    StratA = make_strategy_class("Position_A_纯发散", evs["open_a"], evs["exit"])
    StratB = make_strategy_class("Position_B_共振发散", evs["open_b"], evs["exit"])

    all_results = []
    all_pairs_list = []

    for symbol, df in datasets.items():
        print(f"\n[回测] 正在运行标的：{symbol} (共 {len(df)} 根 K线)...")
        bars = format_standard_kline(df, freq=Freq.F30)

        for strat_name, StratClass in [("方案A(单周期纯发散)", StratA), ("方案B(日线共振过滤)", StratB)]:
            tactic = StratClass(symbol=symbol)
            res = tactic.backtest(bars)

            with ipc.open_file(res.pairs_arrow) as reader:
                pairs = reader.read_all().to_pandas()

            if not pairs.empty:
                pairs["方案"] = strat_name
                pairs["标的"] = symbol
                all_pairs_list.append(pairs)

            metrics = evaluate_pairs_performance(pairs, fee_bp=2.0)
            metrics["标的代码"] = symbol
            metrics["方案名称"] = strat_name
            all_results.append(metrics)

    res_df = pd.DataFrame(all_results)
    # 调整列顺序
    cols_order = [
        "标的代码",
        "方案名称",
        "总交易笔数",
        "交易胜率(%)",
        "盈亏比",
        "盈亏平衡胜率(%)",
        "安全胜率边际(%)",
        "单笔均益(BP)",
        "累计总收益(BP)",
        "平均持仓天数",
        "止损离场占比(%)",
    ]
    res_df = res_df[cols_order]

    print("\n" + "=" * 70)
    print("[三、各标的交易对回测表现汇总 (Pairs Performance)]")
    print("=" * 70)
    print(res_df.to_string(index=False))

    # 汇总各方案跨资产总表现
    combined_pairs = pd.concat(all_pairs_list, ignore_index=True)
    summary_by_scheme = []
    for gname in ["方案A(单周期纯发散)", "方案B(日线共振过滤)"]:
        sub_p = combined_pairs[combined_pairs["方案"] == gname]
        m = evaluate_pairs_performance(sub_p, fee_bp=2.0)
        m["方案名称"] = gname
        summary_by_scheme.append(m)

    scheme_df = pd.DataFrame(summary_by_scheme)[
        [
            "方案名称",
            "总交易笔数",
            "交易胜率(%)",
            "盈亏比",
            "盈亏平衡胜率(%)",
            "安全胜率边际(%)",
            "单笔均益(BP)",
            "累计总收益(BP)",
            "平均持仓天数",
            "止损离场占比(%)",
        ]
    ]
    print("\n" + "=" * 70)
    print("[四、跨资产全样本综合表现对比]")
    print("=" * 70)
    print(scheme_df.to_string(index=False))

    # 保存产物
    res_df.to_csv(OUTPUT_DIR / "phase3_pairs_summary.csv", index=False, encoding="utf-8-sig")
    scheme_df.to_csv(OUTPUT_DIR / "phase3_scheme_comparison.csv", index=False, encoding="utf-8-sig")
    combined_pairs.to_parquet(OUTPUT_DIR / "phase3_all_pairs.parquet")

    with open(OUTPUT_DIR / "phase3_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "symbol_summary": res_df.to_dict(orient="records"),
                "scheme_summary": scheme_df.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n[产物] 阶段三分析数据已保存至：")
    print(f"  - {OUTPUT_DIR / 'phase3_pairs_summary.csv'}")
    print(f"  - {OUTPUT_DIR / 'phase3_scheme_comparison.csv'}")
    print(f"  - {OUTPUT_DIR / 'phase3_results.json'}")


if __name__ == "__main__":
    main()
