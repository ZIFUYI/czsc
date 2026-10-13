"""双顺均线粘合发散信号研究 —— 阶段四：完整策略开发与时序权重回测 (Strategy & WeightBacktest)

流程目标（原作者标准方法论）：
1. 继承 CzscStrategyBase 编写完整策略类 MaJointStrategy
2. 运行 tactic.backtest() 产出 holds / pairs / signals
3. 演示 tactic.replay() 与 tactic.save_positions() 落盘复盘产物与 Position JSON
4. 将 holds 转化为标准时序权重表 (dfw)，分别进行：
   - 单标的基准回测 (510500 真实行情)
   - 多标的组合等权回测 (510500 + 000852 + 159915 宽基组合)
5. 接入 WeightBacktest，扣除双边手续费与滑点 (2 BP)，输出核心量化指标：
   年化收益、年化波动、夏普比率、最大回撤、卡玛比率、换手率
6. 调用 wbt.generate_backtest_report 自动生成交互式 HTML 回测报告

运行：
    uv run python examples/ma_joint_study/04_strategy_backtest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 确保在 Windows 控制台输出中文不乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from wbt import generate_backtest_report

from czsc import (
    CzscStrategyBase,
    Event,
    Freq,
    Position,
    WeightBacktest,
    format_standard_kline,
)
from czsc.mock import generate_symbol_kines

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPLAY_DIR = OUTPUT_DIR / "replay_artifacts"
REPLAY_DIR.mkdir(parents=True, exist_ok=True)

FEE_RATE = 0.0002  # 单边 2 BP (万分之二，覆盖手续费与冲击滑点)
YEARLY_DAYS = 252


# ----------------------------- 1. 策略类定义 ----------------------------- #


def build_ma_joint_position(symbol: str) -> Position:
    """构建双顺均线粘合发散多头持仓策略。"""
    open_event = Event.load(
        {
            "name": "均线发散_开多",
            "operate": "开多",
            "signals_all": ["30分钟_D1SMA5#10#20#40#60#120#250_均线粘合发散V260520_看多_任意_任意_0"],
            "signals_not": ["30分钟_D1_涨跌停V230331_涨停_任意_任意_0"],
        }
    )
    exit_event = Event.load(
        {
            "name": "均线破位_平多",
            "operate": "平多",
            "signals_all": ["30分钟_D1SMA5#10#20#40#60#120#250_均线粘合发散V260520_看空_任意_任意_0"],
            "signals_not": [],
        }
    )
    return Position(
        symbol=symbol,
        name=f"MA粘合发散_{symbol}",
        opens=[open_event],
        exits=[exit_event],
        interval=3600 * 4,  # 最小开仓间隔 4 小时
        timeout=16 * 25,  # 最长持仓 25 个交易日
        stop_loss=300,  # 300 BP (3%) 硬止损
        t0=False,
    )


class MaJointStrategy(CzscStrategyBase):
    """双顺均线粘合发散标准策略类。"""

    @property
    def positions(self) -> list[Position]:
        return [build_ma_joint_position(self.symbol)]


# --------------------------- 2. 工具函数 --------------------------- #


def holds_to_weight_df(holds: pd.DataFrame) -> pd.DataFrame:
    """把 holds 转成 wbt 权重表。"""
    df = holds[["dt", "symbol", "pos", "price"]].rename(columns={"pos": "weight"})
    df["dt"] = pd.to_datetime(df["dt"])
    df["price"] = df["price"].astype(float)
    df["weight"] = df["weight"].astype(float)
    if df.duplicated(subset=["dt", "symbol"]).any():
        df = df.groupby(["dt", "symbol"], as_index=False).agg(
            weight=("weight", "mean"),
            price=("price", "first"),
        )
    return df[["dt", "symbol", "weight", "price"]]


# ------------------------------- 3. 主流程 ------------------------------- #


def main() -> None:
    print("=" * 70)
    print("  CZSC 原作者标准流程 —— 阶段四：完整策略开发与时序权重回测")
    print("=" * 70)

    # 1. 准备多资产数据池
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

    print(f"[数据] 标的池：{list(datasets.keys())} (时间范围 {sdt_str} ~ {edt_str})")

    # 2. 单标的与多标的回测执行
    all_holds_list = []
    symbol_stats_list = []

    for symbol, df in datasets.items():
        print(f"\n[回测] 执行策略回测: {symbol}...")
        bars = format_standard_kline(df, freq=Freq.F30)
        tactic = MaJointStrategy(symbol=symbol)

        # 内存回测
        res = tactic.backtest(bars, sdt="2020-06-01")
        holds = res.holds_df()
        pairs = res.pairs_df()
        print(f"  {symbol}: holds 形状 = {holds.shape}, pairs 数量 = {len(pairs)}")

        dfw_single = holds_to_weight_df(holds)
        all_holds_list.append(holds)

        # 单品种 WeightBacktest
        wb_single = WeightBacktest(data=dfw_single, fee_rate=FEE_RATE, weight_type="ts", yearly_days=YEARLY_DAYS)
        single_stats = dict(wb_single.stats)
        single_stats["标的代码"] = symbol
        symbol_stats_list.append(single_stats)

    # 3. 运行单个典型标的（510500）的 replay 落盘与 Position 序列化
    tactic_500 = MaJointStrategy(symbol="510500.XSHG")
    bars_500 = format_standard_kline(datasets["510500.XSHG"], freq=Freq.F30)
    tactic_500.replay(bars_500, res_path=REPLAY_DIR, sdt="2020-06-01", refresh=True)
    tactic_500.save_positions(REPLAY_DIR / "positions")
    print(f"\n[落盘复盘] 510500 回放产物与 Position JSON 已落盘至：{REPLAY_DIR}")

    # 4. 构建多品种组合等权权重表 (Portfolio Weight Table)
    combined_holds = pd.concat(all_holds_list, ignore_index=True)
    dfw_combined = holds_to_weight_df(combined_holds)

    # 组合层权重：每个标的分配 1/N 权重 (N=3)
    n_symbols = len(datasets)
    dfw_portfolio = dfw_combined.copy()
    dfw_portfolio["weight"] = dfw_portfolio["weight"] / float(n_symbols)

    print(f"\n[组合权重] 宽基 3 标的等权组合 (各分配 1/{n_symbols} 仓位) 权重表：{dfw_portfolio.shape}")

    # 5. 组合 WeightBacktest
    wb_portfolio = WeightBacktest(data=dfw_portfolio, fee_rate=FEE_RATE, weight_type="ts", yearly_days=YEARLY_DAYS)
    portfolio_stats = dict(wb_portfolio.stats)
    portfolio_stats["标的代码"] = "3标的等权组合"

    # 汇总绩效对比表
    all_summary_stats = symbol_stats_list + [portfolio_stats]
    summary_df = pd.DataFrame(all_summary_stats)

    cols_order = [
        "标的代码",
        "绝对收益",
        "年化收益",
        "年化波动率",
        "夏普比率",
        "最大回撤",
        "卡玛比率",
        "日胜率",
        "月胜率",
        "年胜率",
    ]
    avail_cols = [c for c in cols_order if c in summary_df.columns]
    summary_df = summary_df[avail_cols]

    print("\n" + "=" * 70)
    print("[三、单标的 vs 宽基组合 WeightBacktest 绩效对比]")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # 6. 生成交互式 HTML 回测报告
    report_500_path = OUTPUT_DIR / "510500_single_backtest_report.html"
    report_portfolio_path = OUTPUT_DIR / "multi_index_portfolio_report.html"

    dfw_500 = holds_to_weight_df(all_holds_list[0])
    generate_backtest_report(
        df=dfw_500,
        output_path=str(report_500_path),
        title="双顺均线粘合发散策略 - 中证500ETF(510500) 回测报告",
        fee_rate=FEE_RATE,
        weight_type="ts",
        yearly_days=YEARLY_DAYS,
    )

    generate_backtest_report(
        df=dfw_portfolio,
        output_path=str(report_portfolio_path),
        title="双顺均线粘合发散策略 - 宽基多标的等权组合 回测报告",
        fee_rate=FEE_RATE,
        weight_type="ts",
        yearly_days=YEARLY_DAYS,
    )

    print("\n[HTML 报告] 已生成两份自包含交互式报告：")
    print(f"  1. 单标的报告：{report_500_path} (大小: {report_500_path.stat().st_size:,} 字节)")
    print(f"  2. 多标的组合：{report_portfolio_path} (大小: {report_portfolio_path.stat().st_size:,} 字节)")

    # 7. 保存结构化数据
    summary_df.to_csv(OUTPUT_DIR / "phase4_weight_backtest_summary.csv", index=False, encoding="utf-8-sig")
    with open(OUTPUT_DIR / "phase4_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
