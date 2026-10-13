"""双顺均线粘合发散信号研究 —— 阶段五：边际优化与实盘摩擦治理 (Optimization & Robustness)

流程目标（原作者标准方法论）：
1. 入场边际过滤优化 (Opens Marginal Optimization)：
   对比基准发散与叠加不同辅助维度（短期均线顺排、MACD动量多头、日线大趋势）的边际贡献
2. 风控止损与开仓间隔敏感性测试 (Risk Parameter Sensitivity)：
   - stop_loss (150 BP, 300 BP, 500 BP, 1000 BP)
   - interval (0h, 4h, 8h, 24h)
3. 费率与滑点抗压测试 (Friction Stress Test)：
   - fee_rate (0 BP, 1 BP, 2 BP, 5 BP, 10 BP)
4. 输出完整的优化分析矩阵与全流程投研总结

运行：
    uv run python examples/ma_joint_study/05_marginal_optimize.py
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

from czsc import (
    CzscStrategyBase,
    Event,
    Freq,
    Position,
    WeightBacktest,
    format_standard_kline,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_real_bars():
    """加载中证500ETF真实数据。"""
    real_data_path = Path("examples/results/trend_pullback_510500_30m/bars_30m.parquet")
    df = pd.read_parquet(real_data_path)
    bars = format_standard_kline(df, freq=Freq.F30)
    print(f"[数据] 加载 {bars[0].symbol}，共 {len(bars)} 根 30m K线 ({bars[0].dt} ~ {bars[-1].dt})")
    return bars


def holds_to_weight_df(holds: pd.DataFrame) -> pd.DataFrame:
    """转换 holds 为标准权重表。"""
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


class ParamStudyStrategy(CzscStrategyBase):
    """用于参数敏感性与边际测试的通用策略类。"""

    def __init__(
        self,
        symbol: str,
        name: str,
        open_e: Event,
        exit_e: Event,
        interval: int = 3600 * 4,
        stop_loss: int = 300,
    ):
        super().__init__(symbol)
        self._pos_name = name
        self._open_e = open_e
        self._exit_e = exit_e
        self._interval = interval
        self._stop_loss = stop_loss

    @property
    def positions(self) -> list[Position]:
        return [
            Position(
                symbol=self.symbol,
                name=self._pos_name,
                opens=[self._open_e],
                exits=[self._exit_e],
                interval=self._interval,
                timeout=16 * 25,
                stop_loss=self._stop_loss,
                t0=False,
            )
        ]


# ============================================================================
# 实验一：入场边际过滤优化 (Opens Optimization)
# ============================================================================


def run_opens_optimization(bars: list) -> pd.DataFrame:
    """测试不同入场过滤条件对基准策略的边际增强效果。"""
    print("\n" + "=" * 70)
    print("【实验一】入场边际过滤优化 (Opens Optimization)")
    print("=" * 70)

    exit_event = Event.load(
        {
            "name": "平多_破位",
            "operate": "平多",
            "signals_all": ["30分钟_D1SMA5#10#20#40#60#120#250_均线粘合发散V260520_看空_任意_任意_0"],
            "signals_not": [],
        }
    )

    base_sig = "30分钟_D1SMA5#10#20#40#60#120#250_均线粘合发散V260520_看多_任意_任意_0"
    not_zt = "30分钟_D1_涨跌停V230331_涨停_任意_任意_0"

    candidates = {
        "V0_基准(纯发散)": [base_sig],
        "V1_叠加短期均线顺排": [
            base_sig,
            "30分钟_D1SMA5#10#20_均线系统V230513_多头排列_任意_任意_0",
        ],
        "V2_叠加MACD多头向上": [
            base_sig,
            "30分钟_D1MACD12#26#9_BS辅助V221028_多头_向上_任意_0",
        ],
        "V3_叠加日线60MA顺势": [
            base_sig,
            "日线_D1SMA#60_分类V221101_多头_任意_任意_0",
        ],
        "V4_全维共振(发散+短均顺+MACD)": [
            base_sig,
            "30分钟_D1SMA5#10#20_均线系统V230513_多头排列_任意_任意_0",
            "30分钟_D1MACD12#26#9_BS辅助V221028_多头_向上_任意_0",
        ],
    }

    results = []
    for var_name, sigs_all in candidates.items():
        open_e = Event.load(
            {
                "name": f"开多_{var_name}",
                "operate": "开多",
                "signals_all": sigs_all,
                "signals_not": [not_zt],
            }
        )

        tactic = ParamStudyStrategy(
            symbol="510500.XSHG",
            name=f"MA_{var_name}",
            open_e=open_e,
            exit_e=exit_event,
            interval=3600 * 4,
            stop_loss=300,
        )
        res = tactic.backtest(bars, sdt="2020-06-01")
        pairs = res.pairs_df()
        holds = res.holds_df()

        dfw = holds_to_weight_df(holds)
        wb = WeightBacktest(dfw, fee_rate=0.0002, yearly_days=252)

        # 统计指标
        pnl = pairs["盈亏比例"].astype(float) - 2.0 if not pairs.empty else np.array([0])
        win_rate = np.mean(pnl > 0) * 100.0 if len(pnl) > 0 else 0
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        pl_ratio = np.mean(wins) / abs(np.mean(losses)) if len(losses) > 0 and len(wins) > 0 else np.nan

        results.append(
            {
                "优化变体方案": var_name,
                "交易笔数": len(pairs),
                "交易胜率(%)": round(win_rate, 2),
                "盈亏比": round(pl_ratio, 2),
                "单笔均益(BP)": round(float(np.mean(pnl)), 1),
                "年化收益(%)": round(wb.stats["年化收益"] * 100.0, 2),
                "最大回撤(%)": round(wb.stats["最大回撤"] * 100.0, 2),
                "夏普比率": round(wb.stats["夏普比率"], 2),
                "卡玛比率": round(wb.stats["卡玛比率"], 2),
            }
        )

    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    return df_res


# ============================================================================
# 实验二：风控止损与开仓间隔敏感性测试
# ============================================================================


def run_parameter_sensitivity(bars: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """测试止损阈值 (stop_loss) 与开仓保护间隔 (interval) 的敏感性。"""
    print("\n" + "=" * 70)
    print("【实验二】风控止损与开仓保护间隔敏感性测试")
    print("=" * 70)

    open_e = Event.load(
        {
            "name": "开多",
            "operate": "开多",
            "signals_all": ["30分钟_D1SMA5#10#20#40#60#120#250_均线粘合发散V260520_看多_任意_任意_0"],
            "signals_not": ["30分钟_D1_涨跌停V230331_涨停_任意_任意_0"],
        }
    )
    exit_e = Event.load(
        {
            "name": "平多",
            "operate": "平多",
            "signals_all": ["30分钟_D1SMA5#10#20#40#60#120#250_均线粘合发散V260520_看空_任意_任意_0"],
            "signals_not": [],
        }
    )

    # 1. 止损梯度测试 (固定 interval=4小时)
    stop_losses = [150, 300, 500, 1000]
    sl_results = []
    for sl in stop_losses:
        res = ParamStudyStrategy(
            symbol="510500.XSHG",
            name=f"SL_{sl}",
            open_e=open_e,
            exit_e=exit_e,
            interval=3600 * 4,
            stop_loss=sl,
        ).backtest(bars, sdt="2020-06-01")
        pairs = res.pairs_df()
        wb = WeightBacktest(holds_to_weight_df(res.holds_df()), fee_rate=0.0002, yearly_days=252)

        sl_count = pairs["事件序列"].str.contains("止损").sum() if not pairs.empty else 0
        sl_results.append(
            {
                "止损阈值(BP)": sl,
                "止损幅度(%)": f"{sl / 100:.1f}%",
                "总交易笔数": len(pairs),
                "止损触发次数": int(sl_count),
                "止损占比(%)": round(sl_count / len(pairs) * 100.0, 1) if len(pairs) > 0 else 0,
                "年化收益(%)": round(wb.stats["年化收益"] * 100.0, 2),
                "最大回撤(%)": round(wb.stats["最大回撤"] * 100.0, 2),
                "夏普比率": round(wb.stats["夏普比率"], 2),
            }
        )

    df_sl = pd.DataFrame(sl_results)
    print("\n>>> [2.1 止损梯度敏感性 (stop_loss)]")
    print(df_sl.to_string(index=False))

    # 2. 开仓间隔测试 (固定 stop_loss=300)
    intervals = [0, 3600 * 4, 3600 * 8, 3600 * 24]
    int_labels = ["0小时(无保护)", "4小时(半交易日)", "8小时(1个交易日)", "24小时(跨日保护)"]
    int_results = []
    for itv, label in zip(intervals, int_labels, strict=False):
        res = ParamStudyStrategy(
            symbol="510500.XSHG",
            name=f"ITV_{itv}",
            open_e=open_e,
            exit_e=exit_e,
            interval=itv,
            stop_loss=300,
        ).backtest(bars, sdt="2020-06-01")
        pairs = res.pairs_df()
        wb = WeightBacktest(holds_to_weight_df(res.holds_df()), fee_rate=0.0002, yearly_days=252)

        int_results.append(
            {
                "开仓最小间隔": label,
                "总交易笔数": len(pairs),
                "年化收益(%)": round(wb.stats["年化收益"] * 100.0, 2),
                "最大回撤(%)": round(wb.stats["最大回撤"] * 100.0, 2),
                "夏普比率": round(wb.stats["夏普比率"], 2),
                "卡玛比率": round(wb.stats["卡玛比率"], 2),
            }
        )

    df_itv = pd.DataFrame(int_results)
    print("\n>>> [2.2 开仓间隔保护敏感性 (interval)]")
    print(df_itv.to_string(index=False))

    return df_sl, df_itv


# ============================================================================
# 实验三：费率与滑点承压测试 (Friction Stress Test)
# ============================================================================


def run_friction_stress_test(bars: list) -> pd.DataFrame:
    """测试不同费率与滑点冲击下的策略存活与抗压极限。"""
    print("\n" + "=" * 70)
    print("【实验三】实盘费率与滑点承压测试 (Friction Stress Test)")
    print("=" * 70)

    open_e = Event.load(
        {
            "name": "开多",
            "operate": "开多",
            "signals_all": ["30分钟_D1SMA5#10#20#40#60#120#250_均线粘合发散V260520_看多_任意_任意_0"],
            "signals_not": ["30分钟_D1_涨跌停V230331_涨停_任意_任意_0"],
        }
    )
    exit_e = Event.load(
        {
            "name": "平多",
            "operate": "平多",
            "signals_all": ["30分钟_D1SMA5#10#20#40#60#120#250_均线粘合发散V260520_看空_任意_任意_0"],
            "signals_not": [],
        }
    )

    class BaselineStrat(CzscStrategyBase):
        @property
        def positions(self) -> list[Position]:
            return [
                Position(
                    symbol=self.symbol,
                    name="Base",
                    opens=[open_e],
                    exits=[exit_e],
                    interval=3600 * 4,
                    timeout=16 * 25,
                    stop_loss=300,
                    t0=False,
                )
            ]

    res = BaselineStrat(symbol="510500.XSHG").backtest(bars, sdt="2020-06-01")
    dfw = holds_to_weight_df(res.holds_df())

    fee_levels = [0.0, 0.0001, 0.0002, 0.0005, 0.0010]
    fee_labels = [
        "0 BP (理论免摩擦)",
        "1 BP (普通交易规费)",
        "2 BP (实盘基准含滑点)",
        "5 BP (高摩擦环境)",
        "10 BP (极端高冲击)",
    ]

    stress_results = []
    for fee, label in zip(fee_levels, fee_labels, strict=False):
        wb = WeightBacktest(dfw, fee_rate=fee, yearly_days=252)
        stress_results.append(
            {
                "摩擦测试档位": label,
                "单边费率": f"{fee * 10000:.0f} BP",
                "累计收益(%)": round(wb.stats["绝对收益"] * 100.0, 2),
                "年化收益(%)": round(wb.stats["年化收益"] * 100.0, 2),
                "夏普比率": round(wb.stats["夏普比率"], 2),
                "卡玛比率": round(wb.stats["卡玛比率"], 2),
                "最大回撤(%)": round(wb.stats["最大回撤"] * 100.0, 2),
            }
        )

    df_stress = pd.DataFrame(stress_results)
    print(df_stress.to_string(index=False))
    return df_stress


def main() -> None:
    print("=" * 70)
    print("  CZSC 原作者标准流程 —— 阶段五：边际优化与实盘摩擦治理")
    print("=" * 70)

    bars = load_real_bars()

    # 1. 入场边际过滤优化
    df_opens = run_opens_optimization(bars)

    # 2. 参数敏感性
    df_sl, df_itv = run_parameter_sensitivity(bars)

    # 3. 摩擦承压测试
    df_friction = run_friction_stress_test(bars)

    # 保存产物
    df_opens.to_csv(OUTPUT_DIR / "phase5_opens_optimization.csv", index=False, encoding="utf-8-sig")
    df_sl.to_csv(OUTPUT_DIR / "phase5_stoploss_sensitivity.csv", index=False, encoding="utf-8-sig")
    df_itv.to_csv(OUTPUT_DIR / "phase5_interval_sensitivity.csv", index=False, encoding="utf-8-sig")
    df_friction.to_csv(OUTPUT_DIR / "phase5_friction_stress.csv", index=False, encoding="utf-8-sig")

    summary_json = {
        "opens_optimization": df_opens.to_dict(orient="records"),
        "stop_loss_sensitivity": df_sl.to_dict(orient="records"),
        "interval_sensitivity": df_itv.to_dict(orient="records"),
        "friction_stress_test": df_friction.to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "phase5_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("[完成] 阶段五全部优化与敏感性测试完成！产物已落盘至：")
    print(f"  - {OUTPUT_DIR / 'phase5_opens_optimization.csv'}")
    print(f"  - {OUTPUT_DIR / 'phase5_stoploss_sensitivity.csv'}")
    print(f"  - {OUTPUT_DIR / 'phase5_interval_sensitivity.csv'}")
    print(f"  - {OUTPUT_DIR / 'phase5_friction_stress.csv'}")
    print(f"  - {OUTPUT_DIR / 'phase5_results.json'}")


if __name__ == "__main__":
    main()
