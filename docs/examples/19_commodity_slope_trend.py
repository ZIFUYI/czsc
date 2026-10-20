"""案例 19：商品期货线性回归斜率（SLOPE）多空双向趋势跟踪策略

策略逻辑：
1. 线性回归斜率计算：
   对日线过去 N 根 K 线的收盘价计算一元线性回归斜率（SLOPE）；
2. 拐点与阻力/支撑位动态捕捉：
   - 当斜率由负转正（下跌减缓并拐头），记录前 2 根 K 线的最高价最大值作为阻力位 HH；
   - 当斜率由正转负（上涨减缓并拐头），记录前 2 根 K 线的最低价最小值作为支撑位 LL；
3. 多空双向顺势突破与反手规则：
   - 突破多头：当根日线收盘价 Close > HH，若持有空单则平空并反手开多，若空仓则直接开多；
   - 跌破空头：当根日线收盘价 Close < LL，若持有多单则平多并反手开空，若空仓则直接开空；
4. 架构遵循 CZSC 开发宪法：Rust 核心算子（tas_slope_break_V261003）+ Python 策略门面（CzscStrategyBase）。

运行方式：
    uv run python docs/examples/19_commodity_slope_trend.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd  # noqa: E402
import pyarrow.ipc as ipc  # noqa: E402

from czsc import CzscStrategyBase, Event, Freq, Position, format_standard_kline  # noqa: E402
from czsc.mock import generate_symbol_kines  # noqa: E402


def _arrow_to_df(arrow_bytes: bytes) -> pd.DataFrame:
    """把 Arrow IPC 字节流转换为 pandas DataFrame。"""
    with ipc.open_file(arrow_bytes) as reader:
        return reader.read_all().to_pandas()


class CommoditySlopeTrendStrategy(CzscStrategyBase):
    """商品期货线性回归斜率多空双向趋势跟踪策略。"""

    def __init__(self, symbol: str, n: int = 20, **kwargs):
        super().__init__(symbol=symbol, n=n, **kwargs)
        self.n = n

    @property
    def positions(self) -> list[Position]:
        n = self.n

        # 1. 多头仓位：向上突破 HH 开多；向下跌破 LL 平多
        long_opens = [
            Event.load(
                {
                    "operate": "开多",
                    "signals_all": [f"日线_D1N{n}_斜率突破V261003_看多_向上突破_任意_0"],
                }
            )
        ]
        long_exits = [
            Event.load(
                {
                    "operate": "平多",
                    "signals_all": [f"日线_D1N{n}_斜率突破V261003_看空_任意_任意_0"],
                }
            )
        ]
        pos_long = Position(
            name="日线多头",
            symbol=self.symbol,
            opens=long_opens,
            exits=long_exits,
            interval=0,
            timeout=0,
            stop_loss=0,
        )

        # 2. 空头仓位：向下跌破 LL 开空；向上突破 HH 平空
        short_opens = [
            Event.load(
                {
                    "operate": "开空",
                    "signals_all": [f"日线_D1N{n}_斜率突破V261003_看空_向下跌破_任意_0"],
                }
            )
        ]
        short_exits = [
            Event.load(
                {
                    "operate": "平空",
                    "signals_all": [f"日线_D1N{n}_斜率突破V261003_看多_任意_任意_0"],
                }
            )
        ]
        pos_short = Position(
            name="日线空头",
            symbol=self.symbol,
            opens=short_opens,
            exits=short_exits,
            interval=0,
            timeout=0,
            stop_loss=0,
        )

        return [pos_long, pos_short]


def run_commodity_backtest(symbol: str = "RB9999", n: int = 20, seed: int = 42) -> None:
    print("==================================================")
    print(f" 开始回测商品期货趋势策略: {symbol} (SLOPE LENGTH = {n})")
    print("==================================================")

    # 1. 模拟生成 5 年日线行情数据（2019-2024）
    df = generate_symbol_kines(symbol, "日线", "20190101", "20240101", seed=seed)
    bars = format_standard_kline(df, freq=Freq.D)
    print(f"数据加载完成: 共 {len(bars)} 根日线 K 线")

    # 2. 实例化策略
    strategy = CommoditySlopeTrendStrategy(symbol=symbol, n=n)
    print(f"策略派生信号配置: {strategy.signals_config}")

    # 3. 执行内存回测（预热到 2020-01-01 开始交易）
    res = strategy.backtest(bars, sdt="2020-01-01")
    pairs = _arrow_to_df(res.pairs_arrow)

    if pairs.empty:
        print("未产生任何开平仓交易！")
        return

    print(f"\n【回测交易表现总览】(共 {len(pairs)} 笔交易)")
    print("-" * 75)

    # 统计汇总
    summary = []
    for direction in ["多头", "空头"]:
        sub = pairs[pairs["交易方向"] == direction]
        if sub.empty:
            continue
        # 盈亏比例在底层以 BP 为基准（10000 BP = 100%）
        ret_series = sub["盈亏比例"] / 10000.0
        win_trades = ret_series[ret_series > 0]
        loss_trades = ret_series[ret_series < 0]

        win_rate = len(win_trades) / len(sub) if len(sub) > 0 else 0.0
        avg_win = win_trades.mean() if not win_trades.empty else 0.0
        avg_loss = abs(loss_trades.mean()) if not loss_trades.empty else 0.0
        pnl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        total_pnl = ret_series.sum()
        avg_hold_days = sub["持仓天数"].mean()

        summary.append(
            {
                "交易方向": direction,
                "总交易次数": len(sub),
                "胜率": f"{win_rate:.2%}",
                "盈亏比": f"{pnl_ratio:.2f}",
                "单笔平均收益": f"{ret_series.mean():.2%}",
                "累计名义收益": f"{total_pnl:.2%}",
                "平均持仓天数": f"{avg_hold_days:.1f} 天",
            }
        )

    sum_df = pd.DataFrame(summary)
    print(sum_df.to_string(index=False))
    print("-" * 75)

    print("\n【最新 5 笔交易样本明细】")
    display_cols = ["交易方向", "开仓时间", "平仓时间", "开仓价格", "平仓价格", "持仓天数", "盈亏比例"]
    recent_pairs = pairs[display_cols].tail(5).copy()
    recent_pairs["开仓时间"] = pd.to_datetime(recent_pairs["开仓时间"]).dt.strftime("%Y-%m-%d")
    recent_pairs["平仓时间"] = pd.to_datetime(recent_pairs["平仓时间"]).dt.strftime("%Y-%m-%d")
    recent_pairs["盈亏比例"] = (recent_pairs["盈亏比例"] / 100.0).map("{:+.2f}%".format)
    print(recent_pairs.to_string(index=False))
    print("-" * 75)


if __name__ == "__main__":
    run_commodity_backtest(symbol="RB9999", n=20, seed=42)
