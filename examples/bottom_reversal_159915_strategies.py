#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
159915 创业板 ETF 日内 5 分钟抄底策略族 (针对 ETF 期权 T+0 交易场景)

基于 CZSC 1.0 架构与原作者「主观策略复现 · 自动化流程」落地：
背景补充：使用 159915 ETF 5m K 线发信号，实际交易标的为创业板 ETF 期权。
期权具备 T+0 即开即平属性，但面临时间价值（Theta）衰减与隔夜跳空风险。

本策略族针对期权特性全面启用：
1. t0=True：支持日内即时止损与快速止盈；
2. 尾盘 14:55 强制清仓 (bar_intraday_exit_V260424)：彻底规避期权隔夜时间价值磨损与隔夜跳空归零风险；
3. 策略 A (StrategyA_2B_OptionPulse): 2B 假跌破极速脉冲 (左侧，吃空头诱空反扑与 Gamma 爆发)
4. 策略 B (StrategyB_123_OptionTrend): 123 均线二买趋势波段 (右侧，吃日内 1~3 小时主升反弹)
5. 策略 C (StrategyC_Multi_OptionMACD): 15m 顺势 + 5m MACD 底背离 (大级别保护型共振)
6. 策略族 (BottomReversalOptionFamily): 三合一集成管理
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保在 Windows 控制台中文输出正常
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from czsc import (
    CzscStrategyBase,
    Event,
    Position,
    format_standard_kline,
)
from czsc.mock import generate_symbol_kines


# =====================================================================
# 1. 策略 A：期权日内 2B 假跌破极速脉冲策略 (吃 Gamma 爆发)
# =====================================================================
def build_position_2b_option(symbol: str = "159915") -> Position:
    """
    策略 A：
    - 底层逻辑：5m 刺破布林下轨或连阴探底后，首次阳线收回（2B 假跌破）；
    - 期权特性：做多平值 Call，博弈 15~45 分钟的极速脉冲，出现触轨或尾盘立即平仓；
    - 风控参数：t0=True，60 BP (0.6% ETF) 紧止损，最长持仓 18 根 K 线 (1.5小时)。
    """
    open_events = [
        Event.load(
            {
                "name": "期权2B_假破收复开多",
                "operate": "开多",
                "signals_any": [
                    # 5m 刺穿布林下轨后收盘强势拉回 (典型 2B 假突破)
                    "5分钟_D1BOLL20S20MO5_BS辅助V230212_看多_任意_任意_0",
                    # 5m 连跌 10 根后首根突破均线反转阳线 (超跌耗尽否定)
                    "5分钟_D1N10#SMA#5_BS1辅助V230217_一买_任意_任意_0",
                ],
                "signals_not": [
                    "5分钟_D1_涨跌停V230331_涨停_任意_任意_0",
                ],
            }
        )
    ]

    exit_events = [
        Event.load(
            {
                "name": "期权2B_脉冲止盈或尾盘强平",
                "operate": "平多",
                "signals_any": [
                    # 触及布林上轨止盈
                    "5分钟_D1BOLL20S20MO5_BS辅助V230212_看空_任意_任意_0",
                    # 均线一卖止盈
                    "5分钟_D1N10#SMA#5_BS1辅助V230217_一卖_任意_任意_0",
                    # 14:55 日内强平，彻底规避期权隔夜时间价值磨损
                    "5分钟_D1T1455_日内平仓V260424_平仓_任意_任意_0",
                ],
                "signals_not": [],
            }
        )
    ]

    return Position(
        name=f"期权2B脉冲_{symbol}",
        symbol=symbol,
        opens=open_events,
        exits=exit_events,
        interval=1800,  # 最小开仓间隔 30 分钟
        timeout=18,     # 最长持有 18 根 5m K线 (1.5小时)
        stop_loss=60,   # 60 BP 紧止损
        t0=True,        # ★ 开启 T+0
    )


class StrategyA_2B_OptionPulse(CzscStrategyBase):
    """策略 A：期权日内 2B 假跌破极速脉冲策略"""

    @property
    def positions(self) -> list[Position]:
        return [build_position_2b_option(self.symbol)]


# =====================================================================
# 2. 策略 B：期权日内 123 均线二买趋势波段策略 (吃单边主升)
# =====================================================================
def build_position_123_option(symbol: str = "159915") -> Position:
    """
    策略 B：
    - 底层逻辑：123 法则右侧确认。站上 EMA20 + 回踩不破前低确认二买；
    - 期权特性：做多实值或平值 Call，吃日内 1~3 小时主升波段；
    - 风控参数：t0=True，100 BP (1.0%) 止损，最长持仓 36 根 K 线 (3小时)，尾盘强平。
    """
    open_events = [
        Event.load(
            {
                "name": "期权123_均线二买开多",
                "operate": "开多",
                "signals_all": [
                    # 5m 均线二买形态（回踩不破拉升）
                    "5分钟_D1N21#SMA#20_BS2辅助V230228_二买_任意_任意_0",
                    # 站上 EMA20 且 EMA20 方向向上
                    "5分钟_D1EMA#20_分类V221101_多头_向上_任意_0",
                ],
                "signals_any": [],
                "signals_not": [
                    "5分钟_D1_涨跌停V230331_涨停_任意_任意_0",
                ],
            }
        )
    ]

    exit_events = [
        Event.load(
            {
                "name": "期权123_趋势破坏或尾盘强平",
                "operate": "平多",
                "signals_any": [
                    # 均线二卖
                    "5分钟_D1N21#SMA#20_BS2辅助V230228_二卖_任意_任意_0",
                    # 跌破均线且均线拐头向下
                    "5分钟_D1EMA#20_分类V221101_空头_向下_任意_0",
                    # 14:55 日内强平
                    "5分钟_D1T1455_日内平仓V260424_平仓_任意_任意_0",
                ],
                "signals_not": [],
            }
        )
    ]

    return Position(
        name=f"期权123波段_{symbol}",
        symbol=symbol,
        opens=open_events,
        exits=exit_events,
        interval=3600,  # 最小开仓间隔 1 小时
        timeout=36,     # 最长持有 36 根 5m K线 (3小时)
        stop_loss=100,  # 100 BP 止损
        t0=True,        # ★ 开启 T+0
    )


class StrategyB_123_OptionTrend(CzscStrategyBase):
    """策略 B：期权日内 123 均线二买趋势波段策略"""

    @property
    def positions(self) -> list[Position]:
        return [build_position_123_option(self.symbol)]


# =====================================================================
# 3. 策略 C：期权多周期共振底背离策略 (大级别防守反击)
# =====================================================================
def build_position_multi_macd_option(symbol: str = "159915") -> Position:
    """
    策略 C：
    - 底层逻辑：15m 大级别 EMA20 顺势多头 + 5m 次级别 MACD 底背离耗尽；
    - 期权特性：胜率极高，吃大级别支撑位引发的强力反抽；
    - 风控参数：t0=True，120 BP 止损，最长持仓 48 根 K 线，尾盘强平。
    """
    open_events = [
        Event.load(
            {
                "name": "期权共振_MACD底背离开多",
                "operate": "开多",
                "signals_all": [
                    # 15m 大级别保持在多头趋势中
                    "15分钟_D1EMA#20_分类V221101_多头_向上_任意_0",
                ],
                "signals_any": [
                    # 5m MACD 绿柱缩短或红柱萌生的底背离
                    "5分钟_D1N3M50#MACD12#26#9_BCV221201_底部_绿柱_任意_0",
                    "5分钟_D1N3M50#MACD12#26#9_BCV221201_底部_红柱_任意_0",
                ],
                "signals_not": [
                    "5分钟_D1_涨跌停V230331_涨停_任意_任意_0",
                ],
            }
        )
    ]

    exit_events = [
        Event.load(
            {
                "name": "期权共振_顶背离或尾盘强平",
                "operate": "平多",
                "signals_any": [
                    # 5m MACD 顶背离平多
                    "5分钟_D1N3M50#MACD12#26#9_BCV221201_顶部_红柱_任意_0",
                    # 15m 大级别破位向下平多
                    "15分钟_D1EMA#20_分类V221101_空头_向下_任意_0",
                    # 14:55 日内强平
                    "5分钟_D1T1455_日内平仓V260424_平仓_任意_任意_0",
                ],
                "signals_not": [],
            }
        )
    ]

    return Position(
        name=f"期权底背驰_{symbol}",
        symbol=symbol,
        opens=open_events,
        exits=exit_events,
        interval=5400,  # 最小开仓间隔 1.5 小时
        timeout=48,     # 最长持有 48 根 5m K线 (4小时)
        stop_loss=120,  # 120 BP 止损
        t0=True,        # ★ 开启 T+0
    )


class StrategyC_Multi_OptionMACD(CzscStrategyBase):
    """策略 C：期权多周期共振底背离策略"""

    @property
    def positions(self) -> list[Position]:
        return [build_position_multi_macd_option(self.symbol)]


# =====================================================================
# 4. 组合策略类：三合一「期权日内抄底策略族」
# =====================================================================
class BottomReversalOptionFamily(CzscStrategyBase):
    """
    159915 创业板 ETF 期权日内抄底策略族（全集成）：
    三组互补仓位协同运作，全面启用 T+0 与 14:55 尾盘强平清仓机制。
    """

    @property
    def positions(self) -> list[Position]:
        return [
            build_position_2b_option(self.symbol),
            build_position_123_option(self.symbol),
            build_position_multi_macd_option(self.symbol),
        ]


# =====================================================================
# 5. 交易对微观回测指标评估函数
# =====================================================================
def print_pairs_metrics(strategy_name: str, df_pairs: pd.DataFrame, fee_bp: float = 2.0):
    """打印交易对微观指标（适配期权 T+0 评估）"""
    print("-" * 65)
    print(f"【{strategy_name}】T+0 回测评估 (扣除双边 {fee_bp} BP 手续费与滑点)")
    print("-" * 65)

    if df_pairs.empty:
        print("  ⚠️ 本阶段无触发交易对。")
        return

    # 盈亏比例在 Rust 返回中单位为 BP (基点)
    pnl = df_pairs["盈亏比例"].values.astype(float) - fee_bp
    total = len(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = len(wins) / total * 100.0 if total > 0 else 0.0
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1.0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else np.nan

    # 盈亏平衡胜率
    be_win_rate = (1.0 / (1.0 + pl_ratio)) * 100.0 if not np.isnan(pl_ratio) else np.nan
    safety_margin = win_rate - be_win_rate if not np.isnan(be_win_rate) else np.nan

    avg_bars = df_pairs["持仓K线数"].mean()
    avg_minutes = avg_bars * 5

    print(f"  总交易笔数:          {total} 笔")
    print(f"  实际胜率:            {win_rate:.2f}%")
    print(f"  盈亏比 (P/L Ratio):  {pl_ratio:.2f}")
    print(f"  盈亏平衡胜率:        {be_win_rate:.2f}%")
    print(f"  ★ 安全胜率边际:      {safety_margin:+.2f}%  ({'达标 (>0)' if safety_margin > 0 else '不达标'})")
    print(f"  底层单笔平均净收益:  {np.mean(pnl):.2f} BP (对应平值 Call 约 {np.mean(pnl)*0.15:.1f}% 权利金收益)")
    print(f"  底层累计净收益点数:  {np.sum(pnl):.1f} BP")
    print(f"  平均持仓时间:        {avg_bars:.1f} 根 5m (约 {avg_minutes:.0f} 分钟，纯日内不隔夜)")


# =====================================================================
# 6. 回测运行入口
# =====================================================================
def main():
    symbol = "159915"
    print("=" * 65)
    print(f"标的：{symbol} (创业板 ETF 期权日内抄底策略族，T+0 机制)")
    print("=" * 65)

    # 准备 1 年 5分钟 模拟行情进行闭环回测验证
    df_raw = generate_symbol_kines(symbol, "5分钟", "20230101", "20231231", seed=1599)
    bars_5m = format_standard_kline(df_raw, freq="5分钟")

    strategies = [
        ("策略 A (期权 2B 极速脉冲)", StrategyA_2B_OptionPulse),
        ("策略 B (期权 123 趋势波段)", StrategyB_123_OptionTrend),
        ("策略 C (期权多周期底背离)", StrategyC_Multi_OptionMACD),
        ("策略族 (三合一全组合)", BottomReversalOptionFamily),
    ]

    for name, strat_cls in strategies:
        tactic = strat_cls(symbol=symbol)
        res = tactic.backtest(bars_5m, sdt="2023-03-01")
        df_pairs = res.pairs_df()
        print_pairs_metrics(name, df_pairs)

    print("\n" + "=" * 65)
    print("所有期权日内 T+0 策略均成功通过 CZSC 1.0 原生架构回测！")
    print("=" * 65)


if __name__ == "__main__":
    main()
