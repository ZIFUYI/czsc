#!/usr/bin/env python
# ruff: noqa: SIM102, E741, F841
"""
6×5 生态位矩阵候选信号生成器与分簇定制状态机引擎
(MAP-Elites Niche Hypothesis Bank & Archetype-Specific State Machine Engine)

六大正交战略簇 (Archetypes, 按底层收益生成机制 DGP 划分) × 五级独立周期 (1d, 60m, 30m, 15m, 5m)：
1. A_趋势突破与斜率动量簇 (exit_mode="trend")     中枢脱离/通道突破/斜率加速/日内 ORB·DualThrust 顺势突破
2. B_背驰衰竭与极值反转簇 (exit_mode="reversal")  背驰/2B 假突破/TD 九转·十三转/急杀洗盘 V 反
3. C_中枢震荡与锚定回归簇 (exit_mode="range")     对称中枢/BOLL·KC 箱体/日内 VWAP 锚定均值回归
4. D_量价微结构与筹码冲击簇 (连续强度因子, 见 examples/multi_factor_cta/atomic_factors.py)
5. E_波动率压缩与分形弹性簇 (连续强度因子, 同上)
6. F_多周期共振与截面强弱簇 (连续强度因子, 同上; 预留跨品种截面接口, 兼容 Issue #6)

“日内/隔夜”属于交易制度 (由 ExecutionMode 与周期纵轴控制)，不再作为战略簇横轴；
原 D_日内微结构簇已按其实质机制拆入 A/B/C，见 `LEGACY_ARCHETYPE_ALIAS` / `migrate_archetype`。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

import czsc

ARCH_A = "A_趋势突破与斜率动量簇"
ARCH_B = "B_背驰衰竭与极值反转簇"
ARCH_C = "C_中枢震荡与锚定回归簇"
ARCH_D = "D_量价微结构与筹码冲击簇"
ARCH_E = "E_波动率压缩与分形弹性簇"
ARCH_F = "F_多周期共振与截面强弱簇"
ALL_ARCHETYPES: tuple[str, ...] = (ARCH_A, ARCH_B, ARCH_C, ARCH_D, ARCH_E, ARCH_F)

# 旧四簇名 -> 新簇名。旧 D_日内微结构簇 无法一对一映射 (其成员按机制拆入 A/B/C)，
# 需要借助 sig_id 查表，见 `migrate_archetype`。
LEGACY_ARCHETYPE_ALIAS: dict[str, str] = {
    "A_顺势突破簇": ARCH_A,
    "B_背驰反转簇": ARCH_B,
    "C_震荡回归簇": ARCH_C,
}
LEGACY_INTRADAY_ARCHETYPE = "D_日内微结构簇"


@dataclass(frozen=True)
class ArchetypeRuler:
    """分方向定制的法医三重屏障检验标尺与分层池化协议"""

    archetype: str
    ruler_name: str
    tp_atr: float
    sl_atr: float
    max_bars: int
    min_alpha_bp: float
    min_safety_pct: float
    max_boot_pval: float
    quota_max_units: int  # 该战略方向最多允许新入库多少个原子单元 (防单一方向扎堆)
    min_excursion_ratio: float = 0.90  # 最小 MFE/MAE 路径偏离比门槛 (淘汰扛深亏换微利信号)
    min_net_sharpe: float = 0.0  # 单因子扣费后最小样本内 Net Sharpe
    pool_tier: int = 1  # 多标的分层池化检验层级 (1=跨资产普适, 2=同质资产类别, 3=交易制度微结构)
    stop_plugin: str = "pos_atr_stop"  # 绑定的动态结构止损算子标识


ARCHETYPE_RULERS: dict[str, ArchetypeRuler] = {
    ARCH_A: ArchetypeRuler(
        archetype=ARCH_A,
        ruler_name="趋势标尺 (+2.2*ATR止盈 / -1.1*ATR止损 / 16根)",
        tp_atr=2.20,
        sl_atr=1.10,
        max_bars=16,
        min_alpha_bp=12.0,
        min_safety_pct=4.5,
        max_boot_pval=0.20,
        quota_max_units=10,  # 基石 5 个 + 接管原 D 簇 ORB/DualThrust/RBreaker 顺势突破信号
        min_excursion_ratio=1.15,
        min_net_sharpe=0.0,
        pool_tier=1,
        stop_plugin="pos_fx_stop",
    ),
    ARCH_B: ArchetypeRuler(
        archetype=ARCH_B,
        ruler_name="反转标尺 (+1.35*ATR止盈 / -0.95*ATR止损 / 8根快进快出)",
        tp_atr=1.35,
        sl_atr=0.95,
        max_bars=8,
        min_alpha_bp=8.0,
        min_safety_pct=3.5,
        max_boot_pval=0.22,
        quota_max_units=12,  # 接管原 D 簇洗盘 V 反与 TD 序列衰竭反转
        min_excursion_ratio=0.90,
        min_net_sharpe=0.0,
        pool_tier=2,
        stop_plugin="pos_structure_ext",
    ),
    ARCH_C: ArchetypeRuler(
        archetype=ARCH_C,
        ruler_name="回归标尺 (+1.15*ATR止盈 / -0.85*ATR止损 / 6根均值回归)",
        tp_atr=1.15,
        sl_atr=0.85,
        max_bars=6,
        min_alpha_bp=6.0,
        min_safety_pct=3.0,
        max_boot_pval=0.24,
        quota_max_units=9,  # 接管原 D 簇日内 VWAP 锚定回归
        min_excursion_ratio=0.90,
        min_net_sharpe=0.0,
        pool_tier=2,
        stop_plugin="pos_zs_mid",
    ),
    ARCH_D: ArchetypeRuler(
        archetype=ARCH_D,
        ruler_name="量价标尺 (+1.3*ATR止盈 / -0.85*ATR止损 / 8根资金流冲击)",
        tp_atr=1.30,
        sl_atr=0.85,
        max_bars=8,
        min_alpha_bp=8.0,
        min_safety_pct=3.5,
        max_boot_pval=0.22,
        quota_max_units=8,
        min_excursion_ratio=0.90,
        min_net_sharpe=0.0,
        pool_tier=3,
        stop_plugin="pos_flow_stop",
    ),
    ARCH_E: ArchetypeRuler(
        archetype=ARCH_E,
        ruler_name="波动率扩张标尺 (+2.0*ATR止盈 / -0.7*ATR箱体极值止损 / 12根压缩点火)",
        tp_atr=2.00,
        sl_atr=0.70,
        max_bars=12,
        min_alpha_bp=10.0,
        min_safety_pct=4.0,
        max_boot_pval=0.22,
        quota_max_units=8,
        min_excursion_ratio=1.15,
        min_net_sharpe=0.0,
        pool_tier=1,
        stop_plugin="pos_box_stop",
    ),
    ARCH_F: ArchetypeRuler(
        archetype=ARCH_F,
        ruler_name="共振标尺 (+1.8*ATR止盈 / -1.0*ATR止损 / 12根大级别锚定)",
        tp_atr=1.80,
        sl_atr=1.00,
        max_bars=12,
        min_alpha_bp=8.0,
        min_safety_pct=3.5,
        max_boot_pval=0.22,
        quota_max_units=6,
        min_excursion_ratio=1.00,
        min_net_sharpe=0.0,
        pool_tier=1,
        stop_plugin="pos_htf_ma_stop",
    ),
}


@dataclass(frozen=True)
class SignalSpec:
    sig_id: str
    name: str
    archetype: str
    direction: int  # 1 = 多, -1 = 空
    default_weight: float
    eligible_tfs: tuple[str, ...]
    exit_mode: str  # "trend" | "reversal" | "range" | "continuous" (连续强度因子，见 atomic_factors.py)
    is_base_seed: bool = False


# ---------------------------------------------------------------------------
# 1. 基石真金信号 (5个) + 四大战略方向候选池 (Round 1 + Round 2 + Round 3 + Round 4 + Round 5 TD序列)
# ---------------------------------------------------------------------------
BASE_SEED_SIGNALS: list[SignalSpec] = [
    SignalSpec(
        "S1L", "S1L_中枢三买确认", "A_趋势突破与斜率动量簇", 1, 0.40, ("1d", "60m", "30m"), "trend", is_base_seed=True
    ),
    SignalSpec(
        "S1S", "S1S_中枢三卖确认", "A_趋势突破与斜率动量簇", -1, 0.50, ("1d", "60m", "30m"), "trend", is_base_seed=True
    ),
    SignalSpec(
        "S2L",
        "S2L_均线粘合一阳穿线",
        "A_趋势突破与斜率动量簇",
        1,
        0.35,
        ("1d", "60m", "30m"),
        "trend",
        is_base_seed=True,
    ),
    SignalSpec(
        "S3L", "S3L_内孕线放量突破", "A_趋势突破与斜率动量簇", 1, 0.30, ("1d", "60m", "30m"), "trend", is_base_seed=True
    ),
    SignalSpec(
        "S4S",
        "S4S_VCP压缩空头爆发",
        "A_趋势突破与斜率动量簇",
        -1,
        0.50,
        ("1d", "60m", "30m"),
        "trend",
        is_base_seed=True,
    ),
]

CANDIDATE_HYPOTHESIS_POOL: list[SignalSpec] = [
    # === 方向 B：背驰反转簇 (Round 1 候选, 适用 60m / 30m / 15m) ===
    SignalSpec("B01L", "B01L_MACD底背驰放量反包", "B_背驰衰竭与极值反转簇", 1, 0.28, ("60m", "30m", "15m"), "reversal"),
    SignalSpec(
        "B01S", "B01S_MACD顶背驰放量反杀", "B_背驰衰竭与极值反转簇", -1, 0.28, ("60m", "30m", "15m"), "reversal"
    ),
    SignalSpec(
        "B02L", "B02L_恐慌长下影锤子确认反转", "B_背驰衰竭与极值反转簇", 1, 0.26, ("60m", "30m", "15m"), "reversal"
    ),
    SignalSpec(
        "B02S", "B02S_亢奋长上影流星确认反杀", "B_背驰衰竭与极值反转簇", -1, 0.26, ("60m", "30m", "15m"), "reversal"
    ),
    SignalSpec(
        "B03L", "B03L_超跌底分型停顿放量反抽", "B_背驰衰竭与极值反转簇", 1, 0.26, ("60m", "30m", "15m"), "reversal"
    ),
    SignalSpec(
        "B03S", "B03S_超涨顶分型停顿放量回落", "B_背驰衰竭与极值反转簇", -1, 0.26, ("60m", "30m", "15m"), "reversal"
    ),
    # === 方向 C：震荡回归簇 (Round 1 候选, 适用 60m / 30m / 15m) ===
    SignalSpec("C01L", "C01L_震荡市布林下轨2B收回", "C_中枢震荡与锚定回归簇", 1, 0.26, ("60m", "30m", "15m"), "range"),
    SignalSpec("C01S", "C01S_震荡市布林上轨2B跌回", "C_中枢震荡与锚定回归簇", -1, 0.26, ("60m", "30m", "15m"), "range"),
    SignalSpec(
        "C02L", "C02L_宽幅箱体RSI超卖勾头反包", "C_中枢震荡与锚定回归簇", 1, 0.26, ("60m", "30m", "15m"), "range"
    ),
    SignalSpec(
        "C02S", "C02S_宽幅箱体RSI超买勾头反杀", "C_中枢震荡与锚定回归簇", -1, 0.26, ("60m", "30m", "15m"), "range"
    ),
    SignalSpec(
        "C03L", "C03L_Keltner下轨企稳回归中轨", "C_中枢震荡与锚定回归簇", 1, 0.26, ("60m", "30m", "15m"), "range"
    ),
    SignalSpec(
        "C03S", "C03S_Keltner上轨受阻回归中轨", "C_中枢震荡与锚定回归簇", -1, 0.26, ("60m", "30m", "15m"), "range"
    ),
    # === 方向 D：日内分时微结构簇 (Round 1 候选, 适用 15m / 30m) ===
    SignalSpec("D01L", "D01L_早盘ORB首区放量上破", "A_趋势突破与斜率动量簇", 1, 0.26, ("15m", "30m"), "trend"),
    SignalSpec("D01S", "D01S_早盘ORB首区放量下破", "A_趋势突破与斜率动量簇", -1, 0.26, ("15m", "30m"), "trend"),
    SignalSpec("D02L", "D02L_日内VWAP回踩不破阳包", "C_中枢震荡与锚定回归簇", 1, 0.26, ("15m", "30m"), "range"),
    SignalSpec("D02S", "D02S_日内VWAP反抽受阻阴杀", "C_中枢震荡与锚定回归簇", -1, 0.26, ("15m", "30m"), "range"),
    SignalSpec("D03L", "D03L_午盘异动放量突破早盘高点", "A_趋势突破与斜率动量簇", 1, 0.26, ("15m", "30m"), "trend"),
    SignalSpec("D03S", "D03S_午盘异动放量跌破早盘低点", "A_趋势突破与斜率动量簇", -1, 0.26, ("15m", "30m"), "trend"),
    # === 方向 A：顺势突破簇补充候选 (适用 1d / 60m / 30m) ===
    SignalSpec(
        "A01L", "A01L_OBV能量潮领先新高突破", "D_量价微结构与筹码冲击簇", 1, 0.28, ("1d", "60m", "30m"), "trend"
    ),
    SignalSpec(
        "A01S", "A01S_OBV能量潮领先新低破位", "D_量价微结构与筹码冲击簇", -1, 0.28, ("1d", "60m", "30m"), "trend"
    ),
    SignalSpec("A02L", "A02L_Keltner挤压向上点火", "E_波动率压缩与分形弹性簇", 1, 0.30, ("1d", "60m", "30m"), "trend"),
    SignalSpec("A02S", "A02S_Keltner挤压向下破位", "E_波动率压缩与分形弹性簇", -1, 0.32, ("1d", "60m", "30m"), "trend"),
    SignalSpec(
        "A03L", "A03L_缩量星线后放量长阳破局", "E_波动率压缩与分形弹性簇", 1, 0.28, ("1d", "60m", "30m"), "trend"
    ),
    SignalSpec(
        "A03S", "A03S_缩量星线后放量长阴破位", "E_波动率压缩与分形弹性簇", -1, 0.30, ("1d", "60m", "30m"), "trend"
    ),
    SignalSpec(
        "A04L", "A04L_宽幅期唐奇安10期放量突破", "A_趋势突破与斜率动量簇", 1, 0.30, ("1d", "60m", "30m"), "trend"
    ),
    SignalSpec(
        "A04S", "A04S_宽幅期唐奇安10期放量破位", "A_趋势突破与斜率动量簇", -1, 0.30, ("1d", "60m", "30m"), "trend"
    ),
    # =========================================================================
    # === Round 2 二期定向深挖候选池 (聚焦 B/C/D 空白生态位与空头反转/震荡回归/日内共振) ===
    # =========================================================================
    # --- Round 2 方向 B：双底/双顶结构与假突破 2B 反转 (覆盖 1d / 60m / 30m / 15m) ---
    SignalSpec(
        "B04L",
        "B04L_W双底缩量二次回踩不破反包",
        "B_背驰衰竭与极值反转簇",
        1,
        0.26,
        ("1d", "60m", "30m", "15m"),
        "reversal",
    ),
    SignalSpec(
        "B04S",
        "B04S_M双顶二次冲高受阻放量反杀",
        "B_背驰衰竭与极值反转簇",
        -1,
        0.26,
        ("1d", "60m", "30m", "15m"),
        "reversal",
    ),
    SignalSpec(
        "B05L",
        "B05L_极度超跌三连阴缩量止跌阳包",
        "B_背驰衰竭与极值反转簇",
        1,
        0.26,
        ("1d", "60m", "30m", "15m"),
        "reversal",
    ),
    SignalSpec(
        "B05S",
        "B05S_高位假突破2B放量阴吞没反杀",
        "B_背驰衰竭与极值反转簇",
        -1,
        0.26,
        ("1d", "60m", "30m", "15m"),
        "reversal",
    ),
    # --- Round 2 方向 C：箱体双针探底/受阻与通道十字星均值回归 (覆盖 1d / 60m / 30m / 15m) ---
    SignalSpec(
        "C04L", "C04L_箱体下沿缩量双针探底回归", "C_中枢震荡与锚定回归簇", 1, 0.26, ("1d", "60m", "30m", "15m"), "range"
    ),
    SignalSpec(
        "C04S",
        "C04S_弱势箱体反抽Keltner上中轨受阻",
        "C_中枢震荡与锚定回归簇",
        -1,
        0.26,
        ("1d", "60m", "30m", "15m"),
        "range",
    ),
    SignalSpec(
        "C05L",
        "C05L_通道下轨缩量十字星次根阳线回归",
        "C_中枢震荡与锚定回归簇",
        1,
        0.26,
        ("1d", "60m", "30m", "15m"),
        "range",
    ),
    SignalSpec(
        "C05S",
        "C05S_通道上轨缩量十字星次根阴线回归",
        "C_中枢震荡与锚定回归簇",
        -1,
        0.26,
        ("1d", "60m", "30m", "15m"),
        "range",
    ),
    # --- Round 2 方向 D：日内分时洗盘 V 反、诱多跌破 VWAP 与午后企稳脉冲 (覆盖 15m / 30m / 60m) ---
    SignalSpec(
        "D04L", "D04L_早盘急杀洗盘V型放量站回VWAP", "B_背驰衰竭与极值反转簇", 1, 0.26, ("15m", "30m", "60m"), "reversal"
    ),
    SignalSpec(
        "D04S", "D04S_早盘冲高诱多放量跌破VWAP", "B_背驰衰竭与极值反转簇", -1, 0.26, ("15m", "30m", "60m"), "reversal"
    ),
    SignalSpec(
        "D05L", "D05L_午后回踩VWAP缩量企稳拉升", "C_中枢震荡与锚定回归簇", 1, 0.26, ("15m", "30m", "60m"), "range"
    ),
    SignalSpec(
        "D05S", "D05S_午后反抽VWAP无力放量杀跌", "C_中枢震荡与锚定回归簇", -1, 0.26, ("15m", "30m", "60m"), "range"
    ),
    # =========================================================================
    # === Round 3 三期原生 Rust `czsc-signals` 组合挖掘池 (直连 cxt/byi/jcc/bar/obv/vol) ===
    # =========================================================================
    # --- Round 3 方向 A：原生缠论五笔/七笔突破与三买卖辅助 + 通道/OBV/加速 (覆盖 1d / 60m / 30m / 15m) ---
    SignalSpec(
        "CZ01L",
        "CZ01L_缠论五笔颈线突破放量共振",
        "A_趋势突破与斜率动量簇",
        1,
        0.28,
        ("1d", "60m", "30m", "15m"),
        "trend",
    ),
    SignalSpec(
        "CZ01S",
        "CZ01S_缠论五笔下颈线破位通道共振",
        "A_趋势突破与斜率动量簇",
        -1,
        0.28,
        ("1d", "60m", "30m", "15m"),
        "trend",
    ),
    SignalSpec(
        "CZ02L",
        "CZ02L_缠论三买均线新高加速突破",
        "A_趋势突破与斜率动量簇",
        1,
        0.28,
        ("1d", "60m", "30m", "15m"),
        "trend",
    ),
    SignalSpec(
        "CZ02S",
        "CZ02S_缠论三卖均线新低加速破位",
        "A_趋势突破与斜率动量簇",
        -1,
        0.28,
        ("1d", "60m", "30m", "15m"),
        "trend",
    ),
    # --- Round 3 方向 B：原生缠论一二买卖/双中枢背驰 + 分型停顿/验证分型/吞没 (覆盖 1d / 60m / 30m / 15m) ---
    SignalSpec(
        "CZ03L",
        "CZ03L_缠论一买双中枢底背驰停顿反转",
        "B_背驰衰竭与极值反转簇",
        1,
        0.26,
        ("1d", "60m", "30m", "15m"),
        "reversal",
    ),
    SignalSpec(
        "CZ03S",
        "CZ03S_缠论一卖双中枢顶背驰停顿反杀",
        "B_背驰衰竭与极值反转簇",
        -1,
        0.26,
        ("1d", "60m", "30m", "15m"),
        "reversal",
    ),
    SignalSpec(
        "CZ04L",
        "CZ04L_缠论二买MACD回抽零轴锤子确认",
        "B_背驰衰竭与极值反转簇",
        1,
        0.26,
        ("1d", "60m", "30m", "15m"),
        "reversal",
    ),
    SignalSpec(
        "CZ04S",
        "CZ04S_缠论二卖MACD回抽零轴上吊反杀",
        "B_背驰衰竭与极值反转簇",
        -1,
        0.26,
        ("1d", "60m", "30m", "15m"),
        "reversal",
    ),
    # --- Round 3 方向 C：原生对称中枢 + 假突破/神奇九转 TD9 + 酒田战法分型强弱 (覆盖 1d / 60m / 30m / 15m) ---
    SignalSpec(
        "CZ05L",
        "CZ05L_对称中枢假突破九转买点回归",
        "C_中枢震荡与锚定回归簇",
        1,
        0.26,
        ("1d", "60m", "30m", "15m"),
        "range",
    ),
    SignalSpec(
        "CZ05S",
        "CZ05S_对称中枢假突破九转卖点回归",
        "C_中枢震荡与锚定回归簇",
        -1,
        0.26,
        ("1d", "60m", "30m", "15m"),
        "range",
    ),
    SignalSpec(
        "CZ06L",
        "CZ06L_酒田锤子刺透强底分型缩量回归",
        "C_中枢震荡与锚定回归簇",
        1,
        0.26,
        ("1d", "60m", "30m", "15m"),
        "range",
    ),
    SignalSpec(
        "CZ06S",
        "CZ06S_酒田乌云三鸦强顶分型受阻回归",
        "C_中枢震荡与锚定回归簇",
        -1,
        0.26,
        ("1d", "60m", "30m", "15m"),
        "range",
    ),
    # --- Round 3 方向 D：原生 R-Breaker + 双顺中枢中线突破 + 梯量柱/黄金柱脉冲 (覆盖 15m / 30m / 60m) ---
    SignalSpec(
        "CZ07L", "CZ07L_RBreaker做多梯量黄金柱共振", "A_趋势突破与斜率动量簇", 1, 0.26, ("15m", "30m", "60m"), "trend"
    ),
    SignalSpec(
        "CZ07S", "CZ07S_RBreaker做空梯量高量柱反杀", "A_趋势突破与斜率动量簇", -1, 0.26, ("15m", "30m", "60m"), "trend"
    ),
    SignalSpec(
        "CZ08L", "CZ08L_中枢中线上破双顺通道梯量脉冲", "A_趋势突破与斜率动量簇", 1, 0.26, ("15m", "30m", "60m"), "trend"
    ),
    SignalSpec(
        "CZ08S",
        "CZ08S_中枢中线下破双顺通道梯量杀跌",
        "A_趋势突破与斜率动量簇",
        -1,
        0.26,
        ("15m", "30m", "60m"),
        "trend",
    ),
    # =========================================================================
    # === Round 4 四期 5 分钟 (`5m`) 高频跨日锚定与日内微结构原生组合池 ===
    # =========================================================================
    SignalSpec(
        "M501L", "M501L_5m宽幅唐奇安放量突破DualThrust共振", "A_趋势突破与斜率动量簇", 1, 0.26, ("5m",), "trend"
    ),
    SignalSpec(
        "M501S", "M501S_5m宽幅唐奇安放量破位DualThrust共振", "A_趋势突破与斜率动量簇", -1, 0.26, ("5m",), "trend"
    ),
    SignalSpec(
        "M502L", "M502L_5m跨日48根前低2B弹簧停顿吞没反转", "B_背驰衰竭与极值反转簇", 1, 0.26, ("5m",), "reversal"
    ),
    SignalSpec(
        "M502S", "M502S_5m缠论一二卖双中枢吞没停顿反杀", "B_背驰衰竭与极值反转簇", -1, 0.26, ("5m",), "reversal"
    ),
    SignalSpec("M503L", "M503L_5m对称中枢回踩VWAP企稳放量回归", "C_中枢震荡与锚定回归簇", 1, 0.26, ("5m",), "range"),
    SignalSpec(
        "M503S", "M503S_5m对称中枢反抽VWAP受阻MACD背驰回归", "C_中枢震荡与锚定回归簇", -1, 0.26, ("5m",), "range"
    ),
    SignalSpec("M504L", "M504L_5m早盘30m首区上破DualThrust脉冲", "A_趋势突破与斜率动量簇", 1, 0.26, ("5m",), "trend"),
    SignalSpec(
        "M504S", "M504S_5m早盘30m首区下破RBreaker共振杀跌", "A_趋势突破与斜率动量簇", -1, 0.26, ("5m",), "trend"
    ),
    # =========================================================================
    # === Round 5 五期项目原生 TD 序列 (Setup蓄力 + Countdown确认) 状态机策略池 ===
    # =========================================================================
    SignalSpec(
        "TD01L",
        "TD01L_TD序列(4,2,3)蓄力三确认做多",
        "B_背驰衰竭与极值反转簇",
        1,
        0.28,
        ("5m", "15m", "30m"),
        "reversal",
    ),
    SignalSpec(
        "TD01S",
        "TD01S_TD序列(4,2,3)蓄力三确认做空",
        "B_背驰衰竭与极值反转簇",
        -1,
        0.28,
        ("5m", "15m", "30m"),
        "reversal",
    ),
    SignalSpec(
        "TD02L", "TD02L_TD序列(5,3,6)深度六确认做多", "B_背驰衰竭与极值反转簇", 1, 0.28, ("5m", "15m"), "reversal"
    ),
    SignalSpec(
        "TD02S", "TD02S_TD序列(5,3,6)深度六确认做空", "B_背驰衰竭与极值反转簇", -1, 0.28, ("5m", "15m"), "reversal"
    ),
    SignalSpec(
        "TD03L", "TD03L_跨日TD(4,2,3)波段反转做多", "B_背驰衰竭与极值反转簇", 1, 0.28, ("60m", "30m", "15m"), "reversal"
    ),
    SignalSpec(
        "TD03S",
        "TD03S_跨日TD(4,2,3)波段反转做空",
        "B_背驰衰竭与极值反转簇",
        -1,
        0.28,
        ("60m", "30m", "15m"),
        "reversal",
    ),
    SignalSpec(
        "TD04L",
        "TD04L_跨日TD(5,3,6)中线衰竭反转做多",
        "B_背驰衰竭与极值反转簇",
        1,
        0.28,
        ("1d", "60m", "30m", "15m"),
        "reversal",
    ),
    SignalSpec(
        "TD04S",
        "TD04S_跨日TD(5,3,6)中线衰竭反转做空",
        "B_背驰衰竭与极值反转簇",
        -1,
        0.28,
        ("1d", "60m", "30m", "15m"),
        "reversal",
    ),
]

# ---------------------------------------------------------------------------
# Round 6 (Issue #3)：六簇连续强度原子因子 S = d · Q · D ∈ [-1, 1]
# Q 来自 Rust `strength_*` 信号的 Signal.score/100，D 为动态结构衰减机；实现见
# examples/multi_factor_cta/atomic_factors.py (这里只登记元数据，避免循环导入；单测保证两处一致)。
# ---------------------------------------------------------------------------
_TFS_ALL = ("1d", "60m", "30m", "15m", "5m")
_TFS_RESONANCE = ("60m", "30m", "15m", "5m")
_ATOMIC_SPEC_ROWS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("AC01", "三买卖中枢脱离安全边际突破确认", ARCH_A, _TFS_ALL),
    ("AC02", "均线斜率动量唐奇安突破", ARCH_A, _TFS_ALL),
    ("BC01", "笔背驰衰竭分型确认", ARCH_B, _TFS_ALL),
    ("BC02", "TD九转极值2B假突破", ARCH_B, _TFS_ALL),
    ("CC01", "中枢偏离回归", ARCH_C, _TFS_ALL),
    ("CC02", "KC轨道刺破收回", ARCH_C, _TFS_ALL),
    ("DC01", "量价资金流CLV共振", ARCH_D, _TFS_ALL),
    ("DC02", "NTMDK-KCATR多空共振放量突破", ARCH_D, _TFS_ALL),
    ("EC01", "挤压后波动率扩张点火", ARCH_E, _TFS_ALL),
    ("EC02", "NR7窄幅突破", ARCH_E, _TFS_ALL),
    ("FC01", "大级别结构锚定小级别分型扳机", ARCH_F, _TFS_RESONANCE),
)
ATOMIC_EXIT_MODE = "continuous"
CANDIDATE_HYPOTHESIS_POOL += [
    SignalSpec(
        f"{fam}{'L' if direction > 0 else 'S'}",
        f"{fam}{'L' if direction > 0 else 'S'}_{name}{'做多' if direction > 0 else '做空'}",
        arch,
        direction,
        0.26,
        tfs,
        ATOMIC_EXIT_MODE,
    )
    for fam, name, arch, tfs in _ATOMIC_SPEC_ROWS
    for direction in (1, -1)
]

ALL_SIGNAL_SPECS: dict[str, SignalSpec] = {s.sig_id: s for s in (BASE_SEED_SIGNALS + CANDIDATE_HYPOTHESIS_POOL)}

# 6 簇重构中发生簇迁移的信号 -> 其旧 (4 簇时代) 战略簇名；未列出的信号仅是同簇改名 (见 LEGACY_ARCHETYPE_ALIAS)。
LEGACY_ARCHETYPE_BY_SIG: dict[str, str] = {
    # 原 A_顺势突破簇 候选：OBV 领先 -> D，Keltner 挤压 / 缩量星线放量 -> E
    **dict.fromkeys(("A01L", "A01S", "A02L", "A02S", "A03L", "A03S"), "A_顺势突破簇"),
    # 原 C_震荡回归簇：TD(5,3,6) 中线衰竭反转 -> B
    **dict.fromkeys(("TD04L", "TD04S"), "C_震荡回归簇"),
    # 原 D_日内微结构簇：按收益生成机制拆入 A (顺势突破) / B (洗盘 V 反, TD 衰竭) / C (VWAP 锚定回归)
    **dict.fromkeys(
        (
            "D01L",
            "D01S",
            "D02L",
            "D02S",
            "D03L",
            "D03S",
            "D04L",
            "D04S",
            "D05L",
            "D05S",
            "CZ07L",
            "CZ07S",
            "CZ08L",
            "CZ08S",
            "M504L",
            "M504S",
            "TD01L",
            "TD01S",
        ),
        LEGACY_INTRADAY_ARCHETYPE,
    ),
}


def migrate_archetype(name: str, sig_id: str | None = None) -> str:
    """
    将历史账本 / 冠军名册里的旧四簇名归一为 6 簇新名。

    - 已是新名则原样返回；
    - 提供 ``sig_id`` 且在注册表中时，以注册表现状为准 (原 D_日内微结构簇 与部分 A/C 候选已按机制迁移)；
    - 否则旧 A/B/C 名走 `LEGACY_ARCHETYPE_ALIAS`；旧 D_日内微结构簇 缺少 sig_id 无法判定，抛 ``ValueError``。
    """
    if name in ALL_ARCHETYPES:
        return name
    if sig_id is not None and sig_id in ALL_SIGNAL_SPECS:
        return ALL_SIGNAL_SPECS[sig_id].archetype
    if name in LEGACY_ARCHETYPE_ALIAS:
        return LEGACY_ARCHETYPE_ALIAS[name]
    if name == LEGACY_INTRADAY_ARCHETYPE:
        raise ValueError(f"旧 {name} 已拆入 A/B/C，必须提供 sig_id 才能迁移 (sig_id={sig_id!r})")
    raise ValueError(f"未知战略簇名称: {name}")


TF_SCALE_MAP: dict[str, float] = {
    "1d": 1.00,
    "60m": 0.50,
    "30m": 0.35,
    "15m": 0.25,
    "5m": 0.15,
}


# ---------------------------------------------------------------------------
# 2. 统一多维技术与微观结构特征预计算
# ---------------------------------------------------------------------------
def rust_sma(s: pd.Series, n: int) -> pd.Series:
    """Rust 原生简单均线 `czsc.sma`：与 `s.rolling(n).mean()` 数值一致 (前 n-1 个为 NaN)，由 Rust 算子计算。"""
    return pd.Series(czsc.sma(s.to_numpy(dtype=np.float64), n), index=s.index, dtype=np.float64)


def _infer_causal_bars_in_day(df: pd.DataFrame, scale: float = 1.0) -> int:
    """
    严格因果地推断当前周期的单日标准 K 线根数（不依赖当日未来未收盘的 Bar 计数）：
    优先根据开篇已知的相邻 Bar 时间间隔（5m->48, 15m->16, 30m->8, 60m->4, 1d->1）判定，
    若无时间戳则按周期缩放系数 `scale` 映射。
    """
    if "dt" in df.columns and len(df) >= 2:
        head_dts = pd.to_datetime(df["dt"].iloc[: min(len(df), 8)])
        diffs = head_dts.diff().dropna().dt.total_seconds() / 60.0
        pos_diffs = diffs[diffs > 0]
        if len(pos_diffs) > 0:
            min_step = float(pos_diffs.min())
            if min_step <= 5.5:
                return 48
            if min_step <= 16.0:
                return 16
            if min_step <= 32.0:
                return 8
            if min_step <= 125.0:
                return 4
            return 1
    for sc, cnt in ((0.15, 48), (0.25, 16), (0.35, 8), (0.50, 4), (1.00, 1)):
        if abs(scale - sc) < 0.04:
            return cnt
    return 8


def enrich_kline_features(df: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    df["pct_chg"] = df["close"].pct_change().fillna(0.0)
    df["ma5"] = rust_sma(df["close"], 5)
    df["ma10"] = rust_sma(df["close"], 10)
    df["ma20"] = rust_sma(df["close"], 20)
    df["ma30"] = rust_sma(df["close"], 30)
    df["ma60"] = rust_sma(df["close"], 60)
    df["ma60_slope"] = (df["ma60"] / df["ma60"].shift(5) - 1.0).abs()
    df["vol_ma20"] = rust_sma(df["volume"], 20)
    df["total_range"] = df["high"] - df["low"] + 1e-6
    df["body"] = (df["close"] - df["open"]).abs()
    df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["bias20"] = (df["close"] - df["ma20"]) / df["ma20"]

    # RSI(14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0).rolling(14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14).mean() + 1e-9
    rs = gain / loss
    df["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    # 均线粘合度
    ma_max = df[["ma5", "ma10", "ma20", "ma30"]].max(axis=1)
    ma_min = df[["ma5", "ma10", "ma20", "ma30"]].min(axis=1)
    df["ma_dispersion"] = (ma_max - ma_min) / ma_min

    # ATR 与 60期分位数
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = rust_sma(df["tr"], 14)
    df["atr_rank_60"] = df["atr14"].rolling(60).apply(lambda s: pd.Series(s).rank(pct=True).iloc[-1], raw=False)

    # 滚动 20 期箱体
    df["prev_high_box"] = df["high"].shift(1).rolling(20).max()
    df["prev_low_box"] = df["low"].shift(1).rolling(20).min()
    df["box_amp"] = (df["prev_high_box"] - df["prev_low_box"]) / df["prev_low_box"]

    # 布林带与 Keltner 通道
    std20 = df["close"].rolling(20).std()
    df["boll_up"] = df["ma20"] + 2.0 * std20
    df["boll_dn"] = df["ma20"] - 2.0 * std20
    df["kc_up"] = df["ma20"] + 1.5 * df["atr14"]
    df["kc_dn"] = df["ma20"] - 1.5 * df["atr14"]
    df["in_squeeze"] = (df["boll_up"] <= df["kc_up"]) & (df["boll_dn"] >= df["kc_dn"])

    # MACD (12, 26, 9)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["dif"] = ema12 - ema26
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (df["dif"] - df["dea"]) * 2.0

    # OBV 能量潮及 20 期极值
    sign_chg = np.sign(df["close"].diff().fillna(0.0))
    df["obv"] = (sign_chg * df["volume"]).cumsum()
    df["obv_high20"] = df["obv"].shift(1).rolling(20).max()
    df["obv_low20"] = df["obv"].shift(1).rolling(20).min()
    df["high10"] = df["high"].shift(1).rolling(10).max()
    df["low10"] = df["low"].shift(1).rolling(10).min()

    # 日内分时特征 (ORB / 因果早盘累积极值 / 日内 VWAP) —— 严格零前视
    expected_bars = _infer_causal_bars_in_day(df, scale=scale)
    df["bar_idx_in_day"] = df.groupby("date").cumcount()
    df["bars_in_day"] = expected_bars

    df["orb_high"] = df.groupby("date")["high"].transform("first")
    df["orb_low"] = df.groupby("date")["low"].transform("first")

    # 上午盘高低点 (严格因果：早盘窗口内采用 cummax/cummin，午盘冻结早盘极值，彻底消除 transform('max'/'count') 日内前视)
    half_bars = max(1, expected_bars // 2)
    if "dt" in df.columns and expected_bars > 1:
        dt_times = pd.to_datetime(df["dt"]).dt.time
        is_morning = (df["bar_idx_in_day"] < half_bars) & (dt_times <= time(11, 30))
        if not bool(is_morning.any()):
            is_morning = df["bar_idx_in_day"] < half_bars
    else:
        is_morning = df["bar_idx_in_day"] < half_bars

    am_h_cum = df.groupby("date")["high"].cummax()
    am_l_cum = df.groupby("date")["low"].cummin()
    df["am_high"] = am_h_cum.where(is_morning).groupby(df["date"]).ffill().fillna(df["high"])
    df["am_low"] = am_l_cum.where(is_morning).groupby(df["date"]).ffill().fillna(df["low"])

    typ_price = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typ_price * df["volume"]
    cum_tp_vol = tp_vol.groupby(df["date"]).cumsum()
    cum_vol = df["volume"].groupby(df["date"]).cumsum() + 1e-6
    df["intraday_vwap"] = cum_tp_vol / cum_vol

    return df


STRENGTH_SIGNAL_KINDS: dict[str, str] = {
    # kind -> 信号 k3 段 (Rust `strength_*_V260601` 的模板尾段，用于在挂载后的列名里定位)
    "fx": "FX强度V260601",
    "bi_div": "笔背驰强度V260601",
    "zs_revert": "中枢回归强度V260601",
    "zs_break": "中枢脱离强度V260601",
    "flow": "资金流强度V260601",
    "squeeze": "压缩扩张强度V260601",
}

# Issue #3：此前闲置的高价值 Rust 原生算子 (参数均经 `get_signal_template` + 实测校验)。
# 注：amv_up_dw_line / tas_slope 在 mock 与多组参数下恒为"其他"，暂不挂载，避免产生全常数噪声列。
_EXTENDED_NATIVE_SIGNALS: tuple[dict, ...] = (
    # tas: 均线 / 布林 / RSI / ATR 分层
    {"name": "czsc.signals.tas_ma_base_V221203", "di": 1, "ma_type": "SMA", "timeperiod": 20, "th": 100},
    {"name": "czsc.signals.tas_boll_power_V221112", "di": 1, "timeperiod": 20},
    {"name": "czsc.signals.tas_rsi_base_V230227", "di": 1, "timeperiod": 14, "th": 20},
    {"name": "czsc.signals.tas_atr_V230630", "di": 1, "timeperiod": 14},
    # cvolp 筹码动量 / clv 收盘资金流 / ang 的 ADTM / kcatr KC 通道 / ntmdk
    {"name": "czsc.signals.cvolp_up_dw_line_V230612", "di": 1, "n": 20, "m": 10, "up": 10, "dw": 5},
    {"name": "czsc.signals.clv_up_dw_line_V230605", "di": 1, "n": 20},
    {"name": "czsc.signals.adtm_up_dw_line_V230603", "di": 1, "n": 20, "m": 10, "th": 5},
    {"name": "czsc.signals.kcatr_up_dw_line_V230823", "di": 1, "n": 20, "m": 10, "th": 1},
    {"name": "czsc.signals.ntmdk_V230824", "di": 1, "m": 10},
    {"name": "czsc.signals.er_up_dw_line_V230604", "di": 1, "w": 20, "n": 10},
    # xl 相对位置 / pressure 支撑压力 / coo 原生 TD 序列与 CCI
    {"name": "czsc.signals.xl_bar_position_V240328", "n": 20},
    {"name": "czsc.signals.xl_bar_trend_V240329", "n": 20, "m": 5},
    {"name": "czsc.signals.pressure_support_V240222", "di": 1, "w": 20},
    {"name": "czsc.signals.coo_td_V221111", "di": 1},
    {"name": "czsc.signals.coo_cci_V230323", "di": 1, "n": 14, "ma_type": "SMA", "m": 5},
    # zdy 中枢形态 / 中枢空间
    {"name": "czsc.signals.zdy_zs_V230423", "di": 1},
    {"name": "czsc.signals.zdy_zs_space_V230421", "di": 1},
    # strength: 写入 Signal.score 的结构强度信号 (Q ∈ (0, 1] -> score = round(100·Q))
    {"name": "czsc.signals.strength_fx_V260601", "n": 14},
    {"name": "czsc.signals.strength_bi_div_V260601", "n": 10},
    {"name": "czsc.signals.strength_zs_revert_V260601", "k": 15},
    {"name": "czsc.signals.strength_zs_break_V260601", "n": 10},
    {"name": "czsc.signals.strength_flow_V260601", "n": 20, "k": 20},
    {"name": "czsc.signals.strength_squeeze_V260601", "n": 14, "m": 60, "th": 30},
)


def _extended_native_signal_cfgs(freq_str: str) -> list[dict]:
    """给扩展原生信号表注入周期，得到 `generate_czsc_signals` 的配置列表。"""
    return [{**cfg, "freq": freq_str} for cfg in _EXTENDED_NATIVE_SIGNALS]


def decode_strength(df: pd.DataFrame, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """
    把 Rust `strength_*` 信号列 (`v1_v2_v3_score`) 解码为 (方向 d ∈ {-1,0,+1}, 质量 Q ∈ [0,1])。

    仅做边界解析 (宪法第一条允许的 PyO3/DataFrame 边界胶水)：方向取 v1 的 看多/看空，
    质量取 `Signal.score / 100.0`；`其他` 或缺失 (预热期被填充为 "其他") 解码为 (0, 0)。
    """
    k3 = STRENGTH_SIGNAL_KINDS[kind]
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith("cz_") and c.endswith(k3)]
    if not cols:
        n = len(df)
        return np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.float64)
    ser = df[cols[0]].astype(str)
    parts = ser.str.extract(r"^(看多|看空)_[^_]*_[^_]*_(\d+)$")
    direction = parts[0].map({"看多": 1, "看空": -1}).fillna(0).to_numpy(dtype=np.int8)
    quality = pd.to_numeric(parts[1], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) / 100.0
    return direction, np.clip(quality, 0.0, 1.0)


def attach_native_czsc_signals(df: pd.DataFrame, tf: str, workspace_dir, extended: bool = True) -> pd.DataFrame:
    """
    挂载项目内置 Rust `czsc-signals` 引擎预计算的原生信号列：基础 35 个 (cxt/byi/jcc/bar/obv/vol)，
    `extended=True` 时再叠加此前闲置的 tas/cvolp/clv/ang/kcatr/ntmdk/xl/pressure/coo/zdy 原生算子
    与 6 个写入 `Signal.score` 的 `strength_*` 结构强度信号 (列名前缀统一为 `cz_`)，
    并合成为原生信号组合触发布尔列 (`cz_trig_CZ01L` ~ `cz_trig_CZ08S` 及 5m 专属 `cz_trig_M501L` ~ `cz_trig_M504S`)。

    缓存文件名包含扩展配置指纹，新增/修改信号配置后不会误读旧缓存。
    """
    import hashlib
    import json
    from pathlib import Path

    if df.empty:
        return df.copy()
    suffix = ""
    if extended:
        fingerprint = hashlib.md5(json.dumps(_EXTENDED_NATIVE_SIGNALS, sort_keys=True).encode("utf-8")).hexdigest()[:8]
        suffix = f"_x{fingerprint}"
    cache_path = Path(workspace_dir) / f"native_czsc_signals_{tf}{suffix}.pkl"
    sig_df = None
    if cache_path.exists():
        cached = pd.read_pickle(cache_path)
        if len(cached) == len(df):
            sig_df = cached
    if sig_df is None:
        import czsc

        tf_to_freq = {
            "1d": czsc.Freq.D.value,
            "60m": czsc.Freq.F60.value,
            "30m": czsc.Freq.F30.value,
            "15m": czsc.Freq.F15.value,
            "5m": czsc.Freq.F5.value,
        }
        freq_str = tf_to_freq[tf]
        sub = df[["dt", "symbol", "open", "close", "high", "low", "volume"]].copy()
        sub = sub.rename(columns={"volume": "vol"})
        sub["amount"] = sub["vol"] * sub["close"]
        bars = czsc.format_standard_kline(sub, freq=freq_str)
        sigs_cfg = [
            {"name": "czsc.signals.cxt_five_bi_V230619", "freq": freq_str, "di": 1},
            {"name": "czsc.signals.cxt_seven_bi_V230620", "freq": freq_str, "di": 1},
            {"name": "czsc.signals.cxt_three_bi_V230618", "freq": freq_str, "di": 1},
            {
                "name": "czsc.signals.cxt_third_bs_V230319",
                "freq": freq_str,
                "di": 1,
                "ma_type": "SMA",
                "timeperiod": 20,
            },
            {
                "name": "czsc.signals.cxt_third_bs_V230318",
                "freq": freq_str,
                "di": 1,
                "ma_type": "SMA",
                "timeperiod": 20,
            },
            {"name": "czsc.signals.cxt_third_buy_V230228", "freq": freq_str, "di": 1},
            {
                "name": "czsc.signals.cxt_second_bs_V230320",
                "freq": freq_str,
                "di": 1,
                "ma_type": "SMA",
                "timeperiod": 20,
            },
            {"name": "czsc.signals.cxt_first_buy_V221126", "freq": freq_str, "di": 1},
            {"name": "czsc.signals.cxt_first_sell_V221126", "freq": freq_str, "di": 1},
            {"name": "czsc.signals.cxt_double_zs_V230311", "freq": freq_str, "di": 1},
            {"name": "czsc.signals.cxt_ds_midline_break_V260521", "freq": freq_str, "di": 1, "n": 3, "m": 2},
            {"name": "czsc.signals.cxt_bi_base_V230228", "freq": freq_str, "bi_init_length": 9},
            {"name": "czsc.signals.cxt_bi_status_V230101", "freq": freq_str, "di": 1},
            {"name": "czsc.signals.cxt_fx_power_V221107", "freq": freq_str, "di": 1},
            {"name": "czsc.signals.cxt_bi_end_V230312", "freq": freq_str, "di": 1},
            {"name": "czsc.signals.byi_bi_end_V230106", "freq": freq_str, "di": 1},
            {"name": "czsc.signals.byi_bi_end_V230107", "freq": freq_str, "di": 1},
            {"name": "czsc.signals.byi_second_bs_V230324", "freq": freq_str, "di": 1},
            {"name": "czsc.signals.byi_symmetry_zs_V221107", "freq": freq_str, "di": 1},
            {"name": "czsc.signals.jcc_san_xing_xian_V221023", "freq": freq_str, "di": 1, "th": 2},
            {"name": "czsc.signals.jcc_ten_mo_V221028", "freq": freq_str, "di": 1},
            {"name": "czsc.signals.jcc_yun_xian_V221118", "freq": freq_str, "di": 1},
            {"name": "czsc.signals.jcc_ci_tou_V221101", "freq": freq_str, "di": 1, "z": 5, "th": 50},
            {"name": "czsc.signals.jcc_wu_yun_gai_ding_V221101", "freq": freq_str, "di": 1, "z": 5, "th": 50},
            {"name": "czsc.signals.jcc_three_crow_V221108", "freq": freq_str, "di": 1},
            {"name": "czsc.signals.jcc_san_fa_V20221115", "freq": freq_str, "di": 1},
            {"name": "czsc.signals.bar_fake_break_V230204", "freq": freq_str, "di": 1, "n": 10, "m": 3},
            {
                "name": "czsc.signals.bar_fang_liang_break_V221216",
                "freq": freq_str,
                "di": 1,
                "ma_type": "SMA",
                "timeperiod": 20,
                "th": 13,
            },
            {"name": "czsc.signals.bar_dual_thrust_V230403", "freq": freq_str, "di": 1, "n": 4, "k1": 5, "k2": 5},
            {"name": "czsc.signals.bar_r_breaker_V230326", "freq": freq_str, "n": 1},
            {"name": "czsc.signals.bar_td9_V240616", "freq": freq_str, "n": 9},
            {"name": "czsc.signals.bar_accelerate_V221110", "freq": freq_str, "di": 1, "window": 10},
            {
                "name": "czsc.signals.obv_up_dw_line_V230719",
                "freq": freq_str,
                "di": 1,
                "n": 20,
                "m": 10,
                "max_overlap": 3,
            },
            {"name": "czsc.signals.vol_ti_suo_V221216", "freq": freq_str, "di": 1},
            {"name": "czsc.signals.vol_gao_di_V221218", "freq": freq_str, "di": 1},
        ]
        sigs_cfg += _extended_native_signal_cfgs(freq_str) if extended else []
        df_sig = czsc.generate_czsc_signals(bars, sigs_cfg, sdt="2020-01-01", init_n=20, df=True)
        rename_map = {}
        sig_cols = []
        for col in df_sig.columns:
            if col.startswith(f"{freq_str}_"):
                short_col = "cz_" + col[len(freq_str) + 1 :]
                rename_map[col] = short_col
                sig_cols.append(short_col)
        df_sig = df_sig.rename(columns=rename_map)
        df_sig["dt"] = pd.to_datetime(df_sig["dt"]).dt.tz_localize(None)
        merged = pd.merge(df[["dt"]], df_sig[["dt"] + sig_cols], on="dt", how="left").fillna("其他")
        sig_df = merged[sig_cols]
        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            sig_df.to_pickle(cache_path)

    df = pd.concat([df.reset_index(drop=True), sig_df.reset_index(drop=True)], axis=1)

    def _safe_get(target_col: str, fallback_candidates: list[str] = None) -> pd.Series:
        if target_col in df.columns:
            return df[target_col].astype(str)
        if fallback_candidates:
            for cand in fallback_candidates:
                if cand in df.columns:
                    return df[cand].astype(str)
                for col in df.columns:
                    if cand in col:
                        return df[col].astype(str)
        for col in df.columns:
            if target_col in col:
                return df[col].astype(str)
        return pd.Series("其他", index=df.index, dtype=str)

    bi5 = _safe_get("cz_D1五笔_形态V230619", ["五笔"])
    bi7 = _safe_get("cz_D1七笔_形态V230620", ["七笔"])
    bs3_19 = _safe_get("cz_D1#SMA#20_BS3辅助V230319", ["BS3辅助V230319", "BS3"])
    bs3_28 = _safe_get("cz_D1_三买辅助V230228", ["三买辅助V230228"])
    bs2_20 = _safe_get("cz_D1#SMA#20_BS2辅助V230320", ["BS2辅助V230320", "BS2"])
    buy1 = _safe_get("cz_D1B_BUY1", ["BUY1"])
    sell1 = _safe_get("cz_D1B_SELL1", ["SELL1"])
    dzs = _safe_get("cz_D1双中枢_BS1辅助V230311", ["双中枢"])
    mid_brk = _safe_get("cz_D1N3M2_双顺中线突破V260521", ["双顺中线突破"])
    bi_base = _safe_get("cz_D0BL9_V230228", ["BL9"])
    bi_st = _safe_get("cz_D1_表里关系V230101", ["表里关系"])
    fx_pwr = _safe_get("cz_D1F_分型强弱", ["分型强弱"])
    be_macd = _safe_get("cz_D0MACD12#26#9_BE辅助V230312", ["BE辅助V230312"])
    be_stop = _safe_get("cz_D0停顿分型_BE辅助V230106", ["停顿分型", "BE辅助V230106"])
    be_ver = _safe_get("cz_D0验证分型_BE辅助V230107", ["验证分型", "BE辅助V230107"])
    byi_bs2 = _safe_get("cz_D1MACD12#26#9回抽零轴_BS2辅助V230324", ["回抽零轴"])
    sym_zs = _safe_get("cz_D1B_对称中枢", ["对称中枢"])
    jcc_umb = _safe_get("cz_D1TH200_伞形线", ["伞形线"])
    jcc_eng = _safe_get("cz_D1_吞没形态", ["吞没形态"])
    jcc_har = _safe_get("cz_D1_孕线", ["孕线"])
    jcc_prc = _safe_get("cz_D1Z5TH50_刺透形态", ["刺透形态"])
    jcc_drk = _safe_get("cz_D1Z5TH50_乌云盖顶", ["乌云盖顶"])
    jcc_crw = _safe_get("cz_D1_三只乌鸦", ["三只乌鸦"])
    jcc_3f = _safe_get("cz_D1K_三法", ["三法"])
    fake_brk = _safe_get("cz_D1N10M3_假突破", ["假突破"])
    fl_brk = _safe_get("cz_D1TH13#SMA#20_突破V221216", ["突破V221216"])
    dt_bs = _safe_get("cz_D1通道突破#4#5#5_BS辅助V230403", ["通道突破", "BS辅助V230403"])
    rbrk = _safe_get("cz_RBreaker_BS辅助V230326", ["RBreaker"])
    td9 = _safe_get("cz_神奇九转N9_BS辅助V240616", ["神奇九转"])
    acc = _safe_get("cz_D1W10_加速V221110", ["加速V221110"])
    obv_s = _safe_get("cz_D1N20M10MO3_OBV能量V230719", ["OBV能量"])
    vol_ts = _safe_get("cz_D1K_量柱V221216", ["量柱V221216"])
    vol_gd = _safe_get("cz_D1K_量柱V221218", ["量柱V221218"])

    df["cz_trig_CZ01L"] = (
        (bi5.str.startswith("上颈线突破") | bi5.str.startswith("类三买") | bi7.str.startswith("类三买"))
        & (fl_brk.str.startswith("放量突破") | dt_bs.str.startswith("看多"))
        & (be_macd.str.startswith("看多") | bi_base.str.startswith("向上"))
    ).values
    df["cz_trig_CZ01S"] = (
        (bi5.str.startswith("下颈线突破") | bi5.str.startswith("类三卖") | bi7.str.startswith("类三卖"))
        & dt_bs.str.startswith("看空")
        & (be_macd.str.startswith("看空") | bi_base.str.startswith("向下"))
        & (df["close"] < df["ma60"])
    ).values
    df["cz_trig_CZ02L"] = (
        (bs3_19.str.startswith("三买_均线新高") | bs3_28.str.startswith("三买_"))
        & (acc.str.startswith("上涨") | obv_s.str.startswith("看多") | jcc_3f.str.contains("上升三法"))
        & dt_bs.str.startswith("看多")
    ).values
    df["cz_trig_CZ02S"] = (
        bs3_19.str.startswith("三卖_均线新低")
        & (acc.str.startswith("下跌") | obv_s.str.startswith("看空") | jcc_3f.str.contains("下降三法"))
        & dt_bs.str.startswith("看空")
    ).values
    df["cz_trig_CZ03L"] = (
        (
            buy1.str.startswith("一买_")
            | dzs.str.startswith("看多")
            | bi5.str.contains("底背驰")
            | bi7.str.contains("底背驰")
        )
        & (
            be_stop.str.startswith("看多")
            | be_ver.str.startswith("看多")
            | jcc_eng.str.contains("看涨吞没")
            | jcc_prc.str.startswith("满足")
        )
    ).values
    df["cz_trig_CZ03S"] = (
        (sell1.str.startswith("一卖_") | dzs.str.startswith("看空") | bi5.str.contains("顶背驰"))
        & (
            be_stop.str.startswith("看空")
            | be_ver.str.startswith("看空")
            | jcc_eng.str.contains("看跌吞没")
            | jcc_drk.str.startswith("满足")
        )
    ).values
    df["cz_trig_CZ04L"] = (
        bs2_20.str.startswith("二买")
        & (
            byi_bs2.str.startswith("看多")
            | be_stop.str.startswith("看多_强")
            | jcc_umb.str.startswith("满足_锤子")
            | jcc_har.str.startswith("看多")
        )
    ).values
    df["cz_trig_CZ04S"] = (
        (bs2_20.str.startswith("二卖") | sell1.str.startswith("一卖_"))
        & (
            byi_bs2.str.startswith("看空")
            | be_stop.str.startswith("看空_强")
            | jcc_umb.str.startswith("满足_上吊")
            | jcc_har.str.startswith("看空")
            | jcc_drk.str.startswith("满足")
        )
    ).values
    df["cz_trig_CZ05L"] = (
        sym_zs.str.startswith("是_")
        & (fake_brk.str.startswith("看多") | td9.str.startswith("买点_"))
        & (be_macd.str.startswith("看多") | bi_st.str.startswith("向下_底分"))
    ).values
    df["cz_trig_CZ05S"] = (
        sym_zs.str.startswith("是_")
        & (fake_brk.str.startswith("看空") | td9.str.startswith("卖点_"))
        & (be_macd.str.startswith("看空") | bi_st.str.startswith("向上_顶分") | fx_pwr.str.startswith("强顶"))
    ).values
    df["cz_trig_CZ06L"] = (
        (
            jcc_umb.str.startswith("满足_锤子")
            | jcc_eng.str.contains("看涨吞没")
            | jcc_prc.str.startswith("满足")
            | jcc_har.str.startswith("看多")
        )
        & (fx_pwr.str.startswith("强底") | fx_pwr.str.startswith("中底"))
        & (vol_ts.str.startswith("缩量_") | vol_gd.str.startswith("低量柱"))
    ).values
    df["cz_trig_CZ06S"] = (
        sym_zs.str.startswith("是_")
        & (
            jcc_umb.str.startswith("满足_上吊")
            | jcc_eng.str.contains("看跌吞没")
            | jcc_drk.str.startswith("满足")
            | jcc_crw.str.startswith("满足")
        )
        & (fx_pwr.str.startswith("强顶") | fx_pwr.str.startswith("中顶"))
    ).values
    df["cz_trig_CZ07L"] = (
        rbrk.str.startswith("做多_")
        & (vol_ts.str.startswith("梯量_价升") | vol_gd.str.contains("黄金柱") | fl_brk.str.startswith("放量突破"))
        & be_macd.str.startswith("看多")
    ).values
    df["cz_trig_CZ07S"] = (
        rbrk.str.startswith("做空_")
        & dt_bs.str.startswith("看空")
        & (vol_ts.str.startswith("梯量_价跌") | vol_gd.str.startswith("高量柱"))
        & be_macd.str.startswith("看空")
        & (df["close"] < df["ma60"])
    ).values
    df["cz_trig_CZ08L"] = (
        mid_brk.str.startswith("看多_区间_上破中线")
        & dt_bs.str.startswith("看多")
        & (vol_ts.str.startswith("梯量_价升") | fl_brk.str.startswith("放量突破"))
    ).values
    df["cz_trig_CZ08S"] = (
        (mid_brk.str.startswith("看空_区间_下破中线") | mid_brk.str.startswith("看空_下方_向下确认"))
        & dt_bs.str.startswith("看空")
        & vol_ts.str.startswith("梯量_价跌")
        & be_stop.str.startswith("看空")
    ).values

    # Round 4: 5m 专属跨日锚定 (48根全天极值 / 240根五日均线 / 30m首区ORB) 与原生 czsc-signals 组合
    if tf == "5m":
        c = df["close"].values
        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        ma20 = df["ma20"].values
        ma60 = df["ma60"].values
        ma240 = df["close"].rolling(240, min_periods=1).mean().values
        h48 = df["high"].shift(2).rolling(48, min_periods=1).max().bfill().values
        l48 = df["low"].shift(2).rolling(48, min_periods=1).min().bfill().values
        vol = df["volume"].values
        vma20 = df["vol_ma20"].values
        vwap = df["intraday_vwap"].values
        bidx = df["bar_idx_in_day"].values
        h10 = df["high10"].values
        l10 = df["low10"].values
        is_first_30m = df["bar_idx_in_day"] < 6
        orb30_h = (
            df.groupby("date")["high"]
            .cummax()
            .where(is_first_30m)
            .groupby(df["date"])
            .ffill()
            .fillna(df["high"])
            .values
        )
        orb30_l = (
            df.groupby("date")["low"].cummin().where(is_first_30m).groupby(df["date"]).ffill().fillna(df["low"]).values
        )

        m501l = np.asarray(
            (df["box_amp"] > 0.012)
            & (c > ma60)
            & (ma20 > ma60)
            & (c > h10)
            & (c > o)
            & (vol >= 1.45 * vma20)
            & (dt_bs.str.startswith("看多") | fl_brk.str.startswith("放量突破")),
            dtype=bool,
        )
        m501s = np.asarray(
            (df["box_amp"] > 0.012)
            & (c < ma60)
            & (ma20 < ma60)
            & (c < l10)
            & (c < o)
            & (vol >= 1.45 * vma20)
            & dt_bs.str.startswith("看空"),
            dtype=bool,
        )
        m502l = np.asarray(
            (c > ma240)
            & (np.roll(l, 1) < l48)
            & (c > l48)
            & (c > o)
            & (c > np.roll(h, 1))
            & (vol >= 1.28 * vma20)
            & (be_stop.str.startswith("看多") | be_macd.str.startswith("看多") | jcc_eng.str.contains("看涨吞没")),
            dtype=bool,
        )
        m502s = np.asarray(
            (c < ma240)
            & (bs2_20.str.startswith("二卖") | sell1.str.startswith("一卖_") | dzs.str.startswith("看空"))
            & (be_stop.str.startswith("看空") | jcc_eng.str.contains("看跌吞没") | jcc_drk.str.startswith("满足"))
            & (c < ma20)
            & (vol >= 1.20 * vma20),
            dtype=bool,
        )
        m503l = np.asarray(
            sym_zs.str.startswith("是_")
            & (c > ma240)
            & (np.roll(l, 1) <= np.roll(vwap, 1) * 0.999)
            & (np.roll(c, 1) >= np.roll(vwap, 1) * 0.996)
            & (c > o)
            & (c > np.roll(h, 1))
            & (c > vwap)
            & (vol >= 1.35 * vma20)
            & fl_brk.str.startswith("放量突破"),
            dtype=bool,
        )
        m503s = np.asarray(
            sym_zs.str.startswith("是_")
            & (bidx % 6 >= 3)
            & (c < ma240)
            & (c < ma60)
            & (c < vwap)
            & (c < o)
            & (c < np.roll(l, 1))
            & (vol >= 1.28 * vma20)
            & (be_macd.str.startswith("看空") | dt_bs.str.startswith("看空")),
            dtype=bool,
        )
        m504l = np.asarray(
            (bidx >= 6)
            & (bidx <= 16)
            & (c > orb30_h)
            & (np.roll(c, 1) <= orb30_h)
            & (c > vwap)
            & (c > ma60)
            & (vol >= 1.75 * vma20)
            & fl_brk.str.startswith("放量突破")
            & (dt_bs.str.startswith("看多") | rbrk.str.startswith("做多_")),
            dtype=bool,
        )
        m504s = np.asarray(
            (bidx >= 6)
            & (bidx <= 18)
            & (c < orb30_l)
            & (np.roll(c, 1) >= orb30_l)
            & (c < vwap)
            & (c < ma60)
            & (c < ma240)
            & (vol >= 1.65 * vma20)
            & (dt_bs.str.startswith("看空") | rbrk.str.startswith("做空_")),
            dtype=bool,
        )
        for col_name, arr in [
            ("cz_trig_M501L", m501l),
            ("cz_trig_M501S", m501s),
            ("cz_trig_M502L", m502l),
            ("cz_trig_M502S", m502s),
            ("cz_trig_M503L", m503l),
            ("cz_trig_M503S", m503s),
            ("cz_trig_M504L", m504l),
            ("cz_trig_M504S", m504s),
        ]:
            arr[:240] = False
            df[col_name] = arr

    return df


# ---------------------------------------------------------------------------
# 3. 项目原生 TD 序列 (Setup 蓄力 + Countdown 三条件确认) 状态机与分方向状态机
# ---------------------------------------------------------------------------
TD_SIGNAL_PARAM_MAP: dict[str, tuple[int, int, int, int, float, bool, int, bool]] = {
    # sig_id: (direction, n1, n2, n3, sl_threshold, daily_reset, max_hold_bars, trend_filter)
    "TD01L": (1, 4, 2, 3, 0.015, True, 999, False),
    "TD01S": (-1, 4, 2, 3, 0.015, True, 999, True),
    "TD02L": (1, 5, 3, 6, 0.015, True, 999, True),
    "TD02S": (-1, 5, 3, 6, 0.015, True, 999, True),
    "TD03L": (1, 4, 2, 3, 0.018, False, 8, False),
    "TD03S": (-1, 4, 2, 3, 0.018, False, 8, True),
    "TD04L": (1, 5, 3, 6, 0.018, False, 8, False),
    "TD04S": (-1, 5, 3, 6, 0.018, False, 8, False),
}


def compute_td_triggers_and_weights(
    df: pd.DataFrame,
    direction: int,
    n1: int = 4,
    n2: int = 2,
    n3: int = 3,
    sl_threshold: float = 0.015,
    daily_reset: bool = True,
    max_hold_bars: int = 999,
    trend_filter: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    项目原生 TD 序列两阶段状态机 (源自 examples/czsc_td_strategy_backtest.py 与 td_sequential_real_backtest.py)：
    - Stage 1 (Setup 蓄力): close 与 close[i-n1] 比较得到 ud_run，连续满 ±n2 根触发蓄力完成。
    - Stage 2 (Countdown 计数): 蓄力完成后满足三条件微突破 (condA & condB & condC) 累计达 n3 次触发入场，
      记录计数期极值 buy_count_low / sell_count_high 作为结构止损位，配合 sl_threshold 动态跟踪止损。
    """
    n = len(df)
    triggers = np.zeros(n, dtype=bool)
    weights = np.zeros(n, dtype=np.float64)

    c_arr = df["close"].values
    h_arr = df["high"].values
    l_arr = df["low"].values
    ma60_arr = df["ma60"].values
    vwap_arr = df["intraday_vwap"].values
    dates = df["date"].values
    dts = pd.to_datetime(df["dt"]).dt.time.values

    open_t = time(9, 30)
    no_open_t = time(14, 45)
    exit_t = time(14, 55)

    last_date = None
    last_ud = 0
    ud_run = 0

    buy_setup = False
    buy_count = 0
    buy_k1_close = None
    buy_count_low = float("inf")

    sell_setup = False
    sell_count = 0
    sell_k1_close = None
    sell_count_high = float("-inf")

    pos = 0
    entry_idx = -999
    stop_price = None
    highest_price = None
    lowest_price = None
    bar_count_today = 0

    for i in range(n):
        c_now, h_now, l_now = c_arr[i], h_arr[i], l_arr[i]
        cur_date = dates[i]
        cur_time = dts[i]

        if daily_reset and last_date != cur_date:
            last_date = cur_date
            last_ud = 0
            ud_run = 0
            buy_setup = False
            buy_count = 0
            buy_k1_close = None
            buy_count_low = float("inf")
            sell_setup = False
            sell_count = 0
            sell_k1_close = None
            sell_count_high = float("-inf")
            bar_count_today = 0
            pos = 0
            stop_price = None
            highest_price = None
            lowest_price = None

        bar_count_today += 1

        if daily_reset and cur_time >= exit_t:
            pos = 0
            stop_price = None
            highest_price = None
            lowest_price = None
            weights[i] = 0.0
            continue

        if pos != 0 and (i - entry_idx) >= max_hold_bars:
            pos = 0
            stop_price = None
            highest_price = None
            lowest_price = None
            weights[i] = 0.0
            continue

        if pos == 1 and stop_price is not None and l_now <= stop_price:
            pos = 0
            stop_price = None
            highest_price = None
            weights[i] = 0.0
            continue
        elif pos == -1 and stop_price is not None and h_now >= stop_price:
            pos = 0
            stop_price = None
            lowest_price = None
            weights[i] = 0.0
            continue

        if pos == 1 and sl_threshold > 0:
            highest_price = max(highest_price or c_now, h_now)
            if highest_price > 0 and (highest_price - c_now) / highest_price >= sl_threshold:
                pos = 0
                stop_price = None
                highest_price = None
                weights[i] = 0.0
                continue
        elif pos == -1 and sl_threshold > 0:
            lowest_price = min(lowest_price or c_now, l_now)
            if lowest_price > 0 and (c_now - lowest_price) / lowest_price >= sl_threshold:
                pos = 0
                stop_price = None
                lowest_price = None
                weights[i] = 0.0
                continue

        min_bars = max(n1, 2) + 1
        if (daily_reset and bar_count_today < min_bars) or i < max(60, min_bars):
            weights[i] = float(pos)
            continue

        c_lag = c_arr[i - n1]
        ud = 1 if c_now > c_lag else (-1 if c_now < c_lag else 0)
        if ud == last_ud and ud != 0:
            ud_run += ud
        else:
            ud_run = ud
        last_ud = ud

        if ud_run == n2:
            sell_setup = True
            sell_count = 0
            sell_k1_close = None
            sell_count_high = float("-inf")
        elif ud_run == -n2:
            buy_setup = True
            buy_count = 0
            buy_k1_close = None
            buy_count_low = float("inf")

        if buy_setup:
            condA = c_now >= h_arr[i - 2]
            condB = h_now > h_arr[i - 1]
            condC = (buy_k1_close is None) or (c_now > buy_k1_close)
            if condA and condB and condC:
                buy_count += 1
                if buy_count == 1:
                    buy_k1_close = c_now
                buy_count_low = min(buy_count_low, l_now)

        if sell_setup:
            condA = c_now <= l_arr[i - 2]
            condB = l_now < l_arr[i - 1]
            condC = (sell_k1_close is None) or (c_now < sell_k1_close)
            if condA and condB and condC:
                sell_count += 1
                if sell_count == 1:
                    sell_k1_close = c_now
                sell_count_high = max(sell_count_high, h_now)

        in_open_window = (open_t <= cur_time < no_open_t) if daily_reset else True

        if direction == 1 and buy_count >= n3:
            ok_filter = (not trend_filter) or (c_now > ma60_arr[i] and c_now > vwap_arr[i])
            if in_open_window and pos == 0 and ok_filter:
                pos = 1
                entry_idx = i
                stop_price = buy_count_low
                highest_price = c_now
                triggers[i] = True
            buy_setup = False
            buy_count = 0
        elif direction == -1 and sell_count >= n3:
            ok_filter = (not trend_filter) or (c_now < ma60_arr[i] and c_now < vwap_arr[i])
            if in_open_window and pos == 0 and ok_filter:
                pos = -1
                entry_idx = i
                stop_price = sell_count_high
                lowest_price = c_now
                triggers[i] = True
            sell_setup = False
            sell_count = 0

        weights[i] = float(pos)

    return triggers, weights


def compute_signal_triggers_and_weights(
    df: pd.DataFrame,
    sig_id: str,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    针对 A/B/C 三类战略方向 (trend / reversal / range) 采用物理匹配的离场状态机
    (旧 intraday 模式随 D_日内微结构簇 取消，其成员已改用目标簇离场逻辑)：
    - trend:    MA20 趋势跟踪 + 保本提损 + 3%*scale 移动止盈
    - reversal: 从偏离区向 MA20 均值回归 (触及 MA20 或盈利达 +1.35*ATR 立即止盈，或满 8 根时间平仓；5m 信号翻倍持仓根数)
    - range:    通道边界向中轨回归 (触及 MA20 或盈利达 +1.15*ATR 立即止盈，或满 6 根时间平仓；5m 信号翻倍持仓根数)
    D/E/F 三簇为连续强度因子，见 examples/multi_factor_cta/atomic_factors.py。
    - TD序列:   直连项目原生 TD(Setup+Countdown) 两阶段状态机与计数极值结构止损
    """
    spec_atomic = ALL_SIGNAL_SPECS.get(sig_id)
    if spec_atomic is not None and spec_atomic.exit_mode == ATOMIC_EXIT_MODE:
        # 连续强度原子因子 (D/E/F 簇及 A/B/C 簇的 Issue #3 新增因子)：延迟导入以避免循环依赖
        from examples.multi_factor_cta.atomic_factors import compute_atomic_signal

        return compute_atomic_signal(df, sig_id, scale)

    if sig_id in TD_SIGNAL_PARAM_MAP:
        direction, n1, n2, n3, sl_th, daily_rst, max_hold, t_flt = TD_SIGNAL_PARAM_MAP[sig_id]
        return compute_td_triggers_and_weights(
            df,
            direction=direction,
            n1=n1,
            n2=n2,
            n3=n3,
            sl_threshold=sl_th,
            daily_reset=daily_rst,
            max_hold_bars=max_hold,
            trend_filter=t_flt,
        )

    spec = ALL_SIGNAL_SPECS[sig_id]
    n = len(df)
    triggers = np.zeros(n, dtype=bool)
    weights = np.zeros(n, dtype=np.float64)
    is_5m_sig = sig_id.startswith("M5")
    cz_trig_arr = (
        df[f"cz_trig_{sig_id}"].values
        if (sig_id.startswith("CZ") or is_5m_sig) and f"cz_trig_{sig_id}" in df.columns
        else None
    )

    box_amp_th = 0.09 * scale
    disp_th = 0.02 * scale
    be_th = 0.02 * scale
    act_th = 0.03 * scale
    trail_th = 0.03 * scale
    pierce_th = 0.015 * scale

    c_arr = df["close"].values
    o_arr = df["open"].values
    h_arr = df["high"].values
    l_arr = df["low"].values
    ma5_arr = df["ma5"].values
    ma10_arr = df["ma10"].values
    ma20_arr = df["ma20"].values
    ma30_arr = df["ma30"].values
    ma60_arr = df["ma60"].values
    ma60_slp = df["ma60_slope"].values
    vol_arr = df["volume"].values
    vma20_arr = df["vol_ma20"].values
    amp_arr = df["box_amp"].values
    btop_arr = df["prev_high_box"].values
    bbot_arr = df["prev_low_box"].values
    disp_arr = df["ma_dispersion"].values
    atr_arr = df["atr14"].values
    arank_arr = df["atr_rank_60"].values
    rng_arr = df["total_range"].values
    body_arr = df["body"].values
    ushad_arr = df["upper_shadow"].values
    lshad_arr = df["lower_shadow"].values
    macd_arr = df["macd_hist"].values
    obv_arr = df["obv"].values
    obv_h20 = df["obv_high20"].values
    obv_l20 = df["obv_low20"].values
    h10_arr = df["high10"].values
    l10_arr = df["low10"].values
    sqz_arr = df["in_squeeze"].values
    bup_arr = df["boll_up"].values
    bdn_arr = df["boll_dn"].values
    kcup_arr = df["kc_up"].values
    kcdn_arr = df["kc_dn"].values
    bidx_arr = df["bar_idx_in_day"].values
    bmax_arr = df["bars_in_day"].values
    orbh_arr = df["orb_high"].values
    orbl_arr = df["orb_low"].values
    amh_arr = df["am_high"].values
    aml_arr = df["am_low"].values
    vwap_arr = df["intraday_vwap"].values
    rsi_arr = df["rsi14"].values
    bias20_arr = df["bias20"].values

    pos = 0
    entry_idx = -999
    stop_p = entry_p = peak_p = trough_p = None
    pend_l = pend_s = False
    l_idx = s_idx = -999
    l_btop = s_bbot = 0.0
    l_ret = 9999.0
    s_reb = 0.0

    tp_scale = 1.35 if is_5m_sig else 1.0
    sl_scale = 1.20 if is_5m_sig else 1.0
    rev_max_b = 16 if is_5m_sig else 8
    rng_max_b = 12 if is_5m_sig else 6
    start_bar = 240 if is_5m_sig else 60

    for i in range(start_bar, n):
        c, o, h, l = c_arr[i], o_arr[i], h_arr[i], l_arr[i]
        ma20, ma60 = ma20_arr[i], ma60_arr[i]
        atr = max(atr_arr[i], 1e-5)

        # 1. 分方向定制离场状态机
        exited = False
        bars_held = i - entry_idx

        if pos == 1:
            peak_p = max(peak_p or c, h)
            unreal = (peak_p - entry_p) / entry_p
            if spec.exit_mode == "trend":
                if unreal >= be_th:
                    stop_p = max(stop_p, entry_p * (1 + 0.002 * scale))
                if (
                    stop_p is not None
                    and l <= stop_p
                    or unreal >= act_th
                    and (peak_p - c) / peak_p >= trail_th
                    or (c < ma60 if is_5m_sig else c < ma20)
                    or (is_5m_sig and bars_held >= 32)
                ):
                    pos = 0
                    exited = True
            elif spec.exit_mode == "reversal":
                if (
                    stop_p is not None
                    and l <= stop_p
                    or h >= entry_p + 1.35 * tp_scale * atr
                    or (not is_5m_sig and bars_held >= 2 and c >= ma20)
                    or bars_held >= rev_max_b
                ):
                    pos = 0
                    exited = True
            elif spec.exit_mode == "range":
                if (
                    stop_p is not None
                    and l <= stop_p
                    or h >= entry_p + 1.15 * tp_scale * atr
                    or (not is_5m_sig and c >= ma20)
                    or bars_held >= rng_max_b
                ):
                    pos = 0
                    exited = True

        elif pos == -1:
            trough_p = min(trough_p or c, l)
            unreal = (entry_p - trough_p) / entry_p
            if spec.exit_mode == "trend":
                if unreal >= be_th:
                    stop_p = min(stop_p, entry_p * (1 - 0.002 * scale))
                if (
                    stop_p is not None
                    and h >= stop_p
                    or unreal >= act_th
                    and (c - trough_p) / trough_p >= trail_th
                    or (c > ma60 if is_5m_sig else c > ma20)
                    or (is_5m_sig and bars_held >= 32)
                ):
                    pos = 0
                    exited = True
            elif spec.exit_mode == "reversal":
                if (
                    stop_p is not None
                    and h >= stop_p
                    or l <= entry_p - 1.35 * tp_scale * atr
                    or (not is_5m_sig and bars_held >= 2 and c <= ma20)
                    or bars_held >= rev_max_b
                ):
                    pos = 0
                    exited = True
            elif spec.exit_mode == "range":
                if (
                    stop_p is not None
                    and h >= stop_p
                    or l <= entry_p - 1.15 * tp_scale * atr
                    or (not is_5m_sig and c <= ma20)
                    or bars_held >= rng_max_b
                ):
                    pos = 0
                    exited = True

        if exited:
            stop_p = entry_p = peak_p = trough_p = None

        # 2. 各方向信号触发条件
        entered = False

        # === 基石真金信号 (5个) ===
        if sig_id == "S1L":
            if pos == 0 and not pend_l and amp_arr[i] <= box_amp_th and c > ma60 and c > btop_arr[i]:
                pend_l, l_idx, l_btop, l_ret = True, i, btop_arr[i], l
            if pend_l and (i - l_idx > 4):
                pend_l = False
            if pos == 0 and pend_l and i > l_idx:
                l_ret = min(l_ret, l)
                if c < l_btop * (1 - 0.005 * scale):
                    pend_l = False
                elif c > o and c >= h_arr[i - 1]:
                    pos, stop_p, entry_p, peak_p, pend_l = 1, max(l_ret, l_btop * (1 - 0.008 * scale)), c, c, False
                    entered = True

        elif sig_id == "S1S":
            if pos == 0 and not pend_s and amp_arr[i] <= box_amp_th and c < ma60 and c < bbot_arr[i]:
                pend_s, s_idx, s_bbot, s_reb = True, i, bbot_arr[i], h
            if pend_s and (i - s_idx > 4):
                pend_s = False
            if pos == 0 and pend_s and i > s_idx:
                s_reb = max(s_reb, h)
                if c > s_bbot * (1 + 0.005 * scale):
                    pend_s = False
                elif c < o and c <= l_arr[i - 1]:
                    pos, stop_p, entry_p, trough_p, pend_s = -1, min(s_reb, s_bbot * (1 + 0.008 * scale)), c, c, False
                    entered = True

        elif sig_id == "S2L":
            if pos == 0 and disp_arr[i] <= disp_th:
                m_min = min(ma5_arr[i], ma10_arr[i], ma20)
                m_max = max(ma5_arr[i], ma10_arr[i], ma20, ma30_arr[i])
                if o <= m_min * (1 + 0.005 * scale) and c >= m_max and c > o * (1 + pierce_th) and c > ma60:
                    pos, entry_p, peak_p, stop_p = 1, c, c, l - 0.002 * scale
                    entered = True

        elif sig_id == "S3L":
            if (
                (
                    pos == 0
                    and h_arr[i - 1] < h_arr[i - 2]
                    and l_arr[i - 1] > l_arr[i - 2]
                    and vol_arr[i] >= 1.2 * vma20_arr[i]
                )
                and c > h_arr[i - 2]
                and c > o
                and c > ma60
            ):
                pos, entry_p, peak_p, stop_p = 1, c, c, l_arr[i - 1] - 0.002 * scale
                entered = True

        elif sig_id == "S4S":
            if pos == 0 and arank_arr[i] <= 0.20 and rng_arr[i] >= 1.5 * atr:
                if c < ma60 and c < o and c < ma20:
                    pos, entry_p, trough_p, stop_p = -1, c, c, h + 0.002 * scale
                    entered = True

        # === 方向 B：背驰反转簇 ===
        elif sig_id == "B01L":
            # MACD底背驰放量反包
            if pos == 0 and l_arr[i - 1] <= bbot_arr[i - 1] * (1 + 0.003 * scale):
                prev_min_macd = float(np.min(macd_arr[i - 20 : i - 3]))
                if macd_arr[i - 1] > prev_min_macd * 0.85 and macd_arr[i] > macd_arr[i - 1]:
                    if c > o and c >= h_arr[i - 1] and vol_arr[i] >= 1.15 * vma20_arr[i]:
                        pos, entry_p, peak_p, stop_p = 1, c, c, c - 0.95 * atr
                        entered = True

        elif sig_id == "B01S":
            # MACD顶背驰放量反杀
            if pos == 0 and h_arr[i - 1] >= btop_arr[i - 1] * (1 - 0.003 * scale):
                prev_max_macd = float(np.max(macd_arr[i - 20 : i - 3]))
                if macd_arr[i - 1] < prev_max_macd * 0.85 and macd_arr[i] < macd_arr[i - 1]:
                    if c < o and c <= l_arr[i - 1] and vol_arr[i] >= 1.15 * vma20_arr[i]:
                        pos, entry_p, trough_p, stop_p = -1, c, c, c + 0.95 * atr
                        entered = True

        elif sig_id == "B02L":
            # 恐慌长下影锤子确认反转：前一根在 MA20 下方出现放量长下影 (>2倍实体)，当根阳线突破前高
            if pos == 0 and l_arr[i - 1] < bdn_arr[i - 1] and lshad_arr[i - 1] >= 2.0 * max(body_arr[i - 1], 1e-5):
                if c > o and c > h_arr[i - 1] and vol_arr[i - 1] >= 1.15 * vma20_arr[i - 1]:
                    pos, entry_p, peak_p, stop_p = 1, c, c, c - 0.95 * atr
                    entered = True

        elif sig_id == "B02S":
            # 亢奋长上影流星确认反杀：前一根在布林上轨上方出现放量长上影 (>2倍实体)，当根阴线跌破前低
            if pos == 0 and h_arr[i - 1] > bup_arr[i - 1] and ushad_arr[i - 1] >= 2.0 * max(body_arr[i - 1], 1e-5):
                if c < o and c < l_arr[i - 1] and vol_arr[i - 1] >= 1.15 * vma20_arr[i - 1]:
                    pos, entry_p, trough_p, stop_p = -1, c, c, c + 0.95 * atr
                    entered = True

        elif sig_id == "B03L":
            # 超跌底分型停顿放量反抽：前一根 Bias20 <= -0.025*scale 构成底分型低点，当根放量阳线突破前高
            if pos == 0 and bias20_arr[i - 1] <= -0.025 * scale and l_arr[i - 1] < l_arr[i - 2]:
                if c > o and c > h_arr[i - 1] and vol_arr[i] >= 1.15 * vol_arr[i - 1]:
                    pos, entry_p, peak_p, stop_p = 1, c, c, c - 0.95 * atr
                    entered = True

        elif sig_id == "B03S":
            # 超涨顶分型停顿放量回落：前一根 Bias20 >= +0.025*scale 构成顶分型高点，当根放量阴线跌破前低
            if pos == 0 and bias20_arr[i - 1] >= 0.025 * scale and h_arr[i - 1] > h_arr[i - 2]:
                if c < o and c < l_arr[i - 1] and vol_arr[i] >= 1.15 * vol_arr[i - 1]:
                    pos, entry_p, trough_p, stop_p = -1, c, c, c + 0.95 * atr
                    entered = True

        # === 方向 C：震荡回归簇 ===
        elif sig_id == "C01L":
            # 震荡市布林下轨 2B 收回 (MA60 斜率平缓，前根刺破下轨，当根阳线站回下轨之上且过前高)
            if pos == 0 and ma60_slp[i] <= 0.015 * scale and l_arr[i - 1] < bdn_arr[i - 1]:
                if c > bdn_arr[i] and c > o and c >= h_arr[i - 1]:
                    pos, entry_p, peak_p, stop_p = 1, c, c, c - 0.85 * atr
                    entered = True

        elif sig_id == "C01S":
            # 震荡市布林上轨 2B 跌回
            if pos == 0 and ma60_slp[i] <= 0.015 * scale and h_arr[i - 1] > bup_arr[i - 1]:
                if c < bup_arr[i] and c < o and c <= l_arr[i - 1]:
                    pos, entry_p, trough_p, stop_p = -1, c, c, c + 0.85 * atr
                    entered = True

        elif sig_id == "C02L":
            # 宽幅箱体 RSI 超卖勾头反包
            if pos == 0 and rsi_arr[i - 1] <= 31.0 and rsi_arr[i] > rsi_arr[i - 1] + 4.0:
                if c > o and c > h_arr[i - 1] and c > ma60 * (1 - 0.025 * scale):
                    pos, entry_p, peak_p, stop_p = 1, c, c, c - 0.85 * atr
                    entered = True

        elif sig_id == "C02S":
            # 宽幅箱体 RSI 超买勾头反杀
            if pos == 0 and rsi_arr[i - 1] >= 69.0 and rsi_arr[i] < rsi_arr[i - 1] - 4.0:
                if c < o and c < l_arr[i - 1] and c < ma60 * (1 + 0.025 * scale):
                    pos, entry_p, trough_p, stop_p = -1, c, c, c + 0.85 * atr
                    entered = True

        elif sig_id == "C03L":
            # Keltner 下轨企稳回归中轨 (前根触及 Keltner 下轨但未放巨量崩盘，当根阳线过前高)
            if pos == 0 and l_arr[i - 1] <= kcdn_arr[i - 1] and c_arr[i - 1] >= kcdn_arr[i - 1] * (1 - 0.003 * scale):
                if c > o and c > h_arr[i - 1] and macd_arr[i] > macd_arr[i - 1]:
                    pos, entry_p, peak_p, stop_p = 1, c, c, c - 0.85 * atr
                    entered = True

        elif sig_id == "C03S":
            # Keltner 上轨受阻回归中轨
            if pos == 0 and h_arr[i - 1] >= kcup_arr[i - 1] and c_arr[i - 1] <= kcup_arr[i - 1] * (1 + 0.003 * scale):
                if c < o and c < l_arr[i - 1] and macd_arr[i] < macd_arr[i - 1]:
                    pos, entry_p, trough_p, stop_p = -1, c, c, c + 0.85 * atr
                    entered = True

        # === 方向 D：日内分时微结构簇 ===
        elif sig_id == "D01L":
            # 早盘 ORB 首区放量上破
            if pos == 0 and bmax_arr[i] >= 8 and bidx_arr[i] in (1, 2, 3):
                orb_rng = (orbh_arr[i] - orbl_arr[i]) / max(orbl_arr[i], 1e-5)
                if (
                    orb_rng <= 0.038 * scale
                    and c > orbh_arr[i]
                    and c > o
                    and c > ma60
                    and c > vwap_arr[i]
                    and vol_arr[i] >= 1.25 * vma20_arr[i]
                ):
                    pos, entry_p, peak_p, stop_p = 1, c, c, c - 0.85 * atr
                    entered = True

        elif sig_id == "D01S":
            # 早盘 ORB 首区放量下破
            if pos == 0 and bmax_arr[i] >= 8 and bidx_arr[i] in (1, 2, 3):
                orb_rng = (orbh_arr[i] - orbl_arr[i]) / max(orbl_arr[i], 1e-5)
                if (
                    orb_rng <= 0.038 * scale
                    and c < orbl_arr[i]
                    and c < o
                    and c < ma60
                    and c < vwap_arr[i]
                    and vol_arr[i] >= 1.25 * vma20_arr[i]
                ):
                    pos, entry_p, trough_p, stop_p = -1, c, c, c + 0.85 * atr
                    entered = True

        elif sig_id == "D02L":
            # 日内 VWAP 回踩不破阳包
            if pos == 0 and bmax_arr[i] >= 8 and bidx_arr[i] >= 2 and c > ma60 and ma20 > ma60:
                if l_arr[i - 1] <= vwap_arr[i - 1] * (1 + 0.002 * scale) and c_arr[i - 1] >= vwap_arr[i - 1] * (
                    1 - 0.002 * scale
                ):
                    if c > o and c > h_arr[i - 1] and c > vwap_arr[i] and vol_arr[i] >= 1.20 * vol_arr[i - 1]:
                        pos, entry_p, peak_p, stop_p = 1, c, c, c - 0.85 * atr
                        entered = True

        elif sig_id == "D02S":
            # 日内 VWAP 反抽受阻阴杀
            if pos == 0 and bmax_arr[i] >= 8 and bidx_arr[i] >= 2 and c < ma60 and ma20 < ma60:
                if h_arr[i - 1] >= vwap_arr[i - 1] * (1 - 0.002 * scale) and c_arr[i - 1] <= vwap_arr[i - 1] * (
                    1 + 0.002 * scale
                ):
                    if c < o and c < l_arr[i - 1] and c < vwap_arr[i] and vol_arr[i] >= 1.20 * vol_arr[i - 1]:
                        pos, entry_p, trough_p, stop_p = -1, c, c, c + 0.85 * atr
                        entered = True

        elif sig_id == "D03L":
            # 午盘异动放量突破早盘高点
            half_b = bmax_arr[i] // 2
            if (
                pos == 0
                and bmax_arr[i] >= 8
                and bidx_arr[i] in (half_b, half_b + 1)
                and (
                    c > amh_arr[i]
                    and c_arr[i - 1] <= amh_arr[i]
                    and c > ma60
                    and c > o
                    and vol_arr[i] >= 1.35 * vma20_arr[i]
                )
            ):
                pos, entry_p, peak_p, stop_p = 1, c, c, c - 0.85 * atr
                entered = True

        elif sig_id == "D03S":
            # 午盘异动放量跌破早盘低点
            half_b = bmax_arr[i] // 2
            if (
                pos == 0
                and bmax_arr[i] >= 8
                and bidx_arr[i] in (half_b, half_b + 1)
                and (
                    c < aml_arr[i]
                    and c_arr[i - 1] >= aml_arr[i]
                    and c < ma60
                    and c < o
                    and vol_arr[i] >= 1.35 * vma20_arr[i]
                )
            ):
                pos, entry_p, trough_p, stop_p = -1, c, c, c + 0.85 * atr
                entered = True

        # === 方向 A：顺势突破簇补充候选 ===
        elif sig_id == "A01L":
            if pos == 0 and c > ma60 and amp_arr[i] <= box_amp_th * 1.15:
                if obv_arr[i] > obv_h20[i] and c > h10_arr[i] and c > o and vol_arr[i] >= 1.25 * vma20_arr[i]:
                    pos, entry_p, peak_p, stop_p = 1, c, c, l - 0.002 * scale
                    entered = True

        elif sig_id == "A01S":
            if pos == 0 and c < ma60 and amp_arr[i] <= box_amp_th * 1.15:
                if obv_arr[i] < obv_l20[i] and c < l10_arr[i] and c < o and vol_arr[i] >= 1.25 * vma20_arr[i]:
                    pos, entry_p, trough_p, stop_p = -1, c, c, h + 0.002 * scale
                    entered = True

        elif sig_id == "A02L":
            if pos == 0 and c > ma60 and (sqz_arr[i - 1] or sqz_arr[i - 2]):
                if c > bup_arr[i] and c > o and vol_arr[i] >= 1.30 * vma20_arr[i]:
                    pos, entry_p, peak_p, stop_p = 1, c, c, ma20 - 0.002 * scale
                    entered = True

        elif sig_id == "A02S":
            if pos == 0 and c < ma60 and (sqz_arr[i - 1] or sqz_arr[i - 2]):
                if c < bdn_arr[i] and c < o and vol_arr[i] >= 1.30 * vma20_arr[i]:
                    pos, entry_p, trough_p, stop_p = -1, c, c, ma20 + 0.002 * scale
                    entered = True

        elif sig_id == "A03L":
            if (
                (
                    pos == 0
                    and c > ma60
                    and vol_arr[i - 1] <= 0.75 * vma20_arr[i - 1]
                    and body_arr[i - 1] <= 0.55 * atr_arr[i - 1]
                )
                and c > o
                and c > max(h_arr[i - 1], h_arr[i - 2])
                and vol_arr[i] >= 1.30 * vma20_arr[i]
            ):
                pos, entry_p, peak_p, stop_p = 1, c, c, min(l, l_arr[i - 1]) - 0.002 * scale
                entered = True

        elif sig_id == "A03S":
            if (
                (
                    pos == 0
                    and c < ma60
                    and vol_arr[i - 1] <= 0.75 * vma20_arr[i - 1]
                    and body_arr[i - 1] <= 0.55 * atr_arr[i - 1]
                )
                and c < o
                and c < min(l_arr[i - 1], l_arr[i - 2])
                and vol_arr[i] >= 1.30 * vma20_arr[i]
            ):
                pos, entry_p, trough_p, stop_p = -1, c, c, max(h, h_arr[i - 1]) + 0.002 * scale
                entered = True

        elif sig_id == "A04L":
            if pos == 0 and amp_arr[i] > 0.085 * scale and c > ma60 and ma20 > ma60:
                if c > h10_arr[i] and c > o and macd_arr[i] > macd_arr[i - 1] and vol_arr[i] >= 1.25 * vma20_arr[i]:
                    pos, entry_p, peak_p, stop_p = 1, c, c, l - 0.002 * scale
                    entered = True

        elif sig_id == "A04S":
            if pos == 0 and amp_arr[i] > 0.085 * scale and c < ma60 and ma20 < ma60:
                if c < l10_arr[i] and c < o and macd_arr[i] < macd_arr[i - 1] and vol_arr[i] >= 1.25 * vma20_arr[i]:
                    pos, entry_p, trough_p, stop_p = -1, c, c, h + 0.002 * scale
                    entered = True

        # =====================================================================
        # === Round 2 二期新增信号状态机 (聚焦 B/C/D 空白生态位与双向互补) ===
        # =====================================================================
        elif sig_id == "B04L":
            # W双底缩量二次回踩不破前低反包：近期回踩接近10期低点但严格高于20期前低 (抬高底)，缩量止跌后阳线反包
            if pos == 0 and c < ma20:
                recent_low = min(l_arr[i - 1], l_arr[i - 2])
                if recent_low > bbot_arr[i] * (1 + 0.001 * scale) and recent_low <= l10_arr[i] * (1 + 0.004 * scale):
                    if (
                        vol_arr[i - 1] <= 0.92 * vma20_arr[i - 1]
                        and c > o
                        and c > h_arr[i - 1]
                        and macd_arr[i] > macd_arr[i - 1]
                    ):
                        pos, entry_p, peak_p, stop_p = 1, c, c, c - 0.95 * atr
                        entered = True

        elif sig_id == "B04S":
            # M双顶二次冲高受阻放量反杀：近期反抽接近10期高点但严格低于20期前高 (降低顶)，随后放量阴线跌破前低
            if pos == 0 and c > ma20:
                recent_high = max(h_arr[i - 1], h_arr[i - 2])
                if recent_high < btop_arr[i] * (1 - 0.001 * scale) and recent_high >= h10_arr[i] * (1 - 0.004 * scale):
                    if (
                        c < o
                        and c < l_arr[i - 1]
                        and macd_arr[i] < macd_arr[i - 1]
                        and vol_arr[i] >= 1.10 * vma20_arr[i]
                    ):
                        pos, entry_p, trough_p, stop_p = -1, c, c, c + 0.95 * atr
                        entered = True

        elif sig_id == "B05L":
            # 极度超跌三连阴缩量止跌阳包：连续3根阴线将 Bias20 压至超跌区，当根放量阳线反包前高
            if pos == 0 and bias20_arr[i - 1] <= -0.020 * scale:
                if c_arr[i - 1] < o_arr[i - 1] and c_arr[i - 2] < o_arr[i - 2] and c_arr[i - 3] < o_arr[i - 3]:
                    if c > o and c > h_arr[i - 1] and vol_arr[i] >= 1.15 * vol_arr[i - 1]:
                        pos, entry_p, peak_p, stop_p = 1, c, c, c - 0.95 * atr
                        entered = True

        elif sig_id == "B05S":
            # 高位假突破 2B 放量阴吞没反杀：前根或当根刺破 10 期高点后无法站稳，放量中阴线吞没跌破前低
            if (
                pos == 0
                and max(h, h_arr[i - 1]) >= h10_arr[i - 1]
                and ma60_slp[i] <= 0.018 * scale
                and (
                    c < o
                    and c < l_arr[i - 1]
                    and c < o_arr[i - 1]
                    and body_arr[i] >= 0.45 * atr
                    and vol_arr[i] >= 1.15 * vma20_arr[i]
                )
            ):
                pos, entry_p, trough_p, stop_p = -1, c, c, c + 0.95 * atr
                entered = True

        elif sig_id == "C04L":
            # 箱体下沿缩量双针探底回归：箱体震荡期内连续两根K线在下轨区域探底不破，当根阳线向上回归中轨
            if pos == 0 and amp_arr[i] <= 1.35 * box_amp_th and c < ma20:
                if l_arr[i - 1] <= kcdn_arr[i - 1] * (1 + 0.004 * scale) and l >= l_arr[i - 1] * (1 - 0.002 * scale):
                    if (
                        lshad_arr[i - 1] >= 0.8 * body_arr[i - 1]
                        and c > o
                        and c > c_arr[i - 1]
                        and macd_arr[i] >= macd_arr[i - 1]
                    ):
                        pos, entry_p, peak_p, stop_p = 1, c, c, c - 0.85 * atr
                        entered = True

        elif sig_id == "C04S":
            # 弱势箱体反抽 Keltner 上中轨受阻：在中期均线压制下反抽至 MA20 上方受阻，阴线跌破前低回归下沿
            if pos == 0 and c < ma60 and ma20 < ma60 and amp_arr[i] <= 1.35 * box_amp_th:
                if h_arr[i - 1] >= ma20_arr[i - 1] * (1 + 0.003 * scale) and h_arr[i - 1] <= kcup_arr[i - 1] * (
                    1 + 0.005 * scale
                ):
                    if c < o and c < l_arr[i - 1] and macd_arr[i] < macd_arr[i - 1]:
                        pos, entry_p, trough_p, stop_p = -1, c, c, c + 0.85 * atr
                        entered = True

        elif sig_id == "C05L":
            # 通道下轨缩量十字星次根阳线回归：前根在 Keltner 下轨附近收窄幅缩量星线，当根阳线突破前高
            if pos == 0 and l_arr[i - 1] <= kcdn_arr[i - 1] * (1 + 0.003 * scale):
                if body_arr[i - 1] <= 0.38 * rng_arr[i - 1] and vol_arr[i - 1] <= 0.95 * vma20_arr[i - 1]:
                    if c > o and c > h_arr[i - 1] and c < ma20:
                        pos, entry_p, peak_p, stop_p = 1, c, c, c - 0.85 * atr
                        entered = True

        elif sig_id == "C05S":
            # 通道上轨缩量十字星次根阴线回归：前根在 Keltner 上轨附近收窄幅星线，当根阴线跌破前低
            if pos == 0 and h_arr[i - 1] >= kcup_arr[i - 1] * (1 - 0.003 * scale) and ma60_slp[i] <= 0.015 * scale:
                if body_arr[i - 1] <= 0.38 * rng_arr[i - 1] and c < o and c < l_arr[i - 1] and c > ma20:
                    pos, entry_p, trough_p, stop_p = -1, c, c, c + 0.85 * atr
                    entered = True

        elif sig_id == "D04L":
            # 早盘急杀洗盘 V 型放量站回 VWAP：前根急杀偏离日内 VWAP 下方，当根放量阳线强势收复 VWAP 与前高
            if pos == 0 and bmax_arr[i] >= 4 and 1 <= bidx_arr[i] <= max(2, bmax_arr[i] // 2):
                if l_arr[i - 1] <= vwap_arr[i - 1] * (1 - 0.004 * scale) and c > vwap_arr[i]:
                    if c > o and c > h_arr[i - 1] and vol_arr[i] >= 1.15 * vma20_arr[i]:
                        pos, entry_p, peak_p, stop_p = 1, c, c, c - 0.85 * atr
                        entered = True

        elif sig_id == "D04S":
            # 早盘冲高诱多放量跌破 VWAP：前根冲高偏离日内 VWAP 上方，当根放量阴线跌破 VWAP 与前低
            if pos == 0 and bmax_arr[i] >= 4 and 1 <= bidx_arr[i] <= max(2, bmax_arr[i] // 2):
                if h_arr[i - 1] >= vwap_arr[i - 1] * (1 + 0.004 * scale) and c < vwap_arr[i] and c < ma60:
                    if c < o and c < l_arr[i - 1] and vol_arr[i] >= 1.15 * vma20_arr[i]:
                        pos, entry_p, trough_p, stop_p = -1, c, c, c + 0.85 * atr
                        entered = True

        elif sig_id == "D05L":
            # 午后回踩 VWAP 缩量企稳拉升：午后多头趋势中缩量回踩 VWAP 不破，随后阳线突破前高
            half_b = max(1, bmax_arr[i] // 2)
            if pos == 0 and bmax_arr[i] >= 4 and bidx_arr[i] >= half_b and c > ma60:
                if l_arr[i - 1] <= vwap_arr[i - 1] * (1 + 0.003 * scale) and c_arr[i - 1] >= vwap_arr[i - 1] * (
                    1 - 0.002 * scale
                ):
                    if vol_arr[i - 1] <= 0.95 * vma20_arr[i - 1] and c > o and c > h_arr[i - 1] and c > vwap_arr[i]:
                        pos, entry_p, peak_p, stop_p = 1, c, c, c - 0.85 * atr
                        entered = True

        elif sig_id == "D05S":
            # 午后反抽 VWAP 无力放量杀跌：午后弱势格局下反抽至 VWAP 受阻，当根放量阴线跌破前低
            half_b = max(1, bmax_arr[i] // 2)
            if pos == 0 and bmax_arr[i] >= 4 and bidx_arr[i] >= half_b and c < ma60:
                if h_arr[i - 1] >= vwap_arr[i - 1] * (1 - 0.003 * scale) and c_arr[i - 1] <= vwap_arr[i - 1] * (
                    1 + 0.002 * scale
                ):
                    if c < o and c < l_arr[i - 1] and c < vwap_arr[i] and vol_arr[i] >= 1.10 * vol_arr[i - 1]:
                        pos, entry_p, trough_p, stop_p = -1, c, c, c + 0.85 * atr
                        entered = True

        # =====================================================================
        # === Round 3 & Round 4 原生 Rust `czsc-signals` (含 5m 专属) 组合信号状态机 ===
        # =====================================================================
        elif cz_trig_arr is not None:
            if pos == 0 and bool(cz_trig_arr[i]):
                base_sl = 1.10 if spec.exit_mode == "trend" else (0.95 if spec.exit_mode == "reversal" else 0.85)
                stop_mult = base_sl * sl_scale
                if spec.direction == 1:
                    pos, entry_p, peak_p, stop_p = 1, c, c, c - stop_mult * atr
                else:
                    pos, entry_p, trough_p, stop_p = -1, c, c, c + stop_mult * atr
                entered = True

        if entered:
            triggers[i] = True
            entry_idx = i
        weights[i] = float(pos)

    if cz_trig_arr is not None:
        triggers[:] = False
        triggers[start_bar:] = cz_trig_arr[start_bar:]

    return triggers, weights
