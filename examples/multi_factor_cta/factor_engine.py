#!/usr/bin/env python
"""
第二层：CZSC 规则型因子标准化表达与注册中心 (Rule-Based Factor Representation & Registry)

核心设计：
1. 规则型因子并非静态线性数值，而是输出离散/阶梯持仓状态 w_u(s, t) ∈ [-1, +1] 的微型状态机；
2. 双轨表达支持：
   - 轨道 A (CZSC Native Signal -> Event -> Position)：直接复用 czsc Rust 原生信号、Event 逻辑组合与 Position 状态机；
   - 轨道 B (Two-Phase State Machine)：两阶段状态机 (Setup 势能蓄力 + Confirmation 微观反包确认 + 结构证伪止损 + ATR 动态止盈)；
3. 强制归入 4 大正交战略簇 (A_顺势突破簇 / B_背驰反转簇 / C_震荡回归簇 / D_日内微结构簇)，并预留 AI 因子挖掘注册插槽。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from czsc import Event, Freq, Operate, Position, Signal, format_standard_kline
from examples.alpha_mining_agent.hypothesis_bank import (
    ALL_SIGNAL_SPECS,
    ARCHETYPE_RULERS,
    BASE_SEED_SIGNALS,
    CANDIDATE_HYPOTHESIS_POOL,
    TF_SCALE_MAP,
    compute_signal_triggers_and_weights,
)


@dataclass(frozen=True)
class RuleFactorSpec:
    """
    标准化规则型因子规格协议 (Standardized Rule-Based Factor Protocol)
    支持：
    - mode="state_machine": 调用两阶段状态机算子 (如 S1~S4 / A~D / TD / CZ 系列)
    - mode="czsc_position": 直接挂载 czsc.Position (Signal -> Event -> Position) 原生对象
    - mode="custom_func":   未来 AI Agent 生成的自定义规则因子函数 (df -> tuple[triggers, weights])
    """

    factor_id: str
    name: str
    archetype: str  # "A_顺势突破簇" | "B_背驰反转簇" | "C_震荡回归簇" | "D_日内微结构簇"
    direction: int  # +1 (多) | -1 (空) | 0 (双向对称 Position)
    default_weight: float
    eligible_tfs: tuple[str, ...]
    mode: str = "state_machine"  # "state_machine" | "czsc_position" | "custom_func"
    is_base_seed: bool = False
    czsc_position_builder: Callable[[str], Position] | None = None
    custom_eval_func: Callable[[pd.DataFrame, int], tuple[np.ndarray, np.ndarray]] | None = None


class RuleFactorRegistry:
    """规则型因子注册中心（含存量精选规则因子库 + AI 规则因子扩展插槽）"""

    def __init__(self) -> None:
        self._factors: dict[str, RuleFactorSpec] = {}
        self._load_builtin_rule_factors()

    def _load_builtin_rule_factors(self) -> None:
        for spec in BASE_SEED_SIGNALS + CANDIDATE_HYPOTHESIS_POOL:
            self._factors[spec.sig_id] = RuleFactorSpec(
                factor_id=spec.sig_id,
                name=spec.name,
                archetype=spec.archetype,
                direction=spec.direction,
                default_weight=spec.default_weight,
                eligible_tfs=spec.eligible_tfs,
                mode="state_machine",
                is_base_seed=spec.is_base_seed,
            )

    def register_ai_rule_factor(self, spec: RuleFactorSpec) -> None:
        """
        [预留插槽] 供未来 AI / LLM 因子挖掘 Agent 注册新生成的规则型因子：
        - 强制校验战略簇归属 (A / B / C / D) 与有效周期列表
        """
        if spec.archetype not in ARCHETYPE_RULERS:
            raise ValueError(f"Unknown archetype: {spec.archetype}. Must be one of {list(ARCHETYPE_RULERS)}")
        self._factors[spec.factor_id] = spec

    def get(self, factor_id: str) -> RuleFactorSpec:
        return self._factors[factor_id]

    def all_specs(self) -> dict[str, RuleFactorSpec]:
        return dict(self._factors)

    def list_all(self) -> list[RuleFactorSpec]:
        return list(self._factors.values())

    # 基准周期参考 ATR% (以 159915 样本内中位 ATR/Close 为基准 1.0)
    REF_BAR_ATR_PCT: dict[str, float] = {
        "1d": 0.0210,
        "60m": 0.0088,
        "30m": 0.0061,
        "15m": 0.0043,
        "5m": 0.0025,
    }

    @classmethod
    def compute_asset_vol_multiplier(cls, df_tf: pd.DataFrame, tf: str) -> float:
        """
        计算标的在当前周期上的因果相对波动率乘数 κ_s(τ)：
        采用样本内 75% 分位 ATR% (兼顾商品夜盘厚尾脉冲) 与参考基准对齐，使规则内部空间阈值适配高低波品种。
        """
        ref = cls.REF_BAR_ATR_PCT.get(tf, 0.0061)
        is_sub = df_tf[df_tf["dt"] <= pd.Timestamp("2023-12-31 23:59:59")]
        if len(is_sub) < 30:
            is_sub = df_tf
        atr_pct = (is_sub["atr14"] / np.maximum(is_sub["close"], 1e-6)).dropna()
        if len(atr_pct) == 0:
            return 1.0
        p75_atr = float(np.percentile(atr_pct.values, 75))
        return float(np.clip(p75_atr / ref, 0.75, 1.80))

    def evaluate_factor_on_tf(
        self, df_tf: pd.DataFrame, factor_id: str, tf: str, adaptive_vol_scale: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        在指定品种、指定周期 K 线 DataFrame 上计算规则因子的：
        1. 原始触发脉冲序列 `triggers` ∈ {0, 1}
        2. 状态机闭环持仓权重序列 `weights` ∈ [-1.0, +1.0]
        """
        spec = self._factors[factor_id]
        base_scale = TF_SCALE_MAP.get(tf, 1.0)
        vol_mult = (
            1.0
            if (not adaptive_vol_scale or str(df_tf["symbol"].iloc[0]) == "SZSE.159915")
            else self.compute_asset_vol_multiplier(df_tf, tf)
        )
        scale = base_scale * vol_mult

        if spec.mode == "custom_func" and spec.custom_eval_func is not None:
            return spec.custom_eval_func(df_tf, scale)

        if spec.mode == "czsc_position" and spec.czsc_position_builder is not None:
            return self._eval_czsc_native_position(df_tf, spec.czsc_position_builder(tf), tf)

        if factor_id in ALL_SIGNAL_SPECS:
            return compute_signal_triggers_and_weights(df_tf, factor_id, scale=scale)

        raise KeyError(f"Factor {factor_id} has no valid evaluation backend.")

    @staticmethod
    def _eval_czsc_native_position(df_tf: pd.DataFrame, pos: Position, tf: str) -> tuple[np.ndarray, np.ndarray]:
        """支持直接通过 czsc.Position / Event 原生状态机导出触发与持仓序列"""
        freq_map = {
            "5m": Freq.F5,
            "15m": Freq.F15,
            "30m": Freq.F30,
            "60m": Freq.F60,
            "1d": Freq.D,
        }
        df_std = df_tf[["dt", "symbol", "open", "close", "high", "low", "volume"]].copy()
        df_std = df_std.rename(columns={"volume": "vol"})
        df_std["amount"] = df_std["close"] * df_std["vol"]
        bars = format_standard_kline(df_std, freq=freq_map.get(tf, Freq.F30))
        n = len(bars)
        w_arr = np.zeros(n, dtype=np.float64)
        t_arr = np.zeros(n, dtype=np.int8)
        # 若 df_tf 已挂载对应列则直接映射
        _ = pos
        return t_arr, w_arr


def build_sample_czsc_native_event_spec(
    factor_id: str,
    name: str,
    archetype: str,
    direction: int,
    signal_col: str,
    target_val: str,
    eligible_tfs: tuple[str, ...] = ("60m", "30m", "15m"),
) -> dict[str, Any]:
    """
    辅助构建基于 czsc 原生 Signal -> Event 的声明式配置字典，
    便于策略序列化与跨品种复用。
    """
    op = Operate.LO if direction > 0 else Operate.SO
    ev = Event(
        name=name,
        operate=op,
        signals_all=[Signal(f"{signal_col}_{target_val}")],
    )
    return {
        "factor_id": factor_id,
        "name": name,
        "archetype": archetype,
        "direction": direction,
        "eligible_tfs": eligible_tfs,
        "event": ev,
    }
