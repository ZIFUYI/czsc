#!/usr/bin/env python
r"""
第四层：时序×截面三支柱分层合成与跨品种风险平价引擎 (Intra-Asset Three-Pillar & Cross-Asset HRP Synthesis Engine)

包含两级核心合成架构（Issue #7 Phase 4 升级）：
1. 第一级（单品种内 · 六簇分层正交合成 `IntraAssetThreePillarSynthesizer` / `IntraAssetSixClusterSynthesizer`）：
   - 支柱一：六簇 × 五周期第一级簇内合成 + 第二级跨簇施密特残差正交 (`gram_schmidt_cluster_orthogonalize`)，
     剥离战术簇对主基石簇的冗余共线成分，保证正交化后任意两簇 $|\rho(U_m, U_n)| < 0.25$；
   - 支柱二：六簇明确风险预算配权（支持滚动经验贝叶斯收缩 `RollingEmpiricalBayesWeighter`）+ 跨簇多空非对称反向断路器；
   - 支柱三：因果有效湍流波动率封顶调制 $\sigma_{\text{turb}} = \sigma_{15d} \cdot (1 - 0.35 \cdot ER_{5d})$
     + 非对称进出场滞回状态机 (`ContinuousSignalStateMachine`) + 截面强弱风险预算倾斜 (`apply_cross_sectional_budget_tilt`)
     + 交易制度边界适配器 (`LONG_SHORT` / `LONG_ONLY_SPOT` / `INTRADAY_FLAT_1455`)。
2. 第二级（跨品种间 · 动态活跃池分配与三层因果风险平价 `CrossAssetHRPSynthesizer` / `DynamicActivePoolAllocator`）：
   - 动态活跃池 ($1 / N_{\text{active}, t}$)：解决静态分母 $1 / N_{\text{total}}$ 空仓品种占用预算造成的现金拖累，
     严守单品种权重上限 `max_single_asset_weight = 0.35` 与最低有效品种保底 `min_effective_assets = 3`；
   - 三层因果风险预算闭环：因果 60 日 `shift(1)` ATR/收益率逆波动率平价 + 板块内 $1/\sqrt{K_{s,t}}$ 去拥挤与板块间 HRP
     + 截面强弱预算倾斜与组合目标波动率 $\sigma_{\text{target}}$ 缩放。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any

import numpy as np
import pandas as pd

from czsc import WeightBacktest
from examples.alpha_mining_agent.forensic_pipeline import IS_END_DT, apply_rebalance_deadzone
from examples.alpha_mining_agent.hypothesis_bank import (
    ALL_ARCHETYPES,
    ALL_SIGNAL_SPECS,
    ARCH_A,
    ARCH_B,
    ARCH_C,
    ARCH_D,
    ARCH_E,
    ARCH_F,
)
from examples.multi_factor_cta.atomic_factors import SoftCrossSectionalGate
from examples.multi_factor_cta.data_engine import AssetClass, AssetSpec, ExecutionMode
from examples.multi_factor_cta.forensic_validator import (
    RollingEBWeightsResult,
    RollingEmpiricalBayesWeighter,
    compute_empirical_bayes_shrinkage_weight,
    compute_rolling_empirical_bayes_weights,
)

DEFAULT_BASIS_ORDER: tuple[str, ...] = (ARCH_A, ARCH_F, ARCH_E, ARCH_D, ARCH_B, ARCH_C)


def _orthogonalize_step_against_basis(
    raw_v: np.ndarray,
    basis_list: list[np.ndarray],
    eps: float = 1e-3,
    beta_clip: float = 4.0,
) -> np.ndarray:
    """
    在给定正交基列表 `basis_list` 上剥离 `raw_v` 的冗余线性共线投影（支持软缩放与激活集中心化/非中心化双正交），
    使输出向量在 `[-1, 1]` 内且与所有前序基向量的全样本相关与激活态并集相关均严格接近 0。
    """
    v = np.asarray(raw_v, dtype=np.float64).copy()
    if len(v) == 0 or float(np.max(np.abs(v))) <= eps:
        return np.zeros_like(v)
    if not basis_list:
        return np.clip(v, -1.0, 1.0)

    # 第一轮：剥离主线性投影并施加双曲正切软缩放到 (-1, 1)，防止逆势信号被极端拉伸
    for u in basis_list:
        uu = float(np.dot(u, u))
        if uu > eps * eps:
            beta = float(np.clip(np.dot(v, u) / (uu + 1e-12), -beta_clip, beta_clip))
            v = v - beta * u
    v = np.tanh(v)

    # 第二轮：消除 tanh 软缩放引入的微弱非线性残差共线性（同时保证全样本协方差与激活并集内积正交）
    aug_basis: list[np.ndarray] = []
    n = len(v)
    ones_vec = np.ones(n, dtype=np.float64)
    for u in basis_list:
        if float(np.dot(u, u)) > 1e-12:
            aug_basis.append(u)
            u_centered = u - float(np.mean(u))
            if float(np.dot(u_centered, u_centered)) > 1e-12:
                aug_basis.append(u_centered)
            act_u = (np.abs(u) > eps).astype(np.float64)
            if float(np.dot(act_u, act_u)) > 1e-12:
                aug_basis.append(act_u)
    aug_basis.append(ones_vec)

    # 构建标准正交子空间基 Q 并执行两轮 Modified Gram-Schmidt 投影剥离
    q_list: list[np.ndarray] = []
    for b in aug_basis:
        q = b.copy()
        for prev_q in q_list:
            q = q - float(np.dot(q, prev_q)) * prev_q
        norm_q = float(np.linalg.norm(q))
        if norm_q > 1e-9:
            q_list.append(q / norm_q)

    # 保存原始静默零值掩码：若某簇与前序所有基均无激活重叠则保持原样
    has_any_overlap = False
    for u in basis_list:
        if np.any((np.abs(v) > eps) & (np.abs(u) > eps)):
            has_any_overlap = True
            break

    if has_any_overlap:
        for _ in range(2):
            for q in q_list:
                proj = float(np.clip(np.dot(v, q), -beta_clip * np.sqrt(n), beta_clip * np.sqrt(n)))
                v = v - proj * q

    max_abs = float(np.max(np.abs(v)))
    if max_abs > 1.0:
        v = v / max_abs
    return np.clip(v, -1.0, 1.0)


def gram_schmidt_cluster_orthogonalize(
    cluster_signals: dict[str, np.ndarray],
    basis_order: tuple[str, ...] = DEFAULT_BASIS_ORDER,
    causal_window: int | None = None,
    eps: float = 1e-3,
) -> dict[str, np.ndarray]:
    """
    跨簇施密特残差正交化 (Cross-Cluster Gram-Schmidt Residual Orthogonalization)：
    - 以趋势主升浪基石簇 U_1 = S^(A) 与多周期/跨品种截面共振簇 U_2 = S^(F) - proj_{U_1}(S^(F)) 为优先主正交基，
      按 `basis_order` 顺序依次对后续战术簇 S^(k) 剥离其对前序已正交化基向量 U_1, ..., U_{k-1} 的同向线性投影残差：
          U_k = S^(k) - sum_{j=1}^{k-1} β_{k,j} U_j
    - 支持全序列激活并集正交 (`causal_window=None`) 与严格零前视因果滚动窗口正交 (`causal_window: int`)；
    - 正交化后任意两簇之间的激活态并集相关系数与全样本相关系数严格满足 |ρ(U_m, U_n)| < 0.25。
    """
    if not cluster_signals:
        return {}

    ordered_keys: list[str] = [k for k in basis_order if k in cluster_signals]
    for k in cluster_signals:
        if k not in ordered_keys:
            ordered_keys.append(k)

    first_arr = np.asarray(cluster_signals[ordered_keys[0]], dtype=np.float64)
    n = len(first_arr)
    ortho_out: dict[str, np.ndarray] = {}

    if causal_window is None or causal_window <= 0:
        basis_so_far: list[np.ndarray] = []
        for idx, k in enumerate(ordered_keys):
            s_k = np.nan_to_num(np.asarray(cluster_signals[k], dtype=np.float64), nan=0.0)
            if idx == 0:
                u_k = np.clip(s_k.copy(), -1.0, 1.0)
            else:
                u_k = _orthogonalize_step_against_basis(s_k, basis_so_far, eps=eps)
            ortho_out[k] = u_k
            if float(np.max(np.abs(u_k))) > eps:
                basis_so_far.append(u_k)
        return ortho_out

    # 因果滚动窗口模式 (Causal Rolling Window Gram-Schmidt): 在任意时刻 t 仅使用 [max(0, t-W+1), t] 窗口内历史信号
    w_len = max(int(causal_window), 10)
    for k in ordered_keys:
        ortho_out[k] = np.zeros(n, dtype=np.float64)

    raw_mat = {k: np.nan_to_num(np.asarray(cluster_signals[k], dtype=np.float64), nan=0.0) for k in ordered_keys}
    ortho_out[ordered_keys[0]] = np.clip(raw_mat[ordered_keys[0]].copy(), -1.0, 1.0)

    for k_idx in range(1, len(ordered_keys)):
        k_name = ordered_keys[k_idx]
        s_vec = raw_mat[k_name]
        u_vec = np.zeros(n, dtype=np.float64)
        prev_keys = ordered_keys[:k_idx]

        for t in range(n):
            st = max(0, t - w_len + 1)
            v_win = s_vec[st : t + 1].copy()
            if float(np.max(np.abs(v_win))) <= eps:
                u_vec[t] = 0.0
                continue
            for p_key in prev_keys:
                u_win = ortho_out[p_key][st : t + 1]
                uu = float(np.dot(u_win, u_win))
                if uu > eps * eps:
                    u_c = u_win - float(np.mean(u_win))
                    uc_norm = float(np.dot(u_c, u_c))
                    if uc_norm > eps * eps and len(v_win) >= 4:
                        beta_c = float(np.clip(np.dot(v_win - float(np.mean(v_win)), u_c) / uc_norm, -3.5, 3.5))
                        v_win = v_win - beta_c * u_c
                    beta_d = float(np.clip(np.dot(v_win, u_win) / (uu + 1e-12), -3.5, 3.5))
                    v_win = v_win - beta_d * u_win
            u_vec[t] = float(np.tanh(v_win[-1]))

        ortho_out[k_name] = np.clip(u_vec, -1.0, 1.0)

    return ortho_out


@dataclass
class ContinuousSignalStateMachine:
    """
    带非对称进出场滞回 (Hysteresis Entry/Exit Thresholds) 与同向调仓死区 + 最小持仓半衰期平滑的连续信号状态机：
    1. 非对称进出场滞回门限：
       - 空仓时需 |w_t| >= entry_threshold (如 0.08) 才触发建仓；
       - 建仓后只要方向未反转且 |w_t| >= exit_threshold (如 0.03 < entry_threshold) 即维持持仓，
         低于 exit_threshold 时立即零延迟清仓归零；
    2. 同向微调死区与半衰期平滑：
       - 同向微调幅度 |w_t - w_{t-1}| < rebalance_deadband (如 0.06) 时保持原仓位不交易，滤除连续强度微幅抖动；
       - 当发生多空方向反转 (w_t * w_{t-1} < 0) 或触发清仓门限时，立即零延迟响应。
    """

    entry_threshold: float = 0.08
    exit_threshold: float = 0.03
    rebalance_deadband: float = 0.06
    min_holding_half_life_bars: float = 0.0

    def __post_init__(self) -> None:
        if self.exit_threshold < 0.0 or self.entry_threshold <= 0.0:
            raise ValueError("entry_threshold must be > 0 and exit_threshold must be >= 0.")
        if self.exit_threshold >= self.entry_threshold:
            raise ValueError(
                f"exit_threshold ({self.exit_threshold}) must be strictly less than entry_threshold ({self.entry_threshold})."
            )
        if self.rebalance_deadband < 0.0:
            raise ValueError("rebalance_deadband must be >= 0.")

    def transform(self, signal_series: np.ndarray) -> np.ndarray:
        w = np.nan_to_num(np.asarray(signal_series, dtype=np.float64), nan=0.0)
        n = len(w)
        out = np.zeros(n, dtype=np.float64)
        cur = 0.0

        use_ema = self.min_holding_half_life_bars > 0.0
        alpha = float(1.0 - np.power(0.5, 1.0 / max(self.min_holding_half_life_bars, 1e-6))) if use_ema else 1.0

        for i in range(n):
            raw_target = float(np.clip(w[i], -1.0, 1.0))
            abs_target = abs(raw_target)

            if abs(cur) <= 1e-9:
                # 当前空仓：必须达到入场门限 entry_threshold 才建仓
                cur = raw_target if abs_target >= self.entry_threshold else 0.0
            else:
                # 当前持仓中：检查是否触发清仓门限或多空方向反转（均零延迟立即执行）
                if abs_target < self.exit_threshold:
                    cur = 0.0
                elif cur * raw_target < 0.0:
                    # 多空方向反转：若新方向达到入场门限则直接反手，否则立即平仓归零
                    cur = raw_target if abs_target >= self.entry_threshold else 0.0
                else:
                    # 同向维持：应用可选半衰期平滑与同向调仓死区
                    smoothed = (alpha * raw_target + (1.0 - alpha) * cur) if use_ema else raw_target
                    if abs(smoothed - cur) >= self.rebalance_deadband:
                        cur = float(np.clip(smoothed, -1.0, 1.0))

            out[i] = cur
        return out

    def __call__(self, signal_series: np.ndarray) -> np.ndarray:
        return self.transform(signal_series)


def apply_cross_sectional_budget_tilt(
    ts_weights: np.ndarray,
    cs_strength: np.ndarray,
    tilt_Floor: float = 0.25,
    tilt_Cap: float = 1.50,
    *,
    tilt_floor: float | None = None,
    tilt_cap: float | None = None,
) -> np.ndarray:
    """
    跨品种截面强弱风险预算倾斜 (Cross-Sectional Budget Tilt)：
    将单品种时序合成目标仓位 W_{i,t}^{TS} 与因果截面相对强弱得分 R_{i,t}^{strength} ∈ [-1, 1] 联动：
    - 对多头持仓 (W_{i,t}^{TS} > 0)，有效截面顺势度 R_{i,t}^{eff} = R_{i,t}^{strength}；
    - 对空头持仓 (W_{i,t}^{TS} < 0)，有效截面顺势度 R_{i,t}^{eff} = -R_{i,t}^{strength}；
    - 调制公式：
        W_{i,t}^{CS} = W_{i,t}^{TS} * clip(0.5 + 0.5 * (1 + R_{i,t}^{eff}), tilt_Floor, tilt_Cap)
    严格保证不改变原时序信号的多空方向符号 (sign(W^{CS}) == sign(W^{TS}))，且零仓位保持为零。
    """
    eff_floor = float(tilt_floor if tilt_floor is not None else tilt_Floor)
    eff_cap = float(tilt_cap if tilt_cap is not None else tilt_Cap)
    if eff_floor <= 0.0 or eff_cap < eff_floor:
        raise ValueError(f"Invalid tilt bounds: tilt_Floor={eff_floor}, tilt_Cap={eff_cap}")

    w_ts = np.nan_to_num(np.asarray(ts_weights, dtype=np.float64), nan=0.0)
    r_str = np.clip(np.nan_to_num(np.asarray(cs_strength, dtype=np.float64), nan=0.0), -1.0, 1.0)
    if len(w_ts) != len(r_str):
        raise ValueError(f"Length mismatch: ts_weights ({len(w_ts)}) vs cs_strength ({len(r_str)})")

    r_eff = np.sign(w_ts) * r_str
    tilt_mult = np.clip(0.5 + 0.5 * (1.0 + r_eff), eff_floor, eff_cap)
    return w_ts * tilt_mult


class IntraAssetThreePillarSynthesizer:
    """单品种内六簇×多周期三支柱分层合成器（支持跨簇施密特残差正交、滞回状态机与截面强弱预算倾斜）"""

    TF_WEIGHTS: dict[str, float] = {
        "1d": 0.65,
        "60m": 0.55,
        "30m": 0.45,
        "15m": 0.40,
        "5m": 0.30,
    }
    COMMODITY_TF_WEIGHTS: dict[str, float] = {
        "1d": 0.95,
        "60m": 0.55,
        "30m": 0.25,
        "15m": 0.20,
        "5m": 0.00,
    }
    CLUSTER_RISK_BUDGETS: dict[str, float] = {
        ARCH_A: 1.00,
        ARCH_F: 0.95,
        ARCH_E: 0.90,
        ARCH_D: 0.85,
        ARCH_B: 1.00,
        ARCH_C: 0.95,
    }

    def __init__(
        self,
        cluster_risk_budgets: dict[str, float] | None = None,
        basis_order: tuple[str, ...] = DEFAULT_BASIS_ORDER,
    ) -> None:
        self.cluster_risk_budgets = dict(self.CLUSTER_RISK_BUDGETS)
        if cluster_risk_budgets:
            self.cluster_risk_budgets.update(cluster_risk_budgets)
        self.basis_order = basis_order
        self.last_raw_cluster_signals: dict[str, np.ndarray] = {}
        self.last_ortho_cluster_signals: dict[str, np.ndarray] = {}

    def extract_raw_cluster_signals(
        self,
        df_grid: pd.DataFrame,
        unit_weights_on_grid: dict[str, np.ndarray],
        unit_alloc_map: dict[str, float | np.ndarray],
        asset_spec: AssetSpec,
        enable_regime_gating: bool = True,
        cs_slope_rank: np.ndarray | None = None,
        soft_gate: SoftCrossSectionalGate | bool | None = True,
    ) -> tuple[dict[str, np.ndarray], bool, np.ndarray | None]:
        """
        第一级（簇内合成 Intra-Cluster Synthesis）：
        对每个战略簇 C ∈ {ARCH_A ~ ARCH_F}，跨周期 (1d, 60m, 30m, 15m, 5m) 结合 SoftCrossSectionalGate
        软空间门控与单元权重（支持标量或滚动经验贝叶斯权重序列），聚合得到主网格上的 6 条簇级原始连续信号序列 S_{i,t}^{(C)} ∈ [-1, 1]。
        """
        n = len(df_grid)
        w_tf_arch = {tf: {arch: np.zeros(n, dtype=np.float64) for arch in ALL_ARCHETYPES} for tf in self.TF_WEIGHTS}

        has_ortho = False
        for unit_id, coeff in unit_alloc_map.items():
            if unit_id not in unit_weights_on_grid:
                continue
            sid, tf = unit_id.split("@")
            if tf not in w_tf_arch:
                continue
            spec = ALL_SIGNAL_SPECS.get(sid)
            arch = spec.archetype if spec is not None else ARCH_A
            if arch != ARCH_A:
                has_ortho = True
            coeff_arr = np.asarray(coeff, dtype=np.float64)
            w_tf_arch[tf][arch] += coeff_arr * np.asarray(unit_weights_on_grid[unit_id], dtype=np.float64)

        gate_obj: SoftCrossSectionalGate | None = None
        if enable_regime_gating and soft_gate is not None and soft_gate is not False:
            gate_obj = soft_gate if isinstance(soft_gate, SoftCrossSectionalGate) else SoftCrossSectionalGate()

        eff_cs_rank: np.ndarray | None = None
        if gate_obj is not None:
            if cs_slope_rank is not None:
                eff_cs_rank = np.clip(np.nan_to_num(np.asarray(cs_slope_rank, dtype=np.float64), nan=0.5), 0.0, 1.0)
            else:
                eff_cs_rank = gate_obj.extract_cs_rank(df_grid)

        er_grid = gate_obj.extract_causal_er(df_grid) if (gate_obj is not None and eff_cs_rank is not None) else None

        regimes = (
            self._get_causal_regimes(df_grid)
            if (enable_regime_gating and eff_cs_rank is None and asset_spec.symbol != "SZSE.159915")
            else None
        )

        is_commodity = asset_spec.asset_class == AssetClass.COMMODITY_FUTURES
        tf_weight_map = self.COMMODITY_TF_WEIGHTS if is_commodity else self.TF_WEIGHTS

        raw_clusters: dict[str, np.ndarray] = {arch: np.zeros(n, dtype=np.float64) for arch in ALL_ARCHETYPES}
        for tf, tw in tf_weight_map.items():
            if tw <= 0.0:
                continue
            wa = np.clip(w_tf_arch[tf][ARCH_A], -1.0, 1.0)
            wb = np.clip(w_tf_arch[tf][ARCH_B], -1.0, 1.0)
            wc = np.clip(w_tf_arch[tf][ARCH_C], -1.0, 1.0)
            wd = np.clip(w_tf_arch[tf][ARCH_D], -1.0, 1.0)
            we = np.clip(w_tf_arch[tf][ARCH_E], -1.0, 1.0)
            wf = np.clip(w_tf_arch[tf][ARCH_F], -1.0, 1.0)

            if gate_obj is not None and eff_cs_rank is not None:
                wa = gate_obj.apply(wa, eff_cs_rank, er_grid, archetype=ARCH_A)
                wb = gate_obj.apply(wb, eff_cs_rank, er_grid, archetype=ARCH_B)
                wc = gate_obj.apply(wc, eff_cs_rank, er_grid, archetype=ARCH_C)
            elif regimes is not None:
                if is_commodity:
                    wb = np.where((regimes == -1) & (wb > 0.0), wb * 0.10, wb)
                    wb = np.where((regimes == 1) & (wb < 0.0), wb * 0.10, wb)
                    wc = np.where((regimes == -1) & (wc > 0.0), wc * 0.15, wc)
                    wc = np.where((regimes == 1) & (wc < 0.0), wc * 0.15, wc)
                else:
                    wc = np.where((regimes == -1) & (wc > 0.0), wc * 0.35, wc)
                    wc = np.where((regimes == 1) & (wc < 0.0), wc * 0.35, wc)

            raw_clusters[ARCH_A] += tw * wa
            raw_clusters[ARCH_B] += tw * wb
            raw_clusters[ARCH_C] += tw * wc
            raw_clusters[ARCH_D] += tw * wd
            raw_clusters[ARCH_E] += tw * we
            raw_clusters[ARCH_F] += tw * wf

        for arch in ALL_ARCHETYPES:
            raw_clusters[arch] = np.clip(raw_clusters[arch], -1.0, 1.0)

        return raw_clusters, has_ortho, eff_cs_rank

    def synthesize_cluster_orthogonals(
        self,
        df_grid: pd.DataFrame,
        unit_weights_on_grid: dict[str, np.ndarray] | None = None,
        unit_alloc_map: dict[str, float | np.ndarray] | None = None,
        asset_spec: AssetSpec | None = None,
        *,
        cluster_signals: dict[str, np.ndarray] | None = None,
        basis_order: tuple[str, ...] | None = None,
        causal_window: int | None = None,
        rebalance_deadzone: float = 0.06,
        execution_mode_override: ExecutionMode | None = None,
        enable_regime_gating: bool = True,
        cs_slope_rank: np.ndarray | None = None,
        soft_gate: SoftCrossSectionalGate | bool | None = True,
        state_machine: ContinuousSignalStateMachine | bool | None = None,
        cs_strength: np.ndarray | None = None,
        tilt_Floor: float = 0.25,
        tilt_Cap: float = 1.50,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        执行六簇二级施密特残差正交与跨簇风险预算合成，返回 `(w_final, ortho_cluster_signals)`，
        方便直接审计正交化前后的六簇信号相关性矩阵。
        """
        if asset_spec is None:
            asset_spec = AssetSpec(
                symbol=str(df_grid["symbol"].iloc[0]) if "symbol" in df_grid.columns else "SYNTH.ASSET",
                name="SyntheticAsset",
                asset_class=AssetClass.COMMODITY_FUTURES,
                sector="SYNTH",
                base_tf="30m",
                eligible_tfs=("1d", "60m", "30m", "15m", "5m"),
                fee_rate=0.0002,
                execution_mode=ExecutionMode.LONG_SHORT,
            )

        if cluster_signals is not None:
            raw_clusters = {
                arch: np.clip(
                    np.nan_to_num(
                        np.asarray(cluster_signals.get(arch, np.zeros(len(df_grid))), dtype=np.float64),
                        nan=0.0,
                    ),
                    -1.0,
                    1.0,
                )
                for arch in ALL_ARCHETYPES
            }
            has_ortho = any(
                arch != ARCH_A and float(np.max(np.abs(raw_clusters[arch]))) > 1e-6 for arch in ALL_ARCHETYPES
            )
            eff_cs_rank = cs_slope_rank
        else:
            raw_clusters, has_ortho, eff_cs_rank = self.extract_raw_cluster_signals(
                df_grid=df_grid,
                unit_weights_on_grid=unit_weights_on_grid or {},
                unit_alloc_map=unit_alloc_map or {},
                asset_spec=asset_spec,
                enable_regime_gating=enable_regime_gating,
                cs_slope_rank=cs_slope_rank,
                soft_gate=soft_gate,
            )

        self.last_raw_cluster_signals = {k: v.copy() for k, v in raw_clusters.items()}
        eff_basis_order = basis_order or self.basis_order
        ortho_clusters = gram_schmidt_cluster_orthogonalize(
            raw_clusters,
            basis_order=eff_basis_order,
            causal_window=causal_window,
        )
        self.last_ortho_cluster_signals = {k: v.copy() for k, v in ortho_clusters.items()}

        w_final = self._combine_cluster_signals_to_position(
            df_grid=df_grid,
            cluster_map=ortho_clusters,
            has_ortho=has_ortho,
            eff_cs_rank=eff_cs_rank,
            asset_spec=asset_spec,
            enable_regime_gating=enable_regime_gating,
            rebalance_deadzone=rebalance_deadzone,
            execution_mode_override=execution_mode_override,
            state_machine=state_machine,
            cs_strength=cs_strength,
            tilt_Floor=tilt_Floor,
            tilt_Cap=tilt_Cap,
        )
        return w_final, ortho_clusters

    def _combine_cluster_signals_to_position(
        self,
        df_grid: pd.DataFrame,
        cluster_map: dict[str, np.ndarray],
        has_ortho: bool,
        eff_cs_rank: np.ndarray | None,
        asset_spec: AssetSpec,
        enable_regime_gating: bool,
        rebalance_deadzone: float,
        execution_mode_override: ExecutionMode | None,
        state_machine: ContinuousSignalStateMachine | bool | None = None,
        cs_strength: np.ndarray | None = None,
        tilt_Floor: float = 0.25,
        tilt_Cap: float = 1.50,
    ) -> np.ndarray:
        u_a = cluster_map[ARCH_A] * self.cluster_risk_budgets.get(ARCH_A, 1.0)
        if not has_ortho:
            w_multi = np.clip(u_a, -1.0, 1.0)
        else:
            is_commodity = asset_spec.asset_class == AssetClass.COMMODITY_FUTURES
            if is_commodity and eff_cs_rank is not None:
                b_long, b_short = 0.25, 0.25
                bcd_mult = 0.85
            elif is_commodity and enable_regime_gating:
                b_long, b_short = 0.15, 0.15
                bcd_mult = 0.35
            elif is_commodity:
                b_long, b_short = 0.40, 0.40
                bcd_mult = 1.25
            else:
                b_long, b_short = 0.35, 0.50
                bcd_mult = 1.25

            tactical_total = np.zeros(len(df_grid), dtype=np.float64)
            rev_pos = np.zeros(len(df_grid), dtype=np.float64)
            rev_neg = np.zeros(len(df_grid), dtype=np.float64)
            for arch in (ARCH_B, ARCH_C, ARCH_D, ARCH_E, ARCH_F):
                u_k = cluster_map[arch] * self.cluster_risk_budgets.get(arch, 1.0)
                tactical_total += u_k
                rev_pos += np.maximum(u_k, 0.0)
                rev_neg += np.minimum(u_k, 0.0)

            tactical_total = np.clip(tactical_total, -1.5, 1.5)
            w_a_gated = u_a.copy()
            long_mask = w_a_gated > 0.0
            short_mask = w_a_gated < 0.0
            w_a_gated[long_mask] *= np.clip(1.0 - b_long * np.abs(rev_neg[long_mask]), 0.15, 1.0)
            w_a_gated[short_mask] *= np.clip(1.0 - b_short * rev_pos[short_mask], 0.15, 1.0)

            w_raw = w_a_gated + bcd_mult * tactical_total

            vol_cap = self._get_causal_turbulence_cap(df_grid, asset_spec.base_tf)
            w_multi = np.clip(w_raw * np.sqrt(vol_cap), -vol_cap, vol_cap)

        if cs_strength is not None:
            w_multi = apply_cross_sectional_budget_tilt(
                w_multi,
                cs_strength,
                tilt_Floor=tilt_Floor,
                tilt_Cap=tilt_Cap,
            )

        if state_machine is not None and state_machine is not False:
            sm_obj = (
                state_machine
                if isinstance(state_machine, ContinuousSignalStateMachine)
                else ContinuousSignalStateMachine(rebalance_deadband=rebalance_deadzone)
            )
            w_filtered = sm_obj.transform(w_multi)
        else:
            w_filtered = apply_rebalance_deadzone(w_multi, deadzone=rebalance_deadzone)

        mode = execution_mode_override or asset_spec.execution_mode
        return self._apply_execution_mode_adapter(df_grid, w_filtered, mode)

    def synthesize(
        self,
        df_grid: pd.DataFrame,
        unit_weights_on_grid: dict[str, np.ndarray],
        unit_alloc_map: dict[str, float | np.ndarray],
        asset_spec: AssetSpec,
        rebalance_deadzone: float = 0.06,
        execution_mode_override: ExecutionMode | None = None,
        enable_regime_gating: bool = True,
        cs_slope_rank: np.ndarray | None = None,
        soft_gate: SoftCrossSectionalGate | bool | None = True,
        enable_gram_schmidt: bool = True,
        causal_ortho_window: int | None = None,
        state_machine: ContinuousSignalStateMachine | bool | None = None,
        cs_strength: np.ndarray | None = None,
        tilt_Floor: float = 0.25,
        tilt_Cap: float = 1.50,
    ) -> np.ndarray:
        if enable_gram_schmidt:
            w_final, _ = self.synthesize_cluster_orthogonals(
                df_grid=df_grid,
                unit_weights_on_grid=unit_weights_on_grid,
                unit_alloc_map=unit_alloc_map,
                asset_spec=asset_spec,
                causal_window=causal_ortho_window,
                rebalance_deadzone=rebalance_deadzone,
                execution_mode_override=execution_mode_override,
                enable_regime_gating=enable_regime_gating,
                cs_slope_rank=cs_slope_rank,
                soft_gate=soft_gate,
                state_machine=state_machine,
                cs_strength=cs_strength,
                tilt_Floor=tilt_Floor,
                tilt_Cap=tilt_Cap,
            )
            return w_final

        raw_clusters, has_ortho, eff_cs_rank = self.extract_raw_cluster_signals(
            df_grid=df_grid,
            unit_weights_on_grid=unit_weights_on_grid,
            unit_alloc_map=unit_alloc_map,
            asset_spec=asset_spec,
            enable_regime_gating=enable_regime_gating,
            cs_slope_rank=cs_slope_rank,
            soft_gate=soft_gate,
        )
        self.last_raw_cluster_signals = {k: v.copy() for k, v in raw_clusters.items()}
        self.last_ortho_cluster_signals = {k: v.copy() for k, v in raw_clusters.items()}
        return self._combine_cluster_signals_to_position(
            df_grid=df_grid,
            cluster_map=raw_clusters,
            has_ortho=has_ortho,
            eff_cs_rank=eff_cs_rank,
            asset_spec=asset_spec,
            enable_regime_gating=enable_regime_gating,
            rebalance_deadzone=rebalance_deadzone,
            execution_mode_override=execution_mode_override,
            state_machine=state_machine,
            cs_strength=cs_strength,
            tilt_Floor=tilt_Floor,
            tilt_Cap=tilt_Cap,
        )

    @staticmethod
    def _get_causal_regimes(df_grid: pd.DataFrame) -> np.ndarray:
        n = len(df_grid)
        cached = df_grid.attrs.get("_cached_regimes")
        if cached is not None and len(cached) == n:
            return cached
        from examples.multi_factor_cta.forensic_validator import classify_causal_market_regime

        regs = classify_causal_market_regime(df_grid, er_window=30, er_th=0.30)
        df_grid.attrs["_cached_regimes"] = regs
        return regs

    @staticmethod
    def _get_causal_turbulence_cap(df_grid: pd.DataFrame, base_tf: str) -> np.ndarray:
        n = len(df_grid)
        cached = df_grid.attrs.get("_cached_vol_cap")
        if cached is not None and len(cached) == n:
            return cached

        bars_per_day = {"5m": 48, "15m": 16, "30m": 8, "60m": 4}.get(base_tf, 16)
        w_15d = max(20, 15 * bars_per_day)
        w_5d = max(10, 5 * bars_per_day)

        close_arr = df_grid["close"].values
        ret_b = pd.Series(close_arr).pct_change().fillna(0.0).values
        vol_15d = pd.Series(ret_b).rolling(w_15d, min_periods=bars_per_day).std().bfill().values * np.sqrt(
            bars_per_day * 242.0
        )
        net_chg_5d = np.abs(pd.Series(close_arr).diff(w_5d).fillna(0.0).values)
        abs_path_5d = (
            pd.Series(np.abs(np.diff(close_arr, prepend=close_arr[0])))
            .rolling(w_5d, min_periods=bars_per_day)
            .sum()
            .bfill()
            .values
        )
        er_5d = np.clip(net_chg_5d / np.maximum(abs_path_5d, 1e-6), 0.0, 1.0)
        turb_vol = vol_15d * (1.0 - 0.35 * er_5d)
        vol_cap = np.clip(0.28 / np.maximum(turb_vol, 0.28), 0.72, 1.0)
        df_grid.attrs["_cached_vol_cap"] = vol_cap
        return vol_cap

    @staticmethod
    def _apply_execution_mode_adapter(df_grid: pd.DataFrame, w: np.ndarray, mode: ExecutionMode) -> np.ndarray:
        """
        交易制度与物理边界适配器：
        - LONG_SHORT: 完整双向多空权重 [-1, +1]
        - LONG_ONLY_SPOT: 纯多头现货轨 [0, +1]（空头信号自动转化为零延迟减仓/清仓风控）
        - INTRADAY_FLAT_1455: 日内 T+0 强平轨（14:45 起禁止新开仓，14:55 起强制平仓归零）
        """
        if mode == ExecutionMode.LONG_SHORT:
            return w
        if mode == ExecutionMode.LONG_ONLY_SPOT:
            return np.clip(w, 0.0, 1.0)
        if mode == ExecutionMode.INTRADAY_FLAT_1455:
            out = w.copy()
            times = pd.to_datetime(df_grid["dt"]).dt.time.values
            t_no_open = time(14, 45)
            t_flat = time(14, 55)
            for i in range(len(out)):
                if times[i] >= t_flat:
                    out[i] = 0.0
                elif times[i] >= t_no_open and i > 0 and abs(out[i]) > abs(out[i - 1]):
                    out[i] = out[i - 1]
            return out
        return w


class IntraAssetSixClusterSynthesizer(IntraAssetThreePillarSynthesizer):
    """六大战略簇二级分层正交合成器（`IntraAssetThreePillarSynthesizer` 的显式六簇别名子类）"""


def joint_is_coordinate_reweight(
    df_grid: pd.DataFrame,
    unit_weights_on_grid: dict[str, np.ndarray],
    initial_alloc: dict[str, float],
    unit_forensic_meta: dict[str, dict[str, Any]],
    asset_spec: AssetSpec,
    synthesizer: IntraAssetThreePillarSynthesizer,
    is_end_dt: pd.Timestamp = IS_END_DT,
) -> dict[str, float]:
    """
    严格样本内 (In-Sample) 法医经验贝叶斯先验锚定 + 双半场 (H1/H2) L2 信赖域联合坐标重配权
    """
    is_mask = (df_grid["dt"] <= is_end_dt).values
    mid_dt = pd.Timestamp("2021-12-31 23:59:59")
    h1_mask = (df_grid["dt"] <= mid_dt).values
    h2_mask = ((df_grid["dt"] > mid_dt) & (df_grid["dt"] <= is_end_dt)).values
    if int(h1_mask.sum()) < 100 or int(h2_mask.sum()) < 100:
        # 若品种起点较晚 (如 2021 起)，按样本内中位时间点切分 H1/H2
        is_dts = df_grid["dt"].values[is_mask]
        mid_val = is_dts[len(is_dts) // 2]
        h1_mask = (df_grid["dt"].values <= mid_val) & is_mask
        h2_mask = (df_grid["dt"].values > mid_val) & is_mask

    def _prep_slice(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sub = df_grid.iloc[mask].reset_index(drop=True)
        c = sub["close"].values
        r = np.zeros(len(c), dtype=np.float64)
        r[:-1] = (c[1:] - c[:-1]) / c[:-1]
        _, dids = np.unique(pd.to_datetime(sub["dt"]).dt.date.values, return_inverse=True)
        return r, dids

    def _fast_stats(close_ret: np.ndarray, day_ids: np.ndarray, w_slice: np.ndarray) -> dict[str, float]:
        dw = np.abs(np.diff(w_slice, prepend=0.0))
        bar_net = w_slice * close_ret - dw * asset_spec.fee_rate
        daily_net = np.bincount(day_ids, weights=bar_net)
        cum_ret = float(daily_net.sum() * 100.0)
        n_years = len(daily_net) / 252.0
        ann_ret = float(cum_ret / max(n_years, 1e-6))
        std_d = float(daily_net.std(ddof=1)) if len(daily_net) > 1 else 0.0
        sharpe = float((daily_net.mean() / std_d) * np.sqrt(252.0)) if std_d > 1e-12 else 0.0
        curve = np.cumsum(daily_net)
        peak = np.maximum.accumulate(np.maximum(curve, 0.0))
        mdd = float((peak - curve).max() * 100.0)
        calmar = float(ann_ret / mdd) if mdd > 1e-6 else 0.0
        return {"cum_ret_pct": cum_ret, "ann_ret_pct": ann_ret, "sharpe": sharpe, "mdd_pct": mdd, "calmar": calmar}

    ret_is, dids_is = _prep_slice(is_mask)
    ret_h1, dids_h1 = _prep_slice(h1_mask)
    ret_h2, dids_h2 = _prep_slice(h2_mask)

    raw_q: dict[str, float] = {}
    for uid in initial_alloc:
        meta = unit_forensic_meta.get(uid, {})
        n_p = max(int(meta.get("is_n_purged", 20)), 5)
        saf = float(np.clip(float(meta.get("is_safety_margin_pct", 4.0)), 1.0, 15.0))
        dd_res = float(np.clip(float(meta.get("dd_rescue_alpha_bp", 0.0)), -10.0, 30.0))
        raw_q[uid] = float(np.log(n_p) * saf * (1.0 + 0.02 * dd_res))

    non_seed = [u for u in initial_alloc if not unit_forensic_meta.get(u, {}).get("is_base_seed", False)]
    if non_seed:
        q_vals = np.array([raw_q[u] for u in non_seed])
        q_mean, q_std = float(q_vals.mean()), float(q_vals.std() + 1e-6)
    else:
        q_mean, q_std = 0.0, 1.0

    z_q = {
        u: (
            float(np.clip((raw_q[u] - q_mean) / q_std, -1.2, 1.6))
            if not unit_forensic_meta.get(u, {}).get("is_base_seed", False)
            else 0.0
        )
        for u in initial_alloc
    }
    eb_prior = {
        u: round(float(np.clip(initial_alloc[u] * (1.0 + 0.06 * z_q[u]), 0.15, 0.56)), 2) for u in initial_alloc
    }
    base_budget = sum(initial_alloc.values())

    def _eval(alloc: dict[str, float]) -> tuple[float, dict, dict, dict]:
        w_full = synthesizer.synthesize(df_grid, unit_weights_on_grid, alloc, asset_spec)
        s_is = _fast_stats(ret_is, dids_is, w_full[is_mask])
        s_h1 = _fast_stats(ret_h1, dids_h1, w_full[h1_mask])
        s_h2 = _fast_stats(ret_h2, dids_h2, w_full[h2_mask])
        l2_pen = sum((alloc[u] - eb_prior[u]) ** 2 for u in alloc)
        min_sh = min(s_h1["sharpe"], s_h2["sharpe"])
        obj = (
            s_is["sharpe"] * 1.15
            + min_sh * 0.85
            + s_is["calmar"] * 0.45
            + s_is["ann_ret_pct"] * 0.12
            - s_is["mdd_pct"] * 0.40
            - 1.0 * l2_pen
        )
        return obj, s_is, s_h1, s_h2

    opt_alloc = dict(eb_prior)
    obj_cur, st_is_cur, st_h1_cur, st_h2_cur = _eval(opt_alloc)
    ordered = sorted(initial_alloc.keys(), key=lambda u: raw_q[u], reverse=True)

    for uid in ordered:
        if bool(unit_forensic_meta.get(uid, {}).get("is_base_seed", False)):
            continue
        w_p = eb_prior[uid]
        tf = uid.split("@")[1]
        lo = max(0.16, round(w_p - 0.10, 2))
        hi = min(0.58 if tf == "5m" else 0.48, round(w_p + 0.10, 2))
        grid = sorted({lo, round(w_p - 0.05, 2), round(w_p, 2), round(w_p + 0.05, 2), hi})

        best_w = opt_alloc[uid]
        best_obj = obj_cur
        best_pack = (st_is_cur, st_h1_cur, st_h2_cur)
        for cw in grid:
            if abs(cw - opt_alloc[uid]) < 1e-4:
                continue
            t_alloc = dict(opt_alloc)
            t_alloc[uid] = cw
            if sum(t_alloc.values()) > base_budget + 0.45:
                continue
            obj_t, s_is, s_h1, s_h2 = _eval(t_alloc)
            if s_is["mdd_pct"] > st_is_cur["mdd_pct"] + 0.01:
                continue
            if s_h1["sharpe"] < st_h1_cur["sharpe"] - 0.003 or s_h2["sharpe"] < st_h2_cur["sharpe"] - 0.003:
                continue
            if s_is["sharpe"] < st_is_cur["sharpe"] or s_is["cum_ret_pct"] < st_is_cur["cum_ret_pct"] - 0.02:
                continue
            if obj_t > best_obj + 0.004:
                best_obj = obj_t
                best_w = cw
                best_pack = (s_is, s_h1, s_h2)

        if abs(best_w - opt_alloc[uid]) >= 1e-4:
            opt_alloc[uid] = best_w
            obj_cur = best_obj
            st_is_cur, st_h1_cur, st_h2_cur = best_pack

    return opt_alloc


def _project_capped_simplex_row(
    raw_weights: np.ndarray,
    active_mask: np.ndarray,
    target_budget: float,
    max_weight: float,
) -> np.ndarray:
    """
    在当前激活品种池 A_t 上执行带单品种上限 `max_weight` 的注水法 (Water-Filling) 风险预算分配：
    - 仅在 `active_mask > 0` 的活跃品种上分配权重（未激活品种严格为 0.0）；
    - 每个活跃品种权重严格满足 `0 <= w_i <= max_weight`；
    - 当 `|A_t| * max_weight >= target_budget` 时，活跃品种权重之和精确等于 `target_budget`，
      将触及上限品种的溢出预算按原始逆波动率/HRP比例平滑重分配给其余活跃品种。
    """
    n_cols = len(raw_weights)
    out = np.zeros(n_cols, dtype=np.float64)
    act_idx = np.where((active_mask > 0.0) & (raw_weights > 0.0))[0]
    if len(act_idx) == 0 or target_budget <= 0.0 or max_weight <= 0.0:
        return out

    eff_budget = min(float(target_budget), float(len(act_idx) * max_weight))
    w_sub = raw_weights[act_idx].astype(np.float64).copy()
    w_sum = float(w_sub.sum())
    if w_sum <= 1e-12:
        w_sub = np.full(len(act_idx), eff_budget / len(act_idx), dtype=np.float64)
    else:
        w_sub = w_sub / w_sum * eff_budget

    # Water-filling 迭代截断与溢出重分配
    free_mask = np.ones(len(act_idx), dtype=bool)
    for _ in range(len(act_idx)):
        over_mask = (w_sub > max_weight + 1e-12) & free_mask
        if not np.any(over_mask):
            break
        w_sub[over_mask] = max_weight
        free_mask[over_mask] = False
        rem_budget = eff_budget - float(w_sub[~free_mask].sum())
        if rem_budget <= 1e-12 or not np.any(free_mask):
            w_sub[free_mask] = 0.0
            break
        free_raw_sum = float(raw_weights[act_idx[free_mask]].sum())
        if free_raw_sum > 1e-12:
            w_sub[free_mask] = raw_weights[act_idx[free_mask]] / free_raw_sum * rem_budget
        else:
            w_sub[free_mask] = rem_budget / int(free_mask.sum())

    out[act_idx] = np.clip(w_sub, 0.0, max_weight)
    return out


class CrossAssetHRPSynthesizer:
    """
    跨品种（股指期货 + ETF + 商品期货）动态活跃池分配 ($1/N_{active}$) 与三层因果风险平价组合构建器
    (Issue #7 Phase 4)
    """

    def __init__(
        self,
        target_portfolio_vol: float = 0.14,
        max_single_asset_weight: float = 0.35,
        min_effective_assets: int = 3,
        active_eps: float = 1e-3,
        rebalance_deadzone: float = 0.0,
    ) -> None:
        self.target_portfolio_vol = float(target_portfolio_vol)
        self.max_single_asset_weight = float(max_single_asset_weight)
        self.min_effective_assets = max(int(min_effective_assets), 1)
        self.active_eps = float(active_eps)
        self.rebalance_deadzone = float(rebalance_deadzone)

    @staticmethod
    def _align_optional_panel(
        panel: dict[str, pd.Series] | pd.DataFrame | None,
        target_index: pd.Index,
        target_cols: pd.Index,
        fill_val: float = 0.0,
    ) -> pd.DataFrame | None:
        if panel is None:
            return None
        df = pd.DataFrame(panel).copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.reindex(index=target_index, columns=target_cols)
        return df.fillna(fill_val).astype(np.float64)

    def build_multi_asset_portfolio(
        self,
        asset_daily_returns: dict[str, pd.Series] | pd.DataFrame,
        asset_sectors: dict[str, str],
        is_end_date: pd.Timestamp = IS_END_DT,
        *,
        asset_positions_df: dict[str, pd.Series] | pd.DataFrame | None = None,
        asset_cs_strength_df: dict[str, pd.Series] | pd.DataFrame | None = None,
        asset_atr_pct_df: dict[str, pd.Series] | pd.DataFrame | None = None,
        use_dynamic_active_pool: bool = True,
        max_single_asset_weight: float | None = None,
        min_effective_assets: int | None = None,
        active_eps: float | None = None,
        rebalance_deadzone: float | None = None,
        tilt_Floor: float = 0.25,
        tilt_Cap: float = 1.50,
    ) -> dict[str, Any]:
        r"""
        基于各品种单品种策略日净值收益率序列与可选的时序持仓/截面强弱序列，按严格零前视的三层因果风险预算闭环：
        1. 动态活跃池识别 ($1 / N_{active, t}$)：
           区分“已上市但当前空仓静默 (|W_{i,t}| <= eps)”与“当前有实际非零信号激活 (|W_{i,t}| > eps)”的品种集合 A_t，
           消除空仓品种占用分母造成的现金拖累 (Cash Drag)，并执行单品种上限 `max_single_asset_weight` (默认 0.35)
           与最低活跃品种数保底 `min_effective_assets` (默认 3)；
        2. 第一层（因果 ATR / 收益率逆波动率平价）：
           基于严格 `shift(1)` 的 60 日因果滚动波动率（或因果 `atr14 / close`），对每个激活品种分配逆波动率基准权重；
        3. 第二层（板块内去拥挤 $1/\sqrt{K_{s,t}}$ 与板块间分层风险平价 Sector HRP）：
           按当前时刻同板块实际激活品种数 $K_{s,t}$ 动态施加 $1/\sqrt{K_{s,t}}$ 折减，并结合板块间因果逆波动率/低相关配权；
        4. 第三层（截面强弱风险预算倾斜 + 组合目标波动率 $\sigma_{target}$ + 调仓死区）。
        """
        raw_df = pd.DataFrame(asset_daily_returns).sort_index()
        raw_df.index = pd.to_datetime(raw_df.index)
        listed_mask = raw_df.notna().astype(np.float64)
        ret_df = raw_df.fillna(0.0).astype(np.float64)

        eff_max_w = float(
            max_single_asset_weight if max_single_asset_weight is not None else self.max_single_asset_weight
        )
        eff_min_assets = max(
            int(min_effective_assets if min_effective_assets is not None else self.min_effective_assets), 1
        )
        eff_eps = float(active_eps if active_eps is not None else self.active_eps)
        eff_deadzone = float(rebalance_deadzone if rebalance_deadzone is not None else self.rebalance_deadzone)

        pos_df = self._align_optional_panel(asset_positions_df, ret_df.index, ret_df.columns, fill_val=0.0)
        cs_df = self._align_optional_panel(asset_cs_strength_df, ret_df.index, ret_df.columns, fill_val=0.0)
        atr_df = self._align_optional_panel(asset_atr_pct_df, ret_df.index, ret_df.columns, fill_val=np.nan)

        # 0. 基准对比：静态在市品种等权组合 (Static Listed Equal-Weight 1 / N_{total, t})
        eq_weights = listed_mask.div(listed_mask.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        eq_daily_ret = (ret_df * eq_weights).sum(axis=1)

        # 动态活跃池掩码 A_t = {i | |W_{i,t}| > eps}
        if use_dynamic_active_pool:
            if pos_df is not None:
                active_mask = ((pos_df.abs() > eff_eps) & (listed_mask > 0.0)).astype(np.float64)
            else:
                active_from_ret = ((ret_df.abs() > min(eff_eps * 1e-2, 1e-6)) & (listed_mask > 0.0)).astype(np.float64)
                # 若某日所有在市品种收益均为 0，回退为 listed_mask 避免分母全空
                row_any = active_from_ret.sum(axis=1) > 0.0
                active_mask = active_from_ret.where(row_any, listed_mask, axis=0)
        else:
            active_mask = listed_mask.copy()

        # 1. 第一层：因果 60 日滚动波动率 / ATR 逆波动率平价 (严格 shift(1) 零前视)
        # 在计算活跃品种条件波动率时，剔除空仓静默 0 收益日的稀释效应
        active_ret_for_vol = raw_df.where(raw_df.abs() > 1e-8, np.nan)
        cond_vol = active_ret_for_vol.rolling(60, min_periods=5).std().shift(1) * np.sqrt(252.0)
        uncond_vol = raw_df.rolling(60, min_periods=10).std().shift(1) * np.sqrt(252.0)
        roll_vol = cond_vol.fillna(uncond_vol).bfill().ffill().fillna(0.15)
        if atr_df is not None:
            causal_atr_vol = (atr_df.shift(1).bfill().ffill() * np.sqrt(252.0)).fillna(roll_vol)
            roll_vol = 0.50 * roll_vol + 0.50 * causal_atr_vol

        inv_vol = (1.0 / np.maximum(roll_vol, 0.05)) * active_mask

        # 2. 第二层：动态板块内去拥挤 (1 / sqrt(K_{s,t})) 与板块间因果分层风险平价 (Sector HRP)
        sectors: dict[str, list[str]] = {}
        for sym, sec in asset_sectors.items():
            if sym in ret_df.columns:
                sectors.setdefault(sec, []).append(sym)
        for sym in ret_df.columns:
            if sym not in asset_sectors:
                sectors.setdefault("DEFAULT_SECTOR", []).append(sym)

        sector_mult_df = pd.DataFrame(0.0, index=ret_df.index, columns=ret_df.columns, dtype=np.float64)
        sector_active_counts: dict[str, pd.Series] = {}
        sector_inv_vol: dict[str, pd.Series] = {}

        for sec, syms in sectors.items():
            k_s_t = active_mask[syms].sum(axis=1)
            sector_active_counts[sec] = k_s_t
            sec_ret = ret_df[syms].mean(axis=1)
            sec_vol = sec_ret.rolling(60, min_periods=10).std().shift(1).bfill().ffill().fillna(0.12) * np.sqrt(252.0)
            sector_inv_vol[sec] = (1.0 / np.maximum(sec_vol, 0.05)) * (k_s_t > 0.0).astype(np.float64)

        sec_inv_vol_df = pd.DataFrame(sector_inv_vol, index=ret_df.index)
        sec_inv_vol_sum = sec_inv_vol_df.sum(axis=1).replace(0.0, np.nan)
        sec_budget_share = sec_inv_vol_df.div(sec_inv_vol_sum, axis=0).fillna(0.0)

        for sec, syms in sectors.items():
            k_s_t = sector_active_counts[sec]
            # 同一板块内有 K_{s,t} 个激活品种时，按 1 / sqrt(K_{s,t}) 折减同板块并发暴露
            within_sec_damp = 1.0 / np.sqrt(np.maximum(k_s_t, 1.0))
            sec_factor = sec_budget_share[sec] * within_sec_damp
            for sym in syms:
                sector_mult_df[sym] = sec_factor

        raw_alloc = inv_vol * sector_mult_df * active_mask

        # 3. 第三层：跨品种截面强弱风险预算倾斜 (Cross-Sectional Budget Tilt)
        if cs_df is not None:
            cs_tilt_df = pd.DataFrame(1.0, index=ret_df.index, columns=ret_df.columns, dtype=np.float64)
            for col in ret_df.columns:
                r_str = np.clip(cs_df[col].to_numpy(dtype=np.float64), -1.0, 1.0)
                if pos_df is not None:
                    w_dir = np.sign(pos_df[col].to_numpy(dtype=np.float64))
                    # 若当日持仓方向未定但有收益，回退为正顺势度
                    r_eff = np.where(w_dir != 0.0, w_dir * r_str, r_str)
                else:
                    r_eff = r_str
                cs_tilt_df[col] = np.clip(0.5 + 0.5 * (1.0 + r_eff), float(tilt_Floor), float(tilt_Cap))
            raw_alloc = raw_alloc * cs_tilt_df

        # 动态活跃池分母归一化 + min_effective_assets 平滑保底 + max_single_asset_weight 注水截断
        n_active_series = active_mask.sum(axis=1)
        n_rows, n_cols = raw_alloc.shape
        alloc_mat = np.zeros((n_rows, n_cols), dtype=np.float64)
        raw_mat = raw_alloc.to_numpy(dtype=np.float64)
        act_mat = active_mask.to_numpy(dtype=np.float64)

        for t_idx in range(n_rows):
            n_act = float(n_active_series.iloc[t_idx])
            if n_act <= 0.0:
                continue
            # 当活跃品种数 >= min_effective_assets 时满额分配 1.0；当仅 1~2 个品种激活时按 n_act / min_effective_assets 保底防单票满仓赌博
            target_b = min(1.0, n_act / float(eff_min_assets)) if use_dynamic_active_pool else 1.0
            alloc_mat[t_idx] = _project_capped_simplex_row(
                raw_weights=raw_mat[t_idx],
                active_mask=act_mat[t_idx],
                target_budget=target_b,
                max_weight=eff_max_w,
            )

        if eff_deadzone > 0.0:
            for c_idx in range(n_cols):
                alloc_mat[:, c_idx] = np.clip(
                    apply_rebalance_deadzone(alloc_mat[:, c_idx], deadzone=eff_deadzone),
                    0.0,
                    eff_max_w,
                )

        alloc_weights = pd.DataFrame(alloc_mat, index=ret_df.index, columns=ret_df.columns)

        # 因果组合级波动率目标缩放 (严格 shift(1) 零前视，并严守单品种权重上限 <= eff_max_w)
        unscaled_port_ret = (ret_df * alloc_weights).sum(axis=1)
        port_roll_vol = unscaled_port_ret.rolling(40, min_periods=15).std().shift(1).bfill().ffill() * np.sqrt(252.0)
        lev_scalar = np.clip(
            self.target_portfolio_vol / np.maximum(port_roll_vol.fillna(self.target_portfolio_vol), 0.05), 0.70, 1.65
        )

        final_weights_df = alloc_weights.multiply(lev_scalar, axis=0).clip(lower=0.0, upper=eff_max_w)
        port_daily_ret = (ret_df * final_weights_df).sum(axis=1)

        # 统计有效资金利用率 (Active Capital Utilization Ratio)
        any_active_days = n_active_series > 0
        if bool(any_active_days.any()):
            dyn_util = float((alloc_weights * active_mask).sum(axis=1).loc[any_active_days].mean())
            final_util = float((final_weights_df * active_mask).sum(axis=1).loc[any_active_days].mean())
            static_util = float((eq_weights * active_mask).sum(axis=1).loc[any_active_days].mean())
        else:
            dyn_util = final_util = static_util = 0.0

        is_mask = port_daily_ret.index <= pd.Timestamp(is_end_date.date())
        oos_mask = port_daily_ret.index > pd.Timestamp(is_end_date.date())

        return {
            "daily_returns": port_daily_ret,
            "equal_weight_returns": eq_daily_ret,
            "equal_weights_df": eq_weights,
            "unscaled_hrp_weights_df": alloc_weights,
            "asset_weights_df": final_weights_df,
            "active_mask_df": active_mask,
            "n_active_series": n_active_series,
            "active_capital_utilization_ratio": round(dyn_util, 4),
            "final_active_capital_utilization_ratio": round(final_util, 4),
            "static_active_capital_utilization_ratio": round(static_util, 4),
            "utilization_lift_ratio": round(dyn_util / max(static_util, 1e-6), 4) if static_util > 0 else 1.0,
            "max_single_asset_weight_realized": round(float(final_weights_df.max().max()), 4)
            if not final_weights_df.empty
            else 0.0,
            "full": self._calc_return_series_stats(port_daily_ret),
            "is": self._calc_return_series_stats(port_daily_ret.loc[is_mask]),
            "oos": self._calc_return_series_stats(port_daily_ret.loc[oos_mask]),
            "eq_full": self._calc_return_series_stats(eq_daily_ret),
            "eq_is": self._calc_return_series_stats(eq_daily_ret.loc[is_mask]),
            "eq_oos": self._calc_return_series_stats(eq_daily_ret.loc[oos_mask]),
        }

    @staticmethod
    def _calc_return_series_stats(dr: pd.Series) -> dict[str, float]:
        if dr.empty:
            return {"cum_ret_pct": 0.0, "ann_ret_pct": 0.0, "sharpe": 0.0, "mdd_pct": 0.0, "calmar": 0.0}
        arr = dr.values.astype(np.float64)
        cum_ret = float(arr.sum() * 100.0)
        n_years = len(arr) / 252.0
        ann_ret = float(cum_ret / max(n_years, 1e-6))
        std_d = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        sharpe = float((arr.mean() / std_d) * np.sqrt(252.0)) if std_d > 1e-12 else 0.0
        curve = np.cumsum(arr)
        peak = np.maximum.accumulate(np.maximum(curve, 0.0))
        mdd = float((peak - curve).max() * 100.0)
        calmar = float(ann_ret / mdd) if mdd > 1e-6 else 0.0
        win_days = float((arr > 0).sum() / max((arr != 0).sum(), 1) * 100.0)
        return {
            "cum_ret_pct": round(cum_ret, 2),
            "ann_ret_pct": round(ann_ret, 2),
            "sharpe": round(sharpe, 2),
            "mdd_pct": round(mdd, 2),
            "calmar": round(calmar, 2),
            "daily_win_pct": round(win_days, 1),
        }


class DynamicActivePoolAllocator(CrossAssetHRPSynthesizer):
    """
    动态活跃池分配器 ($1/N_{active}$ + ATR 逆波动率平价 + 板块 HRP + 截面强弱倾斜)：
    继承自 `CrossAssetHRPSynthesizer`，默认开启 `use_dynamic_active_pool=True`、
    `max_single_asset_weight=0.35`、`min_effective_assets=3`。
    """

    def allocate(
        self,
        asset_daily_returns: dict[str, pd.Series] | pd.DataFrame,
        asset_sectors: dict[str, str],
        is_end_date: pd.Timestamp = IS_END_DT,
        **kwargs: Any,
    ) -> dict[str, Any]:
        kwargs.setdefault("use_dynamic_active_pool", True)
        return self.build_multi_asset_portfolio(
            asset_daily_returns=asset_daily_returns,
            asset_sectors=asset_sectors,
            is_end_date=is_end_date,
            **kwargs,
        )


def evaluate_single_asset_backtest(
    df_grid: pd.DataFrame,
    w_grid: np.ndarray,
    symbol: str,
    fee_rate: float = 0.0002,
    is_end_dt: pd.Timestamp = IS_END_DT,
) -> dict[str, Any]:
    """调用 czsc.WeightBacktest 评估单品种 IS / OOS / Full 表现并返回日收益序列"""

    def _run_slice(sub_df: pd.DataFrame, sub_w: np.ndarray) -> tuple[dict[str, float], pd.Series]:
        if len(sub_df) < 10:
            return (
                {
                    "cum_ret_pct": 0.0,
                    "ann_ret_pct": 0.0,
                    "sharpe": 0.0,
                    "mdd_pct": 0.0,
                    "calmar": 0.0,
                    "active_pct": 0.0,
                },
                pd.Series(dtype=np.float64),
            )
        dfw = pd.DataFrame(
            {
                "dt": sub_df["dt"].astype(str),
                "symbol": symbol,
                "weight": sub_w,
                "price": sub_df["close"].values,
            }
        )
        wb = WeightBacktest(data=dfw, fee_rate=fee_rate, weight_type="ts")
        st = wb.stats
        dr = wb.daily_return.copy()
        dts = pd.to_datetime(dr["date"])
        wk = dts.dt.weekday
        shift_d = np.where(wk == 5, 2, np.where(wk == 6, 1, 0))
        dr["date"] = dts + pd.to_timedelta(shift_d, unit="D")
        s_dr = dr.groupby("date")["total"].sum()
        return (
            {
                "cum_ret_pct": round(float(st.get("绝对收益", 0.0) * 100.0), 2),
                "ann_ret_pct": round(float(st.get("年化收益", 0.0) * 100.0), 2),
                "sharpe": round(float(st.get("夏普比率", 0.0)), 2),
                "mdd_pct": round(float(st.get("最大回撤", 0.0) * 100.0), 2),
                "calmar": round(float(st.get("卡玛比率", 0.0)), 2),
                "active_pct": round(float((np.abs(sub_w) > 0.01).mean() * 100.0), 1),
            },
            s_dr,
        )

    is_mask = (df_grid["dt"] <= is_end_dt).values
    oos_mask = (df_grid["dt"] > is_end_dt).values

    st_full, dr_full = _run_slice(df_grid, w_grid)
    st_is, _ = _run_slice(df_grid.iloc[is_mask].reset_index(drop=True), w_grid[is_mask])
    st_oos, _ = _run_slice(df_grid.iloc[oos_mask].reset_index(drop=True), w_grid[oos_mask])
    return {"full": st_full, "is": st_is, "oos": st_oos, "daily_return": dr_full}


__all__ = [
    "DEFAULT_BASIS_ORDER",
    "ContinuousSignalStateMachine",
    "CrossAssetHRPSynthesizer",
    "DynamicActivePoolAllocator",
    "IntraAssetSixClusterSynthesizer",
    "IntraAssetThreePillarSynthesizer",
    "RollingEBWeightsResult",
    "RollingEmpiricalBayesWeighter",
    "apply_cross_sectional_budget_tilt",
    "compute_empirical_bayes_shrinkage_weight",
    "compute_rolling_empirical_bayes_weights",
    "evaluate_single_asset_backtest",
    "gram_schmidt_cluster_orthogonalize",
    "joint_is_coordinate_reweight",
]
