#!/usr/bin/env python
"""
第三层：规则型因子专属三级法医检验管线 (Three-Gate Forensic Validator for Rule-Based Factors)

为什么规则型因子不能直接套用股票截面多因子的全样本线性 IC/IR？
1. 规则型因子（分型/笔/中枢/两阶段状态机）在 70%~90% 时间处于 0（静默等待），全样本 IC 会被大量空仓期严重扭曲；
2. 不同战略簇（趋势突破 vs 背驰反转 vs 震荡回归 vs 日内微结构）的持仓半衰期与止盈止损结构截然不同。

本模块实现专用于规则型因子的三级门禁：
- Gate 1 (内在边缘与跨品种普适关)：
  Purged 去重叠独立抽样 + 周期自适应样本量门槛 + 分战略簇三重屏障标尺 + 成本穿透安全垫 ΔW + 跨品种普适率 (≥60% 品种正超额)
- Gate 2 (时序×截面三维互补与防拥挤关)：
  非零激活态交并比 (Signed Jaccard < 0.30) + 时序相关性 (|ρ| < 0.32) + 空仓补盲率 (≥35%) + 回撤期救援 Alpha (>0) + 横截面并发度 CSC
- Gate 3 (双半场稳健边际与留一法新陈代谢关)：
  样本内 H1/H2 双半场夏普一致性 + LOO 后向消融修剪 + MAP-Elites 同生态位 1-for-1 精英替换
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from examples.alpha_mining_agent.forensic_pipeline import (
    IS_END_DT,
    _compute_weighted_edge_metrics,
    compute_active_union_correlation,
    compute_forward_rank_ic_profile,
    compute_single_factor_net_sharpe_and_turnover,
    evaluate_triple_barrier,
    purge_overlapping_triggers,
    run_parameter_smoothness_audit,
)
from examples.alpha_mining_agent.hypothesis_bank import (
    ALL_SIGNAL_SPECS,
    ARCH_A,
    ARCH_B,
    ARCH_C,
    ARCH_D,
    ARCH_E,
    ARCH_F,
    ARCHETYPE_RULERS,
    TF_SCALE_MAP,
    ArchetypeRuler,
    compute_signal_triggers_and_weights,
)
from examples.multi_factor_cta.atomic_factors import SoftCrossSectionalGate

# 周期自适应最小独立样本数门槛（解决日线/60m 低频高质量信号被固定高样本门槛误杀的问题）
TF_MIN_PURGED_SAMPLES: dict[str, int] = {
    "1d": 12,
    "60m": 25,
    "30m": 35,
    "15m": 55,
    "5m": 80,
}

# 三级多标的分层池化检验层级映射 (Tier 1: 跨资产普适簇 | Tier 2: 同质资产类别特征簇 | Tier 3: 交易制度与微结构簇)
STRATIFIED_POOL_TIERS: dict[str, int] = {
    ARCH_A: 1,
    ARCH_E: 1,
    ARCH_F: 1,
    ARCH_B: 2,
    ARCH_C: 2,
    ARCH_D: 3,
}


def get_stratified_pool_tier(archetype: str, timeframe: str = "30m") -> int:
    """根据战略簇与周期返回对应的分层池化检验层级 (Tier 1 / 2 / 3)。"""
    ruler = ARCHETYPE_RULERS.get(archetype)
    base_tier = int(getattr(ruler, "pool_tier", STRATIFIED_POOL_TIERS.get(archetype, 1)))
    if base_tier == 3 or archetype == ARCH_D:
        return 3
    if timeframe in ("5m", "15m") and archetype not in (ARCH_A, ARCH_E, ARCH_F, ARCH_B, ARCH_C):
        return 3
    return base_tier


def infer_asset_pool_group(symbol: str, explicit_group: str | None = None) -> str:
    """
    识别标的所属同质资产池类别 ("equity" 宽基权益池 vs "commodity" 商品期货池)：
    - 若传入 explicit_group，优先归一化为 "equity" 或 "commodity"；
    - 否则按标的代码前缀/特征自动判定。
    """
    if explicit_group:
        g = explicit_group.strip().lower()
        if any(k in g for k in ("eq", "etf", "index", "stock", "if", "ic", "im", "ih")):
            return "equity"
        if any(k in g for k in ("comm", "cmd", "fut", "metal", "chem", "agri", "black", "energy")):
            return "commodity"
    s = symbol.upper()
    if any(
        tag in s
        for tag in (
            "SHFE",
            "DCE",
            "CZCE",
            "INE",
            "GFEX",
            "CMD",
            "COMM",
            "KQ.M@",
            "RB",
            "AG",
            "CU",
            "AU",
            "AL",
            "ZN",
            "NI",
            "TA",
            "MA",
            "M0",
            "RM",
            "CF",
            "SR",
            "SC",
            "I9",
            "JM",
            "J9",
        )
    ):
        return "commodity"
    return "equity"


def compute_signed_jaccard(w_u: np.ndarray, w_v: np.ndarray, eps: float = 1e-4) -> float:
    """
    计算两个稀疏规则型因子在非零激活状态下的同向交并比 (Signed Jaccard Similarity)：
    J_signed(u, v) = sum(I(w_u * w_v > 0)) / sum(I(|w_u| > eps or |w_v| > eps))
    专门解决大量共同 0 值掩盖真实触发重叠的问题。
    """
    active_union = (np.abs(w_u) > eps) | (np.abs(w_v) > eps)
    denom = int(active_union.sum())
    if denom == 0:
        return 0.0
    same_sign_inter = (w_u * w_v) > (eps * eps)
    return float(same_sign_inter.sum() / denom)


def compute_cross_sectional_concurrency(
    asset_weights_dict: dict[str, pd.Series], eps: float = 1e-3
) -> dict[str, float]:
    """
    计算单个规则因子在多品种横截面上的并发拥挤度 (Cross-Sectional Concurrency, CSC)：
    对齐各品种时间戳后，统计同一时刻触发同向持仓的品种占比分布。
    """
    if len(asset_weights_dict) <= 1:
        return {"mean_csc_pct": 0.0, "p95_csc_pct": 0.0, "max_csc_pct": 0.0}

    panel = pd.DataFrame(asset_weights_dict).ffill().fillna(0.0)
    active_mat = (panel.abs() > eps).astype(np.float64)
    any_active = active_mat.sum(axis=1) > 0
    if not np.any(any_active):
        return {"mean_csc_pct": 0.0, "p95_csc_pct": 0.0, "max_csc_pct": 0.0}

    csc_series = (active_mat.loc[any_active].mean(axis=1)) * 100.0
    return {
        "mean_csc_pct": round(float(csc_series.mean()), 2),
        "p95_csc_pct": round(float(np.percentile(csc_series, 95)), 2),
        "max_csc_pct": round(float(csc_series.max()), 2),
    }


class RuleFactorForensicValidator:
    """规则型与连续强度因子四级法医检验器 (Gate 1A + Gate 1B 分层池化 + 参数平原 + 激活态正交 & 边际增益)"""

    def __init__(
        self,
        is_end_dt: pd.Timestamp = IS_END_DT,
        max_pairwise_corr: float = 0.32,
        soft_gate: SoftCrossSectionalGate | bool | None = True,
    ) -> None:
        self.is_end_dt = is_end_dt
        self.max_pairwise_corr = max_pairwise_corr
        self.soft_gate = soft_gate
        self.rng = np.random.default_rng(42)

    def _resolve_soft_gated_inputs(
        self,
        df_tf: pd.DataFrame,
        raw_triggers: np.ndarray,
        weights_tf: np.ndarray | None,
        archetype: str,
        direction: int,
        soft_gate: SoftCrossSectionalGate | bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        eff_gate = self.soft_gate if soft_gate is None else soft_gate
        trig_bool = np.asarray(raw_triggers, dtype=bool)
        if eff_gate is None or eff_gate is False:
            return trig_bool, weights_tf
        gate_obj = eff_gate if isinstance(eff_gate, SoftCrossSectionalGate) else SoftCrossSectionalGate()
        cs_rank = gate_obj.extract_cs_rank(df_tf)
        if cs_rank is None:
            return trig_bool, weights_tf
        er = gate_obj.extract_causal_er(df_tf)
        return gate_obj.apply_to_triggers_and_weights(
            trig_bool,
            weights_tf,
            cs_rank,
            er,
            archetype=archetype,
            direction=direction,
        )

    def run_gate1_single_asset_audit(
        self,
        df_tf: pd.DataFrame,
        raw_triggers: np.ndarray,
        direction: int,
        archetype: str,
        timeframe: str,
        fee_rate: float = 0.0002,
        weights_tf: np.ndarray | None = None,
        soft_gate: SoftCrossSectionalGate | bool | None = None,
    ) -> dict[str, Any]:
        """
        Gate 1A：单品种单周期法医标尺检验（严格样本内 IS）
        - Purged 去重叠独立事件抽样
        - 周期自适应最小样本数门槛
        - 分战略簇三重屏障标尺 (Next-Bar Open + 交易成本扣除 + 连续质量分 Q 加权)
        - MFE / MAE 路径偏离比门槛 + 单因子扣费 Net Sharpe + 前向多期 Rank IC / ICIR 衰减曲线
        """
        ruler: ArchetypeRuler = ARCHETYPE_RULERS[archetype]
        is_mask = (df_tf["dt"] <= self.is_end_dt).values
        if int(is_mask.sum()) < 25:
            is_mask = np.ones(len(df_tf), dtype=bool)
        oos_mask = (df_tf["dt"] > self.is_end_dt).values

        min_samples = 8 if timeframe == "1d" else (12 if timeframe == "60m" else (20 if timeframe == "5m" else 15))
        eff_max_bars = ruler.max_bars * 2 if timeframe == "5m" else ruler.max_bars
        eff_tp_atr = ruler.tp_atr * 1.35 if timeframe == "5m" else ruler.tp_atr
        eff_sl_atr = ruler.sl_atr * 1.20 if timeframe == "5m" else ruler.sl_atr
        min_warmup = min(240 if timeframe == "5m" else 60, max(10, int(is_mask.sum()) // 4))

        trig_bool, weights_tf = self._resolve_soft_gated_inputs(
            df_tf, raw_triggers, weights_tf, archetype, direction, soft_gate
        )
        is_raw_idx = np.where(trig_bool & is_mask)[0]
        is_purged_idx = purge_overlapping_triggers(is_raw_idx, cooldown_bars=eff_max_bars)

        tb_is = evaluate_triple_barrier(
            df_tf,
            is_purged_idx,
            direction=direction,
            max_bars=eff_max_bars,
            tp_atr=eff_tp_atr,
            sl_atr=eff_sl_atr,
            fee_rate=fee_rate,
            weights_tf=weights_tf,
        )
        n_purged = len(tb_is)
        if n_purged < min_samples:
            return {
                "passed": False,
                "reason": f"样本内去重独立触发仅 {n_purged} 次 (< {min_samples} 次周期下限)",
                "is_n_purged": n_purged,
            }

        rets_is = tb_is["net_ret"].values
        q_w = tb_is["quality_w"].values if "quality_w" in tb_is.columns else np.ones(n_purged, dtype=np.float64)

        eq_stats = _compute_weighted_edge_metrics(rets_is, np.ones(n_purged, dtype=np.float64))
        qw_stats = _compute_weighted_edge_metrics(rets_is, q_w)

        use_q_weight = weights_tf is not None and (np.std(q_w) > 1e-4)
        active_stats = qw_stats if use_q_weight else eq_stats
        f_mean = active_stats["mean"]
        f_win = active_stats["win_rate"]
        pl_ratio = active_stats["pl_ratio"]
        safety_margin = active_stats["safety_margin"]

        mfe_arr = tb_is["mfe_atr"].values
        mae_arr = tb_is["mae_atr"].values
        q_norm = q_w / max(float(np.sum(q_w)), 1e-12)
        mean_mfe_atr = float(np.sum(q_norm * mfe_arr))
        mean_mae_atr = float(np.sum(q_norm * mae_arr))
        excursion_ratio = float(mean_mfe_atr / max(mean_mae_atr, 1e-6))

        valid_pool = np.where(
            is_mask & (np.arange(len(df_tf)) >= min_warmup) & (np.arange(len(df_tf)) < len(df_tf) - eff_max_bars - 2)
        )[0]
        if len(valid_pool) > 10:
            rand_pool_idx = self.rng.choice(valid_pool, size=min(800, len(valid_pool)), replace=False)
            rand_tb = evaluate_triple_barrier(
                df_tf,
                rand_pool_idx,
                direction=direction,
                max_bars=eff_max_bars,
                tp_atr=eff_tp_atr,
                sl_atr=eff_sl_atr,
                fee_rate=fee_rate,
            )
            rand_rets = rand_tb["net_ret"].values if not rand_tb.empty else np.zeros(50)
        else:
            rand_rets = np.zeros(50)

        bench_drift = float(np.mean(rand_rets))
        excess_alpha = f_mean - bench_drift
        boot_means = np.array([np.mean(self.rng.choice(rand_rets, size=n_purged, replace=True)) for _ in range(500)])
        boot_pval = float(np.mean(boot_means >= f_mean))

        if weights_tf is not None:
            net_sh_pack = compute_single_factor_net_sharpe_and_turnover(
                df_tf, weights_tf, timeframe, fee_rate=fee_rate, is_end_dt=self.is_end_dt
            )
            ic_pack = compute_forward_rank_ic_profile(df_tf, weights_tf, is_end_dt=self.is_end_dt)
        else:
            synth_w = np.zeros(len(df_tf), dtype=np.float64)
            for _, r_row in tb_is.iterrows():
                st_i = int(r_row["trigger_idx"]) + 1
                ed_i = min(st_i + max(1, int(r_row["bars_held"])), len(df_tf))
                synth_w[st_i:ed_i] = float(direction)
            net_sh_pack = compute_single_factor_net_sharpe_and_turnover(
                df_tf, synth_w, timeframe, fee_rate=fee_rate, is_end_dt=self.is_end_dt
            )
            ic_pack = compute_forward_rank_ic_profile(df_tf, synth_w, is_end_dt=self.is_end_dt)

        is_net_sharpe = float(net_sh_pack["is_net_sharpe"])
        ann_turnover = float(net_sh_pack["ann_turnover"])

        oos_raw_idx = np.where(trig_bool & oos_mask)[0]
        oos_purged_idx = purge_overlapping_triggers(oos_raw_idx, cooldown_bars=eff_max_bars)
        tb_oos = evaluate_triple_barrier(
            df_tf,
            oos_purged_idx,
            direction=direction,
            max_bars=eff_max_bars,
            tp_atr=eff_tp_atr,
            sl_atr=eff_sl_atr,
            fee_rate=fee_rate,
            weights_tf=weights_tf,
        )
        oos_mean = float(np.mean(tb_oos["net_ret"].values)) * 100.0 if len(tb_oos) >= 1 else 0.0

        tf_alpha_scale = {"1d": 1.2, "60m": 1.0, "30m": 0.85, "15m": 0.55, "5m": 0.40}.get(timeframe, 1.0)
        min_alpha_dec = (ruler.min_alpha_bp * tf_alpha_scale) / 10000.0
        min_safety_dec = ruler.min_safety_pct / 100.0
        max_pval = ruler.max_boot_pval + (0.02 if timeframe in ("60m", "1d") else 0.0)
        min_exc_ratio = float(getattr(ruler, "min_excursion_ratio", 0.90))
        min_net_sh = float(getattr(ruler, "min_net_sharpe", 0.0))

        passed = (
            excess_alpha >= min_alpha_dec
            and safety_margin >= min_safety_dec
            and boot_pval <= max_pval
            and excursion_ratio >= min_exc_ratio
            and (weights_tf is None or is_net_sharpe > min_net_sh)
        )
        reasons = []
        if excess_alpha < min_alpha_dec:
            reasons.append(f"超额Alpha={excess_alpha * 100:+.2f}%<+{min_alpha_dec * 100:.2f}%")
        if safety_margin < min_safety_dec:
            reasons.append(f"安全垫={safety_margin * 100:+.1f}%<+{ruler.min_safety_pct:.1f}%")
        if boot_pval > max_pval:
            reasons.append(f"p={boot_pval:.3f}>{max_pval:.2f}")
        if excursion_ratio < min_exc_ratio:
            reasons.append(f"MFE/MAE偏离比={excursion_ratio:.2f}<{min_exc_ratio:.2f}")
        if weights_tf is not None and is_net_sharpe <= min_net_sh:
            reasons.append(f"扣费Net Sharpe={is_net_sharpe:.2f}<={min_net_sh:.2f}")

        return {
            "passed": bool(passed),
            "reason": "通过" if passed else "; ".join(reasons),
            "ruler_name": ruler.archetype,
            "is_n_purged": int(n_purged),
            "is_mean_pct": round(f_mean * 100.0, 3),
            "is_unweighted_mean_pct": round(eq_stats["mean"] * 100.0, 3),
            "is_q_weighted_mean_pct": round(qw_stats["mean"] * 100.0, 3),
            "is_excess_alpha_pct": round(excess_alpha * 100.0, 3),
            "is_win_rate_pct": round(f_win * 100.0, 1),
            "is_unweighted_win_rate_pct": round(eq_stats["win_rate"] * 100.0, 1),
            "is_q_weighted_win_rate_pct": round(qw_stats["win_rate"] * 100.0, 1),
            "is_pl_ratio": round(pl_ratio, 2),
            "is_safety_margin_pct": round(safety_margin * 100.0, 1),
            "is_unweighted_safety_margin_pct": round(eq_stats["safety_margin"] * 100.0, 1),
            "is_q_weighted_safety_margin_pct": round(qw_stats["safety_margin"] * 100.0, 1),
            "is_boot_pval": round(boot_pval, 4),
            "mfe_atr_mean": round(mean_mfe_atr, 3),
            "mae_atr_mean": round(mean_mae_atr, 3),
            "excursion_ratio": round(excursion_ratio, 3),
            "is_net_sharpe": round(is_net_sharpe, 3),
            "ann_turnover": round(ann_turnover, 2),
            "rank_ic_curve": ic_pack["rank_ic_curve"],
            "icir_curve": ic_pack["icir_curve"],
            "peak_rank_ic": ic_pack["peak_rank_ic"],
            "mean_rank_ic": ic_pack["mean_rank_ic"],
            "autocorr_lag1": ic_pack["autocorr_lag1"],
            "half_life_bars": ic_pack["half_life_bars"],
            "bar_turnover": ic_pack["bar_turnover"],
            "oos_n_purged": int(len(tb_oos)),
            "oos_mean_pct": round(oos_mean, 3),
        }

    def run_gate1_cross_asset_universality_audit(
        self,
        asset_audits: dict[str, dict[str, Any]],
        min_positive_ratio: float = 0.60,
    ) -> dict[str, Any]:
        """
        Gate 1B（基础兼容接口）：跨品种普适性检验 (Cross-Asset Universality Audit)
        检验某一规则因子在多品种池中是否具备普适正期望（≥60% 有效品种取得正超额收益），
        严防针对单一品种特定历史噪声的过拟合。
        """
        valid = {sym: res for sym, res in asset_audits.items() if int(res.get("is_n_purged", 0)) >= 5}
        if not valid:
            return {"passed": False, "universal_ratio_pct": 0.0, "positive_assets": []}

        pos_syms = [
            sym
            for sym, res in valid.items()
            if (
                float(res.get("is_mean_pct", res.get("raw_mean_pct", 0.0))) > 0.0
                or float(res.get("norm_r_atr", 0.0)) > 0.0
            )
            and float(res.get("is_safety_margin_pct", res.get("uncond_safety_pct", 1.0))) > 0.0
        ]
        ratio = len(pos_syms) / len(valid)
        total_is_n = sum(int(res.get("is_n_purged", 0)) for res in valid.values())
        avg_safety = float(
            np.mean(
                [float(res.get("is_safety_margin_pct", res.get("uncond_safety_pct", 0.0))) for res in valid.values()]
            )
        )

        return {
            "passed": bool(ratio >= min_positive_ratio and avg_safety > 1.5),
            "universal_ratio_pct": round(ratio * 100.0, 1),
            "total_pool_is_n": total_is_n,
            "mean_pool_safety_pct": round(avg_safety, 2),
            "positive_assets": pos_syms,
            "evaluated_assets": list(valid.keys()),
        }

    def run_stratified_pooled_gate1b_audit(
        self,
        archetype: str = ARCH_A,
        timeframe: str = "30m",
        *,
        pool_tf_dfs: dict[str, pd.DataFrame] | None = None,
        triggers_by_symbol: dict[str, np.ndarray] | None = None,
        weights_by_symbol: dict[str, np.ndarray] | None = None,
        sig_id: str | None = None,
        direction: int = 1,
        eval_fn: Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray]] | None = None,
        asset_audits: dict[str, dict[str, Any]] | None = None,
        per_symbol_gate1: dict[str, dict[str, Any]] | None = None,
        asset_groups: dict[str, str] | None = None,
        target_pool_group: str = "equity",
        fee_rate: float = 0.0002,
        allow_single_asset_fallback: bool = False,
        allow_single_asset_degrade: bool | None = None,
        soft_gate: SoftCrossSectionalGate | bool | None = None,
    ) -> dict[str, Any]:
        """
        Gate 1B（三级多标的分层池化检验协议 · Stratified Multi-Asset Pooling）：
        统一通过 `run_vol_and_regime_normalized_audit` 做 ATR 无量纲标准化 + IPW 状态逆概率重加权：
        - Tier 1 (跨资产普适簇: A_趋势突破 / E_波动率压缩 / F_多周期共振):
          检验全市场跨资产大池（宽基权益 + 商品期货），要求跨标的池正期望占比 `universal_ratio >= 0.60`，
          且池化合并 Purged `t-stat >= 2.5` (当池规模 >= 5 时，否則 >= 2.0)。
        - Tier 2 (同质资产类别特征簇: B_背驰衰竭 / C_中枢震荡):
          检验所属同质大类资产池（`equity` 宽基权益池 vs `commodity` 商品板块池），
          要求同质池内正期望占比 `universal_ratio >= 0.75` (权益池) 或 `>= 0.65` (商品池)，
          严禁只在单标的赚钱而在同池其余标的上亏损。
        - Tier 3 (交易制度与微结构簇: D_量价筹码 及高频微结构):
          检验同交易制度高流动性池，要求正期望占比 `universal_ratio >= 0.70` 且扣费后池化平均 `is_net_sharpe > 0`。
        """
        if per_symbol_gate1 is not None and asset_audits is None:
            asset_audits = per_symbol_gate1
        if allow_single_asset_degrade is not None:
            allow_single_asset_fallback = allow_single_asset_degrade

        tier = get_stratified_pool_tier(archetype, timeframe)
        pool_tier_name = {
            1: "Tier1_CrossAsset",
            2: "Tier2_AssetClass",
            3: "Tier3_Microstructure",
        }.get(tier, "Tier1_CrossAsset")
        norm_target_group = infer_asset_pool_group(target_pool_group, target_pool_group)

        raw_audits: dict[str, dict[str, Any]] = {}
        if asset_audits is not None:
            raw_audits = dict(asset_audits)
        elif pool_tf_dfs is not None:
            for sym, df_sym in pool_tf_dfs.items():
                if triggers_by_symbol is not None and sym in triggers_by_symbol:
                    t_arr = triggers_by_symbol[sym]
                    w_arr = weights_by_symbol.get(sym) if weights_by_symbol is not None else None
                elif eval_fn is not None:
                    t_arr, w_arr = eval_fn(df_sym)
                elif sig_id is not None and sig_id in ALL_SIGNAL_SPECS:
                    spec = ALL_SIGNAL_SPECS[sig_id]
                    direction = spec.direction
                    t_arr, w_arr = compute_signal_triggers_and_weights(
                        df_sym, sig_id, scale=TF_SCALE_MAP.get(timeframe, 1.0)
                    )
                else:
                    raise ValueError(f"Missing triggers/eval_fn/sig_id for symbol {sym} in pooled Gate 1B audit.")

                raw_audits[sym] = self.run_vol_and_regime_normalized_audit(
                    df_sym,
                    t_arr,
                    direction=direction,
                    archetype=archetype,
                    timeframe=timeframe,
                    fee_rate=fee_rate,
                    weights_tf=w_arr,
                    soft_gate=soft_gate,
                )
        else:
            raise ValueError("Must provide either pool_tf_dfs or asset_audits to run_stratified_pooled_gate1b_audit.")

        if not raw_audits:
            return {
                "passed": False,
                "tier": tier,
                "pool_tier": pool_tier_name,
                "pool_size": 0,
                "reason": "标的池为空，无法执行分层池化检验",
                "universal_ratio": 0.0,
                "universal_ratio_pct": 0.0,
                "pooled_t_stat": 0.0,
                "mean_pool_net_sharpe": 0.0,
                "positive_assets": [],
                "evaluated_assets": [],
            }

        # 按 Tier 筛选目标检验池：Tier 1 取全市场跨资产池；Tier 2 / Tier 3 优先取目标同质池
        if tier in (2, 3) and len(raw_audits) > 1:
            matched_syms = [
                sym
                for sym in raw_audits
                if infer_asset_pool_group(sym, (asset_groups or {}).get(sym)) == norm_target_group
            ]
            selected_syms = matched_syms if len(matched_syms) >= 2 else list(raw_audits.keys())
        else:
            selected_syms = list(raw_audits.keys())

        valid_audits = {
            sym: raw_audits[sym] for sym in selected_syms if int(raw_audits[sym].get("is_n_purged", 0)) >= 3
        }
        if not valid_audits:
            return {
                "passed": False,
                "tier": tier,
                "pool_tier": pool_tier_name,
                "pool_size": 0,
                "reason": "池内所有标的独立触发样本均不足 (< 3)",
                "universal_ratio": 0.0,
                "universal_ratio_pct": 0.0,
                "pooled_t_stat": 0.0,
                "mean_pool_net_sharpe": 0.0,
                "positive_assets": [],
                "evaluated_assets": [],
            }

        n_assets = len(valid_audits)
        pos_assets: list[str] = []
        pooled_r_atr_samples: list[float] = []
        net_sharpes: list[float] = []
        ipw_r_atrs: list[float] = []

        for sym, aud in valid_audits.items():
            norm_r = float(aud.get("norm_r_atr", aud.get("is_mean_pct", 0.0)))
            ipw_r = float(aud.get("ipw_r_atr", norm_r))
            raw_m = float(aud.get("raw_mean_pct", aud.get("is_mean_pct", 0.0)))
            net_sh = float(aud.get("is_net_sharpe", 0.0))
            net_sharpes.append(net_sh)
            ipw_r_atrs.append(ipw_r)

            if (norm_r > 0.0 or ipw_r > 0.0) and raw_m > 0.0:
                pos_assets.append(sym)

            if (
                "r_atr_values" in aud
                and isinstance(aud["r_atr_values"], list | np.ndarray)
                and len(aud["r_atr_values"]) > 0
            ):
                pooled_r_atr_samples.extend(float(x) for x in aud["r_atr_values"])
            else:
                n_p = max(1, int(aud.get("is_n_purged", 10)))
                std_r = max(float(aud.get("std_r_atr", 0.5)), 0.1)
                synth = np.linspace(norm_r - std_r * 0.5, norm_r + std_r * 0.5, num=n_p)
                pooled_r_atr_samples.extend(float(x) for x in synth)

        universal_ratio = float(len(pos_assets) / n_assets)
        mean_net_sharpe = float(np.mean(net_sharpes)) if net_sharpes else 0.0
        mean_ipw_r_atr = float(np.mean(ipw_r_atrs)) if ipw_r_atrs else 0.0

        r_pool_arr = np.asarray(pooled_r_atr_samples, dtype=np.float64)
        total_n_purged = len(r_pool_arr)
        if total_n_purged >= 2 and float(r_pool_arr.std(ddof=1)) > 1e-9:
            pooled_t_stat = float(r_pool_arr.mean() / (r_pool_arr.std(ddof=1) / np.sqrt(total_n_purged)))
        else:
            pooled_t_stat = 0.0

        # 单标的平滑降级模式 (仅当本地只有单标的数据且显式允许 fallback 时生效)
        if n_assets == 1 and allow_single_asset_fallback:
            passed = (universal_ratio >= 1.0) and (mean_ipw_r_atr > 0.0) and (pooled_t_stat >= 1.2)
            return {
                "passed": bool(passed),
                "tier": tier,
                "pool_tier": pool_tier_name,
                "pool_size": n_assets,
                "degraded_single_asset": True,
                "reason": "单标的降级检验通过 (建议补充同质池数据)"
                if passed
                else f"单标的池化t={pooled_t_stat:.2f}或IPW期望不足",
                "universal_ratio": round(universal_ratio, 4),
                "universal_ratio_pct": round(universal_ratio * 100.0, 1),
                "pooled_t_stat": round(pooled_t_stat, 3),
                "mean_pool_net_sharpe": round(mean_net_sharpe, 3),
                "mean_ipw_r_atr": round(mean_ipw_r_atr, 3),
                "total_pool_is_n": total_n_purged,
                "positive_assets": pos_assets,
                "evaluated_assets": list(valid_audits.keys()),
            }

        reasons: list[str] = []
        pool_label = "Equity-Pool" if norm_target_group == "equity" else "Commodity-Sector-Pool"
        if tier == 1:
            min_ratio = 0.60
            min_t_stat = 2.5 if n_assets >= 5 else 2.0
            if universal_ratio < min_ratio:
                reasons.append(f"Tier1跨资产普适率={universal_ratio * 100:.1f}% < {min_ratio * 100:.0f}%")
            if pooled_t_stat < min_t_stat:
                reasons.append(f"Tier1池化Purged t-stat={pooled_t_stat:.2f} < {min_t_stat:.2f}")
            if mean_ipw_r_atr <= 0.0:
                reasons.append(f"Tier1状态均衡IPW r_atr={mean_ipw_r_atr:+.3f} <= 0")
        elif tier == 2:
            min_ratio = 0.75 if norm_target_group == "equity" else 0.65
            min_t_stat = 1.96
            if universal_ratio < min_ratio:
                reasons.append(
                    f"Tier2({pool_label})未达标: 同质池({norm_target_group})普适率={universal_ratio * 100:.1f}% < {min_ratio * 100:.0f}% (单品种过拟合拦截)"
                )
            if pooled_t_stat < min_t_stat:
                reasons.append(f"Tier2同质池Purged t-stat={pooled_t_stat:.2f} < {min_t_stat:.2f}")
            if mean_ipw_r_atr <= 0.0:
                reasons.append(f"Tier2状态均衡IPW r_atr={mean_ipw_r_atr:+.3f} <= 0")
        else:
            min_ratio = 0.70
            min_t_stat = 1.80
            if universal_ratio < min_ratio:
                reasons.append(f"Tier3同制度池普适率={universal_ratio * 100:.1f}% < {min_ratio * 100:.0f}%")
            if mean_net_sharpe <= 0.0:
                reasons.append(f"Tier3扣费后池化平均Net Sharpe={mean_net_sharpe:+.2f} <= 0.00")
            if pooled_t_stat < min_t_stat:
                reasons.append(f"Tier3池化Purged t-stat={pooled_t_stat:.2f} < {min_t_stat:.2f}")

        passed = len(reasons) == 0
        return {
            "passed": bool(passed),
            "tier": tier,
            "pool_tier": pool_tier_name,
            "pool_size": n_assets,
            "degraded_single_asset": False,
            "reason": f"通过 Tier {tier} 多标的分层池化检验" if passed else "; ".join(reasons),
            "universal_ratio": round(universal_ratio, 4),
            "universal_ratio_pct": round(universal_ratio * 100.0, 1),
            "pooled_t_stat": round(pooled_t_stat, 3),
            "mean_pool_net_sharpe": round(mean_net_sharpe, 3),
            "mean_ipw_r_atr": round(mean_ipw_r_atr, 3),
            "total_pool_is_n": total_n_purged,
            "positive_assets": pos_assets,
            "evaluated_assets": list(valid_audits.keys()),
            "per_asset_audits": valid_audits,
        }

    def run_parameter_smoothness_audit(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """委托执行参数平原平滑度检验 (Parameter Smoothness Gate)。"""
        if "is_end_dt" not in kwargs:
            kwargs["is_end_dt"] = self.is_end_dt
        return run_parameter_smoothness_audit(*args, **kwargs)

    def run_gate2_complementarity_and_jaccard_audit(
        self,
        df_grid: pd.DataFrame | None = None,
        cand_w_on_grid: np.ndarray | None = None,
        existing_same_dir_weights: dict[str, np.ndarray] | None = None,
        current_portfolio_w: np.ndarray | None = None,
        max_corr_threshold: float | None = None,
        max_jaccard_threshold: float = 0.30,
        min_gap_fill_pct: float = 35.0,
        active_corr_eps: float = 1e-3,
        min_active_bars: int = 30,
        *,
        df_30m: pd.DataFrame | None = None,
        cand_w_30m: np.ndarray | None = None,
        cand_w_on_30m: np.ndarray | None = None,
        zoo_weights_30m: dict[str, np.ndarray] | None = None,
        curr_portfolio_w_30m: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Gate 2 / Gate 3：时序×截面三维真互补与激活态正交门禁（严格样本内 IS）
        1. 激活态并集条件相关性 |ρ_active| < 0.32 且 非零激活态交并比 Signed Jaccard < 0.30；
        2. 空仓填补率 >= 35%；
        3. 原组合回撤/亏损期的雪中送炭 Alpha (dd_rescue_alpha_bp > 0)。
        """
        if df_grid is None:
            df_grid = df_30m
        if cand_w_on_grid is None:
            cand_w_on_grid = cand_w_30m if cand_w_30m is not None else cand_w_on_30m
        if existing_same_dir_weights is None:
            existing_same_dir_weights = zoo_weights_30m or {}
        if current_portfolio_w is None:
            current_portfolio_w = curr_portfolio_w_30m
        if max_corr_threshold is None:
            max_corr_threshold = self.max_pairwise_corr
        assert df_grid is not None and cand_w_on_grid is not None and current_portfolio_w is not None

        is_mask = (df_grid["dt"] <= self.is_end_dt).values
        if int(is_mask.sum()) < 25:
            is_mask = np.ones(len(df_grid), dtype=bool)
        cw_is = cand_w_on_grid[is_mask]
        port_w_is = current_portfolio_w[is_mask]

        max_corr = 0.0
        max_full_corr = 0.0
        max_jaccard = 0.0
        max_peer = "None"
        for peer_id, pw in existing_same_dir_weights.items():
            pw_is = pw[is_mask]
            act_c = float(
                abs(
                    compute_active_union_correlation(cw_is, pw_is, eps=active_corr_eps, min_active_bars=min_active_bars)
                )
            )
            full_c = (
                float(abs(np.corrcoef(cw_is, pw_is)[0, 1])) if (np.std(cw_is) > 1e-6 and np.std(pw_is) > 1e-6) else 0.0
            )
            jac = compute_signed_jaccard(cw_is, pw_is)
            if full_c > max_full_corr:
                max_full_corr = full_c
            if act_c > max_corr or jac > max_jaccard:
                if act_c > max_corr:
                    max_corr = act_c
                    max_peer = peer_id
                if jac > max_jaccard:
                    max_jaccard = jac

        cand_active = np.abs(cw_is) > 0.01
        if int(cand_active.sum()) == 0:
            return {"passed": False, "reason": "样本内无有效持仓激活"}

        port_flat_or_light = np.abs(port_w_is) <= 0.25
        gap_fill_ratio = float((cand_active & port_flat_or_light).sum() / cand_active.sum() * 100.0)

        bar_ret = df_grid["close"].pct_change().fillna(0.0).values[is_mask]
        port_cum_nav = np.cumprod(1.0 + port_w_is * bar_ret)
        port_peak = np.maximum.accumulate(port_cum_nav)
        port_dd_state = (port_cum_nav - port_peak) / port_peak

        fwd_bar_ret = df_grid["close"].pct_change().shift(-1).fillna(0.0).values[is_mask]
        rescue_mask = ((np.abs(port_w_is) <= 0.20) | (port_dd_state <= -0.015)) & cand_active
        bp_scale = 6.0 if len(df_grid) > 40000 else 1.0
        if int(rescue_mask.sum()) >= 5:
            dd_rescue_alpha_bp = float(np.mean(cw_is[rescue_mask] * fwd_bar_ret[rescue_mask]) * 10000.0 * bp_scale)
        else:
            dd_rescue_alpha_bp = 0.0

        passed = (
            (max_corr < max_corr_threshold)
            and (max_jaccard < max_jaccard_threshold)
            and (gap_fill_ratio >= min_gap_fill_pct)
            and (dd_rescue_alpha_bp >= 0.5)
        )
        return {
            "passed": bool(passed),
            "max_corr": round(max_corr, 3),
            "max_full_sample_corr": round(max_full_corr, 3),
            "max_signed_jaccard": round(max_jaccard, 3),
            "max_corr_peer": max_peer,
            "gap_fill_ratio_pct": round(gap_fill_ratio, 1),
            "dd_rescue_alpha_bp": round(dd_rescue_alpha_bp, 2),
        }

    run_gate3_orthogonality_and_marginal_audit = run_gate2_complementarity_and_jaccard_audit

    def run_vol_and_regime_normalized_audit(
        self,
        df_tf: pd.DataFrame,
        raw_triggers: np.ndarray,
        direction: int,
        archetype: str,
        timeframe: str,
        fee_rate: float = 0.0002,
        weights_tf: np.ndarray | None = None,
        soft_gate: SoftCrossSectionalGate | bool | None = None,
    ) -> dict[str, Any]:
        """
        解决【跨标的波动率不同 + 市场状态构成不同】的法医归一化检验：
        1. 波动率无量纲化 (R-Multiple / ATR-Unit Edge)：
           将每笔交易净收益除以入场时刻因果相对波动率 `atr_pct = atr14 / close`，
           转化为“每承担 1 单位 ATR 波动捕获的净 ATR 倍数 (r_atr)”，消除高波/低波标的名义收益率不可比问题；
        2. 因果三态状态机切分 (Bull Trend / Bear Trend / Choppy Range)：
           利用 Kaufman 效率系数 ER(20) 与均线结构将每个标的样本内切分为【多头趋势态(+1) / 空头趋势态(-1) / 震荡盘整态(0)】；
        3. 逆概率重加权状态均衡期望 (IPW Regime-Balanced Edge) 与主场状态安全边际 (Home-Regime ΔW)：
           按标准均衡状态配比 (Bull:Bear:Range = 1/3:1/3:1/3) 计算状态重加权后的期望与安全边际，
           剥离“某标的恰巧在样本内走了三年震荡或三年熊市”的状态构成偏差 (Regime Compositional Bias)。
        """
        ruler: ArchetypeRuler = ARCHETYPE_RULERS[archetype]
        is_mask = (df_tf["dt"] <= self.is_end_dt).values
        if int(is_mask.sum()) < 25:
            is_mask = np.ones(len(df_tf), dtype=bool)
        eff_max_bars = ruler.max_bars * 2 if timeframe == "5m" else ruler.max_bars
        eff_tp_atr = ruler.tp_atr * 1.35 if timeframe == "5m" else ruler.tp_atr
        eff_sl_atr = ruler.sl_atr * 1.20 if timeframe == "5m" else ruler.sl_atr

        regimes = classify_causal_market_regime(df_tf)
        reg_is = regimes[is_mask]
        bull_time_pct = float(np.mean(reg_is == 1) * 100.0)
        bear_time_pct = float(np.mean(reg_is == -1) * 100.0)
        range_time_pct = float(np.mean(reg_is == 0) * 100.0)

        c_arr = df_tf["close"].values
        atr_pct_arr = np.maximum(df_tf["atr14"].bfill().values / np.maximum(c_arr, 1e-6), 1e-4)
        median_atr_pct = float(np.median(atr_pct_arr[is_mask]) * 100.0)

        trig_bool, weights_tf = self._resolve_soft_gated_inputs(
            df_tf, raw_triggers, weights_tf, archetype, direction, soft_gate
        )
        is_raw_idx = np.where(trig_bool & is_mask)[0]
        is_purged_idx = purge_overlapping_triggers(is_raw_idx, cooldown_bars=eff_max_bars)
        tb_is = evaluate_triple_barrier(
            df_tf,
            is_purged_idx,
            direction=direction,
            max_bars=eff_max_bars,
            tp_atr=eff_tp_atr,
            sl_atr=eff_sl_atr,
            fee_rate=fee_rate,
            weights_tf=weights_tf,
        )
        if tb_is.empty:
            return {"is_n_purged": 0, "median_bar_atr_pct": round(median_atr_pct, 3)}

        t_idx = tb_is["trigger_idx"].values.astype(int)
        raw_nets = tb_is["net_ret"].values
        q_w = tb_is["quality_w"].values if "quality_w" in tb_is.columns else np.ones(len(tb_is), dtype=np.float64)
        use_q_w = weights_tf is not None and (np.std(q_w) > 1e-4 or np.mean(q_w) < 0.99)
        q_norm = (q_w / max(float(np.sum(q_w)), 1e-12)) if use_q_w else np.full(len(tb_is), 1.0 / len(tb_is))

        entry_atr_pct = atr_pct_arr[t_idx]
        # 核心变换 1：转化为无量纲 ATR 单位净收益 (R-Multiple in ATR units)
        r_atr = raw_nets / entry_atr_pct
        trig_regimes = regimes[t_idx]

        # 定义该因子战略簇对应的“目标主场状态 (Home Regime)”
        if archetype == ARCH_A:
            home_mask = (trig_regimes == 1) if direction == 1 else (trig_regimes == -1)
        elif archetype == ARCH_C:
            home_mask = trig_regimes == 0
        else:
            # 背驰衰竭 / 量价筹码 / 波动率压缩 / 多周期共振：震荡态或逆势衰竭态均为主场
            home_mask = (trig_regimes == 0) | (trig_regimes == -direction)

        def _calc_safety(rets: np.ndarray) -> float:
            if len(rets) < 2:
                return 0.0
            w = float(np.mean(rets > 0))
            pos = rets[rets > 0]
            neg = rets[rets <= 0]
            avg_w = float(np.mean(pos)) if len(pos) > 0 else 0.0
            avg_l = float(abs(np.mean(neg))) if len(neg) > 0 else 1e-6
            r = avg_w / max(avg_l, 1e-6)
            return (w - 1.0 / (1.0 + r)) * 100.0

        uncond_safety = _calc_safety(raw_nets)
        home_safety = _calc_safety(raw_nets[home_mask]) if int(home_mask.sum()) >= 3 else uncond_safety
        norm_r_atr_val = float(np.sum(q_norm * r_atr))
        home_r_atr = (
            float(np.sum(q_norm[home_mask] * r_atr[home_mask]) / max(float(np.sum(q_norm[home_mask])), 1e-12))
            if int(home_mask.sum()) >= 3
            else norm_r_atr_val
        )

        # 核心变换 2：三态逆概率重加权 (IPW Regime-Balanced Expectancy)
        state_r_atr = []
        state_safeties = []
        for st in (1, 0, -1):
            m = trig_regimes == st
            if int(m.sum()) >= 2:
                w_m = q_norm[m]
                state_r_atr.append(float(np.sum(w_m * r_atr[m]) / max(float(np.sum(w_m)), 1e-12)))
                state_safeties.append(_calc_safety(raw_nets[m]))
        ipw_r_atr = float(np.mean(state_r_atr)) if state_r_atr else norm_r_atr_val
        ipw_safety = float(np.mean(state_safeties)) if state_safeties else uncond_safety

        if weights_tf is not None:
            net_sh_pack = compute_single_factor_net_sharpe_and_turnover(
                df_tf, weights_tf, timeframe, fee_rate=fee_rate, is_end_dt=self.is_end_dt
            )
        else:
            synth_w = np.zeros(len(df_tf), dtype=np.float64)
            for _, r_row in tb_is.iterrows():
                st_i = int(r_row["trigger_idx"]) + 1
                ed_i = min(st_i + max(1, int(r_row["bars_held"])), len(df_tf))
                synth_w[st_i:ed_i] = float(direction)
            net_sh_pack = compute_single_factor_net_sharpe_and_turnover(
                df_tf, synth_w, timeframe, fee_rate=fee_rate, is_end_dt=self.is_end_dt
            )

        std_r_atr = float(np.std(r_atr, ddof=1)) if len(r_atr) > 1 else 0.5
        t_stat_r_atr = float(norm_r_atr_val / (std_r_atr / np.sqrt(len(r_atr)))) if std_r_atr > 1e-9 else 0.0
        mfe_mean = float(np.sum(q_norm * tb_is["mfe_atr"].values))
        mae_mean = float(np.sum(q_norm * tb_is["mae_atr"].values))
        exc_ratio = float(mfe_mean / max(mae_mean, 1e-6))

        return {
            "is_n_purged": int(len(tb_is)),
            "median_bar_atr_pct": round(median_atr_pct, 3),
            "regime_dist_str": f"牛{bull_time_pct:.0f}%/震{range_time_pct:.0f}%/熊{bear_time_pct:.0f}%",
            "raw_mean_pct": round(float(np.sum(q_norm * raw_nets)) * 100.0, 3),
            "norm_r_atr": round(norm_r_atr_val, 3),
            "std_r_atr": round(std_r_atr, 3),
            "t_stat_r_atr": round(t_stat_r_atr, 3),
            "r_atr_values": [round(float(x), 4) for x in r_atr],
            "ipw_r_atr": round(ipw_r_atr, 3),
            "home_r_atr": round(home_r_atr, 3),
            "uncond_safety_pct": round(uncond_safety, 1),
            "ipw_safety_pct": round(ipw_safety, 1),
            "home_safety_pct": round(home_safety, 1),
            "excursion_ratio": round(exc_ratio, 3),
            "is_net_sharpe": round(float(net_sh_pack["is_net_sharpe"]), 3),
            "ann_turnover": round(float(net_sh_pack["ann_turnover"]), 2),
            "cost_to_atr_pct": round(float(fee_rate / (median_atr_pct / 100.0) * 100.0), 1),
        }


def run_stratified_pooled_gate1b_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """模块级函数入口：执行三级多标的分层池化检验 (Pre-Admission Gate 1B)。"""
    is_end_dt = kwargs.pop("is_end_dt", IS_END_DT)
    validator = RuleFactorForensicValidator(is_end_dt=is_end_dt)
    return validator.run_stratified_pooled_gate1b_audit(*args, **kwargs)


def classify_causal_market_regime(df_tf: pd.DataFrame, er_window: int = 20, er_th: float = 0.22) -> np.ndarray:
    """
    严格因果的三态市场状态分类器 (Causal 3-State Market Regime Classifier)：
    基于时刻 t 已收盘的 Kaufman 效率系数 ER(20) 与 MA20/MA60 均线结构，将市场切分为：
    - `+1` (BULL_TREND 多头趋势态): ER_20 >= er_th 且 close > ma60 且 ma20 >= ma60
    - `-1` (BEAR_TREND 空头趋势态): ER_20 >= er_th 且 close < ma60 且 ma20 <= ma60
    - `0`  (CHOPPY_RANGE 震荡盘整态): 其余时间（路径效率低或均线缠绕）
    """
    c = df_tf["close"].values.astype(np.float64)
    c_series = pd.Series(c)
    ma20 = (
        df_tf["ma20"].bfill().values if "ma20" in df_tf.columns else c_series.rolling(20, min_periods=1).mean().values
    )
    ma60 = (
        df_tf["ma60"].bfill().values if "ma60" in df_tf.columns else c_series.rolling(60, min_periods=1).mean().values
    )

    net_chg = np.abs(c_series.diff(er_window).fillna(0.0).values)
    abs_path = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(er_window, min_periods=5).sum().bfill().values
    er = net_chg / np.maximum(abs_path, 1e-6)

    regimes = np.zeros(len(c), dtype=np.int8)
    bull_mask = (er >= er_th) & (c > ma60) & (ma20 >= ma60 * 0.998)
    bear_mask = (er >= er_th) & (c < ma60) & (ma20 <= ma60 * 1.002)
    regimes[bull_mask] = 1
    regimes[bear_mask] = -1
    return regimes


def compute_empirical_bayes_shrinkage_weight(
    local_weight: float,
    cross_asset_prior_weight: float,
    is_n_purged: int,
    shrink_k: int = 60,
) -> tuple[float, float]:
    """
    分层经验贝叶斯部分池化收缩配权 (Hierarchical Empirical Bayes Shrinkage)：
    w_final = (1 - λ) * w_cross_asset_prior + λ * w_local_asset
    其中置信权重 λ = N_purged / (N_purged + κ_0)。
    - 当单品种样本量 N_purged 较小时 (如 60m/1d 低频因子)，自动向跨品种共性先验收缩，防单品种状态过拟合；
    - 当单品种样本量充足时 (如 5m/15m 日内因子)，赋予单品种微观结构个性更高权重。
    """
    lam = float(is_n_purged / max(is_n_purged + shrink_k, 1))
    w_eb = (1.0 - lam) * cross_asset_prior_weight + lam * local_weight
    return round(float(w_eb), 3), round(lam, 3)


class RollingEBWeightsResult(dict[str, np.ndarray]):
    """
    滚动经验贝叶斯配权结果容器：
    - 继承自 `dict[str, np.ndarray]`，可直接按 `res[unit_id]` 获取各单元/簇在时间轴上的因果 EB 权重序列；
    - 同时暴露 `.eb_weights`、`.prior_weights`、`.rolling_perf_weights`、`.rolling_ir` 与 `.prior_anchor_lambda` 属性。
    """

    def __init__(
        self,
        eb_weights: dict[str, np.ndarray],
        prior_weights: dict[str, float],
        rolling_perf_weights: dict[str, np.ndarray],
        rolling_ir: dict[str, np.ndarray],
        prior_anchor_lambda: float = 0.70,
    ) -> None:
        super().__init__(eb_weights)
        self.eb_weights = eb_weights
        self.prior_weights = prior_weights
        self.rolling_perf_weights = rolling_perf_weights
        self.rolling_ir = rolling_ir
        self.prior_anchor_lambda = float(prior_anchor_lambda)

    def __getitem__(self, key: str) -> Any:
        if super().__contains__(key):
            return super().__getitem__(key)
        meta_map = {
            "eb_weights": self.eb_weights,
            "prior_weights": self.prior_weights,
            "rolling_perf_weights": self.rolling_perf_weights,
            "rolling_ir": self.rolling_ir,
            "prior_anchor_lambda": self.prior_anchor_lambda,
        }
        if key in meta_map:
            return meta_map[key]
        raise KeyError(key)

    def to_dataframe(self, index: Any = None) -> pd.DataFrame:
        return pd.DataFrame(self.eb_weights, index=index)


class RollingEmpiricalBayesWeighter:
    """
    零前视滚动/扩展窗口经验贝叶斯部分池化收缩配权器 (Rolling Empirical Bayes Shrinkage Weighter, Issue #7 Phase 4)

    核心机制：
    1. 静态先验层 (Prior Anchor):
       支持结合跨品种池化先验得分/权重 Q_pool,k (如 Gate 1B 的 pooled_r_atr / safety_margin 或先验基准权重)
       与单品种样本内质量得分/权重 Q_local,i,k，按先验收缩比 λ_prior = 0.70
       (70% 锚定全品种池化先验 + 30% 锚定单品种样本量自适应调整) 生成无过拟合的基准权重先验 w_{i,k}^{prior}。
    2. 因果滚动动态更新层 (Causal Rolling / Expanding Update):
       在时间轴 t 上，仅使用严格截至 t-1 (shift(1) 零前视) 的历史已实现净收益序列
       (滚动窗口 lookback_bars 如 60~120，或扩展窗口 lookback_bars=None)，计算各单元/各簇的因果滚动信息比率 / 净夏普得分，
       映射为滚动表现权重 w_{i,k,t-1}^{rolling_perf}，并向基准先验执行 L2 经验贝叶斯收缩：
           w_{i,k,t}^{EB} = λ_prior * w_{i,k}^{prior} + (1 - λ_prior) * w_{i,k,t-1}^{rolling_perf},  λ_prior = 0.70
    """

    def __init__(
        self,
        prior_anchor_lambda: float = 0.70,
        lookback_bars: int | None = 90,
        min_bars: int = 15,
        shrink_k: int = 60,
        annualization_factor: float = 252.0,
        perf_tilt_scale: float = 0.55,
        min_weight: float = 0.05,
        max_weight: float = 1.00,
        normalize_budget: bool = False,
        *,
        lambda_prior: float | None = None,
    ) -> None:
        eff_lam = float(lambda_prior if lambda_prior is not None else prior_anchor_lambda)
        if not (0.0 <= eff_lam <= 1.0):
            raise ValueError(f"prior_anchor_lambda must be in [0, 1], got {eff_lam}")
        self.prior_anchor_lambda = eff_lam
        self.lookback_bars = lookback_bars
        self.min_bars = max(int(min_bars), 2)
        self.shrink_k = max(int(shrink_k), 1)
        self.annualization_factor = float(annualization_factor)
        self.perf_tilt_scale = float(perf_tilt_scale)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.normalize_budget = bool(normalize_budget)

    def compute_prior_anchor(
        self,
        cross_asset_prior: dict[str, float],
        local_quality: dict[str, float] | None = None,
        is_n_purged: dict[str, int] | int | None = None,
        unit_forensic_meta: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        """
        计算静态先验锚定权重 w_{i,k}^{prior}：
        - 70% (λ_prior) 锚定跨品种池化先验权重 w_pool,k；
        - 30% (1 - λ_prior) 锚定单品种样本量自适应调整后的局部权重 w_local,i,k。
        """
        lam_p = self.prior_anchor_lambda
        priors: dict[str, float] = {}

        # 若提供 unit_forensic_meta 但未直接提供 local_quality，从法医元数据提取局部质量修正
        derived_local: dict[str, float] = {}
        if local_quality is None and unit_forensic_meta:
            raw_q: dict[str, float] = {}
            for uid in cross_asset_prior:
                meta = unit_forensic_meta.get(uid, {})
                n_p = max(int(meta.get("is_n_purged", 20)), 5)
                saf = float(np.clip(float(meta.get("is_safety_margin_pct", 4.0)), 0.5, 15.0))
                r_atr = float(np.clip(float(meta.get("pooled_r_atr", meta.get("norm_r_atr", 0.5))), -1.0, 3.0))
                raw_q[uid] = float(np.log(n_p) * saf * (1.0 + 0.15 * r_atr))
            if len(raw_q) > 1:
                q_vals = np.asarray(list(raw_q.values()), dtype=np.float64)
                q_mean, q_std = float(q_vals.mean()), float(q_vals.std() + 1e-6)
            else:
                q_mean, q_std = 0.0, 1.0
            for uid, w_pool in cross_asset_prior.items():
                z_u = float(np.clip((raw_q[uid] - q_mean) / q_std, -1.2, 1.6)) if len(raw_q) > 1 else 0.0
                derived_local[uid] = float(np.clip(w_pool * (1.0 + 0.25 * z_u), self.min_weight, self.max_weight))

        eff_local_map = local_quality if local_quality is not None else (derived_local if derived_local else None)

        for uid, w_pool_raw in cross_asset_prior.items():
            w_pool = float(w_pool_raw)
            if eff_local_map is not None and uid in eff_local_map:
                w_loc_raw = float(eff_local_map[uid])
                n_obs: int | None = None
                if isinstance(is_n_purged, dict) and uid in is_n_purged:
                    n_obs = int(is_n_purged[uid])
                elif isinstance(is_n_purged, int):
                    n_obs = int(is_n_purged)
                elif unit_forensic_meta and uid in unit_forensic_meta and "is_n_purged" in unit_forensic_meta[uid]:
                    n_obs = int(unit_forensic_meta[uid]["is_n_purged"])

                if n_obs is not None:
                    eta = float(max(n_obs, 0) / max(max(n_obs, 0) + self.shrink_k, 1))
                    w_loc_adj = (1.0 - eta) * w_pool + eta * w_loc_raw
                else:
                    w_loc_adj = w_loc_raw
                w_prior = lam_p * w_pool + (1.0 - lam_p) * w_loc_adj
            else:
                w_prior = w_pool
            priors[uid] = float(np.clip(w_prior, self.min_weight, self.max_weight))

        return priors

    @staticmethod
    def build_causal_unit_net_returns(
        df_grid: pd.DataFrame,
        unit_weights_on_grid: dict[str, np.ndarray],
        fee_rate: float = 0.0002,
    ) -> dict[str, np.ndarray]:
        """
        从价格序列与各单元时序仓位构造严格因果的已实现单步净收益序列 r_{k, s}：
        在时刻 s >= 1，已实现净收益为 s-1 时刻持仓在 (s-1 -> s) 价格变动上的收益扣除调仓成本：
            r_{k, s} = w_{k, s-1} * (close_s - close_{s-1}) / close_{s-1} - fee_rate * |w_{k, s-1} - w_{k, s-2}|
        保证 r_{k, s} 仅依赖 <= s 的已知价格与 <= s-1 的持仓。
        """
        close_arr = np.asarray(df_grid["close"].values, dtype=np.float64)
        n = len(close_arr)
        realized_price_ret = np.zeros(n, dtype=np.float64)
        if n > 1:
            realized_price_ret[1:] = (close_arr[1:] - close_arr[:-1]) / np.maximum(close_arr[:-1], 1e-8)

        unit_rets: dict[str, np.ndarray] = {}
        for uid, w_raw in unit_weights_on_grid.items():
            w = np.asarray(w_raw, dtype=np.float64)
            r_u = np.zeros(n, dtype=np.float64)
            if n > 1:
                w_prev = w[:-1]
                w_prev2 = np.concatenate(([0.0], w[:-2])) if n > 2 else np.zeros_like(w_prev)
                dw_prev = np.abs(w_prev - w_prev2)
                r_u[1:] = w_prev * realized_price_ret[1:] - fee_rate * dw_prev
            unit_rets[uid] = r_u
        return unit_rets

    def compute_weights(
        self,
        cross_asset_prior: dict[str, float],
        realized_net_returns: dict[str, np.ndarray] | pd.DataFrame | None = None,
        *,
        local_quality: dict[str, float] | None = None,
        is_n_purged: dict[str, int] | int | None = None,
        unit_forensic_meta: dict[str, dict[str, Any]] | None = None,
        df_grid: pd.DataFrame | None = None,
        unit_weights_on_grid: dict[str, np.ndarray] | None = None,
        fee_rate: float = 0.0002,
        rolling_perf_override: dict[str, np.ndarray] | None = None,
    ) -> RollingEBWeightsResult:
        """
        执行零前视因果滚动/扩展窗口经验贝叶斯收缩配权：
            w_{i,k,t}^{EB} = λ_prior * w_{i,k}^{prior} + (1 - λ_prior) * w_{i,k,t-1}^{rolling_perf}
        """
        prior_map = self.compute_prior_anchor(
            cross_asset_prior=cross_asset_prior,
            local_quality=local_quality,
            is_n_purged=is_n_purged,
            unit_forensic_meta=unit_forensic_meta,
        )
        uids = list(prior_map.keys())

        if realized_net_returns is None:
            if df_grid is not None and unit_weights_on_grid is not None:
                realized_net_returns = self.build_causal_unit_net_returns(
                    df_grid, unit_weights_on_grid, fee_rate=fee_rate
                )
            elif rolling_perf_override is not None:
                first_arr = next(iter(rolling_perf_override.values()))
                realized_net_returns = {u: np.zeros(len(first_arr), dtype=np.float64) for u in uids}
            else:
                raise ValueError(
                    "Must provide either realized_net_returns or (df_grid, unit_weights_on_grid) to compute_weights."
                )

        if isinstance(realized_net_returns, pd.DataFrame):
            ret_dict = {
                col: realized_net_returns[col].to_numpy(dtype=np.float64) for col in uids if col in realized_net_returns
            }
        else:
            ret_dict = {
                u: np.asarray(realized_net_returns[u], dtype=np.float64) for u in uids if u in realized_net_returns
            }

        if not ret_dict:
            raise ValueError("No matching unit keys found between cross_asset_prior and realized_net_returns.")

        n = len(next(iter(ret_dict.values())))
        lam_p = self.prior_anchor_lambda
        sqrt_ann = np.sqrt(self.annualization_factor)
        w_lb = self.lookback_bars if (self.lookback_bars is not None and self.lookback_bars > 0) else None

        rolling_ir_map: dict[str, np.ndarray] = {u: np.zeros(n, dtype=np.float64) for u in uids}
        rolling_perf_map: dict[str, np.ndarray] = {u: np.full(n, prior_map[u], dtype=np.float64) for u in uids}
        eb_weights_map: dict[str, np.ndarray] = {u: np.full(n, prior_map[u], dtype=np.float64) for u in uids}

        if rolling_perf_override is not None:
            # 若显式传入滚动表现目标序列 (调试或直接收缩验证)，严格执行 shift(1) 后按 L2 公式线性收缩
            for u in uids:
                if u in rolling_perf_override:
                    raw_perf = np.asarray(rolling_perf_override[u], dtype=np.float64)
                    shifted_perf = np.empty(n, dtype=np.float64)
                    shifted_perf[0] = prior_map[u]
                    if n > 1:
                        shifted_perf[1:] = np.clip(raw_perf[:-1], self.min_weight, self.max_weight)
                    rolling_perf_map[u] = shifted_perf
                    eb_weights_map[u] = lam_p * prior_map[u] + (1.0 - lam_p) * shifted_perf
            return RollingEBWeightsResult(
                eb_weights=eb_weights_map,
                prior_weights=prior_map,
                rolling_perf_weights=rolling_perf_map,
                rolling_ir=rolling_ir_map,
                prior_anchor_lambda=lam_p,
            )

        prior_sum = sum(prior_map.values())
        for t in range(self.min_bars, n):
            # 严格只取 [start_idx, t) 即截至 t-1 的历史已实现净收益 (shift(1) 零前视)
            start_idx = max(0, t - w_lb) if w_lb is not None else 0
            raw_perf_t: dict[str, float] = {}
            for u in uids:
                r_win = ret_dict[u][start_idx:t]
                if len(r_win) < self.min_bars:
                    ir_val = 0.0
                else:
                    std_v = float(np.std(r_win, ddof=1))
                    ir_val = float((np.mean(r_win) / std_v) * sqrt_ann) if std_v > 1e-10 else 0.0
                rolling_ir_map[u][t] = ir_val
                tilt = float(np.tanh(ir_val / 2.0))
                w_perf = float(
                    np.clip(prior_map[u] * (1.0 + self.perf_tilt_scale * tilt), self.min_weight, self.max_weight)
                )
                raw_perf_t[u] = w_perf

            if self.normalize_budget and prior_sum > 1e-9:
                cur_sum = sum(raw_perf_t.values())
                if cur_sum > 1e-9:
                    scale_f = prior_sum / cur_sum
                    for u in uids:
                        raw_perf_t[u] = float(np.clip(raw_perf_t[u] * scale_f, self.min_weight, self.max_weight))

            for u in uids:
                w_perf_u = raw_perf_t[u]
                rolling_perf_map[u][t] = w_perf_u
                eb_weights_map[u][t] = lam_p * prior_map[u] + (1.0 - lam_p) * w_perf_u

        return RollingEBWeightsResult(
            eb_weights=eb_weights_map,
            prior_weights=prior_map,
            rolling_perf_weights=rolling_perf_map,
            rolling_ir=rolling_ir_map,
            prior_anchor_lambda=lam_p,
        )


def compute_rolling_empirical_bayes_weights(
    cross_asset_prior: dict[str, float],
    realized_net_returns: dict[str, np.ndarray] | pd.DataFrame | None = None,
    *,
    local_quality: dict[str, float] | None = None,
    is_n_purged: dict[str, int] | int | None = None,
    unit_forensic_meta: dict[str, dict[str, Any]] | None = None,
    df_grid: pd.DataFrame | None = None,
    unit_weights_on_grid: dict[str, np.ndarray] | None = None,
    prior_anchor_lambda: float = 0.70,
    lookback_bars: int | None = 90,
    min_bars: int = 15,
    shrink_k: int = 60,
    annualization_factor: float = 252.0,
    perf_tilt_scale: float = 0.55,
    min_weight: float = 0.05,
    max_weight: float = 1.00,
    normalize_budget: bool = False,
    fee_rate: float = 0.0002,
    rolling_perf_override: dict[str, np.ndarray] | None = None,
) -> RollingEBWeightsResult:
    """
    模块级便捷函数：计算零前视滚动/扩展窗口经验贝叶斯收缩权重序列 w_{i,k,t}^{EB}。
    """
    weighter = RollingEmpiricalBayesWeighter(
        prior_anchor_lambda=prior_anchor_lambda,
        lookback_bars=lookback_bars,
        min_bars=min_bars,
        shrink_k=shrink_k,
        annualization_factor=annualization_factor,
        perf_tilt_scale=perf_tilt_scale,
        min_weight=min_weight,
        max_weight=max_weight,
        normalize_budget=normalize_budget,
    )
    return weighter.compute_weights(
        cross_asset_prior=cross_asset_prior,
        realized_net_returns=realized_net_returns,
        local_quality=local_quality,
        is_n_purged=is_n_purged,
        unit_forensic_meta=unit_forensic_meta,
        df_grid=df_grid,
        unit_weights_on_grid=unit_weights_on_grid,
        fee_rate=fee_rate,
        rolling_perf_override=rolling_perf_override,
    )
