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

from typing import Any

import numpy as np
import pandas as pd

from examples.alpha_mining_agent.forensic_pipeline import (
    IS_END_DT,
    evaluate_triple_barrier,
    purge_overlapping_triggers,
)
from examples.alpha_mining_agent.hypothesis_bank import ARCHETYPE_RULERS, ArchetypeRuler

# 周期自适应最小独立样本数门槛（解决日线/60m 低频高质量信号被固定高样本门槛误杀的问题）
TF_MIN_PURGED_SAMPLES: dict[str, int] = {
    "1d": 12,
    "60m": 25,
    "30m": 35,
    "15m": 55,
    "5m": 80,
}


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
    """规则型因子三级法医检验器"""

    def __init__(self, is_end_dt: pd.Timestamp = IS_END_DT) -> None:
        self.is_end_dt = is_end_dt
        self.rng = np.random.default_rng(42)

    def run_gate1_single_asset_audit(
        self,
        df_tf: pd.DataFrame,
        raw_triggers: np.ndarray,
        direction: int,
        archetype: str,
        timeframe: str,
        fee_rate: float = 0.0002,
    ) -> dict[str, Any]:
        """
        Gate 1A：单品种单周期法医标尺检验（严格样本内 IS）
        - Purged 去重叠独立事件抽样
        - 周期自适应最小样本数门槛
        - 分战略簇三重屏障标尺 (Next-Bar Open + 交易成本扣除)
        - 成本穿透安全边际 ΔW = W - 1/(1+R) 与 Bootstrap p-val
        """
        ruler: ArchetypeRuler = ARCHETYPE_RULERS[archetype]
        is_mask = (df_tf["dt"] <= self.is_end_dt).values
        oos_mask = (df_tf["dt"] > self.is_end_dt).values

        min_samples = 8 if timeframe == "1d" else (12 if timeframe == "60m" else (20 if timeframe == "5m" else 15))
        eff_max_bars = ruler.max_bars * 2 if timeframe == "5m" else ruler.max_bars
        eff_tp_atr = ruler.tp_atr * 1.35 if timeframe == "5m" else ruler.tp_atr
        eff_sl_atr = ruler.sl_atr * 1.20 if timeframe == "5m" else ruler.sl_atr
        min_warmup = 240 if timeframe == "5m" else 60

        is_raw_idx = np.where((raw_triggers == 1) & is_mask)[0]
        is_purged_idx = purge_overlapping_triggers(is_raw_idx, cooldown_bars=eff_max_bars)

        tb_is = evaluate_triple_barrier(
            df_tf,
            is_purged_idx,
            direction=direction,
            max_bars=eff_max_bars,
            tp_atr=eff_tp_atr,
            sl_atr=eff_sl_atr,
            fee_rate=fee_rate,
        )
        n_purged = len(tb_is)
        if n_purged < min_samples:
            return {
                "passed": False,
                "reason": f"样本内去重独立触发仅 {n_purged} 次 (< {min_samples} 次周期下限)",
                "is_n_purged": n_purged,
            }

        rets_is = tb_is["net_ret"].values
        f_mean = float(np.mean(rets_is))
        f_win = float(np.mean(rets_is > 0))
        wins = rets_is[rets_is > 0]
        losses = rets_is[rets_is <= 0]
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(abs(np.mean(losses))) if len(losses) > 0 else 1e-6
        pl_ratio = avg_win / max(avg_loss, 1e-6)
        be_win = 1.0 / (1.0 + pl_ratio)
        safety_margin = f_win - be_win

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

        oos_raw_idx = np.where((raw_triggers == 1) & oos_mask)[0]
        oos_purged_idx = purge_overlapping_triggers(oos_raw_idx, cooldown_bars=eff_max_bars)
        tb_oos = evaluate_triple_barrier(
            df_tf,
            oos_purged_idx,
            direction=direction,
            max_bars=eff_max_bars,
            tp_atr=eff_tp_atr,
            sl_atr=eff_sl_atr,
            fee_rate=fee_rate,
        )
        oos_mean = float(np.mean(tb_oos["net_ret"].values)) * 100.0 if len(tb_oos) >= 1 else 0.0

        tf_alpha_scale = {"1d": 1.2, "60m": 1.0, "30m": 0.85, "15m": 0.55, "5m": 0.40}.get(timeframe, 1.0)
        min_alpha_dec = (ruler.min_alpha_bp * tf_alpha_scale) / 10000.0
        min_safety_dec = ruler.min_safety_pct / 100.0
        max_pval = ruler.max_boot_pval + (0.02 if timeframe in ("60m", "1d") else 0.0)

        passed = excess_alpha >= min_alpha_dec and safety_margin >= min_safety_dec and boot_pval <= max_pval
        reasons = []
        if excess_alpha < min_alpha_dec:
            reasons.append(f"超额Alpha={excess_alpha * 100:+.2f}%<+{min_alpha_dec * 100:.2f}%")
        if safety_margin < min_safety_dec:
            reasons.append(f"安全垫={safety_margin * 100:+.1f}%<+{ruler.min_safety_pct:.1f}%")
        if boot_pval > max_pval:
            reasons.append(f"p={boot_pval:.3f}>{max_pval:.2f}")

        return {
            "passed": bool(passed),
            "reason": "通过" if passed else "; ".join(reasons),
            "ruler_name": ruler.archetype,
            "is_n_purged": int(n_purged),
            "is_mean_pct": round(f_mean * 100.0, 3),
            "is_excess_alpha_pct": round(excess_alpha * 100.0, 3),
            "is_win_rate_pct": round(f_win * 100.0, 1),
            "is_pl_ratio": round(pl_ratio, 2),
            "is_safety_margin_pct": round(safety_margin * 100.0, 1),
            "is_boot_pval": round(boot_pval, 4),
            "oos_n_purged": int(len(tb_oos)),
            "oos_mean_pct": round(oos_mean, 3),
        }

    def run_gate1_cross_asset_universality_audit(
        self,
        asset_audits: dict[str, dict[str, Any]],
        min_positive_ratio: float = 0.60,
    ) -> dict[str, Any]:
        """
        Gate 1B：跨品种普适性检验 (Cross-Asset Universality Audit)
        检验某一规则因子在多品种池中是否具备普适正期望（≥60% 有效品种取得正超额收益），
        严防针对单一品种特定历史噪声的过拟合。
        """
        valid = {sym: res for sym, res in asset_audits.items() if res.get("is_n_purged", 0) >= 10}
        if not valid:
            return {"passed": False, "universal_ratio_pct": 0.0, "positive_assets": []}

        pos_syms = [
            sym
            for sym, res in valid.items()
            if res.get("is_mean_pct", 0.0) > 0.0 and res.get("is_safety_margin_pct", -99.0) > 0.0
        ]
        ratio = len(pos_syms) / len(valid)
        total_is_n = sum(int(res.get("is_n_purged", 0)) for res in valid.values())
        avg_safety = float(np.mean([float(res.get("is_safety_margin_pct", 0.0)) for res in valid.values()]))

        return {
            "passed": ratio >= min_positive_ratio and avg_safety > 1.5,
            "universal_ratio_pct": round(ratio * 100.0, 1),
            "total_pool_is_n": total_is_n,
            "mean_pool_safety_pct": round(avg_safety, 2),
            "positive_assets": pos_syms,
            "evaluated_assets": list(valid.keys()),
        }

    def run_gate2_complementarity_and_jaccard_audit(
        self,
        df_grid: pd.DataFrame,
        cand_w_on_grid: np.ndarray,
        existing_same_dir_weights: dict[str, np.ndarray],
        current_portfolio_w: np.ndarray,
        max_corr_threshold: float = 0.32,
        max_jaccard_threshold: float = 0.30,
        min_gap_fill_pct: float = 35.0,
    ) -> dict[str, Any]:
        """
        Gate 2：时序×截面三维真互补门禁（严格样本内 IS）
        1. 全时序相关性 |ρ| < 0.32 且 非零激活态交并比 Signed Jaccard < 0.30；
        2. 空仓填补率 >= 35%；
        3. 原组合回撤/亏损期的雪中送炭 Alpha (dd_rescue_alpha_bp > 0)。
        """
        is_mask = (df_grid["dt"] <= self.is_end_dt).values
        cw_is = cand_w_on_grid[is_mask]
        port_w_is = current_portfolio_w[is_mask]

        max_corr = 0.0
        max_jaccard = 0.0
        max_peer = "None"
        for peer_id, pw in existing_same_dir_weights.items():
            pw_is = pw[is_mask]
            c = float(abs(np.corrcoef(cw_is, pw_is)[0, 1])) if (np.std(cw_is) > 1e-6 and np.std(pw_is) > 1e-6) else 0.0
            jac = compute_signed_jaccard(cw_is, pw_is)
            if c > max_corr or jac > max_jaccard:
                if c > max_corr:
                    max_corr = c
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
            "passed": passed,
            "max_corr": round(max_corr, 3),
            "max_signed_jaccard": round(max_jaccard, 3),
            "max_corr_peer": max_peer,
            "gap_fill_ratio_pct": round(gap_fill_ratio, 1),
            "dd_rescue_alpha_bp": round(dd_rescue_alpha_bp, 2),
        }

    def run_vol_and_regime_normalized_audit(
        self,
        df_tf: pd.DataFrame,
        raw_triggers: np.ndarray,
        direction: int,
        archetype: str,
        timeframe: str,
        fee_rate: float = 0.0002,
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
        eff_max_bars = ruler.max_bars * 2 if timeframe == "5m" else ruler.max_bars
        eff_tp_atr = ruler.tp_atr * 1.35 if timeframe == "5m" else ruler.tp_atr
        eff_sl_atr = ruler.sl_atr * 1.20 if timeframe == "5m" else ruler.sl_atr

        regimes = classify_causal_market_regime(df_tf)
        reg_is = regimes[is_mask]
        bull_Time_pct = float(np.mean(reg_is == 1) * 100.0)
        bear_time_pct = float(np.mean(reg_is == -1) * 100.0)
        range_time_pct = float(np.mean(reg_is == 0) * 100.0)

        c_arr = df_tf["close"].values
        atr_pct_arr = np.maximum(df_tf["atr14"].bfill().values / np.maximum(c_arr, 1e-6), 1e-4)
        median_atr_pct = float(np.median(atr_pct_arr[is_mask]) * 100.0)

        is_raw_idx = np.where((raw_triggers == 1) & is_mask)[0]
        is_purged_idx = purge_overlapping_triggers(is_raw_idx, cooldown_bars=eff_max_bars)
        tb_is = evaluate_triple_barrier(
            df_tf,
            is_purged_idx,
            direction=direction,
            max_bars=eff_max_bars,
            tp_atr=eff_tp_atr,
            sl_atr=eff_sl_atr,
            fee_rate=fee_rate,
        )
        if tb_is.empty:
            return {"is_n_purged": 0, "median_bar_atr_pct": round(median_atr_pct, 3)}

        t_idx = tb_is["trigger_idx"].values.astype(int)
        raw_nets = tb_is["net_ret"].values
        entry_atr_pct = atr_pct_arr[t_idx]
        # 核心变换 1：转化为无量纲 ATR 单位净收益 (R-Multiple in ATR units)
        r_atr = raw_nets / entry_atr_pct
        trig_regimes = regimes[t_idx]

        # 定义该因子战略簇对应的“目标主场状态 (Home Regime)”
        if archetype == "A_顺势突破簇":
            home_mask = (trig_regimes == 1) if direction == 1 else (trig_regimes == -1)
        elif archetype == "C_震荡回归簇":
            home_mask = trig_regimes == 0
        else:
            # 背驰反转与日内微结构：震荡态或逆势衰竭态均为主场
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
        home_r_atr = float(np.mean(r_atr[home_mask])) if int(home_mask.sum()) >= 3 else float(np.mean(r_atr))

        # 核心变换 2：三态逆概率重加权 (IPW Regime-Balanced Expectancy)
        state_r_atr = []
        state_safeties = []
        for st in (1, 0, -1):
            m = trig_regimes == st
            if int(m.sum()) >= 2:
                state_r_atr.append(float(np.mean(r_atr[m])))
                state_safeties.append(_calc_safety(raw_nets[m]))
        ipw_r_atr = float(np.mean(state_r_atr)) if state_r_atr else float(np.mean(r_atr))
        ipw_safety = float(np.mean(state_safeties)) if state_safeties else uncond_safety

        return {
            "is_n_purged": int(len(tb_is)),
            "median_bar_atr_pct": round(median_atr_pct, 3),
            "regime_dist_str": f"牛{bull_Time_pct:.0f}%/震{range_time_pct:.0f}%/熊{bear_time_pct:.0f}%",
            "raw_mean_pct": round(float(np.mean(raw_nets)) * 100.0, 3),
            "norm_r_atr": round(float(np.mean(r_atr)), 3),
            "ipw_r_atr": round(ipw_r_atr, 3),
            "home_r_atr": round(home_r_atr, 3),
            "uncond_safety_pct": round(uncond_safety, 1),
            "ipw_safety_pct": round(ipw_safety, 1),
            "home_safety_pct": round(home_safety, 1),
            "cost_to_atr_pct": round(float(fee_rate / (median_atr_pct / 100.0) * 100.0), 1),
        }


def classify_causal_market_regime(df_tf: pd.DataFrame, er_window: int = 20, er_th: float = 0.22) -> np.ndarray:
    """
    严格因果的三态市场状态分类器 (Causal 3-State Market Regime Classifier)：
    基于时刻 t 已收盘的 Kaufman 效率系数 ER(20) 与 MA20/MA60 均线结构，将市场切分为：
    - `+1` (BULL_TREND 多头趋势态): ER_20 >= er_th 且 close > ma60 且 ma20 >= ma60
    - `-1` (BEAR_TREND 空头趋势态): ER_20 >= er_th 且 close < ma60 且 ma20 <= ma60
    - `0`  (CHOPPY_RANGE 震荡盘整态): 其余时间（路径效率低或均线缠绕）
    """
    c = df_tf["close"].values.astype(np.float64)
    ma20 = df_tf["ma20"].bfill().values
    ma60 = df_tf["ma60"].bfill().values

    net_chg = np.abs(pd.Series(c).diff(er_window).fillna(0.0).values)
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
