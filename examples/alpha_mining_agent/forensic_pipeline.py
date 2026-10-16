#!/usr/bin/env python
"""
分方向定制标尺 + 三维真互补门禁流水线
(Archetype-Specific Forensic Rulers & 3D Complementarity Pipeline)

核心升级：
1. 第一关 [Gate 1 · 分方向定制法医标尺 (Archetype-Specific Ruler)]：
   - 趋势突破簇用趋势尺 (+2.2*ATR / -1.1*ATR / 16根)
   - 背驰反转簇用反转尺 (+1.35*ATR / -0.95*ATR / 8根)
   - 震荡回归簇用回归尺 (+1.15*ATR / -0.85*ATR / 6根)
   - 日内微结构簇用日内尺 (+1.25*ATR / -0.85*ATR / 6根)
   - 在候选信号的专属周期 (1d / 60m / 30m / 15m) 上独立执行 Purged + Next-Open + 2BP + Bootstrap 审计
2. 第二关 [Gate 2 · 三维真互补门禁 (3D Complementarity Gate)]：
   - 维度 A (生态位配额)：检查该战略方向是否已达配额上限 (防止扎堆单一方向)
   - 维度 B (时序低相关)：与已入库同向信号在主时间轴上的最大相关系数 |ρ_max| < 0.32
   - 维度 C (空仓填补 + 回撤期雪中送炭)：空仓期填补率 >= 35% 且 在原组合亏损/回撤时刻具备正超额收益 (dd_rescue_alpha_bp > 0)
3. 第三关 [Gate 3 · 专属周期独立路由与组合边际增益]：
   - 仅在该信号通过检验的专属周期 tf 上挂载权重（不再强制三周期一刀切捆绑），验证样本内 (2020-2023) 边际增益。
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from czsc import WeightBacktest
from examples.alpha_mining_agent.hypothesis_bank import (
    ALL_ARCHETYPES,
    ALL_SIGNAL_SPECS,
    ARCH_A,
    ARCHETYPE_RULERS,
    ATOMIC_EXIT_MODE,
    TF_SCALE_MAP,
    ArchetypeRuler,
    compute_signal_triggers_and_weights,
)

IS_END_DT = pd.Timestamp("2023-12-31 23:59:59")


def purge_overlapping_triggers(trigger_indices: np.ndarray, cooldown_bars: int) -> np.ndarray:
    selected = []
    last_idx = -999999
    for idx in trigger_indices:
        if idx >= last_idx + cooldown_bars:
            selected.append(idx)
            last_idx = idx
    return np.array(selected, dtype=int)


def evaluate_triple_barrier(
    df: pd.DataFrame,
    entry_indices: np.ndarray,
    direction: int,
    max_bars: int,
    tp_atr: float,
    sl_atr: float,
    fee_rate: float = 0.0002,
    weights_tf: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    评估分战略簇三重屏障标尺 (Next-Bar Open 入场 + 止盈/止损/时间屏障 + 双边费率扣除)，
    同步提取：
    1. 触发时刻连续质量权重 `quality_w = |w_{t_k}| ∈ (0, 1]`（供连续强度因子质量加权审计）；
    2. 持仓路径内的最大有利偏离 `mfe_atr` 与最大不利偏离 `mae_atr`（以入场时刻因果 ATR 无量纲化）。
    """
    n = len(df)
    o_arr = df["open"].values
    h_arr = df["high"].values
    l_arr = df["low"].values
    c_arr = df["close"].values
    atr_arr = df["atr14"].values

    records = []
    for idx in entry_indices:
        idx_int = int(idx)
        if idx_int + 1 >= n:
            continue
        entry_p = float(o_arr[idx_int + 1])
        atr = float(atr_arr[idx_int])
        if np.isnan(atr) or atr <= 0 or entry_p <= 0:
            continue

        q_w = 1.0
        if weights_tf is not None and 0 <= idx_int < len(weights_tf):
            raw_q = abs(float(weights_tf[idx_int]))
            if raw_q <= 1e-6 and idx_int + 1 < len(weights_tf):
                raw_q = abs(float(weights_tf[idx_int + 1]))
            q_w = float(np.clip(raw_q, 1e-4, 1.0)) if raw_q > 1e-6 else 1.0

        tp_dist = tp_atr * atr
        sl_dist = sl_atr * atr
        exit_p = entry_p
        end_bar = min(idx_int + 1 + max_bars, n - 1)
        exit_bar = end_bar
        mfe_atr = 0.0
        mae_atr = 0.0

        for j in range(idx_int + 1, end_bar + 1):
            hi, lo, cl = float(h_arr[j]), float(l_arr[j]), float(c_arr[j])
            if direction == 1:
                fav = max(0.0, (hi - entry_p) / atr)
                adv = max(0.0, (entry_p - lo) / atr)
                mfe_atr = max(mfe_atr, fav)
                mae_atr = max(mae_atr, adv)
                if lo <= entry_p - sl_dist:
                    exit_p = entry_p - sl_dist
                    exit_bar = j
                    break
                if hi >= entry_p + tp_dist:
                    exit_p = entry_p + tp_dist
                    exit_bar = j
                    break
            else:
                fav = max(0.0, (entry_p - lo) / atr)
                adv = max(0.0, (hi - entry_p) / atr)
                mfe_atr = max(mfe_atr, fav)
                mae_atr = max(mae_atr, adv)
                if hi >= entry_p + sl_dist:
                    exit_p = entry_p + sl_dist
                    exit_bar = j
                    break
                if lo <= entry_p - tp_dist:
                    exit_p = entry_p - tp_dist
                    exit_bar = j
                    break
            if j == end_bar:
                exit_p = cl
                exit_bar = j

        raw_ret = (exit_p - entry_p) / entry_p if direction == 1 else (entry_p - exit_p) / entry_p
        net_ret = raw_ret - fee_rate
        records.append(
            {
                "trigger_idx": idx_int,
                "net_ret": float(net_ret),
                "quality_w": float(q_w),
                "mfe_atr": float(mfe_atr),
                "mae_atr": float(mae_atr),
                "bars_held": int(exit_bar - idx_int),
            }
        )

    return pd.DataFrame(records)


def compute_forward_rank_ic_profile(
    df_tf: pd.DataFrame,
    weights_tf: np.ndarray,
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20),
    is_end_dt: pd.Timestamp = IS_END_DT,
    eps: float = 1e-4,
) -> dict[str, Any]:
    """
    针对连续强度信号 S_{i,t} ∈ [-1, 1] 计算前向多期 Rank IC / ICIR 衰减曲线 (h ∈ {1, 3, 5, 10, 20})
    以及信号一阶自相关系数、隐含半衰期与单根换手率。
    """
    is_mask = (df_tf["dt"] <= is_end_dt).values
    if int(is_mask.sum()) < 25:
        is_mask = np.ones(len(df_tf), dtype=bool)

    w_is = np.asarray(weights_tf[is_mask], dtype=np.float64)
    c_is = np.asarray(df_tf["close"].values[is_mask], dtype=np.float64)
    n = len(w_is)

    active = np.abs(w_is) > eps
    rank_ic_curve: dict[int, float] = {}
    icir_curve: dict[int, float] = {}

    for h in horizons:
        if n <= h + 5:
            rank_ic_curve[h] = 0.0
            icir_curve[h] = 0.0
            continue
        fwd_ret = (c_is[h:] - c_is[:-h]) / np.maximum(c_is[:-h], 1e-6)
        w_slice = w_is[:-h]
        act_h = active[:-h]
        # 若激活集样本充足且具连续区分度，在激活集上计算 Spearman Rank IC；否则在全样本计算
        if int(act_h.sum()) >= 10 and np.std(w_slice[act_h]) > 1e-6:
            s_vec = w_slice[act_h]
            r_vec = fwd_ret[act_h]
        elif np.std(w_slice) > 1e-6:
            s_vec = w_slice
            r_vec = fwd_ret
        else:
            rank_ic_curve[h] = 0.0
            icir_curve[h] = 0.0
            continue

        s_rank = pd.Series(s_vec).rank(method="average").to_numpy(dtype=np.float64)
        r_rank = pd.Series(r_vec).rank(method="average").to_numpy(dtype=np.float64)
        ic_val = float(np.corrcoef(s_rank, r_rank)[0, 1]) if np.std(s_rank) > 1e-6 and np.std(r_rank) > 1e-6 else 0.0
        rank_ic_curve[h] = round(ic_val if np.isfinite(ic_val) else 0.0, 4)

        n_blocks = min(8, max(2, len(s_vec) // 12))
        if len(s_vec) >= n_blocks * 5:
            chunks_s = np.array_split(s_vec, n_blocks)
            chunks_r = np.array_split(r_vec, n_blocks)
            blk_ics = []
            for cs, cr in zip(chunks_s, chunks_r, strict=False):
                if len(cs) >= 4 and np.std(cs) > 1e-6 and np.std(cr) > 1e-6:
                    rk_s = pd.Series(cs).rank(method="average").to_numpy(dtype=np.float64)
                    rk_r = pd.Series(cr).rank(method="average").to_numpy(dtype=np.float64)
                    c_b = float(np.corrcoef(rk_s, rk_r)[0, 1])
                    if np.isfinite(c_b):
                        blk_ics.append(c_b)
            if len(blk_ics) >= 2:
                ic_arr = np.asarray(blk_ics, dtype=np.float64)
                icir_val = float(ic_arr.mean() / max(float(ic_arr.std(ddof=1)), 1e-6))
            else:
                icir_val = ic_val
        else:
            icir_val = ic_val
        icir_curve[h] = round(icir_val if np.isfinite(icir_val) else 0.0, 4)

    if n > 2 and np.std(w_is[:-1]) > 1e-6 and np.std(w_is[1:]) > 1e-6:
        rho1 = float(np.corrcoef(w_is[:-1], w_is[1:])[0, 1])
        if not np.isfinite(rho1):
            rho1 = 0.0
    else:
        rho1 = 0.0
    half_life = float(-np.log(2.0) / np.log(np.clip(rho1, 0.01, 0.999))) if rho1 > 0.01 else 0.0
    bar_turnover = float(np.mean(np.abs(np.diff(w_is, prepend=0.0)))) if n > 0 else 0.0
    ic_vals = list(rank_ic_curve.values())

    return {
        "rank_ic_curve": rank_ic_curve,
        "icir_curve": icir_curve,
        "peak_rank_ic": round(float(max(ic_vals, default=0.0)), 4),
        "mean_rank_ic": round(float(np.mean(ic_vals)) if ic_vals else 0.0, 4),
        "autocorr_lag1": round(rho1, 4),
        "half_life_bars": round(half_life, 2),
        "bar_turnover": round(bar_turnover, 4),
    }


def compute_single_factor_net_sharpe_and_turnover(
    df_tf: pd.DataFrame,
    weights_tf: np.ndarray,
    timeframe: str = "30m",
    fee_rate: float = 0.0002,
    is_end_dt: pd.Timestamp = IS_END_DT,
) -> dict[str, float]:
    """
    计算单因子时序持仓序列扣除双边交易成本后的样本内 Net Sharpe 与年化换手率，
    拦截高频刷单净亏损因子。
    """
    is_mask = (df_tf["dt"] <= is_end_dt).values
    if int(is_mask.sum()) < 20:
        is_mask = np.ones(len(df_tf), dtype=bool)

    w_is = np.asarray(weights_tf[is_mask], dtype=np.float64)
    c_is = np.asarray(df_tf["close"].values[is_mask], dtype=np.float64)
    n = len(c_is)
    if n < 2 or np.max(np.abs(w_is)) <= 1e-6:
        return {"is_net_sharpe": 0.0, "ann_turnover": 0.0, "is_net_cum_ret_pct": 0.0}

    fwd_ret = np.zeros(n, dtype=np.float64)
    fwd_ret[:-1] = (c_is[1:] - c_is[:-1]) / np.maximum(c_is[:-1], 1e-6)
    dw = np.abs(np.diff(w_is, prepend=0.0))
    bar_net = w_is * fwd_ret - dw * fee_rate

    dts = pd.to_datetime(df_tf["dt"].iloc[is_mask]).dt.date.values
    uniq_dates, day_ids = np.unique(dts, return_inverse=True)
    if len(uniq_dates) >= 5:
        daily_net = np.bincount(day_ids, weights=bar_net)
        std_d = float(daily_net.std(ddof=1)) if len(daily_net) > 1 else 0.0
        net_sharpe = float((daily_net.mean() / std_d) * np.sqrt(252.0)) if std_d > 1e-12 else 0.0
        n_years = max(len(uniq_dates) / 252.0, 1e-6)
        ann_turnover = float(dw.sum() / n_years)
    else:
        bars_per_yr = {"1d": 252.0, "60m": 252.0 * 4, "30m": 252.0 * 8, "15m": 252.0 * 16, "5m": 252.0 * 48}.get(
            timeframe, 252.0 * 8
        )
        std_b = float(bar_net.std(ddof=1)) if len(bar_net) > 1 else 0.0
        net_sharpe = float((bar_net.mean() / std_b) * np.sqrt(bars_per_yr)) if std_b > 1e-12 else 0.0
        ann_turnover = float(dw.mean() * bars_per_yr)

    return {
        "is_net_sharpe": round(net_sharpe, 3),
        "ann_turnover": round(ann_turnover, 2),
        "is_net_cum_ret_pct": round(float(bar_net.sum() * 100.0), 3),
    }


def _compute_weighted_edge_metrics(rets: np.ndarray, q_weights: np.ndarray) -> dict[str, float]:
    """根据质量权重 q_k ∈ (0, 1] 计算加权期望净收益、加权胜率、加权盈亏比与加权安全边际。"""
    if len(rets) == 0:
        return {"mean": 0.0, "win_rate": 0.0, "pl_ratio": 0.0, "safety_margin": 0.0}
    q = np.clip(np.asarray(q_weights, dtype=np.float64), 1e-4, 1.0)
    q_sum = float(np.sum(q))
    if q_sum <= 1e-12:
        q = np.ones_like(rets, dtype=np.float64)
        q_sum = float(len(rets))

    w_mean = float(np.sum(q * rets) / q_sum)
    win_mask = rets > 0
    loss_mask = ~win_mask
    w_win_rate = float(np.sum(q[win_mask]) / q_sum)
    avg_win = float(np.sum(q[win_mask] * rets[win_mask]) / np.sum(q[win_mask])) if np.any(win_mask) else 0.0
    avg_loss = (
        float(np.sum(q[loss_mask] * np.abs(rets[loss_mask])) / np.sum(q[loss_mask])) if np.any(loss_mask) else 1e-6
    )
    pl_ratio = avg_win / max(avg_loss, 1e-6)
    be_win = 1.0 / (1.0 + pl_ratio)
    safety_margin = w_win_rate - be_win
    return {
        "mean": w_mean,
        "win_rate": w_win_rate,
        "pl_ratio": pl_ratio,
        "safety_margin": safety_margin,
    }


def run_gate1_forensic_audit(
    df_tf: pd.DataFrame,
    triggers_tf: np.ndarray,
    direction: int,
    ruler: ArchetypeRuler,
    timeframe: str,
    rng: np.random.Generator | None = None,
    weights_tf: np.ndarray | None = None,
    fee_rate: float = 0.0002,
) -> dict[str, Any]:
    """
    使用该战略方向专属的法医三重屏障标尺 (ruler) 在特定周期 df_tf 上执行样本内 (2020-2023) 审计。
    Phase 2 升级：
    1. 支持传入 `weights_tf` 激活连续质量分 Q_{i,t} 加权期望收益与加权安全边际；
    2. 考核持仓路径 MFE/MAE 偏离比 (`excursion_ratio >= ruler.min_excursion_ratio`)；
    3. 考核单因子扣费后 `is_net_sharpe` 与前向多期 Rank IC / ICIR 衰减曲线。
    """
    del rng
    is_mask = (df_tf["dt"] <= IS_END_DT).values
    if int(is_mask.sum()) < 25:
        is_mask = np.ones(len(df_tf), dtype=bool)
    oos_mask = (df_tf["dt"] > IS_END_DT).values

    min_samples = 8 if timeframe == "1d" else (12 if timeframe == "60m" else (20 if timeframe == "5m" else 15))
    eff_max_bars = ruler.max_bars * 2 if timeframe == "5m" else ruler.max_bars
    eff_tp_atr = ruler.tp_atr * 1.35 if timeframe == "5m" else ruler.tp_atr
    eff_sl_atr = ruler.sl_atr * 1.20 if timeframe == "5m" else ruler.sl_atr
    min_warmup = min(240 if timeframe == "5m" else 60, max(10, int(is_mask.sum()) // 4))

    trig_bool = np.asarray(triggers_tf, dtype=bool)
    is_raw_idx = np.where(trig_bool & is_mask)[0]
    is_purged_idx = purge_overlapping_triggers(is_raw_idx, cooldown_bars=eff_max_bars)

    tb_is = evaluate_triple_barrier(
        df_tf,
        is_purged_idx,
        direction,
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

    # MFE / MAE 路径偏离比审计
    mfe_arr = tb_is["mfe_atr"].values
    mae_arr = tb_is["mae_atr"].values
    q_norm = q_w / max(float(np.sum(q_w)), 1e-12)
    mean_mfe_atr = float(np.sum(q_norm * mfe_arr))
    mean_mae_atr = float(np.sum(q_norm * mae_arr))
    excursion_ratio = float(mean_mfe_atr / max(mean_mae_atr, 1e-6))

    # 500 轮随机入场置换检验 (同样使用该专属标尺)
    valid_pool = np.where(
        is_mask & (np.arange(len(df_tf)) >= min_warmup) & (np.arange(len(df_tf)) < len(df_tf) - eff_max_bars - 2)
    )[0]
    if len(valid_pool) < 5:
        valid_pool = np.arange(max(1, len(df_tf) - eff_max_bars - 2))
    unit_seed = (
        4242
        + int(np.sum(is_raw_idx))
        + sum((idx + 1) * ord(c) for idx, c in enumerate(f"{ruler.archetype}_{timeframe}"))
    )
    local_rng = np.random.default_rng(unit_seed)
    rand_pool_idx = local_rng.choice(valid_pool, size=min(1000, len(valid_pool)), replace=False)
    rand_tb = evaluate_triple_barrier(
        df_tf,
        rand_pool_idx,
        direction,
        max_bars=eff_max_bars,
        tp_atr=eff_tp_atr,
        sl_atr=eff_sl_atr,
        fee_rate=fee_rate,
    )
    rand_rets = rand_tb["net_ret"].values if not rand_tb.empty else np.zeros(50, dtype=np.float64)
    bench_drift = float(np.mean(rand_rets))
    excess_alpha = f_mean - bench_drift

    boot_means = np.array([np.mean(local_rng.choice(rand_rets, size=n_purged, replace=True)) for _ in range(500)])
    boot_pval = float(np.mean(boot_means >= f_mean))

    # 单因子扣费 Net Sharpe 与前向 Rank IC 衰减曲线审计
    if weights_tf is not None:
        net_sh_pack = compute_single_factor_net_sharpe_and_turnover(df_tf, weights_tf, timeframe, fee_rate=fee_rate)
        ic_pack = compute_forward_rank_ic_profile(df_tf, weights_tf)
    else:
        synth_w = np.zeros(len(df_tf), dtype=np.float64)
        for row_idx, r_row in tb_is.iterrows():
            _ = row_idx
            st_i = int(r_row["trigger_idx"]) + 1
            ed_i = min(st_i + max(1, int(r_row["bars_held"])), len(df_tf))
            synth_w[st_i:ed_i] = float(direction)
        net_sh_pack = compute_single_factor_net_sharpe_and_turnover(df_tf, synth_w, timeframe, fee_rate=fee_rate)
        ic_pack = compute_forward_rank_ic_profile(df_tf, synth_w)

    is_net_sharpe = float(net_sh_pack["is_net_sharpe"])
    ann_turnover = float(net_sh_pack["ann_turnover"])

    # 样本外 (2024-2026) 盲测表现跟踪
    oos_raw_idx = np.where(trig_bool & oos_mask)[0]
    oos_purged_idx = purge_overlapping_triggers(oos_raw_idx, cooldown_bars=eff_max_bars)
    tb_oos = evaluate_triple_barrier(
        df_tf,
        oos_purged_idx,
        direction,
        max_bars=eff_max_bars,
        tp_atr=eff_tp_atr,
        sl_atr=eff_sl_atr,
        fee_rate=fee_rate,
        weights_tf=weights_tf,
    )
    if len(tb_oos) >= 2:
        r_oos = tb_oos["net_ret"].values
        q_oos = tb_oos["quality_w"].values if use_q_weight else np.ones(len(r_oos), dtype=np.float64)
        oos_stats = _compute_weighted_edge_metrics(r_oos, q_oos)
        oos_mean = oos_stats["mean"]
        oos_safety = oos_stats["safety_margin"]
        oos_alpha = oos_mean - bench_drift
    else:
        oos_mean = oos_alpha = oos_safety = 0.0

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
        reasons.append(f"超额Alpha={excess_alpha * 100:+.2f}% < +{min_alpha_dec * 100:.2f}%")
    if safety_margin < min_safety_dec:
        reasons.append(f"安全垫={safety_margin * 100:+.1f}% < +{ruler.min_safety_pct:.1f}%")
    if boot_pval > max_pval:
        reasons.append(f"Bootstrap p={boot_pval:.3f} > {max_pval:.2f}")
    if excursion_ratio < min_exc_ratio:
        reasons.append(f"MFE/MAE路径偏离比={excursion_ratio:.2f} < {min_exc_ratio:.2f}")
    if weights_tf is not None and is_net_sharpe <= min_net_sh:
        reasons.append(f"扣费Net Sharpe={is_net_sharpe:.2f} <= {min_net_sh:.2f}")

    return {
        "passed": bool(passed),
        "reason": "; ".join(reasons) if reasons else f"通过 {ruler.archetype} 专属法医标尺",
        "is_n_purged": int(n_purged),
        "is_mean_pct": round(f_mean * 100, 3),
        "is_unweighted_mean_pct": round(eq_stats["mean"] * 100, 3),
        "is_q_weighted_mean_pct": round(qw_stats["mean"] * 100, 3),
        "is_excess_alpha_pct": round(excess_alpha * 100, 3),
        "is_win_rate_pct": round(f_win * 100, 2),
        "is_unweighted_win_rate_pct": round(eq_stats["win_rate"] * 100, 2),
        "is_q_weighted_win_rate_pct": round(qw_stats["win_rate"] * 100, 2),
        "is_pl_ratio": round(pl_ratio, 2),
        "is_safety_margin_pct": round(safety_margin * 100, 2),
        "is_unweighted_safety_margin_pct": round(eq_stats["safety_margin"] * 100, 2),
        "is_q_weighted_safety_margin_pct": round(qw_stats["safety_margin"] * 100, 2),
        "is_boot_pval": round(boot_pval, 4),
        "mfe_atr_mean": round(mean_mfe_atr, 3),
        "mae_atr_mean": round(mean_mae_atr, 3),
        "excursion_ratio": round(excursion_ratio, 3),
        "is_excursion_ratio": round(excursion_ratio, 3),
        "is_net_sharpe": round(is_net_sharpe, 3),
        "ann_turnover": round(ann_turnover, 2),
        "signal_turnover": round(ann_turnover, 2),
        "rank_ic_curve": ic_pack["rank_ic_curve"],
        "rank_ic_1": ic_pack["rank_ic_curve"].get(1, 0.0),
        "rank_ic_3": ic_pack["rank_ic_curve"].get(3, 0.0),
        "rank_ic_5": ic_pack["rank_ic_curve"].get(5, 0.0),
        "rank_ic_10": ic_pack["rank_ic_curve"].get(10, 0.0),
        "rank_ic_20": ic_pack["rank_ic_curve"].get(20, 0.0),
        "icir_curve": ic_pack["icir_curve"],
        "peak_rank_ic": ic_pack["peak_rank_ic"],
        "mean_rank_ic": ic_pack["mean_rank_ic"],
        "autocorr_lag1": ic_pack["autocorr_lag1"],
        "half_life_bars": ic_pack["half_life_bars"],
        "bar_turnover": ic_pack["bar_turnover"],
        "oos_n": int(len(tb_oos)),
        "oos_mean_pct": round(oos_mean * 100, 3),
        "oos_excess_alpha_pct": round(oos_alpha * 100, 3),
        "oos_safety_margin_pct": round(oos_safety * 100, 2),
    }


def compute_active_union_correlation(
    w1: np.ndarray,
    w2: np.ndarray,
    eps: float = 1e-3,
    min_active_bars: int = 30,
) -> float:
    """
    计算两个稀疏规则/连续因子在激活态并集切片 T_12 = {t | |w1_t| > eps or |w2_t| > eps} 上的条件相关系数：
    - 若 |T_12| < min_active_bars，回退到全样本皮尔逊相关；
    - 在激活态并集切片上同时计算中心化 Pearson 相关与零基线同向共振余弦相关，
      彻底解决 90%+ 空仓零值（或静默期底噪）稀释相关性导致的同质因子重复入库问题。
    """
    a1 = np.asarray(w1, dtype=np.float64)
    a2 = np.asarray(w2, dtype=np.float64)
    if len(a1) != len(a2) or len(a1) < 2:
        return 0.0

    active_mask = (np.abs(a1) > eps) | (np.abs(a2) > eps)
    n_active = int(active_mask.sum())
    if n_active < min_active_bars:
        if np.std(a1) > 1e-8 and np.std(a2) > 1e-8:
            c_full = float(np.corrcoef(a1, a2)[0, 1])
            return c_full if np.isfinite(c_full) else 0.0
        return 0.0

    u1 = a1[active_mask]
    u2 = a2[active_mask]
    prod = u1 * u2
    # 若在激活并集上两者从未同刻同时非零（完全错开），则条件相关为 0
    if not np.any(np.abs(prod) > eps * eps):
        return 0.0

    pearson_c = 0.0
    if np.std(u1) > 1e-8 and np.std(u2) > 1e-8:
        val = float(np.corrcoef(u1, u2)[0, 1])
        if np.isfinite(val):
            pearson_c = val

    denom = float(np.linalg.norm(u1) * np.linalg.norm(u2))
    cos_c = float(np.dot(u1, u2) / denom) if denom > 1e-12 else 0.0

    if cos_c >= 0.0:
        return float(np.clip(max(pearson_c, cos_c), -1.0, 1.0))
    return float(np.clip(min(pearson_c, cos_c), -1.0, 1.0))


def run_parameter_smoothness_audit(
    df_tf: pd.DataFrame | Callable[..., Any] | None = None,
    sig_id: str | None = None,
    timeframe: str = "30m",
    *,
    eval_fn: Callable[..., Any] | None = None,
    evaluate_fn: Callable[..., Any] | None = None,
    center_param: float = 1.0,
    perturb_ratios: tuple[float, ...] = (0.80, 0.90, 1.00, 1.10, 1.20),
    deltas: Sequence[float] | None = None,
    neighbor_metrics: Sequence[float] | None = None,
    center_metric: float | None = None,
    min_retention_ratio: float = 0.70,
    max_cv: float = 0.40,
    fee_rate: float = 0.0002,
    is_end_dt: pd.Timestamp = IS_END_DT,
) -> dict[str, Any]:
    """
    参数平原检验 (Parameter Smoothness Gate)：
    对候选信号的核心参数（如 DecayConfig.lam0、scale 或窗口长度）执行 ±20% 共 3~5 组邻域微扰，
    计算邻域净收益/夏普的变异系数 CV = σ / |μ| 与均值保持率 μ_neighbor / μ_center，
    要求保持率 >= 0.70 且 CV <= 0.40，剔除“孤岛参数”。
    """
    if callable(df_tf) and eval_fn is None and evaluate_fn is None:
        eval_fn = df_tf
        df_tf = None
    if evaluate_fn is not None and eval_fn is None:
        eval_fn = evaluate_fn

    use_direct_deltas = deltas is not None
    ratios = tuple(float(d) for d in deltas) if deltas is not None else tuple(float(r) for r in perturb_ratios)
    if neighbor_metrics is not None:
        vals = np.asarray(list(neighbor_metrics), dtype=np.float64)
        c_idx = len(vals) // 2
        c_val = float(center_metric) if center_metric is not None else float(vals[c_idx])
    else:
        metrics_list: list[float] = []
        if use_direct_deltas:
            c_idx = ratios.index(0.0) if 0.0 in ratios else len(ratios) // 2
        else:
            c_idx = ratios.index(1.0) if 1.0 in ratios else len(ratios) // 2

        for r in ratios:
            if eval_fn is not None:
                p_val = r if use_direct_deltas else (center_param * r)
                sig = inspect.signature(eval_fn)
                res = eval_fn(df_tf, p_val) if len(sig.parameters) >= 2 and df_tf is not None else eval_fn(p_val)
                if isinstance(res, int | float | np.floating):
                    m_val = float(res)
                elif isinstance(res, dict):
                    m_val = float(
                        res.get(
                            "is_net_sharpe",
                            res.get("sharpe", res.get("is_mean_pct", res.get("mean_ret", 0.0))),
                        )
                    )
                elif isinstance(res, tuple) and len(res) == 2 and df_tf is not None:
                    _, w_arr = res
                    sh_pack = compute_single_factor_net_sharpe_and_turnover(
                        df_tf, np.asarray(w_arr, dtype=np.float64), timeframe, fee_rate=fee_rate, is_end_dt=is_end_dt
                    )
                    m_val = float(sh_pack["is_net_sharpe"])
                elif isinstance(res, np.ndarray) and df_tf is not None:
                    sh_pack = compute_single_factor_net_sharpe_and_turnover(
                        df_tf, np.asarray(res, dtype=np.float64), timeframe, fee_rate=fee_rate, is_end_dt=is_end_dt
                    )
                    m_val = float(sh_pack["is_net_sharpe"])
                else:
                    m_val = 0.0
                metrics_list.append(m_val)
            elif df_tf is not None and sig_id is not None:
                spec = ALL_SIGNAL_SPECS.get(sig_id)
                base_scale = TF_SCALE_MAP.get(timeframe, 1.0)
                if spec is not None and spec.exit_mode == ATOMIC_EXIT_MODE:
                    from examples.multi_factor_cta.atomic_factors import DecayConfig, compute_atomic_signal

                    trig_arr, w_arr = compute_atomic_signal(
                        df_tf,
                        sig_id,
                        scale=base_scale * r,
                        decay=DecayConfig(lam0=0.10 * r),
                    )
                else:
                    trig_arr, w_arr = compute_signal_triggers_and_weights(df_tf, sig_id, scale=base_scale * r)
                ruler = ARCHETYPE_RULERS[spec.archetype] if spec is not None else ARCHETYPE_RULERS[ARCH_A]
                direction = spec.direction if spec is not None else 1
                g1_r = run_gate1_forensic_audit(
                    df_tf, trig_arr, direction, ruler, timeframe, weights_tf=w_arr, fee_rate=fee_rate
                )
                metrics_list.append(float(g1_r.get("is_mean_pct", 0.0)))
            else:
                raise ValueError("Must provide either neighbor_metrics, eval_fn, or (df_tf, sig_id).")

        vals = np.asarray(metrics_list, dtype=np.float64)
        c_val = float(center_metric) if center_metric is not None else float(vals[c_idx])

    neighbor_only = [float(v) for i, v in enumerate(vals) if i != c_idx] if len(vals) > 1 else [float(vals[0])]
    mu_neighbor = float(np.mean(neighbor_only))
    mu_all = float(np.mean(vals))
    sigma_all = float(np.std(vals, ddof=0))

    retention_ratio = float(mu_neighbor / c_val) if c_val > 1e-9 else 0.0
    cv = float(sigma_all / max(abs(mu_all), 1e-9)) if mu_all > 1e-9 else float("inf")

    passed = (c_val > 0.0) and (retention_ratio >= min_retention_ratio) and (cv <= max_cv)
    reasons = []
    if c_val <= 0.0:
        reasons.append(f"中心参数指标={c_val:.4f} <= 0")
    if retention_ratio < min_retention_ratio:
        reasons.append(f"邻域均值保持率={retention_ratio:.2f} < {min_retention_ratio:.2f} (孤岛参数塌陷)")
    if cv > max_cv:
        reasons.append(f"邻域变异系数CV={cv:.2f} > {max_cv:.2f} (参数敏感不稳定)")

    return {
        "passed": bool(passed),
        "reason": "; ".join(reasons) if reasons else "通过参数平原平滑度检验",
        "center_metric": round(c_val, 4),
        "neighbor_mean": round(mu_neighbor, 4),
        "all_mean": round(mu_all, 4),
        "retention_ratio": round(retention_ratio, 4),
        "cv": round(cv, 4) if np.isfinite(cv) else 999.0,
        "grid_metrics": [round(float(x), 4) for x in vals],
        "perturb_ratios": list(ratios),
    }


def run_gate2_3d_complementarity_audit(
    df_30m: pd.DataFrame,
    cand_w_on_30m: np.ndarray,
    zoo_w_on_30m: dict[str, np.ndarray],
    current_portfolio_w_30m: np.ndarray,
    archetype: str,
    current_archetype_new_count: int,
    ruler: ArchetypeRuler,
    current_arch_same_dir_new_count: int = 0,
    max_corr_threshold: float = 0.32,
    min_gap_fill_pct: float = 35.0,
    active_corr_eps: float = 1e-3,
    min_active_bars: int = 30,
) -> dict[str, Any]:
    """
    执行第二关：三维真互补门禁 (3D Complementarity Gate)
    1. 生态位总配额与多空均衡子配额检查：该方向新增入库单元数 < ruler.quota_max_units 且单边同向新增数 < 4
    2. 激活态条件相关正交：与已入库同向单元的激活态并集条件相关系数 |ρ_active_max| < max_corr_threshold
    3. 空仓填补 + 回撤期雪中送炭：计算空仓填补率 (gap_fill_pct) 与原组合处于空仓或净值回撤状态 (DD <= -1.5%) 时的平均单根净收益 (dd_rescue_alpha_bp)
    """
    if current_archetype_new_count >= ruler.quota_max_units:
        return {
            "passed": False,
            "reason": f"{archetype} 生态位配额已满 ({current_archetype_new_count}/{ruler.quota_max_units})，禁止继续扎堆",
            "max_corr": 0.0,
            "max_full_sample_corr": 0.0,
            "max_corr_peer": "QuotaCap",
            "gap_fill_ratio_pct": 0.0,
            "dd_rescue_alpha_bp": 0.0,
        }

    max_same_dir = max(4, (ruler.quota_max_units + 1) // 2)
    if current_arch_same_dir_new_count >= max_same_dir:
        return {
            "passed": False,
            "reason": f"{archetype} 同向单边配额已满 ({current_arch_same_dir_new_count}/{max_same_dir})，优先预留反向对冲生态位",
            "max_corr": 0.0,
            "max_full_sample_corr": 0.0,
            "max_corr_peer": "DirQuotaCap",
            "gap_fill_ratio_pct": 0.0,
            "dd_rescue_alpha_bp": 0.0,
        }

    is_mask = (df_30m["dt"] <= IS_END_DT).values
    if int(is_mask.sum()) < 25:
        is_mask = np.ones(len(df_30m), dtype=bool)
    c_sub = cand_w_on_30m[is_mask]

    if np.std(c_sub) < 1e-6:
        return {
            "passed": False,
            "reason": "样本内持仓序列标准差为 0",
            "max_corr": 1.0,
            "max_full_sample_corr": 1.0,
            "max_corr_peer": "None",
            "gap_fill_ratio_pct": 0.0,
            "dd_rescue_alpha_bp": 0.0,
        }

    max_corr = 0.0
    max_full_corr = 0.0
    max_peer = "None"
    for peer_id, peer_w in zoo_w_on_30m.items():
        p_sub = peer_w[is_mask]
        if np.std(p_sub) < 1e-6:
            continue
        act_corr = float(
            abs(compute_active_union_correlation(c_sub, p_sub, eps=active_corr_eps, min_active_bars=min_active_bars))
        )
        full_corr = float(abs(np.corrcoef(c_sub, p_sub)[0, 1]))
        if full_corr > max_full_corr:
            max_full_corr = full_corr
        if act_corr > max_corr:
            max_corr = act_corr
            max_peer = peer_id

    # 空仓填补率
    active_cand = np.abs(c_sub) > 0.01
    port_sub = current_portfolio_w_30m[is_mask]
    gap_fill_pct = float(np.mean(np.abs(port_sub[active_cand]) <= 0.25) * 100.0) if np.sum(active_cand) > 0 else 0.0

    # 无未来函数的空仓期/组合回撤期雪中送炭指标 (以时刻 t 已知的组合回撤状态或空仓状态为条件)
    bar_ret = df_30m["close"].pct_change().fillna(0.0).values[is_mask]
    port_cum_nav = np.cumprod(1.0 + port_sub * bar_ret)
    port_peak = np.maximum.accumulate(port_cum_nav)
    port_dd_state = (port_cum_nav - port_peak) / port_peak  # t 时刻已知的历史回撤深度

    fwd_bar_ret = df_30m["close"].pct_change().shift(-1).fillna(0.0).values[is_mask]
    rescue_window_mask = ((np.abs(port_sub) <= 0.20) | (port_dd_state <= -0.015)) & active_cand
    # 若主网格为 5m (约 78,672 根)，将单根 5m 救援超额折算为等效 30m 窗口 (×6) 以保持物理量纲一致
    bp_scale = 6.0 if len(df_30m) > 40000 else 1.0
    if np.sum(rescue_window_mask) >= 5:
        dd_rescue_alpha_bp = float(
            np.mean(c_sub[rescue_window_mask] * fwd_bar_ret[rescue_window_mask]) * 10000.0 * bp_scale
        )
    else:
        dd_rescue_alpha_bp = 0.0

    passed = max_corr < max_corr_threshold and gap_fill_pct >= min_gap_fill_pct and dd_rescue_alpha_bp >= 0.5
    reasons = []
    if max_corr >= max_corr_threshold:
        reasons.append(f"与 {max_peer} 激活态条件相关性 {max_corr:.2f} >= {max_corr_threshold:.2f}")
    if gap_fill_pct < min_gap_fill_pct:
        reasons.append(f"补盲率 {gap_fill_pct:.1f}% < {min_gap_fill_pct:.1f}%")
    if dd_rescue_alpha_bp < 0.5:
        reasons.append(f"空仓/回撤期雪中送炭Alpha={dd_rescue_alpha_bp:+.1f}BP < +0.5BP")

    return {
        "passed": bool(passed),
        "reason": "; ".join(reasons) if reasons else "通过三维真互补门禁",
        "max_corr": round(max_corr, 3),
        "max_full_sample_corr": round(max_full_corr, 3),
        "max_corr_peer": max_peer,
        "gap_fill_ratio_pct": round(gap_fill_pct, 1),
        "dd_rescue_alpha_bp": round(dd_rescue_alpha_bp, 2),
    }


TF_TO_5M_GROUP_SIZE: dict[str, int] = {
    "1d": 48,
    "60m": 12,
    "30m": 6,
    "15m": 3,
    "5m": 1,
}


def map_tf_weight_to_5m_grid(
    df_source: pd.DataFrame,
    w_source: np.ndarray,
    df_target: pd.DataFrame,
    timeframe: str,
) -> np.ndarray:
    """
    将任意周期 (1d / 60m / 30m / 15m / 5m) 的持仓权重无未来函数、零延迟地映射到 5m 原生主时间轴上：
    - 5m: 原生 1:1 即时生效，彻底消除旧版 30m 网格下采样导致的盘中持仓坍缩与最高 25 分钟入场延迟
    - 15m / 30m / 60m / 1d: 严格在该周期 K 线收盘时刻 (即对应第 g 根 5m 收盘 dt) 生效，并向后阶梯保持 (Step-Hold ffill)
    """
    if timeframe not in TF_TO_5M_GROUP_SIZE:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    n_target = len(df_target)
    g = TF_TO_5M_GROUP_SIZE[timeframe]
    if g == 1 and len(w_source) == n_target:
        return w_source.copy()
    if len(w_source) * g == n_target:
        out = np.zeros(n_target, dtype=np.float64)
        out[g - 1 : -1] = np.repeat(w_source[:-1], g)
        out[-1] = w_source[-1]
        return out

    s_src = pd.Series(w_source, index=df_source["dt"])
    return df_target["dt"].map(s_src).ffill().fillna(0.0).values


# 兼容旧函数名调用，自动对齐到传入的主时间轴 (5m 或 30m)
map_tf_weight_to_30m_grid = map_tf_weight_to_5m_grid


def apply_rebalance_deadzone(w: np.ndarray, deadzone: float = 0.06) -> np.ndarray:
    """
    最小调仓死区滤波器 (Rebalance Deadzone Filter)：
    - 当目标仓位归零 (|target| < 1e-4) 或多空方向反转 (cur * target < 0) 时，立即无延迟执行；
    - 当同向微调幅度 |target - cur| < deadzone 时，保持原持仓不变，消除 5m 主网格上的碎步换手磨损。
    """
    if deadzone <= 0.0:
        return w
    out = np.empty_like(w)
    cur = 0.0
    for i in range(len(w)):
        target = w[i]
        if abs(target) < 1e-4:
            cur = 0.0
        elif cur * target < 0.0 or abs(target - cur) >= deadzone:
            cur = target
        out[i] = cur
    return out


def synthesize_atomic_units_portfolio(
    df_grid: pd.DataFrame,
    unit_weights_on_grid: dict[str, np.ndarray],
    unit_alloc_map: dict[str, float],
    rebalance_deadzone: float = 0.06,
) -> np.ndarray:
    """
    三支柱分层复合引擎 (Three-Pillar Hierarchical Portfolio Synthesis Engine)：
    1. 六簇×五周期双通道分层聚合 (Cluster-Aware Dual-Channel Aggregation)：
       在各周期层 (1d, 60m, 30m, 15m, 5m) 内，将 A_趋势突破与斜率动量簇 与 B/C/D/E/F 正交战术簇
       (背驰衰竭/中枢回归/量价筹码/波动率压缩/多周期共振) 分别独立截断，彻底消除 30m/60m 顺势单元饱和截断对正交信号的淹没；
    2. 跨簇多空非对称反向断路器 (Asymmetric Cross-Cluster Reversal Brake)：
       当顺势簇持有多头而正交簇发出顶部背驰/破位空头信号时，对多头底仓施加乘法衰减 (b_long=0.35)；
       当顺势簇持有空头而正交簇发出底部超跌/V型反转多头信号时，对空头底仓施加快速平空保护 (b_short=0.50)；
    3. 因果有效湍流波动率目标调制 (Causal Turbulence-Adjusted Volatility Targeting)：
       结合滚动 15 日年化波动率与 5 日考夫曼趋势效率比 ER_5d 识别“低效率无序巨震湍流”，动态调节风险敞口上限；
    4. 施加最小调仓死区 (rebalance_deadzone=0.06)，滤除碎步换手磨损。
    """
    n = len(df_grid)
    tf_weights = {"1d": 0.65, "60m": 0.55, "30m": 0.45, "15m": 0.40, "5m": 0.30}
    w_tf_arch = {tf: {arch: np.zeros(n, dtype=np.float64) for arch in ALL_ARCHETYPES} for tf in tf_weights}

    has_ortho = False
    for unit_id, coeff in unit_alloc_map.items():
        sid, tf = unit_id.split("@")
        spec = ALL_SIGNAL_SPECS.get(sid)
        arch = spec.archetype if spec is not None else ARCH_A
        if arch != ARCH_A:
            has_ortho = True
        w_tf_arch[tf][arch] += coeff * unit_weights_on_grid[unit_id]

    if not has_ortho:
        # 仅含顺势突破基石种子时 (Step 0)，保持原始五周期分形基准不变
        w_seed = (
            0.65 * np.clip(w_tf_arch["1d"][ARCH_A], -1.0, 1.0)
            + 0.55 * np.clip(w_tf_arch["60m"][ARCH_A], -1.0, 1.0)
            + 0.45 * np.clip(w_tf_arch["30m"][ARCH_A], -1.0, 1.0)
            + 0.35 * np.clip(w_tf_arch["15m"][ARCH_A], -1.0, 1.0)
            + 0.25 * np.clip(w_tf_arch["5m"][ARCH_A], -1.0, 1.0)
        )
        return apply_rebalance_deadzone(np.clip(w_seed, -1.0, 1.0), deadzone=rebalance_deadzone)

    w_a_total = np.zeros(n, dtype=np.float64)
    w_bcd_total = np.zeros(n, dtype=np.float64)
    w_bcd_pos = np.zeros(n, dtype=np.float64)
    w_bcd_neg = np.zeros(n, dtype=np.float64)

    tactical_archs = [a for a in ALL_ARCHETYPES if a != ARCH_A]
    for tf, tw in tf_weights.items():
        wa = np.clip(w_tf_arch[tf][ARCH_A], -1.0, 1.0)
        wbcd = np.clip(sum(w_tf_arch[tf][a] for a in tactical_archs), -1.0, 1.0)
        w_a_total += tw * wa
        w_bcd_total += tw * wbcd
        w_bcd_pos += tw * np.maximum(wbcd, 0.0)
        w_bcd_neg += tw * np.minimum(wbcd, 0.0)

    # 第二支柱：跨簇多空非对称反向断路器
    w_a_gated = w_a_total.copy()
    long_mask = w_a_gated > 0.0
    short_mask = w_a_gated < 0.0
    w_a_gated[long_mask] *= np.clip(1.0 - 0.35 * np.abs(w_bcd_neg[long_mask]), 0.15, 1.0)
    w_a_gated[short_mask] *= np.clip(1.0 - 0.50 * w_bcd_pos[short_mask], 0.15, 1.0)

    w_raw = w_a_gated + 1.25 * w_bcd_total

    # 第三支柱：因果有效湍流波动率目标与敞口封顶调制 (缓存于 df_grid.attrs 避免重复计算)
    vol_cap = df_grid.attrs.get("_cached_vol_cap")
    if vol_cap is None or len(vol_cap) != n:
        close_arr = df_grid["close"].values
        ret_5m = pd.Series(close_arr).pct_change().fillna(0.0).values
        vol_15d = pd.Series(ret_5m).rolling(720, min_periods=48).std().bfill().values * np.sqrt(48.0 * 242.0)
        net_chg_5d = np.abs(pd.Series(close_arr).diff(240).fillna(0.0).values)
        abs_path_5d = (
            pd.Series(np.abs(np.diff(close_arr, prepend=close_arr[0])))
            .rolling(240, min_periods=48)
            .sum()
            .bfill()
            .values
        )
        er_5d = np.clip(net_chg_5d / np.maximum(abs_path_5d, 1e-6), 0.0, 1.0)
        turb_vol = vol_15d * (1.0 - 0.35 * er_5d)
        vol_cap = np.clip(0.28 / np.maximum(turb_vol, 0.28), 0.72, 1.0)
        df_grid.attrs["_cached_vol_cap"] = vol_cap

    w_multi = np.clip(w_raw * np.sqrt(vol_cap), -vol_cap, vol_cap)
    return apply_rebalance_deadzone(w_multi, deadzone=rebalance_deadzone)


def count_round_trips(w: np.ndarray) -> tuple[int, int, int]:
    total = long_c = short_c = 0
    in_p = False
    cur_d = 0
    for val in w:
        d = 1 if val > 0.01 else (-1 if val < -0.01 else 0)
        if not in_p and d != 0:
            in_p = True
            cur_d = d
            total += 1
            if d == 1:
                long_c += 1
            else:
                short_c += 1
        elif in_p and d != cur_d:
            if d != 0:
                cur_d = d
                total += 1
                if d == 1:
                    long_c += 1
                else:
                    short_c += 1
            else:
                in_p = False
    return total, long_c, short_c


def backtest_slice(df_30m_slice: pd.DataFrame, w_slice: np.ndarray) -> dict[str, Any]:
    dfw = pd.DataFrame(
        {
            "dt": df_30m_slice["dt"].astype(str),
            "symbol": "159915",
            "weight": w_slice,
            "price": df_30m_slice["close"].values,
        }
    )
    wb = WeightBacktest(data=dfw, fee_rate=0.0002, weight_type="ts")
    st = wb.stats
    tot_t, l_t, s_t = count_round_trips(w_slice)
    active_pct = float((np.abs(w_slice) > 0.01).mean() * 100.0)

    cum_ret = float(st.get("绝对收益", 0.0) * 100.0)
    ann_ret = float(st.get("年化收益", 0.0) * 100.0)
    sharpe = float(st.get("夏普比率", 0.0))
    mdd = float(st.get("最大回撤", 0.0) * 100.0)
    calmar = float(st.get("卡玛比率", 0.0))
    score = sharpe + 0.35 * calmar + 0.05 * ann_ret

    dr = wb.daily_return.copy()
    dr["date"] = pd.to_datetime(dr["date"])
    ret_s = dr.set_index("date")["total"]
    nav_s = (1 + ret_s).cumprod()
    dd_s = (nav_s - nav_s.cummax()) / nav_s.cummax() * 100

    return {
        "trades": tot_t,
        "long_trades": l_t,
        "short_trades": s_t,
        "active_pct": round(active_pct, 1),
        "cum_ret_pct": round(cum_ret, 2),
        "ann_ret_pct": round(ann_ret, 2),
        "sharpe": round(sharpe, 2),
        "mdd_pct": round(mdd, 2),
        "calmar": round(calmar, 2),
        "composite_score": round(score, 4),
        "nav_series": nav_s,
        "dd_series": dd_s,
    }
