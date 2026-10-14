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

from typing import Any

import numpy as np
import pandas as pd

from czsc import WeightBacktest
from examples.alpha_mining_agent.hypothesis_bank import ArchetypeRuler

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
) -> pd.DataFrame:
    n = len(df)
    o_arr = df["open"].values
    h_arr = df["high"].values
    l_arr = df["low"].values
    c_arr = df["close"].values
    atr_arr = df["atr14"].values

    records = []
    for idx in entry_indices:
        if idx + 1 >= n:
            continue
        entry_p = o_arr[idx + 1]
        atr = atr_arr[idx]
        if np.isnan(atr) or atr <= 0 or entry_p <= 0:
            continue

        tp_dist = tp_atr * atr
        sl_dist = sl_atr * atr
        exit_p = entry_p
        end_bar = min(idx + 1 + max_bars, n - 1)

        for j in range(idx + 1, end_bar + 1):
            hi, lo, cl = h_arr[j], l_arr[j], c_arr[j]
            if direction == 1:
                if lo <= entry_p - sl_dist:
                    exit_p = entry_p - sl_dist
                    break
                if hi >= entry_p + tp_dist:
                    exit_p = entry_p + tp_dist
                    break
            else:
                if hi >= entry_p + sl_dist:
                    exit_p = entry_p + sl_dist
                    break
                if lo <= entry_p - tp_dist:
                    exit_p = entry_p - tp_dist
                    break
            if j == end_bar:
                exit_p = cl

        raw_ret = (exit_p - entry_p) / entry_p if direction == 1 else (entry_p - exit_p) / entry_p
        net_ret = raw_ret - fee_rate
        records.append({"trigger_idx": idx, "net_ret": net_ret})

    return pd.DataFrame(records)


def run_gate1_forensic_audit(
    df_tf: pd.DataFrame,
    triggers_tf: np.ndarray,
    direction: int,
    ruler: ArchetypeRuler,
    timeframe: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    使用该战略方向专属的法医三重屏障标尺 (ruler) 在特定周期 df_tf 上执行样本内 (2020-2023) 审计。
    """
    is_mask = (df_tf["dt"] <= IS_END_DT).values
    oos_mask = (df_tf["dt"] > IS_END_DT).values

    min_samples = 8 if timeframe == "1d" else (12 if timeframe == "60m" else (20 if timeframe == "5m" else 15))
    eff_max_bars = ruler.max_bars * 2 if timeframe == "5m" else ruler.max_bars
    eff_tp_atr = ruler.tp_atr * 1.35 if timeframe == "5m" else ruler.tp_atr
    eff_sl_atr = ruler.sl_atr * 1.20 if timeframe == "5m" else ruler.sl_atr
    min_warmup = 240 if timeframe == "5m" else 60

    is_raw_idx = np.where(triggers_tf & is_mask)[0]
    is_purged_idx = purge_overlapping_triggers(is_raw_idx, cooldown_bars=eff_max_bars)

    tb_is = evaluate_triple_barrier(
        df_tf,
        is_purged_idx,
        direction,
        max_bars=eff_max_bars,
        tp_atr=eff_tp_atr,
        sl_atr=eff_sl_atr,
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

    # 500 轮随机入场置换检验 (同样使用该专属标尺)
    valid_pool = np.where(
        is_mask & (np.arange(len(df_tf)) >= min_warmup) & (np.arange(len(df_tf)) < len(df_tf) - eff_max_bars - 2)
    )[0]
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
    )
    rand_rets = rand_tb["net_ret"].values
    bench_drift = float(np.mean(rand_rets))
    excess_alpha = f_mean - bench_drift

    boot_means = np.array([np.mean(local_rng.choice(rand_rets, size=n_purged, replace=True)) for _ in range(500)])
    boot_pval = float(np.mean(boot_means >= f_mean))

    # 样本外 (2024-2026) 盲测表现跟踪
    oos_raw_idx = np.where(triggers_tf & oos_mask)[0]
    oos_purged_idx = purge_overlapping_triggers(oos_raw_idx, cooldown_bars=eff_max_bars)
    tb_oos = evaluate_triple_barrier(
        df_tf,
        oos_purged_idx,
        direction,
        max_bars=eff_max_bars,
        tp_atr=eff_tp_atr,
        sl_atr=eff_sl_atr,
    )
    if len(tb_oos) >= 2:
        r_oos = tb_oos["net_ret"].values
        oos_mean = float(np.mean(r_oos))
        oos_win = float(np.mean(r_oos > 0))
        w_o = r_oos[r_oos > 0]
        l_o = r_oos[r_oos <= 0]
        pl_o = (float(np.mean(w_o)) if len(w_o) > 0 else 0.0) / max(
            float(abs(np.mean(l_o))) if len(l_o) > 0 else 1e-6, 1e-6
        )
        oos_safety = oos_win - 1.0 / (1.0 + pl_o)
        oos_alpha = oos_mean - bench_drift
    else:
        oos_mean = oos_alpha = oos_safety = 0.0

    tf_alpha_scale = {"1d": 1.2, "60m": 1.0, "30m": 0.85, "15m": 0.55, "5m": 0.40}.get(timeframe, 1.0)
    min_alpha_dec = (ruler.min_alpha_bp * tf_alpha_scale) / 10000.0
    min_safety_dec = ruler.min_safety_pct / 100.0
    max_pval = ruler.max_boot_pval + (0.02 if timeframe in ("60m", "1d") else 0.0)

    passed = excess_alpha >= min_alpha_dec and safety_margin >= min_safety_dec and boot_pval <= max_pval
    reasons = []
    if excess_alpha < min_alpha_dec:
        reasons.append(f"超额Alpha={excess_alpha * 100:+.2f}% < +{min_alpha_dec * 100:.2f}%")
    if safety_margin < min_safety_dec:
        reasons.append(f"安全垫={safety_margin * 100:+.1f}% < +{ruler.min_safety_pct:.1f}%")
    if boot_pval > max_pval:
        reasons.append(f"Bootstrap p={boot_pval:.3f} > {max_pval:.2f}")

    return {
        "passed": bool(passed),
        "reason": "; ".join(reasons) if reasons else f"通过 {ruler.archetype} 专属法医标尺",
        "is_n_purged": int(n_purged),
        "is_mean_pct": round(f_mean * 100, 3),
        "is_excess_alpha_pct": round(excess_alpha * 100, 3),
        "is_win_rate_pct": round(f_win * 100, 2),
        "is_pl_ratio": round(pl_ratio, 2),
        "is_safety_margin_pct": round(safety_margin * 100, 2),
        "is_boot_pval": round(boot_pval, 4),
        "oos_n": int(len(tb_oos)),
        "oos_mean_pct": round(oos_mean * 100, 3),
        "oos_excess_alpha_pct": round(oos_alpha * 100, 3),
        "oos_safety_margin_pct": round(oos_safety * 100, 2),
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
) -> dict[str, Any]:
    """
    执行第二关：三维真互补门禁 (3D Complementarity Gate)
    1. 生态位总配额与多空均衡子配额检查：该方向新增入库单元数 < ruler.quota_max_units 且单边同向新增数 < 4
    2. 持仓时序正交：与已入库同向单元的最大皮尔逊相关系数 |ρ_max| < max_corr_threshold
    3. 空仓填补 + 回撤期雪中送炭：计算空仓填补率 (gap_fill_pct) 与原组合处于空仓或净值回撤状态 (DD <= -1.5%) 时的平均单根净收益 (dd_rescue_alpha_bp)
    """
    if current_archetype_new_count >= ruler.quota_max_units:
        return {
            "passed": False,
            "reason": f"{archetype} 生态位配额已满 ({current_archetype_new_count}/{ruler.quota_max_units})，禁止继续扎堆",
            "max_corr": 0.0,
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
            "max_corr_peer": "DirQuotaCap",
            "gap_fill_ratio_pct": 0.0,
            "dd_rescue_alpha_bp": 0.0,
        }

    is_mask = (df_30m["dt"] <= IS_END_DT).values
    c_sub = cand_w_on_30m[is_mask]

    if np.std(c_sub) < 1e-6:
        return {
            "passed": False,
            "reason": "样本内持仓序列标准差为 0",
            "max_corr": 1.0,
            "max_corr_peer": "None",
            "gap_fill_ratio_pct": 0.0,
            "dd_rescue_alpha_bp": 0.0,
        }

    max_corr = 0.0
    max_peer = "None"
    for peer_id, peer_w in zoo_w_on_30m.items():
        p_sub = peer_w[is_mask]
        if np.std(p_sub) < 1e-6:
            continue
        corr = float(abs(np.corrcoef(c_sub, p_sub)[0, 1]))
        if corr > max_corr:
            max_corr = corr
            max_peer = peer_id

    # 空仓填补率
    active_cand = np.abs(c_sub) > 0.01
    port_sub = current_portfolio_w_30m[is_mask]
    gap_fill_pct = float(np.mean(np.abs(port_sub[active_cand]) <= 0.25) * 100.0) if np.sum(active_cand) > 0 else 0.0

    # 无未来函数的空仓期/組合回撤期雪中送炭指标 (以时刻 t 已知的组合回撤状态或空仓状态为条件)
    bar_ret = df_30m["close"].pct_change().fillna(0.0).values[is_mask]
    port_cum_nav = np.cumprod(1.0 + port_sub * bar_ret)
    port_peak = np.maximum.accumulate(port_cum_nav)
    port_dd_state = (port_cum_nav - port_peak) / port_peak  # t 时刻已知的历史回撤深度

    fwd_bar_ret = df_30m["close"].pct_change().shift(-1).fillna(0.0).values[is_mask]
    rescue_window_mask = ((np.abs(port_sub) <= 0.20) | (port_dd_state <= -0.015)) & active_cand
    if np.sum(rescue_window_mask) >= 5:
        dd_rescue_alpha_bp = float(np.mean(c_sub[rescue_window_mask] * fwd_bar_ret[rescue_window_mask]) * 10000.0)
    else:
        dd_rescue_alpha_bp = 0.0

    passed = max_corr < max_corr_threshold and gap_fill_pct >= min_gap_fill_pct and dd_rescue_alpha_bp >= 0.5
    reasons = []
    if max_corr >= max_corr_threshold:
        reasons.append(f"与 {max_peer} 相关性 {max_corr:.2f} >= {max_corr_threshold:.2f}")
    if gap_fill_pct < min_gap_fill_pct:
        reasons.append(f"补盲率 {gap_fill_pct:.1f}% < {min_gap_fill_pct:.1f}%")
    if dd_rescue_alpha_bp < 0.5:
        reasons.append(f"空仓/回撤期雪中送炭Alpha={dd_rescue_alpha_bp:+.1f}BP < +0.5BP")

    return {
        "passed": bool(passed),
        "reason": "; ".join(reasons) if reasons else "通过三维真互补门禁",
        "max_corr": round(max_corr, 3),
        "max_corr_peer": max_peer,
        "gap_fill_ratio_pct": round(gap_fill_pct, 1),
        "dd_rescue_alpha_bp": round(dd_rescue_alpha_bp, 2),
    }


def map_tf_weight_to_30m_grid(
    df_source: pd.DataFrame,
    w_source: np.ndarray,
    df_30m: pd.DataFrame,
    timeframe: str,
) -> np.ndarray:
    """
    将任意周期 (1d / 60m / 30m / 15m / 5m) 的持仓权重无未来函数映射到 30m 主时间轴上：
    - 30m: 直接返回
    - 1d / 60m: 在收盘时刻 dt 对齐并向后 ffill
    - 15m: 每个 30m K线由 2 根 15m K线组成，按索引分组取均值映射到对应 30m K线
    - 5m: 每个 30m K线由 6 根 5m K线组成，取该 30m K线收盘时刻 (第6根5m) 的已知持仓状态以消除高频碎步换手磨损
    """
    if timeframe == "30m":
        return w_source.copy()
    elif timeframe in ("1d", "60m"):
        s_src = pd.Series(w_source, index=df_source["dt"])
        return df_30m["dt"].map(s_src).ffill().fillna(0.0).values
    elif timeframe == "15m":
        # 每 2 根 15m K 线精确对应 1 根 30m K 线
        n30 = len(df_30m)
        if len(w_source) == 2 * n30:
            return 0.5 * (w_source[0::2] + w_source[1::2])
        s_src = pd.Series(w_source, index=df_source["dt"])
        return df_30m["dt"].map(s_src).ffill().fillna(0.0).values
    elif timeframe == "5m":
        n30 = len(df_30m)
        if len(w_source) == 6 * n30:
            return w_source[5::6].copy()
        s_src = pd.Series(w_source, index=df_source["dt"])
        return df_30m["dt"].map(s_src).ffill().fillna(0.0).values
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def synthesize_atomic_units_portfolio(
    df_30m: pd.DataFrame,
    unit_weights_on_30m: dict[str, np.ndarray],
    unit_alloc_map: dict[str, float],
) -> np.ndarray:
    """
    按原子单元 (sig_id@tf) 独立合成五周期复合持仓权重：
    先在各周期层内部 (1d, 60m, 30m, 15m, 5m) 聚合截断，再按五周期分形共振系数叠加。
    """
    n = len(df_30m)
    w_by_tf = {
        "1d": np.zeros(n, dtype=np.float64),
        "60m": np.zeros(n, dtype=np.float64),
        "30m": np.zeros(n, dtype=np.float64),
        "15m": np.zeros(n, dtype=np.float64),
        "5m": np.zeros(n, dtype=np.float64),
    }

    for unit_id, coeff in unit_alloc_map.items():
        tf = unit_id.split("@")[1]
        w_by_tf[tf] += coeff * unit_weights_on_30m[unit_id]

    w1d = np.clip(w_by_tf["1d"], -1.0, 1.0)
    w60 = np.clip(w_by_tf["60m"], -1.0, 1.0)
    w30 = np.clip(w_by_tf["30m"], -1.0, 1.0)
    w15 = np.clip(w_by_tf["15m"], -1.0, 1.0)
    w5 = np.clip(w_by_tf["5m"], -1.0, 1.0)

    w_multi = np.clip(0.65 * w1d + 0.55 * w60 + 0.45 * w30 + 0.35 * w15 + 0.25 * w5, -1.0, 1.0)
    return w_multi


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
