#!/usr/bin/env python
"""
分方向定制标尺 + 三维真互补门禁流水线
(Archetype-Specific Forensic Rulers & 3D Complementarity Pipeline)

核心升级：
1. 固定 ATR 事件诊断 [Gate 1 · Archetype-Specific Ruler，不能代替实际执行准入]：
   - 趋势突破簇用趋势尺 (+2.2*ATR / -1.1*ATR / 16根)
   - 背驰反转簇用反转尺 (+1.35*ATR / -0.95*ATR / 8根)
   - 震荡回归簇用回归尺 (+1.15*ATR / -0.85*ATR / 6根)
   - 日内微结构簇用日内尺 (+1.25*ATR / -0.85*ATR / 6根)
   - 在候选信号的专属周期 (1d / 60m / 30m / 15m) 上独立执行 Purged + Next-Open + 2BP + Bootstrap 审计
2. 正式执行准入 [Execution]：Rust 实际结构退出交易、完整交易统计与连续扣费账本。
3. 互补性门禁 [Gate 2 · Three-Dimensional Complementarity]：
   - 维度 A (生态位配额)：检查该战略方向是否已达配额上限 (防止扎堆单一方向)
   - 维度 B (时序低相关)：与已入库同向信号在主时间轴上的最大相关系数 |ρ_max| < 0.32
   - 维度 C (空仓填补 + 回撤期雪中送炭)：空仓期填补率 >= 35% 且 在原组合亏损/回撤时刻具备正超额收益 (dd_rescue_alpha_bp > 0)
4. 第三关 [Gate 3 · 专属周期独立路由与组合边际增益]：
   - 仅在该信号通过检验的专属周期 tf 上挂载权重（不再强制三周期一刀切捆绑），验证样本内 (2020-2023) 边际增益。
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from czsc import (
    closed_execution_intervals,
    daily_performance,
    decision_execution_ledger,
    decision_interval_ledger,
    execution_episodes,
    execution_ledger,
)
from examples.alpha_mining_agent.hypothesis_bank import (
    ALL_ARCHETYPES,
    ALL_SIGNAL_SPECS,
    ARCH_A,
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
    *,
    label_end_dt: pd.Timestamp | None = None,
    purge_overlaps: bool = False,
) -> pd.DataFrame:
    """
    评估分战略簇三重屏障标尺 (Next-Bar Open 入场 + 止盈/止损/时间屏障 + 实际成交数量双边计费)，
    同步提取：
    1. 触发时刻连续质量权重 `quality_w = |w_{t_k}| ∈ (0, 1]`（供连续强度因子质量加权审计）；
    2. 持仓路径内的最大有利偏离 `mfe_atr` 与最大不利偏离 `mae_atr`（以入场时刻因果 ATR 无量纲化）。
    """
    n = len(df)
    if label_end_dt is not None:
        n = int((pd.to_datetime(df["dt"]) <= label_end_dt).sum())
    o_arr = df["open"].values
    h_arr = df["high"].values
    l_arr = df["low"].values
    c_arr = df["close"].values
    atr_arr = df["atr14"].values

    records = []
    last_exit = -1
    for idx in np.sort(np.unique(entry_indices)):
        idx_int = int(idx)
        if idx_int < 0 or idx_int + 1 >= n or (purge_overlaps and idx_int < last_exit):
            continue
        entry_p = float(o_arr[idx_int + 1])
        atr = float(atr_arr[idx_int])
        if np.isnan(atr) or atr <= 0 or entry_p <= 0:
            continue

        q_w = 1.0
        if weights_tf is not None and 0 <= idx_int < len(weights_tf):
            raw_q = abs(float(weights_tf[idx_int]))
            if not np.isfinite(raw_q) or raw_q <= 0.0:
                continue
            q_w = float(np.clip(raw_q, 0.0, 1.0))

        tp_dist = tp_atr * atr
        sl_dist = sl_atr * atr
        exit_p = entry_p
        timeout_bar = idx_int + max(1, max_bars)
        end_bar = min(timeout_bar, n - 1)
        completed = False
        exit_bar = end_bar
        mfe_atr = 0.0
        mae_atr = 0.0

        for j in range(idx_int + 1, end_bar + 1):
            bar_open = float(o_arr[j])
            stop_price = entry_p - direction * sl_dist
            if direction * (bar_open - stop_price) <= 0:
                # A gap through the stop is executable only at the opening price, not at
                # a skipped stop level. No later excursion of this bar was held.
                exit_p = bar_open
                exit_bar = j
                completed = True
                opening_move = direction * (bar_open - entry_p) / atr
                mfe_atr = max(mfe_atr, opening_move, 0.0)
                mae_atr = max(mae_atr, -opening_move, 0.0)
                break
            hi, lo, cl = float(h_arr[j]), float(l_arr[j]), float(c_arr[j])
            opening_move = direction * (bar_open - entry_p) / atr
            mfe_atr = max(mfe_atr, opening_move, 0.0)
            mae_atr = max(mae_atr, -opening_move, 0.0)
            favorable = hi if direction == 1 else lo
            adverse = lo if direction == 1 else hi
            stop_hit = direction * (adverse - entry_p) <= -sl_dist
            profit_hit = direction * (favorable - entry_p) >= tp_dist
            if opening_move >= tp_atr:
                # An opening gap is observed before either intrabar extreme.
                exit_p, exit_bar, completed = bar_open, j, True
                break
            if stop_hit:
                # OHLC does not reveal ordering. Stop wins if both barriers hit;
                # only the open and the executable stop were certainly held.
                exit_p, exit_bar, completed = entry_p - direction * sl_dist, j, True
                mae_atr = max(mae_atr, sl_atr)
                break
            if profit_hit:
                exit_p, exit_bar, completed = entry_p + direction * tp_dist, j, True
                mfe_atr = max(mfe_atr, tp_atr)
                break
            mfe_atr = max(mfe_atr, direction * (favorable - entry_p) / atr, 0.0)
            mae_atr = max(mae_atr, -direction * (adverse - entry_p) / atr, 0.0)
            if j == end_bar:
                exit_p = cl
                exit_bar = j
                completed = j == timeout_bar

        if not completed:
            continue  # The outcome is not observable inside this training fold.
        last_exit = exit_bar
        # A barrier event holds its entry quantity until exit, without intermediate
        # rebalancing. Quality remains an independent statistical observation weight.
        ledger = execution_ledger([entry_p, exit_p], [float(direction), 0.0], fee_rate)
        net_ret = sum(ledger.net_returns)
        records.append(
            {
                "trigger_idx": idx_int,
                "entry_idx": idx_int + 1,
                "exit_idx": int(exit_bar),
                "trigger_dt": pd.Timestamp(df["dt"].iloc[idx_int]).isoformat(),
                "entry_dt": pd.Timestamp(df["dt"].iloc[idx_int + 1]).isoformat(),
                "exit_dt": pd.Timestamp(df["dt"].iloc[exit_bar]).isoformat(),
                "atr_pct": atr / entry_p,
                "net_ret": float(net_ret),
                "quality_w": float(q_w),
                "mfe_atr": float(mfe_atr),
                "mae_atr": float(mae_atr),
                "bars_held": int(exit_bar - idx_int),
            }
        )

    return pd.DataFrame(records)


_EXECUTION_EVENT_COLUMNS = (
    "trigger_idx",
    "entry_idx",
    "exit_idx",
    "direction",
    "net_ret",
    "weighted_net_ret",
    "quality_w",
    "atr_pct",
    "mfe_atr",
    "mae_atr",
)


def evaluate_execution_events(
    df: pd.DataFrame,
    weights_tf: np.ndarray,
    fee_rate: float = 0.0002,
    *,
    label_end_dt: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Expose completed Rust execution episodes with their observed timestamps.

    The input weights already include the selected structural exit policy. Same
    direction resizing remains one episode; unfinished positions and decisions
    without an observable next open never become independent completed samples.
    """
    timestamps = pd.DatetimeIndex(pd.to_datetime(df["dt"]))
    if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError("Execution timestamps must be unique and ordered")
    weights = np.asarray(weights_tf, dtype=np.float64)
    if len(weights) != len(df):
        raise ValueError("Execution weights and frame must have equal lengths")
    n = len(df) if label_end_dt is None else int((timestamps <= label_end_dt).sum())
    episodes = execution_episodes(
        *(df[column].iloc[:n].to_numpy(dtype=np.float64) for column in ("open", "high", "low", "close", "atr14")),
        weights[:n],
        fee_rate,
    )
    records = []
    for episode in episodes:
        record = {key: getattr(episode, key) for key in _EXECUTION_EVENT_COLUMNS}
        for key in ("trigger", "entry", "exit"):
            record[f"{key}_dt"] = timestamps[record[f"{key}_idx"]].isoformat()
        record["bars_held"] = record["exit_idx"] - record["entry_idx"]
        records.append(record)
    return pd.DataFrame(records, columns=[*_EXECUTION_EVENT_COLUMNS, "trigger_dt", "entry_dt", "exit_dt", "bars_held"])


def compute_execution_net_metrics(
    df: pd.DataFrame,
    weights_tf: np.ndarray,
    *,
    fee_rate: float = 0.0002,
    is_end_dt: pd.Timestamp = IS_END_DT,
) -> dict[str, float]:
    """Aggregate the shared Rust next-open ledger; never invent a fold-end trade."""
    timestamps = pd.DatetimeIndex(pd.to_datetime(df["dt"]))
    if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError("Execution timestamps must be unique and ordered")
    mask = timestamps <= is_end_dt
    weights = np.asarray(weights_tf, dtype=np.float64)
    if len(weights) != len(df):
        raise ValueError("Execution weights and frame must have equal lengths")
    if not mask.any():
        return {"is_net_sharpe": 0.0, "is_net_cum_ret_pct": 0.0, "is_net_mdd_pct": 0.0, "ann_turnover": 0.0}
    ledger = decision_execution_ledger(df.loc[mask, "open"].to_numpy(dtype=np.float64), weights[mask], fee_rate)
    daily = (
        pd.Series(ledger.net_returns, index=pd.DatetimeIndex(pd.to_datetime(df.loc[mask, "dt"]))).resample("D").sum()
    )
    # Keep actual observation dates, not invented overnight/weekend sessions.
    daily = daily.reindex(pd.DatetimeIndex(pd.to_datetime(df.loc[mask, "dt"])).normalize().unique())
    metrics = daily_performance(daily.to_numpy(dtype=np.float64), yearly_days=252)
    turnover = sum(ledger.turnover)
    return {
        "is_net_sharpe": float(metrics["夏普"]),
        "is_net_cum_ret_pct": float(metrics["绝对收益"]) * 100.0,
        "is_net_mdd_pct": float(metrics["最大回撤"]) * 100.0,
        "ann_turnover": float(turnover * 252.0 / len(daily)),
    }


def run_execution_audit(
    df_tf: pd.DataFrame,
    weights_tf: np.ndarray,
    *,
    fee_rate: float = 0.0002,
    is_end_dt: pd.Timestamp = IS_END_DT,
    min_net_sharpe: float = 0.0,
) -> dict[str, Any]:
    """Apply formal SOP admission to completed structural-exit execution episodes.

    Fixed ATR ruler results are separate diagnostics. Statistical evidence is
    computed from actual resized, fee-paid episodes and their entry quality; the
    continuous net Sharpe comes from the same Rust execution clock. The zero
    exposure benchmark is explicit and is not presented as randomized-entry alpha.
    """
    events = evaluate_execution_events(df_tf, weights_tf, fee_rate, label_end_dt=is_end_dt)
    count = len(events)
    result = {
        "passed": False,
        "reason": "",
        "audit_kind": "structural_execution",
        "benchmark_kind": "zero_exposure",
        "is_n_purged": count,
        "is_effective_n": 0.0,
        "is_mean_ci_low_pct": None,
        "events": [],
        **compute_execution_net_metrics(df_tf, weights_tf, fee_rate=fee_rate, is_end_dt=is_end_dt),
    }
    if not count:
        result["reason"] = "样本内完整执行交易仅 0 次 (< 30)"
        return result
    returns = events["net_ret"].to_numpy(dtype=float)
    quality = events["quality_w"].to_numpy(dtype=float)
    effective_n = float(quality.sum() ** 2 / (quality @ quality))
    edge = _compute_weighted_edge_metrics(returns, quality)
    p_value, ci_low = (1.0, float("-inf"))
    if count >= 30 and effective_n >= 30 - 1e-9:
        p_value, ci_low = _weighted_bootstrap_evidence(
            returns,
            quality,
            returns - float(np.average(returns, weights=quality)),
            np.random.default_rng(4242),
            paired_zero_null=True,
        )
    result.update(
        {
            "is_effective_n": effective_n,
            "is_mean_ci_low_pct": ci_low * 100.0 if np.isfinite(ci_low) else None,
            "is_mean_pct": edge["mean"] * 100.0,
            "is_excess_alpha_pct": edge["mean"] * 100.0,
            "is_win_rate_pct": edge["win_rate"] * 100.0,
            "is_pl_ratio": edge["pl_ratio"],
            "is_safety_margin_pct": edge["safety_margin"] * 100.0,
            "is_boot_pval": p_value,
            "events": events.assign(r_atr=events["net_ret"] / events["atr_pct"]).to_dict("records"),
        }
    )
    reasons = []
    if count < 30:
        reasons.append(f"样本内完整执行交易仅 {count} 次 (< 30)")
    if effective_n < 30 - 1e-9:
        reasons.append(f"Q加权有效执行样本={effective_n:.2f} < 30")
    if edge["win_rate"] <= 0.5:
        reasons.append("实际交易胜率未超过50%")
    if edge["pl_ratio"] <= 1.5:
        reasons.append("实际交易盈亏比未超过1.5")
    if ci_low <= 0.0:
        reasons.append("实际交易净收益95%置信下界不大于0")
    if edge["safety_margin"] <= 0.10:
        reasons.append("实际交易安全边际未超过10%")
    if result["is_net_sharpe"] <= min_net_sharpe:
        reasons.append(f"实际执行扣费Sharpe未超过{min_net_sharpe:.2f}")
    result["passed"] = not reasons
    result["reason"] = "; ".join(reasons) if reasons else "实际结构退出交易通过 SOP 门槛"
    # OOS episodes remain descriptive, and are never used in the acceptance decision.
    all_events = evaluate_execution_events(df_tf, weights_tf, fee_rate)
    oos = all_events.loc[pd.to_datetime(all_events["trigger_dt"]) > is_end_dt]
    oos_edge = _compute_weighted_edge_metrics(
        oos["net_ret"].to_numpy(dtype=float),
        oos["quality_w"].to_numpy(dtype=float),
    )
    result.update(
        {
            "oos_n": len(oos),
            "oos_mean_pct": oos_edge["mean"] * 100.0,
            "oos_excess_alpha_pct": oos_edge["mean"] * 100.0,
            "oos_safety_margin_pct": oos_edge["safety_margin"] * 100.0,
        }
    )
    return result


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


def _prepare_next_open_slice(df_slice: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Precompute a completed fold's execution intervals without reading outside the fold."""
    n = len(df_slice)
    returns = np.zeros(n, dtype=np.float64)
    if not n:
        return returns, np.zeros(0, dtype=np.int64)
    timestamps = pd.DatetimeIndex(pd.to_datetime(df_slice["dt"]))
    if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError("Execution timestamps must be unique and ordered")
    opens = df_slice["open"].to_numpy(dtype=np.float64)
    terminal_close = float(df_slice["close"].iloc[-1])
    if not np.isfinite(opens).all() or (opens <= 0).any() or not np.isfinite(terminal_close) or terminal_close <= 0:
        raise ValueError("Execution prices must be finite and positive")
    returns[1:-1] = opens[2:] / opens[1:-1] - 1.0
    if n > 1:
        returns[-1] = terminal_close / opens[-1] - 1.0
    _, day_ids = np.unique(timestamps.date, return_inverse=True)
    return returns, day_ids


def _fast_next_open_slice_stats(
    interval_returns: np.ndarray,
    day_ids: np.ndarray,
    decision_weights: np.ndarray,
    fee_rate: float,
) -> dict[str, float]:
    """Score next-open fills plus final-close liquidation using the reporting engine's statistics.

    Price intervals are prepared once per fold. Rust shifts each candidate's decisions,
    charges actual rebalancing quantities and calls the native daily metrics routine.
    This is the same execution clock as ``backtest_slice`` without rebuilding a backtest.
    """
    from czsc import daily_performance

    keys = ("cum_ret_pct", "ann_ret_pct", "sharpe", "mdd_pct", "calmar")
    weights = np.asarray(decision_weights, dtype=np.float64)
    if len(weights) != len(interval_returns) or len(weights) != len(day_ids):
        raise ValueError("Execution intervals, dates and weights must have equal lengths")
    if not np.isfinite(fee_rate) or fee_rate < 0 or not np.isfinite(weights).all():
        raise ValueError("Execution weights and nonnegative fee must be finite")
    if not len(weights):
        return dict.fromkeys(keys, 0.0)
    ledger = decision_interval_ledger(interval_returns, weights, fee_rate)
    # The explicit final close belongs to the last observed day. Interval PnL
    # belongs to its ending mark, matching the complete execution report.
    daily = np.bincount(np.r_[day_ids, day_ids[-1]], weights=ledger.net_returns)
    metrics = daily_performance(daily, yearly_days=252)
    return {
        "cum_ret_pct": float(metrics["绝对收益"] * 100),
        "ann_ret_pct": float(metrics["年化"] * 100),
        "sharpe": float(metrics["夏普"]),
        "mdd_pct": float(metrics["最大回撤"] * 100),
        "calmar": float(metrics["卡玛"]),
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

    w_is = np.asarray(weights_tf[is_mask], dtype=np.float64)
    c_is = np.asarray(df_tf["close"].values[is_mask], dtype=np.float64)
    n = len(c_is)
    if n < 2 or np.max(np.abs(w_is)) <= 1e-6:
        return {"is_net_sharpe": 0.0, "ann_turnover": 0.0, "is_net_cum_ret_pct": 0.0}

    ledger = decision_execution_ledger(
        df_tf.loc[is_mask, "open"].to_numpy(dtype=np.float64), w_is, fee_rate, terminal_close=float(c_is[-1])
    )
    dw = np.asarray(ledger.turnover)
    bar_net = np.asarray(ledger.net_returns)

    dts = pd.to_datetime(df_tf["dt"].iloc[is_mask]).dt.date.values
    uniq_dates, day_ids = np.unique(np.r_[dts, dts[-1]], return_inverse=True)
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
    q = np.clip(np.asarray(q_weights, dtype=np.float64), 0.0, 1.0)
    q_sum = float(np.sum(q))
    if q_sum <= 1e-12:
        return {"mean": 0.0, "win_rate": 0.0, "pl_ratio": 0.0, "safety_margin": 0.0}

    w_mean = float(np.sum(q * rets) / q_sum)
    win_mask = rets > 0
    loss_mask = ~win_mask
    w_win_rate = float(np.sum(q[win_mask]) / q_sum)
    avg_win = float(np.sum(q[win_mask] * rets[win_mask]) / np.sum(q[win_mask])) if float(q[win_mask].sum()) > 0 else 0.0
    avg_loss = (
        float(np.sum(q[loss_mask] * np.abs(rets[loss_mask])) / np.sum(q[loss_mask]))
        if float(q[loss_mask].sum()) > 0
        else 1e-6
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


def _weighted_bootstrap_evidence(
    returns: np.ndarray,
    quality: np.ndarray,
    null_returns: np.ndarray,
    rng: np.random.Generator,
    iterations: int = 1000,
    *,
    paired_zero_null: bool = False,
) -> tuple[float, float]:
    """Compare the same Q-weighted statistic under the null and resample adjacent events for its CI."""
    returns = np.asarray(returns, dtype=float)
    quality = np.asarray(quality, dtype=float)
    n = len(returns)
    if n < 2 or quality.sum() <= 0 or len(null_returns) < 2:
        return 1.0, float("-inf")
    q = quality / quality.sum()
    observed = float(q @ returns)
    block = max(2, int(np.ceil(np.sqrt(n))))
    null_means, observed_means = [], []
    for _ in range(iterations):
        starts = rng.integers(0, len(null_returns), size=int(np.ceil(n / block)))
        null_idx = ((starts[:, None] + np.arange(block)) % len(null_returns)).ravel()[:n]
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        idx = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        observed_mean = float(np.average(returns[idx], weights=quality[idx]))
        observed_means.append(observed_mean)
        # For an execution zero-exposure null, resample return/quality pairs and
        # recenter that same statistic. Random-entry ruler diagnostics retain
        # their separate null pool with the observed quality vector.
        null_means.append(observed_mean - observed if paired_zero_null else float(q @ null_returns[null_idx]))
    p_value = (1 + np.count_nonzero(np.asarray(null_means) >= observed)) / (iterations + 1)
    return float(p_value), float(np.quantile(observed_means, 0.025))


def run_gate1_forensic_audit(
    df_tf: pd.DataFrame,
    triggers_tf: np.ndarray,
    direction: int,
    ruler: ArchetypeRuler,
    timeframe: str,
    rng: np.random.Generator | None = None,
    weights_tf: np.ndarray | None = None,
    fee_rate: float = 0.0002,
    is_end_dt: pd.Timestamp = IS_END_DT,
) -> dict[str, Any]:
    """
    使用该战略方向专属的法医三重屏障标尺 (ruler) 在特定周期 df_tf 上执行样本内 (2020-2023) 审计。
    Phase 2 升级：
    1. 支持传入 `weights_tf` 激活连续质量分 Q_{i,t} 加权期望收益与加权安全边际；
    2. 考核持仓路径 MFE/MAE 偏离比 (`excursion_ratio >= ruler.min_excursion_ratio`)；
    3. 考核单因子扣费后 `is_net_sharpe` 与前向多期 Rank IC / ICIR 衰减曲线。
    """
    del rng
    is_mask = (df_tf["dt"] <= is_end_dt).values
    oos_mask = (df_tf["dt"] > is_end_dt).values

    min_samples = 30
    eff_max_bars = ruler.max_bars * 2 if timeframe == "5m" else ruler.max_bars
    eff_tp_atr = ruler.tp_atr * 1.35 if timeframe == "5m" else ruler.tp_atr
    eff_sl_atr = ruler.sl_atr * 1.20 if timeframe == "5m" else ruler.sl_atr
    min_warmup = min(240 if timeframe == "5m" else 60, max(10, int(is_mask.sum()) // 4))

    trig_bool = np.asarray(triggers_tf, dtype=bool)
    is_raw_idx = np.where(trig_bool & is_mask)[0]
    is_purged_idx = is_raw_idx

    tb_is = evaluate_triple_barrier(
        df_tf,
        is_purged_idx,
        direction,
        max_bars=eff_max_bars,
        tp_atr=eff_tp_atr,
        sl_atr=eff_sl_atr,
        fee_rate=fee_rate,
        weights_tf=weights_tf,
        label_end_dt=is_end_dt,
        purge_overlaps=True,
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
    effective_n = float(1.0 / np.sum(q_norm**2))
    mean_mfe_atr = float(np.sum(q_norm * mfe_arr))
    mean_mae_atr = float(np.sum(q_norm * mae_arr))
    excursion_ratio = float(mean_mfe_atr / max(mean_mae_atr, 1e-6))

    # 500 轮随机入场置换检验 (同样使用该专属标尺)
    valid_pool = np.where(
        is_mask & (np.arange(len(df_tf)) >= min_warmup) & (np.arange(len(df_tf)) < int(is_mask.sum()) - eff_max_bars)
    )[0]
    if len(valid_pool) < 5:
        return {"passed": False, "reason": "样本内基准事件不足", "is_n_purged": n_purged}
    unit_seed = (
        4242
        + int(np.sum(is_raw_idx))
        + sum((idx + 1) * ord(c) for idx, c in enumerate(f"{ruler.archetype}_{timeframe}"))
    )
    local_rng = np.random.default_rng(unit_seed)
    rand_pool_idx = np.sort(local_rng.choice(valid_pool, size=min(1000, len(valid_pool)), replace=False))
    rand_tb = evaluate_triple_barrier(
        df_tf,
        rand_pool_idx,
        direction,
        max_bars=eff_max_bars,
        tp_atr=eff_tp_atr,
        sl_atr=eff_sl_atr,
        fee_rate=fee_rate,
        label_end_dt=is_end_dt,
        purge_overlaps=True,
    )
    if rand_tb.empty:
        return {"passed": False, "reason": "样本内基准事件不足", "is_n_purged": n_purged}
    rand_rets = rand_tb["net_ret"].values
    bench_drift = float(np.mean(rand_rets))
    excess_alpha = f_mean - bench_drift

    boot_pval, mean_ci_low = _weighted_bootstrap_evidence(rets_is, q_w, rand_rets, local_rng)

    # 单因子扣费 Net Sharpe 与前向 Rank IC 衰减曲线审计
    if weights_tf is not None:
        net_sh_pack = compute_single_factor_net_sharpe_and_turnover(
            df_tf, weights_tf, timeframe, fee_rate=fee_rate, is_end_dt=is_end_dt
        )
        ic_pack = compute_forward_rank_ic_profile(df_tf, weights_tf, is_end_dt=is_end_dt)
    else:
        synth_w = np.zeros(len(df_tf), dtype=np.float64)
        for row_idx, r_row in tb_is.iterrows():
            _ = row_idx
            st_i = int(r_row["trigger_idx"])
            ed_i = int(r_row["exit_idx"])
            synth_w[st_i:ed_i] = float(direction)
        net_sh_pack = compute_single_factor_net_sharpe_and_turnover(
            df_tf, synth_w, timeframe, fee_rate=fee_rate, is_end_dt=is_end_dt
        )
        ic_pack = compute_forward_rank_ic_profile(df_tf, synth_w, is_end_dt=is_end_dt)

    is_net_sharpe = float(net_sh_pack["is_net_sharpe"])
    ann_turnover = float(net_sh_pack["ann_turnover"])

    # 样本外 (2024-2026) 盲测表现跟踪
    oos_raw_idx = np.where(trig_bool & oos_mask)[0]
    oos_purged_idx = oos_raw_idx
    tb_oos = evaluate_triple_barrier(
        df_tf,
        oos_purged_idx,
        direction,
        max_bars=eff_max_bars,
        tp_atr=eff_tp_atr,
        sl_atr=eff_sl_atr,
        fee_rate=fee_rate,
        weights_tf=weights_tf,
        purge_overlaps=True,
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
    min_safety_dec = max(0.10, ruler.min_safety_pct / 100.0)
    max_pval = min(0.05, ruler.max_boot_pval)
    min_exc_ratio = float(getattr(ruler, "min_excursion_ratio", 0.90))
    min_net_sh = float(getattr(ruler, "min_net_sharpe", 0.0))

    passed = (
        effective_n >= min_samples - 1e-9
        and f_win > 0.50
        and pl_ratio > 1.5
        and mean_ci_low > 0.0
        and excess_alpha >= min_alpha_dec
        and safety_margin > min_safety_dec
        and boot_pval <= max_pval
        and excursion_ratio >= min_exc_ratio
        and (weights_tf is None or is_net_sharpe > min_net_sh)
    )
    reasons = []
    if effective_n < min_samples - 1e-9:
        reasons.append(f"Q加权有效样本数={effective_n:.2f} < {min_samples}")
    if f_win <= 0.50:
        reasons.append("胜率未超过50%")
    if pl_ratio <= 1.5:
        reasons.append("盈亏比未超过1.5")
    if mean_ci_low <= 0.0:
        reasons.append("净收益95%置信下界不大于0")
    if excess_alpha < min_alpha_dec:
        reasons.append(f"超额Alpha={excess_alpha * 100:+.2f}% < +{min_alpha_dec * 100:.2f}%")
    if safety_margin <= min_safety_dec:
        reasons.append(f"安全垫={safety_margin * 100:+.1f}% 未超过 +{min_safety_dec * 100:.1f}%")
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
        "is_effective_n": effective_n,
        "is_mean_ci_low_pct": mean_ci_low * 100.0,
        "events": tb_is.assign(r_atr=tb_is["net_ret"] / tb_is["atr_pct"]).to_dict("records"),
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
    max_cv: float = 0.35,
    soft_gate: bool = False,
    fee_rate: float = 0.0002,
    is_end_dt: pd.Timestamp = IS_END_DT,
) -> dict[str, Any]:
    """
    参数平原检验 (Parameter Smoothness Gate)：
    对候选信号的核心参数（如 DecayConfig.lam0、scale 或窗口长度）执行 ±20% 共 3~5 组邻域微扰，
    计算邻域净收益/夏普的变异系数 CV = σ / |μ| 与均值保持率 μ_neighbor / μ_center，
    要求保持率 >= 0.70 且 CV <= 0.35，剔除“孤岛参数”。
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
                    if "is_net_sharpe" not in res:
                        raise ValueError("Parameter audit requires is_net_sharpe, not mean returns.")
                    m_val = float(res["is_net_sharpe"])
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
                        scale=base_scale,
                        decay=DecayConfig(lam0=0.10 * r),
                        soft_gate=False,
                    )
                else:
                    trig_arr, w_arr = compute_signal_triggers_and_weights(df_tf, sig_id, scale=base_scale * r)
                if soft_gate:
                    from examples.multi_factor_cta.atomic_factors import SoftCrossSectionalGate

                    gate = SoftCrossSectionalGate()
                    rank = gate.extract_cs_rank(df_tf)
                    if rank is None:
                        raise ValueError(
                            "Parameter admission requires the same cross-sectional inputs as the candidate."
                        )
                    trig_arr, w_arr = gate.apply_to_triggers_and_weights(
                        trig_arr,
                        w_arr,
                        rank,
                        gate.extract_causal_er(df_tf),
                        archetype=spec.archetype,
                        direction=spec.direction,
                    )
                sharpe = compute_single_factor_net_sharpe_and_turnover(
                    df_tf, w_arr, timeframe, fee_rate=fee_rate, is_end_dt=is_end_dt
                )
                metrics_list.append(float(sharpe["is_net_sharpe"]))
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


def _next_open_rescue_metrics(
    df_fold: pd.DataFrame,
    candidate_decisions: np.ndarray,
    portfolio_decisions: np.ndarray,
) -> tuple[float, float]:
    """Measure fill-aligned gap coverage and net rescue contribution inside one completed fold.

    The drawdown state at an opening execution uses only returns and costs already
    observed there. Candidate contribution covers that open-to-next-open interval,
    with final-open-to-close marking and explicit terminal liquidation. Closing-only
    trades remain in the rescue sample so zero exposure cannot hide their exit fee.
    """
    interval_returns, _ = _prepare_next_open_slice(df_fold)
    n = len(interval_returns)
    if not n:
        return 0.0, 0.0
    fee = float(df_fold.attrs.get("fee_rate", 0.0002))
    candidate = np.asarray(candidate_decisions, dtype=float)
    portfolio = np.asarray(portfolio_decisions, dtype=float)
    if (
        len(candidate) != n
        or len(portfolio) != n
        or not np.isfinite(candidate).all()
        or not np.isfinite(portfolio).all()
    ):
        raise ValueError("Rescue decisions must be finite and aligned with the training fold")
    if not np.isfinite(fee) or fee < 0:
        raise ValueError("Rescue execution fee must be finite and nonnegative")
    opens = df_fold["open"].to_numpy(dtype=float)
    candidate_ledger = decision_execution_ledger(opens, candidate, fee, terminal_close=float(df_fold["close"].iloc[-1]))
    portfolio_ledger = decision_execution_ledger(opens, portfolio, fee)
    intervals = closed_execution_intervals(candidate_ledger)
    cand_held = np.asarray(candidate_ledger.executed_weights[:-1])
    port_held = np.asarray(portfolio_ledger.executed_weights)
    port_nav = 1.0 + np.cumsum(portfolio_ledger.net_returns)
    peak = np.maximum.accumulate(np.maximum(port_nav, 1.0))
    drawdown = (port_nav - peak) / peak

    active = np.abs(cand_held) > 0.01
    gap_fill = float(np.mean(np.abs(port_held[active]) <= 0.25) * 100) if active.any() else 0.0
    cand_turnover = np.asarray(intervals.turnover)
    candidate_net = np.asarray(intervals.net_returns)
    transaction_active = active | (cand_turnover > 0.01)
    rescue = ((np.abs(port_held) <= 0.20) | (drawdown <= -0.015)) & transaction_active
    rescue_bp = float(np.mean(candidate_net[rescue]) * 10000) if int(rescue.sum()) >= 5 else 0.0
    return gap_fill, rescue_bp


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
        return {
            "passed": False,
            "reason": "样本内历史不足",
            "max_corr": 0.0,
            "max_full_sample_corr": 0.0,
            "max_corr_peer": "None",
            "gap_fill_ratio_pct": 0.0,
            "dd_rescue_alpha_bp": 0.0,
        }
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

    gap_fill_pct, dd_rescue_alpha_bp = _next_open_rescue_metrics(
        df_30m.loc[is_mask].reset_index(drop=True), c_sub, current_portfolio_w_30m[is_mask]
    )

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
    """Map closed-bar decisions by their observed timestamps, retaining prior decisions across missing bars."""
    from examples.multi_factor_cta.data_engine import map_tf_weight_to_base_grid

    if timeframe not in TF_TO_5M_GROUP_SIZE:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return map_tf_weight_to_base_grid(df_source, w_source, df_target)


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

    # Recompute from this data prefix; DataFrame.attrs can survive same-length price edits.
    close_arr = df_grid["close"].values
    ret_5m = pd.Series(close_arr).pct_change().fillna(0.0)
    vol_15d = ret_5m.rolling(720, min_periods=48).std() * np.sqrt(48.0 * 242.0)
    net_chg_5d = np.abs(pd.Series(close_arr).diff(240).fillna(0.0).values)
    abs_path_5d = pd.Series(np.abs(np.diff(close_arr, prepend=close_arr[0]))).rolling(240, min_periods=48).sum()
    er_5d = np.clip(net_chg_5d / np.maximum(abs_path_5d.fillna(np.inf).values, 1e-6), 0.0, 1.0)
    turb_vol = vol_15d.values * (1.0 - 0.35 * er_5d)
    vol_cap = np.nan_to_num(np.clip(0.28 / np.maximum(turb_vol, 0.28), 0.72, 1.0), nan=0.0)

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
    """Execute decisions at next open and liquidate at the final observed close."""
    if len(df_30m_slice) != len(w_slice):
        raise ValueError("Bar and weight lengths must agree.")
    if df_30m_slice.empty:
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], name="date"))
        return {
            "trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "active_pct": 0.0,
            "cum_ret_pct": 0.0,
            "ann_ret_pct": 0.0,
            "sharpe": 0.0,
            "mdd_pct": 0.0,
            "calmar": 0.0,
            "composite_score": 0.0,
            "nav_series": empty,
            "dd_series": empty.copy(),
            "daily_return": pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "total": pd.Series(dtype=float)}),
        }
    symbols = df_30m_slice["symbol"].unique()
    if len(symbols) != 1:
        raise ValueError("backtest_slice requires one asset; use the portfolio allocator for multiple assets.")
    decision_dt = pd.to_datetime(df_30m_slice["dt"]).reset_index(drop=True)
    if decision_dt.isna().any() or not decision_dt.is_monotonic_increasing or decision_dt.duplicated().any():
        raise ValueError("Execution timestamps must be unique and ordered")
    final_price = float(df_30m_slice["close"].iloc[-1])
    ledger = decision_execution_ledger(
        df_30m_slice["open"].to_numpy(dtype=np.float64),
        np.asarray(w_slice, dtype=np.float64),
        float(df_30m_slice.attrs.get("fee_rate", 0.0002)),
        terminal_close=final_price,
    )
    execution_dt = pd.DatetimeIndex([*decision_dt, decision_dt.iloc[-1] + pd.Timedelta(microseconds=1)])
    ret_s = pd.Series(ledger.net_returns, index=execution_dt).groupby(execution_dt.normalize()).sum()
    st = daily_performance(ret_s.to_numpy(dtype=np.float64), yearly_days=252)
    executed = np.asarray(ledger.executed_weights[:-1])
    tot_t, l_t, s_t = count_round_trips(executed)
    active_pct = float((np.abs(executed) > 0.01).mean() * 100.0)

    cum_ret = float(st.get("绝对收益", 0.0) * 100.0)
    ann_ret = float(st.get("年化", 0.0) * 100.0)
    sharpe = float(st.get("夏普", 0.0))
    mdd = float(st.get("最大回撤", 0.0) * 100.0)
    calmar = float(st.get("卡玛", 0.0))
    score = sharpe + 0.35 * calmar + 0.05 * ann_ret

    dr = ret_s.rename_axis("date").reset_index(name="total")
    nav_s = 1 + ret_s.cumsum()
    peak = nav_s.cummax().clip(lower=1.0)
    dd_s = (nav_s - peak) / peak * 100

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
        "daily_return": dr,
        "nav_series": nav_s,
        "dd_series": dd_s,
    }


from examples.alpha_mining_agent.hypothesis_bank import _bootstrap_atomic_ssot as _sync_ssot  # noqa: E402

_sync_ssot()
