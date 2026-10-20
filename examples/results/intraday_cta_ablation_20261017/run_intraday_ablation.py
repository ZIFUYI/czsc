#!/usr/bin/env python
"""
日内周期（15m / 30m / 60m）商品期货多因子 CTA 体系可行性验证与四步控制变量消融实验
(Intraday Multi-Factor CTA Feasibility & Four-Step Ablation Study)

实验设计：
- 资产池：Core-20 主力商品期货（覆盖贵金属、有色、黑色、能化、农产品 5 大板块，2024-01-02 ~ 2026-09-30 完整 15m/30m/60m/1d 数据）
- 滚动样本外（Walk-Forward）：
    * Fold 1: 训练 2024-01-02 ~ 2024-12-31 -> 样本外测试 2025-01-02 ~ 2025-12-31 (2025 全年)
    * Fold 2: 训练 2024-01-02 ~ 2025-12-31 -> 样本外测试 2026-01-05 ~ 2026-09-30 (2026 Q1-Q3)
    * Stitched OOS: 2025-01-02 ~ 2026-09-30 (共 1.75 年、425 个交易日纯样本外)
- 四步控制变量消融：
    * Step 1 (准入漏斗消融): Gate-0 (全量不筛) vs Gate-1 (原版严格一票否决) vs Gate-2 (单品种本地正期望) vs Gate-3 (跨品种池化协同准入)
    * Step 2 (单品种多因子合成消融): Comb-0 (仅A簇趋势) vs Comb-1 (六簇简单等权) vs Comb-2 (六簇分层预算) vs Comb-3 (+施密特正交) vs Comb-4 (+经验贝叶斯动态配权)
    * Step 3 (执行状态机与跨品种分配消融): Alloc-0 (无滞回+静态1/N) vs Alloc-1 (+滞回死区状态机) vs Alloc-2 (+动态活跃池与逆波动率) vs Alloc-3 (+板块HRP) vs Alloc-4 (+截面强弱倾斜)
    * Step 4 (周期拆解与 0~10bp 摩擦压力测试): 15m only vs 30m only vs 60m only vs 多周期联立，以及 0/2/5/10 bp 额外滑点压力测试
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.alpha_mining_agent.forensic_pipeline import (  # noqa: E402
    compute_single_factor_net_sharpe_and_turnover,
    evaluate_execution_events,
)
from examples.alpha_mining_agent.hypothesis_bank import (  # noqa: E402
    ALL_ARCHETYPES,
    ALL_SIGNAL_SPECS,
    ARCH_A,
    BASE_SEED_SIGNALS,
)
from examples.alpha_mining_agent.run_evolution_loop import compute_research_signal  # noqa: E402
from examples.multi_factor_cta.atomic_factors import ATOMIC_SIGNAL_IDS  # noqa: E402
from examples.multi_factor_cta.data_engine import (  # noqa: E402
    AssetClass,
    AssetSpec,
    CrossAssetDataEngine,
    map_tf_weight_to_base_grid,
)
from examples.multi_factor_cta.forensic_validator import RollingEmpiricalBayesWeighter  # noqa: E402
from examples.multi_factor_cta.run_framework_demo import load_research_universe  # noqa: E402
from examples.multi_factor_cta.synthesis_engine import (  # noqa: E402
    ContinuousSignalStateMachine,
    CrossAssetHRPSynthesizer,
    IntraAssetThreePillarSynthesizer,
    account_portfolio_execution,
    build_asset_execution_inputs,
)

OUT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "examples/results/research_rerun_20261017/manifest.json"
CACHE_ROOT = ROOT / "examples/results/research_rerun_20261017/intraday/native_cache"

# Core-20 主力商品期货池（覆盖全部 5 大商品板块，2024.01-2026.09 全周期无缺失）
CORE_20_IDS = (
    # 贵金属 (2)
    "ag",
    "au",
    # 有色金属 (4)
    "cu",
    "al",
    "zn",
    "ni",
    # 黑色建材 (5)
    "rb",
    "hc",
    "i",
    "jm",
    "fg",
    # 能源化工 (5)
    "sc",
    "ta",
    "ma",
    "ru",
    "pp",
    # 农产品与油脂 (4)
    "m",
    "p",
    "oi",
    "sr",
)

FOLDS = (
    {
        "id": "Fold1_2025",
        "train_start": "2024-01-02 00:00:00",
        "cutoff": "2024-12-31 23:59:59",
        "oos_start": "2025-01-02 00:00:00",
        "oos_end": "2025-12-31 23:59:59",
    },
    {
        "id": "Fold2_2026",
        "train_start": "2024-01-02 00:00:00",
        "cutoff": "2025-12-31 23:59:59",
        "oos_start": "2026-01-05 00:00:00",
        "oos_end": "2026-09-30 23:59:59",
    },
)


def get_core_candidates() -> list[str]:
    """获取覆盖 6 大战略簇与 60m/30m/15m 三周期的核心种子与连续强度候选因子 (94个)"""
    seed_ids = {s.sig_id for s in BASE_SEED_SIGNALS} | set(ATOMIC_SIGNAL_IDS)
    return sorted(
        f"{sid}@{tf}"
        for sid, f in ALL_SIGNAL_SPECS.items()
        if sid in seed_ids
        for tf in ("60m", "30m", "15m")
        if tf in f.eligible_tfs
    )


def evaluate_candidate_on_fold(
    df_tf: pd.DataFrame,
    w_tf: np.ndarray,
    tf: str,
    fee_rate: float,
    cutoff: pd.Timestamp,
) -> dict[str, Any]:
    """在训练截止时刻 cutoff 之前评估单个候选因子的 Rust 原生执行事件与时序净 Sharpe"""
    episodes = evaluate_execution_events(df_tf, w_tf, fee_rate=fee_rate, label_end_dt=cutoff)
    ts_stats = compute_single_factor_net_sharpe_and_turnover(
        df_tf, w_tf, timeframe=tf, fee_rate=fee_rate, is_end_dt=cutoff
    )
    n = len(episodes)
    if n == 0:
        return {
            "n": 0,
            "effective_n": 0.0,
            "mean_net_pct": 0.0,
            "q_mean_net_pct": 0.0,
            "win_rate": 0.0,
            "pl_ratio": 0.0,
            "safety_margin": -1.0,
            "t_stat": 0.0,
            "is_net_sharpe": ts_stats["is_net_sharpe"],
            "ann_turnover": ts_stats["ann_turnover"],
            "strict_local_pass": False,
            "event_returns": np.zeros(0, dtype=np.float64),
        }

    rets = episodes["net_ret"].to_numpy(dtype=np.float64)
    q = np.clip(episodes["quality_w"].to_numpy(dtype=np.float64), 1e-6, 1.0)
    q_sum = float(q.sum())
    eff_n = float((q_sum * q_sum) / max(float((q * q).sum()), 1e-12))
    mean_ret = float(rets.mean())
    q_mean_ret = float((q * rets).sum() / q_sum)
    std_ret = float(rets.std(ddof=1)) if n > 1 else 1e-6
    t_stat = float(mean_ret / (std_ret / np.sqrt(n))) if std_ret > 1e-12 and n > 1 else 0.0

    wins = rets[rets > 0]
    losses = np.abs(rets[rets <= 0])
    win_rate = float(len(wins) / n)
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 1e-6
    pl_ratio = avg_win / max(avg_loss, 1e-6)
    be_win = 1.0 / (1.0 + pl_ratio)
    safety_margin = win_rate - be_win

    # CI 下界近似 (95% 单侧 / 97.5% 双侧)
    ci_low = mean_ret - 1.96 * (std_ret / np.sqrt(n)) if n >= 2 else -1.0
    strict_local_pass = bool(
        n >= 30
        and eff_n >= 30.0
        and win_rate > 0.50
        and pl_ratio > 1.50
        and safety_margin > 0.10
        and ci_low > 0.0
        and ts_stats["is_net_sharpe"] > 0.0
    )
    return {
        "n": int(n),
        "effective_n": round(eff_n, 2),
        "mean_net_pct": round(mean_ret * 100.0, 4),
        "q_mean_net_pct": round(q_mean_ret * 100.0, 4),
        "win_rate": round(win_rate * 100.0, 2),
        "pl_ratio": round(pl_ratio, 3),
        "safety_margin": round(safety_margin * 100.0, 2),
        "t_stat": round(t_stat, 3),
        "is_net_sharpe": ts_stats["is_net_sharpe"],
        "ann_turnover": ts_stats["ann_turnover"],
        "strict_local_pass": strict_local_pass,
        "event_returns": rets,
    }


def select_factors_by_gate(
    fold_scores: pd.DataFrame,
    pooled_stats: pd.DataFrame,
    asset_id: str,
    gate_mode: str,
    allowed_tfs: tuple[str, ...] = ("60m", "30m", "15m"),
) -> list[str]:
    """根据不同准入漏斗规则筛选当前资产的入选候选因子 (uid = sid@tf)"""
    sub = fold_scores[(fold_scores["asset"] == asset_id) & (fold_scores["tf"].isin(allowed_tfs))].copy()
    if sub.empty:
        return []

    if gate_mode == "Gate0_All":
        return sorted(sub["uid"].tolist())

    if gate_mode == "Gate1_Strict":
        # 原版严苛准入：本地严格通过 + 全池无一品种缺事件 (min_asset_n >= 30) + 普适率 >= 65% + 池化 t >= 1.96
        valid_pools = set(
            pooled_stats[
                (pooled_stats["min_asset_n"] >= 30)
                & (pooled_stats["pos_asset_ratio"] >= 0.65)
                & (pooled_stats["pooled_t"] >= 1.96)
            ]["uid"]
        )
        passed = sub[sub["strict_local_pass"] & sub["uid"].isin(valid_pools)]
        return sorted(passed["uid"].tolist())

    if gate_mode == "Gate2_Local":
        # 单品种本地正期望准入：n >= 15, mean_net_pct > 0, is_net_sharpe > 0.15
        passed = sub[(sub["n"] >= 15) & (sub["mean_net_pct"] > 0.0) & (sub["is_net_sharpe"] > 0.15)]
        return sorted(passed["uid"].tolist())

    if gate_mode == "Gate3_PooledSynergy":
        # 跨品种池化协同准入（多因子弱预测器正交合成范式）：
        # 1. 池内汇总事件数 total_n >= 150，池化平均净收益 > 0，池化 t_stat >= 1.50（或池化平均 Sharpe >= 0.10），正收益品种占比 >= 45%
        # 2. 按 6 大战略簇分层优选 Top 4 候选，结合本品种未出现严重水土不服 (is_net_sharpe >= -0.15 且 mean_net_pct >= -0.02)
        pool_ok = pooled_stats[
            (pooled_stats["tf"].isin(allowed_tfs))
            & (pooled_stats["total_n"] >= 150)
            & (pooled_stats["pooled_mean_pct"] > 0.0)
            & ((pooled_stats["pooled_t"] >= 1.50) | (pooled_stats["mean_sharpe"] >= 0.10))
            & (pooled_stats["pos_asset_ratio"] >= 0.45)
        ].copy()
        if pool_ok.empty:
            # 保底按各簇池化 t_stat 选最优正均值因子
            pool_ok = pooled_stats[
                (pooled_stats["tf"].isin(allowed_tfs))
                & (pooled_stats["total_n"] >= 100)
                & (pooled_stats["pooled_mean_pct"] > 0.0)
            ].copy()

        selected_uids: set[str] = set()
        for arch in ALL_ARCHETYPES:
            arch_df = pool_ok[pool_ok["archetype"] == arch].sort_values(
                ["pooled_t", "mean_sharpe"], ascending=[False, False]
            )
            selected_uids.update(arch_df.head(4)["uid"].tolist())

        # 本地兼容性过滤：剔除在该品种上明显反向亏损的因子
        local_ok = sub[
            sub["uid"].isin(selected_uids)
            & (sub["n"] >= 5)
            & ((sub["is_net_sharpe"] >= -0.10) | (sub["mean_net_pct"] > 0.0))
        ]
        return sorted(local_ok["uid"].tolist())

    raise ValueError(f"Unknown gate_mode: {gate_mode}")


def synthesize_asset_signal(
    df_grid: pd.DataFrame,
    unit_grid_map: dict[str, np.ndarray],
    selected_uids: list[str],
    spec: AssetSpec,
    comb_mode: str,
    use_state_machine: bool = True,
    rebalance_deadzone: float = 0.06,
) -> np.ndarray:
    """
    Step 2 单品种内多因子合成消融：
    - Comb0_ClusterA: 仅使用入选因子中的 A 簇（趋势突破与斜率动量）
    - Comb1_SimpleEqual: 入选因子简单等权合成
    - Comb2_SixClusterBudget: 六簇分层预算合成（无施密特正交、固定先验权重）
    - Comb3_GramSchmidtOrtho: 六簇分层预算 + 因果滚动施密特残差正交（固定先验权重）
    - Comb4_RollingEB: 六簇分层预算 + 施密特正交 + 滚动经验贝叶斯收缩动态配权
    """
    if not selected_uids:
        return np.zeros(len(df_grid), dtype=np.float64)

    if comb_mode == "Comb0_ClusterA":
        a_uids = [u for u in selected_uids if ALL_SIGNAL_SPECS[u.split("@")[0]].archetype == ARCH_A]
        if not a_uids:
            return np.zeros(len(df_grid), dtype=np.float64)
        raw = np.column_stack([unit_grid_map[u] for u in a_uids]).mean(axis=1)
        raw = np.clip(raw, -1.0, 1.0)
        if use_state_machine:
            return ContinuousSignalStateMachine(rebalance_deadband=rebalance_deadzone).transform(raw)
        return raw

    if comb_mode == "Comb1_SimpleEqual":
        raw = np.column_stack([unit_grid_map[u] for u in selected_uids]).mean(axis=1)
        raw = np.clip(raw, -1.0, 1.0)
        if use_state_machine:
            return ContinuousSignalStateMachine(rebalance_deadband=rebalance_deadzone).transform(raw)
        return raw

    sub_units = {u: unit_grid_map[u] for u in selected_uids}
    priors = {u: float(ALL_SIGNAL_SPECS[u.split("@")[0]].default_weight) for u in selected_uids}

    if comb_mode == "Comb4_RollingEB":
        alloc_map = RollingEmpiricalBayesWeighter().compute_weights(
            cross_asset_prior=priors,
            df_grid=df_grid,
            unit_weights_on_grid=sub_units,
            fee_rate=spec.fee_rate,
        )
    else:
        alloc_map = priors

    enable_gs = comb_mode in ("Comb3_GramSchmidtOrtho", "Comb4_RollingEB")
    synth = IntraAssetThreePillarSynthesizer()
    w = synth.synthesize(
        df_grid=df_grid,
        unit_weights_on_grid=sub_units,
        unit_alloc_map=alloc_map,
        asset_spec=spec,
        rebalance_deadzone=rebalance_deadzone if use_state_machine else 0.0,
        enable_regime_gating=True,
        soft_gate=False,
        enable_gram_schmidt=enable_gs,
        state_machine=ContinuousSignalStateMachine(rebalance_deadband=rebalance_deadzone)
        if use_state_machine
        else False,
    )
    return w


def allocate_and_account_fold(
    frames: dict[str, dict[str, pd.DataFrame]],
    specs: dict[str, AssetSpec],
    asset_targets: dict[str, np.ndarray],
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
    alloc_mode: str,
    extra_fee_bp: float = 0.0,
) -> dict[str, Any]:
    """
    Step 3 跨品种组合分配与 Rust 真实数量账本结算：
    - Alloc0_RawEqual: 静态 1/N 等权（信号端不加滞回状态机由上游控制）
    - Alloc1_SM_Equal: 静态 1/N 等权（带滞回死区状态机）
    - Alloc2_ActiveInvVol: 动态活跃池 (1/N_active) + 因果 60 日 ATR/收益率逆波动率平价
    - Alloc3_SectorHRP: 动态活跃池 + 逆波动率 + 板块内 1/sqrt(K) 去拥挤与板块间 HRP
    - Alloc4_CSTiltFull: 在 Alloc3 基础上叠加跨品种截面强弱风险预算倾斜
    """
    returns, positions, strengths, atr_pcts, sectors, fees = {}, {}, {}, {}, {}, {}
    for aid, spec in specs.items():
        full_grid = frames[spec.symbol]["15m"]
        mask = (full_grid["dt"] <= oos_end).to_numpy()
        grid = full_grid.loc[mask].copy().reset_index(drop=True)
        # 保留 IS 真实因果目标仓位供组合波动率预热，仅在截取 w_oos 时对 OOS 起点做空仓起步
        target = np.asarray(asset_targets[aid], dtype=np.float64)[mask].copy()

        ret_s, pos_s = build_asset_execution_inputs(grid, target)
        returns[aid] = ret_s
        positions[aid] = pos_s
        sectors[aid] = spec.sector
        fees[aid] = float(spec.fee_rate + extra_fee_bp / 10000.0)

        idx = ret_s.index
        if "cs_strength" in grid.columns:
            strengths[aid] = (
                pd.Series(grid["cs_strength"].to_numpy(dtype=np.float64), index=pd.DatetimeIndex(grid["dt"]))
                .shift(1)
                .reindex(idx)
                .ffill()
                .fillna(0.0)
            )
        if "atr14" in grid.columns:
            atr_pct = grid["atr14"].to_numpy(dtype=np.float64) / np.maximum(
                grid["close"].to_numpy(dtype=np.float64), 1e-6
            )
            atr_pcts[aid] = (
                pd.Series(atr_pct, index=pd.DatetimeIndex(grid["dt"])).shift(1).reindex(idx).ffill().fillna(0.005)
            )

    raw_ret_df = pd.DataFrame(returns).sort_index()
    tradable_mask = raw_ret_df.notna()
    ret_df = raw_ret_df.fillna(0.0)
    pos_df = pd.DataFrame(positions).reindex_like(ret_df).fillna(0.0)
    # 缺失报价时保持前一期已执行权重，避免在无报价时点触发虚假调仓
    for col in pos_df.columns:
        t_col = tradable_mask[col].to_numpy()
        p_col = pos_df[col].to_numpy().copy()
        for i in range(1, len(p_col)):
            if not t_col[i]:
                p_col[i] = p_col[i - 1]
        pos_df[col] = p_col

    n_assets = len(specs)
    periods_per_year = 252.0 * 16.0  # 15m 基准网格

    if alloc_mode in ("Alloc0_RawEqual", "Alloc1_SM_Equal"):
        weights_df = pos_df / float(n_assets)
    elif alloc_mode == "Alloc2_ActiveInvVol":
        # 动态活跃池 (1/N_active) + 因果 60 日逆波动率风险平价 + 组合目标波动率 14%
        bars_per_day = 16
        risk_win = 60 * bars_per_day
        roll_vol = raw_ret_df.rolling(risk_win, min_periods=10 * bars_per_day).std().shift(1) * np.sqrt(
            periods_per_year
        )
        roll_vol = roll_vol.ffill().fillna(0.15)
        active_mask = ((pos_df.abs() > 1e-3) & tradable_mask).astype(float)
        inv_vol = (1.0 / np.maximum(roll_vol, 0.05)) * active_mask
        denom = inv_vol.sum(axis=1).replace(0.0, np.nan)
        budget = inv_vol.div(denom, axis=0).fillna(0.0)
        # 单品种上限 0.25，最低有效分母 4
        n_act = active_mask.sum(axis=1).clip(lower=4.0)
        budget = np.minimum(budget, 1.0 / n_act.to_numpy()[:, None] * 1.5).clip(upper=0.25)
        weights_df = pos_df * budget
        # 组合级因果目标波动率缩放 (target_vol = 14%)，按期货交易日(夜盘归属下一交易日)聚合
        from multi_factor_cta.data_engine import assign_trading_date

        t_dates = pd.to_datetime(
            assign_trading_date(pd.Series(ret_df.index, index=ret_df.index), has_night_session=True)
        )
        unlev_bar_ret = (weights_df.shift(1).fillna(0.0) * ret_df).sum(axis=1)
        daily_unlev = unlev_bar_ret.groupby(t_dates).sum()
        realized_vol = daily_unlev.rolling(60, min_periods=15).std().shift(1) * np.sqrt(252.0)
        vol_scalar = (0.14 / realized_vol.replace(0.0, np.nan)).fillna(1.5).clip(lower=0.5, upper=2.5)
        bar_scalar = t_dates.map(vol_scalar).fillna(1.5).to_numpy()
        weights_df = weights_df.mul(bar_scalar, axis=0)
    elif alloc_mode in ("Alloc3_SectorHRP", "Alloc4_CSTiltFull"):
        hrp = CrossAssetHRPSynthesizer(
            target_portfolio_vol=0.14,
            max_single_asset_weight=0.25,
            min_effective_assets=4,
        )
        res = hrp.build_multi_asset_portfolio(
            raw_ret_df,
            sectors,
            asset_positions_df=pos_df,
            asset_fee_rates=fees,
            returns_kind="price",
            positions_timing="execution",
            periods_per_year=periods_per_year,
            asset_cs_strength_df=strengths if alloc_mode == "Alloc4_CSTiltFull" else None,
            asset_atr_pct_df=atr_pcts,
            use_dynamic_active_pool=True,
        )
        weights_df = res["asset_weights_df"]
    else:
        raise ValueError(f"Unknown alloc_mode: {alloc_mode}")

    # 截取 OOS 测试窗口，统一施加总杠杆上限 1.0，确保无报价时点冻结权重，首尾归零
    oos_mask = ret_df.index >= oos_start
    ret_oos = ret_df.loc[oos_mask].copy()
    mask_oos = tradable_mask.loc[oos_mask].copy()
    w_oos = weights_df.loc[oos_mask].copy()

    # 总名义敞口上限 1.0（与日线可行性实验保持可比口径）
    gross_exp = w_oos.abs().sum(axis=1)
    scale_cap = np.maximum(gross_exp, 1.0)
    w_oos = w_oos.div(scale_cap, axis=0)

    # 强制首时点平仓起步、末时点收盘清算，且缺失报价时点维持上一时点已执行权重
    w_oos.iloc[0] = 0.0
    for col in w_oos.columns:
        t_col = mask_oos[col].to_numpy()
        w_col = w_oos[col].to_numpy().copy()
        for i in range(1, len(w_col)):
            if not t_col[i]:
                w_col[i] = w_col[i - 1]
        if t_col[-1]:
            w_col[-1] = 0.0
        w_oos[col] = w_col

    ledger = account_portfolio_execution(
        ret_oos,
        w_oos,
        fees,
        tradable_mask=mask_oos,
    )
    return {
        "daily_returns": ledger["daily_returns"],
        "bar_net_returns": ledger["bar_net_returns"],
        "gross_returns_df": ledger["gross_returns_df"],
        "cost_returns_df": ledger["cost_returns_df"],
        "turnover_df": ledger["turnover_df"],
        "weights_df": w_oos,
    }


def summarize_stitched_oos(fold_ledgers: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """汇总两折拼接后的 2025.01-2026.09 纯样本外绩效指标"""
    daily_s = pd.concat([x["daily_returns"] for x in fold_ledgers]).sort_index()
    daily_s = daily_s.groupby(level=0).sum()

    gross_total = sum(float(x["gross_returns_df"].to_numpy().sum()) for x in fold_ledgers)
    cost_total = sum(float(x["cost_returns_df"].to_numpy().sum()) for x in fold_ledgers)
    net_total = float(daily_s.sum())
    turnover_total = sum(float(x["turnover_df"].to_numpy().sum()) for x in fold_ledgers)
    avg_exposure = float(np.mean([float(x["weights_df"].abs().sum(axis=1).mean()) for x in fold_ledgers]))

    n_days = len(daily_s)
    ann_factor = 252.0 / max(n_days, 1)
    ann_ret = net_total * ann_factor
    std_d = float(daily_s.std(ddof=1)) if n_days > 1 else 0.0
    ann_vol = std_d * np.sqrt(252.0)
    sharpe = float((daily_s.mean() / std_d) * np.sqrt(252.0)) if std_d > 1e-12 else 0.0

    nav = 1.0 + daily_s.cumsum()
    peak = np.maximum.accumulate(np.maximum(nav.to_numpy(), 1.0))
    dd = (nav.to_numpy() - peak) / peak
    max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0
    calmar = float(ann_ret / max_dd) if max_dd > 1e-6 else 0.0
    win_days = float((daily_s > 0).mean()) if len(daily_s) > 0 else 0.0

    # 分年度表现 (2025 vs 2026)
    d2025 = daily_s[daily_s.index.year == 2025]
    d2026 = daily_s[daily_s.index.year == 2026]
    ret_2025 = float(d2025.sum() * 100.0) if len(d2025) else 0.0
    ret_2026 = float(d2026.sum() * 100.0) if len(d2026) else 0.0

    return {
        "experiment": label,
        "oos_days": n_days,
        "gross_ret_pct": round(gross_total * 100.0, 2),
        "cost_ret_pct": round(cost_total * 100.0, 2),
        "net_ret_pct": round(net_total * 100.0, 2),
        "ann_ret_pct": round(ann_ret * 100.0, 2),
        "ann_vol_pct": round(ann_vol * 100.0, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(max_dd * 100.0, 2),
        "calmar": round(calmar, 2),
        "daily_win_pct": round(win_days * 100.0, 1),
        "ann_turnover": round(turnover_total * ann_factor, 1),
        "avg_exposure_pct": round(avg_exposure * 100.0, 1),
        "ret_2025_pct": round(ret_2025, 2),
        "ret_2026_pct": round(ret_2026, 2),
        "daily_series": daily_s,
    }


def main() -> None:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 110)
    print("  启动日内周期 (15m/30m/60m) 商品期货多因子 CTA 体系可行性验证与四步控制变量消融实验")
    print("=" * 110)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    rec_map = {r["id"]: r for r in manifest["intraday"] if r["asset_group"] == "commodity"}
    specs_list = [
        AssetSpec(
            symbol=rec_map[aid]["symbol"],
            name=aid,
            asset_class=AssetClass.COMMODITY_FUTURES,
            sector=rec_map[aid]["sector"],
            base_tf="15m",
            eligible_tfs=("1d", "60m", "30m", "15m"),
            fee_rate=rec_map[aid]["fee_rate"],
            has_night_session=rec_map[aid]["has_night_session"],
            data_path=Path(rec_map[aid]["path"]),
            adjust_rolls=False,
        )
        for aid in CORE_20_IDS
    ]
    specs_by_id = {s.name: s for s in specs_list}

    print("[Phase 0] 加载 Core-20 主力商品期货多周期数据 (15m/30m/60m/1d) 与跨品种因果截面特征...")
    engine = CrossAssetDataEngine(OUT_DIR / "ro_cache_placeholder")
    frames = load_research_universe(engine, specs_list)
    print(f"  -> 数据与截面特征加载完成，耗时 {time.time() - t_start:.1f}s")

    candidates = get_core_candidates()
    print(f"[Phase 1] 计算 20 个品种 × {len(candidates)} 个日内多周期候选因子信号并映射至 15m 执行网格...")
    t_sig = time.time()
    # unit_raw_tf[aid][uid] = (tf, w_on_tf), unit_grid_15m[aid][uid] = w_on_15m
    unit_raw_tf: dict[str, dict[str, np.ndarray]] = {aid: {} for aid in CORE_20_IDS}
    unit_grid_15m: dict[str, dict[str, np.ndarray]] = {aid: {} for aid in CORE_20_IDS}

    for idx, spec in enumerate(specs_list, 1):
        aid = spec.name
        tf_dfs = frames[spec.symbol]
        grid_15m = tf_dfs["15m"]
        for uid in candidates:
            sid, tf = uid.split("@")
            _, w_tf = compute_research_signal(tf_dfs[tf], sid, tf)
            unit_raw_tf[aid][uid] = w_tf
            unit_grid_15m[aid][uid] = map_tf_weight_to_base_grid(tf_dfs[tf], w_tf, grid_15m)
        if idx % 5 == 0 or idx == len(specs_list):
            print(f"  -> 已完成 {idx}/{len(specs_list)} 个品种信号预计算 ({time.time() - t_sig:.1f}s)", flush=True)

    # 评估两折训练期（Fold 1 & Fold 2）所有候选因子的单品种表现与跨品种池化表现
    print("\n[Phase 2] 执行两折滚动训练期 (Fold1: 2024, Fold2: 2024-2025) 原子因子体检与池化统计...")
    all_score_rows = []
    fold_score_dfs: dict[str, pd.DataFrame] = {}
    fold_pooled_dfs: dict[str, pd.DataFrame] = {}

    for fold in FOLDS:
        fid = fold["id"]
        cutoff = pd.Timestamp(fold["cutoff"])
        f_rows = []
        event_pool: dict[str, list[np.ndarray]] = {uid: [] for uid in candidates}

        for spec in specs_list:
            aid = spec.name
            tf_dfs = frames[spec.symbol]
            for uid in candidates:
                sid, tf = uid.split("@")
                res = evaluate_candidate_on_fold(
                    tf_dfs[tf],
                    unit_raw_tf[aid][uid],
                    tf=tf,
                    fee_rate=spec.fee_rate,
                    cutoff=cutoff,
                )
                if len(res["event_returns"]) > 0:
                    event_pool[uid].append(res["event_returns"])
                row = {
                    "fold": fid,
                    "asset": aid,
                    "sector": spec.sector,
                    "uid": uid,
                    "sid": sid,
                    "tf": tf,
                    "archetype": ALL_SIGNAL_SPECS[sid].archetype,
                    **{k: v for k, v in res.items() if k != "event_returns"},
                }
                f_rows.append(row)
                all_score_rows.append(row)

        df_f = pd.DataFrame(f_rows)
        fold_score_dfs[fid] = df_f

        # 汇总跨品种池化指标
        p_rows = []
        for uid in candidates:
            sid, tf = uid.split("@")
            sub_u = df_f[df_f["uid"] == uid]
            all_ev = np.concatenate(event_pool[uid]) if event_pool[uid] else np.zeros(0, dtype=np.float64)
            tot_n = len(all_ev)
            if tot_n > 1:
                p_mean = float(all_ev.mean())
                p_std = float(all_ev.std(ddof=1))
                p_t = float(p_mean / (p_std / np.sqrt(tot_n))) if p_std > 1e-12 else 0.0
            else:
                p_mean, p_t = 0.0, 0.0
            p_rows.append(
                {
                    "fold": fid,
                    "uid": uid,
                    "sid": sid,
                    "tf": tf,
                    "archetype": ALL_SIGNAL_SPECS[sid].archetype,
                    "total_n": int(tot_n),
                    "min_asset_n": int(sub_u["n"].min()),
                    "median_asset_n": float(sub_u["n"].median()),
                    "pos_asset_ratio": round(float((sub_u["mean_net_pct"] > 0).mean()), 3),
                    "pooled_mean_pct": round(p_mean * 100.0, 4),
                    "pooled_t": round(p_t, 3),
                    "mean_sharpe": round(float(sub_u["is_net_sharpe"].mean()), 3),
                    "strict_local_pass_cnt": int(sub_u["strict_local_pass"].sum()),
                }
            )
        df_p = pd.DataFrame(p_rows)
        fold_pooled_dfs[fid] = df_p

    df_all_scores = pd.DataFrame(all_score_rows)
    df_all_pooled = pd.concat(list(fold_pooled_dfs.values()), ignore_index=True)
    df_all_scores.to_csv(OUT_DIR / "intraday_factor_scores.csv", index=False)
    df_all_pooled.to_csv(OUT_DIR / "intraday_factor_pooled_summary.csv", index=False)

    # 打印各战略簇在日内周期上的整体体检表
    print("\n--- [诊断表 1] 六大战略簇在日内周期 (60m/30m/15m) 的池化体检汇总 ---")
    arch_summary = (
        df_all_pooled.groupby(["archetype", "tf"])
        .agg(
            candidates=("uid", "nunique"),
            avg_total_events=("total_n", "mean"),
            pos_mean_ratio=("pooled_mean_pct", lambda s: round(float((s > 0).mean() * 100), 1)),
            max_pooled_t=("pooled_t", "max"),
            top_mean_sharpe=("mean_sharpe", "max"),
        )
        .reset_index()
    )
    print(arch_summary.to_string(index=False))

    # =========================================================================
    # Step 1 ~ Step 4 控制变量消融实验配置矩阵
    # =========================================================================
    experiments = [
        # --- Step 1: 准入漏斗消融 (固定合成=Comb3_GramSchmidtOrtho, 分配=Alloc3_SectorHRP) ---
        {
            "step": "Step1_Gate",
            "name": "1A_Gate0_AllFactors",
            "gate": "Gate0_All",
            "comb": "Comb3_GramSchmidtOrtho",
            "alloc": "Alloc3_SectorHRP",
            "sm": True,
            "tfs": ("60m", "30m", "15m"),
        },
        {
            "step": "Step1_Gate",
            "name": "1B_Gate1_StrictOfficial",
            "gate": "Gate1_Strict",
            "comb": "Comb3_GramSchmidtOrtho",
            "alloc": "Alloc3_SectorHRP",
            "sm": True,
            "tfs": ("60m", "30m", "15m"),
        },
        {
            "step": "Step1_Gate",
            "name": "1C_Gate2_LocalEdge",
            "gate": "Gate2_Local",
            "comb": "Comb3_GramSchmidtOrtho",
            "alloc": "Alloc3_SectorHRP",
            "sm": True,
            "tfs": ("60m", "30m", "15m"),
        },
        {
            "step": "Step1_Gate",
            "name": "1D_Gate3_PooledSynergy",
            "gate": "Gate3_PooledSynergy",
            "comb": "Comb3_GramSchmidtOrtho",
            "alloc": "Alloc3_SectorHRP",
            "sm": True,
            "tfs": ("60m", "30m", "15m"),
        },
        # --- Step 2: 单品种多因子合成消融 (固定准入=Gate3_PooledSynergy, 分配=Alloc3_SectorHRP) ---
        {
            "step": "Step2_Combine",
            "name": "2A_Comb0_ClusterA_Only",
            "gate": "Gate3_PooledSynergy",
            "comb": "Comb0_ClusterA",
            "alloc": "Alloc3_SectorHRP",
            "sm": True,
            "tfs": ("60m", "30m", "15m"),
        },
        {
            "step": "Step2_Combine",
            "name": "2B_Comb1_SixCluster_Equal",
            "gate": "Gate3_PooledSynergy",
            "comb": "Comb1_SimpleEqual",
            "alloc": "Alloc3_SectorHRP",
            "sm": True,
            "tfs": ("60m", "30m", "15m"),
        },
        {
            "step": "Step2_Combine",
            "name": "2C_Comb2_SixCluster_Budget",
            "gate": "Gate3_PooledSynergy",
            "comb": "Comb2_SixClusterBudget",
            "alloc": "Alloc3_SectorHRP",
            "sm": True,
            "tfs": ("60m", "30m", "15m"),
        },
        {
            "step": "Step2_Combine",
            "name": "2D_Comb3_GramSchmidt_Ortho",
            "gate": "Gate3_PooledSynergy",
            "comb": "Comb3_GramSchmidtOrtho",
            "alloc": "Alloc3_SectorHRP",
            "sm": True,
            "tfs": ("60m", "30m", "15m"),
        },
        {
            "step": "Step2_Combine",
            "name": "2E_Comb4_Rolling_EB_Weight",
            "gate": "Gate3_PooledSynergy",
            "comb": "Comb4_RollingEB",
            "alloc": "Alloc3_SectorHRP",
            "sm": True,
            "tfs": ("60m", "30m", "15m"),
        },
        # --- Step 3: 执行状态机与跨品种组合分配消融 (固定准入=Gate3_PooledSynergy, 合成=Comb3_GramSchmidtOrtho) ---
        {
            "step": "Step3_Alloc",
            "name": "3A_Alloc0_NoSM_Static1N",
            "gate": "Gate3_PooledSynergy",
            "comb": "Comb3_GramSchmidtOrtho",
            "alloc": "Alloc0_RawEqual",
            "sm": False,
            "tfs": ("60m", "30m", "15m"),
        },
        {
            "step": "Step3_Alloc",
            "name": "3B_Alloc1_WithSM_Static1N",
            "gate": "Gate3_PooledSynergy",
            "comb": "Comb3_GramSchmidtOrtho",
            "alloc": "Alloc1_SM_Equal",
            "sm": True,
            "tfs": ("60m", "30m", "15m"),
        },
        {
            "step": "Step3_Alloc",
            "name": "3C_Alloc2_ActivePool_InvVol",
            "gate": "Gate3_PooledSynergy",
            "comb": "Comb3_GramSchmidtOrtho",
            "alloc": "Alloc2_ActiveInvVol",
            "sm": True,
            "tfs": ("60m", "30m", "15m"),
        },
        {
            "step": "Step3_Alloc",
            "name": "3D_Alloc3_Sector_HRP",
            "gate": "Gate3_PooledSynergy",
            "comb": "Comb3_GramSchmidtOrtho",
            "alloc": "Alloc3_SectorHRP",
            "sm": True,
            "tfs": ("60m", "30m", "15m"),
        },
        {
            "step": "Step3_Alloc",
            "name": "3E_Alloc4_CS_Tilt_Full",
            "gate": "Gate3_PooledSynergy",
            "comb": "Comb3_GramSchmidtOrtho",
            "alloc": "Alloc4_CSTiltFull",
            "sm": True,
            "tfs": ("60m", "30m", "15m"),
        },
        # --- Step 4A: 日内不同周期拆解消融 (15m only vs 30m only vs 60m only vs Multi-TF) ---
        {
            "step": "Step4_Timeframe",
            "name": "4A_TF_15m_Only",
            "gate": "Gate3_PooledSynergy",
            "comb": "Comb3_GramSchmidtOrtho",
            "alloc": "Alloc3_SectorHRP",
            "sm": True,
            "tfs": ("15m",),
        },
        {
            "step": "Step4_Timeframe",
            "name": "4B_TF_30m_Only",
            "gate": "Gate3_PooledSynergy",
            "comb": "Comb3_GramSchmidtOrtho",
            "alloc": "Alloc3_SectorHRP",
            "sm": True,
            "tfs": ("30m",),
        },
        {
            "step": "Step4_Timeframe",
            "name": "4C_TF_60m_Only",
            "gate": "Gate3_PooledSynergy",
            "comb": "Comb3_GramSchmidtOrtho",
            "alloc": "Alloc3_SectorHRP",
            "sm": True,
            "tfs": ("60m",),
        },
    ]

    print(f"\n[Phase 3] 运行 {len(experiments)} 组四步控制变量消融实验 (Stitched OOS: 2025.01 - 2026.09)...")
    ablation_rows = []
    membership_counts = {}
    champion_targets_by_fold: dict[str, dict[str, np.ndarray]] = {}

    for exp in experiments:
        exp_name = exp["name"]
        fold_ledgers = []
        avg_members = []

        for fold in FOLDS:
            fid = fold["id"]
            df_f = fold_score_dfs[fid]
            df_p = fold_pooled_dfs[fid]
            oos_start = pd.Timestamp(fold["oos_start"])
            oos_end = pd.Timestamp(fold["oos_end"])

            asset_targets = {}
            for aid in CORE_20_IDS:
                spec = specs_by_id[aid]
                df_grid = frames[spec.symbol]["15m"]
                members = select_factors_by_gate(df_f, df_p, aid, gate_mode=exp["gate"], allowed_tfs=exp["tfs"])
                avg_members.append(len(members))
                w_target = synthesize_asset_signal(
                    df_grid=df_grid,
                    unit_grid_map=unit_grid_15m[aid],
                    selected_uids=members,
                    spec=spec,
                    comb_mode=exp["comb"],
                    use_state_machine=exp["sm"],
                    rebalance_deadzone=0.06,
                )
                asset_targets[aid] = w_target

            if exp_name == "1D_Gate3_PooledSynergy":
                champion_targets_by_fold[fid] = asset_targets

            ledger = allocate_and_account_fold(
                frames=frames,
                specs=specs_by_id,
                asset_targets=asset_targets,
                oos_start=oos_start,
                oos_end=oos_end,
                alloc_mode=exp["alloc"],
                extra_fee_bp=0.0,
            )
            fold_ledgers.append(ledger)

        summary = summarize_stitched_oos(fold_ledgers, exp_name)
        summary["step"] = exp["step"]
        summary["avg_admitted_factors"] = round(float(np.mean(avg_members)), 1)
        membership_counts[exp_name] = summary["avg_admitted_factors"]
        ablation_rows.append({k: v for k, v in summary.items() if k != "daily_series"})
        print(
            f"  [{exp['step']}] {exp_name:<28} | 准入均值={summary['avg_admitted_factors']:>4.1f} | "
            f"毛益={summary['gross_ret_pct']:>+6.2f}% | 成本={summary['cost_ret_pct']:>5.2f}% | "
            f"净益={summary['net_ret_pct']:>+6.2f}% | 年化={summary['ann_ret_pct']:>+6.2f}% | "
            f"夏普={summary['sharpe']:>5.2f} | 回撤={summary['max_dd_pct']:>5.2f}% | 卡玛={summary['calmar']:>5.2f}",
            flush=True,
        )

    # Step 4B: 0 / 2 / 5 / 10 BP 摩擦压力测试 (针对推荐链路 Gate3 + Comb3 + Alloc3)
    print("\n[Phase 4] 执行摩擦压力测试 (额外滑点 0 / 2 / 5 / 10 BP)...")
    stress_rows = []
    for extra_bp in (0, 2, 5, 10):
        fold_ledgers = []
        for fold in FOLDS:
            fid = fold["id"]
            ledger = allocate_and_account_fold(
                frames=frames,
                specs=specs_by_id,
                asset_targets=champion_targets_by_fold[fid],
                oos_start=pd.Timestamp(fold["oos_start"]),
                oos_end=pd.Timestamp(fold["oos_end"]),
                alloc_mode="Alloc3_SectorHRP",
                extra_fee_bp=float(extra_bp),
            )
            fold_ledgers.append(ledger)
        s_res = summarize_stitched_oos(fold_ledgers, f"Friction_+{extra_bp}bp")
        s_res["extra_bp"] = extra_bp
        stress_rows.append({k: v for k, v in s_res.items() if k != "daily_series"})
        print(
            f"  [Friction +{extra_bp:>2}bp] 毛益={s_res['gross_ret_pct']:>+6.2f}% | 成本={s_res['cost_ret_pct']:>5.2f}% | "
            f"净益={s_res['net_ret_pct']:>+6.2f}% | 夏普={s_res['sharpe']:>5.2f} | 回撤={s_res['max_dd_pct']:>5.2f}%",
            flush=True,
        )

    df_ablation = pd.DataFrame(ablation_rows)
    df_stress = pd.DataFrame(stress_rows)
    df_ablation.to_csv(OUT_DIR / "ablation_results.csv", index=False)
    df_stress.to_csv(OUT_DIR / "friction_stress_results.csv", index=False)

    # 保存汇总 JSON
    out_summary = {
        "elapsed_seconds": round(time.time() - t_start, 2),
        "universe_size": len(CORE_20_IDS),
        "candidates_count": len(candidates),
        "ablation_results": ablation_rows,
        "friction_stress_results": stress_rows,
        "top_pooled_factors_fold2": fold_pooled_dfs["Fold2_2026"]
        .sort_values("pooled_t", ascending=False)
        .head(20)
        .to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(out_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n实验全部完成！总耗时 {time.time() - t_start:.1f}s，结果已保存至 {OUT_DIR}")


if __name__ == "__main__":
    main()
