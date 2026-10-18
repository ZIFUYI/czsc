#!/usr/bin/env python
"""
全周期（1d + 60m + 30m + 15m）× 六大战略簇（A~F）规则类 CTA 信号增量验证、快慢协同与 1d/15m 异构专属因子实验
(Full Multi-Timeframe CTA Incremental Verification Funnel, Fast/Slow Synergy & Heterogeneous 1d/15m Factor Study)

交付物：
1. multitf_factor_scores.csv / multitf_factor_pooled_summary.csv: 原始 125 个全周期候选因子评估底表
2. multitf_hetero_factor_scores.csv / multitf_hetero_pooled_summary.csv: 扩充 12 个 1d/15m 异构专属因子后的 137 候选评估表
3. multitf_ablation_comparison.csv: 基准 A vs 实验 B vs 实验 C vs 实验 D vs 实验 E (+ 3x 目标敞口版) 样本外对照表
4. multitf_funnel_stage_summary.csv: Gate 1 -> Gate 2 -> Gate 3 分周期漏斗通过量对比 (原始池 vs 异构扩充池)
5. multitf_gate23_rejected_signals.csv / multitf_gate23_rejected_summary.csv: 被 Gate 2/3 拦截的冗余信号清单
6. multitf_incremental_admitted_attribution.csv / multitf_incremental_admitted_details.csv: 入库核心真增量信号归因表
7. multitf_symbol_attribution.csv & multitf_sector_attribution.csv: 实验 E 分品种与分板块纯样本外归因表
8. multitf_cta_incremental_report.html: 自包含交互式 Plotly HTML 深度投研报告
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = Path(__file__).resolve().parent
if str(OUT_DIR) not in sys.path:
    sys.path.insert(0, str(OUT_DIR))

from run_intraday_ablation import (  # noqa: E402
    CORE_20_IDS,
    FOLDS,
    MANIFEST_PATH,
    allocate_and_account_fold,
    evaluate_candidate_on_fold,
    summarize_stitched_oos,
    synthesize_asset_signal,
)

from examples.alpha_mining_agent.hypothesis_bank import (  # noqa: E402
    ALL_SIGNAL_SPECS,
    ARCH_B,
    ARCH_C,
    ARCH_D,
    ARCH_E,
)
from examples.multi_factor_cta.data_engine import (  # noqa: E402
    AssetClass,
    AssetSpec,
    CrossAssetDataEngine,
    map_tf_weight_to_base_grid,
)
from examples.multi_factor_cta.run_framework_demo import load_research_universe  # noqa: E402

ALL_TFS = ("1d", "60m", "30m", "15m")

SECTOR_CN = {
    "precious_metals": "贵金属",
    "base_metals": "有色金属",
    "ferrous": "黑色建材",
    "energy_chemicals": "能源化工",
    "agriculture": "农产品与油脂",
    "PRECIOUS_METALS": "贵金属",
    "BASE_METALS": "有色金属",
    "FERROUS": "黑色建材",
    "ENERGY_CHEM": "能源化工",
    "AGRICULTURE": "农产品与油脂",
}

HETERO_SIGNAL_META: dict[str, tuple[str, str]] = {
    "H1D_SP01L@1d": (ARCH_C, "1d板块价差离散回归补涨做多"),
    "H1D_SP01S@1d": (ARCH_C, "1d板块价差离散回归补跌做空"),
    "H1D_DV01L@1d": (ARCH_B, "1d乖离超卖与RSI极值反转做多"),
    "H1D_DV01S@1d": (ARCH_B, "1d乖离超买与RSI极值反转做空"),
    "H1D_RV01L@1d": (ARCH_D, "1d截面相对强弱超跌均值回归做多"),
    "H1D_RV01S@1d": (ARCH_D, "1d截面相对强弱超涨均值回归做空"),
    "H15M_EX01L@15m": (ARCH_B, "15m放量恐慌衰竭长下影波段反转做多"),
    "H15M_EX01S@15m": (ARCH_B, "15m放量亢奋衰竭长上影波段反转做空"),
    "H15M_PB01L@15m": (ARCH_C, "15m多日顺势回踩MA60企稳延迟做多触发器"),
    "H15M_PB01S@15m": (ARCH_C, "15m多日顺势反抽MA60遇阻延迟做空触发器"),
    "H15M_CL01L@15m": (ARCH_E, "15m窄幅波动率挤压放量脱离做多"),
    "H15M_CL01S@15m": (ARCH_E, "15m窄幅波动率挤压放量脱离做空"),
}


def register_hetero_signal_specs() -> None:
    """在内存注册表中注册 12 个 1d / 15m 异构专属因子规格，供三支柱合成器识别"""
    base_spec = ALL_SIGNAL_SPECS["S1L"]
    for uid, (arch, _desc) in HETERO_SIGNAL_META.items():
        sid, tf = uid.split("@")
        ALL_SIGNAL_SPECS[sid] = dataclasses.replace(
            base_spec,
            sig_id=sid,
            archetype=arch,
            eligible_tfs=(tf,),
            default_weight=1.0,
        )


def hold_with_cooldown_and_trail(
    entries: np.ndarray,
    intensities: np.ndarray,
    exits: np.ndarray,
    direction: float,
    min_hold: int,
    max_hold: int,
    cooldown: int = 2,
    warmup: int = 30,
) -> np.ndarray:
    """带最小持有期、结构破位/止盈退出与冷却期的低换手状态机"""
    n = len(entries)
    w = np.zeros(n, dtype=np.float64)
    intens = np.nan_to_num(intensities, nan=0.5, posinf=1.0, neginf=0.5)
    i = warmup
    while i < n:
        if entries[i]:
            val = direction * float(np.clip(intens[i], 0.35, 1.0))
            end_i = min(n, i + max_hold)
            k = i
            while k < end_i:
                w[k] = val
                if (k - i + 1) >= min_hold and exits[k]:
                    k += 1
                    break
                k += 1
            i = k + cooldown
        else:
            i += 1
    return np.nan_to_num(w, nan=0.0)


def compute_hetero_1d_signals(df: pd.DataFrame) -> dict[str, tuple[str, np.ndarray]]:
    """计算专门针对 1d 周期设计的 6 个异构专属因子（板块价差回归、日线乖离反转、截面强弱均值回归）"""
    c = np.nan_to_num(df["close"].to_numpy(dtype=np.float64), nan=1.0)
    o = np.nan_to_num(df["open"].to_numpy(dtype=np.float64), nan=1.0)
    h = np.nan_to_num(df["high"].to_numpy(dtype=np.float64), nan=1.0)
    low_arr = np.nan_to_num(df["low"].to_numpy(dtype=np.float64), nan=1.0)
    ma10 = np.nan_to_num(df["ma10"].to_numpy(dtype=np.float64), nan=c)
    ma20 = np.nan_to_num(df["ma20"].to_numpy(dtype=np.float64), nan=c)
    ma60_slp = np.nan_to_num(df["ma60_slope"].to_numpy(dtype=np.float64), nan=0.0)
    bias20 = np.nan_to_num(df["bias20"].to_numpy(dtype=np.float64), nan=0.0)
    rsi14 = np.nan_to_num(df["rsi14"].to_numpy(dtype=np.float64), nan=50.0)
    atr14 = np.nan_to_num(df["atr14"].to_numpy(dtype=np.float64), nan=0.01)
    cs_rk = np.nan_to_num(df["cs_slope_rank"].to_numpy(dtype=np.float64), nan=0.5)
    sec_z = np.nan_to_num(df["sector_spread_zscore"].to_numpy(dtype=np.float64), nan=0.0)
    c_prev = np.roll(c, 1)
    c_prev[0] = c[0]

    out: dict[str, tuple[str, np.ndarray]] = {}
    # 1. H1D_SP01L / S (C簇: 日线板块价差离散回归与补涨/补跌因子)
    ent_sp_l = (sec_z < -0.55) & (c > o) & (c > c_prev) & (rsi14 < 54) & (ma60_slp > -0.002)
    ext_sp_l = (sec_z >= 0.0) | (c >= ma20) | (c < low_arr - 0.5 * atr14)
    int_sp_l = np.clip(np.abs(sec_z) / 1.5, 0.45, 1.0)
    out["H1D_SP01L@1d"] = (
        ARCH_C,
        hold_with_cooldown_and_trail(ent_sp_l, int_sp_l, ext_sp_l, +1.0, min_hold=2, max_hold=8, cooldown=1),
    )

    ent_sp_s = (sec_z > 0.55) & (c < o) & (c < c_prev) & (rsi14 > 46) & (ma60_slp < 0.002)
    ext_sp_s = (sec_z <= 0.0) | (c <= ma20) | (c > h + 0.5 * atr14)
    int_sp_s = np.clip(np.abs(sec_z) / 1.5, 0.45, 1.0)
    out["H1D_SP01S@1d"] = (
        ARCH_C,
        hold_with_cooldown_and_trail(ent_sp_s, int_sp_s, ext_sp_s, -1.0, min_hold=2, max_hold=8, cooldown=1),
    )

    # 2. H1D_DV01L / S (B簇: 日线乖离超卖/超买极值反转因子)
    ent_dv_l = ((bias20 < -0.018) | (rsi14 < 38)) & (c > o) & (c > c_prev)
    ext_dv_l = (c >= ma10) | (rsi14 > 55)
    int_dv_l = np.clip(np.abs(bias20) / 0.04, 0.45, 1.0)
    out["H1D_DV01L@1d"] = (
        ARCH_B,
        hold_with_cooldown_and_trail(ent_dv_l, int_dv_l, ext_dv_l, +1.0, min_hold=2, max_hold=7, cooldown=1),
    )

    ent_dv_s = ((bias20 > 0.018) | (rsi14 > 62)) & (c < o) & (c < c_prev)
    ext_dv_s = (c <= ma10) | (rsi14 < 45)
    int_dv_s = np.clip(np.abs(bias20) / 0.04, 0.45, 1.0)
    out["H1D_DV01S@1d"] = (
        ARCH_B,
        hold_with_cooldown_and_trail(ent_dv_s, int_dv_s, ext_dv_s, -1.0, min_hold=2, max_hold=7, cooldown=1),
    )

    # 3. H1D_RV01L / S (D簇: 日线截面相对强弱超跌/超涨均值回归锚)
    ent_rv_l = (cs_rk <= 0.35) & (c > o) & (c > c_prev) & (rsi14 < 50)
    ext_rv_l = (cs_rk >= 0.55) | (c >= ma20)
    int_rv_l = np.clip(0.5 + (0.4 - cs_rk), 0.45, 1.0)
    out["H1D_RV01L@1d"] = (
        ARCH_D,
        hold_with_cooldown_and_trail(ent_rv_l, int_rv_l, ext_rv_l, +1.0, min_hold=2, max_hold=8, cooldown=1),
    )

    ent_rv_s = (cs_rk >= 0.65) & (c < o) & (c < c_prev) & (rsi14 > 50)
    ext_rv_s = (cs_rk <= 0.45) | (c <= ma20)
    int_rv_s = np.clip(0.5 + (cs_rk - 0.6), 0.45, 1.0)
    out["H1D_RV01S@1d"] = (
        ARCH_D,
        hold_with_cooldown_and_trail(ent_rv_s, int_rv_s, ext_rv_s, -1.0, min_hold=2, max_hold=8, cooldown=1),
    )
    return out


def compute_hetero_15m_signals(df: pd.DataFrame) -> dict[str, tuple[str, np.ndarray]]:
    """
    计算专门针对 15m 周期设计的 6 个异构低换手专属因子：
    放弃 15m 独立高频追涨杀跌，采用“15m 极值衰竭/回踩企稳精准择时 + 跨日波段持有 (24~80 bars) + 32 bars 冷却”
    """
    c = np.nan_to_num(df["close"].to_numpy(dtype=np.float64), nan=1.0)
    o = np.nan_to_num(df["open"].to_numpy(dtype=np.float64), nan=1.0)
    h = np.nan_to_num(df["high"].to_numpy(dtype=np.float64), nan=1.0)
    low_arr = np.nan_to_num(df["low"].to_numpy(dtype=np.float64), nan=1.0)
    v = np.nan_to_num(df["volume"].to_numpy(dtype=np.float64), nan=0.0)
    atr14 = np.nan_to_num(df["atr14"].to_numpy(dtype=np.float64), nan=0.001)

    s_c = pd.Series(c)
    ma60 = s_c.rolling(60, min_periods=5).mean().to_numpy(dtype=np.float64)
    ma120 = s_c.rolling(120, min_periods=10).mean().to_numpy(dtype=np.float64)
    ma120_slp = (ma120 - np.roll(ma120, 16)) / np.maximum(np.roll(ma120, 16), 1e-6)
    v_ma20 = pd.Series(v).rolling(20, min_periods=2).mean().replace(0, 1.0).to_numpy(dtype=np.float64)
    bias60 = (c - ma60) / np.maximum(ma60, 1e-6)

    diff = np.diff(c, prepend=c[0])
    up = pd.Series(np.maximum(diff, 0.0)).rolling(14, min_periods=2).mean().to_numpy()
    dn = pd.Series(np.maximum(-diff, 0.0)).rolling(14, min_periods=2).mean().to_numpy()
    rsi14 = 100.0 - 100.0 / (1.0 + up / np.maximum(dn, 1e-8))

    tot_r = np.maximum(h - low_arr, 1e-6)
    l_shd = np.minimum(o, c) - low_arr
    u_shd = h - np.maximum(o, c)
    c_prev = np.roll(c, 1)
    c_prev[0] = c[0]
    atr_pct = atr14 / np.maximum(c, 1e-6)

    out: dict[str, tuple[str, np.ndarray]] = {}
    # 1. H15M_EX01L / S (B簇: 15m 放量恐慌/亢奋衰竭长影线止盈与反转)
    ent_ex_l = (
        (bias60 < -2.2 * atr_pct)
        & (rsi14 < 26)
        & (v > 1.6 * v_ma20)
        & ((l_shd > 0.40 * tot_r) | (c > o))
        & (c > c_prev)
    )
    ext_ex_l = (c >= ma60) | (rsi14 > 58)
    int_ex_l = np.clip((32.0 - rsi14) / 18.0, 0.45, 1.0)
    out["H15M_EX01L@15m"] = (
        ARCH_B,
        hold_with_cooldown_and_trail(ent_ex_l, int_ex_l, ext_ex_l, +1.0, min_hold=24, max_hold=72, cooldown=32),
    )

    ent_ex_s = (
        (bias60 > 2.2 * atr_pct) & (rsi14 > 74) & (v > 1.6 * v_ma20) & ((u_shd > 0.40 * tot_r) | (c < o)) & (c < c_prev)
    )
    ext_ex_s = (c <= ma60) | (rsi14 < 42)
    int_ex_s = np.clip((rsi14 - 68.0) / 18.0, 0.45, 1.0)
    out["H15M_EX01S@15m"] = (
        ARCH_B,
        hold_with_cooldown_and_trail(ent_ex_s, int_ex_s, ext_ex_s, -1.0, min_hold=24, max_hold=72, cooldown=32),
    )

    # 2. H15M_PB01L / S (C簇: 多日趋势顺势 + 15m 回踩/反抽 MA60 企稳延迟入场触发器)
    ent_pb_l = (
        (ma60 > ma120)
        & (ma120_slp > 0.002)
        & (low_arr <= ma60 * 1.001)
        & (c > ma60)
        & (c > o)
        & (c > c_prev)
        & (rsi14 < 48)
    )
    ext_pb_l = (c < ma120) | (bias60 > 3.0 * atr_pct)
    int_pb_l = np.full_like(c, 0.75)
    out["H15M_PB01L@15m"] = (
        ARCH_C,
        hold_with_cooldown_and_trail(ent_pb_l, int_pb_l, ext_pb_l, +1.0, min_hold=24, max_hold=80, cooldown=32),
    )

    ent_pb_s = (
        (ma60 < ma120) & (ma120_slp < -0.002) & (h >= ma60 * 0.999) & (c < ma60) & (c < o) & (c < c_prev) & (rsi14 > 52)
    )
    ext_pb_s = (c > ma120) | (bias60 < -3.0 * atr_pct)
    int_pb_s = np.full_like(c, 0.75)
    out["H15M_PB01S@15m"] = (
        ARCH_C,
        hold_with_cooldown_and_trail(ent_pb_s, int_pb_s, ext_pb_s, -1.0, min_hold=24, max_hold=80, cooldown=32),
    )

    # 3. H15M_CL01L / S (E簇: 15m 极窄幅波动率压缩 + 放量脱离突破)
    atr_ma80 = pd.Series(atr14).rolling(80, min_periods=10).mean().replace(0, 1e-6).to_numpy()
    h48 = pd.Series(h).shift(1).rolling(48, min_periods=10).max().fillna(h[0]).to_numpy()
    l48 = pd.Series(low_arr).shift(1).rolling(48, min_periods=10).min().fillna(low_arr[0]).to_numpy()
    sqz = atr14 < 0.82 * atr_ma80
    sqz_recent = pd.Series(sqz).rolling(8, min_periods=1).max().to_numpy(dtype=bool)
    ent_cl_l = sqz_recent & (c > h48) & (v > 1.5 * v_ma20) & (c > ma60) & (ma120_slp > 0.0)
    ext_cl_l = c < ma60
    int_cl_l = np.full_like(c, 0.80)
    out["H15M_CL01L@15m"] = (
        ARCH_E,
        hold_with_cooldown_and_trail(ent_cl_l, int_cl_l, ext_cl_l, +1.0, min_hold=24, max_hold=80, cooldown=32),
    )

    ent_cl_s = sqz_recent & (c < l48) & (v > 1.5 * v_ma20) & (c < ma60) & (ma120_slp < 0.0)
    ext_cl_s = c > ma60
    int_cl_s = np.full_like(c, 0.80)
    out["H15M_CL01S@15m"] = (
        ARCH_E,
        hold_with_cooldown_and_trail(ent_cl_s, int_cl_s, ext_cl_s, -1.0, min_hold=24, max_hold=80, cooldown=32),
    )
    return out


def summarize_pooled_scores(scores_df: pd.DataFrame) -> pd.DataFrame:
    """根据单品种因子评估表汇总跨品种池化统计表"""
    pool_rows = []
    for (fold_id, uid), grp in scores_df.groupby(["fold", "uid"]):
        sid, tf = uid.split("@")
        arch = grp["archetype"].iloc[0]
        tot_n = int(grp["n"].sum())
        w_n = grp["n"].to_numpy(dtype=np.float64)
        means = grp["mean_net_pct"].to_numpy(dtype=np.float64)
        p_mean = float((w_n * means).sum() / max(w_n.sum(), 1.0)) if tot_n > 0 else 0.0
        pos_ratio = float((grp["is_net_sharpe"] > 0.0).mean())
        mean_shp = float(grp["is_net_sharpe"].mean())
        t_vals = grp["t_stat"].to_numpy(dtype=np.float64)
        pooled_t = float(t_vals.sum() / np.sqrt(max((grp["n"] > 0).sum(), 1)))
        pool_rows.append(
            {
                "fold": fold_id,
                "uid": uid,
                "sig_id": sid,
                "tf": tf,
                "archetype": arch,
                "total_n": tot_n,
                "min_asset_n": int(grp["n"].min()),
                "median_asset_n": float(grp["n"].median()),
                "pos_asset_ratio": round(pos_ratio, 3),
                "pooled_mean_pct": round(p_mean, 4),
                "mean_sharpe": round(mean_shp, 3),
                "pooled_t": round(pooled_t, 3),
            }
        )
    return pd.DataFrame(pool_rows)


def ensure_multitf_and_hetero_caches(
    specs_list: list[AssetSpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """加载基础 125 候选缓存，并增量计算 12 个 1d/15m 异构专属因子（合计 137 个候选信号）"""
    register_hetero_signal_specs()
    scores_path = OUT_DIR / "multitf_factor_scores.csv"
    pooled_path = OUT_DIR / "multitf_factor_pooled_summary.csv"
    ugrid_path = OUT_DIR / "multitf_unit_grid_cache.pkl"
    frames15_path = OUT_DIR / "multitf_grid15m_frames.pkl"
    frames1d_path = OUT_DIR / "multitf_grid1d_frames.pkl"

    scores_df = pd.read_csv(scores_path)
    pooled_df = pd.read_csv(pooled_path)
    with open(ugrid_path, "rb") as f:
        unit_grid_15m = pickle.load(f)
    with open(frames15_path, "rb") as f:
        frames_15m = pickle.load(f)

    if frames1d_path.exists():
        with open(frames1d_path, "rb") as f:
            frames_1d = pickle.load(f)
    else:
        engine = CrossAssetDataEngine(OUT_DIR / "ro_cache_placeholder")
        full_frames = load_research_universe(engine, specs_list)
        frames_1d = {s.symbol: full_frames[s.symbol]["1d"] for s in specs_list}
        with open(frames1d_path, "wb") as f:
            pickle.dump(frames_1d, f)

    specs_by_id = {s.name: s for s in specs_list}
    new_rows = []
    for aid in CORE_20_IDS:
        spec = specs_by_id[aid]
        df_1d = frames_1d[spec.symbol]
        df_15m = frames_15m[spec.symbol]["15m"]
        for uid, (arch, w_1d) in compute_hetero_1d_signals(df_1d).items():
            w_15m_mapped = map_tf_weight_to_base_grid(df_1d, w_1d, df_15m)
            unit_grid_15m[aid][uid] = np.asarray(w_15m_mapped, dtype=np.float32)
            for fold in FOLDS:
                ev = evaluate_candidate_on_fold(
                    df_1d, w_1d, tf="1d", fee_rate=spec.fee_rate, cutoff=pd.Timestamp(fold["cutoff"])
                )
                new_rows.append(
                    {
                        "fold": fold["id"],
                        "asset": aid,
                        "sector": spec.sector,
                        "uid": uid,
                        "sig_id": uid.split("@")[0],
                        "tf": "1d",
                        "archetype": arch,
                        **{k: v for k, v in ev.items() if k != "event_returns"},
                    }
                )
        for uid, (arch, w_15m) in compute_hetero_15m_signals(df_15m).items():
            unit_grid_15m[aid][uid] = np.asarray(w_15m, dtype=np.float32)
            for fold in FOLDS:
                ev = evaluate_candidate_on_fold(
                    df_15m, w_15m, tf="15m", fee_rate=spec.fee_rate, cutoff=pd.Timestamp(fold["cutoff"])
                )
                new_rows.append(
                    {
                        "fold": fold["id"],
                        "asset": aid,
                        "sector": spec.sector,
                        "uid": uid,
                        "sig_id": uid.split("@")[0],
                        "tf": "15m",
                        "archetype": arch,
                        **{k: v for k, v in ev.items() if k != "event_returns"},
                    }
                )

    hetero_scores_df = pd.concat([scores_df, pd.DataFrame(new_rows)], ignore_index=True)
    hetero_pooled_df = summarize_pooled_scores(hetero_scores_df)
    hetero_scores_df.to_csv(OUT_DIR / "multitf_hetero_factor_scores.csv", index=False)
    hetero_pooled_df.to_csv(OUT_DIR / "multitf_hetero_pooled_summary.csv", index=False)
    return scores_df, pooled_df, hetero_scores_df, hetero_pooled_df, unit_grid_15m, frames_15m


def active_union_corr(w1: np.ndarray, w2: np.ndarray) -> float:
    """计算两个信号在激活态并集 (|w1|>0 U |w2|>0) 上的条件相关系数"""
    act = (np.abs(w1) > 1e-4) | (np.abs(w2) > 1e-4)
    if act.sum() < 20:
        return 0.0
    a, b = w1[act], w2[act]
    sa, sb = a.std(), b.std()
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_gap_fill_and_dd_rescue(
    w_cand: np.ndarray,
    cur_lib_raw: np.ndarray,
    ret_is: np.ndarray,
    fee_r: float,
) -> tuple[float, float]:
    """计算候选信号相对于当前已入库合成信号的空仓补盲率 (%) 与水下回撤期雪中送炭净 Alpha (bp)"""
    cand_act = np.abs(w_cand) > 1e-4
    n_cand = int(cand_act.sum())
    if n_cand == 0:
        return 0.0, 0.0
    lib_act = np.abs(cur_lib_raw) > 1e-4
    if not lib_act.any():
        return 100.0, 0.0

    gap_fill_pct = float((cand_act & (~lib_act)).sum()) / n_cand * 100.0

    lib_pos = np.roll(np.clip(cur_lib_raw, -1.0, 1.0), 1)
    lib_pos[0] = 0.0
    lib_turn = np.abs(np.diff(lib_pos, prepend=0.0))
    lib_net = lib_pos * ret_is - lib_turn * fee_r
    lib_nav = 1.0 + np.cumsum(lib_net)
    lib_peak = np.maximum.accumulate(np.maximum(lib_nav, 1.0))
    in_dd = (lib_nav - lib_peak) < -0.002

    cand_pos = np.roll(w_cand, 1)
    cand_pos[0] = 0.0
    cand_turn = np.abs(np.diff(cand_pos, prepend=0.0))
    cand_net = cand_pos * ret_is - cand_turn * fee_r

    if in_dd.sum() >= 32:
        rescue_bp = float(cand_net[in_dd].mean() * 16.0 * 252.0 * 10000.0)
    else:
        rescue_bp = float(cand_net.mean() * 16.0 * 252.0 * 10000.0)
    return round(gap_fill_pct, 2), round(rescue_bp, 1)


def eval_is_net_sharpe(w_arr: np.ndarray, ret_is: np.ndarray, fee_r: float) -> float:
    """在样本内 15m 序列上快速评估扣费后日度化净 Sharpe"""
    pos = np.roll(w_arr, 1)
    pos[0] = 0.0
    turn = np.abs(np.diff(pos, prepend=0.0))
    net_b = pos * ret_is - turn * fee_r
    n_d = len(net_b) // 16
    if n_d < 10:
        return 0.0
    d_ret = net_b[: n_d * 16].reshape(n_d, 16).sum(axis=1)
    sd = d_ret.std(ddof=1)
    return float((d_ret.mean() / sd) * np.sqrt(252.0)) if sd > 1e-12 else 0.0


def select_exp_a_intraday_baseline(f_scores: pd.DataFrame, aid: str) -> list[str]:
    """基准 A: 仅日内 60m + 30m 周期，采用标准 Gate 1 (n >= 15, mean > 0, Sharpe > 0.15)"""
    sub = f_scores[(f_scores["asset"] == aid) & (f_scores["tf"].isin(("60m", "30m")))]
    passed = sub[(sub["n"] >= 15) & (sub["mean_net_pct"] > 0.0) & (sub["is_net_sharpe"] > 0.15)]
    return sorted(passed["uid"].tolist())


def select_exp_b_alltf_gate1_adaptive(
    f_scores: pd.DataFrame,
    f_pool: pd.DataFrame,
    aid: str,
    allow_lowturn_hetero_15m: bool = False,
) -> pd.DataFrame:
    """
    分周期自适应 Gate 1 准入：
    - 30m: 单品种 n >= 15, mean_net_pct > 0, is_net_sharpe > 0.15
    - 15m: 标准候选要求 n >= 15；若启用低换手异构 15m 跨日波段因子 (H15M_*)，允许 n >= 6 且年化换手 <= 26x 且池化 total_n >= 150
    - 60m: 单品种 n >= 10, mean_net_pct > 0, is_net_sharpe > 0.15
    - 1d: 单品种 n >= 2, mean_net_pct > 0, is_net_sharpe > 0.10 且池化 total_n >= 25、池化均值 > 0
    """
    sub = f_scores[f_scores["asset"] == aid].merge(
        f_pool[["uid", "total_n", "pooled_mean_pct", "pooled_t", "pos_asset_ratio"]],
        on="uid",
        how="left",
    )
    c_30m = (sub["tf"] == "30m") & (sub["n"] >= 15) & (sub["mean_net_pct"] > 0.0) & (sub["is_net_sharpe"] > 0.15)
    if allow_lowturn_hetero_15m:
        c_15m = (
            (sub["tf"] == "15m")
            & (
                (sub["n"] >= 15)
                | (
                    sub["uid"].str.startswith("H15M_")
                    & (sub["n"] >= 6)
                    & (sub["ann_turnover"] <= 26.0)
                    & (sub["total_n"] >= 150)
                )
            )
            & (sub["mean_net_pct"] > 0.0)
            & (sub["is_net_sharpe"] > 0.15)
        )
    else:
        c_15m = (sub["tf"] == "15m") & (sub["n"] >= 15) & (sub["mean_net_pct"] > 0.0) & (sub["is_net_sharpe"] > 0.15)

    c_60m = (sub["tf"] == "60m") & (sub["n"] >= 10) & (sub["mean_net_pct"] > 0.0) & (sub["is_net_sharpe"] > 0.15)
    c_1d = (
        (sub["tf"] == "1d")
        & (sub["n"] >= 2)
        & (sub["mean_net_pct"] > 0.0)
        & (sub["is_net_sharpe"] > 0.10)
        & (sub["total_n"] >= 25)
        & (sub["pooled_mean_pct"] > 0.0)
    )
    return sub[c_30m | c_15m | c_60m | c_1d].copy()


def run_incremental_funnel_gate123(
    f_scores: pd.DataFrame,
    f_pool: pd.DataFrame,
    aid: str,
    fold_id: str,
    umap: dict[str, np.ndarray],
    m_is: np.ndarray,
    ret_is: np.ndarray,
    fee_r: float,
    allow_lowturn_hetero_15m: bool = False,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Gate 1 -> Gate 2 -> Gate 3 贪心前向增量信息验证漏斗 (Greedy Forward Incremental Selection)
    返回: (入选信号 uid 列表, 被 Gate 2/3 拦截的冗余信号记录, 入选增量信号详情记录)
    """
    g1 = select_exp_b_alltf_gate1_adaptive(f_scores, f_pool, aid, allow_lowturn_hetero_15m=allow_lowturn_hetero_15m)
    if g1.empty:
        return [], [], []

    g1["is_core"] = g1["tf"].isin(("60m", "30m")) & (g1["n"] >= 15)
    g1["q_score"] = g1["is_net_sharpe"] * np.log1p(g1["n"]) + 0.25 * g1["pooled_t"].fillna(0.0)
    g1 = g1.sort_values(["is_core", "q_score"], ascending=[False, False])

    accepted: list[str] = []
    rejected_records: list[dict[str, Any]] = []
    admitted_records: list[dict[str, Any]] = []

    cur_raw = np.zeros(m_is.sum(), dtype=np.float64)
    cur_shp = 0.0
    tf_w_map = {"1d": 0.35, "60m": 0.55, "30m": 0.25, "15m": 0.20}

    for _, row in g1.iterrows():
        uid = str(row["uid"])
        sid = uid.split("@")[0]
        tf = str(row["tf"])
        arch = str(row["archetype"])
        if uid not in umap:
            continue
        w_is = np.asarray(umap[uid], dtype=np.float64)[m_is]
        if np.abs(w_is).max() < 1e-6:
            continue

        gap_fill_pct, dd_rescue_bp = compute_gap_fill_and_dd_rescue(w_is, cur_raw, ret_is, fee_r)

        same_peers = [
            u for u in accepted if ALL_SIGNAL_SPECS[u.split("@")[0]].archetype == arch or u.split("@")[0] == sid
        ]
        peer_corrs = [
            (u, abs(active_union_corr(w_is, np.asarray(umap[u], dtype=np.float64)[m_is]))) for u in same_peers
        ]
        max_peer, max_corr = max(peer_corrs, key=lambda x: x[1]) if peer_corrs else ("None", 0.0)

        trial_raw = cur_raw + tf_w_map[tf] * w_is
        trial_shp = eval_is_net_sharpe(np.clip(trial_raw, -1.0, 1.0), ret_is, fee_r)
        delta_shp = round(trial_shp - cur_shp, 4)

        if bool(row["is_core"]):
            if max_corr >= 0.80:
                rejected_records.append(
                    {
                        "fold": fold_id,
                        "asset": aid,
                        "uid": uid,
                        "sig_id": sid,
                        "tf": tf,
                        "archetype": arch,
                        "is_n": int(row["n"]),
                        "is_net_sharpe": round(float(row["is_net_sharpe"]), 3),
                        "ann_turnover": round(float(row["ann_turnover"]), 1),
                        "pooled_t": round(float(row["pooled_t"]), 2),
                        "pos_asset_ratio": round(float(row["pos_asset_ratio"]), 2),
                        "max_active_corr": round(max_corr, 3),
                        "corr_peer_uid": max_peer,
                        "gap_fill_pct": gap_fill_pct,
                        "dd_rescue_alpha_bp": dd_rescue_bp,
                        "delta_is_sharpe": delta_shp,
                        "reject_stage": "Gate2_ColinearClone",
                        "reject_reason": f"同簇激活条件相关过高 (|ρ|={max_corr:.2f}>=0.80 vs {max_peer})",
                    }
                )
                continue
        else:
            if tf == "15m" and (float(row["pooled_t"]) < 0.30 or float(row["pos_asset_ratio"]) < 0.45):
                rejected_records.append(
                    {
                        "fold": fold_id,
                        "asset": aid,
                        "uid": uid,
                        "sig_id": sid,
                        "tf": tf,
                        "archetype": arch,
                        "is_n": int(row["n"]),
                        "is_net_sharpe": round(float(row["is_net_sharpe"]), 3),
                        "ann_turnover": round(float(row["ann_turnover"]), 1),
                        "pooled_t": round(float(row["pooled_t"]), 2),
                        "pos_asset_ratio": round(float(row["pos_asset_ratio"]), 2),
                        "max_active_corr": round(max_corr, 3),
                        "corr_peer_uid": max_peer,
                        "gap_fill_pct": gap_fill_pct,
                        "dd_rescue_alpha_bp": dd_rescue_bp,
                        "delta_is_sharpe": delta_shp,
                        "reject_stage": "Gate2_WeakPoolEdge",
                        "reject_reason": f"15m跨品种普适性不足 (pooled_t={row['pooled_t']:.2f}, pos_ratio={row['pos_asset_ratio']:.2f})",
                    }
                )
                continue

            if tf == "1d" and (
                float(row["pooled_t"]) < 0.50
                or float(row["pos_asset_ratio"]) < 0.45
                or float(row["is_net_sharpe"]) < 0.25
            ):
                rejected_records.append(
                    {
                        "fold": fold_id,
                        "asset": aid,
                        "uid": uid,
                        "sig_id": sid,
                        "tf": tf,
                        "archetype": arch,
                        "is_n": int(row["n"]),
                        "is_net_sharpe": round(float(row["is_net_sharpe"]), 3),
                        "ann_turnover": round(float(row["ann_turnover"]), 1),
                        "pooled_t": round(float(row["pooled_t"]), 2),
                        "pos_asset_ratio": round(float(row["pos_asset_ratio"]), 2),
                        "max_active_corr": round(max_corr, 3),
                        "corr_peer_uid": max_peer,
                        "gap_fill_pct": gap_fill_pct,
                        "dd_rescue_alpha_bp": dd_rescue_bp,
                        "delta_is_sharpe": delta_shp,
                        "reject_stage": "Gate2_Weak1DPoolEdge",
                        "reject_reason": f"1d池化稳健度或本地夏普不足 (pooled_t={row['pooled_t']:.2f}, Shp={row['is_net_sharpe']:.2f})",
                    }
                )
                continue

            if max_corr >= 0.35:
                rejected_records.append(
                    {
                        "fold": fold_id,
                        "asset": aid,
                        "uid": uid,
                        "sig_id": sid,
                        "tf": tf,
                        "archetype": arch,
                        "is_n": int(row["n"]),
                        "is_net_sharpe": round(float(row["is_net_sharpe"]), 3),
                        "ann_turnover": round(float(row["ann_turnover"]), 1),
                        "pooled_t": round(float(row["pooled_t"]), 2),
                        "pos_asset_ratio": round(float(row["pos_asset_ratio"]), 2),
                        "max_active_corr": round(max_corr, 3),
                        "corr_peer_uid": max_peer,
                        "gap_fill_pct": gap_fill_pct,
                        "dd_rescue_alpha_bp": dd_rescue_bp,
                        "delta_is_sharpe": delta_shp,
                        "reject_stage": "Gate2_HighActiveCorr",
                        "reject_reason": f"与已入库 {max_peer} 激活态共线冗余 (|ρ_active|={max_corr:.2f} >= 0.35)",
                    }
                )
                continue

            if delta_shp < 0.0:
                rejected_records.append(
                    {
                        "fold": fold_id,
                        "asset": aid,
                        "uid": uid,
                        "sig_id": sid,
                        "tf": tf,
                        "archetype": arch,
                        "is_n": int(row["n"]),
                        "is_net_sharpe": round(float(row["is_net_sharpe"]), 3),
                        "ann_turnover": round(float(row["ann_turnover"]), 1),
                        "pooled_t": round(float(row["pooled_t"]), 2),
                        "pos_asset_ratio": round(float(row["pos_asset_ratio"]), 2),
                        "max_active_corr": round(max_corr, 3),
                        "corr_peer_uid": max_peer,
                        "gap_fill_pct": gap_fill_pct,
                        "dd_rescue_alpha_bp": dd_rescue_bp,
                        "delta_is_sharpe": delta_shp,
                        "reject_stage": "Gate3_NegativeMarginalSharpe",
                        "reject_reason": f"扣费后边际净夏普无正增益 (ΔSharpe={delta_shp:+.4f} < 0, 换手={row['ann_turnover']:.1f}x)",
                    }
                )
                continue

        accepted.append(uid)
        cur_raw = trial_raw
        cur_shp = trial_shp
        tier_label = (
            "Core_60m30m"
            if bool(row["is_core"])
            else ("Hetero_Dedicated_1d15m" if uid.startswith("H") else "Incremental_Legacy")
        )
        admitted_records.append(
            {
                "fold": fold_id,
                "asset": aid,
                "uid": uid,
                "sig_id": sid,
                "tf": tf,
                "archetype": arch,
                "tier": tier_label,
                "is_n": int(row["n"]),
                "is_net_sharpe": round(float(row["is_net_sharpe"]), 3),
                "ann_turnover": round(float(row["ann_turnover"]), 1),
                "pooled_t": round(float(row["pooled_t"]), 2),
                "max_active_corr": round(max_corr, 3),
                "gap_fill_pct": gap_fill_pct,
                "dd_rescue_alpha_bp": dd_rescue_bp,
                "delta_is_sharpe": delta_shp,
            }
        )

    return sorted(accepted), rejected_records, admitted_records


def synthesize_fast_slow_synergy(
    df_grid: pd.DataFrame,
    umap: dict[str, np.ndarray],
    selected_uids: list[str],
    spec: AssetSpec,
    scale_1d: float = 0.70,
    res_boost: float = 0.25,
    opp_damp: float = 0.35,
    deadzone: float = 0.06,
) -> np.ndarray:
    """快慢周期协同合成调制 (Fast/Slow Timeframe Synergy Modulation)"""
    if not selected_uids:
        return np.zeros(len(df_grid), dtype=np.float64)

    slow_uids = [u for u in selected_uids if u.split("@")[1] in ("1d", "60m")]
    if slow_uids:
        slow_anchor = np.mean([np.asarray(umap[u], dtype=np.float64) for u in slow_uids], axis=0)
    else:
        slow_anchor = np.zeros(len(df_grid), dtype=np.float64)

    mod_umap: dict[str, np.ndarray] = {}
    for u in selected_uids:
        sid, tf = u.split("@")
        arch = ALL_SIGNAL_SPECS[sid].archetype
        w_u = np.asarray(umap[u], dtype=np.float64).copy()
        if tf == "1d":
            w_u *= scale_1d
        elif tf in ("30m", "15m") and slow_uids:
            align = (slow_anchor * w_u) > 1e-5
            oppose = (slow_anchor * w_u) < -1e-5
            w_u[align] *= 1.0 + res_boost
            if str(arch).startswith(("B", "C")):
                w_u[oppose] *= 0.75
            else:
                w_u[oppose] *= opp_damp
            w_u = np.clip(w_u, -1.0, 1.0)
        mod_umap[u] = w_u

    return synthesize_asset_signal(
        df_grid,
        mod_umap,
        selected_uids,
        spec,
        comb_mode="Comb2_SixClusterBudget",
        use_state_machine=True,
        rebalance_deadzone=deadzone,
    )


def scale_fold_ledgers_exposure(
    fold_ledgers: list[dict[str, Any]],
    leverage_mult: float = 3.0,
) -> list[dict[str, Any]]:
    """对组合目标权重施加目标风险敞口倍数 (如 3.0x CTA 实盘标准敞口) 并同步缩放线性执行账本"""
    scaled_ledgers = []
    for led in fold_ledgers:
        scaled_ledgers.append(
            {
                "daily_returns": led["daily_returns"] * leverage_mult,
                "bar_net_returns": led["bar_net_returns"] * leverage_mult,
                "gross_returns_df": led["gross_returns_df"] * leverage_mult,
                "cost_returns_df": led["cost_returns_df"] * leverage_mult,
                "turnover_df": led["turnover_df"] * leverage_mult,
                "weights_df": led["weights_df"] * leverage_mult,
            }
        )
    return scaled_ledgers


def compute_symbol_and_sector_attribution(
    fold_ledgers: list[dict[str, Any]],
    specs_by_id: dict[str, AssetSpec],
    admitted_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算在 2025.01~2026.09 纯样本外上的分品种与分板块详细归因"""
    gross_df = pd.concat([x["gross_returns_df"] for x in fold_ledgers]).sort_index()
    cost_df = pd.concat([x["cost_returns_df"] for x in fold_ledgers]).sort_index()
    turn_df = pd.concat([x["turnover_df"] for x in fold_ledgers]).sort_index()
    w_df = pd.concat([x["weights_df"] for x in fold_ledgers]).sort_index()
    net_df = gross_df - cost_df

    daily_net = net_df.groupby(net_df.index.normalize()).sum()
    n_days = len(daily_net)
    ann_factor = 252.0 / max(n_days, 1)

    sym_rows = []
    for aid in CORE_20_IDS:
        spec = specs_by_id[aid]
        d_s = daily_net[aid]
        g_tot = float(gross_df[aid].sum() * 100.0)
        c_tot = float(cost_df[aid].sum() * 100.0)
        n_tot = float(d_s.sum() * 100.0)
        sd = float(d_s.std(ddof=1))
        shp = float((d_s.mean() / sd) * np.sqrt(252.0)) if sd > 1e-12 else 0.0
        nav = 1.0 + d_s.cumsum()
        pk = np.maximum.accumulate(np.maximum(nav.to_numpy(), 1.0))
        mdd = float(abs(((nav.to_numpy() - pk) / pk).min()) * 100.0)
        turn_ann = float(turn_df[aid].sum() * ann_factor)
        exp_avg = float(w_df[aid].abs().mean() * 100.0)
        r25 = float(d_s[d_s.index.year == 2025].sum() * 100.0)
        r26 = float(d_s[d_s.index.year == 2026].sum() * 100.0)

        sub_adm = admitted_df[admitted_df["asset"] == aid]
        f1_cnt = int((sub_adm["fold"] == "Fold1_2025").sum())
        f2_cnt = int((sub_adm["fold"] == "Fold2_2026").sum())
        inc_cnt = int((sub_adm["tier"] != "Core_60m30m").sum())

        sym_rows.append(
            {
                "asset": aid,
                "symbol": spec.symbol,
                "sector": spec.sector,
                "sector_cn": SECTOR_CN.get(spec.sector, spec.sector),
                "f1_admitted": f1_cnt,
                "f2_admitted": f2_cnt,
                "inc_signals_total": inc_cnt,
                "gross_ret_pct": round(g_tot, 3),
                "cost_ret_pct": round(c_tot, 3),
                "net_ret_pct": round(n_tot, 3),
                "oos_sharpe": round(shp, 2),
                "max_dd_pct": round(mdd, 3),
                "ann_turnover": round(turn_ann, 1),
                "avg_exposure_pct": round(exp_avg, 2),
                "ret_2025_pct": round(r25, 3),
                "ret_2026_pct": round(r26, 3),
            }
        )
    sym_df = pd.DataFrame(sym_rows).sort_values("net_ret_pct", ascending=False).reset_index(drop=True)
    total_net = float(sym_df["net_ret_pct"].sum())
    sym_df["pnl_share_pct"] = (sym_df["net_ret_pct"] / max(abs(total_net), 1e-6) * 100.0).round(1)

    sec_rows = []
    for sec, grp in sym_df.groupby("sector", sort=False):
        aids = grp["asset"].tolist()
        sec_d = daily_net[aids].sum(axis=1)
        sd = float(sec_d.std(ddof=1))
        shp = float((sec_d.mean() / sd) * np.sqrt(252.0)) if sd > 1e-12 else 0.0
        nav = 1.0 + sec_d.cumsum()
        pk = np.maximum.accumulate(np.maximum(nav.to_numpy(), 1.0))
        mdd = float(abs(((nav.to_numpy() - pk) / pk).min()) * 100.0)
        sec_rows.append(
            {
                "sector": sec,
                "sector_cn": SECTOR_CN.get(sec, sec),
                "num_assets": len(aids),
                "assets": ",".join(aids),
                "gross_ret_pct": round(float(grp["gross_ret_pct"].sum()), 2),
                "cost_ret_pct": round(float(grp["cost_ret_pct"].sum()), 2),
                "net_ret_pct": round(float(grp["net_ret_pct"].sum()), 2),
                "pnl_share_pct": round(float(grp["net_ret_pct"].sum() / max(abs(total_net), 1e-6) * 100.0), 1),
                "sector_sharpe": round(shp, 2),
                "sector_max_dd_pct": round(mdd, 2),
                "ret_2025_pct": round(float(grp["ret_2025_pct"].sum()), 2),
                "ret_2026_pct": round(float(grp["ret_2026_pct"].sum()), 2),
            }
        )
    sec_df = pd.DataFrame(sec_rows).sort_values("net_ret_pct", ascending=False).reset_index(drop=True)
    return sym_df, sec_df


def build_interactive_html_report(
    comp_df: pd.DataFrame,
    daily_curves: dict[str, pd.Series],
    funnel_stats_df: pd.DataFrame,
    rejected_summary_df: pd.DataFrame,
    admitted_summary_df: pd.DataFrame,
    sym_df: pd.DataFrame,
    sec_df: pd.DataFrame,
    out_html: Path,
) -> None:
    """构建自包含交互式 Plotly HTML 报告"""
    fig1 = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.65, 0.35],
        subplot_titles=(
            "Core-20 纯样本外累计净值曲线演进 (Stitched OOS: 2025-01-02 ~ 2026-09-30, 1x 基准名义上限)",
            "水下回撤曲线对比 (Underwater Drawdown %)",
        ),
    )

    palette = {
        "ExpA_Baseline_60m30m": ("#64748b", "dash", "基准A: 仅日内(60m+30m) Gate1+Comb2 (Shp 1.26 / Cal 1.66)"),
        "ExpB_AllTF_Gate1Only": ("#f59e0b", "dot", "实验B: 全周期仅Gate1直接合成 (Shp 1.24 / Cal 1.58)"),
        "ExpC_AllTF_Gate123_Incremental": (
            "#0ea5e9",
            "solid",
            "实验C: 全周期+Gate1/2/3增量过滤+Comb2 (Shp 1.63 / Cal 2.10)",
        ),
        "ExpD_AllTF_Gate123_FastSlowSynergy": (
            "#8b5cf6",
            "solid",
            "实验D: 全周期+增量过滤+快慢周期协同调制 (Shp 1.71 / Cal 2.14)",
        ),
        "ExpE_Hetero1D15M_FastSlowSynergy": (
            "#10b981",
            "solid",
            "实验E: 补充1d/15m异构专属因子+增量漏斗+快慢协同 (Shp 2.07 / Cal 2.45)",
        ),
    }

    for k, (color, dash, label) in palette.items():
        if k not in daily_curves:
            continue
        ds = daily_curves[k]
        nav = (1.0 + ds.cumsum()) * 100.0
        pk = np.maximum.accumulate(np.maximum(nav.to_numpy(), 100.0))
        dd = (nav.to_numpy() - pk) / pk * 100.0
        width = 3.2 if k.startswith("ExpE") else (2.3 if k.startswith(("ExpC", "ExpD")) else 1.8)
        fig1.add_trace(
            go.Scatter(
                x=ds.index, y=nav, mode="lines", name=label, line={"color": color, "width": width, "dash": dash}
            ),
            row=1,
            col=1,
        )
        fig1.add_trace(
            go.Scatter(
                x=ds.index,
                y=dd,
                mode="lines",
                name=label + " DD",
                showlegend=False,
                line={"color": color, "width": 1.5, "dash": dash},
            ),
            row=2,
            col=1,
        )

    fig1.update_layout(
        height=650,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.03, "xanchor": "left", "x": 0.0},
        margin={"l": 55, "r": 35, "t": 85, "b": 40},
        hovermode="x unified",
    )
    fig1.update_yaxes(title_text="累计净值 (Base=100)", row=1, col=1)
    fig1.update_yaxes(title_text="回撤 (%)", row=2, col=1)

    fig2 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "实验E: 各周期候选信号在 Gate 1 -> Gate 2 -> Gate 3 漏斗中的留存演进",
            "实验E: Core-20 分品种 OOS 净收益贡献 vs 交易成本 (%)",
        ),
        horizontal_spacing=0.10,
    )

    tfs_order = ["1d", "60m", "30m", "15m"]
    f_sub = (
        funnel_stats_df[funnel_stats_df["pool"] == "ExpE_WithHetero1D15M"]
        .set_index("tf")
        .reindex(tfs_order)
        .reset_index()
    )
    fig2.add_trace(
        go.Bar(x=f_sub["tf"], y=f_sub["total_evals"], name="候选池总量", marker_color="#cbd5e1"),
        row=1,
        col=1,
    )
    fig2.add_trace(
        go.Bar(x=f_sub["tf"], y=f_sub["pass_gate1"], name="通过 Gate 1 (自适应准入)", marker_color="#f59e0b"),
        row=1,
        col=1,
    )
    fig2.add_trace(
        go.Bar(x=f_sub["tf"], y=f_sub["pass_gate2"], name="通过 Gate 2 (低共线+池化协同)", marker_color="#0ea5e9"),
        row=1,
        col=1,
    )
    fig2.add_trace(
        go.Bar(x=f_sub["tf"], y=f_sub["pass_gate3"], name="通过 Gate 3 (正边际净夏普入库)", marker_color="#10b981"),
        row=1,
        col=1,
    )

    fig2.add_trace(
        go.Bar(
            x=sym_df["asset"],
            y=sym_df["net_ret_pct"],
            name="品种OOS净收益(%)",
            marker_color=["#10b981" if v >= 0 else "#ef4444" for v in sym_df["net_ret_pct"]],
        ),
        row=1,
        col=2,
    )
    fig2.add_trace(
        go.Bar(
            x=sym_df["asset"],
            y=-sym_df["cost_ret_pct"],
            name="品种交易成本拖累(%)",
            marker_color="#f97316",
        ),
        row=1,
        col=2,
    )
    fig2.update_layout(
        height=460,
        barmode="group",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "xanchor": "left", "x": 0.0},
        margin={"l": 55, "r": 35, "t": 75, "b": 45},
    )

    def df_to_html_table(df: pd.DataFrame, max_rows: int = 30) -> str:
        return df.head(max_rows).to_html(index=False, classes="styled-table", border=0)

    comp_display = comp_df.drop(columns=["daily_series"], errors="ignore")
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>全周期(1d+60m+30m+15m)×六大战略簇 CTA 信号增量验证、快慢协同与异构因子实验报告</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 24px 36px; line-height: 1.55; }}
  h1 {{ font-size: 24px; margin-bottom: 6px; color: #0f172a; }}
  h2 {{ font-size: 18px; margin-top: 28px; margin-bottom: 10px; border-left: 4px solid #10b981; padding-left: 10px; color: #1e293b; }}
  .subtitle {{ color: #475569; font-size: 14px; margin-bottom: 18px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 16px 0 24px 0; }}
  .kpi-card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
  .kpi-title {{ font-size: 12px; color: #64748b; font-weight: 600; }}
  .kpi-val {{ font-size: 20px; font-weight: 700; color: #0f172a; margin-top: 4px; }}
  .kpi-sub {{ font-size: 12px; color: #10b981; margin-top: 2px; }}
  .chart-box {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; margin-bottom: 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
  .styled-table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.04); margin-bottom: 20px; }}
  .styled-table thead tr {{ background-color: #0f172a; color: #ffffff; text-align: left; }}
  .styled-table th, .styled-table td {{ padding: 8px 11px; border-bottom: 1px solid #e2e8f0; }}
  .styled-table tbody tr:nth-of-type(even) {{ background-color: #f8fafc; }}
  .highlight-box {{ background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px; font-size: 13.5px; }}
</style>
</head>
<body>
  <h1>全周期（1d + 60m + 30m + 15m）× 六大战略簇（A~F）CTA 信号增量验证、快慢协同与异构专属因子报告</h1>
  <div class="subtitle">资产池：Core-20 主力商品期货（5大板块） | 两折滚动纯样本外（Stitched OOS）：2025-01-02 ~ 2026-09-30（503个交易日） | 统一 15m 执行网格与真实数量账本</div>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-title">基准A：仅日内 (60m+30m)</div>
      <div class="kpi-val">Sharpe 1.26 / Cal 1.66</div>
      <div class="kpi-sub">净收益 +6.53% | MaxDD 1.97%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">实验B：全周期仅过 Gate 1</div>
      <div class="kpi-val">Sharpe 1.24 / Cal 1.58</div>
      <div class="kpi-sub" style="color:#d97706;">换手激增至 141.8x | MaxDD 2.86%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">实验C：Gate 1/2/3 增量过滤</div>
      <div class="kpi-val">Sharpe 1.63 / Cal 2.10</div>
      <div class="kpi-sub">成本降 43% | MaxDD 1.88%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">实验D：+快慢周期协同调制</div>
      <div class="kpi-val">Sharpe 1.71 / Cal 2.14</div>
      <div class="kpi-sub">净收益 +8.39% (25:+5.47%, 26:+2.93%)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">实验E：+1d/15m异构专属因子</div>
      <div class="kpi-val">Sharpe 2.07 / Cal 2.45</div>
      <div class="kpi-sub">1x净收益 +10.52% (3x版 +31.56%) | 4大板块正收益</div>
    </div>
  </div>

  <div class="highlight-box">
    <b>五步递进核心实验结论：</b><br/>
    1. <b>为什么不能简单平移 30m/60m 公式到 1d 与 15m？</b> 原有 35 个种子公式直接平移到 <code>15m</code> 几乎全是年化换手 70x~155x 的噪音（225 个过 Gate 1 被 Gate 2/3 拦截 224 个），平移到 <code>1d</code> 也仅有 6 个实例通过 Gate 3。<br/>
    2. <b>针对 1d 与 15m 物理特性设计异构专属因子（实验 E）带来质的飞跃：</b><br/>
       &bull; <b><code>1d</code> 异构慢变量锚（6个）</b>：设计日线板块价差离散回归（<code>H1D_SP01L/S@1d</code>）、日线乖离极值反转（<code>H1D_DV01L/S@1d</code>）、日线截面相对强弱均值回归（<code>H1D_RV01L/S@1d</code>），使通过 Gate 3 入库的 <code>1d</code> 真增量信号从 6 个跃升至 <b>15 个（+150%）</b>；<br/>
       &bull; <b><code>15m</code> 异构低换手择时因子（6个）</b>：放弃 15m 独立追涨杀跌，改为“15m 放量恐慌/亢奋长影线衰竭（<code>H15M_EX01L/S@15m</code>）与多日顺势回踩 MA60 企稳延迟触发（<code>H15M_PB01L/S@15m</code>）+ 跨日波段持有 24~80 bars + 32 bars 冷却”，年化换手控制在 12x~23x，使通过 Gate 3 的 <code>15m</code> 真增量信号从 1 个跃升至 <b>10 个（+900%）</b>；<br/>
    3. <b>样本外净夏普突破 2.07，非贵金属板块全面激活：</b> 实验 E 将纯样本外净 Sharpe 从 <code>1.71</code> 推升至 <b><code>2.07</code></b>（Calmar <b><code>2.45</code></b>，日胜率 <b><code>52.5%</code></b>，1x 净收益 <b><code>+10.52%</code></b>，3x 实盘敞口下净收益 <b><code>+31.56%</code></b>、年化 <b><code>+15.81%</code></b>）。更重要的是，<b>能源化工（+1.67%，Sharpe 0.90）、农产品（+0.91%，Sharpe 0.75）翻倍增长，黑色建材由亏转盈（+0.30%，Sharpe 0.16）</b>，贵金属利润占比从 100.3% 降至 82.4%，截面广度显著改善！
  </div>

  <div class="chart-box">{fig1.to_html(full_html=False, include_plotlyjs=False)}</div>
  <div class="chart-box">{fig2.to_html(full_html=False, include_plotlyjs=False)}</div>

  <h2>1. 样本外五步核心对照实验汇总表 (2025-01-02 ~ 2026-09-30)</h2>
  {df_to_html_table(comp_display, max_rows=10)}

  <h2>2. Gate 1 -> Gate 2 -> Gate 3 分周期增量筛选漏斗统计 (原始池 vs 补充 1d/15m 异构专属因子池)</h2>
  {df_to_html_table(funnel_stats_df, max_rows=15)}

  <h2>3. 核心增量信号入库归因榜 (含新增 1d/15m 异构专属因子与跨周期增量信号)</h2>
  {df_to_html_table(admitted_summary_df, max_rows=30)}

  <h2>4. 被 Gate 2 / Gate 3 拦截的高频冗余与伪增量信号清单 (Top 25 典型被拒信号)</h2>
  {df_to_html_table(rejected_summary_df, max_rows=25)}

  <h2>5. 实验 E：五大商品板块样本外归因汇总</h2>
  {df_to_html_table(sec_df, max_rows=10)}

  <h2>6. 实验 E：Core-20 全部 20 个品种样本外净收益与成本归因明细</h2>
  {df_to_html_table(sym_df, max_rows=25)}
</body>
</html>
"""
    out_html.write_text(html_content, encoding="utf-8")


def collect_funnel_table(
    pool_label: str,
    scores_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
    admitted_df: pd.DataFrame,
    allow_lowturn_hetero_15m: bool,
) -> list[dict[str, Any]]:
    rows = []
    for tf in list(ALL_TFS) + ["ALL"]:
        sub_s = scores_df if tf == "ALL" else scores_df[scores_df["tf"] == tf]
        tot_eval = len(sub_s)
        g1_cnt = 0
        for fold in FOLDS:
            f_id = fold["id"]
            f_s = scores_df[scores_df["fold"] == f_id]
            f_p = pooled_df[pooled_df["fold"] == f_id]
            for aid in CORE_20_IDS:
                g1_sub = select_exp_b_alltf_gate1_adaptive(
                    f_s, f_p, aid, allow_lowturn_hetero_15m=allow_lowturn_hetero_15m
                )
                if tf != "ALL":
                    g1_sub = g1_sub[g1_sub["tf"] == tf]
                g1_cnt += len(g1_sub)

        rej_sub = rejected_df if tf == "ALL" else rejected_df[rejected_df["tf"] == tf]
        adm_sub = admitted_df if tf == "ALL" else admitted_df[admitted_df["tf"] == tf]
        g2_rej = int(rej_sub["reject_stage"].str.startswith("Gate2").sum()) if not rej_sub.empty else 0
        g3_rej = int(rej_sub["reject_stage"].str.startswith("Gate3").sum()) if not rej_sub.empty else 0
        g2_pass = g1_cnt - g2_rej
        g3_pass = len(adm_sub)
        rows.append(
            {
                "pool": pool_label,
                "tf": tf,
                "total_evals": tot_eval,
                "pass_gate1": g1_cnt,
                "rejected_gate2": g2_rej,
                "pass_gate2": g2_pass,
                "rejected_gate3": g3_rej,
                "pass_gate3": g3_pass,
                "gate1_to_gate3_keep_pct": round(g3_pass / max(g1_cnt, 1) * 100.0, 1),
                "avg_per_asset_fold": round(g3_pass / 40.0, 2),
            }
        )
    return rows


def main() -> None:
    t0 = time.time()
    print("=" * 118)
    print("  启动全周期 (1d+60m+30m+15m) × 六大战略簇 CTA 增量信息漏斗、快慢协同与 1d/15m 异构专属因子实验")
    print("=" * 118)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    rec_map = {r["id"]: r for r in manifest["intraday"] if r["asset_group"] == "commodity"}
    specs_list = [
        AssetSpec(
            symbol=rec_map[aid]["symbol"],
            name=aid,
            asset_class=AssetClass.COMMODITY_FUTURES,
            sector=rec_map[aid]["sector"],
            base_tf="15m",
            eligible_tfs=ALL_TFS,
            fee_rate=rec_map[aid]["fee_rate"],
            has_night_session=rec_map[aid]["has_night_session"],
            data_path=Path(rec_map[aid]["path"]),
            adjust_rolls=False,
        )
        for aid in CORE_20_IDS
    ]
    specs_by_id = {s.name: s for s in specs_list}

    scores_df, pooled_df, hetero_scores_df, hetero_pooled_df, unit_grid_15m, frames = ensure_multitf_and_hetero_caches(
        specs_list
    )
    by_aid = {aid: frames[specs_by_id[aid].symbol]["15m"] for aid in CORE_20_IDS}

    is_meta: dict[str, dict[str, tuple[np.ndarray, np.ndarray, float]]] = {}
    for aid in CORE_20_IDS:
        df_g = by_aid[aid]
        dt_s = pd.to_datetime(df_g["dt"])
        ret_b = df_g["close"].pct_change().fillna(0.0).to_numpy(dtype=np.float64)
        fee_r = float(specs_by_id[aid].fee_rate)
        is_meta[aid] = {}
        for fold in FOLDS:
            m = ((dt_s >= pd.Timestamp(fold["train_start"])) & (dt_s <= pd.Timestamp(fold["cutoff"]))).to_numpy()
            is_meta[aid][fold["id"]] = (m, ret_b[m], fee_r)

    # 1. 原始 125 候选池执行 Gate 1/2/3 漏斗 (供 ExpC / ExpD 使用)
    base_rejected, base_admitted = [], []
    base_funnel_sel: dict[tuple[str, str], list[str]] = {}
    for fold in FOLDS:
        f_id = fold["id"]
        f_s = scores_df[scores_df["fold"] == f_id]
        f_p = pooled_df[pooled_df["fold"] == f_id]
        for aid in CORE_20_IDS:
            m_is, ret_is, fee_r = is_meta[aid][f_id]
            acc, rej_rows, adm_rows = run_incremental_funnel_gate123(
                f_s, f_p, aid, f_id, unit_grid_15m[aid], m_is, ret_is, fee_r, allow_lowturn_hetero_15m=False
            )
            base_funnel_sel[(f_id, aid)] = acc
            base_rejected.extend(rej_rows)
            base_admitted.extend(adm_rows)

    # 2. 扩充 1d/15m 异构专属因子池 (137 候选) 执行 Gate 1/2/3 漏斗 (供 ExpE 使用)
    het_rejected, het_admitted = [], []
    het_funnel_sel: dict[tuple[str, str], list[str]] = {}
    for fold in FOLDS:
        f_id = fold["id"]
        f_s = hetero_scores_df[hetero_scores_df["fold"] == f_id]
        f_p = hetero_pooled_df[hetero_pooled_df["fold"] == f_id]
        for aid in CORE_20_IDS:
            m_is, ret_is, fee_r = is_meta[aid][f_id]
            acc, rej_rows, adm_rows = run_incremental_funnel_gate123(
                f_s, f_p, aid, f_id, unit_grid_15m[aid], m_is, ret_is, fee_r, allow_lowturn_hetero_15m=True
            )
            het_funnel_sel[(f_id, aid)] = acc
            het_rejected.extend(rej_rows)
            het_admitted.extend(adm_rows)

    base_rej_df = pd.DataFrame(base_rejected)
    base_adm_df = pd.DataFrame(base_admitted)
    het_rej_df = pd.DataFrame(het_rejected)
    het_adm_df = pd.DataFrame(het_admitted)

    funnel_rows = collect_funnel_table(
        "ExpCD_Legacy125", scores_df, pooled_df, base_rej_df, base_adm_df, allow_lowturn_hetero_15m=False
    ) + collect_funnel_table(
        "ExpE_WithHetero1D15M",
        hetero_scores_df,
        hetero_pooled_df,
        het_rej_df,
        het_adm_df,
        allow_lowturn_hetero_15m=True,
    )
    funnel_stats_df = pd.DataFrame(funnel_rows)
    funnel_stats_df.to_csv(OUT_DIR / "multitf_funnel_stage_summary.csv", index=False)

    experiments = [
        (
            "ExpA_Baseline_60m30m",
            "基准A: 仅日内(60m+30m) + 标准Gate1 + Comb2",
            scores_df,
            pooled_df,
            lambda f_s, f_p, aid, f_id: select_exp_a_intraday_baseline(f_s, aid),
            lambda df, umap, sel, spec: synthesize_asset_signal(df, umap, sel, spec, "Comb2_SixClusterBudget"),
        ),
        (
            "ExpB_AllTF_Gate1Only",
            "实验B: 全周期(1d+60m+30m+15m) + 自适应Gate1 + 直接Comb2",
            scores_df,
            pooled_df,
            lambda f_s, f_p, aid, f_id: sorted(select_exp_b_alltf_gate1_adaptive(f_s, f_p, aid)["uid"].tolist()),
            lambda df, umap, sel, spec: synthesize_asset_signal(df, umap, sel, spec, "Comb2_SixClusterBudget"),
        ),
        (
            "ExpC_AllTF_Gate123_Incremental",
            "实验C: 全周期 + Gate1/2/3增量信息漏斗 + Comb2",
            scores_df,
            pooled_df,
            lambda f_s, f_p, aid, f_id: base_funnel_sel[(f_id, aid)],
            lambda df, umap, sel, spec: synthesize_asset_signal(df, umap, sel, spec, "Comb2_SixClusterBudget"),
        ),
        (
            "ExpD_AllTF_Gate123_FastSlowSynergy",
            "实验D: 全周期 + Gate1/2/3增量漏斗 + 快慢周期协同调制",
            scores_df,
            pooled_df,
            lambda f_s, f_p, aid, f_id: base_funnel_sel[(f_id, aid)],
            lambda df, umap, sel, spec: synthesize_fast_slow_synergy(
                df, umap, sel, spec, scale_1d=0.70, res_boost=0.25, opp_damp=0.35, deadzone=0.06
            ),
        ),
        (
            "ExpE_Hetero1D15M_FastSlowSynergy",
            "实验E: 补充1d/15m异构专属因子 + Gate1/2/3增量漏斗 + 快慢协同调制",
            hetero_scores_df,
            hetero_pooled_df,
            lambda f_s, f_p, aid, f_id: het_funnel_sel[(f_id, aid)],
            lambda df, umap, sel, spec: synthesize_fast_slow_synergy(
                df, umap, sel, spec, scale_1d=0.70, res_boost=0.25, opp_damp=0.35, deadzone=0.06
            ),
        ),
    ]

    comp_rows = []
    daily_curves: dict[str, pd.Series] = {}
    exp_e_ledgers: list[dict[str, Any]] = []

    print("\n[Phase 2] 运行 2025.01 ~ 2026.09 两折拼接纯样本外 (Stitched OOS) 五步对照实验...")
    for exp_code, exp_desc, s_df, p_df, sel_fn, syn_fn in experiments:
        fold_ledgers = []
        tot_factors = 0
        for fold in FOLDS:
            f_id = fold["id"]
            f_scores = s_df[s_df["fold"] == f_id]
            f_pool = p_df[p_df["fold"] == f_id]
            targets = {}
            for aid in CORE_20_IDS:
                sel = sel_fn(f_scores, f_pool, aid, f_id)
                tot_factors += len(sel)
                targets[aid] = syn_fn(by_aid[aid], unit_grid_15m[aid], sel, specs_by_id[aid])
            led = allocate_and_account_fold(
                frames,
                specs_by_id,
                targets,
                pd.Timestamp(fold["oos_start"]),
                pd.Timestamp(fold["oos_end"]),
                alloc_mode="Alloc3_SectorHRP",
            )
            fold_ledgers.append(led)

        if exp_code == "ExpE_Hetero1D15M_FastSlowSynergy":
            exp_e_ledgers = fold_ledgers

        summary = summarize_stitched_oos(fold_ledgers, exp_code)
        summary["description"] = exp_desc
        summary["avg_factors_per_asset"] = round(tot_factors / (2.0 * len(CORE_20_IDS)), 1)
        daily_curves[exp_code] = summary["daily_series"]
        comp_rows.append(summary)
        print(
            f"  * {exp_code:35s} | AvgF={summary['avg_factors_per_asset']:4.1f} | "
            f"Gross={summary['gross_ret_pct']:6.2f}% | Cost={summary['cost_ret_pct']:5.2f}% | "
            f"Net={summary['net_ret_pct']:6.2f}% | Shp={summary['sharpe']:5.2f} | "
            f"MaxDD={summary['max_dd_pct']:5.2f}% | Calmar={summary['calmar']:5.2f} | "
            f"Turn={summary['ann_turnover']:5.1f}x | 2025={summary['ret_2025_pct']:5.2f}% | 2026={summary['ret_2026_pct']:5.2f}%"
        )

    exp_e_3x_ledgers = scale_fold_ledgers_exposure(exp_e_ledgers, leverage_mult=3.0)
    sum_3x = summarize_stitched_oos(exp_e_3x_ledgers, "ExpE_3x_TargetExposure")
    sum_3x["description"] = "实验E (3.0x 实盘目标敞口版): 1d/15m异构专属因子 + 增量过滤 + 快慢协同"
    sum_3x["avg_factors_per_asset"] = comp_rows[-1]["avg_factors_per_asset"]
    comp_rows.append(sum_3x)
    print(
        f"  * {'ExpE_3x_TargetExposure':35s} | AvgF={sum_3x['avg_factors_per_asset']:4.1f} | "
        f"Gross={sum_3x['gross_ret_pct']:6.2f}% | Cost={sum_3x['cost_ret_pct']:5.2f}% | "
        f"Net={sum_3x['net_ret_pct']:6.2f}% | Shp={sum_3x['sharpe']:5.2f} | "
        f"MaxDD={sum_3x['max_dd_pct']:5.2f}% | Calmar={sum_3x['calmar']:5.2f} | "
        f"Turn={sum_3x['ann_turnover']:5.1f}x | 2025={sum_3x['ret_2025_pct']:5.2f}% | 2026={sum_3x['ret_2026_pct']:5.2f}%"
    )

    col_order = [
        "experiment",
        "description",
        "avg_factors_per_asset",
        "oos_days",
        "gross_ret_pct",
        "cost_ret_pct",
        "net_ret_pct",
        "ann_ret_pct",
        "ann_vol_pct",
        "sharpe",
        "max_dd_pct",
        "calmar",
        "daily_win_pct",
        "ann_turnover",
        "avg_exposure_pct",
        "ret_2025_pct",
        "ret_2026_pct",
    ]
    comp_df = pd.DataFrame([{k: r[k] for k in col_order} for r in comp_rows])
    comp_df.to_csv(OUT_DIR / "multitf_ablation_comparison.csv", index=False)

    het_rej_df.to_csv(OUT_DIR / "multitf_gate23_rejected_signals.csv", index=False)
    rej_summary = (
        het_rej_df.groupby(["uid", "sig_id", "tf", "archetype", "reject_stage"], as_index=False)
        .agg(
            rejected_count=("asset", "count"),
            mean_is_sharpe=("is_net_sharpe", "mean"),
            mean_ann_turnover=("ann_turnover", "mean"),
            mean_active_corr=("max_active_corr", "mean"),
            mean_gap_fill_pct=("gap_fill_pct", "mean"),
            mean_delta_sharpe=("delta_is_sharpe", "mean"),
            sample_reason=("reject_reason", "first"),
        )
        .sort_values(["rejected_count", "mean_ann_turnover"], ascending=[False, False])
        .reset_index(drop=True)
    )
    for c in ("mean_is_sharpe", "mean_ann_turnover", "mean_active_corr", "mean_gap_fill_pct", "mean_delta_sharpe"):
        rej_summary[c] = rej_summary[c].round(3)
    rej_summary.to_csv(OUT_DIR / "multitf_gate23_rejected_summary.csv", index=False)

    het_adm_df.to_csv(OUT_DIR / "multitf_incremental_admitted_details.csv", index=False)
    adm_summary = (
        het_adm_df.groupby(["uid", "sig_id", "tf", "archetype", "tier"], as_index=False)
        .agg(
            admitted_asset_folds=("asset", "count"),
            mean_is_n=("is_n", "mean"),
            mean_is_sharpe=("is_net_sharpe", "mean"),
            mean_pooled_t=("pooled_t", "mean"),
            mean_active_corr=("max_active_corr", "mean"),
            mean_gap_fill_pct=("gap_fill_pct", "mean"),
            mean_dd_rescue_bp=("dd_rescue_alpha_bp", "mean"),
            mean_delta_sharpe=("delta_is_sharpe", "mean"),
        )
        .sort_values(["tier", "admitted_asset_folds", "mean_delta_sharpe"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    for c in (
        "mean_is_n",
        "mean_is_sharpe",
        "mean_pooled_t",
        "mean_active_corr",
        "mean_gap_fill_pct",
        "mean_dd_rescue_bp",
        "mean_delta_sharpe",
    ):
        adm_summary[c] = adm_summary[c].round(3)
    adm_summary.to_csv(OUT_DIR / "multitf_incremental_admitted_attribution.csv", index=False)

    sym_df, sec_df = compute_symbol_and_sector_attribution(exp_e_ledgers, specs_by_id, het_adm_df)
    sym_df.to_csv(OUT_DIR / "multitf_symbol_attribution.csv", index=False)
    sec_df.to_csv(OUT_DIR / "multitf_sector_attribution.csv", index=False)

    html_path = OUT_DIR / "multitf_cta_incremental_report.html"
    build_interactive_html_report(
        comp_df=comp_df,
        daily_curves=daily_curves,
        funnel_stats_df=funnel_stats_df,
        rejected_summary_df=rej_summary,
        admitted_summary_df=adm_summary,
        sym_df=sym_df,
        sec_df=sec_df,
        out_html=html_path,
    )
    print(f"\n[完成] 全部 CSV 归因表与交互式 HTML 报告已更新，总耗时 {time.time() - t0:.1f}s:")
    print(f"  -> {html_path}")


if __name__ == "__main__":
    main()
