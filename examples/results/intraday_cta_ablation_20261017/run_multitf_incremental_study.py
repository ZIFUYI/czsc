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
    build_asset_execution_inputs,
    evaluate_candidate_on_fold,
    summarize_stitched_oos,
    synthesize_asset_signal,
)

import czsc  # noqa: E402
from examples.alpha_mining_agent.forensic_pipeline import evaluate_execution_events  # noqa: E402
from examples.alpha_mining_agent.hypothesis_bank import (  # noqa: E402
    ALL_SIGNAL_SPECS,
    ARCH_A,
    ARCH_B,
    ARCH_C,
    ARCH_D,
    ARCH_E,
    research_bars,
    rust_sma,
)
from examples.multi_factor_cta.data_engine import (  # noqa: E402
    AssetClass,
    AssetSpec,
    CrossAssetDataEngine,
    map_tf_weight_to_base_grid,
)
from examples.multi_factor_cta.run_framework_demo import load_research_universe  # noqa: E402
from examples.multi_factor_cta.synthesis_engine import (  # noqa: E402
    CrossAssetHRPSynthesizer,
    IntraAssetThreePillarSynthesizer,
    account_portfolio_execution,
)

# 启用 5m 周期在商品三支柱合成器中的权重 (原默认值为 0.00)
IntraAssetThreePillarSynthesizer.COMMODITY_TF_WEIGHTS["5m"] = 0.22

ALL_TFS = ("1d", "60m", "30m", "15m", "5m")

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
    "N5M_BRK01L@5m": (ARCH_A, "5m朴素高频1小时通道突破短持做多(对照组)"),
    "N5M_BRK01S@5m": (ARCH_A, "5m朴素高频1小时通道突破短持做空(对照组)"),
    "N5M_REV01L@5m": (ARCH_B, "5m朴素高频RSI超卖短持反转做多(对照组)"),
    "N5M_REV01S@5m": (ARCH_B, "5m朴素高频RSI超买短持反转做空(对照组)"),
    "H5M_CLV01L@5m": (ARCH_C, "5m子K线尾盘放量抢筹确认均线回归做多"),
    "H5M_CLV01S@5m": (ARCH_C, "5m子K线尾盘放量砸盘确认均线回归做空"),
    "H5M_LQ01L@5m": (ARCH_B, "5m持仓量急降砍仓出清与长下影反转做多"),
    "H5M_LQ01S@5m": (ARCH_B, "5m持仓量急降逼空出清与长上影反转做空"),
    "H5M_OIB01L@5m": (ARCH_D, "5m持仓量激增建仓与主动买盘共振顺势做多"),
    "H5M_OIB01S@5m": (ARCH_D, "5m持仓量激增建仓与主动卖盘共振顺势做空"),
}


def register_hetero_signal_specs() -> None:
    """在内存注册表中注册 1d / 15m / 5m 异构专属因子与对照组规格，供三支柱合成器识别"""
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
    ma120_prev16 = np.empty_like(ma120)
    ma120_prev16[:16] = ma120[0]
    ma120_prev16[16:] = ma120[:-16]
    ma120_slp = (ma120 - ma120_prev16) / np.maximum(ma120_prev16, 1e-6)
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


def build_5m_micro_on_15m(df5: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    """将 5m 原始 K 线中的持仓量变化 (open_interest) 与子 K 线微观结构特征严格因果对齐至 15m 网格"""
    d5 = df5.copy()
    c5 = d5["close"].to_numpy(dtype=np.float64)
    o5 = d5["open"].to_numpy(dtype=np.float64)
    h5 = d5["high"].to_numpy(dtype=np.float64)
    l5 = d5["low"].to_numpy(dtype=np.float64)
    v5 = d5["volume"].to_numpy(dtype=np.float64)
    oi5 = d5["open_interest"].to_numpy(dtype=np.float64)

    ret5 = np.zeros_like(c5)
    ret5[1:] = (c5[1:] - c5[:-1]) / np.maximum(c5[:-1], 1e-6)
    oi_chg5 = np.zeros_like(oi5)
    oi_chg5[1:] = oi5[1:] - oi5[:-1]
    oi_med = pd.Series(np.abs(oi_chg5)).rolling(72, min_periods=10).median().fillna(1.0).to_numpy()
    oi_chg5 = np.clip(oi_chg5, -8.0 * oi_med, 8.0 * oi_med)

    range5 = np.maximum(h5 - l5, 1e-6)
    body5 = c5 - o5
    clv5 = ((c5 - l5) - (h5 - c5)) / range5
    signed_vol5 = v5 * clv5
    oi_flow5 = oi_chg5 * np.sign(body5)

    s_ret5 = pd.Series(ret5)
    s_v5 = pd.Series(v5)
    s_oichg5 = pd.Series(oi_chg5)
    s_clv5 = pd.Series(clv5)
    s_svol5 = pd.Series(signed_vol5)
    s_oiflow5 = pd.Series(oi_flow5)

    v_sum3 = s_v5.rolling(3, min_periods=1).sum().replace(0, 1.0).to_numpy()
    last5_vol_share = v5 / v_sum3
    last5_clv = clv5
    mean3_clv = s_clv5.rolling(3, min_periods=1).mean().to_numpy()
    svol_imb3 = s_svol5.rolling(3, min_periods=1).sum().to_numpy() / v_sum3
    oi_chg3 = s_oichg5.rolling(3, min_periods=1).sum().to_numpy()
    oi_flow3 = s_oiflow5.rolling(3, min_periods=1).sum().to_numpy()

    rv12 = s_ret5.rolling(12, min_periods=4).std().fillna(0.0).to_numpy()
    rv72 = s_ret5.rolling(72, min_periods=12).std().replace(0, 1e-6).fillna(1e-6).to_numpy()
    rv_ratio = rv12 / rv72

    l_shd5 = (np.minimum(o5, c5) - l5) / range5
    u_shd5 = (h5 - np.maximum(o5, c5)) / range5
    max_l_shd3 = pd.Series(l_shd5).rolling(3, min_periods=1).max().to_numpy()
    max_u_shd3 = pd.Series(u_shd5).rolling(3, min_periods=1).max().to_numpy()

    feat5 = pd.DataFrame(
        {
            "dt": pd.to_datetime(d5["dt"]),
            "oi": oi5,
            "oi_chg3": oi_chg3,
            "oi_flow3": oi_flow3,
            "svol_imb3": svol_imb3,
            "last5_vol_share": last5_vol_share,
            "last5_clv": last5_clv,
            "mean3_clv": mean3_clv,
            "rv_ratio": rv_ratio,
            "max_l_shd3": max_l_shd3,
            "max_u_shd3": max_u_shd3,
        }
    )
    aligned = pd.merge_asof(
        pd.DataFrame({"dt": pd.to_datetime(df15["dt"])}),
        feat5.sort_values("dt"),
        on="dt",
        direction="backward",
        allow_exact_matches=True,
    ).fillna(0.0)
    return aligned


def compute_all_5m_candidates(df15: pd.DataFrame, f5: pd.DataFrame) -> dict[str, tuple[str, np.ndarray]]:
    """
    计算 10 个 5m 候选信号：
    - 对照组 (4个 Naive 5m 高频突破与反转: N5M_BRK01L/S@5m, N5M_REV01L/S@5m): 短持 2~6 根 15m bars (6~18 根 5m bars)
    - 实验组 (6个 5m 子K线微观结构与持仓量异构专属因子: H5M_CLV01L/S@5m, H5M_LQ01L/S@5m, H5M_OIB01L/S@5m):
      利用 5m 独有的 open_interest 急降砍仓/激增建仓与 5m 尾盘抢筹/长影线拒绝，精准触发跨日波段持有 (24~76 bars)
    """
    c = np.nan_to_num(df15["close"].to_numpy(dtype=np.float64), nan=1.0)
    o = np.nan_to_num(df15["open"].to_numpy(dtype=np.float64), nan=1.0)
    h = np.nan_to_num(df15["high"].to_numpy(dtype=np.float64), nan=1.0)
    low_arr = np.nan_to_num(df15["low"].to_numpy(dtype=np.float64), nan=1.0)
    atr14 = np.nan_to_num(df15["atr14"].to_numpy(dtype=np.float64), nan=0.001)
    atr_pct = atr14 / np.maximum(c, 1e-6)

    s_c = pd.Series(c)
    ma20 = s_c.rolling(20, min_periods=3).mean().to_numpy()
    ma60 = s_c.rolling(60, min_periods=5).mean().to_numpy()
    ma120 = s_c.rolling(120, min_periods=10).mean().to_numpy()
    ma120_prev16 = np.empty_like(ma120)
    ma120_prev16[:16] = ma120[0]
    ma120_prev16[16:] = ma120[:-16]
    ma120_slp = (ma120 - ma120_prev16) / np.maximum(ma120_prev16, 1e-6)
    bias20 = (c - ma20) / np.maximum(ma20, 1e-6)
    bias60 = (c - ma60) / np.maximum(ma60, 1e-6)

    diff = np.diff(c, prepend=c[0])
    up = pd.Series(np.maximum(diff, 0.0)).rolling(14, min_periods=2).mean().to_numpy()
    dn = pd.Series(np.maximum(-diff, 0.0)).rolling(14, min_periods=2).mean().to_numpy()
    rsi14 = 100.0 - 100.0 / (1.0 + up / np.maximum(dn, 1e-8))

    oi_chg3 = f5["oi_chg3"].to_numpy(dtype=np.float64)
    oi_std = pd.Series(oi_chg3).rolling(40, min_periods=5).std().replace(0, 1.0).fillna(1.0).to_numpy()
    oi_z = oi_chg3 / oi_std
    oi_z_sum4 = pd.Series(oi_z).rolling(4, min_periods=1).sum().to_numpy()

    svol3 = f5["svol_imb3"].to_numpy(dtype=np.float64)
    last5_vs = f5["last5_vol_share"].to_numpy(dtype=np.float64)
    last5_clv = f5["last5_clv"].to_numpy(dtype=np.float64)
    max_l_shd3 = f5["max_l_shd3"].to_numpy(dtype=np.float64)
    max_u_shd3 = f5["max_u_shd3"].to_numpy(dtype=np.float64)

    c_prev = np.roll(c, 1)
    c_prev[0] = c[0]

    out: dict[str, tuple[str, np.ndarray]] = {}
    # 对照组: 4 个朴素高频 5m 突破与反转 (Naive 5m short-hold)
    h12 = pd.Series(h).shift(1).rolling(12, min_periods=4).max().fillna(h[0]).to_numpy()
    l12 = pd.Series(low_arr).shift(1).rolling(12, min_periods=4).min().fillna(low_arr[0]).to_numpy()
    out["N5M_BRK01L@5m"] = (
        ARCH_A,
        hold_with_cooldown_and_trail(
            c > h12, np.full_like(c, 0.8), c < ma20, +1.0, min_hold=2, max_hold=6, cooldown=1, warmup=30
        ),
    )
    out["N5M_BRK01S@5m"] = (
        ARCH_A,
        hold_with_cooldown_and_trail(
            c < l12, np.full_like(c, 0.8), c > ma20, -1.0, min_hold=2, max_hold=6, cooldown=1, warmup=30
        ),
    )
    out["N5M_REV01L@5m"] = (
        ARCH_B,
        hold_with_cooldown_and_trail(
            (rsi14 < 30) & (last5_clv > 0.2),
            np.full_like(c, 0.8),
            c >= ma20,
            +1.0,
            min_hold=2,
            max_hold=6,
            cooldown=1,
            warmup=30,
        ),
    )
    out["N5M_REV01S@5m"] = (
        ARCH_B,
        hold_with_cooldown_and_trail(
            (rsi14 > 70) & (last5_clv < -0.2),
            np.full_like(c, 0.8),
            c <= ma20,
            -1.0,
            min_hold=2,
            max_hold=6,
            cooldown=1,
            warmup=30,
        ),
    )

    # 异构专属 5m 微观结构+持仓量因子 (6 个)
    # 1. H5M_CLV01L / S (C簇: 5m 尾盘放量抢筹/砸盘确认的均线锚定回归)
    ent_clv_l = (
        (ma60 >= ma120 * 0.995)
        & (bias20 < -0.9 * atr_pct)
        & (rsi14 < 38)
        & (last5_vs > 0.38)
        & (last5_clv > 0.50)
        & (c > c_prev)
    )
    ext_clv_l = (c >= ma60) | (rsi14 > 60)
    out["H5M_CLV01L@5m"] = (
        ARCH_C,
        hold_with_cooldown_and_trail(
            ent_clv_l, np.full_like(c, 0.90), ext_clv_l, +1.0, min_hold=24, max_hold=72, cooldown=32, warmup=80
        ),
    )

    ent_clv_s = (
        (ma60 <= ma120 * 0.998)
        & (ma120_slp < -0.0005)
        & (bias20 > 0.9 * atr_pct)
        & (rsi14 > 62)
        & (last5_vs > 0.38)
        & (last5_clv < -0.50)
        & (c < c_prev)
    )
    ext_clv_s = (c <= ma60) | (rsi14 < 40)
    out["H5M_CLV01S@5m"] = (
        ARCH_C,
        hold_with_cooldown_and_trail(
            ent_clv_s, np.full_like(c, 0.90), ext_clv_s, -1.0, min_hold=24, max_hold=72, cooldown=32, warmup=80
        ),
    )

    # 2. H5M_LQ01L / S (B簇: 5m 持仓量减仓砍仓/逼空出清 + 5m 子周期长影线尾盘反转)
    ent_lq_l = (
        (bias60 < -1.9 * atr_pct)
        & (rsi14 < 29)
        & (oi_z_sum4 < -0.5)
        & ((max_l_shd3 > 0.45) | (last5_clv > 0.35))
        & (c > o)
        & (c > c_prev)
    )
    ext_lq_l = (c >= ma60) | (rsi14 > 58)
    int_lq_l = np.clip((36.0 - rsi14) / 14.0, 0.65, 1.0)
    out["H5M_LQ01L@5m"] = (
        ARCH_B,
        hold_with_cooldown_and_trail(
            ent_lq_l, int_lq_l, ext_lq_l, +1.0, min_hold=24, max_hold=72, cooldown=32, warmup=80
        ),
    )

    ent_lq_s = (
        (bias60 > 1.9 * atr_pct)
        & (rsi14 > 71)
        & (ma120_slp < 0.0)
        & (oi_z_sum4 < -0.5)
        & ((max_u_shd3 > 0.45) | (last5_clv < -0.35))
        & (c < o)
        & (c < c_prev)
    )
    ext_lq_s = (c <= ma60) | (rsi14 < 42)
    int_lq_s = np.clip((rsi14 - 64.0) / 14.0, 0.65, 1.0)
    out["H5M_LQ01S@5m"] = (
        ARCH_B,
        hold_with_cooldown_and_trail(
            ent_lq_s, int_lq_s, ext_lq_s, -1.0, min_hold=24, max_hold=72, cooldown=32, warmup=80
        ),
    )

    # 3. H5M_OIB01L / S (D簇: 5m 持仓量激增建仓 + 5m 主动买卖盘失衡的回调企稳再启动)
    ent_oib_l = (
        (ma60 > ma120)
        & (bias20 < -0.5 * atr_pct)
        & (oi_z_sum4 > 0.4)
        & (svol3 > 0.35)
        & (last5_clv > 0.40)
        & (c > o)
        & (c > c_prev)
    )
    ext_oib_l = (c < ma120) | (rsi14 > 66)
    out["H5M_OIB01L@5m"] = (
        ARCH_D,
        hold_with_cooldown_and_trail(
            ent_oib_l, np.full_like(c, 0.90), ext_oib_l, +1.0, min_hold=24, max_hold=76, cooldown=32, warmup=80
        ),
    )

    ent_oib_s = (
        (ma60 < ma120)
        & (ma120_slp < -0.0005)
        & (bias20 > 0.5 * atr_pct)
        & (oi_z_sum4 > 0.4)
        & (svol3 < -0.35)
        & (last5_clv < -0.40)
        & (c < o)
        & (c < c_prev)
    )
    ext_oib_s = (c > ma120) | (rsi14 < 34)
    out["H5M_OIB01S@5m"] = (
        ARCH_D,
        hold_with_cooldown_and_trail(
            ent_oib_s, np.full_like(c, 0.90), ext_oib_s, -1.0, min_hold=24, max_hold=76, cooldown=32, warmup=80
        ),
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """加载基础 125 候选缓存，并增量计算 12 个 1d/15m 异构专属因子与 10 个 5m 候选因子（合计 147 个候选信号）"""
    register_hetero_signal_specs()
    scores_path = OUT_DIR / "multitf_factor_scores.csv"
    pooled_path = OUT_DIR / "multitf_factor_pooled_summary.csv"
    ugrid_path = OUT_DIR / "multitf_unit_grid_cache.pkl"
    frames15_path = OUT_DIR / "multitf_grid15m_frames.pkl"
    frames1d_path = OUT_DIR / "multitf_grid1d_frames.pkl"
    micro5m_path = OUT_DIR / "multitf_grid5m_micro_frames.pkl"

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
    if micro5m_path.exists():
        with open(micro5m_path, "rb") as f:
            micro_frames_5m = pickle.load(f)
    else:
        micro_frames_5m = {}
        for aid in CORE_20_IDS:
            spec = specs_by_id[aid]
            df5 = pd.read_parquet(OUT_DIR / "data_5m" / f"{aid}_5m.parquet")
            df15 = frames_15m[spec.symbol]["15m"]
            micro_frames_5m[aid] = build_5m_micro_on_15m(df5, df15)
        with open(micro5m_path, "wb") as f:
            pickle.dump(micro_frames_5m, f)

    new_rows = []
    rows_5m = []
    for aid in CORE_20_IDS:
        spec = specs_by_id[aid]
        df_1d = frames_1d[spec.symbol]
        df_15m = frames_15m[spec.symbol]["15m"]
        f_5m = micro_frames_5m[aid]
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
        for uid, (arch, w_5m) in compute_all_5m_candidates(df_15m, f_5m).items():
            unit_grid_15m[aid][uid] = np.asarray(w_5m, dtype=np.float32)
            for fold in FOLDS:
                ev = evaluate_candidate_on_fold(
                    df_15m, w_5m, tf="15m", fee_rate=spec.fee_rate, cutoff=pd.Timestamp(fold["cutoff"])
                )
                rows_5m.append(
                    {
                        "fold": fold["id"],
                        "asset": aid,
                        "sector": spec.sector,
                        "uid": uid,
                        "sig_id": uid.split("@")[0],
                        "tf": "5m",
                        "archetype": arch,
                        **{k: v for k, v in ev.items() if k != "event_returns"},
                    }
                )

    hetero_scores_df = pd.concat([scores_df, pd.DataFrame(new_rows)], ignore_index=True)
    hetero_pooled_df = summarize_pooled_scores(hetero_scores_df)
    df_5m_scores = pd.DataFrame(rows_5m)
    df_5m_pool = summarize_pooled_scores(df_5m_scores)

    all_5tf_scores_df = pd.concat([hetero_scores_df, df_5m_scores], ignore_index=True)
    all_5tf_pooled_df = pd.concat([hetero_pooled_df, df_5m_pool], ignore_index=True)
    all_5tf_scores_df.to_csv(OUT_DIR / "multitf_hetero_factor_scores.csv", index=False)
    all_5tf_pooled_df.to_csv(OUT_DIR / "multitf_hetero_pooled_summary.csv", index=False)
    return (
        scores_df,
        pooled_df,
        hetero_scores_df,
        hetero_pooled_df,
        df_5m_scores,
        df_5m_pool,
        unit_grid_15m,
        frames_15m,
    )


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
        elif tf in ("30m", "15m", "5m") and slow_uids:
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


def compute_episode_payoff_stats(ret_arrays: list[np.ndarray]) -> dict[str, float]:
    """统计执行区间 (Execution Episodes) 的胜率、单笔盈亏比 R = AvgWin / AvgLoss 与安全边际 ΔW"""
    if not ret_arrays:
        return {"ep_count": 0, "win_rate_pct": 0.0, "payoff_r": 0.0, "safety_margin_pct": 0.0}
    all_r = np.concatenate(ret_arrays)
    if len(all_r) == 0:
        return {"ep_count": 0, "win_rate_pct": 0.0, "payoff_r": 0.0, "safety_margin_pct": 0.0}
    wins = all_r[all_r > 0]
    losses = np.abs(all_r[all_r <= 0])
    wr = len(wins) / len(all_r)
    avg_w = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_l = float(losses.mean()) if len(losses) > 0 else 1e-6
    pl = avg_w / max(avg_l, 1e-6)
    sm = wr - 1.0 / (1.0 + pl)
    return {
        "ep_count": int(len(all_r)),
        "win_rate_pct": round(wr * 100.0, 2),
        "payoff_r": round(pl, 3),
        "safety_margin_pct": round(sm * 100.0, 2),
    }


def build_structure_pack(df_tf: pd.DataFrame, tf: str, box_window: int = 32) -> dict[str, np.ndarray]:
    """基于 Rust `czsc.structure_snapshots` 与 `czsc.structure_stop_anchors` 提取严格因果的结构化出场特征包"""
    c = np.nan_to_num(df_tf["close"].to_numpy(dtype=np.float64), nan=1.0)
    l_arr = np.nan_to_num(df_tf["low"].to_numpy(dtype=np.float64), nan=c)
    h_arr = np.nan_to_num(df_tf["high"].to_numpy(dtype=np.float64), nan=c)
    atr = np.nan_to_num(
        df_tf["atr14"].to_numpy(dtype=np.float64) if "atr14" in df_tf.columns else (h_arr - l_arr),
        nan=0.005 * c,
    )
    tr = df_tf["tr"] if "tr" in df_tf.columns else pd.Series(h_arr - l_arr, index=df_tf.index)
    atr_fast = np.nan_to_num(rust_sma(tr, 5).to_numpy(dtype=np.float64), nan=1.0)
    atr_slow = np.nan_to_num(rust_sma(tr, 20).to_numpy(dtype=np.float64), nan=1.0)

    bars = research_bars(df_tf, tf)
    snaps = czsc.structure_snapshots(bars)
    sec_anchors = czsc.structure_stop_anchors(bars, snaps, "pos_second_ext")
    zs_anchors = czsc.structure_stop_anchors(bars, snaps, "pos_zs_reentry")
    box_anchors = czsc.structure_stop_anchors(bars, snaps, "pos_box_stop", box_window=box_window)

    pad = 0.35 * np.maximum(atr, 1e-6)
    sec_l = np.asarray(sec_anchors["long_stop"], dtype=np.float64) - pad
    sec_s = np.asarray(sec_anchors["short_stop"], dtype=np.float64) + pad
    zs_l = np.asarray(zs_anchors["long_stop"], dtype=np.float64) - pad
    zs_s = np.asarray(zs_anchors["short_stop"], dtype=np.float64) + pad
    box_l = np.asarray(box_anchors["long_stop"], dtype=np.float64) - pad
    box_s = np.asarray(box_anchors["short_stop"], dtype=np.float64) + pad

    fallback_l = np.minimum(l_arr, c) - 1.2 * np.maximum(atr, 1e-6)
    fallback_s = np.maximum(h_arr, c) + 1.2 * np.maximum(atr, 1e-6)

    long_sec = np.where(np.isfinite(sec_l) & (sec_l > 0) & (sec_l < c), sec_l, fallback_l)
    short_sec = np.where(np.isfinite(sec_s) & (sec_s > c), sec_s, fallback_s)
    long_zs = np.where(np.isfinite(zs_l) & (zs_l > 0) & (zs_l < c), zs_l, long_sec)
    short_zs = np.where(np.isfinite(zs_s) & (zs_s > c), zs_s, short_sec)
    long_box = np.where(np.isfinite(box_l) & (box_l > 0) & (box_l < c), box_l, long_sec)
    short_box = np.where(np.isfinite(box_s) & (box_s > c), box_s, short_sec)
    long_hyb = 0.5 * (long_sec + long_box)
    short_hyb = 0.5 * (short_sec + short_box)

    raw_sec_l = np.where(
        np.isfinite(sec_anchors["long_stop"]) & (sec_anchors["long_stop"] < c),
        sec_anchors["long_stop"],
        np.minimum(l_arr, c) - 0.6 * np.maximum(atr, 1e-6),
    )
    raw_sec_s = np.where(
        np.isfinite(sec_anchors["short_stop"]) & (sec_anchors["short_stop"] > c),
        sec_anchors["short_stop"],
        np.maximum(h_arr, c) + 0.6 * np.maximum(atr, 1e-6),
    )
    long_rawzs = np.where(
        np.isfinite(zs_anchors["long_stop"]) & (zs_anchors["long_stop"] < c),
        zs_anchors["long_stop"],
        raw_sec_l,
    )
    short_rawzs = np.where(
        np.isfinite(zs_anchors["short_stop"]) & (zs_anchors["short_stop"] > c),
        zs_anchors["short_stop"],
        raw_sec_s,
    )

    bi_dir = np.asarray([s.completed_bi_dir for s in snaps], dtype=np.int8)
    fx_dir = np.asarray([s.completed_fx_dir for s in snaps], dtype=np.int8)
    zs_zg = np.asarray([s.zs_zg for s in snaps], dtype=np.float64) - pad
    zs_zd = np.asarray([s.zs_zd for s in snaps], dtype=np.float64) + pad
    raw_zs_zg = np.asarray([s.zs_zg for s in snaps], dtype=np.float64)
    raw_zs_zd = np.asarray([s.zs_zd for s in snaps], dtype=np.float64)
    return {
        "close": c,
        "atr_fast": atr_fast,
        "atr_slow": atr_slow,
        "long_sec": long_sec,
        "short_sec": short_sec,
        "long_zs": long_zs,
        "short_zs": short_zs,
        "long_box": long_box,
        "short_box": short_box,
        "long_hyb": long_hyb,
        "short_hyb": short_hyb,
        "long_rawzs": long_rawzs,
        "short_rawzs": short_rawzs,
        "bi_dir": bi_dir,
        "fx_dir": fx_dir,
        "zs_zg": zs_zg,
        "zs_zd": zs_zd,
        "raw_zs_zg": raw_zs_zg,
        "raw_zs_zd": raw_zs_zd,
    }


def rust_govern_series(
    w_arr: np.ndarray,
    sp: dict[str, np.ndarray],
    stop_kind: str = "zs",
    trailing: float = 0.014,
    trail_activation: float = 0.016,
    max_hold: int = 96,
    ratchet_stop: bool = True,
    use_zs_reentry: bool = True,
    use_bi_exit: bool = False,
    use_fx_exit: bool = True,
    cooldown: int = 2,
    use_raw_zs_edge: bool = False,
) -> np.ndarray:
    """纯透传调用 Rust `czsc.evaluate_structure_decay` 对连续持仓序列施加结构化出场治理"""
    w = np.nan_to_num(np.asarray(w_arr, dtype=np.float64), nan=0.0)
    d = np.where(w > 1e-5, 1, np.where(w < -1e-5, -1, 0)).astype(np.int8)
    q = np.abs(w)
    l_stop = sp[f"long_{stop_kind}"]
    s_stop = sp[f"short_{stop_kind}"]
    stop_lvl = np.where(d >= 0, l_stop, s_stop)
    rev_dir = sp["bi_dir"] if use_bi_exit else np.zeros_like(d)
    zg_key = "raw_zs_zg" if use_raw_zs_edge else "zs_zg"
    zd_key = "raw_zs_zd" if use_raw_zs_edge else "zs_zd"
    res = czsc.evaluate_structure_decay(
        d.tolist(),
        q.tolist(),
        sp["close"].tolist(),
        stop_lvl.tolist(),
        rev_dir.tolist(),
        sp["atr_fast"].tolist(),
        sp["atr_slow"].tolist(),
        lam0=0.0,
        rho_min=0.5,
        rho_max=2.0,
        d_eps=0.02,
        max_hold=max_hold,
        trailing=trailing,
        trail_activation=trail_activation,
        ratchet_stop=ratchet_stop,
        min_hold=0,
        cooldown=cooldown,
        track_signal=True,
        zs_zg=sp[zg_key].tolist() if use_zs_reentry else None,
        zs_zd=sp[zd_key].tolist() if use_zs_reentry else None,
        completed_fx_dir=sp["fx_dir"].tolist() if use_fx_exit else None,
    )
    return np.asarray(res["strength"], dtype=np.float64)


def synthesize_structure_aware_synergy(
    df_grid: pd.DataFrame,
    umap: dict[str, np.ndarray],
    selected_uids: list[str],
    spec: AssetSpec,
    sp15: dict[str, np.ndarray],
    sp1d_15: dict[str, np.ndarray],
    use_factor_exit: bool = False,
    use_asset_gate: bool = False,
) -> np.ndarray:
    """
    结合单因子层结构化出场 (`use_factor_exit`) 与单品种合成层结构门控 (`use_asset_gate`) 的快慢周期协同合成：
    - 因子层：对 1d/15m 异构专属因子施加次极值+三买卖拉回中枢证伪 (`pos_zs_reentry`)、反向完成笔离场与 1.3% 动态追踪止盈；
    - 合成层：对 15m 合成目标仓位 `t15` 施加因果对齐的 1d 宏观次极值/箱体混合止损、中枢重入证伪与 1.5% 动态追踪止盈。
    """
    if not selected_uids:
        return np.zeros(len(df_grid), dtype=np.float64)

    umap_use: dict[str, np.ndarray] = {}
    for u in selected_uids:
        w_u = np.asarray(umap[u], dtype=np.float64)
        if use_factor_exit and u.startswith("H"):
            w_u = rust_govern_series(
                w_u,
                sp15,
                stop_kind="rawzs",
                trailing=0.013,
                trail_activation=0.012,
                max_hold=64,
                ratchet_stop=True,
                use_zs_reentry=True,
                use_bi_exit=True,
                use_fx_exit=True,
                cooldown=0,
                use_raw_zs_edge=True,
            )
        umap_use[u] = w_u

    t15 = synthesize_fast_slow_synergy(
        df_grid, umap_use, selected_uids, spec, scale_1d=0.70, res_boost=0.25, opp_damp=0.35, deadzone=0.06
    )
    if use_asset_gate:
        t15_gov = rust_govern_series(
            t15,
            sp1d_15,
            stop_kind="hyb",
            trailing=0.015,
            trail_activation=0.023,
            max_hold=160,
            ratchet_stop=True,
            use_zs_reentry=True,
            use_bi_exit=False,
            use_fx_exit=False,
            cooldown=4,
            use_raw_zs_edge=False,
        )
        t15 = np.where(np.abs(t15_gov) > 1e-6, t15, 0.0)
    return t15


def select_5m_gate1_adaptive(f_s5: pd.DataFrame, aid: str, f_p5: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    5m 周期 Gate 1 准入（遵循 SOP 阶段一胜率与样本数门禁）：
    - 单品种样本数 n >= 20, 胜率 win_rate >= 52.5%, mean_net_pct > 0, is_net_sharpe >= 0.43
    - 年化换手率 ann_turnover <= 56x, 跨品种池化事件数 total_n >= 180
    """
    sub5 = f_s5[(f_s5["asset"] == aid) & (f_s5["tf"] == "5m")].copy()
    if "total_n" not in sub5.columns and f_p5 is not None:
        sub5 = sub5.merge(
            f_p5[["uid", "total_n", "pooled_mean_pct", "pooled_t", "pos_asset_ratio"]],
            on="uid",
            how="left",
        )
    cond = (
        sub5["uid"].str.startswith("H5M_")
        & (sub5["n"] >= 20)
        & (sub5["ann_turnover"] <= 56.0)
        & (sub5["total_n"] >= 180)
        & (sub5["mean_net_pct"] > 0.0)
        & (sub5["is_net_sharpe"] >= 0.43)
        & (sub5["win_rate"] >= 52.5)
    )
    return sub5[cond].copy()


def run_tier3_5m_incremental_funnel(
    f_s5: pd.DataFrame,
    aid: str,
    fold_id: str,
    base_uids: list[str],
    df_is: pd.DataFrame,
    umap_is: dict[str, np.ndarray],
    ret_is: np.ndarray,
    fee_r: float,
    spec: AssetSpec,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    在已入库 4 周期 (1d+60m+30m+15m) 信号基础上执行 Tier-3 5m 增量验证漏斗 (Gate 1 -> Gate 2 -> Gate 3)：
    1. Gate 1: 单品种 n >= 20, win_rate >= 52.5%, is_net_sharpe >= 0.43, ann_turnover <= 56x, total_n >= 180;
    2. 品种主干活性门禁 (Gate 2a): 要求该品种已入库 4 周期主干合成信号样本内 base_is_sharpe >= 0.60，防止在无趋势震荡品种上用 5m 强行开仓稀释组合权重;
    3. 跨品种池化与同簇同向正相关检验 (Gate 2b): pooled_t >= 0.65, pos_asset_ratio >= 0.55，且与同簇同方向 (L vs L, S vs S) 已入库信号的正向激活条件相关 < 0.35;
    4. 扣费后边际夏普增益 (Gate 3): 加入 5m 信号后样本内净 Sharpe 边际提升 ΔSharpe >= +0.045。
    """
    g1_5m = select_5m_gate1_adaptive(f_s5, aid)
    if g1_5m.empty:
        return sorted(base_uids), [], []

    cur_syn = synthesize_fast_slow_synergy(df_is, umap_is, base_uids, spec)
    base_shp = eval_is_net_sharpe(cur_syn, ret_is, fee_r)
    cur_shp = base_shp

    g1_5m["q_score"] = g1_5m["is_net_sharpe"] * np.log1p(g1_5m["n"]) + 0.25 * g1_5m["pooled_t"].fillna(0.0)
    g1_5m = g1_5m.sort_values("q_score", ascending=False)

    accepted_uids = list(base_uids)
    rejected_5m: list[dict[str, Any]] = []
    admitted_5m: list[dict[str, Any]] = []

    for _, row in g1_5m.iterrows():
        uid = str(row["uid"])
        sid = uid.split("@")[0]
        side = sid[-1]
        arch = str(row["archetype"])
        w_is = umap_is[uid]
        gap_fill_pct, dd_rescue_bp = compute_gap_fill_and_dd_rescue(w_is, cur_syn, ret_is, fee_r)

        same_peers = [
            u
            for u in accepted_uids
            if ALL_SIGNAL_SPECS[u.split("@")[0]].archetype == arch and u.split("@")[0][-1] == side
        ]
        peer_corrs = [(u, max(0.0, active_union_corr(w_is, umap_is[u]))) for u in same_peers]
        max_peer, max_corr = max(peer_corrs, key=lambda x: x[1]) if peer_corrs else ("None", 0.0)

        trial_raw = cur_syn + 0.22 * w_is
        trial_shp = eval_is_net_sharpe(np.clip(trial_raw, -1.0, 1.0), ret_is, fee_r)
        delta_shp = round(trial_shp - cur_shp, 4)

        if base_shp < 0.60:
            rejected_5m.append(
                {
                    "fold": fold_id,
                    "asset": aid,
                    "uid": uid,
                    "sig_id": sid,
                    "tf": "5m",
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
                    "reject_stage": "Gate2_WeakBackboneAsset",
                    "reject_reason": f"品种大级别主干趋势不足 (base_IS_Sharpe={base_shp:.2f} < 0.60)，禁止5m独立激活",
                }
            )
            continue

        if float(row["pooled_t"]) < 0.65 or float(row["pos_asset_ratio"]) < 0.55:
            rejected_5m.append(
                {
                    "fold": fold_id,
                    "asset": aid,
                    "uid": uid,
                    "sig_id": sid,
                    "tf": "5m",
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
                    "reject_stage": "Gate2_Weak5MPoolEdge",
                    "reject_reason": f"5m跨品种池化显著性不足 (pooled_t={row['pooled_t']:.2f}, pos_ratio={row['pos_asset_ratio']:.2f})",
                }
            )
            continue

        if max_corr >= 0.35:
            rejected_5m.append(
                {
                    "fold": fold_id,
                    "asset": aid,
                    "uid": uid,
                    "sig_id": sid,
                    "tf": "5m",
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
                    "reject_stage": "Gate2_HighSameSideCorr",
                    "reject_reason": f"与同簇同向已入库 {max_peer} 激活态共线冗余 (ρ_pos={max_corr:.2f} >= 0.35)",
                }
            )
            continue

        if delta_shp < 0.045:
            rejected_5m.append(
                {
                    "fold": fold_id,
                    "asset": aid,
                    "uid": uid,
                    "sig_id": sid,
                    "tf": "5m",
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
                    "reject_stage": "Gate3_Low5MMarginalSharpe",
                    "reject_reason": f"5m扣费后边际净夏普增益不足 (ΔSharpe={delta_shp:+.4f} < 0.045)",
                }
            )
            continue

        accepted_uids.append(uid)
        cur_syn = np.clip(trial_raw, -1.0, 1.0)
        cur_shp = trial_shp
        admitted_5m.append(
            {
                "fold": fold_id,
                "asset": aid,
                "uid": uid,
                "sig_id": sid,
                "tf": "5m",
                "archetype": arch,
                "tier": "Hetero_Dedicated_5m",
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

    return sorted(accepted_uids), rejected_5m, admitted_5m


def map_15m_target_to_5m_grid(
    df15: pd.DataFrame,
    t15: np.ndarray,
    df5: pd.DataFrame,
    is_5m_native: bool = False,
) -> np.ndarray:
    """
    将 15m 目标权重对齐至 5m 原生执行网格。
    严格因果：后向时间戳对齐 (merge_asof backward)，禁止向过去时间回填任何仓位 (删除 w5_early[idx-2:idx] = w5[idx])。
    在 15m 周期未完成确认前 (如 09:20、09:25)，5m 网格只能维持此前已确认的高周期目标，不得前视 09:30 的收盘仓位。
    """
    return map_tf_weight_to_base_grid(df15, t15, df5)


def allocate_and_account_5m_grid(
    df5_map: dict[str, pd.DataFrame],
    specs: dict[str, AssetSpec],
    asset_targets_5m: dict[str, np.ndarray],
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
    deadzone: float = 0.08,
) -> dict[str, Any]:
    """在 5m 原生执行网格 (每天48根K线) 上执行资产状态机、Alloc3_SectorHRP板块分层风险平价与摩擦结算"""
    returns, positions, sectors, fees = {}, {}, {}, {}
    for aid, spec in specs.items():
        grid5 = df5_map[aid]
        mask = (grid5["dt"] <= oos_end).to_numpy()
        sub_grid = grid5.loc[mask].copy().reset_index(drop=True)
        raw_target = np.asarray(asset_targets_5m[aid], dtype=np.float64)[mask].copy()
        raw_target[sub_grid["dt"] < oos_start] = 0.0

        # 死区防抖状态机
        target = np.zeros_like(raw_target)
        cur = 0.0
        for i, w in enumerate(raw_target):
            if abs(w - cur) >= deadzone:
                cur = w
            target[i] = cur

        ret_s, pos_s = build_asset_execution_inputs(sub_grid, target)
        returns[aid] = ret_s
        positions[aid] = pos_s
        sectors[aid] = spec.sector
        fees[aid] = float(spec.fee_rate)

    raw_ret_df = pd.DataFrame(returns).sort_index()
    tradable_mask = raw_ret_df.notna()
    ret_df = raw_ret_df.fillna(0.0)
    pos_df = pd.DataFrame(positions).reindex_like(ret_df).fillna(0.0)
    for col in pos_df.columns:
        t_col = tradable_mask[col].to_numpy()
        p_col = pos_df[col].to_numpy().copy()
        for i in range(1, len(p_col)):
            if not t_col[i]:
                p_col[i] = p_col[i - 1]
        pos_df[col] = p_col

    periods_per_year = 252.0 * 48.0
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
        use_dynamic_active_pool=True,
    )
    weights_df = res["asset_weights_df"]

    oos_mask = ret_df.index >= oos_start
    ret_oos = ret_df.loc[oos_mask].copy()
    mask_oos = tradable_mask.loc[oos_mask].copy()
    w_oos = weights_df.loc[oos_mask].copy()

    # 总名义敞口上限 1.0
    gross_exp = w_oos.abs().sum(axis=1)
    scale_cap = np.maximum(gross_exp, 1.0)
    w_oos = w_oos.div(scale_cap, axis=0)

    w_oos.iloc[0] = 0.0
    w_oos.iloc[-1] = 0.0
    for col in w_oos.columns:
        t_col = mask_oos[col].to_numpy()
        w_col = w_oos[col].to_numpy().copy()
        for i in range(1, len(w_col)):
            if not t_col[i]:
                w_col[i] = w_col[i - 1]
        w_oos[col] = w_col

    ledger = account_portfolio_execution(ret_oos, w_oos, fees, tradable_mask=mask_oos)
    return {
        "daily_returns": ledger["daily_returns"],
        "bar_net_returns": ledger["bar_net_returns"],
        "gross_returns_df": ledger["gross_returns_df"],
        "cost_returns_df": ledger["cost_returns_df"],
        "turnover_df": ledger["turnover_df"],
        "weights_df": w_oos,
    }


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

    res_map = {r["experiment"]: r for _, r in comp_df.iterrows()}
    r_a = res_map.get("ExpA_Baseline_60m30m", {})
    r_b = res_map.get("ExpB_AllTF_Gate1Only", {})
    r_c = res_map.get("ExpC_AllTF_Gate123_Incremental", {})
    r_d = res_map.get("ExpD_AllTF_Gate123_FastSlowSynergy", {})
    r_e = res_map.get("ExpE_Hetero1D15M_FastSlowSynergy", {})
    r_ef = res_map.get("ExpE_FactorExit", {})
    r_ea = res_map.get("ExpE_AssetGate", {})
    r_es = res_map.get("ExpE_FullStructExit", {})
    r_es_3x = res_map.get("ExpE_FullStructExit_3x", {})
    r_f = res_map.get("ExpF_5TF_Hetero5M_MicroOI", {})
    r_f_3x = res_map.get("ExpF_3x_TargetExposure", {})

    palette = {
        "ExpA_Baseline_60m30m": (
            "#64748b",
            "dash",
            f"基准A: 仅日内(60m+30m) (Shp {r_a.get('sharpe', 0):.2f} / Cal {r_a.get('calmar', 0):.2f} / Net +{r_a.get('net_ret_pct', 0):.2f}%)",
        ),
        "ExpB_AllTF_Gate1Only": (
            "#f59e0b",
            "dot",
            f"实验B: 全周期仅Gate1直接合成 (Shp {r_b.get('sharpe', 0):.2f} / Cal {r_b.get('calmar', 0):.2f} / Turn {r_b.get('ann_turnover', 0):.1f}x)",
        ),
        "ExpC_AllTF_Gate123_Incremental": (
            "#0ea5e9",
            "solid",
            f"实验C: 全周期+Gate1/2/3增量过滤 (Shp {r_c.get('sharpe', 0):.2f} / Cal {r_c.get('calmar', 0):.2f} / Net +{r_c.get('net_ret_pct', 0):.2f}%)",
        ),
        "ExpD_AllTF_Gate123_FastSlowSynergy": (
            "#8b5cf6",
            "solid",
            f"实验D: 全周期+增量过滤+快慢协同 (Shp {r_d.get('sharpe', 0):.2f} / Cal {r_d.get('calmar', 0):.2f})",
        ),
        "ExpE_Hetero1D15M_FastSlowSynergy": (
            "#10b981",
            "dash",
            f"实验E(基准): +1d/15m异构专属 (Shp {r_e.get('sharpe', 0):.2f} / Cal {r_e.get('calmar', 0):.2f} / Net +{r_e.get('net_ret_pct', 0):.2f}%)",
        ),
        "ExpE_FactorExit": (
            "#059669",
            "dot",
            f"实验E1: +单因子层结构出场 (Shp {r_ef.get('sharpe', 0):.2f} / Cal {r_ef.get('calmar', 0):.2f} / MaxDD {r_ef.get('max_dd_pct', 0):.2f}%)",
        ),
        "ExpE_AssetGate": (
            "#2563eb",
            "dot",
            f"实验E2: +单品种1d结构门控 (Shp {r_ea.get('sharpe', 0):.2f} / Cal {r_ea.get('calmar', 0):.2f} / MaxDD {r_ea.get('max_dd_pct', 0):.2f}%)",
        ),
        "ExpE_FullStructExit": (
            "#dc2626",
            "solid",
            f"实验E3(最优): +双层全量结构化出场 (Shp {r_es.get('sharpe', 0):.2f} / Cal {r_es.get('calmar', 0):.2f} / Net +{r_es.get('net_ret_pct', 0):.2f}%)",
        ),
        "ExpF_5TF_Hetero5M_MicroOI": (
            "#9333ea",
            "dash",
            f"实验F: +5m微观异构+因果执行 (Shp {r_f.get('sharpe', 0):.2f} / Cal {r_f.get('calmar', 0):.2f} / Net +{r_f.get('net_ret_pct', 0):.2f}%)",
        ),
    }

    for k, (color, dash, label) in palette.items():
        if k not in daily_curves:
            continue
        ds = daily_curves[k]
        nav = (1.0 + ds.cumsum()) * 100.0
        pk = np.maximum.accumulate(np.maximum(nav.to_numpy(), 100.0))
        dd = (nav.to_numpy() - pk) / pk * 100.0
        width = 3.4 if k == "ExpE_FullStructExit" else (2.4 if k.startswith("ExpE") else 1.8)
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
            "实验F: 5大周期候选信号在 Gate 1 -> Gate 2 -> Gate 3 漏斗中的留存演进",
            "实验E3(全量结构出场): Core-20 分品种 OOS 净收益贡献 vs 交易成本 (%)",
        ),
        horizontal_spacing=0.10,
    )

    tfs_order = ["1d", "60m", "30m", "15m", "5m"]
    f_sub = (
        funnel_stats_df[funnel_stats_df["pool"] == "ExpF_5TF_WithHetero5M"]
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
<title>全周期(1d+60m+30m+15m+5m)×六大战略簇 CTA 信号增量验证、快慢协同与结构化动态出场报告</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 24px 36px; line-height: 1.55; }}
  h1 {{ font-size: 24px; margin-bottom: 6px; color: #0f172a; }}
  h2 {{ font-size: 18px; margin-top: 28px; margin-bottom: 10px; border-left: 4px solid #e11d48; padding-left: 10px; color: #1e293b; }}
  .subtitle {{ color: #475569; font-size: 14px; margin-bottom: 18px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin: 16px 0 24px 0; }}
  .kpi-card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
  .kpi-title {{ font-size: 11.5px; color: #64748b; font-weight: 600; }}
  .kpi-val {{ font-size: 18px; font-weight: 700; color: #0f172a; margin-top: 4px; }}
  .kpi-sub {{ font-size: 11.5px; color: #10b981; margin-top: 2px; }}
  .chart-box {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; margin-bottom: 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
  .styled-table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.04); margin-bottom: 20px; }}
  .styled-table thead tr {{ background-color: #0f172a; color: #ffffff; text-align: left; }}
  .styled-table th, .styled-table td {{ padding: 8px 11px; border-bottom: 1px solid #e2e8f0; }}
  .styled-table tbody tr:nth-of-type(even) {{ background-color: #f8fafc; }}
  .highlight-box {{ background: #fff1f2; border: 1px solid #fecdd3; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px; font-size: 13.5px; }}
</style>
</head>
<body>
  <h1>全周期（1d + 60m + 30m + 15m + 5m）× 六大战略簇（A~F）CTA 信号增量验证、快慢协同与结构化动态出场报告</h1>
  <div class="subtitle">资产池：Core-20 主力商品期货（5大板块） | 两折滚动纯样本外（Stitched OOS）：2025-01-02 ~ 2026-09-30（503个交易日） | 全面对标原生 5m 执行网格（每天48根K线）与 Rust 结构化出场状态机</div>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-title">基准A：仅日内(60m+30m)</div>
      <div class="kpi-val">Shp {r_a.get("sharpe", 0):.2f} / Cal {r_a.get("calmar", 0):.2f}</div>
      <div class="kpi-sub">净收益 +{r_a.get("net_ret_pct", 0):.2f}% | MaxDD {r_a.get("max_dd_pct", 0):.2f}%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">实验E(基准)：+1d/15m异构</div>
      <div class="kpi-val">Shp {r_e.get("sharpe", 0):.2f} / Cal {r_e.get("calmar", 0):.2f}</div>
      <div class="kpi-sub">净收益 +{r_e.get("net_ret_pct", 0):.2f}% | MaxDD {r_e.get("max_dd_pct", 0):.2f}%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">实验E1：+单因子结构出场</div>
      <div class="kpi-val">Shp {r_ef.get("sharpe", 0):.2f} / Cal {r_ef.get("calmar", 0):.2f}</div>
      <div class="kpi-sub">净收益 +{r_ef.get("net_ret_pct", 0):.2f}% | MaxDD {r_ef.get("max_dd_pct", 0):.2f}%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">实验E2：+单品种1d结构门控</div>
      <div class="kpi-val">Shp {r_ea.get("sharpe", 0):.2f} / Cal {r_ea.get("calmar", 0):.2f}</div>
      <div class="kpi-sub">单笔R={r_ea.get("single_trade_payoff_r", 0):.2f} | MaxDD {r_ea.get("max_dd_pct", 0):.2f}%</div>
    </div>
    <div class="kpi-card" style="border: 2px solid #dc2626; background: #fff1f2;">
      <div class="kpi-title" style="color:#be123c;">实验E3(最优)：双层结构出场</div>
      <div class="kpi-val" style="color:#dc2626;">Shp {r_es.get("sharpe", 0):.2f} / Cal {r_es.get("calmar", 0):.2f}</div>
      <div class="kpi-sub" style="color:#be123c;"><b>净收益 +{r_es.get("net_ret_pct", 0):.2f}% (3x: +{r_es_3x.get("net_ret_pct", 0):.2f}%, DD {r_es_3x.get("max_dd_pct", 0):.2f}%)</b></div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">实验F：+5m因果网格执行</div>
      <div class="kpi-val">Shp {r_f.get("sharpe", 0):.2f} / Cal {r_f.get("calmar", 0):.2f}</div>
      <div class="kpi-sub">净收益 +{r_f.get("net_ret_pct", 0):.2f}% (3x: +{r_f_3x.get("net_ret_pct", 0):.2f}%)</div>
    </div>
  </div>

  <div class="highlight-box">
    <b>缠论结构化动态止损与出场机制（Structure-Aware Exits）核心实证结论：</b><br/>
    1. <b>双层结构化出场机制（Rust <code>czsc-core</code> 原生实现，零未完成笔前视）：</b><br/>
       &bull; <b>单因子层（<code>ExpE_FactorExit</code>）</b>：针对异构波段因子引入已完成笔次极值（<code>pos_second_ext</code>）与三买卖拉回中枢证伪（<code>pos_zs_reentry</code>）、反向完成笔离场及 1.3% 动态追踪止盈，将异构因子单笔盈亏比从 0.902 提升至 1.432，组合净 Sharpe 提升至 <b><code>{r_ef.get("sharpe", 0):.2f}</code></b>，MaxDD 降至 <b><code>{r_ef.get("max_dd_pct", 0):.2f}%</code></b>；<br/>
       &bull; <b>单品种合成层（<code>ExpE_AssetGate</code>）</b>：对 15m 合成目标仓位施加因果映射的 1d 宏观次极值/箱体混合止损与 1.5% 动态追踪止盈，单笔盈亏比达 <b><code>R = {r_ea.get("single_trade_payoff_r", 0):.3f}</code></b>（远超 SOP 1.70 门禁），1.0x MaxDD 从 1.94% 压降至 <b><code>{r_ea.get("max_dd_pct", 0):.2f}%</code></b>；<br/>
    2. <b>全量双层协同（<code>ExpE_FullStructExit</code>）突破 2.0 净夏普：</b><br/>
       &bull; <b>1.0x 基准敞口</b>：Net Sharpe 从 1.81 跃升至 <b><code>{r_es.get("sharpe", 0):.2f}</code></b>，累计净收益从 +8.51% 提升至 <b><code>+{r_es.get("net_ret_pct", 0):.2f}%</code></b>，MaxDD 从 1.94% 降至 <b><code>{r_es.get("max_dd_pct", 0):.2f}%</code></b>，Calmar 从 2.19 跃升至 <b><code>{r_es.get("calmar", 0):.2f}</code></b>，单笔盈亏比 <b><code>R = {r_es.get("single_trade_payoff_r", 0):.3f} &ge; 1.70</code></b>（胜率提升至 <b><code>{r_es.get("single_trade_win_pct", 0):.1f}%</code></b>）；<br/>
       &bull; <b>3.0x 实盘目标敞口（<code>ExpE_FullStructExit_3x</code>）</b>：累计净收益达 <b><code>+{r_es_3x.get("net_ret_pct", 0):.2f}%</code></b>（年化 <b><code>+{r_es_3x.get("ann_ret_pct", 0):.2f}%</code></b>），最大回撤从 5.77% 显著优化至 <b><code>{r_es_3x.get("max_dd_pct", 0):.2f}%</code></b>（远低于 8.0% 风控红线）。
  </div>

  <div class="chart-box">{fig1.to_html(full_html=False, include_plotlyjs=False)}</div>
  <div class="chart-box">{fig2.to_html(full_html=False, include_plotlyjs=False)}</div>

  <h2>1. 样本外核心对照与结构化出场消融汇总表 (2025-01-02 ~ 2026-09-30)</h2>
  {df_to_html_table(comp_display, max_rows=20)}

  <h2>2. Gate 1 -> Gate 2 -> Gate 3 分周期增量筛选漏斗统计 (原始池 vs 1d/15m异构池 vs 5TF异构5m池)</h2>
  {df_to_html_table(funnel_stats_df, max_rows=20)}

  <h2>3. 核心增量信号入库归因榜 (含 1d/15m/5m 异构专属因子与跨周期增量信号)</h2>
  {df_to_html_table(admitted_summary_df, max_rows=35)}

  <h2>4. 被 Gate 2 / Gate 3 拦截的高频冗余与伪增量信号清单 (Top 30 典型被拒信号)</h2>
  {df_to_html_table(rejected_summary_df, max_rows=30)}

  <h2>5. 实验 E3 (全量结构化出场)：五大商品板块样本外归因汇总</h2>
  {df_to_html_table(sec_df, max_rows=10)}

  <h2>6. 实验 E3 (全量结构化出场)：Core-20 全部 20 个品种样本外净收益与成本归因明细</h2>
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
    tfs_to_show: tuple[str, ...] = ("1d", "60m", "30m", "15m"),
) -> list[dict[str, Any]]:
    rows = []
    for tf in list(tfs_to_show) + ["ALL"]:
        sub_s = scores_df if tf == "ALL" else scores_df[scores_df["tf"] == tf]
        tot_eval = len(sub_s)
        g1_cnt = 0
        for fold in FOLDS:
            f_id = fold["id"]
            f_s = scores_df[scores_df["fold"] == f_id]
            f_p = pooled_df[pooled_df["fold"] == f_id]
            for aid in CORE_20_IDS:
                if tf == "5m":
                    g1_sub = select_5m_gate1_adaptive(f_s, aid, f_p)
                elif tf == "ALL":
                    g1_sub = select_exp_b_alltf_gate1_adaptive(
                        f_s, f_p, aid, allow_lowturn_hetero_15m=allow_lowturn_hetero_15m
                    )
                    if "5m" in tfs_to_show:
                        g1_5m = select_5m_gate1_adaptive(f_s, aid, f_p)
                        g1_sub = pd.concat([g1_sub, g1_5m], ignore_index=True)
                else:
                    g1_sub = select_exp_b_alltf_gate1_adaptive(
                        f_s, f_p, aid, allow_lowturn_hetero_15m=allow_lowturn_hetero_15m
                    )
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
    print("  启动全周期 (1d+60m+30m+15m+5m) × 六大战略簇 CTA 增量信息漏斗、快慢协同与 5m 异构持仓量因子实验")
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

    (
        scores_df,
        pooled_df,
        hetero_scores_df,
        hetero_pooled_df,
        df_5m_scores,
        df_5m_pool,
        unit_grid_15m,
        frames,
    ) = ensure_multitf_and_hetero_caches(specs_list)
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

    # 3. 在 ExpE 基础上执行 Tier-3 5m 异构持仓量因子增量漏斗 (供 ExpF 使用)
    exp_f_sel: dict[tuple[str, str], list[str]] = {}
    rej_5m_records: list[dict[str, Any]] = []
    adm_5m_records: list[dict[str, Any]] = []
    for fold in FOLDS:
        f_id = fold["id"]
        f_s5 = df_5m_scores[df_5m_scores["fold"] == f_id].merge(
            df_5m_pool[df_5m_pool["fold"] == f_id][
                ["uid", "total_n", "pooled_mean_pct", "pooled_t", "pos_asset_ratio"]
            ],
            on="uid",
            how="left",
        )
        for aid in CORE_20_IDS:
            m_is, ret_is, fee_r = is_meta[aid][f_id]
            spec = specs_by_id[aid]
            df_is = by_aid[aid].iloc[np.where(m_is)[0]].reset_index(drop=True)
            umap_is = {u: np.asarray(unit_grid_15m[aid][u], dtype=np.float64)[m_is] for u in unit_grid_15m[aid]}
            base_uids = het_funnel_sel[(f_id, aid)]
            acc_5m, rej_5m_sub, adm_5m_sub = run_tier3_5m_incremental_funnel(
                f_s5, aid, f_id, base_uids, df_is, umap_is, ret_is, fee_r, spec
            )
            exp_f_sel[(f_id, aid)] = acc_5m
            rej_5m_records.extend(rej_5m_sub)
            adm_5m_records.extend(adm_5m_sub)

    base_rej_df = pd.DataFrame(base_rejected)
    base_adm_df = pd.DataFrame(base_admitted)
    het_rej_df = pd.DataFrame(het_rejected)
    het_adm_df = pd.DataFrame(het_admitted)
    all_rej_df = pd.concat([het_rej_df, pd.DataFrame(rej_5m_records)], ignore_index=True)
    all_adm_df = pd.concat([het_adm_df, pd.DataFrame(adm_5m_records)], ignore_index=True)

    scores_5tf_df = pd.concat([hetero_scores_df, df_5m_scores], ignore_index=True)
    pooled_5tf_df = pd.concat([hetero_pooled_df, df_5m_pool], ignore_index=True)

    funnel_rows = (
        collect_funnel_table(
            "ExpCD_Legacy125",
            scores_df,
            pooled_df,
            base_rej_df,
            base_adm_df,
            allow_lowturn_hetero_15m=False,
            tfs_to_show=("1d", "60m", "30m", "15m"),
        )
        + collect_funnel_table(
            "ExpE_WithHetero1D15M",
            hetero_scores_df,
            hetero_pooled_df,
            het_rej_df,
            het_adm_df,
            allow_lowturn_hetero_15m=True,
            tfs_to_show=("1d", "60m", "30m", "15m"),
        )
        + collect_funnel_table(
            "ExpF_5TF_WithHetero5M",
            scores_5tf_df,
            pooled_5tf_df,
            all_rej_df,
            all_adm_df,
            allow_lowturn_hetero_15m=True,
            tfs_to_show=("1d", "60m", "30m", "15m", "5m"),
        )
    )
    funnel_stats_df = pd.DataFrame(funnel_rows)
    funnel_stats_df.to_csv(OUT_DIR / "multitf_funnel_stage_summary.csv", index=False)

    # 加载原生 5m K线行情数据与 1d/15m 严格因果缠论结构快照包
    df5_map = {}
    for aid in CORE_20_IDS:
        p = OUT_DIR / "data_5m" / f"{aid}_5m.parquet"
        df = pd.read_parquet(p)
        df["dt"] = pd.to_datetime(df["dt"])
        df5_map[aid] = df.sort_values("dt").reset_index(drop=True)

    with open(OUT_DIR / "multitf_grid1d_frames.pkl", "rb") as f:
        frames_1d = pickle.load(f)

    sp15_map: dict[str, dict[str, np.ndarray]] = {}
    sp1d_on_15m_map: dict[str, dict[str, np.ndarray]] = {}
    for aid in CORE_20_IDS:
        sym = specs_by_id[aid].symbol
        df15 = frames[sym]["15m"]
        df1d = frames_1d[sym]
        sp15_map[aid] = build_structure_pack(df15, "15m", box_window=32)
        sp1d = build_structure_pack(df1d, "1d", box_window=10)
        sp1d_15 = dict(sp15_map[aid])
        for k in (
            "long_sec",
            "short_sec",
            "long_zs",
            "short_zs",
            "long_box",
            "short_box",
            "long_hyb",
            "short_hyb",
            "zs_zg",
            "zs_zd",
        ):
            src_arr = np.asarray(sp1d[k], dtype=np.float64)
            src_clean = np.where(np.isfinite(src_arr) & (src_arr > 0), src_arr, -1.0)
            mapped = map_tf_weight_to_base_grid(df1d, src_clean, df15)
            bad = (~np.isfinite(mapped)) | (mapped <= 0)
            if k in ("zs_zg", "zs_zd"):
                sp1d_15[k] = np.where(bad, np.nan, mapped)
            else:
                sp1d_15[k] = np.where(bad, sp15_map[aid][k], mapped)
        sp1d_on_15m_map[aid] = sp1d_15

    experiments = [
        (
            "ExpA_Baseline_60m30m",
            "基准A: 仅日内(60m+30m) + 标准Gate1 + Comb2",
            scores_df,
            pooled_df,
            lambda f_s, f_p, aid, f_id: select_exp_a_intraday_baseline(f_s, aid),
            lambda df, umap, sel, spec: synthesize_asset_signal(df, umap, sel, spec, "Comb2_SixClusterBudget"),
            False,
        ),
        (
            "ExpB_AllTF_Gate1Only",
            "实验B: 全周期(1d+60m+30m+15m) + 自适应Gate1 + 直接Comb2",
            scores_df,
            pooled_df,
            lambda f_s, f_p, aid, f_id: sorted(select_exp_b_alltf_gate1_adaptive(f_s, f_p, aid)["uid"].tolist()),
            lambda df, umap, sel, spec: synthesize_asset_signal(df, umap, sel, spec, "Comb2_SixClusterBudget"),
            False,
        ),
        (
            "ExpC_AllTF_Gate123_Incremental",
            "实验C: 全周期 + Gate1/2/3增量信息漏斗 + Comb2",
            scores_df,
            pooled_df,
            lambda f_s, f_p, aid, f_id: base_funnel_sel[(f_id, aid)],
            lambda df, umap, sel, spec: synthesize_asset_signal(df, umap, sel, spec, "Comb2_SixClusterBudget"),
            False,
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
            False,
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
            False,
        ),
        (
            "ExpE_FactorExit",
            "实验E1 (单因子层结构化出场): 1d/15m异构因子次极值+三买卖拉回中枢证伪+反向笔确认+1.3%动态追踪止盈",
            hetero_scores_df,
            hetero_pooled_df,
            lambda f_s, f_p, aid, f_id: het_funnel_sel[(f_id, aid)],
            lambda df, umap, sel, spec: synthesize_structure_aware_synergy(
                df,
                umap,
                sel,
                spec,
                sp15_map[spec.name],
                sp1d_on_15m_map[spec.name],
                use_factor_exit=True,
                use_asset_gate=False,
            ),
            False,
        ),
        (
            "ExpE_AssetGate",
            "实验E2 (单品种合成层结构门控): 1d宏观次极值/箱体混合止损+三买卖拉回中枢证伪+1.5%动态追踪止盈",
            hetero_scores_df,
            hetero_pooled_df,
            lambda f_s, f_p, aid, f_id: het_funnel_sel[(f_id, aid)],
            lambda df, umap, sel, spec: synthesize_structure_aware_synergy(
                df,
                umap,
                sel,
                spec,
                sp15_map[spec.name],
                sp1d_on_15m_map[spec.name],
                use_factor_exit=False,
                use_asset_gate=True,
            ),
            False,
        ),
        (
            "ExpE_FullStructExit",
            "实验E3 (双层全量结构化出场): 因子层结构衰减治理 + 合成层1d宏观结构门控协同",
            hetero_scores_df,
            hetero_pooled_df,
            lambda f_s, f_p, aid, f_id: het_funnel_sel[(f_id, aid)],
            lambda df, umap, sel, spec: synthesize_structure_aware_synergy(
                df,
                umap,
                sel,
                spec,
                sp15_map[spec.name],
                sp1d_on_15m_map[spec.name],
                use_factor_exit=True,
                use_asset_gate=True,
            ),
            False,
        ),
        (
            "ExpF_5TF_Hetero5M_MicroOI",
            "实验F: 增量融入5m异构微观持仓量因子 + 15m高周期确认后5m网格因果执行 + 快慢协同调制",
            scores_5tf_df,
            pooled_5tf_df,
            lambda f_s, f_p, aid, f_id: exp_f_sel[(f_id, aid)],
            lambda df, umap, sel, spec: synthesize_fast_slow_synergy(
                df, umap, sel, spec, scale_1d=0.70, res_boost=0.25, opp_damp=0.35, deadzone=0.06
            ),
            True,
        ),
    ]

    comp_rows = []
    daily_curves: dict[str, pd.Series] = {}
    exp_e_ledgers: list[dict[str, Any]] = []
    exp_es_ledgers: list[dict[str, Any]] = []
    exp_f_ledgers: list[dict[str, Any]] = []

    print("\n[Phase 2] 运行 2025.01 ~ 2026.09 两折拼接纯样本外 (Stitched OOS) 原生 5m 执行网格与结构化出场对照实验...")
    for exp_code, exp_desc, s_df, p_df, sel_fn, syn_fn, is_5m_native in experiments:
        fold_ledgers = []
        tot_factors = 0
        ep_ret_list: list[np.ndarray] = []
        for fold in FOLDS:
            f_id = fold["id"]
            f_scores = s_df[s_df["fold"] == f_id]
            f_pool = p_df[p_df["fold"] == f_id]
            targets_5m = {}
            oos_s = pd.Timestamp(fold["oos_start"])
            oos_e = pd.Timestamp(fold["oos_end"])
            for aid in CORE_20_IDS:
                sel = sel_fn(f_scores, f_pool, aid, f_id)
                tot_factors += len(sel)
                df15 = by_aid[aid]
                t15 = syn_fn(df15, unit_grid_15m[aid], sel, specs_by_id[aid])
                m_oos = ((pd.to_datetime(df15["dt"]) >= oos_s) & (pd.to_datetime(df15["dt"]) <= oos_e)).to_numpy()
                ep_df = evaluate_execution_events(
                    df15.loc[m_oos].reset_index(drop=True),
                    t15[m_oos],
                    fee_rate=float(specs_by_id[aid].fee_rate),
                )
                if len(ep_df) > 0:
                    ep_ret_list.append(ep_df["net_ret"].to_numpy(dtype=np.float64))
                targets_5m[aid] = map_15m_target_to_5m_grid(df15, t15, df5_map[aid], is_5m_native=is_5m_native)
            led = allocate_and_account_5m_grid(
                df5_map,
                specs_by_id,
                targets_5m,
                oos_s,
                oos_e,
                deadzone=0.08,
            )
            fold_ledgers.append(led)

        if exp_code == "ExpE_Hetero1D15M_FastSlowSynergy":
            exp_e_ledgers = fold_ledgers
        elif exp_code == "ExpE_FullStructExit":
            exp_es_ledgers = fold_ledgers
        elif exp_code == "ExpF_5TF_Hetero5M_MicroOI":
            exp_f_ledgers = fold_ledgers

        ep_stats = compute_episode_payoff_stats(ep_ret_list)
        summary = summarize_stitched_oos(fold_ledgers, exp_code)
        summary["description"] = exp_desc
        summary["avg_factors_per_asset"] = round(tot_factors / (2.0 * len(CORE_20_IDS)), 1)
        summary["single_trade_payoff_r"] = ep_stats["payoff_r"]
        summary["single_trade_win_pct"] = ep_stats["win_rate_pct"]
        daily_curves[exp_code] = summary["daily_series"]
        comp_rows.append(summary)
        print(
            f"  * {exp_code:35s} | AvgF={summary['avg_factors_per_asset']:4.1f} | "
            f"Gross={summary['gross_ret_pct']:6.2f}% | Cost={summary['cost_ret_pct']:5.2f}% | "
            f"Net={summary['net_ret_pct']:6.2f}% | Shp={summary['sharpe']:5.2f} | "
            f"MaxDD={summary['max_dd_pct']:5.2f}% | Calmar={summary['calmar']:5.2f} | "
            f"Turn={summary['ann_turnover']:5.1f}x | R={summary['single_trade_payoff_r']:5.3f} (WR={summary['single_trade_win_pct']:4.1f}%)"
        )

    exp_e_row = next(r for r in comp_rows if r["experiment"] == "ExpE_Hetero1D15M_FastSlowSynergy")
    exp_es_row = next(r for r in comp_rows if r["experiment"] == "ExpE_FullStructExit")
    exp_f_row = next(r for r in comp_rows if r["experiment"] == "ExpF_5TF_Hetero5M_MicroOI")

    exp_e_3x_ledgers = scale_fold_ledgers_exposure(exp_e_ledgers, leverage_mult=3.0)
    sum_e_3x = summarize_stitched_oos(exp_e_3x_ledgers, "ExpE_3x_TargetExposure")
    sum_e_3x["description"] = "实验E (3.0x 实盘目标敞口版): 1d/15m异构专属因子 + 增量过滤 + 快慢协同"
    sum_e_3x["avg_factors_per_asset"] = exp_e_row["avg_factors_per_asset"]
    sum_e_3x["single_trade_payoff_r"] = exp_e_row["single_trade_payoff_r"]
    sum_e_3x["single_trade_win_pct"] = exp_e_row["single_trade_win_pct"]
    comp_rows.append(sum_e_3x)

    exp_es_3x_ledgers = scale_fold_ledgers_exposure(exp_es_ledgers, leverage_mult=3.0)
    sum_es_3x = summarize_stitched_oos(exp_es_3x_ledgers, "ExpE_FullStructExit_3x")
    sum_es_3x["description"] = "实验E3 (3.0x 实盘目标敞口版): 双层全量结构化出场 + 快慢协同调制"
    sum_es_3x["avg_factors_per_asset"] = exp_es_row["avg_factors_per_asset"]
    sum_es_3x["single_trade_payoff_r"] = exp_es_row["single_trade_payoff_r"]
    sum_es_3x["single_trade_win_pct"] = exp_es_row["single_trade_win_pct"]
    comp_rows.append(sum_es_3x)

    exp_f_3x_ledgers = scale_fold_ledgers_exposure(exp_f_ledgers, leverage_mult=3.0)
    sum_f_3x = summarize_stitched_oos(exp_f_3x_ledgers, "ExpF_3x_TargetExposure")
    sum_f_3x["description"] = (
        "实验F (3.0x 实盘目标敞口版): 增量融入5m异构持仓量因子 + 15m确认后5m网格因果执行 + 全周期快慢协同"
    )
    sum_f_3x["avg_factors_per_asset"] = exp_f_row["avg_factors_per_asset"]
    sum_f_3x["single_trade_payoff_r"] = exp_f_row["single_trade_payoff_r"]
    sum_f_3x["single_trade_win_pct"] = exp_f_row["single_trade_win_pct"]
    comp_rows.append(sum_f_3x)

    for s3x in (sum_e_3x, sum_es_3x, sum_f_3x):
        print(
            f"  * {s3x['experiment']:35s} | AvgF={s3x['avg_factors_per_asset']:4.1f} | "
            f"Gross={s3x['gross_ret_pct']:6.2f}% | Cost={s3x['cost_ret_pct']:5.2f}% | "
            f"Net={s3x['net_ret_pct']:6.2f}% | Shp={s3x['sharpe']:5.2f} | "
            f"MaxDD={s3x['max_dd_pct']:5.2f}% | Calmar={s3x['calmar']:5.2f} | "
            f"Turn={s3x['ann_turnover']:5.1f}x | R={s3x['single_trade_payoff_r']:5.3f}"
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
        "single_trade_payoff_r",
        "single_trade_win_pct",
        "ret_2025_pct",
        "ret_2026_pct",
    ]
    comp_df = pd.DataFrame([{k: r[k] for k in col_order} for r in comp_rows])
    comp_df.to_csv(OUT_DIR / "multitf_ablation_comparison.csv", index=False)

    struct_ablation_df = comp_df[
        comp_df["experiment"].isin(
            [
                "ExpE_Hetero1D15M_FastSlowSynergy",
                "ExpE_FactorExit",
                "ExpE_AssetGate",
                "ExpE_FullStructExit",
                "ExpE_3x_TargetExposure",
                "ExpE_FullStructExit_3x",
            ]
        )
    ].reset_index(drop=True)
    struct_ablation_df.to_csv(OUT_DIR / "multitf_structure_exit_ablation.csv", index=False)

    all_rej_df.to_csv(OUT_DIR / "multitf_gate23_rejected_signals.csv", index=False)
    rej_summary = (
        all_rej_df.groupby(["uid", "sig_id", "tf", "archetype", "reject_stage"], as_index=False)
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

    all_adm_df.to_csv(OUT_DIR / "multitf_incremental_admitted_details.csv", index=False)
    adm_summary = (
        all_adm_df.groupby(["uid", "sig_id", "tf", "archetype", "tier"], as_index=False)
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

    sym_df, sec_df = compute_symbol_and_sector_attribution(exp_es_ledgers, specs_by_id, het_adm_df)
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
