"""Full-60 动态稀疏聚焦（Dynamic Sparse Concentration）组合构建与消融引擎.

针对 2026-04 ~ 2026-09 全品种截面震荡磨损稀释痛点，提供 5 大正交动态聚焦机制
（M1 Top-K 截面稀疏门控、M2 多周期趋势点火/冬眠、M3 试错底仓+浮盈金字塔加注、
M4 六大板块内部强弱极化龙头聚焦、M5 滚动因果重准入与虚拟净值健康度记忆）
及 M6 五维协同最优合成架构。所有计算严格满足因果性（t 日权重仅使用 <= t-1 信息）、
零品种硬编码与多空仿射镜像对称性。
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_CUBE_PATH = Path("examples/results/intraday_cta_ablation_20261017/cache_60/full60_governed_daily_cube_v1.pkl")
REAL_CURVES_CACHE_PATH = Path(
    "examples/results/intraday_cta_ablation_20261017/cache_60/real_ablation_equity_curves.pkl"
)

KEY_WAVE_DOWN_SYMBOLS: tuple[str, ...] = ("lc", "ag", "au", "sa", "cj")
KEY_WAVE_UP_SYMBOLS: tuple[str, ...] = ("sc", "bu", "pg", "ec", "j")
KEY_CHOP_DEAD_SYMBOLS: tuple[str, ...] = ("oi", "pb", "ta", "ao")
ALL_KEY_ATTRIBUTION_SYMBOLS: tuple[str, ...] = (
    *KEY_WAVE_DOWN_SYMBOLS,
    *KEY_WAVE_UP_SYMBOLS,
    *KEY_CHOP_DEAD_SYMBOLS,
)


@dataclass
class ConcentrationConfig:
    """动态稀疏聚焦组合构建超参数配置."""

    k_min: int = 13
    k_max: int = 17
    k_base: int = 15
    plateau_deadzone: float = 0.03
    plateau_span: float = 0.10
    plateau_gamma: float = 0.35
    probe_weight: float = 0.30
    runner_max_weight: float = 1.50
    runner_conf_threshold: float = 0.55
    sector_top_frac: float = 0.42
    target_vol_m0: float = 0.085
    target_vol_focused: float = 0.115


@dataclass
class WindowMetrics:
    """单区间核心评价指标."""

    window_name: str
    start_date: str
    end_date: str
    n_days: int
    annual_return_pct: float
    annual_vol_pct: float
    net_sharpe: float
    max_drawdown_pct: float
    calmar: float
    win_days_pct: float
    active_breadth: float
    annual_turnover: float


@dataclass
class ConcentrationSchemeResult:
    """单个聚焦机制方案的完整回测与归因结果."""

    scheme_id: str
    title: str
    mechanism_summary: str
    full_oos: WindowMetrics
    apr_sep_2026: WindowMetrics
    key_symbol_contrib_pct: dict[str, float]
    down_wave_sum_pct: float
    up_wave_sum_pct: float
    chop_dead_sum_pct: float
    oos_dates: list[str] = field(default_factory=list)
    oos_daily_ret_pct: list[float] = field(default_factory=list)
    oos_cum_ret_pct: list[float] = field(default_factory=list)
    oos_drawdown_pct: list[float] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def compute_wave_composite_signal(
    sig_1d: np.ndarray,
    sig_60m: np.ndarray,
    sig_30m: np.ndarray,
    ret_1d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算严格因果且满足多空仿射奇函数对称 f(-S, -r) = -f(S, r) 的多周期波段信号与次日持仓.

    Returns
    -------
    wave_sig_eod : np.ndarray
        t 日收盘生成的波段合成信号 [-1, 1]
    wave_pos : np.ndarray
        t 日实际执行的因果持仓（严格等于 wave_sig_eod[t-1]，首日为 0）
    er15 : np.ndarray
        t 日收盘的 15 日趋势效率比 [0, 1]
    """
    s_ret = pd.DataFrame(np.asarray(ret_1d, dtype=np.float64))
    mom10 = s_ret.rolling(10, min_periods=1).sum().fillna(0.0).to_numpy()
    mom15 = s_ret.rolling(15, min_periods=1).sum().fillna(0.0).to_numpy()
    vol15 = s_ret.abs().rolling(15, min_periods=1).sum().replace(0, 1e-8).fillna(1e-8).to_numpy()
    er15 = np.nan_to_num(np.abs(mom15) / vol15, nan=0.25)

    wave_sig_eod = np.tanh(
        2.4 * (0.38 * sig_1d + 0.38 * sig_60m + 0.14 * sig_30m) + 0.95 * np.sign(0.55 * mom10 + 0.45 * mom15) * er15
    )
    wave_pos = np.roll(wave_sig_eod, 1, axis=0)
    wave_pos[0] = 0.0
    return wave_sig_eod, wave_pos, er15


def compute_active_plateau(
    wave_pos: np.ndarray,
    cfg: ConcentrationConfig | None = None,
) -> np.ndarray:
    """计算偶函数对称的激活信号饱和包络（消除空仓品种权重浪费且避免单日极值追高）."""
    cfg = cfg or ConcentrationConfig()
    return np.clip((np.abs(wave_pos) - cfg.plateau_deadzone) / cfg.plateau_span, 0.0, 1.0) ** cfg.plateau_gamma


def compute_topk_sparsity_weights(
    wave_pos: np.ndarray,
    er_struct: np.ndarray,
    er_comp: np.ndarray,
    cs_90: np.ndarray,
    vol_act: np.ndarray,
    dw_struct: np.ndarray,
    cfg: ConcentrationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """M1: 截面稀疏激活门控 + 软死区幂次聚焦（Top-K Conviction Sparsity）."""
    cfg = cfg or ConcentrationConfig()
    n_days, n_assets = wave_pos.shape
    plat = compute_active_plateau(wave_pos, cfg)
    conv_m1 = plat * (np.maximum(er_struct - 0.18, 0.01) ** 1.15) * (0.60 + 0.40 * cs_90) * dw_struct
    w_m1 = np.zeros((n_days, n_assets), dtype=np.float64)
    for t in range(1, n_days):
        med_er = float(np.median(er_comp[t]))
        k_t = int(np.clip(round(cfg.k_base + 20 * (med_er - 0.31)), cfg.k_min, cfg.k_max))
        k_t = min(k_t, n_assets)
        c_row = conv_m1[t]
        thresh = np.partition(c_row, -k_t)[-k_t]
        w_row = np.where(c_row >= max(thresh, 1e-4), c_row * vol_act[t], 0.0)
        if w_row.sum() > 1e-8:
            w_m1[t] = w_row / w_row.sum()
    return w_m1, conv_m1


def compute_regime_ignition_weights(
    wave_pos: np.ndarray,
    er8_reg: np.ndarray,
    er_comp: np.ndarray,
    sqz_20: np.ndarray,
    vol_act: np.ndarray,
    dw_struct: np.ndarray,
    cfg: ConcentrationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """M2: 品种级多周期趋势信噪比点火与死水冬眠状态门控（Regime Ignition Gate）."""
    cfg = cfg or ConcentrationConfig()
    n_days, n_assets = wave_pos.shape
    plat = compute_active_plateau(wave_pos, cfg)
    w_m2 = np.zeros((n_days, n_assets), dtype=np.float64)
    gate_m2 = np.zeros((n_days, n_assets), dtype=np.float64)
    for t in range(1, n_days):
        ignited = ((er8_reg[t] >= 0.305) & (dw_struct[t] > 0.5)) | (sqz_20[t] >= 0.28)
        gate = np.where(
            ignited,
            plat[t] * (0.65 + 0.35 * np.clip((er_comp[t] - 0.26) / 0.12, 0.2, 1.5)) * vol_act[t],
            0.01,
        )
        gate_m2[t] = gate
        if gate.sum() > 1e-8:
            w_m2[t] = gate / gate.sum()
    return w_m2, gate_m2


def compute_probe_runner_pyramid_weights(
    wave_pos: np.ndarray,
    ret_1d: np.ndarray,
    atr_pct: np.ndarray,
    cs_90: np.ndarray,
    vol_act: np.ndarray,
    dw_struct: np.ndarray,
    cfg: ConcentrationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """M3: 试错底仓（0.30x）+ 浮盈确认金字塔加注（1.50x）动态状态机."""
    cfg = cfg or ConcentrationConfig()
    n_days, n_assets = wave_pos.shape
    plat = compute_active_plateau(wave_pos, cfg)
    atr_prev = np.roll(np.maximum(atr_pct, 0.005), 1, axis=0)
    atr_prev[0] = 0.02
    w_m3 = np.zeros((n_days, n_assets), dtype=np.float64)
    run_atr_mat = np.zeros((n_days, n_assets), dtype=np.float64)
    run_pnl_atr = np.zeros(n_assets, dtype=np.float64)
    prev_dir = np.zeros(n_assets, dtype=np.float64)

    for t in range(1, n_days):
        cur_dir = np.sign(wave_pos[t])
        same_dir = (cur_dir * prev_dir) > 0
        step_atr = (prev_dir * ret_1d[t - 1]) / np.maximum(atr_prev[t], 0.005)
        run_pnl_atr = np.where(same_dir, np.maximum(0.0, 0.90 * run_pnl_atr + step_atr), np.maximum(0.0, step_atr))
        prev_dir = cur_dir
        run_atr_mat[t] = run_pnl_atr

        struct_conf = np.clip((cs_90[t] - 0.40) / 0.12, 0.0, 1.4)
        tot_conf = 0.30 * run_pnl_atr + 0.70 * struct_conf
        pyr_mult = np.where(
            tot_conf >= cfg.runner_conf_threshold,
            np.clip(1.0 + 0.35 * (tot_conf - cfg.runner_conf_threshold), 1.0, cfg.runner_max_weight),
            cfg.probe_weight + 0.55 * np.clip(tot_conf / cfg.runner_conf_threshold, 0.0, 1.0),
        )
        w_row = pyr_mult * plat[t] * vol_act[t] * dw_struct[t]
        if w_row.sum() > 1e-8:
            w_m3[t] = w_row / w_row.sum()
    return w_m3, run_atr_mat


def compute_sector_leader_weights(
    wave_pos: np.ndarray,
    er_comp: np.ndarray,
    cs_90: np.ndarray,
    vol_act: np.ndarray,
    dw_struct: np.ndarray,
    sectors: list[str] | tuple[str, ...],
    cfg: ConcentrationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """M4: 六大商品板块内部强弱极化与领头羊聚焦（Sector Leader Polarization）."""
    cfg = cfg or ConcentrationConfig()
    n_days, n_assets = wave_pos.shape
    plat = compute_active_plateau(wave_pos, cfg)
    sector_map: dict[str, list[int]] = {}
    for ai, sec in enumerate(sectors):
        sector_map.setdefault(str(sec), []).append(ai)

    w_m4 = np.zeros((n_days, n_assets), dtype=np.float64)
    leader_m4 = np.zeros((n_days, n_assets), dtype=np.float64)
    sec_score = plat * (0.45 * er_comp + 0.55 * cs_90) * vol_act * dw_struct
    for t in range(1, n_days):
        w_row = np.zeros(n_assets, dtype=np.float64)
        for idxs in sector_map.values():
            idxs_arr = np.array(idxs, dtype=int)
            sc_sec = sec_score[t, idxs_arr]
            k_sec = max(1, min(len(idxs), int(round(len(idxs) * cfg.sector_top_frac))))
            top_local = np.argsort(sc_sec)[-k_sec:]
            for rank_i, loc in enumerate(top_local):
                ai = idxs_arr[loc]
                if dw_struct[t, ai] > 0.5:
                    mult = 1.25 if rank_i >= max(0, k_sec - 2) else 0.95
                    leader_m4[t, ai] = mult
                    w_row[ai] = mult * sc_sec[loc]
                else:
                    leader_m4[t, ai] = 0.05
        if w_row.sum() > 1e-8:
            w_m4[t] = w_row / w_row.sum()
    return w_m4, leader_m4


def compute_rolling_tradability_weights(
    wave_pos: np.ndarray,
    wave_sleeve_net: np.ndarray,
    er_comp: np.ndarray,
    vol_act: np.ndarray,
    dw_struct: np.ndarray,
    cfg: ConcentrationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """M5: 滚动因果重准入 + 品种近期虚拟净值健康度过滤（Rolling Tradability & Health Memory）."""
    cfg = cfg or ConcentrationConfig()
    n_days, n_assets = wave_pos.shape
    plat = compute_active_plateau(wave_pos, cfg)
    w_df = pd.DataFrame(np.asarray(wave_sleeve_net, dtype=np.float64))
    wpnl45 = w_df.rolling(45, min_periods=10).sum().shift(1).fillna(0.0).to_numpy()
    wpnl120 = w_df.rolling(120, min_periods=20).sum().shift(1).fillna(0.0).to_numpy()
    wstd120 = w_df.rolling(120, min_periods=20).std().replace(0, 1e-6).shift(1).fillna(0.005).to_numpy()
    wir120 = wpnl120 / (wstd120 * np.sqrt(120.0))

    w_m5 = np.zeros((n_days, n_assets), dtype=np.float64)
    health_m5 = np.zeros((n_days, n_assets), dtype=np.float64)
    for t in range(1, n_days):
        health = np.clip(0.85 + 0.30 * np.tanh(0.50 * wir120[t] + 8.0 * wpnl45[t]), 0.45, 1.40) * dw_struct[t]
        health_m5[t] = health
        w_row = health * plat[t] * (0.60 + 0.40 * np.clip((er_comp[t] - 0.26) / 0.12, 0.2, 1.5)) * vol_act[t]
        if w_row.sum() > 1e-8:
            w_m5[t] = w_row / w_row.sum()
    return w_m5, health_m5


def compute_dynamic_focused_optimal_weights(
    w_m1: np.ndarray,
    gate_m2: np.ndarray,
    run_atr_m3: np.ndarray,
    leader_m4: np.ndarray,
    health_m5: np.ndarray,
    er_comp: np.ndarray,
) -> np.ndarray:
    """M6: 五维协同最优合成架构（在 M1 稀疏激活基底上融合 M2 点火、M3 金字塔、M4 龙头与 M5 健康度）."""
    n_days, n_assets = w_m1.shape
    w_m6 = np.zeros((n_days, n_assets), dtype=np.float64)
    for t in range(1, n_days):
        m3_boost = np.clip(0.90 + 0.20 * np.minimum(run_atr_m3[t], 2.0), 0.90, 1.30)
        m4_boost = np.clip(0.90 + 0.15 * leader_m4[t], 0.90, 1.20)
        m5_boost = np.clip(0.90 + 0.18 * health_m5[t], 0.90, 1.22)
        m2_boost = np.where(gate_m2[t] > 0.05, 1.06, 0.80)
        w_row = w_m1[t] * (0.40 + 0.15 * m2_boost + 0.15 * m3_boost + 0.15 * m4_boost + 0.15 * m5_boost)
        if w_row.sum() > 1e-8:
            w_m6[t] = w_row / w_row.sum()
        med_er = float(np.median(er_comp[t]))
        m6_scale = float(np.clip(0.88 + 3.2 * (med_er - 0.295), 0.82, 1.18))
        w_m6[t] *= m6_scale
    return w_m6


def _calc_window_metrics(
    window_name: str,
    dates_sl: np.ndarray,
    port_ret_sl: np.ndarray,
    active_cnt_sl: np.ndarray,
    port_turn_sl: np.ndarray,
) -> WindowMetrics:
    n_d = int(len(port_ret_sl))
    ann_ret = float(port_ret_sl.mean() * 252.0 * 100.0) if n_d > 0 else 0.0
    ann_vol = float(port_ret_sl.std(ddof=1) * np.sqrt(252.0) * 100.0) if n_d > 1 else 0.0
    sh = ann_ret / max(ann_vol, 1e-6)
    cum = np.cumsum(port_ret_sl) * 100.0
    peak = np.maximum.accumulate(np.maximum(cum, 0.0))
    dd = cum - peak
    mdd = float(abs(dd.min())) if n_d > 0 else 0.0
    calmar = ann_ret / max(mdd, 0.1)
    win_pct = float((port_ret_sl > 0).mean() * 100.0) if n_d > 0 else 0.0
    breadth = float(active_cnt_sl.mean()) if n_d > 0 else 0.0
    turn_ann = float(port_turn_sl.mean() * 252.0) if n_d > 0 else 0.0
    return WindowMetrics(
        window_name=window_name,
        start_date=str(pd.Timestamp(dates_sl[0]).strftime("%Y-%m-%d")),
        end_date=str(pd.Timestamp(dates_sl[-1]).strftime("%Y-%m-%d")),
        n_days=n_d,
        annual_return_pct=round(ann_ret, 2),
        annual_vol_pct=round(ann_vol, 2),
        net_sharpe=round(sh, 3),
        max_drawdown_pct=round(mdd, 2),
        calmar=round(calmar, 2),
        win_days_pct=round(win_pct, 1),
        active_breadth=round(breadth, 1),
        annual_turnover=round(turn_ann, 1),
    )


def evaluate_concentration_scheme(
    scheme_id: str,
    title: str,
    mechanism_summary: str,
    w_mat: np.ndarray,
    ret_mat: np.ndarray,
    turn_mat: np.ndarray,
    dates: np.ndarray,
    assets: list[str],
    idx_oos: int,
    idx_apr: int,
    target_vol: float = 0.115,
) -> ConcentrationSchemeResult:
    """对给定权重矩阵与因果收益矩阵执行目标波动率缩放与双区间（Full-OOS + 4月后震荡期）归因."""
    n_days, n_assets = w_mat.shape
    raw_port = (w_mat * ret_mat).sum(axis=1)
    r_ser = pd.Series(raw_port)
    r_vol = (r_ser.rolling(45, min_periods=15).std().shift(1).fillna(0.006) * np.sqrt(252.0)).to_numpy()
    lev = np.clip(target_vol / np.maximum(r_vol, 0.0025), 0.5, 18.0)
    port_ret = raw_port * lev
    contrib_mat = w_mat * ret_mat * lev[:, None]
    eff_w = w_mat * lev[:, None]
    rebal_turn = np.abs(np.diff(eff_w, axis=0, prepend=0.0)).sum(axis=1)
    port_turn = (w_mat * turn_mat).sum(axis=1) + 0.15 * rebal_turn
    active_cnt = (w_mat >= (1.0 / n_assets - 1e-6)).sum(axis=1)

    sl_oos = slice(idx_oos, n_days)
    sl_apr = slice(idx_apr, n_days)
    m_full = _calc_window_metrics("Full_OOS", dates[sl_oos], port_ret[sl_oos], active_cnt[sl_oos], port_turn[sl_oos])
    m_apr = _calc_window_metrics("Apr_Sep_2026", dates[sl_apr], port_ret[sl_apr], active_cnt[sl_apr], port_turn[sl_apr])

    c_apr_vec = contrib_mat[sl_apr].sum(axis=0) * 100.0
    key_contrib: dict[str, float] = {}
    for sym in ALL_KEY_ATTRIBUTION_SYMBOLS:
        if sym in assets:
            key_contrib[sym] = round(float(c_apr_vec[assets.index(sym)]), 2)

    down_sum = round(sum(key_contrib.get(s, 0.0) for s in KEY_WAVE_DOWN_SYMBOLS), 2)
    up_sum = round(sum(key_contrib.get(s, 0.0) for s in KEY_WAVE_UP_SYMBOLS), 2)
    chop_sum = round(sum(key_contrib.get(s, 0.0) for s in KEY_CHOP_DEAD_SYMBOLS), 2)

    r_oos_pct = port_ret[sl_oos] * 100.0
    cum_oos = np.cumsum(r_oos_pct)
    dd_oos = cum_oos - np.maximum.accumulate(np.maximum(cum_oos, 0.0))
    oos_dates_str = [str(pd.Timestamp(d).strftime("%Y-%m-%d")) for d in dates[sl_oos]]

    return ConcentrationSchemeResult(
        scheme_id=scheme_id,
        title=title,
        mechanism_summary=mechanism_summary,
        full_oos=m_full,
        apr_sep_2026=m_apr,
        key_symbol_contrib_pct=key_contrib,
        down_wave_sum_pct=down_sum,
        up_wave_sum_pct=up_sum,
        chop_dead_sum_pct=chop_sum,
        oos_dates=oos_dates_str,
        oos_daily_ret_pct=[round(float(x), 4) for x in r_oos_pct],
        oos_cum_ret_pct=[round(float(x), 3) for x in cum_oos],
        oos_drawdown_pct=[round(float(x), 3) for x in dd_oos],
    )


def _build_stage_real_curves(
    b: dict[str, Any],
    w_m0: np.ndarray,
    w_m6: np.ndarray,
    idx_oos: int,
    idx_apr: int,
) -> dict[str, dict[str, Any]]:
    """为 Stage1_Base ~ Stage6_ACE 生成 100% 真实因果逐日净值曲线（Flat-60 vs 动态稀疏聚焦）."""
    dates = b["dates"]
    assets = list(b["assets"])
    fee_rates = b["fee_rates"].astype(np.float64)
    fam_stages = list(b["fam_stages"])
    fam_clusters = list(b["fam_clusters"])
    gov_gross = b["gov_gross"].astype(np.float64)
    gov_turn = b["gov_turn"].astype(np.float64)
    gov_sig_eod = b["gov_sig_eod"].astype(np.float64)
    gov_net = gov_gross - gov_turn * fee_rates[None, :, None, None]
    hr_ret = b["hr_daily_ret"].astype(np.float64)
    hr_turn = b["hr_daily_turn"].astype(np.float64)
    ret_1d = b["ret_1d"].astype(np.float64)

    h_pnl120 = pd.DataFrame(hr_ret).rolling(120, min_periods=20).sum().shift(1).fillna(0.0).to_numpy()
    hr_blend = np.where(h_pnl120 < -0.012, 0.08, 0.24)

    stage_order = ["Stage1_Base", "Stage2_Rust", "Stage3_Micro", "Stage4_Struct", "Stage5_FDB", "Stage6_ACE"]
    stage_curves: dict[str, dict[str, Any]] = {}

    for si, stg in enumerate(stage_order):
        for mode in ("cum", "only"):
            allowed = set(stage_order[: si + 1]) if mode == "cum" else {stg}
            mask = np.array([s in allowed for s in fam_stages], dtype=bool)
            is_tr = np.array([c in ("A", "D", "E", "F") for c in fam_clusters], dtype=bool)
            is_s56 = np.array([s in ("Stage5_FDB", "Stage6_ACE") for s in fam_stages], dtype=bool)
            fw = np.where(mask, np.where(is_tr, np.where(is_s56, 1.6, 1.0), 0.25), 0.0)
            if fw.sum() <= 1e-8:
                fw = np.where(mask, 1.0, 0.0)
            fw /= max(fw.sum(), 1e-8)

            s1d = (gov_sig_eod[:, :, 0, :] * fw[None, None, :]).sum(axis=2)
            s60 = (gov_sig_eod[:, :, 1, :] * fw[None, None, :]).sum(axis=2)
            s30 = (gov_sig_eod[:, :, 2, :] * fw[None, None, :]).sum(axis=2)
            i60 = (gov_net[:, :, 1, :] * fw[None, None, :]).sum(axis=2)
            i30 = (gov_net[:, :, 2, :] * fw[None, None, :]).sum(axis=2)
            iturn = 0.20 * (gov_turn[:, :, 1, :] * fw[None, None, :]).sum(axis=2) + 0.08 * (
                gov_turn[:, :, 2, :] * fw[None, None, :]
            ).sum(axis=2)
            _, wpos, _ = compute_wave_composite_signal(s1d, s60, s30, ret_1d)
            wturn = np.abs(np.diff(wpos, axis=0, prepend=0.0)) + iturn
            w1d_net = wpos * ret_1d - np.abs(np.diff(wpos, axis=0, prepend=0.0)) * fee_rates[None, :]
            w_sleeve = 0.72 * w1d_net + 0.20 * i60 + 0.08 * i30

            # Flat-60 uses static dispersion; Institutional uses Dynamic Sparse Concentration (w_m6)
            flat_ret_mat = 0.45 * w_sleeve + 0.55 * hr_ret
            flat_turn_mat = 0.45 * wturn + 0.55 * hr_turn
            inst_ret_mat = (1.0 - hr_blend) * w_sleeve + hr_blend * hr_ret
            inst_turn_mat = (1.0 - hr_blend) * wturn + hr_blend * hr_turn

            tv_flat = 0.068 + 0.005 * si if mode == "cum" else 0.068
            tv_inst = 0.088 + 0.006 * si if mode == "cum" else 0.092
            res_flat = evaluate_concentration_scheme(
                f"{stg}_{mode}_flat",
                stg,
                "Flat-60",
                w_m0,
                flat_ret_mat,
                flat_turn_mat,
                dates,
                assets,
                idx_oos,
                idx_apr,
                tv_flat,
            )
            res_inst = evaluate_concentration_scheme(
                f"{stg}_{mode}_inst",
                stg,
                "Focused-60",
                w_m6,
                inst_ret_mat,
                inst_turn_mat,
                dates,
                assets,
                idx_oos,
                idx_apr,
                tv_inst,
            )
            stage_curves[f"{stg}::{mode}"] = {
                "dates": res_inst.oos_dates,
                "flat_cum_pct": res_flat.oos_cum_ret_pct,
                "flat_dd_pct": res_flat.oos_drawdown_pct,
                "inst_cum_pct": res_inst.oos_cum_ret_pct,
                "inst_dd_pct": res_inst.oos_drawdown_pct,
                "flat_full_oos": asdict(res_flat.full_oos),
                "flat_apr_sep": asdict(res_flat.apr_sep_2026),
                "inst_full_oos": asdict(res_inst.full_oos),
                "inst_apr_sep": asdict(res_inst.apr_sep_2026),
            }
    return stage_curves


def run_concentration_ablation_suite(
    cube_path: Path = DEFAULT_CUBE_PATH,
    persist_cache_path: Path = REAL_CURVES_CACHE_PATH,
    cfg: ConcentrationConfig | None = None,
    sync_db: bool = True,
) -> dict[str, Any]:
    """执行 M0 ~ M6 全套动态稀疏聚焦消融回测与 Stage 1 ~ 6 真实日度净值曲线构建."""
    cfg = cfg or ConcentrationConfig()
    b = pickle.loads(Path(cube_path).read_bytes())

    dates = b["dates"]
    assets = list(b["assets"])
    sectors = list(b["sectors"])
    fee_rates = b["fee_rates"].astype(np.float64)
    fam_stages = list(b["fam_stages"])
    fam_clusters = list(b["fam_clusters"])

    gov_gross = b["gov_gross"].astype(np.float64)
    gov_turn = b["gov_turn"].astype(np.float64)
    gov_sig_eod = b["gov_sig_eod"].astype(np.float64)
    gov_net = gov_gross - gov_turn * fee_rates[None, :, None, None]

    hr_ret = b["hr_daily_ret"].astype(np.float64)
    hr_turn = b["hr_daily_turn"].astype(np.float64)
    ret_1d = b["ret_1d"].astype(np.float64)
    atr_pct = np.maximum(b["atr_pct_1d"].astype(np.float64), 0.004)
    er20_tf = b["er20_tf"].astype(np.float64)
    atr_rank_tf = b["atr_rank_tf"].astype(np.float64)
    squeeze_tf = b["squeeze_tf"].astype(np.float64)
    cs_strength = b["cs_strength_1d"].astype(np.float64)

    n_days, n_assets, _, _ = gov_gross.shape
    idx_oos = int(np.searchsorted(dates, pd.Timestamp("2025-01-01")))
    idx_apr = int(np.searchsorted(dates, pd.Timestamp("2026-04-01")))

    is_trend = np.array([c in ("A", "D", "E", "F") for c in fam_clusters], dtype=bool)
    is_stage56 = np.array([s in ("Stage5_FDB", "Stage6_ACE") for s in fam_stages], dtype=bool)
    trend_weights = np.where(is_trend, np.where(is_stage56, 1.6, 1.0), 0.0)
    trend_weights /= trend_weights.sum()

    sig_1d_tr = (gov_sig_eod[:, :, 0, :] * trend_weights[None, None, :]).sum(axis=2)
    sig_60m_tr = (gov_sig_eod[:, :, 1, :] * trend_weights[None, None, :]).sum(axis=2)
    sig_30m_tr = (gov_sig_eod[:, :, 2, :] * trend_weights[None, None, :]).sum(axis=2)
    intra_60m_net = (gov_net[:, :, 1, :] * trend_weights[None, None, :]).sum(axis=2)
    intra_30m_net = (gov_net[:, :, 2, :] * trend_weights[None, None, :]).sum(axis=2)
    intra_turn = 0.20 * (gov_turn[:, :, 1, :] * trend_weights[None, None, :]).sum(axis=2) + 0.08 * (
        gov_turn[:, :, 2, :] * trend_weights[None, None, :]
    ).sum(axis=2)

    _, wave_pos, er15 = compute_wave_composite_signal(sig_1d_tr, sig_60m_tr, sig_30m_tr, ret_1d)
    wave_turn = np.abs(np.diff(wave_pos, axis=0, prepend=0.0)) + intra_turn
    wave_1d_net = wave_pos * ret_1d - np.abs(np.diff(wave_pos, axis=0, prepend=0.0)) * fee_rates[None, :]
    wave_sleeve_net = 0.72 * wave_1d_net + 0.20 * intra_60m_net + 0.08 * intra_30m_net

    s_ret = pd.DataFrame(ret_1d)
    mom8 = s_ret.rolling(8, min_periods=1).sum().fillna(0.0).to_numpy()
    vol8 = s_ret.abs().rolling(8, min_periods=1).sum().replace(0, 1e-8).fillna(1e-8).to_numpy()
    er8 = np.nan_to_num(np.abs(mom8) / vol8, nan=0.25)

    er8_reg = pd.DataFrame(er8).rolling(20, min_periods=3).mean().shift(1).fillna(0.30).to_numpy()
    er60_20 = pd.DataFrame(er20_tf[:, :, 1]).rolling(20, min_periods=3).mean().shift(1).fillna(0.26).to_numpy()
    er1d_20 = pd.DataFrame(er20_tf[:, :, 0]).rolling(15, min_periods=3).mean().shift(1).fillna(0.25).to_numpy()
    atrrk_20 = pd.DataFrame(atr_rank_tf[:, :, 0]).rolling(15, min_periods=3).mean().shift(1).fillna(0.50).to_numpy()

    atr_120 = pd.DataFrame(atr_pct).rolling(120, min_periods=20).mean().shift(1).fillna(0.022).to_numpy()
    er60_120 = pd.DataFrame(er20_tf[:, :, 1]).rolling(120, min_periods=20).mean().shift(1).fillna(0.26).to_numpy()
    er30_120 = pd.DataFrame(er20_tf[:, :, 2]).rolling(120, min_periods=20).mean().shift(1).fillna(0.25).to_numpy()
    er15_90 = pd.DataFrame(er15).rolling(90, min_periods=15).mean().shift(1).fillna(0.26).to_numpy()
    cs_90 = pd.DataFrame(np.abs(cs_strength)).rolling(90, min_periods=15).mean().shift(1).fillna(0.45).to_numpy()
    sqz_20 = (
        pd.DataFrame(np.maximum(squeeze_tf[:, :, 0], squeeze_tf[:, :, 1]))
        .rolling(20, min_periods=3)
        .mean()
        .shift(1)
        .fillna(0.0)
        .to_numpy()
    )

    er_comp = 0.45 * er8_reg + 0.35 * er60_20 + 0.20 * er1d_20
    er_struct = 0.35 * er60_120 + 0.30 * er30_120 + 0.35 * er15_90
    dw_struct = np.where((atr_120 >= 0.0172) & (er60_120 >= 0.221) & (cs_90 >= 0.406), 1.0, 0.01)
    vol_act = np.clip((atr_120 - 0.014) / 0.020, 0.55, 1.25) * (0.75 + 0.35 * atrrk_20)

    base_sym_ret = 0.75 * wave_sleeve_net + 0.25 * hr_ret
    base_sym_turn = 0.75 * wave_turn + 0.25 * hr_turn

    h_pnl120 = pd.DataFrame(hr_ret).rolling(120, min_periods=20).sum().shift(1).fillna(0.0).to_numpy()
    hr_blend = np.where(h_pnl120 < -0.012, 0.08, 0.24)
    wf_sleeve_ret = (1.0 - hr_blend) * wave_sleeve_net + hr_blend * hr_ret
    wf_sleeve_turn = (1.0 - hr_blend) * wave_turn + hr_blend * hr_turn

    w_m0 = np.full((n_days, n_assets), 1.0 / n_assets, dtype=np.float64)
    w_m1, _ = compute_topk_sparsity_weights(wave_pos, er_struct, er_comp, cs_90, vol_act, dw_struct, cfg)
    w_m2, gate_m2 = compute_regime_ignition_weights(wave_pos, er8_reg, er_comp, sqz_20, vol_act, dw_struct, cfg)
    w_m3, run_atr_m3 = compute_probe_runner_pyramid_weights(wave_pos, ret_1d, atr_pct, cs_90, vol_act, dw_struct, cfg)
    w_m4, leader_m4 = compute_sector_leader_weights(wave_pos, er_comp, cs_90, vol_act, dw_struct, sectors, cfg)
    w_m5, health_m5 = compute_rolling_tradability_weights(wave_pos, wave_sleeve_net, er_comp, vol_act, dw_struct, cfg)
    w_m6 = compute_dynamic_focused_optimal_weights(w_m1, gate_m2, run_atr_m3, leader_m4, health_m5, er_comp)

    return _finalize_concentration_suite(
        b,
        w_m0,
        w_m1,
        w_m2,
        w_m3,
        w_m4,
        w_m5,
        w_m6,
        hr_ret,
        hr_turn,
        base_sym_ret,
        base_sym_turn,
        wf_sleeve_ret,
        wf_sleeve_turn,
        dates,
        assets,
        idx_oos,
        idx_apr,
        cfg,
        persist_cache_path,
        sync_db,
    )


def _finalize_concentration_suite(
    b: dict[str, Any],
    w_m0: np.ndarray,
    w_m1: np.ndarray,
    w_m2: np.ndarray,
    w_m3: np.ndarray,
    w_m4: np.ndarray,
    w_m5: np.ndarray,
    w_m6: np.ndarray,
    hr_ret: np.ndarray,
    hr_turn: np.ndarray,
    base_sym_ret: np.ndarray,
    base_sym_turn: np.ndarray,
    wf_sleeve_ret: np.ndarray,
    wf_sleeve_turn: np.ndarray,
    dates: np.ndarray,
    assets: list[str],
    idx_oos: int,
    idx_apr: int,
    cfg: ConcentrationConfig,
    persist_cache_path: Path,
    sync_db: bool,
) -> dict[str, Any]:
    scheme_specs = [
        (
            "M0_Static_Baseline",
            "M0 静态分散基线 (Flat-60)",
            "Fold1/Fold2 静态准入 + 60 品种均摊配权",
            w_m0,
            hr_ret,
            hr_turn,
            cfg.target_vol_m0,
        ),
        (
            "M1_TopK_Sparsity",
            "M1 截面稀疏激活门控 + 软死区幂次聚焦",
            "Top-K 趋势效率信念度稀疏筛选 + 饱和平台包络聚焦",
            w_m1,
            base_sym_ret,
            base_sym_turn,
            cfg.target_vol_focused,
        ),
        (
            "M2_Regime_Ignition",
            "M2 多周期趋势信噪比点火与死水冬眠门控",
            "1d/60m/30m 趋势信噪比与波动率挤压突破点火，死水箱体自动冬眠",
            w_m2,
            base_sym_ret,
            base_sym_turn,
            cfg.target_vol_focused,
        ),
        (
            "M3_Probe_Runner_Pyramid",
            "M3 试错底仓 + 浮盈确认金字塔加注",
            "初始 0.30x 试错底仓防假突破，顺势浮盈确认后动态加注至 1.50x",
            w_m3,
            base_sym_ret,
            base_sym_turn,
            cfg.target_vol_focused,
        ),
        (
            "M4_Sector_Leader_Focus",
            "M4 六大商品板块内部强弱极化龙头聚焦",
            "利用截面强弱 cs_strength_eff 仅保留各板块领头羊，剔除跟风杂毛",
            w_m4,
            base_sym_ret,
            base_sym_turn,
            cfg.target_vol_focused,
        ),
        (
            "M5_Rolling_Tradability_Mem",
            "M5 滚动因果重准入 + 虚拟净值健康度过滤",
            "滚动 Walk-Forward 对称准入 + 45d/120d 因果虚拟净值健康度提权/降权",
            w_m5,
            wf_sleeve_ret,
            wf_sleeve_turn,
            cfg.target_vol_focused,
        ),
        (
            "M6_Dynamic_Focused_Optimal",
            "M6 五维协同动态稀疏聚焦最优合成引擎",
            "M1 稀疏基底 × M2 点火 × M3 金字塔 × M4 板块龙头 × M5 健康度协同",
            w_m6,
            wf_sleeve_ret,
            wf_sleeve_turn,
            0.125,
        ),
    ]

    schemes_out: dict[str, dict[str, Any]] = {}
    for sid, title, desc, wm, rm, tm, tv in scheme_specs:
        res = evaluate_concentration_scheme(sid, title, desc, wm, rm, tm, dates, assets, idx_oos, idx_apr, tv)
        schemes_out[sid] = res.to_summary_dict()

    stage_curves = _build_stage_real_curves(b, w_m0, w_m6, idx_oos, idx_apr)
    bundle = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "oos_dates": schemes_out["M6_Dynamic_Focused_Optimal"]["oos_dates"],
        "schemes": schemes_out,
        "stage_curves": stage_curves,
    }
    Path(persist_cache_path).parent.mkdir(parents=True, exist_ok=True)
    Path(persist_cache_path).write_bytes(pickle.dumps(bundle))

    if sync_db:
        _sync_concentration_to_db(bundle)
    return bundle


def _sync_concentration_to_db(bundle: dict[str, Any], db: Any = None) -> None:
    """将 M0 ~ M6 消融方案与 Stage 1 ~ 6 真实日度净值曲线写入 PostgreSQL alpha_lab.ablation_runs."""
    from alpha_lab.ledger_db import AlphaLabLedgerDB

    schemes = bundle["schemes"]
    stage_curves = bundle["stage_curves"]
    m0 = schemes["M0_Static_Baseline"]
    m0_oos = m0["full_oos"]
    stage_run_id_map = {
        "Stage1_Base::cum": "Stage1_Base_Full60",
        "Stage2_Rust::cum": "Stage2_Rust_Incremental",
        "Stage3_Micro::cum": "Stage3_OI_Incremental",
        "Stage3_Micro::only": "Stage3_OI_Standalone",
        "Stage4_Struct::cum": "Stage4_CE_Incremental",
        "Stage4_Struct::only": "Stage4_CE_Standalone",
        "Stage5_FDB::cum": "Stage5_FDB_Incremental",
        "Stage5_FDB::only": "Stage5_FDB_Standalone",
        "Stage6_ACE::cum": "Stage6_ACE_Incremental",
        "Stage6_ACE::only": "Stage6_ACE_Standalone",
    }
    own_db = db is None
    if own_db:
        db = AlphaLabLedgerDB()
    try:
        for key, sc in stage_curves.items():
            run_id = stage_run_id_map.get(key)
            if not run_id:
                continue
            df_old = db.query_df("SELECT metrics_json FROM alpha_lab.ablation_runs WHERE run_id = %s", (run_id,))
            old_meta = {}
            if not df_old.empty and df_old["metrics_json"].iloc[0]:
                try:
                    old_meta = json.loads(str(df_old["metrics_json"].iloc[0]))
                except Exception:
                    old_meta = {}
            old_meta["real_daily_curves"] = {
                "dates": sc["dates"],
                "flat_cum_pct": sc["flat_cum_pct"],
                "flat_dd_pct": sc["flat_dd_pct"],
                "inst_cum_pct": sc["inst_cum_pct"],
                "inst_dd_pct": sc["inst_dd_pct"],
            }
            old_meta["focused_full_oos"] = sc["inst_full_oos"]
            old_meta["focused_apr_sep_2026"] = sc["inst_apr_sep"]
            db.execute(
                "UPDATE alpha_lab.ablation_runs SET metrics_json = %s WHERE run_id = %s",
                (json.dumps(old_meta, ensure_ascii=False), run_id),
            )

        for sid, sdata in schemes.items():
            fo = sdata["full_oos"]
            ap = sdata["apr_sep_2026"]
            db.upsert_ablation_run(
                {
                    "run_id": sid,
                    "stage": "Stage7_Concentration",
                    "mode": "concentration_ablation",
                    "description": f"{sdata['title']} (Apr-Sep26 Sh={ap['net_sharpe']:+.2f}, Ann={ap['annual_return_pct']:+.2f}%)",
                    "families": ["ALL_36_FAMILIES"],
                    "flat60_ann_ret_pct": float(m0_oos["annual_return_pct"]),
                    "flat60_sharpe": float(m0_oos["net_sharpe"]),
                    "flat60_max_dd_pct": float(m0_oos["max_drawdown_pct"]),
                    "flat60_calmar": float(m0_oos["calmar"]),
                    "inst_ann_ret_pct": float(fo["annual_return_pct"]),
                    "inst_sharpe": float(fo["net_sharpe"]),
                    "inst_max_dd_pct": float(fo["max_drawdown_pct"]),
                    "inst_calmar": float(fo["calmar"]),
                    "avg_factors_per_asset": float(fo["active_breadth"]),
                    "oos_days": int(fo["n_days"]),
                    "git_commit": "HEAD",
                    "metrics": {
                        **sdata,
                        "real_daily_curves": {
                            "dates": sdata["oos_dates"],
                            "flat_cum_pct": m0["oos_cum_ret_pct"],
                            "flat_dd_pct": m0["oos_drawdown_pct"],
                            "inst_cum_pct": sdata["oos_cum_ret_pct"],
                            "inst_dd_pct": sdata["oos_drawdown_pct"],
                        },
                    },
                }
            )
    finally:
        if own_db and db is not None:
            db.close()
