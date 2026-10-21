"""商品期货 CTA 品种池筛选研究脚本：从 Full-60 动态过滤至 30~35 核心交易池的四层因果漏斗实证。

核心流程（严格遵循 <= cutoff 因果训练期评估，零先验硬编码）：
1. Gate 1（流动性与上市成熟度底线）：剔除上市不足 160 交易日及成交量/成交额垫底的冷门品种；
2. Gate 2（价格微观趋势信噪比 KER）：计算 20d/60d 考夫曼效率比与日内实体占比，过滤高频震荡品种；
3. Gate 3（样本内双模式稳健正期望）：结合快慢线合成与滞回止盈合成的样本内夏普、盈亏比与安全边际；
4. Gate 4（同板块产业链强共线去重 + 簇内风险分摊 + 跨期 Buffer 滞回）：输出 30~35 个核心交易品种。
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))
STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from run_full_60_commodities_study import (  # noqa: E402
    FOLDS,
    IDS_60,
    SECTORS_60,
    SPECS_60,
    _execute_sector_conserved_allocation,
    compute_episode_payoff_stats,
    compute_is_liquidity_and_price_returns,
    summarize_stitched_oos,
)

from examples.multi_factor_cta.synthesis_engine import (  # noqa: E402
    compute_causal_sleeve_health_multiplier,
    compute_continuous_cluster_risk_shares,
)


def compute_is_microstructure_snr(frames_1d: dict) -> dict[tuple[str, str], dict[str, float]]:
    """在每个 Fold 训练截止日前 (<= cutoff) 计算品种日线趋势效率比 (KER) 与信噪比特征。"""
    snr_stats: dict[tuple[str, str], dict[str, float]] = {}
    for aid in IDS_60:
        df1 = frames_1d[SPECS_60[aid].symbol].copy()
        df1["dt"] = pd.to_datetime(df1["dt"]).dt.normalize()
        df1 = df1.sort_values("dt").reset_index(drop=True)
        for fold in FOLDS:
            f_id = fold["id"]
            cut = pd.Timestamp(fold["cutoff"]).normalize()
            sub = df1.loc[df1["dt"] <= cut].reset_index(drop=True)
            if len(sub) < 30:
                snr_stats[(f_id, aid)] = {
                    "ker_20": 0.0,
                    "ker_60": 0.0,
                    "intraday_body_ratio": 0.0,
                    "ann_vol": 0.0,
                }
                continue
            close = sub["close"].astype(float)
            diff_abs = close.diff().abs()
            ker20 = ((close - close.shift(20)).abs() / diff_abs.rolling(20).sum().replace(0, np.nan)).dropna()
            ker60 = ((close - close.shift(60)).abs() / diff_abs.rolling(60).sum().replace(0, np.nan)).dropna()
            hl_range = (sub["high"].astype(float) - sub["low"].astype(float)).replace(0, np.nan)
            body_ratio = ((sub["close"].astype(float) - sub["open"].astype(float)).abs() / hl_range).dropna()
            ret = close.pct_change().dropna()
            snr_stats[(f_id, aid)] = {
                "ker_20": float(ker20.mean()) if len(ker20) else 0.0,
                "ker_60": float(ker60.mean()) if len(ker60) else 0.0,
                "intraday_body_ratio": float(body_ratio.mean()) if len(body_ratio) else 0.0,
                "ann_vol": float(ret.std() * np.sqrt(252)) if len(ret) else 0.0,
            }
    return snr_stats


def zscore_dict(d: dict[str, float]) -> dict[str, float]:
    """计算字典数值的横截面 Z-Score 并截断在 [-2.5, 2.5]。"""
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=float)
    mu, sd = float(np.mean(vals)), float(np.std(vals))
    if sd < 1e-9:
        return dict.fromkeys(d, 0.0)
    return {k: float(np.clip((v - mu) / sd, -2.5, 2.5)) for k, v in d.items()}


def sel_all_60(f_id, pit, meta, liq, snr, cp, cs, prev):
    """基准0：全品种无过滤。"""
    return list(pit), dict.fromkeys(pit, 1.0)


def sel_top32_liquidity(f_id, pit, meta, liq, snr, cp, cs, prev):
    """方案1：仅按样本内日均成交手数与成交额复合排名取前 32。"""
    z_vol = zscore_dict({a: np.log1p(liq[(f_id, a)]["is_vol_wan"]) for a in pit})
    z_amt = zscore_dict({a: np.log1p(liq[(f_id, a)]["is_amt_yi"]) for a in pit})
    sc = {a: 0.5 * z_vol[a] + 0.5 * z_amt[a] for a in pit}
    ranked = sorted(pit, key=lambda a: sc[a], reverse=True)[:32]
    return ranked, {a: float(np.exp(0.25 * sc[a])) for a in ranked}


def make_institutional_funnel_selector(
    target_n: int = 32,
    min_vol_wan: float = 4.0,
    min_amt_yi: float = 35.0,
    min_nf: int = 3,
    min_shp_hr: float = 0.35,
    min_r: float = 1.35,
    w_shp: float = 0.45,
    w_sm: float = 0.25,
    w_ker: float = 0.15,
    w_liq: float = 0.15,
    max_per_corr_cluster: int = 2,
    corr_thresh: float = 0.78,
    buffer_bonus: float = 0.15,
):
    """构建机构四层漏斗筛选器（流动性底线 + KER趋势信噪比 + 双模式稳健正期望 + 同板块强共线去重）。"""

    def _selector(f_id, pit, meta, liq, snr, cp, cs, prev):
        cands = []
        for a in pit:
            m = meta[(f_id, a)]
            lq = liq[(f_id, a)]
            if lq["is_vol_wan"] < min_vol_wan and lq["is_amt_yi"] < min_amt_yi:
                continue
            if m["n_factors"] < min_nf or m["is_ep_count"] < 15:
                continue
            eff_shp = max(float(m["is_shp_hr"]), 0.85 * float(m["is_shp_fs"]))
            if eff_shp < min_shp_hr or m["is_payoff_r"] < min_r:
                continue
            cands.append(a)

        if len(cands) < target_n:
            fallback = [
                a
                for a in pit
                if a not in cands
                and (liq[(f_id, a)]["is_vol_wan"] >= min_vol_wan or liq[(f_id, a)]["is_amt_yi"] >= min_amt_yi)
                and meta[(f_id, a)]["n_factors"] >= 1
            ]
            fallback.sort(
                key=lambda a: 0.5 * meta[(f_id, a)]["is_shp_hr"] + 0.5 * snr[(f_id, a)]["ker_20"],
                reverse=True,
            )
            cands.extend(fallback[: target_n + 6 - len(cands)])

        z_shp = zscore_dict(
            {a: 0.55 * meta[(f_id, a)]["is_shp_hr"] + 0.45 * meta[(f_id, a)]["is_shp_fs"] for a in cands}
        )
        z_sm = zscore_dict({a: meta[(f_id, a)]["is_safety_margin"] for a in cands})
        z_ker = zscore_dict({a: 0.6 * snr[(f_id, a)]["ker_20"] + 0.4 * snr[(f_id, a)]["ker_60"] for a in cands})
        z_liq = zscore_dict({a: np.log1p(liq[(f_id, a)]["is_vol_wan"]) for a in cands})

        raw_sc = {
            a: (
                w_shp * z_shp[a]
                + w_sm * z_sm[a]
                + w_ker * z_ker[a]
                + w_liq * z_liq[a]
                + (buffer_bonus if a in prev else 0.0)
            )
            for a in cands
        }

        kept = []
        for _sec, syms in SECTORS_60.items():
            sec_cands = sorted([a for a in syms if a in raw_sc], key=lambda a: raw_sc[a], reverse=True)
            sec_kept = []
            for a in sec_cands:
                high_corr_peers = [k for k in sec_kept if float(cp.loc[a, k]) >= corr_thresh]
                if len(high_corr_peers) >= max_per_corr_cluster:
                    continue
                sec_kept.append(a)
                kept.append(a)

        selected = sorted(kept, key=lambda a: raw_sc[a], reverse=True)[:target_n]
        mults = {}
        for a in selected:
            m = meta[(f_id, a)]
            lq = liq[(f_id, a)]
            liq_w = float(np.clip(np.sqrt(lq["is_vol_wan"] / 20.0), 0.65, 1.40))
            eff_s = 0.55 * float(m["is_shp_hr"]) + 0.45 * float(m["is_shp_fs"])
            q_p = np.clip(eff_s, 0.5, 2.5) * np.clip(float(max(m["is_safety_margin"], 5.0)) / 15.0, 0.5, 2.0) * liq_w
            mults[a] = float(q_p)
        return selected, mults

    return _selector


def evaluate_filtered_universe(
    name: str,
    b: dict,
    is_liq_stats: dict,
    snr_stats: dict,
    price_ret_df: pd.DataFrame,
    selector_fn,
    *,
    use_quality_tilt: bool = False,
    use_cluster_damp: bool = False,
    use_health_mult: bool = False,
    sector_damp_power: float = 0.55,
    symmetric_deadzone: float = 0.02,
    max_single_asset_weight: float = 0.25,
    min_effective_assets: int = 4,
    target_portfolio_vol: float = 0.14,
) -> tuple[dict, dict[str, list[str]]]:
    """在统一的板块风险预算与执行框架下，评估不同品种筛选方案的样本外表现。"""
    oos_fast_cache = b["oos_fast_cache"]
    sleeve_daily_ret = b["sleeve_daily_ret"]
    is_meta_stats = b["is_meta_stats"]
    pit_assets = b["pit_assets"]
    ep_map = b["ep_hr"]

    ledgers = []
    ep_list = []
    selected_per_fold: dict[str, list[str]] = {}
    prev_selected: set[str] = set()

    for fold in FOLDS:
        f_id = fold["id"]
        cut = pd.Timestamp(fold["cutoff"]).normalize()
        fc = oos_fast_cache[f_id]
        fc.setdefault("execution_prices_df", fc["ret_oos"])
        ret_oos = fc["ret_oos"][IDS_60]
        mask_oos = fc["mask_oos"][IDS_60]
        roll_vol = fc["roll_vol_oos"][IDS_60]
        pos_df = fc["pos_oos_map"]["hr"][IDS_60].copy()

        d_all = sleeve_daily_ret[f_id]
        d_is = d_all.loc["2024-01-02":cut]
        sub_p = price_ret_df.loc["2024-01-02":cut]
        cp_mat = sub_p.corr().abs().fillna(0.0)
        cs_mat = d_is.corr().abs().fillna(0.0)

        sel_ids, scores_map = selector_fn(
            f_id, pit_assets[f_id], is_meta_stats, is_liq_stats, snr_stats, cp_mat, cs_mat, prev_selected
        )
        prev_selected = set(sel_ids)
        selected_per_fold[f_id] = sel_ids
        sel_set = set(sel_ids)
        for c in IDS_60:
            if c not in sel_set:
                pos_df[c] = 0.0

        listed_mask = mask_oos.astype(np.float64)
        active_mask = ((pos_df.abs() > 1e-3) & (listed_mask > 0.0)).astype(np.float64)
        inv_vol = (1.0 / np.maximum(roll_vol, 0.05)) * active_mask
        if use_cluster_damp and sel_ids:
            cluster_w = compute_continuous_cluster_risk_shares(
                price_corr_df=cp_mat,
                strategy_corr_df=cs_mat,
                sectors=SECTORS_60,
                candidate_assets=sel_ids,
                cluster_damp_coeff=0.85,
            )
            for c in sel_ids:
                inv_vol[c] *= cluster_w.get(c, 1.0)

        if use_quality_tilt and sel_ids:
            for c in sel_ids:
                inv_vol[c] *= float(np.clip(scores_map.get(c, 1.0), 0.5, 2.2))

        if use_health_mult:
            h_5m = compute_causal_sleeve_health_multiplier(d_all[IDS_60], ret_oos.index, neg_floor=0.35)
            inv_vol = inv_vol * h_5m

        led = _execute_sector_conserved_allocation(
            fc,
            pos_df,
            inv_vol,
            active_mask,
            sector_damp_power=sector_damp_power,
            symmetric_deadzone=symmetric_deadzone,
            max_single_asset_weight=max_single_asset_weight,
            min_effective_assets=min_effective_assets,
            target_portfolio_vol=target_portfolio_vol,
        )
        ledgers.append(led)
        w_oos = led["weights_df"]
        for c in sel_ids:
            if w_oos[c].abs().max() > 1e-6 and len(ep_map.get((f_id, c), [])) > 0:
                ep_list.append(ep_map[(f_id, c)])

    s = summarize_stitched_oos(ledgers, name)
    s.update(compute_episode_payoff_stats(ep_list))
    s["f1_count"] = len(selected_per_fold["Fold1_2025"])
    s["f2_count"] = len(selected_per_fold["Fold2_2026"])
    return s, selected_per_fold


def main() -> None:
    cache_dir = STUDY_DIR / "cache_60"
    b = pickle.loads((cache_dir / "fast_oos_bundle_v1.pkl").read_bytes())
    frames_1d = pickle.loads((cache_dir / "frames_1d_60.pkl").read_bytes())
    is_liq_stats, price_ret_df = compute_is_liquidity_and_price_returns(frames_1d)
    snr_stats = compute_is_microstructure_snr(frames_1d)

    experiments = [
        ("0_Flat60_Baseline", sel_all_60, False, False),
        ("1_Naive_Top32_Liquidity", sel_top32_liquidity, False, False),
        (
            "2_Gate_Liq_SNR_32",
            make_institutional_funnel_selector(
                32,
                min_nf=1,
                min_shp_hr=-1.0,
                min_r=0.5,
                w_shp=0.0,
                w_sm=0.0,
                w_ker=0.70,
                w_liq=0.30,
                max_per_corr_cluster=99,
            ),
            False,
            False,
        ),
        (
            "3_Gate_Liq_SNR_SOP_32",
            make_institutional_funnel_selector(32, max_per_corr_cluster=99),
            True,
            False,
        ),
        ("4_Inst_Funnel_N28", make_institutional_funnel_selector(28), True, True),
        ("5_Inst_Funnel_N30", make_institutional_funnel_selector(30), True, True),
        ("6_Inst_Funnel_N32_SweetSpot", make_institutional_funnel_selector(32), True, True),
        ("7_Inst_Funnel_N35_BestSharpe", make_institutional_funnel_selector(35), True, True),
        ("8_Inst_Funnel_N40", make_institutional_funnel_selector(40), True, True),
    ]

    rows = []
    for name, fn, qt, cd in experiments:
        s, sel_map = evaluate_filtered_universe(
            name,
            b,
            is_liq_stats,
            snr_stats,
            price_ret_df,
            fn,
            use_quality_tilt=qt,
            use_cluster_damp=cd,
            use_health_mult=False,
        )
        s["f1_symbols"] = ",".join(sel_map["Fold1_2025"])
        s["f2_symbols"] = ",".join(sel_map["Fold2_2026"])
        rows.append({k: v for k, v in s.items() if k != "daily_series"})

    df_out = pd.DataFrame(rows)
    out_csv = STUDY_DIR / "universe_filtering_30_ablation.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved ablation table to {out_csv}")
    cols = [
        "experiment",
        "f1_count",
        "f2_count",
        "gross_ret_pct",
        "cost_ret_pct",
        "net_ret_pct",
        "sharpe",
        "max_dd_pct",
        "calmar",
        "ann_turnover",
        "payoff_r",
        "safety_margin_pct",
    ]
    print(df_out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
