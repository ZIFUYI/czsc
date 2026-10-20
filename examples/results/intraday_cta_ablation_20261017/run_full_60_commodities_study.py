"""
全商品池（方案B：60个流动性合格商品期货）多周期 × 6大簇增量筛选 + 缠论结构化出场与核心-卫星因果配置实证脚本
========================================================================================================
本脚本完整复现从 Core-20（20个主力核心商品）扩展至 Scheme B（60个全市场流动性合格商品期货，覆盖6大板块）的：
1. 60品种流动性普查与数据覆盖表 (`full60_universe_summary.csv`)
2. 为什么朴素全量60品种等权HRP会遭遇「板块风险预算膨胀 + 产业链影子品种稀释」陷阱（Sharpe 2.36 -> 1.18）的归因诊断
3. 100% 严格因果的机构级「Core-20旗舰锚定 + 40扩展池SOP阶段三门禁 + 剩余容量因果袖套分配」架构实证
4. 1.0x 与 3.0x 敞口下的全链路控制变量消融对比表 (`full60_ablation_comparison.csv`)
5. 40个扩展卫星品种的样本内 SOP 阶段三门禁指标与样本外独立表现明细表 (`full60_satellite_sop_admission.csv`)
6. 6大板块归因表 (`full60_sector_attribution.csv`) 与 60品种归因表 (`full60_symbol_attribution.csv`)
7. 自包含交互式 Plotly HTML 报告 (`full60_cta_expansion_report.html`)
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from multi_factor_cta.data_engine import AssetClass, AssetSpec  # noqa: E402
from multi_factor_cta.synthesis_engine import (  # noqa: E402
    _project_capped_simplex_row,
    account_portfolio_execution,
)
from run_multitf_incremental_study import (  # noqa: E402
    ALL_TFS,
    CORE_20_IDS,
    FOLDS,
    OUT_DIR,
    compute_episode_payoff_stats,
    scale_fold_ledgers_exposure,
    summarize_stitched_oos,
)

CACHE_60_DIR = OUT_DIR / "cache_60"
DATA_15M_60_DIR = OUT_DIR / "data_15m_60"

# ---------------------------------------------------------------------------
# 1. 方案B：60个流动性合格商品期货元数据定义（6大板块）
# ---------------------------------------------------------------------------
RAW_60_META: list[tuple[str, str, str, str, bool, float]] = [
    # 贵金属 (2)
    ("au", "AU9999.XSGE", "PRECIOUS_METALS", "沪金", True, 0.00010),
    ("ag", "AG9999.XSGE", "PRECIOUS_METALS", "沪银", True, 0.00015),
    # 有色金属 (8)
    ("cu", "CU9999.XSGE", "BASE_METALS", "沪铜", True, 0.00015),
    ("al", "AL9999.XSGE", "BASE_METALS", "沪铝", True, 0.00015),
    ("zn", "ZN9999.XSGE", "BASE_METALS", "沪锌", True, 0.00015),
    ("ni", "NI9999.XSGE", "BASE_METALS", "沪镍", True, 0.00015),
    ("sn", "SN9999.XSGE", "BASE_METALS", "沪锡", True, 0.00015),
    ("pb", "PB9999.XSGE", "BASE_METALS", "沪铅", True, 0.00015),
    ("ao", "AO9999.XSGE", "BASE_METALS", "氧化铝", True, 0.00015),
    ("ss", "SS9999.XSGE", "BASE_METALS", "不锈钢", True, 0.00015),
    # 黑色建材 (9)
    ("rb", "RB9999.XSGE", "FERROUS", "螺纹钢", True, 0.00015),
    ("hc", "HC9999.XSGE", "FERROUS", "热轧卷板", True, 0.00015),
    ("i", "I9999.XDCE", "FERROUS", "铁矿石", True, 0.00015),
    ("jm", "JM9999.XDCE", "FERROUS", "焦煤", True, 0.00015),
    ("j", "J9999.XDCE", "FERROUS", "焦炭", True, 0.00015),
    ("fg", "FG9999.XZCE", "FERROUS", "玻璃", True, 0.00015),
    ("sa", "SA9999.XZCE", "FERROUS", "纯碱", True, 0.00015),
    ("sm", "SM9999.XZCE", "FERROUS", "锰硅", False, 0.00015),
    ("sf", "SF9999.XZCE", "FERROUS", "硅铁", False, 0.00015),
    # 能源化工 (22)
    ("sc", "SC9999.XINE", "ENERGY_CHEM", "原油", True, 0.00015),
    ("fu", "FU9999.XSGE", "ENERGY_CHEM", "燃料油", True, 0.00015),
    ("lu", "LU9999.XINE", "ENERGY_CHEM", "低硫燃料油", True, 0.00015),
    ("bu", "BU9999.XSGE", "ENERGY_CHEM", "沥青", True, 0.00015),
    ("pg", "PG9999.XDCE", "ENERGY_CHEM", "液化石油气", True, 0.00015),
    ("ta", "TA9999.XZCE", "ENERGY_CHEM", "PTA", True, 0.00015),
    ("px", "PX9999.XZCE", "ENERGY_CHEM", "对二甲苯", True, 0.00015),
    ("eg", "EG9999.XDCE", "ENERGY_CHEM", "乙二醇", True, 0.00015),
    ("pf", "PF9999.XZCE", "ENERGY_CHEM", "短纤", True, 0.00015),
    ("pr", "PR9999.XZCE", "ENERGY_CHEM", "瓶片", True, 0.00015),
    ("ma", "MA9999.XZCE", "ENERGY_CHEM", "甲醇", True, 0.00015),
    ("ur", "UR9999.XZCE", "ENERGY_CHEM", "尿素", False, 0.00015),
    ("l", "L9999.XDCE", "ENERGY_CHEM", "塑料", True, 0.00015),
    ("pp", "PP9999.XDCE", "ENERGY_CHEM", "聚丙烯", True, 0.00015),
    ("v", "V9999.XDCE", "ENERGY_CHEM", "PVC", True, 0.00015),
    ("eb", "EB9999.XDCE", "ENERGY_CHEM", "苯乙烯", True, 0.00015),
    ("sh", "SH9999.XZCE", "ENERGY_CHEM", "烧碱", True, 0.00015),
    ("ru", "RU9999.XSGE", "ENERGY_CHEM", "天然橡胶", True, 0.00015),
    ("nr", "NR9999.XINE", "ENERGY_CHEM", "20号胶", True, 0.00015),
    ("br", "BR9999.XSGE", "ENERGY_CHEM", "丁二烯橡胶", True, 0.00015),
    ("sp", "SP9999.XSGE", "ENERGY_CHEM", "纸浆", True, 0.00015),
    ("ec", "EC9999.XINE", "ENERGY_CHEM", "集运指数(欧线)", False, 0.00015),
    # 农产品与油脂 (16)
    ("m", "M9999.XDCE", "AGRICULTURE", "豆粕", True, 0.00015),
    ("rm", "RM9999.XZCE", "AGRICULTURE", "菜籽粕", True, 0.00015),
    ("a", "A9999.XDCE", "AGRICULTURE", "豆一", True, 0.00015),
    ("b", "B9999.XDCE", "AGRICULTURE", "豆二", True, 0.00015),
    ("y", "Y9999.XDCE", "AGRICULTURE", "豆油", True, 0.00015),
    ("p", "P9999.XDCE", "AGRICULTURE", "棕榈油", True, 0.00015),
    ("oi", "OI9999.XZCE", "AGRICULTURE", "菜籽油", True, 0.00015),
    ("pk", "PK9999.XZCE", "AGRICULTURE", "花生", False, 0.00015),
    ("c", "C9999.XDCE", "AGRICULTURE", "玉米", True, 0.00015),
    ("cs", "CS9999.XDCE", "AGRICULTURE", "玉米淀粉", True, 0.00015),
    ("sr", "SR9999.XZCE", "AGRICULTURE", "白糖", True, 0.00015),
    ("cf", "CF9999.XZCE", "AGRICULTURE", "棉花", True, 0.00015),
    ("lh", "LH9999.XDCE", "AGRICULTURE", "生猪", False, 0.00015),
    ("jd", "JD9999.XDCE", "AGRICULTURE", "鸡蛋", False, 0.00015),
    ("ap", "AP9999.XZCE", "AGRICULTURE", "苹果", False, 0.00015),
    ("cj", "CJ9999.XZCE", "AGRICULTURE", "红枣", False, 0.00015),
    # 新能源材料 (3)
    ("si", "SI9999.GFEX", "NEW_ENERGY", "工业硅", False, 0.00015),
    ("lc", "LC9999.GFEX", "NEW_ENERGY", "碳酸锂", False, 0.00015),
    ("ps", "PS9999.GFEX", "NEW_ENERGY", "多晶硅", False, 0.00015),
]

CN_NAME_MAP = {aid: cn for aid, _, _, cn, _, _ in RAW_60_META}
UNIVERSE_60_SPECS = [
    AssetSpec(
        symbol=sym,
        name=aid,
        asset_class=AssetClass.COMMODITY_FUTURES,
        sector=sec,
        base_tf="15m",
        eligible_tfs=ALL_TFS,
        fee_rate=fee,
        has_night_session=has_night,
        data_path=DATA_15M_60_DIR / f"{aid}_15m.parquet",
        adjust_rolls=False,
    )
    for aid, sym, sec, _, has_night, fee in RAW_60_META
]
SPECS_60 = {s.name: s for s in UNIVERSE_60_SPECS}
IDS_60 = [s.name for s in UNIVERSE_60_SPECS]
SECTORS_60: dict[str, list[str]] = {}
for c in IDS_60:
    SECTORS_60.setdefault(SPECS_60[c].sector, []).append(c)

PERIODS_PER_YEAR = 252.0 * 48.0
ANN_SCALE = np.sqrt(PERIODS_PER_YEAR)
BARS_PER_DAY = 48


def build_universe_60_summary_table() -> pd.DataFrame:
    """统计 60 个商品的 15m/5m 流动性、数据起止日期与所属层级（Core-20 vs 扩展池40）。"""
    rows = []
    for aid, sym, sec, cn_name, has_night, fee in RAW_60_META:
        p15 = DATA_15M_60_DIR / f"{aid}_15m.parquet"
        df15 = pd.read_parquet(p15)
        dt = pd.to_datetime(df15["dt"])
        d_norm = dt.dt.normalize()
        vol_col = "volume" if "volume" in df15.columns else "vol"
        amt_series = df15["amount"] if "amount" in df15.columns else pd.Series(0.0, index=df15.index)
        if float(amt_series.sum()) <= 0.0 and vol_col in df15.columns:
            # 历史本地 parquet 若未存成交额字段，使用 close * volume * 10 (标准合约乘数近似) 估算名义成交额
            amt_series = df15["close"].astype(np.float64) * df15[vol_col].astype(np.float64) * 10.0
        daily_amt_yi = amt_series.groupby(d_norm).sum() / 1e8
        daily_vol_wan = df15.groupby(d_norm)[vol_col].sum() / 1e4 if vol_col in df15.columns else pd.Series([0.0])
        rows.append(
            {
                "asset_id": aid,
                "cn_name": cn_name,
                "jq_symbol": sym,
                "sector": sec,
                "tier": "Tier1_Core20" if aid in CORE_20_IDS else "Tier2_Satellite40",
                "has_night": has_night,
                "fee_rate_bp": round(fee * 1e4, 2),
                "start_date": str(d_norm.min().date()),
                "end_date": str(d_norm.max().date()),
                "trading_days": int(d_norm.nunique()),
                "bars_15m": int(len(df15)),
                "median_daily_turnover_yi": round(float(daily_amt_yi.median()), 2),
                "median_daily_vol_wan": round(float(daily_vol_wan.median()), 2),
            }
        )
    return pd.DataFrame(rows)


def run_flat_60_alloc(
    name: str,
    oos_fast_cache: dict,
    pit_assets: dict,
    ep_map: dict,
    scheme: str = "hr",
    sector_damp_power: float = 0.50,
    symmetric_deadzone: float = 0.0,
    max_single_asset_weight: float = 0.25,
    min_effective_assets: int = 4,
    target_portfolio_vol: float = 0.14,
) -> tuple[dict, list[dict]]:
    """运行全量 60 品种扁平化跨资产 HRP 配置（对比朴素等权 vs 板块预算守恒+对称死区）。"""
    ledgers = []
    ep_list = []
    for fold in FOLDS:
        f_id = fold["id"]
        fc = oos_fast_cache[f_id]
        ret_oos = fc["ret_oos"][IDS_60]
        mask_oos = fc["mask_oos"][IDS_60]
        roll_vol = fc["roll_vol_oos"][IDS_60]
        pos_df = fc["pos_oos_map"][scheme][IDS_60].copy()
        pit_set = set(pit_assets[f_id])
        for c in IDS_60:
            if c not in pit_set:
                pos_df[c] = 0.0

        listed_mask = mask_oos.astype(np.float64)
        active_mask = ((pos_df.abs() > 1e-3) & (listed_mask > 0.0)).astype(np.float64)
        inv_vol = (1.0 / np.maximum(roll_vol, 0.05)) * active_mask

        sector_active_counts = {}
        sector_inv_vol = {}
        for sec, syms in SECTORS_60.items():
            k_s_t = active_mask[syms].sum(axis=1)
            sector_active_counts[sec] = k_s_t
            sec_vol = fc["sec_vol_oos"][sec]
            sector_inv_vol[sec] = (1.0 / np.maximum(sec_vol, 0.05)) * (k_s_t > 0.0).astype(np.float64)

        sec_inv_vol_df = pd.DataFrame(sector_inv_vol, index=ret_oos.index)
        sec_inv_vol_sum = sec_inv_vol_df.sum(axis=1).replace(0.0, np.nan)
        sec_budget_share = sec_inv_vol_df.div(sec_inv_vol_sum, axis=0).fillna(0.0)

        sector_mult_df = pd.DataFrame(0.0, index=ret_oos.index, columns=IDS_60, dtype=np.float64)
        for sec, syms in SECTORS_60.items():
            k_s_t = sector_active_counts[sec]
            within_sec_damp = 1.0 / np.power(np.maximum(k_s_t, 1.0), sector_damp_power)
            sec_factor = sec_budget_share[sec] * within_sec_damp
            for sym in syms:
                sector_mult_df[sym] = sec_factor

        raw_alloc = inv_vol * sector_mult_df * active_mask
        n_rows, n_cols = raw_alloc.shape
        alloc_mat = np.zeros((n_rows, n_cols), dtype=np.float64)
        raw_mat = raw_alloc.to_numpy(dtype=np.float64)
        act_mat = active_mask.to_numpy(dtype=np.float64)
        for t_idx in range(n_rows):
            n_act = float(act_mat[t_idx].sum())
            if n_act <= 0.0:
                continue
            target_b = min(1.0, n_act / float(min_effective_assets))
            alloc_mat[t_idx] = _project_capped_simplex_row(
                raw_weights=raw_mat[t_idx],
                active_mask=act_mat[t_idx],
                target_budget=target_b,
                max_weight=max_single_asset_weight,
            )

        alloc_weights = pd.DataFrame(alloc_mat, index=ret_oos.index, columns=IDS_60)
        unscaled_positions = (alloc_weights * pos_df).where(mask_oos).ffill().fillna(0.0)
        unscaled_gross = (unscaled_positions.shift(1).fillna(0.0) * ret_oos).sum(axis=1)
        port_roll_vol = unscaled_gross.rolling(40 * BARS_PER_DAY, min_periods=15 * BARS_PER_DAY).std().shift(1)
        port_roll_vol = (port_roll_vol * ANN_SCALE).ffill().fillna(target_portfolio_vol)
        lev_scalar = np.clip(target_portfolio_vol / np.maximum(port_roll_vol, 0.05), 0.0, 1.65)
        budget_weights = alloc_weights.multiply(lev_scalar, axis=0).clip(lower=0.0, upper=max_single_asset_weight)
        final_weights_df = budget_weights * pos_df

        budget_fraction = active_mask.sum(axis=1).div(min_effective_assets).clip(upper=1.0)
        gross_budget_series = (budget_fraction * lev_scalar).to_numpy(dtype=np.float64)
        final_mat = final_weights_df.to_numpy().copy()
        observed = mask_oos.to_numpy()
        prev = np.zeros(n_cols)
        for t_idx in range(n_rows):
            target = final_mat[t_idx]
            if symmetric_deadzone > 0.0:
                same_dir = (target * prev) > 0.0
                keep = same_dir & (np.abs(target) > 1e-6) & (np.abs(target - prev) < symmetric_deadzone)
                target[keep] = prev[keep]
            target[~observed[t_idx]] = prev[~observed[t_idx]]
            frozen = float(np.abs(prev[~observed[t_idx]]).sum())
            avail = max(0.0, float(gross_budget_series[t_idx]) - frozen)
            prop = float(np.abs(target[observed[t_idx]]).sum())
            if prop > avail and prop > 0:
                target[observed[t_idx]] *= avail / prop
            final_mat[t_idx] = np.clip(target, -max_single_asset_weight, max_single_asset_weight)
            prev = final_mat[t_idx].copy()

        w_oos = pd.DataFrame(final_mat, index=ret_oos.index, columns=IDS_60)
        gross_exp = w_oos.abs().sum(axis=1)
        scale_cap = np.maximum(gross_exp, 1.0)
        w_oos = w_oos.div(scale_cap, axis=0)
        w_oos.iloc[0] = 0.0
        w_oos.iloc[-1] = 0.0
        for col in IDS_60:
            t_col = mask_oos[col].to_numpy()
            w_col = w_oos[col].to_numpy().copy()
            for i in range(1, len(w_col)):
                if not t_col[i]:
                    w_col[i] = w_col[i - 1]
            w_oos[col] = w_col

        fees = {c: float(SPECS_60[c].fee_rate) for c in IDS_60}
        ledger = account_portfolio_execution(ret_oos, w_oos, fees, tradable_mask=mask_oos)
        ledgers.append(
            {
                "daily_returns": ledger["daily_returns"],
                "bar_net_returns": ledger["bar_net_returns"],
                "gross_returns_df": ledger["gross_returns_df"],
                "cost_returns_df": ledger["cost_returns_df"],
                "turnover_df": ledger["turnover_df"],
                "weights_df": w_oos,
            }
        )
        for c in IDS_60:
            if w_oos[c].abs().max() > 1e-6 and len(ep_map.get((f_id, c), [])) > 0:
                ep_list.append(ep_map[(f_id, c)])

    s = summarize_stitched_oos(ledgers, name)
    ep_st = compute_episode_payoff_stats(ep_list)
    s.update(ep_st)
    return s, ledgers


def run_zero_hardcode_sop_stage3(
    name: str,
    oos_fast_cache: dict,
    sleeve_daily_ret: dict,
    is_meta_stats: dict,
    pit_assets: dict,
    ep_map: dict,
    scheme: str = "hr",
    min_nf: int = 6,
    min_shp_fs: float = 1.05,
    min_shp_hr: float = 0.95,
    min_r: float = 1.50,
    min_sm: float = 12.0,
    sector_damp_power: float = 0.50,
    symmetric_deadzone: float = 0.02,
    max_single_asset_weight: float = 0.25,
    min_effective_assets: int = 4,
    target_portfolio_vol: float = 0.14,
) -> tuple[dict, list[dict], dict[str, list[str]]]:
    """
    运行 100% 零先验硬编码 (Zero Hardcoded Core-20) 的全市场 60 品种统一 SOP Stage-3 门禁与因果健康度配置。
    """
    ledgers = []
    ep_list = []
    selected_per_fold: dict[str, list[str]] = {}

    for fold in FOLDS:
        f_id = fold["id"]
        fc = oos_fast_cache[f_id]
        ret_oos = fc["ret_oos"][IDS_60]
        mask_oos = fc["mask_oos"][IDS_60]
        roll_vol = fc["roll_vol_oos"][IDS_60]
        pos_df = fc["pos_oos_map"][scheme][IDS_60].copy()

        sel_ids = []
        for aid in IDS_60:
            if aid not in pit_assets[f_id]:
                continue
            m = is_meta_stats[(f_id, aid)]
            if (
                m["n_factors"] >= min_nf
                and m["is_shp_fs"] >= min_shp_fs
                and m["is_shp_hr"] >= min_shp_hr
                and m["is_payoff_r"] >= min_r
                and m["is_safety_margin"] >= min_sm
                and m["is_ep_count"] >= 6
            ):
                sel_ids.append(aid)
        selected_per_fold[f_id] = sel_ids
        sel_set = set(sel_ids)
        for c in IDS_60:
            if c not in sel_set:
                pos_df[c] = 0.0

        listed_mask = mask_oos.astype(np.float64)
        active_mask = ((pos_df.abs() > 1e-3) & (listed_mask > 0.0)).astype(np.float64)
        inv_vol = (1.0 / np.maximum(roll_vol, 0.05)) * active_mask

        for c in sel_ids:
            m = is_meta_stats[(f_id, c)]
            q_p = np.clip(float(m["is_shp_hr"]), 0.5, 2.5) * np.clip(float(m["is_safety_margin"]) / 15.0, 0.5, 2.0)
            inv_vol[c] *= q_p

        d_df = sleeve_daily_ret[f_id][IDS_60]
        r_m = d_df.rolling(60, min_periods=20).mean().shift(1)
        r_s = d_df.rolling(60, min_periods=20).std(ddof=1).shift(1)
        r_shp = ((r_m / r_s.replace(0.0, np.nan)) * np.sqrt(252.0)).fillna(0.5)
        h_daily = r_shp.clip(lower=-0.5, upper=2.0).apply(lambda s: np.where(s >= 0.0, 1.0 + 0.30 * s, 0.35))
        h_5m = h_daily.reindex(ret_oos.index.normalize()).fillna(1.0)
        h_5m.index = ret_oos.index
        inv_vol = inv_vol * h_5m

        sector_active_counts = {}
        sector_inv_vol = {}
        for sec, syms in SECTORS_60.items():
            k_s_t = active_mask[syms].sum(axis=1)
            sector_active_counts[sec] = k_s_t
            sec_vol = fc["sec_vol_oos"][sec]
            sector_inv_vol[sec] = (1.0 / np.maximum(sec_vol, 0.05)) * (k_s_t > 0.0).astype(np.float64)

        sec_inv_vol_df = pd.DataFrame(sector_inv_vol, index=ret_oos.index)
        sec_inv_vol_sum = sec_inv_vol_df.sum(axis=1).replace(0.0, np.nan)
        sec_budget_share = sec_inv_vol_df.div(sec_inv_vol_sum, axis=0).fillna(0.0)

        sector_mult_df = pd.DataFrame(0.0, index=ret_oos.index, columns=IDS_60, dtype=np.float64)
        for sec, syms in SECTORS_60.items():
            k_s_t = sector_active_counts[sec]
            within_sec_damp = 1.0 / np.power(np.maximum(k_s_t, 1.0), sector_damp_power)
            sec_factor = sec_budget_share[sec] * within_sec_damp
            for sym in syms:
                sector_mult_df[sym] = sec_factor

        raw_alloc = inv_vol * sector_mult_df * active_mask
        n_rows, n_cols = raw_alloc.shape
        alloc_mat = np.zeros((n_rows, n_cols), dtype=np.float64)
        raw_mat = raw_alloc.to_numpy(dtype=np.float64)
        act_mat = active_mask.to_numpy(dtype=np.float64)
        for t_idx in range(n_rows):
            n_act = float(act_mat[t_idx].sum())
            if n_act <= 0.0:
                continue
            target_b = min(1.0, n_act / float(min_effective_assets))
            alloc_mat[t_idx] = _project_capped_simplex_row(
                raw_weights=raw_mat[t_idx],
                active_mask=act_mat[t_idx],
                target_budget=target_b,
                max_weight=max_single_asset_weight,
            )

        alloc_weights = pd.DataFrame(alloc_mat, index=ret_oos.index, columns=IDS_60)
        unscaled_positions = (alloc_weights * pos_df).where(mask_oos).ffill().fillna(0.0)
        unscaled_gross = (unscaled_positions.shift(1).fillna(0.0) * ret_oos).sum(axis=1)
        port_roll_vol = unscaled_gross.rolling(40 * BARS_PER_DAY, min_periods=15 * BARS_PER_DAY).std().shift(1)
        port_roll_vol = (port_roll_vol * ANN_SCALE).ffill().fillna(target_portfolio_vol)
        lev_scalar = np.clip(target_portfolio_vol / np.maximum(port_roll_vol, 0.05), 0.0, 1.65)
        budget_weights = alloc_weights.multiply(lev_scalar, axis=0).clip(lower=0.0, upper=max_single_asset_weight)
        final_weights_df = budget_weights * pos_df

        budget_fraction = active_mask.sum(axis=1).div(min_effective_assets).clip(upper=1.0)
        gross_budget_series = (budget_fraction * lev_scalar).to_numpy(dtype=np.float64)
        final_mat = final_weights_df.to_numpy().copy()
        observed = mask_oos.to_numpy()
        prev = np.zeros(n_cols)
        for t_idx in range(n_rows):
            target = final_mat[t_idx]
            if symmetric_deadzone > 0.0:
                same_dir = (target * prev) > 0.0
                keep = same_dir & (np.abs(target) > 1e-6) & (np.abs(target - prev) < symmetric_deadzone)
                target[keep] = prev[keep]
            target[~observed[t_idx]] = prev[~observed[t_idx]]
            frozen = float(np.abs(prev[~observed[t_idx]]).sum())
            avail = max(0.0, float(gross_budget_series[t_idx]) - frozen)
            prop = float(np.abs(target[observed[t_idx]]).sum())
            if prop > avail and prop > 0:
                target[observed[t_idx]] *= avail / prop
            final_mat[t_idx] = np.clip(target, -max_single_asset_weight, max_single_asset_weight)
            prev = final_mat[t_idx].copy()

        w_oos = pd.DataFrame(final_mat, index=ret_oos.index, columns=IDS_60)
        gross_exp = w_oos.abs().sum(axis=1)
        scale_cap = np.maximum(gross_exp, 1.0)
        w_oos = w_oos.div(scale_cap, axis=0)
        w_oos.iloc[0] = 0.0
        w_oos.iloc[-1] = 0.0
        for col in IDS_60:
            t_col = mask_oos[col].to_numpy()
            w_col = w_oos[col].to_numpy().copy()
            for i in range(1, len(w_col)):
                if not t_col[i]:
                    w_col[i] = w_col[i - 1]
            w_oos[col] = w_col

        fees = {c: float(SPECS_60[c].fee_rate) for c in IDS_60}
        ledger = account_portfolio_execution(ret_oos, w_oos, fees, tradable_mask=mask_oos)
        ledgers.append(
            {
                "daily_returns": ledger["daily_returns"],
                "bar_net_returns": ledger["bar_net_returns"],
                "gross_returns_df": ledger["gross_returns_df"],
                "cost_returns_df": ledger["cost_returns_df"],
                "turnover_df": ledger["turnover_df"],
                "weights_df": w_oos,
            }
        )
        for c in sel_ids:
            if w_oos[c].abs().max() > 1e-6 and len(ep_map.get((f_id, c), [])) > 0:
                ep_list.append(ep_map[(f_id, c)])

    s = summarize_stitched_oos(ledgers, name)
    ep_st = compute_episode_payoff_stats(ep_list)
    s.update(ep_st)
    return s, ledgers, selected_per_fold


def run_core_satellite_sop_stage3(
    name: str,
    oos_fast_cache: dict,
    sleeve_daily_ret: dict,
    is_meta_stats: dict,
    is_corrs_with_core: dict,
    pit_assets: dict,
    exact_core20: dict,
    ep_20_exact: dict,
    ep_60_map_all: dict,
    scheme: str = "hr",
    sat_cap: float = 0.50,
    max_single_sat_w: float = 0.12,
    min_nf: int = 8,
    min_shp_fs: float = 1.15,
    min_shp_hr: float = 1.05,
    min_r: float = 1.60,
    min_sm: float = 7.5,
    max_corr: float = 0.25,
    sat_rebal_deadzone: float = 0.025,
    use_causal_health: bool = True,
) -> tuple[dict, list[dict], dict[str, list[str]]]:
    """
    运行 100% 严格因果的「Core-20 旗舰锚定 + 40 扩展池 SOP 阶段三门禁 + 剩余容量因果袖套分配」架构。
    """
    core_leds = exact_core20[scheme]
    ep_60_map = ep_60_map_all[scheme]
    ep_list = list(ep_20_exact[scheme].values())
    combined_ledgers = []
    admitted_sats: dict[str, list[str]] = {}

    for idx_f, fold in enumerate(FOLDS):
        f_id = fold["id"]
        fc = oos_fast_cache[f_id]
        ret_oos = fc["ret_oos"][IDS_60]
        mask_oos = fc["mask_oos"][IDS_60]
        roll_vol = fc["roll_vol_oos"][IDS_60]
        c_led = core_leds[idx_f]
        w_core = c_led["weights_df"]

        sats = []
        for aid in IDS_60:
            if aid in CORE_20_IDS or aid not in pit_assets[f_id]:
                continue
            m = is_meta_stats[(f_id, aid)]
            c = is_corrs_with_core[(f_id, aid)]
            if (
                m["n_factors"] >= min_nf
                and m["is_shp_fs"] >= min_shp_fs
                and m["is_shp_hr"] >= min_shp_hr
                and m["is_payoff_r"] >= min_r
                and m["is_safety_margin"] >= min_sm
                and m["is_ep_count"] >= 8
                and c <= max_corr
            ):
                sats.append(aid)
        admitted_sats[f_id] = sats

        w_full = pd.DataFrame(0.0, index=ret_oos.index, columns=IDS_60, dtype=np.float64)
        for c in CORE_20_IDS:
            w_full[c] = w_core[c].reindex(ret_oos.index).fillna(0.0)

        if sats and sat_cap > 0.0:
            pos_sat = fc["pos_oos_map"][scheme][sats].copy()
            mask_sat = mask_oos[sats]
            act_sat = ((pos_sat.abs() > 1e-3) & mask_sat).astype(np.float64)
            inv_v = (1.0 / np.maximum(roll_vol[sats], 0.05)) * act_sat

            for s_aid in sats:
                m = is_meta_stats[(f_id, s_aid)]
                q_prior = max(0.25, float(m["is_shp_hr"]) * (float(m["is_safety_margin"]) / 20.0))
                inv_v[s_aid] *= q_prior
                if len(ep_60_map.get((f_id, s_aid), [])) > 0:
                    ep_list.append(ep_60_map[(f_id, s_aid)])

            if use_causal_health:
                d_df = sleeve_daily_ret[f_id][sats]
                r_m = d_df.rolling(60, min_periods=20).mean().shift(1)
                r_s = d_df.rolling(60, min_periods=20).std(ddof=1).shift(1)
                r_shp = ((r_m / r_s.replace(0.0, np.nan)) * np.sqrt(252.0)).fillna(0.5)
                h_daily = r_shp.clip(lower=-0.5, upper=2.0).apply(lambda s: np.where(s >= 0.0, 1.0 + 0.30 * s, 0.25))
                h_5m = h_daily.reindex(ret_oos.index.normalize()).fillna(1.0)
                h_5m.index = ret_oos.index
                inv_v = inv_v * h_5m

            inv_sum = inv_v.sum(axis=1).replace(0.0, np.nan)
            n_act_sat = act_sat.sum(axis=1)
            core_gross = w_full[list(CORE_20_IDS)].abs().sum(axis=1)
            avail_cap = np.minimum(sat_cap * (n_act_sat / 2.0).clip(upper=1.0), np.maximum(0.0, 1.0 - core_gross))
            raw_w_sat = (
                inv_v.div(inv_sum, axis=0).fillna(0.0).multiply(avail_cap, axis=0).clip(upper=max_single_sat_w)
                * pos_sat
            )
            raw_w_sat.iloc[0] = 0.0
            raw_w_sat.iloc[-1] = 0.0

            sat_mat = raw_w_sat.to_numpy(dtype=np.float64).copy()
            obs_sat = mask_sat.to_numpy()
            n_r, n_c = sat_mat.shape
            prev_s = np.zeros(n_c, dtype=np.float64)
            c_gross_arr = core_gross.to_numpy(dtype=np.float64)
            for t_i in range(n_r):
                row = sat_mat[t_i]
                same_d = (row * prev_s) > 0.0
                keep = same_d & (np.abs(row) > 1e-6) & (np.abs(row - prev_s) < sat_rebal_deadzone)
                row[keep] = prev_s[keep]
                row[~obs_sat[t_i]] = prev_s[~obs_sat[t_i]]
                rem = max(0.0, 1.0 - c_gross_arr[t_i])
                s_tot = float(np.abs(row).sum())
                if s_tot > rem and s_tot > 0:
                    row *= rem / s_tot
                sat_mat[t_i] = row
                prev_s = row.copy()
            sat_mat[0] = 0.0
            sat_mat[-1] = 0.0

            sat_fees = {s_aid: float(SPECS_60[s_aid].fee_rate) for s_aid in sats}
            sat_w_df = pd.DataFrame(sat_mat, index=ret_oos.index, columns=sats)
            sat_led = account_portfolio_execution(ret_oos[sats], sat_w_df, sat_fees, tradable_mask=mask_sat)
            for s_aid in sats:
                w_full[s_aid] = sat_w_df[s_aid]

            gross_df = pd.DataFrame(0.0, index=ret_oos.index, columns=IDS_60)
            cost_df = pd.DataFrame(0.0, index=ret_oos.index, columns=IDS_60)
            turn_df = pd.DataFrame(0.0, index=ret_oos.index, columns=IDS_60)
            for c in CORE_20_IDS:
                gross_df[c] = c_led["gross_returns_df"][c]
                cost_df[c] = c_led["cost_returns_df"][c]
                turn_df[c] = c_led["turnover_df"][c]
            for s_aid in sats:
                gross_df[s_aid] = sat_led["gross_returns_df"][s_aid]
                cost_df[s_aid] = sat_led["cost_returns_df"][s_aid]
                turn_df[s_aid] = sat_led["turnover_df"][s_aid]
            bar_net = c_led["bar_net_returns"] + sat_led["bar_net_returns"]
            daily_net = bar_net.groupby(bar_net.index.normalize()).sum()
            combined_ledgers.append(
                {
                    "daily_returns": daily_net,
                    "bar_net_returns": bar_net,
                    "gross_returns_df": gross_df,
                    "cost_returns_df": cost_df,
                    "turnover_df": turn_df,
                    "weights_df": w_full,
                }
            )

    s = summarize_stitched_oos(combined_ledgers, name)
    ep_st = compute_episode_payoff_stats(ep_list)
    s.update(ep_st)
    return s, combined_ledgers, admitted_sats


def build_attribution_tables(
    flat_ledgers: list[dict],
    cs_ledgers: list[dict],
    admitted_sats: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成 6大板块 与 60品种 在 Flat-60 HRP 与 CoreSatellite-60 最优架构下的归因对比表。"""
    flat_gross = pd.concat([led["gross_returns_df"] for led in flat_ledgers]).sum() * 100.0
    flat_cost = pd.concat([led["cost_returns_df"] for led in flat_ledgers]).sum() * 100.0
    flat_net = flat_gross - flat_cost
    flat_turn = pd.concat([led["turnover_df"] for led in flat_ledgers]).sum() / (503.0 / 252.0)

    cs_gross = pd.concat([led["gross_returns_df"] for led in cs_ledgers]).sum() * 100.0
    cs_cost = pd.concat([led["cost_returns_df"] for led in cs_ledgers]).sum() * 100.0
    cs_net = cs_gross - cs_cost
    cs_turn = pd.concat([led["turnover_df"] for led in cs_ledgers]).sum() / (503.0 / 252.0)

    sym_rows = []
    for aid in IDS_60:
        role = (
            "Core-20旗舰"
            if aid in CORE_20_IDS
            else (
                "SOP准入卫星"
                if (aid in admitted_sats["Fold1_2025"] or aid in admitted_sats["Fold2_2026"])
                else "SOP拦截影子/震荡"
            )
        )
        sym_rows.append(
            {
                "asset_id": aid,
                "cn_name": CN_NAME_MAP[aid],
                "sector": SPECS_60[aid].sector,
                "role": role,
                "flat60_gross_pct": round(float(flat_gross[aid]), 3),
                "flat60_cost_pct": round(float(flat_cost[aid]), 3),
                "flat60_net_pct": round(float(flat_net[aid]), 3),
                "flat60_ann_turn": round(float(flat_turn[aid]), 2),
                "coresat60_gross_pct": round(float(cs_gross[aid]), 3),
                "coresat60_cost_pct": round(float(cs_cost[aid]), 3),
                "coresat60_net_pct": round(float(cs_net[aid]), 3),
                "coresat60_ann_turn": round(float(cs_turn[aid]), 2),
                "net_improvement_pct": round(float(cs_net[aid] - flat_net[aid]), 3),
            }
        )
    sym_df = pd.DataFrame(sym_rows)

    sec_rows = []
    for sec, syms in SECTORS_60.items():
        sub = sym_df[sym_df["sector"] == sec]
        sec_rows.append(
            {
                "sector": sec,
                "total_symbols": len(syms),
                "core20_count": sum(1 for c in syms if c in CORE_20_IDS),
                "admitted_sat_count": sum(
                    1 for c in syms if c in set(admitted_sats["Fold1_2025"]) | set(admitted_sats["Fold2_2026"])
                ),
                "flat60_gross_pct": round(float(sub["flat60_gross_pct"].sum()), 2),
                "flat60_cost_pct": round(float(sub["flat60_cost_pct"].sum()), 2),
                "flat60_net_pct": round(float(sub["flat60_net_pct"].sum()), 2),
                "flat60_ann_turn": round(float(sub["flat60_ann_turn"].sum()), 1),
                "coresat60_gross_pct": round(float(sub["coresat60_gross_pct"].sum()), 2),
                "coresat60_cost_pct": round(float(sub["coresat60_cost_pct"].sum()), 2),
                "coresat60_net_pct": round(float(sub["coresat60_net_pct"].sum()), 2),
                "coresat60_ann_turn": round(float(sub["coresat60_ann_turn"].sum()), 1),
                "net_improvement_pct": round(float(sub["coresat60_net_pct"].sum() - sub["flat60_net_pct"].sum()), 2),
            }
        )
    sec_df = pd.DataFrame(sec_rows)
    return sec_df, sym_df


def build_satellite_sop_table(
    is_meta_stats: dict,
    is_corrs_with_core: dict,
    sleeve_daily_ret: dict,
    admitted_sats: dict[str, list[str]],
) -> pd.DataFrame:
    """生成 40 个扩展卫星品种在两折样本内的 SOP Stage-3 门禁指标与样本外独立表现明细表。"""
    rows = []
    for aid in IDS_60:
        if aid in CORE_20_IDS:
            continue
        m1 = is_meta_stats[("Fold1_2025", aid)]
        m2 = is_meta_stats[("Fold2_2026", aid)]
        c1 = is_corrs_with_core[("Fold1_2025", aid)]
        c2 = is_corrs_with_core[("Fold2_2026", aid)]

        # 计算该卫星品种在两折 OOS 的独立单品种袖套净 Sharpe
        oos_d1 = sleeve_daily_ret["Fold1_2025"][aid].loc["2025-01-01":"2025-12-31"]
        oos_d2 = sleeve_daily_ret["Fold2_2026"][aid].loc["2026-01-01":"2026-09-30"]
        oos_all = pd.concat([oos_d1, oos_d2])
        sd = float(oos_all.std(ddof=1))
        oos_shp = float((oos_all.mean() / sd) * np.sqrt(252.0)) if sd > 1e-12 else 0.0
        oos_ret_pct = float(oos_all.sum() * 100.0)

        rows.append(
            {
                "asset_id": aid,
                "cn_name": CN_NAME_MAP[aid],
                "sector": SPECS_60[aid].sector,
                "f1_n_factors": m1["n_factors"],
                "f1_is_shp_fs": round(float(m1["is_shp_fs"]), 2),
                "f1_is_shp_hr": round(float(m1["is_shp_hr"]), 2),
                "f1_is_payoff_r": round(float(m1["is_payoff_r"]), 2),
                "f1_is_safety_margin_pct": round(float(m1["is_safety_margin"]), 1),
                "f1_max_core_corr": round(float(c1), 3),
                "admitted_fold1": aid in admitted_sats["Fold1_2025"],
                "f2_n_factors": m2["n_factors"],
                "f2_is_shp_fs": round(float(m2["is_shp_fs"]), 2),
                "f2_is_shp_hr": round(float(m2["is_shp_hr"]), 2),
                "f2_is_payoff_r": round(float(m2["is_payoff_r"]), 2),
                "f2_is_safety_margin_pct": round(float(m2["is_safety_margin"]), 1),
                "f2_max_core_corr": round(float(c2), 3),
                "admitted_fold2": aid in admitted_sats["Fold2_2026"],
                "standalone_oos_ret_pct": round(oos_ret_pct, 2),
                "standalone_oos_sharpe": round(oos_shp, 2),
            }
        )
    return pd.DataFrame(rows)


def generate_html_report(
    ablation_df: pd.DataFrame,
    sec_df: pd.DataFrame,
    sym_df: pd.DataFrame,
    sat_df: pd.DataFrame,
    ledger_map: dict[str, list[dict]],
    output_path: Path,
) -> None:
    """生成自包含交互式 Plotly HTML 报告。"""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "1.0x 纯样本外累计净值曲线对比 (2025.01 ~ 2026.09, 503交易日)",
            "3.0x 机构目标敞口累计净收益与水下回撤对比",
            "6大板块净收益贡献对比：Flat-60等权HRP vs Core-Satellite-60最优架构",
            "60品种净收益贡献排序（Core-20旗舰 + SOP准入卫星 vs Flat-60稀释）",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    palette = {
        "Core20_ExpE_StaticExit": "#7f8c8d",
        "Core20_ExpE_HysteresisRunner": "#2980b9",
        "Full60_FlatHRP_StaticExit": "#e67e22",
        "Full60_FlatHRP_HysteresisRunner": "#d35400",
        "Full60_SecConserved_SymDZ_HR": "#8e44ad",
        "Full60_ZeroHardcode_SOPStage3": "#16a085",
        "Full60_CoreSatellite_SOPStage3_Optimal": "#27ae60",
    }

    for k, color in palette.items():
        leds = ledger_map[k]
        d_ret = pd.concat([led["daily_returns"] for led in leds]).sort_index()
        cum_ret = d_ret.cumsum() * 100.0
        fig.add_trace(
            go.Scatter(
                x=cum_ret.index,
                y=cum_ret.values,
                mode="lines",
                name=k,
                line={"color": color, "width": 3.0 if "Optimal" in k or "ZeroHardcode" in k else 2.0},
            ),
            row=1,
            col=1,
        )

    for k, color in (
        ("Core20_ExpE_HysteresisRunner_3x", "#2980b9"),
        ("Full60_FlatHRP_HysteresisRunner_3x", "#d35400"),
        ("Full60_ZeroHardcode_SOPStage3_3x", "#16a085"),
        ("Full60_CoreSatellite_SOPStage3_3x", "#27ae60"),
    ):
        leds = ledger_map[k]
        d_ret = pd.concat([led["daily_returns"] for led in leds]).sort_index()
        cum_ret = d_ret.cumsum() * 100.0
        fig.add_trace(
            go.Scatter(
                x=cum_ret.index,
                y=cum_ret.values,
                mode="lines",
                name=k,
                line={"color": color, "width": 3.0 if "CoreSatellite" in k else 2.0, "dash": "solid"},
            ),
            row=1,
            col=2,
        )

    # Subplot 3: Sector Attribution Bar Chart
    fig.add_trace(
        go.Bar(
            x=sec_df["sector"],
            y=sec_df["flat60_net_pct"],
            name="Flat-60 等权HRP (净收益%)",
            marker_color="#e67e22",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=sec_df["sector"],
            y=sec_df["coresat60_net_pct"],
            name="Core-Satellite-60 最优架构 (净收益%)",
            marker_color="#27ae60",
        ),
        row=2,
        col=1,
    )

    # Subplot 4: Top active symbols comparison
    active_sym = sym_df[(sym_df["coresat60_ann_turn"] > 0.01) | (sym_df["flat60_net_pct"].abs() > 0.15)].sort_values(
        "coresat60_net_pct", ascending=False
    )
    labels = [f"{r.asset_id}({r.cn_name})" for _, r in active_sym.iterrows()]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=active_sym["coresat60_net_pct"],
            name="Core-Satellite-60 单品种净贡献(%)",
            marker_color="#27ae60",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=active_sym["flat60_net_pct"],
            name="Flat-60 单品种净贡献(%)",
            marker_color="#e67e22",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="全商品池（方案B：60品种）多周期增量筛选 + 缠论结构化出场与核心-卫星因果配置实证报告",
        height=980,
        template="plotly_white",
        barmode="group",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )

    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>方案B（60商品期货）多周期CTA扩展与核心-卫星SOP阶段三因果配置实证报告</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; color: #2c3e50; background: #f8f9fa; }}
h1, h2, h3 {{ color: #1a252f; }}
.card {{ background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 10px; }}
th, td {{ border: 1px solid #dfe6e9; padding: 8px 10px; text-align: right; }}
th {{ background: #f1f2f6; text-align: center; font-weight: 600; }}
td:first-child, td:nth-child(2), td:nth-child(3) {{ text-align: left; }}
.highlight {{ background: #e8f8f5; font-weight: 600; }}
</style>
</head>
<body>
<h1>方案B（60个流动性合格商品期货）多周期CTA扩展与核心-卫星SOP阶段三因果配置实证报告</h1>
<div class="card">
<h2>一、核心消融实验总表（2025.01 ~ 2026.09，503个纯样本外交易日，原生5m网格计价）</h2>
{ablation_df.to_html(index=False)}
</div>
<div class="card">
{chart_html}
</div>
<div class="card">
<h2>二、6大商品板块样本外归因对比表（Flat-60等权HRP vs Core-Satellite-60最优架构）</h2>
{sec_df.to_html(index=False)}
</div>
<div class="card">
<h2>三、40个扩展池卫星品种样本内 SOP Stage-3 准入门禁指标与样本外表现表</h2>
{sat_df.to_html(index=False)}
</div>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")


def main() -> None:
    t0 = time.time()
    print("=" * 96)
    print("启动方案B（60个流动性合格商品期货）全链路消融实证与归因分析...")
    print("=" * 96)

    univ_df = build_universe_60_summary_table()
    univ_csv = OUT_DIR / "full60_universe_summary.csv"
    univ_df.to_csv(univ_csv, index=False)
    print(f"[1/5] 已生成 60 品种流动性与数据覆盖表: {univ_csv.name} ({len(univ_df)} 个品种)")

    with open(CACHE_60_DIR / "fast_oos_bundle_v1.pkl", "rb") as f:
        b = pickle.load(f)
    with open(CACHE_60_DIR / "exact_core20_ledgers_v3.pkl", "rb") as f:
        d_c20 = pickle.load(f)

    oos_fast_cache = b["oos_fast_cache"]
    sleeve_daily_ret = b["sleeve_daily_ret"]
    is_meta_stats = b["is_meta_stats"]
    ep_fs, ep_full, ep_hr = b["ep_fs"], b["ep_full"], b["ep_hr"]
    pit_assets = b["pit_assets"]
    exact_core20 = d_c20["ledgers"]
    ep_20_exact = d_c20["ep_20_exact"]
    ep_60_map_all = {"fs": ep_fs, "full": ep_full, "hr": ep_hr}

    # 计算扩展品种与同板块 Core-20 旗舰在样本内 (<= cutoff) 的最大绝对相关性
    is_corrs_with_core: dict[tuple[str, str], float] = {}
    for fold in FOLDS:
        f_id = fold["id"]
        cutoff = pd.Timestamp(fold["cutoff"]).normalize()
        d_df = sleeve_daily_ret[f_id]
        d_is = d_df.loc[d_df.index <= cutoff]
        for aid in IDS_60:
            if aid in CORE_20_IDS:
                is_corrs_with_core[(f_id, aid)] = 0.0
                continue
            sec = SPECS_60[aid].sector
            core_peers = [c for c in CORE_20_IDS if SPECS_60[c].sector == sec]
            if not core_peers or d_is[aid].std() < 1e-12:
                is_corrs_with_core[(f_id, aid)] = 0.0
                continue
            corrs = [abs(float(np.corrcoef(d_is[aid], d_is[cp])[0, 1])) for cp in core_peers if d_is[cp].std() > 1e-12]
            is_corrs_with_core[(f_id, aid)] = max(corrs) if corrs else 0.0

    results = []
    ledger_map: dict[str, list[dict]] = {}

    # 1. Core-20 静态出场基准 & HysteresisRunner 结构出场基准
    for code, exp_name in (
        ("fs", "Core20_ExpE_StaticExit"),
        ("full", "Core20_ExpE_FullStructExit"),
        ("hr", "Core20_ExpE_HysteresisRunner"),
    ):
        leds = exact_core20[code]
        s = summarize_stitched_oos(leds, exp_name)
        s.update(compute_episode_payoff_stats(list(ep_20_exact[code].values())))
        s["active_universe"] = "Core-20 (20)"
        results.append(s)
        ledger_map[exp_name] = leds

    # 2. Full-60 朴素等权 HRP 系列 (展示板块风险预算膨胀 + 产业链影子品种稀释陷阱)
    for code, exp_name in (
        ("fs", "Full60_FlatHRP_StaticExit"),
        ("full", "Full60_FlatHRP_FullStructExit"),
        ("hr", "Full60_FlatHRP_HysteresisRunner"),
    ):
        s, leds = run_flat_60_alloc(
            exp_name,
            oos_fast_cache,
            pit_assets,
            ep_60_map_all[code],
            scheme=code,
            sector_damp_power=0.50,
            symmetric_deadzone=0.0,
        )
        s["active_universe"] = "Full-60 Flat (58/60)"
        results.append(s)
        ledger_map[exp_name] = leds

    # 3. Full-60 板块风险预算守恒 (1/K^0.9) + 对称调仓死区 (展示被动再平衡摩擦抑制)
    s_sec, leds_sec = run_flat_60_alloc(
        "Full60_SecConserved_SymDZ_HR",
        oos_fast_cache,
        pit_assets,
        ep_hr,
        scheme="hr",
        sector_damp_power=0.90,
        symmetric_deadzone=0.025,
    )
    s_sec["active_universe"] = "Full-60 SecConserved (58/60)"
    results.append(s_sec)
    ledger_map["Full60_SecConserved_SymDZ_HR"] = leds_sec

    # 4. Full-60 100% 零先验硬编码 (Zero Hardcoded Core-20) 全池统一 SOP 阶段三门禁与因果健康度配置
    s_zhc, leds_zhc, sel_zhc = run_zero_hardcode_sop_stage3(
        "Full60_ZeroHardcode_SOPStage3",
        oos_fast_cache=oos_fast_cache,
        sleeve_daily_ret=sleeve_daily_ret,
        is_meta_stats=is_meta_stats,
        pit_assets=pit_assets,
        ep_map=ep_hr,
        scheme="hr",
        min_nf=6,
        min_shp_fs=1.05,
        min_shp_hr=0.95,
        min_r=1.50,
        min_sm=12.0,
        sector_damp_power=0.50,
        symmetric_deadzone=0.02,
    )
    s_zhc["active_universe"] = f"ZeroHC-SOP (F1:{len(sel_zhc['Fold1_2025'])}, F2:{len(sel_zhc['Fold2_2026'])})"
    results.append(s_zhc)
    ledger_map["Full60_ZeroHardcode_SOPStage3"] = leds_zhc

    # 5. Full-60 机构级「核心20锚定 + 40扩展池SOP阶段三门禁 + 剩余容量因果袖套分配」最优架构
    s_opt, leds_opt, admitted_sats = run_core_satellite_sop_stage3(
        "Full60_CoreSatellite_SOPStage3_Optimal",
        oos_fast_cache=oos_fast_cache,
        sleeve_daily_ret=sleeve_daily_ret,
        is_meta_stats=is_meta_stats,
        is_corrs_with_core=is_corrs_with_core,
        pit_assets=pit_assets,
        exact_core20=exact_core20,
        ep_20_exact=ep_20_exact,
        ep_60_map_all=ep_60_map_all,
        scheme="hr",
        sat_cap=0.50,
        max_single_sat_w=0.12,
        min_nf=8,
        min_shp_fs=1.15,
        min_shp_hr=1.05,
        min_r=1.60,
        min_sm=7.5,
        max_corr=0.25,
        sat_rebal_deadzone=0.025,
        use_causal_health=True,
    )
    s_opt["active_universe"] = (
        f"Core20 + SOP-Sat (F1:{20 + len(admitted_sats['Fold1_2025'])}, F2:{20 + len(admitted_sats['Fold2_2026'])})"
    )
    results.append(s_opt)
    ledger_map["Full60_CoreSatellite_SOPStage3_Optimal"] = leds_opt

    # 6. 3.0x 机构目标敞口对比
    for base_key, name_3x in (
        ("Core20_ExpE_StaticExit", "Core20_ExpE_StaticExit_3x"),
        ("Core20_ExpE_HysteresisRunner", "Core20_ExpE_HysteresisRunner_3x"),
        ("Full60_FlatHRP_HysteresisRunner", "Full60_FlatHRP_HysteresisRunner_3x"),
        ("Full60_ZeroHardcode_SOPStage3", "Full60_ZeroHardcode_SOPStage3_3x"),
        ("Full60_CoreSatellite_SOPStage3_Optimal", "Full60_CoreSatellite_SOPStage3_3x"),
    ):
        leds_3x = scale_fold_ledgers_exposure(ledger_map[base_key], 3.0)
        s3 = summarize_stitched_oos(leds_3x, name_3x)
        base_row = next(r for r in results if r["experiment"] == base_key)
        for k in ("ep_count", "win_rate_pct", "payoff_r", "safety_margin_pct", "active_universe"):
            if k in base_row:
                s3[k] = base_row[k]
        results.append(s3)
        ledger_map[name_3x] = leds_3x

    ablation_df = pd.DataFrame(results)
    ablation_csv = OUT_DIR / "full60_ablation_comparison.csv"
    ablation_df.to_csv(ablation_csv, index=False)
    print(f"[2/5] 已生成消融对比表: {ablation_csv.name}")
    print(
        ablation_df[
            [
                "experiment",
                "active_universe",
                "gross_ret_pct",
                "cost_ret_pct",
                "net_ret_pct",
                "sharpe",
                "max_dd_pct",
                "calmar",
                "ann_turnover",
                "payoff_r",
            ]
        ].to_string(index=False)
    )
    print(
        f"      SOP Stage-3 样本内自动准入卫星品种: Fold1_2025={admitted_sats['Fold1_2025']} | Fold2_2026={admitted_sats['Fold2_2026']}"
    )

    sec_df, sym_df = build_attribution_tables(
        ledger_map["Full60_FlatHRP_HysteresisRunner"],
        ledger_map["Full60_CoreSatellite_SOPStage3_Optimal"],
        admitted_sats,
    )
    sec_csv = OUT_DIR / "full60_sector_attribution.csv"
    sym_csv = OUT_DIR / "full60_symbol_attribution.csv"
    sec_df.to_csv(sec_csv, index=False)
    sym_df.to_csv(sym_csv, index=False)
    print(f"[3/5] 已生成 6大板块归因表 ({sec_csv.name}) 与 60品种归因表 ({sym_csv.name})")

    sat_df = build_satellite_sop_table(is_meta_stats, is_corrs_with_core, sleeve_daily_ret, admitted_sats)
    sat_csv = OUT_DIR / "full60_satellite_sop_admission.csv"
    sat_df.to_csv(sat_csv, index=False)
    print(f"[4/5] 已生成 40个扩展池卫星品种 SOP Stage-3 准入明细表: {sat_csv.name}")

    html_path = OUT_DIR / "full60_cta_expansion_report.html"
    generate_html_report(ablation_df, sec_df, sym_df, sat_df, ledger_map, html_path)
    print(f"[5/5] 已生成交互式 HTML 报告: {html_path} (总耗时 {time.time() - t0:.2f}s)")


if __name__ == "__main__":
    main()
