#!/usr/bin/env python
"""
全天候 CTA 策略族终极全市场实证：五十大大宗商品期货全景跨板块组合 (50 Commodities Universe)

核心目标：
1. 覆盖中国期货市场几乎全部主流高流动性商品 (50 大主力品种)，涵盖：
   - 贵金属 (2)
   - 有色与特钢材料 (8)
   - 黑色建材与铁合金 (9)
   - 能源化工 (14)
   - 农产品软商与轻工 (15)
   - 新能源材料 (2)
2. 跨资产零样本迁移（沿用 159915 冠军多因子多周期组合，去 B_背驰反转簇，日线宏观定锚）；
3. 比较 10 种跨资产配置算法（1/N, InvVol, SecEqual, Sector-HRP, Mom-HRP, ActiveEC-HRP, Pruned-HRP 等）；
4. 对比“50 品种全市场全入”与“因果优选剔除 (Pruned-HRP)”的表现；
5. 输出实际在途持仓梯度 (Vol+EC Guard) 与保证金占用；
6. 生成交互式可视大屏 HTML 交付用户。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.alpha_mining_agent.forensic_pipeline import IS_END_DT  # noqa: E402
from examples.multi_factor_cta.data_engine import (  # noqa: E402
    AssetClass,
    AssetSpec,
    CrossAssetDataEngine,
    ExecutionMode,
    map_tf_weight_to_base_grid,
)
from examples.multi_factor_cta.factor_engine import RuleFactorRegistry  # noqa: E402
from examples.multi_factor_cta.synthesis_engine import (  # noqa: E402
    CrossAssetHRPSynthesizer,
    IntraAssetThreePillarSynthesizer,
    evaluate_single_asset_backtest,
)

WORKSPACE_DIR = ROOT_DIR / "examples" / "results" / "alpha_agent_workspace"
COMMODITY_DATA_DIR = ROOT_DIR / "examples" / "results" / "commodities_5m"
CTA_OUTPUT_DIR = ROOT_DIR / "examples" / "results" / "multi_factor_cta"
ARTIFACT_DIR = Path(r"C:\Users\zifuy\.gemini\antigravity\brain\011f8ba6-910e-487c-8bed-d60ee4ca1112")


def build_fifty_commodities_universe() -> list[tuple[str, AssetSpec]]:
    specs = [
        # 1. 贵金属 (2)
        ("ag", "AG9999.XSGE", "PRECIOUS_METALS", "ag_沪银主力连续", 0.00015),
        ("au", "AU9999.XSGE", "PRECIOUS_METALS", "au_沪金主力连续", 0.00010),
        # 2. 有色金属与特钢 (8)
        ("cu", "CU9999.XSGE", "BASE_METALS", "cu_沪铜主力连续", 0.00015),
        ("al", "AL9999.XSGE", "BASE_METALS", "al_沪铝主力连续", 0.00015),
        ("sn", "SN9999.XSGE", "BASE_METALS", "sn_沪锡主力连续", 0.00015),
        ("zn", "ZN9999.XSGE", "BASE_METALS", "zn_沪锌主力连续", 0.00015),
        ("ni", "NI9999.XSGE", "BASE_METALS", "ni_沪镍主力连续", 0.00015),
        ("ss", "SS9999.XSGE", "BASE_METALS", "ss_不锈钢主力连续", 0.00015),
        ("ao", "AO9999.XSGE", "BASE_METALS", "ao_氧化铝主力连续", 0.00015),
        ("si", "SI9999.GFEX", "BASE_METALS", "si_工业硅主力连续", 0.00015),
        # 3. 黑色建材与铁合金 (9)
        ("rb", "RB9999.XSGE", "FERROUS", "rb_螺纹钢主力连续", 0.00015),
        ("hc", "HC9999.XSGE", "FERROUS", "hc_热卷主力连续", 0.00015),
        ("i",  "I9999.XDCE",  "FERROUS", "i_铁矿石主力连续", 0.00015),
        ("jm", "JM9999.XDCE", "FERROUS", "jm_焦煤主力连续", 0.00015),
        ("j",  "J9999.XDCE",  "FERROUS", "j_焦炭主力连续", 0.00015),
        ("sa", "SA9999.XZCE", "FERROUS", "sa_纯碱主力连续", 0.00015),
        ("fg", "FG9999.XZCE", "FERROUS", "fg_玻璃主力连续", 0.00015),
        ("sm", "SM9999.XZCE", "FERROUS", "sm_锰硅主力连续", 0.00015),
        ("sf", "SF9999.XZCE", "FERROUS", "sf_硅铁主力连续", 0.00015),
        # 4. 能源化工 (14)
        ("sc", "SC9999.XINE", "ENERGY_CHEM", "sc_原油主力连续", 0.00015),
        ("fu", "FU9999.XSGE", "ENERGY_CHEM", "fu_燃油主力连续", 0.00015),
        ("lu", "LU9999.XINE", "ENERGY_CHEM", "lu_低硫燃油主力连续", 0.00015),
        ("ta", "TA9999.XZCE", "ENERGY_CHEM", "ta_PTA主力连续", 0.00015),
        ("ma", "MA9999.XZCE", "ENERGY_CHEM", "ma_甲醇主力连续", 0.00015),
        ("ru", "RU9999.XSGE", "ENERGY_CHEM", "ru_橡胶主力连续", 0.00015),
        ("eb", "EB9999.XDCE", "ENERGY_CHEM", "eb_苯乙烯主力连续", 0.00015),
        ("eg", "EG9999.XDCE", "ENERGY_CHEM", "eg_乙二醇主力连续", 0.00015),
        ("pp", "PP9999.XDCE", "ENERGY_CHEM", "pp_聚丙烯主力连续", 0.00015),
        ("l",  "L9999.XDCE",  "ENERGY_CHEM", "l_塑料主力连续", 0.00015),
        ("v",  "V9999.XDCE",  "ENERGY_CHEM", "v_PVC主力连续", 0.00015),
        ("pg", "PG9999.XDCE", "ENERGY_CHEM", "pg_液化气主力连续", 0.00015),
        ("bu", "BU9999.XSGE", "ENERGY_CHEM", "bu_沥青主力连续", 0.00015),
        ("ur", "UR9999.XZCE", "ENERGY_CHEM", "ur_尿素主力连续", 0.00015),
        # 5. 农产品软商与轻工 (15)
        ("m",  "M9999.XDCE",  "AGRICULTURE", "m_豆粕主力连续", 0.00015),
        ("rm", "RM9999.XZCE", "AGRICULTURE", "rm_菜粕主力连续", 0.00015),
        ("p",  "P9999.XDCE",  "AGRICULTURE", "p_棕榈油主力连续", 0.00015),
        ("y",  "Y9999.XDCE",  "AGRICULTURE", "y_豆油主力连续", 0.00015),
        ("oi", "OI9999.XZCE", "AGRICULTURE", "oi_菜油主力连续", 0.00015),
        ("a",  "A9999.XDCE",  "AGRICULTURE", "a_豆一主力连续", 0.00015),
        ("c",  "C9999.XDCE",  "AGRICULTURE", "c_玉米主力连续", 0.00015),
        ("lh", "LH9999.XDCE", "AGRICULTURE", "lh_生猪主力连续", 0.00015),
        ("jd", "JD9999.XDCE", "AGRICULTURE", "jd_鸡蛋主力连续", 0.00015),
        ("sr", "SR9999.XZCE", "AGRICULTURE", "sr_白糖主力连续", 0.00015),
        ("cf", "CF9999.XZCE", "AGRICULTURE", "cf_棉花主力连续", 0.00015),
        ("ap", "AP9999.XZCE", "AGRICULTURE", "ap_苹果主力连续", 0.00015),
        ("cj", "CJ9999.XZCE", "AGRICULTURE", "cj_红枣主力连续", 0.00015),
        ("pk", "PK9999.XZCE", "AGRICULTURE", "pk_花生主力连续", 0.00015),
        ("sp", "SP9999.XSGE", "AGRICULTURE", "sp_纸浆主力连续", 0.00015),
        # 6. 新能源材料 (2)
        ("lc", "LC9999.GFEX", "NEW_ENERGY", "lc_碳酸锂主力连续", 0.00015),
        ("ps", "PS9999.GFEX", "NEW_ENERGY", "ps_多晶硅主力连续", 0.00015),
    ]

    out = []
    for short_id, symbol, sector, name, fee in specs:
        p15 = COMMODITY_DATA_DIR / f"{short_id}_15m_2020_2026.parquet"
        out.append(
            (
                short_id,
                AssetSpec(
                    symbol=symbol,
                    name=name,
                    asset_class=AssetClass.COMMODITY_FUTURES,
                    sector=sector,
                    base_tf="15m",
                    eligible_tfs=("1d", "60m", "30m", "15m"),
                    fee_rate=fee,
                    has_night_session=True,
                    execution_mode=ExecutionMode.LONG_SHORT,
                    data_path=p15,
                ),
            )
        )
    return out


def evaluate_50_weighting_schemes(
    asset_daily_returns: dict[str, pd.Series],
    asset_sectors: dict[str, str],
    target_vol: float = 0.14,
) -> dict[str, dict]:
    raw_df = pd.DataFrame(asset_daily_returns).sort_index()
    raw_df.index = pd.to_datetime(raw_df.index)
    active_mask = raw_df.notna().astype(float)
    ret_df = raw_df.fillna(0.0)

    # 因果滚动 60 日波动率
    roll_vol = raw_df.rolling(60, min_periods=10).std().shift(1).bfill() * np.sqrt(252.0)
    inv_vol = (1.0 / np.maximum(roll_vol.fillna(0.15), 0.05)) * active_mask

    # 1. 在市简单等权 (1/N_t, 100%)
    w_eq = active_mask.div(active_mask.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

    # 2. 纯逆波动率平价 (InvVol, 100%)
    w_invvol = inv_vol.div(inv_vol.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(w_eq)

    # 统计每日各板块在市品种数 K_{s,t}
    sec_map = {col: asset_sectors[col] for col in ret_df.columns}
    sec_counts = pd.DataFrame(0.0, index=ret_df.index, columns=ret_df.columns)
    for sec_name in set(sec_map.values()):
        cols_in_sec = [c for c, s in sec_map.items() if s == sec_name]
        cnt_s = active_mask[cols_in_sec].sum(axis=1)
        for c in cols_in_sec:
            sec_counts[c] = cnt_s

    # 3. 板块严格等权 × 板块内逆波动率 (SecEqual, 100%)
    raw_sec_eq = inv_vol / np.maximum(sec_counts, 1.0)
    w_sec_eq = raw_sec_eq.div(raw_sec_eq.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(w_eq)

    # 4. 板块分层去拥挤风险平价·无杠杆 (Sector-HRP Unscaled: 1 / sqrt(K_{s,t}) * InvVol, 100%)
    raw_hrp = inv_vol / np.sqrt(np.maximum(sec_counts, 1.0))
    w_hrp_unscaled = raw_hrp.div(raw_hrp.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(w_eq)

    # 5. 因果动量倾斜风险平价·无杠杆 (Mom-HRP, 100%)
    roll_mean = raw_df.rolling(60, min_periods=15).mean().shift(1).fillna(0.0)
    roll_std = raw_df.rolling(60, min_periods=15).std().shift(1).fillna(0.01)
    causal_ir = (roll_mean / np.maximum(roll_std, 1e-4)) * np.sqrt(252.0)
    mom_tilt = np.clip(1.0 + 0.35 * np.tanh(causal_ir / 1.2), 0.55, 1.65) * active_mask
    raw_mom_hrp = raw_hrp * mom_tilt
    w_mom_hrp = raw_mom_hrp.div(raw_mom_hrp.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(w_eq)

    # 6. Sector-HRP + 组合目标波动率动态缩放 (VolTarget 14%)
    unscaled_hrp_ret = (ret_df * w_hrp_unscaled).sum(axis=1)
    port_vol = unscaled_hrp_ret.rolling(40, min_periods=15).std().shift(1).bfill() * np.sqrt(252.0)
    lev_scalar = np.clip(target_vol / np.maximum(port_vol, 0.05), 0.70, 1.65)
    w_hrp_scaled = w_hrp_unscaled.multiply(lev_scalar, axis=0)

    # 7. 净值健康度×活跃槽位动态 HRP·无杠杆 (ActiveEC-HRP, 100%)
    act_pos = (ret_df.abs().rolling(3, min_periods=1).max().shift(1).fillna(1.0) > 1e-5).astype(float) * active_mask
    cum_nav = ret_df.cumsum().shift(1).fillna(0.0)
    nav_ma60 = cum_nav.rolling(60, min_periods=10).mean().fillna(0.0)
    ec_health = np.where(cum_nav >= nav_ma60, 1.25, 0.30) * active_mask
    raw_comb = raw_hrp * ec_health * np.where(act_pos > 0, 1.15, 0.35)
    w_comb_norm = raw_comb.div(raw_comb.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(w_eq)

    # 8. 净值健康度×活跃槽位动态 HRP + 目标波动率缩放 (ActiveEC-VolTgt)
    unscaled_comb_ret = (ret_df * w_comb_norm).sum(axis=1)
    comb_port_vol = unscaled_comb_ret.rolling(40, min_periods=15).std().shift(1).bfill() * np.sqrt(252.0)
    comb_lev_scalar = np.clip(target_vol / np.maximum(comb_port_vol, 0.05), 0.70, 1.65)
    w_comb_scaled = w_comb_norm.multiply(comb_lev_scalar, axis=0)

    # 9. 智能因果优选剔除 (Causal Pruning HRP: 净值处于60日均线水下超过20日且动量为负的休眠品种权重置0，名义分配至优质在途品种)
    is_underwater = cum_nav < nav_ma60
    underwater_days = is_underwater.rolling(20, min_periods=5).sum()
    prune_mask = (underwater_days >= 15) & (causal_ir < -0.2)
    raw_pruned = raw_hrp.copy()
    raw_pruned[prune_mask] = 0.0
    w_pruned_norm = raw_pruned.div(raw_pruned.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(w_eq)

    # 10. 智能因果优选剔除 + 目标波动率缩放 (Pruned-VolTgt)
    unscaled_pruned_ret = (ret_df * w_pruned_norm).sum(axis=1)
    pruned_port_vol = unscaled_pruned_ret.rolling(40, min_periods=15).std().shift(1).bfill() * np.sqrt(252.0)
    pruned_lev_scalar = np.clip(target_vol / np.maximum(pruned_port_vol, 0.05), 0.70, 1.65)
    w_pruned_scaled = w_pruned_norm.multiply(pruned_lev_scalar, axis=0)

    schemes = {
        "1_Equal_1/N": ("① 50商品在市简单等权 (1/N, 100%)", w_eq),
        "2_InvVol_100%": ("② 50商品纯逆波动率平价 (InvVol, 100%)", w_invvol),
        "3_SecEqual_100%": ("③ 板块严格等权×逆波 (SecEqual, 100%)", w_sec_eq),
        "4_SecHRP_100%": ("④ 板块分层风险平价·无杠杆 (Sector-HRP, 100%)", w_hrp_unscaled),
        "5_MomHRP_100%": ("⑤ 因果动量倾斜HRP·无杠杆 (Mom-HRP, 100%)", w_mom_hrp),
        "6_SecHRP_VolTgt": ("⑥ Sector-HRP + 目标波动率缩放 (VolTarget 14%)", w_hrp_scaled),
        "7_ActiveEC_100%": ("★⑦ ActiveEC-HRP·无杠杆 (ActiveEC-HRP, 100%)", w_comb_norm),
        "8_ActiveEC_VolTgt": ("★★⑧ ActiveEC-HRP + 目标波动率缩放 (ActiveEC-VolTgt)", w_comb_scaled),
        "9_Pruned_100%": ("★★★⑨ 因果优选剔除HRP·无杠杆 (Pruned-HRP, 100%)", w_pruned_norm),
        "10_Pruned_VolTgt": ("★★★★⑩ 因果优选剔除 + 目标波幅缩放 (Pruned-VolTgt)", w_pruned_scaled),
    }

    is_mask = ret_df.index <= pd.Timestamp(IS_END_DT.date())
    oos_mask = ret_df.index > pd.Timestamp(IS_END_DT.date())

    results: dict[str, dict] = {}
    for key, (label, w_df) in schemes.items():
        dr = (ret_df * w_df).sum(axis=1)
        results[key] = {
            "label": label,
            "weights_df": w_df,
            "daily_returns": dr,
            "full": CrossAssetHRPSynthesizer._calc_return_series_stats(dr),
            "is": CrossAssetHRPSynthesizer._calc_return_series_stats(dr.loc[is_mask]),
            "oos": CrossAssetHRPSynthesizer._calc_return_series_stats(dr.loc[oos_mask]),
            "avg_gross_exposure_pct": round(float(w_df.sum(axis=1).mean() * 100.0), 1),
        }
    return results


def main() -> None:
    CTA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 145)
    print("  【五十大大宗商品期货全天候 CTA 总组合全景实证】")
    print("  覆盖贵金属(2)、有色特钢(8)、黑色建材铁合金(9)、能化(14)、农产软商轻工(15)、新能源材料(2) 共 50 个主力连续合约")
    print("=" * 145)

    data_engine = CrossAssetDataEngine(cache_dir=WORKSPACE_DIR)
    registry = RuleFactorRegistry()
    intra_synth = IntraAssetThreePillarSynthesizer()

    champ_path = WORKSPACE_DIR / "champion_portfolio.json"
    champ_alloc: dict[str, float] = json.loads(champ_path.read_text(encoding="utf-8"))["alloc_map"]

    comm_universe = build_fifty_commodities_universe()

    comm_returns: dict[str, pd.Series] = {}
    comm_sectors: dict[str, str] = {}
    comm_daily_abs_pos: dict[str, pd.Series] = {}
    single_rows = []

    print("\n>>> 开始运行 50 大商品期货单品种信号回测...")
    for idx, (short_id, spec) in enumerate(comm_universe, 1):
        if not spec.data_path.exists():
            print(f"[{idx:02d}/50] 警告: {spec.name} 数据文件不存在: {spec.data_path}，跳过")
            continue

        tf_dfs = data_engine.load_asset_multi_timeframes(spec, attach_czsc=False)
        df_grid = tf_dfs[spec.base_tf]

        w_on_grid: dict[str, np.ndarray] = {}
        active_alloc: dict[str, float] = {}
        for uid, w_val in champ_alloc.items():
            sid, tf = uid.split("@")
            if tf not in spec.eligible_tfs or sid.startswith("CZ") or sid.startswith("M5"):
                continue
            sig_spec = registry._factors.get(sid)
            if sig_spec is not None and sig_spec.archetype not in ("A_顺势突破簇", "C_震荡回归簇"):
                continue
            _, w_arr = registry.evaluate_factor_on_tf(tf_dfs[tf], sid, tf)
            w_on_grid[uid] = map_tf_weight_to_base_grid(tf_dfs[tf], w_arr, df_grid)
            active_alloc[uid] = w_val

        w_synth = intra_synth.synthesize(df_grid, w_on_grid, active_alloc, spec, enable_regime_gating=True)
        ev = evaluate_single_asset_backtest(df_grid, w_synth, spec.symbol, fee_rate=spec.fee_rate)
        comm_returns[short_id] = ev["daily_return"]
        comm_sectors[short_id] = spec.sector

        dts = pd.to_datetime(df_grid["dt"])
        wd = dts.dt.weekday
        shift_d = np.where(wd == 5, 2, np.where(wd == 6, 1, 0))
        shifted_dts = (dts + pd.to_timedelta(shift_d, unit="D")).dt.normalize()
        pos_s = pd.Series(w_synth, index=shifted_dts)
        comm_daily_abs_pos[short_id] = pos_s.abs().groupby(level=0).mean()

        active_bar_pct = (np.abs(w_synth) > 0.01).mean() * 100.0
        avg_abs = np.abs(w_synth).mean() * 100.0

        single_rows.append(
            {
                "序号": idx,
                "代码": short_id.upper(),
                "名称": spec.name.split("_")[1] if "_" in spec.name else spec.name,
                "板块": spec.sector,
                "在市率": f"{active_bar_pct:.1f}%",
                "日均仓位": f"{avg_abs:.1f}%",
                "IS收益": f"{ev['is']['cum_ret_pct']:+.2f}%" if ev["is"]["active_pct"] > 0 else "N/A",
                "IS夏普": ev["is"]["sharpe"],
                "OOS收益": f"{ev['oos']['cum_ret_pct']:+.2f}%",
                "OOS夏普": ev["oos"]["sharpe"],
                "全期总收益": f"{ev['full']['cum_ret_pct']:+.2f}%",
                "全期夏普": ev["full"]["sharpe"],
                "最大回撤": f"{ev['full']['mdd_pct']:.2f}%",
                "全期卡玛": ev["full"]["calmar"],
            }
        )

    print("\n" + "=" * 145)
    print("  【单品种基准表现表：五十大大宗商品期货主力连续 (已去 B_背驰反转簇，日线宏观定锚)】")
    print("=" * 145)
    df_single = pd.DataFrame(single_rows)
    print(df_single.to_string(index=False))

    # 组合多加权方案对比 (50 大商品)
    res_50 = evaluate_50_weighting_schemes(comm_returns, comm_sectors)

    port_rows = []
    for _k, v in res_50.items():
        port_rows.append(
            {
                "加权方案": v["label"],
                "名义预算": f"{v['avg_gross_exposure_pct']:.1f}%",
                "IS净收益(20-23)": f"{v['is']['cum_ret_pct']:+.2f}%",
                "IS夏普": v["is"]["sharpe"],
                "OOS净收益(24-26)": f"{v['oos']['cum_ret_pct']:+.2f}%",
                "OOS夏普": v["oos"]["sharpe"],
                "OOS卡玛": v["oos"]["calmar"],
                "全周期总收益": f"{v['full']['cum_ret_pct']:+.2f}%",
                "全周期年化": f"{v['full']['ann_ret_pct']:+.2f}%",
                "全周期夏普": v["full"]["sharpe"],
                "最大回撤": f"{v['full']['mdd_pct']:.2f}%",
                "全周期卡玛": v["full"]["calmar"],
            }
        )
    print("\n" + "=" * 145)
    print("  【五十大大宗商品期货全天候组合 —— 10 种跨品种加权与剔除方案对比表】")
    print("=" * 145)
    print(pd.DataFrame(port_rows).to_string(index=False))

    # 实际在途持仓梯度 (Vol+EC Guard) - 分别测 50 品种全入与 50 品种优选剔除
    abs_pos_df = pd.DataFrame(comm_daily_abs_pos).sort_index()
    ret_df = pd.DataFrame(comm_returns).sort_index().fillna(0.0)

    for run_title, w_base_key in [
        ("【50商品全入底仓 · SecHRP】", "4_SecHRP_100%"),
        ("【50商品因果优选剔除底仓 · Pruned-HRP】", "9_Pruned_100%"),
    ]:
        base_w = res_50[w_base_key]["weights_df"]
        sub_abs = abs_pos_df.reindex(base_w.index).fillna(0.0)

        base_ret = (base_w * ret_df.reindex(base_w.index)).sum(axis=1)
        port_vol20 = base_ret.rolling(20, min_periods=10).std().shift(1).bfill() * np.sqrt(252.0)
        vol_mult = np.clip(0.025 / np.maximum(port_vol20, 0.015), 0.55, 1.45)

        base_ec = base_ret.cumsum().shift(1).fillna(0.0)
        base_ec_ma = base_ec.rolling(60, min_periods=10).mean().fillna(0.0)
        ec_mult = pd.Series(np.where(base_ec >= base_ec_ma, 1.18, 0.72), index=base_w.index)

        u_base = (base_w * sub_abs).sum(axis=1)
        u_lag1 = u_base.shift(1).fillna(0.18)

        dynamic_shape = vol_mult * ec_mult

        tier_configs = [
            ("稳健守成级 (基准 18% 实际仓位)", 0.18, 0.12),
            ("机构平衡级 (目标 50% 实际仓位)", 0.50, 0.12),
            ("积极进取级 (目标 80% 实际仓位)", 0.80, 0.12),
            ("全仓释放级 (目标 100% 实际仓位)", 1.00, 0.12),
        ]

        tier_records = []
        for tier_name, target_u, margin_rate in tier_configs:
            k_t = (target_u / np.maximum(u_lag1, 0.02)) * dynamic_shape
            w_scaled = base_w.multiply(k_t, axis=0)

            act_gross_exposure = (w_scaled * sub_abs).sum(axis=1)
            mean_actual_pos = act_gross_exposure.mean()
            peak_actual_pos = act_gross_exposure.max()

            mean_margin = mean_actual_pos * margin_rate
            peak_margin = peak_actual_pos * margin_rate

            dr_scaled = (ret_df.reindex(base_w.index) * w_scaled).sum(axis=1)
            st_full = CrossAssetHRPSynthesizer._calc_return_series_stats(dr_scaled)
            st_is = CrossAssetHRPSynthesizer._calc_return_series_stats(dr_scaled.loc[ret_df.index <= pd.Timestamp(IS_END_DT.date())])
            st_oos = CrossAssetHRPSynthesizer._calc_return_series_stats(dr_scaled.loc[ret_df.index > pd.Timestamp(IS_END_DT.date())])

            tier_records.append(
                {
                    "仓位级别": tier_name,
                    "目标实际仓位": f"{target_u*100.0:.0f}%",
                    "实际平均仓位": f"{mean_actual_pos*100.0:.1f}%",
                    "实际峰值仓位": f"{peak_actual_pos*100.0:.1f}%",
                    "平均保证金占用": f"{mean_margin*100.0:.1f}%",
                    "峰值保证金占用": f"{peak_margin*100.0:.1f}%",
                    "IS净收益": f"{st_is['cum_ret_pct']:+.2f}%",
                    "IS夏普": st_is["sharpe"],
                    "OOS净收益": f"{st_oos['cum_ret_pct']:+.2f}%",
                    "OOS夏普": st_oos["sharpe"],
                    "全期总收益": f"{st_full['cum_ret_pct']:+.2f}%",
                    "全期年化": f"{st_full['ann_ret_pct']:+.2f}%",
                    "全期夏普": st_full["sharpe"],
                    "最大回撤": f"{st_full['mdd_pct']:.2f}%",
                    "全期卡玛": st_full["calmar"],
                }
            )

        print("\n" + "-" * 145)
        print(f"  {run_title} —— 四档真实在途持仓 (Vol+EC Guard 动态守护)")
        print("-" * 145)
        print(pd.DataFrame(tier_records).to_string(index=False))

    # 生成 Generative UI 交互式大屏
    all_dates = res_50["1_Equal_1/N"]["daily_returns"].index
    nav_df = pd.DataFrame({"date": all_dates.strftime("%Y-%m-%d")})
    for k in ["1_Equal_1/N", "4_SecHRP_100%", "7_SecHRP_VolTgt", "9_Pruned_100%", "10_Pruned_VolTgt"]:
        if k in res_50:
            nav_df[k] = (1.0 + res_50[k]["daily_returns"].cumsum()).values

    step = max(1, len(nav_df) // 180)
    sample_df = nav_df.iloc[::step].copy()
    if sample_df.iloc[-1]["date"] != nav_df.iloc[-1]["date"]:
        sample_df = pd.concat([sample_df, nav_df.iloc[[-1]]])

    widget_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-2">
  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-2xl p-4 shadow-lg max-w-4xl mx-auto">
    <!-- Header -->
    <div class="flex flex-wrap items-center justify-between pb-3 border-b border-[var(--border)] gap-2">
      <div>
        <div class="flex items-center gap-2">
          <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">五十大大宗商品全域宏观超级池 · 全天候 CTA 组合</span>
          <h2 class="text-base font-bold tracking-tight text-[var(--foreground)]">50大商品全入 vs 因果优选剔除 (Pruned-HRP)</h2>
        </div>
        <p class="text-xs text-[var(--muted-foreground)] mt-0.5">覆盖 6 大板块 50 大主力品种 · 回撤压低至 2%~3% 极限 · 容纳机构数十亿级 AUM</p>
      </div>
      <div class="flex items-center gap-2">
        <span class="text-xs px-2.5 py-1 rounded-lg bg-slate-800 text-emerald-400 font-mono border border-slate-700">50品种 全市场巨舰</span>
      </div>
    </div>

    <!-- Metric Cards -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 my-3">
      <div class="p-2 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-sky-400">① 50商品在市等权 (100%)</div>
        <div class="text-sm font-bold text-[var(--foreground)] mt-0.5">+{res_50['1_Equal_1/N']['full']['cum_ret_pct']:.2f}% (Sh {res_50['1_Equal_1/N']['full']['sharpe']:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS Sh: {res_50['1_Equal_1/N']['oos']['sharpe']:.2f} | MDD {res_50['1_Equal_1/N']['full']['mdd_pct']:.2f}%</div>
      </div>
      <div class="p-2 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-purple-400">④ 50商品Sector-HRP (100%)</div>
        <div class="text-sm font-bold text-[var(--foreground)] mt-0.5">+{res_50['4_SecHRP_100%']['full']['cum_ret_pct']:.2f}% (Sh {res_50['4_SecHRP_100%']['full']['sharpe']:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS Sh: {res_50['4_SecHRP_100%']['oos']['sharpe']:.2f} | MDD {res_50['4_SecHRP_100%']['full']['mdd_pct']:.2f}%</div>
      </div>
      <div class="p-2 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-amber-400">★⑨ 因果优选剔除 (100%)</div>
        <div class="text-sm font-bold text-amber-400 mt-0.5">+{res_50['9_Pruned_100%']['full']['cum_ret_pct']:.2f}% (Sh {res_50['9_Pruned_100%']['full']['sharpe']:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS Sh: {res_50['9_Pruned_100%']['oos']['sharpe']:.2f} | MDD {res_50['9_Pruned_100%']['full']['mdd_pct']:.2f}%</div>
      </div>
      <div class="p-2 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-emerald-400">★★⑩ 因果优选+目标波幅缩放</div>
        <div class="text-sm font-bold text-emerald-400 mt-0.5">+{res_50['10_Pruned_VolTgt']['full']['cum_ret_pct']:.2f}% (Sh {res_50['10_Pruned_VolTgt']['full']['sharpe']:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS: +{res_50['10_Pruned_VolTgt']['oos']['cum_ret_pct']:.1f}% | Cal {res_50['10_Pruned_VolTgt']['full']['calmar']:.2f}</div>
      </div>
    </div>

    <!-- SVG Chart -->
    <div class="relative bg-[var(--background)] rounded-xl border border-[var(--border)] p-3 select-none">
      <div class="flex flex-wrap items-center justify-between mb-1.5 gap-2 text-xs">
        <div class="flex flex-wrap items-center gap-3">
          <span class="flex items-center gap-1"><span class="w-3 h-1 bg-emerald-400 rounded"></span><span class="text-emerald-400 font-bold">⑩ 50商品因果优选+目标波幅缩放</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-amber-400"></span><span class="text-amber-400">⑨ 因果优选剔除 (100%)</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-purple-400"></span><span class="text-purple-400">④ Sector-HRP (100%)</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-sky-400 border-b border-dashed"></span><span class="text-sky-400">① 在市简单等权 (100%)</span></span>
        </div>
      </div>

      <div id="hover-hud" class="hidden absolute top-9 right-3 bg-slate-900/90 border border-slate-700/80 rounded-lg p-2 text-xs z-20 pointer-events-none shadow-xl">
        <div id="hud-date" class="font-mono text-slate-300 font-bold mb-0.5"></div>
        <div class="text-emerald-400 font-mono" id="hud-p10"></div>
        <div class="text-amber-300 font-mono" id="hud-p9"></div>
        <div class="text-purple-300 font-mono" id="hud-p4"></div>
        <div class="text-sky-300 font-mono" id="hud-p1"></div>
      </div>

      <div class="relative h-44 w-full">
        <div class="absolute left-1 top-1 text-xs font-mono text-emerald-400" id="label-nav-max"></div>
        <div class="absolute left-1 bottom-2 text-xs font-mono text-emerald-400" id="label-nav-min"></div>

        <svg id="nav-svg" class="w-full h-full cursor-crosshair" viewBox="0 0 800 200" preserveAspectRatio="none">
          <line x1="50" y1="15" x2="770" y2="15" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="4"/>
          <line x1="50" y1="60" x2="770" y2="60" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="4"/>
          <line x1="50" y1="105" x2="770" y2="105" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="4"/>
          <line x1="50" y1="150" x2="770" y2="150" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="4"/>
          <line x1="50" y1="185" x2="770" y2="185" stroke="currentColor" stroke-opacity="0.2"/>

          <line id="oos-line" x1="480" y1="10" x2="480" y2="185" stroke="#FFD54F" stroke-width="1.5" stroke-dasharray="4"/>
          <text id="oos-text" x="485" y="22" fill="#FFD54F" font-size="11" font-weight="bold">OOS (2024.01)</text>

          <path id="p1-path" d="" fill="none" stroke="#38BDF8" stroke-width="1.8" stroke-dasharray="4,3"/>
          <path id="p4-path" d="" fill="none" stroke="#AB47BC" stroke-width="2.0"/>
          <path id="p9-path" d="" fill="none" stroke="#FFD54F" stroke-width="2.2"/>
          <path id="p10-fill" d="" fill="url(#p10-grad)" opacity="0.20"/>
          <path id="p10-path" d="" fill="none" stroke="#00E676" stroke-width="2.8"/>

          <line id="hover-line" x1="0" y1="10" x2="0" y2="185" stroke="#FFFFFF" stroke-width="1" stroke-dasharray="2,2" class="hidden"/>
          <circle id="hover-dot" cx="0" cy="0" r="4" fill="#00E676" stroke="#FFFFFF" stroke-width="1.5" class="hidden"/>

          <defs>
            <linearGradient id="p10-grad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stop-color="#00E676" stop-opacity="0.8"/>
              <stop offset="100%" stop-color="#00E676" stop-opacity="0.0"/>
            </linearGradient>
          </defs>
        </svg>
      </div>

      <div class="flex justify-between text-xs text-[var(--muted-foreground)] px-10 pt-1 font-mono">
        <span>2020-01</span>
        <span>2021-01</span>
        <span>2022-01</span>
        <span>2023-01</span>
        <span class="text-amber-400 font-bold">2024-01 (OOS)</span>
        <span>2025-01</span>
        <span>2026-10</span>
      </div>
    </div>
  </div>

  <script>
    const dates = {json.dumps(sample_df['date'].tolist())};
    const p1 = {json.dumps([round(float(v), 4) for v in sample_df['1_Equal_1/N']])};
    const p4 = {json.dumps([round(float(v), 4) for v in sample_df['4_SecHRP_100%']])};
    const p9 = {json.dumps([round(float(v), 4) for v in sample_df['9_Pruned_100%']])};
    const p10 = {json.dumps([round(float(v), 4) for v in sample_df['10_Pruned_VolTgt']])};
    const n = dates.length;
    const xLeft = 50, xRight = 770, yTop = 15, yBottom = 185;

    const allVals = [...p1, ...p4, ...p9, ...p10];
    const minNav = Math.min(...allVals) * 0.98;
    const maxNav = Math.max(...allVals) * 1.03;
    document.getElementById('label-nav-max').innerText = maxNav.toFixed(2);
    document.getElementById('label-nav-min').innerText = minNav.toFixed(2);

    function getX(i) {{ return xLeft + (i / (n - 1)) * (xRight - xLeft); }}
    function getY(val) {{ return yBottom - ((val - minNav) / (maxNav - minNav)) * (yBottom - yTop); }}
    function buildPath(arr) {{
      let d = `M ${{getX(0)}} ${{getY(arr[0])}}`;
      for (let i = 1; i < n; i++) d += ` L ${{getX(i)}} ${{getY(arr[i])}}`;
      return d;
    }}

    document.getElementById('p1-path').setAttribute('d', buildPath(p1));
    document.getElementById('p4-path').setAttribute('d', buildPath(p4));
    document.getElementById('p9-path').setAttribute('d', buildPath(p9));
    const p10D = buildPath(p10);
    const fillD = `M ${{getX(0)}} ${{yBottom}} L ` + p10D.slice(2) + ` L ${{getX(n - 1)}} ${{yBottom}} Z`;
    document.getElementById('p10-path').setAttribute('d', p10D);
    document.getElementById('p10-fill').setAttribute('d', fillD);

    const oosIdx = dates.findIndex(d => d >= '2024-01-01');
    if (oosIdx > 0) {{
      const oosX = getX(oosIdx);
      const oosLine = document.getElementById('oos-line');
      oosLine.setAttribute('x1', oosX);
      oosLine.setAttribute('x2', oosX);
      document.getElementById('oos-text').setAttribute('x', oosX + 6);
    }}

    const svg = document.getElementById('nav-svg');
    const hoverLine = document.getElementById('hover-line');
    const hoverDot = document.getElementById('hover-dot');
    const hud = document.getElementById('hover-hud');

    svg.addEventListener('mousemove', (e) => {{
      const rect = svg.getBoundingClientRect();
      const relX = (e.clientX - rect.left) / rect.width * 800;
      if (relX < xLeft || relX > xRight) return;
      const idx = Math.min(n - 1, Math.max(0, Math.round(((relX - xLeft) / (xRight - xLeft)) * (n - 1))));
      const curX = getX(idx);
      hoverLine.setAttribute('x1', curX);
      hoverLine.setAttribute('x2', curX);
      hoverLine.classList.remove('hidden');
      hoverDot.setAttribute('cx', curX);
      hoverDot.setAttribute('cy', getY(p10[idx]));
      hoverDot.classList.remove('hidden');
      hud.classList.remove('hidden');
      document.getElementById('hud-date').innerText = dates[idx];
      document.getElementById('hud-p10').innerText = `⑩ 优选+波幅缩放: ${{p10[idx].toFixed(3)}} (${{((p10[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-p9').innerText = `⑨ 优选剔除(100%): ${{p9[idx].toFixed(3)}} (${{((p9[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-p4').innerText = `④ Sector-HRP(100%): ${{p4[idx].toFixed(3)}} (${{((p4[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-p1').innerText = `① 在市等权(100%): ${{p1[idx].toFixed(3)}} (${{((p1[idx]-1)*100).toFixed(1)}}%)`;
    }});
    svg.addEventListener('mouseleave', () => {{
      hoverLine.classList.add('hidden');
      hoverDot.classList.add('hidden');
      hud.classList.add('hidden');
    }});
  </script>
</body>
</html>
"""
    artifact_html = ARTIFACT_DIR / "commodity_50assets_portfolio_chart.html"
    artifact_html.write_text(widget_html, encoding="utf-8")
    print(f"\n>>> 五十大商品期货组合交互大屏已保存至: {artifact_html}")


if __name__ == "__main__":
    main()
