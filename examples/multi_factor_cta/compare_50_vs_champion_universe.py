#!/usr/bin/env python
"""
全天候 CTA 策略族：五十大全市场全入 VS 二十四大正阿尔法优选商品池对比实证
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from examples.multi_factor_cta.test_all_weather_50commodities_portfolio import (
    build_fifty_commodities_universe,
    evaluate_50_weighting_schemes,
)
from examples.multi_factor_cta.data_engine import CrossAssetDataEngine, map_tf_weight_to_base_grid
from examples.multi_factor_cta.factor_engine import RuleFactorRegistry
from examples.multi_factor_cta.synthesis_engine import (
    CrossAssetHRPSynthesizer,
    IntraAssetThreePillarSynthesizer,
    evaluate_single_asset_backtest,
)
from examples.alpha_mining_agent.forensic_pipeline import IS_END_DT
from examples.alpha_mining_agent.hypothesis_bank import (
    ARCH_A,
    ARCH_C,
    LEGACY_ARCHETYPE_ALIAS,
    LEGACY_ARCHETYPE_BY_SIG,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = ROOT_DIR / "examples" / "results" / "alpha_agent_workspace"
ARTIFACT_DIR = Path(r"C:\Users\zifuy\.gemini\antigravity\brain\011f8ba6-910e-487c-8bed-d60ee4ca1112")

# 24 大在零样本迁移下产生正超额的优质商品主力
TOP_24_POSITIVE = {
    # 贵金属 (2)
    "ag",
    "au",
    # 有色与特钢材料 (7)
    "cu",
    "al",
    "sn",
    "ao",
    "si",
    # 黑色建材 (4)
    "rb",
    "hc",
    "i",
    "jm",
    "fg",
    # 能化 (4)
    "sc",
    "eb",
    "fu",
    # 农产软商 (8)
    "lh",
    "p",
    "cj",
    "oi",
    "cf",
    "y",
    "m",
    # 新能源材料 (2)
    "lc",
    "ps",
}


def main():
    data_engine = CrossAssetDataEngine(cache_dir=WORKSPACE_DIR)
    registry = RuleFactorRegistry()
    intra_synth = IntraAssetThreePillarSynthesizer()
    champ_alloc = json.loads((WORKSPACE_DIR / "champion_portfolio.json").read_text(encoding="utf-8"))["alloc_map"]

    all_specs = build_fifty_commodities_universe()
    top24_specs = [x for x in all_specs if x[0] in TOP_24_POSITIVE]

    top24_returns = {}
    top24_sectors = {}
    top24_abs_pos = {}

    for short_id, spec in top24_specs:
        tf_dfs = data_engine.load_asset_multi_timeframes(spec, attach_czsc=False)
        df_grid = tf_dfs[spec.base_tf]

        w_on_grid = {}
        active_alloc = {}
        for uid, w_val in champ_alloc.items():
            sid, tf = uid.split("@")
            if tf not in spec.eligible_tfs or sid.startswith("CZ") or sid.startswith("M5"):
                continue
            sig_spec = registry._factors.get(sid)
            # 保持原对比口径：仅纳入旧 A_顺势突破簇 / C_震荡回归簇 (6 簇重构后按迁移前的旧簇过滤)
            if sig_spec is not None:
                legacy_arch = LEGACY_ARCHETYPE_BY_SIG.get(sid, sig_spec.archetype)
                legacy_arch = LEGACY_ARCHETYPE_ALIAS.get(legacy_arch, legacy_arch)
                if legacy_arch not in (ARCH_A, ARCH_C):
                    continue
            _, w_arr = registry.evaluate_factor_on_tf(tf_dfs[tf], sid, tf)
            w_on_grid[uid] = map_tf_weight_to_base_grid(tf_dfs[tf], w_arr, df_grid)
            active_alloc[uid] = w_val

        w_synth = intra_synth.synthesize(df_grid, w_on_grid, active_alloc, spec, enable_regime_gating=True)
        ev = evaluate_single_asset_backtest(df_grid, w_synth, spec.symbol, fee_rate=spec.fee_rate)
        top24_returns[short_id] = ev["daily_return"]
        top24_sectors[short_id] = spec.sector

        dts = pd.to_datetime(df_grid["dt"])
        wd = dts.dt.weekday
        shift_d = np.where(wd == 5, 2, np.where(wd == 6, 1, 0))
        shifted_dts = (dts + pd.to_timedelta(shift_d, unit="D")).dt.normalize()
        pos_s = pd.Series(w_synth, index=shifted_dts)
        top24_abs_pos[short_id] = pos_s.abs().groupby(level=0).mean()

    res_24 = evaluate_50_weighting_schemes(top24_returns, top24_sectors)

    print("\n" + "=" * 145)
    print("  【二十四大优质正阿尔法商品期货优选组合 —— 加权方案对比表】")
    print("=" * 145)
    rows = []
    for _k, v in res_24.items():
        rows.append(
            {
                "加权方案": v["label"],
                "名义预算": f"{v['avg_gross_exposure_pct']:.1f}%",
                "IS净收益": f"{v['is']['cum_ret_pct']:+.2f}%",
                "IS夏普": v["is"]["sharpe"],
                "OOS净收益": f"{v['oos']['cum_ret_pct']:+.2f}%",
                "OOS夏普": v["oos"]["sharpe"],
                "全周期总收益": f"{v['full']['cum_ret_pct']:+.2f}%",
                "全周期年化": f"{v['full']['ann_ret_pct']:+.2f}%",
                "全周期夏普": v["full"]["sharpe"],
                "最大回撤": f"{v['full']['mdd_pct']:.2f}%",
                "全周期卡玛": v["full"]["calmar"],
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    # 实际仓位
    abs_pos_df = pd.DataFrame(top24_abs_pos).sort_index()
    ret_df = pd.DataFrame(top24_returns).sort_index().fillna(0.0)
    base_w = res_24["4_SecHRP_100%"]["weights_df"]
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

    tier_records = []
    for tier_name, target_u in [
        ("基准18%实际仓位", 0.18),
        ("平衡50%实际仓位", 0.50),
        ("进取80%实际仓位", 0.80),
        ("满仓100%实际仓位", 1.00),
    ]:
        k_t = (target_u / np.maximum(u_lag1, 0.02)) * dynamic_shape
        w_scaled = base_w.multiply(k_t, axis=0)
        act_gross_exposure = (w_scaled * sub_abs).sum(axis=1)
        dr_scaled = (ret_df.reindex(base_w.index) * w_scaled).sum(axis=1)
        st_full = CrossAssetHRPSynthesizer._calc_return_series_stats(dr_scaled)
        st_oos = CrossAssetHRPSynthesizer._calc_return_series_stats(
            dr_scaled.loc[ret_df.index > pd.Timestamp(IS_END_DT.date())]
        )
        tier_records.append(
            {
                "仓位级别": tier_name,
                "平均实际仓位": f"{act_gross_exposure.mean() * 100:.1f}%",
                "平均保证金": f"{act_gross_exposure.mean() * 12:.1f}%",
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
    print("  【Top 24 优质正阿尔法组合 · SecHRP】 —— 四档真实在途持仓 (Vol+EC Guard 动态守护)")
    print("-" * 145)
    print(pd.DataFrame(tier_records).to_string(index=False))


if __name__ == "__main__":
    main()
