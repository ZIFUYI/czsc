import json
from pathlib import Path
import pandas as pd
import numpy as np

from examples.multi_factor_cta.test_all_weather_50commodities_portfolio import build_fifty_commodities_universe
from examples.multi_factor_cta.data_engine import CrossAssetDataEngine, map_tf_weight_to_base_grid
from examples.multi_factor_cta.factor_engine import RuleFactorRegistry
from examples.multi_factor_cta.synthesis_engine import IntraAssetThreePillarSynthesizer, CrossAssetHRPSynthesizer, evaluate_single_asset_backtest
from examples.alpha_mining_agent.forensic_pipeline import IS_END_DT

WORKSPACE_DIR = Path(r"d:\czsc\examples\results\alpha_agent_workspace")
data_engine = CrossAssetDataEngine(cache_dir=WORKSPACE_DIR)
registry = RuleFactorRegistry()
intra_synth = IntraAssetThreePillarSynthesizer()
champ_alloc = json.loads((WORKSPACE_DIR / "champion_portfolio.json").read_text(encoding="utf-8"))["alloc_map"]

universe = build_fifty_commodities_universe()
all_returns = {}
all_sectors = {}
all_abs_pos = {}

for short_id, spec in universe:
    if not spec.data_path.exists():
        continue
    tf_dfs = data_engine.load_asset_multi_timeframes(spec, attach_czsc=False)
    df_grid = tf_dfs[spec.base_tf]
    w_on_grid = {}
    active_alloc = {}
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
    all_returns[short_id] = ev["daily_return"]
    all_sectors[short_id] = spec.sector
    dts = pd.to_datetime(df_grid["dt"])
    wd = dts.dt.weekday
    shift_d = np.where(wd == 5, 2, np.where(wd == 6, 1, 0))
    shifted_dts = (dts + pd.to_timedelta(shift_d, unit="D")).dt.normalize()
    pos_s = pd.Series(w_synth, index=shifted_dts)
    all_abs_pos[short_id] = pos_s.abs().groupby(level=0).mean()

# Test 1: Plan A 原版 (23品种)
plan_a_targets = ['ag', 'cu', 'al', 'sn', 'ao', 'si', 'rb', 'hc', 'i', 'jm', 'sa', 'fg', 'sc', 'eb', 'v', 'bu', 'p', 'y', 'oi', 'lh', 'cf', 'cj', 'lc']

# Test 2: Plan A+ (纳回顾宏观底仓黄金 AU, 剔除 SA 纯碱/V PVC/BU 沥青 3个低信噪比品种 -> 21品种)
plan_a_plus_targets = [x for x in plan_a_targets if x not in ('sa', 'v', 'bu')] + ['au']

# Test 3: Plan A++ (更优质的宏观核心组合：保留大趋势品种，加入 AU 黄金 + FU 燃油 + M 豆粕，剔除地产高危 SA/V/BU/I -> 22品种)
plan_a_core_targets = ['ag', 'au', 'cu', 'al', 'sn', 'ao', 'si', 'rb', 'hc', 'jm', 'fg', 'sc', 'eb', 'fu', 'p', 'y', 'oi', 'lh', 'cf', 'cj', 'lc', 'm']

tests = {
    "原版方案A (23品种)": plan_a_targets,
    "方案A+ (纳入AU黄金, 剔除SA/V/BU)": plan_a_plus_targets,
    "方案A++ (宏观精选22品种: AU/AG+LH+HC+SC+P等)": plan_a_core_targets
}

oos_mask = None
res_table = []
for name, targets in tests.items():
    sub_ret = pd.DataFrame({k: all_returns[k] for k in targets}).sort_index().fillna(0.0)
    sub_sec = {k: all_sectors[k] for k in targets}
    mask = sub_ret.notna().astype(float)
    if oos_mask is None:
        oos_mask = sub_ret.index > pd.Timestamp(IS_END_DT.date())

    roll_vol = sub_ret.rolling(60, min_periods=10).std().shift(1).bfill() * np.sqrt(252.0)
    inv_vol = (1.0 / np.maximum(roll_vol.fillna(0.15), 0.05)) * mask
    sec_counts = pd.DataFrame(0.0, index=sub_ret.index, columns=sub_ret.columns)
    for s_name in set(sub_sec.values()):
        cols = [c for c, s in sub_sec.items() if s == s_name]
        cnt = mask[cols].sum(axis=1)
        for c in cols:
            sec_counts[c] = cnt
    raw_hrp = inv_vol / np.sqrt(np.maximum(sec_counts, 1.0))
    w_hrp = raw_hrp.div(raw_hrp.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    dr = (sub_ret * w_hrp).sum(axis=1)

    st_full = CrossAssetHRPSynthesizer._calc_return_series_stats(dr)
    st_is = CrossAssetHRPSynthesizer._calc_return_series_stats(dr.loc[~oos_mask])
    st_oos = CrossAssetHRPSynthesizer._calc_return_series_stats(dr.loc[oos_mask])

    # 50% 实际仓位
    sub_abs = pd.DataFrame({k: all_abs_pos[k] for k in targets}).sort_index()
    p_vol20 = dr.rolling(20, min_periods=10).std().shift(1).bfill() * np.sqrt(252.0)
    v_mult = np.clip(0.025 / np.maximum(p_vol20, 0.015), 0.55, 1.45)
    b_ec = dr.cumsum().shift(1).fillna(0.0)
    b_ec_ma = b_ec.rolling(60, min_periods=10).mean().fillna(0.0)
    e_mult = pd.Series(np.where(b_ec >= b_ec_ma, 1.18, 0.72), index=w_hrp.index)
    u_b = (w_hrp * sub_abs.reindex(w_hrp.index).fillna(0.0)).sum(axis=1)
    u_lag = u_b.shift(1).fillna(0.18)
    dyn_shape = v_mult * e_mult
    kt = (0.50 / np.maximum(u_lag, 0.02)) * dyn_shape
    w_sc = w_hrp.multiply(kt, axis=0)
    dr_sc = (sub_ret.reindex(w_hrp.index) * w_sc).sum(axis=1)
    st_sc_full = CrossAssetHRPSynthesizer._calc_return_series_stats(dr_sc)
    st_sc_oos = CrossAssetHRPSynthesizer._calc_return_series_stats(dr_sc.loc[oos_mask])

    res_table.append({
        "方案名": name,
        "品种数": len(targets),
        "底仓_IS收益": f"{st_is['cum_ret_pct']:+.2f}%",
        "底仓_OOS收益": f"{st_oos['cum_ret_pct']:+.2f}%",
        "底仓_OOS夏普": st_oos['sharpe'],
        "底仓_全期收益": f"{st_full['cum_ret_pct']:+.2f}%",
        "底仓_全期夏普": st_full['sharpe'],
        "底仓_最大回撤": f"{st_full['mdd_pct']:.2f}%",
        "50%仓_OOS收益": f"{st_sc_oos['cum_ret_pct']:+.2f}%",
        "50%仓_OOS夏普": st_sc_oos['sharpe'],
        "50%仓_全期收益": f"{st_sc_full['cum_ret_pct']:+.2f}%",
        "50%仓_全期年化": f"{st_sc_full['ann_ret_pct']:+.2f}%",
        "50%仓_全期夏普": st_sc_full['sharpe'],
        "50%仓_最大回撤": f"{st_sc_full['mdd_pct']:.2f}%",
    })

print(pd.DataFrame(res_table).to_string(index=False))
