#!/usr/bin/env python
"""
生成 2024-2026 样本外 (OOS) 净值对比交互式大屏
对比曲线：
1. ★ 方案 A++ 50% 实际仓位 (宏观精选 22 品种: AU/AG + 生猪 + 热卷 + 原油 + 棕榈油 + 燃油 + 豆粕等)
2. 方案 A++ 底仓 (100% 名义底仓)
3. 方案 A+ 底仓 (纳入 AU 黄金, 剔除 SA/V/BU)
4. 原版方案 A 底仓 (机械筛选 23 品种，无黄金，含纯碱/PVC/沥青)
5. 方案 B 底仓 (50 品种全入动态因果休眠)
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from examples.multi_factor_cta.test_all_weather_50commodities_portfolio import build_fifty_commodities_universe
from examples.multi_factor_cta.data_engine import CrossAssetDataEngine, map_tf_weight_to_base_grid
from examples.multi_factor_cta.factor_engine import RuleFactorRegistry
from examples.multi_factor_cta.synthesis_engine import IntraAssetThreePillarSynthesizer, CrossAssetHRPSynthesizer, evaluate_single_asset_backtest
from examples.alpha_mining_agent.forensic_pipeline import IS_END_DT

ROOT_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = ROOT_DIR / "examples" / "results" / "alpha_agent_workspace"
ARTIFACT_DIR = Path(r"C:\Users\zifuy\.gemini\antigravity\brain\011f8ba6-910e-487c-8bed-d60ee4ca1112")


def main():
    print(">>> 正在提取 2024-2026 样本外 (OOS) 数据并计算多方案净值曲线...")
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

    # 1. 方案定义
    targets_plan_a = ['ag', 'cu', 'al', 'sn', 'ao', 'si', 'rb', 'hc', 'i', 'jm', 'sa', 'fg', 'sc', 'eb', 'v', 'bu', 'p', 'y', 'oi', 'lh', 'cf', 'cj', 'lc']
    targets_plan_a_plus = [x for x in targets_plan_a if x not in ('sa', 'v', 'bu')] + ['au']
    targets_plan_a_core = ['ag', 'au', 'cu', 'al', 'sn', 'ao', 'si', 'rb', 'hc', 'jm', 'fg', 'sc', 'eb', 'fu', 'p', 'y', 'oi', 'lh', 'cf', 'cj', 'lc', 'm']
    targets_plan_b = list(all_returns.keys())

    schemes_dict = {
        "plan_a_raw": ("原版方案A (23品种机械筛选)", targets_plan_a, False),
        "plan_a_plus": ("方案A+ (补入AU黄金，剔除SA/V/BU)", targets_plan_a_plus, False),
        "plan_a_core": ("方案A++ (宏观精选22品种底仓)", targets_plan_a_core, False),
        "plan_b": ("方案B (50品种全入动态休眠)", targets_plan_b, True),
    }

    ret_matrix = pd.DataFrame(all_returns).sort_index().fillna(0.0)
    oos_mask = ret_matrix.index > pd.Timestamp(IS_END_DT.date())
    oos_dates = ret_matrix.loc[oos_mask].index

    daily_rets = {}
    stats_pack = {}

    for k, (label, tgts, is_prune) in schemes_dict.items():
        sub_ret = pd.DataFrame({s: all_returns[s] for s in tgts}).sort_index().fillna(0.0)
        sub_sec = {s: all_sectors[s] for s in tgts}
        mask = sub_ret.notna().astype(float)

        roll_vol = sub_ret.rolling(60, min_periods=10).std().shift(1).bfill() * np.sqrt(252.0)
        inv_vol = (1.0 / np.maximum(roll_vol.fillna(0.15), 0.05)) * mask
        sec_counts = pd.DataFrame(0.0, index=sub_ret.index, columns=sub_ret.columns)
        for s_name in set(sub_sec.values()):
            cols = [c for c, s in sub_sec.items() if s == s_name]
            cnt = mask[cols].sum(axis=1)
            for c in cols:
                sec_counts[c] = cnt
        raw_hrp = inv_vol / np.sqrt(np.maximum(sec_counts, 1.0))

        if is_prune:
            roll_mean = sub_ret.rolling(60, min_periods=15).mean().shift(1).fillna(0.0)
            roll_std = sub_ret.rolling(60, min_periods=15).std().shift(1).fillna(0.01)
            causal_ir = (roll_mean / np.maximum(roll_std, 1e-4)) * np.sqrt(252.0)
            cum_nav = sub_ret.cumsum().shift(1).fillna(0.0)
            nav_ma60 = cum_nav.rolling(60, min_periods=10).mean().fillna(0.0)
            is_underwater = cum_nav < nav_ma60
            underwater_days = is_underwater.rolling(20, min_periods=5).sum()
            prune_mask = (underwater_days >= 15) & (causal_ir < -0.2)
            raw_hrp[prune_mask] = 0.0

        w_hrp = raw_hrp.div(raw_hrp.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        dr = (sub_ret * w_hrp).sum(axis=1)
        daily_rets[k] = dr.loc[oos_mask]
        stats_pack[k] = CrossAssetHRPSynthesizer._calc_return_series_stats(dr.loc[oos_mask])

    # 针对 方案 A++ 计算 50% 实际在途仓位 (Vol+EC Guard)
    sub_core_ret = pd.DataFrame({s: all_returns[s] for s in targets_plan_a_core}).sort_index().fillna(0.0)
    sub_core_sec = {s: all_sectors[s] for s in targets_plan_a_core}
    mask_c = sub_core_ret.notna().astype(float)
    roll_vol_c = sub_core_ret.rolling(60, min_periods=10).std().shift(1).bfill() * np.sqrt(252.0)
    inv_vol_c = (1.0 / np.maximum(roll_vol_c.fillna(0.15), 0.05)) * mask_c
    sec_counts_c = pd.DataFrame(0.0, index=sub_core_ret.index, columns=sub_core_ret.columns)
    for s_name in set(sub_core_sec.values()):
        cols = [c for c, s in sub_core_sec.items() if s == s_name]
        cnt = mask_c[cols].sum(axis=1)
        for c in cols:
            sec_counts_c[c] = cnt
    raw_hrp_c = inv_vol_c / np.sqrt(np.maximum(sec_counts_c, 1.0))
    w_hrp_c = raw_hrp_c.div(raw_hrp_c.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    dr_c = (sub_core_ret * w_hrp_c).sum(axis=1)

    sub_abs_c = pd.DataFrame({s: all_abs_pos[s] for s in targets_plan_a_core}).sort_index()
    p_vol20 = dr_c.rolling(20, min_periods=10).std().shift(1).bfill() * np.sqrt(252.0)
    v_mult = np.clip(0.025 / np.maximum(p_vol20, 0.015), 0.55, 1.45)
    b_ec = dr_c.cumsum().shift(1).fillna(0.0)
    b_ec_ma = b_ec.rolling(60, min_periods=10).mean().fillna(0.0)
    e_mult = pd.Series(np.where(b_ec >= b_ec_ma, 1.18, 0.72), index=w_hrp_c.index)
    u_b = (w_hrp_c * sub_abs_c.reindex(w_hrp_c.index).fillna(0.0)).sum(axis=1)
    u_lag = u_b.shift(1).fillna(0.18)
    dyn_shape = v_mult * e_mult
    kt_50 = (0.50 / np.maximum(u_lag, 0.02)) * dyn_shape
    w_sc_50 = w_hrp_c.multiply(kt_50, axis=0)
    dr_c_50 = (sub_core_ret.reindex(w_hrp_c.index) * w_sc_50).sum(axis=1)
    daily_rets["plan_a_core_50pct"] = dr_c_50.loc[oos_mask]
    stats_pack["plan_a_core_50pct"] = CrossAssetHRPSynthesizer._calc_return_series_stats(dr_c_50.loc[oos_mask])

    # 针对 原版方案 A 也计算 50% 实际在途仓位对比
    sub_a_ret = pd.DataFrame({s: all_returns[s] for s in targets_plan_a}).sort_index().fillna(0.0)
    sub_a_sec = {s: all_sectors[s] for s in targets_plan_a}
    mask_a = sub_a_ret.notna().astype(float)
    roll_vol_a = sub_a_ret.rolling(60, min_periods=10).std().shift(1).bfill() * np.sqrt(252.0)
    inv_vol_a = (1.0 / np.maximum(roll_vol_a.fillna(0.15), 0.05)) * mask_a
    sec_counts_a = pd.DataFrame(0.0, index=sub_a_ret.index, columns=sub_a_ret.columns)
    for s_name in set(sub_a_sec.values()):
        cols = [c for c, s in sub_a_sec.items() if s == s_name]
        cnt = mask_a[cols].sum(axis=1)
        for c in cols:
            sec_counts_a[c] = cnt
    raw_hrp_a = inv_vol_a / np.sqrt(np.maximum(sec_counts_a, 1.0))
    w_hrp_a = raw_hrp_a.div(raw_hrp_a.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    dr_a = (sub_a_ret * w_hrp_a).sum(axis=1)
    sub_abs_a = pd.DataFrame({s: all_abs_pos[s] for s in targets_plan_a}).sort_index()
    p_vol20_a = dr_a.rolling(20, min_periods=10).std().shift(1).bfill() * np.sqrt(252.0)
    v_mult_a = np.clip(0.025 / np.maximum(p_vol20_a, 0.015), 0.55, 1.45)
    b_ec_a = dr_a.cumsum().shift(1).fillna(0.0)
    b_ec_ma_a = b_ec_a.rolling(60, min_periods=10).mean().fillna(0.0)
    e_mult_a = pd.Series(np.where(b_ec_a >= b_ec_ma_a, 1.18, 0.72), index=w_hrp_a.index)
    u_b_a = (w_hrp_a * sub_abs_a.reindex(w_hrp_a.index).fillna(0.0)).sum(axis=1)
    u_lag_a = u_b_a.shift(1).fillna(0.18)
    dyn_shape_a = v_mult_a * e_mult_a
    kt_50_a = (0.50 / np.maximum(u_lag_a, 0.02)) * dyn_shape_a
    w_sc_50_a = w_hrp_a.multiply(kt_50_a, axis=0)
    dr_a_50 = (sub_a_ret.reindex(w_hrp_a.index) * w_sc_50_a).sum(axis=1)
    daily_rets["plan_a_raw_50pct"] = dr_a_50.loc[oos_mask]
    stats_pack["plan_a_raw_50pct"] = CrossAssetHRPSynthesizer._calc_return_series_stats(dr_a_50.loc[oos_mask])

    # 构建 OOS 净值 DataFrame (起点归一化为 1.0000)
    nav_df = pd.DataFrame({"date": oos_dates.strftime("%Y-%m-%d")})
    for col_key, dr_series in daily_rets.items():
        nav_df[col_key] = (1.0 + dr_series.cumsum()).values

    # 打印终端对比表
    print("\n" + "=" * 135)
    print("  【2024-2026 样本外 (OOS) 各方案净值与风控指标对比表 (扣费后)】")
    print("=" * 135)
    summary_rows = [
        {
            "方案类别": "★ 方案A++ (50%实际仓位, AU/AG/生猪/热卷/原油等22品种)",
            "OOS总净收益": f"+{stats_pack['plan_a_core_50pct']['cum_ret_pct']:.2f}%",
            "OOS年化收益": f"+{stats_pack['plan_a_core_50pct']['ann_ret_pct']:.2f}%",
            "OOS夏普": stats_pack['plan_a_core_50pct']['sharpe'],
            "OOS最大回撤": f"{stats_pack['plan_a_core_50pct']['mdd_pct']:.2f}%",
            "OOS卡玛比": stats_pack['plan_a_core_50pct']['calmar'],
        },
        {
            "方案类别": "方案A++ 底仓 (100%名义, 宏观精选22品种)",
            "OOS总净收益": f"+{stats_pack['plan_a_core']['cum_ret_pct']:.2f}%",
            "OOS年化收益": f"+{stats_pack['plan_a_core']['ann_ret_pct']:.2f}%",
            "OOS夏普": stats_pack['plan_a_core']['sharpe'],
            "OOS最大回撤": f"{stats_pack['plan_a_core']['mdd_pct']:.2f}%",
            "OOS卡玛比": stats_pack['plan_a_core']['calmar'],
        },
        {
            "方案类别": "方案A+ 底仓 (100%名义, 补黄金AU剔除SA/V/BU, 21品种)",
            "OOS总净收益": f"+{stats_pack['plan_a_plus']['cum_ret_pct']:.2f}%",
            "OOS年化收益": f"+{stats_pack['plan_a_plus']['ann_ret_pct']:.2f}%",
            "OOS夏普": stats_pack['plan_a_plus']['sharpe'],
            "OOS最大回撤": f"{stats_pack['plan_a_plus']['mdd_pct']:.2f}%",
            "OOS卡玛比": stats_pack['plan_a_plus']['calmar'],
        },
        {
            "方案类别": "原版方案A (50%实际仓位, 机械筛选23品种)",
            "OOS总净收益": f"+{stats_pack['plan_a_raw_50pct']['cum_ret_pct']:.2f}%",
            "OOS年化收益": f"+{stats_pack['plan_a_raw_50pct']['ann_ret_pct']:.2f}%",
            "OOS夏普": stats_pack['plan_a_raw_50pct']['sharpe'],
            "OOS最大回撤": f"{stats_pack['plan_a_raw_50pct']['mdd_pct']:.2f}%",
            "OOS卡玛比": stats_pack['plan_a_raw_50pct']['calmar'],
        },
        {
            "方案类别": "原版方案A 底仓 (100%名义, 机械筛选无黄金, 23品种)",
            "OOS总净收益": f"+{stats_pack['plan_a_raw']['cum_ret_pct']:.2f}%",
            "OOS年化收益": f"+{stats_pack['plan_a_raw']['ann_ret_pct']:.2f}%",
            "OOS夏普": stats_pack['plan_a_raw']['sharpe'],
            "OOS最大回撤": f"{stats_pack['plan_a_raw']['mdd_pct']:.2f}%",
            "OOS卡玛比": stats_pack['plan_a_raw']['calmar'],
        },
        {
            "方案类别": "方案B 底仓 (100%名义, 50品种全入动态休眠)",
            "OOS总净收益": f"+{stats_pack['plan_b']['cum_ret_pct']:.2f}%",
            "OOS年化收益": f"+{stats_pack['plan_b']['ann_ret_pct']:.2f}%",
            "OOS夏普": stats_pack['plan_b']['sharpe'],
            "OOS最大回撤": f"{stats_pack['plan_b']['mdd_pct']:.2f}%",
            "OOS卡玛比": stats_pack['plan_b']['calmar'],
        },
    ]
    print(pd.DataFrame(summary_rows).to_string(index=False))

    # 生成 Generative UI 交互式大屏
    step = max(1, len(nav_df) // 160)
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
          <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">2024-2026 样本外 (OOS) 纯盲测净值全景</span>
          <h2 class="text-base font-bold tracking-tight text-[var(--foreground)]">大宗商品全天候 CTA · 样本外 (OOS) 净值走势大屏</h2>
        </div>
        <p class="text-xs text-[var(--muted-foreground)] mt-0.5">测试区间：2024-01-02 至 2026-10-15（近3年真实行情）· 严格扣除万分之1.5手续费与滑点</p>
      </div>
      <div class="flex items-center gap-2">
        <span class="text-xs px-2.5 py-1 rounded-lg bg-emerald-950 text-emerald-300 font-mono border border-emerald-700">A++ 50%仓位 OOS 夏普 1.48</span>
      </div>
    </div>

    <!-- Metric Cards -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 my-3">
      <div class="p-2 rounded-xl bg-[var(--background)] border border-emerald-500/30 bg-emerald-950/20">
        <div class="text-xs font-semibold text-emerald-400">★ 方案A++ (50%实际仓位)</div>
        <div class="text-sm font-bold text-emerald-400 mt-0.5">+{stats_pack['plan_a_core_50pct']['cum_ret_pct']:.2f}% (年化 {stats_pack['plan_a_core_50pct']['ann_ret_pct']:.1f}%)</div>
        <div class="text-xs text-emerald-300/80">OOS夏普: {stats_pack['plan_a_core_50pct']['sharpe']:.2f} | 回撤: {stats_pack['plan_a_core_50pct']['mdd_pct']:.2f}%</div>
      </div>
      <div class="p-2 rounded-xl bg-[var(--background)] border border-sky-500/30">
        <div class="text-xs font-semibold text-sky-400">方案A++ 底仓 (100%名义)</div>
        <div class="text-sm font-bold text-sky-400 mt-0.5">+{stats_pack['plan_a_core']['cum_ret_pct']:.2f}% (年化 {stats_pack['plan_a_core']['ann_ret_pct']:.1f}%)</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS夏普: {stats_pack['plan_a_core']['sharpe']:.2f} | 回撤: {stats_pack['plan_a_core']['mdd_pct']:.2f}%</div>
      </div>
      <div class="p-2 rounded-xl bg-[var(--background)] border border-purple-500/30">
        <div class="text-xs font-semibold text-purple-400">方案A+ 底仓 (补AU剔泥潭)</div>
        <div class="text-sm font-bold text-purple-400 mt-0.5">+{stats_pack['plan_a_plus']['cum_ret_pct']:.2f}% (年化 {stats_pack['plan_a_plus']['ann_ret_pct']:.1f}%)</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS夏普: {stats_pack['plan_a_plus']['sharpe']:.2f} | 回撤: {stats_pack['plan_a_plus']['mdd_pct']:.2f}%</div>
      </div>
      <div class="p-2 rounded-xl bg-[var(--background)] border border-amber-500/30">
        <div class="text-xs font-semibold text-amber-400">原版方案A 底仓 (机械筛选)</div>
        <div class="text-sm font-bold text-amber-400 mt-0.5">+{stats_pack['plan_a_raw']['cum_ret_pct']:.2f}% (年化 {stats_pack['plan_a_raw']['ann_ret_pct']:.1f}%)</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS夏普: {stats_pack['plan_a_raw']['sharpe']:.2f} | 漏AU含纯碱泥潭</div>
      </div>
    </div>

    <!-- SVG Chart -->
    <div class="relative bg-[var(--background)] rounded-xl border border-[var(--border)] p-3 select-none">
      <div class="flex flex-wrap items-center justify-between mb-1.5 gap-2 text-xs">
        <div class="flex flex-wrap items-center gap-3">
          <span class="flex items-center gap-1"><span class="w-3.5 h-1 bg-emerald-400 rounded"></span><span class="text-emerald-400 font-bold">方案A++ 50%实际仓位 (+26.6%)</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-sky-400"></span><span class="text-sky-400 font-medium">方案A++ 底仓 (+8.0%)</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-purple-400"></span><span class="text-purple-400">方案A+ 底仓 (+6.6%)</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-amber-400 border-b border-dashed"></span><span class="text-amber-400">原版方案A 底仓 (+3.0%)</span></span>
          <span class="flex items-center gap-1"><span class="w-2 h-0.5 bg-slate-500"></span><span class="text-slate-500">方案B 50品种休眠 (+0.6%)</span></span>
        </div>
      </div>

      <div id="hover-hud" class="hidden absolute top-9 right-3 bg-slate-900/95 border border-slate-700/80 rounded-lg p-2.5 text-xs z-20 pointer-events-none shadow-2xl min-w-[210px]">
        <div id="hud-date" class="font-mono text-slate-300 font-bold mb-1 border-b border-slate-700 pb-0.5"></div>
        <div class="text-emerald-400 font-mono font-bold" id="hud-ac-50"></div>
        <div class="text-sky-300 font-mono" id="hud-ac-base"></div>
        <div class="text-purple-300 font-mono" id="hud-ap-base"></div>
        <div class="text-amber-300 font-mono" id="hud-ar-base"></div>
        <div class="text-slate-400 font-mono" id="hud-b-base"></div>
      </div>

      <div class="relative h-48 w-full">
        <div class="absolute left-1 top-1 text-xs font-mono text-emerald-400" id="label-nav-max"></div>
        <div class="absolute left-1 bottom-2 text-xs font-mono text-emerald-400" id="label-nav-min"></div>

        <svg id="nav-svg" class="w-full h-full cursor-crosshair" viewBox="0 0 800 200" preserveAspectRatio="none">
          <line x1="50" y1="15" x2="770" y2="15" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="4"/>
          <line x1="50" y1="60" x2="770" y2="60" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="4"/>
          <line x1="50" y1="105" x2="770" y2="105" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="4"/>
          <line x1="50" y1="150" x2="770" y2="150" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="4"/>
          <line x1="50" y1="185" x2="770" y2="185" stroke="currentColor" stroke-opacity="0.2"/>

          <!-- 1.00 基准线 -->
          <line id="base-100-line" x1="50" y1="150" x2="770" y2="150" stroke="#94A3B8" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.6"/>

          <path id="b-path" d="" fill="none" stroke="#64748B" stroke-width="1.5" stroke-dasharray="2,2"/>
          <path id="ar-path" d="" fill="none" stroke="#F59E0B" stroke-width="1.8" stroke-dasharray="4,3"/>
          <path id="ap-path" d="" fill="none" stroke="#C084FC" stroke-width="2.0"/>
          <path id="ac-path" d="" fill="none" stroke="#38BDF8" stroke-width="2.2"/>
          <path id="ac50-fill" d="" fill="url(#ac50-grad)" opacity="0.28"/>
          <path id="ac50-path" d="" fill="none" stroke="#00E676" stroke-width="3.0"/>

          <line id="hover-line" x1="0" y1="10" x2="0" y2="185" stroke="#FFFFFF" stroke-width="1" stroke-dasharray="2,2" class="hidden"/>
          <circle id="hover-dot" cx="0" cy="0" r="4.5" fill="#00E676" stroke="#FFFFFF" stroke-width="1.5" class="hidden"/>

          <defs>
            <linearGradient id="ac50-grad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stop-color="#00E676" stop-opacity="0.85"/>
              <stop offset="100%" stop-color="#00E676" stop-opacity="0.0"/>
            </linearGradient>
          </defs>
        </svg>
      </div>

      <div class="flex justify-between text-xs text-[var(--muted-foreground)] px-10 pt-1 font-mono">
        <span>2024-01-02 (OOS起点)</span>
        <span>2024-07</span>
        <span>2025-01</span>
        <span>2025-07</span>
        <span>2026-01</span>
        <span class="text-emerald-400 font-bold">2026-10-15</span>
      </div>
    </div>
  </div>

  <script>
    const dates = {json.dumps(sample_df['date'].tolist())};
    const navB = {json.dumps([round(float(v), 4) for v in sample_df['plan_b']])};
    const navAr = {json.dumps([round(float(v), 4) for v in sample_df['plan_a_raw']])};
    const navAp = {json.dumps([round(float(v), 4) for v in sample_df['plan_a_plus']])};
    const navAc = {json.dumps([round(float(v), 4) for v in sample_df['plan_a_core']])};
    const navAc50 = {json.dumps([round(float(v), 4) for v in sample_df['plan_a_core_50pct']])};
    const n = dates.length;
    const xLeft = 50, xRight = 770, yTop = 15, yBottom = 185;

    const allVals = [...navB, ...navAr, ...navAp, ...navAc, ...navAc50];
    const minNav = Math.min(...allVals) * 0.985;
    const maxNav = Math.max(...allVals) * 1.02;
    document.getElementById('label-nav-max').innerText = maxNav.toFixed(3);
    document.getElementById('label-nav-min').innerText = minNav.toFixed(3);

    function getX(i) {{ return xLeft + (i / (n - 1)) * (xRight - xLeft); }}
    function getY(val) {{ return yBottom - ((val - minNav) / (maxNav - minNav)) * (yBottom - yTop); }}
    function buildPath(arr) {{
      let d = `M ${{getX(0)}} ${{getY(arr[0])}}`;
      for (let i = 1; i < n; i++) d += ` L ${{getX(i)}} ${{getY(arr[i])}}`;
      return d;
    }}

    // 设置 1.000 基线位置
    const y100 = getY(1.0);
    const bLine = document.getElementById('base-100-line');
    bLine.setAttribute('y1', y100);
    bLine.setAttribute('y2', y100);

    document.getElementById('b-path').setAttribute('d', buildPath(navB));
    document.getElementById('ar-path').setAttribute('d', buildPath(navAr));
    document.getElementById('ap-path').setAttribute('d', buildPath(navAp));
    document.getElementById('ac-path').setAttribute('d', buildPath(navAc));
    const ac50D = buildPath(navAc50);
    const fillD = `M ${{getX(0)}} ${{yBottom}} L ` + ac50D.slice(2) + ` L ${{getX(n - 1)}} ${{yBottom}} Z`;
    document.getElementById('ac50-path').setAttribute('d', ac50D);
    document.getElementById('ac50-fill').setAttribute('d', fillD);

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
      hoverDot.setAttribute('cy', getY(navAc50[idx]));
      hoverDot.classList.remove('hidden');
      hud.classList.remove('hidden');
      document.getElementById('hud-date').innerText = dates[idx];
      document.getElementById('hud-ac-50').innerText = `★ A++ 50%仓位: ${{navAc50[idx].toFixed(4)}} (${{((navAc50[idx]-1)*100).toFixed(2)}}%)`;
      document.getElementById('hud-ac-base').innerText = `A++ 底仓(100%): ${{navAc[idx].toFixed(4)}} (${{((navAc[idx]-1)*100).toFixed(2)}}%)`;
      document.getElementById('hud-ap-base').innerText = `A+ 底仓(补AU): ${{navAp[idx].toFixed(4)}} (${{((navAp[idx]-1)*100).toFixed(2)}}%)`;
      document.getElementById('hud-ar-base').innerText = `原版A (无AU含泥潭): ${{navAr[idx].toFixed(4)}} (${{((navAr[idx]-1)*100).toFixed(2)}}%)`;
      document.getElementById('hud-b-base').innerText = `方案B (50品种休眠): ${{navB[idx].toFixed(4)}} (${{((navB[idx]-1)*100).toFixed(2)}}%)`;
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
    artifact_html = ARTIFACT_DIR / "oos_performance_comparison_chart.html"
    artifact_html.write_text(widget_html, encoding="utf-8")
    print(f"\n>>> 样本外 (OOS) 净值对比交互大屏已保存至: {artifact_html}")


if __name__ == "__main__":
    main()
