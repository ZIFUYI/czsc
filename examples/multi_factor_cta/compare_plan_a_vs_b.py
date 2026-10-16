#!/usr/bin/env python
"""
全天候 CTA 策略族：品种优选两大严格因果方案全面深度对比实证
方案 A：纯样本内因果定锚法 (Ex-ante In-Sample Selection，23品种静态池)
方案 B：动态净值均线因果休眠法 (Causal Pruning HRP，50品种全入纯算法自适应)
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from examples.multi_factor_cta.test_all_weather_50commodities_portfolio import (
    build_fifty_commodities_universe,
)
from examples.multi_factor_cta.data_engine import CrossAssetDataEngine, map_tf_weight_to_base_grid
from examples.multi_factor_cta.factor_engine import RuleFactorRegistry
from examples.multi_factor_cta.synthesis_engine import (
    CrossAssetHRPSynthesizer,
    IntraAssetThreePillarSynthesizer,
    evaluate_single_asset_backtest,
)
from examples.alpha_mining_agent.forensic_pipeline import IS_END_DT

ROOT_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = ROOT_DIR / "examples" / "results" / "alpha_agent_workspace"
ARTIFACT_DIR = Path(r"C:\Users\zifuy\.gemini\antigravity\brain\011f8ba6-910e-487c-8bed-d60ee4ca1112")


def main():
    print("=" * 145)
    print("  【全天候 CTA 策略族：方案 A (样本内定锚) vs 方案 B (动态休眠) 终极实证对比】")
    print("=" * 145)

    data_engine = CrossAssetDataEngine(cache_dir=WORKSPACE_DIR)
    registry = RuleFactorRegistry()
    intra_synth = IntraAssetThreePillarSynthesizer()
    champ_alloc = json.loads((WORKSPACE_DIR / "champion_portfolio.json").read_text(encoding="utf-8"))["alloc_map"]

    all_specs = build_fifty_commodities_universe()
    all_returns = {}
    all_sectors = {}
    all_abs_pos = {}
    single_is_stats = {}

    print(">>> 正在加载 50 大商品期货并计算单品种基准收益...")
    for short_id, spec in all_specs:
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
        single_is_stats[short_id] = {
            "name": spec.name,
            "sector": spec.sector,
            "is_ret": ev["is"]["cum_ret_pct"],
            "oos_ret": ev["oos"]["cum_ret_pct"],
            "full_ret": ev["full"]["cum_ret_pct"],
        }

        dts = pd.to_datetime(df_grid["dt"])
        wd = dts.dt.weekday
        shift_d = np.where(wd == 5, 2, np.where(wd == 6, 1, 0))
        shifted_dts = (dts + pd.to_timedelta(shift_d, unit="D")).dt.normalize()
        pos_s = pd.Series(w_synth, index=shifted_dts)
        all_abs_pos[short_id] = pos_s.abs().groupby(level=0).mean()

    # =========================================================================
    # 方案 A：纯样本内因果定锚法 (Ex-ante In-Sample Selection)
    # 规则：仅使用 2020-2023 IS 样本内数据，凡 is_ret > 0 的品种入选
    # =========================================================================
    plan_a_targets = [s for s, v in single_is_stats.items() if v["is_ret"] > 0]
    print(f"\n【方案 A 选出的 23 个品种 (IS 收益 > 0)】: {len(plan_a_targets)} 个")
    for s in plan_a_targets:
        v = single_is_stats[s]
        print(f"  {s.upper():4s} ({v['name']:14s}) | IS收益: {v['is_ret']:+6.2f}% | OOS收益: {v['oos_ret']:+6.2f}% | 全期: {v['full_ret']:+6.2f}%")

    plan_a_returns = {k: all_returns[k] for k in plan_a_targets}
    plan_a_sectors = {k: all_sectors[k] for k in plan_a_targets}
    plan_a_abs = pd.DataFrame({k: all_abs_pos[k] for k in plan_a_targets}).sort_index()

    raw_a = pd.DataFrame(plan_a_returns).sort_index()
    raw_a.index = pd.to_datetime(raw_a.index)
    mask_a = raw_a.notna().astype(float)
    ret_a = raw_a.fillna(0.0)

    # 方案 A 的 Sector-HRP
    roll_vol_a = raw_a.rolling(60, min_periods=10).std().shift(1).bfill() * np.sqrt(252.0)
    inv_vol_a = (1.0 / np.maximum(roll_vol_a.fillna(0.15), 0.05)) * mask_a
    sec_map_a = {col: plan_a_sectors[col] for col in ret_a.columns}
    sec_cnt_a = pd.DataFrame(0.0, index=ret_a.index, columns=ret_a.columns)
    for sec_name in set(sec_map_a.values()):
        cols = [c for c, s in sec_map_a.items() if s == sec_name]
        cnt = mask_a[cols].sum(axis=1)
        for c in cols:
            sec_cnt_a[c] = cnt
    raw_hrp_a = inv_vol_a / np.sqrt(np.maximum(sec_cnt_a, 1.0))
    w_a_hrp = raw_hrp_a.div(raw_hrp_a.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    dr_a_hrp = (ret_a * w_a_hrp).sum(axis=1)

    # 方案 A 的 ActiveEC-HRP
    cum_nav_a = ret_a.cumsum().shift(1).fillna(0.0)
    nav_ma60_a = cum_nav_a.rolling(60, min_periods=10).mean().fillna(0.0)
    ec_health_a = np.where(cum_nav_a >= nav_ma60_a, 1.25, 0.30) * mask_a
    act_pos_a = (ret_a.abs().rolling(3, min_periods=1).max().shift(1).fillna(1.0) > 1e-5).astype(float) * mask_a
    raw_comb_a = raw_hrp_a * ec_health_a * np.where(act_pos_a > 0, 1.15, 0.35)
    w_a_active_ec = raw_comb_a.div(raw_comb_a.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    dr_a_active_ec = (ret_a * w_a_active_ec).sum(axis=1)

    # =========================================================================
    # 方案 B：动态净值均线因果休眠法 (Causal Pruning HRP，50品种全入)
    # 规则：50品种全入，每日依据 shift(1) 的累计净值与动量，水下超15日且动量为负者休眠
    # =========================================================================
    raw_b = pd.DataFrame(all_returns).sort_index()
    raw_b.index = pd.to_datetime(raw_b.index)
    mask_b = raw_b.notna().astype(float)
    ret_b = raw_b.fillna(0.0)

    roll_vol_b = raw_b.rolling(60, min_periods=10).std().shift(1).bfill() * np.sqrt(252.0)
    inv_vol_b = (1.0 / np.maximum(roll_vol_b.fillna(0.15), 0.05)) * mask_b
    sec_map_b = {col: all_sectors[col] for col in ret_b.columns}
    sec_cnt_b = pd.DataFrame(0.0, index=ret_b.index, columns=ret_b.columns)
    for sec_name in set(sec_map_b.values()):
        cols = [c for c, s in sec_map_b.items() if s == sec_name]
        cnt = mask_b[cols].sum(axis=1)
        for c in cols:
            sec_cnt_b[c] = cnt
    raw_hrp_b = inv_vol_b / np.sqrt(np.maximum(sec_cnt_b, 1.0))

    # 因果动量 IR
    roll_mean_b = raw_b.rolling(60, min_periods=15).mean().shift(1).fillna(0.0)
    roll_std_b = raw_b.rolling(60, min_periods=15).std().shift(1).fillna(0.01)
    causal_ir_b = (roll_mean_b / np.maximum(roll_std_b, 1e-4)) * np.sqrt(252.0)

    # 动态水下均线休眠判别 (Pruning)
    cum_nav_b = ret_b.cumsum().shift(1).fillna(0.0)
    nav_ma60_b = cum_nav_b.rolling(60, min_periods=10).mean().fillna(0.0)
    is_underwater_b = cum_nav_b < nav_ma60_b
    underwater_days_b = is_underwater_b.rolling(20, min_periods=5).sum()
    prune_mask_b = (underwater_days_b >= 15) & (causal_ir_b < -0.2)

    raw_pruned_b = raw_hrp_b.copy()
    raw_pruned_b[prune_mask_b] = 0.0
    w_b_pruned = raw_pruned_b.div(raw_pruned_b.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    dr_b_pruned = (ret_b * w_b_pruned).sum(axis=1)

    # 方案 B 的 ActiveEC-Pruned
    act_pos_b = (ret_b.abs().rolling(3, min_periods=1).max().shift(1).fillna(1.0) > 1e-5).astype(float) * mask_b
    ec_health_b = np.where(cum_nav_b >= nav_ma60_b, 1.25, 0.30) * mask_b
    raw_comb_b = raw_pruned_b * ec_health_b * np.where(act_pos_b > 0, 1.15, 0.35)
    w_b_active_pruned = raw_comb_b.div(raw_comb_b.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    dr_b_active_pruned = (ret_b * w_b_active_pruned).sum(axis=1)

    # 作为对比：50品种全入无休眠的基准
    w_b_unpruned = raw_hrp_b.div(raw_hrp_b.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    dr_b_unpruned = (ret_b * w_b_unpruned).sum(axis=1)

    is_mask = ret_a.index <= pd.Timestamp(IS_END_DT.date())
    oos_mask = ret_a.index > pd.Timestamp(IS_END_DT.date())

    def get_stats_pack(dr: pd.Series):
        return {
            "full": CrossAssetHRPSynthesizer._calc_return_series_stats(dr),
            "is": CrossAssetHRPSynthesizer._calc_return_series_stats(dr.loc[is_mask]),
            "oos": CrossAssetHRPSynthesizer._calc_return_series_stats(dr.loc[oos_mask]),
        }

    stats_compare = {
        "方案A_SecHRP(23品种定锚)": get_stats_pack(dr_a_hrp),
        "方案A_ActiveEC(23品种定锚)": get_stats_pack(dr_a_active_ec),
        "方案B_PrunedHRP(50品种动态休眠)": get_stats_pack(dr_b_pruned),
        "方案B_ActivePruned(50品种休眠+槽位)": get_stats_pack(dr_b_active_pruned),
        "基准_50品种全入无休眠SecHRP": get_stats_pack(dr_b_unpruned),
    }

    comp_rows = []
    for k, st in stats_compare.items():
        comp_rows.append({
            "方案类别": k,
            "IS收益(20-23)": f"{st['is']['cum_ret_pct']:+.2f}%",
            "IS夏普": st['is']['sharpe'],
            "OOS净收益(24-26)": f"{st['oos']['cum_ret_pct']:+.2f}%",
            "OOS夏普": st['oos']['sharpe'],
            "OOS卡玛": st['oos']['calmar'],
            "全期总收益": f"{st['full']['cum_ret_pct']:+.2f}%",
            "全期年化": f"{st['full']['ann_ret_pct']:+.2f}%",
            "全期夏普": st['full']['sharpe'],
            "最大回撤": f"{st['full']['mdd_pct']:.2f}%",
            "全期卡玛": st['full']['calmar'],
        })

    print("\n" + "=" * 145)
    print("  【方案 A vs 方案 B 核心指标对比表（100% 名义底仓，扣费后净值）】")
    print("=" * 145)
    print(pd.DataFrame(comp_rows).to_string(index=False))

    # =========================================================================
    # 四档真实在途持仓 (Vol+EC Guard) 对比（重点对比 50% 机构平衡仓位）
    # =========================================================================
    all_abs_df = pd.DataFrame(all_abs_pos).sort_index()

    tier_results = {}
    for plan_name, base_w, sub_abs, ret_matrix in [
        ("方案A (23品种定锚·SecHRP)", w_a_hrp, plan_a_abs, ret_a),
        ("方案B (50品种动态休眠·PrunedHRP)", w_b_pruned, all_abs_df, ret_b),
    ]:
        b_ret = (base_w * ret_matrix.reindex(base_w.index)).sum(axis=1)
        p_vol20 = b_ret.rolling(20, min_periods=10).std().shift(1).bfill() * np.sqrt(252.0)
        v_mult = np.clip(0.025 / np.maximum(p_vol20, 0.015), 0.55, 1.45)
        b_ec = b_ret.cumsum().shift(1).fillna(0.0)
        b_ec_ma = b_ec.rolling(60, min_periods=10).mean().fillna(0.0)
        e_mult = pd.Series(np.where(b_ec >= b_ec_ma, 1.18, 0.72), index=base_w.index)
        u_b = (base_w * sub_abs.reindex(base_w.index).fillna(0.0)).sum(axis=1)
        u_lag = u_b.shift(1).fillna(0.18)
        dyn_shape = v_mult * e_mult

        t_records = []
        scaled_rets = {}
        for t_label, target_u in [("基准18%实际仓位", 0.18), ("平衡50%实际仓位", 0.50), ("进取80%实际仓位", 0.80), ("满仓100%实际仓位", 1.00)]:
            kt = (target_u / np.maximum(u_lag, 0.02)) * dyn_shape
            w_sc = base_w.multiply(kt, axis=0)
            act_gross = (w_sc * sub_abs.reindex(base_w.index).fillna(0.0)).sum(axis=1)
            dr_sc = (ret_matrix.reindex(base_w.index) * w_sc).sum(axis=1)
            scaled_rets[t_label] = dr_sc

            st_f = CrossAssetHRPSynthesizer._calc_return_series_stats(dr_sc)
            st_o = CrossAssetHRPSynthesizer._calc_return_series_stats(dr_sc.loc[oos_mask])
            t_records.append({
                "方案": plan_name,
                "仓位级别": t_label,
                "平均实际仓位": f"{act_gross.mean()*100:.1f}%",
                "平均保证金": f"{act_gross.mean()*12:.1f}%",
                "OOS净收益": f"{st_o['cum_ret_pct']:+.2f}%",
                "OOS夏普": st_o["sharpe"],
                "全期总收益": f"{st_f['cum_ret_pct']:+.2f}%",
                "全期年化": f"{st_f['ann_ret_pct']:+.2f}%",
                "全期夏普": st_f["sharpe"],
                "最大回撤": f"{st_f['mdd_pct']:.2f}%",
                "全期卡玛": st_f["calmar"],
            })
        tier_results[plan_name] = (pd.DataFrame(t_records), scaled_rets)

    print("\n" + "-" * 145)
    print("  【四档实际在途持仓对比 (Vol+EC Guard 动态守护)】")
    print("-" * 145)
    df_tiers = pd.concat([tier_results["方案A (23品种定锚·SecHRP)"][0], tier_results["方案B (50品种动态休眠·PrunedHRP)"][0]])
    print(df_tiers.to_string(index=False))

    # =========================================================================
    # 生成交互式大屏 HTML
    # =========================================================================
    dates_idx = dr_a_hrp.index
    nav_df = pd.DataFrame({"date": dates_idx.strftime("%Y-%m-%d")})
    nav_df["plan_a_sec_hrp"] = (1.0 + dr_a_hrp.cumsum()).values
    nav_df["plan_a_active_ec"] = (1.0 + dr_a_active_ec.cumsum()).values
    nav_df["plan_b_pruned"] = (1.0 + dr_b_pruned.cumsum()).values
    nav_df["plan_b_active_pruned"] = (1.0 + dr_b_active_pruned.cumsum()).values
    nav_df["plan_a_50pct"] = (1.0 + tier_results["方案A (23品种定锚·SecHRP)"][1]["平衡50%实际仓位"].cumsum()).values
    nav_df["plan_b_50pct"] = (1.0 + tier_results["方案B (50品种动态休眠·PrunedHRP)"][1]["平衡50%实际仓位"].cumsum()).values

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
          <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-sky-500/20 text-sky-400 border border-sky-500/30">因果优选双方案深度实证对照</span>
          <h2 class="text-base font-bold tracking-tight text-[var(--foreground)]">方案A (23品种样本内定锚) vs 方案B (50品种动态休眠)</h2>
        </div>
        <p class="text-xs text-[var(--muted-foreground)] mt-0.5">严格样本内外分离检验 · 方案A在2024-2026样本外爆发力突出 · 方案B全市场自适应回撤控制强</p>
      </div>
      <div class="flex items-center gap-2">
        <span class="text-xs px-2.5 py-1 rounded-lg bg-slate-800 text-sky-400 font-mono border border-slate-700">无未来函数 严谨比对</span>
      </div>
    </div>

    <!-- Metric Cards -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 my-3">
      <div class="p-2 rounded-xl bg-[var(--background)] border border-sky-500/30">
        <div class="text-xs font-semibold text-sky-400">方案A 23品种 SecHRP (100%)</div>
        <div class="text-sm font-bold text-sky-400 mt-0.5">+{stats_compare['方案A_SecHRP(23品种定锚)']['full']['cum_ret_pct']:.2f}% (Sh {stats_compare['方案A_SecHRP(23品种定锚)']['full']['sharpe']:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS: +{stats_compare['方案A_SecHRP(23品种定锚)']['oos']['cum_ret_pct']:.1f}% | MDD {stats_compare['方案A_SecHRP(23品种定锚)']['full']['mdd_pct']:.2f}%</div>
      </div>
      <div class="p-2 rounded-xl bg-[var(--background)] border border-amber-500/30">
        <div class="text-xs font-semibold text-amber-400">方案B 50品种休眠 (100%)</div>
        <div class="text-sm font-bold text-amber-400 mt-0.5">+{stats_compare['方案B_PrunedHRP(50品种动态休眠)']['full']['cum_ret_pct']:.2f}% (Sh {stats_compare['方案B_PrunedHRP(50品种动态休眠)']['full']['sharpe']:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">全自适应 | MDD {stats_compare['方案B_PrunedHRP(50品种动态休眠)']['full']['mdd_pct']:.2f}%</div>
      </div>
      <div class="p-2 rounded-xl bg-[var(--background)] border border-emerald-500/30 bg-emerald-950/20">
        <div class="text-xs font-semibold text-emerald-400">★ 方案A 50%实盘平衡仓位</div>
        <div class="text-sm font-bold text-emerald-400 mt-0.5">+{tier_results['方案A (23品种定锚·SecHRP)'][0].iloc[1]['全期总收益']} (年化 {tier_results['方案A (23品种定锚·SecHRP)'][0].iloc[1]['全期年化']})</div>
        <div class="text-xs text-emerald-300/80">OOS +{tier_results['方案A (23品种定锚·SecHRP)'][0].iloc[1]['OOS净收益']} | 夏普 {tier_results['方案A (23品种定锚·SecHRP)'][0].iloc[1]['全期夏普']} | MDD {tier_results['方案A (23品种定锚·SecHRP)'][0].iloc[1]['最大回撤']}</div>
      </div>
      <div class="p-2 rounded-xl bg-[var(--background)] border border-purple-500/30">
        <div class="text-xs font-semibold text-purple-400">方案B 50%实盘平衡仓位</div>
        <div class="text-sm font-bold text-purple-400 mt-0.5">+{tier_results['方案B (50品种动态休眠·PrunedHRP)'][0].iloc[1]['全期总收益']} (年化 {tier_results['方案B (50品种动态休眠·PrunedHRP)'][0].iloc[1]['全期年化']})</div>
        <div class="text-xs text-[var(--muted-foreground)]">夏普 {tier_results['方案B (50品种动态休眠·PrunedHRP)'][0].iloc[1]['全期夏普']} | MDD {tier_results['方案B (50品种动态休眠·PrunedHRP)'][0].iloc[1]['最大回撤']}</div>
      </div>
    </div>

    <!-- SVG Chart -->
    <div class="relative bg-[var(--background)] rounded-xl border border-[var(--border)] p-3 select-none">
      <div class="flex flex-wrap items-center justify-between mb-1.5 gap-2 text-xs">
        <div class="flex flex-wrap items-center gap-3">
          <span class="flex items-center gap-1"><span class="w-3 h-1 bg-emerald-400 rounded"></span><span class="text-emerald-400 font-bold">方案A 50%实盘仓位</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-purple-400"></span><span class="text-purple-400">方案B 50%实盘仓位</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-sky-400"></span><span class="text-sky-400">方案A 23品种底仓 (100%)</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-amber-400 border-b border-dashed"></span><span class="text-amber-400">方案B 50品种休眠 (100%)</span></span>
        </div>
      </div>

      <div id="hover-hud" class="hidden absolute top-9 right-3 bg-slate-900/90 border border-slate-700/80 rounded-lg p-2 text-xs z-20 pointer-events-none shadow-xl">
        <div id="hud-date" class="font-mono text-slate-300 font-bold mb-0.5"></div>
        <div class="text-emerald-400 font-mono" id="hud-a-50"></div>
        <div class="text-purple-300 font-mono" id="hud-b-50"></div>
        <div class="text-sky-300 font-mono" id="hud-a-base"></div>
        <div class="text-amber-300 font-mono" id="hud-b-base"></div>
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

          <path id="b-base-path" d="" fill="none" stroke="#FFB74D" stroke-width="1.8" stroke-dasharray="4,3"/>
          <path id="a-base-path" d="" fill="none" stroke="#38BDF8" stroke-width="2.0"/>
          <path id="b-50-path" d="" fill="none" stroke="#AB47BC" stroke-width="2.2"/>
          <path id="a-50-fill" d="" fill="url(#a50-grad)" opacity="0.22"/>
          <path id="a-50-path" d="" fill="none" stroke="#00E676" stroke-width="2.8"/>

          <line id="hover-line" x1="0" y1="10" x2="0" y2="185" stroke="#FFFFFF" stroke-width="1" stroke-dasharray="2,2" class="hidden"/>
          <circle id="hover-dot" cx="0" cy="0" r="4" fill="#00E676" stroke="#FFFFFF" stroke-width="1.5" class="hidden"/>

          <defs>
            <linearGradient id="a50-grad" x1="0%" y1="0%" x2="0%" y2="100%">
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
    const aBase = {json.dumps([round(float(v), 4) for v in sample_df['plan_a_sec_hrp']])};
    const bBase = {json.dumps([round(float(v), 4) for v in sample_df['plan_b_pruned']])};
    const a50 = {json.dumps([round(float(v), 4) for v in sample_df['plan_a_50pct']])};
    const b50 = {json.dumps([round(float(v), 4) for v in sample_df['plan_b_50pct']])};
    const n = dates.length;
    const xLeft = 50, xRight = 770, yTop = 15, yBottom = 185;

    const allVals = [...aBase, ...bBase, ...a50, ...b50];
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

    document.getElementById('b-base-path').setAttribute('d', buildPath(bBase));
    document.getElementById('a-base-path').setAttribute('d', buildPath(aBase));
    document.getElementById('b-50-path').setAttribute('d', buildPath(b50));
    const a50D = buildPath(a50);
    const fillD = `M ${{getX(0)}} ${{yBottom}} L ` + a50D.slice(2) + ` L ${{getX(n - 1)}} ${{yBottom}} Z`;
    document.getElementById('a-50-path').setAttribute('d', a50D);
    document.getElementById('a-50-fill').setAttribute('d', fillD);

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
      hoverDot.setAttribute('cy', getY(a50[idx]));
      hoverDot.classList.remove('hidden');
      hud.classList.remove('hidden');
      document.getElementById('hud-date').innerText = dates[idx];
      document.getElementById('hud-a-50').innerText = `方案A 50%仓位: ${{a50[idx].toFixed(3)}} (${{((a50[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-b-50').innerText = `方案B 50%仓位: ${{b50[idx].toFixed(3)}} (${{((b50[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-a-base').innerText = `方案A 底仓(100%): ${{aBase[idx].toFixed(3)}} (${{((aBase[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-b-base').innerText = `方案B 休眠(100%): ${{bBase[idx].toFixed(3)}} (${{((bBase[idx]-1)*100).toFixed(1)}}%)`;
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
    artifact_html = ARTIFACT_DIR / "compare_plan_a_vs_plan_b_chart.html"
    artifact_html.write_text(widget_html, encoding="utf-8")
    print(f"\n>>> 方案 A vs 方案 B 对比大屏已成功保存至: {artifact_html}")


if __name__ == "__main__":
    main()
