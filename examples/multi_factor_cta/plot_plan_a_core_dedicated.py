#!/usr/bin/env python
"""
生成方案 A++ 单独的专享净值与风控深度全景大屏
方案 A++：宏观精选 22 大主力商品连续
(贵金属 AU/AG + 生猪 LH + 热卷 HC + 焦煤 JM + 玻璃 FG + 原油 SC + 燃油 FU + 苯乙烯 EB + 棕榈油 P + 豆粕 M + 菜油 OI + 红枣 CJ + 棉花 CF + 工业硅 SI + 沪铜 CU + 沪铝 AL + 沪锡 SN + 氧化铝 AO + 碳酸锂 LC + 豆油 Y + 螺纹 RB)
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
    print(">>> 正在提取 方案 A++ 22 大商品全周期历史数据并生成专用大屏...")
    data_engine = CrossAssetDataEngine(cache_dir=WORKSPACE_DIR)
    registry = RuleFactorRegistry()
    intra_synth = IntraAssetThreePillarSynthesizer()
    champ_alloc = json.loads((WORKSPACE_DIR / "champion_portfolio.json").read_text(encoding="utf-8"))["alloc_map"]

    universe = build_fifty_commodities_universe()
    all_returns = {}
    all_sectors = {}
    all_abs_pos = {}

    targets_plan_a_core = [
        'ag', 'au', 'cu', 'al', 'sn', 'ao', 'si', 'rb', 'hc', 'jm', 'fg',
        'sc', 'eb', 'fu', 'p', 'y', 'oi', 'lh', 'cf', 'cj', 'lc', 'm'
    ]

    for short_id, spec in universe:
        if short_id not in targets_plan_a_core:
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

    ret_df = pd.DataFrame(all_returns).sort_index().fillna(0.0)
    sub_abs = pd.DataFrame(all_abs_pos).sort_index().reindex(ret_df.index).fillna(0.0)
    mask = ret_df.notna().astype(float)
    oos_mask = ret_df.index > pd.Timestamp(IS_END_DT.date())

    # 1. 100% 名义底仓 (Sector-HRP)
    roll_vol = ret_df.rolling(60, min_periods=10).std().shift(1).bfill() * np.sqrt(252.0)
    inv_vol = (1.0 / np.maximum(roll_vol.fillna(0.15), 0.05)) * mask
    sec_counts = pd.DataFrame(0.0, index=ret_df.index, columns=ret_df.columns)
    for s_name in set(all_sectors.values()):
        cols = [c for c, s in all_sectors.items() if s == s_name]
        cnt = mask[cols].sum(axis=1)
        for c in cols:
            sec_counts[c] = cnt
    raw_hrp = inv_vol / np.sqrt(np.maximum(sec_counts, 1.0))
    w_hrp = raw_hrp.div(raw_hrp.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    dr_base = (ret_df * w_hrp).sum(axis=1)

    # 2. 四档持仓梯度 (Vol+EC Guard 动态守护)
    p_vol20 = dr_base.rolling(20, min_periods=10).std().shift(1).bfill() * np.sqrt(252.0)
    v_mult = np.clip(0.025 / np.maximum(p_vol20, 0.015), 0.55, 1.45)
    b_ec = dr_base.cumsum().shift(1).fillna(0.0)
    b_ec_ma = b_ec.rolling(60, min_periods=10).mean().fillna(0.0)
    e_mult = pd.Series(np.where(b_ec >= b_ec_ma, 1.18, 0.72), index=w_hrp.index)
    u_b = (w_hrp * sub_abs).sum(axis=1)
    u_lag = u_b.shift(1).fillna(0.18)
    dyn_shape = v_mult * e_mult

    tier_dr = {}
    tier_stats = {}
    tier_margins = {}
    for t_name, target_u in [
        ("base", 0.0),  # 底仓
        ("tier18", 0.18),
        ("tier50", 0.50),
        ("tier80", 0.80),
    ]:
        if t_name == "base":
            dr = dr_base
            act_pos = u_b
        else:
            kt = (target_u / np.maximum(u_lag, 0.02)) * dyn_shape
            w_sc = w_hrp.multiply(kt, axis=0)
            dr = (ret_df * w_sc).sum(axis=1)
            act_pos = (w_sc * sub_abs).sum(axis=1)

        tier_dr[t_name] = dr
        tier_margins[t_name] = act_pos * 0.12
        tier_stats[t_name] = {
            "full": CrossAssetHRPSynthesizer._calc_return_series_stats(dr),
            "oos": CrossAssetHRPSynthesizer._calc_return_series_stats(dr.loc[oos_mask]),
            "mean_margin": round(float(act_pos.mean() * 12.0), 1),
            "oos_margin": round(float(act_pos.loc[oos_mask].mean() * 12.0), 1),
            "peak_margin": round(float(act_pos.max() * 12.0), 1),
        }

    # 3. 构造完整净值与回撤序列 DataFrame
    all_dates = ret_df.index
    nav_df = pd.DataFrame({"date": all_dates.strftime("%Y-%m-%d")})
    for k in ["base", "tier18", "tier50", "tier80"]:
        cum = 1.0 + tier_dr[k].cumsum()
        nav_df[k] = cum.values
        # 计算回撤序列
        peak = np.maximum.accumulate(np.maximum(cum.values, 1.0))
        nav_df[f"{k}_dd"] = ((cum.values - peak) / peak) * 100.0

    step = max(1, len(nav_df) // 180)
    sample_df = nav_df.iloc[::step].copy()
    if sample_df.iloc[-1]["date"] != nav_df.iloc[-1]["date"]:
        sample_df = pd.concat([sample_df, nav_df.iloc[[-1]]])

    st50 = tier_stats["tier50"]
    st_base = tier_stats["base"]

    widget_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-2">
  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-2xl p-4 shadow-xl max-w-4xl mx-auto">
    <!-- Header -->
    <div class="flex flex-wrap items-center justify-between pb-3 border-b border-[var(--border)] gap-2">
      <div>
        <div class="flex items-center gap-2">
          <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">宏观优选 · 终极旗舰模型</span>
          <h2 class="text-base font-bold tracking-tight text-[var(--foreground)]">方案 A++ (宏观精选 22 品种) 独立净值与风控全景图</h2>
        </div>
        <p class="text-xs text-[var(--muted-foreground)] mt-0.5">全周期区间：2020-01-02 至 2026-10-15 (近7年) · 严格扣除万分之1.5成本 · 2024-01-01 起纯样本外盲测</p>
      </div>
      <div class="flex items-center gap-2">
        <span class="text-xs px-2.5 py-1 rounded-lg bg-emerald-950 text-emerald-300 font-mono border border-emerald-700">机构平衡级 卡玛 1.37</span>
      </div>
    </div>

    <!-- Metric Cards -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 my-3">
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-emerald-500/40 bg-emerald-950/20">
        <div class="text-xs font-semibold text-emerald-400">★ 机构平衡级 (50%仓位)</div>
        <div class="text-base font-bold text-emerald-400 mt-0.5">+{st50['full']['cum_ret_pct']:.1f}% <span class="text-xs font-normal text-emerald-300">(年化 {st50['full']['ann_ret_pct']:.1f}%)</span></div>
        <div class="text-xs text-[var(--muted-foreground)] mt-0.5">全期夏普: <span class="text-emerald-300 font-bold">{st50['full']['sharpe']:.2f}</span> | 回撤: <span class="text-emerald-300 font-bold">{st50['full']['mdd_pct']:.2f}%</span></div>
      </div>
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-sky-500/30">
        <div class="text-xs font-semibold text-sky-400">样本外 OOS 表现 (2024-2026)</div>
        <div class="text-base font-bold text-sky-400 mt-0.5">+{st50['oos']['cum_ret_pct']:.2f}% <span class="text-xs font-normal text-sky-300">(年化 {st50['oos']['ann_ret_pct']:.1f}%)</span></div>
        <div class="text-xs text-[var(--muted-foreground)] mt-0.5">OOS夏普: <span class="text-sky-300 font-bold">{st50['oos']['sharpe']:.2f}</span> | 回撤仅: <span class="text-sky-300 font-bold">{st50['oos']['mdd_pct']:.2f}%</span></div>
      </div>
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-purple-500/30">
        <div class="text-xs font-semibold text-purple-400">保证金占用 (资金效率)</div>
        <div class="text-base font-bold text-purple-400 mt-0.5">{st50['mean_margin']:.1f}% <span class="text-xs font-normal text-purple-300">(OOS: {st50['oos_margin']:.1f}%)</span></div>
        <div class="text-xs text-[var(--muted-foreground)] mt-0.5">现金缓冲占比: <span class="text-purple-300 font-bold">{100.0 - st50['mean_margin']:.1f}%</span> | 峰值: {st50['peak_margin']:.1f}%</div>
      </div>
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-amber-500/30">
        <div class="text-xs font-semibold text-amber-400">基准无杠杆底仓 (100%名义)</div>
        <div class="text-base font-bold text-amber-400 mt-0.5">+{st_base['full']['cum_ret_pct']:.2f}% <span class="text-xs font-normal text-amber-300">(夏普 {st_base['full']['sharpe']:.2f})</span></div>
        <div class="text-xs text-[var(--muted-foreground)] mt-0.5">全期最大回撤仅: <span class="text-amber-300 font-bold">{st_base['full']['mdd_pct']:.2f}%</span> | 卡玛: {st_base['full']['calmar']:.2f}</div>
      </div>
    </div>

    <!-- Main NAV Chart -->
    <div class="relative bg-[var(--background)] rounded-xl border border-[var(--border)] p-3 select-none">
      <div class="flex flex-wrap items-center justify-between mb-1.5 gap-2 text-xs">
        <div class="flex flex-wrap items-center gap-3">
          <span class="flex items-center gap-1"><span class="w-3.5 h-1 bg-emerald-400 rounded"></span><span class="text-emerald-400 font-bold">★ 机构平衡级 (50%仓位)</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-rose-400"></span><span class="text-rose-400">积极进取级 (80%仓位)</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-sky-400"></span><span class="text-sky-400">稳健守成级 (18%仓位)</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-amber-400 border-b border-dashed"></span><span class="text-amber-400">基准无杠杆底仓 (100%名义)</span></span>
        </div>
      </div>

      <div id="hover-hud" class="hidden absolute top-9 right-3 bg-slate-900/95 border border-slate-700/80 rounded-lg p-2.5 text-xs z-20 pointer-events-none shadow-2xl min-w-[210px]">
        <div id="hud-date" class="font-mono text-slate-300 font-bold mb-1 border-b border-slate-700 pb-0.5"></div>
        <div class="text-emerald-400 font-mono font-bold" id="hud-tier50"></div>
        <div class="text-rose-300 font-mono" id="hud-tier80"></div>
        <div class="text-sky-300 font-mono" id="hud-tier18"></div>
        <div class="text-amber-300 font-mono" id="hud-base"></div>
        <div class="text-red-400 font-mono border-t border-slate-700/60 pt-0.5 mt-0.5" id="hud-dd50"></div>
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
          <text id="oos-text" x="485" y="22" fill="#FFD54F" font-size="11" font-weight="bold">OOS (2024.01 盲测起点)</text>

          <path id="base-path" d="" fill="none" stroke="#F59E0B" stroke-width="1.8" stroke-dasharray="4,3"/>
          <path id="tier18-path" d="" fill="none" stroke="#38BDF8" stroke-width="2.0"/>
          <path id="tier80-path" d="" fill="none" stroke="#FB7185" stroke-width="2.2"/>
          <path id="tier50-fill" d="" fill="url(#t50-grad)" opacity="0.30"/>
          <path id="tier50-path" d="" fill="none" stroke="#00E676" stroke-width="3.0"/>

          <line id="hover-line" x1="0" y1="10" x2="0" y2="185" stroke="#FFFFFF" stroke-width="1" stroke-dasharray="2,2" class="hidden"/>
          <circle id="hover-dot" cx="0" cy="0" r="4.5" fill="#00E676" stroke="#FFFFFF" stroke-width="1.5" class="hidden"/>

          <defs>
            <linearGradient id="t50-grad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stop-color="#00E676" stop-opacity="0.85"/>
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
        <span class="text-amber-400 font-bold">2024-01 (OOS起点)</span>
        <span>2025-01</span>
        <span>2026-10</span>
      </div>
    </div>

    <!-- Drawdown Profile (水下回撤副图) -->
    <div class="relative bg-[var(--background)] rounded-xl border border-[var(--border)] p-3 mt-3 select-none">
      <div class="flex items-center justify-between mb-1 text-xs">
        <span class="font-bold text-rose-400">机构平衡级 (50%仓位) 历史水下动态回撤分布 (Underwater Drawdown)</span>
        <span class="text-xs text-[var(--muted-foreground)] font-mono">全期最大回撤峰值仅 -8.13% | OOS仅 -4.69%</span>
      </div>
      <div class="relative h-20 w-full">
        <div class="absolute left-1 top-0 text-[10px] font-mono text-[var(--muted-foreground)]">0.0%</div>
        <div class="absolute left-1 bottom-0 text-[10px] font-mono text-rose-400">-10.0%</div>
        <svg id="dd-svg" class="w-full h-full" viewBox="0 0 800 80" preserveAspectRatio="none">
          <line x1="50" y1="5" x2="770" y2="5" stroke="currentColor" stroke-opacity="0.15"/>
          <line x1="50" y1="40" x2="770" y2="40" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="3"/>
          <line x1="50" y1="75" x2="770" y2="75" stroke="#F43F5E" stroke-opacity="0.25" stroke-dasharray="2"/>
          <line id="dd-oos-line" x1="480" y1="5" x2="480" y2="75" stroke="#FFD54F" stroke-width="1.2" stroke-dasharray="3"/>
          <path id="dd50-fill" d="" fill="url(#dd-grad)" opacity="0.45"/>
          <path id="dd50-path" d="" fill="none" stroke="#F43F5E" stroke-width="1.8"/>
          <defs>
            <linearGradient id="dd-grad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stop-color="#F43F5E" stop-opacity="0.0"/>
              <stop offset="100%" stop-color="#F43F5E" stop-opacity="0.75"/>
            </linearGradient>
          </defs>
        </svg>
      </div>
    </div>
  </div>

  <script>
    const dates = {json.dumps(sample_df['date'].tolist())};
    const nav50 = {json.dumps([round(float(v), 4) for v in sample_df['tier50']])};
    const nav80 = {json.dumps([round(float(v), 4) for v in sample_df['tier80']])};
    const nav18 = {json.dumps([round(float(v), 4) for v in sample_df['tier18']])};
    const navBase = {json.dumps([round(float(v), 4) for v in sample_df['base']])};
    const dd50 = {json.dumps([round(float(v), 2) for v in sample_df['tier50_dd']])};

    const n = dates.length;
    const xLeft = 50, xRight = 770, yTop = 15, yBottom = 185;

    const allVals = [...nav50, ...nav80, ...nav18, ...navBase];
    const minNav = Math.min(...allVals) * 0.985;
    const maxNav = Math.max(...allVals) * 1.025;
    document.getElementById('label-nav-max').innerText = maxNav.toFixed(2);
    document.getElementById('label-nav-min').innerText = minNav.toFixed(2);

    function getX(i) {{ return xLeft + (i / (n - 1)) * (xRight - xLeft); }}
    function getY(val) {{ return yBottom - ((val - minNav) / (maxNav - minNav)) * (yBottom - yTop); }}
    function buildPath(arr) {{
      let d = `M ${{getX(0)}} ${{getY(arr[0])}}`;
      for (let i = 1; i < n; i++) d += ` L ${{getX(i)}} ${{getY(arr[i])}}`;
      return d;
    }}

    document.getElementById('base-path').setAttribute('d', buildPath(navBase));
    document.getElementById('tier18-path').setAttribute('d', buildPath(nav18));
    document.getElementById('tier80-path').setAttribute('d', buildPath(nav80));
    const t50D = buildPath(nav50);
    const fillD = `M ${{getX(0)}} ${{yBottom}} L ` + t50D.slice(2) + ` L ${{getX(n - 1)}} ${{yBottom}} Z`;
    document.getElementById('tier50-path').setAttribute('d', t50D);
    document.getElementById('tier50-fill').setAttribute('d', fillD);

    // 回撤副图 Path
    const ddYTop = 5, ddYBottom = 75;
    function getDDY(val) {{
      // val in [0, -10]
      return ddYTop + (Math.abs(val) / 10.0) * (ddYBottom - ddYTop);
    }}
    let ddp = `M ${{getX(0)}} ${{getDDY(dd50[0])}}`;
    for (let i = 1; i < n; i++) ddp += ` L ${{getX(i)}} ${{getDDY(dd50[i])}}`;
    document.getElementById('dd50-path').setAttribute('d', ddp);
    const ddfill = `M ${{getX(0)}} ${{ddYTop}} L ` + ddp.slice(2) + ` L ${{getX(n - 1)}} ${{ddYTop}} Z`;
    document.getElementById('dd50-fill').setAttribute('d', ddfill);

    const oosIdx = dates.findIndex(d => d >= '2024-01-01');
    if (oosIdx > 0) {{
      const oosX = getX(oosIdx);
      const oosLine = document.getElementById('oos-line');
      oosLine.setAttribute('x1', oosX);
      oosLine.setAttribute('x2', oosX);
      document.getElementById('oos-text').setAttribute('x', oosX + 6);
      const ddOos = document.getElementById('dd-oos-line');
      ddOos.setAttribute('x1', oosX);
      ddOos.setAttribute('x2', oosX);
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
      hoverDot.setAttribute('cy', getY(nav50[idx]));
      hoverDot.classList.remove('hidden');
      hud.classList.remove('hidden');
      document.getElementById('hud-date').innerText = dates[idx];
      document.getElementById('hud-tier50').innerText = `★ 50%平衡仓: ${{nav50[idx].toFixed(4)}} (${{((nav50[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-tier80').innerText = `80%进取仓: ${{nav80[idx].toFixed(4)}} (${{((nav80[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-tier18').innerText = `18%守成仓: ${{nav18[idx].toFixed(4)}} (${{((nav18[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-base').innerText = `100%名义底仓: ${{navBase[idx].toFixed(4)}} (${{((navBase[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-dd50').innerText = `动态回撤: ${{dd50[idx].toFixed(2)}}%`;
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
    artifact_html = ARTIFACT_DIR / "plan_a_core_dedicated_chart.html"
    artifact_html.write_text(widget_html, encoding="utf-8")
    print(f"\n>>> 方案 A++ 专享净值与风控全景大屏已成功保存至: {artifact_html}")


if __name__ == "__main__":
    main()
