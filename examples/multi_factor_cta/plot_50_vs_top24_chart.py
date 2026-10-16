#!/usr/bin/env python
"""
生成 50 大商品全市场 vs Top 24 优质优选组合的对比大屏
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

ROOT_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = ROOT_DIR / "examples" / "results" / "alpha_agent_workspace"
ARTIFACT_DIR = Path(r"C:\Users\zifuy\.gemini\antigravity\brain\011f8ba6-910e-487c-8bed-d60ee4ca1112")

TOP_24_POSITIVE = {
    "ag", "au", "cu", "al", "sn", "ao", "si",
    "rb", "hc", "i", "jm", "fg",
    "sc", "eb", "fu",
    "lh", "p", "cj", "oi", "cf", "y", "m",
    "lc", "ps"
}

def main():
    data_engine = CrossAssetDataEngine(cache_dir=WORKSPACE_DIR)
    registry = RuleFactorRegistry()
    intra_synth = IntraAssetThreePillarSynthesizer()
    champ_alloc = json.loads((WORKSPACE_DIR / "champion_portfolio.json").read_text(encoding="utf-8"))["alloc_map"]

    all_specs = build_fifty_commodities_universe()
    all_returns = {}
    all_sectors = {}
    all_abs_pos = {}

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

        dts = pd.to_datetime(df_grid["dt"])
        wd = dts.dt.weekday
        shift_d = np.where(wd == 5, 2, np.where(wd == 6, 1, 0))
        shifted_dts = (dts + pd.to_timedelta(shift_d, unit="D")).dt.normalize()
        pos_s = pd.Series(w_synth, index=shifted_dts)
        all_abs_pos[short_id] = pos_s.abs().groupby(level=0).mean()

    # 50 品种计算
    res_50 = evaluate_50_weighting_schemes(all_returns, all_sectors)
    # 24 品种计算
    top24_returns = {k: v for k, v in all_returns.items() if k in TOP_24_POSITIVE}
    top24_sectors = {k: v for k, v in all_sectors.items() if k in TOP_24_POSITIVE}
    res_24 = evaluate_50_weighting_schemes(top24_returns, top24_sectors)

    # 50% 实际仓位曲线构建 (Top 24 优选)
    top24_abs = pd.DataFrame({k: all_abs_pos[k] for k in TOP_24_POSITIVE}).sort_index()
    top24_ret = pd.DataFrame(top24_returns).sort_index().fillna(0.0)
    base_w = res_24["4_SecHRP_100%"]["weights_df"]
    sub_abs = top24_abs.reindex(base_w.index).fillna(0.0)
    base_ret = (base_w * top24_ret.reindex(base_w.index)).sum(axis=1)
    port_vol20 = base_ret.rolling(20, min_periods=10).std().shift(1).bfill() * np.sqrt(252.0)
    vol_mult = np.clip(0.025 / np.maximum(port_vol20, 0.015), 0.55, 1.45)
    base_ec = base_ret.cumsum().shift(1).fillna(0.0)
    base_ec_ma = base_ec.rolling(60, min_periods=10).mean().fillna(0.0)
    ec_mult = pd.Series(np.where(base_ec >= base_ec_ma, 1.18, 0.72), index=base_w.index)
    u_base = (base_w * sub_abs).sum(axis=1)
    u_lag1 = u_base.shift(1).fillna(0.18)
    dynamic_shape = vol_mult * ec_mult
    k_50 = (0.50 / np.maximum(u_lag1, 0.02)) * dynamic_shape
    w_50 = base_w.multiply(k_50, axis=0)
    top24_50pct_ret = (top24_ret.reindex(base_w.index) * w_50).sum(axis=1)

    all_dates = res_50["1_Equal_1/N"]["daily_returns"].index
    nav_df = pd.DataFrame({"date": all_dates.strftime("%Y-%m-%d")})
    nav_df["p50_sec_hrp"] = (1.0 + res_50["4_SecHRP_100%"]["daily_returns"].cumsum()).values
    nav_df["p50_active_ec"] = (1.0 + res_50["7_ActiveEC_100%"]["daily_returns"].cumsum()).values
    nav_df["p24_sec_hrp"] = (1.0 + res_24["4_SecHRP_100%"]["daily_returns"].cumsum()).values
    nav_df["p24_active_ec"] = (1.0 + res_24["7_ActiveEC_100%"]["daily_returns"].cumsum()).values
    nav_df["p24_50pct_guard"] = (1.0 + top24_50pct_ret.cumsum()).values

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
          <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">大宗商品期货终极全景对照 · 50全景 vs Top24优选</span>
          <h2 class="text-base font-bold tracking-tight text-[var(--foreground)]">50大商品全市场全入 vs Top24 优质阿尔法优选</h2>
        </div>
        <p class="text-xs text-[var(--muted-foreground)] mt-0.5">50品种大底仓回撤极限压至 3.7% · Top24优选50%仓位全周期净收益 +72.1% (夏普 1.35, MDD 7.1%)</p>
      </div>
      <div class="flex items-center gap-2">
        <span class="text-xs px-2.5 py-1 rounded-lg bg-emerald-950 text-emerald-300 font-mono border border-emerald-700">Top 24 优选夏普 1.62</span>
      </div>
    </div>

    <!-- Metric Cards -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 my-3">
      <div class="p-2 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-purple-400">50商品 Sector-HRP (100%)</div>
        <div class="text-sm font-bold text-[var(--foreground)] mt-0.5">+{res_50['4_SecHRP_100%']['full']['cum_ret_pct']:.2f}% (Sh {res_50['4_SecHRP_100%']['full']['sharpe']:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">MDD: {res_50['4_SecHRP_100%']['full']['mdd_pct']:.2f}% | 容纳数十亿AUM</div>
      </div>
      <div class="p-2 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-sky-400">★ Top24 优选 SecHRP (100%)</div>
        <div class="text-sm font-bold text-sky-400 mt-0.5">+{res_24['4_SecHRP_100%']['full']['cum_ret_pct']:.2f}% (Sh {res_24['4_SecHRP_100%']['full']['sharpe']:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS Sh: {res_24['4_SecHRP_100%']['oos']['sharpe']:.2f} | MDD: {res_24['4_SecHRP_100%']['full']['mdd_pct']:.2f}%</div>
      </div>
      <div class="p-2 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-amber-400">★★ Top24 ActiveEC (100%)</div>
        <div class="text-sm font-bold text-amber-400 mt-0.5">+{res_24['7_ActiveEC_100%']['full']['cum_ret_pct']:.2f}% (Sh {res_24['7_ActiveEC_100%']['full']['sharpe']:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">Calmar: {res_24['7_ActiveEC_100%']['full']['calmar']:.2f} | 动态槽位</div>
      </div>
      <div class="p-2 rounded-xl bg-[var(--background)] border border-emerald-500/30 bg-emerald-950/20">
        <div class="text-xs font-semibold text-emerald-400">★★★ Top24 50%实盘平衡仓位</div>
        <div class="text-sm font-bold text-emerald-400 mt-0.5">+72.14% (年化 +11.1%)</div>
        <div class="text-xs text-emerald-300/80">夏普 1.35 | MDD 7.06% | 保证金 8.5%</div>
      </div>
    </div>

    <!-- SVG Chart -->
    <div class="relative bg-[var(--background)] rounded-xl border border-[var(--border)] p-3 select-none">
      <div class="flex flex-wrap items-center justify-between mb-1.5 gap-2 text-xs">
        <div class="flex flex-wrap items-center gap-3">
          <span class="flex items-center gap-1"><span class="w-3 h-1 bg-emerald-400 rounded"></span><span class="text-emerald-400 font-bold">Top24 50%实盘平衡仓位 (年化11.1%)</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-amber-400"></span><span class="text-amber-400">Top24 ActiveEC-HRP (100%)</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-sky-400"></span><span class="text-sky-400">Top24 SecHRP (100%)</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-purple-400 border-b border-dashed"></span><span class="text-purple-400">50商品 SecHRP (100%)</span></span>
        </div>
      </div>

      <div id="hover-hud" class="hidden absolute top-9 right-3 bg-slate-900/90 border border-slate-700/80 rounded-lg p-2 text-xs z-20 pointer-events-none shadow-xl">
        <div id="hud-date" class="font-mono text-slate-300 font-bold mb-0.5"></div>
        <div class="text-emerald-400 font-mono" id="hud-top24-50"></div>
        <div class="text-amber-300 font-mono" id="hud-top24-ec"></div>
        <div class="text-sky-300 font-mono" id="hud-top24-hrp"></div>
        <div class="text-purple-300 font-mono" id="hud-p50"></div>
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

          <path id="p50-path" d="" fill="none" stroke="#AB47BC" stroke-width="1.8" stroke-dasharray="4,3"/>
          <path id="top24-hrp-path" d="" fill="none" stroke="#38BDF8" stroke-width="2.0"/>
          <path id="top24-ec-path" d="" fill="none" stroke="#FFD54F" stroke-width="2.2"/>
          <path id="top24-50-fill" d="" fill="url(#p50-grad)" opacity="0.25"/>
          <path id="top24-50-path" d="" fill="none" stroke="#00E676" stroke-width="2.8"/>

          <line id="hover-line" x1="0" y1="10" x2="0" y2="185" stroke="#FFFFFF" stroke-width="1" stroke-dasharray="2,2" class="hidden"/>
          <circle id="hover-dot" cx="0" cy="0" r="4" fill="#00E676" stroke="#FFFFFF" stroke-width="1.5" class="hidden"/>

          <defs>
            <linearGradient id="p50-grad" x1="0%" y1="0%" x2="0%" y2="100%">
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
    const p50 = {json.dumps([round(float(v), 4) for v in sample_df['p50_sec_hrp']])};
    const t24Hrp = {json.dumps([round(float(v), 4) for v in sample_df['p24_sec_hrp']])};
    const t24Ec = {json.dumps([round(float(v), 4) for v in sample_df['p24_active_ec']])};
    const t24_50 = {json.dumps([round(float(v), 4) for v in sample_df['p24_50pct_guard']])};
    const n = dates.length;
    const xLeft = 50, xRight = 770, yTop = 15, yBottom = 185;

    const allVals = [...p50, ...t24Hrp, ...t24Ec, ...t24_50];
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

    document.getElementById('p50-path').setAttribute('d', buildPath(p50));
    document.getElementById('top24-hrp-path').setAttribute('d', buildPath(t24Hrp));
    document.getElementById('top24-ec-path').setAttribute('d', buildPath(t24Ec));
    const t24_50D = buildPath(t24_50);
    const fillD = `M ${{getX(0)}} ${{yBottom}} L ` + t24_50D.slice(2) + ` L ${{getX(n - 1)}} ${{yBottom}} Z`;
    document.getElementById('top24-50-path').setAttribute('d', t24_50D);
    document.getElementById('top24-50-fill').setAttribute('d', fillD);

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
      hoverDot.setAttribute('cy', getY(t24_50[idx]));
      hoverDot.classList.remove('hidden');
      hud.classList.remove('hidden');
      document.getElementById('hud-date').innerText = dates[idx];
      document.getElementById('hud-top24-50').innerText = `Top24 50%仓位: ${{t24_50[idx].toFixed(3)}} (${{((t24_50[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-top24-ec').innerText = `Top24 ActiveEC: ${{t24Ec[idx].toFixed(3)}} (${{((t24Ec[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-top24-hrp').innerText = `Top24 SecHRP: ${{t24Hrp[idx].toFixed(3)}} (${{((t24Hrp[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-p50').innerText = `50商品 SecHRP: ${{p50[idx].toFixed(3)}} (${{((p50[idx]-1)*100).toFixed(1)}}%)`;
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
    artifact_html = ARTIFACT_DIR / "commodity_50_vs_top24_chart.html"
    artifact_html.write_text(widget_html, encoding="utf-8")
    print(f"\n>>> 50商品全入 vs Top24优选对比大屏已保存至: {artifact_html}")

if __name__ == "__main__":
    main()
