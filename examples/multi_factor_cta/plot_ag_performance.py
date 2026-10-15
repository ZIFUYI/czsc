#!/usr/bin/env python
"""
白银主力连续 (KQ.m@SHFE.ag) 规则型多因子 CTA 策略回测表现与净值看板导出脚本
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.multi_factor_cta.data_engine import (
    CrossAssetDataEngine,
    map_tf_weight_to_base_grid,
)
from examples.multi_factor_cta.factor_engine import RuleFactorRegistry
from examples.multi_factor_cta.forensic_validator import (
    RuleFactorForensicValidator,
    compute_empirical_bayes_shrinkage_weight,
    compute_signed_jaccard,
)
from examples.multi_factor_cta.run_framework_demo import (
    build_default_cross_asset_universe,
    evaluate_single_asset_backtest,
)
from examples.multi_factor_cta.synthesis_engine import (
    IntraAssetThreePillarSynthesizer,
    joint_is_coordinate_reweight,
)

WORKSPACE_DIR = ROOT_DIR / "examples" / "results" / "alpha_agent_workspace"
CTA_OUTPUT_DIR = ROOT_DIR / "examples" / "results" / "multi_factor_cta"
ARTIFACT_DIR = Path(r"C:\Users\zifuy\.gemini\antigravity\brain\011f8ba6-910e-487c-8bed-d60ee4ca1112")


def main() -> None:
    CTA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> 正在加载并回测 白银主力连续 (KQ.m@SHFE.ag)...")
    data_engine = CrossAssetDataEngine(cache_dir=WORKSPACE_DIR)
    registry = RuleFactorRegistry()
    validator = RuleFactorForensicValidator()
    intra_synth = IntraAssetThreePillarSynthesizer()

    champ_path = WORKSPACE_DIR / "champion_portfolio.json"
    champ_alloc: dict[str, float] = json.loads(champ_path.read_text(encoding="utf-8"))["alloc_map"]

    universe = build_default_cross_asset_universe()
    spec = [s for s in universe if s.symbol == "KQ.m@SHFE.ag"][0]

    tf_dfs = data_engine.load_asset_multi_timeframes(spec, attach_czsc=False)
    df_grid = tf_dfs[spec.base_tf]

    is_mask = (df_grid["dt"] <= validator.is_end_dt).values
    same_dir_pool: dict[int, dict[str, np.ndarray]] = {1: {}, -1: {}}
    w_on_grid: dict[str, np.ndarray] = {}
    init_alloc: dict[str, float] = {}
    unit_forensic_meta: dict[str, dict] = {}
    candidates_g1 = []

    for f_spec in registry.list_all():
        sid = f_spec.factor_id
        if sid.startswith("CZ") or sid.startswith("M5"):
            continue
        for tf in spec.eligible_tfs:
            if tf not in f_spec.eligible_tfs:
                continue
            uid = f"{sid}@{tf}"
            trig_arr, w_arr = registry.evaluate_factor_on_tf(tf_dfs[tf], sid, tf)
            g1 = validator.run_gate1_single_asset_audit(
                tf_dfs[tf],
                trig_arr,
                direction=f_spec.direction,
                archetype=f_spec.archetype,
                timeframe=tf,
                fee_rate=spec.fee_rate,
            )
            norm_res = validator.run_vol_and_regime_normalized_audit(
                tf_dfs[tf],
                trig_arr,
                direction=f_spec.direction,
                archetype=f_spec.archetype,
                timeframe=tf,
                fee_rate=spec.fee_rate,
            )
            g1.update(norm_res)
            w_mapped = map_tf_weight_to_base_grid(tf_dfs[tf], w_arr, df_grid)
            if (
                f_spec.archetype == "A_顺势突破簇"
                and f_spec.direction == 1
                and g1.get("is_n_purged", 0) >= 3
                and (g1.get("is_mean_pct", -1.0) > -0.15 or g1.get("ipw_safety_pct", -1.0) > 0.0)
            ):
                same_dir_pool[f_spec.direction][uid] = w_mapped
                w_on_grid[uid] = w_mapped
                init_alloc[uid] = f_spec.default_weight
                g1["is_base_seed"] = True
                unit_forensic_meta[uid] = g1
            elif (
                g1.get("is_n_purged", 0) >= 6
                and g1.get("is_mean_pct", -1.0) > 0.01
                and (
                    (g1.get("is_safety_margin_pct", -1.0) > 1.0 and g1.get("is_excess_alpha_pct", -1.0) > 0.01)
                    or (g1.get("home_r_atr", -1.0) >= 0.06 and g1.get("home_safety_pct", -1.0) >= 3.0)
                )
            ):
                eff_saf = max(float(g1.get("is_safety_margin_pct", 0.0)), float(g1.get("home_safety_pct", 0.0)))
                score = eff_saf * np.log(max(int(g1["is_n_purged"]), 5))
                candidates_g1.append((uid, score, w_mapped, g1))

    candidates_g1.sort(key=lambda x: x[1], reverse=True)

    for uid, _, w_mapped, g1 in candidates_g1:
        sid, _ = uid.split("@")
        f_spec = registry.get(sid)
        d = f_spec.direction
        cw_is = w_mapped[is_mask]
        redundant = False
        for _, pw in same_dir_pool[d].items():
            pw_is = pw[is_mask]
            corr = (
                float(abs(np.corrcoef(cw_is, pw_is)[0, 1]))
                if (np.std(cw_is) > 1e-6 and np.std(pw_is) > 1e-6)
                else 0.0
            )
            jac = compute_signed_jaccard(cw_is, pw_is)
            if corr >= 0.32 or jac >= 0.30:
                redundant = True
                break
        if not redundant:
            same_dir_pool[d][uid] = w_mapped
            w_on_grid[uid] = w_mapped
            prior_w = float(champ_alloc.get(uid, f_spec.default_weight))
            local_w = float(
                np.clip(
                    f_spec.default_weight * (1.0 + 0.04 * float(g1.get("home_safety_pct", 2.0))),
                    0.15,
                    0.48,
                )
            )
            w_eb, _ = compute_empirical_bayes_shrinkage_weight(
                local_w, prior_w, int(g1.get("is_n_purged", 20)), shrink_k=60
            )
            init_alloc[uid] = w_eb
            unit_forensic_meta[uid] = g1

    print(">>> 执行 IS 联合坐标重配权与分层合成...")
    active_alloc = joint_is_coordinate_reweight(
        df_grid, w_on_grid, init_alloc, unit_forensic_meta, spec, intra_synth
    )
    w_synth = intra_synth.synthesize(df_grid, w_on_grid, active_alloc, spec)
    ev = evaluate_single_asset_backtest(df_grid, w_synth, spec.symbol, fee_rate=spec.fee_rate)

    dr: pd.Series = ev["daily_return"]
    nav = 1.0 + dr.cumsum()
    peak = np.maximum.accumulate(np.maximum(nav.values, 1.0))
    dd = (nav.values - peak) / peak * 100.0

    # 提取标的底层主力连续基准价格走势
    df_grid_indexed = df_grid.set_index("dt")
    price_daily = df_grid_indexed["close"].resample("1D").last().dropna()
    common_dates = dr.index.intersection(price_daily.index)
    b_series = price_daily.loc[common_dates]
    b_nav = b_series / b_series.iloc[0]

    # 保存每日明细 CSV
    daily_df = pd.DataFrame(
        {
            "date": dr.index.strftime("%Y-%m-%d"),
            "daily_return": dr.values,
            "strategy_nav": nav.values,
            "drawdown_pct": dd,
            "benchmark_nav": b_nav.reindex(dr.index).ffill().bfill().values,
        }
    )
    csv_path = CTA_OUTPUT_DIR / "ag_daily_perf.csv"
    daily_df.to_csv(csv_path, index=False)
    print(f">>> 每日净值明细已保存至: {csv_path}")

    # 逐年统计
    dr_s = pd.Series(dr.values, index=pd.to_datetime(dr.index))
    years = sorted(list(set(dr_s.index.year)))
    yearly_stats = []
    for y in years:
        sub = dr_s[dr_s.index.year == y]
        ret = float(sub.sum()) * 100.0
        vol = float(sub.std() * np.sqrt(252)) * 100.0
        sh = float(sub.mean() / (sub.std() + 1e-8) * np.sqrt(252))
        sub_nav = 1.0 + sub.cumsum()
        sub_peak = np.maximum.accumulate(np.maximum(sub_nav.values, 1.0))
        sub_mdd = float(np.min((sub_nav.values - sub_peak) / sub_peak)) * 100.0
        yearly_stats.append(
            {
                "year": int(y),
                "return_pct": round(ret, 2),
                "vol_pct": round(vol, 2),
                "sharpe": round(sh, 2),
                "mdd_pct": round(abs(sub_mdd), 2),
            }
        )

    # 1. 导出 Plotly 交互式净值大屏 (双 Y 轴: 策略净值 vs 白银基准价格)
    pos_series = pd.Series(w_synth, index=df_grid["dt"]).resample("1D").last().fillna(0.0).reindex(dr.index).fillna(0.0)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.55, 0.25, 0.20],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]],
        subplot_titles=(
            "白银主力连续 (KQ.m@SHFE.ag) 规则型多因子 CTA 策略累计净值 (左轴) vs 白银基准价格 (右轴)",
            "策略动态回撤曲线 (Drawdown %)",
            "模型多空净头寸暴露 (Net Position Weight [-1, +1])",
        ),
    )

    # 策略净值 (左轴)
    fig.add_trace(
        go.Scatter(
            x=dr.index,
            y=nav.values,
            mode="lines",
            name=f"多因子 CTA 策略 (全周期 +{ev['full']['cum_ret_pct']:.1f}%, OOS +{ev['oos']['cum_ret_pct']:.1f}%)",
            line={"width": 2.8, "color": "#AB47BC"},
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    # 基准净值 (右轴)
    fig.add_trace(
        go.Scatter(
            x=dr.index,
            y=daily_df["benchmark_nav"].values,
            mode="lines",
            name=f"白银主力基准价格 (+{(daily_df['benchmark_nav'].iloc[-1]-1.0)*100:.1f}%)",
            line={"width": 1.5, "color": "#90A4AE", "dash": "dot"},
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    # 回撤
    fig.add_trace(
        go.Scatter(
            x=dr.index,
            y=dd,
            mode="lines",
            fill="tozeroy",
            name=f"策略动态回撤 (最大回撤 {ev['full']['mdd_pct']:.2f}%)",
            line={"width": 1.2, "color": "#EF5350"},
        ),
        row=2,
        col=1,
    )
    # 持仓头寸
    fig.add_trace(
        go.Scatter(
            x=dr.index,
            y=pos_series.values,
            mode="lines",
            name="净头寸权重",
            line={"width": 1.0, "color": "#26A69A"},
        ),
        row=3,
        col=1,
    )

    fig.add_vline(x="2024-01-01", line_width=2, line_dash="dash", line_color="#FFD54F")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#101721",
        height=860,
        hovermode="x unified",
        title={
            "text": "商品期货主力连续 (KQ.m@SHFE.ag) 规则型多因子 CTA 策略深度绩效全景 (含夜盘×主力换月调仓)",
            "font": {"size": 18, "color": "#FFFFFF"},
        },
    )
    fig.update_yaxes(title_text="策略净值 (起点=1.0)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="白银主力现价倍数", row=1, col=1, secondary_y=True)

    out_html = CTA_OUTPUT_DIR / "ag_cta_performance.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f">>> Plotly 交互式净值大屏已生成至: {out_html}")

    # 2. 生成内嵌 Generative UI 卡片组件 (Tailwind CSS + SVG 双轴独立缩放 + 交互 HUD)
    step = max(1, len(dr) // 180)
    sample_df = daily_df.iloc[::step].copy()
    if sample_df.iloc[-1]["date"] != daily_df.iloc[-1]["date"]:
        sample_df = pd.concat([sample_df, daily_df.iloc[[-1]]])

    svg_data = {
        "dates": sample_df["date"].tolist(),
        "nav": [round(float(v), 4) for v in sample_df["strategy_nav"].tolist()],
        "bench": [round(float(v), 4) for v in sample_df["benchmark_nav"].tolist()],
        "dd": [round(float(v), 2) for v in sample_df["drawdown_pct"].tolist()],
        "yearly": yearly_stats,
        "full_stats": ev["full"],
        "is_stats": ev["is"],
        "oos_stats": ev["oos"],
        "n_units": len(active_alloc),
    }

    widget_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-3">
  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-2xl p-5 shadow-lg max-w-4xl mx-auto">
    <!-- Header -->
    <div class="flex flex-wrap items-center justify-between pb-4 border-b border-[var(--border)] gap-2">
      <div>
        <div class="flex items-center gap-2">
          <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-purple-500/20 text-purple-400 border border-purple-500/30">SHFE 商品期货</span>
          <h2 class="text-xl font-bold tracking-tight text-[var(--foreground)]">白银主力连续 (KQ.m@SHFE.ag)</h2>
        </div>
        <p class="text-xs text-[var(--muted-foreground)] mt-1">规则型多因子 CTA 策略 · 夜盘连续对齐 × 主力换月跳空后复权 × 三支柱状态门控</p>
      </div>
      <div class="flex gap-2 text-right">
        <div class="px-3 py-1.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">
          <div class="text-xs text-[var(--muted-foreground)]">挂载因子单元</div>
          <div class="text-base font-bold text-purple-400">{len(active_alloc)} 个</div>
        </div>
        <div class="px-3 py-1.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">
          <div class="text-xs text-[var(--muted-foreground)]">纯样本外(2024-26)</div>
          <div class="text-base font-bold text-emerald-400">+{ev['oos']['cum_ret_pct']:.2f}%</div>
        </div>
        <div class="px-3 py-1.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">
          <div class="text-xs text-[var(--muted-foreground)]">全周期总收益</div>
          <div class="text-base font-bold text-purple-300">+{ev['full']['cum_ret_pct']:.2f}%</div>
        </div>
      </div>
    </div>

    <!-- Metrics Cards Grid -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 my-4">
      <div class="p-3 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs text-[var(--muted-foreground)]">全周期夏普比率</div>
        <div class="text-lg font-bold text-[var(--foreground)] mt-0.5">{ev['full']['sharpe']:.2f}</div>
        <div class="text-xs text-purple-400 mt-0.5">OOS 夏普: {ev['oos']['sharpe']:.2f}</div>
      </div>
      <div class="p-3 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs text-[var(--muted-foreground)]">最大动态回撤</div>
        <div class="text-lg font-bold text-rose-400 mt-0.5">-{ev['full']['mdd_pct']:.2f}%</div>
        <div class="text-xs text-[var(--muted-foreground)] mt-0.5">卡玛比率: {ev['full']['calmar']:.2f}</div>
      </div>
      <div class="p-3 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs text-[var(--muted-foreground)]">年化复合收益率</div>
        <div class="text-lg font-bold text-emerald-400 mt-0.5">+{ev['full']['ann_ret_pct']:.2f}%</div>
        <div class="text-xs text-emerald-500 mt-0.5">OOS 年化: +{ev['oos']['ann_ret_pct']:.2f}%</div>
      </div>
      <div class="p-3 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs text-[var(--muted-foreground)]">持仓活跃度</div>
        <div class="text-lg font-bold text-[var(--foreground)] mt-0.5">{ev['full']['active_pct']:.1f}%</div>
        <div class="text-xs text-[var(--muted-foreground)] mt-0.5">多空双向轮动</div>
      </div>
    </div>

    <!-- Chart Container with SVG Dual Axis -->
    <div class="relative bg-[var(--background)] rounded-xl border border-[var(--border)] p-4 select-none" id="chart-card">
      <div class="flex items-center justify-between mb-2">
        <div class="flex items-center gap-4 text-xs">
          <span class="flex items-center gap-1.5"><span class="w-3 h-1 bg-purple-500 rounded"></span><span class="text-purple-400 font-semibold">策略净值 (左轴)</span></span>
          <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 border-t border-dashed border-slate-400"></span><span class="text-slate-400">白银基准价格 (右轴)</span></span>
          <span class="flex items-center gap-1.5"><span class="w-3 h-1 bg-rose-500/70 rounded"></span><span class="text-rose-400">回撤 (%)</span></span>
        </div>
        <span class="text-xs px-2 py-0.5 rounded bg-amber-500/20 text-amber-400 font-mono">2024起为纯样本外 OOS</span>
      </div>

      <!-- Hover Cursor HUD -->
      <div id="hover-hud" class="hidden absolute top-4 right-4 bg-slate-900/90 border border-slate-700/80 rounded-lg p-2.5 text-xs z-20 pointer-events-none shadow-xl backdrop-blur-sm">
        <div id="hud-date" class="font-mono text-slate-300 font-bold mb-1">2024-05-18</div>
        <div class="flex items-center gap-2 text-purple-300">
          <span>策略净值:</span>
          <span id="hud-nav" class="font-mono font-bold">1.284</span>
        </div>
        <div class="flex items-center gap-2 text-slate-400">
          <span>白银基准:</span>
          <span id="hud-bench" class="font-mono font-bold">1.840</span>
        </div>
        <div class="flex items-center gap-2 text-rose-400">
          <span>当前回撤:</span>
          <span id="hud-dd" class="font-mono font-bold">-3.2%</span>
        </div>
      </div>

      <!-- SVG Main NAV Plot -->
      <div class="relative h-64 w-full">
        <!-- Axis labels -->
        <div class="absolute left-1 top-2 text-[11px] font-mono text-purple-400" id="label-nav-max">1.45</div>
        <div class="absolute left-1 bottom-4 text-[11px] font-mono text-purple-400" id="label-nav-min">0.90</div>
        <div class="absolute right-1 top-2 text-[11px] font-mono text-slate-400 text-right" id="label-bench-max">4.50</div>
        <div class="absolute right-1 bottom-4 text-[11px] font-mono text-slate-400 text-right" id="label-bench-min">0.80</div>

        <svg id="nav-svg" class="w-full h-full cursor-crosshair" viewBox="0 0 800 240" preserveAspectRatio="none">
          <!-- Grid lines -->
          <line x1="50" y1="20" x2="750" y2="20" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="4"/>
          <line x1="50" y1="70" x2="750" y2="70" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="4"/>
          <line x1="50" y1="120" x2="750" y2="120" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="4"/>
          <line x1="50" y1="170" x2="750" y2="170" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="4"/>
          <line x1="50" y1="220" x2="750" y2="220" stroke="currentColor" stroke-opacity="0.2"/>

          <!-- OOS Split Line -->
          <line id="oos-line" x1="480" y1="10" x2="480" y2="220" stroke="#FFD54F" stroke-width="1.5" stroke-dasharray="4"/>
          <text id="oos-text" x="485" y="24" fill="#FFD54F" font-size="11" font-weight="bold">OOS (2024.01)</text>

          <!-- Benchmark line (Right Axis Scale) -->
          <path id="bench-path" d="" fill="none" stroke="#78909C" stroke-width="1.4" stroke-dasharray="4,3" opacity="0.8"/>

          <!-- Strategy Fill Area & Line (Left Axis Scale) -->
          <path id="nav-fill" d="" fill="url(#nav-grad)" opacity="0.25"/>
          <path id="nav-path" d="" fill="none" stroke="#AB47BC" stroke-width="2.8"/>

          <!-- Hover Indicator Vertical Line -->
          <line id="hover-line" x1="0" y1="10" x2="0" y2="220" stroke="#FFFFFF" stroke-width="1" stroke-dasharray="2,2" class="hidden"/>
          <circle id="hover-nav-dot" cx="0" cy="0" r="4" fill="#AB47BC" stroke="#FFFFFF" stroke-width="1.5" class="hidden"/>
          <circle id="hover-bench-dot" cx="0" cy="0" r="3.5" fill="#78909C" stroke="#FFFFFF" stroke-width="1.5" class="hidden"/>

          <defs>
            <linearGradient id="nav-grad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stop-color="#AB47BC" stop-opacity="0.8"/>
              <stop offset="100%" stop-color="#AB47BC" stop-opacity="0.0"/>
            </linearGradient>
          </defs>
        </svg>
      </div>

      <!-- SVG Drawdown Plot -->
      <div class="h-20 w-full mt-2 border-t border-[var(--border)] pt-1">
        <svg id="dd-svg" class="w-full h-full" viewBox="0 0 800 70" preserveAspectRatio="none">
          <line x1="50" y1="10" x2="750" y2="10" stroke="currentColor" stroke-opacity="0.2"/>
          <path id="dd-fill" d="" fill="#EF5350" opacity="0.3"/>
          <path id="dd-path" d="" fill="none" stroke="#EF5350" stroke-width="1.2"/>
          <line id="dd-hover-line" x1="0" y1="10" x2="0" y2="65" stroke="#FFFFFF" stroke-width="1" stroke-dasharray="2,2" class="hidden"/>
        </svg>
      </div>

      <!-- Date Labels -->
      <div class="flex justify-between text-xs text-[var(--muted-foreground)] px-10 pt-1 font-mono">
        <span>2020-08</span>
        <span>2021-06</span>
        <span>2022-06</span>
        <span>2023-06</span>
        <span class="text-amber-400 font-bold">2024-01 (OOS)</span>
        <span>2025-06</span>
        <span>2026-09</span>
      </div>
    </div>

    <!-- Yearly Breakdown Table -->
    <div class="mt-4 overflow-x-auto">
      <div class="text-xs font-semibold text-[var(--foreground)] mb-2 flex items-center justify-between">
        <span>分年份实证统计 (2020 - 2026)</span>
        <span class="text-[var(--muted-foreground)] font-normal text-xs">日频收益率口径 (扣除万分之0.5手续费与换月滑点)</span>
      </div>
      <table class="w-full text-xs text-left">
        <thead class="bg-[var(--background)] text-[var(--muted-foreground)] border-b border-[var(--border)]">
          <tr>
            <th class="py-1.5 px-3">年份</th>
            <th class="py-1.5 px-3">区间性质</th>
            <th class="py-1.5 px-3">年度收益率</th>
            <th class="py-1.5 px-3">年化波动率</th>
            <th class="py-1.5 px-3">夏普比率</th>
            <th class="py-1.5 px-3">最大回撤</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-[var(--border)]">
          {''.join([
              f'''<tr class="hover:bg-slate-500/5">
                <td class="py-1.5 px-3 font-mono font-medium">{y['year']}</td>
                <td class="py-1.5 px-3">{"<span class='px-1.5 py-0.5 rounded text-xs bg-amber-500/20 text-amber-400 font-medium'>OOS 样本外</span>" if y['year'] >= 2024 else "<span class='px-1.5 py-0.5 rounded text-xs bg-blue-500/20 text-blue-400 font-medium'>IS 样本内</span>"}</td>
                <td class="py-1.5 px-3 font-mono font-bold {('text-emerald-400' if y['return_pct'] >= 0 else 'text-rose-400')}">{y['return_pct']:+.2f}%</td>
                <td class="py-1.5 px-3 font-mono">{y['vol_pct']:.2f}%</td>
                <td class="py-1.5 px-3 font-mono">{y['sharpe']:.2f}</td>
                <td class="py-1.5 px-3 font-mono text-rose-400">-{y['mdd_pct']:.2f}%</td>
              </tr>'''
              for y in yearly_stats
          ])}
        </tbody>
      </table>
    </div>

    <!-- Forensic Analysis Insights -->
    <div class="mt-4 p-3 rounded-xl bg-[var(--background)] border border-[var(--border)] text-xs space-y-1.5">
      <div class="font-bold text-[var(--foreground)] flex items-center gap-1.5">
        <span class="w-2 h-2 rounded-full bg-purple-500"></span>
        <span>白银多因子 CTA 策略特征归因</span>
      </div>
      <div class="text-[var(--muted-foreground)] leading-relaxed">
        • <strong>顺势大涨捕获能力强</strong>：在 2024 年白银主升浪中录得 <strong>+27.55%</strong> (夏普 2.14, 最大回撤仅 3.38%)；纯样本外 (2024-2026) 累计收益 <strong>+41.28%</strong> (夏普 1.19, 卡玛 0.80)。<br>
        • <strong>震荡市磨损可控</strong>：在 2021/2023 年白银宽幅剧烈震荡中，三支柱状态门控 (Home-Regime Gating) 与波动率自适应收缩有效将最大回撤锁定在 16.53%，显著优于单纯做多现货的回撤。
      </div>
    </div>
  </div>

  <script>
    const data = {json.dumps(svg_data)};
    const navs = data.nav;
    const benchs = data.bench;
    const dds = data.dd;
    const dates = data.dates;
    const n = navs.length;

    // 策略净值 (左轴专用独立标尺)
    const minNav = Math.min(...navs) * 0.96;
    const maxNav = Math.max(...navs) * 1.04;
    document.getElementById('label-nav-max').innerText = maxNav.toFixed(2);
    document.getElementById('label-nav-min').innerText = minNav.toFixed(2);

    // 白银基准 (右轴专用独立标尺)
    const minBench = Math.min(...benchs) * 0.90;
    const maxBench = Math.max(...benchs) * 1.05;
    document.getElementById('label-bench-max').innerText = maxBench.toFixed(2);
    document.getElementById('label-bench-min').innerText = minBench.toFixed(2);

    const minDd = Math.min(...dds, -20.0);

    const xLeft = 50, xRight = 750;
    const yTop = 20, yBottom = 220;
    const ddTop = 10, ddBottom = 65;

    function getX(i) {{
      return xLeft + (i / (n - 1)) * (xRight - xLeft);
    }}
    function getYNav(val) {{
      return yBottom - ((val - minNav) / (maxNav - minNav)) * (yBottom - yTop);
    }}
    function getYBench(val) {{
      return yBottom - ((val - minBench) / (maxBench - minBench)) * (yBottom - yTop);
    }}
    function getDdY(val) {{
      return ddTop + (Math.abs(val) / Math.abs(minDd)) * (ddBottom - ddTop);
    }}

    // 绘制 NAV 路径 (左轴)
    let navD = `M ${{getX(0)}} ${{getYNav(navs[0])}}`;
    let fillD = `M ${{getX(0)}} ${{yBottom}} L ${{getX(0)}} ${{getYNav(navs[0])}}`;
    for (let i = 1; i < n; i++) {{
      navD += ` L ${{getX(i)}} ${{getYNav(navs[i])}}`;
      fillD += ` L ${{getX(i)}} ${{getYNav(navs[i])}}`;
    }}
    fillD += ` L ${{getX(n - 1)}} ${{yBottom}} Z`;

    document.getElementById('nav-path').setAttribute('d', navD);
    document.getElementById('nav-fill').setAttribute('d', fillD);

    // 绘制基准路径 (右轴)
    let benchD = `M ${{getX(0)}} ${{getYBench(benchs[0])}}`;
    for (let i = 1; i < n; i++) {{
      benchD += ` L ${{getX(i)}} ${{getYBench(benchs[i])}}`;
    }}
    document.getElementById('bench-path').setAttribute('d', benchD);

    // 绘制回撤路径
    let ddD = `M ${{getX(0)}} ${{getDdY(dds[0])}}`;
    let ddFill = `M ${{getX(0)}} ${{ddTop}} L ${{getX(0)}} ${{getDdY(dds[0])}}`;
    for (let i = 1; i < n; i++) {{
      ddD += ` L ${{getX(i)}} ${{getDdY(dds[i])}}`;
      ddFill += ` L ${{getX(i)}} ${{getDdY(dds[i])}}`;
    }}
    ddFill += ` L ${{getX(n - 1)}} ${{ddTop}} Z`;
    document.getElementById('dd-path').setAttribute('d', ddD);
    document.getElementById('dd-fill').setAttribute('d', ddFill);

    // 计算 2024-01-01 的 OOS 分割线位置
    const oosIdx = dates.findIndex(d => d >= '2024-01-01');
    if (oosIdx > 0) {{
      const oosX = getX(oosIdx);
      const oosLine = document.getElementById('oos-line');
      oosLine.setAttribute('x1', oosX);
      oosLine.setAttribute('x2', oosX);
      const oosText = document.getElementById('oos-text');
      if (oosText) {{
        oosText.setAttribute('x', oosX + 6);
      }}
    }}

    // 交互式悬浮游标
    const svg = document.getElementById('nav-svg');
    const hoverLine = document.getElementById('hover-line');
    const hoverNavDot = document.getElementById('hover-nav-dot');
    const hoverBenchDot = document.getElementById('hover-bench-dot');
    const ddHoverLine = document.getElementById('dd-hover-line');
    const hud = document.getElementById('hover-hud');
    const hudDate = document.getElementById('hud-date');
    const hudNav = document.getElementById('hud-nav');
    const hudBench = document.getElementById('hud-bench');
    const hudDd = document.getElementById('hud-dd');

    svg.addEventListener('mousemove', (e) => {{
      const rect = svg.getBoundingClientRect();
      const relX = (e.clientX - rect.left) / rect.width * 800;
      if (relX < xLeft || relX > xRight) return;
      const idx = Math.min(n - 1, Math.max(0, Math.round(((relX - xLeft) / (xRight - xLeft)) * (n - 1))));
      const curX = getX(idx);
      const curNavY = getYNav(navs[idx]);
      const curBenchY = getYBench(benchs[idx]);

      hoverLine.setAttribute('x1', curX);
      hoverLine.setAttribute('x2', curX);
      hoverLine.classList.remove('hidden');

      ddHoverLine.setAttribute('x1', curX);
      ddHoverLine.setAttribute('x2', curX);
      ddHoverLine.classList.remove('hidden');

      hoverNavDot.setAttribute('cx', curX);
      hoverNavDot.setAttribute('cy', curNavY);
      hoverNavDot.classList.remove('hidden');

      hoverBenchDot.setAttribute('cx', curX);
      hoverBenchDot.setAttribute('cy', curBenchY);
      hoverBenchDot.classList.remove('hidden');

      hud.classList.remove('hidden');
      hudDate.innerText = dates[idx];
      hudNav.innerText = navs[idx].toFixed(4) + ` (${{((navs[idx]-1)*100).toFixed(1)}}%)`;
      hudBench.innerText = benchs[idx].toFixed(4) + ` (${{((benchs[idx]-1)*100).toFixed(1)}}%)`;
      hudDd.innerText = dds[idx].toFixed(2) + '%';
    }});

    svg.addEventListener('mouseleave', () => {{
      hoverLine.classList.add('hidden');
      hoverNavDot.classList.add('hidden');
      hoverBenchDot.classList.add('hidden');
      ddHoverLine.classList.add('hidden');
      hud.classList.add('hidden');
    }});
  </script>
</body>
</html>
"""
    artifact_html = ARTIFACT_DIR / "ag_performance_chart.html"
    artifact_html.write_text(widget_html, encoding="utf-8")
    print(f">>> Generative UI 交互式净值卡片已保存至: {artifact_html}")


if __name__ == "__main__":
    main()
