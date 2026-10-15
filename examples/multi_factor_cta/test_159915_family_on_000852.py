#!/usr/bin/env python
"""
将 159915 挖掘的 44 单元规则型策略族【零样本直接迁移】至 000852 (中证1000指数/IM股指衍生) 全周期实证测试
- 不重新挖掘、不重新筛选、不重新配权，100% 沿用 champion_portfolio.json 的 44 个单元与权重；
- 唯一变化：底层 K 线标的由 SZSE.159915 替换为 SHSE.000852 (2020-2026 5分钟全周期数据)。
"""

from __future__ import annotations

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

from examples.multi_factor_cta.data_engine import (  # noqa: E402
    AssetClass,
    AssetSpec,
    CrossAssetDataEngine,
    ExecutionMode,
    map_tf_weight_to_base_grid,
)
from examples.multi_factor_cta.factor_engine import RuleFactorRegistry  # noqa: E402
from examples.multi_factor_cta.synthesis_engine import (  # noqa: E402
    IntraAssetThreePillarSynthesizer,
    evaluate_single_asset_backtest,
)

WORKSPACE_DIR = ROOT_DIR / "examples" / "results" / "alpha_agent_workspace"
CTA_OUTPUT_DIR = ROOT_DIR / "examples" / "results" / "multi_factor_cta"
ARTIFACT_DIR = Path(r"C:\Users\zifuy\.gemini\antigravity\brain\011f8ba6-910e-487c-8bed-d60ee4ca1112")


def main() -> None:
    CTA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("  【零样本跨品种盲测：将 159915 的 44 单元策略族原封不动直接迁移至 000852 (中证1000指数/IM衍生)】")
    print("  约束：不重新挖掘、不重新筛选、不重新配权，100% 复用 159915 冠军组合配置 (champion_portfolio.json)")
    print("=" * 120)

    data_engine = CrossAssetDataEngine(cache_dir=WORKSPACE_DIR)
    registry = RuleFactorRegistry()
    intra_synth = IntraAssetThreePillarSynthesizer()

    # 1. 加载 159915 的 44 个冠军单元及原权重
    champ_path = WORKSPACE_DIR / "champion_portfolio.json"
    champ_data = json.loads(champ_path.read_text(encoding="utf-8"))
    champ_alloc: dict[str, float] = champ_data["alloc_map"]
    print(f"\n[1] 已加载 159915 原生策略族配置: 共 {len(champ_alloc)} 个规则因子单元")

    # 2. 构建 000852 与 159915 完全同构的 5m~1d 五周期资产配置
    spec_000852 = AssetSpec(
        symbol="SHSE.000852",
        name="000852_中证1000指数(直接套用159915策略族)",
        asset_class=AssetClass.INDEX_FUTURES,
        sector="EQUITY_SMALLCAP",
        base_tf="5m",
        eligible_tfs=("1d", "60m", "30m", "15m", "5m"),
        fee_rate=0.0002,
        has_night_session=False,
        execution_mode=ExecutionMode.LONG_SHORT,
        data_path=ROOT_DIR / "examples" / "results" / "000852_real_5m_2020_2026.parquet",
    )

    print("\n[2] 正在加载 000852 (2020-2026) 5m~1d 五周期 K 线并计算 CZSC 原生笔/分型信号...")
    tf_dfs = data_engine.load_asset_multi_timeframes(spec_000852, attach_czsc=True)
    df_grid = tf_dfs[spec_000852.base_tf]
    print(f"    -> 5m 基准网格 K 线总数: {len(df_grid)} 根 ({df_grid['dt'].min()} ~ {df_grid['dt'].max()})")

    # 3. 逐一计算 44 个策略单元在 000852 上的触发与持仓权重
    w_on_grid: dict[str, np.ndarray] = {}
    cluster_counts: dict[str, int] = {}
    for uid, _w_val in champ_alloc.items():
        sid, tf = uid.split("@")
        f_spec = registry.get(sid)
        cluster_counts[f_spec.archetype] = cluster_counts.get(f_spec.archetype, 0) + 1
        _, w_arr = registry.evaluate_factor_on_tf(tf_dfs[tf], sid, tf)
        w_on_grid[uid] = map_tf_weight_to_base_grid(tf_dfs[tf], w_arr, df_grid)

    print(f"    -> 44 个策略单元映射完成，战略簇分布: {cluster_counts}")

    # 4. 评估四种运行模式：
    # 模式 A: 100% 镜像原版 (多空双向 LONG_SHORT, 不开额外状态门控 enable_regime_gating=False)
    w_ls_raw = intra_synth.synthesize(
        df_grid,
        w_on_grid,
        champ_alloc,
        spec_000852,
        execution_mode_override=ExecutionMode.LONG_SHORT,
        enable_regime_gating=False,
    )
    ev_ls_raw = evaluate_single_asset_backtest(df_grid, w_ls_raw, spec_000852.symbol, fee_rate=spec_000852.fee_rate)

    # 模式 B: 多空双向 + 开启跨品种主场状态门控 (LONG_SHORT + Home-Regime Gating)
    w_ls_gated = intra_synth.synthesize(
        df_grid,
        w_on_grid,
        champ_alloc,
        spec_000852,
        execution_mode_override=ExecutionMode.LONG_SHORT,
        enable_regime_gating=True,
    )
    ev_ls_gated = evaluate_single_asset_backtest(
        df_grid, w_ls_gated, spec_000852.symbol, fee_rate=spec_000852.fee_rate
    )

    # 模式 C: 纯多头现货增强模式 (LONG_ONLY_SPOT [0, +1])
    w_long_only = intra_synth.synthesize(
        df_grid,
        w_on_grid,
        champ_alloc,
        spec_000852,
        execution_mode_override=ExecutionMode.LONG_ONLY_SPOT,
        enable_regime_gating=False,
    )
    ev_long_only = evaluate_single_asset_backtest(
        df_grid, w_long_only, spec_000852.symbol, fee_rate=spec_000852.fee_rate
    )

    # 模式 D: 日内 14:55 强平 T+0 模式 (INTRADAY_FLAT_1455, 零隔夜敞口)
    w_intraday = intra_synth.synthesize(
        df_grid,
        w_on_grid,
        champ_alloc,
        spec_000852,
        execution_mode_override=ExecutionMode.INTRADAY_FLAT_1455,
        enable_regime_gating=False,
    )
    ev_intraday = evaluate_single_asset_backtest(
        df_grid, w_intraday, spec_000852.symbol, fee_rate=spec_000852.fee_rate
    )

    # 基准买入持有 (000852 Buy & Hold)
    df_grid_indexed = df_grid.set_index("dt")
    price_daily = df_grid_indexed["close"].resample("1D").last().dropna()
    dr_index = ev_ls_raw["daily_return"].index
    common_dates = dr_index.intersection(price_daily.index)
    b_series = price_daily.loc[common_dates]
    b_ret = b_series.pct_change().fillna(0.0).reindex(dr_index).fillna(0.0)
    b_nav = 1.0 + b_ret.cumsum()
    b_peak = np.maximum.accumulate(np.maximum(b_nav.values, 1.0))
    b_dd = (b_nav.values - b_peak) / b_peak * 100.0

    modes = [
        ("1. 多空双向 (100%原版镜像迁移)", ev_ls_raw, w_ls_raw),
        ("2. 多空双向 (含主场状态门控)", ev_ls_gated, w_ls_gated),
        ("3. 纯多头现货模式 [0, +1]", ev_long_only, w_long_only),
        ("4. 日内 14:55 强平不隔夜", ev_intraday, w_intraday),
    ]

    print("\n" + "=" * 120)
    print("  【三、159915 策略族 (44单元) 零样本直接迁移至 000852 绩效全景汇总 (2020-2026)】")
    print("=" * 120)
    rows = []
    for label, ev, _ in modes:
        rows.append(
            {
                "执行模式": label,
                "IS净收益(20-23)": f"{ev['is']['cum_ret_pct']:+.2f}%",
                "IS夏普": ev["is"]["sharpe"],
                "IS卡玛": ev["is"]["calmar"],
                "OOS净收益(24-26)": f"{ev['oos']['cum_ret_pct']:+.2f}%",
                "OOS夏普": ev["oos"]["sharpe"],
                "OOS卡玛": ev["oos"]["calmar"],
                "全周期总收益": f"{ev['full']['cum_ret_pct']:+.2f}%",
                "全周期年化": f"{ev['full']['ann_ret_pct']:.2f}%",
                "全周期夏普": ev["full"]["sharpe"],
                "最大回撤": f"{ev['full']['mdd_pct']:.2f}%",
                "全周期卡玛": ev["full"]["calmar"],
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    # 分年度统计
    dr_ls = pd.Series(ev_ls_raw["daily_return"].values, index=pd.to_datetime(dr_index))
    dr_lo = pd.Series(ev_long_only["daily_return"].values, index=pd.to_datetime(dr_index))
    dr_bh = pd.Series(b_ret.values, index=pd.to_datetime(dr_index))
    years = sorted(list(set(dr_ls.index.year)))
    yearly_stats = []
    for y in years:
        sub_ls = dr_ls[dr_ls.index.year == y]
        sub_lo = dr_lo[dr_lo.index.year == y]
        sub_bh = dr_bh[dr_bh.index.year == y]

        ret_ls = float(sub_ls.sum()) * 100.0
        vol_ls = float(sub_ls.std() * np.sqrt(252)) * 100.0
        sh_ls = float(sub_ls.mean() / (sub_ls.std() + 1e-8) * np.sqrt(252))
        nav_ls_y = 1.0 + sub_ls.cumsum()
        pk_ls_y = np.maximum.accumulate(np.maximum(nav_ls_y.values, 1.0))
        mdd_ls = float(np.min((nav_ls_y.values - pk_ls_y) / pk_ls_y)) * 100.0

        ret_lo = float(sub_lo.sum()) * 100.0
        sh_lo = float(sub_lo.mean() / (sub_lo.std() + 1e-8) * np.sqrt(252))

        ret_bh = float(sub_bh.sum()) * 100.0

        yearly_stats.append(
            {
                "year": int(y),
                "ls_ret_pct": round(ret_ls, 2),
                "ls_vol_pct": round(vol_ls, 2),
                "ls_sharpe": round(sh_ls, 2),
                "ls_mdd_pct": round(abs(mdd_ls), 2),
                "lo_ret_pct": round(ret_lo, 2),
                "lo_sharpe": round(sh_lo, 2),
                "bh_ret_pct": round(ret_bh, 2),
            }
        )

    print("\n" + "=" * 120)
    print("  【四、000852 分年份绩效明细 (2020 - 2026)】")
    print("=" * 120)
    print(pd.DataFrame(yearly_stats).to_string(index=False))

    # 保存每日净值 CSV
    nav_ls = 1.0 + ev_ls_raw["daily_return"].cumsum()
    pk_ls = np.maximum.accumulate(np.maximum(nav_ls.values, 1.0))
    dd_ls = (nav_ls.values - pk_ls) / pk_ls * 100.0

    nav_lo = 1.0 + ev_long_only["daily_return"].cumsum()
    pk_lo = np.maximum.accumulate(np.maximum(nav_lo.values, 1.0))
    dd_lo = (nav_lo.values - pk_lo) / pk_lo * 100.0

    nav_id = 1.0 + ev_intraday["daily_return"].cumsum()

    daily_df = pd.DataFrame(
        {
            "date": dr_index.strftime("%Y-%m-%d"),
            "ls_daily_return": ev_ls_raw["daily_return"].values,
            "ls_nav": nav_ls.values,
            "ls_drawdown_pct": dd_ls,
            "long_only_nav": nav_lo.values,
            "long_only_dd_pct": dd_lo,
            "intraday_nav": nav_id.values,
            "benchmark_nav": b_nav.values,
            "benchmark_dd_pct": b_dd,
        }
    )
    csv_path = CTA_OUTPUT_DIR / "000852_transfer_daily_perf.csv"
    daily_df.to_csv(csv_path, index=False)
    print(f"\n>>> 每日净值明细已保存至: {csv_path}")

    # 5. 生成 Plotly 交互式净值大屏
    pos_series = (
        pd.Series(w_ls_raw, index=df_grid["dt"]).resample("1D").last().fillna(0.0).reindex(dr_index).fillna(0.0)
    )
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=(
            "159915策略族(44单元) 零样本直接迁移至 000852 (中证1000) 累计净值对比 (2020-2026)",
            "策略动态回撤对比 (Drawdown %)",
            "多空双向模式日末净头寸暴露 (Net Position [-1, +1])",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=dr_index,
            y=nav_ls.values,
            mode="lines",
            name=f"多空双向原版迁移 (Full +{ev_ls_raw['full']['cum_ret_pct']:.1f}%, 夏普 {ev_ls_raw['full']['sharpe']:.2f}, 回撤 {ev_ls_raw['full']['mdd_pct']:.2f}%)",
            line={"width": 2.8, "color": "#00E676"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dr_index,
            y=nav_lo.values,
            mode="lines",
            name=f"纯多头现货 [0,+1] (Full +{ev_long_only['full']['cum_ret_pct']:.1f}%, 夏普 {ev_long_only['full']['sharpe']:.2f}, 回撤 {ev_long_only['full']['mdd_pct']:.2f}%)",
            line={"width": 2.2, "color": "#42A5F5"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dr_index,
            y=nav_id.values,
            mode="lines",
            name=f"日内14:55强平轨 (Full +{ev_intraday['full']['cum_ret_pct']:.1f}%, 夏普 {ev_intraday['full']['sharpe']:.2f}, 回撤 {ev_intraday['full']['mdd_pct']:.2f}%)",
            line={"width": 1.8, "color": "#FFA726"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dr_index,
            y=b_nav.values,
            mode="lines",
            name=f"000852 买入持有基准 ({(b_nav.iloc[-1]-1.0)*100:+.1f}%)",
            line={"width": 1.5, "color": "#90A4AE", "dash": "dot"},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dr_index,
            y=dd_ls,
            mode="lines",
            fill="tozeroy",
            name=f"多空双向回撤 (MDD {ev_ls_raw['full']['mdd_pct']:.2f}%)",
            line={"width": 1.3, "color": "#00E676"},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dr_index,
            y=dd_lo,
            mode="lines",
            name=f"纯多头现货回撤 (MDD {ev_long_only['full']['mdd_pct']:.2f}%)",
            line={"width": 1.2, "color": "#42A5F5"},
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dr_index,
            y=pos_series.values,
            mode="lines",
            name="日末净头寸权重",
            line={"width": 1.0, "color": "#AB47BC"},
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
            "text": "159915 规则型策略族 (44单元) 零样本直接迁移至 000852 (中证1000指数/IM衍生) 全周期实证大屏",
            "font": {"size": 18, "color": "#FFFFFF"},
        },
    )

    out_html = CTA_OUTPUT_DIR / "transfer_159915_to_000852_dashboard.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f">>> Plotly 交互式净值大屏已生成至: {out_html}")

    # 6. 生成内嵌 Generative UI 卡片组件
    step = max(1, len(dr_index) // 180)
    sample_df = daily_df.iloc[::step].copy()
    if sample_df.iloc[-1]["date"] != daily_df.iloc[-1]["date"]:
        sample_df = pd.concat([sample_df, daily_df.iloc[[-1]]])

    svg_data = {
        "dates": sample_df["date"].tolist(),
        "ls_nav": [round(float(v), 4) for v in sample_df["ls_nav"].tolist()],
        "lo_nav": [round(float(v), 4) for v in sample_df["long_only_nav"].tolist()],
        "id_nav": [round(float(v), 4) for v in sample_df["intraday_nav"].tolist()],
        "bench": [round(float(v), 4) for v in sample_df["benchmark_nav"].tolist()],
        "dd": [round(float(v), 2) for v in sample_df["ls_drawdown_pct"].tolist()],
        "yearly": yearly_stats,
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
          <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">零样本跨标的直接迁移</span>
          <h2 class="text-xl font-bold tracking-tight text-[var(--foreground)]">中证1000指数 (SHSE.000852) · 套用 159915 44单元策略族</h2>
        </div>
        <p class="text-xs text-[var(--muted-foreground)] mt-1">不重新挖掘、不改参数、不改权重 · 100% 原封不动迁移 159915 冠军组合 (5m/15m/30m/60m/1d)</p>
      </div>
      <div class="flex gap-2 text-right">
        <div class="px-3 py-1.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">
          <div class="text-xs text-[var(--muted-foreground)]">多空双向总收益</div>
          <div class="text-base font-bold text-emerald-400">+{ev_ls_raw['full']['cum_ret_pct']:.2f}%</div>
        </div>
        <div class="px-3 py-1.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">
          <div class="text-xs text-[var(--muted-foreground)]">纯多头现货[0,+1]</div>
          <div class="text-base font-bold text-sky-400">+{ev_long_only['full']['cum_ret_pct']:.2f}%</div>
        </div>
        <div class="px-3 py-1.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">
          <div class="text-xs text-[var(--muted-foreground)]">2024-26 纯OOS</div>
          <div class="text-base font-bold text-amber-400">+{ev_ls_raw['oos']['cum_ret_pct']:.2f}%</div>
        </div>
      </div>
    </div>

    <!-- Metrics Cards Grid -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 my-4">
      <div class="p-3 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs text-[var(--muted-foreground)]">多空双向全周期夏普</div>
        <div class="text-lg font-bold text-emerald-400 mt-0.5">{ev_ls_raw['full']['sharpe']:.2f}</div>
        <div class="text-xs text-[var(--muted-foreground)] mt-0.5">IS: {ev_ls_raw['is']['sharpe']:.2f} | OOS: {ev_ls_raw['oos']['sharpe']:.2f}</div>
      </div>
      <div class="p-3 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs text-[var(--muted-foreground)]">多空最大动态回撤</div>
        <div class="text-lg font-bold text-rose-400 mt-0.5">-{ev_ls_raw['full']['mdd_pct']:.2f}%</div>
        <div class="text-xs text-[var(--muted-foreground)] mt-0.5">卡玛比率: {ev_ls_raw['full']['calmar']:.2f}</div>
      </div>
      <div class="p-3 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs text-[var(--muted-foreground)]">纯多头现货夏普 / 回撤</div>
        <div class="text-lg font-bold text-sky-400 mt-0.5">{ev_long_only['full']['sharpe']:.2f} / -{ev_long_only['full']['mdd_pct']:.2f}%</div>
        <div class="text-xs text-[var(--muted-foreground)] mt-0.5">卡玛比率: {ev_long_only['full']['calmar']:.2f}</div>
      </div>
      <div class="p-3 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs text-[var(--muted-foreground)]">14:55日内强平夏普 / 回撤</div>
        <div class="text-lg font-bold text-amber-400 mt-0.5">{ev_intraday['full']['sharpe']:.2f} / -{ev_intraday['full']['mdd_pct']:.2f}%</div>
        <div class="text-xs text-[var(--muted-foreground)] mt-0.5">全周期 +{ev_intraday['full']['cum_ret_pct']:.2f}%</div>
      </div>
    </div>

    <!-- Chart Container -->
    <div class="relative bg-[var(--background)] rounded-xl border border-[var(--border)] p-4 select-none">
      <div class="flex flex-wrap items-center justify-between mb-2 gap-2">
        <div class="flex flex-wrap items-center gap-4 text-xs">
          <span class="flex items-center gap-1.5"><span class="w-3 h-1 bg-emerald-400 rounded"></span><span class="text-emerald-400 font-semibold">多空双向迁移 (+{ev_ls_raw['full']['cum_ret_pct']:.1f}%)</span></span>
          <span class="flex items-center gap-1.5"><span class="w-3 h-1 bg-sky-400 rounded"></span><span class="text-sky-400 font-semibold">纯多头现货 (+{ev_long_only['full']['cum_ret_pct']:.1f}%)</span></span>
          <span class="flex items-center gap-1.5"><span class="w-3 h-1 bg-amber-400 rounded"></span><span class="text-amber-400">14:55日内强平 (+{ev_intraday['full']['cum_ret_pct']:.1f}%)</span></span>
          <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 border-t border-dashed border-slate-400"></span><span class="text-slate-400">000852基准 ({(b_nav.iloc[-1]-1.0)*100:+.1f}%)</span></span>
        </div>
        <span class="text-xs px-2 py-0.5 rounded bg-amber-500/20 text-amber-400 font-mono">虚线右端为 2024+ 样本外</span>
      </div>

      <!-- Hover HUD -->
      <div id="hover-hud" class="hidden absolute top-10 right-4 bg-slate-900/90 border border-slate-700/80 rounded-lg p-2.5 text-xs z-20 pointer-events-none shadow-xl backdrop-blur-sm">
        <div id="hud-date" class="font-mono text-slate-300 font-bold mb-1">2024-05-18</div>
        <div class="flex items-center gap-2 text-emerald-400"><span>多空双向:</span><span id="hud-ls" class="font-mono font-bold"></span></div>
        <div class="flex items-center gap-2 text-sky-400"><span>纯多现货:</span><span id="hud-lo" class="font-mono font-bold"></span></div>
        <div class="flex items-center gap-2 text-slate-400"><span>000852基准:</span><span id="hud-bench" class="font-mono font-bold"></span></div>
        <div class="flex items-center gap-2 text-rose-400"><span>多空回撤:</span><span id="hud-dd" class="font-mono font-bold"></span></div>
      </div>

      <div class="relative h-64 w-full">
        <div class="absolute left-1 top-2 text-xs font-mono text-emerald-400" id="label-nav-max"></div>
        <div class="absolute left-1 bottom-4 text-xs font-mono text-emerald-400" id="label-nav-min"></div>

        <svg id="nav-svg" class="w-full h-full cursor-crosshair" viewBox="0 0 800 240" preserveAspectRatio="none">
          <line x1="50" y1="20" x2="770" y2="20" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="4"/>
          <line x1="50" y1="70" x2="770" y2="70" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="4"/>
          <line x1="50" y1="120" x2="770" y2="120" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="4"/>
          <line x1="50" y1="170" x2="770" y2="170" stroke="currentColor" stroke-opacity="0.08" stroke-dasharray="4"/>
          <line x1="50" y1="220" x2="770" y2="220" stroke="currentColor" stroke-opacity="0.2"/>

          <line id="oos-line" x1="480" y1="10" x2="480" y2="220" stroke="#FFD54F" stroke-width="1.5" stroke-dasharray="4"/>
          <text id="oos-text" x="485" y="24" fill="#FFD54F" font-size="11" font-weight="bold">OOS (2024.01)</text>

          <path id="bench-path" d="" fill="none" stroke="#78909C" stroke-width="1.4" stroke-dasharray="4,3" opacity="0.8"/>
          <path id="id-path" d="" fill="none" stroke="#FFA726" stroke-width="1.6" opacity="0.85"/>
          <path id="lo-path" d="" fill="none" stroke="#42A5F5" stroke-width="2.0"/>
          <path id="ls-fill" d="" fill="url(#ls-grad)" opacity="0.22"/>
          <path id="ls-path" d="" fill="none" stroke="#00E676" stroke-width="2.8"/>

          <line id="hover-line" x1="0" y1="10" x2="0" y2="220" stroke="#FFFFFF" stroke-width="1" stroke-dasharray="2,2" class="hidden"/>
          <circle id="hover-ls-dot" cx="0" cy="0" r="4" fill="#00E676" stroke="#FFFFFF" stroke-width="1.5" class="hidden"/>
          <circle id="hover-lo-dot" cx="0" cy="0" r="3.5" fill="#42A5F5" stroke="#FFFFFF" stroke-width="1.5" class="hidden"/>

          <defs>
            <linearGradient id="ls-grad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stop-color="#00E676" stop-opacity="0.8"/>
              <stop offset="100%" stop-color="#00E676" stop-opacity="0.0"/>
            </linearGradient>
          </defs>
        </svg>
      </div>

      <!-- Drawdown Plot -->
      <div class="h-20 w-full mt-2 border-t border-[var(--border)] pt-1">
        <svg id="dd-svg" class="w-full h-full" viewBox="0 0 800 70" preserveAspectRatio="none">
          <line x1="50" y1="10" x2="770" y2="10" stroke="currentColor" stroke-opacity="0.2"/>
          <path id="dd-fill" d="" fill="#EF5350" opacity="0.28"/>
          <path id="dd-path" d="" fill="none" stroke="#EF5350" stroke-width="1.2"/>
          <line id="dd-hover-line" x1="0" y1="10" x2="0" y2="65" stroke="#FFFFFF" stroke-width="1" stroke-dasharray="2,2" class="hidden"/>
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

    <!-- Yearly Table -->
    <div class="mt-4 overflow-x-auto">
      <div class="text-xs font-semibold text-[var(--foreground)] mb-2 flex items-center justify-between">
        <span>000852 分年份实证表现 (2020 - 2026 · 零样本直接套用 159915 策略族)</span>
        <span class="text-[var(--muted-foreground)] font-normal text-xs">单边费率 2BP (0.02%)</span>
      </div>
      <table class="w-full text-xs text-left">
        <thead class="bg-[var(--background)] text-[var(--muted-foreground)] border-b border-[var(--border)]">
          <tr>
            <th class="py-1.5 px-3">年份</th>
            <th class="py-1.5 px-3">多空双向收益</th>
            <th class="py-1.5 px-3">多空夏普</th>
            <th class="py-1.5 px-3">多空最大回撤</th>
            <th class="py-1.5 px-3">纯多头现货收益</th>
            <th class="py-1.5 px-3">纯多头夏普</th>
            <th class="py-1.5 px-3">000852基准收益</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-[var(--border)]">
          {''.join([
              f'''<tr class="hover:bg-slate-500/5">
                <td class="py-1.5 px-3 font-mono font-medium">{y['year']}</td>
                <td class="py-1.5 px-3 font-mono font-bold {('text-emerald-400' if y['ls_ret_pct'] >= 0 else 'text-rose-400')}">{y['ls_ret_pct']:+.2f}%</td>
                <td class="py-1.5 px-3 font-mono">{y['ls_sharpe']:.2f}</td>
                <td class="py-1.5 px-3 font-mono text-rose-400">-{y['ls_mdd_pct']:.2f}%</td>
                <td class="py-1.5 px-3 font-mono font-bold {('text-sky-400' if y['lo_ret_pct'] >= 0 else 'text-rose-400')}">{y['lo_ret_pct']:+.2f}%</td>
                <td class="py-1.5 px-3 font-mono">{y['lo_sharpe']:.2f}</td>
                <td class="py-1.5 px-3 font-mono text-[var(--muted-foreground)]">{y['bh_ret_pct']:+.2f}%</td>
              </tr>'''
              for y in yearly_stats
          ])}
        </tbody>
      </table>
    </div>
  </div>

  <script>
    const data = {json.dumps(svg_data)};
    const lsNav = data.ls_nav;
    const loNav = data.lo_nav;
    const idNav = data.id_nav;
    const benchs = data.bench;
    const dds = data.dd;
    const dates = data.dates;
    const n = lsNav.length;

    const allVals = [...lsNav, ...loNav, ...idNav, ...benchs];
    const minNav = Math.min(...allVals) * 0.96;
    const maxNav = Math.max(...allVals) * 1.04;
    document.getElementById('label-nav-max').innerText = maxNav.toFixed(2);
    document.getElementById('label-nav-min').innerText = minNav.toFixed(2);

    const minDd = Math.min(...dds, -16.0);
    const xLeft = 50, xRight = 770;
    const yTop = 20, yBottom = 220;
    const ddTop = 10, ddBottom = 65;

    function getX(i) {{
      return xLeft + (i / (n - 1)) * (xRight - xLeft);
    }}
    function getY(val) {{
      return yBottom - ((val - minNav) / (maxNav - minNav)) * (yBottom - yTop);
    }}
    function getDdY(val) {{
      return ddTop + (Math.abs(val) / Math.abs(minDd)) * (ddBottom - ddTop);
    }}

    function buildPath(arr) {{
      let d = `M ${{getX(0)}} ${{getY(arr[0])}}`;
      for (let i = 1; i < n; i++) {{
        d += ` L ${{getX(i)}} ${{getY(arr[i])}}`;
      }}
      return d;
    }}

    const lsD = buildPath(lsNav);
    const fillD = `M ${{getX(0)}} ${{yBottom}} L ` + lsD.slice(2) + ` L ${{getX(n - 1)}} ${{yBottom}} Z`;
    document.getElementById('ls-path').setAttribute('d', lsD);
    document.getElementById('ls-fill').setAttribute('d', fillD);
    document.getElementById('lo-path').setAttribute('d', buildPath(loNav));
    document.getElementById('id-path').setAttribute('d', buildPath(idNav));
    document.getElementById('bench-path').setAttribute('d', buildPath(benchs));

    let ddD = `M ${{getX(0)}} ${{getDdY(dds[0])}}`;
    let ddFill = `M ${{getX(0)}} ${{ddTop}} L ${{getX(0)}} ${{getDdY(dds[0])}}`;
    for (let i = 1; i < n; i++) {{
      ddD += ` L ${{getX(i)}} ${{getDdY(dds[i])}}`;
      ddFill += ` L ${{getX(i)}} ${{getDdY(dds[i])}}`;
    }}
    ddFill += ` L ${{getX(n - 1)}} ${{ddTop}} Z`;
    document.getElementById('dd-path').setAttribute('d', ddD);
    document.getElementById('dd-fill').setAttribute('d', ddFill);

    const oosIdx = dates.findIndex(d => d >= '2024-01-01');
    if (oosIdx > 0) {{
      const oosX = getX(oosIdx);
      const oosLine = document.getElementById('oos-line');
      oosLine.setAttribute('x1', oosX);
      oosLine.setAttribute('x2', oosX);
      const oosText = document.getElementById('oos-text');
      if (oosText) oosText.setAttribute('x', oosX + 6);
    }}

    const svg = document.getElementById('nav-svg');
    const hoverLine = document.getElementById('hover-line');
    const hoverLsDot = document.getElementById('hover-ls-dot');
    const hoverLoDot = document.getElementById('hover-lo-dot');
    const ddHoverLine = document.getElementById('dd-hover-line');
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

      ddHoverLine.setAttribute('x1', curX);
      ddHoverLine.setAttribute('x2', curX);
      ddHoverLine.classList.remove('hidden');

      hoverLsDot.setAttribute('cx', curX);
      hoverLsDot.setAttribute('cy', getY(lsNav[idx]));
      hoverLsDot.classList.remove('hidden');

      hoverLoDot.setAttribute('cx', curX);
      hoverLoDot.setAttribute('cy', getY(loNav[idx]));
      hoverLoDot.classList.remove('hidden');

      hud.classList.remove('hidden');
      document.getElementById('hud-date').innerText = dates[idx];
      document.getElementById('hud-ls').innerText = lsNav[idx].toFixed(4) + ` (${{((lsNav[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-lo').innerText = loNav[idx].toFixed(4) + ` (${{((loNav[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-bench').innerText = benchs[idx].toFixed(4) + ` (${{((benchs[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-dd').innerText = dds[idx].toFixed(2) + '%';
    }});

    svg.addEventListener('mouseleave', () => {{
      hoverLine.classList.add('hidden');
      hoverLsDot.classList.add('hidden');
      hoverLoDot.classList.add('hidden');
      ddHoverLine.classList.add('hidden');
      hud.classList.add('hidden');
    }});
  </script>
</body>
</html>
"""
    artifact_html = ARTIFACT_DIR / "transfer_000852_chart.html"
    artifact_html.write_text(widget_html, encoding="utf-8")
    print(f">>> Generative UI 交互式净值卡片已保存至: {artifact_html}")


if __name__ == "__main__":
    main()
