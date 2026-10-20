#!/usr/bin/env python
"""
159915 规则型策略族 -> 九大品种跨资产全天候 CTA 总组合全景实证
覆盖：
- 四大宽基指数/ETF (4 品种)：159915 (创业板ETF)、510500 (中证500ETF)、000852 (中证1000)、588000 (科创50ETF)
- 五大商品期货主力连续 (5 品种，已去掉 B_背驰反转簇)：AG (沪银)、AU (沪金)、SC (原油)、LC (碳酸锂)、PS (多晶硅)
输出：
1. 九大品种单标的绩效、股指子组合、商品子组合、九品种在市等权 (1/N_t) 与跨资产 Sector-HRP 全天候总组合对比；
2. 跨资产策略日收益相关性矩阵与股商子组合低相关性验证；
3. 2020-2026 分年度收益明细、Plotly 全景大屏与 Generative UI 交互式净值看板。
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
)
from examples.multi_factor_cta.run_framework_demo import load_research_universe, synthesize_admitted_asset  # noqa: E402
from examples.multi_factor_cta.synthesis_engine import (  # noqa: E402
    CrossAssetHRPSynthesizer,
    IntraAssetThreePillarSynthesizer,
    build_asset_execution_inputs,
    evaluate_single_asset_backtest,
)

WORKSPACE_DIR = ROOT_DIR / "examples" / "results" / "alpha_agent_workspace"
RESULTS_DIR = ROOT_DIR / "examples" / "results"
COMMODITY_DATA_DIR = RESULTS_DIR / "commodities_5m"
CTA_OUTPUT_DIR = RESULTS_DIR / "multi_factor_cta"
ARTIFACT_DIR = CTA_OUTPUT_DIR


def build_nine_assets_universe() -> list[tuple[str, AssetSpec]]:
    return [
        (
            "159915",
            AssetSpec(
                symbol="SZSE.159915",
                name="159915_创业板ETF",
                asset_class=AssetClass.ETF_T0,
                sector="EQUITY_GROWTH",
                base_tf="5m",
                eligible_tfs=("1d", "60m", "30m", "15m", "5m"),
                fee_rate=0.0002,
                has_night_session=False,
                execution_mode=ExecutionMode.LONG_SHORT,
                data_path=RESULTS_DIR / "159915_real_5m_2020_2026.csv",
            ),
        ),
        (
            "510500",
            AssetSpec(
                symbol="SHSE.510500",
                name="510500_中证500ETF",
                asset_class=AssetClass.ETF_T0,
                sector="EQUITY_MIDCAP",
                base_tf="5m",
                eligible_tfs=("1d", "60m", "30m", "15m", "5m"),
                fee_rate=0.0002,
                has_night_session=False,
                execution_mode=ExecutionMode.LONG_SHORT,
                data_path=RESULTS_DIR / "510500_real_5m_2020_2026.parquet",
            ),
        ),
        (
            "000852",
            AssetSpec(
                symbol="SHSE.000852",
                name="000852_中证1000指数",
                asset_class=AssetClass.ETF_T0,
                sector="EQUITY_SMALLCAP",
                base_tf="5m",
                eligible_tfs=("1d", "60m", "30m", "15m", "5m"),
                fee_rate=0.0002,
                has_night_session=False,
                execution_mode=ExecutionMode.LONG_SHORT,
                data_path=RESULTS_DIR / "000852_real_5m_2020_2026.parquet",
            ),
        ),
        (
            "588000",
            AssetSpec(
                symbol="SHSE.588000",
                name="588000_科创50ETF",
                asset_class=AssetClass.ETF_T0,
                sector="EQUITY_STAR",
                base_tf="5m",
                eligible_tfs=("1d", "60m", "30m", "15m", "5m"),
                fee_rate=0.0002,
                has_night_session=False,
                execution_mode=ExecutionMode.LONG_SHORT,
                data_path=RESULTS_DIR / "588000_real_5m_2020_2026.parquet",
            ),
        ),
        (
            "ag",
            AssetSpec(
                symbol="AG9999.XSGE",
                name="ag_沪银主力连续",
                asset_class=AssetClass.COMMODITY_FUTURES,
                sector="PRECIOUS_METALS",
                base_tf="15m",
                eligible_tfs=("1d", "60m", "30m", "15m"),
                fee_rate=0.00015,
                has_night_session=True,
                execution_mode=ExecutionMode.LONG_SHORT,
                data_path=COMMODITY_DATA_DIR / "ag_15m_2020_2026.parquet",
            ),
        ),
        (
            "au",
            AssetSpec(
                symbol="AU9999.XSGE",
                name="au_沪金主力连续",
                asset_class=AssetClass.COMMODITY_FUTURES,
                sector="PRECIOUS_METALS",
                base_tf="15m",
                eligible_tfs=("1d", "60m", "30m", "15m"),
                fee_rate=0.00015,
                has_night_session=True,
                execution_mode=ExecutionMode.LONG_SHORT,
                data_path=COMMODITY_DATA_DIR / "au_15m_2020_2026.parquet",
            ),
        ),
        (
            "sc",
            AssetSpec(
                symbol="SC9999.XINE",
                name="sc_原油主力连续",
                asset_class=AssetClass.COMMODITY_FUTURES,
                sector="ENERGY",
                base_tf="15m",
                eligible_tfs=("1d", "60m", "30m", "15m"),
                fee_rate=0.00015,
                has_night_session=True,
                execution_mode=ExecutionMode.LONG_SHORT,
                data_path=COMMODITY_DATA_DIR / "sc_15m_2020_2026.parquet",
            ),
        ),
        (
            "lc",
            AssetSpec(
                symbol="LC9999.GFEX",
                name="lc_碳酸锂主力连续",
                asset_class=AssetClass.COMMODITY_FUTURES,
                sector="NEW_ENERGY",
                base_tf="15m",
                eligible_tfs=("1d", "60m", "30m", "15m"),
                fee_rate=0.00015,
                has_night_session=False,
                execution_mode=ExecutionMode.LONG_SHORT,
                data_path=COMMODITY_DATA_DIR / "lc_15m_2020_2026.parquet",
            ),
        ),
        (
            "ps",
            AssetSpec(
                symbol="PS9999.GFEX",
                name="ps_多晶硅主力连续",
                asset_class=AssetClass.COMMODITY_FUTURES,
                sector="NEW_ENERGY",
                base_tf="15m",
                eligible_tfs=("1d", "60m", "30m", "15m"),
                fee_rate=0.00015,
                has_night_session=False,
                execution_mode=ExecutionMode.LONG_SHORT,
                data_path=COMMODITY_DATA_DIR / "ps_15m_2020_2026.parquet",
            ),
        ),
    ]


def main() -> None:
    CTA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 130)
    print("  【159915 规则型策略族 -> 九大品种跨资产全天候 CTA 总组合 (4大宽基指数/ETF + 5大商品期货) 全景实证】")
    print("=" * 130)

    data_engine = CrossAssetDataEngine(cache_dir=WORKSPACE_DIR)
    intra_synth = IntraAssetThreePillarSynthesizer()
    hrp_synth = CrossAssetHRPSynthesizer(target_portfolio_vol=0.14)

    champ_path = WORKSPACE_DIR / "champion_portfolio.json"
    champ_alloc: dict[str, float] = json.loads(champ_path.read_text(encoding="utf-8"))["alloc_map"]

    universe = build_nine_assets_universe()
    pool_frames = load_research_universe(data_engine, [spec for _, spec in universe])
    asset_results: dict[str, dict] = {}
    asset_daily_returns: dict[str, pd.Series] = {}
    price_returns, signed_positions, execution_fees = {}, {}, {}
    asset_sectors: dict[str, str] = {}
    equity_returns: dict[str, pd.Series] = {}
    equity_sectors: dict[str, str] = {}
    comm_returns: dict[str, pd.Series] = {}
    comm_sectors: dict[str, str] = {}

    summary_rows = []

    for short_id, spec in universe:
        is_comm = spec.asset_class == AssetClass.COMMODITY_FUTURES
        print(f"\n>>> [加载与评估] {spec.name} ({spec.symbol}) | 资产类={spec.asset_class.value} | 板块={spec.sector}")
        tf_dfs = pool_frames[spec.symbol]
        df_grid = tf_dfs[spec.base_tf]
        w_synth, _, active_alloc, audits = synthesize_admitted_asset(
            spec,
            pool_frames,
            sorted(champ_alloc),
            intra_synth,
        )
        price_returns[short_id], signed_positions[short_id] = build_asset_execution_inputs(df_grid, w_synth)
        execution_fees[short_id] = spec.fee_rate
        ev = evaluate_single_asset_backtest(df_grid, w_synth, spec.symbol, fee_rate=spec.fee_rate)
        ev["name"] = spec.name
        ev["short_id"] = short_id
        ev["n_units"] = len(active_alloc)
        ev["start_dt"] = str(df_grid["dt"].min())[:10]
        ev["end_dt"] = str(df_grid["dt"].max())[:10]

        asset_results[short_id] = ev
        asset_daily_returns[short_id] = ev["daily_return"]
        asset_sectors[short_id] = spec.sector

        if is_comm:
            comm_returns[short_id] = ev["daily_return"]
            comm_sectors[short_id] = spec.sector
        else:
            equity_returns[short_id] = ev["daily_return"]
            equity_sectors[short_id] = spec.sector

        print(
            f"    区间: {ev['start_dt']} ~ {ev['end_dt']} | 单元数: {len(active_alloc)} | "
            f"OOS(24-26): 净收益={ev['oos']['cum_ret_pct']:+.2f}%, 夏普={ev['oos']['sharpe']:.2f}, 卡玛={ev['oos']['calmar']:.2f} | "
            f"全周期: 净收益={ev['full']['cum_ret_pct']:+.2f}%, 年化={ev['full']['ann_ret_pct']:+.2f}%, "
            f"夏普={ev['full']['sharpe']:.2f}, 最大回撤={ev['full']['mdd_pct']:.2f}%, 卡玛={ev['full']['calmar']:.2f}"
        )

        summary_rows.append(
            {
                "代码": short_id.upper(),
                "标的/组合名称": spec.name,
                "资产大类": "商品期货(统一准入)" if is_comm else "宽基股指/ETF",
                "IS净收益(20-23)": f"{ev['is']['cum_ret_pct']:+.2f}%" if ev["is"]["active_pct"] > 0 else "N/A",
                "IS夏普": ev["is"]["sharpe"] if ev["is"]["active_pct"] > 0 else 0.0,
                "OOS净收益(24-26)": f"{ev['oos']['cum_ret_pct']:+.2f}%",
                "OOS夏普": ev["oos"]["sharpe"],
                "OOS卡玛": ev["oos"]["calmar"],
                "全周期总收益": f"{ev['full']['cum_ret_pct']:+.2f}%",
                "全周期年化": f"{ev['full']['ann_ret_pct']:+.2f}%",
                "全周期夏普": ev["full"]["sharpe"],
                "最大回撤": f"{ev['full']['mdd_pct']:.2f}%",
                "全周期卡玛": ev["full"]["calmar"],
            }
        )

    # 构建三大组合：1. 四大股指子组合 | 2. 五大商品子组合 | 3. 九大品种跨资产全天候总组合 (等权 vs Sector-HRP)
    def allocate_subset(sectors: dict[str, str]) -> dict:
        return hrp_synth.build_multi_asset_portfolio(
            {sid: price_returns[sid] for sid in sectors},
            sectors,
            asset_positions_df={sid: signed_positions[sid] for sid in sectors},
            asset_fee_rates={sid: execution_fees[sid] for sid in sectors},
            returns_kind="price",
            positions_timing="execution",
            periods_per_year=252 * 48,
        )

    eq_sub_res = allocate_subset(equity_sectors)
    cm_sub_res = allocate_subset(comm_sectors)
    all_res = allocate_subset(asset_sectors)

    for code, name, cat, res_dict, k_prefix in [
        ("EQ_SUB", "◆ 四大宽基股指/ETF 子组合 (HRP)", "股指子组合(4品种)", eq_sub_res, ""),
        ("CM_SUB", "◆ 五大商品期货 子组合 (HRP·统一准入)", "商品子组合(5品种)", cm_sub_res, ""),
        ("ALL_1/N", "◇ 九大品种在市等权总组合 (1/N_t)", "全天候总组合(等权)", all_res, "eq_"),
        ("ALL_HRP", "★ 九大品种跨资产全天候总组合 (Sector-HRP)", "全天候总组合(HRP)", all_res, ""),
    ]:
        summary_rows.append(
            {
                "代码": code,
                "标的/组合名称": name,
                "资产大类": cat,
                "IS净收益(20-23)": f"{res_dict[k_prefix + 'is']['cum_ret_pct']:+.2f}%",
                "IS夏普": res_dict[k_prefix + "is"]["sharpe"],
                "OOS净收益(24-26)": f"{res_dict[k_prefix + 'oos']['cum_ret_pct']:+.2f}%",
                "OOS夏普": res_dict[k_prefix + "oos"]["sharpe"],
                "OOS卡玛": res_dict[k_prefix + "oos"]["calmar"],
                "全周期总收益": f"{res_dict[k_prefix + 'full']['cum_ret_pct']:+.2f}%",
                "全周期年化": f"{res_dict[k_prefix + 'full']['ann_ret_pct']:+.2f}%",
                "全周期夏普": res_dict[k_prefix + "full"]["sharpe"],
                "最大回撤": f"{res_dict[k_prefix + 'full']['mdd_pct']:.2f}%",
                "全周期卡玛": res_dict[k_prefix + "full"]["calmar"],
            }
        )

    print("\n" + "=" * 135)
    print("  【159915 规则策略族 -> 九大品种 (4大宽基指数 + 5大商品期货) 单品种及全天候跨资产总组合绩效汇总表】")
    print("=" * 135)
    print(pd.DataFrame(summary_rows).to_string(index=False))

    # 计算股指子组合与商品子组合的相关性
    eq_sub_dr: pd.Series = eq_sub_res["daily_returns"]
    cm_sub_dr: pd.Series = cm_sub_res["daily_returns"]
    all_hrp_dr: pd.Series = all_res["daily_returns"]
    all_eq_dr: pd.Series = all_res["equal_weight_returns"]

    aligned_sub = pd.DataFrame({"股指子组合": eq_sub_dr, "商品子组合": cm_sub_dr}).dropna()
    sub_corr = float(aligned_sub["股指子组合"].corr(aligned_sub["商品子组合"]))
    print(
        f"\n>>> [跨资产相关性验证] 四大宽基股指子组合 vs 五大商品期货子组合 日收益相关系数 = {sub_corr:+.4f} (近乎零相关，天然全天候互补!)"
    )

    # 分年份收益明细表 (2020 - 2026)
    years = sorted(set(all_hrp_dr.index.year))
    yr_rows = []
    for y in years:
        s_eq = eq_sub_dr[eq_sub_dr.index.year == y]
        s_cm = cm_sub_dr[cm_sub_dr.index.year == y]
        s_1n = all_eq_dr[all_eq_dr.index.year == y]
        s_all = all_hrp_dr[all_hrp_dr.index.year == y]

        all_ret = float(s_all.sum() * 100.0)
        all_sh = float(s_all.mean() / (s_all.std() + 1e-8) * np.sqrt(252))
        nav_y = 1.0 + s_all.cumsum()
        pk_y = np.maximum.accumulate(np.maximum(nav_y.values, 1.0))
        mdd_y = float(np.min((nav_y.values - pk_y) / pk_y) * 100.0)

        yr_rows.append(
            {
                "年份": int(y),
                "4大股指子组合": f"{s_eq.sum() * 100:+.2f}%",
                "5大商品子组合(统一准入)": f"{s_cm.sum() * 100:+.2f}%",
                "9品种在市等权(1/N)": f"{s_1n.sum() * 100:+.2f}%",
                "9品种等权夏普": round(float(s_1n.mean() / (s_1n.std() + 1e-8) * np.sqrt(252)), 2),
                "★ 9品种全天候HRP收益": f"{all_ret:+.2f}%",
                "★ 全天候HRP夏普": round(all_sh, 2),
                "★ 全天候HRP年内回撤": f"{mdd_y:.2f}%",
            }
        )

    print("\n" + "=" * 135)
    print("  【四大股指子组合 × 五大商品子组合 × 九品种全天候总组合 分年份收益明细 (2020 - 2026)】")
    print("=" * 135)
    print(pd.DataFrame(yr_rows).to_string(index=False))

    # 导出 CSV 与 Plotly 全景图
    all_dates = all_hrp_dr.index
    nav_df = pd.DataFrame({"date": all_dates.strftime("%Y-%m-%d")})
    nav_df["eq_sub_nav"] = (1.0 + eq_sub_dr.reindex(all_dates).fillna(0.0).cumsum()).values
    nav_df["cm_sub_nav"] = (1.0 + cm_sub_dr.reindex(all_dates).fillna(0.0).cumsum()).values
    nav_df["all_1n_nav"] = (1.0 + all_eq_dr.reindex(all_dates).fillna(0.0).cumsum()).values
    nav_df["all_hrp_nav"] = (1.0 + all_hrp_dr.reindex(all_dates).fillna(0.0).cumsum()).values
    pk_full = np.maximum.accumulate(np.maximum(nav_df["all_hrp_nav"].values, 1.0))
    nav_df["all_hrp_dd_pct"] = (nav_df["all_hrp_nav"].values - pk_full) / pk_full * 100.0

    csv_path = CTA_OUTPUT_DIR / "nine_assets_all_weather_daily_perf.csv"
    nav_df.to_csv(csv_path, index=False)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=(
            "159915策略族 -> 九大品种全天候 CTA 组合净值曲线 (4大股指子组合 + 5大商品子组合 -> 9品种全天候总组合)",
            "九大品种全天候 Sector-HRP 总组合动态回撤 (%)",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=all_dates,
            y=nav_df["eq_sub_nav"].values,
            mode="lines",
            name=f"◆ 四大宽基股指/ETF子组合 (Full {eq_sub_res['full']['cum_ret_pct']:+.1f}%, 夏普 {eq_sub_res['full']['sharpe']:.2f})",
            line={"width": 2.0, "color": "#42A5F5"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=all_dates,
            y=nav_df["cm_sub_nav"].values,
            mode="lines",
            name=f"◆ 五大商品期货子组合·去背驰反转 (Full {cm_sub_res['full']['cum_ret_pct']:+.1f}%, 夏普 {cm_sub_res['full']['sharpe']:.2f})",
            line={"width": 2.0, "color": "#FFA726"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=all_dates,
            y=nav_df["all_1n_nav"].values,
            mode="lines",
            name=f"◇ 九大品种在市等权 (1/N_t) 总组合 (Full {all_res['eq_full']['cum_ret_pct']:+.1f}%, 夏普 {all_res['eq_full']['sharpe']:.2f}, 回撤 {all_res['eq_full']['mdd_pct']:.2f}%)",
            line={"width": 2.4, "color": "#38BDF8", "dash": "dash"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=all_dates,
            y=nav_df["all_hrp_nav"].values,
            mode="lines",
            name=f"★ 九大品种跨资产全天候 Sector-HRP 总组合 (Full {all_res['full']['cum_ret_pct']:+.1f}%, 夏普 {all_res['full']['sharpe']:.2f}, 回撤 {all_res['full']['mdd_pct']:.2f}%)",
            line={"width": 3.4, "color": "#00E676"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=all_dates,
            y=nav_df["all_hrp_dd_pct"].values,
            mode="lines",
            fill="tozeroy",
            name=f"9品种全天候 HRP 回撤 (最大回撤 {all_res['full']['mdd_pct']:.2f}%)",
            line={"width": 1.4, "color": "#00E676"},
        ),
        row=2,
        col=1,
    )
    fig.add_vline(x="2024-01-01", line_width=2, line_dash="dash", line_color="#FFD54F")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#101721",
        height=820,
        hovermode="x unified",
        title={
            "text": "159915 规则型策略族 -> 九大品种 (4大宽基指数/ETF + 5大商品期货) 全天候跨资产 CTA 组合实证大屏",
            "font": {"size": 18, "color": "#FFFFFF"},
        },
    )
    out_html = CTA_OUTPUT_DIR / "nine_assets_all_weather_cta_dashboard.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"\n>>> 九大品种全天候 Plotly 大屏已保存至: {out_html}")

    # 生成内嵌 Generative UI 交互式净值看板
    step = max(1, len(nav_df) // 180)
    sample_df = nav_df.iloc[::step].copy()
    if sample_df.iloc[-1]["date"] != nav_df.iloc[-1]["date"]:
        sample_df = pd.concat([sample_df, nav_df.iloc[[-1]]])

    svg_data = {
        "dates": sample_df["date"].tolist(),
        "eq_sub": [round(float(v), 4) for v in sample_df["eq_sub_nav"].tolist()],
        "cm_sub": [round(float(v), 4) for v in sample_df["cm_sub_nav"].tolist()],
        "all_1n": [round(float(v), 4) for v in sample_df["all_1n_nav"].tolist()],
        "all_hrp": [round(float(v), 4) for v in sample_df["all_hrp_nav"].tolist()],
        "dd": [round(float(v), 2) for v in sample_df["all_hrp_dd_pct"].tolist()],
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
          <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">九大品种股商全天候 CTA</span>
          <h2 class="text-xl font-bold tracking-tight text-[var(--foreground)]">4大宽基指数/ETF × 5大商品期货 (去背驰反转)</h2>
        </div>
        <p class="text-xs text-[var(--muted-foreground)] mt-1">159915 · 510500 · 000852 · 588000 + AG · AU · SC · LC · PS | 股商日收益相关系数仅 {sub_corr:+.3f}</p>
      </div>
      <div class="flex gap-2 text-right">
        <div class="px-3 py-1.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">
          <div class="text-xs text-[var(--muted-foreground)]">9品种全天候HRP</div>
          <div class="text-base font-bold text-emerald-400">+{all_res["full"]["cum_ret_pct"]:.2f}% (Sh {all_res["full"]["sharpe"]:.2f})</div>
        </div>
        <div class="px-3 py-1.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">
          <div class="text-xs text-[var(--muted-foreground)]">9品种在市等权(1/N)</div>
          <div class="text-base font-bold text-sky-400">+{all_res["eq_full"]["cum_ret_pct"]:.2f}% (Sh {all_res["eq_full"]["sharpe"]:.2f})</div>
        </div>
        <div class="px-3 py-1.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">
          <div class="text-xs text-[var(--muted-foreground)]">HRP最大回撤 / 卡玛</div>
          <div class="text-base font-bold text-amber-400">-{all_res["full"]["mdd_pct"]:.2f}% / {all_res["full"]["calmar"]:.2f}</div>
        </div>
      </div>
    </div>

    <!-- Summary Cards -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-2.5 my-4">
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-blue-400">◆ 4大宽基股指子组合</div>
        <div class="text-base font-bold text-[var(--foreground)] mt-0.5">+{eq_sub_res["full"]["cum_ret_pct"]:.1f}% (Sh {eq_sub_res["full"]["sharpe"]:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS: +{eq_sub_res["oos"]["cum_ret_pct"]:.1f}% | MDD {eq_sub_res["full"]["mdd_pct"]:.2f}%</div>
      </div>
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-orange-400">◆ 5大商品子组合(统一准入)</div>
        <div class="text-base font-bold text-[var(--foreground)] mt-0.5">+{cm_sub_res["full"]["cum_ret_pct"]:.1f}% (Sh {cm_sub_res["full"]["sharpe"]:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS: +{cm_sub_res["oos"]["cum_ret_pct"]:.1f}% | MDD {cm_sub_res["full"]["mdd_pct"]:.2f}%</div>
      </div>
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-sky-400">◇ 9品种在市等权(1/N_t)</div>
        <div class="text-base font-bold text-[var(--foreground)] mt-0.5">+{all_res["eq_full"]["cum_ret_pct"]:.1f}% (Sh {all_res["eq_full"]["sharpe"]:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS: +{all_res["eq_oos"]["cum_ret_pct"]:.1f}% | MDD {all_res["eq_full"]["mdd_pct"]:.2f}%</div>
      </div>
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-emerald-400">★ 9品种全天候 Sector-HRP</div>
        <div class="text-base font-bold text-[var(--foreground)] mt-0.5">+{all_res["full"]["cum_ret_pct"]:.1f}% (Sh {all_res["full"]["sharpe"]:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS: +{all_res["oos"]["cum_ret_pct"]:.1f}% (Sh {all_res["oos"]["sharpe"]:.2f})</div>
      </div>
    </div>

    <!-- SVG Chart -->
    <div class="relative bg-[var(--background)] rounded-xl border border-[var(--border)] p-4 select-none">
      <div class="flex flex-wrap items-center justify-between mb-2 gap-2 text-xs">
        <div class="flex flex-wrap items-center gap-3">
          <span class="flex items-center gap-1"><span class="w-3 h-1 bg-emerald-400 rounded"></span><span class="text-emerald-400 font-bold">★ 9品种全天候HRP (+{all_res["full"]["cum_ret_pct"]:.1f}%)</span></span>
          <span class="flex items-center gap-1"><span class="w-3 h-0.5 bg-sky-400 border-b border-dashed"></span><span class="text-sky-400 font-semibold">◇ 9品种在市等权1/N (+{all_res["eq_full"]["cum_ret_pct"]:.1f}%)</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-blue-400"></span><span class="text-blue-400">◆ 4大股指子组合 (+{eq_sub_res["full"]["cum_ret_pct"]:.1f}%)</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-orange-400"></span><span class="text-orange-400">◆ 5大商品子组合 (+{cm_sub_res["full"]["cum_ret_pct"]:.1f}%)</span></span>
        </div>
        <span class="px-2 py-0.5 rounded bg-amber-500/20 text-amber-400 font-mono">虚线右端为 2024+ OOS</span>
      </div>

      <div id="hover-hud" class="hidden absolute top-10 right-4 bg-slate-900/90 border border-slate-700/80 rounded-lg p-2.5 text-xs z-20 pointer-events-none shadow-xl backdrop-blur-sm">
        <div id="hud-date" class="font-mono text-slate-300 font-bold mb-1"></div>
        <div class="text-emerald-400 font-mono" id="hud-hrp"></div>
        <div class="text-sky-400 font-mono" id="hud-1n"></div>
        <div class="text-blue-300 font-mono" id="hud-eq"></div>
        <div class="text-orange-300 font-mono" id="hud-cm"></div>
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

          <path id="eq-sub-path" d="" fill="none" stroke="#42A5F5" stroke-width="1.8" opacity="0.88"/>
          <path id="cm-sub-path" d="" fill="none" stroke="#FFA726" stroke-width="1.8" opacity="0.88"/>
          <path id="all-1n-path" d="" fill="none" stroke="#38BDF8" stroke-width="2.2" stroke-dasharray="5,3" opacity="0.95"/>

          <path id="hrp-fill" d="" fill="url(#hrp-grad)" opacity="0.22"/>
          <path id="hrp-path" d="" fill="none" stroke="#00E676" stroke-width="3.2"/>

          <line id="hover-line" x1="0" y1="10" x2="0" y2="220" stroke="#FFFFFF" stroke-width="1" stroke-dasharray="2,2" class="hidden"/>
          <circle id="hover-hrp-dot" cx="0" cy="0" r="4" fill="#00E676" stroke="#FFFFFF" stroke-width="1.5" class="hidden"/>

          <defs>
            <linearGradient id="hrp-grad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stop-color="#00E676" stop-opacity="0.8"/>
              <stop offset="100%" stop-color="#00E676" stop-opacity="0.0"/>
            </linearGradient>
          </defs>
        </svg>
      </div>

      <!-- Drawdown -->
      <div class="h-16 w-full mt-2 border-t border-[var(--border)] pt-1">
        <svg id="dd-svg" class="w-full h-full" viewBox="0 0 800 60" preserveAspectRatio="none">
          <line x1="50" y1="8" x2="770" y2="8" stroke="currentColor" stroke-opacity="0.2"/>
          <path id="dd-fill" d="" fill="#00E676" opacity="0.25"/>
          <path id="dd-path" d="" fill="none" stroke="#00E676" stroke-width="1.2"/>
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
    const data = {json.dumps(svg_data)};
    const dates = data.dates;
    const n = dates.length;
    const allVals = [...data.eq_sub, ...data.cm_sub, ...data.all_1n, ...data.all_hrp];
    const minNav = Math.min(...allVals) * 0.96;
    const maxNav = Math.max(...allVals) * 1.04;
    document.getElementById('label-nav-max').innerText = maxNav.toFixed(2);
    document.getElementById('label-nav-min').innerText = minNav.toFixed(2);

    const minDd = Math.min(...data.dd, -10.0);
    const xLeft = 50, xRight = 770, yTop = 20, yBottom = 220, ddTop = 8, ddBottom = 55;

    function getX(i) {{ return xLeft + (i / (n - 1)) * (xRight - xLeft); }}
    function getY(val) {{ return yBottom - ((val - minNav) / (maxNav - minNav)) * (yBottom - yTop); }}
    function getDdY(val) {{ return ddTop + (Math.abs(val) / Math.abs(minDd)) * (ddBottom - ddTop); }}

    function buildPath(arr) {{
      let d = `M ${{getX(0)}} ${{getY(arr[0])}}`;
      for (let i = 1; i < n; i++) d += ` L ${{getX(i)}} ${{getY(arr[i])}}`;
      return d;
    }}

    document.getElementById('eq-sub-path').setAttribute('d', buildPath(data.eq_sub));
    document.getElementById('cm-sub-path').setAttribute('d', buildPath(data.cm_sub));
    document.getElementById('all-1n-path').setAttribute('d', buildPath(data.all_1n));

    const hrpD = buildPath(data.all_hrp);
    const fillD = `M ${{getX(0)}} ${{yBottom}} L ` + hrpD.slice(2) + ` L ${{getX(n - 1)}} ${{yBottom}} Z`;
    document.getElementById('hrp-path').setAttribute('d', hrpD);
    document.getElementById('hrp-fill').setAttribute('d', fillD);

    let ddD = `M ${{getX(0)}} ${{getDdY(data.dd[0])}}`;
    let ddFill = `M ${{getX(0)}} ${{ddTop}} L ${{getX(0)}} ${{getDdY(data.dd[0])}}`;
    for (let i = 1; i < n; i++) {{
      ddD += ` L ${{getX(i)}} ${{getDdY(data.dd[i])}}`;
      ddFill += ` L ${{getX(i)}} ${{getDdY(data.dd[i])}}`;
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
      document.getElementById('oos-text').setAttribute('x', oosX + 6);
    }}

    const svg = document.getElementById('nav-svg');
    const hoverLine = document.getElementById('hover-line');
    const hoverDot = document.getElementById('hover-hrp-dot');
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
      hoverDot.setAttribute('cy', getY(data.all_hrp[idx]));
      hoverDot.classList.remove('hidden');
      hud.classList.remove('hidden');
      document.getElementById('hud-date').innerText = dates[idx];
      document.getElementById('hud-hrp').innerText = `★ 9品种HRP: ${{data.all_hrp[idx].toFixed(3)}} (${{((data.all_hrp[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-1n').innerText = `◇ 9品种等权: ${{data.all_1n[idx].toFixed(3)}} (${{((data.all_1n[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-eq').innerText = `◆ 4大股指: ${{data.eq_sub[idx].toFixed(3)}} (${{((data.eq_sub[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-cm').innerText = `◆ 5大商品: ${{data.cm_sub[idx].toFixed(3)}} (${{((data.cm_sub[idx]-1)*100).toFixed(1)}}%)`;
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
    artifact_html = ARTIFACT_DIR / "all_weather_9assets_cta_chart.html"
    artifact_html.write_text(widget_html, encoding="utf-8")
    print(f">>> 九大品种全天候 Generative UI 交互式净值卡片已保存至: {artifact_html}")


if __name__ == "__main__":
    main()
