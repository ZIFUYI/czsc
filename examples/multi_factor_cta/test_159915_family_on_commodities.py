#!/usr/bin/env python
"""
将 159915 规则型策略族推广至五大商品期货主力连续 (ag 沪银 / au 沪金 / sc 原油 / lc 碳酸锂 / ps 多晶硅) 全景实证
1. champion_portfolio.json 只提供候选ID；统一准入重审、滚动配权，IS结束后才交易；
2. 对比单品种表现与五大商品跨板块 Sector-HRP 风险平价组合表现；
3. 导出交互式 Plotly 大屏、每日净值 CSV 与内嵌 Generative UI 净值看板。
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
COMMODITY_DATA_DIR = ROOT_DIR / "examples" / "results" / "commodities_5m"
CTA_OUTPUT_DIR = ROOT_DIR / "examples" / "results" / "multi_factor_cta"
ARTIFACT_DIR = CTA_OUTPUT_DIR


def build_five_commodities_universe() -> list[AssetSpec]:
    return [
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
            data_path=COMMODITY_DATA_DIR / "ag_5m_2020_2026.parquet",
        ),
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
            data_path=COMMODITY_DATA_DIR / "au_5m_2020_2026.parquet",
        ),
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
            data_path=COMMODITY_DATA_DIR / "sc_5m_2020_2026.parquet",
        ),
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
            data_path=COMMODITY_DATA_DIR / "lc_5m_2020_2026.parquet",
        ),
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
            data_path=COMMODITY_DATA_DIR / "ps_5m_2020_2026.parquet",
        ),
    ]


def resample_5m_to_15m_base(raw_5m_path: Path, out_15m_path: Path, has_night: bool) -> Path:
    """将 5m 原始期货数据预聚合为 15m 基准网格缓存，加速多品种回测"""
    if out_15m_path.exists():
        return out_15m_path
    df = pd.read_parquet(raw_5m_path)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").drop_duplicates(subset=["dt"]).reset_index(drop=True)
    from examples.multi_factor_cta.data_engine import assign_trading_date

    df["tdate"] = assign_trading_date(df["dt"], has_night)
    df["b_idx"] = df.groupby("tdate", sort=False).cumcount()
    df["grp15"] = df["b_idx"] // 3
    df15 = (
        df.groupby(["tdate", "grp15"], as_index=False, sort=False)
        .agg(
            dt=("dt", "last"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .sort_values("dt")
        .reset_index(drop=True)
    )
    df15.to_parquet(out_15m_path, index=False)
    return out_15m_path


def main() -> None:
    CTA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 125)
    print(
        "  【159915 规则型策略族 -> 五大商品期货 (ag 沪银 / au 沪金 / sc 原油 / lc 碳酸锂 / ps 多晶硅) 样本内重审后的迁移研究】"
    )
    print("=" * 125)

    data_engine = CrossAssetDataEngine(cache_dir=WORKSPACE_DIR)
    intra_synth = IntraAssetThreePillarSynthesizer()
    hrp_synth = CrossAssetHRPSynthesizer(target_portfolio_vol=0.14)

    champ_path = WORKSPACE_DIR / "champion_portfolio.json"
    champ_alloc: dict[str, float] = json.loads(champ_path.read_text(encoding="utf-8"))["alloc_map"]

    specs = build_five_commodities_universe()
    asset_results: dict[str, dict] = {}
    asset_daily_returns: dict[str, pd.Series] = {}
    price_returns, signed_positions, execution_fees = {}, {}, {}
    asset_sectors: dict[str, str] = {}
    summary_rows = []

    prepared_specs = []
    for orig_spec in specs:
        short_id = orig_spec.name.split("_")[0]
        p15 = COMMODITY_DATA_DIR / f"{short_id}_15m_2020_2026.parquet"
        assert orig_spec.data_path is not None
        resample_5m_to_15m_base(orig_spec.data_path, p15, orig_spec.has_night_session)

        spec = AssetSpec(
            symbol=orig_spec.symbol,
            name=orig_spec.name,
            asset_class=orig_spec.asset_class,
            sector=orig_spec.sector,
            base_tf="15m",
            eligible_tfs=("1d", "60m", "30m", "15m"),
            fee_rate=orig_spec.fee_rate,
            has_night_session=orig_spec.has_night_session,
            execution_mode=orig_spec.execution_mode,
            data_path=p15,
        )

        prepared_specs.append(spec)

    pool_frames = load_research_universe(data_engine, prepared_specs)
    for spec in prepared_specs:
        short_id = spec.name.split("_")[0]
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

        print(
            f"    区间: {ev['start_dt']} ~ {ev['end_dt']} | 通过统一准入单元: {len(active_alloc)} | "
            f"OOS(24-26): 净收益={ev['oos']['cum_ret_pct']:+.2f}%, 夏普={ev['oos']['sharpe']:.2f}, 卡玛={ev['oos']['calmar']:.2f} | "
            f"全周期: 净收益={ev['full']['cum_ret_pct']:+.2f}%, 年化={ev['full']['ann_ret_pct']:+.2f}%, "
            f"夏普={ev['full']['sharpe']:.2f}, 最大回撤={ev['full']['mdd_pct']:.2f}%, 卡玛={ev['full']['calmar']:.2f}"
        )

        summary_rows.append(
            {
                "品种代码": short_id.upper(),
                "品种名称": spec.name,
                "数据起点": ev["start_dt"],
                "IS净收益(20-23)": f"{ev['is']['cum_ret_pct']:+.2f}%" if ev["is"]["active_pct"] > 0 else "N/A(未上市)",
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

    # 构建五大商品跨品种组合 (在市等权 1/N_t vs 跨板块 Sector-HRP)
    hrp_res = hrp_synth.build_multi_asset_portfolio(
        price_returns,
        asset_sectors,
        asset_positions_df=signed_positions,
        asset_fee_rates=execution_fees,
        returns_kind="price",
        positions_timing="execution",
        periods_per_year=252 * 16,
    )
    summary_rows.append(
        {
            "品种代码": "EQ_1/N",
            "品种名称": "◇ 五大商品在市等权组合 (1/N_t)",
            "数据起点": "2020-01-02",
            "IS净收益(20-23)": f"{hrp_res['eq_is']['cum_ret_pct']:+.2f}%",
            "IS夏普": hrp_res["eq_is"]["sharpe"],
            "OOS净收益(24-26)": f"{hrp_res['eq_oos']['cum_ret_pct']:+.2f}%",
            "OOS夏普": hrp_res["eq_oos"]["sharpe"],
            "OOS卡玛": hrp_res["eq_oos"]["calmar"],
            "全周期总收益": f"{hrp_res['eq_full']['cum_ret_pct']:+.2f}%",
            "全周期年化": f"{hrp_res['eq_full']['ann_ret_pct']:+.2f}%",
            "全周期夏普": hrp_res["eq_full"]["sharpe"],
            "最大回撤": f"{hrp_res['eq_full']['mdd_pct']:.2f}%",
            "全周期卡玛": hrp_res["eq_full"]["calmar"],
        }
    )
    summary_rows.append(
        {
            "品种代码": "SEC_HRP",
            "品种名称": "★ 五大商品 Sector-HRP 组合",
            "数据起点": "2020-01-02",
            "IS净收益(20-23)": f"{hrp_res['is']['cum_ret_pct']:+.2f}%",
            "IS夏普": hrp_res["is"]["sharpe"],
            "OOS净收益(24-26)": f"{hrp_res['oos']['cum_ret_pct']:+.2f}%",
            "OOS夏普": hrp_res["oos"]["sharpe"],
            "OOS卡玛": hrp_res["oos"]["calmar"],
            "全周期总收益": f"{hrp_res['full']['cum_ret_pct']:+.2f}%",
            "全周期年化": f"{hrp_res['full']['ann_ret_pct']:+.2f}%",
            "全周期夏普": hrp_res["full"]["sharpe"],
            "最大回撤": f"{hrp_res['full']['mdd_pct']:.2f}%",
            "全周期卡玛": hrp_res["full"]["calmar"],
        }
    )

    print("\n" + "=" * 125)
    print("  【五大商品期货零样本迁移表现 (已去掉背驰反转簇) vs 等权(1/N_t)与Sector-HRP组合汇总表】")
    print("=" * 125)
    print(pd.DataFrame(summary_rows).to_string(index=False))

    # 分年度明细表 (各品种 + 等权组合 + HRP组合年度收益率)
    port_dr: pd.Series = hrp_res["daily_returns"]
    eq_dr: pd.Series = hrp_res["equal_weight_returns"]
    years = sorted(set(port_dr.index.year))
    yr_rows = []
    for y in years:
        row: dict[str, int | float | str] = {"年份": int(y)}
        for sid in ["ag", "au", "sc", "lc", "ps"]:
            s_dr = asset_daily_returns[sid]
            sub = s_dr[s_dr.index.year == y]
            row[f"{sid.upper()}_收益"] = f"{sub.sum() * 100:+.2f}%" if len(sub) > 5 else "—"
        sub_eq = eq_dr[eq_dr.index.year == y]
        eq_ret = float(sub_eq.sum() * 100.0)
        eq_sh = float(sub_eq.mean() / (sub_eq.std() + 1e-8) * np.sqrt(252))
        sub_p = port_dr[port_dr.index.year == y]
        p_ret = float(sub_p.sum() * 100.0)
        p_sh = float(sub_p.mean() / (sub_p.std() + 1e-8) * np.sqrt(252))
        p_nav = 1.0 + sub_p.cumsum()
        p_pk = np.maximum.accumulate(np.maximum(p_nav.values, 1.0))
        p_mdd = float(np.min((p_nav.values - p_pk) / p_pk) * 100.0)
        row["等权(1/N)收益"] = f"{eq_ret:+.2f}%"
        row["等权夏普"] = round(eq_sh, 2)
        row["HRP组合收益"] = f"{p_ret:+.2f}%"
        row["HRP组合夏普"] = round(p_sh, 2)
        row["HRP组合回撤"] = f"{p_mdd:.2f}%"
        yr_rows.append(row)

    print("\n" + "=" * 125)
    print("  【五大商品期货及等权(1/N_t)/Sector-HRP组合分年份收益明细 (2020 - 2026)】")
    print("=" * 125)
    print(pd.DataFrame(yr_rows).to_string(index=False))

    # 打印各时期平均权重占比对比
    eq_w_df: pd.DataFrame = hrp_res["equal_weights_df"]
    hrp_w_df: pd.DataFrame = hrp_res["unscaled_hrp_weights_df"]
    eff_w_df: pd.DataFrame = hrp_res["asset_weights_df"]
    print("\n" + "=" * 125)
    print("  【品种间组合权重机制对比：在市等权(1/N_t) vs Sector-HRP 归一化权重 vs 目标波动率缩放后实际仓位权重】")
    print("=" * 125)
    for label, msk in [
        ("阶段1 (2020.01~2023.07: 仅 AG/AU/SC 3品种在市)", eq_w_df.index < "2023-07-24"),
        (
            "阶段2 (2023.07~2024.12: AG/AU/SC/LC 4品种在市)",
            (eq_w_df.index >= "2023-07-24") & (eq_w_df.index < "2024-12-27"),
        ),
        ("阶段3 (2024.12~2026.10: AG/AU/SC/LC/PS 5品种全在市)", eq_w_df.index >= "2024-12-27"),
    ]:
        print(f"  * {label}:")
        eq_m = eq_w_df.loc[msk].mean() * 100
        hrp_m = hrp_w_df.loc[msk].mean() * 100
        eff_m = eff_w_df.loc[msk].mean() * 100
        print("    - 在市等权 (1/N_t)      : " + ", ".join([f"{k.upper()}={v:.1f}%" for k, v in eq_m.items()]))
        print("    - Sector-HRP 风险预算权重: " + ", ".join([f"{k.upper()}={v:.1f}%" for k, v in hrp_m.items()]))
        print(
            "    - 目标波动率缩放后实际权重: "
            + ", ".join([f"{k.upper()}={v:.1f}%" for k, v in eff_m.items()])
            + f" (合计总敞口={eff_m.sum():.1f}%)"
        )

    # 保存 CSV 与 Plotly 大屏
    all_dates = port_dr.index
    nav_df = pd.DataFrame({"date": all_dates.strftime("%Y-%m-%d")})
    for sid in ["ag", "au", "sc", "lc", "ps"]:
        s_dr = asset_daily_returns[sid].reindex(all_dates).fillna(0.0)
        nav_df[f"{sid}_nav"] = (1.0 + s_dr.cumsum()).values
    nav_df["eq_nav"] = (1.0 + eq_dr.cumsum()).values
    nav_df["combo_nav"] = (1.0 + port_dr.cumsum()).values
    p_pk_full = np.maximum.accumulate(np.maximum(nav_df["combo_nav"].values, 1.0))
    nav_df["combo_dd_pct"] = (nav_df["combo_nav"].values - p_pk_full) / p_pk_full * 100.0

    csv_path = CTA_OUTPUT_DIR / "five_commodities_transfer_daily_perf.csv"
    nav_df.to_csv(csv_path, index=False)

    colors = {
        "ag": "#AB47BC",
        "au": "#FFD54F",
        "sc": "#FF7043",
        "lc": "#26C6DA",
        "ps": "#66BB6A",
    }
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=(
            "159915策略族 -> 五大商品期货 (ag/au/sc/lc/ps) 零样本迁移净值曲线与 Sector-HRP 组合 (2020-2026)",
            "五大商品 Sector-HRP 组合动态回撤 (%)",
        ),
    )
    for sid in ["ag", "au", "sc", "lc", "ps"]:
        ev = asset_results[sid]
        s_dr = asset_daily_returns[sid]
        s_nav = 1.0 + s_dr.cumsum()
        fig.add_trace(
            go.Scatter(
                x=s_dr.index,
                y=s_nav.values,
                mode="lines",
                name=f"{ev['name']} (Full {ev['full']['cum_ret_pct']:+.1f}%, 夏普 {ev['full']['sharpe']:.2f}, OOS {ev['oos']['cum_ret_pct']:+.1f}%)",
                line={"width": 1.8, "color": colors[sid]},
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=all_dates,
            y=nav_df["eq_nav"].values,
            mode="lines",
            name=f"◇ 五大商品在市等权 (1/N_t) 组合 (Full {hrp_res['eq_full']['cum_ret_pct']:+.1f}%, 夏普 {hrp_res['eq_full']['sharpe']:.2f}, 回撤 {hrp_res['eq_full']['mdd_pct']:.2f}%)",
            line={"width": 2.4, "color": "#38BDF8", "dash": "dash"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=all_dates,
            y=nav_df["combo_nav"].values,
            mode="lines",
            name=f"★ 五大商品 Sector-HRP 总组合 (Full {hrp_res['full']['cum_ret_pct']:+.1f}%, 夏普 {hrp_res['full']['sharpe']:.2f}, 回撤 {hrp_res['full']['mdd_pct']:.2f}%)",
            line={"width": 3.2, "color": "#00E676"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=all_dates,
            y=nav_df["combo_dd_pct"].values,
            mode="lines",
            fill="tozeroy",
            name=f"HRP 组合回撤 (最大回撤 {hrp_res['full']['mdd_pct']:.2f}%)",
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
            "text": "159915 规则型策略族 (已去掉背驰反转簇) 推广至五大商品期货 (ag/au/sc/lc/ps) 样本内重审后的迁移研究大屏",
            "font": {"size": 18, "color": "#FFFFFF"},
        },
    )
    out_html = CTA_OUTPUT_DIR / "transfer_159915_to_five_commodities_dashboard.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"\n>>> Plotly 大屏已保存至: {out_html}")

    # 生成内嵌 Generative UI 交互看板
    step = max(1, len(nav_df) // 180)
    sample_df = nav_df.iloc[::step].copy()
    if sample_df.iloc[-1]["date"] != nav_df.iloc[-1]["date"]:
        sample_df = pd.concat([sample_df, nav_df.iloc[[-1]]])

    svg_data = {
        "dates": sample_df["date"].tolist(),
        "ag": [round(float(v), 4) for v in sample_df["ag_nav"].tolist()],
        "au": [round(float(v), 4) for v in sample_df["au_nav"].tolist()],
        "sc": [round(float(v), 4) for v in sample_df["sc_nav"].tolist()],
        "lc": [round(float(v), 4) for v in sample_df["lc_nav"].tolist()],
        "ps": [round(float(v), 4) for v in sample_df["ps_nav"].tolist()],
        "eq": [round(float(v), 4) for v in sample_df["eq_nav"].tolist()],
        "combo": [round(float(v), 4) for v in sample_df["combo_nav"].tolist()],
        "dd": [round(float(v), 2) for v in sample_df["combo_dd_pct"].tolist()],
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
          <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">已去掉背驰反转簇 · 纯顺势+量能结构</span>
          <h2 class="text-xl font-bold tracking-tight text-[var(--foreground)]">AG沪银 · AU沪金 · SC原油 · LC碳酸锂 · PS多晶硅</h2>
        </div>
        <p class="text-xs text-[var(--muted-foreground)] mt-1">159915 规则型策略族 (剔除 B_背驰反转簇，保留 26 个顺势与结构单元) · Sector-HRP vs 在市等权(1/N_t)</p>
      </div>
      <div class="flex gap-2 text-right">
        <div class="px-3 py-1.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">
          <div class="text-xs text-[var(--muted-foreground)]">Sector-HRP总收益</div>
          <div class="text-base font-bold text-emerald-400">+{hrp_res["full"]["cum_ret_pct"]:.2f}% (Sh {hrp_res["full"]["sharpe"]:.2f})</div>
        </div>
        <div class="px-3 py-1.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">
          <div class="text-xs text-[var(--muted-foreground)]">在市等权(1/N)总收益</div>
          <div class="text-base font-bold text-sky-400">+{hrp_res["eq_full"]["cum_ret_pct"]:.2f}% (Sh {hrp_res["eq_full"]["sharpe"]:.2f})</div>
        </div>
        <div class="px-3 py-1.5 rounded-lg bg-[var(--background)] border border-[var(--border)]">
          <div class="text-xs text-[var(--muted-foreground)]">HRP最大回撤</div>
          <div class="text-base font-bold text-rose-400">-{hrp_res["full"]["mdd_pct"]:.2f}%</div>
        </div>
      </div>
    </div>

    <!-- Single Commodity Cards -->
    <div class="grid grid-cols-2 sm:grid-cols-5 gap-2.5 my-4">
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-purple-400">AG 沪银主力</div>
        <div class="text-base font-bold text-[var(--foreground)] mt-0.5">{asset_results["ag"]["full"]["cum_ret_pct"]:+.1f}%</div>
        <div class="text-xs text-[var(--muted-foreground)]">Sh {asset_results["ag"]["full"]["sharpe"]:.2f} | OOS {asset_results["ag"]["oos"]["cum_ret_pct"]:+.1f}%</div>
      </div>
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-amber-400">AU 沪金主力</div>
        <div class="text-base font-bold text-[var(--foreground)] mt-0.5">{asset_results["au"]["full"]["cum_ret_pct"]:+.1f}%</div>
        <div class="text-xs text-[var(--muted-foreground)]">Sh {asset_results["au"]["full"]["sharpe"]:.2f} | OOS {asset_results["au"]["oos"]["cum_ret_pct"]:+.1f}%</div>
      </div>
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-orange-400">SC 原油主力</div>
        <div class="text-base font-bold text-[var(--foreground)] mt-0.5">{asset_results["sc"]["full"]["cum_ret_pct"]:+.1f}%</div>
        <div class="text-xs text-[var(--muted-foreground)]">Sh {asset_results["sc"]["full"]["sharpe"]:.2f} | MDD {asset_results["sc"]["full"]["mdd_pct"]:.1f}%</div>
      </div>
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-cyan-400">LC 碳酸锂(23.07+)</div>
        <div class="text-base font-bold text-[var(--foreground)] mt-0.5">{asset_results["lc"]["full"]["cum_ret_pct"]:+.1f}%</div>
        <div class="text-xs text-[var(--muted-foreground)]">Sh {asset_results["lc"]["full"]["sharpe"]:.2f} | MDD {asset_results["lc"]["full"]["mdd_pct"]:.1f}%</div>
      </div>
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-green-400">PS 多晶硅(24.12+)</div>
        <div class="text-base font-bold text-[var(--foreground)] mt-0.5">{asset_results["ps"]["full"]["cum_ret_pct"]:+.1f}%</div>
        <div class="text-xs text-[var(--muted-foreground)]">Sh {asset_results["ps"]["full"]["sharpe"]:.2f} | MDD {asset_results["ps"]["full"]["mdd_pct"]:.1f}%</div>
      </div>
    </div>

    <!-- SVG Chart -->
    <div class="relative bg-[var(--background)] rounded-xl border border-[var(--border)] p-4 select-none">
      <div class="flex flex-wrap items-center justify-between mb-2 gap-2 text-xs">
        <div class="flex flex-wrap items-center gap-3">
          <span class="flex items-center gap-1"><span class="w-3 h-1 bg-emerald-400 rounded"></span><span class="text-emerald-400 font-bold">★ Sector-HRP组合 ({hrp_res["full"]["cum_ret_pct"]:+.1f}%)</span></span>
          <span class="flex items-center gap-1"><span class="w-3 h-0.5 bg-sky-400 border-b border-dashed"></span><span class="text-sky-400 font-semibold">◇ 在市等权1/N ({hrp_res["eq_full"]["cum_ret_pct"]:+.1f}%)</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-purple-400"></span><span class="text-purple-400">AG沪银</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-amber-400"></span><span class="text-amber-400">AU沪金</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-orange-400"></span><span class="text-orange-400">SC原油</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-cyan-400"></span><span class="text-cyan-400">LC碳酸锂</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-green-400"></span><span class="text-green-400">PS多晶硅</span></span>
        </div>
        <span class="px-2 py-0.5 rounded bg-amber-500/20 text-amber-400 font-mono">虚线右端为 2024+ OOS</span>
      </div>

      <div id="hover-hud" class="hidden absolute top-10 right-4 bg-slate-900/90 border border-slate-700/80 rounded-lg p-2.5 text-xs z-20 pointer-events-none shadow-xl backdrop-blur-sm">
        <div id="hud-date" class="font-mono text-slate-300 font-bold mb-1"></div>
        <div class="text-emerald-400 font-mono" id="hud-combo"></div>
        <div class="text-sky-400 font-mono" id="hud-eq"></div>
        <div class="text-purple-300 font-mono" id="hud-ag"></div>
        <div class="text-amber-300 font-mono" id="hud-au"></div>
        <div class="text-orange-300 font-mono" id="hud-sc"></div>
        <div class="text-cyan-300 font-mono" id="hud-lc"></div>
        <div class="text-green-300 font-mono" id="hud-ps"></div>
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

          <path id="ag-path" d="" fill="none" stroke="#AB47BC" stroke-width="1.5" opacity="0.85"/>
          <path id="au-path" d="" fill="none" stroke="#FFD54F" stroke-width="1.5" opacity="0.85"/>
          <path id="sc-path" d="" fill="none" stroke="#FF7043" stroke-width="1.5" opacity="0.85"/>
          <path id="lc-path" d="" fill="none" stroke="#26C6DA" stroke-width="1.5" opacity="0.85"/>
          <path id="ps-path" d="" fill="none" stroke="#66BB6A" stroke-width="1.5" opacity="0.85"/>

          <path id="eq-path" d="" fill="none" stroke="#38BDF8" stroke-width="2.0" stroke-dasharray="5,3" opacity="0.95"/>
          <path id="combo-fill" d="" fill="url(#combo-grad)" opacity="0.22"/>
          <path id="combo-path" d="" fill="none" stroke="#00E676" stroke-width="3.0"/>

          <line id="hover-line" x1="0" y1="10" x2="0" y2="220" stroke="#FFFFFF" stroke-width="1" stroke-dasharray="2,2" class="hidden"/>
          <circle id="hover-combo-dot" cx="0" cy="0" r="4" fill="#00E676" stroke="#FFFFFF" stroke-width="1.5" class="hidden"/>

          <defs>
            <linearGradient id="combo-grad" x1="0%" y1="0%" x2="0%" y2="100%">
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
    const allVals = [...data.ag, ...data.au, ...data.sc, ...data.lc, ...data.ps, ...data.eq, ...data.combo];
    const minNav = Math.min(...allVals) * 0.96;
    const maxNav = Math.max(...allVals) * 1.04;
    document.getElementById('label-nav-max').innerText = maxNav.toFixed(2);
    document.getElementById('label-nav-min').innerText = minNav.toFixed(2);

    const minDd = Math.min(...data.dd, -12.0);
    const xLeft = 50, xRight = 770, yTop = 20, yBottom = 220, ddTop = 8, ddBottom = 55;

    function getX(i) {{ return xLeft + (i / (n - 1)) * (xRight - xLeft); }}
    function getY(val) {{ return yBottom - ((val - minNav) / (maxNav - minNav)) * (yBottom - yTop); }}
    function getDdY(val) {{ return ddTop + (Math.abs(val) / Math.abs(minDd)) * (ddBottom - ddTop); }}

    function buildPath(arr) {{
      let d = `M ${{getX(0)}} ${{getY(arr[0])}}`;
      for (let i = 1; i < n; i++) d += ` L ${{getX(i)}} ${{getY(arr[i])}}`;
      return d;
    }}

    document.getElementById('ag-path').setAttribute('d', buildPath(data.ag));
    document.getElementById('au-path').setAttribute('d', buildPath(data.au));
    document.getElementById('sc-path').setAttribute('d', buildPath(data.sc));
    document.getElementById('lc-path').setAttribute('d', buildPath(data.lc));
    document.getElementById('ps-path').setAttribute('d', buildPath(data.ps));
    document.getElementById('eq-path').setAttribute('d', buildPath(data.eq));

    const comboD = buildPath(data.combo);
    const fillD = `M ${{getX(0)}} ${{yBottom}} L ` + comboD.slice(2) + ` L ${{getX(n - 1)}} ${{yBottom}} Z`;
    document.getElementById('combo-path').setAttribute('d', comboD);
    document.getElementById('combo-fill').setAttribute('d', fillD);

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
    const hoverDot = document.getElementById('hover-combo-dot');
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
      hoverDot.setAttribute('cy', getY(data.combo[idx]));
      hoverDot.classList.remove('hidden');
      hud.classList.remove('hidden');
      document.getElementById('hud-date').innerText = dates[idx];
      document.getElementById('hud-combo').innerText = `★ HRP组合: ${{data.combo[idx].toFixed(3)}} (${{((data.combo[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-eq').innerText = `◇ 在市等权: ${{data.eq[idx].toFixed(3)}} (${{((data.eq[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-ag').innerText = `AG沪银: ${{data.ag[idx].toFixed(3)}}`;
      document.getElementById('hud-au').innerText = `AU沪金: ${{data.au[idx].toFixed(3)}}`;
      document.getElementById('hud-sc').innerText = `SC原油: ${{data.sc[idx].toFixed(3)}}`;
      document.getElementById('hud-lc').innerText = `LC碳酸锂: ${{data.lc[idx].toFixed(3)}}`;
      document.getElementById('hud-ps').innerText = `PS多晶硅: ${{data.ps[idx].toFixed(3)}}`;
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
    artifact_html = ARTIFACT_DIR / "transfer_five_commodities_chart.html"
    artifact_html.write_text(widget_html, encoding="utf-8")
    print(f">>> Generative UI 交互式净值卡片已保存至: {artifact_html}")


if __name__ == "__main__":
    main()
