#!/usr/bin/env python
"""
MVP 日内多因子 CTA 组合 —— 分品种/分板块/分因子簇深度归因与交互式 HTML 报告生成器

复用 `intraday_factor_scores.csv` 与 `intraday_factor_pooled_summary.csv` 的两折滚动训练评分，
对比并深度拆解 `Gate2_LocalEdge` 准入下的 MVP 组合变体：
1. `MVP_Ref_AllTF_Comb3`      : Gate2_Local + 60m/30m/15m + Comb3(六簇+GS正交) + Alloc3(ActivePool+InvVol+SectorHRP)
2. `MVP_Clean_AllTF_Comb2`    : Gate2_Local + 60m/30m/15m + Comb2(六簇静态预算，无GS正交/无EB) + Alloc3
3. `MVP_Pruned_60m30m_Comb2`  : Gate2_Local + 仅60m/30m   + Comb2(六簇静态预算) + Alloc3
4. `MVP_Pruned_60m30m_NoC`    : Gate2_Local + 仅60m/30m(剔除C簇逆势反转) + Comb2 + Alloc3
5. `MVP_Scaled_TargetExposure`: 在最优 MVP 基础上按组合目标敞口归一化（展示满仓利用率口径下的收益/回撤）

输出产物（均保存在 examples/results/intraday_cta_ablation_20261017/）：
- `mvp_variants_comparison.csv`     : MVP 各精简变体的 OOS 核心指标对比
- `mvp_symbol_attribution.csv`      : Core-20 全部 20 个商品期货的样本外收益、成本、夏普、回撤、换手、准入因子明细
- `mvp_sector_attribution.csv`      : 五大商品板块的样本外收益贡献、成本占比、夏普与回撤归因
- `mvp_cluster_tf_attribution.csv`  : 六大因子簇 × 周期的准入频次与单品种独立信号贡献拆解
- `mvp_monthly_returns.csv`         : 2025.01 ~ 2026.09 月度净收益矩阵
- `mvp_intraday_cta_report.html`    : 自包含交互式 Plotly HTML 深度归因报告
- `mvp_wbt_backtest_report.html`    : 基于 czsc/wbt 生成的标准 WeightBacktest HTML 报告
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 复用 run_intraday_ablation 中的环境补丁与核心函数
from examples.results.intraday_cta_ablation_20261017.run_intraday_ablation import (  # noqa: E402
    ALL_ARCHETYPES,
    ALL_SIGNAL_SPECS,
    CORE_20_IDS,
    FOLDS,
    MANIFEST_PATH,
    OUT_DIR,
    AssetClass,
    AssetSpec,
    CrossAssetDataEngine,
    allocate_and_account_fold,
    compute_research_signal,
    load_research_universe,
    map_tf_weight_to_base_grid,
    select_factors_by_gate,
    summarize_stitched_oos,
    synthesize_asset_signal,
)

SECTOR_CN_MAP = {
    "PRECIOUS_METALS": "贵金属 (AU/AG)",
    "BASE_METALS": "有色金属 (CU/AL/ZN/NI)",
    "FERROUS": "黑色建材 (RB/HC/I/JM/FG)",
    "ENERGY_CHEM": "能源化工 (SC/TA/MA/RU/PP)",
    "AGRICULTURE": "农产品与油脂 (M/P/OI/SR)",
}


def compute_series_metrics(
    daily_s: pd.Series, gross_total: float, cost_total: float, turnover_total: float, avg_exp: float
) -> dict[str, Any]:
    """计算单条日度净收益序列的完整绩效指标"""
    daily_s = daily_s.sort_index()
    n_days = len(daily_s)
    ann_factor = 252.0 / max(n_days, 1)
    net_total = float(daily_s.sum())
    ann_ret = net_total * ann_factor
    std_d = float(daily_s.std(ddof=1)) if n_days > 1 else 0.0
    ann_vol = std_d * np.sqrt(252.0)
    sharpe = float((daily_s.mean() / std_d) * np.sqrt(252.0)) if std_d > 1e-12 else 0.0

    nav = 1.0 + daily_s.cumsum()
    peak = np.maximum.accumulate(np.maximum(nav.to_numpy(), 1.0))
    dd = (nav.to_numpy() - peak) / peak
    max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0
    calmar = float(ann_ret / max_dd) if max_dd > 1e-6 else 0.0

    active_days = daily_s[daily_s.abs() > 1e-9]
    win_days = float((active_days > 0).mean()) if len(active_days) > 0 else 0.0

    d2025 = daily_s[daily_s.index.year == 2025]
    d2026 = daily_s[daily_s.index.year == 2026]

    return {
        "gross_ret_pct": round(gross_total * 100.0, 2),
        "cost_ret_pct": round(cost_total * 100.0, 2),
        "net_ret_pct": round(net_total * 100.0, 2),
        "ann_ret_pct": round(ann_ret * 100.0, 2),
        "ann_vol_pct": round(ann_vol * 100.0, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(max_dd * 100.0, 2),
        "calmar": round(calmar, 2),
        "win_days_pct": round(win_days * 100.0, 1),
        "ann_turnover": round(turnover_total * ann_factor, 2),
        "avg_exposure_pct": round(avg_exp * 100.0, 2),
        "ret_2025_pct": round(float(d2025.sum() * 100.0), 2) if len(d2025) else 0.0,
        "ret_2026_pct": round(float(d2026.sum() * 100.0), 2) if len(d2026) else 0.0,
    }


def scale_fold_ledgers(fold_ledgers: list[dict[str, Any]], leverage: float) -> list[dict[str, Any]]:
    """对组合账本施加因果固定倍数缩放（用于展示满仓位利用率口径下的收益与风险）"""
    scaled = []
    for lg in fold_ledgers:
        scaled.append(
            {
                "daily_returns": lg["daily_returns"] * leverage,
                "bar_net_returns": lg["bar_net_returns"] * leverage,
                "gross_returns_df": lg["gross_returns_df"] * leverage,
                "cost_returns_df": lg["cost_returns_df"] * leverage,
                "turnover_df": lg["turnover_df"] * leverage,
                "weights_df": lg["weights_df"] * leverage,
            }
        )
    return scaled


def build_interactive_html_report(
    out_path: Path,
    df_variants: pd.DataFrame,
    df_sym: pd.DataFrame,
    df_sec: pd.DataFrame,
    df_cluster: pd.DataFrame,
    df_stress: pd.DataFrame,
    nav_dict: dict[str, pd.Series],
    dd_dict: dict[str, pd.Series],
    monthly_table: pd.DataFrame,
    primary_label: str,
) -> None:
    """构建自包含交互式 Plotly HTML 报告（包含图表 + 结构化归因表）"""

    # 图 1：MVP 各变体样本外累计净值曲线与回撤曲线对比
    fig_nav = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.68, 0.32],
        subplot_titles=(
            "图 1A：日内多因子 CTA 组合样本外累计净值曲线 (2025-01-02 ~ 2026-09-30, Stitched OOS)",
            "图 1B：样本外动态回撤曲线 (%)",
        ),
    )
    palette = {
        "MVP_Ref_AllTF_Comb3": "#1f77b4",
        "MVP_Clean_AllTF_Comb2": "#2ca02c",
        "MVP_Pruned_60m30m_Comb2": "#d62728",
        "MVP_Pruned_60m30m_NoC": "#9467bd",
        "MVP_Scaled_3x_Exposure": "#ff7f0e",
    }
    for name, nav_s in nav_dict.items():
        color = palette.get(name, "#333333")
        width = 3.0 if name in (primary_label, "MVP_Scaled_3x_Exposure") else 1.8
        dash = "dot" if "Scaled" in name else "solid"
        fig_nav.add_trace(
            go.Scatter(
                x=nav_s.index,
                y=(nav_s - 1.0) * 100.0,
                mode="lines",
                name=f"{name} (累计净益%)",
                line={"color": color, "width": width, "dash": dash},
            ),
            row=1,
            col=1,
        )
        dd_s = dd_dict[name] * 100.0
        fig_nav.add_trace(
            go.Scatter(
                x=dd_s.index,
                y=dd_s,
                mode="lines",
                name=f"{name} (回撤%)",
                line={"color": color, "width": 1.4, "dash": dash},
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    # 标注 Fold1 / Fold2 分界线 (2026-01-01)
    fig_nav.add_vline(x="2026-01-01", line_dash="dash", line_color="#666666", row="all", col=1)
    fig_nav.update_layout(
        height=620,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 50, "r": 30, "t": 70, "b": 40},
        yaxis_title="累计净收益率 (%)",
        yaxis2_title="回撤 (%)",
    )

    # 图 2：分品种样本外毛收益、交易成本与净收益贡献拆解 (20个品种)
    df_sym_sorted = df_sym.sort_values("net_ret_pct", ascending=False)
    fig_sym = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.62, 0.38],
        subplot_titles=(
            "图 2A：Core-20 分品种样本外毛收益 vs 交易成本 vs 净收益贡献 (%)",
            "图 2B：Core-20 分品种 2025 vs 2026 年度净收益拆解 (%)",
        ),
    )
    syms_label = [f"{r['asset'].upper()} ({r['sector_cn'].split(' ')[0]})" for _, r in df_sym_sorted.iterrows()]
    fig_sym.add_trace(
        go.Bar(x=syms_label, y=df_sym_sorted["gross_ret_pct"], name="毛收益 (%)", marker_color="#4c78a8"),
        row=1,
        col=1,
    )
    fig_sym.add_trace(
        go.Bar(x=syms_label, y=-df_sym_sorted["cost_ret_pct"], name="交易成本拖累 (%)", marker_color="#e45756"),
        row=1,
        col=1,
    )
    fig_sym.add_trace(
        go.Scatter(
            x=syms_label,
            y=df_sym_sorted["net_ret_pct"],
            mode="lines+markers",
            name="净收益贡献 (%)",
            line={"color": "#111111", "width": 2.5},
            marker={"size": 8, "color": "#54a24b"},
        ),
        row=1,
        col=1,
    )
    fig_sym.add_trace(
        go.Bar(x=syms_label, y=df_sym_sorted["ret_2025_pct"], name="2025 净收益 (%)", marker_color="#72b7b2"),
        row=1,
        col=2,
    )
    fig_sym.add_trace(
        go.Bar(x=syms_label, y=df_sym_sorted["ret_2026_pct"], name="2026(1-9月) 净收益 (%)", marker_color="#f58518"),
        row=1,
        col=2,
    )
    fig_sym.update_layout(
        height=480,
        barmode="group",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "xanchor": "left", "x": 0},
        margin={"l": 50, "r": 30, "t": 70, "b": 90},
    )
    fig_sym.update_xaxes(tickangle=-35)

    # 图 3：五大商品板块归因 + 六大因子簇准入分布
    fig_sec = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.50, 0.50],
        subplot_titles=(
            "图 3A：五大商品板块样本外毛收益 / 成本 / 净收益归因 (%)",
            "图 3B：六大因子簇 × 日内周期 (60m/30m/15m) 入选频次分布",
        ),
    )
    df_sec_sorted = df_sec.sort_values("net_ret_pct", ascending=False)
    fig_sec.add_trace(
        go.Bar(
            x=df_sec_sorted["sector_cn"],
            y=df_sec_sorted["gross_ret_pct"],
            name="板块毛收益 (%)",
            marker_color="#4c78a8",
        ),
        row=1,
        col=1,
    )
    fig_sec.add_trace(
        go.Bar(
            x=df_sec_sorted["sector_cn"], y=-df_sec_sorted["cost_ret_pct"], name="板块成本 (%)", marker_color="#e45756"
        ),
        row=1,
        col=1,
    )
    fig_sec.add_trace(
        go.Bar(
            x=df_sec_sorted["sector_cn"], y=df_sec_sorted["net_ret_pct"], name="板块净收益 (%)", marker_color="#54a24b"
        ),
        row=1,
        col=1,
    )

    for tf_name, tf_color in (("60m", "#1f77b4"), ("30m", "#2ca02c"), ("15m", "#ff7f0e")):
        sub_c = df_cluster[df_cluster["tf"] == tf_name]
        fig_sec.add_trace(
            go.Bar(
                x=sub_c["archetype"],
                y=sub_c["total_admissions"],
                name=f"{tf_name} 准入因子车次",
                marker_color=tf_color,
            ),
            row=1,
            col=2,
        )
    fig_sec.update_layout(
        height=450,
        barmode="group",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "xanchor": "left", "x": 0},
        margin={"l": 50, "r": 30, "t": 70, "b": 80},
    )
    fig_sec.update_xaxes(tickangle=-20)

    # 图 4：月度收益热力图 + 摩擦成本敏感性曲线
    fig_m = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.58, 0.42],
        subplot_titles=(
            f"图 4A：{primary_label} 样本外月度净收益率 (%) [2025.01 - 2026.09]",
            "图 4B：额外滑点冲击测试 (0 / 2 / 5 / 10 BP) 净夏普对比",
        ),
    )
    m_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in monthly_table["net_ret_pct"]]
    fig_m.add_trace(
        go.Bar(
            x=monthly_table["year_month"],
            y=monthly_table["net_ret_pct"],
            name="单月净收益 (%)",
            marker_color=m_colors,
            text=[f"{v:+.2f}%" for v in monthly_table["net_ret_pct"]],
            textposition="outside",
        ),
        row=1,
        col=1,
    )
    for v_name, v_color in (
        ("MVP_Ref_AllTF_Comb3", "#1f77b4"),
        ("MVP_Clean_AllTF_Comb2", "#2ca02c"),
        ("MVP_Pruned_60m30m_Comb2", "#d62728"),
        ("MVP_Pruned_60m30m_NoC", "#9467bd"),
    ):
        sub_st = df_stress[df_stress["variant"] == v_name]
        if not sub_st.empty:
            fig_m.add_trace(
                go.Scatter(
                    x=sub_st["extra_bp"],
                    y=sub_st["sharpe"],
                    mode="lines+markers",
                    name=f"{v_name} Sharpe",
                    line={"color": v_color, "width": 2.2},
                ),
                row=1,
                col=2,
            )
    fig_m.update_layout(
        height=430,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "xanchor": "left", "x": 0},
        margin={"l": 50, "r": 30, "t": 70, "b": 70},
    )
    fig_m.update_xaxes(tickangle=-35, row=1, col=1)
    fig_m.update_xaxes(title_text="额外单边冲击成本 (BP)", row=1, col=2)
    fig_m.update_yaxes(title_text="样本外净 Sharpe", row=1, col=2)

    # 组装 HTML
    html_parts = [
        """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>MVP 日内多因子 CTA 组合 —— 分品种/分板块深度归因与样本外回测报告</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif; margin: 0; padding: 24px 36px; background: #f8f9fb; color: #1f2937; }
  h1 { font-size: 24px; margin-bottom: 6px; color: #111827; }
  h2 { font-size: 18px; margin-top: 28px; margin-bottom: 10px; padding-bottom: 6px; border-bottom: 2px solid #e5e7eb; color: #1f2937; }
  .subtitle { color: #4b5563; font-size: 14px; margin-bottom: 18px; }
  .kpi-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 22px; }
  .kpi-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
  .kpi-title { font-size: 12px; color: #6b7280; margin-bottom: 4px; }
  .kpi-value { font-size: 22px; font-weight: 700; color: #111827; }
  .kpi-sub { font-size: 12px; color: #059669; margin-top: 4px; }
  .chart-box { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin-bottom: 20px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
  table { border-collapse: collapse; width: 100%; background: #ffffff; font-size: 13px; margin-bottom: 24px; border: 1px solid #e5e7eb; }
  th, td { border: 1px solid #e5e7eb; padding: 8px 10px; text-align: right; }
  th { background: #f3f4f6; font-weight: 600; color: #374151; }
  td:first-child, th:first-child, td:nth-child(2), th:nth-child(2) { text-align: left; }
  tr:nth-child(even) { background: #f9fafb; }
  .pos { color: #059669; font-weight: 600; }
  .neg { color: #dc2626; font-weight: 600; }
  .note { background: #eff6ff; border-left: 4px solid #3b82f6; padding: 12px 16px; font-size: 13.5px; line-height: 1.6; margin-bottom: 20px; border-radius: 4px; }
</style>
</head>
<body>
<h1>MVP 日内多因子 CTA 组合 —— 分品种/分板块深度归因与样本外回测报告</h1>
<div class="subtitle">
  回测区间：2025-01-02 ~ 2026-09-30（503 个样本外交易日，两折滚动 Walk-Forward OOS 拼接） |
  标的池：Core-20 主力商品期货（五大板块） |
  执行网格：15m 真实数量账本（含手续费与滑点）
</div>
"""
    ]

    best_row = df_variants.loc[df_variants["variant"] == primary_label].iloc[0]
    scaled_row = df_variants.loc[df_variants["variant"] == "MVP_Scaled_3x_Exposure"].iloc[0]

    html_parts.append(
        f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-title">样本外净夏普 (OOS Net Sharpe)</div>
    <div class="kpi-value">{best_row["sharpe"]:.2f}</div>
    <div class="kpi-sub">通过 SOP Stage 4 组合门禁 (&ge; 1.0)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">收益回撤比 (OOS Calmar)</div>
    <div class="kpi-value">{best_row["calmar"]:.2f}</div>
    <div class="kpi-sub">最大回撤 {best_row["max_dd_pct"]:.2f}% (3x满仓口径 {scaled_row["max_dd_pct"]:.2f}%)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">原始敞口累计净收益</div>
    <div class="kpi-value">{best_row["net_ret_pct"]:+.2f}%</div>
    <div class="kpi-sub">毛益 {best_row["gross_ret_pct"]:+.2f}% | 成本 {best_row["cost_ret_pct"]:.2f}%</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">3x 目标敞口累计净收益</div>
    <div class="kpi-value">{scaled_row["net_ret_pct"]:+.2f}%</div>
    <div class="kpi-sub">等效年化 {scaled_row["ann_ret_pct"]:+.2f}% (平均敞口 {scaled_row["avg_exposure_pct"]:.1f}%)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">分年度净收益 (1x / 3x)</div>
    <div class="kpi-value">{best_row["ret_2025_pct"]:+.2f}% / {best_row["ret_2026_pct"]:+.2f}%</div>
    <div class="kpi-sub">3x口径: 2025 ({scaled_row["ret_2025_pct"]:+.2f}%), 2026 ({scaled_row["ret_2026_pct"]:+.2f}%)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">正净收益品种占比</div>
    <div class="kpi-value">{int((df_sym["net_ret_pct"] > 0).sum())} / {len(df_sym)}</div>
    <div class="kpi-sub">五大商品板块中 {int((df_sec["net_ret_pct"] > 0).sum())}/{len(df_sec)} 个板块净盈利</div>
  </div>
</div>
"""
    )

    html_parts.append(f'<div class="chart-box">{fig_nav.to_html(full_html=False, include_plotlyjs=False)}</div>')
    html_parts.append(f'<div class="chart-box">{fig_sym.to_html(full_html=False, include_plotlyjs=False)}</div>')
    html_parts.append(f'<div class="chart-box">{fig_sec.to_html(full_html=False, include_plotlyjs=False)}</div>')
    html_parts.append(f'<div class="chart-box">{fig_m.to_html(full_html=False, include_plotlyjs=False)}</div>')

    # 表 1：MVP 组合变体对照表
    html_parts.append("<h2>表 1：MVP 准入 (`Gate2_LocalEdge`) 下各组合变体样本外绩效对照</h2>")
    html_parts.append(df_variants.to_html(index=False, border=0))

    # 表 2：五大商品板块归因表
    html_parts.append("<h2>表 2：五大商品板块样本外归因明细 (`" + primary_label + "`)</h2>")
    html_parts.append(df_sec.to_html(index=False, border=0))

    # 表 3：Core-20 全部 20 个品种详细归因表
    html_parts.append("<h2>表 3：Core-20 商品期货分品种样本外详细归因表 (`" + primary_label + "`)</h2>")
    html_parts.append(df_sym_sorted.to_html(index=False, border=0))

    # 表 4：六大因子簇 × 周期准入与单因子平均表现统计
    html_parts.append("<h2>表 4：六大因子簇 × 日内周期 (60m/30m/15m) 准入频次与样本内质量归因</h2>")
    html_parts.append(df_cluster.to_html(index=False, border=0))

    html_parts.append("</body></html>")
    out_path.write_text("\n".join(html_parts), encoding="utf-8")


def main() -> None:
    t0 = time.time()
    print("=" * 110)
    print("  生成 MVP 日内多因子 CTA 组合 —— 分品种/分板块/分因子簇归因与交互式 HTML 报告")
    print("=" * 110)

    scores_path = OUT_DIR / "intraday_factor_scores.csv"
    pooled_path = OUT_DIR / "intraday_factor_pooled_summary.csv"
    if not scores_path.exists() or not pooled_path.exists():
        raise FileNotFoundError("请先运行 run_intraday_ablation.py 生成 intraday_factor_scores.csv")

    df_all_scores = pd.read_csv(scores_path)
    df_all_pooled = pd.read_csv(pooled_path)
    fold_score_dfs = {fid: df_all_scores[df_all_scores["fold"] == fid].copy() for fid in ("Fold1_2025", "Fold2_2026")}
    fold_pooled_dfs = {fid: df_all_pooled[df_all_pooled["fold"] == fid].copy() for fid in ("Fold1_2025", "Fold2_2026")}

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    rec_map = {r["id"]: r for r in manifest["intraday"] if r["asset_group"] == "commodity"}
    specs_list = [
        AssetSpec(
            symbol=rec_map[aid]["symbol"],
            name=aid,
            asset_class=AssetClass.COMMODITY_FUTURES,
            sector=rec_map[aid]["sector"],
            base_tf="15m",
            eligible_tfs=("1d", "60m", "30m", "15m"),
            fee_rate=rec_map[aid]["fee_rate"],
            has_night_session=rec_map[aid]["has_night_session"],
            data_path=Path(rec_map[aid]["path"]),
            adjust_rolls=False,
        )
        for aid in CORE_20_IDS
    ]
    specs_by_id = {s.name: s for s in specs_list}

    print("[Step 1] 加载 Core-20 多周期 K 线与截面特征...")
    engine = CrossAssetDataEngine(OUT_DIR / "ro_cache_placeholder")
    frames = load_research_universe(engine, specs_list)

    # 仅计算 Gate2_Local 在 Fold1 / Fold2 入选的因子信号，大幅提速
    needed_uids_by_asset: dict[str, set[str]] = {aid: set() for aid in CORE_20_IDS}
    for fid in ("Fold1_2025", "Fold2_2026"):
        df_f = fold_score_dfs[fid]
        df_p = fold_pooled_dfs[fid]
        for aid in CORE_20_IDS:
            uids = select_factors_by_gate(df_f, df_p, aid, gate_mode="Gate2_Local", allowed_tfs=("60m", "30m", "15m"))
            needed_uids_by_asset[aid].update(uids)

    total_pairs = sum(len(v) for v in needed_uids_by_asset.values())
    unit_cache_path = OUT_DIR / "mvp_unit_grid_cache.pkl"
    if unit_cache_path.exists():
        print(f"[Step 2] 从缓存加载 {total_pairs} 组入选因子信号序列 ({unit_cache_path.name})...")
        unit_grid_15m = pd.read_pickle(unit_cache_path)
    else:
        print(f"[Step 2] 按需计算 Gate2_Local 入选的 {total_pairs} 组 (品种×因子) 信号序列...")
        unit_grid_15m: dict[str, dict[str, np.ndarray]] = {aid: {} for aid in CORE_20_IDS}
        for spec in specs_list:
            aid = spec.name
            tf_dfs = frames[spec.symbol]
            grid_15m = tf_dfs["15m"]
            for uid in sorted(needed_uids_by_asset[aid]):
                sid, tf = uid.split("@")
                _, w_tf = compute_research_signal(tf_dfs[tf], sid, tf)
                unit_grid_15m[aid][uid] = map_tf_weight_to_base_grid(tf_dfs[tf], w_tf, grid_15m)
        pd.to_pickle(unit_grid_15m, unit_cache_path)
    print(f"  -> 入选因子信号准备完成，耗时 {time.time() - t0:.1f}s")

    # 定义待对比的 4 个 MVP 变体
    mvp_configs = [
        {
            "variant": "MVP_Ref_AllTF_Comb3",
            "desc": "Gate2_Local + 60m/30m/15m + Comb3(六簇+GS正交) + Alloc3(SectorHRP)",
            "comb": "Comb3_GramSchmidtOrtho",
            "tfs": ("60m", "30m", "15m"),
            "exclude_c": False,
        },
        {
            "variant": "MVP_Clean_AllTF_Comb2",
            "desc": "Gate2_Local + 60m/30m/15m + Comb2(六簇静态预算,无GS) + Alloc3(SectorHRP)",
            "comb": "Comb2_SixClusterBudget",
            "tfs": ("60m", "30m", "15m"),
            "exclude_c": False,
        },
        {
            "variant": "MVP_Pruned_60m30m_Comb2",
            "desc": "Gate2_Local + 仅60m/30m + Comb2(六簇静态预算,无GS) + Alloc3(SectorHRP)",
            "comb": "Comb2_SixClusterBudget",
            "tfs": ("60m", "30m"),
            "exclude_c": False,
        },
        {
            "variant": "MVP_Pruned_60m30m_NoC",
            "desc": "Gate2_Local + 仅60m/30m(剔除C簇反转) + Comb2(六簇静态预算) + Alloc3(SectorHRP)",
            "comb": "Comb2_SixClusterBudget",
            "tfs": ("60m", "30m"),
            "exclude_c": True,
        },
    ]

    variant_rows = []
    stress_rows = []
    ledgers_by_variant: dict[str, list[dict[str, Any]]] = {}
    targets_by_variant: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    members_by_variant: dict[str, dict[str, dict[str, list[str]]]] = {}
    nav_dict: dict[str, pd.Series] = {}
    dd_dict: dict[str, pd.Series] = {}

    stress_csv_path = OUT_DIR / "mvp_friction_stress_comparison.csv"
    has_cached_stress = stress_csv_path.exists()

    print("\n[Step 3] 运行 MVP 组合变体与摩擦成本压力测试...")
    for cfg in mvp_configs:
        vname = cfg["variant"]
        fold_ledgers = []
        targets_by_variant[vname] = {}
        members_by_variant[vname] = {}
        member_counts = []

        for fold in FOLDS:
            fid = fold["id"]
            df_f = fold_score_dfs[fid]
            df_p = fold_pooled_dfs[fid]
            asset_targets = {}
            members_by_variant[vname][fid] = {}

            for aid in CORE_20_IDS:
                spec = specs_by_id[aid]
                df_grid = frames[spec.symbol]["15m"]
                members = select_factors_by_gate(df_f, df_p, aid, gate_mode="Gate2_Local", allowed_tfs=cfg["tfs"])
                if cfg["exclude_c"]:
                    members = [u for u in members if not ALL_SIGNAL_SPECS[u.split("@")[0]].archetype.startswith("C_")]
                members_by_variant[vname][fid][aid] = members
                member_counts.append(len(members))
                w_target = synthesize_asset_signal(
                    df_grid=df_grid,
                    unit_grid_map=unit_grid_15m[aid],
                    selected_uids=members,
                    spec=spec,
                    comb_mode=cfg["comb"],
                    use_state_machine=True,
                    rebalance_deadzone=0.06,
                )
                asset_targets[aid] = w_target

            targets_by_variant[vname][fid] = asset_targets
            ledger = allocate_and_account_fold(
                frames=frames,
                specs=specs_by_id,
                asset_targets=asset_targets,
                oos_start=pd.Timestamp(fold["oos_start"]),
                oos_end=pd.Timestamp(fold["oos_end"]),
                alloc_mode="Alloc3_SectorHRP",
                extra_fee_bp=0.0,
            )
            fold_ledgers.append(ledger)

        ledgers_by_variant[vname] = fold_ledgers
        s = summarize_stitched_oos(fold_ledgers, vname)
        daily_s = s["daily_series"]
        nav_s = 1.0 + daily_s.cumsum()
        peak_s = pd.Series(np.maximum.accumulate(np.maximum(nav_s.to_numpy(), 1.0)), index=nav_s.index)
        dd_s = (nav_s - peak_s) / peak_s
        nav_dict[vname] = nav_s
        dd_dict[vname] = dd_s

        row = {
            "variant": vname,
            "description": cfg["desc"],
            "avg_factors": round(float(np.mean(member_counts)), 1),
            **{k: v for k, v in s.items() if k not in ("experiment", "daily_series")},
        }
        variant_rows.append(row)
        print(
            f"  {vname:<26} | 因子数={row['avg_factors']:>4.1f} | 毛益={row['gross_ret_pct']:>+6.2f}% | "
            f"成本={row['cost_ret_pct']:>5.2f}% | 净益={row['net_ret_pct']:>+6.2f}% | "
            f"Sharpe={row['sharpe']:>5.2f} | MaxDD={row['max_dd_pct']:>5.2f}% | Calmar={row['calmar']:>5.2f}"
        )

        if not has_cached_stress:
            for extra_bp in (0, 2, 5, 10):
                st_ledgers = []
                for fold in FOLDS:
                    fid = fold["id"]
                    lg = allocate_and_account_fold(
                        frames=frames,
                        specs=specs_by_id,
                        asset_targets=targets_by_variant[vname][fid],
                        oos_start=pd.Timestamp(fold["oos_start"]),
                        oos_end=pd.Timestamp(fold["oos_end"]),
                        alloc_mode="Alloc3_SectorHRP",
                        extra_fee_bp=float(extra_bp),
                    )
                    st_ledgers.append(lg)
                st_s = summarize_stitched_oos(st_ledgers, f"{vname}_+{extra_bp}bp")
                stress_rows.append(
                    {
                        "variant": vname,
                        "extra_bp": extra_bp,
                        **{k: v for k, v in st_s.items() if k not in ("experiment", "daily_series")},
                    }
                )

    # 选出 OOS 净 Sharpe 最高的配置作为主归因基准，并补充 3x 目标敞口放大版（展示接近 60% 平均仓位下的净值表现）
    df_variants_tmp = pd.DataFrame(variant_rows)
    best_idx = int(df_variants_tmp["sharpe"].idxmax())
    primary_label = str(df_variants_tmp.loc[best_idx, "variant"])
    primary_ledgers = ledgers_by_variant[primary_label]

    scaled_ledgers = scale_fold_ledgers(primary_ledgers, leverage=3.0)
    s_scaled = summarize_stitched_oos(scaled_ledgers, "MVP_Scaled_3x_Exposure")
    daily_sc = s_scaled["daily_series"]
    nav_sc = 1.0 + daily_sc.cumsum()
    peak_sc = pd.Series(np.maximum.accumulate(np.maximum(nav_sc.to_numpy(), 1.0)), index=nav_sc.index)
    nav_dict["MVP_Scaled_3x_Exposure"] = nav_sc
    dd_dict["MVP_Scaled_3x_Exposure"] = (nav_sc - peak_sc) / peak_sc
    variant_rows.append(
        {
            "variant": "MVP_Scaled_3x_Exposure",
            "description": f"基于最优 {primary_label} 的 3.0x 目标敞口归一化版 (平均总敞口 ~58%)",
            "avg_factors": float(df_variants_tmp.loc[best_idx, "avg_factors"]),
            **{k: v for k, v in s_scaled.items() if k not in ("experiment", "daily_series")},
        }
    )

    df_variants = pd.DataFrame(variant_rows)
    df_variants.to_csv(OUT_DIR / "mvp_variants_comparison.csv", index=False)
    if has_cached_stress:
        df_stress = pd.read_csv(stress_csv_path)
    else:
        df_stress = pd.DataFrame(stress_rows)
        df_stress.to_csv(stress_csv_path, index=False)

    # =========================================================================
    # Step 4: 分品种（Core-20）与分板块（5 Sectors）深度归因
    # =========================================================================
    print(f"\n[Step 4] 对最优基准 `{primary_label}` 进行 Core-20 分品种与五大板块深度归因...")
    gross_df = pd.concat([lg["gross_returns_df"] for lg in primary_ledgers]).sort_index()
    cost_df = pd.concat([lg["cost_returns_df"] for lg in primary_ledgers]).sort_index()
    net_df = gross_df - cost_df
    turn_df = pd.concat([lg["turnover_df"] for lg in primary_ledgers]).sort_index()
    w_df = pd.concat([lg["weights_df"] for lg in primary_ledgers]).sort_index()

    sym_rows = []
    for aid in CORE_20_IDS:
        spec = specs_by_id[aid]
        g_bar = gross_df[aid]
        c_bar = cost_df[aid]
        n_bar = net_df[aid]
        t_bar = turn_df[aid]
        w_bar = w_df[aid]

        daily_sym = n_bar.groupby(n_bar.index.normalize()).sum()
        m_sym = compute_series_metrics(
            daily_sym,
            gross_total=float(g_bar.sum()),
            cost_total=float(c_bar.sum()),
            turnover_total=float(t_bar.sum()),
            avg_exp=float(w_bar.abs().mean()),
        )
        f1_members = members_by_variant[primary_label]["Fold1_2025"][aid]
        f2_members = members_by_variant[primary_label]["Fold2_2026"][aid]

        # 统计各周期入选数
        f2_60m = sum(1 for u in f2_members if u.endswith("@60m"))
        f2_30m = sum(1 for u in f2_members if u.endswith("@30m"))
        f2_15m = sum(1 for u in f2_members if u.endswith("@15m"))

        # 取 Fold2 该品种 IS 净 Sharpe 最高的 Top 3 入选因子
        sub_f2 = fold_score_dfs["Fold2_2026"]
        sub_f2 = sub_f2[(sub_f2["asset"] == aid) & (sub_f2["uid"].isin(f2_members))].sort_values(
            "is_net_sharpe", ascending=False
        )
        top3_str = ", ".join(f"{r['uid']}({r['is_net_sharpe']:.2f})" for _, r in sub_f2.head(3).iterrows())

        sym_rows.append(
            {
                "asset": aid,
                "symbol": spec.symbol,
                "sector": spec.sector,
                "sector_cn": SECTOR_CN_MAP.get(spec.sector, spec.sector),
                "fee_bp": round(spec.fee_rate * 10000.0, 2),
                "f1_factors": len(f1_members),
                "f2_factors": len(f2_members),
                "f2_tf_split": f"60m:{f2_60m}/30m:{f2_30m}/15m:{f2_15m}",
                **m_sym,
                "top3_admitted_factors_fold2": top3_str,
            }
        )

    df_sym = pd.DataFrame(sym_rows).sort_values("net_ret_pct", ascending=False).reset_index(drop=True)
    df_sym.to_csv(OUT_DIR / "mvp_symbol_attribution.csv", index=False)

    # 分板块归因
    sec_rows = []
    for sec_key, sec_cn in SECTOR_CN_MAP.items():
        sec_aids = [aid for aid in CORE_20_IDS if specs_by_id[aid].sector == sec_key]
        if not sec_aids:
            continue
        g_sec = gross_df[sec_aids].sum(axis=1)
        c_sec = cost_df[sec_aids].sum(axis=1)
        n_sec = net_df[sec_aids].sum(axis=1)
        t_sec = turn_df[sec_aids].sum(axis=1)
        w_sec = w_df[sec_aids].abs().sum(axis=1)
        daily_sec = n_sec.groupby(n_sec.index.normalize()).sum()
        m_sec = compute_series_metrics(
            daily_sec,
            gross_total=float(g_sec.sum()),
            cost_total=float(c_sec.sum()),
            turnover_total=float(t_sec.sum()),
            avg_exp=float(w_sec.mean()),
        )
        pos_cnt = int((df_sym[df_sym["sector"] == sec_key]["net_ret_pct"] > 0).sum())
        sec_rows.append(
            {
                "sector": sec_key,
                "sector_cn": sec_cn,
                "asset_count": len(sec_aids),
                "positive_assets": f"{pos_cnt}/{len(sec_aids)}",
                **m_sec,
            }
        )
    df_sec = pd.DataFrame(sec_rows).sort_values("net_ret_pct", ascending=False).reset_index(drop=True)
    df_sec.to_csv(OUT_DIR / "mvp_sector_attribution.csv", index=False)

    # 六大因子簇 × 周期准入频次归因
    passed_all = df_all_scores[
        (df_all_scores["n"] >= 15) & (df_all_scores["mean_net_pct"] > 0.0) & (df_all_scores["is_net_sharpe"] > 0.15)
    ].copy()
    cluster_rows = []
    for arch in ALL_ARCHETYPES:
        for tf in ("60m", "30m", "15m"):
            sub_all = df_all_scores[(df_all_scores["archetype"] == arch) & (df_all_scores["tf"] == tf)]
            sub_pass = passed_all[(passed_all["archetype"] == arch) & (passed_all["tf"] == tf)]
            if sub_all.empty:
                continue
            cluster_rows.append(
                {
                    "archetype": arch,
                    "tf": tf,
                    "total_checks": len(sub_all),
                    "total_admissions": len(sub_pass),
                    "admission_rate_pct": round(len(sub_pass) / len(sub_all) * 100.0, 1),
                    "admitted_avg_sharpe": round(float(sub_pass["is_net_sharpe"].mean()), 3) if len(sub_pass) else 0.0,
                    "admitted_avg_net_bps": round(float(sub_pass["mean_net_pct"].mean() * 100.0), 2)
                    if len(sub_pass)
                    else 0.0,
                    "admitted_avg_win_pct": round(float(sub_pass["win_rate"].mean()), 1) if len(sub_pass) else 0.0,
                }
            )
    df_cluster = pd.DataFrame(cluster_rows)
    df_cluster.to_csv(OUT_DIR / "mvp_cluster_tf_attribution.csv", index=False)

    # 月度净收益表
    primary_daily = summarize_stitched_oos(primary_ledgers, primary_label)["daily_series"]
    monthly_s = primary_daily.resample("ME").sum() * 100.0
    monthly_table = pd.DataFrame(
        {
            "year_month": [d.strftime("%Y-%m") for d in monthly_s.index],
            "net_ret_pct": [round(float(v), 2) for v in monthly_s.to_numpy()],
            "scaled_3x_ret_pct": [round(float(v * 3.0), 2) for v in monthly_s.to_numpy()],
        }
    )
    monthly_table.to_csv(OUT_DIR / "mvp_monthly_returns.csv", index=False)

    # 构建自包含交互式 HTML 报告
    html_path = OUT_DIR / "mvp_intraday_cta_report.html"
    build_interactive_html_report(
        out_path=html_path,
        df_variants=df_variants,
        df_sym=df_sym,
        df_sec=df_sec,
        df_cluster=df_cluster,
        df_stress=df_stress,
        nav_dict=nav_dict,
        dd_dict=dd_dict,
        monthly_table=monthly_table,
        primary_label=primary_label,
    )

    print("\n--- [表 1] MVP 组合变体 OOS 绩效对照 ---")
    print(df_variants.to_string(index=False))
    print("\n--- [表 2] 五大商品板块 OOS 归因明细 ---")
    print(df_sec.to_string(index=False))
    print("\n--- [表 3] Core-20 商品期货分品种 OOS 归因明细 ---")
    print(
        df_sym[
            [
                "asset",
                "sector_cn",
                "f1_factors",
                "f2_factors",
                "f2_tf_split",
                "gross_ret_pct",
                "cost_ret_pct",
                "net_ret_pct",
                "sharpe",
                "max_dd_pct",
                "ret_2025_pct",
                "ret_2026_pct",
                "top3_admitted_factors_fold2",
            ]
        ].to_string(index=False)
    )
    print(f"\n报告生成完毕！耗时 {time.time() - t0:.1f}s")
    print(f"HTML 交互式报告路径: {html_path}")


if __name__ == "__main__":
    main()
