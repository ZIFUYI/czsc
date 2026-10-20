#!/usr/bin/env python
"""
九大品种跨资产全天候 CTA 实盘信号监控、簇级归因与目标仓位输出器 (Live Target Position & Attribution Monitor)

核心功能：
1. 实时计算 9 大品种 (159915 / 510500 / 000852 / 588000 + AG / AU / SC / LC / PS) 最新 K 线截面上的：
   - 因果主场状态 (Home-Regime: UP_TREND / DOWN_TREND / RANGE)
   - 各战略簇实时信号强度 (A_顺势突破簇 / B_背驰反转簇[商品已剔除] / C_震荡回归簇)
   - 单品种合成目标仓位 w_synth ∈ [-1.0, +1.0] 与近 20 日策略年化波动率
2. 输出两种组合配权方案下的最新目标资金配置：
   - 方案 A：在市动态等权 (1/N_t = 11.11% / 品种) × 单品种信号 = 组合净目标敞口
   - 方案 B：跨板块分层风险平价 (Sector-HRP) × 目标波动率缩放 × 单品种信号 = 组合净目标敞口
3. 计算并输出 9×9 跨品种策略日收益相关性矩阵 (Pairwise Correlation Matrix)，并生成 Generative UI 监控大屏。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.alpha_mining_agent.forensic_pipeline import IS_END_DT  # noqa: E402
from examples.alpha_mining_agent.hypothesis_bank import ALL_SIGNAL_SPECS  # noqa: E402
from examples.multi_factor_cta.data_engine import (  # noqa: E402
    AssetClass,
    CrossAssetDataEngine,
)
from examples.multi_factor_cta.run_framework_demo import (  # noqa: E402
    load_candidate_alloc_map,
    load_research_universe,
    synthesize_admitted_asset,
)
from examples.multi_factor_cta.synthesis_engine import (  # noqa: E402
    CrossAssetHRPSynthesizer,
    IntraAssetThreePillarSynthesizer,
    build_asset_execution_inputs,
    compute_continuous_cluster_risk_shares,
    evaluate_single_asset_backtest,
)
from examples.multi_factor_cta.test_all_weather_9assets_portfolio import (  # noqa: E402
    ARTIFACT_DIR,
    CTA_OUTPUT_DIR,
    WORKSPACE_DIR,
    build_nine_assets_universe,
)

REGIME_NAMES = {
    1: "UP_TREND (多头趋势)",
    -1: "DOWN_TREND (空头趋势)",
    0: "RANGE (震荡盘整)",
}


def compute_latest_pillar_breakdown(
    df_grid: pd.DataFrame,
    w_on_grid: dict[str, np.ndarray],
    active_alloc: dict[str, float],
    intra_synth: IntraAssetThreePillarSynthesizer,
    is_commodity: bool,
) -> dict[str, float | str]:
    """提取最新一根 K 线上的 A / B / C 各簇原始合成信号与因果趋势状态"""
    n = len(df_grid)
    tf_weights = intra_synth.TF_WEIGHTS
    a_last = 0.0
    b_last = 0.0
    c_last = 0.0

    for tf, tw in tf_weights.items():
        wa = 0.0
        wb = 0.0
        wc = 0.0
        for uid, coeff in active_alloc.items():
            if uid not in w_on_grid:
                continue
            sid, u_tf = uid.split("@")
            if u_tf != tf:
                continue
            spec = ALL_SIGNAL_SPECS.get(sid)
            arch = spec.archetype if spec is not None else "A_顺势突破簇"
            val = float(w_on_grid[uid][-1]) * coeff
            if arch == "A_顺势突破簇":
                wa += val
            elif arch == "B_背驰反转簇":
                wb += val
            else:
                wc += val
        a_last += tw * float(np.clip(wa, -1.0, 1.0))
        b_last += tw * float(np.clip(wb, -1.0, 1.0))
        c_last += tw * float(np.clip(wc, -1.0, 1.0))

    regimes = intra_synth._get_causal_regimes(df_grid)
    last_regime = int(regimes[-1]) if n > 0 else 0
    return {
        "regime_code": last_regime,
        "regime_label": REGIME_NAMES.get(last_regime, "RANGE"),
        "A_signal": round(float(np.clip(a_last, -1.0, 1.0)), 3),
        "B_signal": round(float(np.clip(b_last, -1.0, 1.0)), 3),
        "C_signal": round(float(np.clip(c_last, -1.0, 1.0)), 3),
    }


def main() -> None:
    CTA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 135)
    print("  【九大品种跨资产全天候 CTA · 实时信号监控、战略簇归因、9×9 相关性矩阵与目标仓位输出表】")
    print("=" * 135)

    data_engine = CrossAssetDataEngine(cache_dir=WORKSPACE_DIR)
    intra_synth = IntraAssetThreePillarSynthesizer()
    hrp_synth = CrossAssetHRPSynthesizer(target_portfolio_vol=0.14)

    champ_alloc: dict[str, float] = load_candidate_alloc_map(WORKSPACE_DIR / "champion_portfolio.json")

    universe = build_nine_assets_universe()
    pool_frames = load_research_universe(data_engine, [spec for _, spec in universe])
    asset_daily_returns: dict[str, pd.Series] = {}
    execution_returns, execution_positions, execution_strengths, execution_fees = {}, {}, {}, {}
    asset_sectors: dict[str, str] = {}
    live_snapshots: dict[str, dict] = {}

    for short_id, spec in universe:
        is_comm = spec.asset_class == AssetClass.COMMODITY_FUTURES
        tf_dfs = pool_frames[spec.symbol]
        df_grid = tf_dfs[spec.base_tf]
        w_synth, w_on_grid, active_alloc, audits = synthesize_admitted_asset(
            spec,
            pool_frames,
            sorted(champ_alloc),
            intra_synth,
            enable_hysteresis_runner=is_comm,
        )
        # A prospective next-open order snapshot uses information at the latest
        # close; its allocation ledger is not a realized execution backtest.
        execution_returns[short_id], execution_positions[short_id] = build_asset_execution_inputs(
            df_grid, w_synth, execution_clock="close", liquidate_at_end=False
        )
        execution_fees[short_id] = spec.fee_rate
        if "cs_strength" in df_grid:
            execution_strengths[short_id] = pd.Series(
                df_grid["cs_strength"].to_numpy(), index=execution_returns[short_id].index
            )
        ev = evaluate_single_asset_backtest(df_grid, w_synth, spec.symbol, fee_rate=spec.fee_rate)
        asset_daily_returns[short_id] = ev["daily_return"]
        asset_sectors[short_id] = spec.sector

        pillar_info = compute_latest_pillar_breakdown(df_grid, w_on_grid, active_alloc, intra_synth, is_comm)
        recent_20d_ret = float(ev["daily_return"].tail(20).sum() * 100.0)
        recent_60d_vol = float(ev["daily_return"].tail(60).std() * np.sqrt(252.0) * 100.0)

        live_snapshots[short_id] = {
            "short_id": short_id.upper(),
            "name": spec.name,
            "sector": spec.sector,
            "is_commodity": is_comm,
            "latest_dt": str(df_grid["dt"].iloc[-1]),
            "latest_price": round(
                float(df_grid["raw_close"].iloc[-1] if "raw_close" in df_grid.columns else df_grid["close"].iloc[-1]),
                3,
            ),
            "regime_label": pillar_info["regime_label"],
            "A_signal": pillar_info["A_signal"],
            "B_signal": pillar_info["B_signal"],
            "C_signal": pillar_info["C_signal"],
            "w_signal": round(float(w_synth[-1]), 3),
            "recent_20d_ret_pct": round(recent_20d_ret, 2),
            "vol_60d_pct": round(recent_60d_vol, 2),
            "full_ret_pct": ev["full"]["cum_ret_pct"],
            "full_sharpe": ev["full"]["sharpe"],
            "oos_ret_pct": ev["oos"]["cum_ret_pct"],
            "oos_sharpe": ev["oos"]["sharpe"],
        }

    sectors_map: dict[str, list[str]] = {}
    for aid, sec in asset_sectors.items():
        sectors_map.setdefault(sec, []).append(aid)
    daily_price_ret_df = (
        pd.DataFrame(execution_returns)
        .groupby(lambda ts: pd.Timestamp(ts).normalize())
        .sum()
        .loc[: pd.Timestamp(IS_END_DT).normalize()]
    )
    daily_strat_ret_df = pd.DataFrame(asset_daily_returns).sort_index().loc[: pd.Timestamp(IS_END_DT).normalize()]
    cluster_shares = compute_continuous_cluster_risk_shares(
        price_corr_df=daily_price_ret_df.corr().abs().fillna(0.0),
        strategy_corr_df=daily_strat_ret_df.corr().abs().fillna(0.0),
        sectors=sectors_map,
    )

    hrp_res = hrp_synth.build_multi_asset_portfolio(
        execution_returns,
        asset_sectors,
        asset_positions_df=execution_positions,
        asset_cs_strength_df=execution_strengths,
        asset_fee_rates=execution_fees,
        returns_kind="price",
        positions_timing="close",
        periods_per_year=252 * 48,
        asset_prior_multipliers=cluster_shares,
    )
    common_dates = pd.DataFrame(execution_returns).dropna().index
    ref_dt = common_dates[-1] if len(common_dates) > 0 else hrp_res["equal_weights_df"].index[-1]
    eq_w_last = hrp_res["equal_weights_df"].loc[ref_dt]
    hrp_unscaled_last = hrp_res["unscaled_hrp_weights_df"].loc[ref_dt]
    hrp_eff_last = hrp_res["asset_weights_df"].loc[ref_dt]

    monitor_rows = []
    for short_id, _ in universe:
        snap = live_snapshots[short_id]
        w_sig = float(snap["w_signal"])
        w_1n = float(eq_w_last[short_id])
        w_hrp_base = float(hrp_unscaled_last[short_id])
        w_hrp_eff = float(hrp_eff_last[short_id])

        snap["w_1n_pct"] = round(w_1n * 100.0, 1)
        snap["w_hrp_base_pct"] = round(w_hrp_base * 100.0, 1)
        snap["w_hrp_eff_pct"] = round(w_hrp_eff * 100.0, 1)
        snap["net_pos_1n_pct"] = round(w_1n * w_sig * 100.0, 2)
        snap["net_pos_hrp_pct"] = round(w_hrp_eff * 100.0, 2)

        monitor_rows.append(
            {
                "代码": snap["short_id"],
                "品种名称": snap["name"],
                "所属板块": snap["sector"],
                "最新时间": snap["latest_dt"][:16],
                "当前趋势状态": snap["regime_label"],
                "A簇(顺势)": f"{snap['A_signal']:+.2f}",
                "B簇(反转)": f"{snap['B_signal']:+.2f}",
                "C簇(结构)": f"{snap['C_signal']:+.2f}",
                "单品种信号w": f"{w_sig:+.2f}",
                "60日波动率": f"{snap['vol_60d_pct']:.1f}%",
                "等权配置(1/N)": f"{snap['w_1n_pct']:.1f}%",
                "HRP基础权重": f"{snap['w_hrp_base_pct']:.1f}%",
                "HRP缩放权重": f"{snap['w_hrp_eff_pct']:.1f}%",
                "★ HRP目标净仓位": f"{snap['net_pos_hrp_pct']:+.2f}%",
            }
        )

    print("\n" + "=" * 145)
    print("  【1. 九大品种最新截面信号分解与目标仓位指令表 (在市等权 1/N vs Sector-HRP)】")
    print("=" * 145)
    print(pd.DataFrame(monitor_rows).to_string(index=False))

    # 2. 计算 9x9 跨品种策略日收益相关性矩阵
    ret_df = pd.DataFrame({k.upper(): v for k, v in asset_daily_returns.items()}).sort_index()
    corr_df = ret_df.corr().round(3)
    print("\n" + "=" * 145)
    print("  【2. 九大品种策略日收益 9×9 相关性矩阵 (Pairwise Daily Return Correlation Matrix)】")
    print("=" * 145)
    print(corr_df.to_string())

    # 保存监控快照 JSON 与 CSV
    out_csv = CTA_OUTPUT_DIR / "nine_assets_live_target_positions.csv"
    pd.DataFrame(monitor_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 生成 Generative UI 实时信号与相关性矩阵看板
    cards_html = ""
    for short_id, _ in universe:
        s = live_snapshots[short_id]
        w_sig = s["w_signal"]
        badge_cls = (
            "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
            if w_sig > 0.05
            else (
                "bg-rose-500/20 text-rose-400 border-rose-500/30"
                if w_sig < -0.05
                else "bg-slate-500/20 text-slate-300 border-slate-500/30"
            )
        )
        dir_txt = "做多 LONG" if w_sig > 0.05 else ("做空 SHORT" if w_sig < -0.05 else "空仓观望 FLAT")
        cards_html += f"""
        <div class="p-3 rounded-xl bg-[var(--background)] border border-[var(--border)] flex flex-col justify-between">
          <div class="flex items-center justify-between">
            <span class="font-bold text-sm text-[var(--foreground)]">{s["short_id"]} · {s["name"].split("_")[1]}</span>
            <span class="px-2 py-0.5 rounded-full text-[11px] font-semibold border {badge_cls}">{dir_txt} ({w_sig:+.2f})</span>
          </div>
          <div class="text-[11px] text-[var(--muted-foreground)] mt-1">板块: {s["sector"]} | 状态: {s["regime_label"].split(" ")[0]}</div>
          <div class="grid grid-cols-3 gap-1 my-2 text-[11px] font-mono bg-slate-900/50 p-1.5 rounded-lg border border-slate-800">
            <div>A顺势: <span class="text-emerald-400">{s["A_signal"]:+.2f}</span></div>
            <div>B反转: <span class="text-amber-400">{"剔除" if s["is_commodity"] else f"{s['B_signal']:+.2f}"}</span></div>
            <div>C结构: <span class="text-sky-400">{s["C_signal"]:+.2f}</span></div>
          </div>
          <div class="flex items-center justify-between text-xs pt-1 border-t border-[var(--border)]">
            <span class="text-[var(--muted-foreground)]">等权1/N: <b class="text-sky-400">{s["w_1n_pct"]:.1f}%</b> | HRP权重: <b class="text-emerald-400">{s["w_hrp_eff_pct"]:.1f}%</b></span>
            <span class="font-mono font-bold text-[var(--foreground)]">目标仓位: {s["net_pos_hrp_pct"]:+.1f}%</span>
          </div>
        </div>
        """

    cols = list(corr_df.columns)
    corr_header = "".join([f'<th class="p-1.5 text-center font-mono text-xs text-slate-300">{c}</th>' for c in cols])
    corr_rows_html = ""
    for r_name in cols:
        cells = f'<td class="p-1.5 font-mono text-xs font-bold text-slate-300">{r_name}</td>'
        for c_name in cols:
            val = float(corr_df.loc[r_name, c_name])
            if r_name == c_name:
                bg = "bg-emerald-500/30 text-emerald-300 font-bold"
            elif val > 0.25:
                bg = "bg-amber-500/25 text-amber-300"
            elif val < 0.0:
                bg = "bg-sky-500/20 text-sky-300"
            else:
                bg = "bg-slate-800/60 text-slate-400"
            cells += f'<td class="p-1.5 text-center font-mono text-xs rounded {bg}">{val:+.2f}</td>'
        corr_rows_html += f"<tr>{cells}</tr>"

    widget_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-3">
  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-2xl p-5 shadow-lg max-w-4xl mx-auto">
    <div class="flex flex-wrap items-center justify-between pb-3 border-b border-[var(--border)] gap-2">
      <div>
        <div class="flex items-center gap-2">
          <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">实盘截面监控与相关性矩阵</span>
          <h2 class="text-lg font-bold tracking-tight text-[var(--foreground)]">九大品种全天候 CTA · 实时簇级信号归因与 9×9 相关性热力表</h2>
        </div>
        <p class="text-xs text-[var(--muted-foreground)] mt-1">股指保留完整三支柱 (A+B+C) · 商品剔除 B_背驰反转簇仅保留顺势与量能结构 (A+C) · 跨板块 Sector-HRP 动态配权</p>
      </div>
    </div>

    <div class="grid grid-cols-1 sm:grid-cols-3 gap-2.5 my-4">
      {cards_html}
    </div>

    <div class="mt-4 bg-[var(--background)] rounded-xl border border-[var(--border)] p-3 overflow-x-auto">
      <div class="text-xs font-bold text-[var(--foreground)] mb-2">九大品种策略日收益 9×9 相关性矩阵 (股指内部正相关约 0.31~0.42，股指与商品跨资产相关性接近 0.00)</div>
      <table class="w-full border-separate border-spacing-1">
        <thead>
          <tr><th class="p-1.5 text-left text-xs text-slate-400">品种</th>{corr_header}</tr>
        </thead>
        <tbody>
          {corr_rows_html}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""
    monitor_artifact = ARTIFACT_DIR / "nine_assets_live_monitor_card.html"
    monitor_artifact.write_text(widget_html, encoding="utf-8")
    print(f"\n>>> 实时目标仓位指令已保存至: {out_csv}")
    print(f">>> 交互式截面监控与相关性热力看板已保存至: {monitor_artifact}")


if __name__ == "__main__":
    main()
