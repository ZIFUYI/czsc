#!/usr/bin/env python
"""
纯商品期货组合（不含股指）：不同跨品种加权方式全景对比实证
对比两大商品池：
  池 A：五大核心趋势商品 (AG沪银 / AU沪金 / SC原油 / LC碳酸锂 / PS多晶硅)
  池 B：六大板块十大商品 (AG / AU / CU沪铜 / RB螺纹 / I铁矿 / SC原油 / TA_PTA / M豆粕 / LC / PS)

对比 6 种严格无未来函数的跨品种加权方式：
  1. 在市简单等权 (Equal-Weight, 1/N_t, 总权重=100%)
  2. 纯逆波动率平价 (Inverse-Vol Parity, 1/σ_i, 总权重=100%)
  3. 板块等权×内部逆波动率 (Sector-Equal Budget, 1/(N_sec * K_s * σ_i), 总权重=100%)
  4. 板块分层去拥挤风险平价·无杠杆 (Sector-HRP Unscaled, 1/(sqrt(K_s) * σ_i), 总权重=100%)
  5. 因果动量增强型风险平价·无杠杆 (Momentum-Tilted Sector-HRP, 总权重=100%)
  6. 板块分层风险平价 + 目标波动率缩放 (Sector-HRP + VolTarget 14%, 动态敞口上限 165%)
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

from examples.alpha_mining_agent.forensic_pipeline import IS_END_DT  # noqa: E402
from examples.multi_factor_cta.data_engine import (  # noqa: E402
    AssetClass,
    CrossAssetDataEngine,
    map_tf_weight_to_base_grid,
)
from examples.multi_factor_cta.factor_engine import RuleFactorRegistry  # noqa: E402
from examples.multi_factor_cta.synthesis_engine import (  # noqa: E402
    CrossAssetHRPSynthesizer,
    IntraAssetThreePillarSynthesizer,
    evaluate_single_asset_backtest,
)
from examples.multi_factor_cta.test_all_weather_14assets_portfolio import (  # noqa: E402
    ARTIFACT_DIR,
    CTA_OUTPUT_DIR,
    WORKSPACE_DIR,
    build_fourteen_assets_universe,
)


def evaluate_weighting_schemes(
    asset_daily_returns: dict[str, pd.Series],
    asset_sectors: dict[str, str],
    target_vol: float = 0.14,
) -> dict[str, dict]:
    """在完全相同的单品种日收益序列上，计算 6 种不同跨品种加权方式的表现与权重轨迹"""
    raw_df = pd.DataFrame(asset_daily_returns).sort_index()
    raw_df.index = pd.to_datetime(raw_df.index)
    active_mask = raw_df.notna().astype(float)
    ret_df = raw_df.fillna(0.0)

    # 因果滚动 60 日波动率 (shift(1) 严格避免当日未来信息)
    roll_vol = raw_df.rolling(60, min_periods=10).std().shift(1).bfill() * np.sqrt(252.0)
    inv_vol = (1.0 / np.maximum(roll_vol.fillna(0.15), 0.05)) * active_mask

    # 1. 在市简单等权 (1/N_t, sum=100%)
    w_eq = active_mask.div(active_mask.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

    # 2. 纯逆波动率平价 (InvVol, sum=100%)
    w_invvol = inv_vol.div(inv_vol.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(w_eq)

    # 统计每日各板块在市品种数 K_{s,t}
    sec_map = {col: asset_sectors[col] for col in ret_df.columns}
    sec_counts = pd.DataFrame(0.0, index=ret_df.index, columns=ret_df.columns)
    for sec_name in set(sec_map.values()):
        cols_in_sec = [c for c, s in sec_map.items() if s == sec_name]
        cnt_s = active_mask[cols_in_sec].sum(axis=1)
        for c in cols_in_sec:
            sec_counts[c] = cnt_s

    # 3. 板块严格等权 × 板块内逆波动率 (Sector-Equal Budget: 1 / K_{s,t} * InvVol, sum=100%)
    raw_sec_eq = inv_vol / np.maximum(sec_counts, 1.0)
    w_sec_eq = raw_sec_eq.div(raw_sec_eq.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(w_eq)

    # 4. 板块分层去拥挤风险平价·无杠杆 (Sector-HRP Unscaled: 1 / sqrt(K_{s,t}) * InvVol, sum=100%)
    raw_hrp = inv_vol / np.sqrt(np.maximum(sec_counts, 1.0))
    w_hrp_unscaled = raw_hrp.div(raw_hrp.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(w_eq)

    # 5. 因果动量倾斜风险平价·无杠杆 (Mom-Tilted Sector-HRP: 结合过去 60 日因果收益信噪比倾斜, sum=100%)
    roll_mean = raw_df.rolling(60, min_periods=15).mean().shift(1).fillna(0.0)
    roll_std = raw_df.rolling(60, min_periods=15).std().shift(1).fillna(0.01)
    causal_ir = (roll_mean / np.maximum(roll_std, 1e-4)) * np.sqrt(252.0)
    # 温和 Sigmoid/Clip 动量倾斜因子 ∈ [0.55, 1.65]
    mom_tilt = np.clip(1.0 + 0.35 * np.tanh(causal_ir / 1.2), 0.55, 1.65) * active_mask
    raw_mom_hrp = raw_hrp * mom_tilt
    w_mom_hrp = raw_mom_hrp.div(raw_mom_hrp.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(w_eq)

    # 6. Sector-HRP + 组合目标波动率动态缩放 (VolTarget 14%, 上限 165%)
    unscaled_hrp_ret = (ret_df * w_hrp_unscaled).sum(axis=1)
    port_vol = unscaled_hrp_ret.rolling(40, min_periods=15).std().shift(1).bfill() * np.sqrt(252.0)
    lev_scalar = np.clip(target_vol / np.maximum(port_vol, 0.05), 0.70, 1.65)
    w_hrp_scaled = w_hrp_unscaled.multiply(lev_scalar, axis=0)

    # 7. Mom-Tilted Sector-HRP + 组合目标波动率动态缩放 (VolTarget 14%, 上限 165%)
    unscaled_mom_ret = (ret_df * w_mom_hrp).sum(axis=1)
    mom_port_vol = unscaled_mom_ret.rolling(40, min_periods=15).std().shift(1).bfill() * np.sqrt(252.0)
    mom_lev_scalar = np.clip(target_vol / np.maximum(mom_port_vol, 0.05), 0.70, 1.65)
    w_mom_scaled = w_mom_hrp.multiply(mom_lev_scalar, axis=0)

    # 8. 净值健康度×活跃槽位动态 HRP·无杠杆 (Active-EC HRP: 净值在60日均线上方且有在途持仓的品种获集中风险预算, sum=100%)
    act_pos = (ret_df.abs().rolling(3, min_periods=1).max().shift(1).fillna(1.0) > 1e-5).astype(float) * active_mask
    cum_nav = ret_df.cumsum().shift(1).fillna(0.0)
    nav_ma60 = cum_nav.rolling(60, min_periods=10).mean().fillna(0.0)
    ec_health = np.where(cum_nav >= nav_ma60, 1.25, 0.30) * active_mask
    raw_comb = raw_hrp * ec_health * np.where(act_pos > 0, 1.15, 0.35)
    w_comb_norm = raw_comb.div(raw_comb.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(w_eq)

    # 9. 净值健康度×活跃槽位动态 HRP + 组合目标波动率缩放 (VolTarget 14%, 上限 165%)
    unscaled_comb_ret = (ret_df * w_comb_norm).sum(axis=1)
    comb_port_vol = unscaled_comb_ret.rolling(40, min_periods=15).std().shift(1).bfill() * np.sqrt(252.0)
    comb_lev_scalar = np.clip(target_vol / np.maximum(comb_port_vol, 0.05), 0.70, 1.65)
    w_comb_scaled = w_comb_norm.multiply(comb_lev_scalar, axis=0)

    schemes = {
        "1_Equal_1/N": ("① 在市简单等权 (1/N_t, 无杠杆100%)", w_eq),
        "2_InvVol_100%": ("② 纯逆波动率平价 (InvVol, 无杠杆100%)", w_invvol),
        "3_SecEqual_100%": ("③ 板块严格等权×逆波动率 (SecEqual, 无杠杆100%)", w_sec_eq),
        "4_SecHRP_100%": ("④ 板块分层HRP·无杠杆 (SecHRP, 无杠杆100%)", w_hrp_unscaled),
        "5_MomHRP_100%": ("⑤ 因果动量增强HRP·无杠杆 (Mom-HRP, 无杠杆100%)", w_mom_hrp),
        "6_ActiveEC_100%": ("★⑥ 净值状态×活跃槽位HRP·无杠杆 (ActiveEC-HRP, 100%)", w_comb_norm),
        "7_SecHRP_VolTgt": ("⑦ 板块分层HRP + 目标波动率缩放 (SecHRP-VolTgt)", w_hrp_scaled),
        "8_MomHRP_VolTgt": ("⑧ 动量增强HRP + 目标波动率缩放 (Mom-HRP-VolTgt)", w_mom_scaled),
        "9_ActiveEC_VolTgt": ("★★⑨ 净值状态×活跃槽位HRP + 目标波动率缩放 (ActiveEC-VolTgt)", w_comb_scaled),
    }

    is_mask = ret_df.index <= pd.Timestamp(IS_END_DT.date())
    oos_mask = ret_df.index > pd.Timestamp(IS_END_DT.date())

    results: dict[str, dict] = {}
    for key, (label, w_df) in schemes.items():
        dr = (ret_df * w_df).sum(axis=1)
        results[key] = {
            "label": label,
            "weights_df": w_df,
            "daily_returns": dr,
            "full": CrossAssetHRPSynthesizer._calc_return_series_stats(dr),
            "is": CrossAssetHRPSynthesizer._calc_return_series_stats(dr.loc[is_mask]),
            "oos": CrossAssetHRPSynthesizer._calc_return_series_stats(dr.loc[oos_mask]),
            "avg_gross_exposure_pct": round(float(w_df.sum(axis=1).mean() * 100.0), 1),
        }
    return results


def main() -> None:
    CTA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 145)
    print("  【纯商品期货组合 (不含股指)：7 种跨品种加权方式在「5大核心商品」与「10大广谱商品」上的全景对比实证】")
    print("=" * 145)

    data_engine = CrossAssetDataEngine(cache_dir=WORKSPACE_DIR)
    registry = RuleFactorRegistry()
    intra_synth = IntraAssetThreePillarSynthesizer()

    champ_path = WORKSPACE_DIR / "champion_portfolio.json"
    champ_alloc: dict[str, float] = json.loads(champ_path.read_text(encoding="utf-8"))["alloc_map"]

    comm_universe = [
        (sid, spec)
        for sid, spec in build_fourteen_assets_universe()
        if spec.asset_class == AssetClass.COMMODITY_FUTURES
    ]

    comm_returns: dict[str, pd.Series] = {}
    comm_sectors: dict[str, str] = {}
    single_rows = []

    for short_id, spec in comm_universe:
        tf_dfs = data_engine.load_asset_multi_timeframes(spec, attach_czsc=False)
        df_grid = tf_dfs[spec.base_tf]

        w_on_grid: dict[str, np.ndarray] = {}
        active_alloc: dict[str, float] = {}
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
        comm_returns[short_id] = ev["daily_return"]
        comm_sectors[short_id] = spec.sector

        single_rows.append(
            {
                "品种": short_id.upper(),
                "名称": spec.name,
                "板块": spec.sector,
                "IS净收益(20-23)": f"{ev['is']['cum_ret_pct']:+.2f}%" if ev["is"]["active_pct"] > 0 else "N/A",
                "OOS净收益(24-26)": f"{ev['oos']['cum_ret_pct']:+.2f}%",
                "OOS夏普": ev["oos"]["sharpe"],
                "全周期总收益": f"{ev['full']['cum_ret_pct']:+.2f}%",
                "全周期夏普": ev["full"]["sharpe"],
                "最大回撤": f"{ev['full']['mdd_pct']:.2f}%",
            }
        )

    print("\n" + "=" * 145)
    print("  【单品种基准表现：10 大商品期货 (已去掉 B_背驰反转簇，日线宏观定锚)】")
    print("=" * 145)
    print(pd.DataFrame(single_rows).to_string(index=False))

    # 池 A：五大核心商品 (ag, au, sc, lc, ps)
    core5_ids = ["ag", "au", "sc", "lc", "ps"]
    core5_rets = {k: comm_returns[k] for k in core5_ids}
    core5_secs = {k: comm_sectors[k] for k in core5_ids}
    res_core5 = evaluate_weighting_schemes(core5_rets, core5_secs)

    # 池 B：十大全板块商品 (ag, au, cu, rb, i, sc, ta, m, lc, ps)
    res_all10 = evaluate_weighting_schemes(comm_returns, comm_sectors)

    for pool_title, res_dict in [
        ("池 A：五大核心商品 (AG沪银 / AU沪金 / SC原油 / LC碳酸锂 / PS多晶硅)", res_core5),
        ("池 B：六大板块十大商品 (AG / AU / CU / RB / I / SC / TA / M / LC / PS)", res_all10),
    ]:
        rows = []
        for _k, v in res_dict.items():
            rows.append(
                {
                    "加权方式": v["label"],
                    "平均名义总仓位": f"{v['avg_gross_exposure_pct']:.1f}%",
                    "IS净收益(20-23)": f"{v['is']['cum_ret_pct']:+.2f}%",
                    "IS夏普": v["is"]["sharpe"],
                    "OOS净收益(24-26)": f"{v['oos']['cum_ret_pct']:+.2f}%",
                    "OOS夏普": v["oos"]["sharpe"],
                    "OOS卡玛": v["oos"]["calmar"],
                    "全周期总收益": f"{v['full']['cum_ret_pct']:+.2f}%",
                    "全周期年化": f"{v['full']['ann_ret_pct']:+.2f}%",
                    "全周期夏普": v["full"]["sharpe"],
                    "最大回撤": f"{v['full']['mdd_pct']:.2f}%",
                    "全周期卡玛": v["full"]["calmar"],
                }
            )
        print("\n" + "=" * 145)
        print(f"  【{pool_title} —— 7 种加权方式绩效对比表】")
        print("=" * 145)
        print(pd.DataFrame(rows).to_string(index=False))

    # 打印全在市阶段 (2025-2026) 各品种在 5 种无杠杆加权方式下的平均权重百分比对比 (池 A 与 池 B)
    print("\n" + "=" * 145)
    print("  【池 A (5大商品全在市阶段 2025-2026) 各加权方式下的品种权重分配对比 (%)】")
    print("=" * 145)
    w_rows_5 = []
    for k, v in res_core5.items():
        wdf = v["weights_df"]
        sub_w = wdf.loc[wdf.index >= "2025-01-01"].mean() * 100.0
        row = {"加权方式": v["label"]}
        for sid in core5_ids:
            row[sid.upper()] = f"{sub_w[sid]:.1f}%"
        row["合计总敞口"] = f"{sub_w.sum():.1f}%"
        w_rows_5.append(row)
    print(pd.DataFrame(w_rows_5).to_string(index=False))

    print("\n" + "=" * 145)
    print("  【池 B (10大商品全在市阶段 2025-2026) 各加权方式下的品种权重分配对比 (%)】")
    print("=" * 145)
    w_rows_10 = []
    for k, v in res_all10.items():
        wdf = v["weights_df"]
        sub_w = wdf.loc[wdf.index >= "2025-01-01"].mean() * 100.0
        row = {"加权方式": v["label"]}
        for sid, _ in comm_universe:
            row[sid.upper()] = f"{sub_w[sid]:.1f}%"
        row["合计"] = f"{sub_w.sum():.1f}%"
        w_rows_10.append(row)
    print(pd.DataFrame(w_rows_10).to_string(index=False))

    # 生成 Generative UI 交互式对比大屏
    all_dates = res_core5["1_Equal_1/N"]["daily_returns"].index
    nav_df = pd.DataFrame({"date": all_dates.strftime("%Y-%m-%d")})
    for k in res_core5:
        nav_df[f"c5_{k}"] = (1.0 + res_core5[k]["daily_returns"].cumsum()).values
        nav_df[f"c10_{k}"] = (1.0 + res_all10[k]["daily_returns"].cumsum()).values

    step = max(1, len(nav_df) // 180)
    sample_df = nav_df.iloc[::step].copy()
    if sample_df.iloc[-1]["date"] != nav_df.iloc[-1]["date"]:
        sample_df = pd.concat([sample_df, nav_df.iloc[[-1]]])

    svg_data = {
        "dates": sample_df["date"].tolist(),
        "c5_eq": [round(float(v), 4) for v in sample_df["c5_1_Equal_1/N"].tolist()],
        "c5_hrp": [round(float(v), 4) for v in sample_df["c5_4_SecHRP_100%"].tolist()],
        "c5_mom": [round(float(v), 4) for v in sample_df["c5_6_ActiveEC_100%"].tolist()],
        "c5_mom_vt": [round(float(v), 4) for v in sample_df["c5_9_ActiveEC_VolTgt"].tolist()],
        "c10_eq": [round(float(v), 4) for v in sample_df["c10_1_Equal_1/N"].tolist()],
        "c10_hrp": [round(float(v), 4) for v in sample_df["c10_4_SecHRP_100%"].tolist()],
        "c10_mom": [round(float(v), 4) for v in sample_df["c10_2_InvVol_100%"].tolist()],
        "c10_mom_vt": [round(float(v), 4) for v in sample_df["c10_7_SecHRP_VolTgt"].tolist()],
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
          <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30">纯商品期货组合 · 9种跨品种加权方式对比</span>
          <h2 class="text-xl font-bold tracking-tight text-[var(--foreground)]">等权(1/N) vs 逆波动率 vs Sector-HRP vs 净值状态×活跃槽位 ActiveEC-HRP</h2>
        </div>
        <p class="text-xs text-[var(--muted-foreground)] mt-1">不含任何股指 · 对比「5大核心商品 (AG/AU/SC/LC/PS)」与「10大全板块商品」在 9 种加权机制下的净值轨迹</p>
      </div>
      <div class="flex gap-1.5">
        <button id="btn-c5" onclick="switchPool('c5')" class="px-3 py-1.5 rounded-lg text-xs font-bold bg-emerald-500 text-slate-950 transition">池A：5大核心商品</button>
        <button id="btn-c10" onclick="switchPool('c10')" class="px-3 py-1.5 rounded-lg text-xs font-bold bg-slate-800 text-slate-300 border border-slate-700 transition">池B：10大全板块商品</button>
      </div>
    </div>

    <!-- Metric Cards -->
    <div id="cards-c5" class="grid grid-cols-2 sm:grid-cols-4 gap-2.5 my-4">
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-sky-400">① 在市等权 (1/N, 100%)</div>
        <div class="text-base font-bold text-[var(--foreground)] mt-0.5">+{res_core5['1_Equal_1/N']['full']['cum_ret_pct']:.2f}% (Sh {res_core5['1_Equal_1/N']['full']['sharpe']:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS Sh: {res_core5['1_Equal_1/N']['oos']['sharpe']:.2f} | MDD {res_core5['1_Equal_1/N']['full']['mdd_pct']:.2f}%</div>
      </div>
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-purple-400">④ Sector-HRP (无杠杆100%)</div>
        <div class="text-base font-bold text-[var(--foreground)] mt-0.5">+{res_core5['4_SecHRP_100%']['full']['cum_ret_pct']:.2f}% (Sh {res_core5['4_SecHRP_100%']['full']['sharpe']:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS Sh: {res_core5['4_SecHRP_100%']['oos']['sharpe']:.2f} | MDD {res_core5['4_SecHRP_100%']['full']['mdd_pct']:.2f}%</div>
      </div>
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-amber-400">★⑥ ActiveEC-HRP (无杠杆100%)</div>
        <div class="text-base font-bold text-amber-400 mt-0.5">+{res_core5['6_ActiveEC_100%']['full']['cum_ret_pct']:.2f}% (Sh {res_core5['6_ActiveEC_100%']['full']['sharpe']:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS: +{res_core5['6_ActiveEC_100%']['oos']['cum_ret_pct']:.1f}% (Sh {res_core5['6_ActiveEC_100%']['oos']['sharpe']:.2f})</div>
      </div>
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-emerald-400">★★⑨ ActiveEC-HRP + 目标波幅</div>
        <div class="text-base font-bold text-emerald-400 mt-0.5">+{res_core5['9_ActiveEC_VolTgt']['full']['cum_ret_pct']:.2f}% (Sh {res_core5['9_ActiveEC_VolTgt']['full']['sharpe']:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS: +{res_core5['9_ActiveEC_VolTgt']['oos']['cum_ret_pct']:.1f}% (Sh {res_core5['9_ActiveEC_VolTgt']['oos']['sharpe']:.2f})</div>
      </div>
    </div>

    <div id="cards-c10" class="hidden grid grid-cols-2 sm:grid-cols-4 gap-2.5 my-4">
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-sky-400">① 10商品在市等权 (100%)</div>
        <div class="text-base font-bold text-[var(--foreground)] mt-0.5">+{res_all10['1_Equal_1/N']['full']['cum_ret_pct']:.2f}% (Sh {res_all10['1_Equal_1/N']['full']['sharpe']:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS Sh: {res_all10['1_Equal_1/N']['oos']['sharpe']:.2f} | MDD {res_all10['1_Equal_1/N']['full']['mdd_pct']:.2f}%</div>
      </div>
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-amber-400">② 10商品纯逆波动率 (100%)</div>
        <div class="text-base font-bold text-[var(--foreground)] mt-0.5">+{res_all10['2_InvVol_100%']['full']['cum_ret_pct']:.2f}% (Sh {res_all10['2_InvVol_100%']['full']['sharpe']:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS Sh: {res_all10['2_InvVol_100%']['oos']['sharpe']:.2f} | MDD {res_all10['2_InvVol_100%']['full']['mdd_pct']:.2f}%</div>
      </div>
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-purple-400">④ 10商品Sector-HRP (100%)</div>
        <div class="text-base font-bold text-[var(--foreground)] mt-0.5">+{res_all10['4_SecHRP_100%']['full']['cum_ret_pct']:.2f}% (Sh {res_all10['4_SecHRP_100%']['full']['sharpe']:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS Sh: {res_all10['4_SecHRP_100%']['oos']['sharpe']:.2f} | MDD {res_all10['4_SecHRP_100%']['full']['mdd_pct']:.2f}%</div>
      </div>
      <div class="p-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs font-semibold text-emerald-400">★⑦ 10商品Sector-HRP+目标波幅</div>
        <div class="text-base font-bold text-emerald-400 mt-0.5">+{res_all10['7_SecHRP_VolTgt']['full']['cum_ret_pct']:.2f}% (Sh {res_all10['7_SecHRP_VolTgt']['full']['sharpe']:.2f})</div>
        <div class="text-xs text-[var(--muted-foreground)]">OOS: +{res_all10['7_SecHRP_VolTgt']['oos']['cum_ret_pct']:.1f}% | MDD {res_all10['7_SecHRP_VolTgt']['full']['mdd_pct']:.2f}%</div>
      </div>
    </div>

    <!-- SVG Chart -->
    <div class="relative bg-[var(--background)] rounded-xl border border-[var(--border)] p-4 select-none">
      <div class="flex flex-wrap items-center justify-between mb-2 gap-2 text-xs">
        <div class="flex flex-wrap items-center gap-3">
          <span class="flex items-center gap-1"><span class="w-3 h-1 bg-emerald-400 rounded"></span><span class="text-emerald-400 font-bold">★⑦ 动量HRP+目标波动率</span></span>
          <span class="flex items-center gap-1"><span class="w-3 h-0.5 bg-teal-300"></span><span class="text-teal-300">⑥ Sector-HRP+目标波动率</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-amber-400"></span><span class="text-amber-400">⑤ 动量HRP (无杠杆100%)</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-purple-400"></span><span class="text-purple-400">④ Sector-HRP (无杠杆100%)</span></span>
          <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-sky-400 border-b border-dashed"></span><span class="text-sky-400">① 在市等权 (1/N, 100%)</span></span>
        </div>
        <span class="px-2 py-0.5 rounded bg-amber-500/20 text-amber-400 font-mono">点击右上角按钮可切换「5商品」/「10商品」</span>
      </div>

      <div id="hover-hud" class="hidden absolute top-10 right-4 bg-slate-900/90 border border-slate-700/80 rounded-lg p-2.5 text-xs z-20 pointer-events-none shadow-xl backdrop-blur-sm">
        <div id="hud-date" class="font-mono text-slate-300 font-bold mb-1"></div>
        <div class="text-emerald-400 font-mono" id="hud-mom-vt"></div>
        <div class="text-amber-300 font-mono" id="hud-mom"></div>
        <div class="text-purple-300 font-mono" id="hud-hrp"></div>
        <div class="text-sky-300 font-mono" id="hud-eq"></div>
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

          <path id="eq-path" d="" fill="none" stroke="#38BDF8" stroke-width="1.8" stroke-dasharray="4,3"/>
          <path id="hrp-path" d="" fill="none" stroke="#AB47BC" stroke-width="2.0"/>
          <path id="mom-path" d="" fill="none" stroke="#FFD54F" stroke-width="2.2"/>
          <path id="mom-vt-fill" d="" fill="url(#mom-grad)" opacity="0.20"/>
          <path id="mom-vt-path" d="" fill="none" stroke="#00E676" stroke-width="3.0"/>

          <line id="hover-line" x1="0" y1="10" x2="0" y2="220" stroke="#FFFFFF" stroke-width="1" stroke-dasharray="2,2" class="hidden"/>
          <circle id="hover-dot" cx="0" cy="0" r="4" fill="#00E676" stroke="#FFFFFF" stroke-width="1.5" class="hidden"/>

          <defs>
            <linearGradient id="mom-grad" x1="0%" y1="0%" x2="0%" y2="100%">
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
    const data = {json.dumps(svg_data)};
    const dates = data.dates;
    const n = dates.length;
    const xLeft = 50, xRight = 770, yTop = 20, yBottom = 220;
    let curPool = 'c5';
    let minNav = 0.95, maxNav = 1.5;

    function getX(i) {{ return xLeft + (i / (n - 1)) * (xRight - xLeft); }}
    function getY(val) {{ return yBottom - ((val - minNav) / (maxNav - minNav)) * (yBottom - yTop); }}
    function buildPath(arr) {{
      let d = `M ${{getX(0)}} ${{getY(arr[0])}}`;
      for (let i = 1; i < n; i++) d += ` L ${{getX(i)}} ${{getY(arr[i])}}`;
      return d;
    }}

    function renderPool(pool) {{
      curPool = pool;
      const eqArr = data[pool + '_eq'];
      const hrpArr = data[pool + '_hrp'];
      const momArr = data[pool + '_mom'];
      const vtArr = data[pool + '_mom_vt'];
      const allVals = [...eqArr, ...hrpArr, ...momArr, ...vtArr];
      minNav = Math.min(...allVals) * 0.98;
      maxNav = Math.max(...allVals) * 1.03;
      document.getElementById('label-nav-max').innerText = maxNav.toFixed(2);
      document.getElementById('label-nav-min').innerText = minNav.toFixed(2);

      document.getElementById('eq-path').setAttribute('d', buildPath(eqArr));
      document.getElementById('hrp-path').setAttribute('d', buildPath(hrpArr));
      document.getElementById('mom-path').setAttribute('d', buildPath(momArr));
      const vtD = buildPath(vtArr);
      const fillD = `M ${{getX(0)}} ${{yBottom}} L ` + vtD.slice(2) + ` L ${{getX(n - 1)}} ${{yBottom}} Z`;
      document.getElementById('mom-vt-path').setAttribute('d', vtD);
      document.getElementById('mom-vt-fill').setAttribute('d', fillD);
    }}

    function switchPool(pool) {{
      renderPool(pool);
      if (pool === 'c5') {{
        document.getElementById('cards-c5').classList.remove('hidden');
        document.getElementById('cards-c10').classList.add('hidden');
        document.getElementById('btn-c5').className = 'px-3 py-1.5 rounded-lg text-xs font-bold bg-emerald-500 text-slate-950 transition';
        document.getElementById('btn-c10').className = 'px-3 py-1.5 rounded-lg text-xs font-bold bg-slate-800 text-slate-300 border border-slate-700 transition';
      }} else {{
        document.getElementById('cards-c5').classList.add('hidden');
        document.getElementById('cards-c10').classList.remove('hidden');
        document.getElementById('btn-c10').className = 'px-3 py-1.5 rounded-lg text-xs font-bold bg-emerald-500 text-slate-950 transition';
        document.getElementById('btn-c5').className = 'px-3 py-1.5 rounded-lg text-xs font-bold bg-slate-800 text-slate-300 border border-slate-700 transition';
      }}
    }}

    const oosIdx = dates.findIndex(d => d >= '2024-01-01');
    if (oosIdx > 0) {{
      const oosX = getX(oosIdx);
      const oosLine = document.getElementById('oos-line');
      oosLine.setAttribute('x1', oosX);
      oosLine.setAttribute('x2', oosX);
      document.getElementById('oos-text').setAttribute('x', oosX + 6);
    }}

    renderPool('c5');

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
      const vtArr = data[curPool + '_mom_vt'];
      const momArr = data[curPool + '_mom'];
      const hrpArr = data[curPool + '_hrp'];
      const eqArr = data[curPool + '_eq'];
      hoverLine.setAttribute('x1', curX);
      hoverLine.setAttribute('x2', curX);
      hoverLine.classList.remove('hidden');
      hoverDot.setAttribute('cx', curX);
      hoverDot.setAttribute('cy', getY(vtArr[idx]));
      hoverDot.classList.remove('hidden');
      hud.classList.remove('hidden');
      document.getElementById('hud-date').innerText = dates[idx];
      document.getElementById('hud-mom-vt').innerText = `★⑦ 动量HRP+缩放: ${{vtArr[idx].toFixed(3)}} (${{((vtArr[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-mom').innerText = `⑤ 动量HRP(100%): ${{momArr[idx].toFixed(3)}} (${{((momArr[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-hrp').innerText = `④ Sector-HRP(100%): ${{hrpArr[idx].toFixed(3)}} (${{((hrpArr[idx]-1)*100).toFixed(1)}}%)`;
      document.getElementById('hud-eq').innerText = `① 在市等权(100%): ${{eqArr[idx].toFixed(3)}} (${{((eqArr[idx]-1)*100).toFixed(1)}}%)`;
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
    artifact_html = ARTIFACT_DIR / "commodity_weighting_comparison_chart.html"
    artifact_html.write_text(widget_html, encoding="utf-8")
    print(f"\n>>> 纯商品组合加权方式对比交互式看板已保存至: {artifact_html}")


if __name__ == "__main__":
    main()
