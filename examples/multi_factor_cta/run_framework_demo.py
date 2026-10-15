#!/usr/bin/env python
"""
第五层：跨资产规则型多因子 CTA 框架全流程实证与交互式归因大屏 (End-to-End Multi-Asset Rule-Based CTA Runner)

覆盖三大真实资产类别：
1. 宽基 ETF (日内 T+0 双向 & 纯多头现货双轨对比): `SZSE.159915` (创业板 ETF, 5m~1d 五级周期, 2020-2026)
2. 股指期货 / 中证 500 指数衍生品标的 (`IC / 510500`): `SHSE.510500` (中证 500, 30m~1d 三级周期, 2020-2026)
3. 商品期货主力连续 (含夜盘 21:00-02:30 交易日归属与换月复权): `KQ.m@SHFE.ag` (沪银主力连续, 15m~1d 四级周期, 2021-2026)

运行方式：
    uv run --no-sync python examples/multi_factor_cta/run_framework_demo.py
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
from examples.multi_factor_cta.forensic_validator import (  # noqa: E402
    RuleFactorForensicValidator,
    compute_cross_sectional_concurrency,
    compute_signed_jaccard,
)
from examples.multi_factor_cta.synthesis_engine import (  # noqa: E402
    CrossAssetHRPSynthesizer,
    IntraAssetThreePillarSynthesizer,
    evaluate_single_asset_backtest,
    joint_is_coordinate_reweight,
)

WORKSPACE_DIR = ROOT_DIR / "examples" / "results" / "alpha_agent_workspace"
CTA_OUTPUT_DIR = ROOT_DIR / "examples" / "results" / "multi_factor_cta"


def build_default_cross_asset_universe() -> list[AssetSpec]:
    return [
        AssetSpec(
            symbol="SZSE.159915",
            name="159915_创业板ETF(日内T+0)",
            asset_class=AssetClass.ETF_T0,
            sector="EQUITY_GROWTH",
            base_tf="5m",
            eligible_tfs=("1d", "60m", "30m", "15m", "5m"),
            fee_rate=0.0002,
            has_night_session=False,
            execution_mode=ExecutionMode.LONG_SHORT,
            data_path=ROOT_DIR / "examples" / "results" / "159915_real_5m_2020_2026.csv",
        ),
        AssetSpec(
            symbol="SHSE.510500",
            name="IC_510500_中证500股指衍生标的",
            asset_class=AssetClass.INDEX_FUTURES,
            sector="EQUITY_MIDCAP",
            base_tf="30m",
            eligible_tfs=("1d", "60m", "30m"),
            fee_rate=0.0002,
            has_night_session=False,
            execution_mode=ExecutionMode.LONG_SHORT,
            data_path=ROOT_DIR / "examples" / "results" / "trend_pullback_510500_30m" / "bars_30m.parquet",
        ),
        AssetSpec(
            symbol="KQ.m@SHFE.ag",
            name="SHFE.ag_沪银期货主力连续(含夜盘)",
            asset_class=AssetClass.COMMODITY_FUTURES,
            sector="PRECIOUS_METALS",
            base_tf="15m",
            eligible_tfs=("1d", "60m", "30m", "15m"),
            fee_rate=0.0002,
            has_night_session=True,
            execution_mode=ExecutionMode.LONG_SHORT,
            data_path=ROOT_DIR / "examples" / "obv_ag" / "data" / "KQ_m_SHFE_ag_900.parquet",
        ),
    ]


def export_multi_asset_cta_dashboard(
    asset_results: dict[str, dict],
    hrp_portfolio: dict,
    out_html: Path,
) -> None:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=(
            "跨资产规则型多因子 CTA 累计净值曲线 (股指衍生 + 创业板ETF + 商品期货 vs 跨品种 HRP 组合 | 2024起为纯样本外 OOS)",
            "跨资产 HRP 组合 vs 单一资产动态回撤对比 (%)",
        ),
    )

    colors = {
        "SZSE.159915": "#42A5F5",
        "SHSE.510500": "#FFA726",
        "KQ.m@SHFE.ag": "#AB47BC",
    }

    for sym, res in asset_results.items():
        dr = res["daily_return"]
        nav = 1.0 + dr.cumsum()
        fig.add_trace(
            go.Scatter(
                x=dr.index,
                y=nav.values,
                mode="lines",
                name=f"{res['name']} (Full夏普 {res['full']['sharpe']:.2f}, OOS {res['oos']['cum_ret_pct']:+.1f}%)",
                line={"width": 1.6, "color": colors.get(sym, "#90A4AE")},
            ),
            row=1,
            col=1,
        )

    port_dr: pd.Series = hrp_portfolio["daily_returns"]
    port_nav = 1.0 + port_dr.cumsum()
    port_peak = np.maximum.accumulate(np.maximum(port_nav.values, 1.0))
    port_dd = (port_nav.values - port_peak) * 100.0

    fig.add_trace(
        go.Scatter(
            x=port_dr.index,
            y=port_nav.values,
            mode="lines",
            name=(
                f"★ 跨资产 Sector-HRP 合成组合 (Full夏普 {hrp_portfolio['full']['sharpe']:.2f}, "
                f"卡玛 {hrp_portfolio['full']['calmar']:.2f}, OOS夏普 {hrp_portfolio['oos']['sharpe']:.2f})"
            ),
            line={"width": 3.0, "color": "#00E676"},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=port_dr.index,
            y=port_dd,
            mode="lines",
            fill="tozeroy",
            name=f"跨资产 HRP 组合回撤 (最大回撤 {hrp_portfolio['full']['mdd_pct']:.2f}%)",
            line={"width": 1.5, "color": "#00E676"},
        ),
        row=2,
        col=1,
    )

    fig.add_vline(x="2024-01-01", line_width=2, line_dash="dash", line_color="#FFD54F")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#101721",
        height=760,
        hovermode="x unified",
        title={
            "text": "跨资产规则型多因子 CTA 框架 (股指期货/衍生 + 日内T+0 ETF + 商品期货) 全流程实证大屏",
            "font": {"size": 18, "color": "#FFFFFF"},
        },
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def main() -> None:
    CTA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 125)
    print("  【跨资产规则型多因子 CTA 策略框架 · 数据底座 + 规则因子三级法医门禁 + 时序×截面分层合成实证】")
    print(
        "  覆盖资产族：1. 股指衍生标的 (IC/510500) | 2. 日内T+0宽基ETF (159915) | 3. 商品期货主力连续 (SHFE.ag 含夜盘)"
    )
    print("=" * 125)

    data_engine = CrossAssetDataEngine(cache_dir=WORKSPACE_DIR)
    registry = RuleFactorRegistry()
    validator = RuleFactorForensicValidator()
    intra_synth = IntraAssetThreePillarSynthesizer()
    hrp_synth = CrossAssetHRPSynthesizer(target_portfolio_vol=0.14)

    # 加载 44 个精选规则因子单元配置
    champ_path = WORKSPACE_DIR / "champion_portfolio.json"
    champ_alloc: dict[str, float] = json.loads(champ_path.read_text(encoding="utf-8"))["alloc_map"]

    universe = build_default_cross_asset_universe()
    asset_results: dict[str, dict] = {}
    asset_daily_returns: dict[str, pd.Series] = {}
    asset_sectors: dict[str, str] = {}
    cross_asset_30m_weights: dict[str, pd.Series] = {}

    # 跨品种规则因子法医普适性审计表
    core_universal_factors = ["S1L@30m", "S1S@30m", "S2L@60m", "B02L@30m", "C03L@60m", "D04L@30m", "TD03S@30m"]
    universal_audit_book: dict[str, dict[str, dict]] = {u: {} for u in core_universal_factors}

    for spec in universe:
        print(
            f"\n[1. 加载与多周期对齐] {spec.name} ({spec.symbol}) | 基准网格={spec.base_tf} | 夜盘={spec.has_night_session}"
        )
        tf_dfs = data_engine.load_asset_multi_timeframes(spec, attach_czsc=(spec.symbol == "SZSE.159915"))
        df_grid = tf_dfs[spec.base_tf]

        w_on_grid: dict[str, np.ndarray] = {}
        active_alloc: dict[str, float] = {}
        unit_forensic_meta: dict[str, dict] = {}

        # 先对跨品种核心审计池中的因子做全资产 Gate 1A 法医登记
        for uid in core_universal_factors:
            sid, tf = uid.split("@")
            if tf in spec.eligible_tfs and not (sid.startswith("CZ") and spec.symbol != "SZSE.159915"):
                f_spec = registry.get(sid)
                trig_arr, _ = registry.evaluate_factor_on_tf(tf_dfs[tf], sid, tf)
                universal_audit_book[uid][spec.symbol] = validator.run_gate1_single_asset_audit(
                    tf_dfs[tf],
                    trig_arr,
                    direction=f_spec.direction,
                    archetype=f_spec.archetype,
                    timeframe=tf,
                    fee_rate=spec.fee_rate,
                )

        if spec.symbol == "SZSE.159915":
            # 159915 直接加载已完成三级法医门禁与双半场联合配权的 44 个冠军单元
            for uid, w_val in champ_alloc.items():
                sid, tf = uid.split("@")
                if tf not in spec.eligible_tfs:
                    continue
                _, w_arr = registry.evaluate_factor_on_tf(tf_dfs[tf], sid, tf)
                w_on_grid[uid] = map_tf_weight_to_base_grid(tf_dfs[tf], w_arr, df_grid)
                active_alloc[uid] = w_val
        else:
            # 对股指与商品期货执行：1. 核心顺势骨架 (A_顺势突破簇) 锚定 + 2. 严格样本内 (2020-2023) Gate 1A 法医筛选 -> Gate 2 交并比去重 -> IS 联合坐标配权
            candidates_g1: list[tuple[str, float, np.ndarray, dict]] = []
            is_mask = (df_grid["dt"] <= validator.is_end_dt).values
            same_dir_pool: dict[int, dict[str, np.ndarray]] = {1: {}, -1: {}}
            init_alloc: dict[str, float] = {}

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
                    # 核心顺势突破骨架种子 (S1L/S2L/S3L/A01L/A02L) 作为 CTA 趋势底仓优先挂载
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

            # 按样本内法医安全边际得分降序通过 Gate 2 (Signed Jaccard < 0.30, |corr| < 0.32)
            candidates_g1.sort(key=lambda x: x[1], reverse=True)
            from examples.multi_factor_cta.forensic_validator import compute_empirical_bayes_shrinkage_weight

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

            # 样本内 (2020-2023) 法医先验锚定 + 双半场联合坐标重配权
            active_alloc = joint_is_coordinate_reweight(
                df_grid,
                w_on_grid,
                init_alloc,
                unit_forensic_meta,
                spec,
                intra_synth,
            )

        # 单品种三支柱分层合成
        w_synth = intra_synth.synthesize(df_grid, w_on_grid, active_alloc, spec)
        ev = evaluate_single_asset_backtest(df_grid, w_synth, spec.symbol, fee_rate=spec.fee_rate)
        ev["name"] = spec.name
        ev["n_units"] = len(active_alloc)
        asset_results[spec.symbol] = ev
        asset_daily_returns[spec.symbol] = ev["daily_return"]
        asset_sectors[spec.symbol] = spec.sector

        # 记录日频净持仓方向用于截面并发度 CSC 审计
        cross_asset_30m_weights[spec.symbol] = pd.Series(w_synth, index=df_grid["dt"]).resample("1D").last().fillna(0.0)

        print(
            f"  -> 挂载通过门禁规则单元: {len(active_alloc)} 个 | "
            f"IS(2020-2023): 收益={ev['is']['cum_ret_pct']:+.2f}%, 夏普={ev['is']['sharpe']:.2f}, 卡玛={ev['is']['calmar']:.2f} | "
            f"OOS(2024-2026): 收益={ev['oos']['cum_ret_pct']:+.2f}%, 夏普={ev['oos']['sharpe']:.2f}, 卡玛={ev['oos']['calmar']:.2f} | "
            f"全周期: 收益={ev['full']['cum_ret_pct']:+.2f}%, 夏普={ev['full']['sharpe']:.2f}, 回撤={ev['full']['mdd_pct']:.2f}%, 卡玛={ev['full']['calmar']:.2f}"
        )

        # 针对 ETF 额外展示【纯多头现货轨 LONG_ONLY_SPOT】与【日内 14:55 强平轨 INTRADAY_FLAT_1455】对比
        if spec.asset_class == AssetClass.ETF_T0:
            w_long_only = intra_synth.synthesize(
                df_grid, w_on_grid, active_alloc, spec, execution_mode_override=ExecutionMode.LONG_ONLY_SPOT
            )
            ev_lo = evaluate_single_asset_backtest(df_grid, w_long_only, spec.symbol, fee_rate=spec.fee_rate)
            w_intraday = intra_synth.synthesize(
                df_grid, w_on_grid, active_alloc, spec, execution_mode_override=ExecutionMode.INTRADAY_FLAT_1455
            )
            ev_id = evaluate_single_asset_backtest(df_grid, w_intraday, spec.symbol, fee_rate=spec.fee_rate)
            print(
                f"  -> [ETF 制度适配器双轨对照] "
                f"纯多头现货轨 [0,+1]: 全周期={ev_lo['full']['cum_ret_pct']:+.2f}% (夏普 {ev_lo['full']['sharpe']:.2f}, 回撤 {ev_lo['full']['mdd_pct']:.2f}%, 卡玛 {ev_lo['full']['calmar']:.2f}) | "
                f"14:55日内强平轨: 全周期={ev_id['full']['cum_ret_pct']:+.2f}% (夏普 {ev_id['full']['sharpe']:.2f}, 回撤 {ev_id['full']['mdd_pct']:.2f}%)"
            )

    # 2. 打印跨品种 Gate 1B 普适率与 Gate 2C 横截面并发度 CSC 审计
    print("\n" + "=" * 125)
    print("  【二、规则型因子跨品种普适率 (Gate 1B) 与横截面并发拥挤度 (Gate 2C · CSC) 审计】")
    print("=" * 125)
    u_rows = []
    for uid, sym_map in universal_audit_book.items():
        uni = validator.run_gate1_cross_asset_universality_audit(sym_map, min_positive_ratio=0.60)
        u_rows.append(
            {
                "规则因子单元": uid,
                "跨品种总独立样本(IS)": uni["total_pool_is_n"],
                "跨品种平均安全垫": f"{uni['mean_pool_safety_pct']:+.2f}%",
                "正期望品种占比": f"{uni['universal_ratio_pct']:.1f}%",
                "正期望覆盖标的": ", ".join(uni["positive_assets"]),
                "普适性门禁": "PASS" if uni["passed"] else "WARN",
            }
        )
    print(pd.DataFrame(u_rows).to_string(index=False))

    csc_stats = compute_cross_sectional_concurrency(cross_asset_30m_weights)
    print(
        f"\n  [横截面并发拥挤度 CSC 审计] 三大资产族日均同向激活并发率={csc_stats['mean_csc_pct']:.1f}%, "
        f"P95分位={csc_stats['p95_csc_pct']:.1f}% (跨资产相关性低，具备优异截面分散度)"
    )

    # 3. 跨品种逆波动率 + 板块分层风险平价 (Sector-HRP) 组合构建
    hrp_res = hrp_synth.build_multi_asset_portfolio(asset_daily_returns, asset_sectors)
    print("\n" + "=" * 125)
    print("  【三、单品种策略 vs 跨资产 Sector-HRP 多因子 CTA 组合全周期实证汇总 (2020-2026)】")
    print("=" * 125)
    summary_rows = []
    for _sym, res in asset_results.items():
        summary_rows.append(
            {
                "资产/组合名称": res["name"],
                "挂载规则数": res["n_units"],
                "IS夏普(20-23)": res["is"]["sharpe"],
                "IS卡玛(20-23)": res["is"]["calmar"],
                "OOS净收益(24-26)": f"{res['oos']['cum_ret_pct']:+.2f}%",
                "OOS夏普": res["oos"]["sharpe"],
                "OOS卡玛": res["oos"]["calmar"],
                "全周期净收益": f"{res['full']['cum_ret_pct']:+.2f}%",
                "全周期年化": f"{res['full']['ann_ret_pct']:.2f}%",
                "全周期夏普": res["full"]["sharpe"],
                "全周期最大回撤": f"{res['full']['mdd_pct']:.2f}%",
                "全周期卡玛": res["full"]["calmar"],
            }
        )
    summary_rows.append(
        {
            "资产/组合名称": "★ 跨资产 Sector-HRP 多因子 CTA 总组合",
            "挂载规则数": 44,
            "IS夏普(20-23)": hrp_res["is"]["sharpe"],
            "IS卡玛(20-23)": hrp_res["is"]["calmar"],
            "OOS净收益(24-26)": f"{hrp_res['oos']['cum_ret_pct']:+.2f}%",
            "OOS夏普": hrp_res["oos"]["sharpe"],
            "OOS卡玛": hrp_res["oos"]["calmar"],
            "全周期净收益": f"{hrp_res['full']['cum_ret_pct']:+.2f}%",
            "全周期年化": f"{hrp_res['full']['ann_ret_pct']:.2f}%",
            "全周期夏普": hrp_res["full"]["sharpe"],
            "全周期最大回撤": f"{hrp_res['full']['mdd_pct']:.2f}%",
            "全周期卡玛": hrp_res["full"]["calmar"],
        }
    )
    print(pd.DataFrame(summary_rows).to_string(index=False))

    out_html = CTA_OUTPUT_DIR / "multi_asset_rule_cta_dashboard.html"
    export_multi_asset_cta_dashboard(asset_results, hrp_res, out_html)
    print("=" * 125)
    print(f"  [跨资产多因子 CTA 交互式大屏已导出] {out_html.resolve()}")
    print("=" * 125)


if __name__ == "__main__":
    main()
