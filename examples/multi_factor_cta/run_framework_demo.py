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

from examples.alpha_mining_agent.forensic_pipeline import IS_END_DT  # noqa: E402
from examples.alpha_mining_agent.run_evolution_loop import (  # noqa: E402
    ResearchAuditContext,
    audit_research_candidate,
    compute_research_signal,
    prepare_research_pool,
)
from examples.multi_factor_cta.data_engine import (  # noqa: E402
    AssetClass,
    AssetSpec,
    CrossAssetDataEngine,
    ExecutionMode,
    map_tf_weight_to_base_grid,
)
from examples.multi_factor_cta.factor_engine import RuleFactorRegistry  # noqa: E402
from examples.multi_factor_cta.forensic_validator import (  # noqa: E402
    compute_cross_sectional_concurrency,
)
from examples.multi_factor_cta.synthesis_engine import (  # noqa: E402
    ContinuousSignalStateMachine,
    CrossAssetHRPSynthesizer,
    IntraAssetThreePillarSynthesizer,
    RollingEmpiricalBayesWeighter,
    build_asset_execution_inputs,
    evaluate_single_asset_backtest,
    govern_position_with_hysteresis_runner,
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


def load_research_universe(
    data_engine: CrossAssetDataEngine, specs: list[AssetSpec]
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load the complete declared universe before attaching any cross-sectional gate."""
    frames = {spec.symbol: data_engine.load_asset_multi_timeframes(spec, attach_czsc=True) for spec in specs}
    metadata = {
        spec.symbol: {
            "asset_group": "commodity" if spec.asset_class == AssetClass.COMMODITY_FUTURES else "equity",
            "sector": spec.sector,
            "fee_rate": spec.fee_rate,
        }
        for spec in specs
    }
    # No liquidity/session regime is guessed: Tier 3 rejects missing microstructure metadata.
    return prepare_research_pool(frames, metadata=metadata)


def load_candidate_alloc_map(champ_path: Path | None = None) -> dict[str, float]:
    """加载精选规则因子单元配置，若本地缓存 json 不存在则自动回退至 RuleFactorRegistry 内置种子因子池。"""
    p = champ_path if champ_path is not None else (WORKSPACE_DIR / "champion_portfolio.json")
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))["alloc_map"]
    reg = RuleFactorRegistry()
    alloc: dict[str, float] = {}
    for spec in reg.list_all():
        if spec.is_base_seed:
            for tf in spec.eligible_tfs:
                alloc[f"{spec.factor_id}@{tf}"] = float(spec.default_weight)
    return alloc


def synthesize_admitted_asset(
    spec: AssetSpec,
    frames: dict[str, dict[str, pd.DataFrame]],
    candidate_ids: list[str],
    synthesizer: IntraAssetThreePillarSynthesizer,
    *,
    is_end_dt: pd.Timestamp = IS_END_DT,
    audit_context: ResearchAuditContext | None = None,
    enable_hysteresis_runner: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, dict]]:
    """Re-audit every seed/champion, fit through the cutoff and trade only afterwards.

    Historical IS-selected membership/priors are never backfilled into tradable IS
    positions. Candidate IDs are configuration; saved optimized coefficients are
    not treated as historical priors. Defaults anchor strictly rolling estimates.
    """
    registry = RuleFactorRegistry()
    tf_dfs = frames[spec.symbol]
    grid = tf_dfs[spec.base_tf]
    weights, priors, audits = {}, {}, {}
    for uid in candidate_ids:
        sid, tf = uid.split("@")
        if tf not in spec.eligible_tfs or tf not in tf_dfs:
            continue
        factor = registry.get(sid)
        if tf not in factor.eligible_tfs:
            continue
        audit_kwargs = {"is_end_dt": is_end_dt}
        if audit_context is not None:
            audit_kwargs["audit_context"] = audit_context
        audit = audit_research_candidate(sid, tf, spec.symbol, frames, **audit_kwargs)
        audits[uid] = audit
        if not audit["passed"]:
            continue
        _, unit = compute_research_signal(tf_dfs[tf], sid, tf)
        weights[uid] = map_tf_weight_to_base_grid(tf_dfs[tf], unit, grid)
        priors[uid] = float(factor.default_weight)
    if not weights:
        return np.zeros(len(grid)), weights, priors, audits
    eb = RollingEmpiricalBayesWeighter().compute_weights(
        cross_asset_prior=priors,
        df_grid=grid,
        unit_weights_on_grid=weights,
        fee_rate=spec.fee_rate,
    )
    target = synthesizer.synthesize(
        grid,
        weights,
        eb,
        spec,
        soft_gate=False,
        state_machine=False,
        rebalance_deadzone=0.0,
    )
    target[pd.to_datetime(grid["dt"]).to_numpy() <= pd.Timestamp(is_end_dt).to_datetime64()] = 0.0
    if enable_hysteresis_runner:
        target = govern_position_with_hysteresis_runner(grid, target, base_tf=spec.base_tf)
    else:
        target = ContinuousSignalStateMachine().transform(target)
    target = synthesizer._apply_execution_mode_adapter(grid, target, spec.execution_mode)
    return target, weights, {uid: float(eb[uid][-1]) for uid in priors}, audits


def main() -> None:
    CTA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 125)
    print("  【跨资产规则型多因子 CTA 策略框架 · 数据底座 + 规则因子三级法医门禁 + 时序×截面分层合成实证】")
    print(
        "  覆盖资产族：1. 股指衍生标的 (IC/510500) | 2. 日内T+0宽基ETF (159915) | 3. 商品期货主力连续 (SHFE.ag 含夜盘)"
    )
    print("=" * 125)

    data_engine = CrossAssetDataEngine(cache_dir=WORKSPACE_DIR)
    intra_synth = IntraAssetThreePillarSynthesizer()
    hrp_synth = CrossAssetHRPSynthesizer(target_portfolio_vol=0.14)

    # 加载精选规则因子单元配置（缺失本地缓存时自动回退至内置种子因子池）
    champ_alloc: dict[str, float] = load_candidate_alloc_map()

    universe = build_default_cross_asset_universe()
    pool_frames = load_research_universe(data_engine, universe)
    asset_results: dict[str, dict] = {}
    asset_daily_returns: dict[str, pd.Series] = {}
    execution_returns, execution_positions, execution_strengths, execution_fees = {}, {}, {}, {}
    asset_sectors: dict[str, str] = {}
    cross_asset_30m_weights: dict[str, pd.Series] = {}

    # 跨品种规则因子法医普适性审计表
    core_universal_factors = ["S1L@30m", "S1S@30m", "S2L@60m", "B02L@30m", "C03L@60m", "D04L@30m", "TD03S@30m"]
    universal_audit_book: dict[str, dict[str, dict]] = {u: {} for u in core_universal_factors}

    for spec in universe:
        print(
            f"\n[1. 加载与多周期对齐] {spec.name} ({spec.symbol}) | 基准网格={spec.base_tf} | 夜盘={spec.has_night_session}"
        )
        tf_dfs = pool_frames[spec.symbol]
        df_grid = tf_dfs[spec.base_tf]
        w_synth, w_on_grid, active_alloc, audits = synthesize_admitted_asset(
            spec,
            pool_frames,
            sorted(champ_alloc),
            intra_synth,
        )
        for uid in core_universal_factors:
            if uid in audits:
                universal_audit_book[uid][spec.symbol] = audits[uid]["gate1b"]
        failures = {uid: value["reason"] for uid, value in audits.items() if not value["passed"]}
        print(f"  -> 统一准入: 通过 {len(active_alloc)} / 审核 {len(audits)}；不足证据的资产池拒绝准入")
        if failures:
            (CTA_OUTPUT_DIR / (spec.symbol.replace("/", "_") + "_admission.json")).write_text(
                json.dumps(
                    {uid: audit for uid, audit in audits.items() if not audit["passed"]}, ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
        execution_returns[spec.symbol], execution_positions[spec.symbol] = build_asset_execution_inputs(
            df_grid, w_synth
        )
        execution_fees[spec.symbol] = spec.fee_rate
        if "cs_strength" in df_grid:
            execution_strengths[spec.symbol] = (
                pd.Series(df_grid["cs_strength"].to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(df_grid["dt"])))
                .shift(1)
                .reindex(execution_returns[spec.symbol].index)
                .ffill()
                .fillna(0.0)
            )
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

    # 2. 打印跨品种 Gate 1B 普适率与 Gate 2C 横截面并发度 CSC 审计
    print("\n" + "=" * 125)
    print("  【二、规则型因子跨品种普适率 (Gate 1B) 与横截面并发拥挤度 (Gate 2C · CSC) 审计】")
    print("=" * 125)
    u_rows = []
    for uid, sym_map in universal_audit_book.items():
        for primary_symbol, pooled in sym_map.items():
            # These are the exact pre-admission results; re-pooling summaries would
            # lose event timestamps and mix the candidate's sector/trading regime.
            u_rows.append(
                {
                    "规则因子单元": uid,
                    "目标资产": primary_symbol,
                    "检验池": pooled["pool_tier"],
                    "资产数": pooled["pool_size"],
                    "跨品种总事件(IS)": pooled["total_pool_is_n"],
                    "独立时间块": pooled["independent_time_blocks"],
                    "时间块t统计": pooled["pooled_t_stat"],
                    "正期望品种占比": f"{pooled['universal_ratio_pct']:.1f}%",
                    "正期望覆盖标的": ", ".join(pooled["positive_assets"]),
                    "普适性门禁": "PASS" if pooled["passed"] else "REJECT",
                }
            )
    print(pd.DataFrame(u_rows).to_string(index=False))

    csc_stats = compute_cross_sectional_concurrency(cross_asset_30m_weights)
    print(
        f"\n  [横截面并发拥挤度 CSC 审计] 三大资产族日均同向激活并发率={csc_stats['mean_csc_pct']:.1f}%, "
        f"P95分位={csc_stats['p95_csc_pct']:.1f}%（并发率为诊断指标，不单独证明分散化收益）"
    )

    # 3. 跨品种逆波动率 + 板块分层风险平价 (Sector-HRP) 组合构建
    hrp_res = hrp_synth.build_multi_asset_portfolio(
        execution_returns,
        asset_sectors,
        asset_positions_df=execution_positions,
        asset_cs_strength_df=execution_strengths,
        asset_fee_rates=execution_fees,
        returns_kind="price",
        positions_timing="execution",
        periods_per_year=252 * 48,
    )
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
            "挂载规则数": sum(res["n_units"] for res in asset_results.values()),
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
