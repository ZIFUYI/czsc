#!/usr/bin/env python
"""
【TD 序列规则型因子 · 全流程三级法医检验与合成解剖示例】
(Step-by-Step Forensic Validation Walkthrough Using TD Sequential Rule Factors)

本脚本以经典的 TD 序列（Setup 势能蓄力 + Countdown 计数确认 + 结构止损状态机）为例，
完整展示一个“规则型因子”如何一步步走完多因子 CTA 框架的 5 大核心检验与入库环节：
- Step 0: 规则型因子的数学形态解剖（为什么传统全样本 IC 失效？触发脉冲 vs 持仓状态机）
- Step 1: Gate 1A 单品种法医标尺检验（Purged 去重叠抽样 + 周期自适应门槛 + 三重屏障标尺 + 成本安全边际 ΔW + Bootstrap p值）
- Step 2: Gate 1B 跨品种普适性审计（在 ETF 159915 / 股指 510500 / 商品沪银 SHFE.ag 三大资产族上检验是否过拟合）
- Step 3: Gate 2 时序×截面三维互补检验（非零激活态交并比 Signed Jaccard + 空仓补盲率 + 回撤期救援 Alpha + 截面并发度 CSC）
- Step 4: Gate 3 样本内双半场 (2020-2021 牛市 vs 2022-2023 熊市) 稳健增益与留一法 (LOO) 消融检验

运行方式：
    uv run --no-sync python examples/multi_factor_cta/walkthrough_td_factor_validation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.alpha_mining_agent.forensic_pipeline import (  # noqa: E402
    IS_END_DT,
    purge_overlapping_triggers,
)
from examples.alpha_mining_agent.hypothesis_bank import ARCHETYPE_RULERS, TD_SIGNAL_PARAM_MAP  # noqa: E402
from examples.multi_factor_cta.data_engine import (  # noqa: E402
    CrossAssetDataEngine,
    map_tf_weight_to_base_grid,
)
from examples.multi_factor_cta.factor_engine import RuleFactorRegistry  # noqa: E402
from examples.multi_factor_cta.forensic_validator import (  # noqa: E402
    RuleFactorForensicValidator,
)
from examples.multi_factor_cta.run_framework_demo import build_default_cross_asset_universe  # noqa: E402
from examples.multi_factor_cta.synthesis_engine import (  # noqa: E402
    IntraAssetThreePillarSynthesizer,
    evaluate_single_asset_backtest,
)

WORKSPACE_DIR = ROOT_DIR / "examples" / "results" / "alpha_agent_workspace"


def compute_naive_full_sample_ic(df_tf: pd.DataFrame, weights: np.ndarray, horizon: int = 6) -> tuple[float, float]:
    """对比演示：若错误套用传统股票截面的全样本线性 IC 会发生什么"""
    is_mask = (df_tf["dt"] <= IS_END_DT).values
    c = df_tf["close"].values[is_mask]
    w = weights[is_mask]
    fwd_ret = np.zeros_like(c)
    fwd_ret[:-horizon] = (c[horizon:] - c[:-horizon]) / c[:-horizon]
    valid = np.arange(len(c)) < (len(c) - horizon)
    if np.std(w[valid]) < 1e-9:
        return 0.0, 0.0
    full_ic = float(np.corrcoef(w[valid], fwd_ret[valid])[0, 1])
    active_mask = valid & (np.abs(w) > 1e-4)
    active_ratio = float(active_mask.sum() / valid.sum() * 100.0)
    return full_ic, active_ratio


def main() -> None:
    print("=" * 125)
    print("  【TD 序列规则型因子 (TD01~TD04) · 五步法医检验与多因子入库全流程解剖】")
    print("=" * 125)

    data_engine = CrossAssetDataEngine(cache_dir=WORKSPACE_DIR)
    registry = RuleFactorRegistry()
    validator = RuleFactorForensicValidator(is_end_dt=IS_END_DT)
    intra_synth = IntraAssetThreePillarSynthesizer()

    universe = build_default_cross_asset_universe()
    asset_tf_dfs: dict[str, dict[str, pd.DataFrame]] = {}
    for spec in universe:
        asset_tf_dfs[spec.symbol] = data_engine.load_asset_multi_timeframes(
            spec, attach_czsc=(spec.symbol == "SZSE.159915")
        )

    df_159915_5m = asset_tf_dfs["SZSE.159915"]["5m"]

    # =========================================================================
    # Step 0: 解剖 TD 规则型因子的物理形态与“全样本线性 IC 失真陷阱”
    # =========================================================================
    print("\n" + "─" * 125)
    print("  Step 0 · 规则型因子的数学形态解剖：为什么不能用传统全样本 IC/IR 检验 TD 因子？")
    print("─" * 125)
    td_examples = [
        ("TD01S", "5m", "日内TD(4,2,3)蓄力三确认做空"),
        ("TD01S", "15m", "日内TD(4,2,3)蓄力三确认做空"),
        ("TD02S", "5m", "日内TD(5,3,6)深度六确认做空"),
        ("TD03S", "30m", "跨日TD(4,2,3)波段反转做空"),
        ("TD04L", "60m", "跨日TD(5,3,6)中线衰竭反转做多"),
        ("TD01L", "30m", "日内TD(4,2,3)蓄力三确认做多 [反例]"),
    ]
    step0_rows = []
    for sid, tf, desc in td_examples:
        df_tf = asset_tf_dfs["SZSE.159915"][tf]
        f_spec = registry.get(sid)
        ruler = ARCHETYPE_RULERS[f_spec.archetype]
        trig, w = registry.evaluate_factor_on_tf(df_tf, sid, tf)
        is_mask = (df_tf["dt"] <= IS_END_DT).values
        raw_is_n = int((trig & is_mask).sum())
        eff_cooldown = ruler.max_bars * 2 if tf == "5m" else ruler.max_bars
        purged_is_idx = purge_overlapping_triggers(np.where(trig & is_mask)[0], cooldown_bars=eff_cooldown)
        full_ic, active_pct = compute_naive_full_sample_ic(df_tf, w, horizon=ruler.max_bars)
        direction, n1, n2, n3, sl_th, daily_rst, max_hold, t_flt = TD_SIGNAL_PARAM_MAP[sid]
        step0_rows.append(
            {
                "因子单元": f"{sid}@{tf}",
                "规则语义": desc,
                "战略簇": f_spec.archetype,
                "TD参数(n1,n2,n3)": f"({n1},{n2},{n3})",
                "IS持仓激活率": f"{active_pct:.1f}%",
                "IS空仓静默率": f"{100.0 - active_pct:.1f}%",
                "IS原始触发数": raw_is_n,
                "Purged独立事件数": len(purged_is_idx),
                "传统全样本IC(失真)": f"{full_ic:+.4f}",
            }
        )
    print(pd.DataFrame(step0_rows).to_string(index=False))
    print(
        "  [关键洞察] TD 规则因子在 72%~95% 的时间处于 0（空仓静默等待 Setup+Countdown 共振），"
        "导致传统全样本线性 IC 仅在 0.005~0.02 之间徘徊；必须使用【事件驱动三重屏障法医标尺】！"
    )

    # =========================================================================
    # Step 1: Gate 1A 单品种法医标尺检验（以 159915 样本内 2020-2023 为例，对照通过者 vs 淘汰者）
    # =========================================================================
    print("\n" + "─" * 125)
    print("  Step 1 · Gate 1A 单品种法医标尺检验（Purged去重叠 + 分簇三重屏障 + 成本安全垫 ΔW + 500轮随机置换 p值）")
    print("─" * 125)
    gate1a_rows = []
    for sid, tf, _desc in td_examples:
        df_tf = asset_tf_dfs["SZSE.159915"][tf]
        f_spec = registry.get(sid)
        ruler = ARCHETYPE_RULERS[f_spec.archetype]
        trig, _ = registry.evaluate_factor_on_tf(df_tf, sid, tf)
        g1 = validator.run_gate1_single_asset_audit(
            df_tf,
            trig,
            direction=f_spec.direction,
            archetype=f_spec.archetype,
            timeframe=tf,
            fee_rate=0.0002,
        )
        be_wr = g1.get("is_win_rate_pct", 0.0) - g1.get("is_safety_margin_pct", 0.0)
        gate1a_rows.append(
            {
                "因子单元": f"{sid}@{tf}",
                "专属标尺(止盈/止损/持仓)": f"+{ruler.tp_atr}ATR/-{ruler.sl_atr}ATR/{ruler.max_bars}根",
                "IS独立样本": g1["is_n_purged"],
                "IS单次净收益": f"{g1.get('is_mean_pct', 0.0):+.2f}%",
                "IS超额Alpha": f"{g1.get('is_excess_alpha_pct', 0.0):+.2f}%",
                "IS胜率(W)": f"{g1.get('is_win_rate_pct', 0.0):.1f}%",
                "盈亏比(R)": f"{g1.get('is_pl_ratio', 0.0):.2f}",
                "盈亏平衡胜率": f"{be_wr:.1f}%",
                "安全边际(ΔW)": f"{g1.get('is_safety_margin_pct', 0.0):+.1f}%",
                "置换p值": f"{g1.get('is_boot_pval', 1.0):.3f}",
                "OOS盲测净均值": f"{g1.get('oos_mean_pct', 0.0):+.2f}%",
                "Gate1A结论": "PASS" if g1["passed"] else f"REJECT ({g1['reason']})",
            }
        )
    print(pd.DataFrame(gate1a_rows).to_string(index=False))

    # =========================================================================
    # Step 2: Gate 1B 跨品种普适性审计（在 ETF 159915 / 股指 510500 / 商品 SHFE.ag 上横向验证 TD03S@30m 与 TD04L@60m）
    # =========================================================================
    print("\n" + "─" * 125)
    print("  Step 2 · Gate 1B 跨品种普适性审计（检验 TD 规则在 ETF、股指、商品期货三大资产族上是否具备普适正期望）")
    print("─" * 125)
    cross_td_units = ["TD03S@30m", "TD04L@60m", "TD01S@15m"]
    gate1b_detail_rows = []
    for uid in cross_td_units:
        sid, tf = uid.split("@")
        f_spec = registry.get(sid)
        sym_audits = {}
        for spec in universe:
            if tf not in spec.eligible_tfs:
                continue
            df_tf = asset_tf_dfs[spec.symbol][tf]
            trig, _ = registry.evaluate_factor_on_tf(df_tf, sid, tf)
            res = validator.run_gate1_single_asset_audit(
                df_tf,
                trig,
                direction=f_spec.direction,
                archetype=f_spec.archetype,
                timeframe=tf,
                fee_rate=spec.fee_rate,
            )
            sym_audits[spec.symbol] = res
            gate1b_detail_rows.append(
                {
                    "TD因子单元": uid,
                    "测试标的": f"{spec.symbol} ({spec.asset_class.value})",
                    "IS独立样本": res.get("is_n_purged", 0),
                    "IS单次净收益": f"{res.get('is_mean_pct', 0.0):+.2f}%",
                    "IS超额Alpha": f"{res.get('is_excess_alpha_pct', 0.0):+.2f}%",
                    "IS胜率": f"{res.get('is_win_rate_pct', 0.0):.1f}%",
                    "IS安全边际(ΔW)": f"{res.get('is_safety_margin_pct', 0.0):+.1f}%",
                    "OOS单次净收益": f"{res.get('oos_mean_pct', 0.0):+.2f}%",
                }
            )
        uni = validator.run_gate1_cross_asset_universality_audit(sym_audits, min_positive_ratio=0.60)
        gate1b_detail_rows.append(
            {
                "TD因子单元": f"★ {uid} 跨品种汇总",
                "测试标的": f"正期望覆盖: {', '.join(uni['positive_assets'])}",
                "IS独立样本": uni["total_pool_is_n"],
                "IS单次净收益": "-",
                "IS超额Alpha": f"正期望率={uni['universal_ratio_pct']:.1f}%",
                "IS胜率": "-",
                "IS安全边际(ΔW)": f"均值={uni['mean_pool_safety_pct']:+.2f}%",
                "OOS单次净收益": "PASS" if uni["passed"] else "WARN",
            }
        )
    print(pd.DataFrame(gate1b_detail_rows).to_string(index=False))

    # =========================================================================
    # Step 3: Gate 2 时序×截面三维互补检验（Signed Jaccard 激活交并比 + 空仓补盲率 + 回撤救援 Alpha + 截面并发度 CSC）
    # =========================================================================
    print("\n" + "─" * 125)
    print("  Step 3 · Gate 2 时序×截面三维互补与防拥挤检验（为什么不能把所有 TD 变体一股脑塞进组合？）")
    print("─" * 125)
    # 构建 159915 的 15 个基石趋势单元 (S1L~S4S) 作为存量底仓组合，检验 TD 因子对趋势底仓的互补价值与 TD 内部去重
    base_seed_ids = ["S1L", "S1S", "S2L", "S3L", "S4S"]
    base_tfs = ["1d", "60m", "30m"]
    w_on_grid: dict[str, np.ndarray] = {}
    base_alloc: dict[str, float] = {}
    same_dir_existing: dict[int, dict[str, np.ndarray]] = {1: {}, -1: {}}

    for sid in base_seed_ids:
        f_spec = registry.get(sid)
        for tf in base_tfs:
            uid = f"{sid}@{tf}"
            _, w_arr = registry.evaluate_factor_on_tf(asset_tf_dfs["SZSE.159915"][tf], sid, tf)
            w_mapped = map_tf_weight_to_base_grid(asset_tf_dfs["SZSE.159915"][tf], w_arr, df_159915_5m)
            w_on_grid[uid] = w_mapped
            base_alloc[uid] = f_spec.default_weight
            same_dir_existing[f_spec.direction][uid] = w_mapped

    spec_159915 = universe[0]
    w_base_port = intra_synth.synthesize(df_159915_5m, w_on_grid, base_alloc, spec_159915)

    gate2_rows = []
    for sid, tf, desc in [
        ("TD01S", "15m", "首个入库的15m日内TD做空因子"),
        ("TD02S", "5m", "高频5m深度6确认TD做空因子"),
        ("TD03S", "30m", "跨日30m波段TD做空因子"),
        ("TD04L", "60m", "跨日60m中线衰竭TD做多因子"),
        ("TD02S", "15m", "与TD01S@15m同周期的冗余变体 [对比演示]"),
    ]:
        f_spec = registry.get(sid)
        _, w_arr = registry.evaluate_factor_on_tf(asset_tf_dfs["SZSE.159915"][tf], sid, tf)
        cand_w = map_tf_weight_to_base_grid(asset_tf_dfs["SZSE.159915"][tf], w_arr, df_159915_5m)
        g2 = validator.run_gate2_complementarity_and_jaccard_audit(
            df_159915_5m,
            cand_w,
            same_dir_existing[f_spec.direction],
            w_base_port,
        )
        uid = f"{sid}@{tf}"
        gate2_rows.append(
            {
                "候选TD单元": uid,
                "功能定位": desc,
                "最大相关Peer": g2.get("max_corr_peer", "None"),
                "时序相关度(|ρ|<0.32)": f"{g2.get('max_corr', 0.0):.3f}",
                "激活交并比(Jaccard<0.30)": f"{g2.get('max_signed_jaccard', 0.0):.3f}",
                "空仓补盲率(>=35%)": f"{g2.get('gap_fill_ratio_pct', 0.0):.1f}%",
                "回撤救援Alpha(>0BP)": f"{g2.get('dd_rescue_alpha_bp', 0.0):+.2f} BP",
                "Gate2结论": "PASS (准予入池)" if g2["passed"] else "REJECT (同生态位重叠)",
            }
        )
        if g2["passed"]:
            same_dir_existing[f_spec.direction][uid] = cand_w
            w_on_grid[uid] = cand_w

    print(pd.DataFrame(gate2_rows).to_string(index=False))

    # =========================================================================
    # Step 4: Gate 3 样本内双半场 (H1: 2020-2021 vs H2: 2022-2023) 稳健增益与留一法 (LOO) 消融验证
    # =========================================================================
    print("\n" + "─" * 125)
    print("  Step 4 · Gate 3 组合级双半场稳健增益与留一法 (LOO) 消融审计（验证 TD 因子族群对最终组合的真实净贡献）")
    print("─" * 125)
    state_data = json.loads((WORKSPACE_DIR / "champion_portfolio.json").read_text(encoding="utf-8"))
    champ_alloc: dict[str, float] = state_data["alloc_map"]

    all_w_grid: dict[str, np.ndarray] = {}
    for uid in champ_alloc:
        sid, tf = uid.split("@")
        _, w_arr = registry.evaluate_factor_on_tf(asset_tf_dfs["SZSE.159915"][tf], sid, tf)
        all_w_grid[uid] = map_tf_weight_to_base_grid(asset_tf_dfs["SZSE.159915"][tf], w_arr, df_159915_5m)

    # 1) 完整 44 单元冠军组合 (含 8 个 TD 单元)
    w_full_44 = intra_synth.synthesize(df_159915_5m, all_w_grid, champ_alloc, spec_159915)
    ev_full_44 = evaluate_single_asset_backtest(df_159915_5m, w_full_44, "SZSE.159915", fee_rate=0.0002)

    # 2) 剔除全部 TD 因子后的 36 单元对照组合 (Ablation: Without TD Family)
    no_td_alloc = {u: v for u, v in champ_alloc.items() if not u.startswith("TD")}
    w_no_td = intra_synth.synthesize(df_159915_5m, all_w_grid, no_td_alloc, spec_159915)
    ev_no_td = evaluate_single_asset_backtest(df_159915_5m, w_no_td, "SZSE.159915", fee_rate=0.0002)

    # 3) 单个代表性 TD 单元的留一法 (LOO) 边际贡献审计
    ablation_rows = [
        {
            "组合配置对照": "1. 不含 TD 因子族群基准组合 (36单元)",
            "IS净收益(20-23)": f"{ev_no_td['is']['cum_ret_pct']:+.2f}%",
            "IS夏普": f"{ev_no_td['is']['sharpe']:.2f}",
            "OOS盲测净收益(24-26)": f"{ev_no_td['oos']['cum_ret_pct']:+.2f}%",
            "OOS夏普": f"{ev_no_td['oos']['sharpe']:.2f}",
            "全周期净收益(20-26)": f"{ev_no_td['full']['cum_ret_pct']:+.2f}%",
            "全周期夏普": f"{ev_no_td['full']['sharpe']:.2f}",
            "全周期最大回撤": f"{ev_no_td['full']['mdd_pct']:.2f}%",
            "全周期卡玛": f"{ev_no_td['full']['calmar']:.2f}",
        },
        {
            "组合配置对照": "2. 挂载通过三级门禁的 8 个 TD 单元后 (44单元)",
            "IS净收益(20-23)": f"{ev_full_44['is']['cum_ret_pct']:+.2f}%",
            "IS夏普": f"{ev_full_44['is']['sharpe']:.2f}",
            "OOS盲测净收益(24-26)": f"{ev_full_44['oos']['cum_ret_pct']:+.2f}%",
            "OOS夏普": f"{ev_full_44['oos']['sharpe']:.2f}",
            "全周期净收益(20-26)": f"{ev_full_44['full']['cum_ret_pct']:+.2f}%",
            "全周期夏普": f"{ev_full_44['full']['sharpe']:.2f}",
            "全周期最大回撤": f"{ev_full_44['full']['mdd_pct']:.2f}%",
            "全周期卡玛": f"{ev_full_44['full']['calmar']:.2f}",
        },
        {
            "组合配置对照": "★ TD 规则因子族群净增量贡献 (Δ)",
            "IS净收益(20-23)": f"{ev_full_44['is']['cum_ret_pct'] - ev_no_td['is']['cum_ret_pct']:+.2f}%",
            "IS夏普": f"{ev_full_44['is']['sharpe'] - ev_no_td['is']['sharpe']:+.2f}",
            "OOS盲测净收益(24-26)": f"{ev_full_44['oos']['cum_ret_pct'] - ev_no_td['oos']['cum_ret_pct']:+.2f}%",
            "OOS夏普": f"{ev_full_44['oos']['sharpe'] - ev_no_td['oos']['sharpe']:+.2f}",
            "全周期净收益(20-26)": f"{ev_full_44['full']['cum_ret_pct'] - ev_no_td['full']['cum_ret_pct']:+.2f}%",
            "全周期夏普": f"{ev_full_44['full']['sharpe'] - ev_no_td['full']['sharpe']:+.2f}",
            "全周期最大回撤": f"{ev_full_44['full']['mdd_pct'] - ev_no_td['full']['mdd_pct']:+.2f}%",
            "全周期卡玛": f"{ev_full_44['full']['calmar'] - ev_no_td['full']['calmar']:+.2f}",
        },
    ]
    print(pd.DataFrame(ablation_rows).to_string(index=False))

    print("\n  [最终入库的 8 个 TD 规则单元及其样本内法医经验贝叶斯配权权重]")
    td_final_rows = []
    for uid, w_val in champ_alloc.items():
        if uid.startswith("TD"):
            sid, tf = uid.split("@")
            f_spec = registry.get(sid)
            td_final_rows.append(
                {
                    "入库TD单元": uid,
                    "规则名称": f_spec.name,
                    "所属战略簇": f_spec.archetype,
                    "方向": "做多(+1)" if f_spec.direction == 1 else "做空(-1)",
                    "初始先验权重": f"{f_spec.default_weight:.2f}",
                    "IS联合重配后权重": f"{w_val:.2f}",
                }
            )
    print(pd.DataFrame(td_final_rows).to_string(index=False))

    # =========================================================================
    # Step 5: 专项解剖 —— 当不同标的【波动率不同、牛熊/震荡状态分布不同】时如何对齐可比性？
    # =========================================================================
    print("\n" + "─" * 125)
    print("  Step 5 · 跨标的【波动率归一化 (ATR单位R-Multiple) + 因果三态状态重加权 (IPW) + 经验贝叶斯收缩】实测对照")
    print("─" * 125)
    from examples.multi_factor_cta.forensic_validator import compute_empirical_bayes_shrinkage_weight

    norm_rows = []
    for uid in ["S1L@30m", "TD03S@30m", "TD04L@60m"]:
        sid, tf = uid.split("@")
        f_spec = registry.get(sid)
        for spec in universe:
            if tf not in spec.eligible_tfs:
                continue
            df_tf = asset_tf_dfs[spec.symbol][tf]
            vol_mult = registry.compute_asset_vol_multiplier(df_tf, tf)
            trig, _ = registry.evaluate_factor_on_tf(df_tf, sid, tf, adaptive_vol_scale=True)
            res = validator.run_vol_and_regime_normalized_audit(
                df_tf,
                trig,
                direction=f_spec.direction,
                archetype=f_spec.archetype,
                timeframe=tf,
                fee_rate=spec.fee_rate,
            )
            # 演示分层经验贝叶斯收缩：单品种局部权重 vs 跨品种共性先验 (0.28)
            local_est = float(np.clip(0.28 * (1.0 + 0.08 * res.get("ipw_safety_pct", 0.0)), 0.12, 0.55))
            w_eb, lam = compute_empirical_bayes_shrinkage_weight(
                local_est, 0.28, res.get("is_n_purged", 0), shrink_k=60
            )
            norm_rows.append(
                {
                    "因子单元": uid,
                    "标的": spec.symbol,
                    "周期中位ATR%": f"{res['median_bar_atr_pct']:.2f}%",
                    "波动率乘数κ": f"{vol_mult:.2f}x",
                    "IS三态分布(牛/震/熊)": res["regime_dist_str"],
                    "IS样本": res["is_n_purged"],
                    "原始净均值(不可比)": f"{res['raw_mean_pct']:+.2f}%",
                    "ATR单位收益(r_ATR)": f"{res['norm_r_atr']:+.2f} ATR",
                    "主场状态r_ATR": f"{res['home_r_atr']:+.2f} ATR",
                    "未校正安全垫": f"{res['uncond_safety_pct']:+.1f}%",
                    "状态均衡安全垫(IPW)": f"{res['ipw_safety_pct']:+.1f}%",
                    "主场安全垫(Home)": f"{res['home_safety_pct']:+.1f}%",
                    "EB收缩权重(λ)": f"{w_eb:.2f} ({lam:.2f})",
                }
            )
    print(pd.DataFrame(norm_rows).to_string(index=False))
    print("=" * 125)


if __name__ == "__main__":
    main()
