#!/usr/bin/env python
"""
159915 长期自主信号挖掘与 4×4 生态位矩阵进化 Agent 主循环
(MAP-Elites Niche Matrix + Archetype-Specific Rulers + 4-Timeframe Atomic Routing)

核心特性：
1. 搜索单元原子化为 `sig_id@tf` (跨越 1d / 60m / 30m / 15m 四级周期)，哪个信号在哪个周期过关就只挂载在哪个周期；
2. 四大战略方向配置专属法医三重屏障检验尺 (趋势尺 / 反转尺 / 回归尺 / 日内尺)，彻底消除对反转与日内信号的误杀；
3. 三维真互补门禁 (生态位配额防扎堆 + 持仓时序低相关 |ρ|<0.32 + 空仓填补率 & 回撤期雪中送炭收益 > 0)；
4. 持久态存储 signal_zoo.json / niche_matrix.json / graveyard.jsonl / leaderboard.md。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from examples.alpha_mining_agent.forensic_pipeline import (
    IS_END_DT,
    backtest_slice,
    map_tf_weight_to_30m_grid,
    run_gate1_forensic_audit,
    run_gate2_3d_complementarity_audit,
    synthesize_atomic_units_portfolio,
)
from examples.alpha_mining_agent.hypothesis_bank import (
    ALL_SIGNAL_SPECS,
    ARCHETYPE_RULERS,
    BASE_SEED_SIGNALS,
    CANDIDATE_HYPOTHESIS_POOL,
    TF_SCALE_MAP,
    attach_native_czsc_signals,
    compute_signal_triggers_and_weights,
    enrich_kline_features,
)
from examples.alpha_mining_agent.state_ledger import (
    WORKSPACE_DIR,
    AdmittedSignalRecord,
    AgentStateLedger,
)

REAL_5M_CSV = Path("examples/results/159915_real_5m_2020_2026.csv")
DASHBOARD_HTML = WORKSPACE_DIR / "autonomous_evolution_dashboard.html"


def load_all_four_timeframes(csv_path: Path) -> dict[str, pd.DataFrame]:
    df_raw = pd.read_csv(csv_path)
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
    df_raw = (
        df_raw.rename(
            columns={
                "SZSE.159915.open": "open",
                "SZSE.159915.high": "high",
                "SZSE.159915.low": "low",
                "SZSE.159915.close": "close",
                "SZSE.159915.volume": "volume",
            }
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    df_raw["date"] = df_raw["datetime"].dt.date
    df_raw["bar_in_day"] = df_raw.groupby("date").cumcount()

    def _agg(bars_per_group: int, tf: str) -> pd.DataFrame:
        scale = TF_SCALE_MAP[tf]
        tmp = df_raw.copy()
        tmp["grp"] = tmp["bar_in_day"] // bars_per_group
        kdf = tmp.groupby(["date", "grp"], as_index=False).agg(
            dt=("datetime", "last"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        kdf["symbol"] = "159915"
        kdf = enrich_kline_features(kdf, scale=scale)
        return attach_native_czsc_signals(kdf, tf, WORKSPACE_DIR)

    return {
        "1d": _agg(48, "1d"),
        "60m": _agg(12, "60m"),
        "30m": _agg(6, "30m"),
        "15m": _agg(3, "15m"),
    }


def evaluate_full_is_oos(df_30m: pd.DataFrame, w_multi: np.ndarray) -> dict:
    is_mask = (df_30m["dt"] <= IS_END_DT).values
    oos_mask = (df_30m["dt"] > IS_END_DT).values

    st_full = backtest_slice(df_30m, w_multi)
    st_is = backtest_slice(df_30m.iloc[is_mask].reset_index(drop=True), w_multi[is_mask])
    st_oos = backtest_slice(df_30m.iloc[oos_mask].reset_index(drop=True), w_multi[oos_mask])
    return {"full": st_full, "is": st_is, "oos": st_oos}


def build_niche_matrix_summary(zoo: dict[str, dict]) -> pd.DataFrame:
    archetypes = ["A_顺势突破簇", "B_背驰反转簇", "C_震荡回归簇", "D_日内微结构簇"]
    tfs = ["1d", "60m", "30m", "15m"]
    grid = {arch: {tf: [] for tf in tfs} for arch in archetypes}

    for _uid, info in zoo.items():
        arch = info["archetype"]
        tf = info["timeframe"]
        tag = f"{info['sig_id']}({'种子' if info['is_base_seed'] else '新挖'})"
        if arch in grid and tf in grid[arch]:
            grid[arch][tf].append(tag)

    rows = []
    for arch in archetypes:
        r = {"战略方向 (Archetype)": arch}
        total_u = 0
        new_u = 0
        for tf in tfs:
            items = grid[arch][tf]
            total_u += len(items)
            new_u += sum(1 for x in items if "新挖" in x)
            r[f"{tf} 周期生态位"] = ", ".join(items) if items else "—"
        r["总单元数 (含新挖)"] = f"{total_u} ({new_u}新)"
        rows.append(r)
    return pd.DataFrame(rows)


def _df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join(":---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(row[c]) for c in cols) + " |" for _, row in df.iterrows()]
    return "\n".join([header, sep, *body])


def generate_markdown_leaderboard(ledger: AgentStateLedger, niche_df: pd.DataFrame) -> None:
    zoo = ledger.load_signal_zoo()
    grave = ledger.load_graveyard_records()
    hist = ledger.load_evolution_history()

    new_units = [u for u, v in zoo.items() if not v["is_base_seed"]]
    lines = [
        "# 159915 自主信号挖掘 Agent · 4×4 生态位矩阵与实时排行榜 (MAP-Elites Signal Zoo)",
        "",
        f"- **已入库原子信号单元总数 (`sig_id@tf`)**: `{len(zoo)}` 个（含 15 个基石种子单元 + **`{len(new_units)}` 个 Agent 跨四大战略方向与四周期新挖正交单元**）",
        f"- **已证伪/拦截进入墓地的原子单元数**: `{len(grave)}` 个",
        "",
        "## 一、4×4 生态位矩阵分布表 (4 大战略方向 × 4 个独立周期)",
        "",
        _df_to_md_table(niche_df),
        "",
        "## 二、Agent 新挖掘入库的正交真金原子单元明细",
        "",
        "| 原子单元ID | 信号名称 | 战略方向 | 专属周期 | 专属检验标尺 | 样本内独立触发 | 样本内超额Alpha | 样本内安全垫 | 最大相关性 | 空仓填补率 | 回撤雪中送炭(BP) | 样本外超额Alpha |",
        "| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for uid in new_units:
        rec = zoo[uid]
        lines.append(
            f"| `{uid}` | **{rec['name']}** | {rec['archetype']} | `{rec['timeframe']}` | {rec['ruler_desc'].split(' ')[0]} | "
            f"{rec['is_n_purged']} | **{rec['is_excess_alpha_pct']:+.2f}%** | **{rec['is_safety_margin_pct']:+.1f}%** | "
            f"{rec['max_corr_with_zoo']:.2f} (`{rec['max_corr_peer']}`) | {rec['gap_fill_ratio_pct']:.1f}% | "
            f"**{rec['dd_rescue_alpha_bp']:+.1f} BP** | **{rec['oos_excess_alpha_pct']:+.2f}%** |"
        )

    lines.extend(
        [
            "",
            "## 三、组合进化里程碑轨迹 (`evolution_history.jsonl`)",
            "",
            "| 进化轮次 | 新并入原子单元 (`sig@tf`) | 所属方向 | 全周期总笔数 (年均) | 资金在场率 | 样本内夏普 (IS) | 样本外累计收益 (OOS) | 样本外夏普 (OOS) | 全周期累计收益 | 全周期夏普 | 全周期卡玛 |",
            "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for h in hist:
        lines.append(
            f"| Step {h['step']} | **{h['added_unit']}** | {h['archetype']} | "
            f"**{h['full_trades']} 笔** ({h['trades_per_yr']}笔/年) | **{h['full_active_pct']:.1f}%** | "
            f"{h['is_sharpe']:.2f} | **{h['oos_cum_ret_pct']:+.2f}%** | **{h['oos_sharpe']:.2f}** | "
            f"**{h['full_cum_ret_pct']:+.2f}%** | **{h['full_sharpe']:.2f}** | **{h['full_calmar']:.2f}** |"
        )

    ledger.leaderboard_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_evolution_dashboard(
    df_1d: pd.DataFrame,
    base_eval: dict,
    champ_eval: dict,
    history_rows: list[dict],
    out_html: Path,
) -> None:
    nav_base = base_eval["full"]["nav_series"]
    dd_base = base_eval["full"]["dd_series"]
    nav_champ = champ_eval["full"]["nav_series"]
    dd_champ = champ_eval["full"]["dd_series"]

    bench_ret = df_1d.set_index(pd.to_datetime(df_1d["dt"].dt.date))["close"].pct_change().fillna(0.0)
    common_idx = nav_base.index.intersection(nav_champ.index).intersection(bench_ret.index)

    nav_base = nav_base.loc[common_idx]
    dd_base = dd_base.loc[common_idx]
    nav_champ = nav_champ.loc[common_idx]
    dd_champ = dd_champ.loc[common_idx]
    bench_nav = (1.0 + bench_ret.loc[common_idx]).cumprod()
    bench_dd = (bench_nav - bench_nav.cummax()) / bench_nav.cummax() * 100.0

    dates_str = common_idx.strftime("%Y-%m-%d").tolist()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=(
            "【159915 4×4生态位矩阵 Agent】 进化前后四周期复合净值对比 (左侧 2020-2023 样本内 | 右侧 2024-2026 样本外盲测)",
            "动态回撤深度对比 (Drawdown %)",
            "Agent 跨四大战略方向与四周期 (1d/60m/30m/15m) 自主进化阶梯轨迹",
        ),
        row_heights=[0.48, 0.22, 0.30],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]],
    )

    fig.add_trace(
        go.Scatter(
            x=dates_str,
            y=bench_nav.values,
            name="159915 买入持有基准",
            line={"color": "#90A4AE", "width": 1.3, "dash": "dot"},
            opacity=0.6,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates_str,
            y=nav_base.values,
            name="进化前：基石 5真金策略 (仅顺势突破簇)",
            line={"color": "#29B6F6", "width": 2.2},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates_str,
            y=nav_champ.values,
            name=f"★ 进化后：4×4生态位矩阵多方向共振策略 (Step {history_rows[-1]['step']})",
            line={"color": "#00E676", "width": 3.4},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates_str,
            y=bench_dd.values,
            name="基准回撤",
            line={"color": "#90A4AE", "width": 1.0, "dash": "dot"},
            opacity=0.3,
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates_str, y=dd_base.values, name="进化前基石回撤", line={"color": "#29B6F6", "width": 1.5}),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates_str, y=dd_champ.values, name="★ 进化后生态位矩阵回撤", line={"color": "#00E676", "width": 1.8}
        ),
        row=2,
        col=1,
    )

    step_labels = [f"S{r['step']}:{r['added_unit'].split('_')[0]}" for r in history_rows]
    step_trades = [r["full_trades"] for r in history_rows]
    step_rets = [r["full_cum_ret_pct"] for r in history_rows]
    step_oos_rets = [r["oos_cum_ret_pct"] for r in history_rows]

    fig.add_trace(
        go.Bar(x=step_labels, y=step_trades, name="全周期总交易笔数 (左轴)", marker_color="rgba(41, 182, 246, 0.45)"),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=step_labels,
            y=step_rets,
            name="全周期累计净收益 % (右轴)",
            mode="lines+markers",
            line={"color": "#00E676", "width": 3.0},
            marker={"size": 9},
        ),
        row=3,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=step_labels,
            y=step_oos_rets,
            name="样本外 (2024-2026 OOS) 净收益 % (右轴)",
            mode="lines+markers",
            line={"color": "#FFD54F", "width": 2.5, "dash": "dash"},
            marker={"size": 8},
        ),
        row=3,
        col=1,
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A0E14",
        plot_bgcolor="#101721",
        title={
            "text": "159915 创业板 ETF · 4×4 生态位矩阵 (趋势+反转+回归+日内 × 1d/60m/30m/15m) 自主进化大屏",
            "font": {"size": 18, "color": "#FFFFFF"},
            "x": 0.02,
            "y": 0.98,
        },
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(0,0,0,0.6)",
        },
        height=920,
    )
    fig.update_yaxes(title_text="累计净值 (NAV)", gridcolor="#263238", row=1, col=1)
    fig.update_yaxes(title_text="回撤 (%)", gridcolor="#263238", row=2, col=1)
    fig.update_yaxes(title_text="总交易笔数", gridcolor="#263238", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="累计收益率 (%)", gridcolor="#263238", row=3, col=1, secondary_y=True)

    fig.write_html(str(out_html), include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="159915 4×4 生态位矩阵自主信号挖掘与进化 Agent")
    parser.add_argument(
        "--reset", action="store_true", help="清空历史账本，从基石种子开始重新运行完整 4×4 生态位进化循环"
    )
    parser.add_argument("--max-candidates", type=int, default=999, help="本轮最多测试多少个原子候选单元")
    parser.add_argument("--status-only", action="store_true", help="仅打印当前持久态信号库与 4×4 生态位矩阵")
    args = parser.parse_args()

    ledger = AgentStateLedger(WORKSPACE_DIR)
    if args.reset:
        ledger.reset_workspace()

    if args.status_only:
        zoo = ledger.load_signal_zoo()
        grave = ledger.load_graveyard_records()
        print(f"当前已入库原子信号单元: {len(zoo)} 个 | 已淘汰墓地原子单元: {len(grave)} 个")
        if zoo:
            print(build_niche_matrix_summary(zoo).to_string(index=False))
        return

    print("=" * 125)
    print("  【159915 4×4 生态位矩阵 · 长期自主信号挖掘与多方向正交进化 Agent】")
    print("  四大战略方向 (趋势突破 / 背驰反转 / 震荡回归 / 日内微结构) × 四级独立周期 (1d / 60m / 30m / 15m)")
    print("=" * 125)

    tf_dfs = load_all_four_timeframes(REAL_5M_CSV)
    df_1d, df_30m = tf_dfs["1d"], tf_dfs["30m"]
    rng = np.random.default_rng(42)

    # 预计算所有 (sig_id, tf) 原子单元在自身周期上的触发序列，以及映射到 30m 主时间轴上的持仓序列
    trig_by_unit: dict[str, np.ndarray] = {}
    w_on_30m_by_unit: dict[str, np.ndarray] = {}

    for sid, spec in ALL_SIGNAL_SPECS.items():
        for tf in spec.eligible_tfs:
            unit_id = f"{sid}@{tf}"
            t_arr, w_arr = compute_signal_triggers_and_weights(tf_dfs[tf], sid, scale=TF_SCALE_MAP[tf])
            trig_by_unit[unit_id] = t_arr
            w_on_30m_by_unit[unit_id] = map_tf_weight_to_30m_grid(tf_dfs[tf], w_arr, df_30m, tf)

    # 1. 初始化 5 个基石信号在 (1d, 60m, 30m) 上的 15 个种子原子单元
    zoo = ledger.load_signal_zoo()
    if not zoo:
        print("\n[初始化] 正在将 5 大基石真金信号在 (1d, 60m, 30m) 上的种子原子单元写入 signal_zoo.json ...")
        ruler_a = ARCHETYPE_RULERS["A_顺势突破簇"]
        for spec in BASE_SEED_SIGNALS:
            for tf in spec.eligible_tfs:
                unit_id = f"{spec.sig_id}@{tf}"
                g1 = run_gate1_forensic_audit(tf_dfs[tf], trig_by_unit[unit_id], spec.direction, ruler_a, tf, rng)
                rec = AdmittedSignalRecord(
                    unit_id=unit_id,
                    sig_id=spec.sig_id,
                    timeframe=tf,
                    name=f"{spec.name}@{tf}",
                    archetype=spec.archetype,
                    ruler_desc=ruler_a.ruler_name,
                    direction=spec.direction,
                    weight_alloc=spec.default_weight,
                    is_base_seed=True,
                    is_n_purged=g1.get("is_n_purged", 0),
                    is_mean_pct=g1.get("is_mean_pct", 0.0),
                    is_excess_alpha_pct=g1.get("is_excess_alpha_pct", 0.0),
                    is_win_rate_pct=g1.get("is_win_rate_pct", 0.0),
                    is_pl_ratio=g1.get("is_pl_ratio", 0.0),
                    is_safety_margin_pct=g1.get("is_safety_margin_pct", 0.0),
                    is_boot_pval=g1.get("is_boot_pval", 0.0),
                    max_corr_with_zoo=0.0,
                    max_corr_peer="Seed",
                    gap_fill_ratio_pct=100.0,
                    dd_rescue_alpha_bp=0.0,
                    marginal_sharpe_delta=0.0,
                    marginal_calmar_delta=0.0,
                    oos_mean_pct=g1.get("oos_mean_pct", 0.0),
                    oos_excess_alpha_pct=g1.get("oos_excess_alpha_pct", 0.0),
                    oos_safety_margin_pct=g1.get("oos_safety_margin_pct", 0.0),
                )
                ledger.save_signal_to_zoo(rec)
        zoo = ledger.load_signal_zoo()

    base_alloc = {f"{s.sig_id}@{tf}": s.default_weight for s in BASE_SEED_SIGNALS for tf in s.eligible_tfs}
    w_base_multi = synthesize_atomic_units_portfolio(df_30m, w_on_30m_by_unit, base_alloc)
    base_eval = evaluate_full_is_oos(df_30m, w_base_multi)

    current_alloc = {uid: float(info["weight_alloc"]) for uid, info in zoo.items()}
    w_curr_multi = synthesize_atomic_units_portfolio(df_30m, w_on_30m_by_unit, current_alloc)
    curr_eval = evaluate_full_is_oos(df_30m, w_curr_multi)

    if not ledger.load_evolution_history():
        ledger.log_evolution_step(
            {
                "step": 0,
                "added_unit": "Base5_基石五真金(仅突破簇)",
                "archetype": "A_顺势突破簇",
                "timeframe": "1d/60m/30m",
                "zoo_size": len(current_alloc),
                "full_trades": curr_eval["full"]["trades"],
                "trades_per_yr": round(curr_eval["full"]["trades"] / 6.77, 1),
                "full_active_pct": curr_eval["full"]["active_pct"],
                "is_sharpe": curr_eval["is"]["sharpe"],
                "is_calmar": curr_eval["is"]["calmar"],
                "oos_cum_ret_pct": curr_eval["oos"]["cum_ret_pct"],
                "oos_ann_ret_pct": curr_eval["oos"]["ann_ret_pct"],
                "oos_sharpe": curr_eval["oos"]["sharpe"],
                "oos_calmar": curr_eval["oos"]["calmar"],
                "full_cum_ret_pct": curr_eval["full"]["cum_ret_pct"],
                "full_ann_ret_pct": curr_eval["full"]["ann_ret_pct"],
                "full_sharpe": curr_eval["full"]["sharpe"],
                "full_mdd_pct": curr_eval["full"]["mdd_pct"],
                "full_calmar": curr_eval["full"]["calmar"],
            }
        )

    # 2. 构建待测试的 (sig_id@tf) 原子候选队列
    graveyard_ids = ledger.load_graveyard_ids()
    atomic_candidates: list[tuple[str, str]] = []
    for spec in CANDIDATE_HYPOTHESIS_POOL:
        for tf in spec.eligible_tfs:
            uid = f"{spec.sig_id}@{tf}"
            if uid not in zoo and uid not in graveyard_ids:
                atomic_candidates.append((spec.sig_id, tf))
    atomic_candidates = atomic_candidates[: args.max_candidates]

    print(
        f"\n[启动 4×4 生态位进化循环] 待检验原子候选单元 (sig@tf): {len(atomic_candidates)} 个 | "
        f"当前基线: 交易 {curr_eval['full']['trades']} 笔, 全周期收益 {curr_eval['full']['cum_ret_pct']:+.2f}%, "
        f"样本外收益 {curr_eval['oos']['cum_ret_pct']:+.2f}%"
    )
    print("-" * 125)

    step_counter = len(ledger.load_evolution_history()) - 1
    admitted_summary_rows = []
    gate_stats = {"Gate1_Rejected": 0, "Gate2_Rejected": 0, "Gate3_Rejected": 0, "Admitted": 0}

    for sid, tf in atomic_candidates:
        spec = ALL_SIGNAL_SPECS[sid]
        ruler = ARCHETYPE_RULERS[spec.archetype]
        unit_id = f"{sid}@{tf}"

        # 统计当前战略方向已新入库的单元数量与同向单边数量 (用于生态位总配额与多空均衡限制)
        curr_arch_new_count = sum(
            1 for info in zoo.values() if info["archetype"] == spec.archetype and not info["is_base_seed"]
        )
        curr_arch_same_dir_count = sum(
            1
            for info in zoo.values()
            if info["archetype"] == spec.archetype
            and not info["is_base_seed"]
            and int(info["direction"]) == spec.direction
        )

        # --- 第一关：Gate 1 分方向专属标尺法医审计 (2020-2023 IS) ---
        g1 = run_gate1_forensic_audit(tf_dfs[tf], trig_by_unit[unit_id], spec.direction, ruler, tf, rng)
        if not g1["passed"]:
            gate_stats["Gate1_Rejected"] += 1
            ledger.bury_rejected_signal(
                unit_id,
                sid,
                tf,
                f"{spec.name}@{tf}",
                spec.archetype,
                spec.direction,
                failed_gate="Gate 1 (专属法医标尺关)",
                reason=g1["reason"],
                metrics=g1,
            )
            continue

        # --- 第二关：Gate 2 三维真互补门禁 (配额 + 低相关 + 空仓补盲 & 回撤雪中送炭) ---
        same_dir_zoo_w = {
            z_uid: w_on_30m_by_unit[z_uid]
            for z_uid, z_info in zoo.items()
            if int(z_info["direction"]) == spec.direction
        }
        g2 = run_gate2_3d_complementarity_audit(
            df_30m,
            w_on_30m_by_unit[unit_id],
            same_dir_zoo_w,
            w_curr_multi,
            archetype=spec.archetype,
            current_archetype_new_count=curr_arch_new_count,
            ruler=ruler,
            current_arch_same_dir_new_count=curr_arch_same_dir_count,
            max_corr_threshold=0.32,
            min_gap_fill_pct=35.0,
        )
        if not g2["passed"]:
            gate_stats["Gate2_Rejected"] += 1
            ledger.bury_rejected_signal(
                unit_id,
                sid,
                tf,
                f"{spec.name}@{tf}",
                spec.archetype,
                spec.direction,
                failed_gate="Gate 2 (三维真互补与配额关)",
                reason=g2["reason"],
                metrics={**g1, **g2},
            )
            continue

        # --- 第三关：Gate 3 专属周期独立挂载与样本内边际增益审计 (2020-2023 IS) ---
        best_trial_w = spec.default_weight
        best_trial_multi = None
        best_trial_eval = None
        best_delta_score = -999.0

        for cand_w in (0.20, 0.28, 0.36):
            t_alloc = dict(current_alloc)
            t_alloc[unit_id] = cand_w
            w_t_multi = synthesize_atomic_units_portfolio(df_30m, w_on_30m_by_unit, t_alloc)
            t_eval = evaluate_full_is_oos(df_30m, w_t_multi)
            d_score = (
                (t_eval["is"]["ann_ret_pct"] - curr_eval["is"]["ann_ret_pct"]) * 0.15
                + (t_eval["is"]["sharpe"] - curr_eval["is"]["sharpe"]) * 1.0
                + (t_eval["is"]["calmar"] - curr_eval["is"]["calmar"]) * 0.4
            )
            if d_score > best_delta_score:
                best_delta_score = d_score
                best_trial_w = cand_w
                best_trial_multi = w_t_multi
                best_trial_eval = t_eval

        trial_eval = best_trial_eval
        w_trial_multi = best_trial_multi
        trial_alloc = dict(current_alloc)
        trial_alloc[unit_id] = best_trial_w

        delta_trades_is = trial_eval["is"]["trades"] - curr_eval["is"]["trades"]
        delta_active_is = trial_eval["is"]["active_pct"] - curr_eval["is"]["active_pct"]
        delta_ret_is = trial_eval["is"]["cum_ret_pct"] - curr_eval["is"]["cum_ret_pct"]
        delta_mdd_is = trial_eval["is"]["mdd_pct"] - curr_eval["is"]["mdd_pct"]
        delta_sharpe_is = round(trial_eval["is"]["sharpe"] - curr_eval["is"]["sharpe"], 3)
        delta_calmar_is = round(trial_eval["is"]["calmar"] - curr_eval["is"]["calmar"], 3)

        min_delta_ret = 0.05 if tf == "15m" else 0.25
        passed_g3 = (
            (delta_trades_is >= 1 or delta_active_is >= 0.15)
            and (
                delta_ret_is >= min_delta_ret
                or (delta_ret_is >= 0.15 and delta_mdd_is <= 0.10 and best_delta_score >= -0.02)
            )
            and best_delta_score >= -0.04
            and delta_mdd_is <= 0.80
        )
        if not passed_g3:
            gate_stats["Gate3_Rejected"] += 1
            g3_reason = (
                f"样本内边际收益变化={delta_ret_is:+.2f}%, 得分变化={best_delta_score:+.3f}, "
                f"新增笔数={delta_trades_is:+d}, 回撤变化={delta_mdd_is:+.2f}%"
            )
            ledger.bury_rejected_signal(
                unit_id,
                sid,
                tf,
                f"{spec.name}@{tf}",
                spec.archetype,
                spec.direction,
                failed_gate="Gate 3 (组合边际增益关)",
                reason=g3_reason,
                metrics={
                    "delta_ret_is": delta_ret_is,
                    "delta_trades_is": delta_trades_is,
                    "delta_mdd_is": delta_mdd_is,
                },
            )
            continue

        # --- 连闯三关！正式晋升入库 4×4 生态位矩阵 ---
        gate_stats["Admitted"] += 1
        step_counter += 1
        current_alloc = trial_alloc
        w_curr_multi = w_trial_multi
        curr_eval = trial_eval

        admitted_rec = AdmittedSignalRecord(
            unit_id=unit_id,
            sig_id=sid,
            timeframe=tf,
            name=f"{spec.name}@{tf}",
            archetype=spec.archetype,
            ruler_desc=ruler.ruler_name,
            direction=spec.direction,
            weight_alloc=best_trial_w,
            is_base_seed=False,
            is_n_purged=g1["is_n_purged"],
            is_mean_pct=g1["is_mean_pct"],
            is_excess_alpha_pct=g1["is_excess_alpha_pct"],
            is_win_rate_pct=g1["is_win_rate_pct"],
            is_pl_ratio=g1["is_pl_ratio"],
            is_safety_margin_pct=g1["is_safety_margin_pct"],
            is_boot_pval=g1["is_boot_pval"],
            max_corr_with_zoo=g2["max_corr"],
            max_corr_peer=g2["max_corr_peer"],
            gap_fill_ratio_pct=g2["gap_fill_ratio_pct"],
            dd_rescue_alpha_bp=g2["dd_rescue_alpha_bp"],
            marginal_sharpe_delta=delta_sharpe_is,
            marginal_calmar_delta=delta_calmar_is,
            oos_mean_pct=g1["oos_mean_pct"],
            oos_excess_alpha_pct=g1["oos_excess_alpha_pct"],
            oos_safety_margin_pct=g1["oos_safety_margin_pct"],
        )
        ledger.save_signal_to_zoo(admitted_rec)
        zoo = ledger.load_signal_zoo()

        step_payload = {
            "step": step_counter,
            "added_unit": f"{spec.name}@{tf}",
            "archetype": spec.archetype,
            "timeframe": tf,
            "zoo_size": len(current_alloc),
            "full_trades": curr_eval["full"]["trades"],
            "trades_per_yr": round(curr_eval["full"]["trades"] / 6.77, 1),
            "full_active_pct": curr_eval["full"]["active_pct"],
            "is_sharpe": curr_eval["is"]["sharpe"],
            "is_calmar": curr_eval["is"]["calmar"],
            "oos_cum_ret_pct": curr_eval["oos"]["cum_ret_pct"],
            "oos_ann_ret_pct": curr_eval["oos"]["ann_ret_pct"],
            "oos_sharpe": curr_eval["oos"]["sharpe"],
            "oos_calmar": curr_eval["oos"]["calmar"],
            "full_cum_ret_pct": curr_eval["full"]["cum_ret_pct"],
            "full_ann_ret_pct": curr_eval["full"]["ann_ret_pct"],
            "full_sharpe": curr_eval["full"]["sharpe"],
            "full_mdd_pct": curr_eval["full"]["mdd_pct"],
            "full_calmar": curr_eval["full"]["calmar"],
        }
        ledger.log_evolution_step(step_payload)
        ledger.save_champion_portfolio(
            {
                "alloc_map": current_alloc,
                "metrics_summary": step_payload,
            }
        )

        admitted_summary_rows.append(
            {
                "晋升轮次": f"Step {step_counter}",
                "原子单元": unit_id,
                "信号名称": spec.name,
                "战略方向": spec.archetype,
                "周期": tf,
                "IS独立样本": g1["is_n_purged"],
                "IS超额Alpha": f"{g1['is_excess_alpha_pct']:+.2f}%",
                "IS安全垫": f"{g1['is_safety_margin_pct']:+.1f}%",
                "最大相关性": f"{g2['max_corr']:.2f}",
                "空仓补盲率": f"{g2['gap_fill_ratio_pct']:.1f}%",
                "雪中送炭Alpha": f"{g2['dd_rescue_alpha_bp']:+.1f}BP",
                "OOS超额Alpha": f"{g1['oos_excess_alpha_pct']:+.2f}%",
            }
        )
        print(
            f"  [★ 晋升 Step {step_counter:02d}] [{spec.archetype} | {tf:3s}] {spec.name}@{tf} 入库! "
            f"| IS_Alpha={g1['is_excess_alpha_pct']:+.2f}%, 相关性={g2['max_corr']:.2f}, 补盲={g2['gap_fill_ratio_pct']:.1f}%, 雪中送炭={g2['dd_rescue_alpha_bp']:+.1f}BP "
            f"-> 总笔数={curr_eval['full']['trades']}, 全周期={curr_eval['full']['cum_ret_pct']:+.2f}% (夏普 {curr_eval['full']['sharpe']:.2f}), "
            f"样本外={curr_eval['oos']['cum_ret_pct']:+.2f}% (夏普 {curr_eval['oos']['sharpe']:.2f})"
        )

    niche_df = build_niche_matrix_summary(zoo)
    ledger.save_niche_matrix(niche_df.to_dict(orient="records"))

    print("\n" + "=" * 125)
    print(
        f"【一、三级漏斗筛选统计】 总检验原子候选: {len(atomic_candidates)} 个 -> "
        f"Gate1法医关淘汰: {gate_stats['Gate1_Rejected']} | "
        f"Gate2互补与配额关淘汰: {gate_stats['Gate2_Rejected']} | "
        f"Gate3组合边际关淘汰: {gate_stats['Gate3_Rejected']} | "
        f"★ 最终晋升入库: {gate_stats['Admitted']} 个"
    )
    print("=" * 125)
    if admitted_summary_rows:
        print(pd.DataFrame(admitted_summary_rows).to_string(index=False))

    print("\n" + "=" * 125)
    print("【二、4×4 生态位矩阵分布总览 (4 大战略方向 × 4 个独立周期 1d / 60m / 30m / 15m)】")
    print("=" * 125)
    print(niche_df.to_string(index=False))

    history_rows = ledger.load_evolution_history()
    print("\n" + "=" * 125)
    print("【三、Agent 自主进化阶梯轨迹：从单一突破基线到四方向×四周期生态位共振组合 (含 2024-2026 纯样本外盲测)】")
    print("=" * 125)
    hist_df = pd.DataFrame(history_rows)[
        [
            "step",
            "added_unit",
            "archetype",
            "timeframe",
            "full_trades",
            "trades_per_yr",
            "full_active_pct",
            "is_sharpe",
            "oos_cum_ret_pct",
            "oos_sharpe",
            "full_cum_ret_pct",
            "full_ann_ret_pct",
            "full_sharpe",
            "full_mdd_pct",
            "full_calmar",
        ]
    ]
    print(hist_df.to_string(index=False))

    generate_markdown_leaderboard(ledger, niche_df)
    export_evolution_dashboard(df_1d, base_eval, curr_eval, history_rows, DASHBOARD_HTML)

    print("\n" + "=" * 125)
    print(f"  [持久态基因库]   {ledger.zoo_path.resolve()}")
    print(f"  [4x4生态位矩阵]  {ledger.niche_path.resolve()}")
    print(f"  [证伪信号墓地]   {ledger.graveyard_path.resolve()}")
    print(f"  [实时演化榜单]   {ledger.leaderboard_path.resolve()}")
    print(f"  [交互式进化大屏] {DASHBOARD_HTML.resolve()}")
    print("=" * 125)


if __name__ == "__main__":
    main()
