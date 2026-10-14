#!/usr/bin/env python
"""
159915 长期自主信号挖掘与 4×5 全周期生态位矩阵进化 Agent 主循环
(MAP-Elites Niche Matrix + Archetype-Specific Rulers + 5-Timeframe Atomic Routing)

核心特性：
1. 搜索单元原子化为 `sig_id@tf` (跨越 1d / 60m / 30m / 15m / 5m 五级周期)，哪个信号在哪个周期过关就只挂载在哪个周期；
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
    map_tf_weight_to_5m_grid,
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
        if bars_per_group == 1:
            kdf = df_raw[["date", "datetime", "open", "high", "low", "close", "volume"]].copy()
            kdf = kdf.rename(columns={"datetime": "dt"})
            kdf["grp"] = df_raw["bar_in_day"]
        else:
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
        "5m": _agg(1, "5m"),
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
    tfs = ["1d", "60m", "30m", "15m", "5m"]
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
        "# 159915 自主信号挖掘 Agent · 4×5 全周期生态位矩阵与实时排行榜 (MAP-Elites Signal Zoo)",
        "",
        f"- **已入库原子信号单元总数 (`sig_id@tf`)**: `{len(zoo)}` 个（含 15 个基石种子单元 + **`{len(new_units)}` 个 Agent 跨四大战略方向与五级周期新挖正交单元**）",
        f"- **已证伪/拦截进入墓地的原子单元数**: `{len(grave)}` 个",
        "",
        "## 一、4×5 全周期生态位矩阵分布表 (4 大战略方向 × 5 个独立周期 1d / 60m / 30m / 15m / 5m)",
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
            "【159915 4×5全周期生态位矩阵 Agent】 进化前后五周期复合净值对比 (左侧 2020-2023 样本内 | 右侧 2024-2026 样本外盲测)",
            "动态回撤深度对比 (Drawdown %)",
            "Agent 跨四大战略方向与五级周期 (1d/60m/30m/15m/5m) 自主进化阶梯轨迹",
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
            name=f"★ 进化后：4×5全周期生态位矩阵多方向共振策略 (Step {history_rows[-1]['step']})",
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
            "text": "159915 创业板 ETF · 4×5 全周期生态位矩阵 (趋势+反转+回归+日内 × 1d/60m/30m/15m/5m) 自主进化大屏",
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
    parser = argparse.ArgumentParser(description="159915 4×5 全周期生态位矩阵自主信号挖掘与进化 Agent")
    parser.add_argument(
        "--reset", action="store_true", help="清空历史账本，从基石种子开始重新运行完整 4×5 全周期生态位进化循环"
    )
    parser.add_argument(
        "--refresh-grid",
        action="store_true",
        help="在原生 5m 主执行时间轴 (零延迟映射 + 0.06 调仓死区) 上重算并同步 Step 0~N 全量进化账本与大屏",
    )
    parser.add_argument(
        "--prune-and-replace",
        action="store_true",
        help="执行 LOO 后向修剪去冗余 + MAP-Elites 同生态位精英竞争替换与空位锦标赛补入",
    )
    parser.add_argument(
        "--reweight",
        action="store_true",
        help="执行存量活跃单元样本内法医先验锚定 + 双半场 (2020-2021/2022-2023) L2 信赖域联合坐标重配权",
    )
    parser.add_argument("--max-candidates", type=int, default=999, help="本轮最多测试多少个原子候选单元")
    parser.add_argument("--status-only", action="store_true", help="仅打印当前持久态信号库与 4×5 全周期生态位矩阵")
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
    print("  【159915 4×5 全周期生态位矩阵 · 长期自主信号挖掘与多方向正交进化 Agent】")
    print("  四大战略方向 (趋势突破 / 背驰反转 / 震荡回归 / 日内微结构) × 五级独立周期 (1d / 60m / 30m / 15m / 5m)")
    print("  [主时间轴] 原生 5m 无未来函数阶梯对齐网格 (消除 15m/5m 下采样坍缩与延迟 + 0.06 最小调仓死区)")
    print("=" * 125)

    tf_dfs = load_all_four_timeframes(REAL_5M_CSV)
    df_1d = tf_dfs["1d"]
    # 将主回测与合成网格下沉至原生 5m 时间轴 (保持变量名 df_30m 兼容后续调用)
    df_30m = tf_dfs["5m"]
    rng = np.random.default_rng(42)

    # 预计算所有 (sig_id, tf) 原子单元在自身周期上的触发序列，以及无延迟映射到 5m 主时间轴上的持仓序列
    trig_by_unit: dict[str, np.ndarray] = {}
    w_on_30m_by_unit: dict[str, np.ndarray] = {}

    for sid, spec in ALL_SIGNAL_SPECS.items():
        for tf in spec.eligible_tfs:
            unit_id = f"{sid}@{tf}"
            t_arr, w_arr = compute_signal_triggers_and_weights(tf_dfs[tf], sid, scale=TF_SCALE_MAP[tf])
            trig_by_unit[unit_id] = t_arr
            w_on_30m_by_unit[unit_id] = map_tf_weight_to_5m_grid(tf_dfs[tf], w_arr, df_30m, tf)

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

    if args.refresh_grid and zoo:
        print("\n[刷新主网格账本] 正在基于原生 5m 主执行网格与三支柱分层复合引擎重算 Step 0 ~ Step N 全量轨迹 ...")
        old_hist = ledger.load_evolution_history()
        if ledger.history_path.exists():
            ledger.history_path.unlink()
        step_alloc = dict(base_alloc)
        w_step_multi = w_base_multi.copy()
        step_eval = base_eval
        ledger.log_evolution_step(
            {
                "step": 0,
                "added_unit": "Base5_基石五真金(仅突破簇)",
                "archetype": "A_顺势突破簇",
                "timeframe": "1d/60m/30m",
                "zoo_size": len(step_alloc),
                "full_trades": step_eval["full"]["trades"],
                "trades_per_yr": round(step_eval["full"]["trades"] / 6.77, 1),
                "full_active_pct": step_eval["full"]["active_pct"],
                "is_sharpe": step_eval["is"]["sharpe"],
                "is_calmar": step_eval["is"]["calmar"],
                "oos_cum_ret_pct": step_eval["oos"]["cum_ret_pct"],
                "oos_ann_ret_pct": step_eval["oos"]["ann_ret_pct"],
                "oos_sharpe": step_eval["oos"]["sharpe"],
                "oos_calmar": step_eval["oos"]["calmar"],
                "full_cum_ret_pct": step_eval["full"]["cum_ret_pct"],
                "full_ann_ret_pct": step_eval["full"]["ann_ret_pct"],
                "full_sharpe": step_eval["full"]["sharpe"],
                "full_mdd_pct": step_eval["full"]["mdd_pct"],
                "full_calmar": step_eval["full"]["calmar"],
            }
        )

        if len(old_hist) > 1:
            for row in old_hist[1:]:
                s_idx = int(row["step"])
                added_label = str(row["added_unit"])
                tf = str(row["timeframe"])
                arch = str(row["archetype"])
                prev_is_sh = step_eval["is"]["sharpe"]
                prev_is_cal = step_eval["is"]["calmar"]

                if added_label.startswith("✂剔除冗余_"):
                    rm_uid = added_label.split("✂剔除冗余_")[1].strip()
                    step_alloc.pop(rm_uid, None)
                    w_step_multi = synthesize_atomic_units_portfolio(df_30m, w_on_30m_by_unit, step_alloc)
                    step_eval = evaluate_full_is_oos(df_30m, w_step_multi)
                elif added_label.startswith("⚖"):
                    step_alloc = {u: float(zoo[u]["weight_alloc"]) for u in step_alloc if u in zoo}
                    w_step_multi = synthesize_atomic_units_portfolio(df_30m, w_on_30m_by_unit, step_alloc)
                    step_eval = evaluate_full_is_oos(df_30m, w_step_multi)
                elif added_label.startswith("⚔") and "(替" in added_label:
                    inc_uid = added_label.split("(替")[1].rstrip(")").strip()
                    cand_sid = added_label[1:].split("_")[0].strip()
                    uid = f"{cand_sid}@{tf}"
                    step_alloc.pop(inc_uid, None)
                    spec = ALL_SIGNAL_SPECS[cand_sid]
                    ruler = ARCHETYPE_RULERS[spec.archetype]
                    same_dir_peers = {
                        p_uid: w_on_30m_by_unit[p_uid]
                        for p_uid in step_alloc
                        if int(ALL_SIGNAL_SPECS[p_uid.split("@")[0]].direction) == spec.direction
                    }
                    g2_ref = run_gate2_3d_complementarity_audit(
                        df_30m,
                        w_on_30m_by_unit[uid],
                        same_dir_peers,
                        w_step_multi,
                        archetype=spec.archetype,
                        current_archetype_new_count=0,
                        ruler=ruler,
                    )
                    w_alloc = float(zoo[uid]["weight_alloc"]) if uid in zoo else spec.default_weight
                    step_alloc[uid] = w_alloc
                    w_step_multi = synthesize_atomic_units_portfolio(df_30m, w_on_30m_by_unit, step_alloc)
                    step_eval = evaluate_full_is_oos(df_30m, w_step_multi)
                    if uid in zoo:
                        info = zoo[uid]
                        info["max_corr_with_zoo"] = g2_ref["max_corr"]
                        info["max_corr_peer"] = g2_ref["max_corr_peer"]
                        info["gap_fill_ratio_pct"] = g2_ref["gap_fill_ratio_pct"]
                        info["dd_rescue_alpha_bp"] = g2_ref["dd_rescue_alpha_bp"]
                        info["marginal_sharpe_delta"] = round(step_eval["is"]["sharpe"] - prev_is_sh, 3)
                        info["marginal_calmar_delta"] = round(step_eval["is"]["calmar"] - prev_is_cal, 3)
                        ledger.save_signal_to_zoo(AdmittedSignalRecord(**info))
                else:
                    sid = added_label.split("_")[0].strip()
                    uid = f"{sid}@{tf}"
                    spec = ALL_SIGNAL_SPECS[sid]
                    ruler = ARCHETYPE_RULERS[spec.archetype]
                    same_dir_peers = {
                        p_uid: w_on_30m_by_unit[p_uid]
                        for p_uid in step_alloc
                        if int(ALL_SIGNAL_SPECS[p_uid.split("@")[0]].direction) == spec.direction
                    }
                    g2_ref = run_gate2_3d_complementarity_audit(
                        df_30m,
                        w_on_30m_by_unit[uid],
                        same_dir_peers,
                        w_step_multi,
                        archetype=spec.archetype,
                        current_archetype_new_count=0,
                        ruler=ruler,
                    )
                    w_alloc = (
                        float(zoo[uid]["weight_alloc"])
                        if uid in zoo
                        else (0.40 if tf == "5m" else spec.default_weight)
                    )
                    step_alloc[uid] = w_alloc
                    w_step_multi = synthesize_atomic_units_portfolio(df_30m, w_on_30m_by_unit, step_alloc)
                    step_eval = evaluate_full_is_oos(df_30m, w_step_multi)
                    if uid in zoo:
                        info = zoo[uid]
                        info["max_corr_with_zoo"] = g2_ref["max_corr"]
                        info["max_corr_peer"] = g2_ref["max_corr_peer"]
                        info["gap_fill_ratio_pct"] = g2_ref["gap_fill_ratio_pct"]
                        info["dd_rescue_alpha_bp"] = g2_ref["dd_rescue_alpha_bp"]
                        info["marginal_sharpe_delta"] = round(step_eval["is"]["sharpe"] - prev_is_sh, 3)
                        info["marginal_calmar_delta"] = round(step_eval["is"]["calmar"] - prev_is_cal, 3)
                        ledger.save_signal_to_zoo(AdmittedSignalRecord(**info))

                step_payload = {
                    "step": s_idx,
                    "added_unit": added_label,
                    "archetype": arch,
                    "timeframe": tf,
                    "zoo_size": len(step_alloc),
                    "full_trades": step_eval["full"]["trades"],
                    "trades_per_yr": round(step_eval["full"]["trades"] / 6.77, 1),
                    "full_active_pct": step_eval["full"]["active_pct"],
                    "is_sharpe": step_eval["is"]["sharpe"],
                    "is_calmar": step_eval["is"]["calmar"],
                    "oos_cum_ret_pct": step_eval["oos"]["cum_ret_pct"],
                    "oos_ann_ret_pct": step_eval["oos"]["ann_ret_pct"],
                    "oos_sharpe": step_eval["oos"]["sharpe"],
                    "oos_calmar": step_eval["oos"]["calmar"],
                    "full_cum_ret_pct": step_eval["full"]["cum_ret_pct"],
                    "full_ann_ret_pct": step_eval["full"]["ann_ret_pct"],
                    "full_sharpe": step_eval["full"]["sharpe"],
                    "full_mdd_pct": step_eval["full"]["mdd_pct"],
                    "full_calmar": step_eval["full"]["calmar"],
                }
                ledger.log_evolution_step(step_payload)
                ledger.save_champion_portfolio({"alloc_map": step_alloc, "metrics_summary": step_payload})
        zoo = ledger.load_signal_zoo()

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
        f"\n[启动 4×5 生态位进化循环] 待检验原子候选单元 (sig@tf): {len(atomic_candidates)} 个 | "
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

        cand_weights = (0.28, 0.40, 0.52) if tf == "5m" else (0.20, 0.28, 0.36)
        for cand_w in cand_weights:
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

        min_delta_ret = 0.05 if tf in ("15m", "5m") else 0.25
        min_score_th = -0.06 if tf == "5m" else -0.04
        passed_g3 = (
            (delta_trades_is >= 1 or delta_active_is >= 0.15)
            and (
                delta_ret_is >= min_delta_ret
                or (delta_ret_is >= 0.15 and delta_mdd_is <= 0.10 and best_delta_score >= -0.02)
                or (tf == "5m" and delta_ret_is >= 0.0 and delta_mdd_is <= 0.25 and best_delta_score >= -0.06)
            )
            and best_delta_score >= min_score_th
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

        # --- 连闯三关！正式晋升入库 4×5 生态位矩阵 ---
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

    # 3. 执行 LOO 后向修剪去冗余 + MAP-Elites 同生态位精英竞争替换与空位锦标赛补入
    if args.prune_and_replace and zoo:
        print("\n" + "=" * 125)
        print("  [启动架构精炼] Phase 1: LOO 后向消融去冗余 | Phase 2: MAP-Elites 同生态位精英替换与锦标赛补位")
        print("=" * 125)

        def _calc_is_obj(ev: dict) -> float:
            return ev["is"]["ann_ret_pct"] * 0.15 + ev["is"]["sharpe"] * 1.0 + ev["is"]["calmar"] * 0.4

        pruned_uids: set[str] = set()
        # --- Phase 1: 严格样本内 (2020-2023 IS) 迭代 LOO 后向修剪 ---
        while True:
            cur_obj = _calc_is_obj(curr_eval)
            worst_uid = None
            worst_contrib = 0.0
            best_without_ev = None
            best_without_w = None

            for uid in [u for u, inf in zoo.items() if not inf["is_base_seed"]]:
                t_alloc = {k: v for k, v in current_alloc.items() if k != uid}
                w_t = synthesize_atomic_units_portfolio(df_30m, w_on_30m_by_unit, t_alloc)
                ev_t = evaluate_full_is_oos(df_30m, w_t)
                c_obj = cur_obj - _calc_is_obj(ev_t)
                c_ret = curr_eval["is"]["cum_ret_pct"] - ev_t["is"]["cum_ret_pct"]
                c_sh = curr_eval["is"]["sharpe"] - ev_t["is"]["sharpe"]
                if c_obj <= -0.05 and c_sh < 0 and c_ret <= 0.05:
                    if c_obj < worst_contrib:
                        worst_contrib = c_obj
                        worst_uid = uid
                        best_without_ev = ev_t
                        best_without_w = w_t

            if worst_uid is None:
                break

            rm_info = ledger.remove_signal_from_zoo(worst_uid)
            zoo = ledger.load_signal_zoo()
            del current_alloc[worst_uid]
            w_curr_multi = best_without_w
            curr_eval = best_without_ev
            pruned_uids.add(worst_uid)
            step_counter += 1

            ledger.bury_rejected_signal(
                worst_uid,
                rm_info["sig_id"],
                rm_info["timeframe"],
                rm_info["name"],
                rm_info["archetype"],
                int(rm_info["direction"]),
                failed_gate="LOO 后向消融修剪关",
                reason=f"拔除后样本内综合得分提升 {-worst_contrib:+.4f}，消除跨轮次冗余",
                metrics={"loo_contrib_is_obj": round(worst_contrib, 4)},
            )
            step_payload = {
                "step": step_counter,
                "added_unit": f"✂剔除冗余_{worst_uid}",
                "archetype": rm_info["archetype"],
                "timeframe": rm_info["timeframe"],
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
            ledger.save_champion_portfolio({"alloc_map": current_alloc, "metrics_summary": step_payload})
            print(
                f"  [✂ LOO修剪 Step {step_counter:02d}] 拔除冗余单元 {worst_uid} ({rm_info['archetype']}) | "
                f"IS得分净增={-worst_contrib:+.4f} -> IS夏普={curr_eval['is']['sharpe']:.2f}, "
                f"全周期={curr_eval['full']['cum_ret_pct']:+.2f}% (夏普 {curr_eval['full']['sharpe']:.2f}), "
                f"样本外={curr_eval['oos']['cum_ret_pct']:+.2f}% (夏普 {curr_eval['oos']['sharpe']:.2f})"
            )

        # --- Phase 2: MAP-Elites 同生态位 1-for-1 精英竞争替换与空位锦标赛补入 ---
        challengers: list[tuple[str, Any, str, dict]] = []
        for spec in CANDIDATE_HYPOTHESIS_POOL:
            ruler = ARCHETYPE_RULERS[spec.archetype]
            for tf in spec.eligible_tfs:
                uid = f"{spec.sig_id}@{tf}"
                if uid in zoo or uid in pruned_uids:
                    continue
                g1 = run_gate1_forensic_audit(tf_dfs[tf], trig_by_unit[uid], spec.direction, ruler, tf, rng)
                if g1["passed"] and g1["is_n_purged"] >= 80:
                    challengers.append((uid, spec, tf, g1))

        # 按样本深度对数 × 样本内安全垫进行锦标赛种子排序 (完全无样本外信息泄露)
        challengers.sort(key=lambda x: np.log(x[3]["is_n_purged"]) * x[3]["is_safety_margin_pct"], reverse=True)

        for cand_uid, spec, tf, g1 in challengers:
            if cand_uid in zoo:
                continue
            ruler = ARCHETYPE_RULERS[spec.archetype]
            arch_new_uids = [
                u for u, inf in zoo.items() if inf["archetype"] == spec.archetype and not inf["is_base_seed"]
            ]
            same_dir_new_uids = [u for u in arch_new_uids if int(zoo[u]["direction"]) == spec.direction]
            max_same_dir = max(4, (ruler.quota_max_units + 1) // 2)
            has_open_slot = len(arch_new_uids) < ruler.quota_max_units and len(same_dir_new_uids) < max_same_dir

            if has_open_slot:
                same_dir_zoo_w = {
                    z_uid: w_on_30m_by_unit[z_uid]
                    for z_uid, z_info in zoo.items()
                    if int(z_info["direction"]) == spec.direction
                }
                g2 = run_gate2_3d_complementarity_audit(
                    df_30m,
                    w_on_30m_by_unit[cand_uid],
                    same_dir_zoo_w,
                    w_curr_multi,
                    archetype=spec.archetype,
                    current_archetype_new_count=len(arch_new_uids),
                    ruler=ruler,
                    current_arch_same_dir_new_count=len(same_dir_new_uids),
                )
                if not g2["passed"]:
                    continue

                best_cw = spec.default_weight
                best_ev = None
                best_w = None
                best_d_obj = -999.0
                best_rank_score = -999.0
                for cw in (0.28, 0.40, 0.52) if tf == "5m" else (0.20, 0.28, 0.36):
                    t_alloc = dict(current_alloc)
                    t_alloc[cand_uid] = cw
                    w_t = synthesize_atomic_units_portfolio(df_30m, w_on_30m_by_unit, t_alloc)
                    ev_t = evaluate_full_is_oos(df_30m, w_t)
                    d_obj = _calc_is_obj(ev_t) - _calc_is_obj(curr_eval)
                    d_ret_is = ev_t["is"]["cum_ret_pct"] - curr_eval["is"]["cum_ret_pct"]
                    d_mdd_is = ev_t["is"]["mdd_pct"] - curr_eval["is"]["mdd_pct"]
                    if d_obj >= 0.03 and d_ret_is >= 0.50 and d_mdd_is <= 0.15:
                        rk = d_obj + 0.05 * d_ret_is
                        if rk > best_rank_score:
                            best_rank_score = rk
                            best_d_obj = d_obj
                            best_cw = cw
                            best_ev = ev_t
                            best_w = w_t

                if best_ev is not None:
                    prev_is_sh = curr_eval["is"]["sharpe"]
                    prev_is_cal = curr_eval["is"]["calmar"]
                    current_alloc[cand_uid] = best_cw
                    w_curr_multi = best_w
                    curr_eval = best_ev
                    step_counter += 1

                    admitted_rec = AdmittedSignalRecord(
                        unit_id=cand_uid,
                        sig_id=spec.sig_id,
                        timeframe=tf,
                        name=f"{spec.name}@{tf}",
                        archetype=spec.archetype,
                        ruler_desc=ruler.ruler_name,
                        direction=spec.direction,
                        weight_alloc=best_cw,
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
                        marginal_sharpe_delta=round(curr_eval["is"]["sharpe"] - prev_is_sh, 3),
                        marginal_calmar_delta=round(curr_eval["is"]["calmar"] - prev_is_cal, 3),
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
                    ledger.save_champion_portfolio({"alloc_map": current_alloc, "metrics_summary": step_payload})
                    print(
                        f"  [★ 锦标赛补位 Step {step_counter:02d}] [{spec.archetype} | {tf:3s}] {spec.name}@{tf} 入库! "
                        f"| dIS_Obj={best_d_obj:+.4f} -> 总笔数={curr_eval['full']['trades']}, "
                        f"全周期={curr_eval['full']['cum_ret_pct']:+.2f}% (夏普 {curr_eval['full']['sharpe']:.2f}, 卡玛 {curr_eval['full']['calmar']:.2f}), "
                        f"样本外={curr_eval['oos']['cum_ret_pct']:+.2f}% (夏普 {curr_eval['oos']['sharpe']:.2f})"
                    )
            else:
                # 生态位已满：检验同生态位 1-for-1 精英竞争替换 (要求挑战者样本量 >= 2倍被替换者，且替换后显著提升 IS 表现)
                best_inc = None
                best_cw = spec.default_weight
                best_ev = None
                best_w = None
                best_g2 = None
                best_d_obj = 0.0
                for inc_uid in same_dir_new_uids:
                    inc_n = int(zoo[inc_uid].get("is_n_purged", 999))
                    if g1["is_n_purged"] < 2 * inc_n:
                        continue
                    same_dir_zoo_w = {
                        z_uid: w_on_30m_by_unit[z_uid]
                        for z_uid, z_info in zoo.items()
                        if int(z_info["direction"]) == spec.direction and z_uid != inc_uid
                    }
                    alloc_no_inc = {k: v for k, v in current_alloc.items() if k != inc_uid}
                    w_no_inc = synthesize_atomic_units_portfolio(df_30m, w_on_30m_by_unit, alloc_no_inc)
                    g2 = run_gate2_3d_complementarity_audit(
                        df_30m,
                        w_on_30m_by_unit[cand_uid],
                        same_dir_zoo_w,
                        w_no_inc,
                        archetype=spec.archetype,
                        current_archetype_new_count=len(arch_new_uids) - 1,
                        ruler=ruler,
                        current_arch_same_dir_new_count=len(same_dir_new_uids) - 1,
                    )
                    if not g2["passed"]:
                        continue

                    for cw in (0.28, 0.40, 0.52) if tf == "5m" else (0.20, 0.28, 0.36):
                        t_alloc = dict(alloc_no_inc)
                        t_alloc[cand_uid] = cw
                        w_t = synthesize_atomic_units_portfolio(df_30m, w_on_30m_by_unit, t_alloc)
                        ev_t = evaluate_full_is_oos(df_30m, w_t)
                        d_obj = _calc_is_obj(ev_t) - _calc_is_obj(curr_eval)
                        d_ret_is = ev_t["is"]["cum_ret_pct"] - curr_eval["is"]["cum_ret_pct"]
                        d_mdd_is = ev_t["is"]["mdd_pct"] - curr_eval["is"]["mdd_pct"]
                        if d_obj > best_d_obj and d_ret_is >= 0.40 and d_mdd_is <= 0.15:
                            best_d_obj = d_obj
                            best_inc = inc_uid
                            best_cw = cw
                            best_ev = ev_t
                            best_w = w_t
                            best_g2 = g2

                if best_inc is not None:
                    prev_is_sh = curr_eval["is"]["sharpe"]
                    prev_is_cal = curr_eval["is"]["calmar"]
                    rm_inc = ledger.remove_signal_from_zoo(best_inc)
                    del current_alloc[best_inc]
                    pruned_uids.add(best_inc)

                    current_alloc[cand_uid] = best_cw
                    w_curr_multi = best_w
                    curr_eval = best_ev
                    step_counter += 1

                    ledger.bury_rejected_signal(
                        best_inc,
                        rm_inc["sig_id"],
                        rm_inc["timeframe"],
                        rm_inc["name"],
                        rm_inc["archetype"],
                        int(rm_inc["direction"]),
                        failed_gate="MAP-Elites 同生态位精英替换关",
                        reason=f"被同生态位大样本优胜挑战者 {cand_uid} 替换 (dIS_Obj={best_d_obj:+.4f})",
                        metrics={"replaced_by": cand_uid, "swap_d_is_obj": round(best_d_obj, 4)},
                    )

                    admitted_rec = AdmittedSignalRecord(
                        unit_id=cand_uid,
                        sig_id=spec.sig_id,
                        timeframe=tf,
                        name=f"{spec.name}@{tf}",
                        archetype=spec.archetype,
                        ruler_desc=ruler.ruler_name,
                        direction=spec.direction,
                        weight_alloc=best_cw,
                        is_base_seed=False,
                        is_n_purged=g1["is_n_purged"],
                        is_mean_pct=g1["is_mean_pct"],
                        is_excess_alpha_pct=g1["is_excess_alpha_pct"],
                        is_win_rate_pct=g1["is_win_rate_pct"],
                        is_pl_ratio=g1["is_pl_ratio"],
                        is_safety_margin_pct=g1["is_safety_margin_pct"],
                        is_boot_pval=g1["is_boot_pval"],
                        max_corr_with_zoo=best_g2["max_corr"],
                        max_corr_peer=best_g2["max_corr_peer"],
                        gap_fill_ratio_pct=best_g2["gap_fill_ratio_pct"],
                        dd_rescue_alpha_bp=best_g2["dd_rescue_alpha_bp"],
                        marginal_sharpe_delta=round(curr_eval["is"]["sharpe"] - prev_is_sh, 3),
                        marginal_calmar_delta=round(curr_eval["is"]["calmar"] - prev_is_cal, 3),
                        oos_mean_pct=g1["oos_mean_pct"],
                        oos_excess_alpha_pct=g1["oos_excess_alpha_pct"],
                        oos_safety_margin_pct=g1["oos_safety_margin_pct"],
                    )
                    ledger.save_signal_to_zoo(admitted_rec)
                    zoo = ledger.load_signal_zoo()

                    step_payload = {
                        "step": step_counter,
                        "added_unit": f"⚔{spec.name}@{tf}(替{best_inc})",
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
                    ledger.save_champion_portfolio({"alloc_map": current_alloc, "metrics_summary": step_payload})
                    print(
                        f"  [⚔ 精英替换 Step {step_counter:02d}] [{spec.archetype} | {tf:3s}] {cand_uid} 击败并替换 {best_inc}! "
                        f"| dIS_Obj={best_d_obj:+.4f} -> 总笔数={curr_eval['full']['trades']}, "
                        f"全周期={curr_eval['full']['cum_ret_pct']:+.2f}% (夏普 {curr_eval['full']['sharpe']:.2f}, 卡玛 {curr_eval['full']['calmar']:.2f}), "
                        f"样本外={curr_eval['oos']['cum_ret_pct']:+.2f}% (夏普 {curr_eval['oos']['sharpe']:.2f})"
                    )

    # 4. 执行存量活跃单元样本内法医先验锚定 + 双半场 (2020-2021/2022-2023) L2 信赖域联合坐标重配权
    if (args.reweight or args.prune_and_replace) and zoo:
        print("\n" + "=" * 125)
        print("  [启动样本内联合重配权] 基石种子锚定 + 29 个新挖掘单元法医经验贝叶斯先验倾斜与双半场 (H1/H2) L2 坐标精调")
        print("=" * 125)

        is_mask = (df_30m["dt"] <= IS_END_DT).values
        h1_mask = (df_30m["dt"] <= pd.Timestamp("2021-12-31 23:59:59")).values
        h2_mask = ((df_30m["dt"] > pd.Timestamp("2021-12-31 23:59:59")) & (df_30m["dt"] <= IS_END_DT)).values

        def _prep_fast_slice(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            sub = df_30m.iloc[mask].reset_index(drop=True)
            c = sub["close"].values
            r = np.zeros(len(c), dtype=np.float64)
            r[:-1] = (c[1:] - c[:-1]) / c[:-1]
            _, dids = np.unique(sub["dt"].dt.date.values, return_inverse=True)
            return r, dids

        def _fast_slice_stats(close_ret: np.ndarray, day_ids: np.ndarray, w_slice: np.ndarray) -> dict[str, float]:
            dw = np.abs(np.diff(w_slice, prepend=0.0))
            bar_net = w_slice * close_ret - dw * 0.0002
            daily_net = np.bincount(day_ids, weights=bar_net)
            cum_ret = float(daily_net.sum() * 100.0)
            n_years = len(daily_net) / 252.0
            ann_ret = float(cum_ret / max(n_years, 1e-6))
            std_d = float(daily_net.std(ddof=1))
            sharpe = float((daily_net.mean() / std_d) * np.sqrt(252.0)) if std_d > 1e-12 else 0.0
            curve = np.cumsum(daily_net)
            peak = np.maximum.accumulate(np.maximum(curve, 0.0))
            mdd = float((peak - curve).max() * 100.0)
            calmar = float(ann_ret / mdd) if mdd > 1e-6 else 0.0
            return {
                "cum_ret_pct": cum_ret,
                "ann_ret_pct": ann_ret,
                "sharpe": sharpe,
                "mdd_pct": mdd,
                "calmar": calmar,
            }

        ret_is, dids_is = _prep_fast_slice(is_mask)
        ret_h1, dids_h1 = _prep_fast_slice(h1_mask)
        ret_h2, dids_h2 = _prep_fast_slice(h2_mask)

        alloc0 = dict(current_alloc)
        base_budget = sum(alloc0.values())

        raw_q: dict[str, float] = {}
        for uid, info in zoo.items():
            n_p = max(int(info.get("is_n_purged", 20)), 5)
            saf = float(np.clip(float(info.get("is_safety_margin_pct", 4.0)), 1.0, 15.0))
            dd_res = float(np.clip(float(info.get("dd_rescue_alpha_bp", 0.0)), -10.0, 30.0))
            raw_q[uid] = float(np.log(n_p) * saf * (1.0 + 0.02 * dd_res))

        non_seed_uids = [u for u in alloc0 if not zoo[u]["is_base_seed"]]
        q_vals = np.array([raw_q[u] for u in non_seed_uids])
        q_mean, q_std = float(q_vals.mean()), float(q_vals.std() + 1e-6)
        z_q = {
            u: (float(np.clip((raw_q[u] - q_mean) / q_std, -1.2, 1.6)) if not zoo[u]["is_base_seed"] else 0.0)
            for u in alloc0
        }

        eb_prior = {
            u: round(float(np.clip(alloc0[u] * (1.0 + 0.06 * z_q[u]), 0.15, 0.56)), 2)
            for u in alloc0
        }

        def _eval_dual_regime(alloc: dict[str, float]) -> tuple[float, dict, dict, dict]:
            w_full = synthesize_atomic_units_portfolio(df_30m, w_on_30m_by_unit, alloc)
            s_is = _fast_slice_stats(ret_is, dids_is, w_full[is_mask])
            s_h1 = _fast_slice_stats(ret_h1, dids_h1, w_full[h1_mask])
            s_h2 = _fast_slice_stats(ret_h2, dids_h2, w_full[h2_mask])
            l2_pen = sum((alloc[u] - eb_prior[u]) ** 2 for u in alloc)
            min_sh = min(s_h1["sharpe"], s_h2["sharpe"])
            obj = (
                s_is["sharpe"] * 1.15
                + min_sh * 0.85
                + s_is["calmar"] * 0.45
                + s_is["ann_ret_pct"] * 0.12
                - s_is["mdd_pct"] * 0.40
                - 1.0 * l2_pen
            )
            return obj, s_is, s_h1, s_h2

        opt_alloc = dict(eb_prior)
        obj_cur, st_is_cur, st_h1_cur, st_h2_cur = _eval_dual_regime(opt_alloc)
        ordered_uids = sorted(alloc0.keys(), key=lambda u: raw_q[u], reverse=True)
        changed_units: list[str] = []

        for _pass in range(2):
            for uid in ordered_uids:
                if bool(zoo[uid]["is_base_seed"]):
                    continue
                w_p = eb_prior[uid]
                tf = uid.split("@")[1]
                lo = max(0.16, round(w_p - 0.10, 2))
                hi = min(0.58 if tf == "5m" else 0.48, round(w_p + 0.10, 2))
                grid = sorted(set([lo, round(w_p - 0.05, 2), round(w_p, 2), round(w_p + 0.05, 2), hi]))

                best_w = opt_alloc[uid]
                best_obj = obj_cur
                best_pack = (st_is_cur, st_h1_cur, st_h2_cur)

                for cw in grid:
                    if abs(cw - opt_alloc[uid]) < 1e-4:
                        continue
                    t_alloc = dict(opt_alloc)
                    t_alloc[uid] = cw
                    if sum(t_alloc.values()) > base_budget + 0.45:
                        continue
                    obj_t, s_is, s_h1, s_h2 = _eval_dual_regime(t_alloc)
                    if s_is["mdd_pct"] > st_is_cur["mdd_pct"] + 0.01:
                        continue
                    if s_h1["sharpe"] < st_h1_cur["sharpe"] - 0.003 or s_h2["sharpe"] < st_h2_cur["sharpe"] - 0.003:
                        continue
                    if s_is["sharpe"] < st_is_cur["sharpe"] or s_is["cum_ret_pct"] < st_is_cur["cum_ret_pct"] - 0.02:
                        continue
                    if obj_t > best_obj + 0.004:
                        best_obj = obj_t
                        best_w = cw
                        best_pack = (s_is, s_h1, s_h2)

                if abs(best_w - opt_alloc[uid]) >= 1e-4:
                    opt_alloc[uid] = best_w
                    obj_cur = best_obj
                    st_is_cur, st_h1_cur, st_h2_cur = best_pack

        for uid in opt_alloc:
            if abs(opt_alloc[uid] - alloc0[uid]) >= 1e-3:
                changed_units.append(f"{uid}({alloc0[uid]:.2f}->{opt_alloc[uid]:.2f})")
                info = zoo[uid]
                info["weight_alloc"] = opt_alloc[uid]
                ledger.save_signal_to_zoo(AdmittedSignalRecord(**info))

        w_opt_multi = synthesize_atomic_units_portfolio(df_30m, w_on_30m_by_unit, opt_alloc)
        opt_eval = evaluate_full_is_oos(df_30m, w_opt_multi)
        if opt_eval["is"]["sharpe"] > curr_eval["is"]["sharpe"] or opt_eval["is"]["calmar"] > curr_eval["is"]["calmar"]:
            current_alloc = opt_alloc
            w_curr_multi = w_opt_multi
            curr_eval = opt_eval
            zoo = ledger.load_signal_zoo()
            step_counter += 1
            step_payload = {
                "step": step_counter,
                "added_unit": "⚖样本内联合重配权_44单元法医先验双半场精调",
                "archetype": "四簇联合重配",
                "timeframe": "1d~5m",
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
            ledger.save_champion_portfolio({"alloc_map": current_alloc, "metrics_summary": step_payload})
            print(
                f"  [⚖ 联合重配权 Step {step_counter:02d}] 微调单元 {len(changed_units)} 个 | "
                f"IS夏普={curr_eval['is']['sharpe']:.2f}, IS回撤={curr_eval['is']['mdd_pct']:.2f}%, IS卡玛={curr_eval['is']['calmar']:.2f} | "
                f"样本外={curr_eval['oos']['cum_ret_pct']:+.2f}% (夏普 {curr_eval['oos']['sharpe']:.2f}, 卡玛 {curr_eval['oos']['calmar']:.2f}) | "
                f"全周期={curr_eval['full']['cum_ret_pct']:+.2f}% (夏普 {curr_eval['full']['sharpe']:.2f}, 回撤 {curr_eval['full']['mdd_pct']:.2f}%, 卡玛 {curr_eval['full']['calmar']:.2f})"
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
    print("【二、4×5 全周期生态位矩阵分布总览 (4 大战略方向 × 5 个独立周期 1d / 60m / 30m / 15m / 5m)】")
    print("=" * 125)
    print(niche_df.to_string(index=False))

    history_rows = ledger.load_evolution_history()
    print("\n" + "=" * 125)
    print("【三、Agent 自主进化阶梯轨迹：从单一突破基线到四方向×五周期生态位共振组合 (含 2024-2026 纯样本外盲测)】")
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
    print(f"  [4x5生态位矩阵]  {ledger.niche_path.resolve()}")
    print(f"  [证伪信号墓地]   {ledger.graveyard_path.resolve()}")
    print(f"  [实时演化榜单]   {ledger.leaderboard_path.resolve()}")
    print(f"  [交互式进化大屏] {DASHBOARD_HTML.resolve()}")
    print("=" * 125)


if __name__ == "__main__":
    main()
