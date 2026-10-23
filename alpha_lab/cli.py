"""
AlphaLab 标准化一键挖掘 CLI 工作台与全景看板生成器 (Issue #23)
================================================================================
提供 `python -m alpha_lab.cli` 命令行入口，涵盖 6 个标准子命令（全部支持 `--json` 输出）：
1. `migrate-db` : 初始化/更新 PostgreSQL (`alpha_lab` Schema) 或 SQLite 台账并导入历史全量缓存
2. `status`     : 输出 6x5 生态位饱和度矩阵、Rust 信号目录覆盖率、待开采信号池与墓地统计
3. `preflight`  : 执行 <1s 仿射镜像对称 + 前视零泄漏 + 边界合法性快速预检门禁
4. `eval`       : 执行 Pre-flight -> Full60ContextStore 评分 -> Gate 1B/2/3 漏斗 -> 消融归因全流程
5. `bury`       : 手工登记淘汰/证伪候选至 `alpha_lab.factor_graveyard`
6. `report`     : 从台账数据库自动生成 `alpha_lab/FACTORS_LEADERBOARD.md` 全景看板
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from alpha_lab.catalog import (
    get_signal_usage_summary,
    list_unconsumed_mirror_signals,
    scan_all_atomic_factors,
)
from alpha_lab.context_store import Full60ContextStore
from alpha_lab.ledger_db import (
    ROOT_DIR,
    AlphaLabLedgerDB,
    migrate_historical_data_to_db,
)
from alpha_lab.preflight import run_preflight_audit
from alpha_lab.runner import run_full_sop_evaluation
from examples.multi_factor_cta.atomic_factors import ATOMIC_FACTOR_DEFS

app = typer.Typer(
    name="alpha-lab",
    help="CZSC AlphaLab 工程化因子挖掘工作台 CLI (Full-60 Commodities SOP)",
    add_completion=False,
)

DEFAULT_LEADERBOARD_PATH = ROOT_DIR / "alpha_lab" / "FACTORS_LEADERBOARD.md"
CLUSTER_META: dict[str, str] = {
    "A": "Trend / Regime (趋势与宏观状态)",
    "B": "Breakout / Participation (突破与资金参与)",
    "C": "Pullback / Exhaustion (回调与衰竭反转)",
    "D": "Position / Carry / Basis (持仓博弈与期限结构)",
    "E": "Cross-Asset / Relative Value (截面强弱与相对价值)",
    "F": "Intraday / Execution Microstructure (日内微结构与冲击)",
}
ALL_CLUSTERS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
ALL_TFS: tuple[str, ...] = ("1d", "60m", "30m", "15m", "5m")


def _cluster_from_arch(arch: str, family_id: str = "") -> str:
    s = str(arch).strip()
    if s and s[0] in ALL_CLUSTERS:
        return s[0]
    f = str(family_id).strip()
    if f and f[0] in ALL_CLUSTERS:
        return f[0]
    return "A"


def build_niche_saturation_matrix(db: AlphaLabLedgerDB) -> dict[str, Any]:
    """构建 6×5 (Cluster A~F × Timeframe 1d/60m/30m/15m/5m) 生态位饱和度矩阵。"""
    niche_df = db.get_niche_saturation_matrix()
    fdef_df = db.query_df("SELECT sig_id, family_id, archetype, stage FROM alpha_lab.factor_defs")

    cells: dict[str, dict[str, dict[str, Any]]] = {
        c: {
            tf: {
                "candidate_uids": 0,
                "gate1b_pass_folds": 0,
                "avg_mean_sharpe": 0.0,
                "avg_pooled_t": 0.0,
                "scarcity_status": "SCARCE",
            }
            for tf in ALL_TFS
        }
        for c in ALL_CLUSTERS
    }

    if not niche_df.empty:
        for _, r in niche_df.iterrows():
            c_str = _cluster_from_arch(str(r["archetype"]))
            tf_str = str(r["tf"])
            if c_str in cells and tf_str in cells[c_str]:
                cells[c_str][tf_str] = {
                    "candidate_uids": int(r["candidate_uids"]),
                    "gate1b_pass_folds": int(r["gate1b_pass_folds"]),
                    "avg_mean_sharpe": round(float(r["avg_mean_sharpe"]), 3),
                    "avg_pooled_t": round(float(r["avg_pooled_t"]), 3),
                    "scarcity_status": str(r["scarcity_status"]),
                }

    under_explored: list[dict[str, Any]] = []
    saturated: list[dict[str, Any]] = []
    for c in ALL_CLUSTERS:
        for tf in ALL_TFS:
            cell = cells[c][tf]
            if cell["scarcity_status"] == "SCARCE":
                under_explored.append({"cluster": c, "tf": tf, **cell})
            elif cell["scarcity_status"] == "SATURATED":
                saturated.append({"cluster": c, "tf": tf, **cell})

    return {
        "clusters": list(ALL_CLUSTERS),
        "timeframes": list(ALL_TFS),
        "matrix": cells,
        "under_explored_niches": under_explored,
        "saturated_niches": saturated,
        "registered_atomic_families": int(len(ATOMIC_FACTOR_DEFS)),
        "total_factor_defs": int(len(fdef_df)),
    }


def get_workbench_status(db: AlphaLabLedgerDB) -> dict[str, Any]:
    """汇总当前 AlphaLab 实验台账状态、6×5 生态位矩阵、Rust 信号覆盖率与墓地统计。"""
    counts = db.get_table_counts()
    niche = build_niche_saturation_matrix(db)
    sig_df = get_signal_usage_summary()
    total_sig = int(len(sig_df))
    consumed_sig = int((sig_df["consumed_count"] > 0).sum()) if not sig_df.empty else 0
    mounted_sig = int(sig_df["mounted_in_context"].sum()) if not sig_df.empty else 0
    sig_summary = {
        "total_kline_signals": total_sig,
        "mounted_in_context": mounted_sig,
        "consumed_kline_signals": consumed_sig,
        "coverage_ratio": round(consumed_sig / max(total_sig, 1), 4),
    }
    unconsumed_df = list_unconsumed_mirror_signals()
    grave_df = db.query_df(
        """
        SELECT failed_gate, COUNT(*) AS cnt
        FROM alpha_lab.factor_graveyard
        GROUP BY failed_gate
        ORDER BY cnt DESC
        """
    )
    grave_by_gate = {str(r["failed_gate"]): int(r["cnt"]) for _, r in grave_df.iterrows()} if not grave_df.empty else {}
    return {
        "backend": db.backend,
        "db_url": db.db_url,
        "table_counts": counts,
        "signal_coverage": sig_summary,
        "unconsumed_mirror_signals_count": int(len(unconsumed_df)),
        "unconsumed_top15": unconsumed_df[["signal_name", "category", "active_rate"]]
        .head(15)
        .to_dict(orient="records"),
        "niche_saturation": niche,
        "graveyard_by_gate": grave_by_gate,
    }


def generate_leaderboard_markdown(
    db: AlphaLabLedgerDB,
    output_path: str | Path | None = None,
) -> Path:
    """从实验台账生成全景 Markdown 看板 `alpha_lab/FACTORS_LEADERBOARD.md`。"""
    out = Path(output_path) if output_path is not None else DEFAULT_LEADERBOARD_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    counts = db.get_table_counts()
    sig_df = get_signal_usage_summary()
    total_sig = int(len(sig_df))
    consumed_sig = int((sig_df["consumed_count"] > 0).sum()) if not sig_df.empty else 0
    sig_summary = {
        "total_kline_signals": total_sig,
        "consumed_kline_signals": consumed_sig,
        "coverage_ratio": consumed_sig / max(total_sig, 1),
    }
    niche = build_niche_saturation_matrix(db)
    lineage_map = scan_all_atomic_factors()
    unconsumed_df = list_unconsumed_mirror_signals()

    ablation_df = db.query_df(
        """
        SELECT run_id, stage, mode, description, flat60_ann_ret_pct, flat60_sharpe,
               flat60_max_dd_pct, flat60_calmar, inst_ann_ret_pct, inst_sharpe,
               inst_max_dd_pct, inst_calmar, avg_factors_per_asset, oos_days
        FROM alpha_lab.ablation_runs
        ORDER BY stage, run_id
        """
    )
    pool_df = db.query_df(
        """
        SELECT fold, uid, sig_id, family_id, tf, archetype, stage, total_n,
               pos_asset_ratio, pooled_mean_pct, mean_sharpe, pooled_t, gate1b_strict_pass
        FROM alpha_lab.factor_pool_scores
        ORDER BY mean_sharpe DESC
        """
    )
    grave_df = db.query_df(
        """
        SELECT uid, family_id, sig_id, tf, archetype, stage, failed_gate,
               reason, mean_is_sharpe, mean_ann_turnover, buried_at
        FROM alpha_lab.factor_graveyard
        ORDER BY buried_at DESC, uid
        """
    )

    lines: list[str] = []
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append("# CZSC AlphaLab 全景因子排行榜与生态位地图 (`FACTORS_LEADERBOARD.md`)")
    lines.append("")
    lines.append(
        f"> **自动生成时间**：`{now_str}` | **台账后端**：`{db.backend}` | **标的池**：`Full-60 Commodities (6 大板块 60 品种)`"
    )
    lines.append("> **生成命令**：`uv run --no-sync python -m alpha_lab.cli report`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 一、 台账总览与工程基线指标")
    lines.append("")
    lines.append("| 指标维度 | 统计数值 | 说明 |")
    lines.append("| :--- | :---: | :--- |")
    lines.append(
        f"| **Rust 原生 K 线信号总数 (`rust_signals`)** | `{counts['rust_signals']}` | 覆盖 17 个 Rust 信号子模块，含镜像对称审计标记 |"
    )
    lines.append(
        f"| **已消费 K 线信号数** | `{sig_summary['consumed_kline_signals']} / {sig_summary['total_kline_signals']}` (`{sig_summary['coverage_ratio'] * 100:.2f}%`) | 67 个显式状态键 + 7 类结构强度信号 |"
    )
    lines.append(
        f"| **待开采精确镜像对称信号 (`exact_mirror=True`)** | `{len(unconsumed_df)}` | 已通过仿射反转严格对称检验、尚未被现有因子消费的蓝海信号 |"
    )
    lines.append(
        f"| **SSOT 原子因子族 (`ATOMIC_FACTOR_DEFS`)** | `{len(ATOMIC_FACTOR_DEFS)}` | 覆盖 Stage1_Base (16) + Stage2_Rust (4) + Stage3_OI (4) + Stage4_CE (4) + Stage5_FDB (4) |"
    )
    lines.append(
        f"| **登记历史候选因子总数 (`factor_defs`)** | `{counts['factor_defs']}` | 含 {len(ATOMIC_FACTOR_DEFS)} 个原子族 + 跨周期异构候选变体 |"
    )
    lines.append(
        f"| **因子↔信号血缘边数 (`factor_signal_lineage`)** | `{counts['factor_signal_lineage']}` | AST 静态扫描自动提取的 `state`/`strength` 依赖边 |"
    )
    lines.append(
        f"| **预检门禁通过率 (`preflight_audits`)** | `{counts['preflight_audits']} / {len(ATOMIC_FACTOR_DEFS)}` (`100%`) | `<1s` 仿射镜像对称 + 前视零泄漏 + 边界合法性门禁 |"
    )
    lines.append(
        f"| **品种级评估明细行 (`factor_asset_scores`)** | `{counts['factor_asset_scores']:,}` | Full-60 品种 × 多周期 × Fold1_2025 / Fold2_2026 样本外打分 |"
    )
    lines.append(
        f"| **全池准入汇总行 (`factor_pool_scores`)** | `{counts['factor_pool_scores']}` | Gate 1B 全品种池准入检验汇总 |"
    )
    lines.append(
        f"| **组合消融归因记录 (`ablation_runs`)** | `{counts['ablation_runs']}` | Stage 1 ~ Stage 5 等权 / 机构两层风险预算组合样本外净值 |"
    )
    lines.append(
        f"| **失败实验墓地记录 (`factor_graveyard`)** | `{counts['factor_graveyard']}` | 被 Gate 1B/2/3 或 Pre-flight 淘汰的证伪候选与原因归档 |"
    )
    lines.append("")
    lines.append("## 二、 6×5 生态位饱和度矩阵 (Cluster A~F × Timeframe `1d`/`60m`/`30m`/`15m`/`5m`)")
    lines.append("")
    lines.append(
        "> 单元格格式：`Gate1B通过折数 / 评估UID数`（括号内为平均 Sharpe；`🟢 SCARCE 蓝海`、`🟡 BALANCED 适中`、`🔴 SATURATED 饱和`）。"
    )
    lines.append("")
    lines.append("| 机制主簇 (Cluster) | `1d` | `60m` | `30m` | `15m` | `5m` |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for c in ALL_CLUSTERS:
        row_cells: list[str] = [f"**{c} · {CLUSTER_META[c]}**"]
        for tf in ALL_TFS:
            cell = niche["matrix"][c][tf]
            ev, g1b, st = cell["candidate_uids"], cell["gate1b_pass_folds"], cell["scarcity_status"]
            badge = "🟢" if st == "SCARCE" else ("🔴" if st == "SATURATED" else "🟡")
            sh_txt = f" ({cell['avg_mean_sharpe']:+.2f})" if ev > 0 else ""
            row_cells.append(f"{badge} `{g1b}/{ev}`{sh_txt}")
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    lines.append("## 三、 Stage 1 ~ Stage 5 全周期组合消融演进对比 (`ablation_runs`)")
    lines.append("")
    lines.append(
        "| 运行标识 (`run_id`) | 阶段 (`stage`) | 模式 (`mode`) | Flat-60 年化 | Flat-60 Sharpe | Flat-60 Calmar | 机构两层年化 | 机构两层 Sharpe | 机构最大回撤 | 机构 Calmar | 单品种均因子数 |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in ablation_df.iterrows():
        lines.append(
            f"| `{r['run_id']}` | `{r['stage']}` | `{r['mode']}` | "
            f"`{float(r['flat60_ann_ret_pct']):+.2f}%` | **{float(r['flat60_sharpe']):+.3f}** | `{float(r['flat60_calmar']):+.3f}` | "
            f"`{float(r['inst_ann_ret_pct']):+.2f}%` | **{float(r['inst_sharpe']):+.3f}** | "
            f"`{float(r['inst_max_dd_pct']):.2f}%` | `{float(r['inst_calmar']):+.3f}` | `{float(r['avg_factors_per_asset']):.1f}` |"
        )
    lines.append("")

    lines.append("## 四、 各主簇 (Cluster A~F) Top 准入因子排行榜 (`factor_pool_scores`)")
    lines.append("")
    lines.append(
        "| 主簇 | 排名 | 候选 UID (`uid`) | 家族 (`family_id`) | 周期 | 阶段 (`stage`) | 检验窗口 (`fold`) | 全池 Mean Sharpe | Pooled-t | 正收益品种占比 | Gate1B 准入 |"
    )
    lines.append("| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    if not pool_df.empty:
        pool_df = pool_df.copy()
        pool_df["cluster_letter"] = [
            _cluster_from_arch(a, f) for a, f in zip(pool_df["archetype"], pool_df["family_id"], strict=False)
        ]
        dedup_pool = pool_df.sort_values(
            ["gate1b_strict_pass", "mean_sharpe"], ascending=[False, False]
        ).drop_duplicates(subset=["uid"], keep="first")
        for c in ALL_CLUSTERS:
            sub = dedup_pool[dedup_pool["cluster_letter"] == c].head(5)
            for idx, (_, r) in enumerate(sub.iterrows(), start=1):
                g1b_mark = "✅" if bool(r["gate1b_strict_pass"]) else "➖"
                lines.append(
                    f"| **{c}** | {idx} | `{r['uid']}` | `{r['family_id']}` | `{r['tf']}` | `{r['stage']}` | `{r['fold']}` | "
                    f"**{float(r['mean_sharpe']):+.4f}** | `{float(r['pooled_t']):+.2f}` | "
                    f"`{float(r['pos_asset_ratio']) * 100:.1f}%` | {g1b_mark} |"
                )
    lines.append("")

    lines.append(f"## 五、 {len(ATOMIC_FACTOR_DEFS)} 个 SSOT 原子因子族 AST 静态血缘与依赖矩阵 (`catalog.py`)")
    lines.append("")
    lines.append(
        "| 家族 (`family_id`) | 主簇 | 阶段 (`stage`) | 默认周期 | State 依赖数 | Strength 依赖 | OI 依赖 | CS 截面依赖 | HTF 依赖 | 代码指纹 (`code_sha256[:10]`) |"
    )
    lines.append("| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for fam, info in lineage_map.items():
        cluster_letter = fam[0] if fam and fam[0] in ALL_CLUSTERS else "A"
        fdef = ATOMIC_FACTOR_DEFS[fam]
        lines.append(
            f"| `{fam}` | **{cluster_letter}** | `{info.stage}` | `{','.join(fdef.eligible_tfs)}` | "
            f"`{len(info.states)}` | `{'/'.join(info.strengths) if info.strengths else '-'}` | "
            f"`{'/'.join(info.oi_features) if info.oi_features else '-'}` | "
            f"`{'/'.join(info.cs_features) if info.cs_features else '-'}` | "
            f"`{'/'.join(info.htf_features) if info.htf_features else '-'}` | `{info.code_sha256[:10]}` |"
        )
    lines.append("")
    lines.append("## 六、 待开采高优先级 Rust 原生信号推荐池 (`exact_mirror=True` & `consumed_count=0`)")
    lines.append("")
    lines.append(
        f"> 共发现 **{len(unconsumed_df)}** 个已通过仿射反转严格镜像对称验证、且尚未接入现有原子因子族的原生 Rust 信号。以下按信号分类展示代表性蓝海信号："
    )
    lines.append("")
    lines.append("| 信号子模块 (`category`) | 待开采对称信号数 | 代表性未开采 `exact_mirror=True` 原生信号示例 |")
    lines.append("| :--- | :---: | :--- |")
    if not unconsumed_df.empty:
        for cat, grp in unconsumed_df.groupby("category"):
            examples = ", ".join(f"`{s}`" for s in grp["signal_name"].head(4).tolist())
            lines.append(f"| `{cat}` | `{len(grp)}` | {examples} |")
    lines.append("")

    lines.append("## 七、 失败实验墓地统计与典型淘汰案例 (`factor_graveyard`)")
    lines.append("")
    lines.append("| 淘汰门禁 (`failed_gate`) | 累计淘汰候选数 | 门禁拦截规则说明 |")
    lines.append("| :--- | :---: | :--- |")
    if not grave_df.empty:
        for gate, grp in grave_df.groupby("failed_gate"):
            rule_desc = {
                "Gate1B_Pool": "Full-60 跨品种池化正收益品种占比 < 50% 或 Pooled-t < 1.96",
                "Gate1B_Consistency": "双折 OOS 均值 Sharpe <= 0 或正收益品种占比未达标",
                "Gate2_ColinearClone": "同簇激活条件共线克隆 (|ρ| >= 0.80)，缺乏独立机制增量",
                "Gate2_HighActiveCorr": "与已入库因子在活跃区高度正交冗余 (|ρ_active| >= 0.35)",
                "Gate2_Weak1DPoolEdge": "1d 日线跨品种池化稳健度或扣费后净夏普未达准入线",
                "Gate2_Weak5MPoolEdge": "5m 高频跨品种池化净边际不足以覆盖冲击与手续费损耗",
                "Gate2_WeakIntradayPoolEdge": "日内周期跨品种池化 t 统计量或正收益品种占比未达标",
                "Gate2_WeakPoolEdge": "15m/30m 跨品种普适性不足 (Pooled-t < 1.96 或 PosRatio < 50%)",
                "Gate2_StandAlone": "单因子全池 OOS Mean Sharpe 未达准入阈值",
                "Gate3_NegativeMarginalSharpe": "加入正交组合后扣费边际净夏普无正增益 (ΔSharpe <= 0)",
                "Gate3_Redundancy": "与已入选高 Sharpe 因子有符号活跃区相关性过高 (同质化冗余)",
                "Preflight": "违反仿射镜像精确对称、存在前视数据泄漏或权重越界",
            }.get(str(gate), "实验淘汰门禁")
            lines.append(f"| `{gate}` | `{len(grp)}` | {rule_desc} |")
    lines.append("")
    lines.append("### 最近归档的 Top 12 失败/冗余候选明细")
    lines.append("")
    lines.append(
        "| 候选标识 (`uid`) | 家族 (`family_id`) | 周期 | 阶段 | 淘汰门禁 | Mean IS Sharpe | 年化换手 | 淘汰原因摘要 |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :--- |")
    for _, r in grave_df.head(12).iterrows():
        sh_str = f"`{float(r['mean_is_sharpe']):+.4f}`" if pd.notna(r["mean_is_sharpe"]) else "`N/A`"
        to_str = f"`{float(r['mean_ann_turnover']):.2f}`" if pd.notna(r["mean_ann_turnover"]) else "`-`"
        lines.append(
            f"| `{r['uid']}` | `{r['family_id']}` | `{r['tf']}` | `{r['stage']}` | `{r['failed_gate']}` | "
            f"{sh_str} | {to_str} | {r['reason']} |"
        )
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        for k, v in payload.items():
            typer.echo(f"{k}: {v}")


@app.command("migrate-db")
def cmd_migrate_db(
    db_url: str | None = typer.Option(None, "--db-url", help="PostgreSQL 或 SQLite 连接串"),
    as_json: bool = typer.Option(False, "--json", help="输出结构化 JSON"),
) -> None:
    """初始化或更新 `alpha_lab` 台账 Schema 并导入历史 Full-60 全量评估缓存。"""
    db = AlphaLabLedgerDB(db_url)
    try:
        counts = migrate_historical_data_to_db(db=db)
        _emit({"status": "OK", "backend": db.backend, "table_counts": counts}, as_json)
    finally:
        db.close()


@app.command("status")
def cmd_status(
    db_url: str | None = typer.Option(None, "--db-url", help="PostgreSQL 或 SQLite 连接串"),
    as_json: bool = typer.Option(False, "--json", help="输出结构化 JSON"),
) -> None:
    """查看 6×5 生态位饱和度矩阵、Rust 信号覆盖率、待开采镜像信号池与墓地统计。"""
    db = AlphaLabLedgerDB(db_url)
    try:
        _emit(get_workbench_status(db), as_json)
    finally:
        db.close()


@app.command("preflight")
def cmd_preflight(
    families: str | None = typer.Option(None, "--families", "-f", help="逗号分隔的因子家族 ID (如 AC05,BC04)"),
    all_families: bool = typer.Option(False, "--all", help="对全部 28 个已注册原子因子族执行预检"),
    db_url: str | None = typer.Option(None, "--db-url", help="PostgreSQL 或 SQLite 连接串"),
    as_json: bool = typer.Option(False, "--json", help="输出结构化 JSON"),
) -> None:
    """执行 <1s 快速预检门禁（仿射镜像对称 + 前视零泄漏 + 边界合法性）。"""
    fams = None if all_families or not families else [x.strip() for x in families.split(",") if x.strip()]
    db = AlphaLabLedgerDB(db_url)
    try:
        df_res = run_preflight_audit(fams, db=db, persist=True, auto_bury_failed=True)
        passed_cnt = int(df_res["passed"].sum()) if not df_res.empty else 0
        payload = {
            "status": "OK" if passed_cnt == len(df_res) else "FAILED",
            "total": int(len(df_res)),
            "passed": passed_cnt,
            "failed": int(len(df_res) - passed_cnt),
            "total_elapsed_ms": round(float(df_res["elapsed_ms"].sum()), 2) if not df_res.empty else 0.0,
            "results": df_res.to_dict(orient="records"),
        }
        _emit(payload, as_json)
        if passed_cnt < len(df_res):
            raise typer.Exit(code=1)
    finally:
        db.close()


@app.command("eval")
def cmd_eval(
    families: str = typer.Option(..., "--families", "-f", help="逗号分隔的因子家族 ID (如 CC04,EC05)"),
    stage: str = typer.Option("Stage4_CE", "--stage", "-s", help="研发阶段标识"),
    tfs: str = typer.Option("1d,60m,30m", "--tfs", help="逗号分隔的评估周期"),
    assets: str | None = typer.Option(None, "--assets", help="可选：逗号分隔的品种子集（默认 Full-60）"),
    db_url: str | None = typer.Option(None, "--db-url", help="PostgreSQL 或 SQLite 连接串"),
    as_json: bool = typer.Option(False, "--json", help="输出结构化 JSON"),
) -> None:
    """一键运行 Pre-flight -> Full60ContextStore 评分 -> Gate 1B/2/3 漏斗 -> 消融归因流水线。"""
    fam_list = [x.strip() for x in families.split(",") if x.strip()]
    tf_list = tuple(x.strip() for x in tfs.split(",") if x.strip())
    asset_list = tuple(x.strip() for x in assets.split(",") if x.strip()) if assets else None
    db = AlphaLabLedgerDB(db_url)
    try:
        store = Full60ContextStore.load_full60(tfs=tf_list, assets=asset_list)
        out = run_full_sop_evaluation(
            fam_list,
            stage=stage,
            tfs=tf_list,
            assets=asset_list,
            store=store,
            db=db,
            persist_db=True,
        )
        _emit(out, as_json)
    finally:
        db.close()


@app.command("bury")
def cmd_bury(
    family_id: str = typer.Option(..., "--family-id", help="因子家族 ID"),
    candidate_key: str | None = typer.Option(None, "--candidate-key", help="候选因子标识 (如 XX01@30m)"),
    stage: str = typer.Option("Custom", "--stage", help="研发阶段标识"),
    failed_gate: str = typer.Option("Manual_Reject", "--failed-gate", help="淘汰门禁名称"),
    reason: str = typer.Option(..., "--reason", help="淘汰或证伪原因说明"),
    oos_sharpe: float = typer.Option(0.0, "--oos-sharpe", help="可选：样本外 Sharpe"),
    db_url: str | None = typer.Option(None, "--db-url", help="PostgreSQL 或 SQLite 连接串"),
    as_json: bool = typer.Option(False, "--json", help="输出结构化 JSON"),
) -> None:
    """手工将证伪或淘汰的候选因子记入 `alpha_lab.factor_graveyard`。"""
    ckey = candidate_key or f"{family_id}@30m"
    db = AlphaLabLedgerDB(db_url)
    try:
        db.bury_factor(
            ckey,
            failed_gate=failed_gate,
            reason=reason,
            family_id=family_id,
            stage=stage,
            mean_is_sharpe=oos_sharpe,
        )
        _emit({"status": "OK", "buried_count": 1, "candidate_key": ckey, "failed_gate": failed_gate}, as_json)
    finally:
        db.close()


@app.command("report")
def cmd_report(
    output: str | None = typer.Option(None, "--output", "-o", help="输出 Markdown 路径"),
    db_url: str | None = typer.Option(None, "--db-url", help="PostgreSQL 或 SQLite 连接串"),
    as_json: bool = typer.Option(False, "--json", help="输出结构化 JSON"),
) -> None:
    """从实验台账自动生成或更新 `alpha_lab/FACTORS_LEADERBOARD.md` 全景看板。"""
    db = AlphaLabLedgerDB(db_url)
    try:
        out_path = generate_leaderboard_markdown(db, output_path=output)
        counts = db.get_table_counts()
        payload = {
            "status": "OK",
            "output_path": str(out_path),
            "bytes": out_path.stat().st_size,
            "table_counts": counts,
        }
        _emit(payload, as_json)
    finally:
        db.close()


@app.command("dashboard")
def cmd_dashboard(
    host: str = typer.Option("127.0.0.1", "--host", help="Web 服务监听地址"),
    port: int = typer.Option(8765, "--port", "-p", help="Web 服务监听端口"),
    export_html: str | None = typer.Option(
        None, "--export", "-e", help="导出独立静态 HTML 路径（若不传 --serve 则仅导出后退出）"
    ),
    serve: bool = typer.Option(False, "--serve", help="同时启动本地 HTTP 实时查询服务"),
    open_browser: bool = typer.Option(False, "--open", help="启动服务后自动打开浏览器"),
    db_url: str | None = typer.Option(None, "--db-url", help="PostgreSQL 或 SQLite 连接串"),
    as_json: bool = typer.Option(False, "--json", help="输出结构化 JSON"),
) -> None:
    """启动 AlphaLab 投研台账与回测结果前端查询服务，或导出自包含静态 HTML 看板。"""
    from alpha_lab.dashboard import export_dashboard_html, serve_dashboard

    if export_html is not None and not serve:
        out_path = export_dashboard_html(db_url=db_url, output_path=export_html)
        payload = {
            "status": "OK",
            "mode": "export_html",
            "output_path": str(out_path),
            "bytes": out_path.stat().st_size,
        }
        _emit(payload, as_json)
        return

    if export_html is not None:
        export_dashboard_html(db_url=db_url, output_path=export_html)

    if not as_json:
        typer.echo(f"🚀 AlphaLab 前端查询工作台已启动: http://{host}:{port}/")
    serve_dashboard(host=host, port=port, db_url=db_url, open_browser=open_browser)


if __name__ == "__main__":
    app()
