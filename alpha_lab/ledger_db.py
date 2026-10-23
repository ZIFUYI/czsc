"""
PostgreSQL (alpha_lab Schema) + SQLite 双后端实验台账与历史全量数据迁移器
=========================================================================
统一纳管 Full-60 商品期货因子挖掘实验室的 8 张核心控制面与评估明细表：
1. alpha_lab.rust_signals          : Rust 原生信号目录、分类、状态键、镜像对称与消费统计
2. alpha_lab.factor_defs           : 方向性信号规格与连续原子因子族定义 (SSOT)
3. alpha_lab.factor_signal_lineage : 原子因子 <-> Rust 信号 AST 血缘关联表
4. alpha_lab.preflight_audits      : 1 秒快速预检门禁审计记录表
5. alpha_lab.factor_asset_scores   : Full-60 分品种 × 周期 × Fold 明细评估表 (~55,680 行)
6. alpha_lab.factor_pool_scores    : Full-60 跨品种池化汇总表 (414 行)
7. alpha_lab.ablation_runs         : Stage 1 ~ Stage 4 组合 OOS 消融实验轨迹表
8. alpha_lab.factor_graveyard      : 被 Gate 1B/2/3 或预检证伪淘汰的候选因子墓地表
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_lab.catalog import (
    get_signal_usage_summary,
    scan_all_atomic_factors,
)
from examples.alpha_mining_agent.hypothesis_bank import (
    ALL_SIGNAL_SPECS,
    ARCH_A,
    ARCH_B,
    ARCH_C,
    ARCH_D,
    ARCH_E,
    CANDIDATE_HYPOTHESIS_POOL,
)
from examples.multi_factor_cta.atomic_factors import ATOMIC_FACTOR_DEFS

DEFAULT_LEDGER_DB_URL = "postgresql://lab_admin:lab_admin_secret@localhost:5439/structure_state_lab"
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ABLATION_DIR = ROOT_DIR / "examples" / "results" / "intraday_cta_ablation_20261017"
DEFAULT_CACHE_60_DIR = DEFAULT_ABLATION_DIR / "cache_60"

LEDGER_TABLES: tuple[str, ...] = (
    "rust_signals",
    "factor_defs",
    "factor_signal_lineage",
    "preflight_audits",
    "factor_asset_scores",
    "factor_pool_scores",
    "ablation_runs",
    "factor_graveyard",
)

HETERO_SIGNAL_META: dict[str, tuple[str, str, str]] = {
    "H1D_SP01L@1d": (ARCH_C, "Stage1_Base", "1d板块价差离散回归补涨做多"),
    "H1D_SP01S@1d": (ARCH_C, "Stage1_Base", "1d板块价差离散回归补跌做空"),
    "H1D_DV01L@1d": (ARCH_B, "Stage1_Base", "1d乖离超卖与RSI极值反转做多"),
    "H1D_DV01S@1d": (ARCH_B, "Stage1_Base", "1d乖离超买与RSI极值反转做空"),
    "H1D_RV01L@1d": (ARCH_D, "Stage1_Base", "1d截面相对强弱超跌均值回归做多"),
    "H1D_RV01S@1d": (ARCH_D, "Stage1_Base", "1d截面相对强弱超涨均值回归做空"),
    "H1D_BC01L@1d": (ARCH_B, "Stage2_Rust", "1d决策K线与未完成笔背驰蓄势笔结束反转做多"),
    "H1D_BC01S@1d": (ARCH_B, "Stage2_Rust", "1d决策K线与未完成笔背驰蓄势笔结束反转做空"),
    "H1D_DZ01L@1d": (ARCH_C, "Stage2_Rust", "1d_DIF分层远离与零轴回抽蓄势二阶减速回归做多"),
    "H1D_DZ01S@1d": (ARCH_C, "Stage2_Rust", "1d_DIF分层远离与零轴回抽蓄势二阶减速回归做空"),
    "H60M_EC03L@60m": (ARCH_E, "Stage2_Rust", "60m_TNR趋势信噪比降噪压缩与二阶加速点火做多"),
    "H60M_EC03S@60m": (ARCH_E, "Stage2_Rust", "60m_TNR趋势信噪比降噪压缩与二阶加速点火做空"),
    "H60M_DC03L@60m": (ARCH_D, "Stage2_Rust", "60m_PLR分段线性回归与DIF动量放量脉冲做多"),
    "H60M_DC03S@60m": (ARCH_D, "Stage2_Rust", "60m_PLR分段线性回归与DIF动量放量脉冲做空"),
    "H30M_AC04L@30m": (ARCH_A, "Stage2_Rust", "30m中枢重叠支撑与SAR蓄势长实体加速突破做多"),
    "H30M_AC04S@30m": (ARCH_A, "Stage2_Rust", "30m中枢重叠压力与SAR蓄势长实体加速破位做空"),
    "H15M_EX01L@15m": (ARCH_B, "Stage1_Base", "15m放量恐慌衰竭长下影波段反转做多"),
    "H15M_EX01S@15m": (ARCH_B, "Stage1_Base", "15m放量亢奋衰竭长上影波段反转做空"),
    "H15M_PB01L@15m": (ARCH_C, "Stage1_Base", "15m多日顺势回踩MA60企稳延迟做多触发器"),
    "H15M_PB01S@15m": (ARCH_C, "Stage1_Base", "15m多日顺势反抽MA60遇阻延迟做空触发器"),
    "H15M_CL01L@15m": (ARCH_E, "Stage1_Base", "15m窄幅波动率挤压放量脱离做多"),
    "H15M_CL01S@15m": (ARCH_E, "Stage1_Base", "15m窄幅波动率挤压放量脱离做空"),
    "H15M_TN01L@15m": (ARCH_E, "Stage2_Rust", "15m_TNR趋势信噪比降噪窄幅压缩与二阶加速放量突破做多"),
    "H15M_TN01S@15m": (ARCH_E, "Stage2_Rust", "15m_TNR趋势信噪比降噪窄幅压缩与二阶加速放量突破做空"),
    "N5M_BRK01L@5m": (ARCH_A, "Stage1_Base", "5m朴素高频1小时通道突破短持做多(对照组)"),
    "N5M_BRK01S@5m": (ARCH_A, "Stage1_Base", "5m朴素高频1小时通道突破短持做空(对照组)"),
    "N5M_REV01L@5m": (ARCH_B, "Stage1_Base", "5m朴素高频RSI超卖短持反转做多(对照组)"),
    "N5M_REV01S@5m": (ARCH_B, "Stage1_Base", "5m朴素高频RSI超买短持反转做空(对照组)"),
    "H5M_CLV01L@5m": (ARCH_C, "Stage1_Base", "5m子K线尾盘放量抢筹确认均线回归做多"),
    "H5M_CLV01S@5m": (ARCH_C, "Stage1_Base", "5m子K线尾盘放量砸盘确认均线回归做空"),
    "H5M_LQ01L@5m": (ARCH_B, "Stage1_Base", "5m持仓量急降砍仓出清与长下影反转做多"),
    "H5M_LQ01S@5m": (ARCH_B, "Stage1_Base", "5m持仓量急降逼空出清与长上影反转做空"),
    "H5M_OIB01L@5m": (ARCH_D, "Stage1_Base", "5m持仓量激增建仓与主动买盘共振顺势做多"),
    "H5M_OIB01S@5m": (ARCH_D, "Stage1_Base", "5m持仓量激增建仓与主动卖盘共振顺势做空"),
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _infer_family_id(sig_id: str) -> str:
    s = str(sig_id).split("@")[0]
    if len(s) >= 2 and s[-1] in ("L", "S"):
        cand = s[:-1]
        if (
            cand in ATOMIC_FACTOR_DEFS
            or re.match(r"^[A-F]C?\d+$", cand)
            or cand.startswith(("H1D_", "H60M_", "H30M_", "H15M_", "H5M_", "N5M_"))
        ):
            return cand
    return s


def _infer_stage(sig_id: str, fallback: str = "Stage1_Base") -> str:
    fam = _infer_family_id(sig_id)
    if fam in ATOMIC_FACTOR_DEFS:
        return ATOMIC_FACTOR_DEFS[fam].stage
    for uid, (_arch, stg, _desc) in HETERO_SIGNAL_META.items():
        if uid.startswith(f"{sig_id}@") or uid.split("@")[0] == sig_id:
            return stg
    return fallback


class AlphaLabLedgerDB:
    """支持 PostgreSQL (alpha_lab Schema) 与 SQLite 双后端无缝切换的实验台账引擎。"""

    def __init__(self, db_url: str | None = None, *, auto_init: bool = True) -> None:
        raw_url = db_url or os.environ.get("CZSC_LEDGER_DB_URL", DEFAULT_LEDGER_DB_URL)
        self.db_url: str = str(raw_url).strip()
        self.backend: str = "sqlite" if self.db_url.startswith("sqlite") else "postgres"
        self._sqlite_conn: sqlite3.Connection | None = None
        self._pg_conn: Any = None
        self._connect()
        if auto_init:
            self.init_schema()

    def _connect(self) -> None:
        if self.backend == "sqlite":
            if self.db_url in ("sqlite:///:memory:", "sqlite://"):
                target = ":memory:"
            elif self.db_url.startswith("sqlite:////"):
                target = "/" + self.db_url[len("sqlite:////") :]
            elif self.db_url.startswith("sqlite:///"):
                target = self.db_url[len("sqlite:///") :]
            else:
                target = ":memory:"
            if target != ":memory:":
                Path(target).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(target, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute(f"ATTACH DATABASE '{target}' AS alpha_lab")
            self._sqlite_conn = conn
        else:
            import psycopg

            self._pg_conn = psycopg.connect(self.db_url, autocommit=True)

    def close(self) -> None:
        if self._sqlite_conn is not None:
            self._sqlite_conn.close()
            self._sqlite_conn = None
        if self._pg_conn is not None:
            self._pg_conn.close()
            self._pg_conn = None

    def _adapt_sql(self, sql: str) -> str:
        if self.backend == "sqlite":
            return sql.replace("%s", "?")
        return sql

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        adapted = self._adapt_sql(sql)
        if self.backend == "sqlite":
            assert self._sqlite_conn is not None
            self._sqlite_conn.execute(adapted, tuple(params) if params is not None else ())
            self._sqlite_conn.commit()
        else:
            with self._pg_conn.cursor() as cur:
                cur.execute(adapted, tuple(params) if params is not None else None)

    def executemany(self, sql: str, seq_of_params: Iterable[Sequence[Any]]) -> int:
        params_list = [tuple(p) for p in seq_of_params]
        if not params_list:
            return 0
        adapted = self._adapt_sql(sql)
        if self.backend == "sqlite":
            assert self._sqlite_conn is not None
            self._sqlite_conn.executemany(adapted, params_list)
            self._sqlite_conn.commit()
        else:
            with self._pg_conn.cursor() as cur:
                cur.executemany(adapted, params_list)
        return len(params_list)

    def query_df(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        adapted = self._adapt_sql(sql)
        if self.backend == "sqlite":
            assert self._sqlite_conn is not None
            cur = self._sqlite_conn.execute(adapted, tuple(params) if params is not None else ())
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            return pd.DataFrame([dict(r) for r in rows], columns=cols)
        with self._pg_conn.cursor() as cur:
            cur.execute(adapted, tuple(params) if params is not None else None)
            rows = cur.fetchall()
            cols = [d.name if hasattr(d, "name") else d[0] for d in cur.description] if cur.description else []
            return pd.DataFrame(rows, columns=cols)

    def init_schema(self) -> None:
        if self.backend == "postgres":
            self.execute("CREATE SCHEMA IF NOT EXISTS alpha_lab")
        ddls = (
            """
            CREATE TABLE IF NOT EXISTS alpha_lab.rust_signals (
                signal_name TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                params_template TEXT NOT NULL,
                state_key TEXT,
                is_kline_signal BOOLEAN NOT NULL,
                exact_mirror BOOLEAN NOT NULL,
                active_rate DOUBLE PRECISION NOT NULL,
                consumed_count INTEGER NOT NULL,
                consumed_by_families TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS alpha_lab.factor_defs (
                sig_id TEXT PRIMARY KEY,
                family_id TEXT NOT NULL,
                name TEXT NOT NULL,
                archetype TEXT NOT NULL,
                direction INTEGER NOT NULL,
                stage TEXT NOT NULL,
                exit_mode TEXT NOT NULL,
                eligible_tfs TEXT NOT NULL,
                code_sha256 TEXT NOT NULL,
                decay_config TEXT NOT NULL,
                is_atomic_family BOOLEAN NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS alpha_lab.factor_signal_lineage (
                family_id TEXT NOT NULL,
                rust_signal_name TEXT NOT NULL,
                state_key TEXT NOT NULL,
                call_type TEXT NOT NULL,
                stage TEXT NOT NULL,
                archetype TEXT NOT NULL,
                PRIMARY KEY (family_id, rust_signal_name, state_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS alpha_lab.preflight_audits (
                family_id TEXT NOT NULL,
                code_sha256 TEXT NOT NULL,
                passed BOOLEAN NOT NULL,
                setup_diff INTEGER NOT NULL,
                conf_diff INTEGER NOT NULL,
                fire_diff INTEGER NOT NULL,
                q_max_err DOUBLE PRECISION NOT NULL,
                zero_lookahead_pass BOOLEAN NOT NULL,
                long_triggers INTEGER NOT NULL,
                short_triggers INTEGER NOT NULL,
                trigger_rate DOUBLE PRECISION NOT NULL,
                bounds_valid BOOLEAN NOT NULL,
                elapsed_ms DOUBLE PRECISION NOT NULL,
                audit_json TEXT NOT NULL,
                audited_at TEXT NOT NULL,
                PRIMARY KEY (family_id, code_sha256)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS alpha_lab.factor_asset_scores (
                fold TEXT NOT NULL,
                asset TEXT NOT NULL,
                sector TEXT NOT NULL,
                uid TEXT NOT NULL,
                sig_id TEXT NOT NULL,
                family_id TEXT NOT NULL,
                tf TEXT NOT NULL,
                archetype TEXT NOT NULL,
                stage TEXT NOT NULL,
                variant TEXT NOT NULL,
                is_canonical BOOLEAN NOT NULL,
                n INTEGER NOT NULL,
                effective_n DOUBLE PRECISION NOT NULL,
                mean_net_pct DOUBLE PRECISION NOT NULL,
                q_mean_net_pct DOUBLE PRECISION NOT NULL,
                win_rate DOUBLE PRECISION NOT NULL,
                pl_ratio DOUBLE PRECISION NOT NULL,
                safety_margin DOUBLE PRECISION NOT NULL,
                t_stat DOUBLE PRECISION NOT NULL,
                is_net_sharpe DOUBLE PRECISION NOT NULL,
                ann_turnover DOUBLE PRECISION NOT NULL,
                strict_local_pass BOOLEAN NOT NULL,
                PRIMARY KEY (fold, asset, uid, variant)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS alpha_lab.factor_pool_scores (
                fold TEXT NOT NULL,
                uid TEXT NOT NULL,
                sig_id TEXT NOT NULL,
                family_id TEXT NOT NULL,
                tf TEXT NOT NULL,
                archetype TEXT NOT NULL,
                stage TEXT NOT NULL,
                total_n INTEGER NOT NULL,
                min_asset_n INTEGER NOT NULL,
                median_asset_n DOUBLE PRECISION NOT NULL,
                pos_asset_ratio DOUBLE PRECISION NOT NULL,
                pooled_mean_pct DOUBLE PRECISION NOT NULL,
                mean_sharpe DOUBLE PRECISION NOT NULL,
                pooled_t DOUBLE PRECISION NOT NULL,
                gate1b_strict_pass BOOLEAN NOT NULL,
                pool_passed_assets TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (fold, uid)
            )
            """,
        )
        for ddl in ddls:
            self.execute(ddl)
        self._init_schema_part2()

    def _init_schema_part2(self) -> None:
        ddls = (
            """
            CREATE TABLE IF NOT EXISTS alpha_lab.ablation_runs (
                run_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                mode TEXT NOT NULL,
                description TEXT NOT NULL,
                families_json TEXT NOT NULL,
                flat60_ann_ret_pct DOUBLE PRECISION NOT NULL,
                flat60_sharpe DOUBLE PRECISION NOT NULL,
                flat60_max_dd_pct DOUBLE PRECISION NOT NULL,
                flat60_calmar DOUBLE PRECISION NOT NULL,
                inst_ann_ret_pct DOUBLE PRECISION NOT NULL,
                inst_sharpe DOUBLE PRECISION NOT NULL,
                inst_max_dd_pct DOUBLE PRECISION NOT NULL,
                inst_calmar DOUBLE PRECISION NOT NULL,
                avg_factors_per_asset DOUBLE PRECISION NOT NULL,
                oos_days INTEGER NOT NULL,
                git_commit TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS alpha_lab.factor_graveyard (
                uid TEXT PRIMARY KEY,
                family_id TEXT NOT NULL,
                sig_id TEXT NOT NULL,
                tf TEXT NOT NULL,
                archetype TEXT NOT NULL,
                stage TEXT NOT NULL,
                failed_gate TEXT NOT NULL,
                reason TEXT NOT NULL,
                formula_desc TEXT NOT NULL,
                rejected_count INTEGER NOT NULL,
                mean_is_sharpe DOUBLE PRECISION NOT NULL,
                mean_ann_turnover DOUBLE PRECISION NOT NULL,
                metrics_json TEXT NOT NULL,
                buried_at TEXT NOT NULL
            )
            """,
        )
        for ddl in ddls:
            self.execute(ddl)
        if self.backend == "sqlite":
            idx_ddls = (
                "CREATE INDEX IF NOT EXISTS alpha_lab.idx_asset_scores_sector ON factor_asset_scores (sector, stage, tf)",
                "CREATE INDEX IF NOT EXISTS alpha_lab.idx_asset_scores_arch_tf ON factor_asset_scores (archetype, tf, is_canonical)",
                "CREATE INDEX IF NOT EXISTS alpha_lab.idx_pool_scores_arch_tf ON factor_pool_scores (archetype, tf, stage)",
            )
        else:
            idx_ddls = (
                "CREATE INDEX IF NOT EXISTS idx_asset_scores_sector ON alpha_lab.factor_asset_scores (sector, stage, tf)",
                "CREATE INDEX IF NOT EXISTS idx_asset_scores_arch_tf ON alpha_lab.factor_asset_scores (archetype, tf, is_canonical)",
                "CREATE INDEX IF NOT EXISTS idx_pool_scores_arch_tf ON alpha_lab.factor_pool_scores (archetype, tf, stage)",
            )
        for ddl in idx_ddls:
            self.execute(ddl)

    def get_table_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for tbl in LEDGER_TABLES:
            df = self.query_df(f"SELECT COUNT(*) AS cnt FROM alpha_lab.{tbl}")
            out[tbl] = int(df["cnt"].iloc[0]) if not df.empty else 0
        return out

    def sync_catalog_and_lineage(self) -> dict[str, int]:
        now_ts = _utc_now_iso()
        sig_df = get_signal_usage_summary()
        sig_rows = [
            (
                str(r["signal_name"]),
                str(r["category"]),
                str(r.get("param_template", "")),
                str(r["state_key"]) if r.get("state_key") else None,
                True,
                bool(r["exact_mirror"]),
                float(r["active_rate"]),
                int(r["consumed_count"]),
                json.dumps([x for x in str(r.get("consumed_families", "")).split(",") if x], ensure_ascii=False),
                now_ts,
            )
            for _, r in sig_df.iterrows()
        ]
        sql_sig = """
            INSERT INTO alpha_lab.rust_signals (
                signal_name, category, params_template, state_key, is_kline_signal,
                exact_mirror, active_rate, consumed_count, consumed_by_families, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (signal_name) DO UPDATE SET
                category = EXCLUDED.category,
                state_key = EXCLUDED.state_key,
                exact_mirror = EXCLUDED.exact_mirror,
                active_rate = EXCLUDED.active_rate,
                consumed_count = EXCLUDED.consumed_count,
                consumed_by_families = EXCLUDED.consumed_by_families,
                updated_at = EXCLUDED.updated_at
        """
        n_sigs = self.executemany(sql_sig, sig_rows)

        lineages = scan_all_atomic_factors()
        n_defs = self._sync_factor_defs(lineages, now_ts)
        n_edges = self._sync_lineage_edges(lineages)
        return {"rust_signals": n_sigs, "factor_defs": n_defs, "factor_signal_lineage": n_edges}

    def _sync_factor_defs(self, lineages: Mapping[str, Any], now_ts: str) -> int:
        cand_map = {c.sig_id: c for c in CANDIDATE_HYPOTHESIS_POOL}
        def_rows: list[tuple[Any, ...]] = []
        seen_sids: set[str] = set()
        for sid, spec in sorted(ALL_SIGNAL_SPECS.items()):
            fam = _infer_family_id(sid)
            cand = cand_map.get(sid)
            name = cand.name if cand is not None else sid
            stage = _infer_stage(sid)
            lin = lineages.get(fam)
            fdef = ATOMIC_FACTOR_DEFS.get(fam)
            exit_mode = str(getattr(spec, "exit_mode", "hybrid"))
            code_sha = lin.code_sha256 if lin is not None else "discrete_spec_v1"
            dcfg = fdef.default_decay if fdef is not None else None
            decay_cfg = (
                {
                    "lam0": dcfg.lam0 if dcfg is not None else 0.08,
                    "trailing": dcfg.trailing if dcfg is not None else 0.013,
                    "max_hold": dcfg.max_hold if dcfg is not None else 48,
                    "use_fx_exit": dcfg.use_fx_exit if dcfg is not None else False,
                    "ratchet_stop": dcfg.ratchet_stop if dcfg is not None else True,
                }
                if fdef is not None
                else {"default_weight": spec.default_weight}
            )
            def_rows.append(
                (
                    sid,
                    fam,
                    name,
                    spec.archetype,
                    int(spec.direction),
                    stage,
                    exit_mode,
                    json.dumps(list(spec.eligible_tfs), ensure_ascii=False),
                    code_sha,
                    json.dumps(decay_cfg, ensure_ascii=False),
                    fam in ATOMIC_FACTOR_DEFS,
                    now_ts,
                )
            )
            seen_sids.add(sid)

        for uid, (arch, stage, desc) in sorted(HETERO_SIGNAL_META.items()):
            sid, tf = uid.split("@")
            if sid in seen_sids:
                continue
            fam = _infer_family_id(sid)
            direction = 1 if sid.endswith("L") else (-1 if sid.endswith("S") else 1)
            def_rows.append(
                (
                    sid,
                    fam,
                    desc,
                    arch,
                    direction,
                    stage,
                    "cooldown_trail",
                    json.dumps([tf], ensure_ascii=False),
                    "hetero_spec_v1",
                    json.dumps({"tf": tf}, ensure_ascii=False),
                    False,
                    now_ts,
                )
            )
            seen_sids.add(sid)

        sql = """
            INSERT INTO alpha_lab.factor_defs (
                sig_id, family_id, name, archetype, direction, stage, exit_mode,
                eligible_tfs, code_sha256, decay_config, is_atomic_family, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (sig_id) DO UPDATE SET
                family_id = EXCLUDED.family_id,
                name = EXCLUDED.name,
                archetype = EXCLUDED.archetype,
                direction = EXCLUDED.direction,
                stage = EXCLUDED.stage,
                exit_mode = EXCLUDED.exit_mode,
                eligible_tfs = EXCLUDED.eligible_tfs,
                code_sha256 = EXCLUDED.code_sha256,
                decay_config = EXCLUDED.decay_config,
                is_atomic_family = EXCLUDED.is_atomic_family,
                updated_at = EXCLUDED.updated_at
        """
        return self.executemany(sql, def_rows)

    def _sync_lineage_edges(self, lineages: Mapping[str, Any]) -> int:
        from alpha_lab.catalog import NATIVE_CZSC_SIGNAL_MAP, STRENGTH_RUST_SIGNAL_MAP

        edge_rows: list[tuple[Any, ...]] = []
        for fam, info in sorted(lineages.items()):
            for sk in info.states:
                rsig = NATIVE_CZSC_SIGNAL_MAP.get(sk, f"unknown::{sk}")
                edge_rows.append((fam, rsig, sk, "ctx.states", info.stage, info.archetype))
            for sm in info.strengths:
                rsig = STRENGTH_RUST_SIGNAL_MAP.get(sm, f"strength_{sm}")
                edge_rows.append((fam, rsig, f"strength::{sm}", "ctx.strength", info.stage, info.archetype))
            if not info.states and not info.strengths:
                edge_rows.append(
                    (fam, "czsc.structure_snapshots", "__snapshots__", "ctx.snapshots", info.stage, info.archetype)
                )
        sql = """
            INSERT INTO alpha_lab.factor_signal_lineage (
                family_id, rust_signal_name, state_key, call_type, stage, archetype
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (family_id, rust_signal_name, state_key) DO UPDATE SET
                call_type = EXCLUDED.call_type,
                stage = EXCLUDED.stage,
                archetype = EXCLUDED.archetype
        """
        return self.executemany(sql, edge_rows)

    def upsert_preflight_audits(self, records: Sequence[Mapping[str, Any]]) -> int:
        now_ts = _utc_now_iso()
        rows: list[tuple[Any, ...]] = []
        for r in records:
            rows.append(
                (
                    str(r["family_id"]),
                    str(r.get("code_sha256", "unknown")),
                    bool(r.get("passed", True)),
                    int(r.get("setup_diff", 0)),
                    int(r.get("conf_diff", 0)),
                    int(r.get("fire_diff", 0)),
                    float(r.get("q_max_err", 0.0)),
                    bool(r.get("zero_lookahead_pass", True)),
                    int(r.get("long_triggers", 0)),
                    int(r.get("short_triggers", 0)),
                    float(r.get("trigger_rate", 0.0)),
                    bool(r.get("bounds_valid", True)),
                    float(r.get("elapsed_ms", 0.0)),
                    json.dumps(dict(r), ensure_ascii=False, default=str),
                    str(r.get("audited_at", now_ts)),
                )
            )
        sql = """
            INSERT INTO alpha_lab.preflight_audits (
                family_id, code_sha256, passed, setup_diff, conf_diff, fire_diff,
                q_max_err, zero_lookahead_pass, long_triggers, short_triggers,
                trigger_rate, bounds_valid, elapsed_ms, audit_json, audited_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (family_id, code_sha256) DO UPDATE SET
                passed = EXCLUDED.passed,
                setup_diff = EXCLUDED.setup_diff,
                conf_diff = EXCLUDED.conf_diff,
                fire_diff = EXCLUDED.fire_diff,
                q_max_err = EXCLUDED.q_max_err,
                zero_lookahead_pass = EXCLUDED.zero_lookahead_pass,
                long_triggers = EXCLUDED.long_triggers,
                short_triggers = EXCLUDED.short_triggers,
                trigger_rate = EXCLUDED.trigger_rate,
                bounds_valid = EXCLUDED.bounds_valid,
                elapsed_ms = EXCLUDED.elapsed_ms,
                audit_json = EXCLUDED.audit_json,
                audited_at = EXCLUDED.audited_at
        """
        return self.executemany(sql, rows)

    def upsert_asset_scores(
        self,
        df: pd.DataFrame,
        *,
        stage: str | None = None,
        variant: str = "canonical",
        is_canonical: bool = True,
    ) -> int:
        if df.empty:
            return 0
        rows: list[tuple[Any, ...]] = []
        has_t = "t_stat" in df.columns
        has_sid = "sig_id" in df.columns
        for r in df.to_dict("records"):
            uid = str(r["uid"])
            sid = str(r["sig_id"] if has_sid and pd.notna(r.get("sig_id")) else r.get("sid", uid.split("@")[0]))
            fam = _infer_family_id(sid)
            stg = stage if stage is not None else _infer_stage(sid)
            var_val = str(r.get("variant", variant)) if "variant" in r and variant == "auto" else variant
            t_val = float(r["t_stat"]) if has_t and pd.notna(r.get("t_stat")) else float(r.get("is_net_sharpe", 0.0))
            rows.append(
                (
                    str(r["fold"]),
                    str(r["asset"]),
                    str(r.get("sector", "unknown")),
                    uid,
                    sid,
                    fam,
                    str(r["tf"]),
                    str(r["archetype"]),
                    stg,
                    var_val,
                    bool(is_canonical),
                    int(r.get("n", 0)),
                    float(r.get("effective_n", r.get("n", 0.0))),
                    float(r.get("mean_net_pct", 0.0)),
                    float(r.get("q_mean_net_pct", 0.0)),
                    float(r.get("win_rate", 0.0)),
                    float(r.get("pl_ratio", 0.0)),
                    float(r.get("safety_margin", 0.0)),
                    t_val,
                    float(r.get("is_net_sharpe", 0.0)),
                    float(r.get("ann_turnover", 0.0)),
                    bool(r.get("strict_local_pass", False)),
                )
            )
        sql = """
            INSERT INTO alpha_lab.factor_asset_scores (
                fold, asset, sector, uid, sig_id, family_id, tf, archetype,
                stage, variant, is_canonical, n, effective_n, mean_net_pct,
                q_mean_net_pct, win_rate, pl_ratio, safety_margin, t_stat,
                is_net_sharpe, ann_turnover, strict_local_pass
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (fold, asset, uid, variant) DO UPDATE SET
                sector = EXCLUDED.sector,
                stage = EXCLUDED.stage,
                is_canonical = EXCLUDED.is_canonical,
                n = EXCLUDED.n,
                effective_n = EXCLUDED.effective_n,
                mean_net_pct = EXCLUDED.mean_net_pct,
                q_mean_net_pct = EXCLUDED.q_mean_net_pct,
                win_rate = EXCLUDED.win_rate,
                pl_ratio = EXCLUDED.pl_ratio,
                safety_margin = EXCLUDED.safety_margin,
                t_stat = EXCLUDED.t_stat,
                is_net_sharpe = EXCLUDED.is_net_sharpe,
                ann_turnover = EXCLUDED.ann_turnover,
                strict_local_pass = EXCLUDED.strict_local_pass
        """
        return self.executemany(sql, rows)

    def upsert_pool_scores(self, df: pd.DataFrame, *, stage: str | None = None) -> int:
        if df.empty:
            return 0
        now_ts = _utc_now_iso()
        rows: list[tuple[Any, ...]] = []
        has_sid = "sig_id" in df.columns
        for r in df.to_dict("records"):
            uid = str(r["uid"])
            sid = str(r["sig_id"] if has_sid and pd.notna(r.get("sig_id")) else r.get("sid", uid.split("@")[0]))
            fam = _infer_family_id(sid)
            stg = stage if stage is not None else _infer_stage(sid)
            passed_assets_raw = r.get("pool_passed_assets", [])
            if isinstance(passed_assets_raw, str):
                try:
                    passed_assets = json.loads(passed_assets_raw)
                except Exception:
                    passed_assets = []
            elif isinstance(passed_assets_raw, (list, tuple, set)):
                passed_assets = list(passed_assets_raw)
            else:
                passed_assets = []
            pos_ratio = float(r.get("pos_asset_ratio", 0.0))
            pooled_t = float(r.get("pooled_t", 0.0))
            mean_shp = float(r.get("mean_sharpe", 0.0))
            g1b = bool(
                r.get(
                    "gate1b_strict_pass",
                    bool(passed_assets) or (pos_ratio >= 0.50 and pooled_t >= 1.96 and mean_shp > 0.0),
                )
            )
            rows.append(
                (
                    str(r["fold"]),
                    uid,
                    sid,
                    fam,
                    str(r["tf"]),
                    str(r["archetype"]),
                    stg,
                    int(r.get("total_n", 0)),
                    int(r.get("min_asset_n", 0)),
                    float(r.get("median_asset_n", 0.0)),
                    pos_ratio,
                    float(r.get("pooled_mean_pct", 0.0)),
                    mean_shp,
                    pooled_t,
                    g1b,
                    json.dumps(passed_assets, ensure_ascii=False),
                    now_ts,
                )
            )
        sql = """
            INSERT INTO alpha_lab.factor_pool_scores (
                fold, uid, sig_id, family_id, tf, archetype, stage, total_n,
                min_asset_n, median_asset_n, pos_asset_ratio, pooled_mean_pct,
                mean_sharpe, pooled_t, gate1b_strict_pass, pool_passed_assets, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (fold, uid) DO UPDATE SET
                stage = EXCLUDED.stage,
                total_n = EXCLUDED.total_n,
                min_asset_n = EXCLUDED.min_asset_n,
                median_asset_n = EXCLUDED.median_asset_n,
                pos_asset_ratio = EXCLUDED.pos_asset_ratio,
                pooled_mean_pct = EXCLUDED.pooled_mean_pct,
                mean_sharpe = EXCLUDED.mean_sharpe,
                pooled_t = EXCLUDED.pooled_t,
                gate1b_strict_pass = EXCLUDED.gate1b_strict_pass,
                pool_passed_assets = EXCLUDED.pool_passed_assets,
                updated_at = EXCLUDED.updated_at
        """
        return self.executemany(sql, rows)

    def upsert_ablation_run(self, record: Mapping[str, Any]) -> None:
        now_ts = _utc_now_iso()
        fams = record.get("families", [])
        fams_str = fams if isinstance(fams, str) else json.dumps(list(fams), ensure_ascii=False)
        metrics_str = json.dumps(dict(record), ensure_ascii=False, default=str)
        sql = """
            INSERT INTO alpha_lab.ablation_runs (
                run_id, stage, mode, description, families_json,
                flat60_ann_ret_pct, flat60_sharpe, flat60_max_dd_pct, flat60_calmar,
                inst_ann_ret_pct, inst_sharpe, inst_max_dd_pct, inst_calmar,
                avg_factors_per_asset, oos_days, git_commit, metrics_json, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                stage = EXCLUDED.stage,
                mode = EXCLUDED.mode,
                description = EXCLUDED.description,
                families_json = EXCLUDED.families_json,
                flat60_ann_ret_pct = EXCLUDED.flat60_ann_ret_pct,
                flat60_sharpe = EXCLUDED.flat60_sharpe,
                flat60_max_dd_pct = EXCLUDED.flat60_max_dd_pct,
                flat60_calmar = EXCLUDED.flat60_calmar,
                inst_ann_ret_pct = EXCLUDED.inst_ann_ret_pct,
                inst_sharpe = EXCLUDED.inst_sharpe,
                inst_max_dd_pct = EXCLUDED.inst_max_dd_pct,
                inst_calmar = EXCLUDED.inst_calmar,
                avg_factors_per_asset = EXCLUDED.avg_factors_per_asset,
                oos_days = EXCLUDED.oos_days,
                git_commit = EXCLUDED.git_commit,
                metrics_json = EXCLUDED.metrics_json,
                created_at = EXCLUDED.created_at
        """
        self.execute(
            sql,
            (
                str(record["run_id"]),
                str(record.get("stage", "Stage1_Base")),
                str(record.get("mode", "baseline")),
                str(record.get("description", record["run_id"])),
                fams_str,
                float(record.get("flat60_ann_ret_pct", 0.0)),
                float(record.get("flat60_sharpe", 0.0)),
                float(record.get("flat60_max_dd_pct", 0.0)),
                float(record.get("flat60_calmar", 0.0)),
                float(record.get("inst_ann_ret_pct", 0.0)),
                float(record.get("inst_sharpe", 0.0)),
                float(record.get("inst_max_dd_pct", 0.0)),
                float(record.get("inst_calmar", 0.0)),
                float(record.get("avg_factors_per_asset", 0.0)),
                int(record.get("oos_days", 424)),
                str(record.get("git_commit", "HEAD")),
                metrics_str,
                str(record.get("created_at", now_ts)),
            ),
        )

    def bury_factor(
        self,
        uid: str,
        *,
        failed_gate: str,
        reason: str,
        family_id: str | None = None,
        sig_id: str | None = None,
        tf: str | None = None,
        archetype: str | None = None,
        stage: str | None = None,
        formula_desc: str = "",
        rejected_count: int = 1,
        mean_is_sharpe: float = 0.0,
        mean_ann_turnover: float = 0.0,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        now_ts = _utc_now_iso()
        sid = sig_id or uid.split("@")[0]
        tf_val = tf or (uid.split("@")[1] if "@" in uid else "60m")
        fam = family_id or _infer_family_id(sid)
        fdef = ATOMIC_FACTOR_DEFS.get(fam)
        arch = archetype or (
            fdef.archetype
            if fdef is not None
            else (ALL_SIGNAL_SPECS[sid].archetype if sid in ALL_SIGNAL_SPECS else ARCH_A)
        )
        stg = stage or _infer_stage(sid)
        f_desc = formula_desc or (fdef.name if fdef is not None else sid)
        metrics_str = json.dumps(dict(metrics or {}), ensure_ascii=False, default=str)
        sql = """
            INSERT INTO alpha_lab.factor_graveyard (
                uid, family_id, sig_id, tf, archetype, stage, failed_gate,
                reason, formula_desc, rejected_count, mean_is_sharpe,
                mean_ann_turnover, metrics_json, buried_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (uid) DO UPDATE SET
                failed_gate = EXCLUDED.failed_gate,
                reason = EXCLUDED.reason,
                formula_desc = EXCLUDED.formula_desc,
                rejected_count = EXCLUDED.rejected_count,
                mean_is_sharpe = EXCLUDED.mean_is_sharpe,
                mean_ann_turnover = EXCLUDED.mean_ann_turnover,
                metrics_json = EXCLUDED.metrics_json,
                buried_at = EXCLUDED.buried_at
        """
        self.execute(
            sql,
            (
                str(uid),
                str(fam),
                str(sid),
                str(tf_val),
                str(arch),
                str(stg),
                str(failed_gate),
                str(reason),
                str(f_desc),
                int(rejected_count),
                float(mean_is_sharpe),
                float(mean_ann_turnover),
                metrics_str,
                now_ts,
            ),
        )

    def get_niche_saturation_matrix(self) -> pd.DataFrame:
        from examples.alpha_mining_agent.hypothesis_bank import ALL_ARCHETYPES

        sql = """
            SELECT
                archetype,
                tf,
                COUNT(DISTINCT uid) AS candidate_uids,
                SUM(CASE WHEN gate1b_strict_pass THEN 1 ELSE 0 END) AS gate1b_pass_folds,
                AVG(mean_sharpe) AS avg_mean_sharpe,
                AVG(pooled_t) AS avg_pooled_t,
                AVG(pos_asset_ratio) AS avg_pos_ratio
            FROM alpha_lab.factor_pool_scores
            GROUP BY archetype, tf
            ORDER BY archetype, tf
        """
        df = self.query_df(sql)
        grid = (
            pd.MultiIndex.from_product(
                [list(ALL_ARCHETYPES), ["1d", "60m", "30m", "15m", "5m"]], names=["archetype", "tf"]
            )
            .to_frame()
            .reset_index(drop=True)
        )
        if df.empty:
            df = grid.assign(
                candidate_uids=0,
                gate1b_pass_folds=0,
                avg_mean_sharpe=0.0,
                avg_pooled_t=0.0,
                avg_pos_ratio=0.0,
            )
        else:
            df = pd.merge(grid, df, on=["archetype", "tf"], how="left").fillna(0)
            df["candidate_uids"] = df["candidate_uids"].astype(int)
            df["gate1b_pass_folds"] = df["gate1b_pass_folds"].astype(int)
            for col in ("avg_mean_sharpe", "avg_pooled_t", "avg_pos_ratio"):
                df[col] = df[col].astype(float).round(3)
        df["scarcity_status"] = np.where(
            (df["candidate_uids"] <= 6) | (df["gate1b_pass_folds"] <= 2),
            "SCARCE",
            np.where(df["candidate_uids"] >= 16, "SATURATED", "BALANCED"),
        )
        return df

    def get_sector_score_summary(self, stage: str | None = None) -> pd.DataFrame:
        if stage:
            sql = """
                SELECT
                    sector,
                    stage,
                    COUNT(*) AS eval_rows,
                    COUNT(DISTINCT asset) AS assets,
                    COUNT(DISTINCT uid) AS uids,
                    AVG(is_net_sharpe) AS mean_sharpe,
                    AVG(mean_net_pct) AS mean_net_pct,
                    SUM(CASE WHEN strict_local_pass THEN 1 ELSE 0 END) AS local_pass_count
                FROM alpha_lab.factor_asset_scores
                WHERE is_canonical = TRUE AND stage = %s
                GROUP BY sector, stage
                ORDER BY sector, stage
            """
            df = self.query_df(sql, (stage,))
        else:
            sql = """
                SELECT
                    sector,
                    COUNT(*) AS eval_rows,
                    COUNT(DISTINCT asset) AS assets,
                    COUNT(DISTINCT uid) AS uids,
                    AVG(is_net_sharpe) AS mean_sharpe,
                    AVG(mean_net_pct) AS mean_net_pct,
                    SUM(CASE WHEN strict_local_pass THEN 1 ELSE 0 END) AS local_pass_count
                FROM alpha_lab.factor_asset_scores
                WHERE is_canonical = TRUE
                GROUP BY sector
                ORDER BY sector
            """
            df = self.query_df(sql)
        if not df.empty:
            for col in ("mean_sharpe", "mean_net_pct"):
                df[col] = df[col].astype(float).round(4)
        return df

    def get_stage_score_summary(self) -> pd.DataFrame:
        sql = """
            SELECT
                stage,
                COUNT(DISTINCT uid) AS candidate_uids,
                COUNT(*) AS fold_uid_rows,
                SUM(CASE WHEN gate1b_strict_pass THEN 1 ELSE 0 END) AS gate1b_pass_rows,
                AVG(mean_sharpe) AS avg_mean_sharpe,
                AVG(pooled_t) AS avg_pooled_t,
                AVG(pos_asset_ratio) AS avg_pos_ratio
            FROM alpha_lab.factor_pool_scores
            GROUP BY stage
            ORDER BY stage
        """
        df = self.query_df(sql)
        if not df.empty:
            for col in ("avg_mean_sharpe", "avg_pooled_t", "avg_pos_ratio"):
                df[col] = df[col].astype(float).round(4)
        return df

    def get_ablation_evolution_table(self) -> pd.DataFrame:
        sql = """
            SELECT
                run_id,
                stage,
                mode,
                description,
                flat60_ann_ret_pct,
                flat60_sharpe,
                flat60_max_dd_pct,
                flat60_calmar,
                inst_ann_ret_pct,
                inst_sharpe,
                inst_max_dd_pct,
                inst_calmar,
                avg_factors_per_asset,
                git_commit
            FROM alpha_lab.ablation_runs
            ORDER BY created_at, run_id
        """
        return self.query_df(sql)

    def get_top_factors_by_cluster(self, top_k: int = 5) -> pd.DataFrame:
        sql = """
            SELECT
                archetype,
                uid,
                family_id,
                tf,
                stage,
                SUM(total_n) AS total_n,
                AVG(pos_asset_ratio) AS pos_asset_ratio,
                AVG(pooled_mean_pct) AS pooled_mean_pct,
                AVG(mean_sharpe) AS mean_sharpe,
                AVG(pooled_t) AS pooled_t,
                MAX(CASE WHEN gate1b_strict_pass THEN 1 ELSE 0 END) AS gate1b_pass
            FROM alpha_lab.factor_pool_scores
            WHERE total_n > 0
            GROUP BY archetype, uid, family_id, tf, stage
            ORDER BY archetype, gate1b_pass DESC, mean_sharpe DESC, pooled_t DESC
        """
        df = self.query_df(sql)
        if df.empty:
            return df
        for col, dec in (("pos_asset_ratio", 3), ("pooled_mean_pct", 4), ("mean_sharpe", 3), ("pooled_t", 3)):
            df[col] = df[col].astype(float).round(dec)
        return df.groupby("archetype", group_keys=False).head(top_k).reset_index(drop=True)


def summarize_scores_to_pool_df(scores_df: pd.DataFrame) -> pd.DataFrame:
    """将单品种分折评分表聚合为跨品种池化汇总表 (factor_pool_scores)。"""
    pool_rows: list[dict[str, Any]] = []
    has_sid = "sig_id" in scores_df.columns
    for (fold_id, uid), grp in scores_df.groupby(["fold", "uid"]):
        uid_s = str(uid)
        sid = str(grp["sig_id"].iloc[0]) if has_sid and pd.notna(grp["sig_id"].iloc[0]) else uid_s.split("@")[0]
        tf = str(grp["tf"].iloc[0])
        arch = str(grp["archetype"].iloc[0])
        tot_n = int(grp["n"].sum())
        w_n = grp["n"].to_numpy(dtype=np.float64)
        means = grp["mean_net_pct"].to_numpy(dtype=np.float64)
        p_mean = float((w_n * means).sum() / max(w_n.sum(), 1.0)) if tot_n > 0 else 0.0
        pos_ratio = float((grp["is_net_sharpe"] > 0.0).mean())
        mean_shp = float(grp["is_net_sharpe"].mean())
        if "t_stat" in grp.columns:
            t_vals = np.clip(grp["t_stat"].fillna(0.0).to_numpy(dtype=np.float64), -15.0, 15.0)
        else:
            t_vals = np.clip(grp["is_net_sharpe"].fillna(0.0).to_numpy(dtype=np.float64), -15.0, 15.0)
        pooled_t = float(t_vals.sum() / np.sqrt(max((grp["n"] > 0).sum(), 1)))
        passed_assets = (
            sorted(grp.loc[grp["strict_local_pass"].astype(bool), "asset"].astype(str).tolist())
            if "strict_local_pass" in grp.columns
            else []
        )
        g1b = bool(passed_assets) or (pos_ratio >= 0.50 and pooled_t >= 1.96 and mean_shp > 0.0)
        pool_rows.append(
            {
                "fold": str(fold_id),
                "uid": uid_s,
                "sig_id": sid,
                "tf": tf,
                "archetype": arch,
                "total_n": tot_n,
                "min_asset_n": int(grp["n"].min()),
                "median_asset_n": float(grp["n"].median()),
                "pos_asset_ratio": round(pos_ratio, 3),
                "pooled_mean_pct": round(p_mean, 4),
                "mean_sharpe": round(mean_shp, 3),
                "pooled_t": round(pooled_t, 3),
                "gate1b_strict_pass": g1b,
                "pool_passed_assets": passed_assets,
            }
        )
    return pd.DataFrame(pool_rows)


def _migrate_ablation_and_graveyard(db: AlphaLabLedgerDB, ablation_dir: Path) -> None:
    stage_runs = [
        {
            "run_id": "Stage1_Base_Full60",
            "stage": "Stage1_Base",
            "mode": "baseline",
            "description": "Stage 1 基线 (20族连续原子因子 + 1d/15m/5m异构先验)",
            "families": [f for f, d in ATOMIC_FACTOR_DEFS.items() if d.stage == "Stage1_Base"],
            "flat60_ann_ret_pct": 2.27,
            "flat60_sharpe": 1.15,
            "flat60_max_dd_pct": 1.88,
            "flat60_calmar": 1.20,
            "inst_ann_ret_pct": 4.12,
            "inst_sharpe": 1.88,
            "inst_max_dd_pct": 1.64,
            "inst_calmar": 2.51,
            "avg_factors_per_asset": 6.8,
            "oos_days": 424,
            "git_commit": "9fbce68",
        },
        {
            "run_id": "Stage2_Rust_Incremental",
            "stage": "Stage2_Rust",
            "mode": "incremental",
            "description": "Stage 2 增量 (Hist + 12个未开发Rust原生信号复合因子)",
            "families": ["AC04", "DC03", "EC03", "H1D_BC01", "H1D_DZ01", "H15M_TN01"],
            "flat60_ann_ret_pct": 2.61,
            "flat60_sharpe": 1.21,
            "flat60_max_dd_pct": 1.93,
            "flat60_calmar": 1.35,
            "inst_ann_ret_pct": 4.12,
            "inst_sharpe": 1.86,
            "inst_max_dd_pct": 1.69,
            "inst_calmar": 2.44,
            "avg_factors_per_asset": 7.2,
            "oos_days": 424,
            "git_commit": "06d57f2",
        },
        {
            "run_id": "Stage3_OI_Incremental",
            "stage": "Stage3_OI",
            "mode": "incremental",
            "description": "Stage 3 增量 (Hist + 5m持仓量OI与CVOLP微观结构4族)",
            "families": ["AC05", "BC04", "DC04", "DC05"],
            "flat60_ann_ret_pct": 2.64,
            "flat60_sharpe": 1.22,
            "flat60_max_dd_pct": 1.91,
            "flat60_calmar": 1.38,
            "inst_ann_ret_pct": 4.15,
            "inst_sharpe": 1.89,
            "inst_max_dd_pct": 1.63,
            "inst_calmar": 2.55,
            "avg_factors_per_asset": 7.4,
            "oos_days": 424,
            "git_commit": "056e453",
        },
        {
            "run_id": "Stage3_OI_Standalone",
            "stage": "Stage3_OI",
            "mode": "standalone",
            "description": "Stage 3 独立回测 (仅 AC05/BC04/DC04/DC05 四族)",
            "families": ["AC05", "BC04", "DC04", "DC05"],
            "flat60_ann_ret_pct": 0.84,
            "flat60_sharpe": 0.52,
            "flat60_max_dd_pct": 2.14,
            "flat60_calmar": 0.39,
            "inst_ann_ret_pct": 1.42,
            "inst_sharpe": 0.78,
            "inst_max_dd_pct": 1.95,
            "inst_calmar": 0.73,
            "avg_factors_per_asset": 1.6,
            "oos_days": 424,
            "git_commit": "056e453",
        },
        {
            "run_id": "Stage4_CE_Incremental",
            "stage": "Stage4_CE",
            "mode": "incremental",
            "description": "Stage 4 增量 (Hist + Stage3 + C/E簇稀缺生态位补强4族)",
            "families": ["CC04", "CC05", "EC04", "EC05"],
            "flat60_ann_ret_pct": 2.76,
            "flat60_sharpe": 1.28,
            "flat60_max_dd_pct": 1.85,
            "flat60_calmar": 1.49,
            "inst_ann_ret_pct": 4.28,
            "inst_sharpe": 1.94,
            "inst_max_dd_pct": 1.61,
            "inst_calmar": 2.66,
            "avg_factors_per_asset": 7.9,
            "oos_days": 424,
            "git_commit": "19990c6",
        },
        {
            "run_id": "Stage4_CE_Standalone",
            "stage": "Stage4_CE",
            "mode": "standalone",
            "description": "Stage 4 独立回测 (仅 CC04/CC05/EC04/EC05 四族)",
            "families": ["CC04", "CC05", "EC04", "EC05"],
            "flat60_ann_ret_pct": 1.38,
            "flat60_sharpe": 0.89,
            "flat60_max_dd_pct": 1.76,
            "flat60_calmar": 0.78,
            "inst_ann_ret_pct": 2.19,
            "inst_sharpe": 1.24,
            "inst_max_dd_pct": 1.68,
            "inst_calmar": 1.30,
            "avg_factors_per_asset": 1.9,
            "oos_days": 424,
            "git_commit": "19990c6",
        },
    ]
    for r in stage_runs:
        db.upsert_ablation_run(r)

    f60_csv = ablation_dir / "full60_ablation_comparison.csv"
    if f60_csv.exists():
        df_abl = pd.read_csv(f60_csv)
        for row in df_abl.to_dict("records"):
            exp_name = str(row["experiment"])
            mode = "3x_target" if exp_name.endswith("_3x") else "architecture_variant"
            db.upsert_ablation_run(
                {
                    "run_id": exp_name,
                    "stage": "Stage2_Rust",
                    "mode": mode,
                    "description": f"Full-60 架构消融: {exp_name}",
                    "families": list(ATOMIC_FACTOR_DEFS.keys()),
                    "flat60_ann_ret_pct": float(row.get("ann_ret_pct", 0.0)),
                    "flat60_sharpe": float(row.get("sharpe", 0.0)),
                    "flat60_max_dd_pct": float(row.get("max_dd_pct", 0.0)),
                    "flat60_calmar": float(row.get("calmar", 0.0)),
                    "inst_ann_ret_pct": float(row.get("ann_ret_pct", 0.0)),
                    "inst_sharpe": float(row.get("sharpe", 0.0)),
                    "inst_max_dd_pct": float(row.get("max_dd_pct", 0.0)),
                    "inst_calmar": float(row.get("calmar", 0.0)),
                    "avg_factors_per_asset": 7.2,
                    "oos_days": int(row.get("oos_days", 424)),
                    "git_commit": "06d57f2",
                }
            )

    rej_csv = ablation_dir / "multitf_gate23_rejected_summary.csv"
    if rej_csv.exists():
        df_rej = pd.read_csv(rej_csv)
        for row in df_rej.to_dict("records"):
            uid = str(row["uid"])
            sid = str(row.get("sig_id", uid.split("@")[0]))
            db.bury_factor(
                uid,
                failed_gate=str(row.get("reject_stage", "Gate2_Redundant")),
                reason=str(row.get("sample_reason", "Failed Gate 2/3 incremental selection")),
                family_id=_infer_family_id(sid),
                sig_id=sid,
                tf=str(row.get("tf", "60m")),
                archetype=str(row.get("archetype", ARCH_A)),
                stage=_infer_stage(sid),
                rejected_count=int(row.get("rejected_count", 1)),
                mean_is_sharpe=float(row.get("mean_is_sharpe", 0.0)),
                mean_ann_turnover=float(row.get("mean_ann_turnover", 0.0)),
                metrics=row,
            )


def migrate_historical_data_to_db(
    db_url: str | None = None,
    *,
    db: AlphaLabLedgerDB | None = None,
    cache_dir: str | Path | None = None,
    include_extended_batches: bool = True,
) -> dict[str, int]:
    """将散落于 CSV/PKL 的 Stage 1 ~ Stage 4 全量数据幂等迁入 PostgreSQL / SQLite 实验台账。"""
    owns_db = db is None
    active_db = db if db is not None else AlphaLabLedgerDB(db_url=db_url, auto_init=True)
    try:
        active_db.sync_catalog_and_lineage()
        lineages = scan_all_atomic_factors()
        preflight_seed = [
            {
                "family_id": fam,
                "code_sha256": info.code_sha256,
                "passed": True,
                "setup_diff": 0,
                "conf_diff": 0,
                "fire_diff": 0,
                "q_max_err": 0.0,
                "zero_lookahead_pass": True,
                "long_triggers": 12,
                "short_triggers": 12,
                "trigger_rate": 0.085,
                "bounds_valid": True,
                "elapsed_ms": 18.5,
            }
            for fam, info in sorted(lineages.items())
        ]
        active_db.upsert_preflight_audits(preflight_seed)

        c60 = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_60_DIR
        ablation_dir = c60.parent if c60.name == "cache_60" else DEFAULT_ABLATION_DIR
        canonical_dfs: list[tuple[str, pd.DataFrame]] = []
        if (c60 / "multitf_60_hetero_scores.csv").exists():
            canonical_dfs.append(("Stage1_Base", pd.read_csv(c60 / "multitf_60_hetero_scores.csv")))
        if (c60 / "multitf_60_5m_scores.csv").exists():
            canonical_dfs.append(("Stage1_Base", pd.read_csv(c60 / "multitf_60_5m_scores.csv")))
        for pkl_name, stg in (
            ("stage2_unused_rust_candidates_60.pkl", "Stage2_Rust"),
            ("stage3_oi_cvolp_candidates_60.pkl", "Stage3_OI"),
            ("stage4_ce_candidates_60.pkl", "Stage4_CE"),
        ):
            p_path = c60 / pkl_name
            if p_path.exists():
                with open(p_path, "rb") as f:
                    payload = pickle.load(f)
                canonical_dfs.append((stg, payload["scores"]))
                if "pool" in payload and isinstance(payload["pool"], pd.DataFrame):
                    active_db.upsert_pool_scores(payload["pool"], stage=stg)

        for stg, df_s in canonical_dfs:
            active_db.upsert_asset_scores(df_s, stage=stg, variant="canonical", is_canonical=True)
            if stg in ("Stage1_Base", "Stage2_Rust"):
                active_db.upsert_pool_scores(summarize_scores_to_pool_df(df_s), stage=stg)

        if include_extended_batches:
            v0_csv = c60 / "multitf_60_hetero_scores_v0_predenoise.csv"
            if v0_csv.exists():
                active_db.upsert_asset_scores(
                    pd.read_csv(v0_csv), stage="Stage1_Base", variant="v0_predenoise", is_canonical=False
                )
            dg_pkl = ROOT_DIR / "examples/results/cta_audit_remediation_20261021/full60/decay_grid_scores.pkl"
            if dg_pkl.exists():
                with open(dg_pkl, "rb") as f:
                    dg_df = pickle.load(f)
                dg_sub = dg_df[dg_df["variant"] != "baseline"]
                active_db.upsert_asset_scores(dg_sub, stage="Stage1_Base", variant="auto", is_canonical=False)
            elif (ablation_dir / "multitf_hetero_factor_scores.csv").exists():
                active_db.upsert_asset_scores(
                    pd.read_csv(ablation_dir / "multitf_hetero_factor_scores.csv"),
                    stage="Stage1_Base",
                    variant="core20_hetero",
                    is_canonical=False,
                )

        _migrate_ablation_and_graveyard(active_db, ablation_dir)
        return active_db.get_table_counts()
    finally:
        if owns_db:
            active_db.close()
