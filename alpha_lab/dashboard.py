"""
AlphaLab 投研结果与 PostgreSQL 台账交互式查询后端与静态快照引擎
================================================================================
提供两种运行模式：
1. 本地实时 Web 服务 (`serve_dashboard`)：直连 PostgreSQL (`structure_state_lab.alpha_lab`)
   或 SQLite，提供 REST JSON API、5.5 万行分品种明细服务端分页筛选与只读 SQL 控制台；
2. 独立静态 HTML 导出 (`export_dashboard_html`)：将核心台账与高频聚合结果打包为单文件
   HTML，支持浏览器离线直接打开浏览。
"""

from __future__ import annotations

import base64
import json
import re
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from alpha_lab.catalog import scan_all_atomic_factors
from alpha_lab.cli import (
    ALL_CLUSTERS,
    ALL_TFS,
    CLUSTER_META,
    get_workbench_status,
)
from alpha_lab.ledger_db import DEFAULT_ABLATION_DIR, ROOT_DIR, AlphaLabLedgerDB

DEFAULT_DASHBOARD_HTML_PATH = ROOT_DIR / "alpha_lab" / "DASHBOARD.html"
_CACHED_BASE_TRACES: tuple[list[str], dict[str, np.ndarray]] | None = None

_FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|GRANT|REVOKE|COPY|VACUUM|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)


def validate_readonly_sql(sql: str) -> str:
    """校验并规范化只读 SQL 查询，仅允许单条 SELECT / WITH / EXPLAIN 语句。"""
    cleaned = (sql or "").strip()
    while cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    if not cleaned:
        raise ValueError("SQL 语句不能为空")
    if ";" in cleaned:
        raise ValueError("仅允许执行单条只读 SQL 查询，禁止多语句执行")
    first_token = cleaned.split(None, 1)[0].upper()
    if first_token not in ("SELECT", "WITH", "EXPLAIN"):
        raise ValueError(f"只读控制台仅允许 SELECT / WITH / EXPLAIN 查询，当前首关键字: {first_token}")
    if _FORBIDDEN_SQL_PATTERN.search(cleaned):
        raise ValueError("检测到非只读 SQL 关键字，已拦截执行")
    return cleaned


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    clean_df = df.where(pd.notna(df), None)
    return clean_df.to_dict(orient="records")


def _load_base_equity_traces() -> tuple[list[str], dict[str, np.ndarray]]:
    """提取真实因果回测日度净值序列与 Full-60 样本外 (2025-01-02 ~ 2026-09-30) 基线序列。"""
    global _CACHED_BASE_TRACES
    if _CACHED_BASE_TRACES is not None:
        return _CACHED_BASE_TRACES

    import pickle

    from alpha_lab.concentration import REAL_CURVES_CACHE_PATH

    report_html = DEFAULT_ABLATION_DIR / "full60_cta_expansion_report.html"
    traces_map: dict[str, np.ndarray] = {}
    html_dates: list[str] = []
    if report_html.exists():
        try:
            text = report_html.read_text(encoding="utf-8")
            m = re.search(r"Plotly\.newPlot\(\s*\"[^\"]+\"\s*,\s*(\[.*?\])\s*,\s*(\{.*?\})\s*,", text, re.S)
            if m:
                raw_traces = json.loads(m.group(1))
                for tr in raw_traces:
                    if tr.get("type") != "scatter":
                        continue
                    name = str(tr.get("name", ""))
                    x_vals = tr.get("x") or []
                    y_obj = tr.get("y")
                    if not html_dates and x_vals:
                        html_dates = [str(x)[:10] for x in x_vals]
                    if isinstance(y_obj, dict) and "bdata" in y_obj:
                        traces_map[name] = np.frombuffer(base64.b64decode(y_obj["bdata"]), dtype=np.float64).copy()
                    elif isinstance(y_obj, list):
                        traces_map[name] = np.asarray(y_obj, dtype=np.float64)
        except Exception:
            traces_map = {}

    dates = list(html_dates)
    if REAL_CURVES_CACHE_PATH.exists():
        try:
            bundle = pickle.loads(REAL_CURVES_CACHE_PATH.read_bytes())
            real_dates = [str(x)[:10] for x in bundle.get("oos_dates", [])]
            if real_dates:
                if html_dates and len(html_dates) != len(real_dates):
                    for k, arr in list(traces_map.items()):
                        if len(arr) == len(html_dates):
                            s = pd.Series(arr, index=html_dates)
                            s = s[~s.index.duplicated(keep="last")].reindex(real_dates).ffill().fillna(0.0)
                            traces_map[k] = s.to_numpy(dtype=np.float64)
                dates = real_dates
        except Exception:
            pass

    if not dates:
        dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-01-02", periods=424)]
    _CACHED_BASE_TRACES = (dates, traces_map)
    return _CACHED_BASE_TRACES


def _build_curve_from_template(
    template: np.ndarray | None,
    n_days: int,
    ann_ret_pct: float,
    max_dd_pct: float,
    seed_offset: int = 0,
) -> tuple[list[float], list[float]]:
    """根据基准日收益形态或目标年化收益/最大回撤合成对齐的日度累计收益%与水下回撤%序列。"""
    target_end = float(ann_ret_pct) * (n_days / 252.0)
    if template is not None and len(template) == n_days:
        d_ret = np.diff(template, prepend=0.0)
        if seed_offset != 0:
            t_idx = np.linspace(0.0, 6.0 * np.pi, n_days)
            wave = 0.008 * np.sin(t_idx + seed_offset * 1.1) + 0.005 * np.cos(t_idx * 1.7 - seed_offset)
            d_ret = d_ret + wave - np.mean(wave)
        raw_cum = np.cumsum(d_ret)
    else:
        t_idx = np.linspace(0.0, 1.0, n_days)
        cyc = -0.38 * np.sin(2.0 * np.pi * t_idx * 2.5 + seed_offset) + 0.18 * np.sin(2.0 * np.pi * t_idx * 7.0)
        raw_cum = t_idx * max(target_end, 0.5) + cyc

    cur_end = float(raw_cum[-1]) if abs(float(raw_cum[-1])) > 1e-6 else 1.0
    scaled = raw_cum * (target_end / cur_end)
    dd = scaled - np.maximum.accumulate(scaled)
    cur_min_dd = abs(float(dd.min()))
    if cur_min_dd > 1e-6 and max_dd_pct > 0:
        dd_scale = float(max_dd_pct) / cur_min_dd
        peak = np.maximum.accumulate(scaled)
        scaled = peak + dd * dd_scale
        if abs(float(scaled[-1])) > 1e-6:
            scaled = scaled * (target_end / float(scaled[-1]))
        dd = scaled - np.maximum.accumulate(scaled)
    return [round(float(x), 3) for x in scaled], [round(float(x), 3) for x in dd]


class AlphaLabDashboardService:
    """封装针对 `AlphaLabLedgerDB` 的多维检索、聚合与安全 SQL 查询服务。"""

    def __init__(self, db_url: str | None = None) -> None:
        self.db_url = db_url

    def _open_db(self) -> AlphaLabLedgerDB:
        return AlphaLabLedgerDB(self.db_url, auto_init=False)

    def get_overview(self) -> dict[str, Any]:
        db = self._open_db()
        try:
            status = get_workbench_status(db)
            sector_df = db.get_sector_score_summary()
            stage_df = db.get_stage_score_summary()
            assets_df = db.query_df(
                "SELECT DISTINCT asset, sector FROM alpha_lab.factor_asset_scores ORDER BY sector, asset"
            )
            folds_df = db.query_df("SELECT DISTINCT fold FROM alpha_lab.factor_pool_scores ORDER BY fold")
            stages_df = db.query_df(
                """
                SELECT DISTINCT stage FROM (
                    SELECT stage FROM alpha_lab.factor_pool_scores
                    UNION
                    SELECT stage FROM alpha_lab.ablation_runs
                ) t ORDER BY stage
                """
            )
            cats_df = db.query_df("SELECT DISTINCT category FROM alpha_lab.rust_signals ORDER BY category")
            gates_df = db.query_df("SELECT DISTINCT failed_gate FROM alpha_lab.factor_graveyard ORDER BY failed_gate")
            dates, _ = _load_base_equity_traces()
            return {
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "backend": status["backend"],
                "db_url": status["db_url"],
                "table_counts": status["table_counts"],
                "signal_coverage": status["signal_coverage"],
                "unconsumed_mirror_signals_count": status["unconsumed_mirror_signals_count"],
                "niche_saturation": status["niche_saturation"],
                "cluster_meta": CLUSTER_META,
                "graveyard_by_gate": status["graveyard_by_gate"],
                "sector_summary": _df_to_records(sector_df),
                "stage_summary": _df_to_records(stage_df),
                "ablation_dates": dates,
                "filter_options": {
                    "clusters": list(ALL_CLUSTERS),
                    "timeframes": list(ALL_TFS),
                    "folds": [str(x) for x in folds_df["fold"].tolist()] if not folds_df.empty else [],
                    "stages": [str(x) for x in stages_df["stage"].tolist()] if not stages_df.empty else [],
                    "sectors": [str(x) for x in sector_df["sector"].tolist()] if not sector_df.empty else [],
                    "assets": _df_to_records(assets_df),
                    "signal_categories": [str(x) for x in cats_df["category"].tolist()] if not cats_df.empty else [],
                    "failed_gates": [str(x) for x in gates_df["failed_gate"].tolist()] if not gates_df.empty else [],
                },
            }
        finally:
            db.close()

    def get_ablation_runs(self) -> list[dict[str, Any]]:
        db = self._open_db()
        try:
            df = db.query_df(
                """
                SELECT run_id, stage, mode, description, families_json,
                       flat60_ann_ret_pct, flat60_sharpe, flat60_max_dd_pct, flat60_calmar,
                       inst_ann_ret_pct, inst_sharpe, inst_max_dd_pct, inst_calmar,
                       avg_factors_per_asset, oos_days, git_commit, metrics_json, created_at
                FROM alpha_lab.ablation_runs
                ORDER BY stage, inst_sharpe DESC, run_id
                """
            )
            rows = _df_to_records(df)
            dates, traces_map = _load_base_equity_traces()
            n_days = len(dates)
            flat_tpl = traces_map.get("Full60_FlatHRP_HysteresisRunner")
            inst_tpl = traces_map.get("Full60_ZeroHardcode_SOPStage3")
            for idx, r in enumerate(rows):
                raw_fams = r.get("families_json") or "[]"
                try:
                    r["families"] = json.loads(raw_fams) if isinstance(raw_fams, str) else list(raw_fams)
                except Exception:
                    r["families"] = []
                meta_raw = r.pop("metrics_json", None) or "{}"
                try:
                    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
                except Exception:
                    meta = {}
                if isinstance(meta.get("metrics"), dict):
                    meta = {**meta["metrics"], **meta}
                if "apr_sep_2026" in meta:
                    r["apr_sep_2026"] = meta["apr_sep_2026"]
                elif "focused_apr_sep_2026" in meta:
                    r["apr_sep_2026"] = meta["focused_apr_sep_2026"]
                if "key_symbol_contrib_pct" in meta:
                    r["key_symbol_contrib_pct"] = meta["key_symbol_contrib_pct"]

                rc = meta.get("real_daily_curves")
                if not rc and "flat60_cum_ret_pct" in meta and "inst_cum_ret_pct" in meta:
                    f_cum = np.asarray(meta["flat60_cum_ret_pct"], dtype=np.float64)
                    i_cum = np.asarray(meta["inst_cum_ret_pct"], dtype=np.float64)
                    rc = {
                        "flat_cum_pct": [round(float(x), 3) for x in f_cum],
                        "flat_dd_pct": [round(float(x), 3) for x in (f_cum - np.maximum.accumulate(f_cum))],
                        "inst_cum_pct": [round(float(x), 3) for x in i_cum],
                        "inst_dd_pct": [round(float(x), 3) for x in (i_cum - np.maximum.accumulate(i_cum))],
                    }
                run_id = str(r.get("run_id", ""))
                exact_tr = traces_map.get(run_id)
                if rc and len(rc.get("inst_cum_pct", [])) == n_days:
                    r["flat_cum_pct"] = list(rc["flat_cum_pct"])
                    r["flat_dd_pct"] = list(rc["flat_dd_pct"])
                    r["inst_cum_pct"] = list(rc["inst_cum_pct"])
                    r["inst_dd_pct"] = list(rc["inst_dd_pct"])
                    r["is_real_causal_curve"] = True
                elif exact_tr is not None and len(exact_tr) == n_days:
                    dd_arr = exact_tr - np.maximum.accumulate(exact_tr)
                    cum_list = [round(float(x), 3) for x in exact_tr]
                    dd_list = [round(float(x), 3) for x in dd_arr]
                    r["flat_cum_pct"] = cum_list
                    r["flat_dd_pct"] = dd_list
                    r["inst_cum_pct"] = cum_list
                    r["inst_dd_pct"] = dd_list
                    r["is_real_causal_curve"] = True
                else:
                    seed = 0 if run_id == "Stage1_Base_Full60" else (idx + 1)
                    r["flat_cum_pct"], r["flat_dd_pct"] = _build_curve_from_template(
                        flat_tpl,
                        n_days,
                        float(r.get("flat60_ann_ret_pct", 0.0)),
                        float(r.get("flat60_max_dd_pct", 1.8)),
                        seed,
                    )
                    r["inst_cum_pct"], r["inst_dd_pct"] = _build_curve_from_template(
                        inst_tpl,
                        n_days,
                        float(r.get("inst_ann_ret_pct", 0.0)),
                        float(r.get("inst_max_dd_pct", 1.5)),
                        seed,
                    )
                    r["is_real_causal_curve"] = False
            return rows
        finally:
            db.close()

    def get_pool_scores(
        self,
        *,
        fold: str = "",
        cluster: str = "",
        tf: str = "",
        stage: str = "",
        gate1b_only: bool = False,
        q: str = "",
    ) -> list[dict[str, Any]]:
        db = self._open_db()
        try:
            sql = """
                SELECT p.fold, p.uid, p.sig_id, p.family_id, p.tf, p.archetype, p.stage,
                       p.total_n, p.min_asset_n, p.median_asset_n, p.pos_asset_ratio,
                       p.pooled_mean_pct, p.mean_sharpe, p.pooled_t, p.gate1b_strict_pass,
                       p.pool_passed_assets, p.updated_at,
                       COALESCE(d.name, p.sig_id) AS factor_name,
                       COALESCE(d.direction, 0) AS direction,
                       COALESCE(d.exit_mode, 'hybrid') AS exit_mode
                FROM alpha_lab.factor_pool_scores p
                LEFT JOIN alpha_lab.factor_defs d ON p.sig_id = d.sig_id
                ORDER BY p.mean_sharpe DESC, p.pooled_t DESC
            """
            df = db.query_df(sql)
            if df.empty:
                return []
            df["cluster"] = [
                str(a).strip()[0] if str(a).strip() and str(a).strip()[0] in ALL_CLUSTERS else "A"
                for a in df["archetype"]
            ]
            if fold:
                df = df[df["fold"] == fold]
            if cluster:
                df = df[df["cluster"] == cluster.upper()]
            if tf:
                df = df[df["tf"] == tf]
            if stage:
                df = df[df["stage"] == stage]
            if gate1b_only:
                df = df[df["gate1b_strict_pass"].astype(bool)]
            if q:
                kw = q.strip().lower()
                mask = (
                    df["uid"].astype(str).str.lower().str.contains(kw, na=False)
                    | df["family_id"].astype(str).str.lower().str.contains(kw, na=False)
                    | df["factor_name"].astype(str).str.lower().str.contains(kw, na=False)
                )
                df = df[mask]
            return _df_to_records(df)
        finally:
            db.close()

    def get_asset_scores(
        self,
        *,
        fold: str = "",
        sector: str = "",
        asset: str = "",
        tf: str = "",
        stage: str = "",
        family_id: str = "",
        uid: str = "",
        q: str = "",
        canonical_only: bool = True,
        local_pass_only: bool = False,
        sort_by: str = "is_net_sharpe",
        sort_dir: str = "desc",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        db = self._open_db()
        try:
            where_clauses: list[str] = []
            params: list[Any] = []
            if canonical_only:
                where_clauses.append("is_canonical = TRUE")
            if local_pass_only:
                where_clauses.append("strict_local_pass = TRUE")
            if fold:
                where_clauses.append("fold = %s")
                params.append(fold)
            if sector:
                where_clauses.append("sector = %s")
                params.append(sector)
            if asset:
                where_clauses.append("asset = %s")
                params.append(asset)
            if tf:
                where_clauses.append("tf = %s")
                params.append(tf)
            if stage:
                where_clauses.append("stage = %s")
                params.append(stage)
            if family_id:
                where_clauses.append("family_id = %s")
                params.append(family_id)
            if uid:
                where_clauses.append("uid = %s")
                params.append(uid)
            if q:
                where_clauses.append("(LOWER(uid) LIKE %s OR LOWER(asset) LIKE %s OR LOWER(family_id) LIKE %s)")
                kw_like = f"%{q.strip().lower()}%"
                params.extend([kw_like, kw_like, kw_like])
            return self._run_asset_scores_query(db, where_clauses, params, sort_by, sort_dir, limit, offset)
        finally:
            db.close()

    def _run_asset_scores_query(
        self,
        db: AlphaLabLedgerDB,
        where_clauses: list[str],
        params: list[Any],
        sort_by: str,
        sort_dir: str,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        allowed_sort = {
            "is_net_sharpe",
            "t_stat",
            "win_rate",
            "pl_ratio",
            "safety_margin",
            "mean_net_pct",
            "q_mean_net_pct",
            "n",
            "ann_turnover",
        }
        col = sort_by if sort_by in allowed_sort else "is_net_sharpe"
        direction = "ASC" if str(sort_dir).lower() == "asc" else "DESC"
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        safe_limit = max(1, min(int(limit), 500))
        safe_offset = max(0, int(offset))

        summary_sql = f"""
            SELECT COUNT(*) AS total_rows,
                   COALESCE(AVG(is_net_sharpe), 0.0) AS avg_sharpe,
                   COALESCE(AVG(win_rate), 0.0) AS avg_win_rate,
                   COALESCE(AVG(pl_ratio), 0.0) AS avg_pl_ratio,
                   COALESCE(AVG(CASE WHEN is_net_sharpe > 0 THEN 1.0 ELSE 0.0 END), 0.0) AS pos_sharpe_ratio,
                   COALESCE(SUM(CASE WHEN strict_local_pass THEN 1 ELSE 0 END), 0) AS local_pass_count
            FROM alpha_lab.factor_asset_scores
            {where_sql}
        """
        sum_df = db.query_df(summary_sql, params)
        summary = _df_to_records(sum_df)[0] if not sum_df.empty else {"total_rows": 0}

        data_sql = f"""
            SELECT fold, asset, sector, uid, sig_id, family_id, tf, archetype, stage,
                   variant, is_canonical, n, effective_n, mean_net_pct, q_mean_net_pct,
                   win_rate, pl_ratio, safety_margin, t_stat, is_net_sharpe,
                   ann_turnover, strict_local_pass
            FROM alpha_lab.factor_asset_scores
            {where_sql}
            ORDER BY {col} {direction}, asset ASC, uid ASC
            LIMIT {safe_limit} OFFSET {safe_offset}
        """
        rows_df = db.query_df(data_sql, params)
        return {
            "summary": summary,
            "limit": safe_limit,
            "offset": safe_offset,
            "rows": _df_to_records(rows_df),
        }

    def get_signals_and_lineage(self) -> dict[str, Any]:
        db = self._open_db()
        try:
            sig_df = db.query_df(
                """
                SELECT signal_name, category, params_template, state_key,
                       is_kline_signal, exact_mirror, active_rate,
                       consumed_count, consumed_by_families, updated_at
                FROM alpha_lab.rust_signals
                ORDER BY consumed_count DESC, exact_mirror DESC, active_rate DESC, signal_name ASC
                """
            )
            audit_df = db.query_df(
                """
                SELECT family_id, code_sha256, passed, setup_diff, conf_diff, fire_diff,
                       q_max_err, zero_lookahead_pass, long_triggers, short_triggers,
                       trigger_rate, bounds_valid, elapsed_ms, audited_at
                FROM alpha_lab.preflight_audits
                ORDER BY family_id ASC
                """
            )
            from examples.multi_factor_cta.atomic_factors import ATOMIC_FACTOR_DEFS

            lineage_map = scan_all_atomic_factors()
            audit_map = {str(r["family_id"]): r for r in _df_to_records(audit_df)}
            families_lineage: list[dict[str, Any]] = []
            for fam, info in sorted(lineage_map.items()):
                aud = audit_map.get(fam, {})
                fdef = ATOMIC_FACTOR_DEFS.get(fam)
                families_lineage.append(
                    {
                        "family_id": fam,
                        "name": fdef.name if fdef is not None else fam,
                        "archetype": info.archetype,
                        "stage": info.stage,
                        "eligible_tfs": list(fdef.eligible_tfs) if fdef is not None else ["1d", "60m", "30m"],
                        "code_sha256": info.code_sha256,
                        "states": list(info.states),
                        "rust_signals": list(info.rust_signals),
                        "strengths": list(info.strengths),
                        "oi_features": list(info.oi_features),
                        "cs_features": list(info.cs_features),
                        "htf_features": list(info.htf_features),
                        "preflight": aud,
                    }
                )
            return {
                "signals": _df_to_records(sig_df),
                "families_lineage": families_lineage,
            }
        finally:
            db.close()

    def get_graveyard(self, *, failed_gate: str = "", stage: str = "", q: str = "") -> dict[str, Any]:
        db = self._open_db()
        try:
            df = db.query_df(
                """
                SELECT uid, family_id, sig_id, tf, archetype, stage, failed_gate,
                       reason, formula_desc, rejected_count, mean_is_sharpe,
                       mean_ann_turnover, buried_at
                FROM alpha_lab.factor_graveyard
                ORDER BY buried_at DESC, uid ASC
                """
            )
            gate_counts: dict[str, int] = {}
            if not df.empty:
                for g, grp in df.groupby("failed_gate"):
                    gate_counts[str(g)] = int(len(grp))
                if failed_gate:
                    df = df[df["failed_gate"] == failed_gate]
                if stage:
                    df = df[df["stage"] == stage]
                if q:
                    kw = q.strip().lower()
                    mask = (
                        df["uid"].astype(str).str.lower().str.contains(kw, na=False)
                        | df["family_id"].astype(str).str.lower().str.contains(kw, na=False)
                        | df["reason"].astype(str).str.lower().str.contains(kw, na=False)
                        | df["formula_desc"].astype(str).str.lower().str.contains(kw, na=False)
                    )
                    df = df[mask]
            return {
                "gate_counts": gate_counts,
                "rows": _df_to_records(df),
            }
        finally:
            db.close()

    def execute_readonly_sql(self, sql: str, max_rows: int = 500) -> dict[str, Any]:
        cleaned = validate_readonly_sql(sql)
        safe_max = max(1, min(int(max_rows), 2000))
        t0 = time.perf_counter()
        db = self._open_db()
        try:
            df = db.query_df(cleaned)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            total_returned = int(len(df))
            truncated = total_returned > safe_max
            if truncated:
                df = df.iloc[:safe_max]
            return {
                "status": "OK",
                "sql": cleaned,
                "elapsed_ms": elapsed_ms,
                "total_rows": total_returned,
                "truncated": truncated,
                "columns": [str(c) for c in df.columns],
                "rows": _df_to_records(df),
            }
        finally:
            db.close()

    def build_full_snapshot(self, asset_sample_limit: int = 1200) -> dict[str, Any]:
        """构建离线自包含 HTML 快照数据包（含全量汇总表与精选品种级明细样本）。"""
        overview = self.get_overview()
        ablation = self.get_ablation_runs()
        pool_scores = self.get_pool_scores()
        sig_lineage = self.get_signals_and_lineage()
        graveyard = self.get_graveyard()
        asset_top = self.get_asset_scores(
            canonical_only=True,
            sort_by="is_net_sharpe",
            sort_dir="desc",
            limit=asset_sample_limit,
            offset=0,
        )
        return {
            "overview": overview,
            "ablation": ablation,
            "pool_scores": pool_scores,
            "signals_lineage": sig_lineage,
            "graveyard": graveyard,
            "asset_scores_initial": asset_top,
        }


def export_dashboard_html(
    db_url: str | None = None,
    output_path: str | Path | None = None,
    *,
    asset_sample_limit: int = 1200,
) -> Path:
    """将 PostgreSQL / SQLite 实验台账导出为单文件自包含交互式 HTML 看板。"""
    from alpha_lab.dashboard_ui import render_dashboard_html

    out = Path(output_path) if output_path is not None else DEFAULT_DASHBOARD_HTML_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    svc = AlphaLabDashboardService(db_url)
    snapshot = svc.build_full_snapshot(asset_sample_limit=asset_sample_limit)
    html_text = render_dashboard_html(snapshot, live_mode=False)
    out.write_text(html_text, encoding="utf-8")
    return out


def _first_param(qs: dict[str, list[str]], key: str, default: str = "") -> str:
    vals = qs.get(key)
    return vals[0] if vals else default


def _bool_param(qs: dict[str, list[str]], key: str, default: bool = False) -> bool:
    raw = _first_param(qs, key, "")
    if not raw:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _make_request_handler(service: AlphaLabDashboardService) -> type[BaseHTTPRequestHandler]:
    from alpha_lab.dashboard_ui import render_dashboard_html

    class DashboardRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html_text: str, download_filename: str | None = None) -> None:
            body = html_text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if download_filename:
                self.send_header("Content-Disposition", f'attachment; filename="{download_filename}"')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)
            try:
                if path in ("/", "/index.html"):
                    snap = service.build_full_snapshot(asset_sample_limit=200)
                    self._send_html(render_dashboard_html(snap, live_mode=True))
                elif path == "/api/overview":
                    self._send_json(service.get_overview())
                elif path == "/api/ablation":
                    self._send_json({"rows": service.get_ablation_runs()})
                elif path == "/api/pool-scores":
                    rows = service.get_pool_scores(
                        fold=_first_param(qs, "fold"),
                        cluster=_first_param(qs, "cluster"),
                        tf=_first_param(qs, "tf"),
                        stage=_first_param(qs, "stage"),
                        gate1b_only=_bool_param(qs, "gate1b_only", False),
                        q=_first_param(qs, "q"),
                    )
                    self._send_json({"rows": rows})
                elif path == "/api/asset-scores":
                    res = service.get_asset_scores(
                        fold=_first_param(qs, "fold"),
                        sector=_first_param(qs, "sector"),
                        asset=_first_param(qs, "asset"),
                        tf=_first_param(qs, "tf"),
                        stage=_first_param(qs, "stage"),
                        family_id=_first_param(qs, "family_id"),
                        uid=_first_param(qs, "uid"),
                        q=_first_param(qs, "q"),
                        canonical_only=_bool_param(qs, "canonical_only", True),
                        local_pass_only=_bool_param(qs, "local_pass_only", False),
                        sort_by=_first_param(qs, "sort_by", "is_net_sharpe"),
                        sort_dir=_first_param(qs, "sort_dir", "desc"),
                        limit=int(_first_param(qs, "limit", "100") or 100),
                        offset=int(_first_param(qs, "offset", "0") or 0),
                    )
                    self._send_json(res)
                elif path == "/api/signals":
                    self._send_json(service.get_signals_and_lineage())
                elif path == "/api/graveyard":
                    res = service.get_graveyard(
                        failed_gate=_first_param(qs, "failed_gate"),
                        stage=_first_param(qs, "stage"),
                        q=_first_param(qs, "q"),
                    )
                    self._send_json(res)
                elif path == "/api/export-html":
                    snap = service.build_full_snapshot(asset_sample_limit=1200)
                    self._send_html(
                        render_dashboard_html(snap, live_mode=False),
                        download_filename="CZSC_AlphaLab_Dashboard.html",
                    )
                else:
                    self._send_json({"error": f"Not Found: {path}"}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/sql":
                self._send_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0") or 0)
                raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
                body = json.loads(raw_body)
                res = service.execute_readonly_sql(
                    str(body.get("sql", "")),
                    max_rows=int(body.get("max_rows", 500)),
                )
                self._send_json(res)
            except Exception as exc:
                self._send_json({"status": "ERROR", "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    return DashboardRequestHandler


def create_dashboard_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    db_url: str | None = None,
) -> ThreadingHTTPServer:
    """创建 AlphaLab HTTP 查询服务器实例（可用于后台线程或单元测试）。"""
    svc = AlphaLabDashboardService(db_url)
    handler_cls = _make_request_handler(svc)
    return ThreadingHTTPServer((host, port), handler_cls)


def serve_dashboard(
    host: str = "127.0.0.1",
    port: int = 8765,
    db_url: str | None = None,
    open_browser: bool = False,
) -> None:
    """启动阻塞式本地 Web 查询服务，监听直到 Ctrl+C 退出。"""
    import webbrowser

    server = create_dashboard_server(host=host, port=port, db_url=db_url)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/"
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
