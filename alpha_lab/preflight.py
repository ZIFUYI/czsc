"""
1 秒快速预检门禁（Pre-Flight Gate）
=========================================================================
在候选原子因子进入 Full-60 全市场评估前，先在预缓存的基准切片（含跨品种截面池与 OI 序列）上
执行 3 项硬性审计（总耗时 < 1.5 秒）：
1. 价格仿射镜像 0 误差检验 (P' = M - P)：验证 setup_diff == 0, conf_diff == 0, fire_diff == 0；
2. 零前视截断一致性检验 (Zero-Lookahead)：验证 compute_family(ctx[:t]) == compute_family(ctx[:T])[:t] (atol=1e-12)；
3. 触发密度与数值边界检验：验证多空双侧均有触发、输出严格位于 [-1, 1] 且无 NaN/Inf。
"""

from __future__ import annotations

import pickle
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_lab.catalog import compute_factor_code_sha256
from alpha_lab.ledger_db import DEFAULT_CACHE_60_DIR, AlphaLabLedgerDB
from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.hypothesis_bank import (
    attach_native_czsc_signals,
    enrich_kline_features,
)
from examples.multi_factor_cta.atomic_factors import (
    ATOMIC_FACTOR_DEFS,
    STATE_SIGNAL_MANIFEST,
    AtomicFactorDef,
    FactorContext,
    compute_family,
    set_factor_context,
    two_stage_trigger,
)
from examples.multi_factor_cta.data_engine import compute_causal_cross_sectional_features

PRICE_COLS = ("open", "high", "low", "close")
PREFLIGHT_TRUNC_T = 2300
_BENCHMARK_MEM_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class PreflightAuditResult:
    """单个原子因子族的 1 秒快速预检门禁审计结果。"""

    family_id: str
    code_sha256: str
    passed: bool
    setup_diff: int
    conf_diff: int
    fire_diff: int
    q_max_err: float
    zero_lookahead_pass: bool
    trunc_max_err: float
    long_triggers: int
    short_triggers: int
    trigger_rate: float
    bounds_valid: bool
    elapsed_ms: float
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["failure_reasons"] = list(self.failure_reasons)
        return d


def _mock_quantized_raw(seed: int = 5) -> pd.DataFrame:
    df = generate_symbol_kines("000001", "30分钟", "20220101", "20221231", seed=seed)
    df = df.rename(columns={"vol": "volume"})
    df["dt"] = pd.to_datetime(df["dt"])
    df = df[["dt", "symbol", *PRICE_COLS, "volume"]].reset_index(drop=True)
    for col in PRICE_COLS:
        df[col] = np.round(df[col].astype(float) * 100.0)
    return df


def _affine_mirror_df(raw: pd.DataFrame) -> pd.DataFrame:
    m = float(raw["high"].max() + raw["low"].min())
    out = raw.copy()
    out["open"], out["close"] = m - raw["open"], m - raw["close"]
    out["high"], out["low"] = m - raw["low"], m - raw["high"]
    out.attrs["market_mirrored"] = True
    return out


def _make_htf_60m(raw: pd.DataFrame) -> pd.DataFrame:
    m = (len(raw) // 2) * 2
    sub = raw.iloc[:m]
    grp = np.arange(m) // 2
    htf = (
        sub.groupby(grp)
        .agg(
            dt=("dt", "last"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index(drop=True)
    )
    htf["symbol"] = "000001"
    return htf


def _prepare_kdf(raw: pd.DataFrame, tf: str, sig_cache_dir: Path) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = df["dt"].dt.date
    df = enrich_kline_features(df)
    return attach_native_czsc_signals(df, tf, sig_cache_dir)


def _build_single_ctx(raw: pd.DataFrame, sig_cache_dir: Path, oi_series: np.ndarray) -> FactorContext:
    base = _prepare_kdf(raw, "30m", sig_cache_dir)
    htf = _prepare_kdf(_make_htf_60m(raw), "60m", sig_cache_dir)
    pool = {"000001": base}
    for seed in (6, 7, 8):
        peer = _mock_quantized_raw(seed).iloc[: len(raw)].copy()
        if raw.attrs.get("market_mirrored", False):
            peer = _affine_mirror_df(peer)
        pool[f"PEER{seed}"] = peer
    compute_causal_cross_sectional_features(
        pool, specs=dict.fromkeys(pool, "test_sector"), tf="30m", attach_to_dfs=True
    )
    oi_df = pd.DataFrame({"dt": base["dt"], "open_interest": oi_series[: len(base)]})
    base = set_factor_context(base, oi_5m=oi_df)
    return FactorContext.from_df(base, htf)


def get_preflight_benchmark_contexts(
    cache_dir: Path | None = None,
) -> tuple[FactorContext, FactorContext, FactorContext]:
    """加载或构建预检门禁专用的 (ctx_full, ctx_mirror, ctx_trunc) 基准上下文三元组。"""
    manifest_keys = tuple(sorted(STATE_SIGNAL_MANIFEST.keys()))
    if "bundle" in _BENCHMARK_MEM_CACHE and _BENCHMARK_MEM_CACHE.get("manifest_keys") == manifest_keys:
        return _BENCHMARK_MEM_CACHE["bundle"]

    c_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_60_DIR
    c_dir.mkdir(parents=True, exist_ok=True)
    bundle_pkl = c_dir / "preflight_benchmark_ctx_v1.pkl"
    if bundle_pkl.exists():
        try:
            with open(bundle_pkl, "rb") as f:
                saved = pickle.load(f)
            if saved.get("manifest_keys") == manifest_keys:
                bundle = (saved["ctx_full"], saved["ctx_mirror"], saved["ctx_trunc"])
                _BENCHMARK_MEM_CACHE["manifest_keys"] = manifest_keys
                _BENCHMARK_MEM_CACHE["bundle"] = bundle
                return bundle
        except Exception:
            pass

    sig_cache = c_dir / "preflight_sig_cache"
    sig_cache.mkdir(parents=True, exist_ok=True)
    raw_full = _mock_quantized_raw(5)
    raw_mirror = _affine_mirror_df(raw_full)
    raw_trunc = raw_full.iloc[:PREFLIGHT_TRUNC_T].reset_index(drop=True)
    rng = np.random.default_rng(20261022)
    oi_series = np.maximum(100000.0 + np.cumsum(rng.normal(0.0, 250.0, size=len(raw_full))), 1000.0)

    ctx_full = _build_single_ctx(raw_full, sig_cache, oi_series)
    ctx_mirror = _build_single_ctx(raw_mirror, sig_cache / "mirror", oi_series)
    ctx_trunc = _build_single_ctx(raw_trunc, sig_cache / "trunc", oi_series[:PREFLIGHT_TRUNC_T])
    for ctx_obj in (ctx_full, ctx_mirror, ctx_trunc):
        ctx_obj.snapshots = [
            {
                "completed_bi_dir": int(s.completed_bi_dir),
                "completed_fx_dir": int(s.completed_fx_dir),
                "zs_zg": float(s.zs_zg),
                "zs_zd": float(s.zs_zd),
            }
            for s in ctx_obj.snapshots
            if hasattr(s, "completed_bi_dir")
        ]
    bundle = (ctx_full, ctx_mirror, ctx_trunc)
    try:
        with open(bundle_pkl, "wb") as f:
            pickle.dump(
                {
                    "manifest_keys": manifest_keys,
                    "ctx_full": ctx_full,
                    "ctx_mirror": ctx_mirror,
                    "ctx_trunc": ctx_trunc,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except Exception:
        pass
    _BENCHMARK_MEM_CACHE["manifest_keys"] = manifest_keys
    _BENCHMARK_MEM_CACHE["bundle"] = bundle
    return bundle


def audit_single_family(
    family_or_fdef: str | AtomicFactorDef,
    ctx_full: FactorContext,
    ctx_mirror: FactorContext,
    ctx_trunc: FactorContext,
) -> PreflightAuditResult:
    """对单个原子因子族在 (ctx_full, ctx_mirror, ctx_trunc) 上执行 3 项硬性审计。"""
    t0 = time.perf_counter()
    fdef = ATOMIC_FACTOR_DEFS[family_or_fdef] if isinstance(family_or_fdef, str) else family_or_fdef
    fam = fdef.family
    code_sha = compute_factor_code_sha256(fdef.build)
    reasons: list[str] = []

    setup_diff = 0
    conf_diff = 0
    fire_diff = 0
    q_max_err = 0.0
    for side in (1, -1):
        sa, sqa, ca, cqa, wa = fdef.build(ctx_full, side)
        sb, sqb, cb, cqb, wb = fdef.build(ctx_mirror, -side)
        sa_b, sb_b = np.asarray(sa, dtype=bool), np.asarray(sb, dtype=bool)
        ca_b, cb_b = np.asarray(ca, dtype=bool), np.asarray(cb, dtype=bool)
        setup_diff += int(np.sum(sa_b != sb_b))
        conf_diff += int(np.sum(ca_b != cb_b))
        if wa != wb:
            reasons.append(f"window_mismatch(side={side}: {wa} vs {wb})")
        fa, qa = two_stage_trigger(sa_b, np.asarray(sqa, dtype=np.float64), ca_b, np.asarray(cqa, dtype=np.float64), wa)
        fb, qb = two_stage_trigger(sb_b, np.asarray(sqb, dtype=np.float64), cb_b, np.asarray(cqb, dtype=np.float64), wb)
        fire_diff += int(np.sum(fa != fb))
        err_q = float(np.max(np.abs(qa - qb))) if len(qa) else 0.0
        q_max_err = max(q_max_err, err_q)

    if setup_diff > 0 or conf_diff > 0 or fire_diff > 0 or q_max_err > 0.011:
        reasons.append(
            f"affine_mirror_violation(setup_diff={setup_diff}, conf_diff={conf_diff}, fire_diff={fire_diff}, q_max_err={q_max_err:.6f})"
        )

    t = ctx_trunc.n
    if fam in ATOMIC_FACTOR_DEFS:
        sig_full, _ = compute_family(ctx_full, fam)
        sig_trunc, _ = compute_family(ctx_trunc, fam)
    else:
        register_temp = fam not in ATOMIC_FACTOR_DEFS
        if register_temp:
            ATOMIC_FACTOR_DEFS[fam] = fdef
        try:
            sig_full, _ = compute_family(ctx_full, fam)
            sig_trunc, _ = compute_family(ctx_trunc, fam)
        finally:
            if register_temp:
                ATOMIC_FACTOR_DEFS.pop(fam, None)

    trunc_err = float(np.max(np.abs(sig_full[:t] - sig_trunc))) if t > 0 else 0.0
    setup_trunc_diff = 0
    for side in (1, -1):
        sf_s, _, cf_s, _, _ = fdef.build(ctx_full, side)
        st_s, _, ct_s, _, _ = fdef.build(ctx_trunc, side)
        setup_trunc_diff += int(np.sum(np.asarray(sf_s, dtype=bool)[:t] != np.asarray(st_s, dtype=bool)))
        setup_trunc_diff += int(np.sum(np.asarray(cf_s, dtype=bool)[:t] != np.asarray(ct_s, dtype=bool)))
    zero_lookahead_pass = bool(trunc_err <= 1e-12 and setup_trunc_diff == 0)
    if not zero_lookahead_pass:
        reasons.append(f"lookahead_violation(trunc_max_err={trunc_err:.3e}, stage_trunc_diff={setup_trunc_diff})")

    finite_ok = bool(np.isfinite(sig_full).all())
    max_abs = float(np.max(np.abs(sig_full))) if len(sig_full) else 0.0
    long_triggers = int((sig_full > 0.0).sum())
    short_triggers = int((sig_full < 0.0).sum())
    trigger_rate = float((sig_full != 0.0).mean()) if len(sig_full) else 0.0
    bounds_valid = bool(
        finite_ok
        and max_abs <= 1.0 + 1e-12
        and long_triggers > 0
        and short_triggers > 0
        and 0.001 <= trigger_rate <= 0.98
    )
    if not bounds_valid:
        reasons.append(
            f"bounds_or_density_violation(finite={finite_ok}, max_abs={max_abs:.4f}, long={long_triggers}, short={short_triggers}, rate={trigger_rate:.4f})"
        )

    passed = len(reasons) == 0
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return PreflightAuditResult(
        family_id=fam,
        code_sha256=code_sha,
        passed=passed,
        setup_diff=setup_diff,
        conf_diff=conf_diff,
        fire_diff=fire_diff,
        q_max_err=round(q_max_err, 6),
        zero_lookahead_pass=zero_lookahead_pass,
        trunc_max_err=trunc_err,
        long_triggers=long_triggers,
        short_triggers=short_triggers,
        trigger_rate=round(trigger_rate, 4),
        bounds_valid=bounds_valid,
        elapsed_ms=elapsed_ms,
        failure_reasons=tuple(reasons),
    )


def run_preflight_audit(
    families: Sequence[str | AtomicFactorDef] | None = None,
    *,
    db: AlphaLabLedgerDB | None = None,
    persist: bool = True,
    auto_bury_failed: bool = False,
) -> pd.DataFrame:
    """对指定原子因子族列表执行 1 秒快速预检门禁，并可将审计结果自动落库。"""
    target_fams = list(families) if families is not None else list(ATOMIC_FACTOR_DEFS.keys())
    ctx_full, ctx_mirror, ctx_trunc = get_preflight_benchmark_contexts()
    results = [audit_single_family(f, ctx_full, ctx_mirror, ctx_trunc) for f in target_fams]
    dicts = [r.to_dict() for r in results]
    if persist and dicts:
        owns_db = db is None
        active_db = db
        if active_db is None:
            try:
                active_db = AlphaLabLedgerDB()
            except Exception:
                active_db = None
                owns_db = False
        if active_db is not None:
            try:
                active_db.upsert_preflight_audits(dicts)
                if auto_bury_failed:
                    for r in results:
                        if not r.passed:
                            active_db.bury_factor(
                                f"{r.family_id}L@60m",
                                failed_gate="Preflight_Gate",
                                reason="; ".join(r.failure_reasons),
                                family_id=r.family_id,
                            )
            finally:
                if owns_db:
                    active_db.close()
    return pd.DataFrame(dicts)
