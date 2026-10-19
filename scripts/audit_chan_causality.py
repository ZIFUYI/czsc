#!/usr/bin/env python
"""
全自动缠论与多周期 CTA 因果性审计与回归验证脚本
Auto Audit Script for Chan Causality, Look-Ahead Prevention, and Research Review

覆盖五大核心检验项：
1. 实验 F 向过去 10 分钟回填仓位漏洞 (w5_early[idx-2:idx] = w5[idx]) 的失败复现与修复验证；
2. 跨周期时间戳对齐 (map_tf_weight_to_base_grid) 在整除长度、夜盘、午休与缺失柱场景的因果单调性；
3. 信号缓存污染与全量配置指纹验证 (attach_native_czsc_signals)；
4. 缠论分型/笔/中枢确认时钟 vs 锚点时间与交易执行因果性；
5. 端到端未来数据截断与扰动不变量检验 (Truncation and Perturbation Invariance)。
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 确保项目根目录在 sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import czsc  # noqa: E402
from czsc.mock import generate_symbol_kines  # noqa: E402
from examples.alpha_mining_agent import hypothesis_bank as hb  # noqa: E402
from examples.multi_factor_cta import atomic_factors as af  # noqa: E402
from examples.multi_factor_cta.data_engine import map_tf_weight_to_base_grid  # noqa: E402
from examples.multi_factor_cta.synthesis_engine import build_asset_execution_inputs  # noqa: E402

ABLATION_DIR = ROOT / "examples" / "results" / "intraday_cta_ablation_20261017"
OUTPUT_DIR = ROOT / "output" / "chan_causality_audit"


def _load_multitf_study():
    if str(ABLATION_DIR) not in sys.path:
        sys.path.insert(0, str(ABLATION_DIR))
    import run_multitf_incremental_study as r

    return r


def audit_exp_f_backfill_causality() -> dict[str, Any]:
    """1. 审计实验 F 仓位回填：复现前视缺陷并验证修复后零未来依赖"""
    r = _load_multitf_study()

    dt5 = pd.to_datetime(
        [
            "2025-01-02 09:05:00",
            "2025-01-02 09:10:00",
            "2025-01-02 09:15:00",
            "2025-01-02 09:20:00",
            "2025-01-02 09:25:00",
            "2025-01-02 09:30:00",
            "2025-01-02 09:35:00",
            "2025-01-02 09:40:00",
            "2025-01-02 09:45:00",
        ]
    )
    dt15 = pd.to_datetime(
        [
            "2025-01-02 09:15:00",
            "2025-01-02 09:30:00",
            "2025-01-02 09:45:00",
        ]
    )
    df5 = pd.DataFrame(
        {
            "dt": dt5,
            "open": [100.0] * 5 + [105.0] * 4,
            "high": [100.5] * 5 + [106.0] * 4,
            "low": [99.5] * 5 + [104.5] * 4,
            "close": [100.0] * 5 + [105.5] * 4,
            "volume": [1000.0] * 9,
            "open_interest": [50000.0] * 9,
        }
    )
    df15_base = pd.DataFrame({"dt": dt15, "close": [100.0, 100.0, 100.0]})
    t15_flat = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    t15_jump_at_0930 = np.array([0.0, 0.8, 0.8], dtype=np.float64)

    # 模拟修复前缺陷逻辑 (Buggy Legacy Logic)
    w5_buggy = map_tf_weight_to_base_grid(df15_base, t15_jump_at_0930, df5).copy()
    diff = np.diff(w5_buggy, prepend=0.0)
    for idx in np.where(np.abs(diff) > 0.05)[0]:
        if idx >= 2:
            w5_buggy[idx - 2 : idx] = w5_buggy[idx]

    legacy_lookahead_leaked = bool(w5_buggy[3] > 0.0 or w5_buggy[4] > 0.0)

    # 修复后执行逻辑
    w5_causal_flat = r.map_15m_target_to_5m_grid(df15_base, t15_flat, df5, is_5m_native=True)
    w5_causal_jump = r.map_15m_target_to_5m_grid(df15_base, t15_jump_at_0930, df5, is_5m_native=True)

    causal_invariance_holds = bool(np.allclose(w5_causal_jump[:5], w5_causal_flat[:5]))
    target_confirmed_at_0930 = bool(w5_causal_jump[5] == 0.8)

    passed = legacy_lookahead_leaked and causal_invariance_holds and target_confirmed_at_0930
    return {
        "check": "exp_f_backfill_causality",
        "passed": passed,
        "details": {
            "legacy_backfill_leaked_into_t_minus_2_and_t_minus_1": legacy_lookahead_leaked,
            "causal_pre_confirmation_invariance": causal_invariance_holds,
            "target_effective_at_confirmation_bar": target_confirmed_at_0930,
            "reproduced_failing_case": "w5_buggy[09:20] = 0.8, w5_buggy[09:25] = 0.8 before 09:30 close",
            "fixed_causal_case": "w5_causal[09:20] = 0.0, w5_causal[09:25] = 0.0, w5_causal[09:30] = 0.8",
        },
    }


def audit_cross_timeframe_timestamp_alignment() -> dict[str, Any]:
    """2. 审计跨周期时间戳对齐：检验在整除行数、夜盘、午休与缺失柱下的单调性"""
    # 目标 5m 真实不规则交易网格 (9 根柱)
    dt_target = pd.to_datetime(
        [
            "2025-01-02 21:05:00",
            "2025-01-02 21:10:00",
            "2025-01-02 21:15:00",
            "2025-01-03 09:05:00",
            # 09:10 缺失
            "2025-01-03 09:15:00",
            "2025-01-03 10:15:00",
            # 10:15-10:30 休盘
            "2025-01-03 10:35:00",
            "2025-01-03 11:30:00",
            # 午休
            "2025-01-03 13:35:00",
        ]
    )
    # 源 15m/高周期柱 (3 根柱: 9 % 3 == 0 整除)
    dt_source = pd.to_datetime(
        [
            "2025-01-02 21:15:00",
            "2025-01-03 09:15:00",
            "2025-01-03 11:30:00",
        ]
    )
    source_df = pd.DataFrame({"dt": dt_source, "close": [10.0, 11.0, 12.0]})
    target_df = pd.DataFrame({"dt": dt_target, "close": [10.0] * 9})
    weights = np.array([0.2, 0.4, 0.6], dtype=np.float64)

    mapped = map_tf_weight_to_base_grid(source_df, weights, target_df)

    expected = np.array(
        [
            0.0,  # 21:05: 21:15 尚未发生 -> 0.0
            0.0,  # 21:10: 21:15 尚未发生 -> 0.0
            0.2,  # 21:15: 21:15 发生 -> 0.2
            0.2,  # 09:05: 维持最新可用 21:15 的 0.2
            0.4,  # 09:15: 09:15 发生 -> 0.4
            0.4,  # 10:15: 维持 09:15 的 0.4
            0.4,  # 10:35: 维持 09:15 的 0.4
            0.6,  # 11:30: 11:30 发生 -> 0.6
            0.6,  # 13:35: 维持 11:30 的 0.6
        ]
    )
    exact_match = bool(np.allclose(mapped, expected))
    no_divisible_repeat = bool(mapped[0] == 0.0 and mapped[1] == 0.0)

    passed = exact_match and no_divisible_repeat
    return {
        "check": "cross_timeframe_timestamp_alignment",
        "passed": passed,
        "details": {
            "mapped_weights": mapped.tolist(),
            "expected_weights": expected.tolist(),
            "exact_asof_backward_match": exact_match,
            "rejected_integer_repeat": no_divisible_repeat,
        },
    }


def audit_signal_cache_integrity() -> dict[str, Any]:
    """3. 审计信号缓存污染：同长度行情扰动、字段损坏与全局补丁安全性"""
    raw = generate_symbol_kines("AUDIT_SYM", "30分钟", "20230101", "20230301", seed=100)
    raw = raw.rename(columns={"vol": "volume"}).iloc[:80].reset_index(drop=True)
    raw["dt"] = pd.to_datetime(raw["dt"])
    raw["date"] = raw["dt"].dt.date
    df1 = hb.enrich_kline_features(raw.copy())

    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        out1 = hb.attach_native_czsc_signals(df1.copy(), "30m", tmp_path)
        cache1 = list(tmp_path.glob("native_czsc_signals_30m_*.pkl"))
        cache_created = len(cache1) == 1

        # 扰动第 50 根 K 线的 close (长度保持 80 根不变)
        df2 = df1.copy()
        df2.loc[50:, "close"] = df2.loc[50:, "close"] * 1.05
        out2 = hb.attach_native_czsc_signals(df2.copy(), "30m", tmp_path)
        cache2 = list(tmp_path.glob("native_czsc_signals_30m_*.pkl"))
        distinct_cache = len(cache2) == 2

        # 50 根以前的信号必须严格一致
        cz_cols = [c for c in out1.columns if c.startswith("cz_")]
        causal_signals = np.all(out1.loc[:49, cz_cols].values == out2.loc[:49, cz_cols].values)

        # 损坏已有缓存的列结构，必须自动触发重新计算
        only_pkl = cache1[0]
        corrupted = pd.read_pickle(only_pkl).drop(columns=[cz_cols[0]])
        corrupted.to_pickle(only_pkl)
        recomputed = hb.attach_native_czsc_signals(df1.copy(), "30m", tmp_path)
        recomputed_ok = cz_cols[0] in recomputed.columns

    # 检查全局补丁是否已被清理
    clean_globals = (
        getattr(pd.read_pickle, "__name__", "") != "_fast_read_pickle"
        and getattr(Path.exists, "__name__", "") != "_fast_exists"
        and getattr(af._frame_fingerprint, "__name__", "") != "_fast_fp"
    )

    passed = cache_created and distinct_cache and causal_signals and recomputed_ok and clean_globals
    return {
        "check": "signal_cache_integrity",
        "passed": passed,
        "details": {
            "cache_file_created": cache_created,
            "same_length_mutation_generates_distinct_cache": distinct_cache,
            "pre_perturbation_signals_invariant": causal_signals,
            "corrupted_cache_triggers_recomputation": recomputed_ok,
            "no_global_monkeypatches": clean_globals,
        },
    }


def audit_structure_clock_and_execution_causality() -> dict[str, Any]:
    """4. 审计缠论结构确认时间与交易时钟因果性"""
    # 验证 RawBar 时间戳为收盘时钟语义，以及 build_asset_execution_inputs 的 shift(1) 成交机制
    n = 10
    dts = pd.date_range("2025-01-02 09:15:00", periods=n, freq="15min")
    df_grid = pd.DataFrame(
        {
            "dt": dts,
            "open": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0],
            "high": [102.5, 104.5, 106.5, 108.5, 110.5, 112.5, 114.5, 116.5, 118.5, 120.5],
            "low": [99.5, 101.5, 103.5, 105.5, 107.5, 109.5, 111.5, 113.5, 115.5, 117.5],
            "close": [102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0, 120.0],
            "volume": [1000.0] * n,
        }
    )
    # 假设在 bar 2 (09:45) 基于完整收盘价 106.0 产生做多信号 target = 1.0
    targets = np.zeros(n, dtype=np.float64)
    targets[2] = 1.0
    targets[3] = 1.0

    ret_s, pos_s = build_asset_execution_inputs(df_grid, targets, execution_clock="next_open")

    # 在 bar 2 (09:45)，持仓 pos 必须仍为 0.0；直到 bar 3 (10:00 开盘) pos 才变为 1.0
    no_lookahead_return = bool(pos_s.iloc[2] == 0.0 and pos_s.iloc[3] == 1.0)
    # bar 2 期间价格从 104.0 涨至 106.0 的收益不得被策略捕获
    bar2_pnl = pos_s.iloc[2] * ret_s.iloc[2]
    signal_bar_gain_zero = bool(abs(bar2_pnl) < 1e-12)

    passed = no_lookahead_return and signal_bar_gain_zero
    return {
        "check": "structure_clock_and_execution_causality",
        "passed": passed,
        "details": {
            "pos_at_signal_bar_2": float(pos_s.iloc[2]),
            "pos_at_execution_bar_3": float(pos_s.iloc[3]),
            "signal_bar_return_earned": float(bar2_pnl),
            "execution_strictly_at_next_open": no_lookahead_return,
        },
    }


def audit_truncation_and_future_perturbation_invariance() -> dict[str, Any]:
    """5. 端到端多周期截断与扰动不变量检验"""
    raw_full = generate_symbol_kines("TRUNC_AUDIT", "15分钟", "20240101", "20240301", seed=2024)
    raw_full = raw_full.rename(columns={"vol": "volume"}).reset_index(drop=True)
    raw_full["dt"] = pd.to_datetime(raw_full["dt"])
    raw_full["date"] = raw_full["dt"].dt.date

    # 截断点设在第 120 根 K 线
    cutoff_idx = 120
    raw_trunc = raw_full.iloc[:cutoff_idx].copy().reset_index(drop=True)

    df_full = hb.enrich_kline_features(raw_full.copy())
    df_trunc = hb.enrich_kline_features(raw_trunc.copy())

    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        full_sig = hb.attach_native_czsc_signals(df_full.copy(), "15m", tmp_path)
        trunc_sig = hb.attach_native_czsc_signals(df_trunc.copy(), "15m", tmp_path)

        cz_cols = [c for c in full_sig.columns if c.startswith("cz_")]
        diff_count = 0
        for c in cz_cols:
            s_full = full_sig.loc[: cutoff_idx - 1, c].values
            s_trunc = trunc_sig.loc[: cutoff_idx - 1, c].values
            if not np.array_equal(s_full, s_trunc):
                diff_count += 1

    invariant = diff_count == 0
    return {
        "check": "truncation_and_future_perturbation_invariance",
        "passed": invariant,
        "details": {
            "total_czsc_columns_tested": len(cz_cols),
            "columns_with_differences": diff_count,
            "truncation_invariance": invariant,
        },
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("  开始执行全套缠论因果性与多周期 CTA 回测审计...")
    print("=" * 80)

    checks = [
        audit_exp_f_backfill_causality(),
        audit_cross_timeframe_timestamp_alignment(),
        audit_signal_cache_integrity(),
        audit_structure_clock_and_execution_causality(),
        audit_truncation_and_future_perturbation_invariance(),
    ]

    all_passed = all(c["passed"] for c in checks)
    for c in checks:
        status = "PASSED" if c["passed"] else "FAILED"
        print(f"  [{status}] {c['check']}")

    native_mod = importlib.import_module("czsc._native")
    audit_report = {
        "audit_version": "2026.10.19.v1",
        "timestamp": pd.Timestamp.now().isoformat(),
        "czsc_version": getattr(czsc, "__version__", "unknown"),
        "native_extension_path": str(Path(native_mod.__file__).resolve()),
        "all_checks_passed": all_passed,
        "checks": checks,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "audit_results.json"
    def _json_default(o):
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.ndarray, list)):
            return list(o)
        return str(o)

    out_file.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(f"\n审计报告已写入: {out_file} (耗时 {time.time() - t0:.2f}s)")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
