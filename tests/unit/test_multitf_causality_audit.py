"""Causality, cache-isolation, and look-ahead regression suite for Multi-TF CTA backtest."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent import hypothesis_bank as hb
from examples.alpha_mining_agent.forensic_pipeline import map_tf_weight_to_5m_grid
from examples.multi_factor_cta import atomic_factors as af
from examples.multi_factor_cta.data_engine import map_tf_weight_to_base_grid
from examples.multi_factor_cta.synthesis_engine import (
    account_portfolio_execution,
    build_asset_execution_inputs,
)

ABLATION_DIR = Path(__file__).resolve().parents[2] / "examples" / "results" / "intraday_cta_ablation_20261017"


def _load_multitf_study():
    if str(ABLATION_DIR) not in sys.path:
        sys.path.insert(0, str(ABLATION_DIR))
    import run_multitf_incremental_study as r

    return r


def test_exp_f_5m_target_does_not_depend_on_future_15m_close():
    """ExpF 5m target weights at t-2 (09:05) and t-1 (09:10) must not depend on 15m close at t (09:15)."""
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
            "open": [100.0, 100.0, 100.0, 100.0, 100.0, 105.0, 105.0, 105.0, 105.0],
            "high": [100.5, 100.5, 100.5, 100.5, 100.5, 106.0, 105.5, 105.5, 105.5],
            "low": [99.5, 99.5, 99.5, 99.5, 99.5, 100.0, 104.5, 104.5, 104.5],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0, 105.5, 105.0, 105.0, 105.0],
            "volume": [1000.0] * 9,
            "open_interest": [50000.0] * 9,
        }
    )
    df15_base = pd.DataFrame({"dt": dt15, "close": [100.0, 100.0, 100.0]})
    t15_flat = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    t15_jump_at_0930 = np.array([0.0, 0.8, 0.8], dtype=np.float64)

    assert hasattr(r, "map_15m_target_to_5m_grid"), (
        "run_multitf_incremental_study must expose causal map_15m_target_to_5m_grid without backward backfill"
    )
    w5_flat = r.map_15m_target_to_5m_grid(df15_base, t15_flat, df5, is_5m_native=True)
    w5_jump = r.map_15m_target_to_5m_grid(df15_base, t15_jump_at_0930, df5, is_5m_native=True)

    # At 09:20 (idx 3) and 09:25 (idx 4), the 09:30 15m bar has NOT closed yet.
    # Therefore w5[:5] (up to 09:25 inclusive) MUST be identical regardless of what happens at 09:30!
    np.testing.assert_allclose(w5_jump[:5], w5_flat[:5], err_msg="Future 09:30 15m close leaked into 09:20/09:25!")
    assert w5_jump[5] == pytest.approx(0.8)


def test_ablation_scripts_do_not_monkeypatch_global_pickle_or_fingerprint():
    """Importing research scripts must not install unsafe global monkeypatches on pandas/pathlib/atomic_factors."""
    if str(ABLATION_DIR) not in sys.path:
        sys.path.insert(0, str(ABLATION_DIR))
    import run_intraday_ablation  # noqa: F401

    assert pd.read_pickle.__name__ != "_fast_read_pickle"
    assert Path.exists.__name__ != "_fast_exists"
    assert af._frame_fingerprint.__name__ != "_fast_fp"

    # Mutating a DataFrame in-place must immediately change _frame_fingerprint
    df = pd.DataFrame(
        {
            "dt": pd.date_range("2025-01-02 09:15:00", periods=4, freq="15min"),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [100.0, 100.0, 100.0, 100.0],
        }
    )
    fp1 = af._frame_fingerprint(df)
    df.loc[2, "close"] = 999.0
    fp2 = af._frame_fingerprint(df)
    assert fp1 != fp2, "_frame_fingerprint failed to detect in-place OHLCV mutation"


def test_native_signal_cache_validates_full_config_and_same_length_mutations(tmp_path):
    """attach_native_czsc_signals must invalidate on same-length OHLCV/timestamp edits and corrupt cache columns."""
    raw = generate_symbol_kines("CACHE_TEST", "30分钟", "20230101", "20230301", seed=42)
    raw = raw.rename(columns={"vol": "volume"}).iloc[:90].reset_index(drop=True)
    raw["dt"] = pd.to_datetime(raw["dt"])
    raw["date"] = raw["dt"].dt.date
    df1 = hb.enrich_kline_features(raw.copy())

    out1 = hb.attach_native_czsc_signals(df1.copy(), "30m", tmp_path)
    cache_files_1 = set(tmp_path.glob("native_czsc_signals_30m_*.pkl"))
    assert len(cache_files_1) == 1

    # 1. Same length, different OHLCV at a single bar -> must create a new cache entry and not reuse old one
    raw2 = raw.copy()
    raw2.loc[45, "close"] *= 1.25
    raw2.loc[45, "high"] = max(raw2.loc[45, "high"], raw2.loc[45, "close"])
    df2 = hb.enrich_kline_features(raw2)
    out2 = hb.attach_native_czsc_signals(df2.copy(), "30m", tmp_path)
    cache_files_2 = set(tmp_path.glob("native_czsc_signals_30m_*.pkl"))
    assert len(cache_files_2) == 2

    # Perturbation at bar 45 must not alter any native CZSC signal before bar 45
    cz_cols = [c for c in out1.columns if c.startswith("cz_")]
    pd.testing.assert_frame_equal(out1.loc[:44, cz_cols], out2.loc[:44, cz_cols], check_exact=True)

    # 2. Corrupting an existing cache file with same row count but missing expected column must recompute
    only_pkl = next(iter(cache_files_1))
    corrupted = pd.read_pickle(only_pkl).drop(columns=[cz_cols[0]])
    corrupted.to_pickle(only_pkl)
    recomputed = hb.attach_native_czsc_signals(df1.copy(), "30m", tmp_path)
    assert cz_cols[0] in recomputed.columns
    pd.testing.assert_frame_equal(out1[cz_cols], recomputed[cz_cols], check_exact=True)


def test_no_np_roll_future_wraparound_in_hetero_and_hypothesis_signals(monkeypatch):
    """Tail perturbations at the end of a series must never wrap around into index 0..15 even with warmup=0."""
    r = _load_multitf_study()
    orig_hold = r.hold_with_cooldown_and_trail

    def _zero_warmup_hold(*args, **kwargs):
        kwargs["warmup"] = 0
        return orig_hold(*args, **kwargs)

    monkeypatch.setattr(r, "hold_with_cooldown_and_trail", _zero_warmup_hold)

    n = 40
    dts = pd.date_range("2024-01-02 09:15:00", periods=n, freq="15min")
    close_a = np.full(n, 100.0, dtype=np.float64)
    close_a[12] = 102.0
    open_a = np.full(n, 100.0, dtype=np.float64)
    high_a = np.maximum(open_a, close_a) + 0.1
    low_a = np.minimum(open_a, close_a) - 0.1
    vol_a = np.full(n, 1000.0, dtype=np.float64)
    vol_a[12] = 5000.0
    atr_a = np.full(n, 0.5, dtype=np.float64)
    atr_a[12] = 0.1

    df_a = pd.DataFrame(
        {
            "dt": dts,
            "open": open_a,
            "high": high_a,
            "low": low_a,
            "close": close_a,
            "volume": vol_a,
            "atr14": atr_a,
        }
    )
    df_b = df_a.copy()
    # Perturb only the future tail (bars 20..39) to make ma120[-4] much larger than ma120[12]
    df_b.loc[20:, ["open", "high", "low", "close"]] = 500.0

    sigs_a = r.compute_hetero_15m_signals(df_a)
    sigs_b = r.compute_hetero_15m_signals(df_b)
    for uid in sigs_a:
        np.testing.assert_allclose(
            sigs_a[uid][1][:16],
            sigs_b[uid][1][:16],
            err_msg=f"{uid} leaked future tail bars (20..39) into index 0..15 via np.roll(ma120, 16)!",
        )


def test_cross_timeframe_alignment_irregular_sessions_and_divisible_lengths():
    """map_tf_weight_to_base_grid must align by backward timestamp across night/lunch/missing/divisible bars."""
    # Target 5m grid with night session, morning break, lunch break, and a missing bar (09:10 missing)
    dt_target = pd.to_datetime(
        [
            "2025-01-02 21:05:00",
            "2025-01-02 21:10:00",
            "2025-01-02 21:15:00",
            "2025-01-03 09:05:00",
            # 09:10 missing
            "2025-01-03 09:15:00",
            "2025-01-03 10:15:00",
            # 10:15-10:30 break
            "2025-01-03 10:35:00",
            "2025-01-03 11:30:00",
            # lunch break
            "2025-01-03 13:35:00",
        ]
    )
    # 3 source bars -> len(target) == 9 is divisible by len(source) == 3!
    # If integer divisibility np.repeat were used, weights would repeat [w0,w0,w0, w1,w1,w1, w2,w2,w2].
    dt_source = pd.to_datetime(
        [
            "2025-01-02 21:15:00",
            "2025-01-03 09:15:00",
            "2025-01-03 11:30:00",
        ]
    )
    w_source = np.array([0.4, -0.6, 0.9], dtype=np.float64)
    df_src = pd.DataFrame({"dt": dt_source})
    df_tgt = pd.DataFrame({"dt": dt_target})

    mapped = map_tf_weight_to_base_grid(df_src, w_source, df_tgt)
    expected = np.array(
        [
            0.0,   # 21:05 (before 21:15 close)
            0.0,   # 21:10 (before 21:15 close)
            0.4,   # 21:15 (exact match with 21:15 close)
            0.4,   # 09:05 (carries 21:15 close across night break)
            -0.6,  # 09:15 (exact match with 09:15 close)
            -0.6,  # 10:15 (carries 09:15 close)
            -0.6,  # 10:35 (carries 09:15 close across morning break)
            0.9,   # 11:30 (exact match with 11:30 close)
            0.9,   # 13:35 (carries 11:30 close across lunch break)
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(mapped, expected)
    np.testing.assert_allclose(map_tf_weight_to_5m_grid(df_src, w_source, df_tgt, "15m"), expected)


def test_end_to_end_pipeline_truncation_and_perturbation_invariance(tmp_path):
    """Truncating or perturbing bars after t must not change features, signals, targets, or executed positions <= t."""
    raw5 = generate_symbol_kines("CAUSAL_E2E", "5分钟", "20240102", "20240115", seed=108)
    raw5 = raw5.rename(columns={"vol": "volume"}).iloc[:180].reset_index(drop=True)
    raw5["dt"] = pd.to_datetime(raw5["dt"])
    raw5["date"] = raw5["dt"].dt.date

    # Construct 15m completed bars from 5m bars (every 3 bars, dt = last)
    g15 = raw5.groupby(np.arange(len(raw5)) // 3, sort=True)
    raw15 = g15.agg(
        dt=("dt", "last"),
        symbol=("symbol", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index(drop=True)
    raw15["date"] = raw15["dt"].dt.date

    cut15 = 40  # 40th 15m bar -> 120th 5m bar
    cut5 = cut15 * 3
    cutoff_dt = raw15["dt"].iloc[cut15 - 1]

    # Full run
    df15_full = hb.attach_native_czsc_signals(hb.enrich_kline_features(raw15.copy()), "15m", tmp_path / "full")
    _, w15_full = af.compute_atomic_signal(df15_full, "AC01L", soft_gate=False)
    w5_full = map_tf_weight_to_base_grid(df15_full, w15_full, raw5)
    ret_full, pos_full = build_asset_execution_inputs(raw5, w5_full, execution_clock="next_open", liquidate_at_end=False)
    ledger_full = account_portfolio_execution(
        pd.DataFrame({"A": ret_full}),
        pd.DataFrame({"A": pos_full}),
        {"A": 0.0002},
    )

    # Truncated run (only data <= cutoff_dt)
    raw15_trunc = raw15.iloc[:cut15].copy()
    raw5_trunc = raw5.iloc[:cut5].copy()
    df15_trunc = hb.attach_native_czsc_signals(hb.enrich_kline_features(raw15_trunc), "15m", tmp_path / "trunc")
    _, w15_trunc = af.compute_atomic_signal(df15_trunc, "AC01L", soft_gate=False)
    w5_trunc = map_tf_weight_to_base_grid(df15_trunc, w15_trunc, raw5_trunc)
    ret_trunc, pos_trunc = build_asset_execution_inputs(
        raw5_trunc, w5_trunc, execution_clock="next_open", liquidate_at_end=False
    )
    ledger_trunc = account_portfolio_execution(
        pd.DataFrame({"A": ret_trunc}),
        pd.DataFrame({"A": pos_trunc}),
        {"A": 0.0002},
    )

    # Perturbed future run (bars after cutoff_dt multiplied by 3.0)
    raw15_pert = raw15.copy()
    raw15_pert.loc[cut15:, ["open", "high", "low", "close"]] *= 3.0
    raw5_pert = raw5.copy()
    raw5_pert.loc[cut5:, ["open", "high", "low", "close"]] *= 3.0
    df15_pert = hb.attach_native_czsc_signals(hb.enrich_kline_features(raw15_pert), "15m", tmp_path / "pert")
    _, w15_pert = af.compute_atomic_signal(df15_pert, "AC01L", soft_gate=False)
    w5_pert = map_tf_weight_to_base_grid(df15_pert, w15_pert, raw5_pert)
    ret_pert, pos_pert = build_asset_execution_inputs(
        raw5_pert, w5_pert, execution_clock="next_open", liquidate_at_end=False
    )
    ledger_pert = account_portfolio_execution(
        pd.DataFrame({"A": ret_pert}),
        pd.DataFrame({"A": pos_pert}),
        {"A": 0.0002},
    )

    assert raw5_trunc["dt"].iloc[-1] == cutoff_dt
    np.testing.assert_allclose(w15_full[:cut15], w15_trunc)
    np.testing.assert_allclose(w15_full[:cut15], w15_pert[:cut15])
    np.testing.assert_allclose(w5_full[:cut5], w5_trunc)
    np.testing.assert_allclose(w5_full[:cut5], w5_pert[:cut5])
    np.testing.assert_allclose(pos_full.iloc[:cut5].to_numpy(), pos_trunc.to_numpy())
    np.testing.assert_allclose(pos_full.iloc[:cut5].to_numpy(), pos_pert.iloc[:cut5].to_numpy())
    np.testing.assert_allclose(
        ledger_full["executed_weights_df"]["A"].iloc[:cut5].to_numpy(),
        ledger_trunc["executed_weights_df"]["A"].to_numpy(),
    )
    np.testing.assert_allclose(
        ledger_full["bar_net_returns"].iloc[:cut5].to_numpy(),
        ledger_pert["bar_net_returns"].iloc[:cut5].to_numpy(),
    )
