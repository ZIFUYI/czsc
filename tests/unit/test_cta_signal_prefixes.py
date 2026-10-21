"""The signal prefix audit catches future-dependent weights and stale cache entries."""

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from scripts.validate_cta_signal_prefixes import compare_unit_prefixes


def _frames():
    base = generate_symbol_kines("SHFE.RB9999", "15分钟", "20240102", "20240105", seed=17)
    raw5 = generate_symbol_kines("SHFE.RB9999", "5分钟", "20240102", "20240105", seed=17)
    return {"15m": base, "30m": base.iloc[1::2].copy().reset_index(drop=True)}, raw5


def test_prefix_audit_slices_each_clock_and_accepts_causal_weights():
    frames, raw5 = _frames()
    cutoff = frames["15m"].dt.iloc[len(frames["15m"]) // 2]
    calls = []

    def causal(current, micro, candidates):
        calls.append((current, micro))
        weight = (current["15m"].close > current["15m"].open).to_numpy(dtype=np.float32)
        return dict.fromkeys(candidates, weight)

    report = compare_unit_prefixes(frames, raw5, ["AC01L@30m", "H5M_CLV01L@5m"], causal, cutoff=cutoff)
    assert report.passed.all() and len(report) == 2
    assert calls[1][1].dt.max() <= cutoff
    assert all(frame.dt.max() <= cutoff for frame in calls[1][0].values())
    assert all(frame.dt.max() > cutoff for frame in calls[0][0].values())


def test_prefix_audit_detects_future_sensitive_weights_and_cache_mismatch():
    frames, raw5 = _frames()
    cutoff = frames["15m"].dt.iloc[len(frames["15m"]) // 2]

    def future_sensitive(current, micro, candidates):
        value = float(current["15m"].close.mean())
        return {uid: np.full(len(current["15m"]), value, dtype=np.float32) for uid in candidates}

    cache = {"bad@15m": np.zeros(len(frames["15m"]), dtype=np.float32)}
    report = compare_unit_prefixes(frames, raw5, list(cache), future_sensitive, cutoff=cutoff, cached_units=cache)
    assert set(report.comparison) == {"future_prefix", "verified_cache_recompute"}
    assert not report.passed.any()
    assert (report.changed_bars == report.bars_compared).all()
    assert report.first_difference_dt.notna().all()


def test_prefix_audit_rejects_silently_missing_candidates():
    frames, raw5 = _frames()
    with pytest.raises(ValueError, match="complete frozen inventory"):
        compare_unit_prefixes(frames, raw5, ["AC01L@30m"], lambda *_: {}, cutoff=pd.Timestamp("2024-01-03"))
