"""OHLCV research serialization leaves normalization and validation in shared Rust."""

import numpy as np
import pandas as pd
import pytest

import czsc
from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.hypothesis_bank import research_bars


def _frame():
    frame = generate_symbol_kines("TEST", "30分钟", "20220101", "20220110", seed=731).iloc[:6].copy()
    frame = frame.rename(columns={"vol": "volume"}).drop(columns="amount")
    frame["volume"] = frame["volume"].astype(np.int64)
    return frame


def _records(bars):
    return [(b.id, b.symbol, b.freq, b.open, b.high, b.low, b.close, b.vol, b.amount) for b in bars]


def test_missing_amount_is_zero_and_ids_are_contiguous_without_polluting_ohlcv():
    frame = _frame()
    frame.attrs["research_context"] = object()
    before = frame.copy(deep=True)
    bars = research_bars(frame, "30m")
    assert [b.id for b in bars] == list(range(len(frame)))
    assert [b.amount for b in bars] == [0.0] * len(frame)
    np.testing.assert_array_equal([b.vol for b in bars], frame["volume"])
    for field in ["open", "high", "low", "close"]:
        np.testing.assert_array_equal([getattr(b, field) for b in bars], frame[field])
    pd.testing.assert_frame_equal(frame, before)
    assert "research_context" in frame.attrs
    assert "amount" not in frame and "vol" not in frame
    assert research_bars(frame.iloc[:0], "30m") == []


def test_existing_amount_and_equal_volume_aliases_are_preserved():
    frame = _frame()
    frame["amount"] = np.arange(1, len(frame) + 1, dtype=np.int64)
    frame["vol"] = frame["volume"].astype(float)
    before = frame.copy(deep=True)
    bars = research_bars(frame, "30m")
    np.testing.assert_array_equal([b.amount for b in bars], frame["amount"])
    pd.testing.assert_frame_equal(frame, before)
    frame.loc[frame.index[1], "vol"] += 1
    with pytest.raises(ValueError, match="volume and vol conflict"):
        research_bars(frame, "30m")


def test_shared_bytes_boundary_accepts_freq_or_string():
    from czsc import _native

    assert czsc.format_research_kline_bytes is _native.format_research_kline_bytes
    frame = _frame()
    data = czsc.to_arrow(frame)
    via_enum = czsc.format_research_kline_bytes(data, czsc.Freq.F30)
    via_name = czsc.format_research_kline_bytes(data, "30分钟")
    assert _records(via_enum) == _records(via_name) == _records(research_bars(frame, "30m"))
    assert len(czsc.structure_snapshots(via_enum)) == len(frame)


def test_null_required_values_invalid_ohlc_and_missing_columns_are_rejected():
    frame = _frame()
    frame.loc[frame.index[1], "close"] = np.nan
    with pytest.raises(ValueError, match="null"):
        research_bars(frame, "30m")
    frame = _frame()
    frame.loc[frame.index[1], "close"] = frame["high"].iloc[1] + 1
    with pytest.raises(ValueError, match="OHLC"):
        research_bars(frame, "30m")
    with pytest.raises(ValueError, match="symbol"):
        research_bars(_frame().drop(columns="symbol"), "30m")
