"""Shared Rust Position replay, including the research registry data boundary."""

import numpy as np
import pytest

from czsc import Event, Freq, Operate, Position, Signal, format_standard_kline
from czsc.mock import generate_symbol_kines
from examples.multi_factor_cta.factor_engine import RuleFactorRegistry


def _inputs():
    frame = generate_symbol_kines("TEST", "30分钟", "20220101", "20220110", seed=81).iloc[:6].copy()
    frame["close"] = 100.0
    frame["high"] = np.maximum(frame["high"], frame["close"])
    frame["low"] = np.minimum(frame["low"], frame["close"])
    frame["cz_test_state"] = ["多", "其他", "平多", "空", "其他", "平空"]
    bars = format_standard_kline(frame, freq=Freq.F30)
    rows = frame[["cz_test_state"]].astype(str).to_dict("records")
    return frame, bars, rows


def _position(t0=True):
    events = [
        Event(operate=op, signals_all=[Signal(f"30分钟_test_state_{value}_任意_任意_0")])
        for op, value in [(Operate.LO, "多"), (Operate.SO, "空"), (Operate.LE, "平多"), (Operate.SE, "平空")]
    ]
    return Position(symbol="TEST", opens=events[:2], exits=events[2:], t0=t0, stop_loss=0, timeout=100)


def test_native_batch_matches_sequential_and_keeps_template_unchanged():
    frame, bars, rows = _inputs()
    position = _position()
    triggers, weights = position.evaluate_signals(bars, rows)
    assert triggers == [True, False, False, True, False, False]
    assert weights == [1, 1, 0, -1, -1, 0]
    expected = _position()
    held = []
    for i, row in frame.reset_index(drop=True).iterrows():
        expected.update({"dt": row["dt"], "id": i, "close": row.close, "30分钟_test_state": row.cz_test_state})
        held.append(float(expected.pos))
    np.testing.assert_array_equal(weights, held)
    assert float(position.pos) == 0
    assert position.evaluate_signals(bars, rows) == (triggers, weights)


def test_registry_executes_native_position_and_preserves_t1():
    frame, bars, rows = _inputs()
    frame = frame.rename(columns={"vol": "volume"})
    expected = _position(False).evaluate_signals(bars, rows)
    actual = RuleFactorRegistry._eval_czsc_native_position(frame, _position(False), "30m")
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[1][2] == 1  # Intraday exit is forbidden by T+1.


def test_batch_rejects_mismatched_or_unordered_rows():
    _, bars, rows = _inputs()
    with pytest.raises(ValueError, match="length"):
        _position().evaluate_signals(bars, rows[:-1])
    with pytest.raises(ValueError, match="ordered"):
        _position().evaluate_signals(bars[::-1], rows)
    assert _position().evaluate_signals([], []) == ([], [])


def test_batch_rejects_another_symbol_than_the_position_template():
    frame, _, rows = _inputs()
    frame["symbol"] = "OTHER"
    bars = format_standard_kline(frame, freq=Freq.F30)
    with pytest.raises(ValueError, match="Position symbol"):
        _position().evaluate_signals(bars, rows)


def test_full_and_research_keys_are_equivalent_and_ambiguous_rows_are_rejected():
    _, bars, rows = _inputs()
    position = _position()
    expected = position.evaluate_signals(bars, rows)
    full = [{"30分钟_test_state": f"{row['cz_test_state']}_任意_任意_0"} for row in rows]
    assert position.evaluate_signals(bars, full) == expected
    rows[0]["30分钟_test_state"] = "空_任意_任意_0"
    with pytest.raises(ValueError, match="duplicate signal key"):
        position.evaluate_signals(bars, rows)
    rows[0].clear()
    with pytest.raises(ValueError, match="missing required signal"):
        position.evaluate_signals(bars, rows)
    assert float(position.pos) == 0


def test_dataframe_default_ids_do_not_disable_native_timeout():
    frame, bars, rows = _inputs()
    assert [bar.id for bar in bars] == [0] * len(bars)  # Current DataFrame boundary omits optional IDs.
    position = Position(
        symbol="TEST",
        opens=[Event(operate=Operate.LO, signals_all=[Signal("30分钟_test_state_多_任意_任意_0")])],
        t0=True,
        stop_loss=0,
        timeout=2,
    )
    for row in rows[1:]:
        row["cz_test_state"] = "其他"
    assert position.evaluate_signals(bars, rows)[1] == [1, 1, 1, 0, 0, 0]
