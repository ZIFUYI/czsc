"""Rust/Python public inclusion contract for doji and ambiguous initial bars."""

import numpy as np
import pytest

import czsc
from czsc.mock import generate_symbol_kines


def _bars(ohlc):
    frame = generate_symbol_kines("TEST", "30分钟", "20220101", "20220103", seed=321).iloc[:3].copy()
    frame[["open", "high", "low", "close"]] = np.asarray(ohlc, dtype=float)
    frame["amount"] = frame["vol"] * frame["close"]
    mirrored = frame.copy()
    mirrored["open"] = 300.0 - frame["open"]
    mirrored["close"] = 300.0 - frame["close"]
    mirrored["high"] = 300.0 - frame["low"]
    mirrored["low"] = 300.0 - frame["high"]
    return tuple(czsc.format_standard_kline(x, freq=czsc.Freq.F30) for x in (frame, mirrored))


def _new_bar(bar):
    return czsc.NewBar(
        symbol=bar.symbol,
        dt=bar.dt,
        freq=bar.freq,
        id=bar.id,
        open=bar.open,
        close=bar.close,
        high=bar.high,
        low=bar.low,
        vol=bar.vol,
        amount=bar.amount,
        elements=[bar],
    )


def _merge(bars):
    return czsc.remove_include(_new_bar(bars[0]), _new_bar(bars[1]), bars[2])


def _assert_mirror(a, b):
    assert (a.open, a.high, a.low, a.close) == (300 - b.open, 300 - b.low, 300 - b.high, 300 - b.close)
    assert a.dt == b.dt
    assert a.vol == b.vol
    assert len(a.elements) == len(b.elements)


@pytest.mark.parametrize("close", [96.0, 110.0, 129.0])
def test_doji_stays_neutral_and_clamps_to_merged_range(close):
    original, mirror = _bars([(95, 110, 90, 105), (105, 120, 100, 115), (close, 130, 95, close)])
    included, a = _merge(original)
    mirrored_included, b = _merge(mirror)
    assert included and mirrored_included
    assert (a.low, a.high) == (100, 130)
    assert a.open == a.close == max(close, 100)
    assert b.open == b.close
    _assert_mirror(a, b)


@pytest.mark.parametrize(
    "h1,l1,h2,l2",
    [(130, 90, 120, 100), (120, 100, 130, 90), (120, 90, 120, 100), (120, 90, 130, 90), (120, 90, 120, 90)],
)
def test_ambiguous_initial_direction_defers_inclusion(h1, l1, h2, l2):
    original, mirror = _bars([(105, h1, l1, 110), (105, h2, l2, 110), (108, 115, 105, 112)])
    included, a = _merge(original)
    mirrored_included, b = _merge(mirror)
    assert not included and not mirrored_included
    assert (a.open, a.high, a.low, a.close) == (108, 115, 105, 112)
    assert len(a.elements) == 1
    _assert_mirror(a, b)


@pytest.mark.parametrize("open_,close,expected", [(108, 112, (105, 120)), (112, 108, (120, 105))])
def test_known_direction_keeps_existing_body_rule(open_, close, expected):
    original, mirror = _bars([(95, 110, 90, 105), (105, 120, 100, 115), (open_, 115, 105, close)])
    included, a = _merge(original)
    mirrored_included, b = _merge(mirror)
    assert included and mirrored_included
    assert (a.open, a.close) == expected
    _assert_mirror(a, b)
