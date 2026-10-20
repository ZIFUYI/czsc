"""Research feature columns use the same canonical Rust indicators as signals."""

import numpy as np
import pandas as pd
import pytest

import czsc
from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.hypothesis_bank import enrich_kline_features


def _bars():
    frame = generate_symbol_kines("SHFE.RB9999", "30分钟", "20220101", "20220301", seed=317)
    frame = frame.rename(columns={"vol": "volume"}).iloc[:100].reset_index(drop=True)
    frame["dt"] = pd.to_datetime(frame["dt"])
    frame["date"] = frame["dt"].dt.date
    return frame


def test_research_macd_obeys_native_full_warmup_and_undoubled_histogram():
    enriched = enrich_kline_features(_bars())
    assert enriched[["dif", "dea", "macd_hist"]].iloc[:33].isna().all().all()
    np.testing.assert_allclose(enriched["macd_hist"], enriched["dif"] - enriched["dea"], equal_nan=True)


def test_research_columns_are_direct_shared_operator_results():
    frame = _bars()
    enriched = enrich_kline_features(frame)
    close = frame["close"].to_numpy()
    volume = frame["volume"].to_numpy()
    np.testing.assert_allclose(enriched["rsi14"], czsc.rsi(close, 14), equal_nan=True)
    for column, values in zip(["dif", "dea", "macd_hist"], czsc.macd(close, 12, 26, 9), strict=True):
        np.testing.assert_allclose(enriched[column], values, equal_nan=True)
    np.testing.assert_allclose(enriched["obv"], czsc.obv(close, volume))


def test_rsi_wilder_seed_and_decay_shared_reference():
    # A numerical operator fixture, not simulated market-performance evidence.
    result = czsc.rsi([100.0, 102.0, 101.0, 104.0, 102.0, 105.0], 3)
    assert np.isnan(result[:3]).all()
    np.testing.assert_allclose(result[3:], [250 / 3, 500 / 9, 4700 / 63], rtol=0, atol=1e-12)


@pytest.mark.parametrize("period", [0, 4, 8])
def test_rsi_unready_and_zero_periods_preserve_length(period):
    assert np.isnan(czsc.rsi([1.0, 2.0, 3.0, 4.0], period)).all()


def test_rsi_flat_up_down_and_empty_contract():
    assert czsc.rsi([], 14) == []
    assert czsc.rsi([1.0] * 5, 3)[3:] == [0.0, 0.0]
    assert czsc.rsi([1.0, 2.0, 3.0, 4.0, 5.0], 3)[3:] == [100.0, 100.0]
    assert czsc.rsi([5.0, 4.0, 3.0, 2.0, 1.0], 3)[3:] == [0.0, 0.0]


def test_macd_seed_and_histogram_shared_reference():
    dif, dea, hist = czsc.macd([1.0, 2.0, 1.0, 4.0, 2.0, 5.0], 2, 3, 2)
    for values in [dif, dea, hist]:
        assert len(values) == 6
        assert np.isnan(values[:3]).all()
    np.testing.assert_allclose([dif[3], dea[3], hist[3]], [0.5, 1 / 3, 1 / 6], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(hist[3:], np.asarray(dif[3:]) - dea[3:])


@pytest.mark.parametrize("periods", [(0, 3, 2), (2, 0, 2), (2, 3, 0), (12, 26, 9)])
def test_macd_empty_and_invalid_periods_preserve_length(periods):
    assert czsc.macd([], *periods) == ([], [], [])
    for values in czsc.macd([1.0, 2.0, 3.0], *periods):
        assert len(values) == 3
        assert np.isnan(values).all()


def test_obv_close_to_close_seed_ties_empty_and_length_error():
    assert czsc.obv([10.0, 11.0, 11.0, 9.0], [2.0, 3.0, 5.0, 7.0]) == [2.0, 5.0, 5.0, -2.0]
    assert czsc.obv([], []) == []
    with pytest.raises(ValueError, match="close and volume must have the same length"):
        czsc.obv([10.0], [])


def test_research_indicators_do_not_change_when_future_bars_arrive():
    frame = _bars()
    full = enrich_kline_features(frame)
    for count in [14, 40]:
        prefix = enrich_kline_features(frame.iloc[:count])
        for column in ["rsi14", "dif", "dea", "macd_hist", "obv"]:
            np.testing.assert_array_equal(full[column].iloc[:count], prefix[column])


def test_obv_nonfinite_inputs_preserve_prefix_and_invalidate_tail():
    for close, volume in [
        ([10.0, 11.0, np.nan, 9.0], [2.0, 3.0, 5.0, 7.0]),
        ([10.0, 11.0, 11.0, 9.0], [2.0, 3.0, np.inf, 7.0]),
    ]:
        actual = czsc.obv(close, volume)
        assert actual[:2] == [2.0, 5.0]
        assert np.isnan(actual[2:]).all()


def test_public_indicator_exports_are_the_native_operators():
    from czsc import _native

    for name in ["rsi", "macd", "obv"]:
        assert getattr(czsc, name) is getattr(_native, name)
