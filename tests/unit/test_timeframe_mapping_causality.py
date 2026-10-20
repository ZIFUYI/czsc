"""Timeframe alignment must use observed timestamps, never array-length ratios."""

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.forensic_pipeline import map_tf_weight_to_5m_grid
from examples.multi_factor_cta.data_engine import map_tf_weight_to_base_grid


@pytest.fixture(params=["base", "5m", "15m"])
def mapper(request):
    if request.param == "base":
        return map_tf_weight_to_base_grid
    return lambda source, weights, target: map_tf_weight_to_5m_grid(source, weights, target, request.param)


def _bars():
    return generate_symbol_kines("TEST", "5分钟", "20230103", "20230104", seed=64).iloc[:18].copy()


def test_equal_length_does_not_mean_equal_timestamps(mapper):
    target = _bars().iloc[:6]
    source = target.copy()
    source["dt"] += pd.Timedelta(minutes=1)
    weights = np.arange(1, len(source) + 1) / 10
    actual = mapper(source, weights, target)
    np.testing.assert_allclose(actual, np.r_[0.0, weights[:-1]])


def test_future_source_append_cannot_change_historical_mapping(mapper):
    target = _bars().iloc[:9]
    source = target.iloc[[2, 5]].copy()
    weights = np.array([0.25, -0.5])
    before = mapper(source, weights, target)
    extended = pd.concat([source, _bars().iloc[[12]]], ignore_index=True)
    after = mapper(extended, np.r_[weights, 1.0], target)
    np.testing.assert_array_equal(after, before)
    np.testing.assert_allclose(before, [0, 0, 0.25, 0.25, 0.25, -0.5, -0.5, -0.5, -0.5])


def test_missing_target_bar_still_uses_latest_observed_source(mapper):
    bars = _bars()
    source = bars.iloc[[2, 5, 8]].copy()
    target = bars.iloc[[0, 1, 3, 4, 6, 7, 9]].copy()
    weights = np.array([0.2, -0.3, 0.4])
    expected = [0.0, 0.0, 0.2, 0.2, -0.3, -0.3, 0.4]
    np.testing.assert_allclose(mapper(source, weights, target), expected)
    for cutoff in (3, 5, 7):
        known = source["dt"] <= target["dt"].iloc[cutoff - 1]
        prefix = mapper(source.loc[known], weights[known], target.iloc[:cutoff])
        np.testing.assert_allclose(prefix, expected[:cutoff])


def test_empty_source_is_cash(mapper):
    target = _bars()
    np.testing.assert_array_equal(mapper(target.iloc[:0], np.array([]), target), np.zeros(len(target)))


@pytest.mark.parametrize(
    "invalid", ["source_duplicate", "source_unsorted", "target_duplicate", "target_unsorted", "weight_length"]
)
def test_ambiguous_or_misaligned_mapping_inputs_are_rejected(mapper, invalid):
    source = _bars().iloc[::3].copy()
    target = _bars()
    weights = np.ones(len(source))
    if invalid == "source_duplicate":
        source.loc[source.index[1], "dt"] = source["dt"].iloc[0]
    elif invalid == "source_unsorted":
        source = source.iloc[::-1]
    elif invalid == "target_duplicate":
        target.loc[target.index[1], "dt"] = target["dt"].iloc[0]
    elif invalid == "target_unsorted":
        target = target.iloc[::-1]
    else:
        weights = weights[:-1]
    with pytest.raises(ValueError):
        mapper(source, weights, target)
