"""The native admission episodes and portfolio fills share one next-open ledger."""

import numpy as np
import pandas as pd
import pytest

import czsc
from czsc.mock import generate_symbol_kines
from examples.multi_factor_cta.synthesis_engine import account_portfolio_execution, build_asset_execution_inputs


def _inputs(count=12):
    frame = generate_symbol_kines("SHFE.RB9999", "30分钟", "20220101", "20220110", seed=112).iloc[:count].copy()
    frame["dt"] = pd.to_datetime(frame["dt"])
    weights = np.zeros(len(frame))
    weights[:5] = [0.5, 0.25, -0.75, -0.25, 0.0]
    atr = np.maximum(frame["high"].to_numpy() - frame["low"].to_numpy(), 1e-6)
    return frame, weights, atr


def _episodes(frame, weights, atr, fee=0.001):
    return czsc.execution_episodes(
        frame["open"].to_numpy(),
        frame["high"].to_numpy(),
        frame["low"].to_numpy(),
        frame["close"].to_numpy(),
        atr,
        weights,
        fee,
    )


def _ledger(frame, weights, fee=0.001):
    returns, filled = build_asset_execution_inputs(frame, weights, liquidate_at_end=False)
    return account_portfolio_execution(returns.to_frame("asset"), filled.to_frame("asset"), {"asset": fee})


def _record(episode):
    return tuple(
        getattr(episode, field)
        for field in (
            "trigger_idx",
            "entry_idx",
            "exit_idx",
            "direction",
            "net_ret",
            "weighted_net_ret",
            "quality_w",
            "atr_pct",
            "mfe_atr",
            "mae_atr",
        )
    )


def test_native_episodes_and_bar_returns_match_portfolio_ledger():
    frame, weights, atr = _inputs()
    expected = _ledger(frame, weights)
    native = czsc.execution_net_returns(frame["open"].to_numpy(), weights, 0.001)
    np.testing.assert_allclose(native, expected["bar_net_returns"], rtol=0, atol=1e-15)
    episodes = _episodes(frame, weights, atr)
    assert [(e.trigger_idx, e.entry_idx, e.exit_idx, e.direction) for e in episodes] == [(0, 1, 3, 1), (2, 3, 5, -1)]
    assert sum(e.weighted_net_ret for e in episodes) == pytest.approx(sum(native), abs=1e-15)
    for episode in episodes:
        assert episode.net_ret * episode.quality_w == pytest.approx(episode.weighted_net_ret, abs=1e-15)
    # Independently derive actual units: resizing and exit include price drift.
    price = frame.open.to_numpy()
    quantity = np.r_[0.0, weights[:-1]] / price
    turnover = np.abs(np.diff(quantity, prepend=0.0)) * price
    np.testing.assert_allclose(expected["turnover_df"]["asset"], turnover, atol=1e-15)


def test_scaling_positions_scales_all_turnover_and_weighted_returns_once():
    frame, weights, atr = _inputs()
    full, half = _episodes(frame, weights, atr), _episodes(frame, weights * 0.5, atr)
    assert len(full) == len(half) == 2
    for a, b in zip(full, half, strict=True):
        assert b.quality_w == 0.5 * a.quality_w
        assert b.weighted_net_ret == pytest.approx(0.5 * a.weighted_net_ret, abs=1e-15)
        assert b.net_ret == pytest.approx(a.net_ret, abs=1e-15)
    baseline = _ledger(frame, weights)
    scaled = _ledger(frame, weights * 0.5)
    np.testing.assert_allclose(scaled["cost_returns_df"], baseline["cost_returns_df"] * 0.5, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        czsc.execution_net_returns(frame["open"].to_numpy(), weights * 0.5, 0.001),
        baseline["bar_net_returns"] * 0.5,
        rtol=0,
        atol=1e-15,
    )


def test_completed_episode_and_ledger_prefixes_do_not_change_when_future_bars_arrive():
    frame, weights, atr = _inputs()
    full = _episodes(frame, weights, atr)
    whole_ledger = czsc.execution_net_returns(frame["open"].to_numpy(), weights, 0.001)
    for count, finished in [(2, 0), (3, 0), (4, 1), (5, 1), (6, 2)]:
        prefix = _episodes(frame.iloc[:count], weights[:count], atr[:count])
        assert len(prefix) == finished
        assert [_record(e) for e in prefix] == [_record(e) for e in full[:finished]]
        assert (
            czsc.execution_net_returns(frame["open"].iloc[:count].to_numpy(), weights[:count], 0.001)
            == whole_ledger[:count]
        )


def test_open_right_boundary_is_not_falsely_liquidated_or_counted_as_an_episode():
    frame, _, atr = _inputs()
    weights = np.full(len(frame), -0.5)
    assert _episodes(frame, weights, atr) == []
    expected = _ledger(frame, weights)
    actual = czsc.execution_net_returns(frame["open"].to_numpy(), weights, 0.001)
    np.testing.assert_allclose(actual, expected["bar_net_returns"], rtol=0, atol=1e-15)
    price = frame.open.to_numpy()
    expected_resize = abs(-0.5 / price[-1] - (-0.5 / price[-2])) * price[-1]
    assert expected["turnover_df"]["asset"].iloc[-1] == pytest.approx(expected_resize)
    # A final close-time exit signal has no observed next open and therefore no fill.
    weights[-1] = 0
    assert _episodes(frame, weights, atr) == []
    np.testing.assert_array_equal(actual, czsc.execution_net_returns(frame["open"].to_numpy(), weights, 0.001))


def test_exit_bar_extrema_are_not_part_of_an_already_closed_trade():
    frame, _, atr = _inputs()
    weights = np.zeros(len(frame))
    weights[0] = 0.5
    before = _episodes(frame, weights, atr)
    frame.loc[frame.index[2], "high"] *= 1000
    frame.loc[frame.index[2], "low"] /= 1000
    after = _episodes(frame, weights, atr)
    assert [_record(e) for e in before] == [_record(e) for e in after]


def test_shared_execution_validation_and_empty_contract():
    frame, weights, atr = _inputs()
    assert czsc.execution_episodes([], [], [], [], [], [], 0.001) == []
    assert czsc.execution_net_returns([], [], 0.001) == []
    with pytest.raises(ValueError, match="lengths"):
        _episodes(frame, weights[:-1], atr)
    with pytest.raises(ValueError, match="lengths"):
        czsc.execution_net_returns(frame["open"].to_numpy(), weights[:-1], 0.001)
    for fee in [-0.1, np.inf, np.nan]:
        with pytest.raises(ValueError, match="fee_rate"):
            _episodes(frame, weights, atr, fee)
        with pytest.raises(ValueError, match="fee_rate"):
            czsc.execution_net_returns(frame["open"].to_numpy(), weights, fee)
    for bad_weight in [1.1, np.inf, np.nan]:
        invalid = weights.copy()
        invalid[1] = bad_weight
        with pytest.raises(ValueError, match="weights"):
            _episodes(frame, invalid, atr)
        with pytest.raises(ValueError, match="weights"):
            czsc.execution_net_returns(frame["open"].to_numpy(), invalid, 0.001)
