"""Independent quantity/cash-flow contracts for the shared Rust execution ledger."""

import numpy as np
import pytest

import czsc
from czsc import _native
from czsc.mock import generate_symbol_kines


def _prices(ratios=(1.0, 1.0, 1.1, 0.99, 1.2, 1.15)):
    frame = generate_symbol_kines("TEST", "日线", "20220101", "20220201", seed=881)
    return float(frame.open.iloc[0]) * np.array(ratios)


def _independent_fills(prices, targets, capital):
    quantity = 0.0
    previous_price = prices[0]
    gross, turnover, quantities = [], [], []
    for price, target in zip(prices, targets, strict=True):
        gross.append(quantity * (price - previous_price) / capital)
        next_quantity = target * capital / price
        turnover.append(abs(next_quantity - quantity) * price / capital)
        quantities.append(next_quantity)
        quantity, previous_price = next_quantity, price
    return np.array(gross), np.array(turnover), np.array(quantities)


@pytest.mark.parametrize("direction", [1.0, -1.0])
@pytest.mark.parametrize("capital", [1.0, 1_000_000.0])
def test_actual_quantities_restore_targets_and_charge_every_fill(direction, capital):
    prices = _prices()
    weights = np.array([0.0, 0.5, 0.5, 0.25, -0.75, 0.0]) * direction
    gross, turnover, quantities = _independent_fills(prices, weights, capital)
    actual = czsc.execution_ledger(prices, weights, 0.003, initial_capital=capital)
    np.testing.assert_allclose(actual.quantities, quantities, atol=1e-12)
    np.testing.assert_allclose(actual.trade_quantities, np.diff(quantities, prepend=0.0), atol=1e-12)
    np.testing.assert_allclose(actual.gross_returns, gross, atol=1e-14)
    np.testing.assert_allclose(actual.turnover, turnover, atol=1e-14)
    np.testing.assert_allclose(actual.cost_returns, turnover * 0.003, atol=1e-14)
    np.testing.assert_allclose(actual.net_returns, gross - turnover * 0.003, atol=1e-14)
    # Same target after a 10% price move still restores the fixed-capital exposure.
    assert actual.turnover[2] == pytest.approx(0.05)
    assert actual.turnover[-1] == pytest.approx(0.75 * prices[-1] / prices[-2])


def test_resize_can_cost_less_than_weight_delta_without_negative_fees():
    prices = _prices((1.0, 1.0, 0.8, 0.8))
    weights = [0.0, 0.5, 0.4, 0.0]
    ledger = czsc.execution_ledger(prices, weights, 0.001)
    assert ledger.turnover[2] == pytest.approx(0.0, abs=1e-15)
    assert ledger.turnover[-1] == pytest.approx(0.4)
    assert min(ledger.cost_returns) >= 0


def test_return_and_price_inputs_agree_with_missing_quotes_and_prefixes():
    prices = _prices((1.0, 1.0, 1.0, 1.2, 1.2))
    weights = np.array([0.0, 0.5, 0.5, 0.5, 0.0])
    prices[2] = np.nan
    tradable = [True, True, False, True, True]
    returns = [0.0, 0.0, 0.0, 0.2, 0.0]
    actual = czsc.execution_ledger(prices, weights, 0.002, tradable=tradable)
    by_return = czsc.executed_weight_ledger(returns, weights, 0.002, tradable=tradable)
    for field in ("gross_returns", "turnover", "cost_returns", "net_returns", "executed_weights"):
        np.testing.assert_allclose(getattr(actual, field), getattr(by_return, field), atol=1e-14)
    assert actual.quantities[2] == actual.quantities[1]
    assert actual.turnover[2] == 0
    assert actual.turnover[3] == pytest.approx(0.1)
    for length in range(1, len(prices)):
        prefix = czsc.execution_ledger(prices[:length], weights[:length], 0.002, tradable=tradable[:length])
        assert prefix.net_returns == actual.net_returns[:length]
    with pytest.raises(ValueError, match="tradable|quote|frozen|unchanged|weight"):
        czsc.execution_ledger(prices, [0.0, 0.5, 0.0, 0.5, 0.0], 0.002, tradable=tradable)


def test_decisions_terminal_close_and_closed_intervals_preserve_cash_flows():
    prices = _prices()
    decisions = np.array([0.5, 0.5, 0.25, -0.75, -0.75, 1.0])
    terminal = prices[-1] * 1.05
    expected_prices = np.r_[prices, terminal]
    expected_weights = np.r_[0.0, decisions[:-1], 0.0]
    gross, turnover, _ = _independent_fills(expected_prices, expected_weights, 1.0)
    complete = czsc.decision_execution_ledger(prices, decisions, 0.001, terminal_close=terminal)
    np.testing.assert_allclose(complete.net_returns, gross - turnover * 0.001, atol=1e-14)
    open_ended = czsc.decision_execution_ledger(prices, decisions, 0.001)
    assert open_ended.net_returns == complete.net_returns[:-1]
    intervals = czsc.closed_execution_intervals(complete)
    assert sum(intervals.net_returns) == pytest.approx(sum(complete.net_returns), abs=1e-14)
    assert sum(intervals.turnover) == pytest.approx(sum(complete.turnover), abs=1e-14)
    forward = np.r_[0.0, prices[2:] / prices[1:-1] - 1.0, terminal / prices[-1] - 1.0]
    quick = czsc.decision_interval_ledger(forward, decisions, 0.001)
    np.testing.assert_allclose(quick.net_returns, complete.net_returns, atol=1e-14)
    with pytest.raises(ValueError, match="flat|closed|liquidat"):
        czsc.closed_execution_intervals(open_ended)


def test_shared_cost_application_and_public_exports_are_native():
    for name in (
        "ExecutionLedger",
        "ClosedExecutionIntervals",
        "executed_weight_ledger",
        "execution_ledger",
        "decision_execution_ledger",
        "decision_interval_ledger",
        "closed_execution_intervals",
        "standardize_execution_returns",
    ):
        assert getattr(czsc, name) is getattr(_native, name)
    assert czsc.standardize_execution_returns([0.1, -0.2], [0.003, 0.004]) == pytest.approx([0.097, -0.204])
    with pytest.raises(ValueError, match="length"):
        czsc.standardize_execution_returns([0.1], [])
    for fee in [-0.001, np.nan, np.inf]:
        with pytest.raises(ValueError, match="fee_rate"):
            czsc.execution_ledger([1.0], [0.0], fee)
    for capital in [0.0, -1.0, np.nan, np.inf]:
        with pytest.raises(ValueError, match="capital"):
            czsc.execution_ledger([1.0], [0.0], initial_capital=capital)
    for cost in [-0.001, np.nan, np.inf]:
        with pytest.raises(ValueError, match="cost|finite|nonnegative"):
            czsc.standardize_execution_returns([0.1], [cost])


def test_one_bar_fold_has_no_executable_decision_or_terminal_fee():
    from examples.alpha_mining_agent.forensic_pipeline import (
        _fast_next_open_slice_stats,
        _prepare_next_open_slice,
        backtest_slice,
    )

    frame = generate_symbol_kines("TEST", "日线", "20220101", "20220201", seed=882).iloc[:1].copy()
    frame["close"] = frame["open"] * 1.1
    frame["high"] = frame["close"]
    returns, days = _prepare_next_open_slice(frame)
    assert returns.tolist() == [0.0]
    quick = _fast_next_open_slice_stats(returns, days, np.array([1.0]), 0.003)
    full = backtest_slice(frame, np.array([1.0]))
    for key in ("cum_ret_pct", "ann_ret_pct", "sharpe", "mdd_pct", "calmar"):
        assert quick[key] == full[key] == 0.0
    assert full["nav_series"].tolist() == [1.0]
    assert full["trades"] == full["active_pct"] == 0


@pytest.mark.parametrize("invalid_clock", ["reversed", "duplicate"])
def test_research_report_rejects_invalid_execution_order(invalid_clock):
    from examples.alpha_mining_agent.forensic_pipeline import backtest_slice

    frame = generate_symbol_kines("TEST", "日线", "20220101", "20220201", seed=883).iloc[:4].copy()
    if invalid_clock == "reversed":
        frame = frame.iloc[::-1]
    else:
        frame.loc[frame.index[1], "dt"] = frame.dt.iloc[0]
    with pytest.raises(ValueError, match="unique and ordered"):
        backtest_slice(frame, np.ones(len(frame)))


def test_native_executed_weight_ledger_roll_mask_parity():
    returns = [0.0, 0.0, 0.15, 0.02, 0.0]
    weights = [0.0, 0.5, 0.5, 0.5, 0.0]
    roll_mask = [False, False, True, False, False]
    ledger = czsc.executed_weight_ledger(returns, weights, fee_rate=0.001, roll_mask=roll_mask)
    np.testing.assert_allclose(ledger.gross_returns, [0.0, 0.0, 0.0, 0.01, 0.0], atol=1e-14)
    np.testing.assert_allclose(ledger.turnover, [0.0, 0.5, 1.0, 0.01, 0.5], atol=1e-14)
    np.testing.assert_allclose(ledger.cost_returns, [0.0, 0.0005, 0.001, 0.00001, 0.0005], atol=1e-14)
    np.testing.assert_allclose(ledger.net_returns, [0.0, -0.0005, -0.001, 0.00999, -0.0005], atol=1e-14)
