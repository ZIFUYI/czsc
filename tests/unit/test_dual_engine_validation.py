"""Engine plumbing checks using mock prices, never factor admission evidence."""

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.multi_factor_cta.dual_engine_validation import run_dual_engine_validation


def _inputs():
    frame = generate_symbol_kines("SHFE.RB9999", "日线", "20230101", "20230331", seed=813).iloc[:12]
    prices = frame.set_index(pd.to_datetime(frame["dt"]))[["open"]].rename(columns={"open": "A"})
    prices["B"] = prices["A"] * 0.7
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights.iloc[1:5, 0] = 0.4
    weights.iloc[5:8, 0] = -0.3
    weights.iloc[3:8, 1] = 0.2
    return prices, weights


def test_actual_broker_gross_pnl_and_zero_fee_wbt_reconcile():
    prices, weights = _inputs()
    result = run_dual_engine_validation(prices, weights, {"A": 0.0, "B": 0.0})
    summary = result["summary"]
    assert summary["backtrader_fixed_orders"] > 0
    assert summary["gross_execution_reconciled"]
    assert summary["fixed_gross_max_bar_error"] < 1e-10
    assert summary["backtrader_fixed_net_return"] == pytest.approx(summary["wbt_standardized_net_return"], abs=1e-10)
    assert summary["wbt_standardized_formal_net_difference"] == pytest.approx(0.0, abs=1e-10)
    assert summary["terminal_inventory"] == 0.0
    assert summary["backtrader_equity_final_return"] == pytest.approx(
        summary["fixed_bar_return_compounding_diagnostic"]
    )
    assert summary["wbt_standardized_formal_max_bar_error"] < 1e-10
    assert summary["wbt_standardized_formal_max_daily_error"] < 1e-10
    assert summary["daily_metrics"]["wbt_standardized"] == summary["daily_metrics"]["formal"]


def test_broker_costs_are_actual_fill_notional_and_price_drift_is_visible():
    prices, weights = _inputs()
    fees = {"A": 0.001, "B": 0.002}
    result = run_dual_engine_validation(prices, weights, fees)
    orders = result["orders"]
    np.testing.assert_allclose(
        orders["commission"], orders["quantity"].abs() * orders["price"] * orders["symbol"].map(fees)
    )
    fixed = orders[orders["sizing"] == "fixed_capital"]
    summary = result["summary"]
    assert summary["backtrader_fixed_cost_return"] == pytest.approx(fixed["commission"].sum() / 1e6)
    assert summary["backtrader_fixed_cost_return"] == pytest.approx(summary["wbt_standardized_cost_return"], abs=1e-10)
    assert summary["gross_execution_reconciled"]


def test_flat_formal_portfolio_has_no_trades_or_fictional_profit():
    prices, weights = _inputs()
    result = run_dual_engine_validation(prices, weights * 0, {"A": 0.001, "B": 0.002})
    assert result["orders"].empty
    summary = result["summary"]
    assert summary["flat_path"]
    assert summary["formal_net_return"] == summary["wbt_standardized_net_return"] == 0.0
    assert summary["backtrader_fixed_net_return"] == summary["backtrader_equity_final_return"] == 0.0


def test_terminal_one_microsecond_close_is_filled_and_shared_exit_fee_not_dropped():
    prices, weights = _inputs()
    prices = prices.iloc[:5, :1].copy()
    weights = weights.iloc[:5, :1].copy()
    weights.iloc[-1] = 0.0
    prices.index = pd.DatetimeIndex([*prices.index[:-1], prices.index[-2] + pd.Timedelta(microseconds=1)])
    weights.index = prices.index
    result = run_dual_engine_validation(prices, weights, {"A": 0.001})
    summary = result["summary"]
    native = result["native_execution_ledger"]["A"]
    held = native["quantity"].iloc[-2]
    assert native["cost_return"].iloc[-1] == pytest.approx(abs(held) * prices["A"].iloc[-1] * 0.001 / 1e6)
    assert summary["wbt_raw_cost_return"] == 0.0
    assert native["cost_return"].iloc[-1] > 0.0
    assert summary["wbt_standardized_formal_net_difference"] == pytest.approx(0.0, abs=1e-10)
    final_orders = result["orders"].query("sizing == 'fixed_capital'")
    assert final_orders.iloc[-1]["dt"] == prices.index[-1]
    assert final_orders.iloc[-1]["position_after"] == 0.0


def test_final_portfolio_budgets_are_summed_not_equal_weighted_again():
    prices, weights = _inputs()
    together = run_dual_engine_validation(prices, weights, {"A": 0.001, "B": 0.002})
    assert together["summary"]["wbt_standardized_net_return"] == pytest.approx(
        together["wbt_assets"]["standardized_net_return"].sum()
    )
    assert together["summary"]["wbt_standardized_formal_net_difference"] == pytest.approx(0.0, abs=1e-10)


def test_native_wbt_intervals_are_aligned_before_calendar_day_aggregation():
    prices, weights = _inputs()
    dates = pd.date_range("2023-01-02 21:00", periods=len(prices), freq="h")
    prices.index = weights.index = dates
    result = run_dual_engine_validation(prices, weights, {"A": 0.001, "B": 0.002})
    daily = result["daily_ledger"]
    assert len(daily) == 2
    np.testing.assert_allclose(daily["wbt_standardized"], daily["formal"], atol=1e-10)
    assert result["summary"]["wbt_standardized_formal_max_bar_error"] < 1e-10
    for engine in ("backtrader_fixed", "backtrader_equity"):
        assert all(quantity == 0 for quantity in result["summary"][f"{engine}_terminal_positions"].values())


def test_intraday_equity_account_returns_use_actual_nav_compounding():
    prices, weights = _inputs()
    dates = pd.date_range("2023-01-02 21:00", periods=len(prices), freq="h")
    prices.index = weights.index = dates
    result = run_dual_engine_validation(prices, weights, {"A": 0.001, "B": 0.002})
    bars = result["bar_ledger"]["backtrader_equity"]
    actual_daily = result["daily_ledger"]["backtrader_equity"]["net_return"]
    compounded_daily = (1.0 + bars["net_return"]).groupby(bars.index.normalize()).prod() - 1.0
    np.testing.assert_allclose(actual_daily, compounded_daily, atol=1e-12)
    summed_daily = bars["net_return"].groupby(bars.index.normalize()).sum()
    assert (actual_daily - summed_daily).abs().max() > 1e-6
    final_return = bars["equity"].iloc[-1] / 1e6 - 1.0
    summary = result["summary"]
    assert summary["equity_nav_metrics"]["final_account_return"] == pytest.approx(final_return)
    assert (1.0 + actual_daily).prod() - 1.0 == pytest.approx(final_return)
    assert "backtrader_equity" not in summary["daily_metrics"]


def test_strided_daily_column_views_are_copied_at_native_metrics_boundary(monkeypatch):
    import czsc

    prices, weights = _inputs()
    native_performance = czsc.daily_performance
    checked = []

    def check_boundary(daily_returns, yearly_days=None):
        assert daily_returns.flags.c_contiguous
        checked.append(len(daily_returns))
        return native_performance(daily_returns, yearly_days=yearly_days)

    monkeypatch.setattr(czsc, "daily_performance", check_boundary)
    result = run_dual_engine_validation(prices, weights, {"A": 0.001, "B": 0.002})
    assert checked == [len(prices)] * 4
    # Pandas returns strided column views from this grouped two-dimensional
    # ledger. The PyO3 boundary must own a contiguous view of each metric input.
    raw_column = result["daily_ledger"]["formal"]["net_return"].to_numpy(dtype=float)
    assert not raw_column.flags.c_contiguous
    expected = native_performance(np.ascontiguousarray(raw_column), yearly_days=252)
    assert result["summary"]["daily_metrics"]["formal"] == expected


@pytest.mark.parametrize("edge", [0, -1])
def test_requires_explicit_flat_start_and_terminal_liquidation(edge):
    prices, weights = _inputs()
    weights.iloc[edge, 0] = 0.1
    with pytest.raises(ValueError, match="flat|terminal"):
        run_dual_engine_validation(prices, weights, {"A": 0.001, "B": 0.002})


def test_never_silently_reorders_or_fills_invalid_execution_inputs():
    prices, weights = _inputs()
    with pytest.raises(ValueError, match="aligned"):
        run_dual_engine_validation(prices, weights.iloc[::-1], {"A": 0.001, "B": 0.002})
    prices.iloc[2, 0] = np.nan
    with pytest.raises(ValueError, match="positive"):
        run_dual_engine_validation(prices, weights, {"A": 0.001, "B": 0.002})


def test_unchanged_target_still_rebalances_quantities_and_pays_actual_costs():
    prices, weights = _inputs()
    prices = prices[["A"]]
    weights = weights[["A"]] * 0.0
    weights.iloc[1:-1] = 0.4
    result = run_dual_engine_validation(prices, weights, {"A": 0.001})
    native = result["native_execution_ledger"]["A"]
    fixed = result["orders"].query("sizing == 'fixed_capital'")
    assert (native["cost_return"].iloc[2:-1] > 0).all()
    assert len(fixed) == len(prices) - 1
    np.testing.assert_allclose(native["quantity"], weights["A"] * 1e6 / prices["A"])
    assert result["summary"]["execution_reconciled"]
    assert result["summary"]["fixed_cost_max_bar_error"] < 1e-10


def test_weight_change_that_preserves_quantity_needs_no_resize_trade():
    prices, weights = _inputs()
    prices = prices.iloc[:4][["A"]]
    weights = weights.iloc[:4][["A"]] * 0.0
    # Mock quotes have cents; these targets preserve quantity and are exactly
    # represented by the installed WBT without the precision mismatch below.
    weights.iloc[1, 0] = prices.iloc[1, 0] / 100
    weights.iloc[2, 0] = prices.iloc[2, 0] / 100
    result = run_dual_engine_validation(prices, weights, {"A": 0.001}, initial_capital=1.0)
    native = result["native_execution_ledger"]["A"]
    assert weights.iloc[2, 0] != weights.iloc[1, 0]
    assert native["trade_quantity"].iloc[2] == pytest.approx(0.0, abs=1e-15)
    assert native["cost_return"].iloc[2] == pytest.approx(0.0, abs=1e-15)
    fixed = result["orders"].query("sizing == 'fixed_capital'")
    assert list(fixed.execution_row) == [1, 3]
    assert result["summary"]["execution_reconciled"]


def test_flip_charges_both_actual_quantity_legs_and_capital_scales_only_cash():
    prices, weights = _inputs()
    small = run_dual_engine_validation(prices, weights, {"A": 0.001, "B": 0.002}, initial_capital=10_000)
    large = run_dual_engine_validation(prices, weights, {"A": 0.001, "B": 0.002}, initial_capital=1_000_000)
    native = small["native_execution_ledger"]["A"]
    assert native["trade_quantity"].iloc[5] == pytest.approx(native["quantity"].iloc[5] - native["quantity"].iloc[4])
    assert abs(native["trade_quantity"].iloc[5]) == pytest.approx(
        abs(native["quantity"].iloc[5]) + abs(native["quantity"].iloc[4])
    )
    for metric in ("gross_return", "cost_return", "net_return"):
        np.testing.assert_allclose(small["bar_ledger"]["formal"][metric], large["bar_ledger"]["formal"][metric])
    np.testing.assert_allclose(
        large["native_execution_ledger"]["A"]["quantity"], small["native_execution_ledger"]["A"]["quantity"] * 100
    )
    assert small["summary"]["execution_reconciled"] and large["summary"]["execution_reconciled"]


@pytest.mark.parametrize("fee", [0.0, 0.00015, 0.00035, 0.00065, 0.00115])
def test_all_friction_levels_reconcile_bar_and_total_gross_cost_net(fee):
    prices, weights = _inputs()
    result = run_dual_engine_validation(prices, weights, {"A": fee, "B": fee})
    summary = result["summary"]
    assert summary["execution_reconciled"]
    assert summary["wbt_raw_cost_return"] == 0.0
    assert summary["wbt_raw_net_return"] == summary["wbt_raw_gross_return"]
    for engine in summary["reconciliation_checks"].values():
        for check in engine.values():
            assert check["max_bar_absolute_error"] <= summary["absolute_bar_tolerance"]
            assert check["total_absolute_error"] <= summary["absolute_total_tolerance"]
            assert check["passed"]


def test_absolute_cost_mismatch_fails_even_when_total_net_return_is_unchanged(monkeypatch):
    from examples.multi_factor_cta import dual_engine_validation as module

    replay = module._backtrader_replay

    def altered_costs(*args):
        ledger, orders, inventory = replay(*args)
        if args[-1] == "fixed_capital":
            ledger.loc[ledger.index[2], "cost_return"] += 2e-9
        return ledger, orders, inventory

    monkeypatch.setattr(module, "_backtrader_replay", altered_costs)
    prices, weights = _inputs()
    summary = run_dual_engine_validation(prices, weights, {"A": 0.001, "B": 0.002})["summary"]
    assert summary["bt_fixed_wbt_standardized_net_difference"] == pytest.approx(0.0, abs=1e-10)
    assert not summary["execution_reconciled"]
    assert not summary["reconciliation_checks"]["backtrader_fixed"]["cost_return"]["passed"]


def test_external_wbt_precision_is_checked_without_rounding_frozen_targets():
    prices, weights = _inputs()
    prices = prices.iloc[:4][["A"]]
    weights = weights.iloc[:4][["A"]] * 0.0
    weights.iloc[1, 0] = 0.4
    weights.iloc[2, 0] = weights.iloc[1, 0] * prices.iloc[2, 0] / prices.iloc[1, 0]
    frozen = weights.copy()
    result = run_dual_engine_validation(prices, weights, {"A": 0.001}, initial_capital=1.0)
    pd.testing.assert_frame_equal(weights, frozen)
    quantities = weights["A"] / prices["A"]
    independent_pnl = quantities.shift(1, fill_value=0.0) * prices["A"].diff().fillna(0.0)
    np.testing.assert_allclose(result["bar_ledger"]["formal"]["gross_return"], independent_pnl, atol=1e-14)
    np.testing.assert_allclose(result["bar_ledger"]["backtrader_fixed"]["gross_return"], independent_pnl, atol=1e-14)
    summary = result["summary"]
    checks = summary["reconciliation_checks"]
    assert all(check["passed"] for check in checks["backtrader_fixed"].values())
    errors = result["bar_ledger"]["wbt_raw"]["gross_return"] - independent_pnl
    precision_passed = bool(
        errors.abs().max() <= summary["absolute_bar_tolerance"]
        and abs(errors.sum()) <= summary["absolute_total_tolerance"]
    )
    assert checks["wbt_raw"]["gross_return"]["passed"] == precision_passed
    assert summary["execution_reconciled"] == precision_passed
