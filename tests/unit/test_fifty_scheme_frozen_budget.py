"""Every comparison scheme reserves risk budget for untradable held positions."""

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.multi_factor_cta.test_all_weather_50commodities_portfolio import evaluate_50_weighting_schemes


def _inputs():
    mock = generate_symbol_kines("SHFE.RB9999", "日线", "20230101", "20230401", seed=992)
    index = pd.to_datetime(mock["dt"])
    returns = pd.DataFrame({s: mock["close"].to_numpy() * 0.0 for s in "ABCD"}, index=index)
    positions = pd.DataFrame(1.0, index=index, columns=list("ABCD"))
    positions["A"] = -1.0
    return returns, positions


def _evaluate(returns, positions):
    return evaluate_50_weighting_schemes(
        returns,
        {s: s for s in "ABCD"},
        asset_positions=positions,
        asset_fee_rates=dict.fromkeys("ABCD", 0.001),
        periods_per_year=252,
    )


def test_all_ten_schemes_reserve_the_budget_of_a_missing_quote():
    returns, positions = _inputs()
    returns.iloc[5:12, 0] = np.nan
    results = _evaluate(returns, positions)
    assert len(results) == 10
    for key, result in results.items():
        weights = result["weights_df"]
        # All schemes have the same unit budget during this fixed-prior warmup.
        assert weights.iloc[5:12].abs().sum(axis=1).max() <= 1.0 + 1e-12, key
        np.testing.assert_allclose(weights.iloc[5:12, 0], weights.iloc[4, 0])
        assert (result["ledger"]["cost_returns_df"].iloc[5:12, 0] == 0.0).all()
        gross = weights.abs().sum(axis=1)
        assert (gross <= result["target_gross_budget"] + result["untradable_budget_excess"] + 1e-12).all()


def test_all_missing_quotes_report_frozen_excess_without_fictitious_liquidation():
    returns, positions = _inputs()
    returns.iloc[5:8] = np.nan
    for key, result in _evaluate(returns, positions).items():
        weights = result["weights_df"]
        np.testing.assert_allclose(weights.iloc[5:8], np.tile(weights.iloc[4].to_numpy(), (3, 1)))
        assert (result["target_gross_budget"].iloc[5:8] == 0.0).all(), key
        assert (result["untradable_budget_excess"].iloc[5:8] > 0.0).all(), key
        assert (result["ledger"]["cost_returns_df"].iloc[5:8] == 0.0).all().all()


def test_all_schemes_charge_gap_rebalance_only_when_quote_resumes():
    returns, positions = _inputs()
    returns.iloc[5:8, 0] = np.nan
    returns.iloc[8, 0] = 0.2
    for key, result in _evaluate(returns, positions).items():
        weights = result["weights_df"]
        ledger = result["ledger"]
        frozen_quantity = weights.iloc[4, 0]  # Last price normalized to one.
        resume_price = 1.2
        resumed_quantity = weights.iloc[8, 0] / resume_price
        traded_notional = abs(resumed_quantity - frozen_quantity) * resume_price
        assert ledger["cost_returns_df"].iloc[5:8, 0].eq(0.0).all(), key
        assert ledger["turnover_df"].iloc[8, 0] == pytest.approx(traded_notional), key
        assert ledger["cost_returns_df"].iloc[8, 0] == pytest.approx(traded_notional * 0.001), key
        assert traded_notional > 0.0, key


def test_risk_budget_reference_keeps_frozen_quantity_through_quote_gap(monkeypatch):
    from examples.multi_factor_cta import test_all_weather_50commodities_portfolio as portfolio

    returns, positions = _inputs()
    returns.iloc[5:8, 0] = np.nan
    returns.iloc[8, 0] = 0.2
    references = []
    account = portfolio.account_portfolio_execution

    def capture_reference(price_returns, targets, fees, **kwargs):
        ledger = account(price_returns, targets, fees, **kwargs)
        if all(fee == 0.0 for fee in fees.values()):
            references.append((targets.copy(), ledger))
        return ledger

    monkeypatch.setattr(portfolio, "account_portfolio_execution", capture_reference)
    _evaluate(returns, positions)
    assert len(references) == 3
    for targets, ledger in references:
        # No liquidation at absent prices, and the resumed quote marks the actual
        # held short quantity. All references deliberately exclude friction.
        np.testing.assert_array_equal(targets.iloc[5:8, 0], np.repeat(targets.iloc[4, 0], 3))
        held_quantity = targets.iloc[4, 0]  # Last valid price normalized to one.
        assert ledger["gross_returns_df"].iloc[8, 0] == pytest.approx(held_quantity * (1.2 - 1.0))
        assert ledger["gross_returns_df"].iloc[8, 0] != 0.0
        assert ledger["cost_returns_df"].to_numpy().sum() == 0.0
