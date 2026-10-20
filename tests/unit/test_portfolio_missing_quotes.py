"""A halted asset keeps its inventory and cannot finance new positions."""

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.multi_factor_cta.synthesis_engine import CrossAssetHRPSynthesizer


def _panels():
    bars = generate_symbol_kines("TEST", "日线", "20230101", "20230201", seed=71).iloc[:8]
    returns = pd.DataFrame(0.0, index=pd.DatetimeIndex(bars.dt), columns=list("ABCD"))
    return returns, returns + 1.0


def _allocate(returns, positions):
    return CrossAssetHRPSynthesizer().build_multi_asset_portfolio(
        returns,
        {symbol: symbol for symbol in returns},
        asset_positions_df=positions,
        asset_fee_rates=dict.fromkeys(returns, 0.001),
        returns_kind="price",
        positions_timing="close",
    )


def test_missing_quote_reserves_frozen_inventory_budget():
    returns, positions = _panels()
    returns.iloc[5:, 3] = np.nan
    positions.iloc[5:, 3] = np.nan
    result = _allocate(returns, positions)
    weights = result["asset_weights_df"]
    assert weights["D"].iloc[5] == pytest.approx(weights["D"].iloc[4])
    assert result["frozen_gross_exposure"].iloc[5] == pytest.approx(0.25)
    assert weights.abs().sum(axis=1).iloc[5] == pytest.approx(1.0)
    assert (weights.abs().sum(axis=1) <= result["target_gross_budget"] + 1e-12).all()
    assert result["equal_weight_positions_df"].abs().sum(axis=1).max() <= 1.0 + 1e-12
    assert result["turnover_df"]["D"].iloc[5:].eq(0).all()


def test_frozen_inventory_over_budget_blocks_new_risk_without_fake_liquidation():
    returns, positions = _panels()
    returns.iloc[5:, 1:] = np.nan
    positions.iloc[5:, 1:] = np.nan
    result = _allocate(returns, positions)
    weights = result["asset_weights_df"]
    assert weights["A"].iloc[5] == 0
    np.testing.assert_allclose(weights.iloc[5, 1:], weights.iloc[4, 1:])
    assert result["untradable_budget_excess"].iloc[5] == pytest.approx(0.75 - 1 / 3)
    assert result["turnover_df"].iloc[5, 1:].eq(0).all()
