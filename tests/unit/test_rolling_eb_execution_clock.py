"""Rolling allocation learns only already observed next-open executions."""

import numpy as np
import pytest

from czsc.mock import generate_symbol_kines
from examples.multi_factor_cta.forensic_validator import RollingEmpiricalBayesWeighter


def _bars():
    return generate_symbol_kines("TEST", "日线", "20230101", "20230401", seed=42).iloc[:40].copy()


def test_eb_does_not_earn_the_gap_before_its_next_open_entry():
    frame = _bars().iloc[:4].copy()
    base = float(frame["open"].iloc[0])
    frame["open"] = base * np.array([1.0, 1.2, 1.32, 1.32])
    frame["close"] = base
    decisions = np.array([1.0, 1.0, 0.0, 0.0])
    result = RollingEmpiricalBayesWeighter.build_causal_unit_net_returns(frame, {"unit": decisions}, fee_rate=0.001)
    # Enter after the 20% gap. The unchanged unit target requires selling the price
    # appreciation before the eventual exit; initial capital remains exactly one.
    entry_quantity = 1.0 / frame["open"].iloc[1]
    resized_quantity = 1.0 / frame["open"].iloc[2]
    gross = entry_quantity * (frame["open"].iloc[2] - frame["open"].iloc[1])
    rebalance_fee = abs(resized_quantity - entry_quantity) * frame["open"].iloc[2] * 0.001
    np.testing.assert_allclose(result["unit"], [0.0, -0.001, gross - rebalance_fee, -0.001], atol=1e-12)
    assert rebalance_fee == pytest.approx(0.0001)


def test_eb_charges_turnover_on_fill_and_never_liquidates_a_history_prefix():
    frame = _bars().iloc[:6].copy()
    frame["open"] = float(frame["open"].iloc[0])
    decisions = np.array([1.0, -1.0, 0.5, 0.5, 0.5, -1.0])
    full = RollingEmpiricalBayesWeighter.build_causal_unit_net_returns(frame, {"unit": decisions}, fee_rate=0.001)
    np.testing.assert_allclose(full["unit"], [0.0, -0.001, -0.002, -0.0015, 0.0, 0.0])
    prefix = RollingEmpiricalBayesWeighter.build_causal_unit_net_returns(
        frame.iloc[:4], {"unit": decisions[:4]}, fee_rate=0.001
    )
    np.testing.assert_array_equal(prefix["unit"], full["unit"][:4])


def test_eb_requires_actual_open_prices():
    frame = _bars().drop(columns="open")
    with pytest.raises(ValueError, match="open"):
        RollingEmpiricalBayesWeighter.build_causal_unit_net_returns(frame, {"unit": np.ones(len(frame))})


def test_eb_weights_use_only_returns_before_decision_time():
    frame = _bars()
    decisions = np.resize(np.array([1.0, 0.5, -0.5, -1.0]), len(frame))
    weighter = RollingEmpiricalBayesWeighter(min_bars=5, lookback_bars=12)
    kwargs = {"cross_asset_prior": {"unit": 0.4}, "unit_weights_on_grid": {"unit": decisions}}
    full = weighter.compute_weights(df_grid=frame, **kwargs)
    cutoff = 20
    mutated = frame.copy()
    mutated.loc[mutated.index[cutoff] :, "open"] *= 10
    changed = weighter.compute_weights(df_grid=mutated, **kwargs)
    np.testing.assert_allclose(changed["unit"][: cutoff + 1], full["unit"][: cutoff + 1], atol=1e-12)
    prefix = weighter.compute_weights(
        df_grid=frame.iloc[:cutoff],
        cross_asset_prior={"unit": 0.4},
        unit_weights_on_grid={"unit": decisions[:cutoff]},
    )
    np.testing.assert_allclose(prefix["unit"], full["unit"][:cutoff], atol=1e-12)
