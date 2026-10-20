"""Barrier event costs use actual entry and exit quantities, not alpha evidence."""

import numpy as np
import pandas as pd
import pytest

import czsc
from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.forensic_pipeline import evaluate_triple_barrier


def _bars():
    frame = generate_symbol_kines("TEST", "日线", "20230101", "20230201", seed=23).iloc[:4].copy()
    frame["dt"] = pd.to_datetime(frame["dt"])
    entry_price = float(frame["close"].iloc[0])
    frame[["open", "high", "low", "close"]] = entry_price
    frame["atr14"] = entry_price * 0.01
    return frame, entry_price


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("fee", [0.0, 0.0015, 0.0035, 0.0065, 0.0115])
def test_timeout_cost_is_actual_entry_and_exit_notional(direction, fee):
    frame, entry = _bars()
    exit_price = entry * 1.2
    frame.loc[frame.index[2], ["open", "high", "low", "close"]] = exit_price
    events = evaluate_triple_barrier(frame, np.array([0]), direction, 2, 100, 100, fee_rate=fee)
    quantity = direction / entry
    expected = quantity * (exit_price - entry) - abs(quantity) * (entry + exit_price) * fee
    assert events["exit_idx"].tolist() == [2]
    assert events["net_ret"].iloc[0] == pytest.approx(expected)
    native = czsc.execution_ledger([entry, exit_price], [float(direction), 0.0], fee)
    assert events["net_ret"].iloc[0] == pytest.approx(sum(native.net_returns))


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("profitable", [True, False])
def test_gap_barrier_charges_executable_open_and_keeps_quality_separate(direction, profitable):
    frame, entry = _bars()
    price_move = direction * (0.2 if profitable else -0.2)
    exit_price = entry * (1.0 + price_move)
    frame.loc[frame.index[2], ["open", "high", "low", "close"]] = exit_price
    weights = np.full(len(frame), direction * 0.25)
    events = evaluate_triple_barrier(frame, np.array([0]), direction, 3, 1, 1, fee_rate=0.001, weights_tf=weights)
    gross = direction * price_move
    expected = gross - (1.0 + exit_price / entry) * 0.001
    assert events["exit_idx"].tolist() == [2]
    assert events["net_ret"].iloc[0] == pytest.approx(expected)
    assert events["quality_w"].iloc[0] == pytest.approx(0.25)
