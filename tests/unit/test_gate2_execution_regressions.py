"""Gate 2 rescue evidence uses executed next-open exposure and complete transaction costs."""

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.forensic_pipeline import run_gate2_3d_complementarity_audit
from examples.alpha_mining_agent.hypothesis_bank import ARCH_A, ARCHETYPE_RULERS
from examples.multi_factor_cta.forensic_validator import RuleFactorForensicValidator


def _bars(n=50):
    df = generate_symbol_kines("TEST", "日线", "20231001", "20240430", seed=77).iloc[:n].copy()
    df["dt"] = pd.to_datetime(df["dt"])
    price = float(df["close"].iloc[0])
    df[["open", "high", "low", "close"]] = price
    df.attrs["fee_rate"] = 0.001
    return df


def _audit(api, df, candidate, portfolio):
    if api == "pipeline":
        return run_gate2_3d_complementarity_audit(df, candidate, {}, portfolio, ARCH_A, 0, ARCHETYPE_RULERS[ARCH_A])
    return RuleFactorForensicValidator().run_gate2_complementarity_and_jaccard_audit(df, candidate, {}, portfolio)


@pytest.mark.parametrize("api", ["pipeline", "validator"])
def test_gate2_open_gap_rescue_accounts_entry_and_exit_costs(api):
    df = _bars()
    candidate = np.zeros(len(df))
    for idx in range(2, len(df) - 2, 5):
        candidate[idx] = 1
        df.loc[df.index[idx + 2], ["open", "high"]] *= 1.1
    result = _audit(api, df, candidate, np.zeros(len(df)))
    assert result["passed"] is True, result
    # Enter at P, sell the same quantity at 1.1P: fee is 0.001 + 0.0011.
    assert result["dd_rescue_alpha_bp"] == pytest.approx(489.5, abs=0.01)
    # Both sides of each round trip matter; a sufficiently large configured fee removes the edge.
    df.attrs["fee_rate"] = 0.06
    expensive = _audit(api, df, candidate, np.zeros(len(df)))
    assert expensive["dd_rescue_alpha_bp"] == pytest.approx(-130.0, abs=0.01)
    assert expensive["passed"] is False


@pytest.mark.parametrize("api", ["pipeline", "validator"])
def test_gate2_flat_market_reports_cost_losses_including_liquidations(api):
    df = _bars()
    candidate = (np.arange(len(df)) % 5 == 2).astype(float)
    result = _audit(api, df, candidate, np.zeros(len(df)))
    assert result["dd_rescue_alpha_bp"] == pytest.approx(-10.0, abs=0.01)
    assert result["passed"] is False


@pytest.mark.parametrize("api", ["pipeline", "validator"])
def test_gate2_drawdown_state_comes_from_executed_open_price_path(api):
    df = _bars()
    base = float(df["open"].iloc[0])
    df["open"] = base * (1 - np.arange(len(df)) * 0.005)
    df["low"] = df["open"]
    candidate = np.zeros(len(df))
    candidate[10:-5:4] = -1.0
    result = _audit(api, df, candidate, np.full(len(df), 0.5))
    assert result["dd_rescue_alpha_bp"] > 10.0, result


@pytest.mark.parametrize("api", ["pipeline", "validator"])
def test_gate2_open_clock_never_reads_prices_beyond_training_boundary(api):
    df = _bars(100)
    candidate = np.resize(np.array([0.0, 0.5, 1.0]), len(df))
    portfolio = np.resize(np.array([0.0, 0.25, 0.5]), len(df))
    first = _audit(api, df, candidate, portfolio)
    changed = df.copy()
    changed.loc[changed["dt"] > pd.Timestamp("2023-12-31"), ["open", "high", "low", "close"]] *= 100
    assert _audit(api, changed, candidate, portfolio) == first


@pytest.mark.parametrize("api", ["pipeline", "validator"])
def test_gate2_charges_terminal_inventory_liquidation(api):
    df = _bars()
    candidate = np.full(len(df), 0.5)
    candidate[0] = 0
    result = _audit(api, df, candidate, np.zeros(len(df)))
    expected_bp = -(2 * 0.5 * df.attrs["fee_rate"]) / (len(df) - 2) * 10000
    assert result["dd_rescue_alpha_bp"] == pytest.approx(round(expected_bp, 2))
