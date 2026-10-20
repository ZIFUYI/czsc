"""Statistical contracts only; controlled event outcomes are not trading performance evidence."""

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.forensic_pipeline import evaluate_triple_barrier
from examples.alpha_mining_agent.hypothesis_bank import ARCH_B
from examples.multi_factor_cta.forensic_validator import RuleFactorForensicValidator


def _bars():
    df = generate_symbol_kines("TEST", "日线", "20200101", "20231231", seed=19)
    df["dt"] = pd.to_datetime(df["dt"])
    return df


def test_pooled_time_blocks_test_the_regime_balanced_mean():
    dt = _bars()["dt"].iloc[:120].reset_index(drop=True)
    rng = np.random.default_rng(123)
    regimes = np.repeat([-1, 0, 1], [30, 45, 45])
    rng.shuffle(regimes)
    returns = np.where(regimes == -1, -1.99, 1.0) + rng.normal(0, 0.5, len(dt))
    # Make the target three-state mean nearly zero, while the more frequent positive states
    # leave a large misleading unweighted mean. Retain dispersion inside every state.
    returns[regimes == -1] += 0.003 - sum(returns[regimes == state].mean() for state in (-1, 0, 1))
    balanced_mean = np.mean([returns[regimes == state].mean() for state in (-1, 0, 1)])
    assert balanced_mean == pytest.approx(0.001)
    assert returns.mean() > 0.25
    events = pd.DataFrame(
        {
            "trigger_dt": dt - pd.Timedelta(minutes=1),
            "entry_dt": dt,
            "exit_dt": dt,
            "regime": regimes,
            "r_atr": returns,
            "net_ret": returns * 0.01,
            "quality_w": 1.0,
        }
    )
    audit = {"events": events.to_dict("records"), "is_net_sharpe": 1.0}
    validator = RuleFactorForensicValidator()
    result = validator.run_stratified_pooled_gate1b_audit(
        archetype=ARCH_B, asset_audits={f"ETF{i}": audit for i in range(5)}
    )
    assert result["mean_ipw_r_atr"] == pytest.approx(balanced_mean)
    assert abs(result["pooled_t_stat"]) < 0.1
    assert result["passed"] is False
    repeated = validator.run_stratified_pooled_gate1b_audit(
        archetype=ARCH_B, asset_audits={f"ETF{i}": audit for i in range(10)}
    )
    assert repeated["pooled_t_stat"] == result["pooled_t_stat"]


@pytest.mark.parametrize("direction", [1, -1])
def test_gap_through_stop_fills_at_available_open(direction):
    bars = _bars().iloc[:4].copy()
    entry_price = float(bars["close"].iloc[0])
    bars[["open", "high", "low", "close"]] = entry_price
    bars["atr14"] = entry_price * 0.01
    gap_open = entry_price * (1 - direction * 0.2)
    bars.loc[bars.index[2], ["open", "high", "low", "close"]] = gap_open
    # An intrabar rebound happens after the opening stop fill and cannot contribute MFE.
    favorable_extremum = "high" if direction == 1 else "low"
    bars.loc[bars.index[2], favorable_extremum] = entry_price * (1 + direction * 0.2)
    events = evaluate_triple_barrier(bars, np.array([0]), direction, 3, 100, 1, fee_rate=0.001)
    assert events["mfe_atr"].iloc[0] == 0.0
    assert events["exit_idx"].tolist() == [2]
    assert events["net_ret"].iloc[0] == pytest.approx(-0.2 - (1.0 + gap_open / entry_price) * 0.001)


@pytest.mark.parametrize("direction", [1, -1])
def test_regular_intrabar_stop_keeps_its_stop_price(direction):
    bars = _bars().iloc[:4].copy()
    entry_price = float(bars["close"].iloc[0])
    bars[["open", "high", "low", "close"]] = entry_price
    bars["atr14"] = entry_price * 0.01
    extremum = "low" if direction == 1 else "high"
    bars.loc[bars.index[2], extremum] = entry_price * (1 - direction * 0.2)
    events = evaluate_triple_barrier(bars, np.array([0]), direction, 3, 100, 1, fee_rate=0.001)
    stop_price = entry_price * (1 - direction * 0.01)
    assert events["net_ret"].iloc[0] == pytest.approx(-0.01 - (1.0 + stop_price / entry_price) * 0.001)
