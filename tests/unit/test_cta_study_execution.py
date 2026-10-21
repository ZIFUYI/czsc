"""The study selector must use the same executable path as its final portfolio."""

import sys
from pathlib import Path

import numpy as np
import pytest

from czsc.mock import generate_symbol_kines
from examples.multi_factor_cta.synthesis_engine import (
    account_portfolio_execution,
    assign_futures_trading_day,
    build_asset_execution_inputs,
)


def _study():
    directory = Path(__file__).resolve().parents[2] / "examples/results/intraday_cta_ablation_20261017"
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    import run_multitf_incremental_study

    return run_multitf_incremental_study


def test_selector_sharpe_matches_next_open_ledger_on_real_observation_days():
    frame = generate_symbol_kines("AUDIT", "15分钟", "20240102", "20240301", seed=73)
    weights = np.sin(np.arange(len(frame)) / 9).round(1)
    returns, positions = build_asset_execution_inputs(frame, weights)
    ledger = account_portfolio_execution(returns.to_frame("A"), positions.to_frame("A"), {"A": 0.0002})
    daily = ledger["bar_net_returns"].groupby(assign_futures_trading_day(ledger["bar_net_returns"].index)).sum()
    expected = daily.mean() / daily.std(ddof=1) * np.sqrt(252)
    assert _study().eval_is_net_sharpe(weights, frame, 0.0002) == pytest.approx(expected)


def test_selector_rejects_returns_without_execution_prices_and_timestamps():
    frame = generate_symbol_kines("AUDIT", "15分钟", "20240102", "20240301", seed=73)
    returns = frame.close.pct_change().fillna(0).to_numpy()
    with pytest.raises((TypeError, ValueError), match="execution|OHLC|DataFrame"):
        _study().eval_is_net_sharpe(np.ones(len(frame)), returns, 0.0002)


def test_empty_execution_grid_and_cash_have_zero_sharpe():
    frame = generate_symbol_kines("AUDIT", "日线", "20240102", "20240201", seed=74)
    assert _study().eval_is_net_sharpe(np.zeros(len(frame)), frame, 0.0002) == 0
    assert _study().eval_is_net_sharpe(np.zeros(0), frame.iloc[:0], 0.0002) == 0


@pytest.mark.parametrize("uid", ["CC01L@60m", "H60M_EC03L@60m", "H1D_SP01L@1d"])
def test_core_and_stage2_candidates_cannot_bypass_negative_executable_margin(monkeypatch, uid):
    import pandas as pd

    from examples.multi_factor_cta.data_engine import AssetClass, AssetSpec

    study = _study()
    frame = generate_symbol_kines("AUDIT", "15分钟", "20240102", "20240301", seed=73)
    candidate = pd.DataFrame(
        [
            {
                "uid": uid,
                "tf": uid.split("@")[1],
                "archetype": "test",
                "n": 40,
                "is_net_sharpe": 2,
                "ann_turnover": 5,
                "pooled_t": 3,
            }
        ]
    )
    monkeypatch.setattr(study, "select_exp_b_alltf_gate1_adaptive", lambda *a: candidate)
    monkeypatch.setattr(study, "synthesize_fast_slow_synergy", lambda df, units, names, spec: units[names[0]])
    monkeypatch.setattr(study, "eval_is_net_sharpe", lambda weights, df, fee: -1.0 if np.any(weights) else 0.0)
    spec = AssetSpec(
        symbol="AUDIT",
        name="audit",
        asset_class=AssetClass.COMMODITY_FUTURES,
        sector="test",
        base_tf="15m",
        eligible_tfs=("15m",),
        fee_rate=0.0002,
    )
    accepted, rejected, admitted = study.run_incremental_funnel_gate123(
        candidate,
        pd.DataFrame(),
        "audit",
        "fold",
        {uid: np.ones(len(frame))},
        np.ones(len(frame), dtype=bool),
        np.zeros(len(frame)),
        0.0002,
        df_is=frame,
        spec=spec,
    )
    assert accepted == admitted == []
    assert rejected[0]["reject_stage"] == "Gate3"


def test_funnel_all_does_not_double_count_5m(monkeypatch):
    import pandas as pd

    study = _study()
    monkeypatch.setattr(study, "CORE_20_IDS", ["audit"])
    monkeypatch.setattr(study, "FOLDS", [{"id": "fold"}])
    passed = pd.DataFrame([{"asset": "audit", "fold": "fold", "uid": "micro@5m", "tf": "5m"}])
    monkeypatch.setattr(study, "select_exp_b_alltf_gate1_adaptive", lambda *a, **kw: passed)
    monkeypatch.setattr(study, "select_5m_gate1_adaptive", lambda *a: passed)
    empty = pd.DataFrame(columns=["tf", "reject_stage"])
    rows = study.collect_funnel_table("test", passed, passed, empty, empty, False, ("5m",))
    assert rows[-1]["pass_gate1"] == rows[0]["pass_gate1"] == 1


def test_micro_gate_scores_actual_combination_instead_of_linear_proxy(monkeypatch):
    import pandas as pd

    study = _study()
    frame = generate_symbol_kines("AUDIT", "15分钟", "20240102", "20240301", seed=73)
    base, micro = "CC01L@60m", "H5M_TESTL@5m"
    candidate = pd.DataFrame(
        [
            {
                "uid": micro,
                "archetype": "test",
                "n": 50,
                "is_net_sharpe": 2,
                "ann_turnover": 4,
                "pooled_t": 3,
                "pos_asset_ratio": 1,
            }
        ]
    )
    monkeypatch.setattr(study, "select_5m_gate1_adaptive", lambda *a: candidate)
    monkeypatch.setattr(
        study,
        "synthesize_fast_slow_synergy",
        lambda df, units, names, spec: np.full(len(df), 0.5 if names == [base] else -0.5),
    )
    monkeypatch.setattr(study, "eval_is_net_sharpe", lambda weights, df, fee: float(np.mean(weights) * 2))
    monkeypatch.setattr(study, "compute_gap_fill_and_dd_rescue", lambda *a: (50, 1))
    accepted, rejected, admitted = study.run_tier3_5m_incremental_funnel(
        candidate,
        "audit",
        "fold",
        [base],
        frame,
        {base: np.full(len(frame), 0.5), micro: np.full(len(frame), 0.5)},
        np.zeros(len(frame)),
        0.0002,
        object(),
    )
    assert accepted == [base]
    assert admitted == []
    assert rejected[0]["reject_stage"].startswith("Gate3")
