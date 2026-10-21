"""Admission contracts exercised through the actual CTA study entry point."""

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.forensic_pipeline import run_execution_audit
from examples.alpha_mining_agent.hypothesis_bank import ARCH_B
from examples.multi_factor_cta import forensic_validator as validator
from examples.multi_factor_cta.study_execution import execution_sharpe
from examples.results.intraday_cta_ablation_20261017.run_intraday_ablation import evaluate_candidate_on_fold


def _frame():
    frame = generate_symbol_kines("TEST", "日线", "20200101", "20240601", seed=19)
    frame["dt"] = pd.to_datetime(frame["dt"])
    frame["atr14"] = (frame["high"] - frame["low"]).rolling(14, min_periods=1).mean()
    return frame


def _score():
    frame = _frame()
    weights = np.resize(np.array([1.0, 1.0, 0.0, 0.0]), len(frame))
    cutoff = pd.Timestamp("2023-12-31 23:59:59")
    result = evaluate_candidate_on_fold(frame, weights, "1d", 0.0002, cutoff)
    return {
        "fold": "F1",
        "asset": "ETF",
        "uid": "BC01L@1d",
        "tf": "1d",
        "archetype": ARCH_B,
        "pool_group": "equity",
        "microstructure_group": "equity_reference",
        **result,
    }


def test_actual_study_uses_formal_execution_admission_and_exports_complete_evidence():
    frame = _frame()
    weights = np.resize(np.array([1.0, 1.0, 0.0, 0.0]), len(frame))
    cutoff = pd.Timestamp("2023-12-31 23:59:59")
    result = evaluate_candidate_on_fold(frame, weights, "1d", 0.0002, cutoff)
    formal = run_execution_audit(frame, weights, fee_rate=0.0002, is_end_dt=cutoff, min_net_sharpe=-float("inf"))
    mask = frame.dt <= cutoff
    sharpe = execution_sharpe(frame.loc[mask].reset_index(drop=True), weights[mask], 0.0002)
    assert result["strict_local_pass"] == (formal["passed"] and sharpe > 0)
    assert result["is_net_sharpe"] == sharpe
    assert result["mean_ci_low_pct"] == formal["is_mean_ci_low_pct"]
    assert result["effective_n"] == formal["is_effective_n"]
    assert len(result["events"]) == formal["is_n_purged"]
    assert all(pd.Timestamp(event["exit_dt"]) <= cutoff for event in result["events"])
    assert all("regime" in event and "r_atr" in event for event in result["events"])


def test_empty_training_history_is_rejected_with_zero_sharpe():
    frame = _frame()
    result = evaluate_candidate_on_fold(
        frame, np.zeros(len(frame)), "1d", 0.0002, frame.dt.min() - pd.Timedelta(days=1)
    )
    assert result["n"] == 0
    assert result["is_net_sharpe"] == 0
    assert result["ann_turnover"] == 0
    assert result["events"] == []
    assert not result["strict_local_pass"]


def test_actual_study_pool_ignores_summary_scores_and_asset_replication():
    score = _score()
    one = validator.summarize_execution_pooled_scores(pd.DataFrame([score])).iloc[0]
    copies = pd.DataFrame([{**score, "asset": f"ETF{i}", "t_stat": 1e20} for i in range(20)])
    many = validator.summarize_execution_pooled_scores(copies).iloc[0]
    assert many["pooled_t"] == pytest.approx(one["pooled_t"])
    assert many["independent_time_blocks"] == one["independent_time_blocks"]
    assert many["pool_evidence_kind"] == "execution_time_block_hac"


def test_old_summary_csv_cannot_be_admitted_without_event_evidence():
    score = _score()
    score.pop("events")
    score.update(
        n=500,
        effective_n=500,
        strict_local_pass=True,
        mean_net_pct=10.0,
        mean_ci_low_pct=5.0,
        win_rate=80.0,
        pl_ratio=3.0,
        safety_margin=50.0,
        is_net_sharpe=10.0,
    )
    scores = pd.DataFrame([score])
    pooled = validator.summarize_execution_pooled_scores(scores)
    assert not pooled.iloc[0]["pool_passed_assets"]
    assert validator.select_pooled_multitf_gate1_units(scores, pooled, "ETF").empty


def test_actual_study_evidence_ignores_future_prices_and_decisions():
    frame = _frame()
    weights = np.resize(np.array([1.0, 1.0, 0.0, 0.0]), len(frame))
    cutoff = pd.Timestamp("2023-12-31 23:59:59")
    first = evaluate_candidate_on_fold(frame, weights, "1d", 0.0002, cutoff)
    future = frame["dt"] > cutoff
    frame.loc[future, ["open", "high", "low", "close"]] *= 100
    weights[future] = -0.5
    second = evaluate_candidate_on_fold(frame, weights, "1d", 0.0002, cutoff)
    np.testing.assert_array_equal(first.pop("event_returns"), second.pop("event_returns"))
    assert first == second


def test_empty_evidence_pool_has_a_usable_schema():
    pooled = validator.summarize_execution_pooled_scores(pd.DataFrame())
    assert pooled.empty
    assert {"fold", "uid", "pool_passed_assets", "pool_audits", "pooled_t"}.issubset(pooled.columns)


@pytest.mark.parametrize(
    "field,value",
    [
        ("n", 29),
        ("effective_n", 29.99),
        ("win_rate", 50.0),
        ("pl_ratio", 1.5),
        ("safety_margin", 10.0),
        ("mean_ci_low_pct", 0.0),
        ("is_net_sharpe", 0.0),
        ("strict_local_pass", False),
    ],
)
def test_every_local_sop_requirement_applies_to_5m_without_low_turnover_exceptions(field, value):
    score = _score()
    score.update(
        uid="BC01L@5m",
        tf="5m",
        n=100,
        effective_n=100.0,
        win_rate=70.0,
        pl_ratio=2.0,
        safety_margin=30.0,
        mean_ci_low_pct=0.1,
        is_net_sharpe=1.0,
        strict_local_pass=True,
    )
    # This fixture isolates the selector's local contract from the separately tested pooled audit.
    pooled = pd.DataFrame(
        [
            {
                "fold": "F1",
                "uid": "BC01L@5m",
                "total_n": 500,
                "pooled_mean_pct": 0.2,
                "pooled_t": 3.0,
                "pos_asset_ratio": 1.0,
                "pool_evidence_kind": "execution_time_block_hac",
                "pool_passed_assets": ["ETF"],
                "pool_audits": {
                    "ETF": {
                        "passed": True,
                        "pooled_t_stat": 3.0,
                        "universal_ratio": 1.0,
                    }
                },
            }
        ]
    )
    assert len(validator.select_pooled_multitf_gate1_units(pd.DataFrame([score]), pooled, "ETF")) == 1
    score[field] = value
    assert validator.select_pooled_multitf_gate1_units(
        pd.DataFrame([score]), pooled, "ETF", allow_lowturn_hetero_15m=True
    ).empty


def test_legacy_pooled_effective_sample_flag_cannot_relax_per_asset_support():
    score = _score()
    events = score["events"][:5]
    for event in events:
        event["net_ret"] = abs(event["net_ret"]) + 0.01
        event["r_atr"] = event["net_ret"] / event["atr_pct"]
    audits = {f"ETF{i}": {"events": events, "event_source": "execution", "is_net_sharpe": 1.0} for i in range(20)}
    result = validator.RuleFactorForensicValidator().run_stratified_pooled_gate1b_audit(
        archetype=ARCH_B, timeframe="1d", asset_audits=audits, allow_pooled_effective_n=True
    )
    assert result["passed"] is False
    assert "有效独立样本" in result["reason"]


@pytest.mark.parametrize("mode", ["Gate1_Strict", "Gate2_Local", "Gate3_PooledSynergy"])
def test_original_intraday_entry_does_not_bypass_formal_admission(mode):
    from examples.results.intraday_cta_ablation_20261017.run_intraday_ablation import select_factors_by_gate

    score = _score()
    score.update(uid="BC01L@30m", tf="30m", n=15, is_net_sharpe=10, mean_net_pct=10)
    pooled = validator.summarize_execution_pooled_scores(pd.DataFrame([score]))
    assert select_factors_by_gate(pd.DataFrame([score]), pooled, "ETF", mode) == []
