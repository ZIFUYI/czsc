"""Actual structural exits, not hypothetical ATR labels, control formal admission."""

import numpy as np
import pandas as pd
import pytest

import czsc
from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.forensic_pipeline import (
    compute_execution_net_metrics,
    evaluate_execution_events,
    evaluate_triple_barrier,
    run_execution_audit,
)
from examples.alpha_mining_agent.hypothesis_bank import ARCH_A
from examples.multi_factor_cta.forensic_validator import RuleFactorForensicValidator


def _bars(n=50):
    frame = generate_symbol_kines("TEST", "日线", "20220101", "20231231", seed=732).iloc[:n].copy()
    frame["dt"] = pd.to_datetime(frame["dt"])
    price = float(frame["close"].iloc[0])
    frame[["open", "high", "low", "close"]] = price
    frame["atr14"] = price / 100.0
    return frame


@pytest.mark.parametrize("direction", [1, -1])
def test_execution_events_resize_reverse_and_close_match_rust_and_fee_ledger(direction):
    frame = _bars(7)
    weights = np.array([0.5, 0.25, -0.75, -0.25, 0.0, 0.0, 0.0]) * direction
    events = evaluate_execution_events(frame, weights, fee_rate=0.001)
    native = czsc.execution_episodes(
        *(frame[column].to_numpy() for column in ("open", "high", "low", "close", "atr14")), weights, 0.001
    )
    assert len(events) == len(native) == 2
    assert list(events["direction"]) == [direction, -direction]
    np.testing.assert_allclose(events["weighted_net_ret"], [-0.001, -0.0015])
    assert events["exit_dt"].iloc[0] == events["entry_dt"].iloc[1]
    np.testing.assert_allclose(events["net_ret"] * events["quality_w"], events["weighted_net_ret"])
    for index, episode in enumerate(native):
        assert events["net_ret"].iloc[index] == episode.net_ret
    assert sum(czsc.execution_net_returns(frame["open"].to_numpy(), weights, 0.001)) == pytest.approx(
        events["weighted_net_ret"].sum()
    )


def test_execution_fold_does_not_invent_last_bar_or_training_boundary_exit():
    frame = _bars(10)
    weights = np.ones(len(frame))
    weights[4] = 0.0
    cutoff = frame["dt"].iloc[4]
    assert evaluate_execution_events(frame, weights, label_end_dt=cutoff).empty
    completed = evaluate_execution_events(frame, weights, label_end_dt=frame["dt"].iloc[5])
    assert len(completed) == 1
    assert completed["exit_idx"].iloc[0] == 5
    full = evaluate_execution_events(frame, weights)
    assert len(full) == 1  # the later position never closes inside the observed history
    changed = frame.copy()
    changed.loc[changed["dt"] > cutoff, ["open", "high", "low", "close"]] *= 100.0
    assert evaluate_execution_events(changed, weights, label_end_dt=cutoff).empty
    first = compute_execution_net_metrics(frame, weights, fee_rate=0.001, is_end_dt=cutoff)
    later = compute_execution_net_metrics(changed, weights, fee_rate=0.001, is_end_dt=cutoff)
    assert first == later
    assert first["is_net_cum_ret_pct"] == pytest.approx(-0.1)  # only entry fee, not invented exit fee


@pytest.mark.parametrize("direction", [1, -1])
def test_execution_does_not_use_exit_bar_extrema(direction):
    frame = _bars(4)
    frame.loc[frame.index[2], "high"] *= 10.0
    frame.loc[frame.index[2], "low"] *= 0.1
    events = evaluate_execution_events(frame, np.array([1.0, 0.0, 1.0, 0.0]) * direction, fee_rate=0)
    assert len(events) == 1
    assert events["mfe_atr"].iloc[0] == 0.0
    assert events["mae_atr"].iloc[0] == 0.0


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("exit_kind", ["stop", "profit", "both"])
def test_atr_diagnostic_exit_excursions_exclude_unordered_post_exit_extremes(direction, exit_kind):
    frame = _bars(4)
    price, atr = float(frame["open"].iloc[0]), float(frame["atr14"].iloc[0])
    favorable = price + direction * (10 * atr if exit_kind in {"profit", "both"} else 0.75 * atr)
    adverse = price - direction * (10 * atr if exit_kind in {"stop", "both"} else 0.75 * atr)
    frame.loc[frame.index[1], "high"] = max(favorable, adverse)
    frame.loc[frame.index[1], "low"] = min(favorable, adverse)
    events = evaluate_triple_barrier(frame, np.array([0]), direction, 3, 2.0, 1.0, fee_rate=0)
    assert events["exit_idx"].iloc[0] == 1
    if exit_kind == "profit":
        assert events["mfe_atr"].iloc[0] == 2.0
        assert events["mae_atr"].iloc[0] == 0.0
    else:
        assert events["mfe_atr"].iloc[0] == 0.0
        assert events["mae_atr"].iloc[0] == 1.0


def test_refreshes_cannot_manufacture_independent_execution_samples():
    frame = _bars(100)
    weights = np.resize(np.array([0.2, 0.8, 0.4]), len(frame))
    result = run_execution_audit(frame, weights)
    assert result["passed"] is False
    assert result["is_n_purged"] == 0
    weights[-2:] = 0.0
    result = run_execution_audit(frame, weights)
    assert result["is_n_purged"] == 1
    assert result["passed"] is False
    assert result["is_net_cum_ret_pct"] < 0


def test_normalized_formal_events_ignore_triggers_and_do_not_gate_executed_weights_again(monkeypatch):
    frame = _bars(100)
    weights = np.resize(np.array([0.5, 0.0]), len(frame))
    validator = RuleFactorForensicValidator()

    def forbidden(*args, **kwargs):
        pytest.fail("Executed weights must not be transformed again")

    monkeypatch.setattr(validator, "_resolve_soft_gated_inputs", forbidden)
    result = validator.run_vol_and_regime_normalized_audit(
        frame, np.ones(len(frame), dtype=bool), 1, ARCH_A, "1d", weights_tf=weights, event_source="execution"
    )
    expected = evaluate_execution_events(frame, weights)
    assert result["event_source"] == "execution"
    assert result["is_n_purged"] == len(expected)
    np.testing.assert_allclose([event["net_ret"] for event in result["events"]], expected["net_ret"])
    with pytest.raises(ValueError, match="require signed weights"):
        validator.run_vol_and_regime_normalized_audit(
            frame, np.ones(len(frame), dtype=bool), 1, ARCH_A, "1d", event_source="execution"
        )


@pytest.mark.parametrize("execution_passed,diagnostic_passed", [(False, True), (True, False)])
def test_formal_admission_uses_execution_and_keeps_atr_result_diagnostic(
    execution_passed, diagnostic_passed, monkeypatch
):
    from examples.alpha_mining_agent import run_evolution_loop as entry

    frame = _bars()
    monkeypatch.setattr(entry, "compute_research_signal", lambda *args: (np.zeros(len(frame)), np.zeros(len(frame))))
    monkeypatch.setattr(
        entry, "run_gate1_forensic_audit", lambda *args, **kwargs: {"passed": diagnostic_passed, "reason": "diagnostic"}
    )
    monkeypatch.setattr(
        entry, "run_execution_audit", lambda *args, **kwargs: {"passed": execution_passed, "reason": "execution"}
    )
    monkeypatch.setattr(
        entry, "evaluate_candidate_pooled_gate1b", lambda *args, **kwargs: {"passed": True, "reason": "pooled"}
    )
    monkeypatch.setattr(
        entry, "run_parameter_smoothness_audit", lambda *args, **kwargs: {"passed": True, "reason": "smoothness"}
    )
    result = entry.audit_research_candidate("AC01L", "1d", "TEST", {"TEST": {"1d": frame}})
    assert result["passed"] is execution_passed
    assert result["gate1"]["diagnostic_only"] is True
    assert result["failed_gate"] == (None if execution_passed else "execution")


def test_execution_admission_checks_effective_samples_win_ratio_ci_and_net_sharpe(monkeypatch):
    """Controlled event statistics test the gate, not economic profitability of mock prices."""
    from examples.alpha_mining_agent import forensic_pipeline as pipeline

    frame = _bars(130)
    weights = np.zeros(len(frame))
    dates = pd.to_datetime(frame["dt"])

    def audit_with(returns, quality=None, sharpe=1.0):
        count = len(returns)
        q = np.ones(count) if quality is None else np.asarray(quality)
        events = pd.DataFrame(
            {
                "net_ret": returns,
                "quality_w": q,
                "weighted_net_ret": q * returns,
                "atr_pct": np.full(count, 0.01),
                "trigger_dt": dates.iloc[:count].map(pd.Timestamp.isoformat).to_numpy(),
            }
        )
        monkeypatch.setattr(pipeline, "evaluate_execution_events", lambda *args, **kwargs: events.copy())
        monkeypatch.setattr(
            pipeline, "compute_execution_net_metrics", lambda *args, **kwargs: {"is_net_sharpe": sharpe}
        )
        return pipeline.run_execution_audit(frame, weights)

    good = np.resize(np.array([0.003, 0.003, -0.001, 0.003, -0.001]), 100)
    assert audit_with(good)["passed"] is True
    poor_quality = np.full(100, 0.001)
    poor_quality[:10] = 1.0
    assert "有效执行样本" in audit_with(good, poor_quality)["reason"]
    assert "胜率" in audit_with(np.resize(np.array([0.003, -0.001]), 100))["reason"]
    assert "盈亏比" in audit_with(np.resize(np.array([0.001, 0.001, -0.001]), 100))["reason"]
    assert "Sharpe" in audit_with(good, sharpe=0.0)["reason"]
    concentrated_outlier = np.r_[0.10, np.full(17, 0.001), np.full(12, -0.003)]
    assert "置信下界" in audit_with(concentrated_outlier)["reason"]


def test_pooled_formal_route_requests_actual_execution_events(monkeypatch):
    from examples.alpha_mining_agent import run_evolution_loop as entry

    frame = _bars()
    calls = []
    monkeypatch.setattr(entry, "compute_research_signal", lambda *args: (np.zeros(len(frame)), np.zeros(len(frame))))

    def normalized(self, *args, **kwargs):
        calls.append(kwargs)
        return {"is_n_purged": 0, "events": []}

    monkeypatch.setattr(RuleFactorForensicValidator, "run_vol_and_regime_normalized_audit", normalized)
    entry.evaluate_candidate_pooled_gate1b(
        "AC01L", "1d", "TEST", {}, {"TEST": {"1d": frame}}, np.random.default_rng(42)
    )
    assert len(calls) == 1
    assert calls[0]["event_source"] == "execution"
    assert calls[0]["soft_gate"] is False


def test_only_execution_reversals_may_share_an_exit_and_entry_timestamp():
    from examples.alpha_mining_agent.hypothesis_bank import ARCH_B

    frame = _bars(100)
    weights = np.resize(np.array([1.0, -1.0]), len(frame))
    validator = RuleFactorForensicValidator(soft_gate=False)
    events = validator.run_vol_and_regime_normalized_audit(
        frame, np.ones(len(frame), dtype=bool), 1, ARCH_B, "1d", weights_tf=weights, event_source="execution"
    )
    execution = validator.run_stratified_pooled_gate1b_audit(archetype=ARCH_B, asset_audits={"ETF": events})
    diagnostic = validator.run_stratified_pooled_gate1b_audit(
        archetype=ARCH_B, asset_audits={"ETF": {**events, "event_source": "diagnostic"}}
    )
    assert "events持有期重叠" not in execution["reason"]
    assert "events持有期重叠" in diagnostic["reason"]
