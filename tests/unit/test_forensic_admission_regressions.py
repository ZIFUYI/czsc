"""Regression contracts for causal, evidence-backed research admission."""

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.forensic_pipeline import (
    compute_single_factor_net_sharpe_and_turnover,
    evaluate_triple_barrier,
    run_gate1_forensic_audit,
    run_parameter_smoothness_audit,
)
from examples.alpha_mining_agent.hypothesis_bank import ARCH_A, ARCH_B, ARCHETYPE_RULERS
from examples.multi_factor_cta.forensic_validator import RuleFactorForensicValidator


def _bars(start="20230101", end="20240601"):
    frame = generate_symbol_kines("TEST", "日线", start, end, seed=19)
    frame["dt"] = pd.to_datetime(frame["dt"])
    frame["atr14"] = (frame["high"] - frame["low"]).rolling(14, min_periods=1).mean()
    return frame


def test_empty_training_never_reuses_oos():
    df = _bars("20240101", "20240601")
    trig = np.ones(len(df), dtype=bool)
    weights = np.ones(len(df))
    gate = run_gate1_forensic_audit(df, trig, 1, ARCHETYPE_RULERS[ARCH_A], "1d", weights_tf=weights)
    assert gate["passed"] is False
    assert gate["is_n_purged"] == 0
    normalized = RuleFactorForensicValidator(soft_gate=False).run_vol_and_regime_normalized_audit(
        df, trig, 1, ARCH_A, "1d", weights_tf=weights
    )
    assert normalized["is_n_purged"] == 0
    assert compute_single_factor_net_sharpe_and_turnover(df, weights)["is_net_cum_ret_pct"] == 0


def test_summary_only_pool_cannot_manufacture_significance():
    audits = {
        f"ETF{i}": {"is_n_purged": 100, "norm_r_atr": 0.08, "std_r_atr": 1, "raw_mean_pct": 0.08, "ipw_r_atr": 0.08}
        for i in range(5)
    }
    result = RuleFactorForensicValidator().run_stratified_pooled_gate1b_audit(archetype=ARCH_B, asset_audits=audits)
    assert result["passed"] is False
    assert result["pooled_t_stat"] == 0
    assert "events" in result["reason"]


def test_homogeneous_pool_never_substitutes_other_assets():
    audits = {"ETF": {"is_n_purged": 100, "norm_r_atr": -0.2, "raw_mean_pct": -0.2}}
    audits.update({f"SHFE.RB{i}": {"is_n_purged": 100, "norm_r_atr": 1, "raw_mean_pct": 1} for i in range(4)})
    result = RuleFactorForensicValidator().run_stratified_pooled_gate1b_audit(
        archetype=ARCH_B, asset_audits=audits, target_pool_group="equity", allow_single_asset_degrade=True
    )
    assert result["passed"] is False
    assert result["pool_size"] == 1
    assert result["evaluated_assets"] == ["ETF"]


def test_round_trip_pays_two_sides():
    df = _bars().iloc[:10].copy()
    price = float(df["close"].iloc[0])
    df[["open", "high", "low", "close"]] = price
    result = evaluate_triple_barrier(df, np.array([0]), 1, 2, 10, 10, fee_rate=0.001)
    assert result["net_ret"].iloc[0] == pytest.approx(-0.002)


def test_parameter_smoothness_rejects_cv_between_old_and_required_limit():
    result = run_parameter_smoothness_audit(neighbor_metrics=[0.5, 0.7, 1.0, 1.3, 1.5])
    assert 0.35 < result["cv"] < 0.4
    assert result["passed"] is False


def test_parameter_smoothness_applies_gate_once_even_with_attached_gate():
    from examples.alpha_mining_agent.hypothesis_bank import enrich_kline_features
    from examples.multi_factor_cta.atomic_factors import compute_atomic_signal, set_factor_context

    frame = _bars("20220101", "20231231").rename(columns={"vol": "volume"})
    frame["date"] = frame["dt"].dt.date
    frame = enrich_kline_features(frame)
    frame["cs_slope_rank"] = np.linspace(0.1, 0.9, len(frame))
    _, weights = compute_atomic_signal(frame, "CC02L", soft_gate=False)
    assert np.any(weights != 0.0)
    baseline = run_parameter_smoothness_audit(frame, "CC02L", "1d", soft_gate=True)
    set_factor_context(frame, soft_gate=True)
    attached = run_parameter_smoothness_audit(frame, "CC02L", "1d", soft_gate=True)
    assert attached["grid_metrics"] == baseline["grid_metrics"]


def test_training_labels_and_statistics_ignore_changed_future():
    df = _bars()
    boundary = pd.Timestamp("2023-12-31 23:59:59")
    idx = int((df["dt"] <= boundary).sum()) - 3
    # Force the timeout, which would end outside the training fold.
    df["atr14"] = df["close"] * 100
    first = evaluate_triple_barrier(df, np.array([idx]), 1, 8, 2, 1, label_end_dt=boundary)
    assert first.empty
    changed = df.copy()
    changed.loc[changed["dt"] > boundary, "high"] *= 1000
    second = evaluate_triple_barrier(changed, np.array([idx]), 1, 8, 2, 1, label_end_dt=boundary)
    pd.testing.assert_frame_equal(first, second)
    w = np.ones(len(df))
    assert compute_single_factor_net_sharpe_and_turnover(df, w) == compute_single_factor_net_sharpe_and_turnover(
        changed, w
    )


def test_purge_uses_actual_exit_and_quality_never_reads_next_bar():
    df = _bars().iloc[:12].copy()
    df["atr14"] = (df["high"] - df["low"]) * 0.00001
    indices = np.arange(8)
    result = evaluate_triple_barrier(df, indices, 1, 8, 1, 1, purge_overlaps=True)
    assert len(result) == len(indices)
    assert (result["entry_idx"].to_numpy()[1:] > result["exit_idx"].to_numpy()[:-1]).all()
    zero_at_decision = np.ones(len(df))
    zero_at_decision[0] = 0
    empty = evaluate_triple_barrier(df, np.array([0]), 1, 8, 1, 1, weights_tf=zero_at_decision)
    assert empty.empty


def test_null_bootstrap_uses_the_same_quality_weighted_statistic():
    from examples.alpha_mining_agent.forensic_pipeline import _weighted_bootstrap_evidence

    df = _bars().iloc[:30]
    returns = np.full(len(df), -0.01)
    returns[0] = 0.01
    quality = np.full(len(df), 0.0001)
    quality[0] = 1
    null_returns = np.resize(np.array([-0.01, 0.01]), 100)
    p_value, ci_low = _weighted_bootstrap_evidence(returns, quality, null_returns, np.random.default_rng(81))
    assert p_value > 0.25
    assert ci_low < 0


def _real_event_audit():
    df = _bars("20200101", "20231231")
    indices = np.arange(20, len(df) - 8, 8)
    events = evaluate_triple_barrier(df, indices, 1, 3, 2, 1, purge_overlaps=True)
    events["r_atr"] = events["net_ret"] / events["atr_pct"]
    # Regime is categorical metadata for this statistical-contract fixture.
    events["regime"] = np.resize(np.array([-1, 0, 1]), len(events))
    return {"events": events.to_dict("records"), "is_n_purged": len(events), "is_net_sharpe": 0.1}


def test_replicating_an_asset_does_not_create_independent_evidence():
    audit = _real_event_audit()
    validator = RuleFactorForensicValidator()
    single = validator.run_stratified_pooled_gate1b_audit(archetype=ARCH_B, asset_audits={"ETF": audit})
    copied = validator.run_stratified_pooled_gate1b_audit(
        archetype=ARCH_B, asset_audits={f"ETF{i}": audit for i in range(10)}
    )
    assert copied["independent_time_blocks"] == single["independent_time_blocks"]
    assert copied["pooled_t_stat"] == single["pooled_t_stat"]
    assert single["passed"] is False
    assert single["required_pool_size"] == 5


def test_display_summaries_cannot_change_event_statistics():
    audit = _real_event_audit()
    validator = RuleFactorForensicValidator()
    result = validator.run_stratified_pooled_gate1b_audit(archetype=ARCH_B, asset_audits={"ETF": audit})
    altered = {**audit, "norm_r_atr": 1e20, "ipw_r_atr": 1e20, "raw_mean_pct": 1e20, "std_r_atr": 0.000001}
    changed = validator.run_stratified_pooled_gate1b_audit(archetype=ARCH_B, asset_audits={"ETF": altered})
    for key in ("pooled_t_stat", "positive_assets", "universal_ratio"):
        assert result[key] == changed[key]


def test_parameter_audit_rejects_mean_return_as_sharpe():
    with pytest.raises(ValueError, match="is_net_sharpe"):
        run_parameter_smoothness_audit(eval_fn=lambda _: {"is_mean_pct": 10.0})


def test_constant_price_single_factor_pays_entry_and_liquidation():
    df = _bars().iloc[:20].copy()
    df[["open", "high", "low", "close"]] = float(df["close"].iloc[0])
    result = compute_single_factor_net_sharpe_and_turnover(df, np.ones(len(df)), fee_rate=0.001)
    assert result["is_net_cum_ret_pct"] == pytest.approx(-0.2)


def test_complete_positive_event_evidence_can_pass_admission():
    # Controlled statistical contract only: these edited outcomes are not economic performance evidence.
    audit = _real_event_audit()
    for event in audit["events"]:
        event["net_ret"] = abs(event["net_ret"]) + 0.002
        event["r_atr"] = event["net_ret"] / event["atr_pct"]
    result = RuleFactorForensicValidator().run_stratified_pooled_gate1b_audit(
        archetype=ARCH_B, asset_audits={f"ETF{i}": audit for i in range(5)}
    )
    assert result["passed"] is True, result["reason"]
    assert result["pool_size"] == 5
    assert result["pooled_t_stat"] > 1.96
    assert result["universal_ratio"] == 1


def test_native_cache_is_invalidated_by_same_length_price_or_symbol_edit(monkeypatch, tmp_path):
    import czsc
    from examples.alpha_mining_agent.hypothesis_bank import attach_native_czsc_signals, enrich_kline_features

    df = _bars().iloc[:40].rename(columns={"vol": "volume"})
    df["date"] = df["dt"].dt.date
    df = enrich_kline_features(df)
    calls = []

    def generate(bars, configs, **kwargs):
        calls.append(len(bars))
        return pd.DataFrame({"dt": [bar.dt for bar in bars], "日线_contract": "其他_任意_任意_0"})

    monkeypatch.setattr(czsc, "generate_czsc_signals", generate)
    attach_native_czsc_signals(df, "1d", tmp_path)
    attach_native_czsc_signals(df, "1d", tmp_path)
    assert len(calls) == 1
    changed = df.copy()
    changed.loc[changed.index[-1], "close"] *= 1.01
    attach_native_czsc_signals(changed, "1d", tmp_path)
    changed["symbol"] = "ANOTHER"
    attach_native_czsc_signals(changed, "1d", tmp_path)
    assert len(calls) == 3


def test_backtest_slice_handles_empty_fold_and_charges_terminal_liquidation():
    from examples.alpha_mining_agent.forensic_pipeline import backtest_slice

    df = _bars().iloc[:20].copy()
    empty = backtest_slice(df.iloc[:0], np.array([]))
    assert empty["cum_ret_pct"] == 0
    assert list(empty["daily_return"].columns) == ["date", "total"]
    df[["open", "high", "low", "close"]] = float(df["close"].iloc[0])
    df.attrs["fee_rate"] = 0.001
    result = backtest_slice(df, np.ones(len(df)))
    assert result["cum_ret_pct"] == pytest.approx(-0.2)


def test_gate2_never_reads_first_oos_return():
    from examples.alpha_mining_agent.forensic_pipeline import run_gate2_3d_complementarity_audit

    df = _bars()
    weights = np.resize(np.array([0.0, 0.5, 1.0]), len(df))
    portfolio = np.zeros(len(df))
    args = (weights, {}, portfolio, ARCH_A, 0, ARCHETYPE_RULERS[ARCH_A])
    result = run_gate2_3d_complementarity_audit(df, *args)
    changed = df.copy()
    changed.loc[changed["dt"] > pd.Timestamp("2023-12-31"), "close"] *= 100
    assert result == run_gate2_3d_complementarity_audit(changed, *args)


def test_single_asset_report_reuses_next_open_ledger_and_charges_liquidation():
    from examples.alpha_mining_agent.forensic_pipeline import backtest_slice
    from examples.multi_factor_cta.synthesis_engine import evaluate_single_asset_backtest

    df = _bars().iloc[:20].copy()
    df[["open", "high", "low", "close"]] = float(df["close"].iloc[0])
    weights = np.ones(len(df))
    weights[-1] = 0
    df.attrs["fee_rate"] = 0.001
    report = evaluate_single_asset_backtest(df, weights, "TEST", fee_rate=0.001, is_end_dt=df.dt.iloc[10])
    expected = backtest_slice(df, weights)
    pd.testing.assert_series_equal(report["daily_return"], expected["daily_return"].set_index("date")["total"])
    assert report["full"]["cum_ret_pct"] == pytest.approx(-0.2)
    assert report["is"]["cum_ret_pct"] == pytest.approx(-0.2)
    assert report["oos"]["cum_ret_pct"] == pytest.approx(-0.2)
    assert report["execution_clock"] == "next_open"


def test_select_pooled_multitf_gate1_units_unifies_pooled_and_single_degrees_of_freedom():
    from examples.multi_factor_cta.forensic_validator import select_pooled_multitf_gate1_units

    fold_scores = pd.DataFrame(
        [
            {
                "fold": "F1",
                "asset": "rb",
                "uid": "AC01L@1d",
                "tf": "1d",
                "n": 5,
                "is_net_sharpe": 0.45,
                "mean_net_pct": 0.15,
                "ann_turnover": 12.0,
            },
            {
                "fold": "F1",
                "asset": "rb",
                "uid": "BC01L@15m",
                "tf": "15m",
                "n": 8,
                "is_net_sharpe": 0.45,
                "mean_net_pct": 0.15,
                "ann_turnover": 60.0,
            },
        ]
    )
    fold_pooled = pd.DataFrame(
        [
            {
                "fold": "F1",
                "uid": "AC01L@1d",
                "pos_asset_ratio": 0.75,
                "pooled_mean_pct": 0.12,
                "pooled_t": 2.40,
                "total_n": 180,
            },
            {
                "fold": "F1",
                "uid": "BC01L@15m",
                "pos_asset_ratio": 0.40,
                "pooled_mean_pct": 0.05,
                "pooled_t": 0.50,
                "total_n": 25,
            },
        ]
    )
    selected = select_pooled_multitf_gate1_units(fold_scores, fold_pooled, "rb", allow_lowturn_hetero_15m=False)
    # AC01L@1d passes via pooled effective degrees of freedom (total_n=180 >= 25 and n=5 >= 2);
    # BC01L@15m fails single-asset event count threshold (n=8 < 15).
    assert list(selected["uid"]) == ["AC01L@1d"]


def test_roll_mask_strips_fake_roll_gap_and_charges_two_sided_rollover_cost():
    from examples.multi_factor_cta.synthesis_engine import (
        account_portfolio_execution,
        build_asset_execution_inputs,
        detect_contract_roll_mask,
    )

    df = _bars().iloc[:6].copy().reset_index(drop=True)
    p0 = float(df["open"].iloc[0])
    df[["open", "high", "low", "close"]] = p0
    # Simulate a +12% non-trading contract rollover gap at bar 3 onwards
    df.loc[3:, ["open", "high", "low", "close"]] = p0 * 1.12
    df["underlying_symbol"] = ["RB2501", "RB2501", "RB2501", "RB2505", "RB2505", "RB2505"]
    roll_mask = detect_contract_roll_mask(df, has_night_session=False)
    assert roll_mask.tolist() == [False, False, False, True, False, False]

    # Hold +0.5 across the roll bar (bar 3), then liquidate
    signed_targets = np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
    ret_s, pos_s = build_asset_execution_inputs(df, signed_targets, roll_mask=roll_mask)
    assert float(ret_s.iloc[3]) == 0.0
    ledger = account_portfolio_execution(ret_s.to_frame("RB"), pos_s.to_frame("RB"), {"RB": 0.001})
    # Zero fake roll-gap gross PnL across the entire path
    assert float(ledger["gross_returns_df"]["RB"].sum()) == pytest.approx(0.0, abs=1e-15)
    # Entry at bar 1 (0.5), rollover at bar 3 (2 * 0.5 = 1.0), exit at bar 5 (0.5) => total turnover 2.0
    np.testing.assert_allclose(
        ledger["turnover_df"]["RB"].to_numpy(),
        [0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0],
        atol=1e-14,
    )
    assert float(ledger["cost_returns_df"]["RB"].iloc[3]) == pytest.approx(2.0 * 0.5 * 0.001)
    assert float(ledger["bar_net_returns"].sum()) == pytest.approx(-2.0 * 0.001)

    # Also verify 2.0x fee_multiplier stress scales rollover and trade costs linearly
    ledger_2x = account_portfolio_execution(
        ret_s.to_frame("RB"), pos_s.to_frame("RB"), {"RB": 0.001}, fee_multiplier=2.0
    )
    assert float(ledger_2x["bar_net_returns"].sum()) == pytest.approx(-4.0 * 0.001)
