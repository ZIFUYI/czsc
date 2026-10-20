"""Causal allocation and execution accounting; mock prices are not alpha evidence."""

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.hypothesis_bank import ARCH_A, ARCH_B
from examples.multi_factor_cta.synthesis_engine import (
    CrossAssetHRPSynthesizer,
    IntraAssetThreePillarSynthesizer,
    gram_schmidt_cluster_orthogonalize,
)


def _prices():
    df = generate_symbol_kines("SHFE.RB9999", "日线", "20230101", "20231231", seed=809)
    return df.set_index(pd.to_datetime(df["dt"]))["close"]


def _inputs():
    price = _prices()
    returns = pd.DataFrame({"A": price.pct_change().fillna(0.0), "B": price.pct_change().fillna(0.0) * 0.7})
    positions = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
    return returns, positions


def _quantity_reference(prices, targets, fee, tradable=None):
    """Independent order book: monetary PnL and fees divided by fixed starting cash."""
    capital = 250_000.0
    quantities = np.zeros(prices.shape[1])
    previous_prices = prices.iloc[0].to_numpy()
    gross, costs, turnover = [], [], []
    for row, (_, prices_at_mark) in enumerate(prices.iterrows()):
        current_prices = prices_at_mark.to_numpy()
        gross.append(quantities * (current_prices - previous_prices) / capital)
        target_quantities = targets.iloc[row].to_numpy() * capital / current_prices
        if tradable is not None:
            target_quantities = np.where(tradable.iloc[row], target_quantities, quantities)
        order_notionals = np.abs(target_quantities - quantities) * current_prices
        turnover.append(order_notionals / capital)
        costs.append(order_notionals * fee / capital)
        quantities = target_quantities
        previous_prices = current_prices
    return tuple(
        pd.DataFrame(values, index=prices.index, columns=prices.columns) for values in (gross, costs, turnover)
    )


def _build(returns, positions, **kwargs):
    return CrossAssetHRPSynthesizer(min_effective_assets=1, max_single_asset_weight=1.0).build_multi_asset_portfolio(
        returns,
        {"A": "X", "B": "Y"},
        asset_positions_df=positions,
        asset_fee_rates={"A": 0.001, "B": 0.001},
        returns_kind="price",
        positions_timing="close",
        **kwargs,
    )


def test_default_orthogonalization_is_prefix_invariant():
    p = _prices()
    a = np.tanh(p.pct_change().fillna(0.0).to_numpy() * 100)
    b = np.tanh(p.pct_change(3).fillna(0.0).to_numpy() * 100)
    cut = len(p) // 2
    full = gram_schmidt_cluster_orthogonalize({ARCH_A: a, ARCH_B: b})
    prefix = gram_schmidt_cluster_orthogonalize({ARCH_A: a[:cut], ARCH_B: b[:cut]})
    np.testing.assert_allclose(full[ARCH_B][:cut], prefix[ARCH_B], atol=1e-12)


def test_turbulence_warmup_does_not_read_future():
    p = _prices()
    frame = pd.DataFrame({"close": p.to_numpy()})
    full = IntraAssetThreePillarSynthesizer._get_causal_turbulence_cap(frame, "30m")
    prefix = IntraAssetThreePillarSynthesizer._get_causal_turbulence_cap(frame.iloc[:4].copy(), "30m")
    np.testing.assert_allclose(full[:4], prefix, atol=1e-12)


def test_returns_without_positions_and_clock_are_rejected():
    returns, _ = _inputs()
    with pytest.raises(ValueError, match="price|positions|仓位"):
        CrossAssetHRPSynthesizer().build_multi_asset_portfolio(returns, {"A": "X", "B": "Y"})


def test_allocator_all_history_including_warmup_is_prefix_invariant():
    returns, positions = _inputs()
    full = _build(returns, positions)
    for cut in (3, 12, 80):
        prefix = _build(returns.iloc[:cut], positions.iloc[:cut])
        pd.testing.assert_frame_equal(full["asset_weights_df"].iloc[:cut], prefix["asset_weights_df"])
        pd.testing.assert_series_equal(full["bar_net_returns"].iloc[:cut], prefix["bar_net_returns"])


def test_entry_exit_reversal_and_budget_changes_charge_final_weight_turnover():
    returns, positions = _inputs()
    returns *= 0.0
    positions.iloc[3:6, 1] = 0.0  # budget rotation changes A even though its signal is unchanged
    positions.iloc[6:9, 0] = -1.0
    positions.iloc[10:] = 0.0
    result = _build(returns, positions)
    w = result["asset_weights_df"]
    expected = w.diff().fillna(w).abs() * 0.001
    pd.testing.assert_frame_equal(result["cost_returns_df"], expected)
    np.testing.assert_allclose(result["bar_net_returns"], -expected.sum(axis=1))
    assert result["cost_returns_df"].iloc[10].sum() > 0.0
    assert result["cost_returns_df"].iloc[3, 0] > 0.0
    assert w.iloc[6, 0] < 0.0


def test_close_target_earns_only_next_price_move():
    returns, positions = _inputs()
    positions.iloc[:5] = 0.0
    result = _build(returns, positions)
    expected = result["asset_weights_df"].shift(1).fillna(0.0) * returns
    pd.testing.assert_frame_equal(result["gross_returns_df"], expected)
    assert result["gross_returns_df"].iloc[5].sum() == 0.0


def test_final_deadband_cannot_keep_stale_positions_on_exit_or_reversal():
    returns, positions = _inputs()
    positions.iloc[4:8, 0] = -0.1
    positions.iloc[8:, 0] = 0.0
    result = _build(returns, positions, rebalance_deadzone=0.9)
    w = result["asset_weights_df"]
    assert w.iloc[4, 0] < 0.0
    assert (w.iloc[8:, 0] == 0.0).all()


def test_daily_returns_sum_the_bar_ledger_without_second_fee_charge():
    returns, positions = _inputs()
    result = _build(returns, positions)
    expected = (result["gross_returns_df"] - result["cost_returns_df"]).sum(axis=1)
    expected = expected.groupby(expected.index.normalize()).sum().rename("net_return")
    pd.testing.assert_series_equal(result["daily_returns"], expected)


def test_future_tactical_cluster_activation_cannot_change_early_positions():
    price = _prices()
    grid = pd.DataFrame({"dt": price.index, "close": price.to_numpy()})
    trend = np.tanh(price.pct_change().fillna(0.0).to_numpy() * 100)
    tactical = trend.copy()
    cut = len(grid) // 2
    tactical[:cut] = 0.0
    synth = IntraAssetThreePillarSynthesizer()
    full, _ = synth.synthesize_cluster_orthogonals(grid, cluster_signals={ARCH_A: trend, ARCH_B: tactical})
    prefix, _ = synth.synthesize_cluster_orthogonals(
        grid.iloc[:cut],
        cluster_signals={ARCH_A: trend[:cut], ARCH_B: tactical[:cut]},
    )
    np.testing.assert_allclose(full[:cut], prefix, atol=1e-12)


def test_actual_weightbacktest_agrees_on_completed_path_gross():
    from czsc import WeightBacktest
    from examples.multi_factor_cta.synthesis_engine import account_portfolio_execution

    price = _prices().iloc[:40]
    target = np.tanh(price.pct_change().fillna(0.0).to_numpy() * 10).round(6)
    target[-2:] = 0.0
    frame = pd.DataFrame({"dt": price.index, "symbol": "A", "price": price.to_numpy(), "weight": target})
    wb = WeightBacktest(frame, fee_rate=0.0, weight_type="ts", digits=6)
    ledger = account_portfolio_execution(
        price.pct_change().fillna(0.0).to_frame("A"),
        pd.DataFrame({"A": target}, index=price.index),
        {"A": 0.0},
    )
    # WBT labels the holding interval by its start; this ledger books realized
    # price PnL at its end. Raw WBT's legacy weight-delta fee is deliberately disabled:
    # only gross is comparable before shared Rust quantity costs are applied.
    assert ledger["daily_returns"].sum() == pytest.approx(wb.daily_return["total"].sum(), abs=len(price) * 1e-6)


def test_formal_entry_reaudits_champions_and_never_backfills_selected_is(monkeypatch):
    from examples.multi_factor_cta import run_framework_demo as entry
    from examples.multi_factor_cta.data_engine import AssetClass, AssetSpec, ExecutionMode

    price = _prices()
    frame = generate_symbol_kines("SHFE.RB9999", "日线", "20230101", "20231231", seed=809)
    spec = AssetSpec(
        symbol="A",
        name="A",
        asset_class=AssetClass.COMMODITY_FUTURES,
        sector="X",
        base_tf="60m",
        eligible_tfs=("60m",),
        fee_rate=0.001,
        execution_mode=ExecutionMode.LONG_SHORT,
    )
    calls = []

    def audit(sid, tf, symbol, pool, *, is_end_dt):
        calls.append((sid, tf, symbol, is_end_dt))
        return {"passed": sid == "AC01L", "reason": "regression fixture"}

    monkeypatch.setattr(entry, "audit_research_candidate", audit)
    monkeypatch.setattr(entry, "compute_research_signal", lambda df, sid, tf: (np.ones(len(df)), np.ones(len(df))))
    cutoff = price.index[len(price) // 2]
    target, _, admitted, audits = entry.synthesize_admitted_asset(
        spec,
        {"A": {"60m": frame}},
        ["AC01L@60m", "BC01L@60m"],
        IntraAssetThreePillarSynthesizer(),
        is_end_dt=cutoff,
    )
    assert len(calls) == 2
    assert set(admitted) == {"AC01L@60m"}
    assert not audits["BC01L@60m"]["passed"]
    assert (target[price.index <= cutoff] == 0.0).all()
    assert (target[price.index > cutoff] > 0.0).any()


def test_formal_entry_stays_flat_if_all_admission_checks_fail(monkeypatch):
    from examples.multi_factor_cta import run_framework_demo as entry
    from examples.multi_factor_cta.data_engine import AssetClass, AssetSpec, ExecutionMode

    frame = generate_symbol_kines("SHFE.RB9999", "日线", "20230101", "20231231", seed=809)
    spec = AssetSpec(
        symbol="A",
        name="A",
        asset_class=AssetClass.COMMODITY_FUTURES,
        sector="X",
        base_tf="60m",
        eligible_tfs=("60m",),
        fee_rate=0.001,
        execution_mode=ExecutionMode.LONG_SHORT,
    )
    monkeypatch.setattr(
        entry, "audit_research_candidate", lambda *a, **k: {"passed": False, "reason": "insufficient pool"}
    )
    target, weights, priors, audits = entry.synthesize_admitted_asset(
        spec,
        {"A": {"60m": frame}},
        ["AC01L@60m"],
        IntraAssetThreePillarSynthesizer(),
    )
    assert not weights and not priors
    assert (target == 0.0).all()
    assert audits["AC01L@60m"]["reason"] == "insufficient pool"


def test_fixed_execution_example_charges_entry_reversal_and_exit():
    from examples.multi_factor_cta.synthesis_engine import account_portfolio_execution

    price = _prices().iloc[:4]
    # Freeze mock prices to isolate the exact execution book: 0 -> +1 -> -1 -> 0.
    returns = (price * 0.0).to_frame("A")
    weights = pd.DataFrame({"A": [0.0, 1.0, -1.0, 0.0]}, index=price.index)
    ledger = account_portfolio_execution(returns, weights, {"A": 0.001})
    np.testing.assert_allclose(ledger["cost_returns_df"]["A"], [0.0, 0.001, 0.002, 0.001])
    assert ledger["daily_returns"].sum() == pytest.approx(-0.004)


@pytest.mark.parametrize("exposure_scale", [1.0, -1.0, 4.0, -4.0])
def test_unchanged_target_rebalances_actual_quantity_against_fixed_initial_capital(exposure_scale):
    from examples.multi_factor_cta.synthesis_engine import account_portfolio_execution

    price = _prices().iloc[:6].copy()
    # Shape a deterministic stress case from mock bars, with both drift and resizing.
    price.iloc[:] = price.iloc[0] * np.array([1.0, 1.1, 1.21, 0.99, 1.05, 0.94])
    prices = price.to_frame("A")
    targets = pd.DataFrame({"A": exposure_scale * np.array([0.0, 0.4, 0.4, 0.2, -0.3, 0.0])}, index=price.index)
    result = account_portfolio_execution(prices.pct_change().fillna(0.0), targets, {"A": 0.003})
    gross, costs, turnover = _quantity_reference(prices, targets, 0.003)
    np.testing.assert_allclose(result["gross_returns_df"], gross, atol=1e-14)
    np.testing.assert_allclose(result["cost_returns_df"], costs, atol=1e-14)
    np.testing.assert_allclose(result["turnover_df"], turnover, atol=1e-14)
    # The second nonzero mark restores the same target after a 10% price move.
    assert result["turnover_df"].iloc[2, 0] == pytest.approx(0.04 * abs(exposure_scale))
    assert result["cost_returns_df"].iloc[2, 0] == pytest.approx(0.00012 * abs(exposure_scale))
    np.testing.assert_allclose(result["executed_weights_df"], targets, atol=1e-14)


def test_missing_quotes_freeze_quantity_then_charge_price_drift_on_resumption():
    from examples.multi_factor_cta.synthesis_engine import account_portfolio_execution

    price = _prices().iloc[:6].copy()
    price.iloc[:] = price.iloc[0] * np.array([1.0, 1.1, 1.1, 1.1, 1.32, 1.25])
    prices = price.to_frame("A")
    targets = pd.DataFrame({"A": [0.0, 0.4, 0.4, 0.4, 0.4, 0.0]}, index=price.index)
    tradable = pd.DataFrame({"A": [True, True, False, False, True, True]}, index=price.index)
    result = account_portfolio_execution(prices.pct_change().fillna(0.0), targets, {"A": 0.003}, tradable_mask=tradable)
    gross, costs, turnover = _quantity_reference(prices, targets, 0.003, tradable)
    np.testing.assert_allclose(result["gross_returns_df"], gross, atol=1e-14)
    np.testing.assert_allclose(result["cost_returns_df"], costs, atol=1e-14)
    np.testing.assert_allclose(result["turnover_df"], turnover, atol=1e-14)
    np.testing.assert_array_equal(result["executed_weights_df"].iloc[2:4, 0], [0.4, 0.4])
    assert result["turnover_df"].iloc[4, 0] == pytest.approx(0.08)
    assert result["pretrade_weights_df"].iloc[4, 0] == pytest.approx(0.48)


def test_missing_quote_rejects_an_unexecutable_target_change():
    from examples.multi_factor_cta.synthesis_engine import account_portfolio_execution

    prices = _prices().iloc[:4].to_frame("A")
    returns = prices.pct_change().fillna(0.0)
    returns.iloc[2, 0] = 0.0
    targets = pd.DataFrame({"A": [0.0, 0.4, 0.0, 0.0]}, index=prices.index)
    tradable = pd.DataFrame({"A": [True, True, False, True]}, index=prices.index)
    with pytest.raises(ValueError, match="tradable|untradable|weight|target"):
        account_portfolio_execution(returns, targets, {"A": 0.001}, tradable_mask=tradable)


def test_next_open_execution_does_not_capture_pre_entry_gap_and_closes_at_end():
    from examples.multi_factor_cta.synthesis_engine import account_portfolio_execution, build_asset_execution_inputs

    frame = generate_symbol_kines("SHFE.RB9999", "日线", "20230101", "20230110", seed=801).iloc[:4].copy()
    # Engineering fixture: all movement is before the first executable entry.
    frame[["open", "high", "low", "close"]] = 200.0
    frame.loc[frame.index[0], ["open", "high", "low", "close"]] = 100.0
    returns, targets = build_asset_execution_inputs(frame, np.ones(len(frame)))
    assert len(returns) == len(frame) + 1
    np.testing.assert_array_equal(targets, [0.0, 1.0, 1.0, 1.0, 0.0])
    ledger = account_portfolio_execution(returns.to_frame("A"), targets.to_frame("A"), {"A": 0.001})
    assert ledger["gross_returns_df"].to_numpy().sum() == pytest.approx(0.0)
    assert ledger["bar_net_returns"].sum() == pytest.approx(-0.002)


def test_next_open_ledger_matches_shared_backtest_on_mock_prices():
    from examples.alpha_mining_agent.forensic_pipeline import backtest_slice
    from examples.multi_factor_cta.synthesis_engine import account_portfolio_execution, build_asset_execution_inputs

    frame = generate_symbol_kines("SHFE.RB9999", "30分钟", "20230102", "20230113", seed=817)
    targets = np.resize(np.array([0.0, 0.5, 1.0, -0.5, 0.0]), len(frame))
    frame.attrs["fee_rate"] = 0.001
    returns, executed = build_asset_execution_inputs(frame, targets)
    ledger = account_portfolio_execution(returns.to_frame("A"), executed.to_frame("A"), {"A": 0.001})
    shared = backtest_slice(frame, targets)
    # Both paths use the shared Rust fixed-capital quantity ledger.
    assert ledger["daily_returns"].sum() == pytest.approx(shared["daily_return"]["total"].sum(), abs=len(frame) * 1e-6)


def test_asset_listed_in_future_does_not_change_existing_budget_history():
    returns, positions = _inputs()
    baseline = _build(returns, positions)
    cut = 100
    extended_r, extended_p = returns.copy(), positions.copy()
    extended_r["C"] = returns["A"]
    extended_p["C"] = 1.0
    extended_r.loc[extended_r.index[:cut], "C"] = np.nan
    extended_p.loc[extended_p.index[:cut], "C"] = np.nan
    extended = CrossAssetHRPSynthesizer(
        min_effective_assets=1, max_single_asset_weight=1.0
    ).build_multi_asset_portfolio(
        extended_r,
        {"A": "X", "B": "Y", "C": "X"},
        asset_positions_df=extended_p,
        asset_fee_rates={"A": 0.001, "B": 0.001, "C": 0.001},
        returns_kind="price",
        positions_timing="close",
    )
    pd.testing.assert_frame_equal(
        baseline["asset_weights_df"].iloc[:cut],
        extended["asset_weights_df"].iloc[:cut][["A", "B"]],
    )


def test_risk_reductions_take_precedence_over_deadband():
    returns, positions = _inputs()
    positions.iloc[5:, 0] = 0.1
    result = _build(returns, positions, rebalance_deadzone=0.9)
    assert result["asset_weights_df"].iloc[5, 0] < result["asset_weights_df"].iloc[4, 0]


def test_fifty_commodity_comparison_uses_same_signed_execution_ledger():
    from examples.multi_factor_cta.test_all_weather_50commodities_portfolio import evaluate_50_weighting_schemes

    returns, positions = _inputs()
    positions.iloc[3:8, 0] = -1.0
    positions.iloc[8:15, 1] = 0.0
    kwargs = {"asset_positions": positions, "asset_fee_rates": {"A": 0.001, "B": 0.001}, "periods_per_year": 252}
    results = evaluate_50_weighting_schemes(returns, {"A": "X", "B": "Y"}, **kwargs)
    prefix = evaluate_50_weighting_schemes(
        returns.iloc[:30],
        {"A": "X", "B": "Y"},
        **{**kwargs, "asset_positions": positions.iloc[:30]},
    )
    for name, result in results.items():
        pd.testing.assert_frame_equal(result["weights_df"].iloc[:30], prefix[name]["weights_df"])
        w = result["weights_df"]
        gross, costs, _ = _quantity_reference((1.0 + returns).cumprod(), w, 0.001)
        expected = (gross - costs).sum(axis=1)
        np.testing.assert_allclose(result["daily_returns"], expected, atol=1e-14)
        assert result["weights_df"].iloc[4, 0] < 0.0


def test_empty_is_warmup_does_not_trigger_max_leverage_boost_at_oos_start():
    returns, positions = _inputs()
    # Zero out the first 30 days of positions (simulating flat pre-OOS warmup).
    positions_zero_warmup = positions.copy()
    positions_zero_warmup.iloc[:30] = 0.0
    result = _build(returns, positions_zero_warmup, periods_per_year=252)
    # On bar 30 (first active bar after flat warmup), active_exp_bars < 15 days,
    # so portfolio rolling vol must fall back to target_portfolio_vol (lev_scalar == 1.0),
    # never spiking to 1.65x due to zero variance of flat returns.
    gross_at_oos_start = float(result["asset_weights_df"].iloc[30].abs().sum())
    assert gross_at_oos_start == pytest.approx(1.0, abs=1e-6)


def test_daily_turbulence_cap_uses_one_bar_per_day():
    p = _prices()
    frame = pd.DataFrame({"close": p.to_numpy()})
    cap_1d = IntraAssetThreePillarSynthesizer._get_causal_turbulence_cap(frame, "1d")
    cap_15m = IntraAssetThreePillarSynthesizer._get_causal_turbulence_cap(frame, "15m")
    # Daily bars must use bars_per_day=1, whereas 15m uses bars_per_day=16 (4x higher annualized vol on same series).
    assert float(np.mean(cap_1d[30:])) > float(np.mean(cap_15m[30:]))


def test_hysteresis_runner_governance_is_prefix_invariant():
    from examples.multi_factor_cta.synthesis_engine import govern_position_with_hysteresis_runner

    frame = generate_symbol_kines("SHFE.RB9999", "15分钟", "20230102", "20230215", seed=811).iloc[:240].copy()
    raw_w = np.tanh(frame["close"].pct_change(4).fillna(0.0).to_numpy(dtype=np.float64) * 50.0)
    cut = 160
    full = govern_position_with_hysteresis_runner(frame, raw_w, base_tf="15m")
    prefix = govern_position_with_hysteresis_runner(frame.iloc[:cut].copy(), raw_w[:cut], base_tf="15m")
    np.testing.assert_allclose(full[:cut], prefix, atol=1e-12)


def test_continuous_cluster_risk_shares_smoothly_dampen_collinear_peers():
    from examples.multi_factor_cta.synthesis_engine import compute_continuous_cluster_risk_shares

    price_corr = pd.DataFrame(
        [[1.0, 0.90, 0.20], [0.90, 1.0, 0.20], [0.20, 0.20, 1.0]],
        index=["rb", "hc", "au"],
        columns=["rb", "hc", "au"],
    )
    strat_corr = pd.DataFrame(
        [[1.0, 0.80, 0.10], [0.80, 1.0, 0.10], [0.10, 0.10, 1.0]],
        index=["rb", "hc", "au"],
        columns=["rb", "hc", "au"],
    )
    shares = compute_continuous_cluster_risk_shares(
        price_corr_df=price_corr,
        strategy_corr_df=strat_corr,
        sectors={"FERROUS": ["rb", "hc"], "PRECIOUS": ["au"]},
    )
    assert shares["au"] == pytest.approx(1.0)
    assert 0.5 < shares["rb"] < 0.65
    assert shares["rb"] == pytest.approx(shares["hc"])


def test_causal_sleeve_health_multiplier_lags_night_session_to_prev_trading_day():
    from examples.multi_factor_cta.synthesis_engine import compute_causal_sleeve_health_multiplier

    days = pd.date_range("2025-01-01", periods=30, freq="B")
    daily_ret = pd.DataFrame({"A": np.linspace(0.001, 0.01, len(days))}, index=days)
    # 2025-02-10 21:15 belongs to trading day 2025-02-11; its health multiplier must equal 2025-02-11's lagged value
    intraday_idx = pd.DatetimeIndex(["2025-02-10 14:45:00", "2025-02-10 21:15:00", "2025-02-11 09:30:00"])
    h_intraday = compute_causal_sleeve_health_multiplier(daily_ret, intraday_idx)
    assert h_intraday.loc["2025-02-10 21:15:00", "A"] == pytest.approx(h_intraday.loc["2025-02-11 09:30:00", "A"])
    # Changing the return on 2025-02-11 must not affect 2025-02-10 21:15 or 2025-02-11 09:30 (strictly shifted by 1 day)
    daily_ret_perturbed = daily_ret.copy()
    daily_ret_perturbed.loc["2025-02-11", "A"] = -0.50
    h_perturbed = compute_causal_sleeve_health_multiplier(daily_ret_perturbed, intraday_idx)
    pd.testing.assert_frame_equal(h_intraday, h_perturbed)
