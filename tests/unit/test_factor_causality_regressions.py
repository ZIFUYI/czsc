"""Regression checks for factor causality, context provenance and soft gate accounting."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.hypothesis_bank import ARCH_B, enrich_kline_features
from examples.multi_factor_cta.atomic_factors import (
    BI_BASE_COL,
    DataFrameCrossSectionProvider,
    FactorContext,
    SoftCrossSectionalGate,
    _build_fc01,
    _unbox,
    compute_atomic_signal,
    compute_family,
    set_factor_context,
)
from examples.multi_factor_cta.data_engine import adjust_continuous_roll_gaps, compute_causal_cross_sectional_features
from examples.multi_factor_cta.factor_engine import RuleFactorRegistry


def _bars(n=80, seed=317):
    df = generate_symbol_kines("SHFE.RB9999", "30分钟", "20220101", "20220301", seed=seed)
    df = df.rename(columns={"vol": "volume"}).iloc[:n].reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["dt"])
    df["date"] = df["dt"].dt.date
    return enrich_kline_features(df)


def test_fc01_future_htf_availability_cannot_change_past_setups():
    df = _bars()
    ctx = FactorContext.from_df(df)
    cut = len(df) // 2
    no_state = replace(ctx, htf_dir=np.zeros(ctx.n, dtype=np.int8), htf_q=np.zeros(ctx.n))
    later_state = replace(no_state, htf_dir=no_state.htf_dir.copy())
    later_state.htf_dir[cut:] = 1
    for side in (-1, 1):
        before = _build_fc01(no_state, side)
        after = _build_fc01(later_state, side)
        np.testing.assert_array_equal(before[0][:cut], after[0][:cut])
        np.testing.assert_allclose(before[1][:cut], after[1][:cut])


def test_completed_bi_events_do_not_depend_on_unfinished_direction_strings():
    import czsc

    df = _bars(80)
    expected = FactorContext.from_df(df).bi_flip
    df[BI_BASE_COL] = np.where(np.arange(len(df)) % 2, "向上_中继", "向下_转折")
    ctx = FactorContext.from_df(df)
    snapshots = czsc.structure_snapshots(
        czsc.format_standard_kline(df.rename(columns={"volume": "vol"}), freq="30分钟")
    )
    np.testing.assert_array_equal(ctx.bi_flip, expected)
    np.testing.assert_array_equal(ctx.bi_flip, [snapshot.completed_bi_dir for snapshot in snapshots])


def test_soft_gate_never_hides_an_event_with_nonzero_exposure():
    df = _bars(8)
    triggers = np.ones(len(df), dtype=bool)
    weights = -np.ones(len(df))
    actual_triggers, gated = SoftCrossSectionalGate().apply_to_triggers_and_weights(
        triggers, weights, np.full(len(df), 0.95), None, ARCH_B, -1
    )
    assert (np.abs(gated) > 0).all()
    np.testing.assert_array_equal(actual_triggers, triggers)


@pytest.mark.parametrize("family", ["FC02", "FC03", "FC04"])
def test_cross_asset_factors_require_real_context(family):
    with pytest.raises(ValueError, match="missing_context"):
        compute_family(FactorContext.from_df(_bars()), family)


def test_atomic_cache_invalidates_non_close_input_and_htf_inplace_updates():
    df = _bars()
    htf = _bars(40)
    htf[BI_BASE_COL] = "向上_中继"
    set_factor_context(df, htf=htf)
    compute_atomic_signal(df, "AC01L")
    original = _unbox(df, "_atomic_ctx")[1]
    df.loc[10, "low"] -= 1.0
    compute_atomic_signal(df, "AC01L")
    revised = _unbox(df, "_atomic_ctx")[1]
    assert revised is not original
    assert revised.l[10] == df.loc[10, "low"]
    htf[BI_BASE_COL] = "向下_转折"
    compute_atomic_signal(df, "AC01L")
    assert _unbox(df, "_atomic_ctx")[1] is not revised


def test_cross_section_provider_fingerprint_covers_all_tfs_and_timestamps():
    a, b = _bars(30), _bars(30, 318)
    cs = compute_causal_cross_sectional_features({"A": a, "B": b}, tf="30m")
    frames = {"30m": cs["A"], "60m": cs["B"].copy()}
    first = DataFrameCrossSectionProvider(frames, default_tf="30m").cache_fingerprint()
    frames["60m"]["dt"] += pd.Timedelta("1min")
    second = DataFrameCrossSectionProvider(frames, default_tf="30m").cache_fingerprint()
    assert first != second


def test_stopped_asset_quotes_expire_and_membership_mask_is_point_in_time():
    a, b = _bars(50), _bars(50, 318)
    stopped = b.iloc[:10].copy()
    stopped["close"] = stopped["close"] * np.arange(1, 11)
    result = compute_causal_cross_sectional_features({"A": a, "B": stopped}, tf="30m")
    assert result["A"]["cs_active_count"].iloc[-1] == 1
    assert result["A"]["sector_leader_dir"].iloc[-1] == 0
    membership = pd.Series(np.arange(len(b)) >= 25, index=b["dt"])
    result = compute_causal_cross_sectional_features({"A": a, "B": b}, tf="30m", membership_masks={"B": membership})
    assert (result["A"]["cs_active_count"].iloc[:25] == 1).all()
    assert (result["A"]["cs_active_count"].iloc[25:] == 2).all()


def test_roll_adjustment_has_no_future_underlying_regime_branch():
    df = _bars(12)
    df["date"] = pd.date_range("2022-01-01", periods=len(df)).date
    df["underlying_symbol"] = "RB2201"
    # A large early price gap must not be retrospectively reclassified when the first later roll appears.
    df.loc[2, "open"] = df.loc[1, "close"] * 2
    extended = df.copy()
    extended.loc[8:, "underlying_symbol"] = "RB2205"
    expected = adjust_continuous_roll_gaps(df.iloc[:8])
    actual = adjust_continuous_roll_gaps(extended)
    np.testing.assert_allclose(expected["adj_factor"], actual["adj_factor"].iloc[:8])


def test_registry_applies_attr_and_explicit_soft_gate_once():
    from examples.multi_factor_cta.atomic_factors import CS_FEATURE_COLS

    df = _bars()
    for col in CS_FEATURE_COLS:
        df[col] = 0.8 if col == "cs_slope_rank" else 0.0
    gate = SoftCrossSectionalGate()
    set_factor_context(df, soft_gate=gate)
    registry = RuleFactorRegistry()
    plain_triggers, plain_weights = registry.evaluate_factor_on_tf(df, "CC02L", "30m", soft_gate=False)
    raw_triggers, raw_weights = compute_atomic_signal(df, "CC02L", soft_gate=False)
    np.testing.assert_array_equal(plain_triggers, raw_triggers)
    np.testing.assert_allclose(plain_weights, raw_weights)
    triggers, weights = registry.evaluate_factor_on_tf(df, "CC02L", "30m", soft_gate=gate)
    _, attr_weights = registry.evaluate_factor_on_tf(df, "CC02L", "30m")
    np.testing.assert_allclose(attr_weights, weights)
    assert np.any(plain_weights != 0), "fixture must exercise a live signal"
    expected = gate.apply(
        plain_weights, df["cs_slope_rank"].to_numpy(), gate.extract_causal_er(df), registry.get("CC02L").archetype
    )
    np.testing.assert_allclose(weights, expected)
    np.testing.assert_array_equal(triggers, plain_triggers)


def test_adaptive_vol_scale_requires_pre_sample_frozen_calibration():
    df = _bars()
    with pytest.raises(ValueError, match="calibration"):
        RuleFactorRegistry().evaluate_factor_on_tf(df, "S1L", "30m", adaptive_vol_scale=True)
    df.attrs["vol_calibration"] = {"multiplier": 1.2, "calibrated_until": df["dt"].iloc[-1]}
    with pytest.raises(ValueError, match="calibration"):
        RuleFactorRegistry().evaluate_factor_on_tf(df, "S1L", "30m", adaptive_vol_scale=True)
    df.attrs["vol_calibration"] = {"multiplier": 1.2, "calibrated_until": df["dt"].iloc[0] - pd.Timedelta("1min")}
    assert RuleFactorRegistry.compute_asset_vol_multiplier(df, "30m") == pytest.approx(1.2)


def test_unknown_sector_does_not_fabricate_common_sector_peers():
    frames = {"A": _bars(40), "B": _bars(40, 318)}
    result = compute_causal_cross_sectional_features(frames, tf="30m")
    for frame in result.values():
        assert not frame["cs_available"].any()
        assert (frame["sector_leader_dir"] == 0).all()
        assert (frame["sector_spread_zscore"] == 0).all()


def test_cross_section_timeframe_selection_does_not_depend_on_future_length():
    a, b = _bars(40), _bars(40, 318)
    frames = compute_causal_cross_sectional_features({"A": a, "B": b}, specs={"A": "X", "B": "X"}, tf="30m")
    selected = frames["A"].copy()
    other = frames["B"].copy()
    other["cs_slope_rank"] = 1.0 - selected["cs_slope_rank"]
    provider = DataFrameCrossSectionProvider({"30m": other, "60m": selected}, default_tf="60m")
    full = provider.features(a)["cs_slope_rank"]
    prefix = provider.features(a.iloc[:20])["cs_slope_rank"]
    np.testing.assert_allclose(full[:20], prefix)
    np.testing.assert_allclose(full, selected["cs_slope_rank"])
    other_default = DataFrameCrossSectionProvider({"30m": other, "60m": selected}, default_tf="30m")
    assert provider.cache_fingerprint() != other_default.cache_fingerprint()
