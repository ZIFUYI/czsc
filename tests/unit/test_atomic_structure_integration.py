"""Formal atomic-factor integration with native structural provenance and exits."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.hypothesis_bank import (
    ARCHETYPE_RULERS,
    attach_native_czsc_signals,
    enrich_kline_features,
)
from examples.multi_factor_cta.atomic_factors import (
    ATOMIC_FACTOR_DEFS,
    FactorContext,
    _unbox,
    compute_atomic_signal,
    compute_family,
)


def _frames():
    raw = generate_symbol_kines("TEST", "30分钟", "20220101", "20220601", seed=951)
    raw = raw.rename(columns={"vol": "volume"}).iloc[:600].reset_index(drop=True)
    raw["dt"] = pd.to_datetime(raw["dt"])
    raw["date"] = raw["dt"].dt.date
    pairs = raw.groupby(np.arange(len(raw)) // 2, sort=True)
    htf = pairs.agg(
        {
            "dt": "last",
            "symbol": "first",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).reset_index(drop=True)
    htf["date"] = htf["dt"].dt.date
    base, htf = enrich_kline_features(raw), enrich_kline_features(htf)
    base.attrs["factor_timeframe"] = "30m"
    htf.attrs["factor_timeframe"] = "60m"
    return base, htf


def _arm_real_structure(ctx, family, monkeypatch, *, distinguish_confirmation_extreme=False):
    """Control only boolean setup/confirmation, preserving all actual mock OHLC and native anchors."""
    definition = ATOMIC_FACTOR_DEFS[family]
    anchors = ctx.structure_anchors[definition.archetype]
    stop = np.asarray(anchors["long_stop"])
    confirmation_low = pd.Series(ctx.l).rolling(3).min().to_numpy()
    candidates = np.flatnonzero(np.isfinite(stop[:-2]) & (ctx.c[:-2] > stop[:-2]) & (ctx.c[1:-1] > stop[:-2]))
    if distinguish_confirmation_extreme:
        candidates = np.asarray([i for i in candidates if not np.isclose(stop[i], confirmation_low[i + 1])])
    assert candidates.size, f"mock fixture needs a valid {family} native structure"
    setup_idx = int(candidates[len(candidates) // 2])
    confirm_idx = setup_idx + 1
    setup = np.zeros(ctx.n, dtype=bool)
    confirm = np.zeros(ctx.n, dtype=bool)
    setup[setup_idx], confirm[confirm_idx] = True, True

    def controlled_build(context, side):
        if side == 1:
            return setup, np.ones(context.n), confirm, np.ones(context.n), 6
        return (
            np.zeros(context.n, dtype=bool),
            np.ones(context.n),
            np.zeros(context.n, dtype=bool),
            np.ones(context.n),
            6,
        )

    monkeypatch.setitem(ATOMIC_FACTOR_DEFS, family, replace(definition, build=controlled_build))
    # Exit-event timing is controlled independently of structure price discovery.
    ctx.bi_flip = np.zeros(ctx.n, dtype=np.int8)
    return setup_idx, confirm_idx, anchors


def test_formal_family_freezes_native_setup_fx_instead_of_confirmation_three_bar_low(monkeypatch):
    base, htf = _frames()
    ctx = FactorContext.from_df(base, htf)
    setup, confirm, anchors = _arm_real_structure(ctx, "AC01", monkeypatch, distinguish_confirmation_extreme=True)
    strength, trigger = compute_family(ctx, "AC01", sides=(1,), soft_gate=False)
    result = ctx.structure_results[("AC01", (1,))]
    assert trigger[confirm] and strength[confirm] > 0
    assert result["setup_index"][confirm] == setup
    assert result["stop"][confirm] == anchors["long_stop"][setup]
    assert result["structure_id"][confirm] == anchors["long_id"][setup]
    assert not np.isclose(result["stop"][confirm], ctx.l[confirm - 2 : confirm + 1].min())


def test_exit_row_preserves_direction_and_source_structure_identity(monkeypatch):
    base, htf = _frames()
    ctx = FactorContext.from_df(base, htf)
    setup, confirm, anchors = _arm_real_structure(ctx, "AC01", monkeypatch)
    ctx.bi_flip[confirm + 1] = -1
    strength, _ = compute_family(ctx, "AC01", sides=(1,), soft_gate=False)
    result = ctx.structure_results[("AC01", (1,))]
    assert strength[confirm] > 0 and strength[confirm + 1] == 0
    assert result["position_direction"][confirm + 1] == 1
    assert result["structure_id"][confirm + 1] == anchors["long_id"][setup]
    assert result["setup_index"][confirm + 1] == setup
    assert result["exit_reason"][confirm + 1] in {"reverse_bi", "structure_break"}


@pytest.mark.parametrize("family", ["AC01", "BC01", "CC01", "DC01", "EC01", "FC01"])
def test_every_stop_plugin_reaches_the_formal_state_machine(family, monkeypatch):
    base, htf = _frames()
    ctx = FactorContext.from_df(base, htf)
    setup, confirm, anchors = _arm_real_structure(ctx, family, monkeypatch)
    _, trigger = compute_family(ctx, family, sides=(1,), soft_gate=False)
    result = ctx.structure_results[(family, (1,))]
    assert trigger[confirm]
    assert result["stop_plugin"] == ARCHETYPE_RULERS[ATOMIC_FACTOR_DEFS[family].archetype].stop_plugin
    assert result["stop"][confirm] == anchors["long_stop"][setup]
    assert result["structure_id"][confirm] == anchors["long_id"][setup]


def test_htf_stop_is_unavailable_until_actual_htf_data_is_supplied(monkeypatch):
    base, htf = _frames()
    with_htf = FactorContext.from_df(base, htf)
    _, confirm, _ = _arm_real_structure(with_htf, "FC01", monkeypatch)
    _, valid_trigger = compute_family(with_htf, "FC01", sides=(1,), soft_gate=False)
    assert valid_trigger[confirm]
    without_htf = FactorContext.from_df(base)
    without_htf.bi_flip = np.zeros(without_htf.n, dtype=np.int8)
    strength, trigger = compute_family(without_htf, "FC01", sides=(1,), soft_gate=False)
    anchors = without_htf.structure_anchors[ATOMIC_FACTOR_DEFS["FC01"].archetype]
    assert np.isnan(anchors["long_stop"]).all()
    assert not trigger.any()
    assert not strength.any()


def test_native_structure_environment_invalidates_both_caches(monkeypatch, tmp_path):
    base, _ = _frames()
    base = base.iloc[:120].copy()
    compute_atomic_signal(base, "AC01L", soft_gate=False)
    previous = _unbox(base, "_atomic_ctx")[1]
    attach_native_czsc_signals(base, "30m", tmp_path, extended=False)
    count = len(list(tmp_path.glob("native_czsc_signals_30m_*.pkl")))
    for name, value in (
        ("CZSC_MIN_BI_LEN", "8"),
        ("czsc_min_bi_len", "9"),
        ("CZSC_MAX_BI_NUM", "14"),
        ("czsc_max_bi_num", "15"),
    ):
        monkeypatch.setenv(name, value)
        compute_atomic_signal(base, "AC01L", soft_gate=False)
        current = _unbox(base, "_atomic_ctx")[1]
        assert current is not previous
        previous = current
        attach_native_czsc_signals(base, "30m", tmp_path, extended=False)
        count += 1
        assert len(list(tmp_path.glob("native_czsc_signals_30m_*.pkl"))) == count
