"""Run-scoped audit caching preserves formal admission and its evidence boundaries."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent import run_evolution_loop as entry
from examples.alpha_mining_agent.hypothesis_bank import enrich_kline_features
from examples.multi_factor_cta.forensic_validator import RuleFactorForensicValidator


def _pool():
    frames = {}
    for index, symbol in enumerate(("ETF_A", "ETF_B")):
        frame = generate_symbol_kines(symbol, "日线", "20220101", "20231231", seed=731 + index)
        frame["date"] = pd.to_datetime(frame["dt"]).dt.date
        frame = enrich_kline_features(frame.rename(columns={"vol": "volume"}).iloc[:80].copy())
        frames[symbol] = {"1d": frame}
    return entry.prepare_research_pool(frames, {symbol: {"asset_group": "equity"} for symbol in frames})


def _count_calls(monkeypatch):
    calls = Counter()
    for owner, name, label in (
        (entry, "compute_research_signal", "signal"),
        (entry, "run_execution_audit", "execution"),
        (RuleFactorForensicValidator, "run_vol_and_regime_normalized_audit", "normalized"),
        (RuleFactorForensicValidator, "run_stratified_pooled_gate1b_audit", "pooled"),
    ):
        original = getattr(owner, name)

        def counted(*args, _original=original, _label=label, **kwargs):
            calls[_label] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(owner, name, counted)
    return calls


def test_cached_admission_matches_real_uncached_pipeline_and_reuses_pool_evidence(monkeypatch):
    pool = _pool()
    expected = {symbol: entry.audit_research_candidate("AC01L", "1d", symbol, pool) for symbol in pool}
    calls = _count_calls(monkeypatch)
    context = entry.ResearchAuditContext(pool)
    actual = {
        symbol: entry.audit_research_candidate("AC01L", "1d", symbol, pool, audit_context=context) for symbol in pool
    }
    assert actual == expected
    assert calls == {"signal": 2, "normalized": 2, "pooled": 1, "execution": 2}
    before = calls.copy()
    for symbol in pool:
        assert entry.audit_research_candidate("AC01L", "1d", symbol, pool, audit_context=context) == expected[symbol]
    assert calls == before
    assert context.cache_info() == {"signals": 2, "normalized": 2, "pooled": 1, "admissions": 2}
    # A report consumer cannot poison subsequent calls by annotating its copy.
    actual["ETF_A"]["execution"]["reason"] = "report annotation"
    actual["ETF_A"]["gate1b"]["reason"] = "report annotation"
    assert entry.audit_research_candidate("AC01L", "1d", "ETF_A", pool, audit_context=context) == expected["ETF_A"]


def test_cached_nonempty_execution_events_are_identical_and_cannot_be_mutated(monkeypatch):
    pool = _pool()

    def positions(frame, sid, tf):
        weights = np.resize(np.array([0.5, 0.0]), len(frame))
        return weights != 0, weights

    # Controlled targets exercise real Rust executions and real statistical audits,
    # without claiming that simulated prices establish economic profitability.
    monkeypatch.setattr(entry, "compute_research_signal", positions)
    expected = entry.audit_research_candidate("AC01L", "1d", "ETF_A", pool)
    context = entry.ResearchAuditContext(pool)
    actual = entry.audit_research_candidate("AC01L", "1d", "ETF_A", pool, audit_context=context)
    assert actual == expected
    assert len(actual["execution"]["events"]) >= 30
    assert actual["gate1b"]["total_pool_is_n"] >= 60
    actual["execution"]["events"][0]["net_ret"] = 999.0
    actual["gate1b"]["per_asset_audits"]["ETF_A"]["events"][0]["net_ret"] = 999.0
    assert entry.audit_research_candidate("AC01L", "1d", "ETF_A", pool, audit_context=context) == expected


@pytest.mark.parametrize("change", ["cutoff", "fee", "spec", "ruler", "scale", "native_env", "pool", "regime"])
def test_cache_isolates_cutoff_fees_and_research_configuration(change, monkeypatch):
    pool = _pool()
    context = entry.ResearchAuditContext(pool)
    calls = _count_calls(monkeypatch)
    entry.audit_research_candidate("AC01L", "1d", "ETF_A", pool, audit_context=context)
    cutoff = entry.IS_END_DT
    if change == "cutoff":
        cutoff = pd.Timestamp(pool["ETF_A"]["1d"]["dt"].iloc[30])
    elif change == "fee":
        pool["ETF_B"]["1d"].attrs["fee_rate"] = 0.001
    elif change == "spec":
        monkeypatch.setitem(
            entry.ALL_SIGNAL_SPECS, "AC01L", replace(entry.ALL_SIGNAL_SPECS["AC01L"], default_weight=0.77)
        )
    elif change == "ruler":
        archetype = entry.ALL_SIGNAL_SPECS["AC01L"].archetype
        ruler = entry.ARCHETYPE_RULERS[archetype]
        monkeypatch.setitem(entry.ARCHETYPE_RULERS, archetype, replace(ruler, min_net_sharpe=10.0))
    elif change == "scale":
        monkeypatch.setitem(entry.TF_SCALE_MAP, "1d", entry.TF_SCALE_MAP["1d"] * 1.1)
    elif change == "native_env":
        monkeypatch.setenv("CZSC_MIN_BI_LEN", "8")
    elif change == "pool":
        pool["ETF_B"]["1d"].attrs["pool_group"] = "commodity:metals"
    else:
        pool["ETF_B"]["1d"].attrs["microstructure_group"] = "explicit-test-regime"
    actual = entry.audit_research_candidate("AC01L", "1d", "ETF_A", pool, is_end_dt=cutoff, audit_context=context)
    assert calls == {"signal": 4, "normalized": 4, "pooled": 2, "execution": 2}
    expected = entry.audit_research_candidate("AC01L", "1d", "ETF_A", pool, is_end_dt=cutoff)
    assert actual == expected


def test_cache_is_bound_to_its_prepared_pool_and_releases_batches():
    pool = _pool()
    context = entry.ResearchAuditContext(pool)
    entry.audit_research_candidate("AC01L", "1d", "ETF_A", pool, audit_context=context)
    context.clear()
    assert context.cache_info() == {"signals": 0, "normalized": 0, "pooled": 0, "admissions": 0}
    with pytest.raises(ValueError, match="original frozen pool"):
        entry.audit_research_candidate("AC01L", "1d", "ETF_A", dict(pool), audit_context=context)
    pool["ETF_A"]["1d"] = pool["ETF_A"]["1d"].iloc[:-1].copy()
    with pytest.raises(ValueError, match="original frozen pool"):
        entry.audit_research_candidate("AC01L", "1d", "ETF_A", pool, audit_context=context)
    # Rebuilding the context is the explicit boundary for edited/replaced data.
    fresh = entry.ResearchAuditContext(pool)
    assert not entry.audit_research_candidate("AC01L", "1d", "ETF_A", pool, audit_context=fresh)["passed"]


def test_explicit_event_overrides_never_reuse_or_populate_pool_cache():
    pool = _pool()
    context = entry.ResearchAuditContext(pool)
    overrides = {symbol: {"is_n_purged": 100, "events": []} for symbol in pool}
    expected = entry.evaluate_candidate_pooled_gate1b(
        "AC01L", "1d", "ETF_A", {}, pool, np.random.default_rng(42), per_symbol_gate1_overrides=overrides
    )
    actual = entry.evaluate_candidate_pooled_gate1b(
        "AC01L",
        "1d",
        "ETF_A",
        {},
        pool,
        np.random.default_rng(42),
        per_symbol_gate1_overrides=overrides,
        audit_context=context,
    )
    assert actual == expected
    assert all(value == 0 for value in context.cache_info().values())


def test_formal_synthesis_passes_explicit_context_to_admission(monkeypatch):
    from examples.multi_factor_cta import run_framework_demo as runner
    from examples.multi_factor_cta.data_engine import AssetClass, AssetSpec
    from examples.multi_factor_cta.synthesis_engine import IntraAssetThreePillarSynthesizer

    pool = _pool()
    context = entry.ResearchAuditContext(pool)
    spec = AssetSpec("ETF_A", "test", AssetClass.INDEX_FUTURES, "equity", "1d", ("1d",), 0.0002)
    observed = []
    original = runner.audit_research_candidate

    def capture(*args, **kwargs):
        observed.append(kwargs["audit_context"])
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "audit_research_candidate", capture)
    targets, _, _, audits = runner.synthesize_admitted_asset(
        spec, pool, ["AC01L@1d"], IntraAssetThreePillarSynthesizer(), audit_context=context
    )
    assert observed == [context]
    assert not targets.any()
    assert not audits["AC01L@1d"]["passed"]
