"""Full-60 study orchestration must rebuild verified evidence and preserve empty folds."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from czsc.mock import generate_symbol_kines
from examples.multi_factor_cta.synthesis_engine import account_portfolio_execution, build_asset_execution_inputs

SOURCE = Path(__file__).resolve().parents[2] / "examples/results/intraday_cta_ablation_20261017"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
study = importlib.import_module("run_full_60_commodities_study")


def _frame():
    frame = generate_symbol_kines("TEST", "日线", "20240101", "20250201", seed=37)
    frame["dt"] = pd.to_datetime(frame["dt"])
    frame["atr14"] = (frame["high"] - frame["low"]).rolling(14, min_periods=1).mean()
    return frame


def test_core_satellite_preserves_fold_when_no_satellites_are_admitted(monkeypatch):
    frame = _frame()
    returns, positions = build_asset_execution_inputs(frame, np.zeros(len(frame)))
    ret = returns.to_frame("ag")
    weights = positions.to_frame("ag")
    ledger = account_portfolio_execution(ret, weights, {"ag": 0.00015})
    ledger["weights_df"] = weights
    prices = pd.Series(frame.open.to_numpy(), index=pd.DatetimeIndex(frame.dt))
    prices.loc[prices.index[-1] + pd.Timedelta(microseconds=1)] = frame.close.iloc[-1]
    ledger.update(execution_prices_df=prices.to_frame("ag"), fee_rates={"ag": 0.00042})
    fold = {"id": "F1"}
    monkeypatch.setattr(study, "FOLDS", (fold,))
    monkeypatch.setattr(study, "IDS_60", ["ag"])
    monkeypatch.setattr(study, "CORE_20_IDS", ("ag",))
    fc = {
        "ret_oos": ret,
        "mask_oos": ret.notna(),
        "roll_vol_oos": ret * 0 + 0.15,
        "execution_prices_df": prices.to_frame("ag"),
    }
    result, ledgers, admitted = study.run_core_satellite_sop_stage3(
        "empty",
        {"F1": fc},
        {},
        {},
        {},
        {"F1": ["ag"]},
        {"hr": [ledger]},
        {"hr": {}},
        {"hr": {}},
    )
    assert admitted == {"F1": []}
    assert len(ledgers) == 1
    assert ledgers[0]["bar_net_returns"].eq(0).all()
    assert ledgers[0]["fee_rates"]["ag"] == 0.00042
    pd.testing.assert_frame_equal(ledgers[0]["execution_prices_df"], prices.to_frame("ag"))
    pd.testing.assert_frame_equal(ledgers[0]["returns_df"], ret)
    pd.testing.assert_frame_equal(ledgers[0]["tradable_mask"], ret.notna())
    assert result["net_ret_pct"] == 0.0


def test_sleeve_training_metrics_ignore_future_prices_and_weights():
    frame = _frame()
    weights = np.resize(np.array([1.0, 1.0, 0.0, 0.0]), len(frame))
    cutoff = pd.Timestamp("2024-12-31 23:59:59")
    first = study._sleeve_training_metrics(frame, weights, 0.00015, cutoff, pd.Timestamp("2024-01-01"))
    mask = frame.dt > cutoff
    frame.loc[mask, ["open", "high", "low", "close"]] *= 100
    weights[mask] = -0.8
    second = study._sleeve_training_metrics(frame, weights, 0.00015, cutoff, pd.Timestamp("2024-01-01"))
    assert first == second


def test_full60_rebuilds_from_complete_prepared_evidence_and_only_reuses_verified_bundles(monkeypatch, tmp_path):
    import dataclasses
    import json

    from examples.multi_factor_cta.audit_protocol import (
        ArtifactFingerprint,
        build_fingerprint,
        canonical_json,
        read_verified_pickle,
    )
    from examples.multi_factor_cta.forensic_validator import summarize_execution_pooled_scores

    frame = _frame()
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "data_5m").mkdir(parents=True)
    source_code = source / "entry.py"
    source_code.write_text("# frozen test input\n")
    raw = source / "ag_15m.parquet"
    frame.to_parquet(raw)
    frame.to_parquet(source / "data_5m/ag_5m.parquet")
    full_spec = dataclasses.replace(study.SPECS_60["ag"], data_path=raw)
    core_spec = dataclasses.replace(full_spec, fee_rate=0.00042, sector="precious_metals")
    fold = {
        "id": "F1",
        "train_start": "2024-01-01",
        "cutoff": "2024-12-31 23:59:59",
        "oos_start": "2025-01-01",
        "oos_end": "2025-02-01",
    }
    for key, value in {
        "SOURCE_DIR": source,
        "OUT_DIR": output,
        "CACHE_60_DIR": output / "cache_60",
        "UNIVERSE_60_SPECS": [full_spec],
        "SPECS_60": {"ag": full_spec},
        "IDS_60": ["ag"],
        "CORE_20_IDS": ("ag",),
        "FOLDS": (fold,),
        "SECTORS_60": {full_spec.sector: ["ag"]},
    }.items():
        monkeypatch.setattr(study, key, value)
    monkeypatch.setattr(study, "_core20_specs_from_manifest", lambda: {"ag": core_spec})
    calls = []
    scores = pd.DataFrame(columns=["fold", "asset", "uid", "tf", "archetype"])
    pool = summarize_execution_pooled_scores(scores)

    def prepare(specs, *, source_dir, output_dir):
        calls.append(specs[0].fee_rate)
        fingerprint = build_fingerprint(
            source_paths=[source_code], data_paths=[raw], config={"fee": specs[0].fee_rate, "sector": specs[0].sector}
        )
        return {
            "frames": {full_spec.symbol: {"15m": frame, "1d": frame}},
            "units": {"ag": {}},
            "scores": scores,
            "pool": pool,
            "fingerprint": fingerprint,
            "settings": {},
        }

    monkeypatch.setattr(study, "prepare_study_inputs", prepare)
    full, core = study.ensure_full60_oos_bundles(force_rebuild=True)
    assert calls == [full_spec.fee_rate, core_spec.fee_rate]
    assert full["sel_60_std"] == {("F1", "ag"): []}
    assert full["is_meta_stats"][("F1", "ag")]["is_execution_passed"] is False
    assert len(core["ledgers"]["hr"]) == 1
    assert full["oos_fast_cache"]["F1"]["execution_prices_df"].notna().any().all()
    assert core["ledgers"]["hr"][0]["bar_net_returns"].eq(0).all()
    assert (
        read_verified_pickle(output / "cache_60/audited_full60_bundle.pkl", full["fingerprint"])["pit_assets"]
        == full["pit_assets"]
    )
    study.save_execution_ledgers({"core": core["ledgers"]["hr"]}, full["fingerprint"], core["fingerprint"])
    path = output / "ledgers/core_F1.pkl"
    metadata = json.loads(path.with_name(path.name + ".provenance.json").read_text())
    fingerprint = ArtifactFingerprint(metadata["fingerprint"], canonical_json(metadata["manifest"]))
    saved = read_verified_pickle(path, fingerprint)
    assert saved["fold"] == "F1"
    assert saved["variant"] == "core"
    assert saved["fee_rates"] == {"ag": core_spec.fee_rate}
    assert set(metadata["manifest"]["config"]["input_studies"]) == {"full60", "core20"}
    monkeypatch.setattr(
        study, "study_execution_ledger", lambda *args: (_ for _ in ()).throw(AssertionError("cache miss"))
    )
    reused_full, reused_core = study.ensure_full60_oos_bundles()
    assert reused_full["fingerprint"] == full["fingerprint"]
    assert reused_core["fingerprint"] == core["fingerprint"]


def test_composite_sleeve_cannot_use_old_low_sample_stage3_exception():
    metrics = {
        "is_execution_passed": True,
        "is_ep_count": 8,
        "is_effective_n": 8,
        "is_win_rate": 80,
        "is_payoff_r": 3,
        "is_safety_margin": 40,
        "is_mean_ci_low_pct": 1,
    }
    assert not study._sleeve_sop_passed(metrics)
