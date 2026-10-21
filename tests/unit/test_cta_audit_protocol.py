"""Research artifacts cannot silently reuse stale data or select on held-out dates."""

import json
import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.hypothesis_bank import enrich_kline_features
from examples.multi_factor_cta.audit_protocol import (
    FoldBoundary,
    FrozenResearchProtocol,
    UnverifiedArtifactError,
    build_fingerprint,
    freeze_protocol,
    read_verified_artifact,
    read_verified_pickle,
    validate_output_directory,
    write_verified_artifact,
)


def _inputs(tmp_path):
    source = tmp_path / "implementation.py"
    source.write_text("VERSION = 1\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    frame = generate_symbol_kines("SHFE.RB9999", "30分钟", "20240102", "20240110", seed=11)
    data = raw / "bars.csv"
    frame.to_csv(data, index=False)
    return source, data, raw


def _protocol():
    return FrozenResearchProtocol(
        candidates=("AC01L@30m", "AC01S@30m"),
        variants=("fixed_baseline",),
        folds=(
            FoldBoundary("2025", "2024-01-01", "2024-12-31 23:59:59", "2025-01-01", "2025-12-31 23:59:59"),
            FoldBoundary("2026", "2024-01-01", "2025-12-31 23:59:59", "2026-01-01", "2026-09-30 23:59:59"),
        ),
    )


def test_fingerprint_covers_middle_bytes_source_data_and_configuration(tmp_path):
    source, data, _ = _inputs(tmp_path)

    def identity(config=None):
        return build_fingerprint(source_paths=[source], data_paths=[data], config=config or {"fee": 0.0002})

    initial = identity()
    assert identity({"fee": 0.0003}).sha256 != initial.sha256
    source.write_text("VERSION = 2\n")
    assert identity().sha256 != initial.sha256
    source.write_text("VERSION = 1\n")
    payload = data.read_bytes()
    midpoint = len(payload) // 2
    data.write_bytes(payload[:midpoint] + b"X" + payload[midpoint + 1 :])
    assert identity().sha256 != initial.sha256


def test_fingerprint_is_order_independent_and_rejects_nonfinite_config(tmp_path):
    source, data, _ = _inputs(tmp_path)
    first = build_fingerprint(source_paths=[source], data_paths=[data], config={"a": 1, "b": 2})
    second = build_fingerprint(source_paths=[source, source], data_paths=[data], config={"b": 2, "a": 1})
    assert first == second
    with pytest.raises(ValueError):
        build_fingerprint(source_paths=[source], data_paths=[data], config={"fee": float("nan")})


def test_legacy_pickle_is_rejected_before_unpickling(tmp_path, monkeypatch):
    source, data, _ = _inputs(tmp_path)
    fingerprint = build_fingerprint(source_paths=[source], data_paths=[data], config={})
    legacy = tmp_path / "legacy.pkl"
    legacy.write_bytes(b"must not be decoded")

    def forbidden_decode(_payload):
        pytest.fail("Unverified pickle was decoded")

    monkeypatch.setattr(pickle, "loads", forbidden_decode)
    with pytest.raises(UnverifiedArtifactError, match="regenerate"):
        read_verified_pickle(legacy, fingerprint)


def test_verified_artifact_checks_payload_and_config_and_preserves_raw(tmp_path):
    source, data, raw = _inputs(tmp_path)
    original = data.read_bytes()
    fingerprint = build_fingerprint(source_paths=[source], data_paths=[data], config={"fee": 0.0002})
    output = tmp_path / "fresh_run"
    target = output / "scores.csv"
    payload = b"candidate,score\nAC01L@30m,0.25\n"
    write_verified_artifact(target, payload, fingerprint, output_dir=output, readonly_roots=[raw])
    assert read_verified_artifact(target, fingerprint) == payload
    assert data.read_bytes() == original
    changed = build_fingerprint(source_paths=[source], data_paths=[data], config={"fee": 0.001})
    with pytest.raises(UnverifiedArtifactError, match="changed"):
        read_verified_artifact(target, changed)
    target.write_bytes(payload.replace(b"0.25", b"9.99"))
    with pytest.raises(UnverifiedArtifactError, match="checksum"):
        read_verified_artifact(target, fingerprint)


def test_verified_pickle_roundtrip_and_missing_score_provenance(tmp_path):
    source, data, raw = _inputs(tmp_path)
    fingerprint = build_fingerprint(source_paths=[source], data_paths=[data], config={})
    output = tmp_path / "fresh_run"
    target = output / "candidate.pkl"
    record = {"candidate": "AC01L@30m", "events": []}
    write_verified_artifact(target, pickle.dumps(record), fingerprint, output_dir=output, readonly_roots=[raw])
    assert read_verified_pickle(target, fingerprint) == record
    target.with_name(f"{target.name}.provenance.json").unlink()
    with pytest.raises(UnverifiedArtifactError, match="regenerate"):
        read_verified_artifact(target, fingerprint)


def test_output_cannot_alias_raw_cache_or_escape_output(tmp_path):
    source, data, raw = _inputs(tmp_path)
    alias = tmp_path / "cache_alias"
    alias.symlink_to(raw, target_is_directory=True)
    for path in (raw, raw / "run", tmp_path, alias / "run"):
        with pytest.raises(ValueError, match="separate"):
            validate_output_directory(path, [raw])
    output = tmp_path / "fresh_run"
    fingerprint = build_fingerprint(source_paths=[source], data_paths=[data], config={})
    with pytest.raises(ValueError, match="inside"):
        write_verified_artifact(data, b"bad", fingerprint, output_dir=output, readonly_roots=[raw])


def test_protocol_labels_history_and_freezes_candidates_before_evaluation(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "fresh_run"
    protocol = _protocol()
    path = freeze_protocol(protocol, output_dir=output, readonly_roots=[raw])
    frozen = json.loads(path.read_bytes())
    assert frozen["historical_label"] == "retrospective_oos_validation"
    assert frozen["holdout_start"] == "2026-10-01 00:00:00"
    assert freeze_protocol(protocol, output_dir=output, readonly_roots=[raw]) == path
    with pytest.raises(ValueError, match="fresh independent"):
        freeze_protocol(replace(protocol, candidates=("changed",)), output_dir=output, readonly_roots=[raw])


@pytest.mark.parametrize("forbidden", ["2025-01-01", "2026-09-30", "2026-10-01", "2026-10-21"])
def test_protocol_never_selects_from_validation_or_holdout(forbidden):
    protocol = _protocol()
    protocol.assert_selection_timestamps(["2024-01-01", "2024-12-31 23:59:59"], fold_id="2025")
    with pytest.raises(ValueError, match="outside training"):
        protocol.assert_selection_timestamps([forbidden], fold_id="2025")


def test_protocol_rejects_holdout_overlap_and_blind_test_claims():
    protocol = _protocol()
    with pytest.raises(ValueError, match="overlap with holdout"):
        replace(protocol, folds=(replace(protocol.folds[1], test_end="2026-10-01"),)).validate()
    with pytest.raises(ValueError, match="retrospective"):
        replace(protocol, historical_label="final_blind_test").validate()
    with pytest.raises(ValueError, match="unique"):
        replace(protocol, candidates=("same", "same")).validate()


def test_signed_ma_slope_retains_direction_and_historical_abs_contract():
    frame = generate_symbol_kines("SHFE.RB9999", "30分钟", "20240102", "20240228", seed=317)
    frame = frame.rename(columns={"vol": "volume"}).iloc[:180].copy()
    frame["dt"] = pd.to_datetime(frame["dt"])
    frame["date"] = frame["dt"].dt.date
    original = enrich_kline_features(frame)
    np.testing.assert_allclose(original.ma60_slope, original.signed_ma60_slope.abs(), equal_nan=True)
    mirror = frame.copy()
    anchor = 2 * frame.high.max()
    mirror["open"] = anchor - frame.open
    mirror["close"] = anchor - frame.close
    mirror["high"] = anchor - frame.low
    mirror["low"] = anchor - frame.high
    reflected = enrich_kline_features(mirror)
    valid = original.signed_ma60_slope.notna() & reflected.signed_ma60_slope.notna()
    np.testing.assert_array_equal(
        np.sign(original.loc[valid, "signed_ma60_slope"]),
        -np.sign(reflected.loc[valid, "signed_ma60_slope"]),
    )
    prefix = enrich_kline_features(frame.iloc[:100])
    np.testing.assert_array_equal(original.signed_ma60_slope.iloc[:100], prefix.signed_ma60_slope)
