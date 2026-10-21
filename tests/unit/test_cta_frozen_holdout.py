"""A holdout can only replay its frozen decisions and must preserve an auditable zero ledger."""

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.multi_factor_cta import run_frozen_holdout as holdout
from examples.multi_factor_cta.audit_protocol import (
    UnverifiedArtifactError,
    build_fingerprint,
    file_sha256,
    write_verified_artifact,
)
from examples.multi_factor_cta.data_engine import AssetClass, AssetSpec


def _fixture(tmp_path, selected=()):
    source = tmp_path / "historical"
    source.mkdir()
    implementation = source / "implementation.py"
    implementation.write_text("VERSION = 1\n")
    incoming = tmp_path / "downloaded"
    incoming.mkdir()
    assets, data = {}, []
    for tf, frequency in (("5m", "5分钟"), ("15m", "15分钟")):
        history = generate_symbol_kines("SHFE.RB9999", frequency, "20240102", "20240105", seed=29)
        directory = source / f"data_{tf}"
        directory.mkdir()
        path = directory / f"rb_{tf}.parquet"
        history.to_parquet(path, index=False)
        data.append(path)
        fresh = generate_symbol_kines("SHFE.RB9999", frequency, "20261008", "20261020", seed=37)
        fresh_path = incoming / f"rb_{tf}.parquet"
        fresh.to_parquet(fresh_path, index=False)
        assets[tf] = {"path": str(fresh_path), "sha256": file_sha256(fresh_path)}
    spec = AssetSpec(
        symbol="SHFE.RB9999",
        name="rb",
        asset_class=AssetClass.COMMODITY_FUTURES,
        sector="black",
        base_tf="15m",
        eligible_tfs=("1d", "60m", "30m", "15m"),
        fee_rate=0.0002,
        has_night_session=True,
        data_path=source / "data_15m/rb_15m.parquet",
    )
    config = {
        "protocol": {
            "candidates": ["AC01L@30m"],
            "folds": [
                {
                    "fold_id": holdout.SELECTION_FOLD,
                    "train_start": holdout.TRAINING_START,
                    "train_end": holdout.TRAINING_END,
                }
            ],
        }
    }
    fingerprint = build_fingerprint(source_paths=[implementation], data_paths=data, config=config)
    historical_output = tmp_path / "historical_output"
    evidence = historical_output / "prepared_evidence.pkl"
    write_verified_artifact(
        evidence, pickle.dumps({"scores": []}), fingerprint, output_dir=historical_output, readonly_roots=[source]
    )
    selection = historical_output / "frozen_primary_selection.pkl"
    holdout.export_frozen_selection(
        selection,
        {"rb": list(selected)},
        [spec],
        fingerprint,
        evidence_path=evidence,
        output_dir=historical_output,
        source_dir=source,
    )
    (incoming / "download_manifest.json").write_text(
        json.dumps({"status": "complete", "holdout_performance_computed": False, "assets": {"rb": {"frames": assets}}})
    )
    return source, implementation, incoming, selection


def test_frozen_selection_preserves_rejections_and_rejects_source_drift(tmp_path):
    _, implementation, _, path = _fixture(tmp_path)
    record, _ = holdout.load_frozen_selection(path)
    assert record["selections"] == {"rb": []}
    assert record["selection_fold"] == "Fold2_2026"
    assert record["parameters"] == holdout.PRIMARY_PARAMETERS
    implementation.write_text("VERSION = 2\n")
    with pytest.raises(UnverifiedArtifactError, match="changed"):
        holdout.load_frozen_selection(path)


def test_selection_cannot_add_unregistered_candidate(tmp_path):
    with pytest.raises(ValueError, match="outside the frozen"):
        _fixture(tmp_path, ["UNSEEN@30m"])


def test_holdout_uses_trading_dates_and_rejects_contaminated_history():
    frame = generate_symbol_kines("SHFE.RB9999", "5分钟", "20240102", "20240103", seed=29).iloc[:5].copy()
    frame["dt"] = pd.to_datetime(
        [
            "2026-10-08 09:05",
            "2026-10-16 21:05",
            "2026-10-17 00:05",
            "2026-10-20 15:00",
            "2026-10-20 21:05",
        ]
    )
    kept = holdout.restrict_to_holdout(frame)
    assert kept.dt.tolist() == frame.dt.iloc[:4].tolist()
    with pytest.raises(ValueError, match="already contains"):
        holdout.concatenate_raw_history(frame.iloc[:1], frame)


def test_zero_selection_still_runs_real_execution_ledger_without_selection(tmp_path, monkeypatch):
    source, _, incoming, selection = _fixture(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Zero-admission holdout must not calculate factors, scores or select candidates")

    monkeypatch.setattr(holdout, "_selected_units", forbidden)
    output = tmp_path / "holdout_run"
    summary = holdout.run_holdout(selection, data_dir=incoming, output_dir=output, source_dir=source)
    assert summary["observed_trading_days"] == 9
    assert summary["admitted_factor_count"] == 0
    assert summary["bar_rows"] > 9
    assert summary["net_return"] == summary["cost_return"] == summary["max_gross_exposure"] == 0
    assert summary["annualized_sharpe"] is None and summary["sharpe_status"] == "no_trades"
    ledger = pickle.loads((output / "holdout_ledger.pkl").read_bytes())
    assert len(ledger["returns_df"]) == summary["bar_rows"]
    assert ledger["execution_prices_df"].notna().all().all()
    assert ledger["tradable_mask"].all().all()
    assert ledger["fee_rates"] == {"rb": 0.0002}
    assert np.all(ledger["turnover_df"].to_numpy() == 0)
    for name in ("execution_prices_df.csv", "returns_df.csv", "tradable_mask.csv"):
        assert (output / name).is_file()
    provenance = json.loads((output / "holdout_frozen_inputs.json").read_text())
    sources = {item["path"].split("/")[-1] for item in provenance["sources"]}
    assert {"run_frozen_holdout.py", "download_holdout.py"}.issubset(sources)


def test_holdout_download_payload_change_is_rejected_before_execution(tmp_path, monkeypatch):
    source, _, incoming, selection = _fixture(tmp_path)
    data = incoming / "rb_15m.parquet"
    data.write_bytes(data.read_bytes() + b"changed")
    monkeypatch.setattr(holdout, "execute_zero_selection", lambda *_: pytest.fail("Executed changed holdout"))
    with pytest.raises(UnverifiedArtifactError, match="Downloaded holdout changed"):
        holdout.run_holdout(selection, data_dir=incoming, output_dir=tmp_path / "run", source_dir=source)


def test_nonempty_selection_recomputes_only_frozen_candidates_without_scores(monkeypatch):
    from examples.alpha_mining_agent import run_evolution_loop

    base = generate_symbol_kines("SHFE.RB9999", "15分钟", "20240102", "20240105", seed=29)
    higher = base.iloc[1::2].reset_index(drop=True)
    calls = []

    def compute(frame, signal_id, timeframe):
        calls.append((signal_id, timeframe))
        return "test", np.ones(len(frame), dtype=float)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Holdout must not evaluate or rank candidates")

    monkeypatch.setattr(run_evolution_loop, "compute_research_signal", compute)
    monkeypatch.setattr(run_evolution_loop, "audit_research_candidate", forbidden)
    selected = ["AC01L@30m"]
    units = holdout._selected_units({"15m": base, "30m": higher}, base, selected, SimpleNamespace(), SimpleNamespace())
    assert calls == [("AC01L", "30m")]
    assert list(units) == selected
    assert len(units[selected[0]]) == len(base)
    assert units[selected[0]][0] == 0  # The first higher-timeframe bar has not yet completed.
    assert np.all(units[selected[0]][1:] == 1)
