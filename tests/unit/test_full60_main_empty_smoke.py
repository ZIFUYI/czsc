"""Run the full report/export path when the formal admission rejects every candidate."""

import dataclasses
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from czsc.mock import generate_symbol_kines
from examples.multi_factor_cta.audit_protocol import (
    ArtifactFingerprint,
    build_fingerprint,
    canonical_json,
    read_verified_pickle,
)
from examples.multi_factor_cta.forensic_validator import summarize_execution_pooled_scores

SOURCE = Path(__file__).resolve().parents[2] / "examples/results/intraday_cta_ablation_20261017"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
study = importlib.import_module("run_full_60_commodities_study")


def test_full60_main_all_rejected_writes_all_variants_ledgers_csvs_and_html(monkeypatch, tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    for subdir in ("data_5m", "data_15m_60"):
        (source / subdir).mkdir(parents=True)
    source_code = source / "frozen_entry.py"
    source_code.write_text("# immutable smoke-test implementation input\n")
    ids, frames, specs, raw_paths = ["ag", "ao"], {}, {}, []
    for number, aid in enumerate(ids):
        frame = generate_symbol_kines(aid, "日线", "20240101", "20260930", seed=71 + number)
        frame["dt"] = pd.to_datetime(frame.dt).dt.normalize() + pd.Timedelta(hours=15)
        frame["atr14"] = (frame.high - frame.low).rolling(14, min_periods=1).mean()
        raw = source / "data_15m_60" / f"{aid}_15m.parquet"
        frame.to_parquet(raw)
        frame.to_parquet(source / "data_5m" / f"{aid}_5m.parquet")
        raw_paths.append(raw)
        specs[aid] = dataclasses.replace(study.SPECS_60[aid], data_path=raw)
        frames[specs[aid].symbol] = {"15m": frame, "1d": frame}
    sectors = {spec.sector: [aid for aid in ids if specs[aid].sector == spec.sector] for spec in specs.values()}
    for key, value in {
        "SOURCE_DIR": source,
        "OUT_DIR": output,
        "CACHE_60_DIR": output / "cache_60",
        "DATA_15M_60_DIR": source / "data_15m_60",
        "UNIVERSE_60_SPECS": list(specs.values()),
        "SPECS_60": specs,
        "IDS_60": ids,
        "SECTORS_60": sectors,
        "RAW_60_META": [row for row in study.RAW_60_META if row[0] in ids],
    }.items():
        monkeypatch.setattr(study, key, value)
    empty_scores = pd.DataFrame(columns=["fold", "asset", "uid", "tf", "archetype"])

    def prepared(selected_specs, *, source_dir, output_dir):
        return {
            "frames": {spec.symbol: frames[spec.symbol] for spec in selected_specs},
            "units": {spec.name: {} for spec in selected_specs},
            "scores": empty_scores,
            "pool": summarize_execution_pooled_scores(empty_scores),
            "fingerprint": build_fingerprint(
                source_paths=[source_code],
                data_paths=raw_paths,
                config={"asset_fees": {spec.name: spec.fee_rate for spec in selected_specs}},
            ),
            "settings": {},
        }

    monkeypatch.setattr(study, "prepare_study_inputs", prepared)
    bundles = study.ensure_full60_oos_bundles(force_rebuild=True)
    monkeypatch.setattr(study, "ensure_full60_oos_bundles", lambda **kwargs: bundles)

    # Keep all twelve Full-60 allocation variants, native accounting, report builders,
    # provenance checks and file writers live; only replace prepared market inputs.
    study.main()

    summary = pd.read_csv(output / "full60_ablation_comparison.csv")
    assert len(summary) == 12
    assert summary["experiment"].nunique() == 12
    assert np.isfinite(summary[["net_ret_pct", "sharpe", "max_dd_pct", "ann_turnover"]]).all().all()
    assert summary[["net_ret_pct", "sharpe", "max_dd_pct", "ann_turnover"]].eq(0).all().all()
    assert summary.execution_evidence.eq("no_executed_positions").all()
    paths = sorted((output / "ledgers").glob("*.pkl"))
    assert len(paths) == 24
    for path in paths:
        sidecar = json.loads(path.with_name(path.name + ".provenance.json").read_text())
        fingerprint = ArtifactFingerprint(sidecar["fingerprint"], canonical_json(sidecar["manifest"]))
        ledger = read_verified_pickle(path, fingerprint)
        assert ledger["bar_net_returns"].eq(0).all()
        assert ledger["weights_df"].eq(0).all().all()
        assert ledger["execution_prices_df"].notna().any().all()
        assert ledger["fold"] in {"Fold1_2025", "Fold2_2026"}
    for name in (
        "full60_universe_summary",
        "full60_sector_attribution",
        "full60_symbol_attribution",
        "full60_satellite_sop_admission",
    ):
        assert not pd.read_csv(output / f"{name}.csv").empty
    html = (output / "full60_cta_expansion_report.html").read_text()
    assert "Plotly.newPlot" in html
    assert "Full60_ZeroHardcode_ClusterConserved_3x" in html
