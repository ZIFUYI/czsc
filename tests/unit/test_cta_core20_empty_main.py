"""Run the real Core-20 entrypoint through every report with no admitted signal."""

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.hypothesis_bank import enrich_kline_features
from examples.multi_factor_cta.audit_protocol import build_fingerprint, read_verified_pickle
from examples.multi_factor_cta.forensic_validator import summarize_execution_pooled_scores


def _mock_window(frequency):
    frames = [
        generate_symbol_kines("KQ.m@SHFE.ag", frequency, "20241223", "20250106", seed=11),
        generate_symbol_kines("KQ.m@SHFE.ag", frequency, "20251229", "20260109", seed=17),
    ]
    frame = pd.concat(frames, ignore_index=True).rename(columns={"vol": "volume"})
    frame["date"] = pd.to_datetime(frame.dt).dt.date
    return frame


@pytest.mark.parametrize("empty_evidence", [True, False], ids=["empty_inventory", "all_candidates_rejected"])
def test_core20_empty_admission_main_writes_all_reports_and_real_zero_ledgers(tmp_path, monkeypatch, empty_evidence):
    source_module_dir = Path(__file__).resolve().parents[2] / "examples/results/intraday_cta_ablation_20261017"
    monkeypatch.syspath_prepend(str(source_module_dir))
    study = importlib.import_module("run_multitf_incremental_study")
    holdout = importlib.import_module("examples.multi_factor_cta.run_frozen_holdout")
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "data_5m").mkdir(parents=True)
    raw5, frame15 = _mock_window("5分钟"), _mock_window("15分钟")
    raw5_path, raw15_path = source / "data_5m/ag_5m.parquet", source / "ag_15m.parquet"
    raw5.to_parquet(raw5_path, index=False)
    frame15.to_parquet(raw15_path, index=False)
    daily = _mock_window("日线")
    frames = {"KQ.m@SHFE.ag": {"15m": enrich_kline_features(frame15), "1d": enrich_kline_features(daily)}}
    manifest_path = source / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "intraday": [
                    {
                        "id": "ag",
                        "asset_group": "commodity",
                        "symbol": "KQ.m@SHFE.ag",
                        "sector": "precious_metals",
                        "fee_rate": 0.0002,
                        "has_night_session": True,
                        "path": str(raw15_path),
                    }
                ]
            }
        )
    )
    implementation = source / "fixture.py"
    implementation.write_text("# Provenance for the mock-only orchestration regression.\n")
    fingerprint = build_fingerprint(source_paths=[implementation], data_paths=[raw5_path, raw15_path], config={})
    scores = pd.DataFrame(columns=["fold", "asset", "uid", "sig_id", "tf", "archetype"])
    pooled = summarize_execution_pooled_scores(scores)
    prepared = {"fingerprint": fingerprint}
    monkeypatch.setattr(
        study,
        "ensure_multitf_and_hetero_caches",
        lambda _specs: (
            scores.loc[scores.tf != "5m"].copy(),
            pooled.loc[pooled.tf != "5m"].copy(),
            scores.loc[scores.tf != "5m"].copy(),
            pooled.loc[pooled.tf != "5m"].copy(),
            scores.loc[scores.tf == "5m"].copy(),
            pooled.loc[pooled.tf == "5m"].copy(),
            {"ag": {}},
            frames,
            prepared,
        ),
    )
    folds = (
        {
            "id": "Fold1_2025",
            "train_start": "2024-12-23",
            "cutoff": "2024-12-31 23:59:59",
            "oos_start": "2025-01-02",
            "oos_end": "2025-01-06 23:59:59",
        },
        {
            "id": "Fold2_2026",
            "train_start": "2024-12-23",
            "cutoff": "2025-12-31 23:59:59",
            "oos_start": "2026-01-05",
            "oos_end": "2026-01-09 23:59:59",
        },
    )
    if not empty_evidence:
        records = []
        for fold in folds:
            enriched = frames["KQ.m@SHFE.ag"]["15m"]
            training = enriched.loc[pd.to_datetime(enriched.dt) <= pd.Timestamp(fold["cutoff"])].reset_index(drop=True)
            evaluation = study.evaluate_candidate_on_fold(
                training, np.zeros(len(training)), "15m", 0.0002, pd.Timestamp(fold["cutoff"])
            )
            for uid in ("AC01L@30m", "H5M_CLV01L@5m"):
                signal, tf = uid.split("@")
                records.append(
                    {
                        "fold": fold["id"],
                        "asset": "ag",
                        "uid": uid,
                        "sig_id": signal,
                        "tf": tf,
                        "archetype": study.ARCH_A,
                        "pool_group": "commodity:precious_metals",
                        **evaluation,
                    }
                )
        scores = pd.DataFrame(records)
        pooled = summarize_execution_pooled_scores(scores)
    for key, value in {
        "OUT_DIR": output,
        "SOURCE_DIR": source,
        "MANIFEST_PATH": manifest_path,
        "CORE_20_IDS": ("ag",),
        "FOLDS": folds,
    }.items():
        monkeypatch.setattr(study, key, value)
    exported = []
    monkeypatch.setattr(
        holdout, "export_frozen_selection", lambda _path, selections, *_args, **_kwargs: exported.append(selections)
    )

    # Keep actual selectors, native structure, synthesis, execution, tables and Plotly export.
    study.main()

    assert exported == [{"ag": []}]
    comparison = pd.read_csv(output / "multitf_ablation_comparison.csv")
    assert len(comparison) == 16
    for column in ("net_ret_pct", "cost_ret_pct", "avg_factors_per_asset", "ann_turnover"):
        assert comparison[column].eq(0).all()
    assert comparison.oos_days.gt(0).all()
    ledgers = list((output / "ledgers").glob("*.pkl"))
    assert len(ledgers) == 24
    for path in ledgers:
        ledger = read_verified_pickle(path, fingerprint)
        assert len(ledger["bar_net_returns"]) > 0
        assert ledger["bar_net_returns"].eq(0).all()
        assert ledger["weights_df"].eq(0).all().all()
    for name in (
        "multitf_funnel_stage_summary.csv",
        "multitf_structure_exit_ablation.csv",
        "multitf_gate23_rejected_signals.csv",
        "multitf_gate23_rejected_summary.csv",
        "multitf_incremental_admitted_details.csv",
        "multitf_incremental_admitted_attribution.csv",
        "multitf_symbol_attribution.csv",
        "multitf_sector_attribution.csv",
    ):
        assert (output / name).is_file()
        pd.read_csv(output / name)  # Empty tables must retain headers and remain parseable.
    html = (output / "multitf_cta_incremental_report.html").read_text()
    assert html.count("Plotly.newPlot") == 2
    assert "ExpE_StructExit_HysteresisRunner" in html
