"""Rebuild CTA study evidence from frozen real bars, never from legacy scores.

The original score CSV supplies candidate names only. Every feature, decision,
completed execution event, admission statistic and pooled test is recomputed.
The holdout is physically excluded before feature calculation and selection.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from examples.alpha_mining_agent.hypothesis_bank import ALL_SIGNAL_SPECS
from examples.alpha_mining_agent.run_evolution_loop import prepare_research_pool
from examples.multi_factor_cta.audit_protocol import (
    FoldBoundary,
    FrozenResearchProtocol,
    UnverifiedArtifactError,
    build_fingerprint,
    freeze_protocol,
    read_verified_pickle,
    write_verified_artifact,
)
from examples.multi_factor_cta.data_engine import AssetSpec, CrossAssetDataEngine
from examples.multi_factor_cta.run_frozen_holdout import _selected_units

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_COLUMNS = ("events", "event_returns", "pool_audits", "pool_passed_assets")


def display_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove nested evidence only from presentation CSVs, retaining it in verified artifacts."""
    return frame.drop(columns=list(EVIDENCE_COLUMNS), errors="ignore")


def study_fingerprint(specs: list[AssetSpec], source_dir: Path, config: dict):
    """Include all research sources, the native binary, and every actual input file."""
    import czsc._native as native

    # Downloading does not compute historical features. The frozen replay module
    # also owns the shared candidate dispatcher, so its source must be included.
    separate_holdout_sources = {"download_holdout.py"}
    paths = [
        path
        for path in [*ROOT.glob("examples/alpha_mining_agent/*.py"), *ROOT.glob("examples/multi_factor_cta/*.py")]
        if path.name not in separate_holdout_sources
    ]
    paths += list(source_dir.glob("*.py")) + [Path(native.__file__), ROOT / "uv.lock", ROOT / "Cargo.lock"]
    data = [spec.data_path for spec in specs] + [source_dir / "data_5m" / f"{spec.name}_5m.parquet" for spec in specs]
    data += [source_dir / "multitf_factor_scores.csv"]
    for spec in specs:
        for companion in (
            spec.data_path.parent / "calendar.parquet",
            spec.data_path.with_name("warmup_" + spec.data_path.name),
        ):
            if companion.exists():
                data.append(companion)
    return build_fingerprint(source_paths=paths, data_paths=data, config=config)


def _write(path: Path, value, fingerprint, output_dir: Path, source_dir: Path) -> None:
    write_verified_artifact(
        path,
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
        fingerprint,
        output_dir=output_dir,
        readonly_roots=(source_dir,),
    )


def _load_asset(args):
    spec, cache_dir = args
    return spec.symbol, CrossAssetDataEngine(cache_dir).load_asset_multi_timeframes(spec, attach_czsc=True)


def _compute_asset_evidence(args):
    spec, tf_frames, _base_uids, source_dir, protocol = args
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    import run_multitf_incremental_study as study
    from run_intraday_ablation import evaluate_candidate_on_fold

    study.register_hetero_signal_specs()
    rows = []
    base = tf_frames["15m"]
    unit = _selected_units(
        tf_frames,
        pd.read_parquet(source_dir / "data_5m" / f"{spec.name}_5m.parquet"),
        list(protocol.candidates),
        study,
    )
    # Score every candidate on the same decision/execution grid as portfolio selection.
    for fold in study.FOLDS:
        mask = pd.to_datetime(base.dt).between(fold["train_start"], fold["cutoff"]).to_numpy()
        training = base.loc[mask].reset_index(drop=True)
        protocol.assert_selection_timestamps(training.dt.astype(str), fold_id=fold["id"])
        for uid, weight in unit.items():
            sid, tf = uid.split("@")
            evaluation = evaluate_candidate_on_fold(
                training, weight[mask], tf, spec.fee_rate, pd.Timestamp(fold["cutoff"])
            )
            rows.append(
                {
                    "fold": fold["id"],
                    "asset": spec.name,
                    "uid": uid,
                    "sig_id": sid,
                    "tf": tf,
                    "archetype": ALL_SIGNAL_SPECS[sid].archetype,
                    "sector": spec.sector,
                    **{key: value for key, value in evaluation.items() if key != "event_returns"},
                    "pool_group": f"commodity:{spec.sector.lower()}",
                    "microstructure_group": "commodity_futures",
                }
            )
    return spec.name, unit, rows


def prepare_study_inputs(specs: list[AssetSpec], *, source_dir: Path, output_dir: Path) -> dict:
    """Produce fresh multi-timeframe decisions and complete training-fold evidence."""
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    import run_multitf_incremental_study as study

    study.register_hetero_signal_specs()
    # Candidate inventory is frozen without reading historical performance columns.
    base_uids = tuple(sorted(pd.read_csv(source_dir / "multitf_factor_scores.csv", usecols=["uid"]).uid.unique()))
    core_hetero = {u for u in study.HETERO_SIGNAL_META if u.split("@")[0] not in getattr(study, "STAGE2_SIG_IDS", ())}
    candidates = tuple(sorted(set(base_uids) | core_hetero))
    protocol = FrozenResearchProtocol(
        candidates=candidates,
        variants=(
            "baseline",
            "all_strict",
            "incremental",
            "fast_slow",
            "factor_exit",
            "asset_exit",
            "full_exit",
            "reentry",
            "partial_exit",
            "hysteresis_runner",
            "micro_5m",
        ),
        folds=tuple(
            FoldBoundary(f["id"], f["train_start"], f["cutoff"], f["oos_start"], f["oos_end"]) for f in study.FOLDS
        ),
    )
    freeze_protocol(protocol, output_dir=output_dir, readonly_roots=(source_dir,))
    specs = [dataclasses.replace(spec, eligible_tfs=("1d", "60m", "30m", "15m")) for spec in specs]
    settings = {
        "protocol": protocol.as_dict(),
        "specs": json.loads(json.dumps([dataclasses.asdict(s) for s in specs], default=str)),
        "execution": "shared_rust_next_open",
        "evidence_version": 2,
        "scope": "continuous_reference_price_only",
        "primary_variant": "hysteresis_runner",
    }
    fingerprint = study_fingerprint(specs, source_dir, settings)
    cache_path = output_dir / "prepared_evidence.pkl"
    try:
        prepared = read_verified_pickle(cache_path, fingerprint)
        print(f"[verified] {len(specs)} assets: {cache_path}", flush=True)
        return prepared
    except UnverifiedArtifactError:
        pass

    start = time.monotonic()
    print(f"[rebuild] {len(specs)} assets; {len(candidates)} candidates; loading real bars", flush=True)
    # Source data in this historical run must not include the sealed holdout.
    for spec in specs:
        raw = pd.read_parquet(spec.data_path)
        if (pd.to_datetime(raw.dt) >= pd.Timestamp(protocol.holdout_start)).any():
            raise ValueError(f"Historical study source contains sealed holdout: {spec.data_path}")
    workers = min(len(specs), max(1, int(os.getenv("CZSC_CTA_WORKERS", "4"))))
    # Pool-wide causal features must be attached after every asset is loaded.
    # Only independent per-asset operations run in separate processes.
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        mapper = pool.map if pool is not None else map
        # Native feature files independently validate their prices, signal config,
        # implementation, environment and binary. A report change must not force
        # unrelated native features to be regenerated; all admission evidence is
        # still protected by the complete study fingerprint above.
        cache_dir = output_dir / "native_cache"
        frames = dict(mapper(_load_asset, [(spec, cache_dir) for spec in specs]))
        metadata = {
            s.symbol: {
                "asset_group": "commodity",
                "sector": s.sector,
                "microstructure_group": "commodity_futures",
                "fee_rate": s.fee_rate,
            }
            for s in specs
        }
        frames = prepare_research_pool(frames, metadata=metadata)
        print(f"[loaded] {len(specs)} assets; generating executable evidence with {workers} workers", flush=True)
        jobs = [(spec, frames[spec.symbol], base_uids, source_dir, protocol) for spec in specs]
        units, rows = {}, []
        for number, (aid, unit, evidence_rows) in enumerate(mapper(_compute_asset_evidence, jobs), 1):
            units[aid] = unit
            rows.extend(evidence_rows)
            print(
                f"[evidence {number}/{len(specs)}] {aid}: {len(unit)} candidates ({time.monotonic() - start:.1f}s)",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    all_scores = pd.DataFrame(rows)
    base_scores = all_scores.loc[all_scores.uid.isin(base_uids)].copy()
    scores_5m = all_scores.loc[all_scores.tf == "5m"].copy()
    scores = all_scores.loc[all_scores.tf != "5m"].copy()
    # Each pool audit is independent per fold/unit; the base inventory is an
    # exact subset and must not repeat identical event-table audits.
    pooled_scores = study.summarize_pooled_scores(scores)
    prepared = {
        "frames": frames,
        "units": units,
        "scores": scores,
        "pool": pooled_scores,
        "base_scores": base_scores,
        "base_pool": pooled_scores.loc[pooled_scores.uid.isin(base_uids)].copy(),
        "scores_5m": scores_5m,
        "pool_5m": study.summarize_pooled_scores(scores_5m),
        "fingerprint": fingerprint,
        "settings": settings,
    }
    if study_fingerprint(specs, source_dir, settings) != fingerprint:
        raise ValueError("Source or data changed during evidence generation; rerun the frozen implementation")
    _write(cache_path, prepared, fingerprint, output_dir, source_dir)
    for name, value in (
        ("multitf_grid15m_frames.pkl", frames),
        ("multitf_grid1d_frames.pkl", {s.symbol: frames[s.symbol]["1d"] for s in specs}),
        ("multitf_unit_grid_cache.pkl", units),
    ):
        _write(output_dir / name, value, fingerprint, output_dir, source_dir)
    for name, value in (
        ("multitf_factor_scores", base_scores),
        ("multitf_factor_pooled_summary", prepared["base_pool"]),
        ("multitf_hetero_factor_scores", all_scores),
        ("multitf_hetero_pooled_summary", pd.concat([prepared["pool"], prepared["pool_5m"]])),
    ):
        display_table(value).to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "run_provenance.json").write_text(json.dumps(fingerprint.manifest(), ensure_ascii=False, indent=2))
    print(f"[rebuilt] complete evidence saved in {time.monotonic() - start:.1f}s", flush=True)
    return prepared
