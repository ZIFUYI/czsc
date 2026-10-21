"""Verify all frozen CTA signal weights are invariant to appending future bars.

The input must be a verified prepared_evidence.pkl. Feature frames are held
fixed and sliced by their completed-bar timestamps; this is a signal-layer
prefix test, not an independent proof of the upstream feature construction.
No admission score or holdout performance is calculated.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from examples.multi_factor_cta.audit_protocol import (
    UnverifiedArtifactError,
    build_fingerprint,
    canonical_json,
    file_sha256,
    read_verified_pickle,
    validate_output_directory,
    write_verified_artifact,
)
from examples.multi_factor_cta.run_frozen_holdout import _selected_units, fingerprint_from_verified_artifact


def compare_unit_prefixes(tf_frames, raw5, candidates, generator, *, cutoff, cached_units=None) -> pd.DataFrame:
    """Compare fresh full/prefix weights and, when supplied, the verified full-run cache."""
    candidates = tuple(candidates)
    boundary = pd.Timestamp(cutoff)
    shortened = {}
    for tf, frame in tf_frames.items():
        dts = pd.DatetimeIndex(pd.to_datetime(frame.dt))
        if dts.has_duplicates or not dts.is_monotonic_increasing:
            raise ValueError(f"{tf} timestamps must be ordered and unique")
        shortened[tf] = frame.loc[dts <= boundary].copy().reset_index(drop=True)
        if shortened[tf].empty:
            raise ValueError(f"Cutoff leaves no completed {tf} bars")
    micro_prefix = raw5.loc[pd.to_datetime(raw5.dt) <= boundary].copy().reset_index(drop=True)
    full = generator(tf_frames, raw5, candidates)
    prefix = generator(shortened, micro_prefix, candidates)
    if set(full) != set(candidates) or set(prefix) != set(candidates):
        raise ValueError("Generated candidates differ from the complete frozen inventory")
    if cached_units is not None and set(cached_units) != set(candidates):
        raise ValueError("Verified cache differs from the complete frozen inventory")
    full_dt, prefix_dt = pd.DatetimeIndex(tf_frames["15m"].dt), pd.DatetimeIndex(shortened["15m"].dt)
    records = []

    def compare(uid, left, right, dts, comparison):
        left, right = np.asarray(left), np.asarray(right)
        if left.shape != right.shape or left.shape != (len(dts),):
            raise ValueError(f"{uid}: weights do not align to the completed 15m grid")
        finite = np.isfinite(left) & np.isfinite(right)
        same = finite & (left == right)
        changed = np.flatnonzero(~same)
        delta = np.abs(left.astype(float) - right.astype(float))
        records.append(
            {
                "uid": uid,
                "comparison": comparison,
                "bars_compared": len(dts),
                "changed_bars": len(changed),
                "nonfinite_bars": int((~finite).sum()),
                "max_abs_difference": float(delta[finite].max()) if finite.any() else None,
                "first_difference_dt": str(dts[changed[0]]) if len(changed) else None,
                "passed": bool(same.all()),
            }
        )

    for uid in candidates:
        compare(uid, full[uid][: len(prefix_dt)], prefix[uid], prefix_dt, "future_prefix")
        if cached_units is not None:
            compare(uid, full[uid], cached_units[uid], full_dt, "verified_cache_recompute")
    return pd.DataFrame(records)


def generate_frozen_units(tf_frames, raw5, candidates, *, source_dir: Path):
    """Reuse the production dispatcher for the exact frozen candidate inventory."""
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    study = importlib.import_module("run_multitf_incremental_study")
    study.register_hetero_signal_specs()
    return _selected_units(tf_frames, raw5, list(candidates), study)


def validate_signal_prefixes(evidence: Path, output_dir: Path, *, asset="ag", cutoff="2025-12-31 23:59:59") -> dict:
    """Load verified source evidence, compare every frozen candidate and save exact results."""
    historical_fingerprint = fingerprint_from_verified_artifact(evidence)
    prepared = read_verified_pickle(evidence, historical_fingerprint)
    settings = historical_fingerprint.manifest()["config"]
    candidates = settings["protocol"]["candidates"]
    spec = next((item for item in settings["specs"] if item["name"] == asset), None)
    if spec is None:
        raise ValueError(f"Asset is absent from the frozen universe: {asset}")
    source_files = historical_fingerprint.manifest()["sources"]
    study_path = next(
        Path(item["path"]) for item in source_files if Path(item["path"]).name == "run_multitf_incremental_study.py"
    )
    source_dir = study_path.parent
    validate_output_directory(output_dir, (source_dir, evidence.parent))
    raw5_path = source_dir / "data_5m" / f"{asset}_5m.parquet"
    data_files = historical_fingerprint.manifest()["data"]
    if str(raw5_path.resolve()) not in {item["path"] for item in data_files}:
        raise UnverifiedArtifactError("The 5m microstructure source is not covered by the verified evidence")
    fingerprint = build_fingerprint(
        source_paths=[item["path"] for item in source_files]
        + [Path(__file__), Path(_selected_units.__code__.co_filename)],
        data_paths=[item["path"] for item in data_files] + [evidence],
        config={
            "historical_fingerprint": historical_fingerprint.sha256,
            "asset": asset,
            "cutoff": cutoff,
            "candidates": candidates,
            "comparison": "exact_float32_signal_weight_prefix",
        },
    )
    print(f"[prefix] {asset}: {len(candidates)} candidates, cutoff={cutoff}", flush=True)
    records = compare_unit_prefixes(
        prepared["frames"][spec["symbol"]],
        pd.read_parquet(raw5_path),
        candidates,
        lambda frames, raw5, frozen: generate_frozen_units(frames, raw5, frozen, source_dir=source_dir),
        cutoff=cutoff,
        cached_units=prepared["units"][asset],
    )
    # Recheck the original provenance after calculation so a source edit cannot be certified.
    if fingerprint_from_verified_artifact(evidence) != historical_fingerprint:
        raise UnverifiedArtifactError("Historical evidence inputs changed during prefix validation")
    for item in fingerprint.manifest()["sources"] + fingerprint.manifest()["data"]:
        if file_sha256(item["path"]) != item["sha256"]:
            raise UnverifiedArtifactError("Validation inputs changed during prefix comparison")
    summary = {
        "asset": asset,
        "cutoff": cutoff,
        "candidates": len(candidates),
        "prefix_bars": int(records.loc[records.comparison == "future_prefix", "bars_compared"].iloc[0]),
        "failed_comparisons": int((~records.passed).sum()),
        "passed": bool(records.passed.all()),
        "historical_fingerprint": historical_fingerprint.sha256,
        "scope": "all frozen signal weights, including heterogeneous/stage2/5m; verified feature frames held fixed",
        "limitation": "This does not independently regenerate or prove upstream cross-sectional feature causality",
        "holdout_performance_computed": False,
    }
    for name, payload in (
        ("prefix_comparison.csv", records.to_csv(index=False).encode()),
        ("prefix_summary.json", canonical_json(summary)),
    ):
        write_verified_artifact(
            output_dir / name, payload, fingerprint, output_dir=output_dir, readonly_roots=(source_dir, evidence.parent)
        )
    return summary


def main():
    """Validate a completed frozen evidence artifact without changing its source tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset", default="ag")
    parser.add_argument("--cutoff", default="2025-12-31 23:59:59")
    args = parser.parse_args()
    result = validate_signal_prefixes(args.evidence, args.output_dir, asset=args.asset, cutoff=args.cutoff)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
