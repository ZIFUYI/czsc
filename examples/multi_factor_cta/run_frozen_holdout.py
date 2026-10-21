"""Execute one frozen E4c strategy on sealed real bars, without any selection step."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from examples.multi_factor_cta.audit_protocol import (
    ArtifactFingerprint,
    UnverifiedArtifactError,
    build_fingerprint,
    canonical_json,
    file_sha256,
    read_verified_artifact,
    read_verified_pickle,
    validate_output_directory,
    write_verified_artifact,
)
from examples.multi_factor_cta.data_engine import AssetClass, AssetSpec, CrossAssetDataEngine, ExecutionMode
from examples.multi_factor_cta.synthesis_engine import (
    account_portfolio_execution,
    assign_futures_trading_day,
    build_asset_execution_inputs,
)

PRIMARY_VARIANT = "ExpE_StructExit_HysteresisRunner"
SELECTION_FOLD = "Fold2_2026"
TRAINING_START = "2024-01-02 00:00:00"
TRAINING_END = "2025-12-31 23:59:59"
HOLDOUT_START = "2026-10-08"
HOLDOUT_END = "2026-10-20"
PRIMARY_PARAMETERS = {
    "use_factor_exit": True,
    "use_asset_gate": True,
    "asset_trail_activation": 0.016,
    "asset_min_hold": 3,
    "asset_cooldown": 5,
    "asset_reentry_mode": "bi_or_peak",
    "asset_partial_exit_ratio": 0.35,
    "asset_breakeven_buffer": 0.0005,
    "asset_runner_trail_mult": 1.5,
    "asset_use_fx_exit": True,
}


def fingerprint_from_verified_artifact(path: Path) -> ArtifactFingerprint:
    """Rehash current sources and historical data before trusting an artifact's provenance."""
    sidecar = path.with_name(path.name + ".provenance.json")
    try:
        manifest = json.loads(sidecar.read_text())["manifest"]
        expected = build_fingerprint(
            source_paths=[item["path"] for item in manifest["sources"]],
            data_paths=[item["path"] for item in manifest["data"]],
            config=manifest["config"],
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise UnverifiedArtifactError("Selection requires complete verified historical provenance") from error
    read_verified_artifact(path, expected)
    return expected


def export_frozen_selection(
    path: Path,
    selections: dict[str, list[str]],
    specs: list[AssetSpec],
    fingerprint: ArtifactFingerprint | None = None,
    *,
    evidence_path: Path,
    output_dir: Path,
    source_dir: Path,
) -> Path:
    """Freeze Fold2 training selections and the primary variant before opening holdout results."""
    expected = fingerprint_from_verified_artifact(evidence_path)
    if fingerprint is not None and fingerprint != expected:
        raise UnverifiedArtifactError("Selection and training evidence fingerprints differ")
    if set(selections) != {spec.name for spec in specs}:
        raise ValueError("Selection export must include every asset, including rejected empty lists")
    protocol = expected.manifest()["config"]["protocol"]
    training = next((f for f in protocol["folds"] if f["fold_id"] == SELECTION_FOLD), None)
    if training is None or training["train_end"] != TRAINING_END or training["train_start"] != TRAINING_START:
        raise ValueError("Primary holdout selection must use the frozen 2024–2025 Fold2 training window")
    candidates = set(protocol["candidates"])
    if any(uid not in candidates for selected in selections.values() for uid in selected):
        raise ValueError("Selection contains a candidate outside the frozen inventory")
    payload = {
        "variant": PRIMARY_VARIANT,
        "selection_fold": SELECTION_FOLD,
        "training_start": TRAINING_START,
        "training_end": TRAINING_END,
        "parameters": PRIMARY_PARAMETERS,
        "deadzone": 0.08,
        "holdout_start": HOLDOUT_START,
        "holdout_end_trading_date": HOLDOUT_END,
        "selections": selections,
        "specs": json.loads(json.dumps([dataclasses.asdict(spec) for spec in specs], default=str)),
        "training_evidence_path": str(evidence_path.resolve()),
        "training_evidence_sha256": file_sha256(evidence_path),
        "execution_scope": "continuous_reference_only",
    }
    return write_verified_artifact(
        path, pickle.dumps(payload), expected, output_dir=output_dir, readonly_roots=(source_dir,)
    )


def load_frozen_selection(path: Path) -> tuple[dict, ArtifactFingerprint]:
    """Load the exact exported strategy, rejecting code, evidence or variant drift."""
    fingerprint = fingerprint_from_verified_artifact(path)
    record = read_verified_pickle(path, fingerprint)
    for key, expected in (
        ("variant", PRIMARY_VARIANT),
        ("selection_fold", SELECTION_FOLD),
        ("training_start", TRAINING_START),
        ("training_end", TRAINING_END),
        ("parameters", PRIMARY_PARAMETERS),
        ("holdout_start", HOLDOUT_START),
        ("holdout_end_trading_date", HOLDOUT_END),
        ("deadzone", 0.08),
    ):
        if record.get(key) != expected:
            raise ValueError(f"Frozen primary strategy contract changed: {key}")
    if file_sha256(record["training_evidence_path"]) != record["training_evidence_sha256"]:
        raise UnverifiedArtifactError("Training evidence changed after selection freeze")
    return record, fingerprint


def restrict_to_holdout(frame: pd.DataFrame, start: str = HOLDOUT_START, end: str = HOLDOUT_END) -> pd.DataFrame:
    """Use trading dates, excluding the last calendar day's next-session night bars."""
    dates = assign_futures_trading_day(pd.to_datetime(frame.dt))
    keep = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return frame.loc[keep].copy().reset_index(drop=True)


def concatenate_raw_history(history: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    """Join raw price units before common normalization, rejecting overlap or contaminated history."""
    history, holdout = history.copy(), restrict_to_holdout(holdout)
    history["dt"] = pd.to_datetime(history.dt)
    holdout["dt"] = pd.to_datetime(holdout.dt)
    if (history.dt >= pd.Timestamp("2026-10-01")).any():
        raise ValueError("Historical source already contains the sealed holdout")
    if history.dt.duplicated().any() or holdout.dt.duplicated().any():
        raise ValueError("Raw input timestamps must be unique")
    return pd.concat([history, holdout], ignore_index=True).sort_values("dt").reset_index(drop=True)


def execute_zero_selection(df5_map: dict[str, pd.DataFrame], specs: dict[str, AssetSpec]) -> dict:
    """Produce the actual shared-Rust zero-position ledger on every observed holdout bar."""
    returns, positions, prices = {}, {}, {}
    for aid in specs:
        frame = restrict_to_holdout(df5_map[aid])
        if frame.empty:
            raise ValueError(f"No observed holdout bars for {aid}")
        returns[aid], positions[aid] = build_asset_execution_inputs(frame, np.zeros(len(frame)))
        price = pd.Series(frame["open"].to_numpy(dtype=float), index=pd.DatetimeIndex(frame.dt))
        price.loc[price.index[-1] + pd.Timedelta(microseconds=1)] = float(frame["close"].iloc[-1])
        prices[aid] = price
    raw_returns = pd.DataFrame(returns).sort_index()
    observed = raw_returns.notna()
    weights = pd.DataFrame(positions).reindex_like(raw_returns).fillna(0.0)
    ledger = account_portfolio_execution(
        raw_returns.fillna(0.0), weights, {aid: spec.fee_rate for aid, spec in specs.items()}, tradable_mask=observed
    )
    ledger["weights_df"] = weights
    ledger["returns_df"] = raw_returns.fillna(0.0)
    ledger["execution_prices_df"] = pd.DataFrame(prices).reindex_like(raw_returns)
    ledger["tradable_mask"] = observed
    ledger["fee_rates"] = {aid: float(spec.fee_rate) for aid, spec in specs.items()}
    ledger["daily_returns"] = (
        ledger["bar_net_returns"].groupby(assign_futures_trading_day(ledger["bar_net_returns"].index)).sum()
    )
    return ledger


def _selected_units(
    tf_frames: dict, df5: pd.DataFrame, selected: list[str], study, stage2=None
) -> dict[str, np.ndarray]:
    """Recompute only frozen decisions; no scores, p-values, gates or rankings are called."""
    from examples.alpha_mining_agent.run_evolution_loop import compute_research_signal
    from examples.multi_factor_cta.atomic_factors import compute_atomic_signal
    from examples.multi_factor_cta.data_engine import map_tf_weight_to_base_grid

    base, units, specialized = tf_frames["15m"], {}, {}
    extended_cache = {}

    def extension():
        nonlocal stage2
        if stage2 is None:
            try:
                stage2 = importlib.import_module("precompute_stage2_60_cache")
            except ModuleNotFoundError as error:
                if error.name != "precompute_stage2_60_cache":
                    raise
                raise ValueError(
                    "Frozen inventory requests an optional Stage-2 extension; "
                    "provide its source before rebuilding this inventory"
                ) from error
        return stage2

    def extended(tf, config_name):
        key = (tf, config_name)
        if key not in extended_cache:
            module = extension()
            extended_cache[key] = module.attach_sigs(tf_frames[tf], tf, getattr(module, config_name))
        return extended_cache[key]

    if any(uid.startswith("H1D_") for uid in selected):
        specialized.update(study.compute_hetero_1d_signals(tf_frames["1d"]))
        if any(uid.startswith("H1D_") and uid not in specialized for uid in selected):
            specialized.update(extension().compute_stage2_1d(extended("1d", "CFGS_1D")))
    if any(uid.startswith("H15M_") for uid in selected):
        specialized.update(study.compute_hetero_15m_signals(base))
        if any(uid.startswith("H15M_") and uid not in specialized for uid in selected):
            specialized.update(extension().compute_stage2_15m(extended("15m", "CFGS_15M")))
    if any(uid.endswith("@5m") for uid in selected):
        specialized.update(study.compute_all_5m_candidates(base, study.build_5m_micro_on_15m(df5, base)))
    for uid in selected:
        sid, tf = uid.split("@")
        if uid in specialized:
            weight = specialized[uid][1]
            source = tf_frames["1d"] if tf == "1d" else base
        elif sid.startswith(("H30M_", "H60M_")):
            source = extended(tf, "CFGS_INTRA")
            _, weight = compute_atomic_signal(source, sid.split("_", 1)[1], soft_gate=False)
        else:
            source = tf_frames[tf]
            _, weight = compute_research_signal(source, sid, tf)
        units[uid] = np.asarray(map_tf_weight_to_base_grid(source, weight, base), dtype=np.float32)
    return units


def run_holdout(selection_path: Path, *, data_dir: Path, output_dir: Path, source_dir: Path) -> dict:
    """Replay exactly the frozen E4c selection once, using raw history for causal warmup."""
    validate_output_directory(output_dir, (source_dir, data_dir, selection_path))
    frozen, historical_fingerprint = load_frozen_selection(selection_path)
    manifest = json.loads((data_dir / "download_manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("holdout_performance_computed") is not False:
        raise ValueError("Holdout requires a completed unopened download manifest")
    specs, frames5, merged_specs, data_paths = {}, {}, [], [selection_path, data_dir / "download_manifest.json"]
    selections = frozen["selections"]
    for row in frozen["specs"]:
        row = {
            **row,
            "data_path": Path(row["data_path"]),
            "eligible_tfs": tuple(row["eligible_tfs"]),
            "asset_class": AssetClass(row["asset_class"]),
            "execution_mode": ExecutionMode(row["execution_mode"]),
        }
        spec = AssetSpec(**row)
        specs[spec.name] = spec
        merged = {}
        for tf in ("15m", "5m"):
            metadata = manifest["assets"][spec.name]["frames"][tf]
            incoming = Path(metadata["path"])
            if file_sha256(incoming) != metadata["sha256"]:
                raise UnverifiedArtifactError(f"Downloaded holdout changed: {incoming}")
            historical = spec.data_path if tf == "15m" else source_dir / "data_5m" / f"{spec.name}_5m.parquet"
            raw = pd.read_parquet(historical)
            warmup = historical.with_name("warmup_" + historical.name)
            if warmup.exists():
                raw = pd.concat([pd.read_parquet(warmup), raw], ignore_index=True)
                data_paths.append(warmup)
            merged[tf] = concatenate_raw_history(raw, pd.read_parquet(incoming))
            data_paths.extend([historical, incoming])
        frames5[spec.name] = merged["5m"]
        raw_path = output_dir / "combined_raw" / f"{spec.name}_15m.parquet"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        merged["15m"].to_parquet(raw_path, index=False)
        merged_specs.append(dataclasses.replace(spec, data_path=raw_path, eligible_tfs=("1d", "60m", "30m", "15m")))
    if set(specs) != set(selections):
        raise ValueError("Frozen selections and asset specifications differ")
    settings = {
        "variant": PRIMARY_VARIANT,
        "parameters": PRIMARY_PARAMETERS,
        "deadzone": 0.08,
        "selection_sha256": file_sha256(selection_path),
        "historical_fingerprint": historical_fingerprint.sha256,
        "holdout_start": HOLDOUT_START,
        "holdout_end_trading_date": HOLDOUT_END,
    }
    fingerprint = build_fingerprint(
        source_paths=[item["path"] for item in historical_fingerprint.manifest()["sources"]]
        + [Path(__file__), Path(__file__).with_name("download_holdout.py")],
        data_paths=data_paths,
        config=settings,
    )
    freeze = output_dir / "holdout_frozen_inputs.json"
    if freeze.exists() and freeze.read_bytes() != fingerprint.canonical_manifest:
        raise ValueError("Holdout inputs changed; preserve the prior evaluation and use a new output directory")
    freeze.write_bytes(fingerprint.canonical_manifest)
    if not any(selections.values()):
        ledger = execute_zero_selection(frames5, specs)
    else:
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        study = importlib.import_module("run_multitf_incremental_study")
        full = importlib.import_module("run_full_60_commodities_study")
        from examples.multi_factor_cta.run_framework_demo import load_research_universe

        study.register_hetero_signal_specs()
        frames = load_research_universe(CrossAssetDataEngine(output_dir / "native_cache"), merged_specs)
        sp15, sp1 = full._structure_inputs(frames, specs, {(SELECTION_FOLD, aid): u for aid, u in selections.items()})
        targets = {}
        for aid, spec in specs.items():
            df15, selected = frames[spec.symbol]["15m"], selections[aid]
            if selected:
                units = _selected_units(frames[spec.symbol], frames5[aid], selected, study)
                target15 = study.synthesize_structure_aware_synergy(
                    df15, units, selected, spec, sp15[aid], sp1[aid], **PRIMARY_PARAMETERS
                )
            else:
                target15 = np.zeros(len(df15))
            targets[aid] = study.map_15m_target_to_5m_grid(df15, target15, frames5[aid])
        ledger = study.allocate_and_account_5m_grid(
            frames5,
            specs,
            targets,
            pd.Timestamp(HOLDOUT_START),
            pd.Timestamp(HOLDOUT_END) + pd.Timedelta(hours=15),
            deadzone=0.08,
        )
        ledger["daily_returns"] = (
            ledger["bar_net_returns"].groupby(assign_futures_trading_day(ledger["bar_net_returns"].index)).sum()
        )
    for record in fingerprint.manifest()["sources"] + fingerprint.manifest()["data"]:
        if file_sha256(record["path"]) != record["sha256"]:
            raise UnverifiedArtifactError("Frozen inputs changed during holdout execution")
    daily_std = float(ledger["daily_returns"].std(ddof=1))
    traded = bool((ledger["turnover_df"].abs() > 0).any().any())
    sharpe = float(ledger["daily_returns"].mean() / daily_std * np.sqrt(252)) if daily_std > 0 else None
    summary = {
        **settings,
        "evaluation_label": "sealed_holdout_validation",
        "execution_scope": "continuous_reference_only",
        "assets": len(specs),
        "admitted_asset_count": sum(bool(value) for value in selections.values()),
        "admitted_factor_count": sum(map(len, selections.values())),
        "bar_rows": len(ledger["bar_net_returns"]),
        "observed_trading_days": len(ledger["daily_returns"]),
        "net_return": float(ledger["bar_net_returns"].sum()),
        "cost_return": float(ledger["cost_returns_df"].sum().sum()),
        "max_gross_exposure": float(ledger["weights_df"].abs().sum(axis=1).max()),
        "nonzero_turnover_cells": int((ledger["turnover_df"].abs() > 0).sum().sum()),
        "annualized_sharpe": sharpe,
        "sharpe_status": "estimated_from_short_window"
        if sharpe is not None
        else "no_trades"
        if not traded
        else "zero_variance",
        "short_window_limit": "Nine trading days cannot establish long-run statistical or live-trading viability",
    }
    artifacts = {"holdout_ledger.pkl": pickle.dumps(ledger), "holdout_summary.json": canonical_json(summary)}
    for key in (
        "daily_returns",
        "bar_net_returns",
        "weights_df",
        "gross_returns_df",
        "cost_returns_df",
        "turnover_df",
        "returns_df",
        "execution_prices_df",
        "tradable_mask",
    ):
        artifacts[f"{key}.csv"] = ledger[key].to_csv().encode()
    for name, payload in artifacts.items():
        write_verified_artifact(
            output_dir / name,
            payload,
            fingerprint,
            output_dir=output_dir,
            readonly_roots=(source_dir, data_dir, selection_path),
        )
    return summary


def main() -> None:
    """Read the frozen selection artifact and execute its sole declared holdout variant."""
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=root / "examples/results/cta_research/holdout_data")
    parser.add_argument("--output-dir", type=Path, default=root / "examples/results/cta_research/holdout_primary")
    parser.add_argument("--source-dir", type=Path, default=root / "examples/results/intraday_cta_ablation_20261017")
    args = parser.parse_args()
    print(
        json.dumps(
            run_holdout(args.selection, data_dir=args.data_dir, output_dir=args.output_dir, source_dir=args.source_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
