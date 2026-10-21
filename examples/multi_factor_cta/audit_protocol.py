"""Frozen research boundaries and content-addressed artifacts for CTA experiments.

Historical 2025–2026 results are retrospective validation. A separate holdout
must never enter candidate, parameter, allocation or report-variant selection.
This module intentionally cannot import an unverified legacy pickle.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
HISTORICAL_VALIDATION_LABEL = "retrospective_oos_validation"
DEFAULT_HOLDOUT_START = "2026-10-01 00:00:00"


class UnverifiedArtifactError(ValueError):
    """An artifact has no valid provenance or belongs to a different experiment."""


def canonical_json(value: Any) -> bytes:
    """Encode finite JSON data deterministically; unsupported objects must be explicit."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: str | Path) -> str:
    """Hash every byte of a source, data file or native binary without loading it all."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactFingerprint:
    """Canonical provenance including all caller-declared sources, data and settings."""

    sha256: str
    canonical_manifest: bytes

    def manifest(self) -> dict[str, Any]:
        """Return a new manifest copy so callers cannot mutate the fingerprint."""
        return json.loads(self.canonical_manifest)


def build_fingerprint(
    *, source_paths: Iterable[str | Path], data_paths: Iterable[str | Path], config: Mapping[str, Any]
) -> ArtifactFingerprint:
    """Hash complete files and configuration, including native binaries when relevant.

    Callers must list the complete transitive research implementation and native
    extension, not merely the entry script. Directory paths are rejected so a
    silently incomplete directory scan cannot become experiment provenance.
    """

    def records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
        resolved = sorted({Path(path).resolve(strict=True) for path in paths})
        result = []
        for path in resolved:
            if not path.is_file():
                raise ValueError(f"Fingerprint inputs must be files: {path}")
            result.append({"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
        if not result:
            raise ValueError("Fingerprint requires nonempty source and data file sets")
        return result

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sources": records(source_paths),
        "data": records(data_paths),
        "config": dict(config),
    }
    payload = canonical_json(manifest)
    return ArtifactFingerprint(hashlib.sha256(payload).hexdigest(), payload)


def validate_output_directory(output_dir: str | Path, readonly_roots: Iterable[str | Path]) -> Path:
    """Require an independent output tree, resolving symlinks before overlap checks."""
    output = Path(output_dir).resolve()
    for value in readonly_roots:
        root = Path(value).resolve()
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError(f"Output must be separate from read-only inputs: {output} overlaps {root}")
    return output


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sidecar(path: Path) -> Path:
    return path.with_name(f"{path.name}.provenance.json")


def write_verified_artifact(
    path: str | Path,
    payload: bytes,
    fingerprint: ArtifactFingerprint,
    *,
    output_dir: str | Path,
    readonly_roots: Iterable[str | Path],
) -> Path:
    """Write a new run artifact and its payload checksum only inside its output tree."""
    output = validate_output_directory(output_dir, readonly_roots)
    target = Path(path).resolve()
    if not target.is_relative_to(output) or target == output:
        raise ValueError(f"Artifact must be inside its independent output directory: {target}")
    if hashlib.sha256(fingerprint.canonical_manifest).hexdigest() != fingerprint.sha256:
        raise UnverifiedArtifactError("Fingerprint manifest checksum mismatch")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint.sha256,
        "manifest": fingerprint.manifest(),
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
    }
    _atomic_write(target, payload)
    _atomic_write(_sidecar(target), canonical_json(metadata))
    return target


def read_verified_artifact(path: str | Path, expected: ArtifactFingerprint) -> bytes:
    """Reject legacy, changed or mismatched cached data before decoding its payload."""
    target = Path(path)
    try:
        metadata = json.loads(_sidecar(target).read_bytes())
        if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION:
            raise UnverifiedArtifactError(f"Unsupported artifact provenance: {target}")
        manifest = canonical_json(metadata["manifest"])
        if (
            metadata["fingerprint"] != expected.sha256
            or manifest != expected.canonical_manifest
            or hashlib.sha256(manifest).hexdigest() != expected.sha256
        ):
            raise UnverifiedArtifactError(f"Artifact source/data/configuration changed: {target}")
        payload = target.read_bytes()
        if hashlib.sha256(payload).hexdigest() != metadata["artifact_sha256"]:
            raise UnverifiedArtifactError(f"Artifact payload checksum mismatch: {target}")
        return payload
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise UnverifiedArtifactError(f"Missing or invalid artifact provenance; regenerate: {target}") from error


def read_verified_pickle(path: str | Path, expected: ArtifactFingerprint) -> Any:
    """Decode a locally generated pickle only after checking provenance and bytes."""
    return pickle.loads(read_verified_artifact(path, expected))


def _timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is not None:
        raise ValueError("Protocol timestamps must be naive Asia/Shanghai wall-clock values")
    return result


@dataclass(frozen=True)
class FoldBoundary:
    """One historical training window followed by a nonoverlapping validation window."""

    fold_id: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    def validate(self, holdout_start: str) -> None:
        """Reject reversed, overlapping or holdout-contaminating historical windows."""
        dates = [_timestamp(value) for value in (self.train_start, self.train_end, self.test_start, self.test_end)]
        if not self.fold_id or not (dates[0] <= dates[1] < dates[2] <= dates[3] < _timestamp(holdout_start)):
            raise ValueError(f"Invalid chronological fold or overlap with holdout: {self.fold_id}")


@dataclass(frozen=True)
class FrozenResearchProtocol:
    """An immutable candidate, variant and date contract fixed before rerunning results."""

    candidates: tuple[str, ...]
    variants: tuple[str, ...]
    folds: tuple[FoldBoundary, ...]
    holdout_start: str = DEFAULT_HOLDOUT_START
    historical_label: str = HISTORICAL_VALIDATION_LABEL
    selection_scope: str = "training_only"

    def validate(self) -> None:
        """Validate frozen candidate identity, historical labeling and holdout isolation."""
        for name, values in (("candidates", self.candidates), ("variants", self.variants)):
            if not values or any(not value for value in values) or len(set(values)) != len(values):
                raise ValueError(f"Protocol {name} must be nonempty, named and unique")
        if self.historical_label != HISTORICAL_VALIDATION_LABEL or self.selection_scope != "training_only":
            raise ValueError("Historical research must remain retrospective and selection must use training only")
        if not self.folds or len({fold.fold_id for fold in self.folds}) != len(self.folds):
            raise ValueError("Protocol requires uniquely named chronological folds")
        previous_end = None
        for fold in self.folds:
            fold.validate(self.holdout_start)
            if previous_end is not None and _timestamp(fold.test_start) <= previous_end:
                raise ValueError("Historical validation windows must not overlap or be out of order")
            previous_end = _timestamp(fold.test_end)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, validated protocol document."""
        self.validate()
        return {"schema_version": SCHEMA_VERSION, "timezone": "Asia/Shanghai", **asdict(self)}

    def assert_selection_timestamps(self, timestamps: Iterable[str], *, fold_id: str) -> None:
        """Require every selection observation to fall inside the named training window."""
        self.validate()
        fold = next((item for item in self.folds if item.fold_id == fold_id), None)
        if fold is None:
            raise ValueError(f"Unknown selection fold: {fold_id}")
        lower, upper = _timestamp(fold.train_start), _timestamp(fold.train_end)
        for value in timestamps:
            if not lower <= _timestamp(str(value)) <= upper:
                raise ValueError(f"Selection observation outside training window {fold_id}: {value}")


def freeze_protocol(
    protocol: FrozenResearchProtocol, *, output_dir: str | Path, readonly_roots: Iterable[str | Path]
) -> Path:
    """Persist the protocol before evaluation and refuse to overwrite a different one."""
    output = validate_output_directory(output_dir, readonly_roots)
    target = output / "frozen_protocol.json"
    payload = canonical_json(protocol.as_dict())
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError("Existing frozen protocol differs; use a fresh independent output directory")
    else:
        _atomic_write(target, payload)
    return target
