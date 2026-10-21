"""Portable input/output locations for standalone CTA research scripts."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def research_data_dir() -> Path:
    """Read the explicit data override, otherwise use the repository's historical cache."""
    return Path(
        os.environ.get("CZSC_CTA_DATA_DIR", ROOT / "examples/results/continuous_signal_commodity_cta/data_cache")
    ).resolve()


def research_output_path(name: str, override: str | Path | None = None) -> Path:
    """Keep generated files in an explicit output directory, separate from source inputs."""
    directory = Path(os.environ.get("CZSC_CTA_OUTPUT_DIR", ROOT / "examples/results/cta_research"))
    output = Path(override) if override is not None else directory / name
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
