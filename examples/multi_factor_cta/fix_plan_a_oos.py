#!/usr/bin/env python
"""Re-evaluate Plan A/B with the common admission and cost ledger."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.multi_factor_cta.compare_plan_a_vs_b import run_plan_comparison  # noqa: E402
from examples.multi_factor_cta.legacy_research import ARTIFACT_DIR, WORKSPACE_DIR, save_experiment_report  # noqa: E402


def main() -> None:
    """Recompute only on explicit execution; imports never read data or create reports."""
    results, inputs, metadata = run_plan_comparison(workspace_dir=WORKSPACE_DIR)
    save_experiment_report(
        results,
        inputs,
        ARTIFACT_DIR / "plan_a_oos_revalidation.html",
        title="统一准入后的样本外研究曲线（含最终调仓成本）",
        metadata=metadata,
        oos_only=True,
    )


if __name__ == "__main__":
    main()
