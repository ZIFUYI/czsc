#!/usr/bin/env python
"""Compare an IS-coverage-frozen universe (A) with full-pool causal pruning (B).

All saved candidates are re-audited. OOS winner lists and future-calibrated
exposure tiers are retired; baseline and ex-ante volatility budgets remain.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.multi_factor_cta.legacy_research import (  # noqa: E402
    ARTIFACT_DIR,
    WORKSPACE_DIR,
    compare_training_subset,  # noqa: E402
    load_experiment_inputs,
    save_experiment_report,
)
from examples.multi_factor_cta.test_all_weather_50commodities_portfolio import (  # noqa: E402
    build_fifty_commodities_universe,
)


def run_plan_comparison(
    *, workspace_dir: Path = WORKSPACE_DIR
) -> tuple[dict[str, dict], dict[str, Any], dict[str, Any]]:
    """Use the same pool and final-position accounting for every Plan A/B view."""
    inputs = load_experiment_inputs(build_fifty_commodities_universe(), workspace_dir=workspace_dir)
    results, metadata = compare_training_subset(inputs)
    return results, inputs, metadata


def main() -> None:
    """Save the all-period Plan A/B comparison and selection record."""
    results, inputs, metadata = run_plan_comparison(workspace_dir=WORKSPACE_DIR)
    save_experiment_report(
        results,
        inputs,
        ARTIFACT_DIR / "compare_plan_a_vs_plan_b_chart.html",
        title="方案 A 与 B：冻结训练资产池与动态休眠的扣费比较",
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
