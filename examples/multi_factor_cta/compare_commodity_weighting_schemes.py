#!/usr/bin/env python
"""Compare commodity allocation schemes on a complete pool and an IS-selected subset.

The historical five-asset winner list is replaced by five assets selected solely
from available training history. Both pools share admission and final-fill fees.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.multi_factor_cta.compare_50_vs_champion_universe import select_training_universe  # noqa: E402
from examples.multi_factor_cta.data_engine import AssetClass  # noqa: E402
from examples.multi_factor_cta.legacy_research import (  # noqa: E402
    ARTIFACT_DIR,
    WORKSPACE_DIR,
    evaluate_experiment,  # noqa: E402
    load_experiment_inputs,
    save_experiment_report,
)
from examples.multi_factor_cta.test_all_weather_14assets_portfolio import build_fourteen_assets_universe  # noqa: E402
from examples.multi_factor_cta.test_all_weather_50commodities_portfolio import (  # noqa: E402
    evaluate_50_weighting_schemes as evaluate_weighting_schemes,
)

__all__ = ["evaluate_weighting_schemes", "main"]


def main() -> None:
    """Compare the shared allocation methods without rescaling already-net returns."""
    universe = [
        (sid, spec)
        for sid, spec in build_fourteen_assets_universe()
        if spec.asset_class == AssetClass.COMMODITY_FUTURES
    ]
    inputs = load_experiment_inputs(universe, workspace_dir=WORKSPACE_DIR)
    selected = select_training_universe(inputs["price_returns"], subset_size=5)
    results = {}
    for name, symbols in (("full", None), ("training_subset", selected)):
        for key, result in evaluate_experiment(inputs, symbols).items():
            results[f"{name}/{key}"] = {**result, "label": f"{name} · {result['label']}"}
    save_experiment_report(
        results,
        inputs,
        ARTIFACT_DIR / "commodity_weighting_comparison_chart.html",
        title="完整商品池与 IS 历史覆盖子集：扣费加权方式对比",
        metadata={"selected_symbols": selected, "selection_rule": "IS history coverage; symbol tie-break"},
    )


if __name__ == "__main__":
    main()
