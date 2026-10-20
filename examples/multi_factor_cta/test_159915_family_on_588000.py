#!/usr/bin/env python
"""Re-audit transferred candidate IDs on 588000 reference prices.

This remains a signal research diagnostic. Reference ETF/index prices do not
establish a tradable futures/option mapping or account inventory constraints.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.multi_factor_cta.data_engine import AssetClass  # noqa: E402
from examples.multi_factor_cta.legacy_research import (  # noqa: E402
    ARTIFACT_DIR,
    WORKSPACE_DIR,
    evaluate_experiment,
    load_experiment_inputs,
    save_experiment_report,
)
from examples.multi_factor_cta.test_all_weather_14assets_portfolio import build_fourteen_assets_universe  # noqa: E402


def main() -> None:
    """Use the declared equity pool for admission and report the selected reference asset."""
    universe = [
        (sid, spec)
        for sid, spec in build_fourteen_assets_universe()
        if spec.asset_class != AssetClass.COMMODITY_FUTURES
    ]
    inputs = load_experiment_inputs(universe, workspace_dir=WORKSPACE_DIR)
    if "588000" not in inputs["price_returns"]:
        raise FileNotFoundError("Missing 588000 reference price history")
    results = evaluate_experiment(inputs, ["588000"], periods_per_year=252 * 48)
    save_experiment_report(
        results,
        inputs,
        ARTIFACT_DIR / "transfer_588000_chart.html",
        title="588000 参考价格上的因子迁移诊断（非实盘执行验收）",
        metadata={"execution_scope": "Reference prices only; no instrument mapping or inventory/short-sale validation"},
    )


if __name__ == "__main__":
    main()
