#!/usr/bin/env python
"""Silver research report using full-pool admission and final execution fees."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.multi_factor_cta.legacy_research import (  # noqa: E402
    ARTIFACT_DIR,
    WORKSPACE_DIR,
    evaluate_experiment,
    load_experiment_inputs,
    save_experiment_report,
)
from examples.multi_factor_cta.test_all_weather_50commodities_portfolio import (  # noqa: E402
    build_fifty_commodities_universe,
)


def main() -> None:
    """Audit silver against the declared commodity pool and save computed daily returns."""
    inputs = load_experiment_inputs(build_fifty_commodities_universe(), workspace_dir=WORKSPACE_DIR)
    if "ag" not in inputs["price_returns"]:
        raise FileNotFoundError("Missing silver commodity price history")
    results = evaluate_experiment(inputs, ["ag"])
    save_experiment_report(
        results,
        inputs,
        ARTIFACT_DIR / "ag_performance_chart.html",
        title="沪银：统一准入后的扣费净值与实际敞口",
    )
    baseline = results["1_Equal_1/N"]["daily_returns"]
    daily = baseline.rename("daily_return").to_frame()
    daily["strategy_nav"] = 1.0 + baseline.cumsum()
    daily.to_csv(ARTIFACT_DIR / "ag_daily_perf.csv", index_label="date")


if __name__ == "__main__":
    main()
