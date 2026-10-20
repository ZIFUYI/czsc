#!/usr/bin/env python
"""Compare the full commodity universe with an IS-history-selected subset.

The historical file name is retained. A hard-coded list of eventual profitable
assets is not a valid OOS universe: the subset is selected by available training
history, without using later returns. Every factor still passes common admission.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.alpha_mining_agent.forensic_pipeline import IS_END_DT  # noqa: E402
from examples.multi_factor_cta.data_engine import CrossAssetDataEngine  # noqa: E402
from examples.multi_factor_cta.run_framework_demo import load_research_universe, synthesize_admitted_asset  # noqa: E402
from examples.multi_factor_cta.synthesis_engine import (  # noqa: E402
    IntraAssetThreePillarSynthesizer,
    build_asset_execution_inputs,
)
from examples.multi_factor_cta.test_all_weather_50commodities_portfolio import (  # noqa: E402
    build_fifty_commodities_universe,
    evaluate_50_weighting_schemes,
)

WORKSPACE_DIR = ROOT_DIR / "examples" / "results" / "alpha_agent_workspace"
ARTIFACT_DIR = ROOT_DIR / "examples" / "results" / "multi_factor_cta"


def select_training_universe(
    price_returns: dict[str, pd.Series],
    *,
    cutoff: pd.Timestamp = IS_END_DT,
    subset_size: int = 24,
    min_observations: int = 30,
) -> list[str]:
    """Select by known training coverage, breaking ties by symbol (never OOS profit)."""
    counts = {
        symbol: int(series.loc[pd.to_datetime(series.index) <= cutoff].notna().sum())
        for symbol, series in price_returns.items()
    }
    eligible = [symbol for symbol, count in counts.items() if count >= min_observations]
    return sorted(eligible, key=lambda symbol: (-counts[symbol], symbol))[:subset_size]


def build_comparison_inputs(data_engine: CrossAssetDataEngine, candidate_ids: list[str]) -> dict[str, Any]:
    """Prepare real prices and signed positions once for both comparison universes."""
    specs = [(sid, spec) for sid, spec in build_fifty_commodities_universe() if spec.data_path.exists()]
    if not specs:
        raise FileNotFoundError("No real commodity data available for universe comparison")
    frames = load_research_universe(data_engine, [spec for _, spec in specs])
    synthesizer = IntraAssetThreePillarSynthesizer()
    prices, positions, sectors, fees, admission = {}, {}, {}, {}, {}
    for short_id, spec in specs:
        target, _, _, audits = synthesize_admitted_asset(spec, frames, candidate_ids, synthesizer)
        grid = frames[spec.symbol][spec.base_tf]
        prices[short_id], positions[short_id] = build_asset_execution_inputs(grid, target)
        sectors[short_id], fees[short_id], admission[short_id] = spec.sector, spec.fee_rate, audits
    return {"price_returns": prices, "positions": positions, "sectors": sectors, "fees": fees, "admission": admission}


def evaluate_universe_comparison(inputs: dict[str, Any], *, cutoff: pd.Timestamp = IS_END_DT) -> dict[str, Any]:
    """Use identical accounting for the full universe and the frozen training subset."""
    selected = select_training_universe(inputs["price_returns"], cutoff=cutoff)
    if not selected:
        raise ValueError("No assets have enough training history to define the comparison subset")

    def evaluate(symbols: list[str]) -> dict:
        return evaluate_50_weighting_schemes(
            {s: inputs["price_returns"][s] for s in symbols},
            {s: inputs["sectors"][s] for s in symbols},
            asset_positions={s: inputs["positions"][s] for s in symbols},
            asset_fee_rates={s: inputs["fees"][s] for s in symbols},
            periods_per_year=252 * 16,
        )

    return {
        "full_universe": evaluate(sorted(inputs["price_returns"])),
        "training_subset": evaluate(selected),
        "selected_symbols": selected,
        "selection_cutoff": pd.Timestamp(cutoff).isoformat(),
        "selection_rule": "IS observed bar count descending; symbol ascending on ties",
    }


def run_comparison() -> dict[str, Any]:
    """Run the existing candidate library through current admission and cost accounting."""
    allocation = json.loads((WORKSPACE_DIR / "champion_portfolio.json").read_text(encoding="utf-8"))["alloc_map"]
    inputs = build_comparison_inputs(CrossAssetDataEngine(cache_dir=WORKSPACE_DIR), sorted(allocation))
    return evaluate_universe_comparison(inputs)


def main() -> None:
    results = run_comparison()
    print(f"IS 截止时间: {results['selection_cutoff']}；历史覆盖子集: {', '.join(results['selected_symbols'])}")
    rows = []
    for universe in ("full_universe", "training_subset"):
        for key, value in results[universe].items():
            rows.append({"universe": universe, "scheme": key, **value["oos"]})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(ARTIFACT_DIR / "universe_comparison_oos.csv", index=False)
    metadata = {key: results[key] for key in ("selected_symbols", "selection_cutoff", "selection_rule")}
    (ARTIFACT_DIR / "universe_comparison_selection.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
