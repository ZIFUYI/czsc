"""Shared orchestration for historical commodity experiments.

Old filenames remain usable, but all experiments re-audit their candidate IDs on
one complete research pool. Saved champion coefficients are not historical
priors. Allocation uses raw execution returns and signed positions, so costs are
charged after the final portfolio rebalance. This module contains no allocator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from examples.alpha_mining_agent.forensic_pipeline import IS_END_DT
from examples.multi_factor_cta.compare_50_vs_champion_universe import select_training_universe
from examples.multi_factor_cta.data_engine import AssetSpec, CrossAssetDataEngine
from examples.multi_factor_cta.run_framework_demo import load_research_universe, synthesize_admitted_asset
from examples.multi_factor_cta.synthesis_engine import IntraAssetThreePillarSynthesizer, build_asset_execution_inputs
from examples.multi_factor_cta.test_all_weather_50commodities_portfolio import evaluate_50_weighting_schemes

ROOT_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = ROOT_DIR / "examples" / "results" / "alpha_agent_workspace"
ARTIFACT_DIR = ROOT_DIR / "examples" / "results" / "multi_factor_cta"


def load_experiment_inputs(
    universe: list[tuple[str, AssetSpec]],
    *,
    workspace_dir: Path = WORKSPACE_DIR,
    candidate_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Load the declared universe and re-audit all candidates before generating positions.

    Missing data is reported explicitly. A partially available universe is kept
    in metadata; an empty universe is an error, never a simulated replacement.
    """
    available = [(sid, spec) for sid, spec in universe if spec.data_path is not None and spec.data_path.exists()]
    missing = [sid for sid, spec in universe if spec.data_path is None or not spec.data_path.exists()]
    if not available:
        raise FileNotFoundError("No real commodity data available for this experiment")
    if candidate_ids is None:
        allocation = json.loads((workspace_dir / "champion_portfolio.json").read_text(encoding="utf-8"))["alloc_map"]
        candidate_ids = sorted(allocation)
    frames = load_research_universe(CrossAssetDataEngine(cache_dir=workspace_dir), [spec for _, spec in available])
    synthesizer = IntraAssetThreePillarSynthesizer()
    prices, positions, sectors, fees, admission = {}, {}, {}, {}, {}
    for short_id, spec in available:
        target, _, _, audits = synthesize_admitted_asset(spec, frames, candidate_ids, synthesizer)
        grid = frames[spec.symbol][spec.base_tf]
        prices[short_id], positions[short_id] = build_asset_execution_inputs(grid, target)
        sectors[short_id], fees[short_id], admission[short_id] = spec.sector, spec.fee_rate, audits
    return {
        "price_returns": prices,
        "positions": positions,
        "sectors": sectors,
        "fees": fees,
        "admission": admission,
        "available_symbols": sorted(prices),
        "missing_symbols": missing,
        "candidate_ids": candidate_ids,
        "selection_cutoff": pd.Timestamp(IS_END_DT).isoformat(),
        "execution_clock": "next_open",
    }


def evaluate_experiment(
    inputs: dict[str, Any],
    symbols: list[str] | None = None,
    *,
    target_vol: float = 0.14,
    periods_per_year: float = 252 * 16,
) -> dict[str, dict]:
    """Reuse the shared comparison allocator and its final-position fee ledger."""
    symbols = sorted(inputs["price_returns"]) if symbols is None else symbols
    if not symbols:
        raise ValueError("No assets have enough training history for this experiment")
    return evaluate_50_weighting_schemes(
        {symbol: inputs["price_returns"][symbol] for symbol in symbols},
        {symbol: inputs["sectors"][symbol] for symbol in symbols},
        target_vol=target_vol,
        asset_positions={symbol: inputs["positions"][symbol] for symbol in symbols},
        asset_fee_rates={symbol: inputs["fees"][symbol] for symbol in symbols},
        periods_per_year=periods_per_year,
    )


def compare_training_subset(
    inputs: dict[str, Any],
    *,
    subset_size: int = 23,
    cutoff: pd.Timestamp = IS_END_DT,
    periods_per_year: float = 252 * 16,
) -> tuple[dict[str, dict], dict[str, Any]]:
    """Compare a frozen IS-coverage subset with causal pruning of the full universe.

    Historical A+/A++ hand-picked winner lists are intentionally retired. No
    post-cutoff price or realized strategy return participates in membership.
    """
    selected = select_training_universe(inputs["price_returns"], subset_size=subset_size, cutoff=cutoff)
    training = evaluate_experiment(inputs, selected, periods_per_year=periods_per_year)
    full = evaluate_experiment(inputs, periods_per_year=periods_per_year)
    plans = {
        "plan_a_base": {**training["4_SecHRP_100%"], "label": "方案 A：IS 历史覆盖子集 · 分层风险预算"},
        "plan_a_vol_target": {**training["6_SecHRP_VolTgt"], "label": "方案 A：IS 历史覆盖子集 · 目标波动率"},
        "plan_b_base": {**full["9_Pruned_100%"], "label": "方案 B：全资产池 · 因果休眠"},
        "plan_b_vol_target": {**full["10_Pruned_VolTgt"], "label": "方案 B：全资产池 · 因果休眠及目标波动率"},
    }
    metadata = {
        "selected_symbols": selected,
        "selection_cutoff": pd.Timestamp(cutoff).isoformat(),
        "selection_rule": "IS observed bar count descending; symbol ascending on ties",
        "retired_variants": "A+/A++ retrospective symbol lists and full-sample exposure normalization",
    }
    return plans, metadata


def render_experiment(results: dict[str, dict], *, title: str, oos_only: bool = False) -> go.Figure:
    """Render the computed net-return and actual exposure series without recomputing PnL."""
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("扣费净值", "实际总敞口"))
    for key, result in results.items():
        label = result["label"].replace("50商品", f"{len(result['weights_df'].columns)} 商品")
        daily = result["daily_returns"]
        gross = result["weights_df"].abs().sum(axis=1).resample("D").mean().dropna()
        if oos_only:
            daily = daily.loc[daily.index > pd.Timestamp(IS_END_DT.date())]
            gross = gross.loc[gross.index > pd.Timestamp(IS_END_DT.date())]
        figure.add_trace(go.Scatter(x=daily.index, y=1.0 + daily.cumsum(), name=label, legendgroup=key), row=1, col=1)
        figure.add_trace(
            go.Scatter(x=gross.index, y=gross * 100.0, name=label, legendgroup=key, showlegend=False),
            row=2,
            col=1,
        )
    figure.update_layout(title=title, template="plotly_white", height=850, hovermode="x unified")
    figure.update_yaxes(title_text="净值（固定名义本金）", row=1, col=1)
    figure.update_yaxes(title_text="敞口 %", row=2, col=1)
    return figure


def save_experiment_report(
    results: dict[str, dict],
    inputs: dict[str, Any],
    destination: Path,
    *,
    title: str,
    metadata: dict[str, Any] | None = None,
    oos_only: bool = False,
) -> None:
    """Save a self-contained chart, split performance table and auditable selection record."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    render_experiment(results, title=title, oos_only=oos_only).write_html(destination, include_plotlyjs=True)
    rows = []
    for key, result in results.items():
        for split in ("is", "oos", "full"):
            rows.append({"scheme": key, "label": result["label"], "split": split, **result[split]})
    table = pd.DataFrame(rows)
    table.to_csv(destination.with_suffix(".csv"), index=False)
    record = {
        key: inputs[key]
        for key in (
            "available_symbols",
            "missing_symbols",
            "candidate_ids",
            "selection_cutoff",
            "execution_clock",
            "admission",
        )
        if key in inputs
    }
    record.update(metadata or {})
    record["accounting"] = "raw fill-to-fill returns; signed final positions; fees on every final position change"
    destination.with_suffix(".json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(table.loc[table["split"] == "oos"].to_string(index=False))
    print(f"已保存：{destination}")
