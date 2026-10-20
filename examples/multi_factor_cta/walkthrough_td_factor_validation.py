#!/usr/bin/env python
"""Trace TD candidate admission and compare a cost-accounted TD-family ablation.

The walkthrough reports actual accepted/rejected states. A diagnostic Gate 1A
result or a saved champion coefficient cannot bypass the common admission path.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from examples.alpha_mining_agent.forensic_pipeline import IS_END_DT  # noqa: E402
from examples.multi_factor_cta.legacy_research import (  # noqa: E402
    ARTIFACT_DIR,
    WORKSPACE_DIR,
    evaluate_experiment,
    load_experiment_inputs,
    save_experiment_report,
)
from examples.multi_factor_cta.test_all_weather_14assets_portfolio import build_fourteen_assets_universe  # noqa: E402
from examples.multi_factor_cta.test_all_weather_50commodities_portfolio import (  # noqa: E402
    build_fifty_commodities_universe,
)


def compute_naive_full_sample_ic(df_tf: pd.DataFrame, weights: np.ndarray, horizon: int = 6) -> tuple[float, float]:
    """对比演示：若错误套用传统股票截面的全样本线性 IC 会发生什么"""
    is_mask = (df_tf["dt"] <= IS_END_DT).values
    c = df_tf["close"].values[is_mask]
    w = weights[is_mask]
    fwd_ret = np.zeros_like(c)
    fwd_ret[:-horizon] = (c[horizon:] - c[:-horizon]) / c[:-horizon]
    valid = np.arange(len(c)) < (len(c) - horizon)
    if np.std(w[valid]) < 1e-9:
        return 0.0, 0.0
    full_ic = float(np.corrcoef(w[valid], fwd_ret[valid])[0, 1])
    active_mask = valid & (np.abs(w) > 1e-4)
    active_ratio = float(active_mask.sum() / valid.sum() * 100.0)
    return full_ic, active_ratio


def main() -> None:
    """Re-audit all TD candidates, then ablate the TD family using identical accounting."""
    allocation = json.loads((WORKSPACE_DIR / "champion_portfolio.json").read_text(encoding="utf-8"))["alloc_map"]
    td_examples = ["TD01S@5m", "TD01S@15m", "TD02S@5m", "TD03S@30m", "TD04L@60m", "TD01L@30m"]
    candidates = sorted(set(allocation) | set(td_examples))
    universe = dict(build_fourteen_assets_universe())
    universe.update(build_fifty_commodities_universe())
    inputs = load_experiment_inputs(list(universe.items()), workspace_dir=WORKSPACE_DIR, candidate_ids=candidates)
    without_td = [uid for uid in candidates if not uid.startswith("TD")]
    baseline_inputs = load_experiment_inputs(
        list(universe.items()),
        workspace_dir=WORKSPACE_DIR,
        candidate_ids=without_td,
    )
    # Keep unlike execution frequencies out of the same bar-risk estimator.
    commodity_ids = [sid for sid, _ in build_fifty_commodities_universe() if sid in inputs["price_returns"]]
    results = {}
    for label, current in (("with_td", inputs), ("without_td", baseline_inputs)):
        for key, result in evaluate_experiment(current, commodity_ids).items():
            results[f"{label}/{key}"] = {**result, "label": f"{label} · {result['label']}"}
    save_experiment_report(
        results,
        inputs,
        ARTIFACT_DIR / "td_factor_validation.html",
        title="TD 因子族：统一准入记录与扣费消融",
        metadata={"baseline_candidate_ids": without_td, "baseline_admission": baseline_inputs["admission"]},
    )
    rows = [
        {"asset": asset, "unit": uid, "passed": bool(audit.get("passed", False)), "reason": audit.get("reason", "")}
        for asset, audits in inputs["admission"].items()
        for uid, audit in audits.items()
        if uid.startswith("TD")
    ]
    table = pd.DataFrame(rows, columns=["asset", "unit", "passed", "reason"])
    table.to_csv(ARTIFACT_DIR / "td_admission.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
