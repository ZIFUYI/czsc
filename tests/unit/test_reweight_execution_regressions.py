"""Both production optimizers must score the same next-open, fully costed execution path."""

import ast
import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent import forensic_pipeline, run_evolution_loop
from examples.multi_factor_cta import synthesis_engine


def _bars(n=24):
    df = generate_symbol_kines("TEST", "日线", "20231201", "20240228", seed=729).iloc[:n].copy()
    df["dt"] = pd.to_datetime(df["dt"])
    df.attrs["fee_rate"] = 0.003
    return df


def _production_fast_functions(path, frame):
    """Execute the actual nested scoring functions, with only their enclosing slice context supplied."""
    module = run_evolution_loop if path == "runner" else synthesis_engine
    prep_name, stats_name = (
        ("_prep_fast_slice", "_fast_slice_stats") if path == "runner" else ("_prep_slice", "_fast_stats")
    )
    syntax = ast.parse(inspect.getsource(module))
    nodes = [
        node for node in ast.walk(syntax) if isinstance(node, ast.FunctionDef) and node.name in {prep_name, stats_name}
    ]
    assert len(nodes) == 2
    namespace = dict(vars(module))
    namespace.update(df_30m=frame, df_grid=frame, asset_spec=SimpleNamespace(fee_rate=frame.attrs["fee_rate"]))
    # Helpers are imported by the optimizer block to avoid changing unrelated module dependencies.
    namespace.update(vars(forensic_pipeline))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), inspect.getfile(module), "exec"), namespace)
    return namespace[prep_name], namespace[stats_name]


def _quantity_net_total(frame, decisions):
    """Independent monetary order book, including the final actual close."""
    capital = 250_000.0
    marks = [*frame["open"], frame["close"].iloc[-1]]
    filled_targets = [0.0, *decisions[:-1], 0.0]
    quantity, previous_price, net = 0.0, marks[0], 0.0
    for price, target in zip(marks, filled_targets, strict=True):
        net += quantity * (price - previous_price)
        target_quantity = target * capital / price
        net -= abs(target_quantity - quantity) * price * frame.attrs["fee_rate"]
        quantity, previous_price = target_quantity, price
    return net / capital * 100.0


@pytest.mark.parametrize("path", ["runner", "joint"])
def test_reweight_scores_next_open_gaps_and_orders_candidates(path):
    df = _bars(8)
    price = float(df["close"].iloc[0])
    df[["open", "high", "low", "close"]] = price
    df.loc[df.index[2], "open"] = price * 1.2
    prep, score = _production_fast_functions(path, df)
    returns, days = prep(np.ones(len(df), dtype=bool))
    before_gap, after_gap = np.zeros(len(df)), np.zeros(len(df))
    before_gap[0] = 1.0  # decision 0 fills at open 1 and sells at open 2
    after_gap[1] = 1.0  # decision 1 buys at open 2, after the positive gap
    early = score(returns, days, before_gap)
    late = score(returns, days, after_gap)
    assert early["cum_ret_pct"] == pytest.approx(_quantity_net_total(df, before_gap), abs=0.01)
    assert late["cum_ret_pct"] == pytest.approx(_quantity_net_total(df, after_gap), abs=0.01)
    assert early["cum_ret_pct"] > late["cum_ret_pct"]


@pytest.mark.parametrize("path", ["runner", "joint"])
def test_reweight_scores_configured_fee_and_terminal_liquidation(path):
    df = _bars()
    df[["open", "high", "low", "close"]] = float(df["close"].iloc[0])
    prep, score = _production_fast_functions(path, df)
    returns, days = prep(np.ones(len(df), dtype=bool))
    stats = score(returns, days, np.ones(len(df)))
    assert stats["cum_ret_pct"] == pytest.approx(-0.6)
    half = score(returns, days, np.full(len(df), 0.5))
    assert half["cum_ret_pct"] == pytest.approx(-0.3)
    assert half["cum_ret_pct"] > stats["cum_ret_pct"]


@pytest.mark.parametrize("path", ["runner", "joint"])
def test_reweight_scores_match_full_execution_report(path):
    df = _bars()
    weights = np.resize(np.array([0.0, 0.25, 0.75, -0.5]), len(df))
    prep, score = _production_fast_functions(path, df)
    returns, days = prep(np.ones(len(df), dtype=bool))
    fast = score(returns, days, weights)
    full = forensic_pipeline.backtest_slice(df, weights)
    assert fast["cum_ret_pct"] == pytest.approx(_quantity_net_total(df, weights), abs=0.011)
    for key in ("cum_ret_pct", "ann_ret_pct", "sharpe", "mdd_pct", "calmar"):
        assert fast[key] == pytest.approx(full[key], abs=0.011), key


@pytest.mark.parametrize("path", ["runner", "joint"])
def test_reweight_empty_fold_and_changed_oos_prices(path):
    df = _bars(60)
    mask = (df["dt"] <= pd.Timestamp("2023-12-31 23:59:59")).to_numpy()
    prep, score = _production_fast_functions(path, df)
    empty_ret, empty_days = prep(np.zeros(len(df), dtype=bool))
    assert all(value == 0 for value in score(empty_ret, empty_days, np.zeros(0)).values())
    returns, days = prep(mask)
    changed = df.copy()
    changed.loc[~mask, ["open", "high", "low", "close"]] *= 100
    changed_prep, _ = _production_fast_functions(path, changed)
    changed_returns, changed_days = changed_prep(mask)
    np.testing.assert_array_equal(returns, changed_returns)
    np.testing.assert_array_equal(days, changed_days)
