"""Historical experiments must share admission, causal selection and execution costs."""

import importlib
import json

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines

ENTRY_POINTS = [
    "test_all_weather_20commodities_portfolio",
    "test_all_weather_32commodities_portfolio",
    "compare_commodity_weighting_schemes",
    "compare_plan_a_vs_b",
    "fix_plan_a_oos",
    "plot_plan_a_core_dedicated",
    "plot_oos_nav_chart",
]


def _inputs():
    frame = generate_symbol_kines("SHFE.RB9999", "日线", "20230101", "20241231", seed=193)
    index = pd.to_datetime(frame["dt"])
    returns = pd.Series(frame["open"].pct_change().fillna(0.0).to_numpy(), index=index)
    positions = pd.Series(np.sign(frame["close"].diff().fillna(0.0).to_numpy()), index=index).shift(1).fillna(0.0)
    positions.iloc[-1] = 0.0
    return {
        "price_returns": {"A": returns, "B": returns * 0.8},
        "positions": {"A": positions, "B": -positions},
        "sectors": {"A": "METALS", "B": "AGRICULTURE"},
        "fees": {"A": 0.001, "B": 0.002},
        "admission": {},
    }


@pytest.mark.parametrize(
    "module,function",
    [
        ("test_all_weather_20commodities_portfolio", "evaluate_20_weighting_schemes"),
        ("test_all_weather_32commodities_portfolio", "evaluate_32_weighting_schemes"),
        ("compare_commodity_weighting_schemes", "evaluate_weighting_schemes"),
    ],
)
def test_legacy_weighting_requires_actual_positions_and_accounts_reallocation_costs(module, function):
    evaluate = getattr(importlib.import_module(f"examples.multi_factor_cta.{module}"), function)
    inputs = _inputs()
    results = evaluate(
        inputs["price_returns"],
        inputs["sectors"],
        asset_positions=inputs["positions"],
        asset_fee_rates=inputs["fees"],
        periods_per_year=252,
    )
    for result in results.values():
        ledger = result["ledger"]
        positions = result["weights_df"]
        assert positions.min().min() < 0.0
        for col, fee in inputs["fees"].items():
            np.testing.assert_allclose(
                ledger["cost_returns_df"][col],
                ledger["turnover_df"][col] * fee,
                atol=1e-14,
            )
        assert float(ledger["cost_returns_df"].to_numpy().sum()) > 0.0
        np.testing.assert_allclose(
            ledger["bar_net_returns"],
            (ledger["gross_returns_df"] - ledger["cost_returns_df"]).sum(axis=1),
        )
    with pytest.raises(TypeError):
        evaluate(inputs["price_returns"], inputs["sectors"])


def test_shared_loader_reaudits_ids_for_complete_pool_and_preserves_signed_execution(monkeypatch, tmp_path):
    from examples.multi_factor_cta import legacy_research as research
    from examples.multi_factor_cta.data_engine import AssetClass, AssetSpec, ExecutionMode

    frame = generate_symbol_kines("SHFE.RB9999", "15分钟", "20240101", "20240115", seed=194)
    frame["dt"] = pd.to_datetime(frame["dt"])
    specs = [
        AssetSpec(
            symbol=symbol,
            name=symbol,
            sector=symbol,
            base_tf="15m",
            eligible_tfs=("15m",),
            fee_rate=0.001,
            execution_mode=ExecutionMode.LONG_SHORT,
            asset_class=AssetClass.COMMODITY_FUTURES,
            data_path=tmp_path / f"{symbol}.parquet",
        )
        for symbol in ("A", "B", "MISSING")
    ]
    for spec in specs[:2]:
        spec.data_path.touch()
    (tmp_path / "champion_portfolio.json").write_text(json.dumps({"alloc_map": {"AC01L@15m": 999.0}}))
    frames = {spec.symbol: {"15m": frame.copy()} for spec in specs[:2]}
    events = []

    def load(engine, universe):
        events.append(("pool", [spec.symbol for spec in universe]))
        return frames

    def synthesize(spec, pool, candidates, synthesizer):
        assert pool is frames
        assert candidates == ["AC01L@15m"]
        events.append(("admission", spec.symbol))
        if spec.symbol == "B":
            return np.zeros(len(frame)), {}, {}, {"AC01L@15m": {"passed": False}}
        target = np.sign(frame["close"].diff().fillna(0.0).to_numpy()) * 0.4
        return target, {}, {}, {"AC01L@15m": {"passed": True}}

    monkeypatch.setattr(research, "load_research_universe", load)
    monkeypatch.setattr(research, "synthesize_admitted_asset", synthesize)
    inputs = research.load_experiment_inputs([(spec.symbol, spec) for spec in specs], workspace_dir=tmp_path)
    assert events == [("pool", ["A", "B"]), ("admission", "A"), ("admission", "B")]
    assert inputs["missing_symbols"] == ["MISSING"]
    assert inputs["positions"]["A"].min() < 0
    assert inputs["positions"]["B"].eq(0).all()
    assert inputs["positions"]["A"].iloc[-1] == 0.0
    expected = frame["open"].pct_change().fillna(0.0)
    np.testing.assert_allclose(inputs["price_returns"]["A"].iloc[:-1], expected)
    assert inputs["fees"] == {"A": 0.001, "B": 0.001}


def test_no_data_is_explicit_and_importing_fix_script_has_no_side_effect(tmp_path, monkeypatch):
    from examples.multi_factor_cta import legacy_research as research

    with pytest.raises(FileNotFoundError, match="No real commodity data"):
        research.load_experiment_inputs([], workspace_dir=tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Importing a legacy script must not start a research run")

    from examples.multi_factor_cta import compare_plan_a_vs_b

    monkeypatch.setattr(compare_plan_a_vs_b, "run_plan_comparison", forbidden)
    imported = importlib.import_module("examples.multi_factor_cta.fix_plan_a_oos")
    importlib.reload(imported)
    assert callable(imported.main)


def test_plan_selection_and_past_ledger_are_invariant_to_future_prices_and_positions():
    from examples.multi_factor_cta.legacy_research import compare_training_subset

    inputs = _inputs()
    cutoff = pd.Timestamp("2023-12-31")
    before, metadata = compare_training_subset(inputs, subset_size=1, cutoff=cutoff, periods_per_year=252)
    changed = {name: values.copy() for name, values in inputs.items()}
    for field in ("price_returns", "positions"):
        changed[field] = {symbol: series.copy() for symbol, series in inputs[field].items()}
        for series in changed[field].values():
            series.loc[series.index > cutoff] *= -0.5 if field == "price_returns" else -1.0
    after, after_metadata = compare_training_subset(changed, subset_size=1, cutoff=cutoff, periods_per_year=252)
    assert metadata == after_metadata
    assert metadata["selected_symbols"] == ["A"]
    for key in before:
        pd.testing.assert_frame_equal(before[key]["weights_df"].loc[:cutoff], after[key]["weights_df"].loc[:cutoff])
        pd.testing.assert_series_equal(
            before[key]["ledger"]["bar_net_returns"].loc[:cutoff],
            after[key]["ledger"]["bar_net_returns"].loc[:cutoff],
        )


@pytest.mark.parametrize("module_name", ENTRY_POINTS)
def test_migrated_main_reaches_shared_reports(module_name, monkeypatch, tmp_path):
    entry = importlib.import_module(f"examples.multi_factor_cta.{module_name}")
    from examples.multi_factor_cta.legacy_research import compare_training_subset

    inputs = _inputs()
    inputs.update(available_symbols=["A", "B"], missing_symbols=[], candidate_ids=["AC01L@15m"])
    plans, metadata = compare_training_subset(inputs, periods_per_year=252)
    calls = []

    def report(results, report_inputs, destination, **kwargs):
        assert report_inputs is inputs
        assert destination.parent == tmp_path
        assert destination.suffix == ".html"
        assert results
        for result in results.values():
            assert result["ledger"]["cost_returns_df"].to_numpy().sum() > 0
        calls.append((results, kwargs))

    monkeypatch.setattr(entry, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(entry, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(entry, "save_experiment_report", report)
    if hasattr(entry, "run_plan_comparison"):
        monkeypatch.setattr(entry, "run_plan_comparison", lambda **kwargs: (plans, inputs, metadata))
    else:
        monkeypatch.setattr(entry, "load_experiment_inputs", lambda *args, **kwargs: inputs)
    entry.main()
    assert len(calls) == 1
    if module_name == "plot_plan_a_core_dedicated":
        assert set(calls[0][0]) == {"plan_a_base", "plan_a_vol_target"}


def test_plan_report_uses_computed_oos_net_curves_and_saves_selection(tmp_path):
    from examples.multi_factor_cta.legacy_research import (
        compare_training_subset,
        render_experiment,
        save_experiment_report,
    )

    inputs = _inputs()
    plans, metadata = compare_training_subset(inputs, periods_per_year=252)
    figure = render_experiment(plans, title="Regression", oos_only=True)
    expected = plans["plan_a_base"]["daily_returns"].loc["2024-01-01":]
    np.testing.assert_allclose(figure.data[0].y, 1.0 + expected.cumsum())
    destination = tmp_path / "result.html"
    save_experiment_report(plans, inputs, destination, title="Regression", metadata=metadata, oos_only=True)
    assert destination.exists()
    stored = json.loads(destination.with_suffix(".json").read_text())
    assert stored["selected_symbols"] == metadata["selected_symbols"]
    table = pd.read_csv(destination.with_suffix(".csv"))
    assert len(table) == len(plans) * 3
    assert set(table["split"]) == {"is", "oos", "full"}


@pytest.mark.parametrize(
    "module_name,symbol",
    [
        ("test_159915_family_on_510500", "510500"),
        ("test_159915_family_on_000852", "000852"),
        ("test_159915_family_on_588000", "588000"),
        ("plot_ag_performance", "ag"),
        ("walkthrough_td_factor_validation", "ag"),
    ],
)
def test_single_asset_and_td_walkthrough_reuse_admission_and_ledger(module_name, symbol, monkeypatch, tmp_path):
    entry = importlib.import_module(f"examples.multi_factor_cta.{module_name}")
    inputs = _inputs()
    for field in ("price_returns", "positions", "sectors", "fees"):
        inputs[field] = {symbol: inputs[field]["A"]}
    inputs["admission"] = {symbol: {"TD03S@30m": {"passed": False, "reason": "test rejected"}}}
    (tmp_path / "champion_portfolio.json").write_text(json.dumps({"alloc_map": {"AC01L@15m": 999.0}}))
    load_calls = []
    reports = []

    def load(*args, **kwargs):
        load_calls.append(kwargs)
        return inputs

    def report(results, report_inputs, destination, **kwargs):
        assert report_inputs is inputs
        assert destination.parent == tmp_path
        for result in results.values():
            assert set(result["weights_df"].columns) == {symbol}
            assert result["ledger"]["cost_returns_df"].sum().sum() > 0
        reports.append(kwargs)

    monkeypatch.setattr(entry, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(entry, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(entry, "load_experiment_inputs", load)
    monkeypatch.setattr(entry, "save_experiment_report", report)
    entry.main()
    assert len(reports) == 1
    if symbol in {"510500", "000852", "588000"}:
        assert "Reference prices only" in reports[0]["metadata"]["execution_scope"]
    if module_name == "walkthrough_td_factor_validation":
        assert len(load_calls) == 2
        assert any(uid.startswith("TD") for uid in load_calls[0]["candidate_ids"])
        assert all(not uid.startswith("TD") for uid in load_calls[1]["candidate_ids"])
        admission = pd.read_csv(tmp_path / "td_admission.csv")
        assert not admission["passed"].any()
        assert admission["reason"].tolist() == ["test rejected"]
