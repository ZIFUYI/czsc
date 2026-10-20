"""Legacy research entry points must use the common admission and execution contract."""

import importlib
import json

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.multi_factor_cta.data_engine import AssetClass, AssetSpec, ExecutionMode
from examples.multi_factor_cta.synthesis_engine import CrossAssetHRPSynthesizer


@pytest.mark.parametrize(
    "module_name,builder,is_family,expected_calls",
    [
        ("test_all_weather_9assets_portfolio", "build_nine_assets_universe", False, 3),
        ("test_all_weather_14assets_portfolio", "build_fourteen_assets_universe", False, 3),
        ("test_159915_family_on_commodities", "build_five_commodities_universe", True, 1),
    ],
)
def test_legacy_main_passes_signed_targets_raw_returns_and_fees(
    module_name,
    builder,
    is_family,
    expected_calls,
    monkeypatch,
    tmp_path,
):
    entry = importlib.import_module(f"examples.multi_factor_cta.{module_name}")
    frame = generate_symbol_kines("SHFE.RB9999", "15分钟", "20240101", "20240115", seed=913)
    frame["dt"] = pd.to_datetime(frame["dt"])
    frame["date"] = frame["dt"].dt.date
    specs = [
        AssetSpec(
            symbol=symbol,
            name=f"{symbol}_fixture",
            sector=symbol,
            base_tf="15m",
            eligible_tfs=("15m",),
            fee_rate=0.001,
            execution_mode=ExecutionMode.LONG_SHORT,
            asset_class=AssetClass.COMMODITY_FUTURES if (is_family or i) else AssetClass.INDEX_FUTURES,
            data_path=tmp_path / f"{symbol}.parquet",
        )
        for i, symbol in enumerate(("A", "B"))
    ]
    frames = {spec.symbol: {"15m": frame.copy()} for spec in specs}
    (tmp_path / "champion_portfolio.json").write_text(json.dumps({"alloc_map": {"AC01L@15m": 0.9}}))
    for attr in ("WORKSPACE_DIR", "CTA_OUTPUT_DIR", "ARTIFACT_DIR", "COMMODITY_DATA_DIR"):
        monkeypatch.setattr(entry, attr, tmp_path, raising=False)
    monkeypatch.setattr(entry, builder, lambda: specs if is_family else [(s.symbol, s) for s in specs])
    if is_family:
        monkeypatch.setattr(entry, "resample_5m_to_15m_base", lambda source, destination, night: destination)

    load_calls, admission_calls, ledger_calls = [], [], []

    def load(engine, universe):
        load_calls.append([spec.symbol for spec in universe])
        return frames

    def synthesize(spec, pool, candidates, synthesizer):
        admission_calls.append((spec.symbol, candidates))
        target = np.sign(frame["close"].pct_change().fillna(0.0).to_numpy()) * 0.4
        target[-2:] = 0.0
        return target, {}, {"AC01L@15m": 0.3}, {"AC01L@15m": {"passed": True}}

    original = CrossAssetHRPSynthesizer.build_multi_asset_portfolio

    class LedgerReached(Exception):
        pass

    def allocate(self, returns, sectors, **kwargs):
        assert kwargs["returns_kind"] == "price"
        assert kwargs["positions_timing"] == "execution"
        for symbol, price_return in returns.items():
            opening = frame["open"].to_numpy()
            expected = np.r_[0.0, np.diff(opening) / opening[:-1], frame["close"].iloc[-1] / opening[-1] - 1.0]
            np.testing.assert_allclose(price_return, expected)
            assert kwargs["asset_positions_df"][symbol].min() < 0.0
            assert kwargs["asset_fee_rates"][symbol] == 0.001
        result = original(self, returns, sectors, **kwargs)
        assert result["cost_returns_df"].to_numpy().sum() > 0.0
        ledger_calls.append(result)
        if len(ledger_calls) == expected_calls:
            raise LedgerReached
        return result

    monkeypatch.setattr(entry, "load_research_universe", load)
    monkeypatch.setattr(entry, "synthesize_admitted_asset", synthesize)
    monkeypatch.setattr(CrossAssetHRPSynthesizer, "build_multi_asset_portfolio", allocate)
    with pytest.raises(LedgerReached):
        entry.main()
    assert load_calls == [["A", "B"]]
    assert len(admission_calls) == 2
    assert len(ledger_calls) == expected_calls


def _comparison_inputs():
    frame = generate_symbol_kines("SHFE.RB9999", "日线", "20230101", "20241231", seed=914)
    index = pd.to_datetime(frame["dt"])
    returns = pd.Series(frame["close"].pct_change().fillna(0.0).to_numpy(), index=index)
    target = np.sign(returns) * 0.4
    return {
        "price_returns": {"A": returns, "B": returns * 0.7},
        "positions": {"A": target, "B": -target},
        "sectors": {"A": "X", "B": "Y"},
        "fees": {"A": 0.001, "B": 0.001},
    }


def test_comparison_selection_never_reads_oos_or_future_listing():
    from examples.multi_factor_cta.compare_50_vs_champion_universe import select_training_universe

    inputs = _comparison_inputs()
    cutoff = pd.Timestamp("2023-12-31")
    before = select_training_universe(inputs["price_returns"], cutoff=cutoff, subset_size=1)
    altered = {key: series.copy() for key, series in inputs["price_returns"].items()}
    altered["A"].loc[altered["A"].index > cutoff] = -0.99
    altered["B"].loc[altered["B"].index > cutoff] = 10.0
    altered["FUTURE"] = altered["A"].loc[altered["A"].index > cutoff]
    assert select_training_universe(altered, cutoff=cutoff, subset_size=1) == before


def test_comparison_and_plot_share_fee_paid_ledgers():
    from examples.multi_factor_cta.compare_50_vs_champion_universe import evaluate_universe_comparison
    from examples.multi_factor_cta.plot_50_vs_top24_chart import render_universe_comparison

    inputs = _comparison_inputs()
    results = evaluate_universe_comparison(inputs)
    assert results["selected_symbols"] == ["A", "B"]
    for universe in ("full_universe", "training_subset"):
        for result in results[universe].values():
            ledger = result["ledger"]
            assert ledger["cost_returns_df"].to_numpy().sum() > 0.0
            np.testing.assert_allclose(
                ledger["bar_net_returns"],
                (ledger["gross_returns_df"] - ledger["cost_returns_df"]).sum(axis=1),
            )
    figure = render_universe_comparison(results)
    assert len(figure.data) == 4
    expected = results["full_universe"]["4_SecHRP_100%"]["daily_returns"]
    np.testing.assert_allclose(figure.data[0].y, 1.0 + expected.cumsum())
