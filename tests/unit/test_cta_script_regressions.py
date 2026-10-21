"""Exercise the actual research script entry points against causal execution contracts."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import czsc
from czsc.mock import generate_symbol_kines


class _Captured(Exception):
    pass


@pytest.mark.parametrize("module_name", ["run_slope_continuous_signal_cta", "test_slope_60m_unchanged_params"])
def test_intraday_target_vol_cannot_read_later_bars(monkeypatch, module_name):
    module = import_module(f"scripts.{module_name}")
    frame = generate_symbol_kines("TEST", "15分钟", "20240101", "20240531", seed=93)
    frame = frame.rename(columns={"vol": "volume"})
    if module_name == "test_slope_60m_unchanged_params":
        price_columns = ["open", "high", "low", "close"]
        frame[price_columns] = 100.0 + (frame[price_columns] - 100.0) * 0.15
    cut = pd.Timestamp("2024-05-20 13:30")
    changed = frame.copy()
    later = changed.dt > cut
    changed.loc[later, ["open", "high", "low", "close"]] *= 1.20
    captured = []

    def capture(weights, band):
        captured.append(weights.copy())
        raise _Captured

    monkeypatch.setattr(Path, "glob", lambda *_: [Path("TEST_15m.parquet")])
    if module_name == "run_slope_continuous_signal_cta":
        monkeypatch.setattr(module, "apply_rebalance_deadband", capture)

        def run():
            return module.run_slope_continuous_system(module.SlopeContinuousConfig(target_vol=0.15))
    else:
        monkeypatch.setattr(module, "apply_rebalance_deadband_60m", capture)
        config = module.SlopeContinuous60mConfig(target_vol=0.15)
        monkeypatch.setattr(module, "SlopeContinuous60mConfig", lambda: config)
        run = module.run_test
    for data in [frame, changed]:
        monkeypatch.setattr(module.pd, "read_parquet", lambda *_, data=data: data.copy())
        with pytest.raises(_Captured):
            run()
    pd.testing.assert_frame_equal(captured[0].loc[:cut], captured[1].loc[:cut])


@pytest.mark.parametrize("module_name", ["run_slope_continuous_signal_cta", "test_slope_60m_unchanged_params"])
def test_friday_night_and_saturday_morning_share_monday(monkeypatch, module_name):
    module = import_module(f"scripts.{module_name}")
    assert module.get_trading_day(pd.Timestamp("2024-05-17 21:15")) == "2024-05-20"
    assert module.get_trading_day(pd.Timestamp("2024-05-18 00:30")) == "2024-05-20"


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    [
        ("calc_overall_nav_multiprocess", "backtest_one_file"),
        ("build_complete_dashboard_data", "backtest_task"),
        ("run_portfolio_weighting_comparison", "backtest_one"),
        ("run_all_commodities_backtest", "backtest_symbol"),
    ],
)
def test_signed_position_reports_match_native_next_open(monkeypatch, module_name, function_name):
    module = import_module(f"scripts.{module_name}")
    frame = generate_symbol_kines("TEST", "日线", "20220101", "20231231", seed=73).iloc[:240].copy()
    frame["id"] = np.arange(len(frame))
    bars = czsc.format_standard_kline(frame, freq=czsc.Freq.D)
    position = czsc.Position(
        symbol="TEST",
        opens=[czsc.Event(operate=czsc.Operate.SO, signals_all=[czsc.Signal("日线_test_state_空_任意_任意_0")])],
        exits=[czsc.Event(operate=czsc.Operate.SE, signals_all=[czsc.Signal("日线_test_state_平空_任意_任意_0")])],
        t0=True,
        stop_loss=100000,
        timeout=0,
    )
    rows = [{"日线_test_state": "空" if i % 12 == 0 else "平空" if i % 12 == 7 else "其他"} for i in range(len(frame))]
    _, weights = position.evaluate_signals(bars, rows)
    assert min(weights) == -1
    tables = {}
    for freq in ("日线", "60分钟"):
        tables[freq] = pd.concat(
            [
                pd.DataFrame(
                    {
                        "dt": frame.dt,
                        "price": frame.close,
                        "pos": pos,
                        "pos_name": freq + name,
                        "n1b": (frame.close.shift(-1) / frame.close - 1) * 10000,
                    }
                )
                for name, pos in [("多头", 0), ("空头", weights)]
            ],
            ignore_index=True,
        )
    tables["pairs"] = pd.DataFrame()
    source = frame.rename(columns={"vol": "volume"})
    monkeypatch.setattr(module.pd, "read_parquet", lambda *_: source.copy())
    monkeypatch.setattr(module, "resample_bars", lambda *_args, **_kwargs: bars, raising=False)
    monkeypatch.setattr(module, "futures_daily_bars", lambda *_args, **_kwargs: bars)
    strategy = getattr(module, "CommodityStrategy", getattr(module, "CommoditySlopeTrendStrategy", None))
    monkeypatch.setattr(
        strategy,
        "backtest",
        lambda self, *_args, **_kwargs: SimpleNamespace(
            holds_arrow=getattr(self, "freq_name", "日线"), pairs_arrow="pairs"
        ),
    )
    monkeypatch.setattr(module, "_arrow_to_df", lambda token: tables[token].copy())
    actual = getattr(module, function_name)(Path("TEST_15m.parquet"))
    expected = czsc.decision_execution_ledger(
        frame.open.tolist(), weights, fee_rate=0.0001, terminal_close=float(frame.close.iloc[-1])
    )
    net = np.asarray(expected.net_returns[:-1])
    net[-1] += expected.net_returns[-1]
    if module_name == "run_all_commodities_backtest":
        np.testing.assert_allclose(actual[1]["strat_ret"], net)
    else:
        assert actual is not None
        np.testing.assert_allclose(list(actual["daily_rets"].values()), net)


def test_capital_warmup_never_backfills_future_volatility():
    from scripts.run_capital_management_simulation import causal_capital_leverage

    frame = generate_symbol_kines("TEST", "日线", "20220101", "20220201", seed=82)
    returns = frame.close.pct_change().fillna(0)
    changed = returns.copy()
    changed.iloc[4:] *= 10
    np.testing.assert_array_equal(causal_capital_leverage(returns).iloc[:5], [4.0] * 5)
    pd.testing.assert_series_equal(causal_capital_leverage(returns).iloc[:5], causal_capital_leverage(changed).iloc[:5])


def test_futures_daily_input_combines_friday_night_and_saturday_without_partial_future():
    from scripts.cta_execution import futures_daily_frame

    frame = generate_symbol_kines("TEST", "15分钟", "20240517", "20240520", seed=82).iloc[:5].copy()
    frame["dt"] = pd.to_datetime(
        ["2024-05-17 15:00", "2024-05-17 21:15", "2024-05-18 00:30", "2024-05-20 09:15", "2024-05-20 15:00"]
    )
    result = futures_daily_frame(frame)
    assert result.dt.astype(str).tolist() == ["2024-05-17 15:00:00", "2024-05-20 15:00:00"]
    assert result.open.iloc[-1] == frame.open.iloc[1]
    assert result.close.iloc[-1] == frame.close.iloc[-1]
    assert len(futures_daily_frame(frame.iloc[:-1])) == 1


def test_weighting_stats_keep_fixed_initial_capital_returns():
    from scripts.run_portfolio_weighting_comparison import calc_stats

    frame = generate_symbol_kines("TEST", "日线", "20220101", "20220201", seed=82)
    returns = frame.close.pct_change().fillna(0.0).to_numpy()
    stats = calc_stats(returns)
    assert stats["total_return"] == round(float(returns.sum() * 100), 2)


def test_weighting_main_warmup_cannot_use_fifth_day(monkeypatch, tmp_path):
    import io
    import json

    from scripts import run_portfolio_weighting_comparison as module

    frame = generate_symbol_kines("TEST", "日线", "20220101", "20220201", seed=82).iloc[:12]
    dates = frame.dt.dt.strftime("%Y-%m-%d").tolist()
    returns = frame.close.pct_change().fillna(0).to_numpy()
    captured = []

    class Sink(io.StringIO):
        def close(self):
            if not self.closed:
                captured.append(json.loads(self.getvalue()))
            super().close()

    class Executor:
        def __init__(self, **_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def map(self, *_):
            return results

    monkeypatch.setattr(module, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(module, "open", lambda *_args, **_kwargs: Sink(), raising=False)
    for changed in (False, True):
        results = []
        for i in range(20):
            r = returns.copy() * (1 + i / 20)
            if changed and i == 0:
                r[4] *= 50
            data = dict(zip(dates, r, strict=True))
            inputs = frame.copy().assign(symbol=f"TEST{i}", pos=1.0)
            inputs.loc[inputs.index[-1], "pos"] = 0.0
            results.append(
                {
                    "symbol": f"TEST{i}",
                    "code": str(i),
                    "sector": "测试",
                    "daily_rets": data,
                    "hourly_rets": data,
                    "daily_inputs": inputs,
                    "hourly_inputs": inputs,
                }
            )
        module.main(output_path=tmp_path / "weighting.json")
    np.testing.assert_allclose(
        captured[0]["strategies"]["risk_parity"]["stats"]["nav"][:4],
        captured[1]["strategies"]["risk_parity"]["stats"]["nav"][:4],
        atol=0.0,
    )


def test_daily_portfolio_and_capital_entrypoints_share_actual_execution(tmp_path):
    from scripts import (
        build_complete_dashboard_data,
        calc_overall_nav_multiprocess,
        run_capital_management_simulation,
        run_portfolio_weighting_comparison,
    )

    frame = generate_symbol_kines("TEST", "日线", "20220101", "20220201", seed=82).iloc[:12].copy()
    positions = np.sign(frame.close.diff().fillna(0.0)).to_numpy()
    frame["pos"] = positions
    native = czsc.decision_execution_ledger(
        frame.open.tolist(), positions.tolist(), fee_rate=0.0001, terminal_close=float(frame.close.iloc[-1])
    )
    net = np.asarray(native.net_returns[:-1])
    net[-1] += native.net_returns[-1]
    dates = frame.dt.dt.strftime("%Y-%m-%d").tolist()
    result_rows = [
        {
            "symbol": f"TEST{i}",
            "code": str(i),
            "sector": "测试",
            "daily_rets": dict(zip(dates, net, strict=True)),
            "hourly_rets": dict(zip(dates, net, strict=True)),
            "daily_inputs": frame.copy(),
            "hourly_inputs": frame.copy(),
        }
        for i in range(20)
    ]
    overall = calc_overall_nav_multiprocess.main(output_path=tmp_path / "overall.json", results=result_rows)
    dashboard = build_complete_dashboard_data.build_data(output_path=tmp_path / "dashboard.json", results=result_rows)
    weights = run_portfolio_weighting_comparison.main(output_path=tmp_path / "weighting.json", results=result_rows)
    capital = run_capital_management_simulation.run_simulation(
        input_path=tmp_path / "weighting.json", output_path=tmp_path / "capital.json"
    )
    expected = round(float(net.sum() * 100), 2)
    assert overall["daily"]["total_return"] == expected
    assert dashboard["aligned"]["metrics"]["daily"]["total_return"] == expected
    assert weights["strategies"]["equal_weight"]["stats"]["total_return"] == expected
    assert capital["models"][0]["total_return"] == expected
    assert capital["models"][2]["total_return"] == round(float(net.sum() * 400), 2)
    assert capital["models"][2]["fee_return_pct"] == pytest.approx(sum(native.cost_returns) * 400)


def test_daily_continuous_sizing_executes_next_open_and_adds_returns():
    from scripts.run_slope_with_continuous_sizing import run_continuous_portfolio

    frame = generate_symbol_kines("TEST", "日线", "20220101", "20220501", seed=82)
    frame["dt"] = frame.dt.dt.strftime("%Y-%m-%d")
    frame = frame.assign(signal=1.0, atr_mult=1.0).set_index("dt")
    result = run_continuous_portfolio({f"TEST{i}": frame.copy() for i in range(20)}, enable_target_vol=False)
    observed = frame.iloc[25:]
    native = czsc.decision_execution_ledger(
        observed.open.tolist(), [1.0] * len(observed), fee_rate=0.0001, terminal_close=float(observed.close.iloc[-1])
    )
    assert result["tot_ret"] == round(float(sum(native.net_returns) * 100), 2)
    assert result["fee_return_pct"] == pytest.approx(sum(native.cost_returns) * 100)


def test_next_open_panel_charges_price_drift_and_terminal_exit():
    from scripts.cta_execution import next_open_panel_ledger

    frame = generate_symbol_kines("TEST", "30分钟", "20220101", "20220103", seed=82).iloc[:12].copy()
    decisions = pd.DataFrame({"TEST": 0.7}, index=pd.DatetimeIndex(frame.dt))
    actual = next_open_panel_ledger(frame, decisions, 0.001)
    expected = czsc.decision_execution_ledger(
        frame.open.tolist(), [0.7] * len(frame), fee_rate=0.001, terminal_close=float(frame.close.iloc[-1])
    )
    assert actual["fee"].to_numpy().sum() == pytest.approx(sum(expected.cost_returns))
    assert actual["net"].to_numpy().sum() == pytest.approx(sum(expected.net_returns))
    assert actual["fee"].to_numpy().sum() > 2 * 0.7 * 0.001


@pytest.mark.parametrize("module_name", ["run_slope_continuous_signal_cta", "test_slope_60m_unchanged_params"])
def test_slope_full_entrypoint_writes_reconciled_native_ledger(monkeypatch, tmp_path, module_name):
    import json

    module = import_module(f"scripts.{module_name}")

    frame = generate_symbol_kines("TEST", "15分钟", "20240101", "20240301", seed=93)
    frame = frame.rename(columns={"vol": "volume"})
    monkeypatch.setattr(Path, "glob", lambda *_: [Path("TEST_15m.parquet")])
    monkeypatch.setattr(module.pd, "read_parquet", lambda *_: frame.copy())
    output = tmp_path / "run.json"
    if module_name == "run_slope_continuous_signal_cta":
        result = module.run_slope_continuous_system(module.SlopeContinuousConfig(), output_path=output)
    else:
        result = module.run_test(output_path=output)
    saved = json.loads(output.read_text())
    assert result == saved
    np.testing.assert_allclose(
        np.asarray(result["daily_gross"]) - result["daily_fees"], result["daily_net"], atol=1e-12
    )
    assert result["accounting"] == "fixed_initial_capital_next_open"
    assert result["summary"]["tot_ret"] == pytest.approx(round(sum(result["daily_net"]) * 100, 2))


@pytest.mark.parametrize("all_flat", [False, True])
def test_saved_execution_validation_replays_sparse_quotes_and_friction(tmp_path, all_flat):
    import pickle

    from examples.multi_factor_cta.audit_protocol import (
        UnverifiedArtifactError,
        build_fingerprint,
        write_verified_artifact,
    )
    from scripts.validate_cta_execution_artifacts import validate_bundle, validate_directory

    frame = generate_symbol_kines("TEST", "30分钟", "20220101", "20220105", seed=82).iloc[:24]
    index = pd.DatetimeIndex(frame.dt)
    prices = pd.DataFrame({"A": frame.open.to_numpy(), "B": frame.close.to_numpy()}, index=index)
    prices.loc[index[:3], "B"] = np.nan
    prices.loc[index[5::4], "B"] = np.nan
    mask = prices.notna()
    returns = pd.DataFrame(0.0, index=index, columns=prices.columns)
    weights = returns.copy()
    for symbol in prices:
        observed = prices[symbol].dropna()
        returns.loc[observed.index, symbol] = observed.pct_change().fillna(0.0)
        target = np.where(observed.diff().fillna(1.0) >= 0, 0.25, -0.25)
        if all_flat:
            target[:] = 0.0
        target[-1] = 0.0
        weights[symbol] = pd.Series(target, index=observed.index).reindex(index).ffill().fillna(0.0)
    fees = {"A": 0.0002, "B": 0.0003}
    native = {
        symbol: czsc.executed_weight_ledger(
            returns[symbol].tolist(), weights[symbol].tolist(), fee_rate=fees[symbol], tradable=mask[symbol].tolist()
        )
        for symbol in prices
    }
    bundle = {
        "returns_df": returns,
        "execution_prices_df": prices,
        "weights_df": weights,
        "tradable_mask": mask,
        "fee_rates": fees,
        "variant": "regression",
        "fold": 1,
    }
    for key, field in [
        ("gross_returns_df", "gross_returns"),
        ("cost_returns_df", "cost_returns"),
        ("turnover_df", "turnover"),
    ]:
        bundle[key] = pd.DataFrame({symbol: getattr(ledger, field) for symbol, ledger in native.items()}, index=index)
    source = tmp_path / "saved.pkl"
    fingerprint = build_fingerprint(source_paths=[__file__], data_paths=[__file__], config={"regression": True})
    write_verified_artifact(source, pickle.dumps(bundle), fingerprint, output_dir=tmp_path, readonly_roots=())
    result = validate_bundle(source, tmp_path / "validation")
    assert result["passed"], result
    stresses = pd.DataFrame(result["friction"])
    turnover = float(bundle["turnover_df"].to_numpy().sum())
    np.testing.assert_allclose(
        stresses.net_return_pct.iloc[0] - stresses.net_return_pct,
        stresses.extra_single_side_bp * turnover / 100,
        atol=1e-10,
    )
    fills = pd.read_parquet(tmp_path / "validation/backtrader_orders.parquet")
    if all_flat:
        assert fills.empty
    else:
        assert fills.loc[fills.symbol == "B", "dt"].min() == prices.B.first_valid_index()
    assets = pd.read_csv(tmp_path / "validation/asset_reconciliation.csv")
    np.testing.assert_allclose(assets.terminal_quantity, 0.0, atol=1e-9)
    assert validate_directory(tmp_path, tmp_path / "batch")[0]["passed"]
    stitched = pd.read_csv(tmp_path / "batch/variant_friction_sensitivity.csv")
    np.testing.assert_allclose(stitched.net_return_pct, stresses.net_return_pct)
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(UnverifiedArtifactError, match="checksum mismatch"):
        validate_bundle(source, tmp_path / "changed")
