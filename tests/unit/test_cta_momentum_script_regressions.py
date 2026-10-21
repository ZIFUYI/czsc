"""Public momentum backtests must use executable orders and complete costs."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

import czsc
from czsc.mock import generate_symbol_kines
from scripts import run_advanced_cta_momentum_strategy as advanced


def test_advanced_entry_uses_next_actual_open_and_charges_terminal_exit():
    frame = generate_symbol_kines("TEST", "日线", "20240101", "20240201", seed=41).iloc[:6].copy()
    frame["dt"] = pd.to_datetime(frame.dt).dt.normalize() + pd.Timedelta(hours=15)
    frame["atr"] = (frame.high - frame.low).mean()
    frame["slope"] = 0.0
    frame["hh20"] = np.inf
    frame["ll20"] = 0.0
    frame["hh10"] = np.inf
    frame["ll10"] = 0.0
    frame.loc[frame.index[0], ["slope", "hh20"]] = [0.1, frame.close.iloc[0] * 0.99]
    data = {f"S{i:02}": frame.set_index("dt").copy() for i in range(20)}
    fee = 0.0002
    result = advanced.run_cta_backtest(data, top_k=1, leverage=1, trailing_mult=1e8, stop_atr=1e8, fee_rate=fee)
    expected = czsc.decision_execution_ledger(
        frame.open.tolist(), [1.0] * len(frame), fee_rate=fee, terminal_close=float(frame.close.iloc[-1])
    )
    native = np.asarray(expected.net_returns)
    native[-2] += native[-1]
    np.testing.assert_allclose(result["daily_net"], native[:-1])
    assert result["daily_net"][0] == 0
    assert sum(result["daily_fees"]) > 0
    assert result["nav"][-1] == 1 + sum(result["daily_net"])


def test_momentum_dashboard_import_does_not_load_data_or_overwrite_outputs(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Importing the dashboard must not run a backtest or write artifacts")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "glob", forbidden)
        guard.setattr(Path, "write_text", forbidden)
        dashboard = import_module("scripts.render_top_momentum_dashboard")
    assert callable(dashboard.run_momentum_backtest)


def test_top_momentum_does_not_earn_the_signal_close_to_next_open_gap():
    dashboard = import_module("scripts.render_top_momentum_dashboard")
    data = {}
    for i in range(20):
        frame = generate_symbol_kines(f"S{i:02}", "日线", "20240101", "20240201", seed=41 + i).iloc[:6].copy()
        frame["dt"] = pd.to_datetime(frame.dt).dt.normalize() + pd.Timedelta(hours=15)
        frame["slope"] = 10 - i
        data[f"S{i:02}"] = frame.set_index("dt")
    fee = 0.0002
    result = dashboard.run_momentum_backtest(data, k=1, fee_rate=fee, warmup_days=0)
    expected_net = np.zeros(6)
    for symbol, target in [("S00", 0.5), ("S19", -0.5)]:
        frame = data[symbol]
        ledger = czsc.decision_execution_ledger(
            frame.open.tolist(), [target] * len(frame), fee_rate=fee, terminal_close=float(frame.close.iloc[-1])
        )
        net = np.asarray(ledger.net_returns)
        net[-2] += net[-1]
        expected_net += net[:-1]
    np.testing.assert_allclose(result["daily_net"], expected_net)
    assert result["daily_net"][0] == 0
    assert result["fee_return_pct"] > 0


def test_daily_loader_preserves_atr_warmup_and_uses_completed_futures_days(tmp_path):
    frame = generate_symbol_kines("TEST", "15分钟", "20240101", "20240701", seed=48)
    frame["dt"] = pd.to_datetime(frame.dt) - pd.Timedelta(minutes=15)
    frame = frame.rename(columns={"vol": "volume"})
    frame.to_parquet(tmp_path / "TEST_15m.parquet")
    result = advanced.load_all_daily_data(tmp_path)["TEST"]
    assert result.index.hour.tolist() == [15] * len(result)
    assert not (result.index.dayofweek >= 5).any()
    assert result.atr.iloc[:4].isna().all()
    assert result.atr.iloc[4:].notna().all()


def test_future_prices_cannot_change_advanced_targets_or_warmup():
    frame = generate_symbol_kines("TEST", "日线", "20240101", "20240701", seed=84).set_index("dt")
    cutoff = frame.index[70]
    changed = frame.copy()
    changed.loc[changed.index > cutoff, ["open", "high", "low", "close"]] *= 1.5
    first = {f"S{i:02}": advanced.add_daily_indicators(frame) for i in range(20)}
    second = {f"S{i:02}": advanced.add_daily_indicators(changed) for i in range(20)}
    pd.testing.assert_frame_equal(
        advanced.build_advanced_targets(first).loc[:cutoff], advanced.build_advanced_targets(second).loc[:cutoff]
    )
