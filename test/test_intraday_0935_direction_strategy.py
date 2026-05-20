# -*- coding: utf-8 -*-
import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

for mod in [
    "clickhouse_connect",
    "clickhouse_connect.driver",
    "clickhouse_connect.driver.client",
    "clickhouse_connect.driver.httpclient",
    "clickhouse_connect.driver.compression",
]:
    sys.modules.setdefault(mod, MagicMock())

from czsc.core import Freq, RawBar
from czsc.traders.base import generate_czsc_signals


def load_example_module():
    file_path = Path(__file__).resolve().parents[1] / "examples" / "intraday_0935_direction_strategy.py"
    spec = importlib.util.spec_from_file_location("intraday_0935_direction_strategy", file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_intraday_bars(day: str, open_price: float, price_map: dict[str, float]) -> list[RawBar]:
    """生成单日 A 股 1 分钟 K 线。"""
    day = pd.Timestamp(day)
    morning = pd.date_range(f"{day:%Y-%m-%d} 09:31:00", f"{day:%Y-%m-%d} 11:30:00", freq="1min")
    afternoon = pd.date_range(f"{day:%Y-%m-%d} 13:01:00", f"{day:%Y-%m-%d} 15:00:00", freq="1min")
    dts = morning.append(afternoon)

    bars = []
    prev_close = open_price
    for i, dt in enumerate(dts):
        hhmm = dt.strftime("%H%M")
        close = float(price_map.get(hhmm, prev_close))
        high = max(prev_close, close) + 0.05
        low = min(prev_close, close) - 0.05
        bars.append(
            RawBar(
                symbol="000852.XSHG",
                id=i,
                freq=Freq.F1,
                dt=dt,
                open=prev_close,
                close=close,
                high=high,
                low=low,
                vol=1000,
                amount=1000 * close,
            )
        )
        prev_close = close
    return bars


def test_intraday_signals_emit_at_expected_minutes():
    module = load_example_module()
    bars = make_intraday_bars("2024-01-02", 100, {"0933": 101, "1455": 101.2})

    tactic = module.Intraday0935DirectionStrategy(symbol="000852.XSHG")
    df = generate_czsc_signals(bars, signals_config=tactic.signals_config, sdt=bars[0].dt, init_n=1, df=True)

    direction_col = "1分钟_D1T0933_日内方向V260424"
    exit_col = "1分钟_D1T1455_日内平仓V260424"

    df = df.copy()
    df.loc[:, "dt"] = pd.to_datetime(df["dt"])
    row_0932 = df[df["dt"].dt.strftime("%H%M") == "0932"].iloc[-1]
    row_0933 = df[df["dt"].dt.strftime("%H%M") == "0933"].iloc[-1]
    row_1455 = df[df["dt"].dt.strftime("%H%M") == "1455"].iloc[-1]

    assert row_0932[direction_col] == "其他_任意_任意_0"
    assert row_0933[direction_col] == "看多_任意_任意_0"
    assert row_1455[exit_col] == "平仓_任意_任意_0"


def test_intraday_long_stop_loss_closes_once_without_reentry():
    module = load_example_module()
    bars = make_intraday_bars("2024-01-02", 100, {"0933": 101, "0934": 99.5})

    tactic = module.Intraday0935DirectionStrategy(symbol="000852.XSHG")
    trader = tactic.init_trader(bars, sdt=bars[0].dt, n=1)
    long_pos = next(x for x in trader.positions if "多头" in x.name)

    assert len(long_pos.pairs) == 1
    assert long_pos.pairs[0]["平仓时间"].strftime("%H%M") == "0934"
    assert "100BP止损" in long_pos.pairs[0]["事件序列"]
    assert len([x for x in long_pos.operates if x["op"].value == "开多"]) == 1


def test_intraday_short_position_holds_until_1455():
    module = load_example_module()
    bars = make_intraday_bars("2024-01-02", 100, {"0933": 99, "1455": 98.7, "1500": 98.8})

    tactic = module.Intraday0935DirectionStrategy(symbol="000852.XSHG")
    trader = tactic.init_trader(bars, sdt=bars[0].dt, n=1)
    short_pos = next(x for x in trader.positions if "空头" in x.name)

    assert len(short_pos.pairs) == 1
    assert short_pos.pairs[0]["开仓时间"].strftime("%H%M") == "0933"
    assert short_pos.pairs[0]["平仓时间"].strftime("%H%M") == "1455"
    assert "平空" in short_pos.pairs[0]["事件序列"]
