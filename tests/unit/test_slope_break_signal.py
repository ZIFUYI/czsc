"""测试 tas_slope_break_V261003 线性回归斜率突破信号与策略回测。"""

from __future__ import annotations

import pandas as pd
import pyarrow.ipc as ipc

import czsc
from czsc import CZSC, CzscStrategyBase, Event, Freq, Position, format_standard_kline
from czsc.mock import generate_symbol_kines


def _arrow_to_df(arrow_bytes: bytes) -> pd.DataFrame:
    with ipc.open_file(arrow_bytes) as reader:
        return reader.read_all().to_pandas()


def test_slope_break_signal_direct_call():
    """测试通过 czsc._native.call_signal 直接调用信号。"""
    df = generate_symbol_kines("RB9999", "日线", "20230101", "20240101", seed=42)
    bars = format_standard_kline(df, freq=Freq.D)
    c = CZSC(bars)

    sigs = czsc._native.call_signal("tas_slope_break_V261003", c, {"n": 20, "di": 1})
    assert len(sigs) == 1
    sig_str = str(sigs[0])
    assert "日线_D1N20_斜率突破V261003" in sig_str
    assert any(k in sig_str for k in ["看多", "看空", "其他"])


def test_slope_break_signal_config_parsing():
    """测试从完整 7 段式信号字符串逆向推导 signals_config。"""
    sig_doc = "日线_D1N20_斜率突破V261003_看多_向上突破_斜率向上_0"
    cfg = czsc.get_signals_config([sig_doc])
    assert len(cfg) == 1
    assert cfg[0] == {"name": "tas_slope_break_V261003", "freq": "日线", "n": 20, "di": 1}


class CommoditySlopeTrendStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        long_opens = [
            Event.load({"operate": "开多", "signals_all": ["日线_D1N20_斜率突破V261003_看多_向上突破_任意_0"]})
        ]
        long_exits = [Event.load({"operate": "平多", "signals_all": ["日线_D1N20_斜率突破V261003_看空_任意_任意_0"]})]
        pos_long = Position(name="日线多头", symbol=self.symbol, opens=long_opens, exits=long_exits)

        short_opens = [
            Event.load({"operate": "开空", "signals_all": ["日线_D1N20_斜率突破V261003_看空_向下跌破_任意_0"]})
        ]
        short_exits = [Event.load({"operate": "平空", "signals_all": ["日线_D1N20_斜率突破V261003_看多_任意_任意_0"]})]
        pos_short = Position(name="日线空头", symbol=self.symbol, opens=short_opens, exits=short_exits)
        return [pos_long, pos_short]


def test_commodity_slope_strategy_backtest():
    """测试商品期货斜率趋势策略回测运行。"""
    df = generate_symbol_kines("RB9999", "日线", "20190101", "20240101", seed=42)
    bars = format_standard_kline(df, freq=Freq.D)

    strat = CommoditySlopeTrendStrategy(symbol="RB9999")
    res = strat.backtest(bars, sdt="2020-01-01")

    pairs = _arrow_to_df(res.pairs_arrow)
    signals = _arrow_to_df(res.signals_arrow)

    assert not signals.empty
    assert "日线_D1N20_斜率突破V261003" in signals.columns
    assert len(pairs) > 0
    assert set(pairs["交易方向"].unique()).issubset({"多头", "空头"})
