"""Intraday time signal parsing must preserve HHMM as strings."""

from __future__ import annotations

from czsc import get_signals_config


def test_intraday_direction_time_params_are_strings() -> None:
    """HHMM placeholders are consumed by native signals via ``params.str``."""
    cfg = get_signals_config(["15分钟_D1T0945_日内方向V260424_看多_任意_任意_0"])
    assert cfg == [
        {
            "name": "bar_intraday_direction_V260424",
            "freq": "15分钟",
            "decision_time": "0945",
        }
    ]

    cfg = get_signals_config(["15分钟_D1T1500_日内平仓V260424_平仓_任意_任意_0"])
    assert cfg == [
        {
            "name": "bar_intraday_exit_V260424",
            "freq": "15分钟",
            "exit_time": "1500",
        }
    ]
