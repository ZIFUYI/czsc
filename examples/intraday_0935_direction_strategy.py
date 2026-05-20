# -*- coding: utf-8 -*-
"""
1分钟日内 9:33 方向策略示例

策略规则：
1. 每天 9:33 比较当日首根分钟K线开盘价与 9:33 收盘价。
2. 涨跌幅 >= 0 开多，否则开空。
3. 14:55 强制平仓，不隔夜。
4. 盘中固定 1% 止损；由于开仓信号只在单点触发，止损后当日不会再次开仓。
"""
import os
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pandas as pd
from loguru import logger

# 兼容当前仓库在部分 Python 版本下的可选依赖导入问题
for mod in [
    "clickhouse_connect",
    "clickhouse_connect.driver",
    "clickhouse_connect.driver.client",
    "clickhouse_connect.driver.httpclient",
    "clickhouse_connect.driver.compression",
]:
    sys.modules.setdefault(mod, MagicMock())

from czsc import BarGenerator, CzscTrader, Event, Operate, Position, RawBar, Signal
from czsc.strategies import CzscStrategyBase


def event_from_factor(name: str, operate: Operate, factor: dict) -> Event:
    """用单个因子创建事件，显式保留 Signal -> Factor -> Event 的组织方式。"""
    return Event(
        name=name,
        operate=operate,
        signals_all=[Signal(x) for x in factor.get("signals_all", [])],
        signals_any=[Signal(x) for x in factor.get("signals_any", [])],
        signals_not=[Signal(x) for x in factor.get("signals_not", [])],
    )


def create_intraday_factors(base_freq: str = "1分钟", decision_time: str = "0933", exit_time: str = "1455"):
    """创建日内方向策略使用的因子集合。"""
    direction_key = f"{base_freq}_D1T{decision_time}_日内方向V260424"
    exit_key = f"{base_freq}_D1T{exit_time}_日内平仓V260424"

    return {
        "long_open": {
            "name": f"{decision_time}方向看多",
            "signals_all": [f"{direction_key}_看多_任意_任意_0"],
            "signals_any": [],
            "signals_not": [],
        },
        "short_open": {
            "name": f"{decision_time}方向看空",
            "signals_all": [f"{direction_key}_看空_任意_任意_0"],
            "signals_any": [],
            "signals_not": [],
        },
        "time_exit": {
            "name": "1455尾盘平仓",
            "signals_all": [f"{exit_key}_平仓_任意_任意_0"],
            "signals_any": [],
            "signals_not": [],
        },
    }


def create_intraday_events(base_freq: str = "1分钟", decision_time: str = "0933", exit_time: str = "1455"):
    """创建开多、开空、平多、平空事件。"""
    factors = create_intraday_factors(base_freq=base_freq, decision_time=decision_time, exit_time=exit_time)
    return {
        "long_open": event_from_factor(f"{decision_time}开多", Operate.LO, factors["long_open"]),
        "short_open": event_from_factor(f"{decision_time}开空", Operate.SO, factors["short_open"]),
        "long_exit": event_from_factor("1455平多", Operate.LE, factors["time_exit"]),
        "short_exit": event_from_factor("1455平空", Operate.SE, factors["time_exit"]),
    }


def create_intraday_positions(
    symbol: str,
    base_freq: str = "1分钟",
    decision_time: str = "0933",
    exit_time: str = "1455",
    stop_loss_bp: int = 100,
):
    """创建多头和空头两个独立持仓。"""
    events = create_intraday_events(base_freq=base_freq, decision_time=decision_time, exit_time=exit_time)

    # interval 使用 15 小时作为额外保护，避免未来扩展额外开仓信号时出现同日重复开仓。
    common_kwargs = {
        "symbol": symbol,
        "interval": 15 * 3600,
        "timeout": 1000,
        "stop_loss": stop_loss_bp,
        "t0": True,
    }
    return [
        Position(
            name=f"{base_freq}{decision_time}方向多头",
            opens=[events["long_open"]],
            exits=[events["long_exit"]],
            **common_kwargs,
        ),
        Position(
            name=f"{base_freq}{decision_time}方向空头",
            opens=[events["short_open"]],
            exits=[events["short_exit"]],
            **common_kwargs,
        ),
    ]


class Intraday0935DirectionStrategy(CzscStrategyBase):
    """日内 9:33 方向策略。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_freq_value = kwargs.get("base_freq", "1分钟")
        self.decision_time = kwargs.get("decision_time", "0933")
        self.exit_time = kwargs.get("exit_time", "1455")
        self.stop_loss_bp = int(kwargs.get("stop_loss_bp", 100))

    @property
    def positions(self):
        return create_intraday_positions(
            symbol=self.symbol,
            base_freq=self.base_freq_value,
            decision_time=self.decision_time,
            exit_time=self.exit_time,
            stop_loss_bp=self.stop_loss_bp,
        )

    @property
    def signals_config(self):
        return [
            {
                "name": "bar_intraday_direction_V260424",
                "freq": self.base_freq_value,
                "decision_time": self.decision_time,
            },
            {
                "name": "bar_intraday_exit_V260424",
                "freq": self.base_freq_value,
                "exit_time": self.exit_time,
            },
        ]

    def init_trader(self, bars: List[RawBar], **kwargs):
        """创建并回放一个 CzscTrader，兼容旧示例的调用方式。"""
        bg = BarGenerator(base_freq=self.base_freq, freqs=self.freqs, max_count=int(kwargs.get("max_count", 5000)))
        trader = CzscTrader(bg, positions=self.positions, signals_config=self.signals_config)
        sdt = pd.to_datetime(kwargs.get("sdt")) if kwargs.get("sdt") else None
        for bar in bars:
            if sdt is not None and pd.to_datetime(bar.dt) < sdt:
                bg.update(bar)
                continue
            trader.update(bar)
        return trader


def ensure_jq_token():
    """优先使用环境变量初始化聚宽 token，否则沿用本机已有 token。"""
    from czsc.connectors.jq_connector import set_token

    user = os.getenv("JQDATA_USERNAME")
    password = os.getenv("JQDATA_PASSWORD")
    token_path = Path.home() / "jq.token"

    if user and password:
        set_token(user, password)
        logger.info("已使用环境变量初始化聚宽 token")
        return

    if token_path.exists():
        logger.info(f"检测到已有聚宽 token 文件：{token_path}")
        return

    raise RuntimeError("未找到聚宽凭证，请设置 JQDATA_USERNAME / JQDATA_PASSWORD，或先执行 set_token")


def read_bars(symbol, freq, sdt, edt, fq="前复权", **kwargs):
    """读取聚宽分钟K线，返回 RawBar 列表。"""
    from czsc.connectors.jq_connector import get_kline_period, freq_cn2jq

    jq_freq = freq_cn2jq[str(freq)]
    fq_flag = fq == "前复权"
    bars = get_kline_period(symbol=symbol, start_date=sdt, end_date=edt, freq=jq_freq, fq=fq_flag)
    logger.info(f"{symbol} {freq} 读取完成，共 {len(bars)} 根K线")
    return bars


def validate_trader(trader, decision_time="0933", exit_time="1455"):
    """检查交易记录是否符合策略时间约束。"""
    results = []
    for pos in trader.positions:
        if not pos.pairs:
            continue

        df_pairs = pd.DataFrame(pos.pairs)
        open_times = pd.to_datetime(df_pairs["开仓时间"]).dt.strftime("%H%M")
        close_times = pd.to_datetime(df_pairs["平仓时间"]).dt.strftime("%H%M")
        is_stop = df_pairs["事件序列"].str.contains("止损", regex=False)

        results.append(
            {
                "position": pos.name,
                "trades": len(df_pairs),
                "open_time_ok": open_times.eq(decision_time).all(),
                "close_time_ok": (close_times.eq(exit_time) | is_stop).all(),
            }
        )
    return pd.DataFrame(results)


def run_single_symbol(symbol="000852.XSHG", sdt="20240101", edt="20240430"):
    """单标的回测和规则校验。"""
    ensure_jq_token()
    tactic = Intraday0935DirectionStrategy(symbol=symbol)
    bars = read_bars(symbol, tactic.base_freq, sdt, edt, fq="不复权")
    trader = tactic.init_trader(bars, sdt=sdt, n=200)

    df_check = validate_trader(trader, tactic.decision_time, tactic.exit_time)
    logger.info(f"规则校验结果：\n{df_check}")
    return trader


def run_batch_backtest(symbols, results_path, bar_sdt="20240101", sdt="20240201", edt="20240430", max_workers=2):
    """批量回测示例；上游 1.0 架构已移除 CTAResearch，这里保留轻量循环版本。"""
    ensure_jq_token()
    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        tactic = Intraday0935DirectionStrategy(symbol=symbol)
        bars = read_bars(symbol, tactic.base_freq, bar_sdt, edt, fq="不复权")
        res = tactic.backtest(bars, sdt=sdt)
        logger.info(f"{symbol} 回测完成：{res}")


if __name__ == "__main__":
    results_path = Path(__file__).parent / "results" / "intraday_0935_direction"
    results_path.mkdir(parents=True, exist_ok=True)
    logger.add(results_path / "intraday_0935_direction.log", rotation="1 week", encoding="utf-8")

    trader = run_single_symbol()
    for pos in trader.positions:
        if not pos.pairs:
            logger.info(f"{pos.name} 没有触发交易")
            continue

        df_pairs = pd.DataFrame(pos.pairs)
        out_file = results_path / f"{pos.name}_pairs.csv"
        df_pairs.to_csv(out_file, index=False, encoding="utf-8-sig")
        logger.info(f"{pos.name} 交易次数：{len(df_pairs)}；结果保存到：{out_file}")
