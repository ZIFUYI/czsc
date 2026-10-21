# ruff: noqa: E402
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc

from czsc import CzscStrategyBase, Event, Freq, Position, format_standard_kline, resample_bars
from scripts.cta_execution import (
    equal_weight_execution,
    futures_daily_bars,
    signed_holds_execution,
    trading_day_returns,
)
from scripts.cta_paths import research_data_dir, research_output_path


def _arrow_to_df(arrow_bytes: bytes) -> pd.DataFrame:
    with ipc.open_file(arrow_bytes) as reader:
        return reader.read_all().to_pandas()


class CommodityStrategy(CzscStrategyBase):
    def __init__(self, symbol: str, freq_name: str = "日线", n: int = 20, **kwargs):
        super().__init__(symbol=symbol, freq_name=freq_name, n=n, **kwargs)
        self.freq_name = freq_name
        self.n = n

    @property
    def positions(self) -> list[Position]:
        f = self.freq_name
        n = self.n
        pos_long = Position(
            name=f"{f}多头",
            symbol=self.symbol,
            opens=[
                Event.load({"operate": "开多", "signals_all": [f"{f}_D1N{n}_斜率突破V261003_看多_向上突破_任意_0"]})
            ],
            exits=[Event.load({"operate": "平多", "signals_all": [f"{f}_D1N{n}_斜率突破V261003_看空_任意_任意_0"]})],
            interval=0,
            timeout=0,
            stop_loss=0,
        )
        pos_short = Position(
            name=f"{f}空头",
            symbol=self.symbol,
            opens=[
                Event.load({"operate": "开空", "signals_all": [f"{f}_D1N{n}_斜率突破V261003_看空_向下跌破_任意_0"]})
            ],
            exits=[Event.load({"operate": "平空", "signals_all": [f"{f}_D1N{n}_斜率突破V261003_看多_任意_任意_0"]})],
            interval=0,
            timeout=0,
            stop_loss=0,
        )
        return [pos_long, pos_short]


def backtest_one_file(file_path_str: str):
    f = Path(file_path_str)
    sym = f.stem.replace("_15m", "")
    try:
        df_15m = pd.read_parquet(f)
        df_15m["dt"] = pd.to_datetime(df_15m["dt"]) + pd.Timedelta(minutes=15)
        if len(df_15m) < 200:
            return None
        df_15m["symbol"] = sym
        df_15m["vol"] = df_15m["volume"]
        df_15m["amount"] = df_15m["vol"] * df_15m["close"] * 10
        bars_15m = format_standard_kline(df_15m, freq=Freq.F15)
        bars_d = futures_daily_bars(df_15m)
        bars_60m = resample_bars(bars_15m, target_freq=Freq.F60)

        # 1. 日线回测 (N=20)
        s_d = CommodityStrategy(symbol=sym, freq_name="日线", n=20)
        res_d = s_d.backtest(bars_d, sdt="2024-03-01")
        h_d = _arrow_to_df(res_d.holds_arrow)
        daily_series = {}
        daily_inputs = pd.DataFrame()
        if not h_d.empty and "pos_name" in h_d.columns:
            p_l = h_d[h_d["pos_name"] == "日线多头"].set_index("dt")
            p_s = h_d[h_d["pos_name"] == "日线空头"].set_index("dt")
            if not p_l.empty:
                daily_execution = signed_holds_execution(p_l, p_s, bars_d)
                daily_inputs = daily_execution.reset_index()
                day_pnl = daily_execution["strat_ret"]
                for dt_val, ret_val in day_pnl.items():
                    d_str = dt_val.strftime("%Y-%m-%d")
                    daily_series[d_str] = float(ret_val)

        # 2. 60m 回测 (N=80)
        s_h = CommodityStrategy(symbol=sym, freq_name="60分钟", n=80)
        res_h = s_h.backtest(bars_60m, sdt="2024-03-01")
        h_h = _arrow_to_df(res_h.holds_arrow)
        hourly_series = {}
        hourly_inputs = pd.DataFrame()
        if not h_h.empty and "pos_name" in h_h.columns:
            p_l = h_h[h_h["pos_name"] == "60分钟多头"].set_index("dt")
            p_s = h_h[h_h["pos_name"] == "60分钟空头"].set_index("dt")
            if not p_l.empty:
                hourly_execution = signed_holds_execution(p_l, p_s, bars_60m)
                hourly_inputs = hourly_execution.reset_index()
                day_h_pnl = trading_day_returns(hourly_execution["strat_ret"])
                for d_obj, ret_val in day_h_pnl.items():
                    d_str = d_obj.strftime("%Y-%m-%d")
                    hourly_series[d_str] = float(ret_val)

        return {
            "symbol": sym,
            "daily_rets": daily_series,
            "hourly_rets": hourly_series,
            "daily_inputs": daily_inputs,
            "hourly_inputs": hourly_inputs,
        }
    except Exception:
        return None


def main(*, output_path=None, results=None):
    data_dir = research_data_dir()
    files = sorted(data_dir.glob("*.parquet"))
    print("开始并行计算 56 个品种整体净值，进程数: 8...")

    if results is None:
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = [r for r in executor.map(backtest_one_file, [str(f) for f in files]) if r is not None]

    print(f"成功计算有效品种数: {len(results)}")

    daily_by_date = {}
    hourly_by_date = {}
    symbol_stats = []

    for item in results:
        sym = item["symbol"]
        d_rets = item["daily_rets"]
        h_rets = item["hourly_rets"]

        d_tot = np.sum(list(d_rets.values())) * 100 if d_rets else 0
        h_tot = np.sum(list(h_rets.values())) * 100 if h_rets else 0
        symbol_stats.append(
            {
                "symbol": sym,
                "daily_return": round(float(d_tot), 2),
                "hourly_return": round(float(h_tot), 2),
                "diff": round(float(h_tot - d_tot), 2),
            }
        )

        for d_str, val in d_rets.items():
            daily_by_date.setdefault(d_str, []).append(val)
        for d_str, val in h_rets.items():
            hourly_by_date.setdefault(d_str, []).append(val)

    common_dates = sorted(set(daily_by_date.keys()).intersection(set(hourly_by_date.keys())))
    # 过滤掉初始阶段品种极少（例如小于10个）的极端日期
    valid_dates = [d for d in common_dates if len(daily_by_date[d]) >= 20]

    daily_execution = equal_weight_execution(results, valid_dates, "daily")["daily"].reindex(
        valid_dates, fill_value=0.0
    )
    hourly_execution = equal_weight_execution(results, valid_dates, "hourly")["daily"].reindex(
        valid_dates, fill_value=0.0
    )
    d_port_rets = daily_execution.net.to_numpy()
    h_port_rets = hourly_execution.net.to_numpy()

    d_nav = 1.0 + np.cumsum(np.array(d_port_rets))
    h_nav = 1.0 + np.cumsum(np.array(h_port_rets))

    d_peak = np.maximum.accumulate(np.maximum(d_nav, 1.0))
    d_dd = (d_nav - d_peak) / d_peak * 100.0

    h_peak = np.maximum.accumulate(np.maximum(h_nav, 1.0))
    h_dd = (h_nav - h_peak) / h_peak * 100.0

    d_tot = (d_nav[-1] - 1.0) * 100.0
    h_tot = (h_nav[-1] - 1.0) * 100.0

    d_std = np.std(d_port_rets)
    h_std = np.std(h_port_rets)

    d_sharpe = float(np.mean(d_port_rets) / d_std * np.sqrt(252)) if d_std > 0 else 0.0
    h_sharpe = float(np.mean(h_port_rets) / h_std * np.sqrt(252)) if h_std > 0 else 0.0

    d_max_dd = float(np.min(d_dd))
    h_max_dd = float(np.min(h_dd))

    d_calmar = abs(d_tot / d_max_dd) if d_max_dd != 0 else 0.0
    h_calmar = abs(h_tot / h_max_dd) if h_max_dd != 0 else 0.0

    d_win_rate = float(np.mean([1 if r > 0 else 0 for r in d_port_rets])) * 100.0
    h_win_rate = float(np.mean([1 if r > 0 else 0 for r in h_port_rets])) * 100.0

    print("================== 整体组合表现对比 ==================")
    print(f"回测交易日: {len(valid_dates)} 天 ({valid_dates[0]} ~ {valid_dates[-1]})")
    print(
        f"日线组合 (N=20): 累计收益 = {d_tot:+.2f}%, 最大回撤 = {d_max_dd:.2f}%, 年化夏普 = {d_sharpe:.2f}, 卡玛 = {d_calmar:.2f}, 日胜率 = {d_win_rate:.1f}%"
    )
    print(
        f"60m 组合 (N=80): 累计收益 = {h_tot:+.2f}%, 最大回撤 = {h_max_dd:.2f}%, 年化夏普 = {h_sharpe:.2f}, 卡玛 = {h_calmar:.2f}, 日胜率 = {h_win_rate:.1f}%"
    )

    out_data = {
        "accounting": "fixed_initial_capital_next_open",
        "daily_returns": d_port_rets.tolist(),
        "hourly_returns": h_port_rets.tolist(),
        "daily_fee_return_pct": float(daily_execution.fee.sum() * 100),
        "hourly_fee_return_pct": float(hourly_execution.fee.sum() * 100),
        "dates": valid_dates,
        "daily": {
            "nav": [round(float(x), 4) for x in d_nav],
            "dd": [round(float(x), 2) for x in d_dd],
            "total_return": round(float(d_tot), 2),
            "max_dd": round(float(d_max_dd), 2),
            "sharpe": round(float(d_sharpe), 2),
            "calmar": round(float(d_calmar), 2),
            "win_rate": round(float(d_win_rate), 1),
            "symbol_count": [len(daily_by_date[d]) for d in valid_dates],
        },
        "hourly": {
            "nav": [round(float(x), 4) for x in h_nav],
            "dd": [round(float(x), 2) for x in h_dd],
            "total_return": round(float(h_tot), 2),
            "max_dd": round(float(h_max_dd), 2),
            "sharpe": round(float(h_sharpe), 2),
            "calmar": round(float(h_calmar), 2),
            "win_rate": round(float(h_win_rate), 1),
            "symbol_count": [len(hourly_by_date[d]) for d in valid_dates],
        },
        "symbol_stats": symbol_stats,
    }

    out_path = research_output_path("overall_portfolio_nav.json", output_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)
    print(f"整体净值对比数据成功写入: {out_path}")
    return out_data


if __name__ == "__main__":
    main()
