# ruff: noqa: E402
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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


SECTOR_MAP = {
    "黑色系": ["rb", "hc", "i", "j", "jm", "ss", "SF", "SM", "wr"],
    "有色金属": ["cu", "al", "zn", "pb", "ni", "sn", "ao", "bc"],
    "能源化工": ["sc", "fu", "lu", "bu", "ru", "sp", "ta", "eg", "MA", "pp", "l", "v", "eb", "sa", "ur", "pg", "br"],
    "农产品": ["m", "y", "a", "b", "p", "c", "cs", "RM", "OI", "CF", "SR", "AP", "CJ", "PK", "JD", "LH"],
    "贵金属": ["au", "ag"],
    "新能源": ["lc", "si", "px"],
}


def get_sector(code: str) -> str:
    for sec, syms in SECTOR_MAP.items():
        if code in syms:
            return sec
    return "其他"


def backtest_task(f_path: str):
    f = Path(f_path)
    sym = f.stem.replace("_15m", "")
    code = sym.split(".")[-1]
    sector = get_sector(code)
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
                    daily_series[dt_val.strftime("%Y-%m-%d")] = float(ret_val)

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
                    hourly_series[d_obj.strftime("%Y-%m-%d")] = float(ret_val)

        return {
            "symbol": sym,
            "code": code,
            "sector": sector,
            "daily_rets": daily_series,
            "hourly_rets": hourly_series,
            "daily_inputs": daily_inputs,
            "hourly_inputs": hourly_inputs,
        }
    except Exception:
        return None


def build_data(*, output_path=None, results=None):
    data_dir = research_data_dir()
    files = sorted(data_dir.glob("*.parquet"))
    if results is None:
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = [r for r in executor.map(backtest_task, [str(f) for f in files]) if r is not None]

    print(f"有效处理品种: {len(results)}")

    daily_by_date = {}
    hourly_by_date = {}
    sector_results = {}

    for item in results:
        sym = item["symbol"]
        sec = item["sector"]
        d_rets = item["daily_rets"]
        h_rets = item["hourly_rets"]

        sector_results.setdefault(sec, {"daily": [], "hourly": []})
        d_tot = np.sum(list(d_rets.values())) * 100.0 if d_rets else 0.0
        h_tot = np.sum(list(h_rets.values())) * 100.0 if h_rets else 0.0
        sector_results[sec]["daily"].append(d_tot)
        sector_results[sec]["hourly"].append(h_tot)

        for d, v in d_rets.items():
            daily_by_date.setdefault(d, []).append(v)
        for d, v in h_rets.items():
            hourly_by_date.setdefault(d, []).append(v)

    # 1. 对齐共同交易日 (2026-01 ~ 2026-09)
    common_dates = sorted(set(daily_by_date.keys()).intersection(set(hourly_by_date.keys())))
    valid_common_dates = [d for d in common_dates if len(daily_by_date[d]) >= 20]

    daily_execution = equal_weight_execution(results, valid_common_dates, "daily")["daily"].reindex(
        valid_common_dates, fill_value=0.0
    )
    hourly_execution = equal_weight_execution(results, valid_common_dates, "hourly")["daily"].reindex(
        valid_common_dates, fill_value=0.0
    )
    d_rets = daily_execution.net.to_numpy()
    h_rets_aligned = hourly_execution.net.to_numpy()

    d_nav = 1.0 + np.cumsum(d_rets)
    h_nav_aligned = 1.0 + np.cumsum(h_rets_aligned)

    d_peak = np.maximum.accumulate(np.maximum(d_nav, 1.0))
    d_dd = (d_nav - d_peak) / d_peak * 100.0

    h_peak_aligned = np.maximum.accumulate(np.maximum(h_nav_aligned, 1.0))
    h_dd_aligned = (h_nav_aligned - h_peak_aligned) / h_peak_aligned * 100.0

    # 2. 60m 完整生命周期 (2024 ~ 2026，共 700+ 天)
    full_h_dates = sorted([d for d, v in hourly_by_date.items() if len(v) >= 20])
    full_hourly_execution = equal_weight_execution(results, full_h_dates, "hourly")["daily"].reindex(
        full_h_dates, fill_value=0.0
    )
    full_h_rets = full_hourly_execution.net.to_numpy()
    full_h_nav = 1.0 + np.cumsum(full_h_rets)
    full_h_peak = np.maximum.accumulate(np.maximum(full_h_nav, 1.0))
    full_h_dd = (full_h_nav - full_h_peak) / full_h_peak * 100.0

    # 3. 品种统计
    symbol_stats = []
    for item in results:
        sym = item["symbol"]
        code = item["code"]
        sec = item["sector"]
        d_sub = [item["daily_rets"].get(d, 0.0) for d in valid_common_dates]
        h_sub = [item["hourly_rets"].get(d, 0.0) for d in valid_common_dates]
        d_tot = np.sum(d_sub) * 100.0
        h_tot = np.sum(h_sub) * 100.0
        symbol_stats.append(
            {
                "symbol": sym,
                "code": code,
                "sector": sec,
                "daily_return": round(float(d_tot), 2),
                "hourly_return": round(float(h_tot), 2),
                "diff": round(float(h_tot - d_tot), 2),
            }
        )

    # 4. 板块汇总
    sector_summary = []
    for sec, data in sector_results.items():
        sector_summary.append(
            {
                "sector": sec,
                "count": len(data["daily"]),
                "daily_avg": round(float(np.mean(data["daily"])), 2),
                "hourly_avg": round(float(np.mean(data["hourly"])), 2),
                "diff": round(float(np.mean(data["hourly"]) - np.mean(data["daily"])), 2),
            }
        )
    sector_summary.sort(key=lambda x: x["hourly_avg"], reverse=True)

    out_payload = {
        "accounting": "fixed_initial_capital_next_open",
        "daily_fee_return_pct": float(daily_execution.fee.sum() * 100),
        "hourly_fee_return_pct": float(hourly_execution.fee.sum() * 100),
        "aligned": {
            "dates": valid_common_dates,
            "daily_nav": [round(float(x), 4) for x in d_nav],
            "daily_dd": [round(float(x), 2) for x in d_dd],
            "hourly_nav": [round(float(x), 4) for x in h_nav_aligned],
            "hourly_dd": [round(float(x), 2) for x in h_dd_aligned],
            "metrics": {
                "daily": {
                    "total_return": round(float((d_nav[-1] - 1.0) * 100), 2),
                    "max_dd": round(float(np.min(d_dd)), 2),
                    "sharpe": round(float(np.mean(d_rets) / np.std(d_rets) * np.sqrt(252)), 2)
                    if np.std(d_rets) > 0
                    else 0,
                    "calmar": round(abs(float((d_nav[-1] - 1.0) * 100 / np.min(d_dd))), 2) if np.min(d_dd) != 0 else 0,
                    "win_rate": round(float(np.mean(d_rets > 0) * 100), 1),
                },
                "hourly": {
                    "total_return": round(float((h_nav_aligned[-1] - 1.0) * 100), 2),
                    "max_dd": round(float(np.min(h_dd_aligned)), 2),
                    "sharpe": round(float(np.mean(h_rets_aligned) / np.std(h_rets_aligned) * np.sqrt(252)), 2)
                    if np.std(h_rets_aligned) > 0
                    else 0,
                    "calmar": round(abs(float((h_nav_aligned[-1] - 1.0) * 100 / np.min(h_dd_aligned))), 2)
                    if np.min(h_dd_aligned) != 0
                    else 0,
                    "win_rate": round(float(np.mean(h_rets_aligned > 0) * 100), 1),
                },
            },
        },
        "full_hourly": {
            "dates": full_h_dates,
            "nav": [round(float(x), 4) for x in full_h_nav],
            "dd": [round(float(x), 2) for x in full_h_dd],
            "metrics": {
                "total_return": round(float((full_h_nav[-1] - 1.0) * 100), 2),
                "max_dd": round(float(np.min(full_h_dd)), 2),
                "sharpe": round(float(np.mean(full_h_rets) / np.std(full_h_rets) * np.sqrt(252)), 2)
                if np.std(full_h_rets) > 0
                else 0,
                "calmar": round(abs(float((full_h_nav[-1] - 1.0) * 100 / np.min(full_h_dd))), 2)
                if np.min(full_h_dd) != 0
                else 0,
                "win_rate": round(float(np.mean(full_h_rets > 0) * 100), 1),
                "total_days": len(full_h_dates),
            },
        },
        "symbol_stats": symbol_stats,
        "sector_summary": sector_summary,
    }

    out_file = research_output_path("overall_portfolio_payload.json", output_path)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False)
    print(f"整体看板数据已保存至 {out_file}")
    return out_payload


if __name__ == "__main__":
    build_data()
