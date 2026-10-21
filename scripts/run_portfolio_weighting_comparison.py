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
    allocated_execution_ledger,
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


def backtest_one(f_path: str):
    f = Path(f_path)
    sym = f.stem.replace("_15m", "")
    code = sym.split(".")[-1]
    sec = get_sector(code)
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
            "sector": sec,
            "daily_rets": daily_series,
            "hourly_rets": hourly_series,
            "daily_inputs": daily_inputs,
            "hourly_inputs": hourly_inputs,
        }
    except Exception:
        return None


def calc_stats(port_rets: np.ndarray):
    nav = 1.0 + np.cumsum(port_rets)
    peak = np.maximum.accumulate(np.maximum(nav, 1.0))
    dd = (nav - peak) / peak * 100.0
    tot = (nav[-1] - 1.0) * 100.0
    max_dd = float(np.min(dd))
    mean_ret = float(np.mean(port_rets))
    std_ret = float(np.std(port_rets))
    ann_ret = mean_ret * 252 * 100.0
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
    calmar = abs(tot / max_dd) if max_dd != 0 else 0.0
    win_rate = float(np.mean(port_rets > 0) * 100.0)
    return {
        "nav": [round(float(x), 4) for x in nav],
        "dd": [round(float(x), 2) for x in dd],
        "total_return": round(float(tot), 2),
        "ann_return": round(float(ann_ret), 2),
        "max_dd": round(float(max_dd), 2),
        "sharpe": round(float(sharpe), 2),
        "calmar": round(float(calmar), 2),
        "win_rate": round(float(win_rate), 1),
    }


def main(*, output_path=None, results=None):
    data_dir = research_data_dir()
    files = sorted(data_dir.glob("*.parquet"))
    print(f"并行计算全品种收益矩阵，文件数: {len(files)}...")

    if results is None:
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = [r for r in executor.map(backtest_one, [str(f) for f in files]) if r is not None]

    print(f"成功回测品种数: {len(results)}")

    # 提取有效交易日：使用所有品种日期的频次统计，筛选活跃交易日（有>=20个品种有数据）
    date_counts = {}
    for r in results:
        for d in r["daily_rets"]:
            date_counts[d] = date_counts.get(d, 0) + 1

    common_dates = sorted([d for d, cnt in date_counts.items() if cnt >= 20])
    print(f"有效对齐交易日天数: {len(common_dates)} ({common_dates[0]} ~ {common_dates[-1]})")

    # 构建 DataFrame: 列为各品种代码
    df_daily = pd.DataFrame(index=common_dates)
    df_hourly = pd.DataFrame(index=common_dates)
    symbol_sector = {}

    for r in results:
        sym = r["symbol"]
        symbol_sector[sym] = r["sector"]
        df_daily[sym] = [r["daily_rets"].get(d, 0.0) for d in common_dates]
        df_hourly[sym] = [r["hourly_rets"].get(d, 0.0) for d in common_dates]

    n_symbols = df_daily.shape[1]
    print(f"对齐矩阵维度: {df_daily.shape[0]} 天 x {n_symbols} 品种")

    # 1. 方案一：全市场等权组合 (Equal Weight 1/N)
    rets_ew = df_daily.mean(axis=1).values
    stats_ew = calc_stats(rets_ew)

    # 2. 方案二：板块平价组合 (Sector Parity)
    # 板块等权：各板块等权分配 1/len(sectors)，板块内部各品种平分
    sectors = sorted(set(symbol_sector.values()))
    sector_weights = pd.Series(0.0, index=df_daily.columns)
    sec_weight_each = 1.0 / len(sectors)
    for sec in sectors:
        syms_in_sec = [s for s in df_daily.columns if symbol_sector[s] == sec]
        if syms_in_sec:
            w_sym = sec_weight_each / len(syms_in_sec)
            for s in syms_in_sec:
                sector_weights[s] = w_sym
    sector_weights = sector_weights / sector_weights.sum()
    rets_sector = df_daily.dot(sector_weights).values
    stats_sector = calc_stats(rets_sector)

    # 3. 方案三：波动率倒数风险平价组合 (Risk Parity on Daily)
    # 动态滚动 20 日历史波动率
    rolling_vol = df_daily.rolling(window=20, min_periods=5).std().fillna(0.01)
    # 避免除以 0
    inv_vol = 1.0 / np.maximum(rolling_vol, 0.0001)
    rp_weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    # 滞后 1 天避免前瞻偏差
    rp_weights_lagged = rp_weights.shift(1).fillna(1.0 / n_symbols)
    rets_rp = (df_daily * rp_weights_lagged).sum(axis=1).values
    stats_rp = calc_stats(rets_rp)

    # 4. 方案四：异构周期 + 风险平价组合 (Optimal Hybrid Multi-Freq + Risk Parity)
    # 能化、贵金属、有色采用 60m 策略；黑色系、农产品、新能源采用日线策略
    df_hybrid = pd.DataFrame(index=common_dates)
    for s in df_daily.columns:
        sec = symbol_sector[s]
        if sec in ["能源化工", "贵金属", "有色金属"]:
            df_hybrid[s] = df_hourly[s]
        else:
            df_hybrid[s] = df_daily[s]

    hybrid_rolling_vol = df_hybrid.rolling(window=20, min_periods=5).std().fillna(0.01)
    hybrid_inv_vol = 1.0 / np.maximum(hybrid_rolling_vol, 0.0001)
    hybrid_weights = hybrid_inv_vol.div(hybrid_inv_vol.sum(axis=1), axis=0)
    hybrid_weights_lagged = hybrid_weights.shift(1).fillna(1.0 / n_symbols)
    rets_hybrid_rp = (df_hybrid * hybrid_weights_lagged).sum(axis=1).values
    stats_hybrid_rp = calc_stats(rets_hybrid_rp)

    allocations = {
        "equal_weight": pd.DataFrame(1.0 / n_symbols, index=common_dates, columns=df_daily.columns),
        "sector_parity": pd.DataFrame([sector_weights] * len(common_dates), index=common_dates),
        "risk_parity": rp_weights_lagged,
        "hybrid_risk_parity": hybrid_weights_lagged,
    }
    ledgers, actual_stats = {}, {}
    for name, budget in allocations.items():
        frequencies = {
            s: "hourly"
            if name == "hybrid_risk_parity" and symbol_sector[s] in ["能源化工", "贵金属", "有色金属"]
            else "daily"
            for s in df_daily
        }
        ledgers[name] = allocated_execution_ledger(results, budget, frequencies)
        daily = ledgers[name]["daily"].reindex(common_dates, fill_value=0.0)
        actual_stats[name] = {
            **calc_stats(daily.net.to_numpy()),
            "daily_returns": daily.net.tolist(),
            "gross_return_pct": float(daily.gross.sum() * 100),
            "fee_return_pct": float(daily.fee.sum() * 100),
            "turnover": float(daily.turnover.sum()),
        }
    stats_ew, stats_sector, stats_rp, stats_hybrid_rp = [actual_stats[name] for name in allocations]

    # 打印对比结果
    print("====================== 组合加权方案对比 ======================")
    print(
        f"1. 基础 1/N 算术等权: 累计收益 = {stats_ew['total_return']:+.2f}%, 最大回撤 = {stats_ew['max_dd']:.2f}%, 夏普 = {stats_ew['sharpe']:.2f}, 卡玛 = {stats_ew['calmar']:.2f}"
    )
    print(
        f"2. 板块平价加权 (6大板块均衡): 累计收益 = {stats_sector['total_return']:+.2f}%, 最大回撤 = {stats_sector['max_dd']:.2f}%, 夏普 = {stats_sector['sharpe']:.2f}, 卡玛 = {stats_sector['calmar']:.2f}"
    )
    print(
        f"3. 波动率倒数风险平价 (Risk Parity): 累计收益 = {stats_rp['total_return']:+.2f}%, 最大回撤 = {stats_rp['max_dd']:.2f}%, 夏普 = {stats_rp['sharpe']:.2f}, 卡玛 = {stats_rp['calmar']:.2f}"
    )
    print(
        f"4. 异构周期 + 风险平价 (Hybrid Risk Parity): 累计收益 = {stats_hybrid_rp['total_return']:+.2f}%, 最大回撤 = {stats_hybrid_rp['max_dd']:.2f}%, 夏普 = {stats_hybrid_rp['sharpe']:.2f}, 卡玛 = {stats_hybrid_rp['calmar']:.2f}"
    )

    out_data = {
        "accounting": "fixed_initial_capital_next_open",
        "trading_day_clock": "futures_session",
        "dates": common_dates,
        "strategies": {
            "equal_weight": {
                "name": "1/N 算术等权基准",
                "stats": stats_ew,
            },
            "sector_parity": {
                "name": "板块平价加权 (Sector Budgeting)",
                "stats": stats_sector,
            },
            "risk_parity": {
                "name": "波动率倒数风险平价 (Risk Parity)",
                "stats": stats_rp,
            },
            "hybrid_risk_parity": {
                "name": "异构周期 + 风险平价 (多周期优选)",
                "stats": stats_hybrid_rp,
            },
        },
        "sectors": sectors,
    }

    import hashlib
    import pickle

    out_file = research_output_path("portfolio_weighting_comparison.json", output_path)
    execution_file = out_file.with_suffix(".execution.pkl")
    execution_bytes = pickle.dumps(ledgers, protocol=pickle.HIGHEST_PROTOCOL)
    execution_file.write_bytes(execution_bytes)
    out_data["execution_ledger"] = {
        "path": str(execution_file.resolve()),
        "sha256": hashlib.sha256(execution_bytes).hexdigest(),
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)
    print(f"加权对比结果已保存至: {out_file}")
    return out_data


if __name__ == "__main__":
    main()
