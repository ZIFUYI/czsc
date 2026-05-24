"""双顺 15分钟一笔策略在 000852.XSHG 上的 Position/Event 回测。

周期搭配是 60分钟版本整体下沉一级：

- 60分钟：均线粘合发散，用作大方向过滤；
- 15分钟：双顺中线突破，用作 15分钟一笔的结构触发；
- 5分钟：MA60 支撑/压制，用作入场质量过滤；
- 1分钟：均线粘合发散 + N20 通道突破，用作执行确认。

Run:
    uv run --no-sync python examples/signals_dev/backtest_double_shun_15m_000852.py \\
        --warmup-sdt 20180101 --sdt 20200101 --edt 20260520
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for mod in [
    "clickhouse_connect",
    "clickhouse_connect.driver",
    "clickhouse_connect.driver.client",
    "clickhouse_connect.driver.httpclient",
    "clickhouse_connect.driver.compression",
]:
    sys.modules.setdefault(mod, MagicMock())

from czsc import CzscStrategyBase, Event, Freq, Position, RawBar, format_standard_kline  # noqa: E402
from examples.signals_dev.backtest_double_shun_60m_000852 import (  # noqa: E402
    FEE_RATE,
    LARGE_STOP_LOSS_BP,
    LARGE_TIMEOUT,
    SYMBOL,
    bars_to_frame,
    holds_to_weight_df,
    next_date,
    normalize_symbol,
    signal_clause,
)
from examples.signals_dev.test_zdy_macd_bc_000852 import read_jq_sdk_bars  # noqa: E402

DEFAULT_WARMUP_SDT = "20180101"
DEFAULT_SDT = "20200101"
DEFAULT_EDT = "20260520"
OUTPUT_DIR = ROOT / "examples" / "results" / "double_shun_15m_000852"
STRATEGY_NAME = "DoubleShun_15mStroke_60mDir_5mMA60_1mExec"

DIRECTION_60M = "60分钟_D1SMA5#10#20#40#60#120#250_均线粘合发散V260520"
MIDLINE_15M = "15分钟_D1N3M3_双顺中线突破V260521"
SUPPORT_5M = "5分钟_D1SMA#60T50_双顺均线支撑V260521"
EXEC_MA_1M = "1分钟_D1SMA5#10#20#40#60#120#250_均线粘合发散V260520"
BREAKOUT_1M = "1分钟_N20通道_突破信号V240623"


def frame_to_bars(data: pd.DataFrame) -> list[RawBar]:
    """Convert cached OHLCV rows to 1-minute RawBar objects."""
    return format_standard_kline(
        data[["dt", "symbol", "open", "close", "high", "low", "vol", "amount"]],
        freq=Freq.F1,
    )


def get_output_dir(symbol: str, sdt: str, edt: str) -> Path:
    """Return output directory for one symbol and one validation range."""
    if symbol == SYMBOL and sdt == DEFAULT_SDT and edt == DEFAULT_EDT:
        return OUTPUT_DIR
    name = f"double_shun_15m_{symbol.replace('.', '_')}_{sdt}_{edt}"
    return ROOT / "examples" / "results" / name


def load_1m_bars(source: str, symbol: str, warmup_sdt: str, edt: str, output_dir: Path) -> list[RawBar]:
    """Load 1-minute bars from local cache or JQData."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / f"{symbol.replace('.', '_')}_1m_ohlc_{warmup_sdt}_{edt}.csv"
    if source in {"auto", "cache"} and cache_file.exists():
        data = pd.read_csv(cache_file, parse_dates=["dt"])
        data["dt"] = pd.to_datetime(data["dt"]).dt.tz_localize(None)
        data = data.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)
        return frame_to_bars(data)

    if source == "cache":
        raise FileNotFoundError(f"Local 1m cache not found: {cache_file}; rerun with --source jq")

    bars = read_jq_sdk_bars(freq="1m", czsc_freq=Freq.F1, sdt=warmup_sdt, edt=next_date(edt), symbol=symbol)
    if not bars:
        raise RuntimeError(f"JQData returned empty 1m bars for {symbol}: {warmup_sdt} -> {edt}")

    stat_edt = pd.Timestamp(edt) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    bars = [bar for bar in bars if pd.Timestamp(warmup_sdt) <= pd.Timestamp(bar.dt).tz_localize(None) <= stat_edt]
    data = bars_to_frame(bars).drop_duplicates("dt").sort_values("dt").reset_index(drop=True)
    data.to_csv(cache_file, index=False, encoding="utf-8-sig")
    return frame_to_bars(data)


class DoubleShun15mStrategy(CzscStrategyBase):
    """双顺 15分钟一笔版本，使用项目原生 Position/Event 表达开平仓事件。"""

    @property
    def positions(self) -> list[Position]:
        long_pos = Position(
            name="DS15_Long_60mNotBear_15mMid_5mMA60_1mExec",
            symbol=self.symbol,
            opens=[
                Event.load(
                    {
                        "name": "15分钟一笔_开多",
                        "operate": "开多",
                        "signals_all": [
                            signal_clause(SUPPORT_5M, "看多"),
                            signal_clause(EXEC_MA_1M, "看多"),
                            signal_clause(BREAKOUT_1M, "做多"),
                        ],
                        "signals_any": [
                            signal_clause(MIDLINE_15M, "看多", v3="上破中线"),
                            signal_clause(MIDLINE_15M, "看多", v3="向上确认"),
                        ],
                        "signals_not": [signal_clause(DIRECTION_60M, "看空")],
                    }
                )
            ],
            exits=[
                Event.load(
                    {
                        "name": "结构或执行转弱_平多",
                        "operate": "平多",
                        "signals_any": [
                            signal_clause(MIDLINE_15M, "看空"),
                            signal_clause(SUPPORT_5M, "看空"),
                            signal_clause(SUPPORT_5M, "其他", v2="跌破"),
                            signal_clause(EXEC_MA_1M, "看空"),
                        ],
                    }
                )
            ],
            interval=0,
            timeout=LARGE_TIMEOUT,
            stop_loss=LARGE_STOP_LOSS_BP,
            t0=True,
        )
        short_pos = Position(
            name="DS15_Short_60mNotBull_15mMid_5mMA60_1mExec",
            symbol=self.symbol,
            opens=[
                Event.load(
                    {
                        "name": "15分钟一笔_开空",
                        "operate": "开空",
                        "signals_all": [
                            signal_clause(SUPPORT_5M, "看空"),
                            signal_clause(EXEC_MA_1M, "看空"),
                            signal_clause(BREAKOUT_1M, "做空"),
                        ],
                        "signals_any": [
                            signal_clause(MIDLINE_15M, "看空", v3="下破中线"),
                            signal_clause(MIDLINE_15M, "看空", v3="向下确认"),
                        ],
                        "signals_not": [signal_clause(DIRECTION_60M, "看多")],
                    }
                )
            ],
            exits=[
                Event.load(
                    {
                        "name": "结构或执行转强_平空",
                        "operate": "平空",
                        "signals_any": [
                            signal_clause(MIDLINE_15M, "看多"),
                            signal_clause(SUPPORT_5M, "看多"),
                            signal_clause(SUPPORT_5M, "其他", v2="上破"),
                            signal_clause(EXEC_MA_1M, "看多"),
                        ],
                    }
                )
            ],
            interval=0,
            timeout=LARGE_TIMEOUT,
            stop_loss=LARGE_STOP_LOSS_BP,
            t0=True,
        )
        return [long_pos, short_pos]


def evaluate_weight_df(dfw: pd.DataFrame, fee_rate: float = FEE_RATE) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate a 1-minute weight curve with daily aggregated returns."""
    data = dfw.sort_values("dt").copy()
    data["price_ret"] = data.groupby("symbol")["price"].pct_change().fillna(0.0)
    data["prev_weight"] = data.groupby("symbol")["weight"].shift(1).fillna(0.0)
    data["turnover"] = data.groupby("symbol")["weight"].diff().abs().fillna(data["weight"].abs())
    data["ret"] = data["prev_weight"] * data["price_ret"] - data["turnover"] * fee_rate

    daily = data.groupby(data["dt"].dt.normalize())["ret"].sum().sort_index().to_frame("ret")
    if daily.empty:
        raise RuntimeError("Daily return series is empty")

    daily["nav"] = (1 + daily["ret"]).cumprod()
    daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1

    years = (daily.index[-1] - daily.index[0]).days / 365.25
    final_nav = float(daily["nav"].iloc[-1])
    annual_return = final_nav ** (1 / years) - 1 if years > 0 and final_nav > 0 else float("nan")
    max_drawdown = float(daily["drawdown"].min())
    cumulative_return = final_nav - 1
    stats = {
        "symbol": str(data["symbol"].iloc[0]),
        "strategy": STRATEGY_NAME,
        "start": str(daily.index[0].date()),
        "end": str(daily.index[-1].date()),
        "annual_return": annual_return,
        "cumulative_return": cumulative_return,
        "final_nav": final_nav,
        "max_drawdown": max_drawdown,
        "calmar": annual_return / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
        "return_drawdown": cumulative_return / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
        "daily_win_rate": float((daily["ret"] > 0).mean()),
        "turnover": float(data["turnover"].sum()),
        "active_frac": float((data["weight"] != 0).mean()),
        "fee_rate": fee_rate,
        "direction_filter": DIRECTION_60M,
        "structure_signal": MIDLINE_15M,
        "support_signal": SUPPORT_5M,
        "execution_signals": f"{EXEC_MA_1M}; {BREAKOUT_1M}",
    }
    return stats, daily, data


def yearly_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate yearly return and max drawdown."""
    rows = []
    for year, data in daily.groupby(daily.index.year):
        nav = (1 + data["ret"]).cumprod()
        drawdown = nav / nav.cummax() - 1
        rows.append(
            {
                "year": int(year),
                "return": float(nav.iloc[-1] - 1),
                "max_drawdown": float(drawdown.min()),
            }
        )
    return pd.DataFrame(rows)


def save_outputs(
    output_dir: Path,
    tactic: DoubleShun15mStrategy,
    res,
    bars: list[RawBar],
    dfw: pd.DataFrame,
    holds: pd.DataFrame,
    stats: dict,
    daily: pd.DataFrame,
    data: pd.DataFrame,
) -> None:
    """Persist strategy artifacts and a compact markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tactic.save_positions(output_dir / "positions")
    bars_to_frame(bars).to_csv(output_dir / "bars_1m.csv", index=False, encoding="utf-8-sig")
    dfw.to_csv(output_dir / "weights.csv", index=False, encoding="utf-8-sig")
    holds.to_csv(output_dir / "holds.csv", index=False, encoding="utf-8-sig")
    res.pairs_df().to_csv(output_dir / "pairs.csv", index=False, encoding="utf-8-sig")
    res.signals_df().to_csv(output_dir / "signals.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output_dir / "daily.csv", index_label="dt", encoding="utf-8-sig")
    data.to_csv(output_dir / "bar_returns.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([stats]).to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    yearly_stats(daily).to_csv(output_dir / "yearly_stats.csv", index=False, encoding="utf-8-sig")

    pair_count = int(len(res.pairs_df()))
    report = f"""# 双顺 15分钟一笔策略

## 信号搭配

- 方向过滤：`{DIRECTION_60M}`，开多时排除 60分钟 `看空`，开空时排除 60分钟 `看多`。
- 结构触发：`{MIDLINE_15M}`，使用 15分钟 `上破中线/向上确认` 与 `下破中线/向下确认`。
- 入场过滤：`{SUPPORT_5M}`，开多要求 `看多`，开空要求 `看空`。
- 执行确认：`{EXEC_MA_1M}` + `{BREAKOUT_1M}`。
- 平仓事件：15分钟结构反向、5分钟 MA60 反向/跌破/上破、1分钟均线粘合发散反向。

## 结果

- 标的：`{stats["symbol"]}`
- 统计区间：`{stats["start"]}` 至 `{stats["end"]}`
- 年化收益：{stats["annual_return"]:.2%}
- 累计收益：{stats["cumulative_return"]:.2%}
- 最大回撤：{stats["max_drawdown"]:.2%}
- 期末净值：{stats["final_nav"]:.4f}
- Calmar：{stats["calmar"]:.2f}
- 换手次数：{stats["turnover"]:.0f}
- 持仓占比：{stats["active_frac"]:.2%}
- 交易对数量：{pair_count}
"""
    (output_dir / "strategy_report.md").write_text(report, encoding="utf-8")


def run_backtest(symbol: str, warmup_sdt: str, sdt: str, edt: str, source: str, output_dir: Path) -> None:
    """Run the native Position/Event backtest."""
    bars = load_1m_bars(source, symbol, warmup_sdt, edt, output_dir)
    if not bars:
        raise RuntimeError("No 1-minute bars loaded")

    first_dt = pd.Timestamp(bars[0].dt).tz_localize(None)
    last_dt = pd.Timestamp(bars[-1].dt).tz_localize(None)
    print(f"bars: {len(bars)} | {first_dt} -> {last_dt}")

    tactic = DoubleShun15mStrategy(symbol=symbol, include_sdt_bar=True)
    print("unique signals:")
    for signal in tactic.unique_signals:
        print(f"  - {signal}")

    res = tactic.backtest(bars, sdt=sdt, include_sdt_bar=True, emit_signals=True)
    dfw, holds = holds_to_weight_df(res.holds_df())
    stats, daily, data = evaluate_weight_df(dfw)
    save_outputs(output_dir, tactic, res, bars, dfw, holds, stats, daily, data)

    print("\nsummary:")
    print(pd.DataFrame([stats]).to_string(index=False))
    print("\nyearly stats:")
    print(yearly_stats(daily).to_string(index=False))
    print(f"\noutputs: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["auto", "cache", "jq"], default="auto")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--warmup-sdt", default=DEFAULT_WARMUP_SDT)
    parser.add_argument("--sdt", default=DEFAULT_SDT)
    parser.add_argument("--edt", default=DEFAULT_EDT)
    args = parser.parse_args()

    symbol = normalize_symbol(args.symbol)
    output_dir = get_output_dir(symbol, args.sdt, args.edt)
    run_backtest(symbol, args.warmup_sdt, args.sdt, args.edt, args.source, output_dir)


if __name__ == "__main__":
    main()
