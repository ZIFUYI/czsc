"""Conditional multi-frequency resonance strategy on 000852.XSHG.

This is a condition-style multi-frequency strategy, not a signal scoring model.
It uses native CZSC signals through ``generate_czsc_signals`` and then applies a
state machine:

- Large-period filter: 60-minute SMA#5 trend.
- Small-period entry: 15-minute N64 absolute momentum reaches an extreme.
- Small-period exit: 15-minute N16 absolute momentum reaches the opposite
  extreme, or the 60-minute filter flips.

Rules:

- Open long: 60m SMA#5 bullish AND 15m N64 momentum is ``超强``.
- Exit long: 15m N16 momentum is ``超弱`` OR 60m SMA#5 turns bearish.
- Open short: 60m SMA#5 bearish AND 15m N64 momentum is ``超弱``.
- Exit short: 15m N16 momentum is ``超强`` OR 60m SMA#5 turns bullish.

Run:
    uv run --no-sync python examples/signals_dev/backtest_conditional_resonance_000852.py

Use JQData instead of the local 000852 cache:
    uv run --no-sync python examples/signals_dev/backtest_conditional_resonance_000852.py --source jq
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
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

from czsc import Freq, format_standard_kline, generate_czsc_signals  # noqa: E402
from examples.signals_dev.test_zdy_macd_bc_000852 import read_jq_sdk_bars  # noqa: E402

SYMBOL = "000852.XSHG"
DATA_SDT = "20180101"
STAT_SDT = "20200101"
STAT_EDT = "20260520"
FETCH_EDT = "20260521"
FEE_RATE = 0.0002
OUTPUT_DIR = ROOT / "examples" / "results" / "conditional_resonance_000852"
LOCAL_CACHE_FILE = ROOT / "examples" / "results" / "multilevel_resonance_000852" / "000852_XSHG_15m_ohlc.csv"

STRATEGY_NAME = "CR_60mFilter_15mN64ExtremeEntry_15mN16ExtremeExit"
LONG_FILTER_COL = "60分钟_D1SMA#5_分类V221101"
ENTRY_COL = "15分钟_D1N64T10_绝对动量V230227"
EXIT_COL = "15分钟_D1N16T10_绝对动量V230227"
SIGNALS_CONFIG = [
    {"name": "tas_ma_base_V221101", "freq": "60分钟", "di": 1, "timeperiod": 5, "ma_type": "SMA"},
    {"name": "bar_bpm_V230227", "freq": "15分钟", "di": 1, "n": 64, "th": 10},
    {"name": "bar_bpm_V230227", "freq": "15分钟", "di": 1, "n": 16, "th": 10},
]


def normalize_symbol(symbol: str) -> str:
    """Normalize common index codes to JoinQuant symbols."""
    symbol = symbol.strip().upper()
    if "." in symbol:
        return symbol
    known = {
        "000852": SYMBOL,
        "000905": "000905.XSHG",
        "159915": "159915.XSHE",
    }
    return known.get(symbol, f"{symbol}.XSHG")


def get_output_dir(symbol: str) -> Path:
    """Return output directory for one symbol."""
    if symbol == SYMBOL:
        return OUTPUT_DIR
    return ROOT / "examples" / "results" / f"conditional_resonance_{symbol.replace('.', '_')}"


def load_ohlc(source: str, symbol: str, output_dir: Path) -> pd.DataFrame:
    """Load 15-minute OHLCV data from local cache or JQData."""
    if source == "cache":
        cache_file = output_dir / f"{symbol.replace('.', '_')}_15m_ohlc.csv"
        source_file = cache_file if cache_file.exists() else LOCAL_CACHE_FILE if symbol == SYMBOL else cache_file
        if not source_file.exists():
            raise FileNotFoundError(f"Local 15m cache not found: {source_file}; rerun with --source jq")
        data = pd.read_csv(source_file, parse_dates=["dt"])
    else:
        bars = read_jq_sdk_bars(freq="15m", czsc_freq=Freq.F15, sdt=DATA_SDT, edt=FETCH_EDT, symbol=symbol)
        data = pd.DataFrame(
            [
                {
                    "dt": pd.Timestamp(bar.dt).tz_localize(None),
                    "symbol": bar.symbol,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "vol": float(bar.vol),
                    "amount": float(bar.amount),
                }
                for bar in bars
            ]
        )

    data["dt"] = pd.to_datetime(data["dt"]).dt.tz_localize(None)
    stat_edt = pd.Timestamp(STAT_EDT) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    data = data[(data["dt"] >= pd.Timestamp(DATA_SDT)) & (data["dt"] <= stat_edt)].copy()
    data = data.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / f"{symbol.replace('.', '_')}_15m_ohlc.csv", index=False, encoding="utf-8-sig")
    return data


def build_signal_frame(data: pd.DataFrame, use_cache: bool, output_dir: Path) -> pd.DataFrame:
    """Generate or load native CZSC signals."""
    signal_file = output_dir / "native_signals.csv"
    if use_cache and signal_file.exists():
        sig = pd.read_csv(signal_file, parse_dates=["dt"])
        sig["dt"] = pd.to_datetime(sig["dt"]).dt.tz_localize(None)
        return sig

    bars = format_standard_kline(data[["dt", "symbol", "open", "close", "high", "low", "vol", "amount"]], freq=Freq.F15)
    sig = generate_czsc_signals(bars, signals_config=SIGNALS_CONFIG, sdt=STAT_SDT, init_n=500, df=True)
    sig["dt"] = pd.to_datetime(sig["dt"]).dt.tz_localize(None)
    sig = sig.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)
    sig.to_csv(signal_file, index=False, encoding="utf-8-sig")
    return sig


def build_condition_weights(sig: pd.DataFrame) -> pd.DataFrame:
    """Build weights with large-period filters and small-period entry/exit conditions."""
    missing = [col for col in [LONG_FILTER_COL, ENTRY_COL, EXIT_COL] if col not in sig.columns]
    if missing:
        raise KeyError(f"Missing expected signal columns: {missing}")

    filter_value = sig[LONG_FILTER_COL].astype(str)
    entry_value = sig[ENTRY_COL].astype(str)
    exit_value = sig[EXIT_COL].astype(str)

    long_filter = filter_value.str.startswith("多头").to_numpy()
    short_filter = filter_value.str.startswith("空头").to_numpy()
    long_entry = entry_value.str.startswith("超强").to_numpy()
    short_entry = entry_value.str.startswith("超弱").to_numpy()
    long_exit = exit_value.str.startswith("超弱").to_numpy()
    short_exit = exit_value.str.startswith("超强").to_numpy()

    weight = np.zeros(len(sig), dtype=float)
    pos = 0.0
    for i in range(len(sig)):
        if (pos == 1.0 and (long_exit[i] or short_filter[i])) or (pos == -1.0 and (short_exit[i] or long_filter[i])):
            pos = 0.0

        if pos == 0.0:
            if long_filter[i] and long_entry[i]:
                pos = 1.0
            elif short_filter[i] and short_entry[i]:
                pos = -1.0
        weight[i] = pos

    out = sig[["dt", "symbol", "close", LONG_FILTER_COL, ENTRY_COL, EXIT_COL]].copy()
    out["weight"] = weight
    out["price"] = out["close"].astype(float)
    return out[["dt", "symbol", "weight", "price", LONG_FILTER_COL, ENTRY_COL, EXIT_COL]]


def evaluate_weight_df(dfw: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate a 15-minute position curve with daily aggregated returns."""
    data = dfw.sort_values("dt").copy()
    data["price_ret"] = data.groupby("symbol")["price"].pct_change().fillna(0.0)
    data["prev_weight"] = data.groupby("symbol")["weight"].shift(1).fillna(0.0)
    data["turnover"] = data.groupby("symbol")["weight"].diff().abs().fillna(data["weight"].abs())
    data["ret"] = data["prev_weight"] * data["price_ret"] - data["turnover"] * FEE_RATE

    daily = data.groupby(data["dt"].dt.normalize())["ret"].sum().sort_index().to_frame("ret")
    daily["nav"] = (1 + daily["ret"]).cumprod()
    daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1

    years = (daily.index[-1] - daily.index[0]).days / 365.25
    final_nav = float(daily["nav"].iloc[-1])
    annual_return = final_nav ** (1 / years) - 1 if years > 0 and final_nav > 0 else float("nan")
    max_drawdown = float(daily["drawdown"].min())
    stats = {
        "symbol": str(data["symbol"].iloc[0]),
        "strategy": STRATEGY_NAME,
        "start": str(daily.index[0].date()),
        "end": str(daily.index[-1].date()),
        "annual_return": annual_return,
        "cumulative_return": final_nav - 1,
        "final_nav": final_nav,
        "max_drawdown": max_drawdown,
        "calmar": annual_return / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
        "return_drawdown": (final_nav - 1) / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
        "daily_win_rate": float((daily["ret"] > 0).mean()),
        "turnover": float(data["turnover"].sum()),
        "active_frac": float((data["weight"] != 0).mean()),
        "fee_rate": FEE_RATE,
        "large_period_filter": LONG_FILTER_COL,
        "small_period_entry": ENTRY_COL,
        "small_period_exit": EXIT_COL,
    }
    return stats, daily, data


def period_stats(daily: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Calculate full and common subperiod stats."""
    rows = [
        {
            "period": "full",
            "sdt": stats["start"],
            "annual_return": stats["annual_return"],
            "final_nav": stats["final_nav"],
            "max_drawdown": stats["max_drawdown"],
            "calmar": stats["calmar"],
            "return_drawdown": stats["return_drawdown"],
        }
    ]
    for period, sdt in [("2020_2022", "2020-01-01"), ("2023_2024", "2023-01-01"), ("last_2y", "2024-05-20")]:
        data = daily[daily.index >= pd.Timestamp(sdt)].copy()
        if period == "2020_2022":
            data = data[data.index <= pd.Timestamp("2022-12-31")]
        elif period == "2023_2024":
            data = data[data.index <= pd.Timestamp("2024-12-31")]
        if len(data) < 2:
            continue
        nav = (1 + data["ret"]).cumprod()
        drawdown = nav / nav.cummax() - 1
        years = (data.index[-1] - data.index[0]).days / 365.25
        final_nav = float(nav.iloc[-1])
        annual = final_nav ** (1 / years) - 1 if years > 0 and final_nav > 0 else float("nan")
        max_drawdown = float(drawdown.min())
        rows.append(
            {
                "period": period,
                "sdt": str(data.index[0].date()),
                "annual_return": annual,
                "final_nav": final_nav,
                "max_drawdown": max_drawdown,
                "calmar": annual / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
                "return_drawdown": (final_nav - 1) / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def extract_pairs(data: pd.DataFrame) -> pd.DataFrame:
    """Extract continuous non-zero position segments from weight data."""
    rows = []
    seg_id = (data["weight"] != data["weight"].shift()).cumsum()
    for _, seg in data.groupby(seg_id, sort=True):
        weight = float(seg["weight"].iloc[0])
        if weight == 0:
            continue
        rows.append(
            {
                "symbol": seg["symbol"].iloc[0],
                "direction": "long" if weight > 0 else "short",
                "entry_dt": seg["dt"].iloc[0],
                "exit_dt": seg["dt"].iloc[-1],
                "entry_price": float(seg["price"].iloc[0]),
                "exit_price": float(seg["price"].iloc[-1]),
                "hold_bars": len(seg),
                "ret": float((1 + seg["ret"]).prod() - 1),
            }
        )
    return pd.DataFrame(rows)


def save_outputs(dfw: pd.DataFrame, stats: dict, daily: pd.DataFrame, data: pd.DataFrame, output_dir: Path) -> None:
    """Persist strategy artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dfw.to_csv(output_dir / "weights.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output_dir / "daily.csv", index_label="dt", encoding="utf-8-sig")
    extract_pairs(data).to_csv(output_dir / "pairs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([stats]).to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    period_stats(daily, stats).to_csv(output_dir / "period_stats.csv", index=False, encoding="utf-8-sig")
    report = f"""# 条件式多周期共振策略

## 规则

- 大周期过滤：`{LONG_FILTER_COL}`，多头只允许做多，空头只允许做空。
- 开多：60分钟过滤为 `多头` 且 `{ENTRY_COL}` 为 `超强`。
- 平多：`{EXIT_COL}` 为 `超弱`，或 60分钟过滤转为 `空头`。
- 开空：60分钟过滤为 `空头` 且 `{ENTRY_COL}` 为 `超弱`。
- 平空：`{EXIT_COL}` 为 `超强`，或 60分钟过滤转为 `多头`。

## 绩效

- 年化收益：{stats["annual_return"]:.2%}
- 最大回撤：{stats["max_drawdown"]:.2%}
- 期末净值：{stats["final_nav"]:.4f}
- Calmar：{stats["calmar"]:.2f}
"""
    (output_dir / "strategy_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["cache", "jq"], default="cache")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--use-signal-cache", action="store_true")
    args = parser.parse_args()

    symbol = normalize_symbol(args.symbol)
    output_dir = get_output_dir(symbol)
    ohlc = load_ohlc(args.source, symbol, output_dir)
    print(f"ohlc rows: {len(ohlc)} | {ohlc['dt'].iloc[0]} -> {ohlc['dt'].iloc[-1]}")

    sig = build_signal_frame(ohlc, use_cache=args.use_signal_cache, output_dir=output_dir)
    print(f"signal rows: {len(sig)} | {sig['dt'].iloc[0]} -> {sig['dt'].iloc[-1]}")
    dfw = build_condition_weights(sig)
    stats, daily, data = evaluate_weight_df(dfw)
    save_outputs(dfw, stats, daily, data, output_dir)
    print(pd.DataFrame([stats]).to_string(index=False))
    print("\nperiod stats:")
    print(period_stats(daily, stats).to_string(index=False))
    print(f"\noutputs: {output_dir}")


if __name__ == "__main__":
    main()
