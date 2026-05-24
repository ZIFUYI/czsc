"""双顺 15分钟一笔策略参数对比表。

该脚本用于参数初筛：先一次性生成所有候选参数需要的信号列，再用同一套
开平仓规则做轻量状态机对比。手续费固定万2，不做更高手续费压力测试。

Run:
    uv run --no-sync python examples/signals_dev/compare_double_shun_15m_params_000852.py
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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

from czsc import generate_czsc_signals, get_signals_config  # noqa: E402
from examples.signals_dev.backtest_double_shun_15m_000852 import (  # noqa: E402
    DEFAULT_EDT,
    DEFAULT_SDT,
    DEFAULT_WARMUP_SDT,
    DIRECTION_60M,
    EXEC_MA_1M,
    FEE_RATE,
    OUTPUT_DIR as BASE_OUTPUT_DIR,
    SYMBOL,
    load_1m_bars,
    normalize_symbol,
    signal_clause,
)

OUTPUT_DIR = ROOT / "examples" / "results" / "double_shun_15m_params_000852"


@dataclass(frozen=True)
class Variant:
    """One 15-minute stroke strategy parameter variant."""

    name: str
    direction_mode: str = "not_opposite"
    mid_n: int = 3
    mid_m: int = 3
    support_th: int = 50
    breakout_n: int = 20

    @property
    def midline(self) -> str:
        return f"15分钟_D1N{self.mid_n}M{self.mid_m}_双顺中线突破V260521"

    @property
    def support(self) -> str:
        return f"5分钟_D1SMA#60T{self.support_th}_双顺均线支撑V260521"

    @property
    def breakout(self) -> str:
        return f"1分钟_N{self.breakout_n}通道_突破信号V240623"


VARIANTS = [
    Variant("base"),
    Variant("direction_same", direction_mode="same"),
    Variant("direction_none", direction_mode="none"),
    Variant("mid_n5", mid_n=5),
    Variant("support_t30", support_th=30),
    Variant("support_t80", support_th=80),
    Variant("breakout_n10", breakout_n=10),
    Variant("breakout_n30", breakout_n=30),
]


def direction_clauses(variant: Variant, side: str) -> tuple[list[str], list[str]]:
    """Return (signals_all, signals_not) for one side."""
    if variant.direction_mode == "none":
        return [], []
    same_v1 = "看多" if side == "long" else "看空"
    opposite_v1 = "看空" if side == "long" else "看多"
    if variant.direction_mode == "same":
        return [signal_clause(DIRECTION_60M, same_v1)], []
    if variant.direction_mode == "not_opposite":
        return [], [signal_clause(DIRECTION_60M, opposite_v1)]
    raise ValueError(f"Unsupported direction_mode: {variant.direction_mode}")


def variant_signal_clauses(variant: Variant) -> list[str]:
    """Collect all signal clauses used by one variant."""
    long_direction_all, long_direction_not = direction_clauses(variant, "long")
    short_direction_all, short_direction_not = direction_clauses(variant, "short")
    return [
        signal_clause(variant.support, "看多"),
        signal_clause(EXEC_MA_1M, "看多"),
        signal_clause(variant.breakout, "做多"),
        signal_clause(variant.midline, "看多", v3="上破中线"),
        signal_clause(variant.midline, "看多", v3="向上确认"),
        *long_direction_all,
        *long_direction_not,
        signal_clause(variant.midline, "看空"),
        signal_clause(variant.support, "看空"),
        signal_clause(variant.support, "其他", v2="跌破"),
        signal_clause(EXEC_MA_1M, "看空"),
        signal_clause(variant.support, "看空"),
        signal_clause(variant.breakout, "做空"),
        signal_clause(variant.midline, "看空", v3="下破中线"),
        signal_clause(variant.midline, "看空", v3="向下确认"),
        *short_direction_all,
        *short_direction_not,
        signal_clause(variant.midline, "看多"),
        signal_clause(variant.support, "看多"),
        signal_clause(variant.support, "其他", v2="上破"),
    ]


def signal_columns() -> list[str]:
    """Return unique signal clauses required by all variants."""
    clauses = []
    for variant in VARIANTS:
        clauses.extend(variant_signal_clauses(variant))
    return sorted(set(clauses))


def split_value(value: str) -> tuple[str, str, str, int]:
    """Split a CZSC signal value into v1/v2/v3/score."""
    parts = str(value).split("_")
    if len(parts) != 4:
        return "其他", "任意", "任意", 0
    try:
        score = int(parts[3])
    except ValueError:
        score = 0
    return parts[0], parts[1], parts[2], score


def match_value(value: str, v1: str, v2: str = "任意", v3: str = "任意", score: int = 0) -> bool:
    """Match a signal value with wildcard semantics."""
    rv1, rv2, rv3, rscore = split_value(value)
    return (
        rscore >= score and (v1 == "任意" or rv1 == v1) and (v2 == "任意" or rv2 == v2) and (v3 == "任意" or rv3 == v3)
    )


def match_col(sig: pd.DataFrame, col: str, v1: str, v2: str = "任意", v3: str = "任意") -> pd.Series:
    """Vectorized signal value matching for one column."""
    if col not in sig.columns:
        raise KeyError(f"Missing signal column: {col}")
    return sig[col].map(lambda x: match_value(x, v1, v2, v3))


def generate_signal_frame(bars, sdt: str, output_dir: Path, use_cache: bool) -> pd.DataFrame:
    """Generate or load the union signal frame for all variants."""
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_file = output_dir / "union_signals.csv"
    if use_cache and signal_file.exists():
        sig = pd.read_csv(signal_file, parse_dates=["dt"])
        sig["dt"] = pd.to_datetime(sig["dt"]).dt.tz_localize(None)
        return sig

    clauses = signal_columns()
    configs = get_signals_config(clauses)
    sig = generate_czsc_signals(bars, signals_config=configs, sdt=sdt, init_n=1000, df=True)
    sig["dt"] = pd.to_datetime(sig["dt"]).dt.tz_localize(None)
    sig = sig.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)
    sig.to_csv(signal_file, index=False, encoding="utf-8-sig")
    return sig


def simulate_variant(sig: pd.DataFrame, variant: Variant) -> tuple[pd.DataFrame, dict]:
    """Simulate one variant's position curve from generated signal columns."""
    long_direction_all, long_direction_not = direction_clauses(variant, "long")
    short_direction_all, short_direction_not = direction_clauses(variant, "short")

    long_open = (
        match_col(sig, variant.support, "看多")
        & match_col(sig, EXEC_MA_1M, "看多")
        & match_col(sig, variant.breakout, "做多")
        & (
            match_col(sig, variant.midline, "看多", v3="上破中线")
            | match_col(sig, variant.midline, "看多", v3="向上确认")
        )
    )
    for clause in long_direction_all:
        col, v1, v2, v3, _ = clause.rsplit("_", 4)
        long_open &= match_col(sig, col, v1, v2, v3)
    for clause in long_direction_not:
        col, v1, v2, v3, _ = clause.rsplit("_", 4)
        long_open &= ~match_col(sig, col, v1, v2, v3)

    short_open = (
        match_col(sig, variant.support, "看空")
        & match_col(sig, EXEC_MA_1M, "看空")
        & match_col(sig, variant.breakout, "做空")
        & (
            match_col(sig, variant.midline, "看空", v3="下破中线")
            | match_col(sig, variant.midline, "看空", v3="向下确认")
        )
    )
    for clause in short_direction_all:
        col, v1, v2, v3, _ = clause.rsplit("_", 4)
        short_open &= match_col(sig, col, v1, v2, v3)
    for clause in short_direction_not:
        col, v1, v2, v3, _ = clause.rsplit("_", 4)
        short_open &= ~match_col(sig, col, v1, v2, v3)

    long_exit = (
        match_col(sig, variant.midline, "看空")
        | match_col(sig, variant.support, "看空")
        | match_col(sig, variant.support, "其他", v2="跌破")
        | match_col(sig, EXEC_MA_1M, "看空")
    )
    short_exit = (
        match_col(sig, variant.midline, "看多")
        | match_col(sig, variant.support, "看多")
        | match_col(sig, variant.support, "其他", v2="上破")
        | match_col(sig, EXEC_MA_1M, "看多")
    )

    weights = []
    trades = []
    pos = 0
    entry_dt = None
    entry_price = None
    entry_side = None
    both_open_count = int((long_open & short_open).sum())
    for i, row in sig.iterrows():
        dt = row["dt"]
        price = float(row["close"])
        if pos == 1 and bool(long_exit.iloc[i]):
            trades.append({"side": "long", "entry_dt": entry_dt, "exit_dt": dt, "ret": price / entry_price - 1})
            pos = 0
            entry_dt = None
            entry_price = None
            entry_side = None
        elif pos == -1 and bool(short_exit.iloc[i]):
            trades.append({"side": "short", "entry_dt": entry_dt, "exit_dt": dt, "ret": entry_price / price - 1})
            pos = 0
            entry_dt = None
            entry_price = None
            entry_side = None

        if pos == 0:
            lo = bool(long_open.iloc[i])
            so = bool(short_open.iloc[i])
            if lo and not so:
                pos = 1
                entry_dt = dt
                entry_price = price
                entry_side = "long"
            elif so and not lo:
                pos = -1
                entry_dt = dt
                entry_price = price
                entry_side = "short"
        weights.append(pos)

    if pos != 0 and entry_price is not None:
        last = sig.iloc[-1]
        last_price = float(last["close"])
        ret = last_price / entry_price - 1 if entry_side == "long" else entry_price / last_price - 1
        trades.append({"side": entry_side, "entry_dt": entry_dt, "exit_dt": last["dt"], "ret": ret})

    dfw = sig[["dt", "symbol", "close"]].rename(columns={"close": "price"}).copy()
    dfw["price"] = dfw["price"].astype(float)
    dfw["weight"] = weights
    dfw = dfw[["dt", "symbol", "weight", "price"]]
    meta = {"both_open_count": both_open_count, "trades": trades}
    return dfw, meta


def evaluate_weight_df(
    dfw: pd.DataFrame, variant: Variant, trades: list[dict], both_open_count: int
) -> tuple[dict, pd.DataFrame]:
    """Evaluate one variant's weight curve."""
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
    cumulative_return = final_nav - 1
    pair_count = len(trades)
    long_count = sum(1 for x in trades if x["side"] == "long")
    short_count = sum(1 for x in trades if x["side"] == "short")
    rets_bp = pd.Series([x["ret"] * 10000 for x in trades], dtype="float64")
    stats = {
        "variant": variant.name,
        "direction_mode": variant.direction_mode,
        "mid_n": variant.mid_n,
        "mid_m": variant.mid_m,
        "support_th": variant.support_th,
        "breakout_n": variant.breakout_n,
        "annual_return": annual_return,
        "cumulative_return": cumulative_return,
        "final_nav": final_nav,
        "max_drawdown": max_drawdown,
        "calmar": annual_return / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
        "return_drawdown": cumulative_return / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
        "daily_win_rate": float((daily["ret"] > 0).mean()),
        "turnover": float(data["turnover"].sum()),
        "active_frac": float((data["weight"] != 0).mean()),
        "pair_count": int(pair_count),
        "long_count": int(long_count),
        "short_count": int(short_count),
        "pair_win_rate": float((rets_bp > 0).mean()) if pair_count else float("nan"),
        "avg_pnl_bp": float(rets_bp.mean()) if pair_count else float("nan"),
        "median_pnl_bp": float(rets_bp.median()) if pair_count else float("nan"),
        "both_open_count": both_open_count,
    }
    return stats, daily


def yearly_returns(daily: pd.DataFrame, variant: str) -> list[dict]:
    """Return yearly rows for one variant."""
    rows = []
    for year, data in daily.groupby(daily.index.year):
        nav = (1 + data["ret"]).cumprod()
        drawdown = nav / nav.cummax() - 1
        rows.append(
            {
                "variant": variant,
                "year": int(year),
                "return": float(nav.iloc[-1] - 1),
                "max_drawdown": float(drawdown.min()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["auto", "cache", "jq"], default="cache")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--warmup-sdt", default=DEFAULT_WARMUP_SDT)
    parser.add_argument("--sdt", default=DEFAULT_SDT)
    parser.add_argument("--edt", default=DEFAULT_EDT)
    parser.add_argument("--refresh-signals", action="store_true")
    args = parser.parse_args()

    symbol = normalize_symbol(args.symbol)
    output_dir = OUTPUT_DIR if symbol == SYMBOL else ROOT / "examples" / "results" / f"double_shun_15m_params_{symbol}"
    bars = load_1m_bars(args.source, symbol, args.warmup_sdt, args.edt, BASE_OUTPUT_DIR)
    print(f"bars: {len(bars)} | {bars[0].dt} -> {bars[-1].dt}")

    sig = generate_signal_frame(bars, args.sdt, output_dir, use_cache=not args.refresh_signals)
    print(f"signals: {len(sig)} | {sig['dt'].iloc[0]} -> {sig['dt'].iloc[-1]}")

    rows = []
    yearly = []
    for variant in VARIANTS:
        print(f"running {variant.name}", flush=True)
        dfw, meta = simulate_variant(sig, variant)
        stats, daily = evaluate_weight_df(dfw, variant, meta["trades"], meta["both_open_count"])
        rows.append(stats)
        yearly.extend(yearly_returns(daily, variant.name))

    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values(["calmar", "annual_return"], ascending=[False, False]).reset_index(drop=True)
    yearly_table = pd.DataFrame(yearly)
    table.to_csv(output_dir / "parameter_comparison.csv", index=False, encoding="utf-8-sig")
    yearly_table.to_csv(output_dir / "yearly_comparison.csv", index=False, encoding="utf-8-sig")

    display_cols = [
        "variant",
        "direction_mode",
        "mid_n",
        "support_th",
        "breakout_n",
        "annual_return",
        "max_drawdown",
        "calmar",
        "final_nav",
        "pair_count",
        "avg_pnl_bp",
        "turnover",
        "active_frac",
        "both_open_count",
    ]
    print("\nparameter comparison:")
    print(
        table[display_cols].to_string(
            index=False,
            formatters={
                "annual_return": "{:.2%}".format,
                "max_drawdown": "{:.2%}".format,
                "calmar": "{:.2f}".format,
                "final_nav": "{:.4f}".format,
                "avg_pnl_bp": "{:.2f}".format,
                "active_frac": "{:.2%}".format,
            },
        )
    )
    print(f"\noutputs: {output_dir}")


if __name__ == "__main__":
    main()
