"""用完整 Position/Event 精确验证双顺 60分钟一笔的方向过滤候选。

三个候选和 15分钟一笔版本一致，只是方向过滤周期换成日线：

- ``direction_none``：不使用日线方向过滤；
- ``base``：日线不反向过滤；
- ``direction_same``：日线必须同向。

Run:
    uv run --no-sync python examples/signals_dev/verify_double_shun_60m_position_event_000852.py
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

from czsc import CzscStrategyBase, Event, Position  # noqa: E402
from examples.signals_dev.backtest_double_shun_60m_000852 import (  # noqa: E402
    BREAKOUT_5M,
    DAILY_MA_JOINT,
    DEFAULT_EDT,
    EXEC_MA_5M,
    FEE_RATE,
    LARGE_STOP_LOSS_BP,
    LARGE_TIMEOUT,
    MIDLINE_60M,
    SUPPORT_15M,
    SYMBOL,
    holds_to_weight_df,
    load_5m_bars,
    normalize_symbol,
    signal_clause,
)

DEFAULT_WARMUP_SDT = "20180101"
DEFAULT_SDT = "20200101"
OUTPUT_DIR = ROOT / "examples" / "results" / "double_shun_60m_position_event_verify_000852"
BASE_CACHE_DIR = ROOT / "examples" / "results" / "double_shun_60m_000852_XSHG_20200101_20260520"


@dataclass(frozen=True)
class Variant:
    """One 60-minute stroke Position/Event candidate."""

    name: str
    direction_mode: str = "not_opposite"


CANDIDATES = [
    Variant("direction_none", direction_mode="none"),
    Variant("base", direction_mode="not_opposite"),
    Variant("direction_same", direction_mode="same"),
]


def direction_clauses(variant: Variant, side: str) -> tuple[list[str], list[str]]:
    """Return (signals_all, signals_not) for one side."""
    if variant.direction_mode == "none":
        return [], []
    same_v1 = "看多" if side == "long" else "看空"
    opposite_v1 = "看空" if side == "long" else "看多"
    if variant.direction_mode == "same":
        return [signal_clause(DAILY_MA_JOINT, same_v1)], []
    if variant.direction_mode == "not_opposite":
        return [], [signal_clause(DAILY_MA_JOINT, opposite_v1)]
    raise ValueError(f"Unsupported direction_mode: {variant.direction_mode}")


class DoubleShun60mCandidateStrategy(CzscStrategyBase):
    """Parametric 60-minute stroke strategy for exact verification."""

    def __init__(self, symbol: str, variant: Variant):
        super().__init__(symbol=symbol, name=f"DS60_PE_{variant.name}", include_sdt_bar=True)
        self.variant = variant

    @property
    def positions(self) -> list[Position]:
        long_direction_all, long_direction_not = direction_clauses(self.variant, "long")
        short_direction_all, short_direction_not = direction_clauses(self.variant, "short")
        long_pos = Position(
            name=f"DS60_Long_{self.variant.name}",
            symbol=self.symbol,
            opens=[
                Event.load(
                    {
                        "name": "60分钟一笔_开多",
                        "operate": "开多",
                        "signals_all": [
                            signal_clause(SUPPORT_15M, "看多"),
                            signal_clause(EXEC_MA_5M, "看多"),
                            signal_clause(BREAKOUT_5M, "做多"),
                            *long_direction_all,
                        ],
                        "signals_any": [
                            signal_clause(MIDLINE_60M, "看多", v3="上破中线"),
                            signal_clause(MIDLINE_60M, "看多", v3="向上确认"),
                        ],
                        "signals_not": long_direction_not,
                    }
                )
            ],
            exits=[
                Event.load(
                    {
                        "name": "结构或执行转弱_平多",
                        "operate": "平多",
                        "signals_any": [
                            signal_clause(MIDLINE_60M, "看空"),
                            signal_clause(SUPPORT_15M, "看空"),
                            signal_clause(SUPPORT_15M, "其他", v2="跌破"),
                            signal_clause(EXEC_MA_5M, "看空"),
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
            name=f"DS60_Short_{self.variant.name}",
            symbol=self.symbol,
            opens=[
                Event.load(
                    {
                        "name": "60分钟一笔_开空",
                        "operate": "开空",
                        "signals_all": [
                            signal_clause(SUPPORT_15M, "看空"),
                            signal_clause(EXEC_MA_5M, "看空"),
                            signal_clause(BREAKOUT_5M, "做空"),
                            *short_direction_all,
                        ],
                        "signals_any": [
                            signal_clause(MIDLINE_60M, "看空", v3="下破中线"),
                            signal_clause(MIDLINE_60M, "看空", v3="向下确认"),
                        ],
                        "signals_not": short_direction_not,
                    }
                )
            ],
            exits=[
                Event.load(
                    {
                        "name": "结构或执行转强_平空",
                        "operate": "平空",
                        "signals_any": [
                            signal_clause(MIDLINE_60M, "看多"),
                            signal_clause(SUPPORT_15M, "看多"),
                            signal_clause(SUPPORT_15M, "其他", v2="上破"),
                            signal_clause(EXEC_MA_5M, "看多"),
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


def evaluate_weight_df(dfw: pd.DataFrame, variant: Variant) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate one full Position/Event weight curve."""
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
    stats = {
        "variant": variant.name,
        "direction_mode": variant.direction_mode,
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
    }
    return stats, daily, data


def summarize_pairs(pairs: pd.DataFrame) -> dict:
    """Summarize native pairs output."""
    if pairs.empty:
        return {
            "pair_count": 0,
            "long_count": 0,
            "short_count": 0,
            "pair_win_rate": float("nan"),
            "avg_pnl_bp": float("nan"),
            "median_pnl_bp": float("nan"),
        }
    directions = pairs["交易方向"].value_counts()
    return {
        "pair_count": int(len(pairs)),
        "long_count": int(directions.get("多头", 0)),
        "short_count": int(directions.get("空头", 0)),
        "pair_win_rate": float((pairs["盈亏比例"] > 0).mean()),
        "avg_pnl_bp": float(pairs["盈亏比例"].mean()),
        "median_pnl_bp": float(pairs["盈亏比例"].median()),
    }


def run_variant(bars, symbol: str, sdt: str, variant: Variant, output_dir: Path) -> dict:
    """Run one exact Position/Event candidate and save artifacts."""
    variant_dir = output_dir / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    tactic = DoubleShun60mCandidateStrategy(symbol=symbol, variant=variant)
    print(f"running {variant.name}: {len(tactic.unique_signals)} signals", flush=True)

    res = tactic.backtest(bars, sdt=sdt, include_sdt_bar=True, emit_signals=False)
    dfw, holds = holds_to_weight_df(res.holds_df())
    stats, daily, data = evaluate_weight_df(dfw, variant)
    pairs = res.pairs_df()
    stats.update(summarize_pairs(pairs))

    tactic.save_positions(variant_dir / "positions")
    dfw.to_csv(variant_dir / "weights.csv", index=False, encoding="utf-8-sig")
    holds.to_csv(variant_dir / "holds.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(variant_dir / "pairs.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(variant_dir / "daily.csv", index_label="dt", encoding="utf-8-sig")
    data.to_csv(variant_dir / "bar_returns.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([stats]).to_csv(variant_dir / "summary.csv", index=False, encoding="utf-8-sig")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["auto", "cache", "jq"], default="cache")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--warmup-sdt", default=DEFAULT_WARMUP_SDT)
    parser.add_argument("--sdt", default=DEFAULT_SDT)
    parser.add_argument("--edt", default=DEFAULT_EDT)
    args = parser.parse_args()

    symbol = normalize_symbol(args.symbol)
    output_dir = (
        OUTPUT_DIR if symbol == SYMBOL else ROOT / "examples" / "results" / f"double_shun_60m_pe_verify_{symbol}"
    )
    cache_dir = BASE_CACHE_DIR if symbol == SYMBOL else output_dir
    bars = load_5m_bars(args.source, symbol, args.warmup_sdt, args.edt, cache_dir)
    print(f"bars: {len(bars)} | {bars[0].dt} -> {bars[-1].dt}", flush=True)

    rows = [run_variant(bars, symbol, args.sdt, variant, output_dir) for variant in CANDIDATES]
    table = pd.DataFrame(rows).sort_values(["calmar", "annual_return"], ascending=[False, False]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "position_event_verification.csv", index=False, encoding="utf-8-sig")

    display_cols = [
        "variant",
        "direction_mode",
        "annual_return",
        "max_drawdown",
        "calmar",
        "final_nav",
        "pair_count",
        "avg_pnl_bp",
        "turnover",
        "active_frac",
    ]
    print("\nPosition/Event verification:")
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
