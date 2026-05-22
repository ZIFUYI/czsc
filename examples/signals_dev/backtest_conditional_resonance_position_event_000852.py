"""Position/Event implementation of the conditional resonance strategy.

This script validates that the 40.89% / -9.93% condition-style strategy can be
implemented with the project-native Signal -> Event -> Position -> backtest
path.

The key point is to split long and short logic into two Positions. A single
long/short Position can let a direction-irrelevant exit event match first and
block the opposite-side exit. Split Positions avoid that event priority issue;
the portfolio weight is the sum of the long and short holds.

Run:
    uv run --no-sync python examples/signals_dev/backtest_conditional_resonance_position_event_000852.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from czsc import CzscStrategyBase, Event, Freq, Position, format_standard_kline  # noqa: E402
from examples.signals_dev.backtest_conditional_resonance_000852 import (  # noqa: E402
    ENTRY_COL,
    EXIT_COL,
    LONG_FILTER_COL,
    STAT_SDT,
    SYMBOL,
    build_condition_weights,
    build_signal_frame,
    evaluate_weight_df,
    get_output_dir,
    load_ohlc,
    normalize_symbol,
    period_stats,
)

POSITION_EVENT_STRATEGY_NAME = "CR_PositionEvent_SplitLongShort_60m15m"
POSITION_EVENT_OUTPUT_DIR = ROOT / "examples" / "results" / "conditional_resonance_position_event_000852"
LARGE_TIMEOUT = 1_000_000
LARGE_STOP_LOSS_BP = 1_000_000.0


def signal_clause(signal_col: str, v1: str) -> str:
    """Build a CZSC event signal clause from a signal column and v1 value."""
    return f"{signal_col}_{v1}_任意_任意_0"


class ConditionalResonancePositionEventStrategy(CzscStrategyBase):
    """Project-native Position/Event version of the conditional resonance strategy."""

    @property
    def positions(self) -> list[Position]:
        long_pos = Position(
            name="CR_Long_60mFilter_15mN64Entry_15mN16Exit",
            symbol=self.symbol,
            opens=[
                Event.load(
                    {
                        "name": "60m多头_N64超强_开多",
                        "operate": "开多",
                        "signals_all": [
                            signal_clause(LONG_FILTER_COL, "多头"),
                            signal_clause(ENTRY_COL, "超强"),
                        ],
                    }
                )
            ],
            exits=[
                Event.load(
                    {
                        "name": "N16超弱或60m空头_平多",
                        "operate": "平多",
                        "signals_any": [
                            signal_clause(EXIT_COL, "超弱"),
                            signal_clause(LONG_FILTER_COL, "空头"),
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
            name="CR_Short_60mFilter_15mN64Entry_15mN16Exit",
            symbol=self.symbol,
            opens=[
                Event.load(
                    {
                        "name": "60m空头_N64超弱_开空",
                        "operate": "开空",
                        "signals_all": [
                            signal_clause(LONG_FILTER_COL, "空头"),
                            signal_clause(ENTRY_COL, "超弱"),
                        ],
                    }
                )
            ],
            exits=[
                Event.load(
                    {
                        "name": "N16超强或60m多头_平空",
                        "operate": "平空",
                        "signals_any": [
                            signal_clause(EXIT_COL, "超强"),
                            signal_clause(LONG_FILTER_COL, "多头"),
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


def get_position_event_output_dir(symbol: str) -> Path:
    """Return the output directory for one symbol."""
    if symbol == SYMBOL:
        return POSITION_EVENT_OUTPUT_DIR
    return ROOT / "examples" / "results" / f"conditional_resonance_position_event_{symbol.replace('.', '_')}"


def holds_to_weight_df(holds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine split long/short Position holds into one portfolio weight series."""
    data = holds[["dt", "symbol", "pos", "price", "pos_name"]].copy()
    data["dt"] = pd.to_datetime(data["dt"]).dt.tz_localize(None)
    data = data.sort_values(["dt", "pos_name"]).reset_index(drop=True)

    wide = (
        data.pivot_table(index=["dt", "symbol"], columns="pos_name", values="pos", aggfunc="last")
        .fillna(0.0)
        .reset_index()
    )
    pos_cols = [col for col in wide.columns if col not in {"dt", "symbol"}]
    wide["weight"] = wide[pos_cols].sum(axis=1)

    prices = data.groupby(["dt", "symbol"], as_index=False)["price"].first()
    weights = wide.merge(prices, on=["dt", "symbol"], how="left")
    overlap = weights[weights["weight"].abs() > 1].copy()
    if not overlap.empty:
        raise ValueError(f"Split Positions produced overlapping exposure on {len(overlap)} bars")

    return weights[["dt", "symbol", "weight", "price"]].sort_values("dt").reset_index(drop=True), data


def run_position_event_backtest(ohlc: pd.DataFrame, symbol: str):
    """Run the native strategy backtest and return weights plus raw artifacts."""
    bars = format_standard_kline(
        ohlc[["dt", "symbol", "open", "close", "high", "low", "vol", "amount"]],
        freq=Freq.F15,
    )
    tactic = ConditionalResonancePositionEventStrategy(symbol=symbol, include_sdt_bar=True)
    res = tactic.backtest(bars, sdt=STAT_SDT, include_sdt_bar=True, emit_signals=True)
    dfw, holds = holds_to_weight_df(res.holds_df())
    return tactic, res, dfw, holds


def compare_weights(custom_dfw: pd.DataFrame, event_dfw: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Compare the prototype state-machine weights with Position/Event weights."""
    compare = custom_dfw[["dt", "symbol", "weight", "price"]].merge(
        event_dfw[["dt", "weight"]],
        on="dt",
        how="outer",
        suffixes=("_custom", "_event"),
    )
    compare[["weight_custom", "weight_event"]] = compare[["weight_custom", "weight_event"]].fillna(0.0)
    compare["weight_diff"] = compare["weight_event"] - compare["weight_custom"]
    stats = {
        "bars": int(len(compare)),
        "diff_bars": int((compare["weight_diff"] != 0).sum()),
        "abs_diff_sum": float(compare["weight_diff"].abs().sum()),
        "max_abs_diff": float(compare["weight_diff"].abs().max()),
    }
    return stats, compare.sort_values("dt").reset_index(drop=True)


def save_outputs(
    output_dir: Path,
    tactic: ConditionalResonancePositionEventStrategy,
    res,
    event_dfw: pd.DataFrame,
    event_holds: pd.DataFrame,
    event_stats: dict,
    event_daily: pd.DataFrame,
    custom_stats: dict,
    compare_stats: dict,
    compare_df: pd.DataFrame,
) -> None:
    """Persist Position/Event artifacts and comparison results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tactic.save_positions(output_dir / "positions")
    event_dfw.to_csv(output_dir / "weights_position_event.csv", index=False, encoding="utf-8-sig")
    event_holds.to_csv(output_dir / "holds_position_event.csv", index=False, encoding="utf-8-sig")
    res.pairs_df().to_csv(output_dir / "pairs_position_event.csv", index=False, encoding="utf-8-sig")
    res.signals_df().to_csv(output_dir / "signals_position_event.csv", index=False, encoding="utf-8-sig")
    event_daily.to_csv(output_dir / "daily_position_event.csv", index_label="dt", encoding="utf-8-sig")
    compare_df.to_csv(output_dir / "weights_compare.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([event_stats]).to_csv(output_dir / "summary_position_event.csv", index=False, encoding="utf-8-sig")
    period_stats(event_daily, event_stats).to_csv(
        output_dir / "period_stats_position_event.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([compare_stats]).to_csv(output_dir / "comparison_summary.csv", index=False, encoding="utf-8-sig")

    report = f"""# Position/Event 条件式多周期共振策略

## 架构

- 信号：`{LONG_FILTER_COL}` / `{ENTRY_COL}` / `{EXIT_COL}`
- 事件：多头和空头分别定义开仓、平仓事件
- 持仓：拆成 `CR_Long` 与 `CR_Short` 两个 Position
- 组合权重：两个 Position 的 holds.pos 求和

## 为什么拆成两个 Position

单个多空 Position 会按 `opens + exits` 固定顺序寻找第一个匹配事件；方向无关的平多事件可能在空头持仓时先匹配，从而挡住平空事件。拆分多头 / 空头 Position 后，每个 Position 只处理自己的方向，事件优先级与条件式状态机一致。

## 绩效

- Position/Event 年化收益：{event_stats["annual_return"]:.2%}
- Position/Event 最大回撤：{event_stats["max_drawdown"]:.2%}
- Position/Event 期末净值：{event_stats["final_nav"]:.4f}
- 原条件式状态机年化收益：{custom_stats["annual_return"]:.2%}
- 原条件式状态机最大回撤：{custom_stats["max_drawdown"]:.2%}

## 权重对齐

- 对比K线数：{compare_stats["bars"]}
- 差异K线数：{compare_stats["diff_bars"]}
- 权重绝对差合计：{compare_stats["abs_diff_sum"]:.6f}
"""
    (output_dir / "strategy_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["cache", "jq"], default="cache")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--use-signal-cache", action="store_true")
    args = parser.parse_args()

    symbol = normalize_symbol(args.symbol)
    custom_output_dir = get_output_dir(symbol)
    output_dir = get_position_event_output_dir(symbol)

    ohlc = load_ohlc(args.source, symbol, custom_output_dir)
    sig = build_signal_frame(ohlc, use_cache=args.use_signal_cache, output_dir=custom_output_dir)
    custom_dfw = build_condition_weights(sig)
    custom_stats, _, _ = evaluate_weight_df(custom_dfw)

    tactic, res, event_dfw, event_holds = run_position_event_backtest(ohlc, symbol)
    event_stats, event_daily, _ = evaluate_weight_df(event_dfw)
    event_stats["strategy"] = POSITION_EVENT_STRATEGY_NAME

    compare_stats, compare_df = compare_weights(custom_dfw, event_dfw)
    save_outputs(
        output_dir,
        tactic,
        res,
        event_dfw,
        event_holds,
        event_stats,
        event_daily,
        custom_stats,
        compare_stats,
        compare_df,
    )

    print("Position/Event stats:")
    print(pd.DataFrame([event_stats]).to_string(index=False))
    print("\nCustom state-machine stats:")
    print(pd.DataFrame([custom_stats]).to_string(index=False))
    print("\nWeight comparison:")
    print(pd.DataFrame([compare_stats]).to_string(index=False))
    print(f"\noutputs: {output_dir}")


if __name__ == "__main__":
    main()
