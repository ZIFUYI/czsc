"""15-minute intraday direction CTA research on 000852.XSHG.

The strategy uses only native CZSC signals:

- ``bar_intraday_direction_V260424`` decides the intraday direction at a fixed
  15-minute bar close.
- ``bar_intraday_exit_V260424`` forces an end-of-day exit.
- ``Position.stop_loss`` handles fixed BP stop loss inside the Rust trader
  engine.

Default selected variant:

- decision time: 11:15
- exit time: 15:00
- direction: long/short
- stop loss: 50BP
- short filter: 60-minute close below SMA60

Run:
    uv run --no-sync python examples/signals_dev/backtest_intraday_direction_000852_15m.py

Run the parameter sweep:
    uv run --no-sync python examples/signals_dev/backtest_intraday_direction_000852_15m.py --sweep
"""

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

# Avoid optional ClickHouse/lz4 import issues when importing czsc top-level modules.
for mod in [
    "clickhouse_connect",
    "clickhouse_connect.driver",
    "clickhouse_connect.driver.client",
    "clickhouse_connect.driver.httpclient",
    "clickhouse_connect.driver.compression",
]:
    sys.modules.setdefault(mod, MagicMock())

from czsc import CzscStrategyBase, Event, Freq, Position  # noqa: E402
from examples.signals_dev.test_zdy_macd_bc_000852 import read_jq_sdk_bars  # noqa: E402

SYMBOL = "000852.XSHG"
DATA_SDT = "20180101"  # warm-up for engine state; stats start at STAT_SDT
STAT_SDT = "20200101"
STAT_EDT = "20260520"
FETCH_EDT = "20260521"  # JQData intraday end_date is effectively exclusive
FEE_RATE = 0.0002
OUTPUT_DIR = ROOT / "examples" / "results" / "intraday_direction_000852_15m"

DEFAULT_DECISION_TIME = "1115"
DEFAULT_EXIT_TIME = "1500"
DEFAULT_MODE = "LS"
DEFAULT_STOP_LOSS_BP = 50
DEFAULT_SHORT_FILTER = "60m_sma60_bear"

SHORT_FILTERS: dict[str, list[str]] = {
    "none": [],
    "60m_sma5_bear": ["60分钟_D1SMA#5_分类V221101_空头_任意_任意_0"],
    "60m_sma60_bear": ["60分钟_D1SMA#60_分类V221101_空头_任意_任意_0"],
    "60m_dma30_120_short": ["60分钟_N30M120#SMA_双均线过滤V240330_看空_任意_任意_0"],
    "daily_ma30_bear": ["日线_D1SMA#30_分类V221101_空头_任意_任意_0"],
    "daily_dma20_60_short": ["日线_N20M60#SMA_双均线过滤V240330_看空_任意_任意_0"],
}

SHORT_FILTER_SUFFIX = {
    "none": "",
    "60m_sma5_bear": "SF60MA5",
    "60m_sma60_bear": "SF60MA60",
    "60m_dma30_120_short": "SF60DMA30_120",
    "daily_ma30_bear": "SFDMA30",
    "daily_dma20_60_short": "SFDDMA20_60",
}


@dataclass(frozen=True)
class Variant:
    """One intraday direction strategy variant."""

    decision_time: str
    mode: str = DEFAULT_MODE
    stop_loss_bp: int = DEFAULT_STOP_LOSS_BP
    exit_time: str = DEFAULT_EXIT_TIME
    short_filter: str = DEFAULT_SHORT_FILTER

    @property
    def name(self) -> str:
        base = f"IDIR{self.decision_time}_{self.mode}_SL{self.stop_loss_bp}"
        suffix = SHORT_FILTER_SUFFIX[self.short_filter]
        return f"{base}_{suffix}" if suffix else base

    @property
    def short_filter_signals(self) -> list[str]:
        return list(SHORT_FILTERS[self.short_filter])


def intraday_direction_signal(decision_time: str, value: str) -> str:
    """Build a native intraday direction signal string."""
    return f"15分钟_D1T{decision_time}_日内方向V260424_{value}_任意_任意_0"


def intraday_exit_signal(exit_time: str) -> str:
    """Build a native intraday fixed-exit signal string."""
    return f"15分钟_D1T{exit_time}_日内平仓V260424_平仓_任意_任意_0"


class IntradayDirectionStrategy(CzscStrategyBase):
    """15-minute intraday direction CTA strategy."""

    def __init__(self, variant: Variant):
        super().__init__(symbol=SYMBOL, name=variant.name, bg_max_count=2000)
        self.variant = variant

    @property
    def positions(self) -> list[Position]:
        opens = []
        exits = []
        exit_signal = intraday_exit_signal(self.variant.exit_time)

        if self.variant.mode in {"LS", "LONG"}:
            opens.append(
                Event.load(
                    {
                        "name": "日内方向_开多",
                        "operate": "开多",
                        "signals_all": [intraday_direction_signal(self.variant.decision_time, "看多")],
                    }
                )
            )
            exits.append(Event.load({"name": "日内收盘_平多", "operate": "平多", "signals_any": [exit_signal]}))

        if self.variant.mode in {"LS", "SHORT"}:
            opens.append(
                Event.load(
                    {
                        "name": "日内方向_开空",
                        "operate": "开空",
                        "signals_all": [
                            intraday_direction_signal(self.variant.decision_time, "看空"),
                            *self.variant.short_filter_signals,
                        ],
                    }
                )
            )
            exits.append(Event.load({"name": "日内收盘_平空", "operate": "平空", "signals_any": [exit_signal]}))

        return [
            Position(
                name=self.variant.name,
                symbol=SYMBOL,
                opens=opens,
                exits=exits,
                interval=0,
                timeout=16,
                stop_loss=self.variant.stop_loss_bp,
                t0=False,
            )
        ]


def load_15m_bars():
    """Load JQData 15-minute bars with enough warm-up history."""
    bars = read_jq_sdk_bars(freq="15m", czsc_freq=Freq.F15, sdt=DATA_SDT, edt=FETCH_EDT)
    stat_edt = pd.Timestamp(STAT_EDT) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return [bar for bar in bars if pd.Timestamp(bar.dt).tz_localize(None) <= stat_edt]


def holds_to_weight_df(holds: pd.DataFrame) -> pd.DataFrame:
    """Convert CZSC holds output to a single-symbol weight table."""
    dfw = holds[["dt", "symbol", "pos", "price"]].rename(columns={"pos": "weight"}).copy()
    dfw["dt"] = pd.to_datetime(dfw["dt"]).dt.tz_localize(None)
    return dfw[["dt", "symbol", "weight", "price"]]


def evaluate_weight_df(dfw: pd.DataFrame, fee_rate: float = FEE_RATE) -> tuple[dict, pd.DataFrame]:
    """Evaluate a 15-minute weight curve with daily aggregated returns."""
    data = dfw.sort_values("dt").copy()
    data["price_ret"] = data.groupby("symbol")["price"].pct_change().fillna(0)
    data["prev_weight"] = data.groupby("symbol")["weight"].shift(1).fillna(0)
    data["turnover"] = data.groupby("symbol")["weight"].diff().abs().fillna(data["weight"].abs())
    data["ret"] = data["prev_weight"] * data["price_ret"] - data["turnover"] * fee_rate

    daily = data.groupby(data["dt"].dt.normalize())["ret"].sum().sort_index().to_frame("ret")
    daily["nav"] = (1 + daily["ret"]).cumprod()
    daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1

    years = (daily.index[-1] - daily.index[0]).days / 365.25
    final_nav = float(daily["nav"].iloc[-1])
    annual_return = final_nav ** (1 / years) - 1 if years > 0 and final_nav > 0 else float("nan")
    max_drawdown = float(daily["drawdown"].min())
    cumulative_return = final_nav - 1
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else float("nan")
    return_drawdown = cumulative_return / abs(max_drawdown) if max_drawdown < 0 else float("nan")
    win_rate = float((daily["ret"] > 0).mean())

    stats = {
        "annual_return": annual_return,
        "cumulative_return": cumulative_return,
        "final_nav": final_nav,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "return_drawdown": return_drawdown,
        "daily_win_rate": win_rate,
        "fee_rate": fee_rate,
        "start": str(daily.index[0].date()),
        "end": str(daily.index[-1].date()),
    }
    return stats, daily


def run_variant(bars, variant: Variant, save_details: bool = False) -> dict:
    """Run one variant and optionally persist its detailed artifacts."""
    strategy = IntradayDirectionStrategy(variant)
    result = strategy.backtest(bars, sdt=STAT_SDT, emit_signals=False)
    pairs = result.pairs_df()
    holds = result.holds_df()
    dfw = holds_to_weight_df(holds)
    stats, daily = evaluate_weight_df(dfw)
    stats.update(
        {
            "variant": variant.name,
            "decision_time": variant.decision_time,
            "exit_time": variant.exit_time,
            "mode": variant.mode,
            "stop_loss_bp": variant.stop_loss_bp,
            "short_filter": variant.short_filter,
            "short_filter_signals": " | ".join(variant.short_filter_signals),
            "pairs": len(pairs),
            "active_bars": int((dfw["weight"] != 0).sum()),
        }
    )

    if save_details:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(OUTPUT_DIR / f"{variant.name}_pairs.csv", index=False, encoding="utf-8-sig")
        holds.to_csv(OUTPUT_DIR / f"{variant.name}_holds.csv", index=False, encoding="utf-8-sig")
        dfw.to_csv(OUTPUT_DIR / f"{variant.name}_weights.csv", index=False, encoding="utf-8-sig")
        daily.to_csv(OUTPUT_DIR / f"{variant.name}_daily.csv", index_label="dt", encoding="utf-8-sig")
        pd.DataFrame([stats]).to_csv(OUTPUT_DIR / f"{variant.name}_summary.csv", index=False, encoding="utf-8-sig")

    return stats


def run_sweep(bars, short_filter: str) -> pd.DataFrame:
    """Run a compact decision-time / direction / stop-loss sweep."""
    rows = []
    times = ["0945", "1000", "1015", "1030", "1045", "1100", "1115", "1130"]
    modes = ["LS", "LONG", "SHORT"]
    stop_losses = [50, 100, 150, 200, 300, 500, 800, 1000]
    total = len(times) * len(modes) * len(stop_losses)
    i = 0
    for decision_time in times:
        for mode in modes:
            for stop_loss_bp in stop_losses:
                i += 1
                variant = Variant(
                    decision_time=decision_time,
                    mode=mode,
                    stop_loss_bp=stop_loss_bp,
                    short_filter=short_filter,
                )
                stats = run_variant(bars, variant, save_details=False)
                rows.append(stats)
                print(
                    f"[{i:03d}/{total}] {variant.name} "
                    f"annual={stats['annual_return']:.2%} "
                    f"calmar={stats['calmar']:.2f} "
                    f"return/dd={stats['return_drawdown']:.2f} "
                    f"max_dd={stats['max_drawdown']:.2%}"
                )

    df = pd.DataFrame(rows).sort_values(["calmar", "annual_return"], ascending=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "sweep_summary.csv", index=False, encoding="utf-8-sig")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="run the parameter sweep and save sweep_summary.csv")
    parser.add_argument(
        "--short-filter",
        choices=sorted(SHORT_FILTERS),
        default=DEFAULT_SHORT_FILTER,
        help="extra trend filter applied only to short opens",
    )
    args = parser.parse_args()

    bars = load_15m_bars()
    print(f"bars: {len(bars)} | {bars[0].dt} -> {bars[-1].dt}")

    selected = Variant(
        decision_time=DEFAULT_DECISION_TIME,
        mode=DEFAULT_MODE,
        stop_loss_bp=DEFAULT_STOP_LOSS_BP,
        exit_time=DEFAULT_EXIT_TIME,
        short_filter=args.short_filter,
    )
    stats = run_variant(bars, selected, save_details=True)
    print("\nselected:")
    print(pd.DataFrame([stats]).to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")

    if args.sweep:
        print("\nsweep:")
        sweep = run_sweep(bars, args.short_filter)
        cols = [
            "variant",
            "annual_return",
            "calmar",
            "return_drawdown",
            "max_drawdown",
            "final_nav",
            "pairs",
        ]
        print("\ntop 20:")
        print(sweep[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
