"""Backtest intraday structure-filter strategy on 000852.XSHG.

Scheme:
- Previous trading day's 30m#daily intraday structure filters direction.
- Current day's fixed-time 15m intraday direction confirms entry.
- Fixed end-of-day exit and BP stop loss keep positions intraday-only.

Run:
    uv run --no-sync python examples/signals_dev/backtest_intraday_structure_filter_000852_15m.py
"""

from __future__ import annotations

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

from czsc import CzscStrategyBase, Event, Freq, Position  # noqa: E402
from examples.signals_dev.test_zdy_macd_bc_000852 import read_jq_sdk_bars  # noqa: E402

SYMBOL = "000852.XSHG"
DATA_SDT = "20180101"
STAT_SDT = "20200101"
STAT_EDT = "20260520"
FETCH_EDT = "20260521"
FEE_RATE = 0.0002
OUTPUT_DIR = ROOT / "examples" / "results" / "intraday_structure_filter_000852_15m"

LONG_STRUCTURES = ["无中枢上涨", "强平衡市"]
SHORT_STRUCTURES = ["无中枢下跌", "弱平衡市"]


@dataclass(frozen=True)
class Variant:
    """One structure-filtered intraday strategy variant."""

    decision_time: str = "1115"
    exit_time: str = "1500"
    stop_loss_bp: int = 50
    mode: str = "LS"
    structure_di: int = 2

    @property
    def name(self) -> str:
        return f"ISTF{self.decision_time}_{self.mode}_D{self.structure_di}_SL{self.stop_loss_bp}"


def intraday_direction_signal(decision_time: str, value: str) -> str:
    """Build native fixed-time intraday direction signal string."""
    return f"15分钟_D1T{decision_time}_日内方向V260424_{value}_任意_任意_0"


def intraday_exit_signal(exit_time: str) -> str:
    """Build native fixed-time intraday exit signal string."""
    return f"15分钟_D1T{exit_time}_日内平仓V260424_平仓_任意_任意_0"


def structure_signal(di: int, value: str) -> str:
    """Build previous-day 30m#daily intraday structure signal string."""
    return f"30分钟#日线_D{di}日_走势分类V230701_{value}_任意_任意_0"


class IntradayStructureFilterStrategy(CzscStrategyBase):
    """Previous-day structure filter plus current-day intraday direction."""

    def __init__(self, variant: Variant):
        super().__init__(symbol=SYMBOL, name=variant.name, bg_max_count=3000, market="A股")
        self.variant = variant

    @property
    def positions(self) -> list[Position]:
        positions = []
        exit_signal = intraday_exit_signal(self.variant.exit_time)
        common_kwargs = {
            "symbol": SYMBOL,
            "interval": 0,
            "timeout": 16,
            "stop_loss": self.variant.stop_loss_bp,
            "t0": True,
        }

        if self.variant.mode in {"LS", "LONG"}:
            long_open = Event.load(
                {
                    "name": "前日结构过滤_日内方向开多",
                    "operate": "开多",
                    "signals_all": [intraday_direction_signal(self.variant.decision_time, "看多")],
                    "signals_any": [structure_signal(self.variant.structure_di, x) for x in LONG_STRUCTURES],
                }
            )
            long_exit = Event.load({"name": "日内收盘平多", "operate": "平多", "signals_any": [exit_signal]})
            positions.append(
                Position(name=f"{self.variant.name}_Long", opens=[long_open], exits=[long_exit], **common_kwargs)
            )

        if self.variant.mode in {"LS", "SHORT"}:
            short_open = Event.load(
                {
                    "name": "前日结构过滤_日内方向开空",
                    "operate": "开空",
                    "signals_all": [intraday_direction_signal(self.variant.decision_time, "看空")],
                    "signals_any": [structure_signal(self.variant.structure_di, x) for x in SHORT_STRUCTURES],
                }
            )
            short_exit = Event.load({"name": "日内收盘平空", "operate": "平空", "signals_any": [exit_signal]})
            positions.append(
                Position(name=f"{self.variant.name}_Short", opens=[short_open], exits=[short_exit], **common_kwargs)
            )

        return positions


def load_15m_bars() -> list:
    """Load JQData 15-minute bars with warm-up history."""
    bars = read_jq_sdk_bars(freq="15m", czsc_freq=Freq.F15, sdt=DATA_SDT, edt=FETCH_EDT, symbol=SYMBOL)
    stat_edt = pd.Timestamp(STAT_EDT) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return [bar for bar in bars if pd.Timestamp(bar.dt).tz_localize(None) <= stat_edt]


def holds_to_weight_df(holds: pd.DataFrame) -> pd.DataFrame:
    """Combine split long/short Position holds into one weight table."""
    data = holds[["dt", "symbol", "pos", "price", "pos_name"]].copy()
    data["dt"] = pd.to_datetime(data["dt"]).dt.tz_localize(None)
    wide = (
        data.pivot_table(index=["dt", "symbol"], columns="pos_name", values="pos", aggfunc="last")
        .fillna(0.0)
        .reset_index()
    )
    pos_cols = [x for x in wide.columns if x not in {"dt", "symbol"}]
    wide["weight"] = wide[pos_cols].sum(axis=1)
    if (wide["weight"].abs() > 1).any():
        overlap = wide.loc[wide["weight"].abs() > 1, ["dt", "symbol", "weight"]].head()
        raise ValueError(f"long/short Positions overlap:\n{overlap}")

    prices = data.groupby(["dt", "symbol"], as_index=False)["price"].first()
    dfw = wide.merge(prices, on=["dt", "symbol"], how="left")
    return dfw[["dt", "symbol", "weight", "price"]].sort_values("dt").reset_index(drop=True)


def evaluate_weight_df(dfw: pd.DataFrame, fee_rate: float = FEE_RATE) -> tuple[dict, pd.DataFrame]:
    """Evaluate bar-level weights after cost, aggregated by day."""
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
    return {
        "start": str(daily.index[0].date()),
        "end": str(daily.index[-1].date()),
        "fee_rate": fee_rate,
        "final_nav": final_nav,
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "return_drawdown": return_drawdown,
        "daily_win_rate": float((daily["ret"] > 0).mean()),
    }, daily


def run_variant(bars: list, variant: Variant, save_details: bool = True) -> dict:
    """Run one strategy variant and optionally save artifacts."""
    strategy = IntradayStructureFilterStrategy(variant)
    result = strategy.backtest(bars, sdt=STAT_SDT, include_sdt_bar=True, emit_signals=save_details)
    pairs = result.pairs_df()
    holds = result.holds_df()
    signals = result.signals_df() if save_details else pd.DataFrame()
    dfw = holds_to_weight_df(holds)
    stats, daily = evaluate_weight_df(dfw)
    stats.update(
        {
            "variant": variant.name,
            "decision_time": variant.decision_time,
            "exit_time": variant.exit_time,
            "mode": variant.mode,
            "structure_di": variant.structure_di,
            "stop_loss_bp": variant.stop_loss_bp,
            "pairs": int(len(pairs)),
            "active_bars": int((dfw["weight"] != 0).sum()),
            "long_structures": "|".join(LONG_STRUCTURES),
            "short_structures": "|".join(SHORT_STRUCTURES),
        }
    )

    if save_details:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(OUTPUT_DIR / f"{variant.name}_pairs.csv", index=False, encoding="utf-8-sig")
        holds.to_csv(OUTPUT_DIR / f"{variant.name}_holds.csv", index=False, encoding="utf-8-sig")
        signals.to_csv(OUTPUT_DIR / f"{variant.name}_signals.csv", index=False, encoding="utf-8-sig")
        dfw.to_csv(OUTPUT_DIR / f"{variant.name}_weights.csv", index=False, encoding="utf-8-sig")
        daily.to_csv(OUTPUT_DIR / f"{variant.name}_daily.csv", index_label="dt", encoding="utf-8-sig")
        pd.DataFrame([stats]).to_csv(OUTPUT_DIR / f"{variant.name}_summary.csv", index=False, encoding="utf-8-sig")

    return stats


def main() -> None:
    """Run the baseline scheme-2 backtest."""
    bars = load_15m_bars()
    print(f"bars: {len(bars)} | {bars[0].dt} -> {bars[-1].dt}")
    variant = Variant()
    strategy = IntradayStructureFilterStrategy(variant)
    print("signals_config:")
    print(pd.DataFrame(strategy.signals_config).to_string(index=False))
    stats = run_variant(bars, variant, save_details=True)
    print("\nsummary:")
    print(pd.DataFrame([stats]).to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
