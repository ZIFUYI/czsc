"""Optimized conditional resonance strategy on 000905.XSHG.

This is a strategy-layer optimization of ``CR_PositionEvent_SplitLongShort_60m15m``
for 000905.XSHG. The structure is intentionally kept close to the original:

- Large-period filter: 60-minute SMA trend and same-direction slope.
- Small-period entry: 15-minute N32/T10 absolute momentum reaches an extreme.
- Small-period exit: 15-minute N8/T20 absolute momentum reaches the opposite
  extreme, or the 60-minute filter flips.
- Strategy-layer exposure scale: 0.79.

Run:
    uv run --no-sync python examples/signals_dev/backtest_conditional_resonance_000905_optimized.py
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

from czsc import CzscStrategyBase, Event, Freq, Position, format_standard_kline, generate_czsc_signals  # noqa: E402
from examples.signals_dev.test_zdy_macd_bc_000852 import read_jq_sdk_bars  # noqa: E402

SYMBOL = "000905.XSHG"
DATA_SDT = "20180101"
STAT_SDT = "20200101"
DEFAULT_STAT_EDT = None
DEFAULT_FETCH_EDT = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y%m%d")
FEE_RATE = 0.0002
RISK_SCALE = 0.79
OUTPUT_DIR = ROOT / "examples" / "results" / "conditional_resonance_000905_optimized"
SOURCE_CACHE_FILE = ROOT / "examples" / "results" / "conditional_resonance_000905_XSHG" / "000905_XSHG_15m_ohlc.csv"

STRATEGY_NAME = "CR_000905_SMA5Slope_N32T10_N8T20_S79"
LONG_FILTER_COL = "60分钟_D1SMA#5_分类V221101"
ENTRY_COL = "15分钟_D1N32T10_绝对动量V230227"
EXIT_COL = "15分钟_D1N8T20_绝对动量V230227"
SIGNALS_CONFIG = [
    {"name": "tas_ma_base_V221101", "freq": "60分钟", "di": 1, "timeperiod": 5, "ma_type": "SMA"},
    {"name": "bar_bpm_V230227", "freq": "15分钟", "di": 1, "n": 32, "th": 10},
    {"name": "bar_bpm_V230227", "freq": "15分钟", "di": 1, "n": 8, "th": 20},
]
LARGE_TIMEOUT = 1_000_000
LARGE_STOP_LOSS_BP = 1_000_000.0


def load_ohlc(source: str, stat_edt: str | None, fetch_edt: str | None) -> pd.DataFrame:
    """Load 15-minute OHLCV data from local cache or JQData."""
    if source == "cache":
        if not SOURCE_CACHE_FILE.exists():
            raise FileNotFoundError(f"Local 15m cache not found: {SOURCE_CACHE_FILE}; rerun with --source jq")
        data = pd.read_csv(SOURCE_CACHE_FILE, parse_dates=["dt"])
    else:
        bars = read_jq_sdk_bars(
            freq="15m",
            czsc_freq=Freq.F15,
            sdt=DATA_SDT,
            edt=fetch_edt or DEFAULT_FETCH_EDT,
            symbol=SYMBOL,
        )
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
    data = data[data["dt"] >= pd.Timestamp(DATA_SDT)].copy()
    if stat_edt:
        stat_edt_ts = pd.Timestamp(stat_edt) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        data = data[data["dt"] <= stat_edt_ts].copy()
    data = data.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_DIR / "000905_XSHG_15m_ohlc.csv", index=False, encoding="utf-8-sig")
    return data


def build_signal_frame(data: pd.DataFrame, use_cache: bool) -> pd.DataFrame:
    """Generate or load native CZSC signals for the optimized rule."""
    signal_file = OUTPUT_DIR / "native_signals.csv"
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


def signal_clause(signal_col: str, v1: str, v2: str = "任意") -> str:
    """Build a CZSC event signal clause."""
    return f"{signal_col}_{v1}_{v2}_任意_0"


class Optimized000905ConditionalResonanceStrategy(CzscStrategyBase):
    """Position/Event implementation of the optimized 000905 conditional resonance rule."""

    @property
    def positions(self) -> list[Position]:
        long_pos = Position(
            name="CR_000905_Long_SMA5Up_N32T10_N8T20",
            symbol=self.symbol,
            opens=[
                Event.load(
                    {
                        "name": "60m多头向上_N32超强_开多",
                        "operate": "开多",
                        "signals_all": [
                            signal_clause(LONG_FILTER_COL, "多头", "向上"),
                            signal_clause(ENTRY_COL, "超强"),
                        ],
                    }
                )
            ],
            exits=[
                Event.load(
                    {
                        "name": "N8超弱或60m空头向下_平多",
                        "operate": "平多",
                        "signals_any": [
                            signal_clause(EXIT_COL, "超弱"),
                            signal_clause(LONG_FILTER_COL, "空头", "向下"),
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
            name="CR_000905_Short_SMA5Down_N32T10_N8T20",
            symbol=self.symbol,
            opens=[
                Event.load(
                    {
                        "name": "60m空头向下_N32超弱_开空",
                        "operate": "开空",
                        "signals_all": [
                            signal_clause(LONG_FILTER_COL, "空头", "向下"),
                            signal_clause(ENTRY_COL, "超弱"),
                        ],
                    }
                )
            ],
            exits=[
                Event.load(
                    {
                        "name": "N8超强或60m多头向上_平空",
                        "operate": "平空",
                        "signals_any": [
                            signal_clause(EXIT_COL, "超强"),
                            signal_clause(LONG_FILTER_COL, "多头", "向上"),
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


def build_condition_weights(sig: pd.DataFrame) -> pd.DataFrame:
    """Build optimized state-machine weights from generated signals."""
    missing = [col for col in [LONG_FILTER_COL, ENTRY_COL, EXIT_COL] if col not in sig.columns]
    if missing:
        raise KeyError(f"Missing expected signal columns: {missing}")

    filter_value = sig[LONG_FILTER_COL].astype(str)
    entry_value = sig[ENTRY_COL].astype(str)
    exit_value = sig[EXIT_COL].astype(str)

    long_filter = filter_value.str.startswith("多头_向上").to_numpy()
    short_filter = filter_value.str.startswith("空头_向下").to_numpy()
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
        weight[i] = pos * RISK_SCALE

    out = sig[["dt", "symbol", "close", LONG_FILTER_COL, ENTRY_COL, EXIT_COL]].copy()
    out["weight"] = weight
    out["price"] = out["close"].astype(float)
    return out[["dt", "symbol", "weight", "price", LONG_FILTER_COL, ENTRY_COL, EXIT_COL]]


def holds_to_weight_df(holds: pd.DataFrame) -> pd.DataFrame:
    """Combine split long/short Position holds into one scaled portfolio weight series."""
    data = holds[["dt", "symbol", "pos", "price", "pos_name"]].copy()
    data["dt"] = pd.to_datetime(data["dt"]).dt.tz_localize(None)
    data = data.sort_values(["dt", "pos_name"]).reset_index(drop=True)
    wide = (
        data.pivot_table(index=["dt", "symbol"], columns="pos_name", values="pos", aggfunc="last")
        .fillna(0.0)
        .reset_index()
    )
    pos_cols = [col for col in wide.columns if col not in {"dt", "symbol"}]
    wide["weight"] = wide[pos_cols].sum(axis=1) * RISK_SCALE
    prices = data.groupby(["dt", "symbol"], as_index=False)["price"].first()
    weights = wide.merge(prices, on=["dt", "symbol"], how="left")
    overlap = weights[weights["weight"].abs() > RISK_SCALE]
    if not overlap.empty:
        raise ValueError(f"Split Positions produced overlapping exposure on {len(overlap)} bars")
    return weights[["dt", "symbol", "weight", "price"]].sort_values("dt").reset_index(drop=True)


def run_position_event_backtest(
    ohlc: pd.DataFrame,
) -> tuple[Optimized000905ConditionalResonanceStrategy, object, pd.DataFrame]:
    """Run project-native Position/Event backtest."""
    bars = format_standard_kline(
        ohlc[["dt", "symbol", "open", "close", "high", "low", "vol", "amount"]],
        freq=Freq.F15,
    )
    tactic = Optimized000905ConditionalResonanceStrategy(symbol=SYMBOL, include_sdt_bar=True)
    res = tactic.backtest(bars, sdt=STAT_SDT, include_sdt_bar=True, emit_signals=True)
    dfw = holds_to_weight_df(res.holds_df())
    return tactic, res, dfw


def evaluate_weight_df(dfw: pd.DataFrame, strategy: str = STRATEGY_NAME) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
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
        "symbol": SYMBOL,
        "strategy": strategy,
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
        "risk_scale": RISK_SCALE,
        "large_period_filter": f"{LONG_FILTER_COL}=多头_向上 / 空头_向下",
        "small_period_entry": ENTRY_COL,
        "small_period_exit": EXIT_COL,
    }
    return stats, daily, data


def yearly_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate calendar-year performance."""
    rows = []
    for year, data in daily.groupby(daily.index.year):
        if len(data) < 2:
            continue
        nav = (1 + data["ret"]).cumprod()
        drawdown = nav / nav.cummax() - 1
        years = (data.index[-1] - data.index[0]).days / 365.25
        final_nav = float(nav.iloc[-1])
        annual = final_nav ** (1 / years) - 1 if years > 0 and final_nav > 0 else float("nan")
        rows.append(
            {
                "year": int(year),
                "start": str(data.index[0].date()),
                "end": str(data.index[-1].date()),
                "return": final_nav - 1,
                "annual_return": annual,
                "max_drawdown": float(drawdown.min()),
                "win_rate": float((data["ret"] > 0).mean()),
                "days": int(len(data)),
            }
        )
    return pd.DataFrame(rows)


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
    end_date = pd.Timestamp(stats["end"])
    for period, sdt in [
        ("2020_2022", "2020-01-01"),
        ("2023_2024", "2023-01-01"),
        ("last_2y", (end_date - pd.DateOffset(years=2)).strftime("%Y-%m-%d")),
        ("since_2025", "2025-01-01"),
        ("2026_ytd", "2026-01-01"),
    ]:
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


def compare_weights(custom_dfw: pd.DataFrame, event_dfw: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Compare optimized state-machine weights with Position/Event weights."""
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
        "diff_bars": int((compare["weight_diff"].abs() > 1e-12).sum()),
        "abs_diff_sum": float(compare["weight_diff"].abs().sum()),
        "max_abs_diff": float(compare["weight_diff"].abs().max()),
    }
    return stats, compare.sort_values("dt").reset_index(drop=True)


def save_equity_svg(daily: pd.DataFrame, output_file: Path) -> None:
    """Save a lightweight SVG equity curve."""
    width, height = 1200, 700
    pad_l, pad_r, pad_t = 80, 40, 65
    nav_x, nav_y, nav_w, nav_h = pad_l, pad_t, width - pad_l - pad_r, 360
    dd_x, dd_y, dd_w, dd_h = pad_l, 505, width - pad_l - pad_r, 120

    def points(xs: pd.Series, ys: pd.Series, x0: int, y0: int, w: int, h: int, ymin: float, ymax: float) -> str:
        xs_num = xs.astype("int64") / 1e9
        xmin, xmax = float(xs_num.min()), float(xs_num.max())
        xmax = xmax if xmax != xmin else xmin + 1
        ymax = ymax if ymax != ymin else ymin + 1
        px = x0 + (xs_num - xmin) / (xmax - xmin) * w
        py = y0 + h - (ys - ymin) / (ymax - ymin) * h
        return " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py, strict=True))

    frame = daily.reset_index().rename(columns={"index": "dt"})
    nav = frame["nav"].astype(float)
    dd = frame["drawdown"].astype(float)
    nav_min, nav_max = float(nav.min()), float(nav.max())
    nav_pad = (nav_max - nav_min) * 0.08 or 0.1
    nav_min -= nav_pad
    nav_max += nav_pad
    dd_min = min(float(dd.min()) * 1.15, -0.01)
    nav_pts = points(frame["dt"], nav, nav_x, nav_y, nav_w, nav_h, nav_min, nav_max)
    dd_pts = points(frame["dt"], dd, dd_x, dd_y, dd_w, dd_h, dd_min, 0.0)
    ret = nav.iloc[-1] - 1
    subtitle = (
        f"{frame['dt'].iloc[0].date()} to {frame['dt'].iloc[-1].date()} | Return {ret:.2%} | Max DD {dd.min():.2%}"
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fff"/>
<text x="{pad_l}" y="34" fill="#0f172a" font-size="24" font-family="Arial, sans-serif" font-weight="700">{STRATEGY_NAME}</text>
<text x="{pad_l}" y="56" fill="#475569" font-size="14" font-family="Arial, sans-serif">{subtitle}</text>
<rect x="{nav_x}" y="{nav_y}" width="{nav_w}" height="{nav_h}" fill="none" stroke="#cbd5e1"/>
<polyline points="{nav_pts}" fill="none" stroke="#0f766e" stroke-width="2.3"/>
<rect x="{dd_x}" y="{dd_y}" width="{dd_w}" height="{dd_h}" fill="none" stroke="#cbd5e1"/>
<polygon points="{dd_x},{dd_y} {dd_pts} {dd_x + dd_w},{dd_y}" fill="#ef4444" opacity="0.34"/>
<polyline points="{dd_pts}" fill="none" stroke="#b91c1c" stroke-width="1.4"/>
</svg>"""
    output_file.write_text(svg, encoding="utf-8")


def save_outputs(
    tactic, res, dfw: pd.DataFrame, stats: dict, daily: pd.DataFrame, data: pd.DataFrame, compare: tuple
) -> None:
    """Persist optimized strategy artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    compare_stats, compare_df = compare
    tactic.save_positions(OUTPUT_DIR / "positions")
    dfw.to_csv(OUTPUT_DIR / "weights_position_event.csv", index=False, encoding="utf-8-sig")
    res.holds_df().to_csv(OUTPUT_DIR / "holds_position_event.csv", index=False, encoding="utf-8-sig")
    res.pairs_df().to_csv(OUTPUT_DIR / "pairs_position_event.csv", index=False, encoding="utf-8-sig")
    res.signals_df().to_csv(OUTPUT_DIR / "signals_position_event.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUTPUT_DIR / "daily_position_event.csv", index_label="dt", encoding="utf-8-sig")
    data.to_csv(OUTPUT_DIR / "bar_returns_position_event.csv", index=False, encoding="utf-8-sig")
    compare_df.to_csv(OUTPUT_DIR / "weights_compare.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([stats]).to_csv(OUTPUT_DIR / "summary_position_event.csv", index=False, encoding="utf-8-sig")
    period_stats(daily, stats).to_csv(OUTPUT_DIR / "period_stats_position_event.csv", index=False, encoding="utf-8-sig")
    yearly_stats(daily).to_csv(OUTPUT_DIR / "yearly_stats_position_event.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([compare_stats]).to_csv(OUTPUT_DIR / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    save_equity_svg(daily, OUTPUT_DIR / "equity_position_event_full.svg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["cache", "jq"], default="cache")
    parser.add_argument("--edt", default=DEFAULT_STAT_EDT, help="statistics end date, e.g. 20260723")
    parser.add_argument("--fetch-edt", default=None, help="JQData fetch end date; intraday end_date is exclusive")
    parser.add_argument("--use-signal-cache", action="store_true")
    args = parser.parse_args()

    ohlc = load_ohlc(args.source, stat_edt=args.edt, fetch_edt=args.fetch_edt)
    print(f"ohlc rows: {len(ohlc)} | {ohlc['dt'].iloc[0]} -> {ohlc['dt'].iloc[-1]}")
    sig = build_signal_frame(ohlc, use_cache=args.use_signal_cache)
    print(f"signal rows: {len(sig)} | {sig['dt'].iloc[0]} -> {sig['dt'].iloc[-1]}")

    custom_dfw = build_condition_weights(sig)
    tactic, res, event_dfw = run_position_event_backtest(ohlc)
    compare = compare_weights(custom_dfw, event_dfw)
    stats, daily, data = evaluate_weight_df(event_dfw)
    save_outputs(tactic, res, event_dfw, stats, daily, data, compare)

    print(pd.DataFrame([stats]).to_string(index=False))
    print("\nperiod stats:")
    print(period_stats(daily, stats).to_string(index=False))
    print("\nyearly stats:")
    print(yearly_stats(daily).to_string(index=False))
    print("\nweight comparison:")
    print(pd.DataFrame([compare[0]]).to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
