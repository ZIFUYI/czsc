"""Overnight trend-following research on 000852.XSHG with 15-minute bars.

Default strategy:

- Ensemble four nearby 15-minute momentum legs: ``32/30BP``, ``48/40BP``,
  ``64/10BP`` and ``64/70BP``.
- Each leg holds long / short until the trend flips.
- Average the four leg weights and apply an 0.8 risk scale, so disagreement
  among legs automatically reduces exposure.

The script can use the saved 15-minute price series from the intraday research
or refetch JQData bars.

Run selected strategy from saved prices:
    .venv/bin/python examples/signals_dev/backtest_overnight_trend_000852_15m.py

Run parameter sweep:
    .venv/bin/python examples/signals_dev/backtest_overnight_trend_000852_15m.py --sweep

Run from JQData:
    .venv/bin/python examples/signals_dev/backtest_overnight_trend_000852_15m.py --source jq
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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

from czsc import Freq  # noqa: E402
from examples.signals_dev.test_zdy_macd_bc_000852 import read_jq_sdk_bars  # noqa: E402

DEFAULT_SYMBOL = "000852.XSHG"
DATA_SDT = "20180101"
STAT_SDT = "20200101"
STAT_EDT = "20260520"
FETCH_EDT = "20260521"
FEE_RATE = 0.0002
DEFAULT_LOOKBACK = 64
DEFAULT_THRESHOLD_BP = 20
DEFAULT_LEGS = "32:30,48:40,64:10,64:70"
DEFAULT_RISK_SCALE = 0.8
DEFAULT_OUTPUT_DIR = ROOT / "examples" / "results" / "overnight_trend_000852_15m"
SAVED_PRICE_FILE = ROOT / "examples" / "results" / "intraday_direction_000852_15m" / "IDIR1115_LS_SL50_weights.csv"


@dataclass(frozen=True)
class TrendVariant:
    """One overnight momentum trend variant."""

    lookback: int = DEFAULT_LOOKBACK
    threshold_bp: int = DEFAULT_THRESHOLD_BP
    legs: tuple[tuple[int, int], ...] = ()
    risk_scale: float = DEFAULT_RISK_SCALE

    @property
    def name(self) -> str:
        if self.legs:
            body = "_".join(f"N{lookback}B{threshold_bp}" for lookback, threshold_bp in self.legs)
            return f"ENS{len(self.legs)}_{body}_S{int(round(self.risk_scale * 100))}"
        return f"MOM_N{self.lookback}_B{self.threshold_bp}"


def normalize_jq_symbol(symbol: str) -> str:
    """Normalize common index / ETF codes to JoinQuant symbols."""
    symbol = symbol.strip().upper()
    if "." in symbol:
        return symbol
    known = {
        "000852": "000852.XSHG",
        "000905": "000905.XSHG",
        "159915": "159915.XSHE",
    }
    if symbol in known:
        return known[symbol]
    if symbol.startswith(("159", "150", "161", "162")):
        return f"{symbol}.XSHE"
    if symbol.startswith(("510", "511", "512", "513", "515", "516", "517", "518", "560", "561", "562", "563", "588")):
        return f"{symbol}.XSHG"
    return f"{symbol}.XSHG"


def get_output_dir(symbol: str) -> Path:
    """Return the output directory for a symbol."""
    if symbol == DEFAULT_SYMBOL:
        return DEFAULT_OUTPUT_DIR
    safe_symbol = symbol.replace(".", "_")
    return ROOT / "examples" / "results" / f"overnight_trend_{safe_symbol}_15m"


def load_price_frame(source: str, symbol: str) -> pd.DataFrame:
    """Load 15-minute close prices."""
    if source == "saved":
        if symbol != DEFAULT_SYMBOL:
            raise ValueError("saved source only contains 000852.XSHG; use --source jq for other symbols")
        data = pd.read_csv(SAVED_PRICE_FILE, parse_dates=["dt"])
        data = data.drop_duplicates("dt").sort_values("dt")
        data = data.rename(columns={"price": "close"})
        return data[["dt", "symbol", "close"]].copy()

    bars = read_jq_sdk_bars(freq="15m", czsc_freq=Freq.F15, sdt=DATA_SDT, edt=FETCH_EDT, symbol=symbol)
    rows = [
        {
            "dt": pd.Timestamp(bar.dt).tz_localize(None),
            "symbol": bar.symbol,
            "close": float(bar.close),
        }
        for bar in bars
    ]
    data = pd.DataFrame(rows).drop_duplicates("dt").sort_values("dt")
    stat_edt = pd.Timestamp(STAT_EDT) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return data[data["dt"] <= stat_edt].copy()


def parse_legs(raw: str) -> tuple[tuple[int, int], ...]:
    """Parse ``lookback:threshold_bp`` legs from CLI input."""
    if not raw.strip():
        return ()
    legs = []
    for item in raw.split(","):
        lookback, threshold_bp = item.strip().split(":", 1)
        legs.append((int(lookback), int(threshold_bp)))
    return tuple(legs)


def make_single_momentum_weight(price: pd.Series, lookback: int, threshold_bp: int) -> pd.Series:
    """Create one long / short momentum leg."""
    momentum = price / price.shift(lookback) - 1
    threshold = threshold_bp / 10000
    weight = pd.Series(0.0, index=price.index)
    weight[momentum > threshold] = 1.0
    weight[momentum < -threshold] = -1.0
    return weight.replace(0.0, np.nan).ffill().fillna(0.0)


def make_momentum_weight(price: pd.Series, variant: TrendVariant) -> pd.Series:
    """Create overnight trend weights from time-series momentum."""
    if not variant.legs:
        return make_single_momentum_weight(price, variant.lookback, variant.threshold_bp) * variant.risk_scale

    weights = [make_single_momentum_weight(price, lookback, threshold_bp) for lookback, threshold_bp in variant.legs]
    ensemble = sum(weights) / len(weights)
    return ensemble * variant.risk_scale


def evaluate_weight_df(dfw: pd.DataFrame, fee_rate: float = FEE_RATE) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
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

    stats = {
        "annual_return": annual_return,
        "cumulative_return": cumulative_return,
        "final_nav": final_nav,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "return_drawdown": return_drawdown,
        "daily_win_rate": float((daily["ret"] > 0).mean()),
        "turnover": float(data["turnover"].sum()),
        "active_bars": int((data["weight"] != 0).sum()),
        "fee_rate": fee_rate,
        "start": str(daily.index[0].date()),
        "end": str(daily.index[-1].date()),
    }
    return stats, daily, data


def extract_pairs(data: pd.DataFrame) -> pd.DataFrame:
    """Extract continuous non-zero position segments."""
    segments = []
    seg_id = (data["weight"] != data["weight"].shift()).cumsum()
    for _, seg in data.groupby(seg_id, sort=True):
        weight = float(seg["weight"].iloc[0])
        if weight == 0:
            continue
        ret = float((1 + seg["ret"]).prod() - 1)
        segments.append(
            {
                "symbol": seg["symbol"].iloc[0],
                "direction": "long" if weight > 0 else "short",
                "entry_dt": seg["dt"].iloc[0],
                "exit_dt": seg["dt"].iloc[-1],
                "entry_price": float(seg["price"].iloc[0]),
                "exit_price": float(seg["price"].iloc[-1]),
                "hold_bars": len(seg),
                "hold_days": (seg["dt"].iloc[-1] - seg["dt"].iloc[0]).total_seconds() / 86400,
                "ret": ret,
                "ret_bp": ret * 10000,
            }
        )
    return pd.DataFrame(segments)


def period_stats(daily: pd.DataFrame, sdt: str) -> dict:
    """Evaluate a subperiod from daily returns."""
    data = daily[daily.index >= pd.Timestamp(sdt)].copy()
    nav = (1 + data["ret"]).cumprod()
    drawdown = nav / nav.cummax() - 1
    years = (data.index[-1] - data.index[0]).days / 365.25
    annual = nav.iloc[-1] ** (1 / years) - 1 if years > 0 and nav.iloc[-1] > 0 else float("nan")
    max_drawdown = float(drawdown.min())
    return {
        "annual_return": float(annual),
        "final_nav": float(nav.iloc[-1]),
        "max_drawdown": max_drawdown,
        "calmar": annual / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
        "return_drawdown": (float(nav.iloc[-1]) - 1) / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
    }


def yearly_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate yearly performance from daily returns."""
    rows = []
    for year, dfg in daily.groupby(daily.index.year):
        stats = period_stats(dfg, str(dfg.index[0].date()))
        rows.append({"year": year, **stats})
    return pd.DataFrame(rows)


def selected_period_stats(daily: pd.DataFrame, full_stats: dict) -> pd.DataFrame:
    """Calculate common full / recent performance slices."""
    rows = [
        {
            "period": "full",
            "sdt": full_stats["start"],
            "annual_return": full_stats["annual_return"],
            "final_nav": full_stats["final_nav"],
            "max_drawdown": full_stats["max_drawdown"],
            "calmar": full_stats["calmar"],
            "return_drawdown": full_stats["return_drawdown"],
        }
    ]
    for period, sdt in [
        ("last_2y", "2024-05-20"),
        ("since_2025", "2025-01-01"),
        ("last_1y", "2025-05-20"),
    ]:
        stats = period_stats(daily, sdt)
        rows.append({"period": period, "sdt": sdt, **stats})
    return pd.DataFrame(rows)


def make_weight_frame(price_frame: pd.DataFrame, variant: TrendVariant) -> pd.DataFrame:
    """Build the standard weight table for one variant."""
    data = price_frame.copy()
    data["dt"] = pd.to_datetime(data["dt"]).dt.tz_localize(None)
    data = data.drop_duplicates("dt").sort_values("dt").set_index("dt")
    weight = make_momentum_weight(data["close"], variant)
    dfw = data.assign(weight=weight, price=data["close"]).reset_index()
    stat_edt = pd.Timestamp(STAT_EDT) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    dfw = dfw[(dfw["dt"] >= pd.Timestamp(STAT_SDT)) & (dfw["dt"] <= stat_edt)]
    return dfw[["dt", "symbol", "weight", "price"]]


def run_variant(
    price_frame: pd.DataFrame, variant: TrendVariant, output_dir: Path, save_details: bool = False
) -> dict:
    """Run and optionally persist one trend variant."""
    dfw = make_weight_frame(price_frame, variant)
    stats, daily, data = evaluate_weight_df(dfw)
    daily.attrs["symbol"] = str(dfw["symbol"].iloc[0])
    pairs = extract_pairs(data)
    stats.update(
        {
            "variant": variant.name,
            "lookback": variant.lookback,
            "threshold_bp": variant.threshold_bp,
            "legs": ",".join(f"{lookback}:{threshold_bp}" for lookback, threshold_bp in variant.legs),
            "risk_scale": variant.risk_scale,
            "pairs": len(pairs),
            "avg_hold_bars": float(pairs["hold_bars"].mean()) if not pairs.empty else 0.0,
            "median_hold_bars": float(pairs["hold_bars"].median()) if not pairs.empty else 0.0,
        }
    )

    if save_details:
        output_dir.mkdir(parents=True, exist_ok=True)
        dfw.to_csv(output_dir / f"{variant.name}_weights.csv", index=False, encoding="utf-8-sig")
        daily.to_csv(output_dir / f"{variant.name}_daily.csv", index_label="dt", encoding="utf-8-sig")
        pairs.to_csv(output_dir / f"{variant.name}_pairs.csv", index=False, encoding="utf-8-sig")
        yearly_stats(daily).to_csv(output_dir / f"{variant.name}_yearly.csv", index=False, encoding="utf-8-sig")
        selected_period_stats(daily, stats).to_csv(
            output_dir / f"{variant.name}_period_stats.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame([stats]).to_csv(output_dir / f"{variant.name}_summary.csv", index=False, encoding="utf-8-sig")
        make_equity_html(daily, variant, output_dir)

    return stats


def run_sweep(price_frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Run a compact lookback / threshold sweep."""
    rows = []
    lookbacks = [32, 48, 64, 96, 128, 160, 240, 320]
    thresholds = [0, 10, 20, 30, 40, 50, 70, 100, 150, 200]
    for lookback in lookbacks:
        for threshold_bp in thresholds:
            variant = TrendVariant(lookback=lookback, threshold_bp=threshold_bp, legs=(), risk_scale=1.0)
            stats = run_variant(price_frame, variant, output_dir, save_details=False)
            rows.append(stats)
            print(
                f"{variant.name} annual={stats['annual_return']:.2%} "
                f"max_dd={stats['max_drawdown']:.2%} "
                f"return/dd={stats['return_drawdown']:.2f} "
                f"pairs={stats['pairs']}"
            )
    df = pd.DataFrame(rows).sort_values(["annual_return", "return_drawdown"], ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "momentum_sweep_summary.csv", index=False, encoding="utf-8-sig")
    return df


def make_equity_html(daily: pd.DataFrame, variant: TrendVariant, output_dir: Path) -> None:
    """Write a compact SVG/HTML equity chart."""
    width, height = 1120, 680
    left, right, top, bottom = 72, 28, 76, 52
    plot_w = width - left - right
    nav_h = 390
    gap = 34
    dd_top = top + nav_h + gap
    dd_h = height - dd_top - bottom
    x_values = daily.index.map(pd.Timestamp.toordinal).astype(float)
    x_min, x_max = float(x_values.min()), float(x_values.max())
    nav_min = min(0.95, float(daily["nav"].min())) * 0.99
    nav_max = float(daily["nav"].max()) * 1.05
    dd_min = min(float(daily["drawdown"].min()) * 1.1, -0.02)

    def sx(dt) -> float:
        return left + (float(pd.Timestamp(dt).toordinal()) - x_min) / (x_max - x_min) * plot_w

    def sy_nav(value: float) -> float:
        return top + (nav_max - value) / (nav_max - nav_min) * nav_h

    def sy_dd(value: float) -> float:
        return dd_top + (0 - value) / (0 - dd_min) * dd_h

    nav_path = "M " + " L ".join(f"{sx(dt):.2f} {sy_nav(v):.2f}" for dt, v in daily["nav"].items())
    dd_path = "M " + " L ".join(f"{sx(dt):.2f} {sy_dd(v):.2f}" for dt, v in daily["drawdown"].items())
    symbol = str(daily.attrs.get("symbol", "000852.XSHG"))
    stats = pd.read_csv(output_dir / f"{variant.name}_summary.csv").iloc[0].to_dict()
    subtitle = (
        f"annual {stats['annual_return']:.2%}, max drawdown {stats['max_drawdown']:.2%}, "
        f"return/dd {stats['return_drawdown']:.2f}"
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fff"/>
<style>
text{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;fill:#1f2937}}
.title{{font-size:24px;font-weight:700}} .sub{{font-size:14px;fill:#64748b}}
.grid{{stroke:#e7ecf3;stroke-width:1}} .axis{{stroke:#cfd8e3;stroke-width:1}}
</style>
<text class="title" x="{left}" y="32">{symbol} Overnight Trend {variant.name}</text>
<text class="sub" x="{left}" y="56">{subtitle}</text>
<line class="grid" x1="{left}" x2="{width-right}" y1="{sy_nav(1):.2f}" y2="{sy_nav(1):.2f}"/>
<line class="axis" x1="{left}" x2="{width-right}" y1="{top+nav_h}" y2="{top+nav_h}"/>
<line class="axis" x1="{left}" x2="{width-right}" y1="{dd_top}" y2="{dd_top}"/>
<text class="sub" x="{left}" y="{top-14}">NAV</text>
<text class="sub" x="{left}" y="{dd_top-12}">Drawdown</text>
<path d="{nav_path}" fill="none" stroke="#c2410c" stroke-width="2.6"/>
<path d="{dd_path}" fill="none" stroke="#0f766e" stroke-width="2.2"/>
</svg>"""
    html = f"<!doctype html><meta charset='utf-8'><title>{variant.name}</title><body style='margin:0;background:#f8fafc'>{svg}</body>"
    (output_dir / f"{variant.name}_equity.svg").write_text(svg, encoding="utf-8")
    (output_dir / f"{variant.name}_equity.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["saved", "jq"], default="saved")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="JoinQuant symbol, e.g. 000905.XSHG or 159915.XSHE")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--threshold-bp", type=int, default=DEFAULT_THRESHOLD_BP)
    parser.add_argument(
        "--legs",
        default=DEFAULT_LEGS,
        help="comma separated lookback:threshold_bp legs; empty string runs the single lookback strategy",
    )
    parser.add_argument("--risk-scale", type=float, default=DEFAULT_RISK_SCALE)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    symbol = normalize_jq_symbol(args.symbol)
    output_dir = get_output_dir(symbol)
    price_frame = load_price_frame(args.source, symbol)
    print(f"price rows: {len(price_frame)} | {price_frame['dt'].iloc[0]} -> {price_frame['dt'].iloc[-1]}")

    variant = TrendVariant(
        lookback=args.lookback,
        threshold_bp=args.threshold_bp,
        legs=parse_legs(args.legs),
        risk_scale=args.risk_scale,
    )
    stats = run_variant(price_frame, variant, output_dir, save_details=True)
    print("\nselected:")
    print(pd.DataFrame([stats]).to_string(index=False))
    print(f"\noutputs: {output_dir}")

    if args.sweep:
        print("\nsweep:")
        sweep = run_sweep(price_frame, output_dir)
        cols = ["variant", "annual_return", "max_drawdown", "return_drawdown", "calmar", "final_nav", "pairs"]
        print(sweep[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
