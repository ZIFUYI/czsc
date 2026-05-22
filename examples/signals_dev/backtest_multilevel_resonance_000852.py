"""Multi-frequency resonance strategy research on 000852.XSHG.

The strategy uses native CZSC signal functions only:

- 15-minute short momentum: ``bar_bpm_V230227`` N16/T10
- 15-minute medium momentum: ``bar_bpm_V230227`` N64/T10
- 60-minute trend: ``tas_ma_base_V221101`` SMA#5
- Daily trend regime: ``tas_ma_joint_V260520``

Each signal is mapped to ``{-1, 0, 1}``; the final weight is the average signal
score multiplied by ``0.9``. This keeps exposure high when periods resonate and
automatically cuts exposure when the periods disagree.

Run:
    uv run --no-sync python examples/signals_dev/backtest_multilevel_resonance_000852.py

Use JQData instead of the local cache:
    uv run --no-sync python examples/signals_dev/backtest_multilevel_resonance_000852.py --source jq
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

from czsc import Freq, format_standard_kline, generate_czsc_signals  # noqa: E402
from examples.signals_dev.test_zdy_macd_bc_000852 import read_jq_sdk_bars  # noqa: E402

SYMBOL = "000852.XSHG"
DATA_SDT = "20180101"
STAT_SDT = "20200101"
STAT_EDT = "20260520"
FETCH_EDT = "20260521"
FEE_RATE = 0.0002
RISK_SCALE = 0.9
DEFAULT_OUTPUT_DIR = ROOT / "examples" / "results" / "multilevel_resonance_000852"
LOCAL_CACHE_FILE = ROOT / "examples" / "results" / "overnight_trend_000852_signal_search" / "000852_XSHG_15m_ohlc.csv"

SIGNALS_CONFIG = [
    {"name": "bar_bpm_V230227", "freq": "15分钟", "di": 1, "n": 16, "th": 10},
    {"name": "bar_bpm_V230227", "freq": "15分钟", "di": 1, "n": 64, "th": 10},
    {"name": "tas_ma_base_V221101", "freq": "60分钟", "di": 1, "timeperiod": 5, "ma_type": "SMA"},
    {"name": "tas_ma_joint_V260520", "freq": "日线", "di": 1, "ma_seq": "5#10#20#40#60#120#250"},
]

DAILY_MA_JOINT_SIGNAL = "日线_D1SMA5#10#20#40#60#120#250_均线粘合发散V260520"

SIGNAL_COLUMNS = [
    "15分钟_D1N16T10_绝对动量V230227",
    "15分钟_D1N64T10_绝对动量V230227",
    "60分钟_D1SMA#5_分类V221101",
    DAILY_MA_JOINT_SIGNAL,
]

LONG_PREFIXES = ("多头", "看多", "做多", "强势", "超强")
SHORT_PREFIXES = ("空头", "看空", "做空", "弱势", "超弱")
SIGNAL_ALIASES = {
    DAILY_MA_JOINT_SIGNAL.replace("粘合发散", "联合发散"): DAILY_MA_JOINT_SIGNAL,
}


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
    if symbol in known:
        return known[symbol]
    return f"{symbol}.XSHG"


def get_output_dir(symbol: str) -> Path:
    """Return output directory for one symbol."""
    if symbol == SYMBOL:
        return DEFAULT_OUTPUT_DIR
    return ROOT / "examples" / "results" / f"multilevel_resonance_{symbol.replace('.', '_')}"


def load_ohlc(source: str, symbol: str, output_dir: Path) -> pd.DataFrame:
    """Load 15-minute OHLCV data from local cache or JQData."""
    if source == "cache":
        cache_file = output_dir / f"{symbol.replace('.', '_')}_15m_ohlc.csv"
        source_file = cache_file if cache_file.exists() else LOCAL_CACHE_FILE if symbol == SYMBOL else cache_file
        if not source_file.exists():
            raise FileNotFoundError(f"Local 15m cache not found: {source_file}")
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


def signal_to_score(values: pd.Series) -> pd.Series:
    """Map native signal values to directional scores."""
    text = values.astype(str)
    score = pd.Series(0.0, index=values.index)
    score[text.str.startswith(LONG_PREFIXES)] = 1.0
    score[text.str.startswith(SHORT_PREFIXES)] = -1.0
    return score


def build_weights(sig: pd.DataFrame) -> pd.DataFrame:
    """Build the final multi-frequency resonance weight table."""
    sig = sig.rename(columns=SIGNAL_ALIASES)
    missing = [col for col in SIGNAL_COLUMNS if col not in sig.columns]
    if missing:
        raise KeyError(f"Missing expected signal columns: {missing}")

    out = sig[["dt", "symbol", "close", *SIGNAL_COLUMNS]].copy()
    scores = pd.DataFrame({col: signal_to_score(out[col]) for col in SIGNAL_COLUMNS}, index=out.index)
    out["score_sum"] = scores.sum(axis=1)
    out["score_avg"] = scores.mean(axis=1)
    out["weight"] = out["score_avg"] * RISK_SCALE
    out["price"] = out["close"].astype(float)
    return out[["dt", "symbol", "weight", "price", "score_sum", "score_avg", *SIGNAL_COLUMNS]]


def evaluate_weights(weights: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Evaluate a timestamped weight table with turnover cost."""
    data = weights.sort_values("dt").copy()
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
        "strategy": "MLR_15mBPM16_15mBPM64_60mSMA5_DailyMAJoint_S90",
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
        "signals": " | ".join(SIGNAL_COLUMNS),
    }
    return stats, daily


def period_stats(daily: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Calculate common full and subperiod performance slices."""
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
    for period, sdt in [
        ("2020_2022", "2020-01-01"),
        ("2023_2024", "2023-01-01"),
        ("last_2y", "2024-05-20"),
        ("since_2025", "2025-01-01"),
        ("last_1y", "2025-05-20"),
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


def write_equity_svg(daily: pd.DataFrame, stats: dict, output_dir: Path) -> None:
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
    subtitle = (
        f"annual {stats['annual_return']:.2%}, max drawdown {stats['max_drawdown']:.2%}, calmar {stats['calmar']:.2f}"
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fff"/>
<style>
text{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;fill:#172033}}
.title{{font-size:24px;font-weight:700}} .sub{{font-size:14px;fill:#526173}}
.grid{{stroke:#e7ecf3;stroke-width:1}} .axis{{stroke:#cfd8e3;stroke-width:1}}
</style>
<text class="title" x="{left}" y="32">{stats["symbol"]} Multi-Level Resonance</text>
<text class="sub" x="{left}" y="56">{subtitle}</text>
<line class="grid" x1="{left}" x2="{width - right}" y1="{sy_nav(1):.2f}" y2="{sy_nav(1):.2f}"/>
<line class="axis" x1="{left}" x2="{width - right}" y1="{top + nav_h}" y2="{top + nav_h}"/>
<line class="axis" x1="{left}" x2="{width - right}" y1="{dd_top}" y2="{dd_top}"/>
<text class="sub" x="{left}" y="{top - 14}">NAV</text>
<text class="sub" x="{left}" y="{dd_top - 12}">Drawdown</text>
<path d="{nav_path}" fill="none" stroke="#0f766e" stroke-width="2.6"/>
<path d="{dd_path}" fill="none" stroke="#b45309" stroke-width="2.2"/>
</svg>"""
    html = f"<!doctype html><meta charset='utf-8'><title>{stats['strategy']}</title><body style='margin:0'>{svg}</body>"
    (output_dir / "equity.svg").write_text(svg, encoding="utf-8")
    (output_dir / "equity.html").write_text(html, encoding="utf-8")


def save_outputs(weights: pd.DataFrame, daily: pd.DataFrame, stats: dict, output_dir: Path) -> None:
    """Persist strategy research artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    weights.to_csv(output_dir / "weights.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output_dir / "daily.csv", index_label="dt", encoding="utf-8-sig")
    pd.DataFrame([stats]).to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    period_stats(daily, stats).to_csv(output_dir / "period_stats.csv", index=False, encoding="utf-8-sig")
    counts = []
    for col in SIGNAL_COLUMNS:
        for value, count in weights[col].value_counts(dropna=False).items():
            counts.append({"signal": col, "value": value, "count": int(count)})
    pd.DataFrame(counts).to_csv(output_dir / "signal_counts.csv", index=False, encoding="utf-8-sig")
    write_equity_svg(daily, stats, output_dir)


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
    weights = build_weights(sig)
    stats, daily = evaluate_weights(weights)
    save_outputs(weights, daily, stats, output_dir)

    print(pd.DataFrame([stats]).to_string(index=False))
    print("\nperiod stats:")
    print(period_stats(daily, stats).to_string(index=False))
    print(f"\noutputs: {output_dir}")


if __name__ == "__main__":
    main()
