"""Search overnight trend-following variants on 000852.XSHG with 15-minute bars.

This script expands the earlier overnight momentum research with several
trend signal families:

- time-series momentum with hold-through-neutral bands;
- EMA cross trends;
- Donchian breakouts;
- MACD-style EMA spread trends;
- EMA slope trends.

It evaluates both single-signal strategies and multi-signal average / vote
ensembles. The evaluation uses 15-minute close-to-close returns, applies the
previous bar weight to the current bar return, and charges turnover cost on
weight changes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
DEFAULT_OUTPUT_DIR = ROOT / "examples" / "results" / "overnight_trend_000852_signal_search"
BALANCED_ANCHORS = [
    "EMA_F8_S32_B0",
    "MACD_F8_S32_B0",
    "MOM_N16_B0",
    "MOM_N16_B10",
    "ESLOPE_N16_L8_B0",
    "MOM_N32_B30",
    "MOM_N48_B40",
    "MOM_N64_B0",
    "MOM_N64_B10",
    "MOM_N64_B20",
    "MOM_N64_B70",
    "MACD_F12_S26_B10",
    "EMA_F12_S32_B5",
    "ESLOPE_N16_L8_B20",
]
FIXED_CANDIDATES = [
    {
        "name": "BAL_AVG4_S90_c9bd249e",
        "mode": "avg",
        "risk_scale": 0.9,
        "vote_threshold": None,
        "signals": ("MOM_N16_B10", "MOM_N64_B10", "MACD_F12_S26_B10", "EMA_F12_S32_B5"),
    },
    {
        "name": "BAL_VOTE3_S100_V34_1987472d",
        "mode": "vote",
        "risk_scale": 1.0,
        "vote_threshold": 0.34,
        "signals": ("EMA_F8_S32_B0", "MOM_N16_B10", "MOM_N64_B10"),
    },
    {
        "name": "BAL_VOTE4_S85_V67_483076a0",
        "mode": "vote",
        "risk_scale": 0.85,
        "vote_threshold": 0.67,
        "signals": ("MOM_N16_B10", "MOM_N48_B40", "MOM_N64_B0", "MACD_F12_S26_B10"),
    },
]


@dataclass(frozen=True)
class SignalSpec:
    """A base trend signal."""

    name: str
    family: str
    weight: pd.Series


@dataclass(frozen=True)
class Candidate:
    """A candidate strategy composed from one or more base signals."""

    name: str
    mode: str
    signal_names: tuple[str, ...]
    risk_scale: float
    vote_threshold: float | None
    weight: pd.Series


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
    return f"{symbol}.XSHG"


def get_output_dir(symbol: str) -> Path:
    """Return output directory for one symbol."""
    if symbol == DEFAULT_SYMBOL:
        return DEFAULT_OUTPUT_DIR
    return ROOT / "examples" / "results" / f"overnight_trend_{symbol.replace('.', '_')}_signal_search"


def get_cache_file(output_dir: Path, symbol: str) -> Path:
    """Return cache file path for one symbol."""
    return output_dir / f"{symbol.replace('.', '_')}_15m_ohlc.csv"


def load_ohlc(source: str, symbol: str = DEFAULT_SYMBOL, output_dir: Path | None = None) -> pd.DataFrame:
    """Load or fetch 15-minute OHLCV bars."""
    output_dir = output_dir or get_output_dir(symbol)
    cache_file = get_cache_file(output_dir, symbol)
    output_dir.mkdir(parents=True, exist_ok=True)
    if source == "cache" and cache_file.exists():
        data = pd.read_csv(cache_file, parse_dates=["dt"])
        return data.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)

    bars = read_jq_sdk_bars(freq="15m", czsc_freq=Freq.F15, sdt=DATA_SDT, edt=FETCH_EDT, symbol=symbol)
    rows = [
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
    data = pd.DataFrame(rows).drop_duplicates("dt").sort_values("dt")
    stat_edt = pd.Timestamp(STAT_EDT) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    data = data[data["dt"] <= stat_edt].reset_index(drop=True)
    data.to_csv(cache_file, index=False, encoding="utf-8-sig")
    return data


def hold_score(score: pd.Series, threshold: float) -> pd.Series:
    """Convert a continuous score to long / short weights with neutral holding."""
    weight = pd.Series(0.0, index=score.index)
    weight[score > threshold] = 1.0
    weight[score < -threshold] = -1.0
    return weight.replace(0.0, np.nan).ffill().fillna(0.0)


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def make_base_signals(data: pd.DataFrame) -> list[SignalSpec]:
    """Build base trend signal weights."""
    close = data["close"]
    high = data["high"]
    low = data["low"]
    signals: list[SignalSpec] = []

    for n in [16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 160]:
        momentum = close / close.shift(n) - 1
        for bp in [0, 10, 20, 30, 40, 50, 70, 100, 150]:
            signals.append(SignalSpec(f"MOM_N{n}_B{bp}", "momentum", hold_score(momentum, bp / 10000)))

    for fast in [8, 12, 16, 24, 32]:
        for slow in [32, 48, 64, 96, 128, 160]:
            if fast >= slow:
                continue
            spread = ema(close, fast) / ema(close, slow) - 1
            for bp in [0, 5, 10, 20, 30, 50]:
                signals.append(SignalSpec(f"EMA_F{fast}_S{slow}_B{bp}", "ema_cross", hold_score(spread, bp / 10000)))

    for n in [16, 24, 32, 48, 64, 80, 96, 128]:
        upper = high.shift(1).rolling(n, min_periods=n).max()
        lower = low.shift(1).rolling(n, min_periods=n).min()
        weight = pd.Series(0.0, index=data.index)
        weight[close > upper] = 1.0
        weight[close < lower] = -1.0
        signals.append(SignalSpec(f"DON_N{n}", "donchian", weight.replace(0.0, np.nan).ffill().fillna(0.0)))

    for fast, slow in [(8, 32), (12, 26), (16, 48), (24, 64), (32, 96)]:
        spread = (ema(close, fast) - ema(close, slow)) / close
        for bp in [0, 5, 10, 20, 30]:
            signals.append(SignalSpec(f"MACD_F{fast}_S{slow}_B{bp}", "macd_spread", hold_score(spread, bp / 10000)))

    for n in [16, 24, 32, 48, 64, 96, 128]:
        base = ema(close, n)
        for lag in [8, 16, 32]:
            slope = base / base.shift(lag) - 1
            for bp in [0, 5, 10, 20, 30]:
                signals.append(SignalSpec(f"ESLOPE_N{n}_L{lag}_B{bp}", "ema_slope", hold_score(slope, bp / 10000)))

    return signals


def make_eval_context(data: pd.DataFrame) -> dict:
    """Precompute arrays shared by all candidate evaluations."""
    stat_edt = pd.Timestamp(STAT_EDT) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    mask = (data["dt"] >= pd.Timestamp(STAT_SDT)) & (data["dt"] <= stat_edt)
    stat = data.loc[mask].copy().reset_index(drop=False)
    price = stat["close"].to_numpy(dtype=float)
    price_ret = np.zeros(len(stat), dtype=float)
    price_ret[1:] = price[1:] / price[:-1] - 1
    dates = stat["dt"].dt.normalize()
    day_index = pd.Index(dates.drop_duplicates())
    day_codes = day_index.get_indexer(dates)
    years = (day_index[-1] - day_index[0]).days / 365.25
    return {
        "stat": stat,
        "original_index": stat["index"].to_numpy(),
        "price_ret": price_ret,
        "day_codes": day_codes,
        "day_index": day_index,
        "years": years,
    }


def evaluate_weight(weight: pd.Series, ctx: dict, fee_rate: float = FEE_RATE) -> tuple[dict, pd.DataFrame]:
    """Evaluate one candidate weight series."""
    w = weight.iloc[ctx["original_index"]].to_numpy(dtype=float)
    prev_w = np.empty_like(w)
    prev_w[0] = 0.0
    prev_w[1:] = w[:-1]
    turnover = np.empty_like(w)
    turnover[0] = abs(w[0])
    turnover[1:] = np.abs(w[1:] - w[:-1])
    ret = prev_w * ctx["price_ret"] - turnover * fee_rate
    daily_ret = np.bincount(ctx["day_codes"], weights=ret, minlength=len(ctx["day_index"]))

    daily = pd.DataFrame({"ret": daily_ret}, index=ctx["day_index"])
    daily["nav"] = (1 + daily["ret"]).cumprod()
    daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1

    final_nav = float(daily["nav"].iloc[-1])
    annual_return = final_nav ** (1 / ctx["years"]) - 1 if ctx["years"] > 0 and final_nav > 0 else float("nan")
    max_drawdown = float(daily["drawdown"].min())
    cumulative_return = final_nav - 1
    stats = {
        "annual_return": annual_return,
        "cumulative_return": cumulative_return,
        "final_nav": final_nav,
        "max_drawdown": max_drawdown,
        "calmar": annual_return / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
        "return_drawdown": cumulative_return / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
        "daily_win_rate": float((daily["ret"] > 0).mean()),
        "turnover": float(turnover.sum()),
        "active_frac": float(np.mean(w != 0)),
        "start": str(ctx["day_index"][0].date()),
        "end": str(ctx["day_index"][-1].date()),
    }
    return stats, daily


def period_stats(daily: pd.DataFrame, sdt: str) -> dict:
    """Evaluate one subperiod from daily returns."""
    data = daily[daily.index >= pd.Timestamp(sdt)]
    nav = (1 + data["ret"]).cumprod()
    drawdown = nav / nav.cummax() - 1
    years = (data.index[-1] - data.index[0]).days / 365.25
    final_nav = float(nav.iloc[-1])
    annual = final_nav ** (1 / years) - 1 if years > 0 and final_nav > 0 else float("nan")
    max_drawdown = float(drawdown.min())
    return {
        "annual_return": annual,
        "final_nav": final_nav,
        "max_drawdown": max_drawdown,
        "calmar": annual / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
        "return_drawdown": (final_nav - 1) / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
    }


def candidate_stats(candidate: Candidate, ctx: dict) -> tuple[dict, pd.DataFrame]:
    """Evaluate and annotate candidate stats."""
    stats, daily = evaluate_weight(candidate.weight, ctx)
    stats.update(
        {
            "name": candidate.name,
            "mode": candidate.mode,
            "risk_scale": candidate.risk_scale,
            "vote_threshold": candidate.vote_threshold,
            "signal_count": len(candidate.signal_names),
            "signals": ";".join(candidate.signal_names),
        }
    )
    for label, sdt in [("last_2y", "2024-05-20"), ("since_2025", "2025-01-01"), ("last_1y", "2025-05-20")]:
        ps = period_stats(daily, sdt)
        stats[f"{label}_annual_return"] = ps["annual_return"]
        stats[f"{label}_max_drawdown"] = ps["max_drawdown"]
        stats[f"{label}_calmar"] = ps["calmar"]
        stats[f"{label}_return_drawdown"] = ps["return_drawdown"]
    stats["qualified_full"] = bool(stats["annual_return"] >= 0.20 and stats["return_drawdown"] >= 3)
    stats["qualified_lowdd"] = bool(stats["qualified_full"] and stats["max_drawdown"] >= -0.10)
    stats["qualified_recent"] = bool(stats["since_2025_annual_return"] >= 0.20 and stats["since_2025_return_drawdown"] >= 3)
    stats["score"] = (
        stats["annual_return"] * 2.0
        + stats["calmar"] * 0.25
        + stats["since_2025_annual_return"] * 1.2
        - abs(stats["max_drawdown"]) * 0.8
    )
    return stats, daily


def make_candidate_name(mode: str, names: tuple[str, ...], risk_scale: float, vote_threshold: float | None) -> str:
    """Build a compact candidate name."""
    digest = hashlib.sha1("|".join(names).encode("utf-8")).hexdigest()[:8]
    vote = "" if vote_threshold is None else f"_V{int(round(vote_threshold * 100))}"
    return f"{mode.upper()}{len(names)}_S{int(round(risk_scale * 100))}{vote}_{digest}"


def combine_weights(
    specs: tuple[SignalSpec, ...], risk_scale: float, mode: str, vote_threshold: float | None
) -> pd.Series:
    """Combine base signal weights."""
    mat = pd.concat([x.weight for x in specs], axis=1)
    avg = mat.mean(axis=1)
    if mode == "avg":
        return avg * risk_scale

    threshold = 1 / len(specs) if vote_threshold is None else vote_threshold
    voted = pd.Series(0.0, index=avg.index)
    voted[avg >= threshold] = 1.0
    voted[avg <= -threshold] = -1.0
    return voted * risk_scale


def run_search(data: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, dict[str, Candidate]]:
    """Search single and multi-signal candidates."""
    ctx = make_eval_context(data)
    base_signals = make_base_signals(data)
    rows: list[dict] = []
    candidates: dict[str, Candidate] = {}

    for spec in base_signals:
        for scale in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            candidate = Candidate(
                name=f"SINGLE_{spec.name}_S{int(round(scale * 100))}",
                mode="single",
                signal_names=(spec.name,),
                risk_scale=scale,
                vote_threshold=None,
                weight=spec.weight * scale,
            )
            stats, _ = candidate_stats(candidate, ctx)
            rows.append(stats)
            candidates[candidate.name] = candidate

    singles = pd.DataFrame(rows)
    eligible = singles[(singles["annual_return"] >= 0.16) & (singles["return_drawdown"] >= 4)].copy()
    eligible = eligible.sort_values(["score", "annual_return"], ascending=False)
    selected_names = list(dict.fromkeys(x.split(";")[0] for x in eligible["signals"].head(top_n)))
    spec_map = {x.name: x for x in base_signals}
    selected_specs = [spec_map[x] for x in selected_names if x in spec_map]

    for size in [2, 3, 4, 5]:
        for combo in itertools.combinations(selected_specs, size):
            signal_names = tuple(x.name for x in combo)
            for mode, vote_thresholds in [("avg", [None]), ("vote", [0.34, 0.50, 0.67])]:
                for vote_threshold in vote_thresholds:
                    for scale in [0.6, 0.7, 0.8, 0.85, 0.9, 1.0]:
                        weight = combine_weights(combo, scale, mode, vote_threshold)
                        name = make_candidate_name(mode, signal_names, scale, vote_threshold)
                        candidate = Candidate(name, mode, signal_names, scale, vote_threshold, weight)
                        stats, _ = candidate_stats(candidate, ctx)
                        rows.append(stats)
                        candidates[name] = candidate

    results = pd.DataFrame(rows)
    results = results.sort_values(["qualified_lowdd", "score", "annual_return"], ascending=False)
    return results, candidates


def run_balanced_search(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Candidate]]:
    """Search a smaller anchor pool that balances full-period and recent behavior."""
    ctx = make_eval_context(data)
    base_signals = make_base_signals(data)
    spec_map = {x.name: x for x in base_signals}
    selected_specs = [spec_map[x] for x in BALANCED_ANCHORS if x in spec_map]
    rows: list[dict] = []
    candidates: dict[str, Candidate] = {}

    for size in [2, 3, 4]:
        for combo in itertools.combinations(selected_specs, size):
            signal_names = tuple(x.name for x in combo)
            for mode, vote_thresholds in [("avg", [None]), ("vote", [0.34, 0.50, 0.67])]:
                for vote_threshold in vote_thresholds:
                    for scale in [0.6, 0.7, 0.8, 0.85, 0.9, 1.0]:
                        weight = combine_weights(combo, scale, mode, vote_threshold)
                        name = make_candidate_name(f"bal_{mode}", signal_names, scale, vote_threshold)
                        candidate = Candidate(name, mode, signal_names, scale, vote_threshold, weight)
                        stats, _ = candidate_stats(candidate, ctx)
                        rows.append(stats)
                        candidates[name] = candidate

    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["qualified_recent", "qualified_lowdd", "score", "since_2025_annual_return"], ascending=False
    )
    return results, candidates


def build_fixed_candidates(data: pd.DataFrame) -> dict[str, Candidate]:
    """Build the selected A/B/C candidates on any symbol."""
    spec_map = {x.name: x for x in make_base_signals(data)}
    candidates = {}
    for item in FIXED_CANDIDATES:
        combo = tuple(spec_map[x] for x in item["signals"])
        weight = combine_weights(combo, item["risk_scale"], item["mode"], item["vote_threshold"])
        candidate = Candidate(
            name=item["name"],
            mode=item["mode"],
            signal_names=item["signals"],
            risk_scale=item["risk_scale"],
            vote_threshold=item["vote_threshold"],
            weight=weight,
        )
        candidates[candidate.name] = candidate
    return candidates


def select_candidates_for_details(results: pd.DataFrame, save_top: int) -> list[str]:
    """Select candidates from several useful lenses for detailed persistence."""
    selected_names: list[str] = []
    lenses = [
        results[results["qualified_lowdd"]].sort_values(["score", "annual_return"], ascending=False),
        results[(results["qualified_full"]) & (results["since_2025_annual_return"] >= 0.20)].sort_values(
            ["max_drawdown", "annual_return"], ascending=[False, False]
        ),
        results[
            (results["qualified_full"]) & (results["since_2025_annual_return"] >= 0.15) & (results["max_drawdown"] >= -0.12)
        ].sort_values(["score", "annual_return"], ascending=False),
    ]
    for lens in lenses:
        selected_names.extend(lens.head(save_top)["name"].tolist())
    return list(dict.fromkeys(selected_names))[: max(save_top, 12)]


def write_equity_svg(daily: pd.DataFrame, stats: dict, output_file: Path) -> None:
    """Write a simple equity / drawdown SVG."""
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
<text class="title" x="{left}" y="32">{stats.get('symbol', '000852.XSHG')} Signal Search {stats['name']}</text>
<text class="sub" x="{left}" y="56">{subtitle}</text>
<line class="grid" x1="{left}" x2="{width-right}" y1="{sy_nav(1):.2f}" y2="{sy_nav(1):.2f}"/>
<line class="axis" x1="{left}" x2="{width-right}" y1="{top+nav_h}" y2="{top+nav_h}"/>
<line class="axis" x1="{left}" x2="{width-right}" y1="{dd_top}" y2="{dd_top}"/>
<text class="sub" x="{left}" y="{top-14}">NAV</text>
<text class="sub" x="{left}" y="{dd_top-12}">Drawdown</text>
<path d="{nav_path}" fill="none" stroke="#b45309" stroke-width="2.6"/>
<path d="{dd_path}" fill="none" stroke="#2563eb" stroke-width="2.2"/>
</svg>"""
    output_file.write_text(svg, encoding="utf-8")


def save_candidate_details(data: pd.DataFrame, candidate: Candidate, stats: dict, ctx: dict, output_dir: Path) -> None:
    """Persist detailed files for one selected candidate."""
    detail_dir = output_dir / "selected"
    detail_dir.mkdir(parents=True, exist_ok=True)
    stats, daily = candidate_stats(candidate, ctx)
    stats["symbol"] = str(data["symbol"].iloc[0])
    stat = ctx["stat"].copy()
    stat["weight"] = candidate.weight.iloc[ctx["original_index"]].to_numpy(dtype=float)
    stat = stat.rename(columns={"close": "price"})
    weights = stat[["dt", "symbol", "weight", "price"]].copy()
    weights.to_csv(detail_dir / f"{candidate.name}_weights.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(detail_dir / f"{candidate.name}_daily.csv", index_label="dt", encoding="utf-8-sig")
    pd.DataFrame([stats]).to_csv(detail_dir / f"{candidate.name}_summary.csv", index=False, encoding="utf-8-sig")
    period_rows = [{"period": "full", "sdt": stats["start"], **{k: stats[k] for k in [
        "annual_return", "final_nav", "max_drawdown", "calmar", "return_drawdown"
    ]}}]
    for period, sdt in [("last_2y", "2024-05-20"), ("since_2025", "2025-01-01"), ("last_1y", "2025-05-20")]:
        period_rows.append({"period": period, "sdt": sdt, **period_stats(daily, sdt)})
    pd.DataFrame(period_rows).to_csv(detail_dir / f"{candidate.name}_period_stats.csv", index=False, encoding="utf-8-sig")
    write_equity_svg(daily, stats, detail_dir / f"{candidate.name}_equity.svg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["cache", "jq"], default="cache")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="JoinQuant symbol, e.g. 000905.XSHG or 159915.XSHE")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--save-top", type=int, default=5)
    parser.add_argument("--skip-balanced", action="store_true")
    parser.add_argument("--selected-only", action="store_true", help="Only backtest the fixed A/B/C candidates")
    args = parser.parse_args()

    symbol = normalize_jq_symbol(args.symbol)
    output_dir = get_output_dir(symbol)
    data = load_ohlc(args.source, symbol=symbol, output_dir=output_dir)
    print(f"data rows: {len(data)} | {data['dt'].iloc[0]} -> {data['dt'].iloc[-1]}")
    ctx = make_eval_context(data)

    if args.selected_only:
        candidates = build_fixed_candidates(data)
        rows = []
        for candidate in candidates.values():
            stats, _ = candidate_stats(candidate, ctx)
            stats["symbol"] = symbol
            rows.append(stats)
            save_candidate_details(data, candidate, stats, ctx, output_dir)
        summary = pd.DataFrame(rows).sort_values(["annual_return", "return_drawdown"], ascending=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_dir / "selected_candidates_summary.csv", index=False, encoding="utf-8-sig")
        cols = [
            "symbol",
            "name",
            "annual_return",
            "max_drawdown",
            "return_drawdown",
            "calmar",
            "since_2025_annual_return",
            "since_2025_max_drawdown",
            "last_1y_annual_return",
            "last_1y_max_drawdown",
            "signals",
        ]
        print(summary[cols].to_string(index=False))
        print(f"\noutputs: {output_dir}")
        return

    results, candidates = run_search(data, top_n=args.top_n)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "all_candidates.csv", index=False, encoding="utf-8-sig")

    qualified = results[results["qualified_full"]].copy()
    qualified.to_csv(output_dir / "qualified_candidates.csv", index=False, encoding="utf-8-sig")
    lowdd = results[results["qualified_lowdd"]].copy()
    lowdd.to_csv(output_dir / "qualified_lowdd_candidates.csv", index=False, encoding="utf-8-sig")

    selected_names = select_candidates_for_details(results, args.save_top)
    for name in selected_names:
        row = results.loc[results["name"].eq(name)].iloc[0].to_dict()
        save_candidate_details(data, candidates[name], row, ctx, output_dir)

    balanced = pd.DataFrame()
    if not args.skip_balanced:
        balanced, balanced_candidates = run_balanced_search(data)
        balanced.to_csv(output_dir / "balanced_candidates.csv", index=False, encoding="utf-8-sig")
        for name in select_candidates_for_details(balanced, args.save_top):
            row = balanced.loc[balanced["name"].eq(name)].iloc[0].to_dict()
            save_candidate_details(data, balanced_candidates[name], row, ctx, output_dir)

    cols = [
        "name",
        "mode",
        "signal_count",
        "annual_return",
        "max_drawdown",
        "return_drawdown",
        "calmar",
        "since_2025_annual_return",
        "since_2025_max_drawdown",
        "last_1y_annual_return",
        "last_1y_max_drawdown",
        "risk_scale",
        "vote_threshold",
        "signals",
    ]
    print("\nTop low-drawdown qualified candidates:")
    print(lowdd[cols].head(20).to_string(index=False))
    if not balanced.empty:
        print("\nTop balanced candidates with full qualification and since-2025 annual >= 20%:")
        q = balanced[(balanced["qualified_full"]) & (balanced["since_2025_annual_return"] >= 0.20)]
        q = q.sort_values(["max_drawdown", "annual_return"], ascending=[False, False])
        print(q[cols].head(20).to_string(index=False))
    print(f"\noutputs: {output_dir}")


if __name__ == "__main__":
    main()
