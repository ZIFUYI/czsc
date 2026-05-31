"""Optimized pure daily CTA variants for 000905.XSHG and 159915.XSHE.

These variants keep the same daily-only execution contract as
``backtest_daily_cta_000852.py``:

- all signals use daily OHLCV only;
- the current close generates the target weight;
- the next trading day's close-to-close return uses the previous day's weight.

Run:
    .venv/bin/python examples/signals_dev/backtest_daily_cta_cross_symbol_optimized.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "examples" / "results" / "daily_holding_cross_symbol"
OUTPUT_DIR = DATA_DIR / "optimized_fixed"
STAT_SDT = "20200101"
STAT_EDT = "20260529"
FEE_RATE = 0.0002


@dataclass(frozen=True)
class StrategyConfig:
    """One optimized daily CTA variant."""

    symbol: str
    data_file: Path
    legs: tuple[str, ...]
    vote: bool
    vol_window: int
    target_vol: float
    leverage_cap: float

    @property
    def name(self) -> str:
        mode = "VOTE" if self.vote else "AVG"
        return f"DailyCTA_{self.symbol.replace('.', '_')}_{mode}{len(self.legs)}_VT{int(self.target_vol * 100)}_C{self.leverage_cap:g}"

    @property
    def out_dir(self) -> Path:
        return OUTPUT_DIR / self.symbol.replace(".", "_")


CONFIGS = [
    StrategyConfig(
        symbol="000905.XSHG",
        data_file=DATA_DIR / "000905_XSHG_daily_20100101_20260529.csv",
        legs=("MA_F_N4_B80", "EMA_H_F12_S200_B300", "MA_F_N5_B100", "MA_R_N30_B500"),
        vote=True,
        vol_window=15,
        target_vol=0.20,
        leverage_cap=3.0,
    ),
    StrategyConfig(
        symbol="159915.XSHE",
        data_file=DATA_DIR / "159915_XSHE_daily_20100101_20260529.csv",
        legs=("MA_H_N320_B150", "EMA_F_F5_S24_B100"),
        vote=True,
        vol_window=40,
        target_vol=0.15,
        leverage_cap=1.2,
    ),
]


def hold_score(score: pd.Series, threshold: float) -> pd.Series:
    """Convert a score to long / short state and hold through neutral zones."""
    weight = pd.Series(0.0, index=score.index)
    weight[score > threshold] = 1.0
    weight[score < -threshold] = -1.0
    return weight.replace(0.0, np.nan).ffill().fillna(0.0)


def flat_score(score: pd.Series, threshold: float) -> pd.Series:
    """Convert a score to long / short / flat state."""
    weight = pd.Series(0.0, index=score.index)
    weight[score > threshold] = 1.0
    weight[score < -threshold] = -1.0
    return weight


def parse_leg(leg: str) -> tuple[str, str, dict[str, int]]:
    """Parse a compact leg name such as ``EMA_H_F12_S200_B300``."""
    parts = leg.split("_")
    family = parts[0]
    mode = parts[1]
    params: dict[str, int] = {}
    for item in parts[2:]:
        params[item[0]] = int(item[1:])
    return family, mode, params


def build_leg(data: pd.DataFrame, leg: str) -> pd.Series:
    """Build one named directional leg."""
    family, mode, params = parse_leg(leg)
    close = data["close"].astype(float)
    threshold = params.get("B", 0) / 10000

    if family == "MA":
        ma = close.rolling(params["N"], min_periods=params["N"]).mean()
        score = close / ma - 1
    elif family == "EMA":
        fast = close.ewm(span=params["F"], adjust=False, min_periods=params["F"]).mean()
        slow = close.ewm(span=params["S"], adjust=False, min_periods=params["S"]).mean()
        score = fast / slow - 1
    else:
        raise ValueError(f"Unsupported leg family: {family}")

    if mode == "H":
        return hold_score(score, threshold)
    if mode == "F":
        return flat_score(score, threshold)
    if mode == "R":
        return -hold_score(score, threshold)
    raise ValueError(f"Unsupported leg mode: {mode}")


def load_daily_bars(config: StrategyConfig) -> pd.DataFrame:
    """Load daily OHLCV data for one symbol."""
    if not config.data_file.exists():
        raise FileNotFoundError(f"Daily data file not found: {config.data_file}")
    data = pd.read_csv(config.data_file, parse_dates=["dt"])
    data["dt"] = pd.to_datetime(data["dt"]).dt.tz_localize(None)
    data["symbol"] = config.symbol
    return data.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)


def build_weight_frame(data: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Build daily target weights from optimized legs."""
    legs = pd.DataFrame({leg: build_leg(data, leg) for leg in config.legs}, index=data.index).fillna(0.0)
    score = legs.mean(axis=1)
    if config.vote:
        score = np.sign(score)

    daily_ret = data["close"].astype(float).pct_change().fillna(0.0)
    realized_vol = daily_ret.rolling(config.vol_window, min_periods=config.vol_window).std() * np.sqrt(252)
    leverage = (config.target_vol / realized_vol).replace([np.inf, -np.inf], np.nan)
    leverage = leverage.clip(upper=config.leverage_cap).fillna(0.0)

    out = data[["dt", "symbol", "close"]].copy()
    out = pd.concat([out, legs], axis=1)
    out["score"] = score
    out["leverage"] = leverage
    out["weight"] = out["score"] * out["leverage"]
    out["price"] = out["close"].astype(float)

    stat_edt = pd.Timestamp(STAT_EDT) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    mask = (out["dt"] >= pd.Timestamp(STAT_SDT)) & (out["dt"] <= stat_edt)
    return out.loc[mask].reset_index(drop=True)


def evaluate(weights: pd.DataFrame, config: StrategyConfig) -> tuple[dict, pd.DataFrame]:
    """Evaluate close-to-close daily returns with previous-day weight."""
    daily = weights[["dt", "symbol", "price", "weight"]].copy()
    daily["price_ret"] = daily.groupby("symbol")["price"].pct_change().fillna(0.0)
    daily["prev_weight"] = daily.groupby("symbol")["weight"].shift(1).fillna(0.0)
    daily["turnover"] = daily.groupby("symbol")["weight"].diff().abs().fillna(daily["weight"].abs())
    daily["ret"] = daily["prev_weight"] * daily["price_ret"] - daily["turnover"] * FEE_RATE
    daily["nav"] = (1 + daily["ret"]).cumprod()
    daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1

    years = (daily["dt"].iloc[-1] - daily["dt"].iloc[0]).days / 365.25
    final_nav = float(daily["nav"].iloc[-1])
    annual_return = final_nav ** (1 / years) - 1 if final_nav > 0 else float("nan")
    max_drawdown = float(daily["drawdown"].min())
    stats = {
        "symbol": config.symbol,
        "strategy": config.name,
        "start": str(daily["dt"].iloc[0].date()),
        "end": str(daily["dt"].iloc[-1].date()),
        "annual_return": annual_return,
        "cumulative_return": final_nav - 1,
        "final_nav": final_nav,
        "max_drawdown": max_drawdown,
        "calmar": annual_return / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
        "daily_win_rate": float((daily["ret"] > 0).mean()),
        "turnover": float(daily["turnover"].sum()),
        "avg_abs_weight": float(daily["weight"].abs().mean()),
        "max_abs_weight": float(daily["weight"].abs().max()),
        "fee_rate": FEE_RATE,
        "target_vol": config.target_vol,
        "vol_window": config.vol_window,
        "leverage_cap": config.leverage_cap,
        "mode": "vote" if config.vote else "avg",
        "legs": " | ".join(config.legs),
    }
    return stats, daily


def yearly_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate calendar-year performance."""
    rows = []
    for year, group in daily.groupby(daily["dt"].dt.year):
        data = group.copy()
        nav = (1 + data["ret"]).cumprod()
        drawdown = nav / nav.cummax() - 1
        years = (data["dt"].iloc[-1] - data["dt"].iloc[0]).days / 365.25
        final_nav = float(nav.iloc[-1])
        annual = final_nav ** (1 / years) - 1 if years > 0 and final_nav > 0 else final_nav - 1
        max_drawdown = float(drawdown.min())
        rows.append(
            {
                "year": int(year),
                "start": str(data["dt"].iloc[0].date()),
                "end": str(data["dt"].iloc[-1].date()),
                "days": int(len(data)),
                "return": final_nav - 1,
                "annual_return": annual,
                "max_drawdown": max_drawdown,
                "calmar": annual / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
                "win_rate": float((data["ret"] > 0).mean()),
                "turnover": float(data["turnover"].sum()),
                "avg_abs_weight": float(data["weight"].abs().mean()),
                "max_abs_weight": float(data["weight"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def write_equity_svg(daily: pd.DataFrame, stats: dict, output_dir: Path) -> None:
    """Write a compact self-contained equity SVG/HTML."""
    width, height = 1120, 680
    left, right, top, bottom = 72, 28, 76, 52
    plot_w = width - left - right
    nav_h = 390
    dd_top = top + nav_h + 34
    dd_h = height - dd_top - bottom
    x_values = daily["dt"].map(pd.Timestamp.toordinal).astype(float)
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

    nav_path = "M " + " L ".join(f"{sx(row.dt):.2f} {sy_nav(row.nav):.2f}" for row in daily.itertuples())
    dd_path = "M " + " L ".join(f"{sx(row.dt):.2f} {sy_dd(row.drawdown):.2f}" for row in daily.itertuples())
    subtitle = (
        f"annual {stats['annual_return']:.2%}, max drawdown {stats['max_drawdown']:.2%}, "
        f"final NAV {stats['final_nav']:.4f}"
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fff"/>
<style>
text{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;fill:#172033}}
.title{{font-size:24px;font-weight:700}} .sub{{font-size:14px;fill:#526173}}
.grid{{stroke:#e7ecf3;stroke-width:1}} .axis{{stroke:#cfd8e3;stroke-width:1}}
</style>
<text class="title" x="{left}" y="32">{stats["symbol"]} Optimized Daily CTA</text>
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


def run_one(config: StrategyConfig) -> dict:
    """Run one optimized strategy and save outputs."""
    data = load_daily_bars(config)
    weights = build_weight_frame(data, config)
    stats, daily = evaluate(weights, config)
    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    weights.to_csv(out_dir / "weights.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(out_dir / "daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([stats]).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    yearly_stats(daily).to_csv(out_dir / "yearly_stats.csv", index=False, encoding="utf-8-sig")
    write_equity_svg(daily, stats, out_dir)
    return stats


def main() -> None:
    rows = [run_one(config) for config in CONFIGS]
    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
