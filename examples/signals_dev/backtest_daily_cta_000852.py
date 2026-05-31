"""Pure daily CTA strategy research on 000852.XSHG.

The selected strategy uses daily OHLCV only:

- Five daily directional legs are averaged into a signal score.
- The score is scaled by 30-day realized volatility targeting.
- The next trading day's close-to-close return uses the previous day's weight.

Run:
    .venv/bin/python examples/signals_dev/backtest_daily_cta_000852.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SYMBOL = "000852.XSHG"
STAT_SDT = "20200101"
STAT_EDT = "20260529"
FEE_RATE = 0.0002
TARGET_VOL = 0.40
VOL_WINDOW = 30
LEVERAGE_CAP = 4.0
DATA_FILE = ROOT / "examples" / "results" / "daily_holding_000852" / "000852_XSHG_daily_20100101_20260529.csv"
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_holding_000852"
STRATEGY_NAME = "DailyCTA_AVG5_MA_MOM_DON_VT40_C4"


def load_daily_bars(file_csv: Path = DATA_FILE) -> pd.DataFrame:
    """Load local daily OHLCV data."""
    if not file_csv.exists():
        raise FileNotFoundError(f"Daily data cache not found: {file_csv}")
    data = pd.read_csv(file_csv, parse_dates=["dt"])
    data["dt"] = pd.to_datetime(data["dt"]).dt.tz_localize(None)
    return data.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)


def hold_score(score: pd.Series, threshold: float) -> pd.Series:
    """Convert a score series to long / short state with neutral holding."""
    weight = pd.Series(0.0, index=score.index)
    weight[score > threshold] = 1.0
    weight[score < -threshold] = -1.0
    return weight.replace(0.0, np.nan).ffill().fillna(0.0)


def flat_score(score: pd.Series, threshold: float) -> pd.Series:
    """Convert a score series to long / short / flat states."""
    weight = pd.Series(0.0, index=score.index)
    weight[score > threshold] = 1.0
    weight[score < -threshold] = -1.0
    return weight


def build_direction_legs(data: pd.DataFrame) -> pd.DataFrame:
    """Build the five selected daily CTA directional legs."""
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)

    ma3 = close.rolling(3, min_periods=3).mean()
    ma20 = close.rolling(20, min_periods=20).mean()
    ma50 = close.rolling(50, min_periods=50).mean()
    mom20 = close / close.shift(20) - 1
    don320_upper = high.shift(1).rolling(320, min_periods=320).max()
    don320_lower = low.shift(1).rolling(320, min_periods=320).min()

    legs = pd.DataFrame(index=data.index)
    legs["MA_H_N3_B50"] = hold_score(close / ma3 - 1, 0.005)
    legs["MA_R_N20_B500"] = -hold_score(close / ma20 - 1, 0.05)
    legs["MOM_R_N20_B1500"] = -hold_score(mom20, 0.15)
    legs["MA_H_N50_B100"] = hold_score(close / ma50 - 1, 0.01)
    legs["DON_F_N320"] = flat_score(pd.Series(np.where(close > don320_upper, 1.0, 0.0), index=data.index), 0.5)
    legs.loc[close < don320_lower, "DON_F_N320"] = -1.0
    return legs.fillna(0.0)


def build_weight_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Build daily target weights from selected direction legs."""
    out = data[["dt", "symbol", "close"]].copy()
    legs = build_direction_legs(data)
    out = pd.concat([out, legs], axis=1)
    out["score_avg"] = legs.mean(axis=1)

    daily_ret = out["close"].pct_change().fillna(0.0)
    realized_vol = daily_ret.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std() * np.sqrt(252)
    leverage = (TARGET_VOL / realized_vol).replace([np.inf, -np.inf], np.nan).clip(upper=LEVERAGE_CAP).fillna(0.0)
    out["leverage"] = leverage
    out["weight"] = out["score_avg"] * out["leverage"]
    out["price"] = out["close"].astype(float)

    stat_edt = pd.Timestamp(STAT_EDT) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    mask = (out["dt"] >= pd.Timestamp(STAT_SDT)) & (out["dt"] <= stat_edt)
    return out.loc[mask].reset_index(drop=True)


def evaluate(weights: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
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
        "symbol": SYMBOL,
        "strategy": STRATEGY_NAME,
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
        "target_vol": TARGET_VOL,
        "vol_window": VOL_WINDOW,
        "leverage_cap": LEVERAGE_CAP,
        "legs": "MA_H_N3_B50 | MA_R_N20_B500 | MOM_R_N20_B1500 | MA_H_N50_B100 | DON_F_N320",
    }
    return stats, daily


def period_stats(daily: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Calculate common period stats."""
    rows = [
        {
            "period": "full",
            "sdt": stats["start"],
            **{k: stats[k] for k in ["annual_return", "final_nav", "max_drawdown", "calmar"]},
        }
    ]
    periods = [
        ("2020_2022", "2020-01-01", "2022-12-31"),
        ("2023_2024", "2023-01-01", "2024-12-31"),
        ("last_2y", "2024-05-29", None),
        ("since_2025", "2025-01-01", None),
        ("last_1y", "2025-05-29", None),
    ]
    for name, sdt, edt in periods:
        part = daily[daily["dt"] >= pd.Timestamp(sdt)].copy()
        if edt:
            part = part[part["dt"] <= pd.Timestamp(edt)]
        if len(part) < 2:
            continue
        nav = (1 + part["ret"]).cumprod()
        drawdown = nav / nav.cummax() - 1
        years = (part["dt"].iloc[-1] - part["dt"].iloc[0]).days / 365.25
        final_nav = float(nav.iloc[-1])
        annual = final_nav ** (1 / years) - 1 if final_nav > 0 else float("nan")
        max_drawdown = float(drawdown.min())
        rows.append(
            {
                "period": name,
                "sdt": str(part["dt"].iloc[0].date()),
                "annual_return": annual,
                "final_nav": final_nav,
                "max_drawdown": max_drawdown,
                "calmar": annual / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def write_equity_svg(daily: pd.DataFrame, stats: dict, output_dir: Path) -> None:
    """Write a self-contained SVG/HTML equity chart."""
    width, height = 1120, 680
    left, right, top, bottom = 72, 28, 76, 52
    plot_w = width - left - right
    nav_h = 390
    gap = 34
    dd_top = top + nav_h + gap
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
<text class="title" x="{left}" y="32">{SYMBOL} Pure Daily CTA</text>
<text class="sub" x="{left}" y="56">{subtitle}</text>
<line class="grid" x1="{left}" x2="{width - right}" y1="{sy_nav(1):.2f}" y2="{sy_nav(1):.2f}"/>
<line class="axis" x1="{left}" x2="{width - right}" y1="{top + nav_h}" y2="{top + nav_h}"/>
<line class="axis" x1="{left}" x2="{width - right}" y1="{dd_top}" y2="{dd_top}"/>
<text class="sub" x="{left}" y="{top - 14}">NAV</text>
<text class="sub" x="{left}" y="{dd_top - 12}">Drawdown</text>
<path d="{nav_path}" fill="none" stroke="#0f766e" stroke-width="2.6"/>
<path d="{dd_path}" fill="none" stroke="#b45309" stroke-width="2.2"/>
</svg>"""
    html = f"<!doctype html><meta charset='utf-8'><title>{STRATEGY_NAME}</title><body style='margin:0'>{svg}</body>"
    (output_dir / "daily_cta_equity.svg").write_text(svg, encoding="utf-8")
    (output_dir / "daily_cta_equity.html").write_text(html, encoding="utf-8")


def save_outputs(weights: pd.DataFrame, daily: pd.DataFrame, stats: dict, output_dir: Path = OUTPUT_DIR) -> None:
    """Save strategy artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    weights.to_csv(output_dir / "daily_cta_weights.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output_dir / "daily_cta_daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([stats]).to_csv(output_dir / "daily_cta_summary.csv", index=False, encoding="utf-8-sig")
    period_stats(daily, stats).to_csv(output_dir / "daily_cta_period_stats.csv", index=False, encoding="utf-8-sig")
    write_equity_svg(daily, stats, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=DATA_FILE)
    args = parser.parse_args()

    data = load_daily_bars(args.file)
    weights = build_weight_frame(data)
    stats, daily = evaluate(weights)
    save_outputs(weights, daily, stats)
    print(pd.DataFrame([stats]).to_string(index=False))
    print("\nperiod stats:")
    print(period_stats(daily, stats).to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
