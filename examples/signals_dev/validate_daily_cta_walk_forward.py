"""Walk-forward validation for pure daily CTA variants.

This script is a deliberately stricter anti-overfitting check than the full
sample optimizers:

- each symbol is tested separately;
- each test year selects parameters from the previous four calendar years only;
- the selected parameters are then frozen for the next test year;
- all signals use daily OHLCV only.

Run:
    .venv/bin/python examples/signals_dev/validate_daily_cta_walk_forward.py
"""

from __future__ import annotations

import itertools
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILES = {
    "000852.XSHG": ROOT / "examples" / "results" / "daily_holding_000852" / "000852_XSHG_daily_20100101_20260529.csv",
    "000905.XSHG": ROOT
    / "examples"
    / "results"
    / "daily_holding_cross_symbol"
    / "000905_XSHG_daily_20100101_20260529.csv",
    "159915.XSHE": ROOT
    / "examples"
    / "results"
    / "daily_holding_cross_symbol"
    / "159915_XSHE_daily_20100101_20260529.csv",
}
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_walk_forward"
FEE_RATE = 0.0002
RANDOM_SEED = 20260531

FOLDS = [
    ("2016-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("2017-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2019-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2021-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("2022-01-01", "2025-12-31", "2026-01-01", "2026-05-29"),
]

VOL_WINDOWS = (20, 40)
TARGET_VOLS = (0.15, 0.20, 0.25)
LEVERAGE_CAPS = (1.2, 2.5)


@dataclass(frozen=True)
class Structure:
    """One daily signal structure before volatility targeting."""

    name: str
    legs: tuple[str, ...]
    mode: str
    score: np.ndarray


@dataclass(frozen=True)
class Candidate:
    """One full candidate including volatility targeting parameters."""

    structure: Structure
    vol_window: int
    target_vol: float
    leverage_cap: float

    @property
    def name(self) -> str:
        return (
            f"{self.structure.mode}{len(self.structure.legs)}"
            f"_VW{self.vol_window}_TV{int(self.target_vol * 100)}_C{self.leverage_cap:g}"
            f"__{'|'.join(self.structure.legs)}"
        )


def hold_score(score: pd.Series, threshold: float) -> np.ndarray:
    """Convert a score to long / short state and hold through neutral zones."""
    weight = pd.Series(0.0, index=score.index)
    weight[score > threshold] = 1.0
    weight[score < -threshold] = -1.0
    return weight.replace(0.0, np.nan).ffill().fillna(0.0).to_numpy(float)


def flat_score(score: pd.Series, threshold: float) -> np.ndarray:
    """Convert a score to long / short / flat state."""
    weight = pd.Series(0.0, index=score.index)
    weight[score > threshold] = 1.0
    weight[score < -threshold] = -1.0
    return weight.to_numpy(float)


def load_daily_data(symbol: str) -> pd.DataFrame:
    """Load one symbol daily OHLCV data."""
    file_csv = DATA_FILES[symbol]
    if not file_csv.exists():
        raise FileNotFoundError(f"Daily data file not found: {file_csv}")
    data = pd.read_csv(file_csv, parse_dates=["dt"])
    data["dt"] = pd.to_datetime(data["dt"]).dt.tz_localize(None)
    data["symbol"] = symbol
    return data.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)


def build_leg_library(data: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build a generic daily signal leg library."""
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    legs: dict[str, np.ndarray] = {}

    for n in [3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 50, 60, 80, 120, 160, 240, 320]:
        ma = close.rolling(n, min_periods=n).mean()
        score = close / ma - 1
        for bp in [0, 20, 50, 80, 100, 150, 200, 300, 500, 800, 1200]:
            threshold = bp / 10000
            legs[f"MA_H_N{n}_B{bp}"] = hold_score(score, threshold)
            legs[f"MA_F_N{n}_B{bp}"] = flat_score(score, threshold)
            legs[f"MA_R_N{n}_B{bp}"] = -hold_score(score, threshold)

    for n in [5, 10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320]:
        score = close / close.shift(n) - 1
        for bp in [0, 50, 100, 200, 500, 1000, 1500, 2000]:
            threshold = bp / 10000
            legs[f"MOM_H_N{n}_B{bp}"] = hold_score(score, threshold)
            legs[f"MOM_F_N{n}_B{bp}"] = flat_score(score, threshold)
            legs[f"MOM_R_N{n}_B{bp}"] = -hold_score(score, threshold)

    for fast in [3, 5, 8, 12, 16, 24, 32]:
        ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        for slow in [16, 24, 32, 48, 64, 96, 128, 200, 320]:
            if fast >= slow:
                continue
            score = ema_fast / close.ewm(span=slow, adjust=False, min_periods=slow).mean() - 1
            for bp in [0, 20, 50, 100, 200, 300]:
                threshold = bp / 10000
                legs[f"EMA_H_F{fast}_S{slow}_B{bp}"] = hold_score(score, threshold)
                legs[f"EMA_F_F{fast}_S{slow}_B{bp}"] = flat_score(score, threshold)

    for n in [20, 40, 60, 80, 120, 160, 240, 320]:
        upper = high.shift(1).rolling(n, min_periods=n).max()
        lower = low.shift(1).rolling(n, min_periods=n).min()
        state = pd.Series(0.0, index=data.index)
        state[close > upper] = 1.0
        state[close < lower] = -1.0
        legs[f"DON_H_N{n}"] = state.replace(0.0, np.nan).ffill().fillna(0.0).to_numpy(float)
        legs[f"DON_F_N{n}"] = state.to_numpy(float)

    return legs


def annualized_stats(dates: pd.Series, returns: np.ndarray, weights: np.ndarray) -> dict:
    """Calculate core performance statistics for a date slice."""
    if len(returns) == 0:
        return {}
    nav = np.cumprod(1 + returns)
    if nav[-1] <= 0:
        annual_return = float("nan")
    else:
        years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
        annual_return = nav[-1] ** (1 / years) - 1 if years > 0 else nav[-1] - 1
    drawdown = nav / np.maximum.accumulate(nav) - 1
    max_drawdown = float(drawdown.min())
    return {
        "annual_return": float(annual_return),
        "cumulative_return": float(nav[-1] - 1),
        "final_nav": float(nav[-1]),
        "max_drawdown": max_drawdown,
        "calmar": float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else float("nan"),
        "win_rate": float((returns > 0).mean()),
        "avg_abs_weight": float(np.abs(weights).mean()),
        "max_abs_weight": float(np.abs(weights).max()),
    }


def period_returns(
    raw_score: np.ndarray, data: pd.DataFrame, candidate: Candidate
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build full-period returns from one candidate raw score."""
    close = data["close"].to_numpy(float)
    price_ret = np.zeros(len(close), dtype=float)
    price_ret[1:] = close[1:] / close[:-1] - 1
    realized_vol = pd.Series(price_ret).rolling(candidate.vol_window, min_periods=candidate.vol_window).std().to_numpy()
    realized_vol = realized_vol * np.sqrt(252)
    leverage = np.divide(candidate.target_vol, realized_vol, out=np.zeros_like(realized_vol), where=realized_vol > 0)
    leverage = np.nan_to_num(np.clip(leverage, 0, candidate.leverage_cap), nan=0.0, posinf=0.0, neginf=0.0)
    weight = raw_score * leverage
    prev_weight = np.r_[0.0, weight[:-1]]
    turnover = np.abs(weight - prev_weight)
    returns = prev_weight * price_ret - turnover * FEE_RATE
    return returns, weight, turnover


def evaluate_candidate(data: pd.DataFrame, candidate: Candidate, mask: np.ndarray) -> dict:
    """Evaluate a candidate on a date mask."""
    returns, weights, turnover = period_returns(candidate.structure.score, data, candidate)
    stats = annualized_stats(data.loc[mask, "dt"].reset_index(drop=True), returns[mask], weights[mask])
    if not stats:
        return {}
    stats["turnover"] = float(turnover[mask].sum())
    stats["candidate"] = candidate.name
    stats["legs"] = "|".join(candidate.structure.legs)
    stats["mode"] = candidate.structure.mode
    stats["vol_window"] = candidate.vol_window
    stats["target_vol"] = candidate.target_vol
    stats["leverage_cap"] = candidate.leverage_cap
    return stats


def yearly_loss_count(data: pd.DataFrame, returns: np.ndarray, mask: np.ndarray) -> int:
    """Count losing calendar years inside a mask."""
    part = pd.DataFrame({"dt": data.loc[mask, "dt"].to_numpy(), "ret": returns[mask]})
    losses = 0
    for _, group in part.groupby(pd.to_datetime(part["dt"]).dt.year):
        nav = np.cumprod(1 + group["ret"].to_numpy())
        losses += int(nav[-1] < 1)
    return losses


def rank_legs(data: pd.DataFrame, legs: dict[str, np.ndarray], train_mask: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Rank individual legs on the training period."""
    ranked = []
    for name, raw in legs.items():
        candidate = Candidate(Structure(name, (name,), "RAW", raw), 30, 0.20, 2.0)
        returns, weights, _ = period_returns(raw, data, candidate)
        stats = annualized_stats(
            data.loc[train_mask, "dt"].reset_index(drop=True), returns[train_mask], weights[train_mask]
        )
        if stats and np.isfinite(stats["calmar"]) and stats["annual_return"] > 0:
            ranked.append((stats["calmar"], stats["annual_return"], name, raw))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [(name, raw) for _, _, name, raw in ranked[:28]]


def make_structures(symbol: str, test_start: str, ranked_legs: list[tuple[str, np.ndarray]]) -> list[Structure]:
    """Create deterministic structures from training-ranked legs."""
    structures: list[Structure] = []
    for name, raw in ranked_legs[:8]:
        structures.append(Structure(name, (name,), "RAW", raw))

    seed_text = f"{symbol}|{test_start}|{RANDOM_SEED}"
    seed = zlib.crc32(seed_text.encode("utf-8"))
    rng = np.random.default_rng(seed)
    pool = ranked_legs[:16]
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for k in [2, 3, 4]:
        for combo in itertools.combinations(pool[: min(4, len(pool))], k):
            names = tuple(item[0] for item in combo)
            arr = np.vstack([item[1] for item in combo])
            for mode, score in [("AVG", arr.mean(axis=0)), ("VOTE", np.sign(arr.mean(axis=0)))]:
                key = (mode, names)
                if key not in seen:
                    seen.add(key)
                    structures.append(Structure(f"{mode}{k}", names, mode, score))
        for _ in range(10):
            if len(pool) < k:
                continue
            picked = [pool[i] for i in sorted(rng.choice(len(pool), size=k, replace=False))]
            names = tuple(item[0] for item in picked)
            arr = np.vstack([item[1] for item in picked])
            for mode, score in [("AVG", arr.mean(axis=0)), ("VOTE", np.sign(arr.mean(axis=0)))]:
                key = (mode, names)
                if key not in seen:
                    seen.add(key)
                    structures.append(Structure(f"{mode}{k}", names, mode, score))
    return structures


def select_candidate(
    data: pd.DataFrame, symbol: str, legs: dict[str, np.ndarray], fold: tuple[str, str, str, str]
) -> dict:
    """Select the best candidate using training data only."""
    train_start, train_end, test_start, _ = fold
    train_mask = ((data["dt"] >= pd.Timestamp(train_start)) & (data["dt"] <= pd.Timestamp(train_end))).to_numpy()
    ranked_legs = rank_legs(data, legs, train_mask)
    structures = make_structures(symbol, test_start, ranked_legs)

    rows = []
    returns_cache = {}
    for structure in structures:
        for vol_window in VOL_WINDOWS:
            for target_vol in TARGET_VOLS:
                for leverage_cap in LEVERAGE_CAPS:
                    candidate = Candidate(structure, vol_window, target_vol, leverage_cap)
                    stats = evaluate_candidate(data, candidate, train_mask)
                    if not stats:
                        continue
                    returns, weights, _ = period_returns(candidate.structure.score, data, candidate)
                    stats["loss_years"] = yearly_loss_count(data, returns, train_mask)
                    stats["recent_year_return"] = recent_year_return(data, returns, train_mask)
                    stats["leg_count"] = len(structure.legs)
                    stats["score"] = (
                        stats["calmar"]
                        + 0.5 * stats["recent_year_return"]
                        - 2.0 * max(0.0, abs(stats["max_drawdown"]) - 0.10)
                        - 0.05 * len(structure.legs)
                    )
                    rows.append(stats)
                    returns_cache[stats["candidate"]] = (candidate, returns, weights)

    ranked = pd.DataFrame(rows)
    if ranked.empty:
        raise RuntimeError(f"No candidates generated for {symbol} {fold}")
    turnover_limit = ranked["turnover"].quantile(0.8)
    eligible = ranked[
        (ranked["annual_return"] >= 0.15)
        & (ranked["max_drawdown"] >= -0.12)
        & (ranked["loss_years"] <= 1)
        & (ranked["max_abs_weight"] <= 3.0)
        & (ranked["turnover"] <= turnover_limit)
    ].copy()
    if eligible.empty:
        eligible = ranked[(ranked["annual_return"] > 0) & (ranked["max_drawdown"] >= -0.20)].copy()
    if eligible.empty:
        eligible = ranked.copy()
    selected = eligible.sort_values(["score", "calmar", "annual_return"], ascending=False).iloc[0].to_dict()
    candidate, returns, weights = returns_cache[selected["candidate"]]
    selected["selected_candidates"] = len(ranked)
    selected["eligible_candidates"] = len(eligible)
    selected["train_start"] = train_start
    selected["train_end"] = train_end
    selected["test_start"] = fold[2]
    selected["test_end"] = fold[3]
    selected["_candidate_obj"] = candidate
    selected["_returns"] = returns
    selected["_weights"] = weights
    return selected


def recent_year_return(data: pd.DataFrame, returns: np.ndarray, train_mask: np.ndarray) -> float:
    """Return the last calendar year's return inside the training window."""
    dates = data.loc[train_mask, "dt"]
    if dates.empty:
        return 0.0
    last_year = int(dates.dt.year.max())
    mask = train_mask & (data["dt"].dt.year.to_numpy() == last_year)
    if not mask.any():
        return 0.0
    nav = np.cumprod(1 + returns[mask])
    return float(nav[-1] - 1)


def summarize_oos(daily: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Summarize stitched out-of-sample performance."""
    stats = annualized_stats(daily["dt"], daily["ret"].to_numpy(), daily["weight"].to_numpy())
    stats["symbol"] = symbol
    stats["start"] = str(daily["dt"].iloc[0].date())
    stats["end"] = str(daily["dt"].iloc[-1].date())
    stats["turnover_proxy"] = float(daily["weight"].diff().abs().fillna(daily["weight"].abs()).sum())
    stats["loss_years"] = int((summarize_yearly(daily)["return"] < 0).sum())
    return pd.DataFrame([stats])


def summarize_yearly(daily: pd.DataFrame) -> pd.DataFrame:
    """Summarize out-of-sample performance by calendar year."""
    rows = []
    for year, group in daily.groupby(daily["dt"].dt.year):
        stats = annualized_stats(
            group["dt"].reset_index(drop=True), group["ret"].to_numpy(), group["weight"].to_numpy()
        )
        stats["year"] = int(year)
        stats["start"] = str(group["dt"].iloc[0].date())
        stats["end"] = str(group["dt"].iloc[-1].date())
        stats["return"] = stats["cumulative_return"]
        rows.append(stats)
    return pd.DataFrame(rows)


def write_outputs(
    symbol: str, summary: pd.DataFrame, selected: pd.DataFrame, yearly: pd.DataFrame, daily: pd.DataFrame
) -> None:
    """Persist symbol outputs."""
    out_dir = OUTPUT_DIR / symbol.replace(".", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "walk_forward_summary.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(out_dir / "selected_params_by_year.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "walk_forward_yearly.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(out_dir / "out_sample_daily_nav.csv", index=False, encoding="utf-8-sig")


def save_nav_plot(daily: pd.DataFrame) -> None:
    """Save a combined walk-forward net-value plot for all symbols."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import plotly.graph_objects as go

        fig = go.Figure()
        for symbol, group in daily.groupby("symbol"):
            fig.add_trace(go.Scatter(x=group["dt"], y=group["nav"], mode="lines", name=symbol))
        fig.update_layout(
            title="Daily CTA walk-forward net value by symbol",
            xaxis_title="date",
            yaxis_title="net value",
            template="plotly_white",
        )
        fig.write_html(OUTPUT_DIR / "walk_forward_separate_nav.html", include_plotlyjs="cdn")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    for symbol, group in daily.groupby("symbol"):
        ax.plot(group["dt"], group["nav"], linewidth=1.5, label=symbol)
    ax.set_title("Daily CTA walk-forward net value by symbol")
    ax.set_xlabel("date")
    ax.set_ylabel("net value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "walk_forward_separate_nav.svg")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    daily_rows = []
    for symbol in DATA_FILES:
        summary, selected, yearly, daily = run_symbol_with_daily(symbol)
        write_outputs(symbol, summary, selected, yearly, daily)
        summaries.append(summary)
        daily_rows.append(daily)
        print(f"\n== {symbol} ==")
        print(summary.to_string(index=False))
        print(yearly[["year", "return", "max_drawdown", "annual_return", "calmar"]].to_string(index=False))
    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(OUTPUT_DIR / "walk_forward_summary.csv", index=False, encoding="utf-8-sig")
    save_nav_plot(pd.concat(daily_rows, ignore_index=True))
    print(f"\noutputs: {OUTPUT_DIR}")


def run_symbol_with_daily(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run walk-forward validation and return the daily frame."""
    data = load_daily_data(symbol)
    legs = build_leg_library(data)
    selected_rows = []
    daily_parts = []
    for fold in FOLDS:
        selected = select_candidate(data, symbol, legs, fold)
        test_start, test_end = pd.Timestamp(fold[2]), pd.Timestamp(fold[3])
        test_mask = ((data["dt"] >= test_start) & (data["dt"] <= test_end)).to_numpy()
        returns = selected.pop("_returns")
        weights = selected.pop("_weights")
        selected.pop("_candidate_obj")
        test_stats = annualized_stats(
            data.loc[test_mask, "dt"].reset_index(drop=True), returns[test_mask], weights[test_mask]
        )
        selected.update({f"test_{k}": v for k, v in test_stats.items()})
        selected_rows.append(selected)
        daily_parts.append(
            pd.DataFrame(
                {
                    "dt": data.loc[test_mask, "dt"].to_numpy(),
                    "symbol": symbol,
                    "ret": returns[test_mask],
                    "weight": weights[test_mask],
                    "test_year": test_start.year,
                    "candidate": selected["candidate"],
                    "legs": selected["legs"],
                }
            )
        )

    selected_df = pd.DataFrame(selected_rows)
    daily = pd.concat(daily_parts, ignore_index=True).sort_values("dt").reset_index(drop=True)
    daily["nav"] = (1 + daily["ret"]).cumprod()
    daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1
    summary = summarize_oos(daily, symbol)
    yearly = summarize_yearly(daily)
    return summary, selected_df, yearly, daily


if __name__ == "__main__":
    main()
