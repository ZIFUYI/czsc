"""Cross-sectional daily CTA walk-forward validation for three symbols.

The strategy uses only daily OHLCV data for 000852, 000905 and 159915. Each
outer fold selects a cross-sectional momentum/trend rule from the previous four
years, then freezes that rule for the next test year.

Run:
    .venv/bin/python examples/signals_dev/validate_daily_cta_cross_section_walk_forward.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from validate_daily_cta_walk_forward import (
    DATA_FILES,
    FEE_RATE,
    FOLDS,
    annualized_stats,
    load_daily_data,
    summarize_yearly,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_cross_section_walk_forward"
SYMBOLS = tuple(DATA_FILES)
LOOKBACK_SETS = ((20,), (60,), (120,), (20, 60), (20, 60, 120))
THRESHOLDS = (0.0, 0.01, 0.03)
SCORE_TYPES = ("mom", "blend")
MODES = ("rank_ls", "rank_long", "market_switch", "trend_equal")
VOL_WINDOWS = (20, 60)
TARGET_VOLS = (0.10, 0.15, 0.20, 0.25, 0.30)
LEVERAGE_CAPS = (1.0, 1.2, 1.5, 2.0)
ENSEMBLE_TOP_N = 6


@dataclass(frozen=True)
class CrossCandidate:
    """One cross-sectional rule with portfolio-level volatility targeting."""

    score_type: str
    lookbacks: tuple[int, ...]
    threshold: float
    mode: str
    vol_window: int
    target_vol: float
    leverage_cap: float

    @property
    def name(self) -> str:
        lbs = "-".join(str(x) for x in self.lookbacks)
        bp = int(self.threshold * 10000)
        tv = int(self.target_vol * 100)
        return f"{self.score_type}_{self.mode}_N{lbs}_B{bp}_VW{self.vol_window}_TV{tv}_C{self.leverage_cap:g}"


@dataclass(frozen=True)
class SelectionPolicy:
    """Training-only selection policy for a cross-sectional ensemble."""

    name: str
    top_n: int
    min_annual_return: float
    min_max_drawdown: float
    max_abs_weight: float
    max_loss_years: int
    min_recent_year_return: float
    drawdown_threshold: float
    drawdown_penalty: float
    train_drawdown_budget: float | None = None
    allowed_vol_windows: tuple[int, ...] | None = None
    allowed_target_vols: tuple[float, ...] | None = None
    allowed_leverage_caps: tuple[float, ...] | None = None


POLICIES = (
    SelectionPolicy(
        "aggressive_baseline",
        6,
        0.10,
        -0.13,
        2.2,
        1,
        -1.0,
        0.10,
        2.5,
        None,
        (20, 60),
        (0.20, 0.30),
        (1.5, 2.0),
    ),
    SelectionPolicy(
        "aggressive_budget_10",
        6,
        0.10,
        -0.13,
        2.2,
        1,
        -1.0,
        0.10,
        2.5,
        0.10,
        (20, 60),
        (0.20, 0.30),
        (1.5, 2.0),
    ),
    SelectionPolicy("expanded_baseline", 6, 0.10, -0.13, 2.2, 1, -1.0, 0.10, 2.5, None),
    SelectionPolicy("low_risk_top3", 3, 0.04, -0.08, 1.2, 1, -1.0, 0.08, 6.0, None),
    SelectionPolicy("low_risk_top10", 10, 0.04, -0.08, 1.0, 1, -1.0, 0.08, 6.0, None),
)
NESTED_BRAKE_MODE = "aggressive_nested_brake"
MDD_BRAKE_RESET_MODE = "aggressive_mdd_brake_reset"
TARGET06_BRAKE_RESET_MODE = "aggressive_target06_brake_reset"
BRAKE_BLEND_50_MODE = "aggressive_brake_blend_50"
STATE_SWITCH_MODE = "aggressive_state_switch"
STATE_SWITCH_X110_MODE = "aggressive_state_switch_x110"
STATE_SWITCH_TIERED_MODE = "aggressive_state_switch_tiered"
STATE_SWITCH_RECENT_GUARD_MODE = "aggressive_state_switch_recent_guard"
STATE_SWITCH_ADAPTIVE_MIX_MODE = "aggressive_state_switch_adaptive_mix"
STATE_SWITCH_TRAIN_ANNUAL_THRESHOLD = 0.17
STATE_SWITCH_SCALE = 1.10
STATE_SWITCH_STRONG_ANNUAL_THRESHOLD = 0.20
STATE_SWITCH_RECENT_GUARD_THRESHOLD = 0.20
STATE_SWITCH_ADAPTIVE_HOT_RECENT_THRESHOLD = 0.45
STATE_SWITCH_WEAK_SCALE = 1.10
STATE_SWITCH_MID_SCALE = 1.40
STATE_SWITCH_STRONG_SCALE = 1.23
STATE_SWITCH_RECENT_GUARD_WEAK_SCALE = 1.11
STATE_SWITCH_RECENT_GUARD_MID_SCALE = 1.40
STATE_SWITCH_RECENT_GUARD_STRONG_COOL_SCALE = 1.37
STATE_SWITCH_RECENT_GUARD_STRONG_HOT_SCALE = 1.23
BRAKE_GRID = tuple(
    (stop, resume, scale)
    for stop in (-0.04, -0.05, -0.06, -0.08, -0.10)
    for resume in (-0.01, -0.02, -0.03, -0.04, -0.06)
    for scale in (0.0, 0.25, 0.5, 0.75)
    if resume > stop
)


def load_panel() -> pd.DataFrame:
    """Load and align close prices for all symbols."""
    frames = []
    for symbol in SYMBOLS:
        data = load_daily_data(symbol)[["dt", "close"]].rename(columns={"close": symbol})
        frames.append(data)
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on="dt", how="inner")
    return panel.sort_values("dt").reset_index(drop=True)


def build_scores(close: pd.DataFrame, score_type: str, lookbacks: tuple[int, ...]) -> pd.DataFrame:
    """Build cross-sectional scores from daily closes."""
    if score_type == "mom":
        parts = [close / close.shift(n) - 1 for n in lookbacks]
    elif score_type == "ma":
        parts = [close / close.rolling(n, min_periods=n).mean() - 1 for n in lookbacks]
    elif score_type == "blend":
        parts = []
        for n in lookbacks:
            parts.append(close / close.shift(n) - 1)
            parts.append(close / close.rolling(n, min_periods=n).mean() - 1)
    else:
        raise ValueError(f"Unsupported score_type: {score_type}")
    return sum(parts) / len(parts)


def raw_weights_from_scores(scores: pd.DataFrame, mode: str, threshold: float) -> pd.DataFrame:
    """Convert scores into symbol weights before volatility targeting."""
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    values = scores.to_numpy(float)
    cols = list(scores.columns)
    for i, row in enumerate(values):
        valid = np.isfinite(row)
        if valid.sum() < 2:
            continue
        valid_values = row[valid]
        valid_cols = [col for col, ok in zip(cols, valid, strict=True) if ok]
        top_idx = int(np.argmax(valid_values))
        bottom_idx = int(np.argmin(valid_values))
        top_col = valid_cols[top_idx]
        bottom_col = valid_cols[bottom_idx]
        top = valid_values[top_idx]
        bottom = valid_values[bottom_idx]
        spread = top - bottom
        market = float(np.nanmean(valid_values))

        if mode == "rank_ls" and spread > threshold:
            weights.at[i, top_col] = 0.5
            weights.at[i, bottom_col] = -0.5
        elif mode == "rank_long" and top > threshold:
            weights.at[i, top_col] = 1.0
        elif mode == "rank_short" and bottom < -threshold:
            weights.at[i, bottom_col] = -1.0
        elif mode == "market_switch":
            if market > threshold:
                weights.at[i, top_col] = 1.0
            elif market < -threshold:
                weights.at[i, bottom_col] = -1.0
        elif mode == "trend_equal":
            longs = [valid_cols[j] for j, value in enumerate(valid_values) if value > threshold]
            shorts = [valid_cols[j] for j, value in enumerate(valid_values) if value < -threshold]
            if longs and not shorts:
                for col in longs:
                    weights.at[i, col] = 1.0 / len(longs)
            elif shorts and not longs:
                for col in shorts:
                    weights.at[i, col] = -1.0 / len(shorts)
            elif longs and shorts:
                for col in longs:
                    weights.at[i, col] = 0.5 / len(longs)
                for col in shorts:
                    weights.at[i, col] = -0.5 / len(shorts)
    return weights


def candidate_returns(panel: pd.DataFrame, candidate: CrossCandidate) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Calculate full-period strategy returns, exposure and symbol weights."""
    close = panel[list(SYMBOLS)].astype(float)
    price_ret = close.pct_change().fillna(0.0)
    scores = build_scores(close, candidate.score_type, candidate.lookbacks)
    raw_weights = raw_weights_from_scores(scores, candidate.mode, candidate.threshold)
    raw_port_ret = (raw_weights.shift(1).fillna(0.0) * price_ret).sum(axis=1)
    realized_vol = raw_port_ret.rolling(candidate.vol_window, min_periods=candidate.vol_window).std() * np.sqrt(252)
    leverage = candidate.target_vol / realized_vol.replace(0, np.nan)
    leverage = leverage.clip(lower=0, upper=candidate.leverage_cap).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = raw_weights.multiply(leverage, axis=0)
    prev_weights = weights.shift(1).fillna(0.0)
    turnover = (weights - prev_weights).abs().sum(axis=1)
    returns = (prev_weights * price_ret).sum(axis=1) - turnover * FEE_RATE
    exposure = weights.abs().sum(axis=1)
    return returns.to_numpy(float), exposure.to_numpy(float), weights


def build_raw_weight_cache(panel: pd.DataFrame) -> dict[tuple[str, tuple[int, ...], float, str], pd.DataFrame]:
    """Cache raw symbol weights before volatility targeting."""
    close = panel[list(SYMBOLS)].astype(float)
    score_cache = {}
    weight_cache = {}
    for score_type in SCORE_TYPES:
        for lookbacks in LOOKBACK_SETS:
            score_cache[(score_type, lookbacks)] = build_scores(close, score_type, lookbacks)
    for (score_type, lookbacks), scores in score_cache.items():
        for threshold in THRESHOLDS:
            for mode in MODES:
                weight_cache[(score_type, lookbacks, threshold, mode)] = raw_weights_from_scores(
                    scores, mode, threshold
                )
    return weight_cache


def cached_candidate_returns(
    panel: pd.DataFrame,
    candidate: CrossCandidate,
    weight_cache: dict[tuple[str, tuple[int, ...], float, str], pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Calculate candidate returns from cached raw symbol weights."""
    close = panel[list(SYMBOLS)].astype(float)
    price_ret = close.pct_change().fillna(0.0)
    raw_weights = weight_cache[(candidate.score_type, candidate.lookbacks, candidate.threshold, candidate.mode)]
    raw_port_ret = (raw_weights.shift(1).fillna(0.0) * price_ret).sum(axis=1)
    realized_vol = raw_port_ret.rolling(candidate.vol_window, min_periods=candidate.vol_window).std() * np.sqrt(252)
    leverage = candidate.target_vol / realized_vol.replace(0, np.nan)
    leverage = leverage.clip(lower=0, upper=candidate.leverage_cap).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = raw_weights.multiply(leverage, axis=0)
    prev_weights = weights.shift(1).fillna(0.0)
    turnover = (weights - prev_weights).abs().sum(axis=1)
    returns = (prev_weights * price_ret).sum(axis=1) - turnover * FEE_RATE
    exposure = weights.abs().sum(axis=1)
    return returns.to_numpy(float), exposure.to_numpy(float), weights


def loss_years(dates: pd.Series, returns: np.ndarray, mask: np.ndarray) -> int:
    """Count losing calendar years in a date mask."""
    part = pd.DataFrame({"dt": dates.loc[mask].to_numpy(), "ret": returns[mask]})
    losses = 0
    for _, group in part.groupby(pd.to_datetime(part["dt"]).dt.year):
        nav = np.cumprod(1 + group["ret"].to_numpy())
        losses += int(nav[-1] < 1)
    return losses


def recent_return(dates: pd.Series, returns: np.ndarray, mask: np.ndarray) -> float:
    """Return the most recent calendar year's return in the mask."""
    years = pd.to_datetime(dates.loc[mask]).dt.year
    if years.empty:
        return 0.0
    last_year = int(years.max())
    year_mask = mask & (pd.to_datetime(dates).dt.year.to_numpy() == last_year)
    nav = np.cumprod(1 + returns[year_mask])
    return float(nav[-1] - 1) if len(nav) else 0.0


def apply_drawdown_brake(
    returns: np.ndarray,
    exposure: np.ndarray,
    stop: float,
    resume: float,
    scale: float,
    start_nav: float = 1.0,
    start_peak: float = 1.0,
    start_risk: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Apply a causal net-value drawdown brake to a return stream."""
    nav = start_nav
    peak = start_peak
    risk = start_risk
    adjusted_returns = []
    adjusted_exposure = []
    risk_states = []
    for daily_return, daily_exposure in zip(returns, exposure, strict=True):
        adjusted_return = risk * daily_return
        adjusted_returns.append(adjusted_return)
        adjusted_exposure.append(risk * daily_exposure)
        risk_states.append(risk)
        nav *= 1 + adjusted_return
        peak = max(peak, nav)
        drawdown = nav / peak - 1
        if risk >= 1 and drawdown <= stop:
            risk = scale
        elif risk < 1 and drawdown >= resume:
            risk = 1.0
    return (
        np.asarray(adjusted_returns),
        np.asarray(adjusted_exposure),
        np.asarray(risk_states),
        nav,
        peak,
        risk,
    )


def select_training_brake(
    panel: pd.DataFrame,
    returns: np.ndarray,
    exposure: np.ndarray,
    train_mask: np.ndarray,
    objective: str = "balanced",
) -> dict:
    """Select drawdown brake parameters on the training period only."""
    best = None
    train_dates = panel.loc[train_mask, "dt"].reset_index(drop=True)
    for stop, resume, scale in BRAKE_GRID:
        adjusted_returns, adjusted_exposure, _, _, _, _ = apply_drawdown_brake(
            returns[train_mask], exposure[train_mask], stop, resume, scale
        )
        stats = annualized_stats(train_dates, adjusted_returns, adjusted_exposure)
        if objective == "balanced":
            feasible = stats["annual_return"] > 0 and stats["max_drawdown"] >= -0.10
            score = (
                (1000 if feasible else 0)
                + stats["annual_return"]
                + 0.5 * stats["calmar"]
                - 5 * max(0.0, abs(stats["max_drawdown"]) - 0.10)
            )
        elif objective == "target06":
            feasible = stats["annual_return"] > 0 and stats["max_drawdown"] >= -0.06
            score = (
                (1000 if feasible else 0)
                + 0.25 * stats["annual_return"]
                + stats["calmar"]
                - 10 * max(0.0, abs(stats["max_drawdown"]) - 0.06)
            )
        elif objective == "min_mdd_ann05":
            feasible = stats["annual_return"] >= 0.05
            score = (1000 if feasible else 0) + stats["max_drawdown"] + 0.1 * stats["annual_return"]
        else:
            raise ValueError(f"Unsupported brake objective: {objective}")
        stats["score"] = score
        stats["stop"] = stop
        stats["resume"] = resume
        stats["scale"] = scale
        stats["train_feasible"] = feasible
        if best is None or stats["score"] > best["score"]:
            best = stats
    if best is None:
        raise RuntimeError("No drawdown brake parameters generated")
    return best


def evaluate_candidate(
    panel: pd.DataFrame,
    candidate: CrossCandidate,
    mask: np.ndarray,
    weight_cache: dict[tuple[str, tuple[int, ...], float, str], pd.DataFrame],
) -> dict:
    """Evaluate one candidate on a date mask."""
    returns, exposure, _ = cached_candidate_returns(panel, candidate, weight_cache)
    stats = annualized_stats(panel.loc[mask, "dt"].reset_index(drop=True), returns[mask], exposure[mask])
    stats["candidate"] = candidate.name
    stats["score_type"] = candidate.score_type
    stats["lookbacks"] = "-".join(str(x) for x in candidate.lookbacks)
    stats["threshold"] = candidate.threshold
    stats["mode"] = candidate.mode
    stats["vol_window"] = candidate.vol_window
    stats["target_vol"] = candidate.target_vol
    stats["leverage_cap"] = candidate.leverage_cap
    stats["_returns"] = returns
    stats["_exposure"] = exposure
    return stats


def build_candidate_rows(
    panel: pd.DataFrame,
    fold: tuple[str, str, str, str],
    weight_cache: dict[tuple[str, tuple[int, ...], float, str], pd.DataFrame],
) -> tuple[np.ndarray, list[dict]]:
    """Build all candidate rows for one training fold."""
    train_start, train_end, test_start, test_end = fold
    train_mask = ((panel["dt"] >= pd.Timestamp(train_start)) & (panel["dt"] <= pd.Timestamp(train_end))).to_numpy()
    rows = []
    for score_type in SCORE_TYPES:
        for lookbacks in LOOKBACK_SETS:
            for threshold in THRESHOLDS:
                for mode in MODES:
                    for vol_window in VOL_WINDOWS:
                        for target_vol in TARGET_VOLS:
                            for leverage_cap in LEVERAGE_CAPS:
                                candidate = CrossCandidate(
                                    score_type, lookbacks, threshold, mode, vol_window, target_vol, leverage_cap
                                )
                                stats = evaluate_candidate(panel, candidate, train_mask, weight_cache)
                                stats["loss_years"] = loss_years(panel["dt"], stats["_returns"], train_mask)
                                stats["recent_year_return"] = recent_return(panel["dt"], stats["_returns"], train_mask)
                                stats["score"] = (
                                    stats["calmar"]
                                    + 0.4 * stats["recent_year_return"]
                                    - 2.5 * max(0.0, abs(stats["max_drawdown"]) - 0.10)
                                )
                                stats["train_start"] = train_start
                                stats["train_end"] = train_end
                                stats["test_start"] = test_start
                                stats["test_end"] = test_end
                                rows.append(stats)
    return train_mask, rows


def select_candidate_ensemble(
    panel: pd.DataFrame, policy: SelectionPolicy, rows: list[dict], train_mask: np.ndarray
) -> dict:
    """Select a top-N cross-sectional ensemble from training data only."""
    ranked = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows])
    if policy.allowed_vol_windows is not None:
        ranked = ranked[ranked["vol_window"].isin(policy.allowed_vol_windows)].copy()
    if policy.allowed_target_vols is not None:
        ranked = ranked[ranked["target_vol"].isin(policy.allowed_target_vols)].copy()
    if policy.allowed_leverage_caps is not None:
        ranked = ranked[ranked["leverage_cap"].isin(policy.allowed_leverage_caps)].copy()
    ranked["policy_score"] = (
        ranked["calmar"]
        + 0.4 * ranked["recent_year_return"]
        - policy.drawdown_penalty * (ranked["max_drawdown"].abs() - policy.drawdown_threshold).clip(lower=0)
    )
    eligible = ranked[
        (ranked["annual_return"] >= policy.min_annual_return)
        & (ranked["max_drawdown"] >= policy.min_max_drawdown)
        & (ranked["loss_years"] <= policy.max_loss_years)
        & (ranked["max_abs_weight"] <= policy.max_abs_weight)
        & (ranked["recent_year_return"] >= policy.min_recent_year_return)
    ].copy()
    if eligible.empty:
        eligible = ranked[
            (ranked["annual_return"] > 0)
            & (ranked["max_drawdown"] >= policy.min_max_drawdown * 1.5)
            & (ranked["max_abs_weight"] <= policy.max_abs_weight)
        ].copy()
    if eligible.empty:
        eligible = ranked.copy()

    eligible = eligible.sort_values(["policy_score", "calmar", "annual_return"], ascending=False).head(policy.top_n)
    returns_lookup = {row["candidate"]: row["_returns"] for row in rows}
    exposure_lookup = {row["candidate"]: row["_exposure"] for row in rows}
    returns = np.vstack([returns_lookup[name] for name in eligible["candidate"]]).mean(axis=0)
    exposure = np.vstack([exposure_lookup[name] for name in eligible["candidate"]]).mean(axis=0)
    raw_train_stats = annualized_stats(
        panel.loc[train_mask, "dt"].reset_index(drop=True), returns[train_mask], exposure[train_mask]
    )
    train_scale = 1.0
    if policy.train_drawdown_budget is not None and raw_train_stats["max_drawdown"] < 0:
        train_scale = min(1.0, policy.train_drawdown_budget / abs(raw_train_stats["max_drawdown"]))
    returns = returns * train_scale
    exposure = exposure * train_scale
    train_stats = annualized_stats(
        panel.loc[train_mask, "dt"].reset_index(drop=True), returns[train_mask], exposure[train_mask]
    )
    train_stats["mode"] = policy.name
    train_stats["loss_years"] = loss_years(panel["dt"], returns, train_mask)
    train_stats["recent_year_return"] = recent_return(panel["dt"], returns, train_mask)
    train_stats["candidate"] = f"ENS{len(eligible)}__{eligible.iloc[0]['candidate']}"
    train_stats["top_candidates"] = "|".join(eligible["candidate"].head(3))
    train_stats["selected_candidates"] = len(ranked)
    train_stats["eligible_candidates"] = len(eligible)
    train_stats["train_scale"] = train_scale
    train_stats["raw_train_annual_return"] = raw_train_stats["annual_return"]
    train_stats["raw_train_max_drawdown"] = raw_train_stats["max_drawdown"]
    train_stats["train_start"] = rows[0]["train_start"]
    train_stats["train_end"] = rows[0]["train_end"]
    train_stats["test_start"] = rows[0]["test_start"]
    train_stats["test_end"] = rows[0]["test_end"]
    train_stats["_returns"] = returns
    train_stats["_exposure"] = exposure
    return train_stats


def summarize_daily(daily: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Summarize stitched out-of-sample returns."""
    stats = annualized_stats(daily["dt"], daily["ret"].to_numpy(), daily["weight"].to_numpy())
    stats["mode"] = mode
    stats["start"] = str(daily["dt"].iloc[0].date())
    stats["end"] = str(daily["dt"].iloc[-1].date())
    stats["loss_years"] = int((summarize_yearly(daily)["return"] < 0).sum())
    return pd.DataFrame([stats])


def save_nav_plot(daily: pd.DataFrame) -> None:
    """Save a net-value plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import plotly.graph_objects as go

        fig = go.Figure()
        for mode, group in daily.groupby("mode"):
            fig.add_trace(go.Scatter(x=group["dt"], y=group["nav"], mode="lines", name=mode))
        fig.update_layout(
            title="Daily CTA cross-section walk-forward net value",
            xaxis_title="date",
            yaxis_title="net value",
            template="plotly_white",
        )
        fig.write_html(OUTPUT_DIR / "cross_section_walk_forward_nav.html", include_plotlyjs="cdn")
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    for mode, group in daily.groupby("mode"):
        ax.plot(group["dt"], group["nav"], linewidth=1.5, label=mode)
    ax.set_title("Daily CTA cross-section walk-forward net value")
    ax.set_xlabel("date")
    ax.set_ylabel("net value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cross_section_walk_forward_nav.svg")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    weight_cache = build_raw_weight_cache(panel)
    selected_rows = []
    daily_parts: dict[str, list[pd.DataFrame]] = {policy.name: [] for policy in POLICIES}
    daily_parts[NESTED_BRAKE_MODE] = []
    daily_parts[MDD_BRAKE_RESET_MODE] = []
    daily_parts[TARGET06_BRAKE_RESET_MODE] = []
    for fold in FOLDS:
        train_mask, rows = build_candidate_rows(panel, fold, weight_cache)
        test_start, test_end = pd.Timestamp(fold[2]), pd.Timestamp(fold[3])
        test_mask = ((panel["dt"] >= test_start) & (panel["dt"] <= test_end)).to_numpy()
        for policy in POLICIES:
            selected = select_candidate_ensemble(panel, policy, rows, train_mask)
            returns = selected.pop("_returns")
            exposure = selected.pop("_exposure")
            test_stats = annualized_stats(
                panel.loc[test_mask, "dt"].reset_index(drop=True), returns[test_mask], exposure[test_mask]
            )
            selected.update({f"test_{k}": v for k, v in test_stats.items()})
            selected_rows.append(selected)
            daily_parts[policy.name].append(
                pd.DataFrame(
                    {
                        "dt": panel.loc[test_mask, "dt"].to_numpy(),
                        "mode": policy.name,
                        "ret": returns[test_mask],
                        "weight": exposure[test_mask],
                        "test_year": test_start.year,
                        "candidate": selected["candidate"],
                    }
                )
            )
            if policy.name == "aggressive_baseline":
                brake_specs = (
                    (NESTED_BRAKE_MODE, "balanced", False),
                    (MDD_BRAKE_RESET_MODE, "min_mdd_ann05", True),
                    (TARGET06_BRAKE_RESET_MODE, "target06", True),
                )
                for brake_mode, objective, reset_state in brake_specs:
                    brake = select_training_brake(panel, returns, exposure, train_mask, objective)
                    if reset_state:
                        train_nav, train_peak, train_risk = 1.0, 1.0, 1.0
                    else:
                        _, _, _, train_nav, train_peak, train_risk = apply_drawdown_brake(
                            returns[train_mask],
                            exposure[train_mask],
                            brake["stop"],
                            brake["resume"],
                            brake["scale"],
                        )
                    brake_returns, brake_exposure, risk_states, _, _, _ = apply_drawdown_brake(
                        returns[test_mask],
                        exposure[test_mask],
                        brake["stop"],
                        brake["resume"],
                        brake["scale"],
                        train_nav,
                        train_peak,
                        train_risk,
                    )
                    brake_selected = selected.copy()
                    brake_selected["mode"] = brake_mode
                    brake_selected["brake_objective"] = objective
                    brake_selected["brake_reset_state"] = reset_state
                    brake_selected["brake_stop"] = brake["stop"]
                    brake_selected["brake_resume"] = brake["resume"]
                    brake_selected["brake_scale"] = brake["scale"]
                    brake_selected["brake_train_annual_return"] = brake["annual_return"]
                    brake_selected["brake_train_max_drawdown"] = brake["max_drawdown"]
                    brake_selected["brake_train_feasible"] = brake["train_feasible"]
                    brake_stats = annualized_stats(
                        panel.loc[test_mask, "dt"].reset_index(drop=True), brake_returns, brake_exposure
                    )
                    brake_selected.update({f"test_{k}": v for k, v in brake_stats.items()})
                    selected_rows.append(brake_selected)
                    daily_parts[brake_mode].append(
                        pd.DataFrame(
                            {
                                "dt": panel.loc[test_mask, "dt"].to_numpy(),
                                "mode": brake_mode,
                                "ret": brake_returns,
                                "weight": brake_exposure,
                                "test_year": test_start.year,
                                "candidate": selected["candidate"],
                                "risk_state": risk_states,
                            }
                        )
                    )

    mdd_daily = pd.concat(daily_parts[MDD_BRAKE_RESET_MODE], ignore_index=True)
    target06_daily = pd.concat(daily_parts[TARGET06_BRAKE_RESET_MODE], ignore_index=True)
    blend_daily = mdd_daily.merge(
        target06_daily,
        on=["dt", "test_year"],
        suffixes=("_mdd", "_target06"),
    )
    daily_parts[BRAKE_BLEND_50_MODE] = [
        pd.DataFrame(
            {
                "dt": blend_daily["dt"],
                "mode": BRAKE_BLEND_50_MODE,
                "ret": 0.5 * blend_daily["ret_mdd"] + 0.5 * blend_daily["ret_target06"],
                "weight": 0.5 * blend_daily["weight_mdd"] + 0.5 * blend_daily["weight_target06"],
                "test_year": blend_daily["test_year"],
                "candidate": f"{MDD_BRAKE_RESET_MODE}|{TARGET06_BRAKE_RESET_MODE}",
            }
        )
    ]
    selected_df = pd.DataFrame(selected_rows)
    baseline_by_year = selected_df[selected_df["mode"] == "aggressive_baseline"].copy()
    baseline_by_year["test_year"] = pd.to_datetime(baseline_by_year["test_start"]).dt.year
    switch_weight_by_year = {
        int(row.test_year): float(row.annual_return >= STATE_SWITCH_TRAIN_ANNUAL_THRESHOLD)
        for row in baseline_by_year.itertuples()
    }
    switch_weights = blend_daily["test_year"].map(switch_weight_by_year).astype(float)
    switch_daily = pd.DataFrame(
        {
            "dt": blend_daily["dt"],
            "mode": STATE_SWITCH_MODE,
            "ret": switch_weights * blend_daily["ret_mdd"] + (1 - switch_weights) * blend_daily["ret_target06"],
            "weight": switch_weights * blend_daily["weight_mdd"]
            + (1 - switch_weights) * blend_daily["weight_target06"],
            "test_year": blend_daily["test_year"],
            "candidate": f"{MDD_BRAKE_RESET_MODE}|{TARGET06_BRAKE_RESET_MODE}",
            "switch_weight_mdd": switch_weights,
        }
    )
    daily_parts[STATE_SWITCH_MODE] = [switch_daily]
    switch_scaled_daily = switch_daily.copy()
    switch_scaled_daily["mode"] = STATE_SWITCH_X110_MODE
    switch_scaled_daily["ret"] = switch_scaled_daily["ret"] * STATE_SWITCH_SCALE
    switch_scaled_daily["weight"] = switch_scaled_daily["weight"] * STATE_SWITCH_SCALE
    switch_scaled_daily["candidate"] = f"{STATE_SWITCH_MODE}*{STATE_SWITCH_SCALE:g}"
    switch_scaled_daily["switch_scale"] = STATE_SWITCH_SCALE
    daily_parts[STATE_SWITCH_X110_MODE] = [switch_scaled_daily]
    tiered_scale_by_year = {}
    for row in baseline_by_year.itertuples():
        if row.annual_return >= STATE_SWITCH_STRONG_ANNUAL_THRESHOLD:
            scale = STATE_SWITCH_STRONG_SCALE
        elif switch_weight_by_year[int(row.test_year)] >= 1:
            scale = STATE_SWITCH_MID_SCALE
        else:
            scale = STATE_SWITCH_WEAK_SCALE
        tiered_scale_by_year[int(row.test_year)] = scale
    tiered_scales = blend_daily["test_year"].map(tiered_scale_by_year).astype(float)
    switch_tiered_daily = switch_daily.copy()
    switch_tiered_daily["mode"] = STATE_SWITCH_TIERED_MODE
    switch_tiered_daily["ret"] = switch_tiered_daily["ret"] * tiered_scales.to_numpy()
    switch_tiered_daily["weight"] = switch_tiered_daily["weight"] * tiered_scales.to_numpy()
    switch_tiered_daily["candidate"] = f"{STATE_SWITCH_MODE}_tiered"
    switch_tiered_daily["switch_scale"] = tiered_scales.to_numpy()
    daily_parts[STATE_SWITCH_TIERED_MODE] = [switch_tiered_daily]
    recent_guard_scale_by_year = {}
    for row in baseline_by_year.itertuples():
        if row.annual_return < STATE_SWITCH_TRAIN_ANNUAL_THRESHOLD:
            scale = STATE_SWITCH_RECENT_GUARD_WEAK_SCALE
        elif row.annual_return < STATE_SWITCH_STRONG_ANNUAL_THRESHOLD:
            scale = STATE_SWITCH_RECENT_GUARD_MID_SCALE
        elif row.recent_year_return < STATE_SWITCH_RECENT_GUARD_THRESHOLD:
            scale = STATE_SWITCH_RECENT_GUARD_STRONG_COOL_SCALE
        else:
            scale = STATE_SWITCH_RECENT_GUARD_STRONG_HOT_SCALE
        recent_guard_scale_by_year[int(row.test_year)] = scale
    recent_guard_scales = blend_daily["test_year"].map(recent_guard_scale_by_year).astype(float)
    switch_recent_guard_daily = switch_daily.copy()
    switch_recent_guard_daily["mode"] = STATE_SWITCH_RECENT_GUARD_MODE
    switch_recent_guard_daily["ret"] = switch_recent_guard_daily["ret"] * recent_guard_scales.to_numpy()
    switch_recent_guard_daily["weight"] = switch_recent_guard_daily["weight"] * recent_guard_scales.to_numpy()
    switch_recent_guard_daily["candidate"] = f"{STATE_SWITCH_MODE}_recent_guard"
    switch_recent_guard_daily["switch_scale"] = recent_guard_scales.to_numpy()
    daily_parts[STATE_SWITCH_RECENT_GUARD_MODE] = [switch_recent_guard_daily]
    baseline_daily = pd.concat(daily_parts["aggressive_baseline"], ignore_index=True)
    low_risk_top3_daily = pd.concat(daily_parts["low_risk_top3"], ignore_index=True)
    adaptive_parts = []
    adaptive_mode_by_year = {}
    for row in baseline_by_year.itertuples():
        test_year = int(row.test_year)
        if row.annual_return < 0.05 and row.max_drawdown > -0.08:
            source_mode = "aggressive_baseline"
            source_daily = baseline_daily
        elif row.annual_return < STATE_SWITCH_TRAIN_ANNUAL_THRESHOLD and row.recent_year_return < 0:
            source_mode = "low_risk_top3"
            source_daily = low_risk_top3_daily
        elif (
            row.annual_return >= STATE_SWITCH_STRONG_ANNUAL_THRESHOLD
            and row.recent_year_return >= STATE_SWITCH_ADAPTIVE_HOT_RECENT_THRESHOLD
        ):
            source_mode = "aggressive_baseline"
            source_daily = baseline_daily
        else:
            source_mode = STATE_SWITCH_RECENT_GUARD_MODE
            source_daily = switch_recent_guard_daily
        adaptive_mode_by_year[test_year] = source_mode
        part = source_daily[source_daily["test_year"] == test_year].copy()
        part["mode"] = STATE_SWITCH_ADAPTIVE_MIX_MODE
        part["source_mode"] = source_mode
        part["candidate"] = f"{STATE_SWITCH_ADAPTIVE_MIX_MODE}:{source_mode}"
        adaptive_parts.append(part)
    daily_parts[STATE_SWITCH_ADAPTIVE_MIX_MODE] = [pd.concat(adaptive_parts, ignore_index=True)]
    for test_year, group in switch_daily.groupby("test_year"):
        stats = annualized_stats(
            group["dt"].reset_index(drop=True), group["ret"].to_numpy(), group["weight"].to_numpy()
        )
        base_row = baseline_by_year[baseline_by_year["test_year"] == test_year].iloc[0].to_dict()
        switch_row = base_row.copy()
        switch_row["mode"] = STATE_SWITCH_MODE
        switch_row["switch_rule"] = f"train_annual_return>={STATE_SWITCH_TRAIN_ANNUAL_THRESHOLD:g}"
        switch_row["switch_weight_mdd"] = switch_weight_by_year[int(test_year)]
        switch_row.update({f"test_{key}": value for key, value in stats.items()})
        selected_rows.append(switch_row)
    for test_year, group in switch_scaled_daily.groupby("test_year"):
        stats = annualized_stats(
            group["dt"].reset_index(drop=True), group["ret"].to_numpy(), group["weight"].to_numpy()
        )
        base_row = baseline_by_year[baseline_by_year["test_year"] == test_year].iloc[0].to_dict()
        switch_row = base_row.copy()
        switch_row["mode"] = STATE_SWITCH_X110_MODE
        switch_row["switch_rule"] = f"train_annual_return>={STATE_SWITCH_TRAIN_ANNUAL_THRESHOLD:g}"
        switch_row["switch_weight_mdd"] = switch_weight_by_year[int(test_year)]
        switch_row["switch_scale"] = STATE_SWITCH_SCALE
        switch_row.update({f"test_{key}": value for key, value in stats.items()})
        selected_rows.append(switch_row)
    for test_year, group in switch_tiered_daily.groupby("test_year"):
        stats = annualized_stats(
            group["dt"].reset_index(drop=True), group["ret"].to_numpy(), group["weight"].to_numpy()
        )
        base_row = baseline_by_year[baseline_by_year["test_year"] == test_year].iloc[0].to_dict()
        switch_row = base_row.copy()
        switch_row["mode"] = STATE_SWITCH_TIERED_MODE
        switch_row["switch_rule"] = (
            f"weak={STATE_SWITCH_WEAK_SCALE:g},mid={STATE_SWITCH_MID_SCALE:g},"
            f"strong={STATE_SWITCH_STRONG_SCALE:g}@train_annual>={STATE_SWITCH_STRONG_ANNUAL_THRESHOLD:g}"
        )
        switch_row["switch_weight_mdd"] = switch_weight_by_year[int(test_year)]
        switch_row["switch_scale"] = tiered_scale_by_year[int(test_year)]
        switch_row.update({f"test_{key}": value for key, value in stats.items()})
        selected_rows.append(switch_row)
    for test_year, group in switch_recent_guard_daily.groupby("test_year"):
        stats = annualized_stats(
            group["dt"].reset_index(drop=True), group["ret"].to_numpy(), group["weight"].to_numpy()
        )
        base_row = baseline_by_year[baseline_by_year["test_year"] == test_year].iloc[0].to_dict()
        switch_row = base_row.copy()
        switch_row["mode"] = STATE_SWITCH_RECENT_GUARD_MODE
        switch_row["switch_rule"] = (
            f"weak={STATE_SWITCH_RECENT_GUARD_WEAK_SCALE:g},"
            f"mid={STATE_SWITCH_RECENT_GUARD_MID_SCALE:g},"
            f"strong_cool={STATE_SWITCH_RECENT_GUARD_STRONG_COOL_SCALE:g},"
            f"strong_hot={STATE_SWITCH_RECENT_GUARD_STRONG_HOT_SCALE:g}@recent>="
            f"{STATE_SWITCH_RECENT_GUARD_THRESHOLD:g}"
        )
        switch_row["switch_weight_mdd"] = switch_weight_by_year[int(test_year)]
        switch_row["switch_scale"] = recent_guard_scale_by_year[int(test_year)]
        switch_row.update({f"test_{key}": value for key, value in stats.items()})
        selected_rows.append(switch_row)
    for test_year, group in daily_parts[STATE_SWITCH_ADAPTIVE_MIX_MODE][0].groupby("test_year"):
        stats = annualized_stats(
            group["dt"].reset_index(drop=True), group["ret"].to_numpy(), group["weight"].to_numpy()
        )
        base_row = baseline_by_year[baseline_by_year["test_year"] == test_year].iloc[0].to_dict()
        switch_row = base_row.copy()
        switch_row["mode"] = STATE_SWITCH_ADAPTIVE_MIX_MODE
        switch_row["switch_rule"] = (
            "baseline_if_train_ann_lt_5_and_mdd_gt_-8;"
            "low_risk_if_train_ann_lt_17_and_recent_lt_0;"
            f"baseline_if_train_ann_ge_20_and_recent_ge_{STATE_SWITCH_ADAPTIVE_HOT_RECENT_THRESHOLD:g};"
            "else_recent_guard"
        )
        switch_row["source_mode"] = adaptive_mode_by_year[int(test_year)]
        switch_row.update({f"test_{key}": value for key, value in stats.items()})
        selected_rows.append(switch_row)

    all_daily = []
    summaries = []
    yearly_rows = []
    for mode, parts in daily_parts.items():
        daily = pd.concat(parts, ignore_index=True).sort_values("dt").reset_index(drop=True)
        daily["nav"] = (1 + daily["ret"]).cumprod()
        daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1
        all_daily.append(daily)
        summaries.append(summarize_daily(daily, mode))
        yearly = summarize_yearly(daily)
        yearly["mode"] = mode
        yearly_rows.append(yearly)

    all_daily_df = pd.concat(all_daily, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True).sort_values(["max_drawdown", "annual_return"], ascending=False)
    yearly = pd.concat(yearly_rows, ignore_index=True)

    pd.DataFrame(selected_rows).to_csv(
        OUTPUT_DIR / "selected_cross_section_by_fold.csv", index=False, encoding="utf-8-sig"
    )
    all_daily_df.to_csv(OUTPUT_DIR / "cross_section_daily_nav.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "cross_section_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUTPUT_DIR / "cross_section_yearly.csv", index=False, encoding="utf-8-sig")
    save_nav_plot(all_daily_df)

    print(summary.to_string(index=False))
    print(yearly[["mode", "year", "return", "max_drawdown", "annual_return", "calmar"]].to_string(index=False))
    print(f"outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
