"""Anti-overfitting checks for the commodity futures main-continuous CTA.

This script validates the optimized candidate in
``backtest_commodity_futures_cta_main_continuous.py`` with three checks:

1. walk-forward selection: train on prior years, select legs using train data only,
   then evaluate the next year out of sample;
2. parameter perturbation: perturb windows and risk parameters around the frozen
   optimized candidate;
3. leave-one stress: remove one optimized leg or one calendar year at a time.

Run:
    uv run --no-sync python examples/signals_dev/validate_commodity_futures_cta_anti_overfit.py
"""

from __future__ import annotations

from dataclasses import replace

import backtest_commodity_futures_cta_main_continuous as cta
import numpy as np
import pandas as pd

OUTPUT_DIR = cta.BASE_OUTPUT_DIR / "anti_overfit_validation"
WFO_TRAIN_TESTS = (
    ("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2021-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("2022-01-01", "2025-12-31", "2026-01-01", "2026-05-29"),
)
CANDIDATE_FAMILIES = ("trend", "rev", "ma_trend", "ma_rev")
CANDIDATE_WINDOWS = (10, 20, 40, 60, 90, 120, 180, 240)
CANDIDATE_LEVERAGE_CAPS = (0.75, 1.0, 1.5, 2.0, 3.0)
WFO_MAX_LEGS = 7


def load_cached_close() -> pd.DataFrame:
    """Load cached TQSDK adj-F daily close matrix without fetching data."""
    data = {}
    for symbol in cta.DEFAULT_SYMBOLS:
        file = cta.cache_file(symbol, cta.DEFAULT_START, cta.DEFAULT_END, cta.ADJ_TYPE)
        if not file.exists():
            continue
        bars = pd.read_csv(file, parse_dates=["dt"]).dropna(subset=["close"])
        if len(bars) >= 260:
            data[symbol] = bars.reset_index(drop=True)
    if len(data) < cta.MIN_SYMBOLS:
        raise RuntimeError(f"Not enough cached symbols: {len(data)}")
    close = cta.make_close_matrix(data)
    mask = (close.index >= pd.Timestamp(cta.DEFAULT_START)) & (close.index <= pd.Timestamp(cta.DEFAULT_END))
    return close.loc[mask]


def evaluate_weights(
    close: pd.DataFrame, weights: pd.DataFrame, config: cta.BacktestConfig
) -> tuple[pd.DataFrame, dict]:
    """Evaluate weights and return daily frame plus stats."""
    daily, stats = cta.evaluate(close, weights, config)
    stats["ok_20_10"] = bool(stats["annual_return"] >= 0.20 and stats["max_drawdown"] >= -0.10)
    return daily, stats


def build_custom_optimized_weights(
    close: pd.DataFrame,
    config: cta.BacktestConfig,
    legs: tuple[cta.OptimizedLeg, ...],
) -> pd.DataFrame:
    """Build optimized-style weights for a custom leg set."""
    returns = close.pct_change(fill_method=None).fillna(0.0)
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for leg in legs:
        if leg.symbol not in close.columns:
            continue
        vol = returns[leg.symbol].rolling(config.vol_window, min_periods=config.vol_window).std() * np.sqrt(252)
        leverage = (config.optimized_leg_target_vol / vol).clip(upper=leg.leverage_cap).fillna(0.0)
        signal = cta.directional_signal(close[leg.symbol], leg.family, leg.window)
        weights[leg.symbol] += signal * leverage / max(len(legs), 1)

    raw_returns = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    portfolio_vol = raw_returns.rolling(config.vol_window, min_periods=config.vol_window).std() * np.sqrt(252)
    overlay = (config.optimized_portfolio_target_vol / portfolio_vol).clip(
        upper=config.optimized_portfolio_leverage_cap
    )
    weights = weights.mul(overlay.shift(1).fillna(0.0), axis=0)
    gross = weights.abs().sum(axis=1)
    weights = weights.mul((config.optimized_gross_cap / gross).clip(upper=1.0).fillna(1.0), axis=0)
    return weights.where(close.notna(), 0.0)


def stats_on_mask(daily: pd.DataFrame, mask: pd.Series | np.ndarray) -> dict:
    """Calculate performance statistics on a subset of daily returns."""
    data = daily.loc[mask].copy().reset_index(drop=True)
    if data.empty:
        return {}
    nav = (1 + data["ret"]).cumprod()
    drawdown = nav / nav.cummax() - 1
    years = (data["dt"].iloc[-1] - data["dt"].iloc[0]).days / 365.25
    final_nav = float(nav.iloc[-1])
    annual_return = final_nav ** (1 / years) - 1 if years > 0 and final_nav > 0 else final_nav - 1
    max_drawdown = float(drawdown.min())
    daily_std = float(data["ret"].std())
    return {
        "start": str(data["dt"].iloc[0].date()),
        "end": str(data["dt"].iloc[-1].date()),
        "days": int(len(data)),
        "annual_return": annual_return,
        "cumulative_return": final_nav - 1,
        "final_nav": final_nav,
        "max_drawdown": max_drawdown,
        "calmar": annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan,
        "sharpe": float(data["ret"].mean() / daily_std * np.sqrt(252)) if daily_std > 0 else np.nan,
        "turnover": float(data["turnover"].sum()),
        "avg_gross": float(data["gross"].mean()),
        "max_gross": float(data["gross"].max()),
        "ok_20_10": bool(annual_return >= 0.20 and max_drawdown >= -0.10),
    }


def candidate_return_and_weight(
    close: pd.DataFrame, symbol: str, family: str, window: int, leverage_cap: float, config: cta.BacktestConfig
) -> tuple[pd.Series, pd.Series]:
    """Build one candidate leg's standalone return and weight."""
    returns = close.pct_change(fill_method=None).fillna(0.0)
    vol = returns[symbol].rolling(config.vol_window, min_periods=config.vol_window).std() * np.sqrt(252)
    leverage = (config.optimized_leg_target_vol / vol).clip(upper=leverage_cap).fillna(0.0)
    signal = cta.directional_signal(close[symbol], family, window)
    weight = (signal * leverage).fillna(0.0)
    turnover = weight.diff().abs().fillna(weight.abs())
    ret = weight.shift(1).fillna(0.0) * returns[symbol] - turnover * config.fee_rate
    return ret, weight


def rank_candidate_library(close: pd.DataFrame, train_mask: pd.Series, config: cta.BacktestConfig) -> list[dict]:
    """Rank single-leg candidates using training data only."""
    rows = []
    for symbol in close.columns:
        if close.loc[train_mask, symbol].notna().sum() < 520:
            continue
        for family in CANDIDATE_FAMILIES:
            for window in CANDIDATE_WINDOWS:
                for leverage_cap in CANDIDATE_LEVERAGE_CAPS:
                    ret, weight = candidate_return_and_weight(close, symbol, family, window, leverage_cap, config)
                    candidate_daily = pd.DataFrame(
                        {
                            "dt": close.index,
                            "ret": ret.to_numpy(),
                            "turnover": weight.diff().abs().fillna(weight.abs()).to_numpy(),
                            "gross": np.abs(weight.to_numpy()),
                        }
                    )
                    stats = stats_on_mask(candidate_daily, train_mask)
                    if not stats or stats["annual_return"] <= 0:
                        continue
                    stats["score"] = (
                        stats["annual_return"]
                        + 0.03 * stats["sharpe"]
                        - 0.40 * max(0.0, abs(stats["max_drawdown"]) - 0.18)
                        - 0.02 * max(0.0, stats["turnover"] / 100)
                    )
                    rows.append(
                        {
                            "symbol": symbol,
                            "family": family,
                            "window": window,
                            "leverage_cap": leverage_cap,
                            **stats,
                        }
                    )
    return sorted(rows, key=lambda x: x["score"], reverse=True)


def greedy_select_legs(
    close: pd.DataFrame, train_mask: pd.Series, ranked: list[dict], config: cta.BacktestConfig
) -> tuple[cta.OptimizedLeg, ...]:
    """Select a diversified leg set using training data only."""
    selected: list[cta.OptimizedLeg] = []
    used_symbols = set()
    while len(selected) < WFO_MAX_LEGS:
        best = None
        for row in ranked:
            if row["symbol"] in used_symbols:
                continue
            leg = cta.OptimizedLeg(row["symbol"], row["family"], int(row["window"]), float(row["leverage_cap"]))
            legs = tuple(selected + [leg])
            weights = build_custom_optimized_weights(close, config, legs)
            daily, _ = evaluate_weights(close, weights, config)
            stats = stats_on_mask(daily, train_mask)
            if not stats:
                continue
            objective = (
                stats["annual_return"]
                + 0.03 * stats["sharpe"]
                - 0.85 * max(0.0, abs(stats["max_drawdown"]) - 0.10)
                - 0.004 * len(legs)
            )
            if best is None or objective > best[0]:
                best = (objective, leg, stats)
        if best is None:
            break
        _, leg, _ = best
        selected.append(leg)
        used_symbols.add(leg.symbol)
    return tuple(selected)


def run_walk_forward(close: pd.DataFrame, config: cta.BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run train-only selection and next-period out-of-sample tests."""
    fold_rows = []
    selection_rows = []
    for fold_id, (train_start, train_end, test_start, test_end) in enumerate(WFO_TRAIN_TESTS, start=1):
        train_mask = (close.index >= pd.Timestamp(train_start)) & (close.index <= pd.Timestamp(train_end))
        test_mask = (close.index >= pd.Timestamp(test_start)) & (close.index <= pd.Timestamp(test_end))
        ranked = rank_candidate_library(close, train_mask, config)
        legs = greedy_select_legs(close, train_mask, ranked[:120], config)
        weights = build_custom_optimized_weights(close, config, legs)
        daily, _ = evaluate_weights(close, weights, config)
        train_stats = stats_on_mask(daily, train_mask)
        test_stats = stats_on_mask(daily, test_mask)
        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "selected_legs": len(legs),
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
            }
        )
        for rank, leg in enumerate(legs, start=1):
            selection_rows.append(
                {
                    "fold": fold_id,
                    "rank": rank,
                    "symbol": leg.symbol,
                    "family": leg.family,
                    "window": leg.window,
                    "leverage_cap": leg.leverage_cap,
                }
            )
    return pd.DataFrame(fold_rows), pd.DataFrame(selection_rows)


def run_parameter_perturbation(close: pd.DataFrame, config: cta.BacktestConfig) -> pd.DataFrame:
    """Perturb windows and risk parameters around the optimized candidate."""
    rows = []

    def add_case(name: str, case_config: cta.BacktestConfig, legs: tuple[cta.OptimizedLeg, ...]) -> None:
        weights = build_custom_optimized_weights(close, case_config, legs)
        _, stats = evaluate_weights(close, weights, case_config)
        rows.append({"case": name, **stats})

    add_case("baseline", config, cta.OPTIMIZED_LEGS)
    for scale in (0.8, 0.9, 1.1, 1.2):
        legs = tuple(replace(leg, window=max(5, int(round(leg.window * scale / 5) * 5))) for leg in cta.OPTIMIZED_LEGS)
        add_case(f"all_windows_x{scale:g}", config, legs)
    for leg_index, leg in enumerate(cta.OPTIMIZED_LEGS, start=1):
        for scale in (0.8, 1.2):
            legs = list(cta.OPTIMIZED_LEGS)
            legs[leg_index - 1] = replace(leg, window=max(5, int(round(leg.window * scale / 5) * 5)))
            add_case(f"leg{leg_index}_{leg.symbol}_window_x{scale:g}", config, tuple(legs))
    for target_vol in (0.16, 0.18, 0.22, 0.24):
        add_case(
            f"leg_target_vol_{target_vol:g}", replace(config, optimized_leg_target_vol=target_vol), cta.OPTIMIZED_LEGS
        )
    for target_vol in (0.12, 0.135, 0.165, 0.18):
        add_case(
            f"portfolio_target_vol_{target_vol:g}",
            replace(config, optimized_portfolio_target_vol=target_vol),
            cta.OPTIMIZED_LEGS,
        )
    for leverage_cap in (1.2, 1.35, 1.8):
        add_case(
            f"portfolio_leverage_cap_{leverage_cap:g}",
            replace(config, optimized_portfolio_leverage_cap=leverage_cap),
            cta.OPTIMIZED_LEGS,
        )
    for gross_cap in (1.5, 1.8, 2.2, 2.5):
        add_case(f"gross_cap_{gross_cap:g}", replace(config, optimized_gross_cap=gross_cap), cta.OPTIMIZED_LEGS)
    return pd.DataFrame(rows)


def run_leave_one_stress(close: pd.DataFrame, config: cta.BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run leave-one-leg and leave-one-year stress tests."""
    leg_rows = []
    for leg in cta.OPTIMIZED_LEGS:
        legs = tuple(x for x in cta.OPTIMIZED_LEGS if x != leg)
        weights = build_custom_optimized_weights(close, config, legs)
        _, stats = evaluate_weights(close, weights, config)
        leg_rows.append(
            {
                "removed_symbol": leg.symbol,
                "removed_family": leg.family,
                "removed_window": leg.window,
                **stats,
            }
        )

    weights = build_custom_optimized_weights(close, config, cta.OPTIMIZED_LEGS)
    daily, _ = evaluate_weights(close, weights, config)
    year_rows = []
    for year in sorted(daily["dt"].dt.year.unique()):
        mask = daily["dt"].dt.year != year
        stats = stats_on_mask(daily, mask)
        year_rows.append({"removed_year": int(year), **stats})
    return pd.DataFrame(leg_rows), pd.DataFrame(year_rows)


def summarize_validation(
    baseline: dict,
    walk_forward: pd.DataFrame,
    perturbation: pd.DataFrame,
    leave_leg: pd.DataFrame,
    leave_year: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact validation summary table."""
    rows = [
        {
            "check": "baseline_in_sample",
            "cases": 1,
            "pass_rate_20_10": float(baseline["ok_20_10"]),
            "min_annual_return": baseline["annual_return"],
            "median_annual_return": baseline["annual_return"],
            "min_max_drawdown": baseline["max_drawdown"],
            "worst_max_drawdown": baseline["max_drawdown"],
        },
        {
            "check": "walk_forward_oos",
            "cases": len(walk_forward),
            "pass_rate_20_10": float(walk_forward["test_ok_20_10"].mean()),
            "min_annual_return": float(walk_forward["test_annual_return"].min()),
            "median_annual_return": float(walk_forward["test_annual_return"].median()),
            "min_max_drawdown": float(walk_forward["test_max_drawdown"].max()),
            "worst_max_drawdown": float(walk_forward["test_max_drawdown"].min()),
        },
        {
            "check": "parameter_perturbation",
            "cases": len(perturbation),
            "pass_rate_20_10": float(perturbation["ok_20_10"].mean()),
            "min_annual_return": float(perturbation["annual_return"].min()),
            "median_annual_return": float(perturbation["annual_return"].median()),
            "min_max_drawdown": float(perturbation["max_drawdown"].max()),
            "worst_max_drawdown": float(perturbation["max_drawdown"].min()),
        },
        {
            "check": "leave_one_leg",
            "cases": len(leave_leg),
            "pass_rate_20_10": float(leave_leg["ok_20_10"].mean()),
            "min_annual_return": float(leave_leg["annual_return"].min()),
            "median_annual_return": float(leave_leg["annual_return"].median()),
            "min_max_drawdown": float(leave_leg["max_drawdown"].max()),
            "worst_max_drawdown": float(leave_leg["max_drawdown"].min()),
        },
        {
            "check": "leave_one_year",
            "cases": len(leave_year),
            "pass_rate_20_10": float(leave_year["ok_20_10"].mean()),
            "min_annual_return": float(leave_year["annual_return"].min()),
            "median_annual_return": float(leave_year["annual_return"].median()),
            "min_max_drawdown": float(leave_year["max_drawdown"].max()),
            "worst_max_drawdown": float(leave_year["max_drawdown"].min()),
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    """Run all anti-overfitting checks and save CSV outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = cta.BacktestConfig(signal_style="optimized")
    close = load_cached_close()

    baseline_weights = cta.build_optimized_weights(close, config)
    baseline_daily, baseline = evaluate_weights(close, baseline_weights, config)
    baseline_daily.to_csv(OUTPUT_DIR / "baseline_daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([baseline]).to_csv(OUTPUT_DIR / "baseline_summary.csv", index=False, encoding="utf-8-sig")

    walk_forward, wfo_selection = run_walk_forward(close, config)
    perturbation = run_parameter_perturbation(close, config)
    leave_leg, leave_year = run_leave_one_stress(close, config)
    summary = summarize_validation(baseline, walk_forward, perturbation, leave_leg, leave_year)

    walk_forward.to_csv(OUTPUT_DIR / "walk_forward_oos.csv", index=False, encoding="utf-8-sig")
    wfo_selection.to_csv(OUTPUT_DIR / "walk_forward_selected_legs.csv", index=False, encoding="utf-8-sig")
    perturbation.to_csv(OUTPUT_DIR / "parameter_perturbation.csv", index=False, encoding="utf-8-sig")
    leave_leg.to_csv(OUTPUT_DIR / "leave_one_leg.csv", index=False, encoding="utf-8-sig")
    leave_year.to_csv(OUTPUT_DIR / "leave_one_year.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "validation_summary.csv", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
