"""Portfolio walk-forward validation for pure daily CTA variants.

This script extends ``validate_daily_cta_walk_forward.py`` by making the
out-of-sample decision at the portfolio level:

- every fold still selects symbol candidates from the previous four years only;
- top-N candidate ensembles are used to reduce single-parameter selection risk;
- portfolio variants are stitched from frozen test-year returns;
- no test-year information is used by candidate selection or symbol gating.

Run:
    .venv/bin/python examples/signals_dev/validate_daily_cta_portfolio_walk_forward.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from validate_daily_cta_walk_forward import (
    DATA_FILES,
    FOLDS,
    LEVERAGE_CAPS,
    TARGET_VOLS,
    VOL_WINDOWS,
    Candidate,
    annualized_stats,
    build_leg_library,
    evaluate_candidate,
    load_daily_data,
    make_structures,
    period_returns,
    rank_legs,
    recent_year_return,
    summarize_yearly,
    yearly_loss_count,
)
from validate_daily_cta_walk_forward import (
    OUTPUT_DIR as SYMBOL_WFO_DIR,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_portfolio_walk_forward"
ENSEMBLE_TOP_N = 8


def build_candidate_rank(data: pd.DataFrame, symbol: str, legs: dict[str, np.ndarray], fold: tuple[str, str, str, str]):
    """Rank all candidates using training data only."""
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

    turnover_limit = ranked["turnover"].quantile(0.7)
    eligible = ranked[
        (ranked["annual_return"] >= 0.12)
        & (ranked["max_drawdown"] >= -0.14)
        & (ranked["loss_years"] <= 1)
        & (ranked["max_abs_weight"] <= 2.5)
        & (ranked["turnover"] <= turnover_limit)
    ].copy()
    if eligible.empty:
        eligible = ranked[(ranked["annual_return"] > 0) & (ranked["max_drawdown"] >= -0.20)].copy()
    if eligible.empty:
        eligible = ranked.copy()

    ranked = ranked.sort_values(["score", "calmar", "annual_return"], ascending=False).reset_index(drop=True)
    eligible = eligible.sort_values(["score", "calmar", "annual_return"], ascending=False).reset_index(drop=True)
    return ranked, eligible, returns_cache, train_mask


def select_symbol_ensemble(
    data: pd.DataFrame, symbol: str, legs: dict[str, np.ndarray], fold: tuple[str, str, str, str]
) -> dict:
    """Select one top-N ensemble for a symbol using training data only."""
    train_start, train_end, test_start, test_end = fold
    ranked, eligible, returns_cache, train_mask = build_candidate_rank(data, symbol, legs, fold)
    top = eligible.head(ENSEMBLE_TOP_N).copy()

    top_returns = []
    top_weights = []
    for candidate_name in top["candidate"]:
        _, returns, weights = returns_cache[candidate_name]
        top_returns.append(returns)
        top_weights.append(weights)
    ensemble_returns = np.vstack(top_returns).mean(axis=0)
    ensemble_weights = np.vstack(top_weights).mean(axis=0)

    train_stats = annualized_stats(
        data.loc[train_mask, "dt"].reset_index(drop=True), ensemble_returns[train_mask], ensemble_weights[train_mask]
    )
    train_stats["loss_years"] = yearly_loss_count(data, ensemble_returns, train_mask)
    train_stats["recent_year_return"] = recent_year_return(data, ensemble_returns, train_mask)
    train_stats["score"] = (
        train_stats["calmar"]
        + 0.5 * train_stats["recent_year_return"]
        - 2.5 * max(0.0, abs(train_stats["max_drawdown"]) - 0.10)
    )

    top1 = top.iloc[0].to_dict()
    top1_returns = returns_cache[top1["candidate"]][1]
    top1_weights = returns_cache[top1["candidate"]][2]

    return {
        "symbol": symbol,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "candidate": f"ENS{len(top)}__{top.iloc[0]['candidate']}",
        "top_candidates": "|".join(top["candidate"].head(3).astype(str)),
        "selected_candidates": len(ranked),
        "eligible_candidates": len(eligible),
        "annual_return": train_stats["annual_return"],
        "cumulative_return": train_stats["cumulative_return"],
        "max_drawdown": train_stats["max_drawdown"],
        "calmar": train_stats["calmar"],
        "loss_years": train_stats["loss_years"],
        "recent_year_return": train_stats["recent_year_return"],
        "avg_abs_weight": train_stats["avg_abs_weight"],
        "max_abs_weight": train_stats["max_abs_weight"],
        "score": train_stats["score"],
        "_returns": ensemble_returns,
        "_weights": ensemble_weights,
        "_top1_returns": top1_returns,
        "_top1_weights": top1_weights,
        "_top1_candidate": top1["candidate"],
    }


def passes_symbol_gate(row: dict) -> bool:
    """Symbol inclusion gate based only on training-period ensemble behavior."""
    return bool(
        row["annual_return"] >= 0.18
        and row["max_drawdown"] >= -0.11
        and row["loss_years"] == 0
        and row["recent_year_return"] >= 0.15
        and row["eligible_candidates"] >= ENSEMBLE_TOP_N
    )


def candidate_allocations(symbols: list[str]) -> list[dict[str, float]]:
    """Build simple long-only capital allocations across symbol strategies."""
    allocations = [dict.fromkeys(symbols, 0.0)]
    for size in range(1, len(symbols) + 1):
        for subset in itertools.combinations(symbols, size):
            weight = 1.0 / size
            allocations.append({symbol: (weight if symbol in subset else 0.0) for symbol in symbols})
    return allocations


def portfolio_stats_for_allocation(
    dates: pd.Series,
    returns_by_symbol: dict[str, np.ndarray],
    weights_by_symbol: dict[str, np.ndarray],
    alloc: dict[str, float],
    mask: np.ndarray,
) -> dict:
    """Evaluate a portfolio allocation on one date mask."""
    returns = np.zeros(mask.sum(), dtype=float)
    exposure = np.zeros(mask.sum(), dtype=float)
    for symbol, weight in alloc.items():
        returns += weight * returns_by_symbol[symbol][mask]
        exposure += weight * np.abs(weights_by_symbol[symbol][mask])
    return annualized_stats(dates.loc[mask].reset_index(drop=True), returns, exposure)


def choose_train_allocation(
    dates: pd.Series,
    selections: list[dict],
    returns_by_symbol: dict[str, np.ndarray],
    weights_by_symbol: dict[str, np.ndarray],
    train_mask: np.ndarray,
) -> tuple[str, dict[str, float], dict]:
    """Choose a frozen symbol allocation using training data only."""
    symbols = [row["symbol"] for row in selections]
    rows = []
    for alloc in candidate_allocations(symbols):
        stats = portfolio_stats_for_allocation(dates, returns_by_symbol, weights_by_symbol, alloc, train_mask)
        if not stats:
            continue
        active = [symbol for symbol, weight in alloc.items() if weight > 0]
        stats["active_symbols"] = ",".join(active) if active else "FLAT"
        stats["allocation"] = alloc
        stats["loss_years"] = portfolio_loss_years(dates, returns_by_symbol, alloc, train_mask)
        stats["score"] = (
            stats["calmar"]
            + 0.25 * stats["annual_return"]
            - 2.5 * max(0.0, abs(stats["max_drawdown"]) - 0.10)
            - 0.02 * max(0, len(active) - 1)
        )
        rows.append(stats)

    ranked = pd.DataFrame(rows)
    eligible = ranked[
        (ranked["annual_return"] >= 0.12)
        & (ranked["max_drawdown"] >= -0.11)
        & (ranked["loss_years"] == 0)
        & (ranked["active_symbols"] != "FLAT")
    ].copy()
    if eligible.empty:
        eligible = ranked[(ranked["annual_return"] > 0) & (ranked["max_drawdown"] >= -0.16)].copy()
    if eligible.empty:
        chosen = ranked[ranked["active_symbols"] == "FLAT"].iloc[0].to_dict()
    else:
        chosen = eligible.sort_values(["score", "calmar", "annual_return"], ascending=False).iloc[0].to_dict()
    return str(chosen["active_symbols"]), chosen["allocation"], chosen


def portfolio_loss_years(
    dates: pd.Series, returns_by_symbol: dict[str, np.ndarray], alloc: dict[str, float], mask: np.ndarray
) -> int:
    """Count losing calendar years for a portfolio allocation."""
    returns = np.zeros(mask.sum(), dtype=float)
    for symbol, weight in alloc.items():
        returns += weight * returns_by_symbol[symbol][mask]
    part = pd.DataFrame({"dt": dates.loc[mask].to_numpy(), "ret": returns})
    losses = 0
    for _, group in part.groupby(pd.to_datetime(part["dt"]).dt.year):
        nav = np.cumprod(1 + group["ret"].to_numpy())
        losses += int(nav[-1] < 1)
    return losses


def append_fold_daily(
    daily_parts: dict[str, list[pd.DataFrame]],
    mode: str,
    data: pd.DataFrame,
    test_mask: np.ndarray,
    returns_by_symbol: dict[str, np.ndarray],
    weights_by_symbol: dict[str, np.ndarray],
    alloc: dict[str, float],
    test_year: int,
) -> None:
    """Append one fold portfolio daily return series."""
    returns = np.zeros(test_mask.sum(), dtype=float)
    exposure = np.zeros(test_mask.sum(), dtype=float)
    for symbol, weight in alloc.items():
        returns += weight * returns_by_symbol[symbol][test_mask]
        exposure += weight * np.abs(weights_by_symbol[symbol][test_mask])
    daily_parts[mode].append(
        pd.DataFrame(
            {
                "dt": data.loc[test_mask, "dt"].to_numpy(),
                "mode": mode,
                "ret": returns,
                "weight": exposure,
                "test_year": test_year,
                "active_symbols": ",".join([s for s, w in alloc.items() if w > 0]) or "FLAT",
            }
        )
    )


def summarize_portfolio(daily: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Summarize stitched portfolio out-of-sample performance."""
    stats = annualized_stats(daily["dt"], daily["ret"].to_numpy(), daily["weight"].to_numpy())
    stats["mode"] = mode
    stats["start"] = str(daily["dt"].iloc[0].date())
    stats["end"] = str(daily["dt"].iloc[-1].date())
    stats["loss_years"] = int((summarize_yearly(daily)["return"] < 0).sum())
    return pd.DataFrame([stats])


def save_nav_plot(all_daily: pd.DataFrame) -> None:
    """Save a net-value comparison plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import plotly.graph_objects as go

        fig = go.Figure()
        for mode, group in all_daily.groupby("mode"):
            fig.add_trace(go.Scatter(x=group["dt"], y=group["nav"], mode="lines", name=mode))
        fig.update_layout(
            title="Daily CTA portfolio walk-forward net value",
            xaxis_title="date",
            yaxis_title="net value",
            template="plotly_white",
        )
        fig.write_html(OUTPUT_DIR / "portfolio_walk_forward_nav.html", include_plotlyjs="cdn")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    for mode, group in all_daily.groupby("mode"):
        ax.plot(group["dt"], group["nav"], label=mode, linewidth=1.4)
    ax.set_title("Daily CTA portfolio walk-forward net value")
    ax.set_xlabel("date")
    ax.set_ylabel("net value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "portfolio_walk_forward_nav.svg")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_by_symbol = {symbol: load_daily_data(symbol) for symbol in DATA_FILES}
    legs_by_symbol = {symbol: build_leg_library(data) for symbol, data in data_by_symbol.items()}
    base_data = data_by_symbol[next(iter(DATA_FILES))]
    symbols = list(DATA_FILES)

    selection_rows = []
    allocation_rows = []
    daily_parts = {
        "top1_all_equal": [],
        "ensemble_all_equal": [],
        "ensemble_gate_equal": [],
        "ensemble_train_alloc": [],
    }

    for fold in FOLDS:
        train_start, train_end, test_start, test_end = fold
        train_mask = (
            (base_data["dt"] >= pd.Timestamp(train_start)) & (base_data["dt"] <= pd.Timestamp(train_end))
        ).to_numpy()
        test_mask = (
            (base_data["dt"] >= pd.Timestamp(test_start)) & (base_data["dt"] <= pd.Timestamp(test_end))
        ).to_numpy()
        selections = []
        for symbol in symbols:
            selected = select_symbol_ensemble(data_by_symbol[symbol], symbol, legs_by_symbol[symbol], fold)
            selected["pass_gate"] = passes_symbol_gate(selected)
            selection_rows.append({k: v for k, v in selected.items() if not k.startswith("_")})
            selections.append(selected)

        ens_returns = {row["symbol"]: row["_returns"] for row in selections}
        ens_weights = {row["symbol"]: row["_weights"] for row in selections}
        top1_returns = {row["symbol"]: row["_top1_returns"] for row in selections}
        top1_weights = {row["symbol"]: row["_top1_weights"] for row in selections}

        equal_alloc = {symbol: 1.0 / len(symbols) for symbol in symbols}
        gate_symbols = [row["symbol"] for row in selections if row["pass_gate"]]
        gate_alloc = {symbol: (1.0 / len(gate_symbols) if symbol in gate_symbols else 0.0) for symbol in symbols}
        if not gate_symbols:
            gate_alloc = dict.fromkeys(symbols, 0.0)
        active_name, train_alloc, train_alloc_stats = choose_train_allocation(
            base_data["dt"], selections, ens_returns, ens_weights, train_mask
        )

        modes = [
            ("top1_all_equal", top1_returns, top1_weights, equal_alloc),
            ("ensemble_all_equal", ens_returns, ens_weights, equal_alloc),
            ("ensemble_gate_equal", ens_returns, ens_weights, gate_alloc),
            ("ensemble_train_alloc", ens_returns, ens_weights, train_alloc),
        ]
        for mode, returns_by_symbol, weights_by_symbol, alloc in modes:
            append_fold_daily(
                daily_parts,
                mode,
                base_data,
                test_mask,
                returns_by_symbol,
                weights_by_symbol,
                alloc,
                pd.Timestamp(test_start).year,
            )
            allocation_rows.append(
                {
                    "mode": mode,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "active_symbols": ",".join([s for s, w in alloc.items() if w > 0]) or "FLAT",
                    "train_selected_active": active_name if mode == "ensemble_train_alloc" else "",
                    "train_annual_return": train_alloc_stats.get("annual_return", np.nan)
                    if mode == "ensemble_train_alloc"
                    else np.nan,
                    "train_max_drawdown": train_alloc_stats.get("max_drawdown", np.nan)
                    if mode == "ensemble_train_alloc"
                    else np.nan,
                    "train_calmar": train_alloc_stats.get("calmar", np.nan)
                    if mode == "ensemble_train_alloc"
                    else np.nan,
                    **{f"alloc_{symbol}": alloc[symbol] for symbol in symbols},
                }
            )

    all_daily = []
    summaries = []
    yearly_rows = []
    for mode, parts in daily_parts.items():
        daily = pd.concat(parts, ignore_index=True).sort_values("dt").reset_index(drop=True)
        daily["nav"] = (1 + daily["ret"]).cumprod()
        daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1
        all_daily.append(daily)
        summaries.append(summarize_portfolio(daily, mode))
        yearly = summarize_yearly(daily)
        yearly["mode"] = mode
        yearly_rows.append(yearly)

    all_daily_df = pd.concat(all_daily, ignore_index=True)
    summary_df = pd.concat(summaries, ignore_index=True).sort_values("calmar", ascending=False)
    yearly_df = pd.concat(yearly_rows, ignore_index=True)

    pd.DataFrame(selection_rows).to_csv(
        OUTPUT_DIR / "selected_symbol_ensembles_by_fold.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(allocation_rows).to_csv(
        OUTPUT_DIR / "portfolio_allocations_by_fold.csv", index=False, encoding="utf-8-sig"
    )
    all_daily_df.to_csv(OUTPUT_DIR / "portfolio_daily_nav.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_DIR / "portfolio_summary.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(OUTPUT_DIR / "portfolio_yearly.csv", index=False, encoding="utf-8-sig")
    save_nav_plot(all_daily_df)

    print(f"symbol walk-forward source: {SYMBOL_WFO_DIR}")
    print(summary_df.to_string(index=False))
    print(f"outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
