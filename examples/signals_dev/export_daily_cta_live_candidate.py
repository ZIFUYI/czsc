# ruff: noqa: E402
"""Export the current daily CTA live candidate package.

The live candidate is ``cap15_harsh_realistic`` from
``validate_daily_cta_realistic_execution_stress.py``:

- 1.5x per-symbol net exposure cap;
- 10% no-trade band;
- 0.5 per-symbol daily turnover cap;
- tiered slippage;
- 10% annual short carry;
- 5bp extra cost on external overlay turnover.

Run:
    uv run --no-sync python examples/signals_dev/export_daily_cta_live_candidate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_daily_cta_cross_section_walk_forward as cs
from validate_daily_cta_layered_low_corr_overlay import (
    TARGET_ANNUAL_RETURN,
    TARGET_MAX_DRAWDOWN,
    evaluate_window_stress,
)
from validate_daily_cta_walk_forward import annualized_stats, summarize_yearly

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "examples" / "results" / "daily_cta_realistic_execution_stress"
SOURCE_DAILY = SOURCE_DIR / "realistic_execution_stress_daily.csv"
SOURCE_SUMMARY = SOURCE_DIR / "realistic_execution_stress_summary.csv"
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_live_candidate"
LIVE_SCHEME = "cap15_harsh_realistic"


def load_live_daily() -> pd.DataFrame:
    """Load the selected live-candidate path."""
    if not SOURCE_DAILY.exists():
        raise FileNotFoundError(f"Missing source daily file: {SOURCE_DAILY}")
    daily = pd.read_csv(SOURCE_DAILY, parse_dates=["dt"])
    daily["dt"] = pd.to_datetime(daily["dt"]).dt.tz_localize(None)
    live = daily[daily["scheme"].eq(LIVE_SCHEME)].copy()
    if live.empty:
        raise ValueError(f"No rows found for {LIVE_SCHEME} in {SOURCE_DAILY}")
    return live.sort_values("dt").reset_index(drop=True)


def live_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Build the one-row live-candidate summary."""
    if SOURCE_SUMMARY.exists():
        summary = pd.read_csv(SOURCE_SUMMARY)
        summary = summary[summary["scheme"].eq(LIVE_SCHEME)].copy()
    else:
        summary = pd.DataFrame()
    stats = annualized_stats(daily["dt"].reset_index(drop=True), daily["ret"].to_numpy(), daily["weight"].to_numpy())
    if summary.empty:
        summary = pd.DataFrame([{"scheme": LIVE_SCHEME}])
    for key, value in stats.items():
        summary[key] = value
    summary["pass_20_10"] = bool(
        stats["annual_return"] >= TARGET_ANNUAL_RETURN and stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
    )
    return summary


def orderable_weights(daily: pd.DataFrame) -> pd.DataFrame:
    """Export executable three-symbol holdings and daily trade deltas."""
    out = daily[["dt", "rebalance", "unfilled_drift", "core_weight", "external_raw_weight", "weight"]].copy()
    for symbol in cs.SYMBOLS:
        out[f"weight_{symbol}"] = daily[symbol].astype(float)
        out[f"trade_{symbol}"] = daily[symbol].astype(float).diff().fillna(0.0)
    out["trade_abs_sum"] = out[[f"trade_{symbol}" for symbol in cs.SYMBOLS]].abs().sum(axis=1)
    out["max_symbol_trade_abs"] = out[[f"trade_{symbol}" for symbol in cs.SYMBOLS]].abs().max(axis=1)
    out["net_core_weight"] = out[[f"weight_{symbol}" for symbol in cs.SYMBOLS]].sum(axis=1)
    out["gross_core_weight"] = out[[f"weight_{symbol}" for symbol in cs.SYMBOLS]].abs().sum(axis=1)
    return out


def top_rebalances(weights: pd.DataFrame) -> pd.DataFrame:
    """List the largest executable rebalance days."""
    cols = [
        "dt",
        "rebalance",
        "trade_abs_sum",
        "max_symbol_trade_abs",
        "gross_core_weight",
        "net_core_weight",
        *[f"weight_{symbol}" for symbol in cs.SYMBOLS],
        *[f"trade_{symbol}" for symbol in cs.SYMBOLS],
    ]
    return weights[weights["trade_abs_sum"].gt(0)].sort_values("trade_abs_sum", ascending=False)[cols].reset_index(drop=True)


def save_nav_plot(daily: pd.DataFrame) -> None:
    """Save final live-candidate net-value plot."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["dt"], y=daily["nav"], mode="lines", name=LIVE_SCHEME))
    fig.add_trace(go.Scatter(x=daily["dt"], y=1 + daily["drawdown"], mode="lines", name="1 + drawdown"))
    fig.update_layout(
        title="Daily CTA live candidate: cap15 harsh realistic",
        xaxis_title="date",
        yaxis_title="net value",
        template="plotly_white",
    )
    fig.write_html(OUTPUT_DIR / "live_candidate_nav.html", include_plotlyjs="cdn")


def save_requirements(summary: pd.DataFrame, weights: pd.DataFrame) -> None:
    """Save a concise execution requirements note."""
    latest = weights.iloc[-1]
    row = summary.iloc[0]
    lines = [
        "# Daily CTA live candidate",
        "",
        f"- Scheme: `{LIVE_SCHEME}`",
        "- Universe: `000852.XSHG`, `000905.XSHG`, `159915.XSHE` plus external overlay sleeves from the research path.",
        "- Core execution: 1.5x per-symbol net cap, 10% no-trade band, 0.5 max per-symbol daily turnover.",
        "- Cost stress included: tiered slippage, 10% annual short carry, 5bp external overlay turnover cost.",
        "- Hard dependency: strategy requires hedge / short capability or reliable substitutes; all-long-only failed the drawdown gate.",
        "",
        "## Performance",
        "",
        f"- Annual return: {row['annual_return']:.2%}",
        f"- Max drawdown: {row['max_drawdown']:.2%}",
        f"- Final NAV: {row['final_nav']:.3f}",
        f"- Pass 20% / 10% gate: {bool(row['pass_20_10'])}",
        "",
        "## Latest executable core weights",
        "",
        f"- Date: {latest['dt'].date()}",
        *[f"- `{symbol}`: {latest[f'weight_{symbol}']:.6f}" for symbol in cs.SYMBOLS],
        "",
        "Positive weights are long exposure. Negative weights require short / hedge implementation.",
    ]
    (OUTPUT_DIR / "live_candidate_execution_requirements.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Export live candidate artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = load_live_daily()
    summary = live_summary(daily)
    yearly = summarize_yearly(daily[["dt", "ret", "weight"]])
    windows = evaluate_window_stress(daily[["dt", "ret", "weight"]].copy(), LIVE_SCHEME)
    failed = windows[~windows["pass_20_10"]].sort_values("annual_return")
    weights = orderable_weights(daily)
    rebalances = top_rebalances(weights)

    daily.to_csv(OUTPUT_DIR / "live_candidate_daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "live_candidate_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUTPUT_DIR / "live_candidate_yearly.csv", index=False, encoding="utf-8-sig")
    windows.to_csv(OUTPUT_DIR / "live_candidate_window_stress.csv", index=False, encoding="utf-8-sig")
    failed.to_csv(OUTPUT_DIR / "live_candidate_failed_windows.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(OUTPUT_DIR / "live_candidate_three_symbol_weights.csv", index=False, encoding="utf-8-sig")
    rebalances.to_csv(OUTPUT_DIR / "live_candidate_rebalances.csv", index=False, encoding="utf-8-sig")
    weights.tail(1).to_csv(OUTPUT_DIR / "live_candidate_latest_weights.csv", index=False, encoding="utf-8-sig")
    save_nav_plot(daily)
    save_requirements(summary, weights)

    print("Live candidate summary:")
    print(summary[["scheme", "annual_return", "max_drawdown", "final_nav", "pass_20_10"]].to_string(index=False))
    print("\nYearly:")
    print(yearly[["year", "return", "max_drawdown", "annual_return", "max_abs_weight"]].to_string(index=False))
    print("\nTop failed windows:")
    print(failed[["window", "annual_return", "max_drawdown"]].head(12).to_string(index=False))
    print("\nLatest weights:")
    print(weights.tail(1).to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
