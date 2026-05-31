"""Anti-overfitting validation for the pure daily CTA on each symbol.

This script keeps the original walk-forward selector unchanged, then adds
post-selection checks that do not use future test-period data:

- each symbol is tested independently;
- each test year is selected from the previous four calendar years only;
- the selected daily strategy is frozen through the next test year;
- the stitched out-of-sample path is stressed with sub-windows and year drops;
- the selected risk parameters are perturbed locally to check parameter
  neighborhood stability.

Run:
    .venv/bin/python examples/signals_dev/validate_daily_cta_separate_anti_overfit.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_daily_cta_walk_forward import (  # noqa: E402
    DATA_FILES,
    FOLDS,
    Candidate,
    annualized_stats,
    build_leg_library,
    load_daily_data,
    period_returns,
    run_symbol_with_daily,
    select_candidate,
)

OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_separate_anti_overfit"
TARGET_ANNUAL_RETURN = 0.20
TARGET_MAX_DRAWDOWN = -0.10
TARGET_VOL_SCALES = (0.8, 0.9, 1.0, 1.1, 1.2)
CAP_SCALES = (0.8, 1.0, 1.2)
VOL_WINDOWS = (20, 40)


def add_pass_flags(stats: dict) -> dict:
    """Add target pass flags to one stats row."""
    stats["pass_20_10"] = bool(
        stats["annual_return"] >= TARGET_ANNUAL_RETURN and stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
    )
    return stats


def summarize_path(daily: pd.DataFrame, label: str, symbol: str, window_type: str) -> dict:
    """Summarize one daily return path."""
    stats = annualized_stats(daily["dt"].reset_index(drop=True), daily["ret"].to_numpy(), daily["weight"].to_numpy())
    stats.update(
        {
            "symbol": symbol,
            "window_type": window_type,
            "label": label,
            "start": str(daily["dt"].iloc[0].date()),
            "end": str(daily["dt"].iloc[-1].date()),
            "days": int(len(daily)),
        }
    )
    return add_pass_flags(stats)


def stress_windows(daily: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Build moving-window, truncated-end, and drop-year stress tests."""
    rows = [summarize_path(daily, "full_2020_2026", symbol, "full")]
    years = sorted(int(x) for x in daily["dt"].dt.year.unique())

    for year in years[:-1]:
        part = daily[daily["dt"].dt.year >= year].copy()
        if len(part) >= 120:
            rows.append(summarize_path(part, f"moving_start_{year}", symbol, "moving_start"))

    for year in years[1:]:
        part = daily[daily["dt"].dt.year <= year].copy()
        if len(part) >= 120:
            rows.append(summarize_path(part, f"moving_end_{year}", symbol, "moving_end"))

    for start_idx in range(len(years)):
        for end_idx in range(start_idx + 1, len(years)):
            if end_idx - start_idx + 1 < 2:
                continue
            start_year, end_year = years[start_idx], years[end_idx]
            part = daily[(daily["dt"].dt.year >= start_year) & (daily["dt"].dt.year <= end_year)].copy()
            if len(part) >= 240:
                rows.append(summarize_path(part, f"rolling_{start_year}_{end_year}", symbol, "rolling"))

    for year in years:
        part = daily[daily["dt"].dt.year != year].copy()
        if len(part) >= 240:
            rows.append(summarize_path(part, f"drop_{year}", symbol, "drop_year"))

    return pd.DataFrame(rows)


def selected_candidates(symbol: str) -> list[tuple[tuple[str, str, str, str], Candidate]]:
    """Recreate fold-level selected candidates with the original selector."""
    data = load_daily_data(symbol)
    legs = build_leg_library(data)
    selected = []
    for fold in FOLDS:
        row = select_candidate(data, symbol, legs, fold)
        selected.append((fold, row["_candidate_obj"]))
    return selected


def stitch_perturbed_path(
    symbol: str,
    selected: list[tuple[tuple[str, str, str, str], Candidate]],
    target_vol_scale: float,
    cap_scale: float,
    vol_window: int,
) -> pd.DataFrame:
    """Build one stitched OOS path from locally perturbed risk parameters."""
    data = load_daily_data(symbol)
    parts = []
    for fold, candidate in selected:
        perturbed = replace(
            candidate,
            vol_window=vol_window,
            target_vol=float(candidate.target_vol * target_vol_scale),
            leverage_cap=float(candidate.leverage_cap * cap_scale),
        )
        returns, weights, _ = period_returns(perturbed.structure.score, data, perturbed)
        test_start, test_end = pd.Timestamp(fold[2]), pd.Timestamp(fold[3])
        mask = ((data["dt"] >= test_start) & (data["dt"] <= test_end)).to_numpy()
        parts.append(
            pd.DataFrame(
                {
                    "dt": data.loc[mask, "dt"].to_numpy(),
                    "symbol": symbol,
                    "ret": returns[mask],
                    "weight": weights[mask],
                    "test_year": test_start.year,
                    "candidate": perturbed.name,
                    "target_vol_scale": target_vol_scale,
                    "cap_scale": cap_scale,
                    "vol_window": vol_window,
                }
            )
        )
    daily = pd.concat(parts, ignore_index=True).sort_values("dt").reset_index(drop=True)
    daily["nav"] = (1 + daily["ret"]).cumprod()
    daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1
    return daily


def perturbation_grid(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the local risk-parameter neighborhood of the selected path."""
    selected = selected_candidates(symbol)
    daily_rows = []
    summary_rows = []
    for vol_window in VOL_WINDOWS:
        for target_vol_scale in TARGET_VOL_SCALES:
            for cap_scale in CAP_SCALES:
                daily = stitch_perturbed_path(symbol, selected, target_vol_scale, cap_scale, vol_window)
                daily_rows.append(daily)
                label = f"VW{vol_window}_TVx{target_vol_scale:g}_CAPx{cap_scale:g}"
                stats = summarize_path(daily, label, symbol, "risk_parameter_perturbation")
                stats.update(
                    {
                        "vol_window": vol_window,
                        "target_vol_scale": target_vol_scale,
                        "cap_scale": cap_scale,
                    }
                )
                summary_rows.append(stats)
    return pd.DataFrame(summary_rows), pd.concat(daily_rows, ignore_index=True)


def save_nav_plot(daily: pd.DataFrame) -> None:
    """Save base out-of-sample net value lines."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for symbol, group in daily.groupby("symbol"):
        fig.add_trace(go.Scatter(x=group["dt"], y=group["nav"], mode="lines", name=symbol))
    fig.update_layout(
        title="Daily CTA independent walk-forward net value",
        xaxis_title="date",
        yaxis_title="net value",
        template="plotly_white",
    )
    fig.write_html(OUTPUT_DIR / "separate_walk_forward_nav.html", include_plotlyjs="cdn")


def main() -> None:
    """Run the independent anti-overfitting validation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    yearly_rows = []
    stress_rows = []
    perturb_rows = []
    daily_rows = []

    for symbol in DATA_FILES:
        summary, selected, yearly, daily = run_symbol_with_daily(symbol)
        out_dir = OUTPUT_DIR / symbol.replace(".", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "base_walk_forward_summary.csv", index=False, encoding="utf-8-sig")
        selected.to_csv(out_dir / "base_selected_params_by_year.csv", index=False, encoding="utf-8-sig")
        yearly.to_csv(out_dir / "base_walk_forward_yearly.csv", index=False, encoding="utf-8-sig")
        daily.to_csv(out_dir / "base_out_sample_daily_nav.csv", index=False, encoding="utf-8-sig")

        stress = stress_windows(daily, symbol)
        perturb, perturb_daily = perturbation_grid(symbol)
        stress.to_csv(out_dir / "window_stress.csv", index=False, encoding="utf-8-sig")
        perturb.to_csv(out_dir / "risk_parameter_perturbation.csv", index=False, encoding="utf-8-sig")
        perturb_daily.to_csv(out_dir / "risk_parameter_perturbation_daily.csv", index=False, encoding="utf-8-sig")

        summary = summary.copy()
        summary["pass_20_10"] = (summary["annual_return"] >= TARGET_ANNUAL_RETURN) & (
            summary["max_drawdown"] >= TARGET_MAX_DRAWDOWN
        )
        yearly = yearly.copy()
        yearly["symbol"] = symbol

        summary_rows.append(summary)
        yearly_rows.append(yearly)
        stress_rows.append(stress)
        perturb_rows.append(perturb)
        daily_rows.append(daily)

        print(f"\n== {symbol} ==")
        print(summary.to_string(index=False))
        print(yearly[["year", "return", "max_drawdown", "annual_return", "calmar"]].to_string(index=False))
        print(
            {
                "stress_pass_rate": float(stress["pass_20_10"].mean()),
                "perturbation_pass_rate": float(perturb["pass_20_10"].mean()),
                "perturbation_annual_median": float(perturb["annual_return"].median()),
                "perturbation_mdd_median": float(perturb["max_drawdown"].median()),
            }
        )

    summary_all = pd.concat(summary_rows, ignore_index=True)
    yearly_all = pd.concat(yearly_rows, ignore_index=True)
    stress_all = pd.concat(stress_rows, ignore_index=True)
    perturb_all = pd.concat(perturb_rows, ignore_index=True)
    daily_all = pd.concat(daily_rows, ignore_index=True)

    summary_all.to_csv(OUTPUT_DIR / "base_walk_forward_summary.csv", index=False, encoding="utf-8-sig")
    yearly_all.to_csv(OUTPUT_DIR / "base_walk_forward_yearly.csv", index=False, encoding="utf-8-sig")
    stress_all.to_csv(OUTPUT_DIR / "window_stress.csv", index=False, encoding="utf-8-sig")
    perturb_all.to_csv(OUTPUT_DIR / "risk_parameter_perturbation.csv", index=False, encoding="utf-8-sig")
    daily_all.to_csv(OUTPUT_DIR / "base_out_sample_daily_nav.csv", index=False, encoding="utf-8-sig")
    save_nav_plot(daily_all)

    aggregate = []
    for symbol, group in stress_all.groupby("symbol"):
        perturb = perturb_all[perturb_all["symbol"] == symbol]
        base = summary_all[summary_all["symbol"] == symbol].iloc[0]
        aggregate.append(
            {
                "symbol": symbol,
                "base_annual_return": base["annual_return"],
                "base_max_drawdown": base["max_drawdown"],
                "base_final_nav": base["final_nav"],
                "base_pass_20_10": bool(base["pass_20_10"]),
                "stress_windows": int(len(group)),
                "stress_pass_rate": float(group["pass_20_10"].mean()),
                "stress_annual_median": float(group["annual_return"].median()),
                "stress_mdd_median": float(group["max_drawdown"].median()),
                "perturbations": int(len(perturb)),
                "perturbation_pass_rate": float(perturb["pass_20_10"].mean()),
                "perturbation_annual_median": float(perturb["annual_return"].median()),
                "perturbation_mdd_median": float(perturb["max_drawdown"].median()),
            }
        )
    aggregate_out = pd.DataFrame(aggregate)
    aggregate_out.to_csv(OUTPUT_DIR / "anti_overfit_aggregate.csv", index=False, encoding="utf-8-sig")

    print("\nAggregate:")
    print(aggregate_out.to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
