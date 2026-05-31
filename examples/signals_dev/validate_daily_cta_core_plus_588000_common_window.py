# ruff: noqa: E402
"""Compare the core daily CTA with and without 588000.XSHG.

588000.XSHG only has valid local daily data from 2020-11-16, so it cannot be
inserted into the original 2020-start 4-year WFO without changing the training
history. This script uses the common feasible WFO window:

- 2024 test: train from 2020-11-16 to 2023-12-31;
- 2025 test: train from 2021-01-01 to 2024-12-31;
- 2026 test: train from 2022-01-01 to 2025-12-31.

It runs the same cross-sectional daily CTA selection code for:

- baseline: 000852 / 000905 / 159915;
- plus_588000: baseline + 588000.

Run:
    uv run --no-sync python examples/signals_dev/validate_daily_cta_core_plus_588000_common_window.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_daily_cta_cross_section_walk_forward as xsec
from validate_daily_cta_walk_forward import DATA_FILES as BASE_DATA_FILES
from validate_daily_cta_walk_forward import annualized_stats, summarize_yearly

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_core_plus_588000_common_window"
DATA_588000 = ROOT / "examples" / "results" / "daily_holding_core_plus_588000" / "588000_XSHG_daily_20100101_20260529.csv"
COMMON_FOLDS = [
    ("2020-11-16", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2021-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("2022-01-01", "2025-12-31", "2026-01-01", "2026-05-29"),
]
ORIGINAL_LOAD_PANEL = xsec.load_panel
ORIGINAL_SYMBOLS = xsec.SYMBOLS
ORIGINAL_FOLDS = xsec.FOLDS
ORIGINAL_OUTPUT_DIR = xsec.OUTPUT_DIR
ORIGINAL_MODES = xsec.MODES
ORIGINAL_RAW_WEIGHTS_FROM_SCORES = xsec.raw_weights_from_scores


def load_one_symbol(symbol: str, file_csv: Path) -> pd.DataFrame:
    """Load one daily close series and drop pre-listing empty rows."""
    if not file_csv.exists():
        raise FileNotFoundError(f"Missing daily data file for {symbol}: {file_csv}")
    data = pd.read_csv(file_csv, parse_dates=["dt"])
    data["dt"] = pd.to_datetime(data["dt"]).dt.tz_localize(None)
    data = data.dropna(subset=["close"]).drop_duplicates("dt").sort_values("dt")
    return data[["dt", "close"]].rename(columns={"close": symbol}).reset_index(drop=True)


def make_panel_loader(data_files: dict[str, Path]):
    """Create a close-price panel loader for a fixed universe."""

    def load_panel() -> pd.DataFrame:
        frames = [load_one_symbol(symbol, file_csv) for symbol, file_csv in data_files.items()]
        panel = frames[0]
        for frame in frames[1:]:
            panel = panel.merge(frame, on="dt", how="inner")
        return panel.sort_values("dt").reset_index(drop=True)

    return load_panel


def run_universe(label: str, data_files: dict[str, Path]) -> Path:
    """Run the original cross-sectional WFO for one universe."""
    out_dir = OUTPUT_DIR / label
    xsec.SYMBOLS = tuple(data_files)
    xsec.FOLDS = COMMON_FOLDS
    xsec.OUTPUT_DIR = out_dir
    xsec.MODES = ("rank_ls", "rank_long", "market_switch", "trend_equal")
    xsec.raw_weights_from_scores = ORIGINAL_RAW_WEIGHTS_FROM_SCORES
    xsec.load_panel = make_panel_loader(data_files)
    xsec.main()
    return out_dir


def restore_xsec_globals() -> None:
    """Restore global state in the imported cross-sectional module."""
    xsec.load_panel = ORIGINAL_LOAD_PANEL
    xsec.SYMBOLS = ORIGINAL_SYMBOLS
    xsec.FOLDS = ORIGINAL_FOLDS
    xsec.OUTPUT_DIR = ORIGINAL_OUTPUT_DIR
    xsec.MODES = ORIGINAL_MODES
    xsec.raw_weights_from_scores = ORIGINAL_RAW_WEIGHTS_FROM_SCORES


def summarize_reference_core() -> pd.DataFrame:
    """Summarize the existing final core-only path over the same test years."""
    daily_file = ROOT / "examples" / "results" / "daily_cta_live_candidate" / "live_candidate_daily.csv"
    daily = pd.read_csv(daily_file, parse_dates=["dt"])
    daily["dt"] = pd.to_datetime(daily["dt"]).dt.tz_localize(None)
    part = daily[(daily["dt"] >= "2024-01-01") & (daily["dt"] <= "2026-05-29")].copy()
    stats = annualized_stats(part["dt"].reset_index(drop=True), part["core_ret"].to_numpy(), part["core_weight"].to_numpy())
    stats["universe"] = "current_final_core_reference"
    stats["mode"] = "cap15_harsh_realistic_core_only"
    stats["start"] = str(part["dt"].iloc[0].date())
    stats["end"] = str(part["dt"].iloc[-1].date())
    stats["loss_years"] = int((summarize_yearly(part.rename(columns={"core_ret": "ret", "core_weight": "weight"}))["return"] < 0).sum())
    return pd.DataFrame([stats])


def build_comparison(result_dirs: dict[str, Path]) -> None:
    """Build summary, yearly and focused mode comparisons."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_frames = []
    yearly_frames = []
    daily_frames = []
    for label, out_dir in result_dirs.items():
        summary = pd.read_csv(out_dir / "cross_section_summary.csv")
        summary["universe"] = label
        yearly = pd.read_csv(out_dir / "cross_section_yearly.csv")
        yearly["universe"] = label
        daily = pd.read_csv(out_dir / "cross_section_daily_nav.csv", parse_dates=["dt"])
        daily["universe"] = label
        summary_frames.append(summary)
        yearly_frames.append(yearly)
        daily_frames.append(daily)

    comparison = pd.concat(summary_frames, ignore_index=True)
    yearly_comparison = pd.concat(yearly_frames, ignore_index=True)
    daily_comparison = pd.concat(daily_frames, ignore_index=True)
    reference = summarize_reference_core()

    focus_modes = [
        "aggressive_state_switch_adaptive_mix",
        "aggressive_state_switch_recent_guard",
        "aggressive_baseline",
        "aggressive_brake_blend_50",
        "low_risk_top3",
    ]
    focus = comparison[comparison["mode"].isin(focus_modes)].copy()
    focus = pd.concat([focus, reference], ignore_index=True, sort=False)
    focus = focus.sort_values(["mode", "universe"]).reset_index(drop=True)

    wide_rows = []
    for mode in focus_modes:
        base = focus[(focus["universe"] == "baseline_3_symbol") & (focus["mode"] == mode)]
        plus = focus[(focus["universe"] == "plus_588000") & (focus["mode"] == mode)]
        if base.empty or plus.empty:
            continue
        base_row = base.iloc[0]
        plus_row = plus.iloc[0]
        wide_rows.append(
            {
                "mode": mode,
                "baseline_annual_return": base_row["annual_return"],
                "plus_588000_annual_return": plus_row["annual_return"],
                "annual_return_delta": plus_row["annual_return"] - base_row["annual_return"],
                "baseline_max_drawdown": base_row["max_drawdown"],
                "plus_588000_max_drawdown": plus_row["max_drawdown"],
                "max_drawdown_delta": plus_row["max_drawdown"] - base_row["max_drawdown"],
                "baseline_final_nav": base_row["final_nav"],
                "plus_588000_final_nav": plus_row["final_nav"],
                "final_nav_delta": plus_row["final_nav"] - base_row["final_nav"],
                "baseline_max_abs_weight": base_row["max_abs_weight"],
                "plus_588000_max_abs_weight": plus_row["max_abs_weight"],
            }
        )
    wide = pd.DataFrame(wide_rows)

    comparison.to_csv(OUTPUT_DIR / "core_plus_588000_all_summary.csv", index=False, encoding="utf-8-sig")
    yearly_comparison.to_csv(OUTPUT_DIR / "core_plus_588000_yearly.csv", index=False, encoding="utf-8-sig")
    daily_comparison.to_csv(OUTPUT_DIR / "core_plus_588000_daily_nav.csv", index=False, encoding="utf-8-sig")
    focus.to_csv(OUTPUT_DIR / "core_plus_588000_focus_summary.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(OUTPUT_DIR / "core_plus_588000_mode_delta.csv", index=False, encoding="utf-8-sig")

    print("\nFocused comparison:")
    print(
        focus[
            [
                "universe",
                "mode",
                "annual_return",
                "max_drawdown",
                "calmar",
                "final_nav",
                "avg_abs_weight",
                "max_abs_weight",
                "loss_years",
            ]
        ].to_string(index=False)
    )
    print("\nMode deltas:")
    print(wide.to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


def main() -> None:
    """Run the common-window comparison."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = dict(BASE_DATA_FILES)
    plus_588000 = {**baseline, "588000.XSHG": DATA_588000}
    result_dirs: dict[str, Path] = {}
    try:
        result_dirs["baseline_3_symbol"] = run_universe("baseline_3_symbol", baseline)
        result_dirs["plus_588000"] = run_universe("plus_588000", plus_588000)
    finally:
        restore_xsec_globals()
    build_comparison(result_dirs)


if __name__ == "__main__":
    main()
