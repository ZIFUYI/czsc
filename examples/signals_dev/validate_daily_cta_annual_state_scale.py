"""Annual training-state risk scaling for the current best daily CTA mix.

The script starts from the aggressive market-state overlay and applies a yearly
scale selected only from each fold's training statistics. The purpose is to use
more risk budget in years where the training window was weak-but-controlled,
while leaving already hot / high-drawdown training states unchanged.

Run:
    .venv/bin/python examples/signals_dev/validate_daily_cta_annual_state_scale.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from validate_daily_cta_walk_forward import annualized_stats, summarize_yearly

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "examples" / "results" / "daily_cta_cross_section_walk_forward"
OUTPUT_DIR = INPUT_DIR / "annual_state_scale"
BASE_DAILY_FILE = INPUT_DIR / "market_state_overlay" / "market_state_overlay_daily.csv"
SELECTED_FILE = INPUT_DIR / "selected_cross_section_by_fold.csv"
BASE_PRESET = "aggressive"
BASE_MODE = "aggressive_state_switch_adaptive_mix"

PRESETS = {
    "capital_efficient": {
        "boost_scale": 1.20,
        "train_annual_max": 0.10,
        "recent_return_max": 0.35,
        "train_drawdown_min": -0.14,
    },
    "cautious": {
        "boost_scale": 1.05,
        "train_annual_max": 0.13,
        "recent_return_max": 0.35,
        "train_drawdown_min": -0.14,
    },
    "balanced": {
        "boost_scale": 1.10,
        "train_annual_max": 0.13,
        "recent_return_max": 0.35,
        "train_drawdown_min": -0.14,
    },
    "stress_balanced": {
        "boost_scale": 1.125,
        "train_annual_max": 0.13,
        "recent_return_max": 0.35,
        "train_drawdown_min": -0.14,
    },
    "aggressive": {
        "boost_scale": 1.20,
        "train_annual_max": 0.13,
        "recent_return_max": 0.35,
        "train_drawdown_min": -0.14,
    },
}
ROBUSTNESS_GRID = {
    "boost_scale": (1.05, 1.075, 1.10, 1.125, 1.15, 1.175, 1.20),
    "train_annual_max": (0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18),
    "recent_return_max": (0.20, 0.25, 0.30, 0.35, 0.40, 0.45),
    "train_drawdown_min": (-0.10, -0.12, -0.14, -0.16),
}


def load_base_daily() -> pd.DataFrame:
    """Load the aggressive market-state overlay daily returns."""
    daily = pd.read_csv(BASE_DAILY_FILE, parse_dates=["dt"])
    daily = daily[daily["preset"] == BASE_PRESET].sort_values("dt").reset_index(drop=True)
    if daily.empty:
        raise ValueError(f"No rows found for preset={BASE_PRESET!r} in {BASE_DAILY_FILE}")
    return daily


def load_training_state() -> pd.DataFrame:
    """Load fold-level training state for the adaptive mix."""
    selected = pd.read_csv(SELECTED_FILE)
    selected = selected[selected["mode"] == BASE_MODE].copy()
    selected["test_year"] = pd.to_datetime(selected["test_start"]).dt.year
    cols = [
        "test_year",
        "annual_return",
        "max_drawdown",
        "recent_year_return",
        "test_start",
        "test_end",
        "source_mode",
    ]
    return selected[cols].sort_values("test_year").reset_index(drop=True)


def scale_by_year(training_state: pd.DataFrame, params: dict[str, float]) -> dict[int, float]:
    """Build a causal yearly scale map from training-period statistics."""
    scale = {}
    for row in training_state.itertuples():
        weak_but_controlled = (
            row.annual_return <= params["train_annual_max"]
            and row.recent_year_return <= params["recent_return_max"]
            and row.max_drawdown >= params["train_drawdown_min"]
        )
        scale[int(row.test_year)] = params["boost_scale"] if weak_but_controlled else 1.0
    return scale


def evaluate_scaled(base: pd.DataFrame, scale_map: dict[int, float]) -> tuple[pd.DataFrame, dict, dict]:
    """Apply yearly scales and summarize full / pre-2026 windows."""
    scaled = base.copy()
    scaled["annual_state_scale"] = scaled["test_year"].map(scale_map).astype(float)
    scaled["ret"] = scaled["ret"] * scaled["annual_state_scale"]
    scaled["weight"] = scaled["weight"] * scaled["annual_state_scale"]
    scaled["nav"] = (1 + scaled["ret"]).cumprod()
    scaled["drawdown"] = scaled["nav"] / scaled["nav"].cummax() - 1
    full_stats = annualized_stats(scaled["dt"], scaled["ret"].to_numpy(), scaled["weight"].to_numpy())
    pre_mask = scaled["test_year"].to_numpy() <= 2025
    pre_stats = annualized_stats(
        scaled.loc[pre_mask, "dt"].reset_index(drop=True),
        scaled.loc[pre_mask, "ret"].to_numpy(),
        scaled.loc[pre_mask, "weight"].to_numpy(),
    )
    return scaled, full_stats, pre_stats


def robustness_rows(base: pd.DataFrame, training_state: pd.DataFrame) -> pd.DataFrame:
    """Evaluate local threshold perturbations around the annual scale rule."""
    rows = []
    for boost_scale in ROBUSTNESS_GRID["boost_scale"]:
        for train_annual_max in ROBUSTNESS_GRID["train_annual_max"]:
            for recent_return_max in ROBUSTNESS_GRID["recent_return_max"]:
                for train_drawdown_min in ROBUSTNESS_GRID["train_drawdown_min"]:
                    params = {
                        "boost_scale": boost_scale,
                        "train_annual_max": train_annual_max,
                        "recent_return_max": recent_return_max,
                        "train_drawdown_min": train_drawdown_min,
                    }
                    scale_map = scale_by_year(training_state, params)
                    scaled, full_stats, pre_stats = evaluate_scaled(base, scale_map)
                    rows.append(
                        {
                            **params,
                            **{f"full_{key}": value for key, value in full_stats.items()},
                            **{f"pre2026_{key}": value for key, value in pre_stats.items()},
                            "scaled_years": ",".join(
                                str(year) for year, scale in sorted(scale_map.items()) if scale > 1
                            ),
                            "pass_full_and_pre2026": bool(
                                full_stats["annual_return"] >= 0.20
                                and full_stats["max_drawdown"] >= -0.10
                                and pre_stats["annual_return"] >= 0.20
                                and pre_stats["max_drawdown"] >= -0.10
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    """Run annual training-state risk scaling validation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_daily()
    training_state = load_training_state()

    daily_rows = []
    yearly_rows = []
    summary_rows = []
    scale_rows = []
    for preset, params in PRESETS.items():
        scale_map = scale_by_year(training_state, params)
        scaled, full_stats, pre_stats = evaluate_scaled(base, scale_map)
        scaled["preset"] = preset
        scaled["mode"] = f"{BASE_MODE}_{BASE_PRESET}_overlay_{preset}_annual_state_scale"
        daily_rows.append(scaled)

        yearly = summarize_yearly(scaled)
        yearly["preset"] = preset
        yearly["mode"] = f"{BASE_MODE}_{BASE_PRESET}_overlay_{preset}_annual_state_scale"
        yearly_rows.append(yearly)

        summary_rows.append(
            {
                "preset": preset,
                **{f"full_{key}": value for key, value in full_stats.items()},
                **{f"pre2026_{key}": value for key, value in pre_stats.items()},
                **params,
            }
        )
        for year, scale in scale_map.items():
            state = training_state[training_state["test_year"] == year].iloc[0].to_dict()
            scale_rows.append({"preset": preset, "test_year": year, "annual_state_scale": scale, **state})

    daily_out = pd.concat(daily_rows, ignore_index=True)
    yearly_out = pd.concat(yearly_rows, ignore_index=True)
    summary_out = pd.DataFrame(summary_rows)
    scale_out = pd.DataFrame(scale_rows)
    robustness = robustness_rows(base, training_state)

    daily_out.to_csv(OUTPUT_DIR / "annual_state_scale_daily.csv", index=False, encoding="utf-8-sig")
    yearly_out.to_csv(OUTPUT_DIR / "annual_state_scale_yearly.csv", index=False, encoding="utf-8-sig")
    summary_out.to_csv(OUTPUT_DIR / "annual_state_scale_summary.csv", index=False, encoding="utf-8-sig")
    scale_out.to_csv(OUTPUT_DIR / "annual_state_scale_by_year.csv", index=False, encoding="utf-8-sig")
    robustness.to_csv(OUTPUT_DIR / "annual_state_scale_robustness.csv", index=False, encoding="utf-8-sig")

    print(summary_out.to_string(index=False))
    print("\nScale by year:")
    print(
        scale_out[["preset", "test_year", "annual_state_scale", "annual_return", "max_drawdown"]].to_string(index=False)
    )
    print("\nYearly:")
    print(yearly_out[["preset", "year", "return", "max_drawdown", "annual_return", "calmar"]].to_string(index=False))
    passed = robustness[robustness["pass_full_and_pre2026"]].copy()
    print("\nRobustness:")
    print(
        {
            "total": int(len(robustness)),
            "passed": int(len(passed)),
            "pass_rate": float(len(passed) / len(robustness)),
            "pass_pre2026_annual_min": float(passed["pre2026_annual_return"].min()),
            "pass_pre2026_annual_median": float(passed["pre2026_annual_return"].median()),
            "pass_max_abs_weight_median": float(passed["full_max_abs_weight"].median()),
        }
    )
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
