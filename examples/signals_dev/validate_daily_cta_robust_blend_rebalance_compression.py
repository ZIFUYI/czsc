# ruff: noqa: E402
"""Rebalance compression tests for ``robust_50rv_blend``.

This script uses component-level desired weights and return contributions from
``validate_daily_cta_robust_blend_component_audit.py``. The simulation keeps
the component signal return stream unchanged, but changes how often portfolio
component exposure is reset to the desired weights. This estimates whether the
final candidate can tolerate practical no-trade bands or weekly execution.

Run:
    uv run --no-sync python examples/signals_dev/validate_daily_cta_robust_blend_rebalance_compression.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_daily_cta_layered_low_corr_overlay import TARGET_ANNUAL_RETURN, TARGET_MAX_DRAWDOWN
from validate_daily_cta_walk_forward import annualized_stats, summarize_yearly

ROOT = Path(__file__).resolve().parents[2]
COMPONENT_DIR = ROOT / "examples" / "results" / "daily_cta_robust_blend_component_audit"
COMPONENT_DAILY = COMPONENT_DIR / "robust_50rv_component_daily.csv"
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_robust_blend_rebalance_compression"

EXTRA_COST_RATES = (0.0, 0.0002, 0.0005, 0.0010)


@dataclass(frozen=True)
class CompressionScheme:
    """One component-level rebalance rule."""

    name: str
    weekly: bool
    threshold: float | None = None


SCHEMES = (
    CompressionScheme("daily_raw", weekly=False, threshold=None),
    CompressionScheme("weekly", weekly=True, threshold=None),
    CompressionScheme("threshold_5pct", weekly=False, threshold=0.05),
    CompressionScheme("threshold_10pct", weekly=False, threshold=0.10),
    CompressionScheme("threshold_20pct", weekly=False, threshold=0.20),
    CompressionScheme("weekly_threshold_5pct", weekly=True, threshold=0.05),
    CompressionScheme("weekly_threshold_10pct", weekly=True, threshold=0.10),
)


def load_component_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load component desired weights and return contributions as wide panels."""
    if not COMPONENT_DAILY.exists():
        raise FileNotFoundError(f"Missing component daily file: {COMPONENT_DAILY}")
    data = pd.read_csv(COMPONENT_DAILY, parse_dates=["dt"])
    data["dt"] = pd.to_datetime(data["dt"]).dt.tz_localize(None)
    data["component_id"] = (
        data["source_side"].astype(str) + "::" + data["selected_label"].astype(str) + "::" + data["component"].astype(str)
    )
    weights = (
        data.groupby(["dt", "component_id"], as_index=False)["component_weight"].sum()
        .pivot(index="dt", columns="component_id", values="component_weight")
        .fillna(0.0)
        .sort_index()
    )
    returns = (
        data.groupby(["dt", "component_id"], as_index=False)["component_ret"].sum()
        .pivot(index="dt", columns="component_id", values="component_ret")
        .fillna(0.0)
        .reindex(index=weights.index, columns=weights.columns)
        .fillna(0.0)
    )
    return weights, returns


def component_unit_returns(desired_weights: pd.DataFrame, component_returns: pd.DataFrame) -> pd.DataFrame:
    """Convert component return contributions into per-unit exposure returns."""
    denominator = desired_weights.shift(1)
    year_changed = desired_weights.index.to_series().dt.year.ne(desired_weights.index.to_series().dt.year.shift())
    denominator.loc[year_changed] = desired_weights.loc[year_changed]
    values = np.divide(
        component_returns.to_numpy(float),
        denominator.to_numpy(float),
        out=np.zeros_like(component_returns.to_numpy(float)),
        where=np.abs(denominator.to_numpy(float)) > 1e-12,
    )
    return pd.DataFrame(values, index=component_returns.index, columns=component_returns.columns)


def is_week_start(dates: pd.DatetimeIndex) -> np.ndarray:
    """Mark first trading day of each calendar week."""
    week = dates.to_series().dt.strftime("%G-%V")
    return week.ne(week.shift()).to_numpy()


def should_rebalance(
    scheme: CompressionScheme,
    idx: int,
    week_start: np.ndarray,
    current_weights: np.ndarray,
    desired_weights: np.ndarray,
) -> bool:
    """Decide whether to rebalance at the end of a day."""
    if scheme.name == "daily_raw":
        return True
    if scheme.weekly and not week_start[idx]:
        return False
    if scheme.threshold is None:
        return True
    drift = float(np.abs(desired_weights - current_weights).sum())
    return drift >= scheme.threshold


def simulate_scheme(
    desired_weights: pd.DataFrame,
    unit_returns: pd.DataFrame,
    scheme: CompressionScheme,
) -> pd.DataFrame:
    """Simulate one rebalance compression rule."""
    dates = desired_weights.index
    week_start = is_week_start(dates)
    desired = desired_weights.to_numpy(float)
    unit_ret = unit_returns.to_numpy(float)
    current = desired[0].copy()
    rows = []

    for i, dt in enumerate(dates):
        year_changed = i == 0 or dates[i].year != dates[i - 1].year
        turnover = 0.0
        pre_rebalance = False
        if year_changed:
            turnover = float(np.abs(desired[i] - current).sum())
            current = desired[i].copy()
            pre_rebalance = True

        daily_ret = float(np.dot(current, unit_ret[i]))
        desired_today = desired[i]
        rebalance = pre_rebalance
        if not pre_rebalance:
            rebalance = should_rebalance(scheme, i, week_start, current, desired_today)
        if rebalance:
            post_turnover = float(np.abs(desired_today - current).sum())
            turnover += post_turnover
            if post_turnover > 0:
                current = desired_today.copy()
        rows.append(
            {
                "dt": dt,
                "scheme": scheme.name,
                "ret": daily_ret,
                "turnover": turnover,
                "weight": float(np.abs(current).sum()),
                "rebalance": bool(rebalance),
                "desired_weight": float(np.abs(desired_today).sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["nav"] = (1 + out["ret"]).cumprod()
    out["drawdown"] = out["nav"] / out["nav"].cummax() - 1
    return out


def summarize_scheme(path: pd.DataFrame) -> dict:
    """Summarize one compressed path before extra cost."""
    stats = annualized_stats(path["dt"].reset_index(drop=True), path["ret"].to_numpy(), path["weight"].to_numpy())
    years = (path["dt"].iloc[-1] - path["dt"].iloc[0]).days / 365.25
    yearly = summarize_yearly(path[["dt", "ret", "weight"]])
    stats.update(
        {
            "scheme": path["scheme"].iloc[0],
            "rebalance_days": int(path["rebalance"].sum()),
            "turnover_sum": float(path["turnover"].sum()),
            "annualized_turnover": float(path["turnover"].sum() / years) if years > 0 else float("nan"),
            "avg_abs_weight": float(path["weight"].mean()),
            "max_abs_weight": float(path["weight"].max()),
            "tracking_ret_abs_sum": np.nan,
            "loss_years": int((yearly["return"] < 0).sum()),
            "pass_20_10": bool(
                stats["annual_return"] >= TARGET_ANNUAL_RETURN
                and stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
            ),
        }
    )
    return stats


def cost_sensitivity(paths: pd.DataFrame) -> pd.DataFrame:
    """Apply extra transaction costs to compressed paths."""
    rows = []
    for scheme, group in paths.groupby("scheme", sort=False):
        for cost_rate in EXTRA_COST_RATES:
            ret = group["ret"] - group["turnover"] * cost_rate
            stats = annualized_stats(group["dt"].reset_index(drop=True), ret.to_numpy(), group["weight"].to_numpy())
            rows.append(
                {
                    "scheme": scheme,
                    "extra_cost_bps": cost_rate * 10000,
                    "annual_return": stats["annual_return"],
                    "max_drawdown": stats["max_drawdown"],
                    "final_nav": stats["final_nav"],
                    "calmar": stats["calmar"],
                    "pass_20_10": bool(
                        stats["annual_return"] >= TARGET_ANNUAL_RETURN
                        and stats["max_drawdown"] >= TARGET_MAX_DRAWDOWN
                    ),
                }
            )
    return pd.DataFrame(rows)


def add_tracking_error(summary: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    """Add simple daily return tracking error versus daily raw execution."""
    raw = paths[paths["scheme"].eq("daily_raw")][["dt", "ret"]].rename(columns={"ret": "raw_ret"})
    rows = []
    for scheme, group in paths.groupby("scheme", sort=False):
        joined = group[["dt", "ret"]].merge(raw, on="dt", how="inner")
        rows.append(
            {
                "scheme": scheme,
                "tracking_ret_abs_sum": float((joined["ret"] - joined["raw_ret"]).abs().sum()),
                "tracking_ret_rmse": float(np.sqrt(np.mean((joined["ret"] - joined["raw_ret"]) ** 2))),
            }
        )
    return summary.drop(columns=["tracking_ret_abs_sum"], errors="ignore").merge(pd.DataFrame(rows), on="scheme")


def save_nav_plot(paths: pd.DataFrame) -> None:
    """Save net-value comparison plot."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for scheme, group in paths.groupby("scheme", sort=False):
        fig.add_trace(go.Scatter(x=group["dt"], y=group["nav"], mode="lines", name=scheme))
    fig.update_layout(
        title="robust_50rv_blend rebalance compression",
        xaxis_title="date",
        yaxis_title="net value",
        template="plotly_white",
    )
    fig.write_html(OUTPUT_DIR / "robust_50rv_rebalance_compression_nav.html", include_plotlyjs="cdn")


def force_raw_path(paths: pd.DataFrame, desired_weights: pd.DataFrame, component_returns: pd.DataFrame) -> pd.DataFrame:
    """Force daily_raw to the exact original component contribution path."""
    out = paths.copy()
    raw_mask = out["scheme"].eq("daily_raw")
    raw_ret = component_returns.sum(axis=1).to_numpy(float)
    raw_weight = desired_weights.sum(axis=1).to_numpy(float)
    raw_turnover = desired_weights.diff().abs().sum(axis=1).fillna(desired_weights.iloc[0].abs().sum()).to_numpy(float)
    out.loc[raw_mask, "ret"] = raw_ret
    out.loc[raw_mask, "weight"] = raw_weight
    out.loc[raw_mask, "desired_weight"] = raw_weight
    out.loc[raw_mask, "turnover"] = raw_turnover
    raw_nav = np.cumprod(1 + raw_ret)
    out.loc[raw_mask, "nav"] = raw_nav
    out.loc[raw_mask, "drawdown"] = raw_nav / np.maximum.accumulate(raw_nav) - 1
    out.loc[raw_mask, "rebalance"] = True
    return out


def main() -> None:
    """Run rebalance compression tests."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    desired_weights, component_returns = load_component_panel()
    unit_returns = component_unit_returns(desired_weights, component_returns)

    paths = pd.concat([simulate_scheme(desired_weights, unit_returns, scheme) for scheme in SCHEMES], ignore_index=True)
    paths = force_raw_path(paths, desired_weights, component_returns)
    summary = pd.DataFrame([summarize_scheme(group) for _, group in paths.groupby("scheme", sort=False)])
    summary = add_tracking_error(summary, paths)
    costs = cost_sensitivity(paths)

    paths.to_csv(OUTPUT_DIR / "robust_50rv_rebalance_compression_daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "robust_50rv_rebalance_compression_summary.csv", index=False, encoding="utf-8-sig")
    costs.to_csv(OUTPUT_DIR / "robust_50rv_rebalance_compression_costs.csv", index=False, encoding="utf-8-sig")
    save_nav_plot(paths)

    print("Rebalance compression summary:")
    print(
        summary[
            [
                "scheme",
                "annual_return",
                "max_drawdown",
                "final_nav",
                "rebalance_days",
                "turnover_sum",
                "annualized_turnover",
                "tracking_ret_abs_sum",
                "pass_20_10",
            ]
        ].to_string(index=False)
    )
    print("\nCost sensitivity:")
    print(
        costs[costs["extra_cost_bps"].isin([0.0, 2.0, 5.0, 10.0])][
            ["scheme", "extra_cost_bps", "annual_return", "max_drawdown", "final_nav", "pass_20_10"]
        ].to_string(index=False)
    )
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
