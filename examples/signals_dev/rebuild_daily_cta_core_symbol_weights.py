# ruff: noqa: E402
"""Rebuild true three-symbol weights for the core daily CTA sleeve.

The final robust blend component audit keeps the core CTA sleeve as an
aggregate exposure because earlier research outputs did not persist
symbol-level weights. This script reruns the same cross-sectional walk-forward
selection logic and exports the real 000852 / 000905 / 159915 weights for the
``aggressive_state_switch_adaptive_mix`` core sleeve after market-state overlay
and annual-state scaling.

Run:
    uv run --no-sync python examples/signals_dev/rebuild_daily_cta_core_symbol_weights.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_daily_cta_cross_section_walk_forward as cs

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_core_symbol_weights"
MARKET_STATE_DAILY = (
    ROOT / "examples" / "results" / "daily_cta_cross_section_walk_forward" / "market_state_overlay" / "market_state_overlay_daily.csv"
)
ANNUAL_STATE_DAILY = (
    ROOT
    / "examples"
    / "results"
    / "daily_cta_cross_section_walk_forward"
    / "annual_state_scale"
    / "annual_state_scale_daily.csv"
)

CANDIDATE_RE = re.compile(
    r"^(?P<score_type>[^_]+)_(?P<mode>rank_ls|rank_long|market_switch|trend_equal)_"
    r"N(?P<lookbacks>[0-9-]+)_B(?P<threshold_bp>\d+)_VW(?P<vol_window>\d+)_"
    r"TV(?P<target_vol>\d+)_C(?P<leverage_cap>[\d.]+)$"
)


def parse_candidate(name: str) -> cs.CrossCandidate:
    """Parse a serialized cross-sectional candidate name."""
    match = CANDIDATE_RE.match(name)
    if match is None:
        raise ValueError(f"Unsupported candidate name: {name}")
    groups = match.groupdict()
    return cs.CrossCandidate(
        score_type=groups["score_type"],
        lookbacks=tuple(int(x) for x in groups["lookbacks"].split("-")),
        threshold=int(groups["threshold_bp"]) / 10000,
        mode=groups["mode"],
        vol_window=int(groups["vol_window"]),
        target_vol=int(groups["target_vol"]) / 100,
        leverage_cap=float(groups["leverage_cap"]),
    )


def select_candidate_ensemble_with_weights(
    panel: pd.DataFrame,
    policy: cs.SelectionPolicy,
    rows: list[dict],
    train_mask: np.ndarray,
    weight_cache: dict[tuple[str, tuple[int, ...], float, str], pd.DataFrame],
) -> dict:
    """Select a top-N ensemble and rebuild its symbol weights."""
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

    symbol_weights = []
    for name in eligible["candidate"]:
        candidate = parse_candidate(name)
        _, _, weights = cs.cached_candidate_returns(panel, candidate, weight_cache)
        symbol_weights.append(weights)
    weights = sum(symbol_weights) / len(symbol_weights)
    gross_weights = sum(frame.abs() for frame in symbol_weights) / len(symbol_weights)

    raw_train_stats = cs.annualized_stats(
        panel.loc[train_mask, "dt"].reset_index(drop=True), returns[train_mask], exposure[train_mask]
    )
    train_scale = 1.0
    if policy.train_drawdown_budget is not None and raw_train_stats["max_drawdown"] < 0:
        train_scale = min(1.0, policy.train_drawdown_budget / abs(raw_train_stats["max_drawdown"]))
    returns = returns * train_scale
    exposure = exposure * train_scale
    weights = weights * train_scale
    gross_weights = gross_weights * train_scale

    train_stats = cs.annualized_stats(
        panel.loc[train_mask, "dt"].reset_index(drop=True), returns[train_mask], exposure[train_mask]
    )
    train_stats["mode"] = policy.name
    train_stats["loss_years"] = cs.loss_years(panel["dt"], returns, train_mask)
    train_stats["recent_year_return"] = cs.recent_return(panel["dt"], returns, train_mask)
    train_stats["candidate"] = f"ENS{len(eligible)}__{eligible.iloc[0]['candidate']}"
    train_stats["top_candidates"] = "|".join(eligible["candidate"].head(3))
    train_stats["selected_candidate_names"] = "|".join(eligible["candidate"])
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
    train_stats["_weights"] = weights
    train_stats["_gross_weights"] = gross_weights
    return train_stats


def symbol_weight_frame(
    panel: pd.DataFrame,
    mask: np.ndarray,
    weights: pd.DataFrame,
    gross_weights: pd.DataFrame,
    mode: str,
    test_year: int,
    candidate: str,
    risk_state: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build one long daily symbol-weight frame."""
    part = weights.loc[mask, list(cs.SYMBOLS)].copy()
    gross_part = gross_weights.loc[mask, list(cs.SYMBOLS)].copy()
    part.insert(0, "dt", panel.loc[mask, "dt"].to_numpy())
    part["mode"] = mode
    part["test_year"] = test_year
    part["candidate"] = candidate
    if risk_state is not None:
        part[list(cs.SYMBOLS)] = part[list(cs.SYMBOLS)].multiply(risk_state, axis=0)
        gross_part[list(cs.SYMBOLS)] = gross_part[list(cs.SYMBOLS)].multiply(risk_state, axis=0)
        part["risk_state"] = risk_state
    for symbol in cs.SYMBOLS:
        part[f"{symbol}_gross_abs"] = gross_part[symbol].abs().to_numpy(float)
    return part


def combine_weight_frames(left: pd.DataFrame, right: pd.DataFrame, left_weight: pd.Series | float, mode: str) -> pd.DataFrame:
    """Linearly combine two symbol-weight frames."""
    joined = left.merge(right, on=["dt", "test_year"], suffixes=("_left", "_right"))
    if isinstance(left_weight, pd.Series):
        lw = joined["test_year"].map(left_weight).astype(float).to_numpy()
    else:
        lw = np.full(len(joined), float(left_weight))
    out = joined[["dt", "test_year"]].copy()
    for symbol in cs.SYMBOLS:
        out[symbol] = lw * joined[f"{symbol}_left"] + (1 - lw) * joined[f"{symbol}_right"]
        out[f"{symbol}_gross_abs"] = lw * joined[f"{symbol}_gross_abs_left"] + (1 - lw) * joined[
            f"{symbol}_gross_abs_right"
        ]
    out["mode"] = mode
    out["candidate"] = mode
    return out


def scale_weight_frame(frame: pd.DataFrame, scale: pd.Series | np.ndarray | float, mode: str) -> pd.DataFrame:
    """Scale symbol weights by a scalar or aligned vector."""
    out = frame.copy()
    if isinstance(scale, pd.Series):
        values = out["test_year"].map(scale).astype(float).to_numpy()
    elif isinstance(scale, np.ndarray):
        values = scale.astype(float)
    else:
        values = float(scale)
    out[list(cs.SYMBOLS)] = out[list(cs.SYMBOLS)].multiply(values, axis=0)
    gross_cols = [f"{symbol}_gross_abs" for symbol in cs.SYMBOLS]
    out[gross_cols] = out[gross_cols].multiply(values, axis=0)
    out["mode"] = mode
    out["candidate"] = mode
    return out


def rebuild_adaptive_symbol_weights() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild symbol weights for the adaptive core before overlays."""
    panel = cs.load_panel()
    weight_cache = cs.build_raw_weight_cache(panel)
    selected_rows = []
    policy_frames: dict[str, list[pd.DataFrame]] = {policy.name: [] for policy in cs.POLICIES}
    mdd_frames = []
    target06_frames = []

    for fold in cs.FOLDS:
        train_mask, rows = cs.build_candidate_rows(panel, fold, weight_cache)
        test_start, test_end = pd.Timestamp(fold[2]), pd.Timestamp(fold[3])
        test_mask = ((panel["dt"] >= test_start) & (panel["dt"] <= test_end)).to_numpy()
        test_year = int(test_start.year)
        for policy in cs.POLICIES:
            selected = select_candidate_ensemble_with_weights(panel, policy, rows, train_mask, weight_cache)
            returns = selected["_returns"]
            exposure = selected["_exposure"]
            weights = selected["_weights"]
            gross_weights = selected["_gross_weights"]
            test_stats = cs.annualized_stats(
                panel.loc[test_mask, "dt"].reset_index(drop=True), returns[test_mask], exposure[test_mask]
            )
            selected_row = {k: v for k, v in selected.items() if not k.startswith("_")}
            selected_row.update({f"test_{key}": value for key, value in test_stats.items()})
            selected_rows.append(selected_row)
            policy_frames[policy.name].append(
                symbol_weight_frame(
                    panel, test_mask, weights, gross_weights, policy.name, test_year, selected["candidate"]
                )
            )

            if policy.name == "aggressive_baseline":
                for brake_mode, objective, reset_state in (
                    (cs.MDD_BRAKE_RESET_MODE, "min_mdd_ann05", True),
                    (cs.TARGET06_BRAKE_RESET_MODE, "target06", True),
                ):
                    brake = cs.select_training_brake(panel, returns, exposure, train_mask, objective)
                    if reset_state:
                        train_nav, train_peak, train_risk = 1.0, 1.0, 1.0
                    else:
                        _, _, _, train_nav, train_peak, train_risk = cs.apply_drawdown_brake(
                            returns[train_mask],
                            exposure[train_mask],
                            brake["stop"],
                            brake["resume"],
                            brake["scale"],
                        )
                    _, _, risk_states, _, _, _ = cs.apply_drawdown_brake(
                        returns[test_mask],
                        exposure[test_mask],
                        brake["stop"],
                        brake["resume"],
                        brake["scale"],
                        train_nav,
                        train_peak,
                        train_risk,
                    )
                    frame = symbol_weight_frame(
                        panel,
                        test_mask,
                        weights,
                        gross_weights,
                        brake_mode,
                        test_year,
                        selected["candidate"],
                        risk_states,
                    )
                    if brake_mode == cs.MDD_BRAKE_RESET_MODE:
                        mdd_frames.append(frame)
                    else:
                        target06_frames.append(frame)

    selected_df = pd.DataFrame(selected_rows)
    baseline_by_year = selected_df[selected_df["mode"] == "aggressive_baseline"].copy()
    baseline_by_year["test_year"] = pd.to_datetime(baseline_by_year["test_start"]).dt.year
    switch_weight_by_year = pd.Series(
        {
            int(row.test_year): float(row.annual_return >= cs.STATE_SWITCH_TRAIN_ANNUAL_THRESHOLD)
            for row in baseline_by_year.itertuples()
        }
    )
    recent_guard_scale_by_year = {}
    for row in baseline_by_year.itertuples():
        if row.annual_return < cs.STATE_SWITCH_TRAIN_ANNUAL_THRESHOLD:
            scale = cs.STATE_SWITCH_RECENT_GUARD_WEAK_SCALE
        elif row.annual_return < cs.STATE_SWITCH_STRONG_ANNUAL_THRESHOLD:
            scale = cs.STATE_SWITCH_RECENT_GUARD_MID_SCALE
        elif row.recent_year_return < cs.STATE_SWITCH_RECENT_GUARD_THRESHOLD:
            scale = cs.STATE_SWITCH_RECENT_GUARD_STRONG_COOL_SCALE
        else:
            scale = cs.STATE_SWITCH_RECENT_GUARD_STRONG_HOT_SCALE
        recent_guard_scale_by_year[int(row.test_year)] = scale
    recent_guard_scale_by_year = pd.Series(recent_guard_scale_by_year)

    baseline = pd.concat(policy_frames["aggressive_baseline"], ignore_index=True)
    low_risk = pd.concat(policy_frames["low_risk_top3"], ignore_index=True)
    mdd = pd.concat(mdd_frames, ignore_index=True)
    target06 = pd.concat(target06_frames, ignore_index=True)
    switch = combine_weight_frames(mdd, target06, switch_weight_by_year, cs.STATE_SWITCH_MODE)
    recent_guard = scale_weight_frame(switch, recent_guard_scale_by_year, cs.STATE_SWITCH_RECENT_GUARD_MODE)

    adaptive_parts = []
    for row in baseline_by_year.itertuples():
        test_year = int(row.test_year)
        if row.annual_return < 0.05 and row.max_drawdown > -0.08:
            source = baseline
            source_mode = "aggressive_baseline"
        elif row.annual_return < cs.STATE_SWITCH_TRAIN_ANNUAL_THRESHOLD and row.recent_year_return < 0:
            source = low_risk
            source_mode = "low_risk_top3"
        elif (
            row.annual_return >= cs.STATE_SWITCH_STRONG_ANNUAL_THRESHOLD
            and row.recent_year_return >= cs.STATE_SWITCH_ADAPTIVE_HOT_RECENT_THRESHOLD
        ):
            source = baseline
            source_mode = "aggressive_baseline"
        else:
            source = recent_guard
            source_mode = cs.STATE_SWITCH_RECENT_GUARD_MODE
        part = source[source["test_year"] == test_year].copy()
        part["mode"] = cs.STATE_SWITCH_ADAPTIVE_MIX_MODE
        part["source_mode"] = source_mode
        adaptive_parts.append(part)

    adaptive = pd.concat(adaptive_parts, ignore_index=True).sort_values("dt").reset_index(drop=True)
    adaptive["total_abs_weight"] = adaptive[list(cs.SYMBOLS)].abs().sum(axis=1)
    return adaptive, selected_df


def apply_overlays(adaptive: pd.DataFrame) -> pd.DataFrame:
    """Apply saved market-state overlay scale and annual-state scale."""
    market = pd.read_csv(MARKET_STATE_DAILY, parse_dates=["dt"])
    market["dt"] = pd.to_datetime(market["dt"]).dt.tz_localize(None)
    market = market[market["preset"].eq("aggressive")][["dt", "overlay_scale"]].copy()

    annual = pd.read_csv(ANNUAL_STATE_DAILY, parse_dates=["dt"])
    annual["dt"] = pd.to_datetime(annual["dt"]).dt.tz_localize(None)
    annual = annual[annual["preset"].isin(["capital_efficient", "stress_balanced"])][
        ["dt", "preset", "annual_state_scale", "weight"]
    ].copy()

    rows = []
    for preset, annual_part in annual.groupby("preset", sort=False):
        joined = adaptive.merge(market, on="dt", how="inner").merge(annual_part, on="dt", how="inner")
        scale = joined["overlay_scale"] * joined["annual_state_scale"]
        for symbol in cs.SYMBOLS:
            joined[symbol] = joined[symbol] * scale
            joined[f"{symbol}_gross_abs"] = joined[f"{symbol}_gross_abs"] * scale
        joined["preset"] = preset
        joined["saved_total_abs_weight"] = joined["weight"].abs()
        joined["rebuilt_net_abs_weight"] = joined[list(cs.SYMBOLS)].abs().sum(axis=1)
        joined["rebuilt_gross_abs_weight"] = joined[[f"{symbol}_gross_abs" for symbol in cs.SYMBOLS]].sum(axis=1)
        joined["gross_abs_diff"] = (joined["rebuilt_gross_abs_weight"] - joined["saved_total_abs_weight"]).abs()
        rows.append(
            joined[
                [
                    "dt",
                    "preset",
                    "test_year",
                    "source_mode",
                    "overlay_scale",
                    "annual_state_scale",
                    *cs.SYMBOLS,
                    *(f"{symbol}_gross_abs" for symbol in cs.SYMBOLS),
                    "rebuilt_net_abs_weight",
                    "rebuilt_gross_abs_weight",
                    "saved_total_abs_weight",
                    "gross_abs_diff",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values(["preset", "dt"]).reset_index(drop=True)


def symbol_summary(core: pd.DataFrame) -> pd.DataFrame:
    """Summarize symbol-level exposure and turnover by annual-state preset."""
    rows = []
    for preset, group in core.groupby("preset", sort=False):
        for symbol in cs.SYMBOLS:
            weights = group[symbol].astype(float)
            gross = group[f"{symbol}_gross_abs"].astype(float)
            turnover = weights.diff().abs().fillna(weights.abs())
            gross_turnover = gross.diff().abs().fillna(gross.abs())
            rows.append(
                {
                    "preset": preset,
                    "symbol": symbol,
                    "avg_net_abs_weight": float(weights.abs().mean()),
                    "net_weight_p95": float(weights.abs().quantile(0.95)),
                    "net_weight_p99": float(weights.abs().quantile(0.99)),
                    "max_net_abs_weight": float(weights.abs().max()),
                    "avg_gross_abs_weight": float(gross.mean()),
                    "gross_weight_p95": float(gross.quantile(0.95)),
                    "gross_weight_p99": float(gross.quantile(0.99)),
                    "max_gross_abs_weight": float(gross.max()),
                    "net_turnover_sum": float(turnover.sum()),
                    "net_turnover_p95": float(turnover.quantile(0.95)),
                    "max_net_daily_turnover": float(turnover.max()),
                    "gross_turnover_sum": float(gross_turnover.sum()),
                    "gross_turnover_p95": float(gross_turnover.quantile(0.95)),
                    "max_gross_daily_turnover": float(gross_turnover.max()),
                    "days_abs_weight_gt_0_5": int((weights.abs() > 0.5).sum()),
                    "days_abs_weight_gt_1_0": int((weights.abs() > 1.0).sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["preset", "max_gross_abs_weight"], ascending=[True, False])


def top_turnover_days(core: pd.DataFrame) -> pd.DataFrame:
    """List largest symbol-level turnover dates."""
    rows = []
    for _, group in core.groupby("preset", sort=False):
        group = group.sort_values("dt").copy()
        turnover = group[list(cs.SYMBOLS)].diff().abs().fillna(group[list(cs.SYMBOLS)].abs())
        gross_cols = [f"{symbol}_gross_abs" for symbol in cs.SYMBOLS]
        gross_turnover = group[gross_cols].diff().abs().fillna(group[gross_cols].abs())
        out = group[
            ["dt", "preset", "test_year", "source_mode", "rebuilt_net_abs_weight", "rebuilt_gross_abs_weight"]
        ].copy()
        for symbol in cs.SYMBOLS:
            out[f"{symbol}_turnover"] = turnover[symbol]
            out[f"{symbol}_weight"] = group[symbol]
            out[f"{symbol}_gross_turnover"] = gross_turnover[f"{symbol}_gross_abs"]
            out[f"{symbol}_gross_abs"] = group[f"{symbol}_gross_abs"]
        out["total_net_symbol_turnover"] = turnover.sum(axis=1)
        out["total_gross_symbol_turnover"] = gross_turnover.sum(axis=1)
        rows.append(out)
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values("total_net_symbol_turnover", ascending=False)
        .reset_index(drop=True)
    )


def reconciliation(core: pd.DataFrame) -> pd.DataFrame:
    """Summarize rebuilt vs saved aggregate core exposure."""
    rows = []
    for preset, group in core.groupby("preset", sort=False):
        rows.append(
            {
                "preset": preset,
                "days": int(len(group)),
                "rebuilt_net_weight_mean": float(group["rebuilt_net_abs_weight"].mean()),
                "rebuilt_gross_weight_mean": float(group["rebuilt_gross_abs_weight"].mean()),
                "saved_weight_mean": float(group["saved_total_abs_weight"].mean()),
                "max_gross_abs_diff": float(group["gross_abs_diff"].max()),
                "p99_gross_abs_diff": float(group["gross_abs_diff"].quantile(0.99)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Rebuild and save core CTA symbol weights."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    adaptive, selected = rebuild_adaptive_symbol_weights()
    core = apply_overlays(adaptive)
    recon = reconciliation(core)
    sym_summary = symbol_summary(core)
    turnover_days = top_turnover_days(core)

    adaptive.to_csv(OUTPUT_DIR / "core_adaptive_mix_symbol_weights.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUTPUT_DIR / "core_rebuilt_selected_by_fold.csv", index=False, encoding="utf-8-sig")
    core.to_csv(OUTPUT_DIR / "core_annual_state_symbol_weights.csv", index=False, encoding="utf-8-sig")
    recon.to_csv(OUTPUT_DIR / "core_symbol_weight_reconciliation.csv", index=False, encoding="utf-8-sig")
    sym_summary.to_csv(OUTPUT_DIR / "core_symbol_weight_summary.csv", index=False, encoding="utf-8-sig")
    turnover_days.to_csv(OUTPUT_DIR / "core_symbol_top_turnover_days.csv", index=False, encoding="utf-8-sig")

    print("Reconciliation:")
    print(recon.to_string(index=False))
    print("\nSymbol summary:")
    print(sym_summary.to_string(index=False))
    print("\nTop turnover days:")
    print(turnover_days.head(12).to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
