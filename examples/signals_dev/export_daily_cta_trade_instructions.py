# ruff: noqa: E402
"""Export order-ready instructions for the daily CTA live candidate.

This turns the live candidate weights into unit-NAV exposure instructions. It
does not assume a broker or account size. Amounts scale linearly:

    notional = weight * account_equity

The index symbols 000852 / 000905 are exposure targets, not directly tradable
shares. Use the generated implementation map to bind them to index futures,
ETF substitutes or another approved hedge instrument before live trading.

Run:
    uv run --no-sync python examples/signals_dev/export_daily_cta_trade_instructions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_daily_cta_cross_section_walk_forward as cs

ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = ROOT / "examples" / "results" / "daily_cta_live_candidate"
WEIGHTS_FILE = LIVE_DIR / "live_candidate_three_symbol_weights.csv"
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_trade_instructions"

EXAMPLE_EQUITY = (1_000_000, 10_000_000, 100_000_000)

INSTRUMENT_MAP = {
    "000852.XSHG": {
        "asset_name": "中证1000指数敞口",
        "preferred_instrument": "IM futures / 中证1000 ETF substitute",
        "tradability": "index_exposure",
        "short_dependency": "requires futures short or borrowable ETF substitute when weight < 0",
        "default_margin_ratio": 0.15,
    },
    "000905.XSHG": {
        "asset_name": "中证500指数敞口",
        "preferred_instrument": "IC futures / 中证500 ETF substitute",
        "tradability": "index_exposure",
        "short_dependency": "requires futures short or borrowable ETF substitute when weight < 0",
        "default_margin_ratio": 0.15,
    },
    "159915.XSHE": {
        "asset_name": "创业板ETF",
        "preferred_instrument": "159915 ETF spot / borrowable ETF short substitute",
        "tradability": "etf",
        "short_dependency": "long via ETF spot; short requires borrow / margin / hedge substitute when weight < 0",
        "default_margin_ratio": 1.00,
    },
}


def load_weights() -> pd.DataFrame:
    """Load live candidate executable weights."""
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(f"Missing live candidate weights: {WEIGHTS_FILE}")
    weights = pd.read_csv(WEIGHTS_FILE, parse_dates=["dt"])
    weights["dt"] = pd.to_datetime(weights["dt"]).dt.tz_localize(None)
    return weights.sort_values("dt").reset_index(drop=True)


def side_from_value(value: float) -> str:
    """Map signed exposure to an order side."""
    if value > 0:
        return "BUY_OR_LONG"
    if value < 0:
        return "SELL_OR_SHORT"
    return "HOLD"


def latest_target_table(weights: pd.DataFrame) -> pd.DataFrame:
    """Build latest unit-NAV target exposure table."""
    latest = weights.iloc[-1]
    rows = []
    for symbol in cs.SYMBOLS:
        weight = float(latest[f"weight_{symbol}"])
        mapping = INSTRUMENT_MAP[symbol]
        rows.append(
            {
                "dt": latest["dt"],
                "symbol": symbol,
                "asset_name": mapping["asset_name"],
                "target_weight": weight,
                "target_notional_per_1_nav": weight,
                "side": side_from_value(weight),
                "requires_short_or_hedge": bool(weight < 0),
                "preferred_instrument": mapping["preferred_instrument"],
                "tradability": mapping["tradability"],
                "short_dependency": mapping["short_dependency"],
                "default_margin_ratio": mapping["default_margin_ratio"],
                "estimated_margin_per_1_nav": abs(weight) * mapping["default_margin_ratio"],
            }
        )
    return pd.DataFrame(rows)


def latest_order_table(weights: pd.DataFrame) -> pd.DataFrame:
    """Build latest model delta and flat-start orders in unit-NAV terms."""
    latest = weights.iloc[-1]
    rows = []
    for symbol in cs.SYMBOLS:
        target_weight = float(latest[f"weight_{symbol}"])
        model_delta = float(latest[f"trade_{symbol}"])
        rows.append(
            {
                "dt": latest["dt"],
                "symbol": symbol,
                "target_weight": target_weight,
                "model_delta_weight": model_delta,
                "model_delta_side": side_from_value(model_delta),
                "flat_start_delta_weight": target_weight,
                "flat_start_side": side_from_value(target_weight),
                "max_single_day_weight_change_rule": 0.5,
                "note": "model_delta is the latest scheduled rebalance delta; flat_start_delta is for a new account starting from zero.",
            }
        )
    return pd.DataFrame(rows)


def daily_instruction_table(weights: pd.DataFrame) -> pd.DataFrame:
    """Build historical daily target and trade instruction table."""
    rows = []
    for _, row in weights.iterrows():
        for symbol in cs.SYMBOLS:
            weight = float(row[f"weight_{symbol}"])
            trade = float(row[f"trade_{symbol}"])
            mapping = INSTRUMENT_MAP[symbol]
            rows.append(
                {
                    "dt": row["dt"],
                    "symbol": symbol,
                    "target_weight": weight,
                    "trade_delta_weight": trade,
                    "trade_side": side_from_value(trade),
                    "requires_short_or_hedge": bool(weight < 0 or trade < 0),
                    "preferred_instrument": mapping["preferred_instrument"],
                    "estimated_margin_per_1_nav": abs(weight) * mapping["default_margin_ratio"],
                }
            )
    return pd.DataFrame(rows)


def example_notional_table(latest_orders: pd.DataFrame) -> pd.DataFrame:
    """Build notional examples for common account equity sizes."""
    rows = []
    for _, row in latest_orders.iterrows():
        for equity in EXAMPLE_EQUITY:
            rows.append(
                {
                    "dt": row["dt"],
                    "account_equity": equity,
                    "symbol": row["symbol"],
                    "target_notional": row["target_weight"] * equity,
                    "model_delta_notional": row["model_delta_weight"] * equity,
                    "flat_start_delta_notional": row["flat_start_delta_weight"] * equity,
                }
            )
    return pd.DataFrame(rows)


def portfolio_diagnostics(weights: pd.DataFrame) -> pd.DataFrame:
    """Summarize exposure, short dependency and margin proxy over time."""
    rows = []
    for _, row in weights.iterrows():
        target_values = {symbol: float(row[f"weight_{symbol}"]) for symbol in cs.SYMBOLS}
        trade_values = {symbol: float(row[f"trade_{symbol}"]) for symbol in cs.SYMBOLS}
        margin_proxy = sum(abs(target_values[symbol]) * INSTRUMENT_MAP[symbol]["default_margin_ratio"] for symbol in cs.SYMBOLS)
        rows.append(
            {
                "dt": row["dt"],
                "gross_core_weight": sum(abs(v) for v in target_values.values()),
                "net_core_weight": sum(target_values.values()),
                "long_core_weight": sum(max(v, 0.0) for v in target_values.values()),
                "short_core_weight_abs": sum(abs(min(v, 0.0)) for v in target_values.values()),
                "trade_abs_sum": sum(abs(v) for v in trade_values.values()),
                "max_symbol_trade_abs": max(abs(v) for v in trade_values.values()),
                "estimated_margin_per_1_nav": margin_proxy,
                "requires_short_or_hedge": any(v < 0 for v in target_values.values()),
            }
        )
    return pd.DataFrame(rows)


def save_execution_readme(targets: pd.DataFrame, latest_orders: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    """Save execution instructions in markdown."""
    latest_dt = pd.to_datetime(targets["dt"].iloc[0]).date()
    diag = diagnostics.iloc[-1]
    rows = []
    for _, row in targets.iterrows():
        rows.append(f"- `{row['symbol']}` target {row['target_weight']:.6f} via {row['preferred_instrument']}")
    order_rows = []
    for _, row in latest_orders.iterrows():
        order_rows.append(
            f"- `{row['symbol']}` model delta {row['model_delta_weight']:.6f}; flat-start delta {row['flat_start_delta_weight']:.6f}"
        )
    lines = [
        "# Daily CTA Trade Instructions",
        "",
        f"Latest signal date: {latest_dt}",
        "",
        "All quantities are expressed per 1.0 account NAV. Multiply by account equity to get notional.",
        "",
        "## Latest Targets",
        "",
        *rows,
        "",
        "## Latest Orders",
        "",
        *order_rows,
        "",
        "## Portfolio Diagnostics",
        "",
        f"- Gross core exposure: {diag['gross_core_weight']:.6f}",
        f"- Net core exposure: {diag['net_core_weight']:.6f}",
        f"- Long core exposure: {diag['long_core_weight']:.6f}",
        f"- Short core exposure abs: {diag['short_core_weight_abs']:.6f}",
        f"- Estimated margin proxy per NAV: {diag['estimated_margin_per_1_nav']:.6f}",
        f"- Requires short / hedge: {bool(diag['requires_short_or_hedge'])}",
        "",
        "## Implementation Notes",
        "",
        "- 000852 and 000905 are index exposure targets, not direct stock orders.",
        "- Bind index exposure to approved futures, ETF, swap or other hedge instruments before live execution.",
        "- Negative target weights require short / hedge capability. The all-long-only stress test failed the drawdown gate.",
        "- `model_delta_weight` is the latest scheduled trade from the backtest path; use `flat_start_delta_weight` only when initializing a new account from zero.",
    ]
    (OUTPUT_DIR / "trade_instruction_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Export unit-NAV trade instructions."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weights = load_weights()
    targets = latest_target_table(weights)
    latest_orders = latest_order_table(weights)
    daily = daily_instruction_table(weights)
    examples = example_notional_table(latest_orders)
    diagnostics = portfolio_diagnostics(weights)

    targets.to_csv(OUTPUT_DIR / "latest_target_exposures_unit_nav.csv", index=False, encoding="utf-8-sig")
    latest_orders.to_csv(OUTPUT_DIR / "latest_orders_unit_nav.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUTPUT_DIR / "daily_trade_instructions_unit_nav.csv", index=False, encoding="utf-8-sig")
    examples.to_csv(OUTPUT_DIR / "latest_order_notional_examples.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(OUTPUT_DIR / "portfolio_exposure_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame.from_dict(INSTRUMENT_MAP, orient="index").reset_index(names="symbol").to_csv(
        OUTPUT_DIR / "implementation_instrument_map.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_execution_readme(targets, latest_orders, diagnostics)

    print("Latest target exposures:")
    print(targets[["symbol", "target_weight", "side", "preferred_instrument", "estimated_margin_per_1_nav"]].to_string(index=False))
    print("\nLatest orders:")
    print(latest_orders.to_string(index=False))
    print("\nLatest diagnostics:")
    print(diagnostics.tail(1).to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
