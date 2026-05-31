# ruff: noqa: E402
"""Export broker-style order blotters for the daily CTA live candidate.

The input is the unit-NAV instruction package. This script adds an account
equity assumption, latest local prices and instrument sizing rules to produce
order-blotter templates. Defaults are examples only; replace the config with
broker-approved contract specs, margin rates and lot sizes before live trading.

Run:
    uv run --no-sync python examples/signals_dev/export_daily_cta_broker_blotter.py
    uv run --no-sync python examples/signals_dev/export_daily_cta_broker_blotter.py --account-equity 50000000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_daily_cta_cross_section_walk_forward as cs

ROOT = Path(__file__).resolve().parents[2]
TRADE_DIR = ROOT / "examples" / "results" / "daily_cta_trade_instructions"
LATEST_ORDERS = TRADE_DIR / "latest_orders_unit_nav.csv"
LATEST_TARGETS = TRADE_DIR / "latest_target_exposures_unit_nav.csv"
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_broker_blotter"
DEFAULT_ACCOUNT_EQUITY = 10_000_000.0


@dataclass(frozen=True)
class InstrumentSpec:
    """Minimal sizing config for one tradable implementation."""

    symbol: str
    exposure_symbol: str
    instrument_name: str
    instrument_type: str
    multiplier: float
    lot_size: int
    min_trade_lots: int
    margin_ratio: float
    allow_short: bool
    price_source: str
    notes: str


DEFAULT_SPECS = (
    InstrumentSpec(
        symbol="IM_PLACEHOLDER",
        exposure_symbol="000852.XSHG",
        instrument_name="CSI1000 exposure placeholder",
        instrument_type="futures_or_substitute",
        multiplier=200.0,
        lot_size=1,
        min_trade_lots=1,
        margin_ratio=0.15,
        allow_short=True,
        price_source="000852.XSHG index close",
        notes="Replace with the actual IM contract or approved ETF/swap substitute.",
    ),
    InstrumentSpec(
        symbol="IC_PLACEHOLDER",
        exposure_symbol="000905.XSHG",
        instrument_name="CSI500 exposure placeholder",
        instrument_type="futures_or_substitute",
        multiplier=200.0,
        lot_size=1,
        min_trade_lots=1,
        margin_ratio=0.15,
        allow_short=True,
        price_source="000905.XSHG index close",
        notes="Replace with the actual IC contract or approved ETF/swap substitute.",
    ),
    InstrumentSpec(
        symbol="159915.XSHE",
        exposure_symbol="159915.XSHE",
        instrument_name="ChiNext ETF",
        instrument_type="etf",
        multiplier=1.0,
        lot_size=100,
        min_trade_lots=1,
        margin_ratio=1.00,
        allow_short=False,
        price_source="159915.XSHE close",
        notes="Long ETF spot by default; short requires a borrowable or hedge substitute.",
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export daily CTA broker blotter templates.")
    parser.add_argument("--account-equity", type=float, default=DEFAULT_ACCOUNT_EQUITY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load latest unit-NAV orders and targets."""
    if not LATEST_ORDERS.exists():
        raise FileNotFoundError(f"Missing latest orders: {LATEST_ORDERS}")
    if not LATEST_TARGETS.exists():
        raise FileNotFoundError(f"Missing latest targets: {LATEST_TARGETS}")
    orders = pd.read_csv(LATEST_ORDERS, parse_dates=["dt"])
    targets = pd.read_csv(LATEST_TARGETS, parse_dates=["dt"])
    orders["dt"] = pd.to_datetime(orders["dt"]).dt.tz_localize(None)
    targets["dt"] = pd.to_datetime(targets["dt"]).dt.tz_localize(None)
    return orders, targets


def latest_prices(dt: pd.Timestamp) -> dict[str, float]:
    """Load latest available local close prices for all exposure symbols."""
    panel = cs.load_panel().sort_values("dt")
    panel = panel[panel["dt"].le(dt)]
    if panel.empty:
        raise ValueError(f"No local price data at or before {dt}")
    row = panel.iloc[-1]
    return {symbol: float(row[symbol]) for symbol in cs.SYMBOLS}


def specs_frame() -> pd.DataFrame:
    """Return default instrument specs as a DataFrame."""
    return pd.DataFrame([spec.__dict__ for spec in DEFAULT_SPECS])


def round_quantity(raw_quantity: float, lot_size: int) -> int:
    """Round a signed quantity to the nearest tradable lot."""
    if raw_quantity == 0:
        return 0
    lots = int(round(abs(raw_quantity) / lot_size))
    if lots == 0:
        return 0
    return int(np.sign(raw_quantity) * lots * lot_size)


def action_from_quantity(quantity: int) -> str:
    """Return a broker-style action label from signed quantity."""
    if quantity > 0:
        return "BUY"
    if quantity < 0:
        return "SELL_OR_SHORT"
    return "HOLD"


def build_blotter(
    orders: pd.DataFrame,
    account_equity: float,
    prices: dict[str, float],
    order_column: str,
    blotter_name: str,
) -> pd.DataFrame:
    """Build one broker-style blotter from a unit-NAV order column."""
    spec_by_exposure = {spec.exposure_symbol: spec for spec in DEFAULT_SPECS}
    rows = []
    for _, order in orders.iterrows():
        exposure_symbol = str(order["symbol"])
        spec = spec_by_exposure[exposure_symbol]
        weight_delta = float(order[order_column])
        target_weight = float(order["target_weight"])
        price = prices[exposure_symbol]
        notional_delta = weight_delta * account_equity
        contract_value = price * spec.multiplier
        raw_quantity = notional_delta / contract_value if contract_value else 0.0
        rounded_quantity = round_quantity(raw_quantity, spec.lot_size)
        rounded_notional = rounded_quantity * contract_value
        residual_notional = notional_delta - rounded_notional
        side = action_from_quantity(rounded_quantity)
        violates_short_rule = bool(rounded_quantity < 0 and not spec.allow_short)
        rows.append(
            {
                "blotter": blotter_name,
                "dt": order["dt"],
                "account_equity": account_equity,
                "exposure_symbol": exposure_symbol,
                "instrument_symbol": spec.symbol,
                "instrument_name": spec.instrument_name,
                "instrument_type": spec.instrument_type,
                "target_weight": target_weight,
                "order_weight_delta": weight_delta,
                "price": price,
                "multiplier": spec.multiplier,
                "contract_or_share_value": contract_value,
                "target_notional_delta": notional_delta,
                "raw_quantity": raw_quantity,
                "rounded_quantity": rounded_quantity,
                "action": side,
                "rounded_notional": rounded_notional,
                "residual_notional": residual_notional,
                "estimated_margin": abs(rounded_notional) * spec.margin_ratio,
                "margin_ratio": spec.margin_ratio,
                "allow_short": spec.allow_short,
                "violates_short_rule": violates_short_rule,
                "notes": spec.notes,
            }
        )
    return pd.DataFrame(rows)


def target_positions(targets: pd.DataFrame, account_equity: float, prices: dict[str, float]) -> pd.DataFrame:
    """Build target positions from target weights."""
    spec_by_exposure = {spec.exposure_symbol: spec for spec in DEFAULT_SPECS}
    rows = []
    for _, row in targets.iterrows():
        exposure_symbol = str(row["symbol"])
        spec = spec_by_exposure[exposure_symbol]
        weight = float(row["target_weight"])
        price = prices[exposure_symbol]
        target_notional = weight * account_equity
        contract_value = price * spec.multiplier
        raw_quantity = target_notional / contract_value if contract_value else 0.0
        rounded_quantity = round_quantity(raw_quantity, spec.lot_size)
        rounded_notional = rounded_quantity * contract_value
        rows.append(
            {
                "dt": row["dt"],
                "account_equity": account_equity,
                "exposure_symbol": exposure_symbol,
                "instrument_symbol": spec.symbol,
                "target_weight": weight,
                "price": price,
                "target_notional": target_notional,
                "raw_quantity": raw_quantity,
                "rounded_quantity": rounded_quantity,
                "rounded_notional": rounded_notional,
                "residual_notional": target_notional - rounded_notional,
                "estimated_margin": abs(rounded_notional) * spec.margin_ratio,
                "requires_short_or_hedge": bool(weight < 0),
            }
        )
    return pd.DataFrame(rows)


def portfolio_summary(positions: pd.DataFrame, model_blotter: pd.DataFrame, flat_blotter: pd.DataFrame) -> pd.DataFrame:
    """Summarize target and order-level exposure."""
    return pd.DataFrame(
        [
            {
                "dt": positions["dt"].iloc[0],
                "account_equity": positions["account_equity"].iloc[0],
                "target_gross_notional": float(positions["target_notional"].abs().sum()),
                "target_net_notional": float(positions["target_notional"].sum()),
                "rounded_gross_notional": float(positions["rounded_notional"].abs().sum()),
                "rounded_net_notional": float(positions["rounded_notional"].sum()),
                "estimated_margin": float(positions["estimated_margin"].sum()),
                "model_order_abs_notional": float(model_blotter["rounded_notional"].abs().sum()),
                "flat_start_order_abs_notional": float(flat_blotter["rounded_notional"].abs().sum()),
                "short_rule_violations": int(
                    model_blotter["violates_short_rule"].sum() + flat_blotter["violates_short_rule"].sum()
                ),
            }
        ]
    )


def save_readme(
    output_dir: Path,
    account_equity: float,
    positions: pd.DataFrame,
    model_blotter: pd.DataFrame,
    flat_blotter: pd.DataFrame,
) -> None:
    """Save broker blotter readme."""
    latest_dt = pd.to_datetime(positions["dt"].iloc[0]).date()
    lines = [
        "# Daily CTA Broker Blotter",
        "",
        f"Latest signal date: {latest_dt}",
        f"Account equity assumption: {account_equity:,.2f}",
        "",
        "This is a broker-blotter template, not a connected broker order.",
        "Replace placeholder instruments, contract multipliers, margin ratios and lot sizes with broker-approved values.",
        "",
        "## Files",
        "",
        "- `broker_order_blotter_model_delta.csv`: orders implied by the latest model rebalance.",
        "- `broker_order_blotter_flat_start.csv`: orders for a new account starting from zero exposure.",
        "- `target_positions_blotter.csv`: target positions after rounding.",
        "- `portfolio_margin_summary.csv`: notional and margin proxy summary.",
        "- `broker_instrument_config_template.csv`: editable sizing and instrument config.",
        "",
        "## Latest Model Orders",
        "",
    ]
    for _, row in model_blotter.iterrows():
        lines.append(
            f"- {row['instrument_symbol']}: {row['action']} {int(row['rounded_quantity'])}, "
            f"target delta {row['target_notional_delta']:,.2f}, rounded {row['rounded_notional']:,.2f}"
        )
    lines.extend(["", "## Flat-Start Orders", ""])
    for _, row in flat_blotter.iterrows():
        lines.append(
            f"- {row['instrument_symbol']}: {row['action']} {int(row['rounded_quantity'])}, "
            f"target delta {row['target_notional_delta']:,.2f}, rounded {row['rounded_notional']:,.2f}"
        )
    lines.extend(
        [
            "",
            "## Cautions",
            "",
            "- 000852 and 000905 are index exposure targets. Bind placeholders to real futures, ETF, swap or hedge instruments.",
            "- `model_delta` is zero when the strategy has no new rebalance on the latest signal date.",
            "- Use `flat_start` only to initialize a new account from zero exposure.",
            "- Rounding residuals are expected; review them against account size and broker lot rules.",
        ]
    )
    (output_dir / "broker_blotter_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Export broker blotters."""
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    orders, targets = load_inputs()
    signal_dt = pd.to_datetime(orders["dt"].max())
    prices = latest_prices(signal_dt)

    model_blotter = build_blotter(orders, args.account_equity, prices, "model_delta_weight", "model_delta")
    flat_blotter = build_blotter(orders, args.account_equity, prices, "flat_start_delta_weight", "flat_start")
    positions = target_positions(targets, args.account_equity, prices)
    summary = portfolio_summary(positions, model_blotter, flat_blotter)
    config = specs_frame()

    model_blotter.to_csv(output_dir / "broker_order_blotter_model_delta.csv", index=False, encoding="utf-8-sig")
    flat_blotter.to_csv(output_dir / "broker_order_blotter_flat_start.csv", index=False, encoding="utf-8-sig")
    positions.to_csv(output_dir / "target_positions_blotter.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "portfolio_margin_summary.csv", index=False, encoding="utf-8-sig")
    config.to_csv(output_dir / "broker_instrument_config_template.csv", index=False, encoding="utf-8-sig")
    save_readme(output_dir, args.account_equity, positions, model_blotter, flat_blotter)

    print("Model delta blotter:")
    print(model_blotter[["instrument_symbol", "action", "rounded_quantity", "target_notional_delta", "rounded_notional"]].to_string(index=False))
    print("\nFlat-start blotter:")
    print(flat_blotter[["instrument_symbol", "action", "rounded_quantity", "target_notional_delta", "rounded_notional"]].to_string(index=False))
    print("\nPortfolio summary:")
    print(summary.to_string(index=False))
    print(f"\noutputs: {output_dir}")


if __name__ == "__main__":
    main()
