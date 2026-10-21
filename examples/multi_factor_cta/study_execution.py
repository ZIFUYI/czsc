"""Research orchestration over the shared Rust executable portfolio ledger."""

from __future__ import annotations

import numpy as np
import pandas as pd

from examples.multi_factor_cta.synthesis_engine import (
    account_portfolio_execution,
    assign_futures_trading_day,
    build_asset_execution_inputs,
)


def study_execution_ledger(frame: pd.DataFrame, weights: np.ndarray, fee_rate: float) -> dict:
    """Execute close decisions at subsequent opens and liquidate at the final close.

    The frame must already be restricted to the training fold. All PnL, quantity
    rebalancing and fees come from the same Rust ledger used by final portfolios.
    """
    if not {"dt", "open", "close"}.issubset(getattr(frame, "columns", ())):
        raise TypeError("An execution OHLC DataFrame with timestamps is required")
    returns, positions = build_asset_execution_inputs(frame, weights)
    ledger = account_portfolio_execution(returns.to_frame("asset"), positions.to_frame("asset"), {"asset": fee_rate})
    net = ledger["bar_net_returns"]
    ledger["daily_returns"] = net.groupby(assign_futures_trading_day(net.index)).sum()
    return ledger


def execution_sharpe(frame: pd.DataFrame, weights: np.ndarray, fee_rate: float) -> float:
    """Compute the study's annualized Sharpe on actual futures trading days."""
    daily = study_execution_ledger(frame, weights, fee_rate)["daily_returns"]
    sd = float(daily.std(ddof=1))
    return float(daily.mean() / sd * np.sqrt(252)) if len(daily) >= 2 and sd > 1e-12 else 0.0
