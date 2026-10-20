"""Replay final research exposures through actual Backtrader and CZSC brokers.

This is an engine IO/diagnostic adapter, not a signal generator. Inputs are the
formal portfolio's *executed* signed weights and the corresponding execution
prices, including its explicit terminal close. Fractional units and a fixed unit
cash margin model allow continuous futures-style exposures; contract lots, exchange
margin changes, financing, delivery and roll costs require a separate audit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import czsc


def _validate_inputs(prices: pd.DataFrame, weights: pd.DataFrame, fees: dict[str, float]) -> None:
    if not prices.index.equals(weights.index) or not prices.columns.equals(weights.columns):
        raise ValueError("Execution prices and final signed weights must be exactly aligned")
    if len(prices) < 2 or not len(prices.columns):
        raise ValueError("At least two execution marks and one asset are required")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Execution timestamps must be strictly increasing and unique")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Execution timestamps must be a DatetimeIndex")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("All execution prices must be finite and positive")
    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("All executed signed weights must be finite")
    if (weights.iloc[[0, -1]].abs() > 1e-12).any().any():
        raise ValueError("The replay must start flat and include explicit terminal liquidation")
    if any(symbol not in fees or not np.isfinite(fees[symbol]) or fees[symbol] < 0 for symbol in prices):
        raise ValueError("Every asset requires a finite nonnegative single-side fee")


def _backtrader_replay(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    fees: dict[str, float],
    initial_capital: float,
    sizing: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    import backtrader as bt

    class ExecutionFeed(bt.feeds.PandasData):
        lines = ("target_weight",)
        params = (("target_weight", -1),)

    class FinalWeightReplay(bt.Strategy):
        def __init__(self):
            self.rows: list[dict[str, Any]] = []
            self.fills: list[dict[str, Any]] = []
            self.previous_equity = initial_capital
            self.previous_prices = prices.iloc[0].to_numpy(dtype=float)
            self.commission_by_bar: dict[int, float] = {}

        def next_open(self):
            # Weights are already decisions shifted to their executable opens.
            # Only execution prices are inspected here; no signal is evaluated.
            current = np.array([data.open[0] for data in self.datas])
            held = np.array([self.getposition(data).size for data in self.datas])
            marked_equity = self.previous_equity + float(np.dot(held, current - self.previous_prices))
            budget = initial_capital if sizing == "fixed_capital" else marked_equity
            if budget <= 0:
                raise ValueError("Backtrader account exhausted its sizing capital")
            for data in self.datas:
                quantity = float(data.target_weight[0]) * budget / float(data.open[0])
                difference = quantity - self.getposition(data).size
                if abs(difference) <= 1e-12:
                    continue
                if difference > 0:
                    self.buy(data=data, size=difference)
                else:
                    self.sell(data=data, size=-difference)

        def notify_order(self, order):
            if order.status in (order.Submitted, order.Accepted):
                return
            i = len(self.data0) - 1
            if order.status != order.Completed:
                raise RuntimeError(f"Backtrader order {order.ref} was {order.getstatusname()} at execution row {i}")
            self.commission_by_bar[i] = self.commission_by_bar.get(i, 0.0) + order.executed.comm
            self.fills.append(
                {
                    "sizing": sizing,
                    "execution_row": i,
                    "dt": prices.index[i],
                    "symbol": order.data._name,
                    "quantity": order.executed.size,
                    "price": order.executed.price,
                    "commission": order.executed.comm,
                    "position_after": self.getposition(order.data).size,
                }
            )

        def next(self):
            i = len(self.data0) - 1
            equity = float(self.broker.getvalue())
            commission = self.commission_by_bar.get(i, 0.0)
            pnl = equity - self.previous_equity
            denominator = initial_capital if sizing == "fixed_capital" else self.previous_equity
            self.rows.append(
                {
                    "equity": equity,
                    "net_return": pnl / denominator,
                    "gross_return": (pnl + commission) / denominator,
                    "cost_return": commission / denominator,
                    "commission": commission,
                }
            )
            self.previous_equity = equity
            self.previous_prices = np.array([data.close[0] for data in self.datas])

    cerebro = bt.Cerebro(stdstats=False, cheat_on_open=True)
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.set_coo(True)
    # Backtrader's floating-point date representation cannot distinguish a
    # terminal close one microsecond after an open label. The ordinal engine
    # clock preserves execution order; every artifact retains the real label.
    clock = pd.date_range("2000-01-01", periods=len(prices), freq="min")
    for symbol in prices:
        px = prices[symbol].to_numpy(dtype=float)
        frame = pd.DataFrame(
            {
                "open": px,
                "high": px,
                "low": px,
                "close": px,
                "volume": 0.0,
                "openinterest": 0.0,
                "target_weight": weights[symbol].to_numpy(dtype=float),
            },
            index=clock,
        )
        cerebro.adddata(ExecutionFeed(dataname=frame), name=str(symbol))
        cerebro.broker.addcommissioninfo(
            bt.CommInfoBase(
                commission=fees[symbol],
                commtype=bt.CommInfoBase.COMM_PERC,
                stocklike=False,
                percabs=True,
                margin=1.0,
                mult=1.0,
            ),
            name=str(symbol),
        )
    cerebro.addstrategy(FinalWeightReplay)
    strategy = cerebro.run()[0]
    inventory = {data._name: float(strategy.getposition(data).size) for data in strategy.datas}
    if any(abs(quantity) > 1e-9 for quantity in inventory.values()):
        raise RuntimeError("Backtrader replay did not liquidate all positions")
    ledger = pd.DataFrame(strategy.rows, index=prices.index)
    fills = pd.DataFrame(
        strategy.fills,
        columns=["sizing", "execution_row", "dt", "symbol", "quantity", "price", "commission", "position_after"],
    )
    return ledger, fills, inventory


def run_dual_engine_validation(
    execution_prices: pd.DataFrame,
    signed_weights: pd.DataFrame,
    fee_rates: dict[str, float],
    *,
    initial_capital: float = 1_000_000.0,
) -> dict[str, Any]:
    """Reconcile fixed-capital fills, including price-drift rebalancing costs.

    Every executable mark targets ``weight * initial_capital / price`` units.
    Rust owns this accounting contract and the actual traded-notional costs.
    Backtrader independently sizes and executes orders against its broker.

    External WBT supports weight-delta fees only. Its raw replay therefore uses
    zero fees and independently verifies gross PnL. ``wbt_standardized`` passes
    that raw PnL and the shared Rust costs back to Rust for net accounting; it is
    explicitly a shared-cost result, not an independent WBT commission engine.
    Equity-fraction Backtrader sizing remains a separate diagnostic account.
    """
    _validate_inputs(execution_prices, signed_weights, fee_rates)
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("Initial capital must be finite and positive")
    absolute_bar_tolerance = 1e-10
    absolute_total_tolerance = 1e-9
    reference_assets = {}
    for symbol in execution_prices:
        native = czsc.execution_ledger(
            execution_prices[symbol].to_list(),
            signed_weights[symbol].to_list(),
            fee_rate=fee_rates[symbol],
            initial_capital=initial_capital,
        )
        reference_assets[symbol] = pd.DataFrame(
            {
                "gross_return": native.gross_returns,
                "cost_return": native.cost_returns,
                "net_return": native.net_returns,
                "turnover": native.turnover,
                "pretrade_weight": native.pretrade_weights,
                "executed_weight": native.executed_weights,
                "quantity": native.quantities,
                "trade_quantity": native.trade_quantities,
            },
            index=execution_prices.index,
        )
    metrics = ["gross_return", "cost_return", "net_return"]
    reference = pd.concat([frame[metrics] for frame in reference_assets.values()]).groupby(level=0).sum()
    fixed, fixed_orders, fixed_inventory = _backtrader_replay(
        execution_prices, signed_weights, fee_rates, initial_capital, "fixed_capital"
    )
    equity, equity_orders, equity_inventory = _backtrader_replay(
        execution_prices, signed_weights, fee_rates, initial_capital, "equity_fraction"
    )
    wbt_summaries = []
    wbt_dailys = []
    raw_assets = {}
    standardized_assets = {}
    # WBT exposes daily rather than bar rows. One ordinal day per execution mark
    # exposes its intervals; all reports are aggregated on the original clock.
    wbt_clock = pd.date_range("2000-01-01", periods=len(execution_prices) + 1, freq="D", unit="us")
    for symbol in execution_prices:
        frame = pd.DataFrame(
            {
                "dt": wbt_clock[:-1],
                "symbol": str(symbol),
                "price": execution_prices[symbol].to_numpy(dtype=float),
                "weight": signed_weights[symbol].to_numpy(dtype=float),
            }
        )
        sentinel = frame.iloc[[-1]].copy()
        sentinel["dt"] = wbt_clock[-1]
        checked = czsc.WeightBacktest(
            pd.concat([frame, sentinel], ignore_index=True), digits=12, fee_rate=0.0, weight_type="ts"
        )
        native_bars = checked.dailys.sort_values("date").reset_index(drop=True)
        if len(native_bars) != len(execution_prices):
            raise RuntimeError("WBT did not preserve the ordinal execution marks")
        raw = pd.DataFrame(
            {
                "gross_return": native_bars["edge"].shift(1, fill_value=0.0).to_numpy(),
                "cost_return": native_bars["cost"].to_numpy(),
                "net_return": native_bars["return"].shift(1, fill_value=0.0).to_numpy(),
            },
            index=execution_prices.index,
        )
        standardized = pd.DataFrame(
            {
                "gross_return": raw["gross_return"],
                "cost_return": reference_assets[symbol]["cost_return"],
                "net_return": czsc.standardize_execution_returns(
                    raw["gross_return"].to_list(), reference_assets[symbol]["cost_return"].to_list()
                ),
            },
            index=execution_prices.index,
        )
        raw_assets[symbol] = raw
        standardized_assets[symbol] = standardized
        for engine, ledger in (("wbt_raw", raw), ("wbt_standardized", standardized)):
            calendar = ledger.groupby(ledger.index.normalize()).sum().reset_index(names="dt")
            calendar["symbol"] = symbol
            calendar["engine"] = engine
            wbt_dailys.append(calendar)
        wbt_summaries.append(
            {
                "symbol": symbol,
                **{f"raw_{metric}": float(raw[metric].sum()) for metric in metrics},
                **{f"standardized_{metric}": float(standardized[metric].sum()) for metric in metrics},
            }
        )
    wbt_assets = pd.DataFrame(wbt_summaries).set_index("symbol")
    bars = pd.concat(
        {
            "formal": reference,
            "backtrader_fixed": fixed,
            "backtrader_equity": equity,
            "wbt_raw": pd.concat(raw_assets.values()).groupby(level=0).sum(),
            "wbt_standardized": pd.concat(standardized_assets.values()).groupby(level=0).sum(),
        },
        axis=1,
    )
    daily = bars.loc[:, bars.columns.get_level_values(1).isin(metrics)]
    daily = daily.groupby(daily.index.normalize()).sum()
    # This independent diagnostic account is sized from actual broker NAV.
    # Account percentages compound within a day; fixed-capital returns add.
    account_daily_equity = equity["equity"].groupby(equity.index.normalize()).last()
    account_day_start = account_daily_equity.shift(1, fill_value=initial_capital)
    account_daily_returns = account_daily_equity / account_day_start - 1.0
    account_daily_costs = equity["commission"].groupby(equity.index.normalize()).sum() / account_day_start
    daily[("backtrader_equity", "net_return")] = account_daily_returns
    daily[("backtrader_equity", "cost_return")] = account_daily_costs
    daily[("backtrader_equity", "gross_return")] = account_daily_returns + account_daily_costs
    account_nav = np.r_[initial_capital, equity["equity"].to_numpy(dtype=float)]
    account_final_multiple = account_nav[-1] / initial_capital
    account_daily_stats = czsc.daily_performance(
        np.ascontiguousarray(account_daily_returns.to_numpy(dtype=float)), yearly_days=252
    )
    comparisons = {}
    for engine, checked_metrics in (
        ("backtrader_fixed", metrics),
        ("wbt_raw", ["gross_return"]),
        ("wbt_standardized", metrics),
    ):
        comparisons[engine] = {}
        for metric in checked_metrics:
            errors = bars[engine][metric] - reference[metric]
            bar_error = float(errors.abs().max())
            total_error = abs(float(bars[engine][metric].sum() - reference[metric].sum()))
            comparisons[engine][metric] = {
                "max_bar_absolute_error": bar_error,
                "total_absolute_error": total_error,
                "passed": bar_error <= absolute_bar_tolerance and total_error <= absolute_total_tolerance,
            }
    net_difference = float(fixed["net_return"].sum() - bars["wbt_standardized"]["net_return"].sum())
    wbt_net = float(bars["wbt_standardized"]["net_return"].sum())
    summary = {
        "execution_marks": len(execution_prices),
        "assets": len(execution_prices.columns),
        "initial_capital": initial_capital,
        "capital_basis": "fixed_initial_capital",
        "rebalance_policy": "target_quantity_at_every_executable_mark",
        "quantity_rule": "target_weight * initial_capital / execution_price",
        "cost_rule": "abs(target_quantity - held_quantity) * execution_price * single_side_fee / initial_capital",
        "wbt_raw_fee_rate": 0.0,
        "wbt_standardization": "WBT raw gross plus shared Rust actual-quantity costs; Rust computes net returns",
        "nonzero_weight_marks": int((signed_weights.abs() > 1e-12).to_numpy().sum()),
        **{
            f"{engine}_{metric}": float(bars[engine][metric].sum())
            for engine in ("formal", "wbt_raw", "wbt_standardized", "backtrader_fixed")
            for metric in metrics
        },
        "backtrader_fixed_orders": len(fixed_orders),
        "backtrader_equity_final_return": float(equity["equity"].iloc[-1] / initial_capital - 1),
        "backtrader_equity_commission": float(equity["commission"].sum()),
        "fixed_bar_return_compounding_diagnostic": float((1 + reference["net_return"]).prod() - 1),
        "absolute_bar_tolerance": absolute_bar_tolerance,
        "absolute_total_tolerance": absolute_total_tolerance,
        "reconciliation_checks": comparisons,
        "execution_reconciled": all(check["passed"] for engine in comparisons.values() for check in engine.values()),
        "gross_execution_reconciled": comparisons["backtrader_fixed"]["gross_return"]["passed"],
        "fixed_gross_max_bar_error": comparisons["backtrader_fixed"]["gross_return"]["max_bar_absolute_error"],
        "fixed_cost_max_bar_error": comparisons["backtrader_fixed"]["cost_return"]["max_bar_absolute_error"],
        "fixed_net_max_bar_error": comparisons["backtrader_fixed"]["net_return"]["max_bar_absolute_error"],
        "wbt_standardized_formal_net_difference": wbt_net - float(reference["net_return"].sum()),
        "wbt_standardized_formal_max_bar_error": comparisons["wbt_standardized"]["net_return"][
            "max_bar_absolute_error"
        ],
        "wbt_standardized_formal_max_daily_error": float(
            (daily["wbt_standardized"]["net_return"] - daily["formal"]["net_return"]).abs().max()
        ),
        "bt_fixed_wbt_standardized_net_difference": net_difference,
        "bt_fixed_wbt_standardized_relative_net_difference": (
            abs(net_difference) / abs(wbt_net) if abs(wbt_net) > 1e-12 else None
        ),
        "flat_path": not bool((signed_weights.abs() > 1e-12).any().any()),
        "terminal_inventory": max(abs(value) for value in [*fixed_inventory.values(), *equity_inventory.values()]),
        "backtrader_fixed_terminal_positions": fixed_inventory,
        "backtrader_equity_terminal_positions": equity_inventory,
        "daily_metrics": {
            engine: czsc.daily_performance(
                np.ascontiguousarray(daily[engine]["net_return"].to_numpy(dtype=float)), yearly_days=252
            )
            for engine in ("formal", "wbt_standardized", "backtrader_fixed")
        },
        "equity_nav_metrics": {
            "final_account_return": float(account_final_multiple - 1),
            "account_max_drawdown": float((1.0 - account_nav / np.maximum.accumulate(account_nav)).max()),
            "annualized_account_return_252_days": (
                float(account_final_multiple ** (252.0 / len(account_daily_equity)) - 1)
                if account_final_multiple > 0
                else None
            ),
            "calendar_daily_sharpe": account_daily_stats["夏普"],
            "calendar_days_observed": len(account_daily_equity),
        },
        "notes": [
            "Formal Rust and Backtrader fixed-capital accounts rebalance quantities at every execution mark and charge actual traded notional.",
            "External WBT runs with zero fee; wbt_standardized uses the shared Rust cost ledger, not an independent WBT commission implementation.",
            "WBT interval PnL is moved from its start to its end mark; a flat same-price sentinel preserves terminal accounting.",
            "Per-asset final portfolio budgets are summed, not equally weighted a second time.",
            "Installed WBT 0.2.1 has an observed high-precision gross mismatch despite digits=12; strict checks reject affected paths without rounding frozen targets.",
            "Fractional units, fixed 1-currency-unit simulation margin, zero spread/slippage beyond the supplied combined fee; no lot/roll modelling.",
            "The ordinal broker clock only preserves execution ordering; orders retain the original execution timestamps.",
            "Equity-account NAV sizing and metrics are separate diagnostics, never substituted for the fixed-capital contract.",
        ],
    }
    return {
        "summary": summary,
        "bar_ledger": bars,
        "daily_ledger": daily,
        "native_execution_ledger": pd.concat(reference_assets, axis=1),
        "wbt_bar_ledger": pd.concat(standardized_assets, axis=1),
        "wbt_raw_bar_ledger": pd.concat(raw_assets, axis=1),
        "orders": pd.concat([fixed_orders, equity_orders], ignore_index=True),
        "wbt_assets": wbt_assets,
        "wbt_daily": pd.concat(wbt_dailys, ignore_index=True),
    }
