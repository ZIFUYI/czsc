"""DataFrame adapters to the shared Rust execution ledger for research scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd

import czsc
from examples.multi_factor_cta.data_engine import assign_trading_date


def futures_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completed intraday bars by futures session; omit the unfinished final day."""
    data = frame.sort_values("dt").copy()
    data["trading_day"] = pd.to_datetime(assign_trading_date(data.dt, True))
    daily = data.groupby("trading_day", sort=True).agg(
        dt=("dt", "max"),
        symbol=("symbol", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        vol=("vol", "sum"),
        amount=("amount", "sum"),
    )
    complete = daily.dt >= daily.index + pd.Timedelta(hours=15)
    return daily.loc[complete].reset_index(drop=True)


def futures_daily_bars(frame: pd.DataFrame) -> list:
    """Convert explicitly session-aggregated research inputs to native daily bars."""
    return czsc.format_standard_kline(futures_daily_frame(frame), freq=czsc.Freq.D)


def trading_day_returns(returns: pd.Series) -> pd.Series:
    """Aggregate fixed-capital PnL on the real futures session clock."""
    days = pd.DatetimeIndex(assign_trading_date(pd.Series(returns.index), True))
    return returns.groupby(days).sum()


def allocated_execution_ledger(results: list[dict], allocations: pd.DataFrame, frequencies: dict[str, str]) -> dict:
    """Apply daily capital budgets to signed positions and recompute actual trades and fees in Rust."""
    prices, weights = {}, {}
    for item in results:
        symbol = item["symbol"]
        frame = item[frequencies[symbol] + "_inputs"].copy()
        if frame.empty:
            continue
        frame = frame.set_index("dt").sort_index()
        executed = frame.pos.shift(1).fillna(0.0)
        days = pd.Index(assign_trading_date(pd.Series(frame.index), True).astype(str))
        keep = days.isin(allocations.index)
        frame, executed, days = frame.loc[keep], executed.loc[keep], days[keep]
        if frame.empty:
            continue
        budget = allocations[symbol].reindex(days).to_numpy(dtype=float)
        price = frame.open.copy()
        target = executed * budget
        terminal = frame.index[-1] + pd.Timedelta(microseconds=1)
        price.loc[terminal] = frame.close.iloc[-1]
        target.loc[terminal] = 0.0
        prices[symbol], weights[symbol] = price, target
    price_df = pd.DataFrame(prices).sort_index()
    mask = price_df.notna()
    weight_df = pd.DataFrame(weights).reindex_like(price_df).ffill().fillna(0.0)
    returns = (
        pd.DataFrame({symbol: price.dropna().pct_change().fillna(0.0) for symbol, price in prices.items()})
        .reindex_like(price_df)
        .fillna(0.0)
    )
    return replay_allocated_execution(
        {
            "execution_prices_df": price_df,
            "weights_df": weight_df,
            "returns_df": returns,
            "tradable_mask": mask,
            "fee_rates": dict.fromkeys(price_df.columns, 0.0001),
        }
    )


def replay_allocated_execution(bundle: dict) -> dict:
    """Reuse native fixed-capital accounting for frozen, already executable allocations."""
    returns, weights, mask = bundle["returns_df"], bundle["weights_df"], bundle["tradable_mask"]
    ledgers = {
        s: czsc.executed_weight_ledger(
            returns[s].tolist(), weights[s].tolist(), fee_rate=bundle["fee_rates"][s], tradable=mask[s].tolist()
        )
        for s in returns
    }
    panels = {
        name: pd.DataFrame({s: getattr(led, field) for s, led in ledgers.items()}, index=returns.index)
        for name, field in {
            "gross": "gross_returns",
            "fee": "cost_returns",
            "net": "net_returns",
            "turnover": "turnover",
        }.items()
    }
    daily = pd.DataFrame({name: trading_day_returns(panel.sum(axis=1)) for name, panel in panels.items()})
    daily.index = daily.index.strftime("%Y-%m-%d")
    return {**bundle, "panels": panels, "daily": daily}


def equal_weight_execution(results: list[dict], dates: list[str], frequency: str) -> dict:
    """Execute a fixed equal capital allocation, retaining unavailable assets as cash."""
    symbols = [item["symbol"] for item in results]
    allocations = pd.DataFrame(1.0 / len(symbols), index=dates, columns=symbols)
    return allocated_execution_ledger(results, allocations, dict.fromkeys(symbols, frequency))


def get_trading_day(dt: pd.Timestamp) -> str:
    """Use the research futures clock, including the Saturday continuation of Friday night."""
    return str(assign_trading_date(pd.Series([pd.Timestamp(dt)]), has_night_session=True).iloc[0])


def lagged_daily_volatility_multiplier(
    bar_returns: pd.Series,
    *,
    target_vol: float,
    window: int = 120,
    min_periods: int = 20,
    annual_days: int = 252,
    warmup: float = 2.0,
) -> pd.Series:
    """Expose each complete trading day's volatility only to subsequent trading days."""
    dates = pd.Series(
        assign_trading_date(pd.Series(bar_returns.index), True).astype(str).to_numpy(), index=bar_returns.index
    )
    daily = bar_returns.groupby(dates, sort=True).sum()
    volatility = daily.rolling(window, min_periods=min_periods).std().shift(1) * np.sqrt(annual_days)
    multiplier = (target_vol / volatility.replace(0, np.nan)).fillna(warmup).clip(0.5, 4.0)
    return dates.map(multiplier).astype(float)


def next_open_panel_ledger(
    combined: pd.DataFrame,
    targets: pd.DataFrame,
    fee_rate: float,
    *,
    liquidate: bool = True,
) -> dict[str, pd.DataFrame]:
    """Account close decisions on each asset's next actual open using Rust; never trade a missing quote.

    Returns use fixed initial capital. An optional terminal close is charged and
    included in the final observed row; it does not invent another trading date.
    """
    fields = {"gross": "gross_returns", "fee": "cost_returns", "net": "net_returns", "turnover": "turnover"}
    panels = {name: pd.DataFrame(0.0, index=targets.index, columns=targets.columns) for name in fields}
    for symbol in targets.columns:
        observed = combined.loc[combined.symbol == symbol].set_index("dt").sort_index()
        observed = observed.loc[observed.index.intersection(targets.index)]
        if observed.empty:
            continue
        decisions = targets.loc[observed.index, symbol].to_numpy(dtype=float)
        prices = observed.open.to_numpy(dtype=float).tolist()
        executed = np.r_[0.0, decisions[:-1]].tolist()
        if liquidate:
            prices.append(float(observed.close.iloc[-1]))
            executed.append(0.0)
        ledger = czsc.execution_ledger(prices, executed, fee_rate=fee_rate)
        for name, field in fields.items():
            values = np.asarray(getattr(ledger, field), dtype=float)
            if liquidate:
                values[-2] += values[-1]
                values = values[:-1]
            panels[name].loc[observed.index, symbol] = values
    return panels


def signed_holds_execution(
    long_holds: pd.DataFrame,
    short_holds: pd.DataFrame,
    bars: list,
    *,
    fee_rate: float = 0.0001,
) -> pd.DataFrame:
    """Join signed native holds to observable opens and reuse native next-open accounting."""
    if not long_holds.index.equals(short_holds.index):
        raise ValueError("Long and short holds must have the same decision timestamps")
    data = pd.DataFrame(
        {"dt": [bar.dt for bar in bars], "open": [bar.open for bar in bars], "close": [bar.close for bar in bars]}
    ).set_index("dt")
    data = data.loc[long_holds.index].copy()
    weights = (long_holds.pos + short_holds.pos).astype(float)
    ledger = czsc.decision_execution_ledger(
        data.open.tolist(), weights.tolist(), fee_rate=fee_rate, terminal_close=float(data.close.iloc[-1])
    )
    result = pd.DataFrame({"price": data.close, "pos": weights, "open": data.open, "close": data.close})
    for name, field in {
        "gross_ret": "gross_returns",
        "cost_ret": "cost_returns",
        "strat_ret": "net_returns",
        "turnover": "turnover",
    }.items():
        values = np.asarray(getattr(ledger, field), dtype=float)
        values[-2] += values[-1]
        result[name] = values[:-1]
    return result
