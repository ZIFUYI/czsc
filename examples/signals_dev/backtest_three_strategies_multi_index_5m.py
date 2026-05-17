# -*- coding: utf-8 -*-
"""Backtest RBreaker, Dual Thrust, and ATR intraday strategies on multiple indices."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "examples" / "results" / "three_strategy_multi_index_jq_sdk"


def load_jq_credentials() -> None:
    """Load JQData credentials from .env when present."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_jq_price(symbol: str, frequency: str, sdt: str, edt: str) -> pd.DataFrame:
    """Read JoinQuant price data and normalize columns."""
    import jqdatasdk as jq

    load_jq_credentials()
    user = os.getenv("JQDATA_USERNAME")
    password = os.getenv("JQDATA_PASSWORD")
    if not user or not password:
        raise RuntimeError("请先设置 JQDATA_USERNAME / JQDATA_PASSWORD")

    jq.auth(user, password)
    df = jq.get_price(
        symbol,
        start_date=pd.to_datetime(sdt).strftime("%Y-%m-%d"),
        end_date=pd.to_datetime(edt).strftime("%Y-%m-%d"),
        frequency=frequency,
        fields=["open", "close", "high", "low", "volume", "money"],
        fq=None,
    )
    if df is None or df.empty:
        raise RuntimeError(f"{symbol} {frequency} 没有返回数据")

    df = df.sort_index().reset_index(names="dt")
    df.loc[:, "symbol"] = symbol
    df.loc[:, "dt"] = pd.to_datetime(df["dt"])
    df.loc[:, "trade_date"] = df["dt"].dt.normalize()
    df = df.rename(columns={"volume": "vol", "money": "amount"})
    return df[["symbol", "dt", "trade_date", "open", "close", "high", "low", "vol", "amount"]].copy()


def make_rbreaker_levels(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate RBreaker levels for each trading day from previous day HLC."""
    rows = []
    daily = daily.sort_values("dt").reset_index(drop=True)
    for i in range(1, len(daily)):
        prev = daily.iloc[i - 1]
        cur = daily.iloc[i]
        high, close, low = float(prev["high"]), float(prev["close"]), float(prev["low"])
        pivot = (high + close + low) / 3
        rows.append(
            {
                "trade_date": cur["trade_date"],
                "range_date": prev["dt"],
                "break_buy": high + 2 * pivot - 2 * low,
                "break_sell": low - 2 * (high - pivot),
            }
        )
    return pd.DataFrame(rows)


def make_dual_thrust_levels(daily: pd.DataFrame, n: int = 5, k1: float = 0.2, k2: float = 0.2) -> pd.DataFrame:
    """Calculate classic Dual Thrust levels from previous N daily bars."""
    rows = []
    daily = daily.sort_values("dt").reset_index(drop=True)
    for i in range(n, len(daily)):
        window = daily.iloc[i - n : i]
        cur = daily.iloc[i]
        hh = float(window["high"].max())
        hc = float(window["close"].max())
        lc = float(window["close"].min())
        ll = float(window["low"].min())
        range_ = max(hh - lc, hc - ll)
        rows.append({"trade_date": cur["trade_date"], "range": range_, "K1": k1, "K2": k2})
    return pd.DataFrame(rows)


def add_atr(df: pd.DataFrame, n: int) -> pd.Series:
    """Calculate simple rolling ATR."""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (close.shift(1) - low).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def make_atr_signals(intraday: pd.DataFrame, timeperiod: int = 5, th: int = 30) -> pd.DataFrame:
    """Generate ATR breakout signals following tas_atr_break_V230424."""
    out = intraday.sort_values("dt").reset_index(drop=True).copy()
    atr = add_atr(out, timeperiod)
    hh = out["high"].rolling(timeperiod).max()
    ll = out["low"].rolling(timeperiod).min()
    k = th / 10
    upper = hh - k * atr
    lower = ll + k * atr
    central = (upper > out["close"]) & (out["close"] > lower)

    out.loc[:, "strategy"] = f"ATR_BREAK_N{timeperiod}_T{th}"
    out.loc[:, "signal"] = None
    out.loc[(~central) & (out["close"] > lower), "signal"] = "做多_ATR突破"
    out.loc[(~central) & (out["close"] < upper), "signal"] = "做空_ATR突破"
    return out


def simulate_level_breakout(
    intraday: pd.DataFrame,
    levels: pd.DataFrame,
    strategy: str,
    level_type: str,
    exit_time: str = "14:55",
    rbreaker_scale: float = 0.3,
) -> pd.DataFrame:
    """Simulate RBreaker or Dual Thrust one-trade-per-day breakout."""
    assert level_type in {"rbreaker", "dual_thrust"}
    exit_at = pd.to_datetime(exit_time).time()
    level_map = {row["trade_date"]: row for _, row in levels.iterrows()}
    trades = []

    for trade_date in sorted(set(intraday["trade_date"]).intersection(level_map)):
        day = intraday[intraday["trade_date"] == trade_date].sort_values("dt").reset_index(drop=True)
        if len(day) < 5:
            continue

        lv = level_map[trade_date]
        day_open = float(day.iloc[0]["open"])
        if level_type == "rbreaker":
            buy_line = day_open + rbreaker_scale * (float(lv["break_buy"]) - day_open)
            sell_line = day_open + rbreaker_scale * (float(lv["break_sell"]) - day_open)
        else:
            buy_line = day_open + float(lv["range"]) * float(lv["K1"])
            sell_line = day_open - float(lv["range"]) * float(lv["K2"])

        for i in range(len(day) - 1):
            bar = day.iloc[i]
            signal = None
            direction = None
            if float(bar["close"]) > buy_line:
                signal, direction = "做多_趋势", "多头"
            elif float(bar["close"]) < sell_line:
                signal, direction = "做空_趋势", "空头"

            if not signal:
                continue

            entry = day.iloc[i + 1]
            if entry["dt"].time() >= exit_at:
                continue

            exit_candidates = day[(day.index >= i + 1) & (day["dt"].dt.time >= exit_at)]
            exit_bar = exit_candidates.iloc[0] if not exit_candidates.empty else day.iloc[-1]
            trades.append(make_trade(strategy, entry, exit_bar, bar, signal, direction, buy_line, sell_line))
            break

    return pd.DataFrame(trades)


def simulate_signal_trades(signals: pd.DataFrame, strategy: str, exit_time: str = "14:55") -> pd.DataFrame:
    """Enter next bar open on first daily signal and exit near close."""
    exit_at = pd.to_datetime(exit_time).time()
    trades = []
    signals = signals.sort_values("dt").reset_index(drop=True)

    for _, day in signals.groupby("trade_date", sort=True):
        day = day.sort_values("dt").reset_index(drop=True)
        for i in range(len(day) - 1):
            bar = day.iloc[i]
            signal = bar["signal"]
            if not isinstance(signal, str) or not signal:
                continue

            direction = "多头" if signal.startswith("做多") else "空头"
            entry = day.iloc[i + 1]
            if entry["dt"].time() >= exit_at:
                continue

            exit_candidates = day[(day.index >= i + 1) & (day["dt"].dt.time >= exit_at)]
            exit_bar = exit_candidates.iloc[0] if not exit_candidates.empty else day.iloc[-1]
            trades.append(make_trade(strategy, entry, exit_bar, bar, signal, direction))
            break

    return pd.DataFrame(trades)


def make_trade(
    strategy: str,
    entry: pd.Series,
    exit_bar: pd.Series,
    signal_bar: pd.Series,
    signal: str,
    direction: str,
    buy_line: float | None = None,
    sell_line: float | None = None,
) -> dict:
    """Create a normalized trade record."""
    entry_price = float(entry["open"])
    exit_price = float(exit_bar["close"])
    ret = exit_price / entry_price - 1 if direction == "多头" else 1 - exit_price / entry_price
    return {
        "strategy": strategy,
        "symbol": entry["symbol"],
        "trade_date": entry["trade_date"],
        "signal_dt": signal_bar["dt"],
        "signal": signal,
        "direction": direction,
        "entry_dt": entry["dt"],
        "exit_dt": exit_bar["dt"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "ret": ret,
        "ret_bp": round(ret * 10000, 2),
        "buy_line": buy_line,
        "sell_line": sell_line,
    }


def summarize_trades(symbol: str, trades: pd.DataFrame, available_days: int) -> pd.DataFrame:
    """Summarize strategy trades."""
    rows = []
    for strategy, df in trades.groupby("strategy", sort=False):
        equity = (1 + df["ret"]).cumprod()
        drawdown = equity / equity.cummax() - 1
        rows.append(
            {
                "symbol": symbol,
                "strategy": strategy,
                "available_days": available_days,
                "trades": len(df),
                "coverage": round(len(df) / available_days, 4) if available_days else 0,
                "win_rate": round((df["ret"] > 0).mean(), 4),
                "avg_bp": round(df["ret_bp"].mean(), 2),
                "median_bp": round(df["ret_bp"].median(), 2),
                "final_nav": round(equity.iloc[-1], 4),
                "cum_ret": round(equity.iloc[-1] - 1, 4),
                "max_drawdown": round(drawdown.min(), 4),
            }
        )
    return pd.DataFrame(rows)


def align_daily(trades: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align daily returns and directions for a symbol."""
    all_days = pd.date_range(start=start, end=end, freq="D")
    returns = {}
    directions = {}
    dmap = {"多头": 1, "空头": -1}
    for strategy, df in trades.items():
        daily_ret = df.groupby("trade_date")["ret"].sum().reindex(all_days, fill_value=0.0)
        daily_dir = df.groupby("trade_date")["direction"].first().map(dmap).reindex(all_days, fill_value=0).astype(int)
        returns[strategy] = daily_ret
        directions[strategy] = daily_dir
    ret_df = pd.DataFrame(returns)
    dir_df = pd.DataFrame(directions)
    ret_df.index.name = "trade_date"
    dir_df.index.name = "trade_date"
    return ret_df, dir_df


def compare_strategies(symbol: str, returns: pd.DataFrame, directions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create pairwise correlation, direction overlap, and combo summaries."""
    pair_rows = []
    for i, a in enumerate(returns.columns):
        for b in returns.columns[i + 1 :]:
            common = (directions[a] != 0) & (directions[b] != 0)
            monthly = returns[[a, b]].groupby(returns.index.to_period("M")).sum()
            pair_rows.extend(
                [
                    {
                        "symbol": symbol,
                        "strategy_a": a,
                        "strategy_b": b,
                        "sample": "all_calendar",
                        "days": len(returns),
                        "pearson": round(returns[a].corr(returns[b]), 4),
                        "spearman": round(returns[a].corr(returns[b], method="spearman"), 4),
                    },
                    {
                        "symbol": symbol,
                        "strategy_a": a,
                        "strategy_b": b,
                        "sample": "common_active",
                        "days": int(common.sum()),
                        "pearson": round(returns.loc[common, a].corr(returns.loc[common, b]), 4),
                        "spearman": round(returns.loc[common, a].corr(returns.loc[common, b], method="spearman"), 4),
                    },
                    {
                        "symbol": symbol,
                        "strategy_a": a,
                        "strategy_b": b,
                        "sample": "monthly",
                        "days": len(monthly),
                        "pearson": round(monthly[a].corr(monthly[b]), 4),
                        "spearman": round(monthly[a].corr(monthly[b], method="spearman"), 4),
                    },
                ]
            )

    direction_rows = []
    for i, a in enumerate(directions.columns):
        for b in directions.columns[i + 1 :]:
            common = (directions[a] != 0) & (directions[b] != 0)
            same = common & (directions[a] == directions[b])
            direction_rows.append(
                {
                    "symbol": symbol,
                    "strategy_a": a,
                    "strategy_b": b,
                    "a_active_days": int((directions[a] != 0).sum()),
                    "b_active_days": int((directions[b] != 0).sum()),
                    "common_active_days": int(common.sum()),
                    "same_direction_ratio": round(same.sum() / common.sum(), 4) if common.sum() else 0,
                }
            )

    combo = returns.copy()
    combo["EqualWeight_3"] = returns.mean(axis=1)
    combo_rows = []
    for name in combo.columns:
        nav = (1 + combo[name]).cumprod()
        drawdown = nav / nav.cummax() - 1
        combo_rows.append(
            {
                "symbol": symbol,
                "strategy_or_combo": name,
                "active_days": int((combo[name] != 0).sum()),
                "final_nav": round(nav.iloc[-1], 4),
                "cum_ret": round(nav.iloc[-1] - 1, 4),
                "max_drawdown": round(drawdown.min(), 4),
                "ret_dd_ratio": round((nav.iloc[-1] - 1) / abs(drawdown.min()), 2) if drawdown.min() < 0 else None,
            }
        )

    return pd.DataFrame(pair_rows), pd.DataFrame(direction_rows), pd.DataFrame(combo_rows)


def plot_symbol(symbol: str, returns: pd.DataFrame, out_dir: Path) -> None:
    """Plot equity curves."""
    plot_df = returns.copy()
    plot_df["EqualWeight_3"] = returns.mean(axis=1)
    plt.figure(figsize=(12, 5))
    for col in plot_df.columns:
        nav = (1 + plot_df[col]).cumprod()
        plt.plot(nav.index, nav, label=col)
    plt.title(f"{symbol} 5m Intraday Strategy Equity")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{symbol.replace('.', '_')}_three_strategy_equity.png", dpi=160)
    plt.close()


def run_symbol(symbol: str, sdt: str, edt: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all strategies on one symbol."""
    daily = read_jq_price(symbol, "daily", sdt, edt)
    intraday = read_jq_price(symbol, "5m", sdt, edt)
    available_days = intraday["trade_date"].nunique()

    rb = simulate_level_breakout(
        intraday,
        make_rbreaker_levels(daily),
        strategy="RBreaker_S0.3",
        level_type="rbreaker",
        rbreaker_scale=0.3,
    )
    dt = simulate_level_breakout(
        intraday,
        make_dual_thrust_levels(daily, n=5, k1=0.2, k2=0.2),
        strategy="DualThrust_N5K0.2",
        level_type="dual_thrust",
    )
    atr = simulate_signal_trades(make_atr_signals(intraday, timeperiod=5, th=30), strategy="ATR_BREAK_N5_T30")

    symbol_dir = OUT_DIR / symbol.replace(".", "_")
    symbol_dir.mkdir(parents=True, exist_ok=True)
    for name, df in {"RBreaker_S0.3": rb, "DualThrust_N5K0.2": dt, "ATR_BREAK_N5_T30": atr}.items():
        df.to_csv(symbol_dir / f"{name}_trades.csv", index=False, encoding="utf-8-sig")

    all_trades = pd.concat([rb, dt, atr], ignore_index=True)
    summary = summarize_trades(symbol, all_trades, available_days)
    returns, directions = align_daily(
        {"RBreaker_S0.3": rb, "DualThrust_N5K0.2": dt, "ATR_BREAK_N5_T30": atr},
        start=intraday["trade_date"].min(),
        end=intraday["trade_date"].max(),
    )
    pair_corr, direction_overlap, combo = compare_strategies(symbol, returns, directions)
    returns.to_csv(symbol_dir / "daily_returns.csv", encoding="utf-8-sig")
    directions.to_csv(symbol_dir / "daily_directions.csv", encoding="utf-8-sig")
    plot_symbol(symbol, returns, symbol_dir)
    return summary, pair_corr, direction_overlap, combo


def main():
    sdt, edt = "20200101", "20260517"
    symbols = ["000905.XSHG", "399006.XSHE"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries, corrs, directions, combos = [], [], [], []
    for symbol in symbols:
        summary, pair_corr, direction_overlap, combo = run_symbol(symbol, sdt=sdt, edt=edt)
        summaries.append(summary)
        corrs.append(pair_corr)
        directions.append(direction_overlap)
        combos.append(combo)

    summary_all = pd.concat(summaries, ignore_index=True)
    corr_all = pd.concat(corrs, ignore_index=True)
    direction_all = pd.concat(directions, ignore_index=True)
    combo_all = pd.concat(combos, ignore_index=True)

    summary_all.to_csv(OUT_DIR / "multi_index_strategy_summary.csv", index=False, encoding="utf-8-sig")
    corr_all.to_csv(OUT_DIR / "multi_index_pairwise_correlation.csv", index=False, encoding="utf-8-sig")
    direction_all.to_csv(OUT_DIR / "multi_index_direction_overlap.csv", index=False, encoding="utf-8-sig")
    combo_all.to_csv(OUT_DIR / "multi_index_combo_summary.csv", index=False, encoding="utf-8-sig")

    print("strategy summary")
    print(summary_all.to_string(index=False))
    print("\npairwise correlation")
    keep = corr_all[corr_all["sample"].isin(["all_calendar", "common_active", "monthly"])]
    print(keep.to_string(index=False))
    print("\ndirection overlap")
    print(direction_all.to_string(index=False))
    print("\ncombo summary")
    print(combo_all.to_string(index=False))
    print(f"\noutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
