"""Daily breakout decisions executed at the next actual open with the shared Rust ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cta_execution import futures_daily_frame, next_open_panel_ledger, trading_day_returns  # noqa: E402

DATA_DIR = ROOT / "examples/results/continuous_signal_commodity_cta/data_cache"
OUTPUT_DIR = ROOT / "examples/results/cta_research/momentum"


def add_daily_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the original ATR, slope and channel parameters without future backfilling."""
    result = frame.copy()
    previous = result.close.shift(1)
    true_range = pd.concat(
        [result.high - result.low, (result.high - previous).abs(), (result.low - previous).abs()], axis=1
    ).max(axis=1)
    result["atr"] = true_range.rolling(20, min_periods=5).mean()
    closes = result.close.to_numpy(dtype=float)
    slopes = np.zeros(len(closes))
    for i in range(19, len(closes)):
        slopes[i] = np.polyfit(np.arange(20), closes[i - 19 : i + 1], 1)[0] / closes[i] * 100
    result["slope"] = slopes
    for window, minimum in [(20, 10), (10, 5)]:
        result[f"hh{window}"] = result.high.shift(1).rolling(window, min_periods=minimum).max()
        result[f"ll{window}"] = result.low.shift(1).rolling(window, min_periods=minimum).min()
    return result


def load_all_daily_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Read start-labelled 15m bars and aggregate complete futures trading days."""
    result = {}
    for path in sorted(Path(data_dir).glob("*.parquet")):
        frame = pd.read_parquet(path)
        if len(frame) < 300:
            continue
        frame["dt"] = pd.to_datetime(frame.dt) + pd.Timedelta(minutes=15)
        frame["symbol"] = path.stem.replace("_15m", "")
        frame["vol"] = frame["volume"]
        frame["amount"] = frame.vol * frame.close * 10
        daily = futures_daily_frame(frame)
        if len(daily) < 100:
            continue
        result[frame.symbol.iloc[0]] = add_daily_indicators(daily).set_index("dt")
    if not result:
        raise ValueError(f"No usable daily histories in {data_dir}")
    return result


def build_advanced_targets(symbol_data, top_k=8, leverage=3.0, trailing_mult=2.5, stop_atr=2.0):
    """Keep the original breakout/ranking rules; update filled risk state only at opens.

    Close exit and entry intentions are netted before the next open. An unchanged
    same-direction target never creates a fictitious closed/reopened trade.
    """
    all_dates = sorted({date for frame in symbol_data.values() for date in frame.index})
    all_dates = [date for date in all_dates if sum(date in frame.index for frame in symbol_data.values()) >= 20]
    targets = pd.DataFrame(0.0, index=pd.DatetimeIndex(all_dates), columns=sorted(symbol_data))
    positions, pending_direction, pending_atr = {}, {}, {}
    for date in targets.index:
        observed = {symbol: frame.loc[date] for symbol, frame in symbol_data.items() if date in frame.index}
        for symbol, row in observed.items():
            desired = pending_direction.get(symbol, 0)
            current = positions.get(symbol, {}).get("direction", 0)
            if desired != current:
                positions.pop(symbol, None)
                if desired:
                    positions[symbol] = {
                        "direction": desired,
                        "entry": row.open,
                        "peak": row.open,
                        "valley": row.open,
                        "atr": pending_atr[symbol],
                    }
        intentions = {symbol: state["direction"] for symbol, state in positions.items()}
        for symbol, state in positions.items():
            if symbol not in observed:
                continue
            row = observed[symbol]
            state["peak"] = max(state["peak"], row.high)
            state["valley"] = min(state["valley"], row.low)
            entry, atr, direction = state["entry"], state["atr"], state["direction"]
            if direction > 0:
                leave = (
                    row.close < entry - stop_atr * atr
                    or (state["peak"] - entry > 1.5 * atr and row.close < state["peak"] - trailing_mult * atr)
                    or row.close < row.ll10
                )
            else:
                leave = (
                    row.close > entry + stop_atr * atr
                    or (entry - state["valley"] > 1.5 * atr and row.close > state["valley"] + trailing_mult * atr)
                    or row.close > row.hh10
                )
            if leave:
                intentions[symbol] = 0
        candidates = []
        for symbol, row in observed.items():
            if (
                intentions.get(symbol, 0)
                or not np.isfinite([row.hh20, row.ll20, row.slope, row.atr]).all()
                or row.atr <= 0
            ):
                continue
            direction = (
                1
                if row.slope > 0.05 and row.close > row.hh20
                else -1
                if row.slope < -0.05 and row.close < row.ll20
                else 0
            )
            if direction:
                candidates.append((symbol, direction, abs(row.slope), row.atr))
        candidates.sort(key=lambda value: (-value[2], value[0]))
        slots = max(0, top_k - sum(direction != 0 for direction in intentions.values()))
        for symbol, direction, _, atr in candidates[:slots]:
            intentions[symbol], pending_atr[symbol] = direction, atr
        for symbol in observed:
            pending_direction[symbol] = intentions.get(symbol, 0)
        for symbol in targets:
            targets.loc[date, symbol] = pending_direction.get(symbol, 0) * leverage / top_k
    return targets


def summarize_execution(symbol_data: dict, targets: pd.DataFrame, fee_rate: float) -> tuple[dict, dict]:
    """Report fixed-initial-capital returns, including rebalancing fees and liquidation."""
    combined = pd.concat(
        [frame.reset_index().assign(symbol=symbol) for symbol, frame in symbol_data.items()], ignore_index=True
    )
    ledger = next_open_panel_ledger(combined, targets, fee_rate)
    daily = {key: trading_day_returns(value.sum(axis=1)) for key, value in ledger.items()}
    net = daily["net"]
    nav = 1 + net.cumsum()
    peak = nav.clip(lower=1).cummax()
    dd = (nav / peak - 1) * 100
    total = float(net.sum() * 100)
    annual = total * 252 / max(len(net), 1)
    max_dd = float(dd.min()) if len(dd) else 0.0
    std = float(net.std(ddof=1))
    sharpe = float(net.mean() / std * np.sqrt(252)) if len(net) > 1 and std > 1e-12 else 0.0
    result = {
        "dates": [str(value) for value in net.index],
        "nav": nav.tolist(),
        "dd": dd.tolist(),
        "tot_ret": total,
        "ann_ret": annual,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "calmar": annual / abs(max_dd) if max_dd else 0.0,
        "gross_return_pct": float(daily["gross"].sum() * 100),
        "fee_return_pct": float(daily["fee"].sum() * 100),
        "turnover": float(daily["turnover"].sum()),
        "fee_rate": fee_rate,
        "daily_net": net.tolist(),
        "daily_gross": daily["gross"].tolist(),
        "daily_fees": daily["fee"].tolist(),
        "daily_turnover": daily["turnover"].tolist(),
        "trading_days": len(net),
        "execution_clock": "next_actual_open",
        "accounting": "fixed_initial_capital",
        "terminal_liquidation": True,
        "validation_scope": "retrospective_continuous_reference_price_replay",
    }
    return result, ledger


def run_cta_backtest(symbol_data, top_k=8, leverage=3.0, trailing_mult=2.5, stop_atr=2.0, fee_rate=0.0001):
    """Evaluate the unchanged three breakout configurations through executable prices."""
    targets = build_advanced_targets(symbol_data, top_k, leverage, trailing_mult, stop_atr)
    result, _ = summarize_execution(symbol_data, targets, fee_rate)
    result["parameters"] = {"top_k": top_k, "leverage": leverage, "trailing_mult": trailing_mult, "stop_atr": stop_atr}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    data = load_all_daily_data(args.data_dir)
    results = {}
    for name, top_k, leverage in [("conservative", 10, 2.0), ("standard", 8, 3.5), ("aggressive", 6, 5.0)]:
        results[name] = run_cta_backtest(data, top_k=top_k, leverage=leverage)
        value = results[name]
        print(
            f"{name}: return={value['tot_ret']:.3f}% Sharpe={value['sharpe']:.3f} maxDD={value['max_dd']:.3f}% fees={value['fee_return_pct']:.3f}%"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "advanced_cta_momentum_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
