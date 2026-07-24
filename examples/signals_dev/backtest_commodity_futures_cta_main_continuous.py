"""Backtest a daily CTA strategy on Chinese commodity futures main continuous contracts.

The script uses TQSDK ``KQ.m@`` main continuous contracts directly. The default
signal is a fixed optimized commodity CTA candidate:

- combine seven single-commodity daily CTA legs selected from trend and reversal families;
- size each leg by its own 20-day volatility;
- apply a portfolio-level 20-day volatility overlay and gross exposure cap;
- previous close target weight is applied to the next close-to-close return.

Use ``--signal-style reversal`` to replay the cross-sectional reversal baseline.
Use ``--signal-style trend`` to replay a conventional trend-following baseline.

Run:
    uv run --no-sync python examples/signals_dev/backtest_commodity_futures_cta_main_continuous.py

If cached CSV data already exists under the output directory, the script reuses it.
Use ``--refresh-data`` to refetch TQSDK daily bars.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE_OUTPUT_DIR = ROOT / "examples" / "results" / "commodity_futures_cta_main_continuous"
DATA_DIR = BASE_OUTPUT_DIR / "daily_cache"

DEFAULT_START = "2020-01-01"
DEFAULT_END = "2026-05-29"
FEE_RATE = 0.00025
ADJ_TYPE = "F"
SIGNAL_STYLE = "optimized"
TARGET_VOL = 0.12
VOL_WINDOW = 20
GROSS_CAP = 1.00
SYMBOL_CAP = 0.12
SECTOR_CAP = 0.35
REBALANCE_BAND = 0.015
MAX_LONGS = 8
MAX_SHORTS = 8
MIN_SYMBOLS = 8
SHORT_SCALE = 1.0
OPTIMIZED_LEG_TARGET_VOL = 0.20
OPTIMIZED_PORTFOLIO_TARGET_VOL = 0.15
OPTIMIZED_PORTFOLIO_LEVERAGE_CAP = 1.50
OPTIMIZED_GROSS_CAP = 2.00

DEFAULT_SYMBOLS = (
    "KQ.m@SHFE.rb",
    "KQ.m@SHFE.fu",
    "KQ.m@SHFE.ag",
    "KQ.m@SHFE.hc",
    "KQ.m@SHFE.sp",
    "KQ.m@SHFE.ru",
    "KQ.m@SHFE.bu",
    "KQ.m@SHFE.ni",
    "KQ.m@SHFE.ss",
    "KQ.m@SHFE.au",
    "KQ.m@SHFE.sn",
    "KQ.m@SHFE.al",
    "KQ.m@SHFE.ao",
    "KQ.m@SHFE.zn",
    "KQ.m@SHFE.cu",
    "KQ.m@SHFE.pb",
    "KQ.m@SHFE.br",
    "KQ.m@CZCE.SA",
    "KQ.m@CZCE.FG",
    "KQ.m@CZCE.TA",
    "KQ.m@CZCE.MA",
    "KQ.m@CZCE.RM",
    "KQ.m@CZCE.CF",
    "KQ.m@CZCE.OI",
    "KQ.m@CZCE.SR",
    "KQ.m@CZCE.UR",
    "KQ.m@CZCE.PF",
    "KQ.m@CZCE.AP",
    "KQ.m@CZCE.SF",
    "KQ.m@CZCE.PX",
    "KQ.m@CZCE.CJ",
    "KQ.m@CZCE.PK",
    "KQ.m@CZCE.SM",
    "KQ.m@CZCE.CY",
    "KQ.m@DCE.m",
    "KQ.m@DCE.p",
    "KQ.m@DCE.i",
    "KQ.m@DCE.v",
    "KQ.m@DCE.y",
    "KQ.m@DCE.eg",
    "KQ.m@DCE.c",
    "KQ.m@DCE.pp",
    "KQ.m@DCE.l",
    "KQ.m@DCE.cs",
    "KQ.m@DCE.a",
    "KQ.m@DCE.eb",
    "KQ.m@DCE.jm",
    "KQ.m@DCE.b",
    "KQ.m@DCE.pg",
    "KQ.m@DCE.jd",
    "KQ.m@DCE.j",
    "KQ.m@DCE.lh",
    "KQ.m@GFEX.si",
    "KQ.m@GFEX.lc",
    "KQ.m@INE.lu",
    "KQ.m@INE.sc",
    "KQ.m@INE.nr",
    "KQ.m@INE.bc",
    "KQ.m@INE.ec",
)

SECTOR_BY_PRODUCT = {
    "au": "precious",
    "ag": "precious",
    "cu": "base_metal",
    "al": "base_metal",
    "zn": "base_metal",
    "pb": "base_metal",
    "ni": "base_metal",
    "sn": "base_metal",
    "ao": "base_metal",
    "bc": "base_metal",
    "rb": "black",
    "hc": "black",
    "i": "black",
    "j": "black",
    "jm": "black",
    "sf": "black",
    "sm": "black",
    "ss": "black",
    "fg": "black",
    "sc": "energy",
    "lu": "energy",
    "fu": "energy",
    "bu": "energy",
    "pg": "energy",
    "sa": "chemical",
    "ta": "chemical",
    "ma": "chemical",
    "eg": "chemical",
    "eb": "chemical",
    "v": "chemical",
    "pp": "chemical",
    "l": "chemical",
    "ru": "chemical",
    "br": "chemical",
    "nr": "chemical",
    "sp": "chemical",
    "pf": "chemical",
    "px": "chemical",
    "si": "chemical",
    "lc": "chemical",
    "m": "agri",
    "a": "agri",
    "b": "agri",
    "p": "agri",
    "y": "agri",
    "oi": "agri",
    "rm": "agri",
    "c": "agri",
    "cs": "agri",
    "cf": "agri",
    "sr": "agri",
    "jd": "agri",
    "lh": "agri",
    "ap": "agri",
    "cj": "agri",
    "pk": "agri",
    "cy": "agri",
    "ur": "agri",
}


@dataclass(frozen=True)
class OptimizedLeg:
    """One fixed single-commodity CTA leg."""

    symbol: str
    family: str
    window: int
    leverage_cap: float


OPTIMIZED_LEGS = (
    OptimizedLeg("KQ.m@DCE.jd", "ma_rev", 90, 1.5),
    OptimizedLeg("KQ.m@SHFE.pb", "ma_rev", 180, 1.5),
    OptimizedLeg("KQ.m@SHFE.au", "trend", 60, 3.0),
    OptimizedLeg("KQ.m@SHFE.br", "trend", 10, 2.0),
    OptimizedLeg("KQ.m@SHFE.sn", "trend", 120, 3.0),
    OptimizedLeg("KQ.m@DCE.pg", "rev", 90, 2.0),
    OptimizedLeg("KQ.m@DCE.lh", "trend", 20, 3.0),
)


@dataclass(frozen=True)
class BacktestConfig:
    """Fixed commodity CTA configuration."""

    start: str = DEFAULT_START
    end: str = DEFAULT_END
    signal_style: str = SIGNAL_STYLE
    fee_rate: float = FEE_RATE
    adj_type: str | None = ADJ_TYPE
    target_vol: float = TARGET_VOL
    vol_window: int = VOL_WINDOW
    gross_cap: float = GROSS_CAP
    symbol_cap: float = SYMBOL_CAP
    sector_cap: float = SECTOR_CAP
    rebalance_band: float = REBALANCE_BAND
    max_longs: int = MAX_LONGS
    max_shorts: int = MAX_SHORTS
    short_scale: float = SHORT_SCALE
    optimized_leg_target_vol: float = OPTIMIZED_LEG_TARGET_VOL
    optimized_portfolio_target_vol: float = OPTIMIZED_PORTFOLIO_TARGET_VOL
    optimized_portfolio_leverage_cap: float = OPTIMIZED_PORTFOLIO_LEVERAGE_CAP
    optimized_gross_cap: float = OPTIMIZED_GROSS_CAP


def load_env_file() -> None:
    """Load local .env keys without printing secrets."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def safe_symbol(symbol: str) -> str:
    """Convert a TQ symbol to a filesystem-safe stem."""
    return symbol.replace("@", "_").replace(".", "_")


def product_code(symbol: str) -> str:
    """Extract product code from a TQ main continuous symbol."""
    return symbol.rsplit(".", maxsplit=1)[-1].lower()


def symbol_sector(symbol: str) -> str:
    """Map symbol to a coarse commodity sector."""
    return SECTOR_BY_PRODUCT.get(product_code(symbol), "other")


def cache_file(symbol: str, start: str, end: str, adj_type: str | None = ADJ_TYPE) -> Path:
    """Return the CSV cache path for one symbol."""
    adj = adj_type or "N"
    return DATA_DIR / f"{safe_symbol(symbol)}_daily_adj{adj}_{start.replace('-', '')}_{end.replace('-', '')}.csv"


def strategy_output_dir(config: BacktestConfig) -> Path:
    """Return the output directory for one signal style."""
    return BASE_OUTPUT_DIR / config.signal_style


def fetch_tq_daily(symbol: str, start: str, end: str, tq_user: str, tq_pass: str, adj_type: str | None) -> pd.DataFrame:
    """Fetch one symbol's daily bars from TQSDK."""
    try:
        from tqsdk import TqApi, TqAuth
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少 tqsdk 依赖；请先执行 `uv sync --extra tq`，或用 `uv run --extra tq ...` 运行") from exc

    api = TqApi(auth=TqAuth(user_name=tq_user, password=tq_pass), web_gui=False)
    try:
        kline = api.get_kline_serial(symbol, duration_seconds=86400, data_length=8000, adj_type=adj_type)
        data = kline.copy()
    finally:
        api.close()

    data = data.dropna(subset=["close"]).copy()
    if "volume" in data.columns:
        data = data[data["volume"].fillna(0) > 0].copy()
    data["dt"] = pd.to_datetime(data["datetime"], unit="ns", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    data["dt"] = data["dt"] + timedelta(days=1)
    data["dt"] = data["dt"].dt.normalize()
    data["symbol"] = symbol
    data = data.rename(columns={"volume": "vol"})
    data["amount"] = data.get("amount", data["vol"] * data["close"])
    data = data[["dt", "symbol", "open", "high", "low", "close", "vol", "amount"]]
    mask = (data["dt"] >= pd.Timestamp(start)) & (data["dt"] <= pd.Timestamp(end))
    return data.loc[mask].drop_duplicates("dt").sort_values("dt").reset_index(drop=True)


def load_or_fetch_daily(symbols: list[str], config: BacktestConfig, refresh_data: bool) -> dict[str, pd.DataFrame]:
    """Load cached daily bars, fetching missing files from TQSDK."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    load_env_file()
    tq_user = os.getenv("TQ_USER")
    tq_pass = os.getenv("TQ_PASS")
    data: dict[str, pd.DataFrame] = {}
    rows = []

    for symbol in symbols:
        file = cache_file(symbol, config.start, config.end, config.adj_type)
        if refresh_data or not file.exists():
            if not tq_user or not tq_pass:
                raise RuntimeError("缺少 TQ_USER / TQ_PASS，且本地没有完整缓存；请先配置天勤账号或放入缓存 CSV")
            df = fetch_tq_daily(symbol, config.start, config.end, tq_user, tq_pass, config.adj_type)
            df.to_csv(file, index=False, encoding="utf-8-sig")
        else:
            df = pd.read_csv(file, parse_dates=["dt"])
            df["symbol"] = symbol

        valid = df.dropna(subset=["close"])
        if len(valid) >= 260:
            data[symbol] = valid.reset_index(drop=True)
        rows.append(
            {
                "symbol": symbol,
                "sector": symbol_sector(symbol),
                "file": str(file),
                "start": str(valid["dt"].iloc[0].date()) if not valid.empty else None,
                "end": str(valid["dt"].iloc[-1].date()) if not valid.empty else None,
                "rows": len(df),
                "valid_rows": len(valid),
                "used": symbol in data,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(BASE_OUTPUT_DIR / "data_summary.csv", index=False, encoding="utf-8-sig")
    if len(data) < MIN_SYMBOLS:
        raise RuntimeError(f"可用合约数量不足：{len(data)} < {MIN_SYMBOLS}")
    return data


def make_close_matrix(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a date-symbol close matrix."""
    series = []
    for symbol, df in data.items():
        item = df[["dt", "close"]].copy()
        item = item.rename(columns={"close": symbol}).set_index("dt")
        series.append(item)
    close = pd.concat(series, axis=1).sort_index()
    close = close.loc[~close.index.duplicated(keep="last")]
    return close


def trend_scores(close: pd.DataFrame) -> pd.DataFrame:
    """Build fixed trend scores in [-1, 1]."""
    mom20 = close / close.shift(20) - 1
    mom60 = close / close.shift(60) - 1
    mom120 = close / close.shift(120) - 1
    ma60 = close / close.rolling(60, min_periods=60).mean() - 1
    ma120 = close / close.rolling(120, min_periods=120).mean() - 1
    score = (
        0.20 * np.sign(mom20)
        + 0.25 * np.sign(mom60)
        + 0.25 * np.sign(mom120)
        + 0.15 * np.sign(ma60)
        + 0.15 * np.sign(ma120)
    )
    return score.fillna(0.0)


def directional_signal(price: pd.Series, family: str, window: int) -> pd.Series:
    """Build one single-symbol directional signal."""
    if family == "trend":
        return np.sign(price / price.shift(window) - 1).fillna(0.0)
    if family == "rev":
        return -np.sign(price / price.shift(window) - 1).fillna(0.0)
    if family == "ma_trend":
        score = price / price.rolling(window, min_periods=window).mean() - 1
        return np.sign(score).replace(0.0, np.nan).ffill().fillna(0.0)
    if family == "ma_rev":
        score = price / price.rolling(window, min_periods=window).mean() - 1
        return -np.sign(score).replace(0.0, np.nan).ffill().fillna(0.0)
    raise ValueError(f"Unsupported optimized leg family: {family}")


def build_optimized_weights(close: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Build the optimized multi-commodity CTA candidate weights."""
    missing = [leg.symbol for leg in OPTIMIZED_LEGS if leg.symbol not in close.columns]
    if missing:
        raise ValueError(f"Optimized legs missing symbols: {missing}")

    returns = close.pct_change(fill_method=None).fillna(0.0)
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for leg in OPTIMIZED_LEGS:
        vol = returns[leg.symbol].rolling(config.vol_window, min_periods=config.vol_window).std() * np.sqrt(252)
        leverage = (config.optimized_leg_target_vol / vol).clip(upper=leg.leverage_cap).fillna(0.0)
        signal = directional_signal(close[leg.symbol], leg.family, leg.window)
        weights[leg.symbol] += signal * leverage / len(OPTIMIZED_LEGS)

    raw_returns = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    portfolio_vol = raw_returns.rolling(config.vol_window, min_periods=config.vol_window).std() * np.sqrt(252)
    overlay = (config.optimized_portfolio_target_vol / portfolio_vol).clip(
        upper=config.optimized_portfolio_leverage_cap
    )
    weights = weights.mul(overlay.shift(1).fillna(0.0), axis=0)

    gross = weights.abs().sum(axis=1)
    weights = weights.mul((config.optimized_gross_cap / gross).clip(upper=1.0).fillna(1.0), axis=0)
    return weights.where(close.notna(), 0.0)


def build_trend_raw_signal(close: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Build a conventional trend-following raw signal before volatility sizing."""
    score = trend_scores(close)
    rank = score.rank(axis=1, pct=True, na_option="keep") * 2 - 1
    strength = score.abs() * (1 + 0.3 * rank.abs().fillna(0.0))
    raw = np.sign(score) * strength
    raw = raw.where(score.abs() >= 0.25, 0.0)
    raw = raw.where(close.notna(), 0.0)

    selected_rows = []
    for _, row in raw.iterrows():
        selected = pd.Series(0.0, index=raw.columns)
        longs = row[row > 0].sort_values(ascending=False).head(config.max_longs)
        shorts = row[row < 0].sort_values(ascending=True).head(config.max_shorts)
        selected.loc[longs.index] = longs
        selected.loc[shorts.index] = shorts
        selected_rows.append(selected)
    return pd.DataFrame(selected_rows, index=raw.index, columns=raw.columns)


def build_reversal_raw_signal(close: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Build the default commodity cross-sectional reversal raw signal."""
    momentum = ((close / close.shift(120) - 1) + (close / close.shift(240) - 1)) / 2
    reversal = -momentum
    broad = close.mean(axis=1)
    broad_weak = (broad / broad.rolling(120, min_periods=120).mean() - 1).fillna(0.0) < 0

    selected_rows = []
    for dt, row in reversal.fillna(0.0).iterrows():
        selected = pd.Series(0.0, index=close.columns)
        longs = row[row > 0].sort_values(ascending=False).head(config.max_longs)
        selected.loc[longs.index] = longs

        if broad_weak.loc[dt] and config.short_scale > 0:
            trend_row = momentum.loc[dt].fillna(0.0)
            shorts = trend_row[trend_row > 0].sort_values(ascending=False).head(config.max_shorts)
            selected.loc[shorts.index] = -config.short_scale * shorts

        selected_rows.append(selected.where(close.loc[dt].notna(), 0.0))
    return pd.DataFrame(selected_rows, index=close.index, columns=close.columns)


def build_raw_signal(close: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Build raw long/short signals before volatility sizing."""
    if config.signal_style == "trend":
        return build_trend_raw_signal(close, config)
    if config.signal_style == "reversal":
        return build_reversal_raw_signal(close, config)
    raise ValueError(f"Unsupported signal_style: {config.signal_style}")


def cap_by_symbol(weights: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Apply a per-symbol absolute weight cap."""
    return weights.clip(lower=-cap, upper=cap)


def cap_by_sector(weights: pd.DataFrame, sector_cap: float) -> pd.DataFrame:
    """Apply a gross exposure cap to each commodity sector."""
    out = weights.copy()
    sectors = {sector: [s for s in weights.columns if symbol_sector(s) == sector] for sector in set(map(symbol_sector, weights))}
    for symbols in sectors.values():
        if not symbols:
            continue
        gross = out[symbols].abs().sum(axis=1)
        scale = (sector_cap / gross).clip(upper=1.0).replace([np.inf, -np.inf], 1.0).fillna(1.0)
        out.loc[:, symbols] = out[symbols].mul(scale, axis=0)
    return out


def normalize_weights(raw: pd.DataFrame, close: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Convert raw signals to capped volatility-targeted weights."""
    returns = close.pct_change(fill_method=None)
    vol = returns.rolling(config.vol_window, min_periods=config.vol_window).std() * np.sqrt(252)
    inv_vol = 1 / vol.clip(lower=0.05, upper=0.80)
    base = (raw * inv_vol).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    gross = base.abs().sum(axis=1)
    unit = base.div(gross.replace(0, np.nan), axis=0).fillna(0.0)
    estimated_vol = np.sqrt(((unit * vol.fillna(0.0)) ** 2).sum(axis=1))
    scale = (config.target_vol / estimated_vol.replace(0, np.nan)).clip(upper=config.gross_cap).fillna(0.0)
    desired = unit.mul(scale, axis=0)
    desired = cap_by_symbol(desired, config.symbol_cap)
    desired = cap_by_sector(desired, config.sector_cap)
    gross = desired.abs().sum(axis=1)
    desired = desired.mul((config.gross_cap / gross).clip(upper=1.0).fillna(1.0), axis=0)
    desired = desired.where(close.notna(), 0.0)
    return desired


def apply_rebalance_band(desired: pd.DataFrame, band: float) -> pd.DataFrame:
    """Keep prior weights when changes are smaller than the rebalance band."""
    rows = []
    previous = pd.Series(0.0, index=desired.columns)
    for _, row in desired.iterrows():
        current = previous.copy()
        changed = (row - previous).abs() >= band
        current.loc[changed] = row.loc[changed]
        current.loc[row.abs() == 0] = 0.0
        rows.append(current)
        previous = current
    return pd.DataFrame(rows, index=desired.index, columns=desired.columns)


def build_weights(close: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Build final target weights."""
    if config.signal_style == "optimized":
        return build_optimized_weights(close, config)
    raw = build_raw_signal(close, config)
    desired = normalize_weights(raw, close, config)
    return apply_rebalance_band(desired, config.rebalance_band)


def annualized_stats(daily: pd.DataFrame, config: BacktestConfig) -> dict:
    """Calculate aggregate performance statistics."""
    years = (daily["dt"].iloc[-1] - daily["dt"].iloc[0]).days / 365.25
    final_nav = float(daily["nav"].iloc[-1])
    annual_return = final_nav ** (1 / years) - 1 if years > 0 and final_nav > 0 else float("nan")
    max_drawdown = float(daily["drawdown"].min())
    daily_std = float(daily["ret"].std())
    sharpe = float(daily["ret"].mean() / daily_std * np.sqrt(252)) if daily_std > 0 else float("nan")
    return {
        "start": str(daily["dt"].iloc[0].date()),
        "end": str(daily["dt"].iloc[-1].date()),
        "signal_style": config.signal_style,
        "days": int(len(daily)),
        "symbols": int(daily["active_symbols"].max()),
        "annual_return": annual_return,
        "cumulative_return": final_nav - 1,
        "final_nav": final_nav,
        "max_drawdown": max_drawdown,
        "calmar": annual_return / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
        "sharpe": sharpe,
        "daily_win_rate": float((daily["ret"] > 0).mean()),
        "turnover": float(daily["turnover"].sum()),
        "avg_daily_turnover": float(daily["turnover"].mean()),
        "avg_gross": float(daily["gross"].mean()),
        "max_gross": float(daily["gross"].max()),
        "avg_net": float(daily["net"].mean()),
        "fee_rate": config.fee_rate,
        "adj_type": config.adj_type or "N",
        "target_vol": config.target_vol,
        "vol_window": config.vol_window,
        "gross_cap": config.gross_cap,
        "symbol_cap": config.symbol_cap,
        "sector_cap": config.sector_cap,
        "rebalance_band": config.rebalance_band,
        "short_scale": config.short_scale,
        "optimized_leg_target_vol": config.optimized_leg_target_vol,
        "optimized_portfolio_target_vol": config.optimized_portfolio_target_vol,
        "optimized_portfolio_leverage_cap": config.optimized_portfolio_leverage_cap,
        "optimized_gross_cap": config.optimized_gross_cap,
    }


def evaluate(close: pd.DataFrame, weights: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, dict]:
    """Evaluate close-to-close returns with previous-day weights."""
    returns = close.pct_change(fill_method=None).fillna(0.0)
    previous_weights = weights.shift(1).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    ret = (previous_weights * returns).sum(axis=1) - turnover * config.fee_rate

    daily = pd.DataFrame(
        {
            "dt": close.index,
            "ret": ret.to_numpy(),
            "turnover": turnover.to_numpy(),
            "gross": weights.abs().sum(axis=1).to_numpy(),
            "net": weights.sum(axis=1).to_numpy(),
            "long_gross": weights.clip(lower=0).sum(axis=1).to_numpy(),
            "short_gross": weights.clip(upper=0).abs().sum(axis=1).to_numpy(),
            "active_symbols": (weights.abs() > 0).sum(axis=1).to_numpy(),
        }
    )
    daily["nav"] = (1 + daily["ret"]).cumprod()
    daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1
    return daily, annualized_stats(daily, config)


def yearly_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate calendar-year performance."""
    rows = []
    for year, group in daily.groupby(daily["dt"].dt.year):
        data = group.copy()
        nav = (1 + data["ret"]).cumprod()
        drawdown = nav / nav.cummax() - 1
        years = (data["dt"].iloc[-1] - data["dt"].iloc[0]).days / 365.25
        final_nav = float(nav.iloc[-1])
        annual_return = final_nav ** (1 / years) - 1 if years > 0 and final_nav > 0 else final_nav - 1
        max_drawdown = float(drawdown.min())
        rows.append(
            {
                "year": int(year),
                "start": str(data["dt"].iloc[0].date()),
                "end": str(data["dt"].iloc[-1].date()),
                "days": int(len(data)),
                "return": final_nav - 1,
                "annual_return": annual_return,
                "max_drawdown": max_drawdown,
                "calmar": annual_return / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
                "win_rate": float((data["ret"] > 0).mean()),
                "turnover": float(data["turnover"].sum()),
                "avg_gross": float(data["gross"].mean()),
                "max_gross": float(data["gross"].max()),
            }
        )
    return pd.DataFrame(rows)


def write_equity_svg(daily: pd.DataFrame, stats: dict, output_dir: Path) -> None:
    """Write a self-contained SVG/HTML chart of NAV, drawdown and gross exposure."""
    width, height = 1180, 760
    left, right, top, bottom = 76, 30, 76, 52
    plot_w = width - left - right
    nav_h = 390
    dd_top = top + nav_h + 36
    dd_h = 140
    gross_top = dd_top + dd_h + 34
    gross_h = height - gross_top - bottom
    x_values = daily["dt"].map(pd.Timestamp.toordinal).astype(float)
    x_min, x_max = float(x_values.min()), float(x_values.max())
    nav_min = min(0.95, float(daily["nav"].min())) * 0.99
    nav_max = float(daily["nav"].max()) * 1.05
    dd_min = min(float(daily["drawdown"].min()) * 1.1, -0.02)
    gross_max = max(float(daily["gross"].max()) * 1.1, 0.5)

    def sx(dt) -> float:
        return left + (float(pd.Timestamp(dt).toordinal()) - x_min) / (x_max - x_min) * plot_w

    def sy_nav(value: float) -> float:
        return top + (nav_max - value) / (nav_max - nav_min) * nav_h

    def sy_dd(value: float) -> float:
        return dd_top + (0 - value) / (0 - dd_min) * dd_h

    def sy_gross(value: float) -> float:
        return gross_top + (gross_max - value) / gross_max * gross_h

    nav_path = "M " + " L ".join(f"{sx(row.dt):.2f} {sy_nav(row.nav):.2f}" for row in daily.itertuples())
    dd_path = "M " + " L ".join(f"{sx(row.dt):.2f} {sy_dd(row.drawdown):.2f}" for row in daily.itertuples())
    gross_path = "M " + " L ".join(f"{sx(row.dt):.2f} {sy_gross(row.gross):.2f}" for row in daily.itertuples())
    subtitle = (
        f"annual {stats['annual_return']:.2%}, max drawdown {stats['max_drawdown']:.2%}, "
        f"final NAV {stats['final_nav']:.4f}, symbols {stats['symbols']}"
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fff"/>
<style>
text{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;fill:#172033}}
.title{{font-size:24px;font-weight:700}} .sub{{font-size:14px;fill:#526173}}
.grid{{stroke:#e7ecf3;stroke-width:1}} .axis{{stroke:#cfd8e3;stroke-width:1}}
</style>
<text class="title" x="{left}" y="32">Commodity Futures Main Continuous CTA</text>
<text class="sub" x="{left}" y="56">{subtitle}</text>
<line class="grid" x1="{left}" x2="{width - right}" y1="{sy_nav(1):.2f}" y2="{sy_nav(1):.2f}"/>
<line class="axis" x1="{left}" x2="{width - right}" y1="{top + nav_h}" y2="{top + nav_h}"/>
<line class="axis" x1="{left}" x2="{width - right}" y1="{dd_top}" y2="{dd_top}"/>
<line class="axis" x1="{left}" x2="{width - right}" y1="{gross_top + gross_h}" y2="{gross_top + gross_h}"/>
<text class="sub" x="{left}" y="{top - 14}">NAV</text>
<text class="sub" x="{left}" y="{dd_top - 12}">Drawdown</text>
<text class="sub" x="{left}" y="{gross_top - 12}">Gross Exposure</text>
<path d="{nav_path}" fill="none" stroke="#0f766e" stroke-width="2.6"/>
<path d="{dd_path}" fill="none" stroke="#b45309" stroke-width="2.2"/>
<path d="{gross_path}" fill="none" stroke="#2563eb" stroke-width="2.0"/>
</svg>"""
    html = f"<!doctype html><meta charset='utf-8'><title>Commodity Futures CTA</title><body style='margin:0'>{svg}</body>"
    (output_dir / "equity.svg").write_text(svg, encoding="utf-8")
    (output_dir / "equity.html").write_text(html, encoding="utf-8")


def save_outputs(close: pd.DataFrame, weights: pd.DataFrame, daily: pd.DataFrame, stats: dict, config: BacktestConfig) -> Path:
    """Save backtest artifacts."""
    output_dir = strategy_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    close.to_csv(output_dir / "close_matrix.csv", encoding="utf-8-sig")
    weights.to_csv(output_dir / "weights.csv", encoding="utf-8-sig")
    daily.to_csv(output_dir / "daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([stats]).to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    yearly_stats(daily).to_csv(output_dir / "yearly_stats.csv", index=False, encoding="utf-8-sig")

    latest = weights.iloc[-1].rename("weight").reset_index()
    latest.columns = ["symbol", "weight"]
    latest = latest[latest["weight"].abs() > 0].copy()
    latest["sector"] = latest["symbol"].map(symbol_sector)
    latest = latest.sort_values("weight", ascending=False)
    latest.to_csv(output_dir / "latest_weights.csv", index=False, encoding="utf-8-sig")
    write_equity_svg(daily, stats, output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Commodity futures main continuous CTA backtest")
    parser.add_argument("--start", default=DEFAULT_START, help="Backtest start date, e.g. 2020-01-01")
    parser.add_argument("--end", default=DEFAULT_END, help="Backtest end date, e.g. 2026-05-29")
    parser.add_argument("--refresh-data", action="store_true", help="Refetch TQSDK daily bars even when cache exists")
    parser.add_argument("--symbols", default="", help="Comma-separated TQ symbols. Empty means the default commodity pool")
    parser.add_argument(
        "--signal-style", choices=["optimized", "reversal", "trend"], default=SIGNAL_STYLE, help="Signal style"
    )
    return parser.parse_args()


def main() -> None:
    """Run the commodity futures CTA backtest."""
    args = parse_args()
    symbols = [x.strip() for x in args.symbols.split(",") if x.strip()] if args.symbols else list(DEFAULT_SYMBOLS)
    config = BacktestConfig(start=args.start, end=args.end, signal_style=args.signal_style)
    data = load_or_fetch_daily(symbols, config, refresh_data=args.refresh_data)
    close = make_close_matrix(data)
    close = close[(close.index >= pd.Timestamp(config.start)) & (close.index <= pd.Timestamp(config.end))]
    weights = build_weights(close, config)
    daily, stats = evaluate(close, weights, config)
    output_dir = save_outputs(close, weights, daily, stats, config)

    print(pd.DataFrame([stats]).to_string(index=False))
    print(f"\noutputs: {output_dir}")


if __name__ == "__main__":
    main()
