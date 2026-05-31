"""Audit whether local data can support an expanded daily CTA symbol pool.

The current cross-sectional daily CTA pipeline requires raw daily OHLCV data
covering at least 2016-01-01 to 2026-05-29, because each OOS year uses the
previous four calendar years for training. This script checks local artifacts
and classifies them as directly usable, short-history only, or not raw price
data.

Run:
    .venv/bin/python examples/signals_dev/validate_daily_cta_extended_pool_feasibility.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from validate_daily_cta_walk_forward import DATA_FILES

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "examples" / "results" / "daily_cta_cross_section_walk_forward" / "extended_pool_feasibility"
REQUIRED_START = pd.Timestamp("2016-01-01")
REQUIRED_END = pd.Timestamp("2026-05-29")
REQUIRED_COLUMNS = {"dt", "open", "high", "low", "close", "vol"}

EXTRA_CANDIDATES = {
    "000300.XSHG": {
        "path": ROOT
        / "examples"
        / "results"
        / "daily_holding_extended_pool"
        / "000300_XSHG_daily_20100101_20260529.csv",
        "kind": "daily_ohlcv",
    },
    "000016.XSHG": {
        "path": ROOT
        / "examples"
        / "results"
        / "daily_holding_extended_pool"
        / "000016_XSHG_daily_20100101_20260529.csv",
        "kind": "daily_ohlcv",
    },
    "399006.XSHE": {
        "path": ROOT
        / "examples"
        / "results"
        / "daily_holding_extended_pool"
        / "399006_XSHE_daily_20100101_20260529.csv",
        "kind": "daily_ohlcv",
    },
    "000906.XSHG": {
        "path": ROOT
        / "examples"
        / "results"
        / "daily_holding_extended_pool"
        / "000906_XSHG_daily_20100101_20260529.csv",
        "kind": "daily_ohlcv",
    },
    "000985.XSHG": {
        "path": ROOT
        / "examples"
        / "results"
        / "daily_holding_extended_pool"
        / "000985_XSHG_daily_20100101_20260529.csv",
        "kind": "daily_ohlcv",
    },
    "000688.XSHG": {
        "path": ROOT
        / "examples"
        / "results"
        / "daily_holding_extended_pool"
        / "000688_XSHG_daily_20100101_20260529.csv",
        "kind": "daily_ohlcv",
    },
    "399006.XSHE_strategy_returns": {
        "path": ROOT
        / "examples"
        / "results"
        / "three_strategy_multi_index_jq_sdk"
        / "399006_XSHE"
        / "daily_returns.csv",
        "kind": "strategy_returns",
    },
    "588000.XSHG_5m_ohlc": {
        "path": ROOT
        / "examples"
        / "results"
        / "double_shun_60m_pe_verify_588000.XSHG"
        / "588000_XSHG_5m_ohlc_20220101_20260525.csv",
        "kind": "intraday_ohlc",
    },
}


def read_price_frame(path: Path, kind: str) -> pd.DataFrame:
    """Read a raw price artifact into a daily-like frame for coverage checks."""
    if kind == "daily_ohlcv":
        return pd.read_csv(path, parse_dates=["dt"])
    if kind == "intraday_ohlc":
        frame = pd.read_csv(path, parse_dates=["dt"])
        frame["date"] = frame["dt"].dt.normalize()
        volume_col = "vol" if "vol" in frame.columns else "volume" if "volume" in frame.columns else None
        aggregations = {
            "open": ("open", "first"),
            "high": ("high", "max"),
            "low": ("low", "min"),
            "close": ("close", "last"),
        }
        if volume_col is not None:
            aggregations["vol"] = (volume_col, "sum")
        daily = frame.groupby("date", as_index=False).agg(**aggregations).rename(columns={"date": "dt"})
        if "vol" not in daily.columns:
            daily["vol"] = 0.0
        return daily
    raise ValueError(f"Unsupported raw price kind: {kind}")


def audit_raw_price(symbol: str, path: Path, kind: str) -> dict:
    """Audit one raw price candidate."""
    row = {
        "symbol": symbol,
        "path": str(path),
        "kind": kind,
        "exists": path.exists(),
        "usable_for_current_wfo": False,
        "reason": "",
    }
    if not path.exists():
        row["reason"] = "file_missing"
        return row

    frame = read_price_frame(path, kind)
    columns = set(frame.columns)
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        row.update(
            {
                "start": None,
                "end": None,
                "rows": len(frame),
                "missing_columns": ",".join(missing),
                "reason": "missing_required_ohlcv_columns",
            }
        )
        return row

    frame = frame.dropna(subset=["dt"]).sort_values("dt").drop_duplicates("dt")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    start = pd.to_datetime(frame["dt"]).min()
    end = pd.to_datetime(frame["dt"]).max()
    enough_history = start <= REQUIRED_START and end >= REQUIRED_END
    row.update(
        {
            "start": str(start.date()),
            "end": str(end.date()),
            "rows": len(frame),
            "missing_columns": "",
            "usable_for_current_wfo": bool(enough_history),
            "reason": "ok" if enough_history else "history_too_short_for_4y_train_2020_2026_wfo",
        }
    )
    return row


def audit_non_price(symbol: str, path: Path, kind: str) -> dict:
    """Audit one artifact that is not raw OHLCV price data."""
    row = {
        "symbol": symbol,
        "path": str(path),
        "kind": kind,
        "exists": path.exists(),
        "usable_for_current_wfo": False,
        "reason": "not_raw_daily_ohlcv",
    }
    if path.exists():
        frame = pd.read_csv(path)
        row["rows"] = len(frame)
        date_col = "dt" if "dt" in frame.columns else "trade_date" if "trade_date" in frame.columns else None
        if date_col is not None:
            dates = pd.to_datetime(frame[date_col], errors="coerce").dropna()
            row["start"] = str(dates.min().date()) if not dates.empty else None
            row["end"] = str(dates.max().date()) if not dates.empty else None
        row["columns"] = ",".join(frame.columns)
    return row


def main() -> None:
    """Run extended pool feasibility audit."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for symbol, path in DATA_FILES.items():
        rows.append(audit_raw_price(symbol, path, "daily_ohlcv"))
    for symbol, info in EXTRA_CANDIDATES.items():
        if info["kind"] in {"daily_ohlcv", "intraday_ohlc"}:
            rows.append(audit_raw_price(symbol, info["path"], info["kind"]))
        else:
            rows.append(audit_non_price(symbol, info["path"], info["kind"]))

    audit = pd.DataFrame(rows)
    audit.to_csv(OUTPUT_DIR / "extended_pool_data_audit.csv", index=False, encoding="utf-8-sig")

    usable = audit[audit["usable_for_current_wfo"]].copy()
    aggregate = {
        "required_start": str(REQUIRED_START.date()),
        "required_end": str(REQUIRED_END.date()),
        "total_candidates": int(len(audit)),
        "usable_raw_daily_ohlcv": int(len(usable)),
        "usable_symbols": ",".join(usable["symbol"]),
        "can_expand_beyond_current_three": bool(len(usable) > len(DATA_FILES)),
    }
    pd.DataFrame([aggregate]).to_csv(OUTPUT_DIR / "extended_pool_data_audit_summary.csv", index=False)

    print(pd.DataFrame([aggregate]).to_string(index=False))
    print("\nData audit:")
    display_cols = ["symbol", "kind", "exists", "start", "end", "rows", "usable_for_current_wfo", "reason"]
    print(audit.reindex(columns=display_cols).to_string(index=False))
    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
