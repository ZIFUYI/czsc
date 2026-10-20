"""Freeze the audited local real-price inputs without modifying source caches.

This is dataset preparation, not a signal or execution implementation. Running
this module records source hashes and deterministic timestamp corrections in an
independent research directory. It never fetches market data or infers a tradable
contract, exchange calendar, borrowing permission, or microstructure group.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from examples.multi_factor_cta.test_all_weather_50commodities_portfolio import build_fifty_commodities_universe

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "examples" / "results"
RUN_DIR = RESULTS / "research_rerun_20261017"
DAILY_SOURCE = RESULTS / "commodity_futures_cta_main_continuous" / "daily_cache"
INTRADAY_SOURCE = RESULTS / "continuous_signal_commodity_cta" / "data_cache"
EQUITY_DAILY = {
    "000016": "daily_holding_extended_pool/000016_XSHG_daily_20100101_20260529.csv",
    "000300": "daily_holding_extended_pool/000300_XSHG_daily_20100101_20260529.csv",
    "000688": "daily_holding_extended_pool/000688_XSHG_daily_20100101_20260529.csv",
    "000906": "daily_holding_extended_pool/000906_XSHG_daily_20100101_20260529.csv",
    "000985": "daily_holding_extended_pool/000985_XSHG_daily_20100101_20260529.csv",
    "399006": "daily_holding_extended_pool/399006_XSHE_daily_20100101_20260529.csv",
    "000852": "daily_holding_000852/000852_XSHG_daily_20100101_20260904.csv",
    "000905": "daily_holding_cross_symbol/000905_XSHG_daily_20100101_20260904.csv",
    "159915": "daily_holding_cross_symbol/159915_XSHE_daily_20100101_20260904.csv",
    "588000": "daily_holding_core_plus_588000/588000_XSHG_daily_20100101_20260529.csv",
}
EQUITY_INTRADAY = {
    "000852": "conditional_resonance_000852/000852_XSHG_15m_ohlc.csv",
    "000905": "conditional_resonance_000905_XSHG/000905_XSHG_15m_ohlc.csv",
    "399006": "conditional_resonance_399006_XSHE/399006_XSHE_15m_ohlc.csv",
    "159915": "conditional_resonance_159915_XSHE/159915_XSHE_15m_ohlc.csv",
    "588000": "double_shun_60m_pe_verify_588000.XSHG/588000_XSHG_5m_ohlc_20220101_20260525.csv",
}
PRICE_COLS = ["open", "high", "low", "close"]
FRAME_COLS = ["dt", "symbol", *PRICE_COLS, "volume", "amount"]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": _hash(path), "bytes": path.stat().st_size}


def _read(path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    original_rows = len(frame)
    frame["dt"] = pd.to_datetime(frame["dt"])
    if "volume" not in frame:
        frame = frame.rename(columns={"vol": "volume"})
    valid = np.isfinite(frame[PRICE_COLS]).all(axis=1) & frame[PRICE_COLS].gt(0).all(axis=1)
    # Equity caches have explicit pre-listing NaNs; never interpolate them.
    if valid.any() and not valid.loc[valid.idxmax() :].all():
        raise ValueError(f"Interior invalid OHLC in {path}")
    frame = frame.loc[valid].copy().sort_values("dt").reset_index(drop=True)
    if frame.empty or frame.dt.duplicated().any() or frame.dt.isna().any():
        raise ValueError(f"Empty/duplicate/invalid clock in {path}")
    bad_envelope = (frame.high < frame[["open", "low", "close"]].max(axis=1)) | (
        frame.low > frame[["open", "high", "close"]].min(axis=1)
    )
    quarantined = frame.loc[bad_envelope, ["dt", *PRICE_COLS]].copy()
    quarantined["dt"] = quarantined.dt.astype(str)
    frame = frame.loc[~bad_envelope].reset_index(drop=True)
    metadata = _source(path) | {
        "source_rows": original_rows,
        "removed_prelisting_rows": int((~valid).sum()),
        "quarantined_invalid_ohlc": quarantined.to_dict("records"),
        "source_columns": list(frame.columns),
        "source_start": str(frame.dt.min()),
        "source_end": str(frame.dt.max()),
    }
    return frame, metadata


def _overlap(daily: pd.DataFrame, intraday: pd.DataFrame) -> dict:
    # Only completed local daytime closes are comparable; no partial tail bars.
    closes = intraday.loc[intraday.dt.dt.strftime("%H:%M") == "15:00", ["dt", "close"]].copy()
    closes["date"] = closes.dt.dt.normalize()
    daily_closes = daily[["dt", "close"]].assign(date=daily.dt.dt.normalize())
    joined = daily_closes.merge(closes[["date", "close"]], on="date", suffixes=("_daily", "_intraday"))
    equal = np.isclose(joined.close_daily, joined.close_intraday, rtol=0, atol=1e-8)
    differences = joined.loc[~equal, ["date", "close_daily", "close_intraday"]].copy()
    differences["date"] = differences.date.astype(str)
    return {
        "comparison": "raw daily close versus completed 15:00 local intraday close",
        "comparable_rows": len(joined),
        "matching_rows": int(equal.sum()),
        "match_ratio": float(equal.mean()) if len(equal) else None,
        "differences": differences.to_dict("records"),
    }


def _save(frame: pd.DataFrame, out: Path, metadata: dict) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = frame[FRAME_COLS].copy()
    if out.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(out), frame, check_dtype=True)
    else:
        frame.to_parquet(out, index=False)
    return metadata | {
        "path": str(out.resolve()),
        "derived_path": str(out.resolve()),
        "short_id": metadata["id"],
        "type": "daily" if metadata["frequency"] == "1d" else "intraday",
        "adjust_rolls": False,
        "sha256": _hash(out),
        "rows": len(frame),
        "start": str(frame.dt.min()),
        "end": str(frame.dt.max()),
        "is_2020_2023_rows": int(frame.dt.between("2020-01-01", "2023-12-31 23:59:59").sum()),
        "oos_2024_plus_rows": int((frame.dt >= "2024-01-01").sum()),
        "timezone": "Asia/Shanghai; timezone-naive local wall clock",
        "timestamp_semantics": "completed bar availability; next bar open executes preceding target",
        "microstructure_group": None,
    }


def build_manifest(run_dir: Path = RUN_DIR) -> dict:
    """Materialize a fixed 50 commodity / 10 equity daily reference-price pool."""
    commodities = build_fifty_commodities_universe()
    intraday_files = {p.name.split(".")[1].split("_")[0].lower(): p for p in INTRADAY_SOURCE.glob("*.parquet")}
    daily_files = {p.name.split("_")[3].lower(): p for p in DAILY_SOURCE.glob("*_daily_20200101_20260529.csv")}
    manifest = {
        "version": 1,
        "run_id": "research_rerun_20261017",
        "selection_rule": "Fixed existing 50-commodity specification and 10 named local equity caches; no return screening",
        "training_start": "2020-01-01",
        "training_end": "2023-12-31 23:59:59",
        "oos_start": "2024-01-01",
        "daily": [],
        "intraday": [],
        "missing_intraday_commodities": [],
        "limitations": [
            "Reference continuous prices lack deliverable-contract mappings, rollover orders and true rollover costs.",
            "No exchange calendar, quote depth, limit-lock execution or historical microstructure metadata is available.",
            "Equity indices are reference prices; ETF long-short results are not executable spot strategies.",
            "TQ daily amount was generated from volume*close by the historical downloader, not actual turnover.",
            "Raw daily and 15m snapshots have isolated close discrepancies retained explicitly; neither overwrites the other.",
            "The requested commodity intraday pool lacks cj/lu/pk and usable 2020-2023 training history.",
            "PS starts in 2024, so it has zero observations in the original training window.",
        ],
        "source_code_evidence": [
            _source(ROOT / "examples/signals_dev/backtest_commodity_futures_cta_main_continuous.py")
            | {"detail": "fetch_tq_daily converts epoch to Asia/Shanghai then incorrectly adds one calendar day"},
            _source(ROOT / "examples/signals_dev/sync_full_history_15m.py")
            | {"detail": "TqSdk DataDownloader KQ.m@ symbols, 900-second bar-start timestamps"},
            _source(ROOT / "examples/signals_dev/fetch_daily_ohlcv_jq_extended_pool.py")
            | {"detail": "JQ get_price daily fq=None; date labels represent trading day"},
            _source(ROOT / "examples/signals_dev/test_zdy_macd_bc_000852.py")
            | {"detail": "JQ raw intraday OHLCV and amount, fq=None"},
        ],
    }
    for short_id, spec in commodities:
        common = {
            "id": short_id,
            "symbol": spec.symbol,
            "name": spec.name,
            "asset_group": "commodity",
            "asset_class": "COMMODITY_FUTURES",
            "sector": spec.sector,
            "fee_rate": spec.fee_rate,
            "execution_scope": "continuous_reference_price_only",
        }
        intraday = None
        if short_id in intraday_files:
            intraday, source = _read(intraday_files[short_id])
            has_night = bool(((intraday.dt.dt.hour >= 18) | (intraday.dt.dt.hour < 3)).any())
            intraday["dt"] += pd.Timedelta(minutes=15)
            intraday["symbol"] = spec.symbol
            intraday["amount"] = 0.0  # Explicit unknown sentinel, not an estimate of monetary turnover.
            manifest["intraday"].append(
                _save(
                    intraday,
                    run_dir / "data/intraday" / f"{short_id}_15m.parquet",
                    common
                    | {
                        "source": source,
                        "source_provider": "TqSdk KQ.m main continuous",
                        "frequency": "15m",
                        "has_night_session": has_night,
                        "adjustments": ["bar-start +15min to completed-bar clock", "unknown amount=0"],
                    },
                )
            )
        else:
            manifest["missing_intraday_commodities"].append(short_id)
        if short_id in daily_files:
            daily, source = _read(daily_files[short_id])
            daily["dt"] = daily.dt.dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=15)
            adjusted_path = daily_files[short_id].with_name(
                daily_files[short_id].name.replace("_daily_", "_daily_adjF_")
            )
            adjusted, adjusted_source = _read(adjusted_path)
            # A filename is not proof that a cache actually implements adjustment.
            adjusted_comparison = {
                "source": adjusted_source,
                "same_row_count": len(adjusted) == len(daily),
                "close_difference_rows": int((adjusted.close != daily.close).sum()),
                "choice": "raw source retained; no unverified retrospective adjustment imported",
            }
            adjustments = ["undo documented downloader +1day", "trading-date availability at15:00"]
        else:
            if short_id != "ps" or intraday is None:
                raise FileNotFoundError(f"Missing requested daily commodity: {short_id}")
            # PS has only day sessions, so this grouping needs no inferred night calendar.
            daily = (
                intraday.assign(day=intraday.dt.dt.normalize())
                .groupby("day", as_index=False)
                .agg(
                    dt=("dt", "last"),
                    open=("open", "first"),
                    high=("high", "max"),
                    low=("low", "min"),
                    close=("close", "last"),
                    volume=("volume", "sum"),
                    amount=("amount", "sum"),
                )
            )
            daily = daily.loc[daily.dt.dt.strftime("%H:%M") == "15:00"].reset_index(drop=True)
            source = _source(intraday_files[short_id])
            adjusted_comparison = None
            adjustments = ["aggregate actual PS day-session 15m OHLCV; retain only completed15:00 days"]
        daily["symbol"] = spec.symbol
        overlap = _overlap(daily, intraday) if intraday is not None else None
        manifest["daily"].append(
            _save(
                daily,
                run_dir / "data/daily" / f"{short_id}_1d.parquet",
                common
                | {
                    "source": source,
                    "source_provider": "TqSdk KQ.m main continuous",
                    "frequency": "1d",
                    "has_night_session": False,  # Already a complete trading-day aggregate.
                    "adjustments": adjustments,
                    "adjusted_cache_comparison": adjusted_comparison,
                    "intraday_overlap": overlap,
                },
            )
        )
    for short_id, relative in EQUITY_DAILY.items():
        daily, source = _read(RESULTS / relative)
        daily["dt"] = daily.dt.dt.normalize() + pd.Timedelta(hours=15)
        common = {
            "id": short_id,
            "symbol": str(daily.symbol.iloc[0]),
            "name": str(daily.symbol.iloc[0]),
            "asset_group": "equity",
            "asset_class": "EQUITY_REFERENCE",
            "sector": "EQUITY_REFERENCE",
            "fee_rate": 0.0002,
            "execution_scope": "ETF_reference_only_T1_not_modelled"
            if short_id in ("159915", "588000")
            else "nontradeable_index_reference",
        }
        intraday = None
        if short_id in EQUITY_INTRADAY:
            intraday, intraday_source = _read(RESULTS / EQUITY_INTRADAY[short_id])
            transforms = ["JQ completed-bar timestamps retained"]
            if short_id == "588000":
                grouped = intraday.assign(
                    day=intraday.dt.dt.normalize(), group=intraday.groupby(intraday.dt.dt.date).cumcount() // 3
                )
                intraday = grouped.groupby(["day", "group"], as_index=False).agg(
                    dt=("dt", "last"),
                    symbol=("symbol", "first"),
                    open=("open", "first"),
                    high=("high", "max"),
                    low=("low", "min"),
                    close=("close", "last"),
                    volume=("volume", "sum"),
                    amount=("amount", "sum"),
                    count=("close", "count"),
                )
                intraday = intraday.loc[intraday["count"] == 3].reset_index(drop=True)
                transforms.append("complete consecutive3x5m bars aggregated per trading day")
            manifest["intraday"].append(
                _save(
                    intraday,
                    run_dir / "data/intraday" / f"{short_id}_15m.parquet",
                    common
                    | {
                        "source": intraday_source,
                        "source_provider": "JQData raw local cache",
                        "frequency": "15m",
                        "has_night_session": False,
                        "adjustments": transforms,
                    },
                )
            )
        manifest["daily"].append(
            _save(
                daily,
                run_dir / "data/daily" / f"{short_id}_1d.parquet",
                common
                | {
                    "source": source,
                    "source_provider": "JQData raw local cache",
                    "frequency": "1d",
                    "has_night_session": False,
                    "adjustments": ["remove only pre-listing empty OHLC rows", "same trading date15:00 availability"],
                    "intraday_overlap": _overlap(daily, intraday) if intraday is not None else None,
                },
            )
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    table = pd.DataFrame(
        [
            {
                k: row.get(k)
                for k in (
                    "id",
                    "symbol",
                    "asset_group",
                    "frequency",
                    "rows",
                    "start",
                    "end",
                    "is_2020_2023_rows",
                    "oos_2024_plus_rows",
                    "path",
                    "sha256",
                )
            }
            for row in manifest["daily"] + manifest["intraday"]
        ]
    )
    table.to_csv(run_dir / "data_inventory.csv", index=False)
    return manifest


if __name__ == "__main__":
    result = build_manifest()
    print(
        json.dumps(
            {
                "daily": len(result["daily"]),
                "intraday": len(result["intraday"]),
                "missing_intraday_commodities": result["missing_intraday_commodities"],
            }
        )
    )
