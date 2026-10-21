"""Holdout download transforms timestamps and audits coverage without computing returns."""

import numpy as np
import pandas as pd

from czsc.mock import generate_symbol_kines
from examples.multi_factor_cta.download_holdout import _safe_error, normalize_tq_bars, read_tq_universe


def _raw_bars():
    frame = generate_symbol_kines("SHFE.RB9999", "5分钟", "20240102", "20240103", seed=17)
    frame = frame.rename(columns={"vol": "volume"}).iloc[:12].copy()
    completed = pd.to_datetime(frame.dt)
    starts = (completed - pd.Timedelta(minutes=5)).dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")
    frame["datetime"] = starts.astype("int64")
    return frame.drop(columns=["dt"]), completed


def test_holdout_download_uses_completed_bar_clock_and_discards_unfinished_tail():
    raw, completed = _raw_bars()
    frame, audit = normalize_tq_bars(
        raw,
        duration_seconds=300,
        start=str(completed.iloc[0]),
        end=str(completed.iloc[-1]),
        observed_at=str(completed.iloc[7]),
    )
    np.testing.assert_array_equal(frame.dt, completed.iloc[:8])
    assert audit["rows"] == 8
    assert audit["execution_scope"] == "continuous_reference_only"
    assert not any("return" in key or "sharpe" in key for key in audit)


def test_holdout_download_rejects_invalid_prices_and_removes_duplicates():
    raw, completed = _raw_bars()
    raw.loc[raw.index[2], "high"] = raw.loc[raw.index[2], "low"] - 1
    raw = pd.concat([raw, raw.iloc[[4]]], ignore_index=True)
    frame, audit = normalize_tq_bars(
        raw,
        duration_seconds=300,
        start=str(completed.iloc[0]),
        end=str(completed.iloc[-1]),
        observed_at=str(completed.iloc[-1]),
    )
    assert audit["invalid_ohlcv_rows"] == 1
    assert audit["duplicate_timestamps_removed"] == 1
    assert len(frame) == 11
    assert frame.dt.is_monotonic_increasing and not frame.dt.duplicated().any()


def test_universe_has_all_sixty_exchange_correct_continuous_symbols():
    universe = {row["id"]: row for row in read_tq_universe()}
    assert len(universe) == 60
    assert universe["ag"]["tq_symbol"] == "KQ.m@SHFE.ag"
    assert universe["ta"]["tq_symbol"] == "KQ.m@CZCE.TA"
    assert universe["m"]["tq_symbol"] == "KQ.m@DCE.m"
    assert universe["sc"]["tq_symbol"] == "KQ.m@INE.sc"
    assert universe["ps"]["tq_symbol"] == "KQ.m@GFEX.ps"


def test_download_errors_never_expose_credentials_or_service_urls():
    error = RuntimeError("user_example password_example https://service.invalid/?token=private token=another")
    record = _safe_error(error, ("user_example", "password_example"))
    for value in ("user_example", "password_example", "private", "another", "https://"):
        assert value not in record["message"]
