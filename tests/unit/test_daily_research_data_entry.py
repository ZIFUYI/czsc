"""Closed daily source bars enter the same formal research loader as intraday data."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.multi_factor_cta.data_engine import AssetClass, AssetSpec, CrossAssetDataEngine


def test_daily_source_preserves_dates_and_raw_returns(tmp_path):
    raw = generate_symbol_kines("DAILY", "日线", "20200101", "20211231", seed=83)
    raw = raw.rename(columns={"vol": "volume"})
    raw["dt"] = pd.to_datetime(raw["dt"]).dt.normalize() + pd.Timedelta(hours=15)
    path = tmp_path / "daily.parquet"
    raw.to_parquet(path)
    spec = AssetSpec(
        symbol="DAILY",
        name="Daily reference",
        asset_class=AssetClass.COMMODITY_FUTURES,
        sector="TEST",
        base_tf="1d",
        eligible_tfs=("1d",),
        fee_rate=0.00015,
        data_path=path,
        adjust_rolls=False,
    )
    engine = CrossAssetDataEngine(tmp_path / "cache")
    result = engine.load_asset_multi_timeframes(spec, attach_czsc=False)["1d"]
    assert result["dt"].tolist() == raw["dt"].tolist()
    for column in ("open", "high", "low", "close"):
        np.testing.assert_allclose(result[column] / result[column].iloc[0], raw[column] / raw[column].iloc[0])
    with pytest.raises(ValueError, match="finer than"):
        engine.load_asset_multi_timeframes(replace(spec, eligible_tfs=("30m", "1d")))


def test_equity_reference_has_explicit_non_futures_identity():
    assert AssetClass.EQUITY_REFERENCE.value == "EQUITY_REFERENCE"
