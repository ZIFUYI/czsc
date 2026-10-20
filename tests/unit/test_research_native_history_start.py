"""Research native features must preserve history when later bars cross 2020."""

import numpy as np
import pandas as pd

import czsc
from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.hypothesis_bank import attach_native_czsc_signals, enrich_kline_features
from examples.alpha_mining_agent.run_evolution_loop import compute_research_signal


def test_native_history_start_and_factor_prefix_survive_2020_boundary(tmp_path, monkeypatch):
    raw = generate_symbol_kines("HISTORY", "日线", "20170101", "20210131", seed=83)
    raw = raw.rename(columns={"vol": "volume"})
    raw["dt"] = pd.to_datetime(raw["dt"]).dt.normalize() + pd.Timedelta(hours=15)
    raw["date"] = raw["dt"].dt.date
    raw["grp"] = 0
    cutoff = pd.Timestamp("2019-12-31 23:59:59")
    prefix = raw.loc[raw["dt"] <= cutoff].copy()
    observed_starts = []
    generate = czsc.generate_czsc_signals

    def record_start(bars, signals_config, **kwargs):
        observed_starts.append(pd.Timestamp(kwargs["sdt"]))
        return generate(bars, signals_config, **kwargs)

    monkeypatch.setattr(czsc, "generate_czsc_signals", record_start)
    complete = attach_native_czsc_signals(enrich_kline_features(raw), "1d", tmp_path / "complete")
    truncated = attach_native_czsc_signals(enrich_kline_features(prefix), "1d", tmp_path / "truncated")

    assert observed_starts == [raw["dt"].iloc[0], prefix["dt"].iloc[0]]
    native_columns = [column for column in truncated if column.startswith("cz_")]
    assert native_columns
    pd.testing.assert_frame_equal(
        complete.loc[: len(prefix) - 1, native_columns], truncated[native_columns], check_exact=True
    )
    categorical = truncated[native_columns].select_dtypes(include="object")
    assert not categorical.empty
    assert categorical.ne("其他").to_numpy().any(), "Prefix equality must include computed native observations"
    for candidate in ("AC01L", "AC01S"):
        full_trigger, full_weight = compute_research_signal(complete, candidate, "1d")
        prefix_trigger, prefix_weight = compute_research_signal(truncated, candidate, "1d")
        np.testing.assert_array_equal(full_trigger[: len(prefix)], prefix_trigger)
        np.testing.assert_array_equal(full_weight[: len(prefix)], prefix_weight)
