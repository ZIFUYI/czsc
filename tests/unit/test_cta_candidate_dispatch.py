"""Published candidates must rebuild without local, untracked signal extensions."""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from czsc.mock import generate_symbol_kines
from examples.alpha_mining_agent.hypothesis_bank import enrich_kline_features
from examples.alpha_mining_agent.run_evolution_loop import prepare_research_pool
from examples.multi_factor_cta.run_frozen_holdout import _selected_units


def _inputs(monkeypatch):
    source = Path(__file__).resolve().parents[2] / "examples/results/intraday_cta_ablation_20261017"
    monkeypatch.syspath_prepend(str(source))
    study = importlib.import_module("run_multitf_incremental_study")
    study.register_hetero_signal_specs()
    frames = {}
    for tf, frequency in (("1d", "日线"), ("60m", "30分钟"), ("30m", "30分钟"), ("15m", "15分钟")):
        frame = generate_symbol_kines("TEST", frequency, "20240101", "20240305", seed=94)
        if tf == "60m":
            frame = frame.iloc[1::2].reset_index(drop=True)
        frame = frame.rename(columns={"vol": "volume"})
        frame["date"] = pd.to_datetime(frame.dt).dt.date
        frames[tf] = enrich_kline_features(frame)
    frames = prepare_research_pool({"TEST": frames}, metadata={"TEST": {"asset_group": "commodity", "sector": "test"}})[
        "TEST"
    ]
    raw5 = generate_symbol_kines("TEST", "5分钟", "20240101", "20240305", seed=94)
    raw5["open_interest"] = raw5.vol.cumsum()
    return source, study, frames, raw5.rename(columns={"vol": "volume"})


def test_builtin_inventory_recomputes_without_untracked_stage2(monkeypatch):
    source, study, frames, raw5 = _inputs(monkeypatch)
    builtin = set(pd.read_csv(source / "multitf_factor_scores.csv", usecols=["uid"]).uid)
    builtin.update(study.compute_hetero_1d_signals(frames["1d"]))
    builtin.update(study.compute_hetero_15m_signals(frames["15m"]))
    micro = study.build_5m_micro_on_15m(raw5, frames["15m"])
    builtin.update(study.compute_all_5m_candidates(frames["15m"], micro))

    def unavailable(name):
        raise ModuleNotFoundError(name, name=name)

    monkeypatch.setattr("examples.multi_factor_cta.run_frozen_holdout.importlib.import_module", unavailable)
    units = _selected_units(frames, raw5, sorted(builtin), study)
    assert set(units) == builtin
    assert all(weight.shape == (len(frames["15m"]),) for weight in units.values())
    assert all(np.isfinite(weight).all() for weight in units.values())


def test_explicit_extension_request_never_silently_drops_candidate(monkeypatch):
    _, study, frames, raw5 = _inputs(monkeypatch)

    def unavailable(name):
        raise ModuleNotFoundError(name, name=name)

    monkeypatch.setattr("examples.multi_factor_cta.run_frozen_holdout.importlib.import_module", unavailable)
    with pytest.raises(ValueError, match="optional Stage-2 extension"):
        _selected_units(frames, raw5, ["H1D_BC01L@1d"], study)
