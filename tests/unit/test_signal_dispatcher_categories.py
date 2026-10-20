"""Rust 信号分发器分类（17 个子模块）与 ``strength_*`` 信号的 Python 公共调用路径回归测试。

对应 Rust 侧 ``czsc-signals/src/registry.rs`` 中的 ``KLINE_CATEGORY_PREFIXES`` 与
``crates/czsc-signals/src/strength.rs`` 的单测（宪法第一条：两端都必须有等价用例）。
"""

from __future__ import annotations

import pandas as pd
import pytest

import czsc
from czsc import CZSC, Freq, format_standard_kline
from czsc import _native as native
from czsc.mock import generate_symbol_kines

EXPECTED_CATEGORIES = [
    "bar",
    "cxt",
    "tas",
    "vol",
    "pressure",
    "obv",
    "cvolp",
    "ang",
    "byi",
    "clv",
    "coo",
    "jcc",
    "kcatr",
    "ntmdk",
    "xl",
    "zdy",
    "strength",
]
STRENGTH_SIGNALS = [
    "strength_fx_V260601",
    "strength_bi_div_V260601",
    "strength_zs_revert_V260601",
    "strength_zs_break_V260601",
    "strength_flow_V260601",
    "strength_squeeze_V260601",
]


@pytest.fixture(scope="module")
def czsc_obj() -> CZSC:
    df = generate_symbol_kines("000001", "30分钟", "20230101", "20231231", seed=7)
    bars = format_standard_kline(df, freq=Freq.F30)
    return CZSC(bars)


def test_seventeen_submodules_exposed_and_nonempty():
    for cat in EXPECTED_CATEGORIES:
        mod = getattr(native.signals, cat)
        assert mod.__category__ == cat
        assert native.list_signal_names(cat), f"分类 {cat} 不应为空"


def test_categories_partition_all_kline_signals():
    all_names = native.list_signal_names()
    seen: dict[str, str] = {}
    for cat in EXPECTED_CATEGORIES:
        for name in native.list_signal_names(cat):
            assert name not in seen, f"{name} 同时属于 {seen[name]} 与 {cat}"
            seen[name] = cat
    assert set(seen) == set(all_names)


def test_categories_follow_rust_module_boundaries():
    assert "amv_up_dw_line_V230603" in native.list_signal_names("ang")
    assert "adtm_up_dw_line_V230603" in native.list_signal_names("ang")
    assert not native.list_signal_names("amv")  # 前缀不再是分类
    assert any(n.startswith("cci_") for n in native.list_signal_names("tas"))
    assert any(n.startswith("obvm_") for n in native.list_signal_names("obv"))
    assert native.get_signal_category("amv_up_dw_line_V230603") == "ang"
    assert native.get_signal_category("clv_up_dw_line_V230605") == "clv"
    # trader 状态信号没有 K 线模块，回退为名字前缀
    assert native.get_signal_category("pos_ma_V230414") == "pos"
    assert native.get_signal_category("not_exists_V000000") is None


@pytest.mark.parametrize("name", STRENGTH_SIGNALS)
def test_strength_signal_registered_with_template(name: str):
    assert name in native.list_signal_names("strength")
    tpl = native.get_signal_template(name)
    assert tpl and tpl.startswith("{freq}_")


@pytest.mark.parametrize("name", STRENGTH_SIGNALS)
def test_strength_signal_score_contract(name: str, czsc_obj: CZSC):
    """call_signal 返回的 Signal：无信号 score==0；有信号 v1 为看多/看空且 score∈[1,100]。"""
    sigs = native.call_signal(name, czsc_obj)
    assert len(sigs) == 1
    sig = sigs[0]
    assert 0 <= sig.score <= 100
    if sig.v1 == "其他":
        assert sig.score == 0
    else:
        assert sig.v1 in ("看多", "看空")
        assert 1 <= sig.score <= 100


def test_strength_signals_reachable_via_generate_czsc_signals():
    """通过公共 API（generate_czsc_signals）能解析并计算 strength 信号，score 以第 4 段落盘。"""
    df = generate_symbol_kines("000001", "30分钟", "20230101", "20230630", seed=11)
    bars = format_standard_kline(df, freq=Freq.F30)
    cfg = [{"name": name.removesuffix("_V260601"), "freq": Freq.F30.value} for name in STRENGTH_SIGNALS]
    out = czsc.generate_czsc_signals(bars, cfg, sdt="2023-02-01", init_n=100, df=True)
    legacy_cfg = [{"name": name, "freq": Freq.F30.value} for name in STRENGTH_SIGNALS]
    legacy = czsc.generate_czsc_signals(bars, legacy_cfg, sdt="2023-02-01", init_n=100, df=True)
    pd.testing.assert_frame_equal(out.sort_index(axis=1), legacy.sort_index(axis=1))
    sig_cols = [c for c in out.columns if "V260601" in c]
    assert len(sig_cols) == len(STRENGTH_SIGNALS)
    for col in sig_cols:
        parts = out[col].astype(str).str.split("_", expand=True)
        assert parts.shape[1] == 4  # v1_v2_v3_score
        scores = parts[3].astype(int)
        assert scores.between(0, 100).all()


@pytest.mark.parametrize("legacy_name", STRENGTH_SIGNALS)
def test_strength_canonical_name_matches_legacy(legacy_name: str, czsc_obj: CZSC):
    canonical = legacy_name.removesuffix("_V260601")
    assert canonical in native.list_signal_names("strength")
    assert native.get_signal_template(canonical) == native.get_signal_template(legacy_name)
    old = native.call_signal(legacy_name, czsc_obj)
    new = native.call_signal(canonical, czsc_obj)
    assert [signal.to_json() for signal in old] == [signal.to_json() for signal in new]


def test_research_default_strength_config_uses_canonical_names():
    from examples.alpha_mining_agent.hypothesis_bank import _extended_native_signal_cfgs

    names = [item["name"].rsplit(".", 1)[-1] for item in _extended_native_signal_cfgs("30分钟")]
    assert {name.removesuffix("_V260601") for name in STRENGTH_SIGNALS}.issubset(names)
    assert not set(STRENGTH_SIGNALS).intersection(names)


@pytest.mark.parametrize("legacy_name", STRENGTH_SIGNALS)
def test_strength_reverse_config_prefers_canonical_name(legacy_name: str, czsc_obj: CZSC):
    signal = native.call_signal(legacy_name, czsc_obj)[0]
    # Use a non-default target value so the public parser includes the signal.
    configs = czsc.get_signals_config([f"{signal.key}_看多_任意_任意_50"])
    assert len(configs) == 1
    assert configs[0]["name"].rsplit(".", 1)[-1] == legacy_name.removesuffix("_V260601")
