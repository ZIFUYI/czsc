"""Rust/Python structural state-machine contracts; price fixtures come from czsc.mock."""

import numpy as np
import pytest

import czsc
from czsc.mock import generate_symbol_kines


def _input():
    # Transition vectors mirror crates/czsc-core/tests/structure_exit.rs; these are
    # algebraic state-machine cases, not simulated market performance evidence.
    return {
        "setup_long": [True, False, False, False],
        "setup_short": [False] * 4,
        "setup_quality_long": [1.0] * 4,
        "setup_quality_short": [1.0] * 4,
        "confirm_long": [False, False, True, False],
        "confirm_short": [False] * 4,
        "confirm_quality_long": [1.0] * 4,
        "confirm_quality_short": [1.0] * 4,
        "close": [100.0, 101.0, 102.0, 99.0],
        "long_stop": [90.0, 98.0, 101.0, 98.0],
        "short_stop": [110.0] * 4,
        "completed_bi_dir": [0] * 4,
        "atr_fast": [1.0] * 4,
        "atr_slow": [1.0] * 4,
    }


def _bars():
    df = generate_symbol_kines("TEST", "30分钟", "20220101", "20220301", seed=905)
    return czsc.format_standard_kline(df.iloc[:250], freq=czsc.Freq.F30)


def test_native_surface_is_pure_reexport_and_freezes_setup_anchor():
    from czsc import _native

    assert czsc.evaluate_structure_factor is _native.evaluate_structure_factor
    result = czsc.evaluate_structure_factor(**_input())
    assert result["trigger"] == [False, False, True, False]
    assert result["setup_index"][2] == 0
    assert result["stop"][2] == 90.0
    assert result["strength"][3] == 0.0
    assert result["exit_reason"][3] == "trailing_stop"
    assert result["position_direction"] == [0, 0, 1, 1]


@pytest.mark.parametrize("change", ["broken", "same_bar", "missing"])
def test_invalid_setup_never_fires(change):
    args = _input()
    if change == "broken":
        args["close"][1] = 89.0
    elif change == "same_bar":
        args["setup_long"] = args["confirm_long"].copy()
    else:
        args["long_stop"] = [np.nan] * 4
    assert not any(czsc.evaluate_structure_factor(**args)["trigger"])


def test_repeated_setup_preserves_structure_identity():
    args = _input()
    args["setup_long"][1] = True
    result = czsc.evaluate_structure_factor(**args)
    assert result["stop"][2] == 90.0
    assert result["setup_index"][2] == 0


def test_opposite_completed_bi_exits_without_changing_past():
    args = _input()
    args["close"][3] = 102.0
    args["completed_bi_dir"][3] = -1
    result = czsc.evaluate_structure_factor(**args)
    assert result["exit_reason"][3] == "reverse_bi"
    args["close"][3] = 120.0
    changed = czsc.evaluate_structure_factor(**args)
    assert result["strength"][:3] == changed["strength"][:3]
    assert result["stop"][2] == changed["stop"][2]


def test_future_setup_and_structure_changes_cannot_rewrite_decisions():
    args = _input()
    before = czsc.evaluate_structure_factor(**args)
    for key in ("setup_long", "setup_short", "confirm_long", "confirm_short"):
        args[key][3] = True
    args["long_stop"][3] = 1.0
    args["short_stop"][3] = 1000.0
    args["setup_quality_long"][3] = 0.1
    args["completed_bi_dir"][3] = -1
    after = czsc.evaluate_structure_factor(**args)
    for field in ("strength", "trigger", "setup_index", "position_direction", "stop"):
        np.testing.assert_allclose(before[field][:3], after[field][:3])


def test_long_short_symmetry_and_shared_validation():
    long = _input()
    short = _input()
    short["setup_short"] = long["setup_long"].copy()
    short["setup_long"] = [False] * 4
    short["confirm_short"] = long["confirm_long"].copy()
    short["confirm_long"] = [False] * 4
    short["close"] = [100.0, 99.0, 98.0, 101.0]
    short["short_stop"] = [110.0, 102.0, 99.0, 102.0]
    a, b = czsc.evaluate_structure_factor(**long), czsc.evaluate_structure_factor(**short)
    assert a["trigger"] == b["trigger"]
    assert a["strength"][2] == -b["strength"][2]
    assert a["exit_reason"] == b["exit_reason"]
    assert a["position_direction"][3] == -b["position_direction"][3]
    with pytest.raises(ValueError, match="equal lengths"):
        czsc.evaluate_structure_factor(**{**long, "close": long["close"][:-1]})
    with pytest.raises(ValueError, match="trailing"):
        czsc.evaluate_structure_factor(**long, trailing=0.1)


def test_snapshots_are_prefix_causal_and_completed_bis_are_native_events():
    bars = _bars()
    all_rows = czsc.structure_snapshots(bars)
    prefix = czsc.structure_snapshots(bars[:125])
    for field in ("fx_low", "fx_high", "bi_low", "bi_high", "zs_mid", "completed_bi_dir"):
        np.testing.assert_allclose([getattr(r, field) for r in all_rows[:125]], [getattr(r, field) for r in prefix])
    events = [r for r in all_rows if r.completed_bi_dir]
    assert events
    assert len({r.completed_bi_id for r in events}) == len(events)
    for i, row in enumerate(all_rows):
        if row.completed_bi_dir:
            analyzer = czsc.CZSC(bars[: i + 1])
            bi = analyzer.finished_bis[-1]
            assert row.completed_bi_dir == (1 if bi.direction == czsc.Direction.Up else -1)
    assert czsc.structure_snapshots([]) == []
    with pytest.raises(ValueError, match="increasing"):
        czsc.structure_snapshots([bars[0], bars[0]])


@pytest.mark.parametrize(
    "plugin,long_field,short_field",
    [
        ("pos_fx_stop", "fx_low", "fx_high"),
        ("pos_structure_ext", "bi_low", "bi_high"),
        ("pos_zs_mid", "zs_low", "zs_high"),
    ],
)
def test_plugin_dispatch_reads_actual_structures(plugin, long_field, short_field):
    bars = _bars()
    snapshots = czsc.structure_snapshots(bars)
    anchors = czsc.structure_stop_anchors(bars, snapshots, plugin)
    np.testing.assert_allclose(anchors["long_stop"], [getattr(s, long_field) for s in snapshots])
    np.testing.assert_allclose(anchors["short_stop"], [getattr(s, short_field) for s in snapshots])


def test_flow_box_and_htf_have_distinct_anchor_semantics():
    bars = _bars()
    snapshots = czsc.structure_snapshots(bars)
    flow = czsc.structure_stop_anchors(bars, snapshots, "pos_flow_stop")
    assert flow["long_stop"][20] == bars[20].low
    box = czsc.structure_stop_anchors(bars, snapshots, "pos_box_stop")
    assert np.isnan(box["long_stop"][18])
    assert box["long_stop"][19] == min(b.low for b in bars[:20])
    missing = czsc.structure_stop_anchors(bars, snapshots, "pos_htf_ma_stop")
    assert np.isnan(missing["long_stop"]).all()
    ma = [b.close for b in bars]
    htf = czsc.structure_stop_anchors(bars, snapshots, "pos_htf_ma_stop", htf_ma=ma)
    assert htf["long_stop"] == ma
    with pytest.raises(ValueError, match="unknown stop_plugin"):
        czsc.structure_stop_anchors(bars, snapshots, "unknown")


def test_compatibility_trigger_and_decay_keep_shared_clock():
    fire, q = czsc.two_stage_trigger([True, False, False], [0.5] * 3, [True, True, False], [1.0] * 3, 2)
    assert fire == [False, True, False]
    assert q == [0.0, 0.5, 0.0]
    result = czsc.evaluate_structure_decay([1, 0], [1.0, 0.0], [100.0, 98.0], [90.0] * 2, [0] * 2, [1.0] * 2, [1.0] * 2)
    assert result["strength"][1] == pytest.approx(np.exp(-0.1))
    assert result["exit_reason"][1] == ""


@pytest.mark.parametrize("reason", ["reverse_bi", "structure_break", "trailing_stop"])
def test_risk_exit_blocks_same_bar_same_side_reentry(reason):
    args = _input()
    args["setup_long"] = [True, False, True, False]
    args["confirm_long"] = [False, True, False, True]
    args["long_stop"] = [90.0] * 4
    args["close"] = [100.0, 101.0, 102.0, 102.0]
    if reason == "reverse_bi":
        args["completed_bi_dir"][3] = -1
    elif reason == "structure_break":
        args["close"][3] = 89.0
    else:
        args["close"][3] = 99.0
    result = czsc.evaluate_structure_factor(**args)
    assert result["exit_reason"][3] == reason
    assert result["strength"][3] == 0.0 and not result["trigger"][3]
    decay = czsc.evaluate_structure_decay(
        [0, 1, 0, 1],
        [1.0] * 4,
        args["close"],
        [90.0] * 4,
        args["completed_bi_dir"],
        [1.0] * 4,
        [1.0] * 4,
        trailing=0.013,
    )
    assert decay["strength"][3] == 0.0 and not decay["trigger"][3]


def test_snapshot_asof_rejects_anchors_from_another_window():
    bars = _bars()
    snapshots = czsc.structure_snapshots(bars)
    snapshots[0], snapshots[1] = snapshots[1], snapshots[0]
    with pytest.raises(ValueError, match="asof"):
        czsc.structure_stop_anchors(bars, snapshots, "pos_fx_stop")


def test_snapshot_defaults_follow_the_same_native_configuration_as_signals(monkeypatch):
    monkeypatch.setenv("CZSC_MIN_BI_LEN", "7")
    monkeypatch.setenv("CZSC_MAX_BI_NUM", "12")
    bars = _bars()
    expected = czsc.structure_snapshots(bars, min_bi_len=7, max_bi_num=12)
    actual = czsc.structure_snapshots(bars)
    for field in ("fx_low", "fx_high", "bi_low", "bi_high", "zs_mid", "completed_bi_dir"):
        np.testing.assert_allclose([getattr(s, field) for s in actual], [getattr(s, field) for s in expected])


def test_zs_anchor_uses_the_twenty_bi_window_of_reversion_setup():
    df = generate_symbol_kines("TEST", "30分钟", "20220101", "20221231", seed=913)
    bars = czsc.format_standard_kline(df.iloc[:1500], freq=czsc.Freq.F30)
    snapshots = czsc.structure_snapshots(bars, min_bi_len=6, max_bi_num=50)
    analyzer = czsc.CZSC(bars, min_bi_len=6, max_bi_num=50)
    bis = analyzer.finished_bis
    assert len(bis) > 20
    zones = czsc.get_zs_seq(bis[-20:])
    zone = next(z for z in reversed(zones) if z.is_valid())
    assert snapshots[-1].zs_low == zone.dd
    assert snapshots[-1].zs_high == zone.gg
    assert snapshots[-1].zs_mid == zone.zz


@pytest.mark.parametrize("side", [1, -1])
def test_default_trailing_exits_exactly_at_one_point_three_percent(side):
    best = 102.0 if side == 1 else 98.0
    boundary = best * (1.0 - side * 0.013)
    args = _input()
    for values in args.values():
        values.append(values[-1])
    args["setup_long"] = [side == 1, False, False, False, False]
    args["setup_short"] = [side == -1, False, False, False, False]
    args["confirm_long"] = [False, side == 1, False, False, False]
    args["confirm_short"] = [False, side == -1, False, False, False]
    args["close"] = [100.0, 100.0, best, boundary + side * 1e-8, boundary]
    args["long_stop"], args["short_stop"] = [90.0] * 5, [110.0] * 5
    result = czsc.evaluate_structure_factor(**args)
    assert result["trigger"][1]
    assert np.sign(result["strength"][3]) == side
    assert result["strength"][4] == 0.0
    assert result["exit_reason"][4] == "trailing_stop"


def test_reverse_bi_exit_cannot_be_cancelled_by_armed_same_direction_setup():
    args = _input()
    args["close"] = [100.0, 101.0, 102.0, 102.0]
    args["setup_long"] = [True, False, True, False]
    args["confirm_long"] = [False, True, False, True]
    args["long_stop"] = [90.0] * 4
    args["completed_bi_dir"][3] = -1
    result = czsc.evaluate_structure_factor(**args)
    assert result["exit_reason"][3] == "reverse_bi"
    assert result["strength"][3] == 0.0
    assert not result["trigger"][3]


def test_secondary_extrema_and_zs_reentry_plugins_and_snapshot_fields():
    bars = _bars()
    all_rows = czsc.structure_snapshots(bars)
    prefix = czsc.structure_snapshots(bars[:125])
    for field in ("bi_second_low", "bi_second_high", "zs_zd", "zs_zg", "completed_fx_dir"):
        np.testing.assert_allclose([getattr(r, field) for r in all_rows[:125]], [getattr(r, field) for r in prefix])
    assert any(r.completed_fx_dir != 0 for r in all_rows)
    sec = czsc.structure_stop_anchors(bars, all_rows, "pos_second_ext")
    zs_re = czsc.structure_stop_anchors(bars, all_rows, "pos_zs_reentry")
    np.testing.assert_allclose(sec["long_stop"], [r.bi_second_low for r in all_rows])
    np.testing.assert_allclose(sec["short_stop"], [r.bi_second_high for r in all_rows])
    for b, r, l_s, s_s in zip(bars, all_rows, zs_re["long_stop"], zs_re["short_stop"], strict=True):
        if np.isfinite(r.zs_zg) and b.close > r.zs_zg:
            assert l_s >= r.zs_zg
        if np.isfinite(r.zs_zd) and b.close < r.zs_zd:
            assert s_s <= r.zs_zd


def test_ratchet_stop_zs_reentry_trailing_fx_and_continuous_track_signal():
    args = _input()
    args["close"] = [100.0, 102.0, 105.0, 100.5]
    args["confirm_long"] = [False, True, False, False]
    args["long_stop"] = [95.0, 95.0, 101.0, 101.0]
    res_ratchet = czsc.evaluate_structure_factor(**args, ratchet_stop=True, trailing=0.015, trail_activation=0.05)
    assert res_ratchet["stop"][1] == 95.0
    assert res_ratchet["stop"][2] == 101.0
    assert res_ratchet["exit_reason"][3] == "structure_break"
    assert res_ratchet["strength"][3] == 0.0

    args_zs = _input()
    args_zs["close"] = [100.0, 102.0, 105.0, 102.8]
    args_zs["confirm_long"] = [False, True, False, False]
    args_zs["long_stop"] = [95.0] * 4
    res_zs = czsc.evaluate_structure_factor(**args_zs, trail_activation=0.10, zs_zg=[103.0] * 4)
    assert res_zs["exit_reason"][3] == "zs_reentry_break"
    assert res_zs["strength"][3] == 0.0

    args_fx = _input()
    args_fx["close"] = [100.0, 100.0, 103.0, 102.4]
    args_fx["confirm_long"] = [False, True, False, False]
    args_fx["long_stop"] = [95.0] * 4
    res_fx = czsc.evaluate_structure_factor(**args_fx, trail_activation=0.02, completed_fx_dir=[0, 0, 0, -1])
    assert res_fx["exit_reason"][3] == "trailing_fx_exit"

    trk = czsc.evaluate_structure_decay(
        [1, 1, 1, 1, 1],
        [0.6, 0.8, 0.8, 0.8, 0.8],
        [100.0, 102.0, 100.5, 101.0, 101.5],
        [95.0, 98.0, 98.0, 98.0, 98.0],
        [0, 0, 0, 0, 0],
        [1.0] * 5,
        [1.0] * 5,
        lam0=0.0,
        trailing=0.013,
        trail_activation=0.015,
        ratchet_stop=True,
        track_signal=True,
        completed_fx_dir=[0, 0, 0, 0, 1],
    )
    assert trk["strength"] == [0.6, 0.8, 0.0, 0.0, 0.8]
    assert trk["exit_reason"][2] == "trailing_stop"
