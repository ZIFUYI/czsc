use czsc_core::structure_exit::{
    StructureFactorConfig, StructureFactorInput, evaluate_structure_factor,
};

fn input() -> StructureFactorInput {
    StructureFactorInput {
        setup_long: vec![true, false, false, false],
        setup_short: vec![false; 4],
        setup_quality_long: vec![1.0; 4],
        setup_quality_short: vec![1.0; 4],
        confirm_long: vec![false, false, true, false],
        confirm_short: vec![false; 4],
        confirm_quality_long: vec![1.0; 4],
        confirm_quality_short: vec![1.0; 4],
        close: vec![100.0, 101.0, 102.0, 99.0],
        long_stop: vec![90.0, 98.0, 101.0, 98.0],
        short_stop: vec![110.0; 4],
        completed_bi_dir: vec![0; 4],
        atr_fast: vec![1.0; 4],
        atr_slow: vec![1.0; 4],
    }
}

#[test]
fn freezes_setup_anchor_and_exits_on_trailing_without_retroactive_entry() {
    let out = evaluate_structure_factor(&input(), &StructureFactorConfig::default()).unwrap();
    assert_eq!(out.trigger, vec![false, false, true, false]);
    assert_eq!(out.setup_index[2], 0);
    assert_eq!(out.stop[2], 90.0);
    assert_eq!(out.strength[3], 0.0);
    assert_eq!(out.exit_reason[3], "trailing_stop");
    assert_eq!(out.position_direction, [0, 0, 1, 1]);
}

#[test]
fn broken_setup_cannot_be_confirmed_and_same_bar_setup_does_not_fire() {
    let mut x = input();
    x.close[1] = 89.0;
    let out = evaluate_structure_factor(&x, &StructureFactorConfig::default()).unwrap();
    assert!(out.trigger.iter().all(|x| !x));
    x = input();
    x.setup_long = x.confirm_long.clone();
    let out = evaluate_structure_factor(&x, &StructureFactorConfig::default()).unwrap();
    assert!(out.trigger.iter().all(|x| !x));
}

#[test]
fn completed_opposite_bi_exits_and_prefix_is_unchanged() {
    let mut x = input();
    x.close[3] = 102.0;
    x.completed_bi_dir[3] = -1;
    let out = evaluate_structure_factor(&x, &StructureFactorConfig::default()).unwrap();
    assert_eq!(out.exit_reason[3], "reverse_bi");
    x.close[3] = 120.0;
    let changed = evaluate_structure_factor(&x, &StructureFactorConfig::default()).unwrap();
    assert_eq!(out.strength[..3], changed.strength[..3]);
    assert_eq!(out.stop[2], changed.stop[2]);
}

#[test]
fn long_short_rules_are_symmetric_and_validation_is_shared() {
    let long = input();
    let mut short = input();
    short.setup_short = long.setup_long.clone();
    short.setup_long = vec![false; 4];
    short.confirm_short = long.confirm_long.clone();
    short.confirm_long = vec![false; 4];
    // Reflect around 100, keeping the same relative trailing drawdown at the exit bar.
    short.close = vec![100.0, 99.0, 98.0, 101.0];
    short.short_stop = vec![110.0, 102.0, 99.0, 102.0];
    let cfg = StructureFactorConfig::default();
    let a = evaluate_structure_factor(&long, &cfg).unwrap();
    let b = evaluate_structure_factor(&short, &cfg).unwrap();
    assert_eq!(a.trigger, b.trigger);
    assert_eq!(a.strength[2], -b.strength[2]);
    assert_eq!(a.exit_reason, b.exit_reason);
    assert_eq!(a.position_direction[3], -b.position_direction[3]);
    short.close.pop();
    assert!(evaluate_structure_factor(&short, &cfg).is_err());
    let invalid = StructureFactorConfig {
        trailing: 0.1,
        ..cfg
    };
    assert!(evaluate_structure_factor(&long, &invalid).is_err());
}

#[test]
fn repeated_setup_does_not_move_frozen_anchor_and_missing_structure_cannot_fire() {
    let mut x = input();
    x.setup_long[1] = true;
    let out = evaluate_structure_factor(&x, &StructureFactorConfig::default()).unwrap();
    assert_eq!(out.stop[2], 90.0);
    assert_eq!(out.setup_index[2], 0);
    x.long_stop.fill(f64::NAN);
    let out = evaluate_structure_factor(&x, &StructureFactorConfig::default()).unwrap();
    assert!(out.trigger.iter().all(|v| !v));
}

#[test]
fn future_setup_and_anchor_changes_cannot_rewrite_past_decisions() {
    let original = input();
    let mut changed = original.clone();
    changed.setup_long[3] = true;
    changed.setup_short[3] = true;
    changed.confirm_long[3] = true;
    changed.confirm_short[3] = true;
    changed.long_stop[3] = 1.0;
    changed.short_stop[3] = 1000.0;
    changed.setup_quality_long[3] = 0.1;
    changed.completed_bi_dir[3] = -1;
    let a = evaluate_structure_factor(&original, &StructureFactorConfig::default()).unwrap();
    let b = evaluate_structure_factor(&changed, &StructureFactorConfig::default()).unwrap();
    assert_eq!(a.strength[..3], b.strength[..3]);
    assert_eq!(a.trigger[..3], b.trigger[..3]);
    assert_eq!(a.setup_index[..3], b.setup_index[..3]);
    assert_eq!(a.position_direction[..3], b.position_direction[..3]);
    for i in 0..3 {
        assert_eq!(a.stop[i].to_bits(), b.stop[i].to_bits());
    }
}

fn bars() -> Vec<czsc_core::objects::bar::RawBar> {
    use chrono::{TimeZone, Utc};
    use czsc_core::objects::{bar::RawBarBuilder, freq::Freq};
    (0i32..160)
        .map(|i| {
            let close = 100.0 + 10.0 * (f64::from(i) * std::f64::consts::PI / 10.0).sin();
            RawBarBuilder::default()
                .symbol("TEST")
                .id(i)
                .freq(Freq::F30)
                .dt(Utc
                    .timestamp_opt(1_700_000_000 + i64::from(i) * 1800, 0)
                    .unwrap())
                .open(close)
                .close(close)
                .high(close + 1.0)
                .low(close - 1.0)
                .vol(100.0)
                .amount(close * 100.0)
                .build()
                .unwrap()
        })
        .collect()
}

#[test]
fn snapshots_preserve_prefix_and_only_emit_new_completed_native_bis() {
    use czsc_core::analyze::CZSC;
    use czsc_core::structure_exit::structure_snapshots;
    let bars = bars();
    let all = structure_snapshots(&bars, 6, 50).unwrap();
    let prefix = structure_snapshots(&bars[..80], 6, 50).unwrap();
    let mut c = CZSC::new(vec![bars[0].clone()], 50, 6);
    let mut last_id = -1;
    let mut emitted = 0;
    for (i, row) in all.iter().enumerate() {
        if i > 0 {
            c.update_bar(bars[i].clone());
        }
        if row.completed_bi_dir != 0 {
            let bi = c.get_finished_bis().last().unwrap().clone();
            assert_eq!(row.completed_bi_id, bi.end_dt().timestamp_millis());
            assert!(row.completed_bi_id > last_id);
            last_id = row.completed_bi_id;
            emitted += 1;
        }
        if i < prefix.len() {
            assert_eq!(row.completed_bi_dir, prefix[i].completed_bi_dir);
            assert_eq!(row.fx_low_id, prefix[i].fx_low_id);
            assert_eq!(row.bi_id, prefix[i].bi_id);
            assert_eq!(row.zs_id, prefix[i].zs_id);
            assert_eq!(row.fx_low.to_bits(), prefix[i].fx_low.to_bits());
        }
    }
    assert!(emitted >= 3);
    assert!(all.iter().any(|x| x.fx_low.is_finite()));
    assert!(all.iter().any(|x| x.zs_mid.is_finite()));
    assert!(structure_snapshots(&bars[..0], 6, 50).unwrap().is_empty());
    assert!(structure_snapshots(&[bars[0].clone(), bars[0].clone()], 6, 50).is_err());
}

#[test]
fn stop_plugins_use_actual_source_structures_and_fail_closed_without_htf() {
    use czsc_core::structure_exit::{structure_snapshots, structure_stop_anchors};
    let bars = bars();
    let snapshots = structure_snapshots(&bars, 6, 50).unwrap();
    for (plugin, field) in [
        ("pos_fx_stop", 0),
        ("pos_structure_ext", 1),
        ("pos_zs_mid", 2),
    ] {
        let out = structure_stop_anchors(&bars, &snapshots, plugin, None, 20).unwrap();
        for (i, s) in snapshots.iter().enumerate() {
            let expected = [s.fx_low, s.bi_low, s.zs_low][field];
            assert_eq!(out.long_stop[i].to_bits(), expected.to_bits());
        }
    }
    let flow = structure_stop_anchors(&bars, &snapshots, "pos_flow_stop", None, 20).unwrap();
    assert_eq!(flow.long_stop[20], bars[20].low);
    let bx = structure_stop_anchors(&bars, &snapshots, "pos_box_stop", None, 20).unwrap();
    assert!(bx.long_stop[18].is_nan());
    assert_eq!(
        bx.long_stop[19],
        bars[..20]
            .iter()
            .map(|b| b.low)
            .fold(f64::INFINITY, f64::min)
    );
    let htf = structure_stop_anchors(&bars, &snapshots, "pos_htf_ma_stop", None, 20).unwrap();
    assert!(htf.long_stop.iter().all(|x| x.is_nan()));
    assert!(structure_stop_anchors(&bars, &snapshots, "unknown", None, 20).is_err());
}

#[test]
fn compatibility_entries_keep_old_clock_and_optional_trailing() {
    use czsc_core::structure_exit::{evaluate_structure_decay, two_stage_trigger};
    let (fire, q) = two_stage_trigger(
        &[true, false, false],
        &[0.5; 3],
        &[true, true, false],
        &[1.0; 3],
        2,
    )
    .unwrap();
    assert_eq!(fire, [false, true, false]);
    assert_eq!(q, [0.0, 0.5, 0.0]);
    let cfg = StructureFactorConfig {
        trailing: 0.0,
        ..Default::default()
    };
    let out = evaluate_structure_decay(
        &[1, 0],
        &[1.0, 0.0],
        &[100.0, 98.0],
        &[90.0; 2],
        &[0; 2],
        &[1.0; 2],
        &[1.0; 2],
        &cfg,
    )
    .unwrap();
    assert!((out.strength[1] - (-0.1_f64).exp()).abs() < 1e-12);
    assert!(out.exit_reason[1].is_empty());
}

#[test]
fn risk_exit_cannot_reopen_same_direction_on_the_same_bar() {
    use czsc_core::structure_exit::evaluate_structure_decay;
    for reason in ["reverse_bi", "structure_break", "trailing_stop"] {
        let mut x = input();
        x.setup_long = vec![true, false, true, false];
        x.confirm_long = vec![false, true, false, true];
        x.long_stop.fill(90.0);
        x.close = vec![100.0, 101.0, 102.0, 102.0];
        if reason == "reverse_bi" {
            x.completed_bi_dir[3] = -1;
        }
        if reason == "structure_break" {
            x.close[3] = 89.0;
        }
        if reason == "trailing_stop" {
            x.close[3] = 99.0;
        }
        let out = evaluate_structure_factor(&x, &StructureFactorConfig::default()).unwrap();
        assert_eq!(out.exit_reason[3], reason);
        assert_eq!(out.strength[3], 0.0);
        assert!(!out.trigger[3]);
        let out = evaluate_structure_decay(
            &[0, 1, 0, 1],
            &[1.0; 4],
            &x.close,
            &[90.0; 4],
            &x.completed_bi_dir,
            &[1.0; 4],
            &[1.0; 4],
            &StructureFactorConfig::default(),
        )
        .unwrap();
        assert_eq!(out.strength[3], 0.0);
        assert!(!out.trigger[3]);
    }
}

#[test]
fn snapshot_asof_prevents_cross_window_anchors() {
    use czsc_core::structure_exit::{structure_snapshots, structure_stop_anchors};
    let bars = bars();
    let mut snapshots = structure_snapshots(&bars, 6, 50).unwrap();
    snapshots.swap(0, 1);
    assert!(
        structure_stop_anchors(&bars, &snapshots, "pos_fx_stop", None, 20)
            .unwrap_err()
            .contains("asof")
    );
}

#[test]
fn zero_configuration_uses_same_core_resolvers_as_signal_generation() {
    use czsc_core::analyze::{resolve_max_bi_num, resolve_min_bi_len};
    use czsc_core::structure_exit::structure_snapshots;
    let bars = bars();
    let automatic = structure_snapshots(&bars, 0, 0).unwrap();
    let explicit =
        structure_snapshots(&bars, resolve_min_bi_len(0), resolve_max_bi_num(0)).unwrap();
    for (a, b) in automatic.iter().zip(explicit) {
        assert_eq!(a.completed_bi_id, b.completed_bi_id);
        assert_eq!(a.completed_bi_dir, b.completed_bi_dir);
        assert_eq!(a.bi_low.to_bits(), b.bi_low.to_bits());
        assert_eq!(a.zs_mid.to_bits(), b.zs_mid.to_bits());
    }
}

#[test]
fn zs_anchor_uses_the_same_twenty_bi_window_as_reversion_setup() {
    use chrono::Duration;
    use czsc_core::analyze::{CZSC, utils::get_zs_seq};
    use czsc_core::structure_exit::structure_snapshots;
    let template = bars();
    let bars: Vec<_> = (0..640)
        .map(|i| {
            let mut b = template[i % template.len()].clone();
            b.id = i as i32;
            b.dt = template[0].dt + Duration::minutes(i as i64 * 30);
            let close = 100.0
                + (30.0 - 20.0 * i as f64 / 640.0) * (i as f64 * std::f64::consts::PI / 10.0).sin();
            b.open = close;
            b.close = close;
            b.high = close + 1.0;
            b.low = close - 1.0;
            b
        })
        .collect();
    let snapshots = structure_snapshots(&bars, 6, 50).unwrap();
    let c = CZSC::new(bars, 50, 6);
    let bis = c.get_finished_bis();
    assert!(bis.len() > 20);
    let zones = get_zs_seq(&bis[bis.len() - 20..]);
    let zs = zones.iter().rev().find(|z| z.is_valid()).unwrap();
    let actual = snapshots.last().unwrap();
    assert_eq!(actual.zs_low, zs.dd);
    assert_eq!(actual.zs_high, zs.gg);
    assert_eq!(actual.zs_mid, zs.zz);
}

#[test]
fn default_trailing_exits_exactly_at_one_point_three_percent() {
    for side in [1, -1] {
        let best = if side == 1 { 102.0 } else { 98.0 };
        let boundary = if side == 1 {
            best * (1.0 - 0.013)
        } else {
            best * (1.0 + 0.013)
        };
        let mut setup_long = vec![false; 5];
        let mut setup_short = vec![false; 5];
        let mut confirm_long = vec![false; 5];
        let mut confirm_short = vec![false; 5];
        if side == 1 {
            setup_long[0] = true;
            confirm_long[1] = true;
        } else {
            setup_short[0] = true;
            confirm_short[1] = true;
        }
        let x = StructureFactorInput {
            setup_long,
            setup_short,
            confirm_long,
            confirm_short,
            setup_quality_long: vec![1.0; 5],
            setup_quality_short: vec![1.0; 5],
            confirm_quality_long: vec![1.0; 5],
            confirm_quality_short: vec![1.0; 5],
            close: vec![
                100.0,
                100.0,
                best,
                boundary + f64::from(side) * 1e-8,
                boundary,
            ],
            long_stop: vec![90.0; 5],
            short_stop: vec![110.0; 5],
            completed_bi_dir: vec![0; 5],
            atr_fast: vec![1.0; 5],
            atr_slow: vec![1.0; 5],
        };
        let result = evaluate_structure_factor(&x, &StructureFactorConfig::default()).unwrap();
        assert!(result.trigger[1]);
        assert_eq!(result.strength[3].signum(), f64::from(side));
        assert_eq!(result.strength[4], 0.0);
        assert_eq!(result.exit_reason[4], "trailing_stop");
    }
}

#[test]
fn reverse_bi_exit_cannot_be_cancelled_by_an_armed_same_direction_setup() {
    let mut x = input();
    x.close = vec![100., 101., 102., 102.];
    x.setup_long = vec![true, false, true, false];
    x.confirm_long = vec![false, true, false, true];
    x.long_stop = vec![90.; 4];
    x.completed_bi_dir[3] = -1;
    let out = evaluate_structure_factor(&x, &StructureFactorConfig::default()).unwrap();
    assert_eq!(out.exit_reason[3], "reverse_bi");
    assert_eq!(out.strength[3], 0.0);
    assert!(!out.trigger[3]);
}

#[test]
fn secondary_extrema_and_zs_reentry_plugins_are_causal_and_symmetric() {
    use czsc_core::structure_exit::{structure_snapshots, structure_stop_anchors};
    let bars = bars();
    let all = structure_snapshots(&bars, 6, 50).unwrap();
    let prefix = structure_snapshots(&bars[..90], 6, 50).unwrap();
    for i in 0..prefix.len() {
        assert_eq!(
            all[i].bi_second_low.to_bits(),
            prefix[i].bi_second_low.to_bits()
        );
        assert_eq!(
            all[i].bi_second_high.to_bits(),
            prefix[i].bi_second_high.to_bits()
        );
        assert_eq!(all[i].zs_zd.to_bits(), prefix[i].zs_zd.to_bits());
        assert_eq!(all[i].zs_zg.to_bits(), prefix[i].zs_zg.to_bits());
        assert_eq!(all[i].completed_fx_dir, prefix[i].completed_fx_dir);
    }
    assert!(all.iter().any(|s| s.completed_fx_dir != 0));
    let sec = structure_stop_anchors(&bars, &all, "pos_second_ext", None, 20).unwrap();
    let zs_re = structure_stop_anchors(&bars, &all, "pos_zs_reentry", None, 20).unwrap();
    for (i, s) in all.iter().enumerate() {
        assert_eq!(sec.long_stop[i].to_bits(), s.bi_second_low.to_bits());
        assert_eq!(sec.short_stop[i].to_bits(), s.bi_second_high.to_bits());
        if s.zs_zg.is_finite() && bars[i].close > s.zs_zg {
            assert!(zs_re.long_stop[i] >= s.zs_zg);
        }
        if s.zs_zd.is_finite() && bars[i].close < s.zs_zd {
            assert!(zs_re.short_stop[i] <= s.zs_zd);
        }
    }
}

#[test]
fn ratchet_stop_and_zs_reentry_and_activation_trailing_and_track_signal() {
    use czsc_core::structure_exit::evaluate_structure_decay;
    // 1. Stop ratcheting tightens stop monotonically and exits when ratcheted secondary low breaks
    let mut x = input();
    x.close = vec![100.0, 102.0, 105.0, 100.5];
    x.confirm_long = vec![false, true, false, false];
    x.long_stop = vec![95.0, 95.0, 101.0, 101.0];
    let cfg_ratchet = StructureFactorConfig {
        ratchet_stop: true,
        trailing: 0.015,
        trail_activation: 0.05, // not yet activated at +2.9% gain
        ..Default::default()
    };
    let out = evaluate_structure_factor(&x, &cfg_ratchet).unwrap();
    assert_eq!(out.stop[1], 95.0);
    assert_eq!(out.stop[2], 101.0); // ratcheted up to secondary low 101.0
    assert_eq!(out.exit_reason[3], "structure_break");
    assert_eq!(out.strength[3], 0.0);

    // 2. ZS reentry break (3rd buy pulling back into ZG=103.0)
    let mut x_zs = x.clone();
    x_zs.close = vec![100.0, 102.0, 105.0, 102.8];
    x_zs.long_stop = vec![95.0; 4];
    let cfg_zs = StructureFactorConfig {
        trail_activation: 0.10,
        zs_zg: Some(vec![103.0; 4]),
        ..Default::default()
    };
    let out_zs = evaluate_structure_factor(&x_zs, &cfg_zs).unwrap();
    assert_eq!(out_zs.exit_reason[3], "zs_reentry_break");
    assert_eq!(out_zs.strength[3], 0.0);

    // 3. Opposite FX confirmation while trailing is activated
    let mut x_fx = x.clone();
    x_fx.close = vec![100.0, 100.0, 103.0, 102.4]; // +3% peak gain >= 2% activation; -0.58% retrace >= 0.35*1.3%
    x_fx.long_stop = vec![95.0; 4];
    let cfg_fx = StructureFactorConfig {
        trail_activation: 0.02,
        completed_fx_dir: Some(vec![0, 0, 0, -1]),
        ..Default::default()
    };
    let out_fx = evaluate_structure_factor(&x_fx, &cfg_fx).unwrap();
    assert_eq!(out_fx.exit_reason[3], "trailing_fx_exit");

    // 4. Continuous signal governance (track_signal=true) latches after risk exit until new structure event
    let cfg_track = StructureFactorConfig {
        lam0: 0.0,
        trailing: 0.013,
        trail_activation: 0.015,
        ratchet_stop: true,
        track_signal: true,
        completed_fx_dir: Some(vec![0, 0, 0, 0, 1]),
        ..Default::default()
    };
    let trk = evaluate_structure_decay(
        &[1, 1, 1, 1, 1],
        &[0.6, 0.8, 0.8, 0.8, 0.8],
        &[100.0, 102.0, 100.5, 101.0, 101.5],
        &[95.0, 98.0, 98.0, 98.0, 98.0],
        &[0, 0, 0, 0, 0],
        &[1.0; 5],
        &[1.0; 5],
        &cfg_track,
    )
    .unwrap();
    assert_eq!(trk.strength[0], 0.6);
    assert_eq!(trk.strength[1], 0.8);
    assert_eq!(trk.exit_reason[2], "trailing_stop");
    assert_eq!(trk.strength[2], 0.0);
    assert_eq!(trk.strength[3], 0.0); // latched flat on bar 3 despite upstream direction[3]==1
    assert_eq!(trk.strength[4], 0.8); // re-armed on bar 4 when completed_fx_dir[4]==1
}
