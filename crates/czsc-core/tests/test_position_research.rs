use chrono::{Duration, TimeZone, Utc};
use czsc_core::objects::bar::RawBarBuilder;
use czsc_core::objects::position::{LiteBar, Position};
use czsc_core::objects::signal::Signal;
use czsc_core::objects::{bar::RawBar, freq::Freq};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
    str::FromStr,
};

fn position(t0: bool) -> Position {
    let events = [("开多", "多"), ("开空", "空"), ("平多", "平多"), ("平空", "平空")].map(|(op, value)| serde_json::json!({
        "name": op, "operate": op, "signals_all": [format!("30分钟_test_state_{value}_任意_任意_0")],
        "signals_any": [], "signals_not": [], "factors": []
    }));
    let mut p: Position = serde_json::from_value(serde_json::json!({
        "name":"research", "symbol":"TEST", "opens":events[..2], "exits":events[2..],
        "interval":0, "timeout":100, "stop_loss":0., "T0":t0
    }))
    .unwrap();
    p.normalize_runtime_fields();
    p
}
fn inputs() -> (Vec<RawBar>, Vec<HashMap<String, String>>) {
    let bars = (0i32..6)
        .map(|i| {
            RawBarBuilder::default()
                .symbol("TEST")
                .id(i)
                .dt(Utc.with_ymd_and_hms(2022, 1, 4, 9, 30, 0).unwrap()
                    + Duration::minutes(i as i64 * 30))
                .freq(Freq::F30)
                .open(100.)
                .close(100.)
                .high(101.)
                .low(99.)
                .vol(1000.)
                .amount(100000.)
                .build()
                .unwrap()
        })
        .collect();
    let rows = ["多", "其他", "平多", "空", "其他", "平空"]
        .iter()
        .map(|v| HashMap::from([("cz_test_state".into(), v.to_string())]))
        .collect();
    (bars, rows)
}
#[test]
fn batch_matches_stepwise_and_keeps_template() {
    let (bars, rows) = inputs();
    let p = position(true);
    let (triggers, weights) = p.evaluate_signals(&bars, &rows).unwrap();
    assert_eq!(triggers, [true, false, false, true, false, false]);
    assert_eq!(weights, [1., 1., 0., -1., -1., 0.]);
    let mut step = position(true);
    for (i, (b, r)) in bars.iter().zip(&rows).enumerate() {
        let signal = Signal::from_str(&format!(
            "30分钟_test_state_{}_任意_任意_0",
            r["cz_test_state"]
        ))
        .unwrap();
        step.update(
            LiteBar {
                id: b.id,
                dt: b.dt.fixed_offset(),
                price: b.close,
            },
            Rc::new(RefCell::new(HashSet::from([signal]))),
        );
        assert_eq!(step.get_pos().to_f64(), weights[i]);
    }
    assert_eq!(p.get_pos().to_f64(), 0.);
    assert_eq!(
        p.evaluate_signals(&bars, &rows).unwrap(),
        (triggers, weights)
    );
}
#[test]
fn batch_preserves_t1_and_rejects_invalid_inputs() {
    let (bars, rows) = inputs();
    assert_eq!(
        position(false).evaluate_signals(&bars, &rows).unwrap().1[2],
        1.
    );
    assert!(
        position(true)
            .evaluate_signals(&bars, &rows[..5])
            .unwrap_err()
            .contains("length")
    );
    let mut reversed = bars.clone();
    reversed.reverse();
    assert!(
        position(true)
            .evaluate_signals(&reversed, &rows)
            .unwrap_err()
            .contains("ordered")
    );
    assert_eq!(
        position(true).evaluate_signals(&[], &[]).unwrap(),
        (vec![], vec![])
    );
}

#[test]
fn batch_rejects_a_different_symbol_than_its_position_template() {
    let (bars, rows) = inputs();
    let mut p = position(true);
    p.symbol = "OTHER".into();
    assert!(
        p.evaluate_signals(&bars, &rows)
            .unwrap_err()
            .contains("Position symbol")
    );
}

#[test]
fn batch_full_and_research_keys_share_values_and_reject_ambiguity() {
    let (bars, mut rows) = inputs();
    let p = position(true);
    let expected = p.evaluate_signals(&bars, &rows).unwrap();
    let full: Vec<HashMap<String, String>> = rows
        .iter()
        .map(|r| {
            HashMap::from([(
                "30分钟_test_state".into(),
                format!("{}_任意_任意_0", r["cz_test_state"]),
            )])
        })
        .collect();
    assert_eq!(p.evaluate_signals(&bars, &full).unwrap(), expected);
    rows[0].insert("30分钟_test_state".into(), "空_任意_任意_0".into());
    assert!(
        p.evaluate_signals(&bars, &rows)
            .unwrap_err()
            .contains("duplicate signal key")
    );
    rows[0].clear();
    assert!(
        p.evaluate_signals(&bars, &rows)
            .unwrap_err()
            .contains("missing required signal")
    );
    assert_eq!(p.get_pos().to_f64(), 0.);
}

#[test]
fn batch_uses_local_bar_sequence_for_timeout_with_dataframe_default_ids() {
    let (mut bars, mut rows) = inputs();
    for bar in &mut bars {
        bar.id = 0;
    }
    for row in &mut rows[1..] {
        row.insert("cz_test_state".into(), "其他".into());
    }
    let mut p = position(true);
    p.timeout = 2;
    assert_eq!(
        p.evaluate_signals(&bars, &rows).unwrap().1,
        [1., 1., 1., 0., 0., 0.]
    );
}
