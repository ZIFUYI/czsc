//! Inclusion must preserve a doji and defer direction for ambiguous initial bars.

use chrono::{TimeZone, Utc};
use czsc_core::analyze::utils::remove_include;
use czsc_core::objects::bar::{NewBar, RawBar, RawBarBuilder};
use czsc_core::objects::freq::Freq;

fn bar(i: i32, high: f64, low: f64, open: f64, close: f64) -> RawBar {
    RawBarBuilder::default()
        .symbol("TEST")
        .dt(Utc.timestamp_opt(i64::from(i) * 1800, 0).unwrap())
        .freq(Freq::F30)
        .id(i)
        .open(open)
        .close(close)
        .high(high)
        .low(low)
        .vol(100.0)
        .amount(1000.0)
        .build()
        .unwrap()
}

fn mirrored(b: &RawBar) -> RawBar {
    bar(
        b.id,
        300.0 - b.low,
        300.0 - b.high,
        300.0 - b.open,
        300.0 - b.close,
    )
}

fn merge(bars: &[RawBar; 3]) -> (bool, NewBar) {
    remove_include(
        &NewBar::new_from_raw(&bars[0]),
        &NewBar::new_from_raw(&bars[1]),
        bars[2].clone(),
    )
    .unwrap()
}

fn assert_mirror(a: &NewBar, b: &NewBar) {
    assert_eq!(a.high, 300.0 - b.low);
    assert_eq!(a.low, 300.0 - b.high);
    assert_eq!(a.open, 300.0 - b.open);
    assert_eq!(a.close, 300.0 - b.close);
    assert_eq!(a.dt, b.dt);
    assert_eq!(a.vol, b.vol);
    assert_eq!(a.elements.len(), b.elements.len());
}

#[test]
fn inclusion_keeps_doji_neutral_and_clamps_only_to_merged_range() {
    for close in [96.0, 110.0, 129.0] {
        let bars = [
            bar(1, 110., 90., 95., 105.),
            bar(2, 120., 100., 105., 115.),
            bar(3, 130., 95., close, close),
        ];
        let mirrored_bars = bars.each_ref().map(mirrored);
        let (included, a) = merge(&bars);
        let (included_mirror, b) = merge(&mirrored_bars);
        assert!(included && included_mirror);
        assert_eq!((a.low, a.high), (100., 130.));
        assert_eq!(a.open, close.clamp(100., 130.));
        assert_eq!(a.close, a.open);
        assert_eq!(b.close, b.open);
        assert_mirror(&a, &b);
    }
}

#[test]
fn unknown_initial_direction_does_not_invent_an_inclusion_trend() {
    // Inner/outer bars, equal highs, equal lows, and identical ranges are all
    // ambiguous. Swapping high/low with a mirror must not invent a direction.
    for (h1, l1, h2, l2) in [
        (130., 90., 120., 100.),
        (120., 100., 130., 90.),
        (120., 90., 120., 100.),
        (120., 90., 130., 90.),
        (120., 90., 120., 90.),
    ] {
        let bars = [
            bar(1, h1, l1, 105., 110.),
            bar(2, h2, l2, 105., 110.),
            bar(3, 115., 105., 108., 112.),
        ];
        let (included, a) = merge(&bars);
        let (included_mirror, b) = merge(&bars.each_ref().map(mirrored));
        assert!(!included && !included_mirror);
        assert_eq!((a.open, a.high, a.low, a.close), (108., 115., 105., 112.));
        assert_eq!(a.elements.len(), 1);
        assert_mirror(&a, &b);
    }
}

#[test]
fn known_direction_keeps_existing_bullish_and_bearish_merge_rules() {
    for (open, close, expected_open, expected_close) in
        [(108., 112., 105., 120.), (112., 108., 120., 105.)]
    {
        let bars = [
            bar(1, 110., 90., 95., 105.),
            bar(2, 120., 100., 105., 115.),
            bar(3, 115., 105., open, close),
        ];
        let (included, a) = merge(&bars);
        let (included_mirror, b) = merge(&bars.each_ref().map(mirrored));
        assert!(included && included_mirror);
        assert_eq!((a.open, a.close), (expected_open, expected_close));
        assert_mirror(&a, &b);
    }
}
