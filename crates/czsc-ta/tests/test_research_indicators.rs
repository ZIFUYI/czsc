//! Shared research/signal indicator contracts, independent of the Python binding.
use czsc_ta::pure::{macd, obv, rsi};

#[test]
fn rsi_wilder_seed_and_decay_are_not_a_rolling_mean() {
    let output = rsi(&[100.0, 102.0, 101.0, 104.0, 102.0, 105.0], 3);
    assert!(output[..3].iter().all(|x| x.is_nan()));
    for (actual, expected) in output[3..]
        .iter()
        .zip([250.0 / 3.0, 500.0 / 9.0, 4700.0 / 63.0])
    {
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn rsi_flat_up_down_empty_and_unready_contract() {
    assert!(rsi(&[], 14).is_empty());
    for n in [0, 4, 8] {
        assert!(rsi(&[1.0, 2.0, 3.0, 4.0], n).iter().all(|x| x.is_nan()));
    }
    assert_eq!(&rsi(&[1.0; 5], 3)[3..], &[0.0, 0.0]);
    assert_eq!(&rsi(&[1.0, 2.0, 3.0, 4.0, 5.0], 3)[3..], &[100.0, 100.0]);
    assert_eq!(&rsi(&[5.0, 4.0, 3.0, 2.0, 1.0], 3)[3..], &[0.0, 0.0]);
}

#[test]
fn macd_uses_sma_seed_full_warmup_and_undoubled_histogram() {
    let (dif, dea, hist) = macd(&[1.0, 2.0, 1.0, 4.0, 2.0, 5.0], 2, 3, 2);
    for values in [&dif, &dea, &hist] {
        assert_eq!(values.len(), 6);
        assert!(values[..3].iter().all(|x| x.is_nan()));
    }
    assert!((dif[3] - 0.5).abs() < 1e-12);
    assert!((dea[3] - 1.0 / 3.0).abs() < 1e-12);
    assert!((hist[3] - 1.0 / 6.0).abs() < 1e-12);
    for i in 3..6 {
        assert_eq!(hist[i], dif[i] - dea[i]);
    }
}

#[test]
fn macd_empty_and_invalid_periods_preserve_length() {
    assert_eq!(macd(&[], 12, 26, 9), (vec![], vec![], vec![]));
    for periods in [(0, 3, 2), (2, 0, 2), (2, 3, 0), (12, 26, 9)] {
        let (dif, dea, hist) = macd(&[1.0, 2.0, 3.0], periods.0, periods.1, periods.2);
        for values in [dif, dea, hist] {
            assert_eq!(values.len(), 3);
            assert!(values.iter().all(|x| x.is_nan()));
        }
    }
}

#[test]
fn obv_is_close_to_close_with_initial_volume_and_unchanged_close_zero_flow() {
    assert_eq!(
        obv(&[10.0, 11.0, 11.0, 9.0], &[2.0, 3.0, 5.0, 7.0]).unwrap(),
        vec![2.0, 5.0, 5.0, -2.0]
    );
    assert_eq!(obv(&[], &[]).unwrap(), Vec::<f64>::new());
    assert_eq!(
        obv(&[10.0], &[]).unwrap_err(),
        "close and volume must have the same length"
    );
}

#[test]
fn indicator_future_changes_cannot_revise_prefix() {
    let close: Vec<_> = (0..80).map(|i| 100.0 + (i as f64 * 0.7).sin()).collect();
    let volume = vec![10.0; close.len()];
    for n in [14, 40] {
        let r = rsi(&close, 14);
        let prefix = rsi(&close[..n], 14);
        for (a, b) in r[..n].iter().zip(prefix) {
            assert!(a == &b || a.is_nan() && b.is_nan());
        }
        let full = macd(&close, 12, 26, 9);
        let prefix = macd(&close[..n], 12, 26, 9);
        for (a, b) in [full.0, full.1, full.2]
            .iter()
            .zip([prefix.0, prefix.1, prefix.2])
        {
            for (a, b) in a[..n].iter().zip(b) {
                assert!(a == &b || a.is_nan() && b.is_nan());
            }
        }
        assert_eq!(
            obv(&close[..n], &volume[..n]).unwrap(),
            obv(&close, &volume).unwrap()[..n]
        );
    }
}

#[test]
fn obv_nonfinite_inputs_preserve_prefix_and_invalidate_tail() {
    for (close, volume) in [
        (vec![10.0, 11.0, f64::NAN, 9.0], vec![2.0, 3.0, 5.0, 7.0]),
        (
            vec![10.0, 11.0, 11.0, 9.0],
            vec![2.0, 3.0, f64::INFINITY, 7.0],
        ),
    ] {
        let actual = obv(&close, &volume).unwrap();
        assert_eq!(&actual[..2], &[2.0, 5.0]);
        assert!(actual[2..].iter().all(|x| x.is_nan()));
    }
}
