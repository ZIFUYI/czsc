use czsc_core::analyze::utils::format_research_kline;
use czsc_core::objects::freq::Freq;
use polars::prelude::*;

fn frame() -> DataFrame {
    df! {
        "symbol" => ["TEST", "TEST"],
        "dt" => [1_700_000_000_000i64, 1_700_001_800_000],
        "open" => [100i64, 101], "high" => [102i64, 103],
        "low" => [99i64, 100], "close" => [101i64, 102],
        "volume" => [10i64, 20],
    }
    .unwrap()
    .lazy()
    .with_column(col("dt").cast(DataType::Datetime(TimeUnit::Milliseconds, None)))
    .collect()
    .unwrap()
}

#[test]
fn missing_amount_integer_prices_and_volume_alias_share_native_boundary() {
    let df = frame();
    let bars = format_research_kline(df.clone(), Freq::F30).unwrap();
    assert_eq!(bars.iter().map(|b| b.id).collect::<Vec<_>>(), [0, 1]);
    assert_eq!(bars.iter().map(|b| b.amount).collect::<Vec<_>>(), [0., 0.]);
    assert_eq!(bars.iter().map(|b| b.vol).collect::<Vec<_>>(), [10., 20.]);
    assert_eq!(bars[0].open, 100.);
    assert_eq!(bars[1].close, 102.);
    assert!(df.column("amount").is_err());
    assert!(df.column("vol").is_err());
    assert_eq!(df.column("open").unwrap().dtype(), &DataType::Int64);
}

#[test]
fn existing_amount_is_preserved_and_alias_conflicts_are_rejected() {
    let mut df = frame();
    df.with_column(Column::new("amount".into(), [1001i64, 2002]))
        .unwrap();
    df.with_column(Column::new("vol".into(), [10f64, 20.]))
        .unwrap();
    let bars = format_research_kline(df.clone(), Freq::F30).unwrap();
    assert_eq!(
        bars.iter().map(|b| b.amount).collect::<Vec<_>>(),
        [1001., 2002.]
    );
    df.with_column(Column::new("vol".into(), [10f64, 21.]))
        .unwrap();
    assert!(
        format_research_kline(df, Freq::F30)
            .unwrap_err()
            .to_string()
            .contains("volume and vol conflict")
    );
}

#[test]
fn null_required_values_and_invalid_ohlc_fail_before_standard_converter() {
    let mut df = frame();
    df.with_column(Column::new("close".into(), [Some(101.), None]))
        .unwrap();
    assert!(
        format_research_kline(df, Freq::F30)
            .unwrap_err()
            .to_string()
            .contains("null")
    );
    let mut df = frame();
    df.with_column(Column::new("close".into(), [101., 104.]))
        .unwrap();
    assert!(
        format_research_kline(df, Freq::F30)
            .unwrap_err()
            .to_string()
            .contains("OHLC")
    );
    assert!(
        format_research_kline(frame().head(Some(0)), Freq::F30)
            .unwrap()
            .is_empty()
    );
    // Arrow represents an empty pandas object/string column as Null.
    let mut empty = frame().head(Some(0));
    empty
        .with_column(Column::full_null("symbol".into(), 0, &DataType::Null))
        .unwrap();
    assert!(format_research_kline(empty, Freq::F30).unwrap().is_empty());
}
