use czsc_core::research_execution::{
    closed_execution_intervals, decision_execution_ledger, decision_interval_ledger,
    executed_weight_ledger, execution_episodes, execution_ledger, execution_net_returns,
    standardize_execution_returns,
};

fn assert_near(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len());
    for (a, b) in left.iter().zip(right) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }
}

#[test]
fn fixed_capital_quantity_fills_match_independent_cash_ledger() {
    let prices = [100., 110., 88., 96., 108., 105.];
    let weights = [0., 0.5, 0.5, 0.25, -0.75, 0.];
    let capital = 100_000.;
    let fee = 0.001;
    let actual = execution_ledger(&prices, &weights, fee, capital, None).unwrap();
    let quantities = actual.quantities.as_ref().unwrap();
    let trades = actual.trade_quantities.as_ref().unwrap();
    let mut previous_qty = 0.;
    for i in 0..prices.len() {
        let q = weights[i] * capital / prices[i];
        let cash_pnl = if i > 0 {
            previous_qty * (prices[i] - prices[i - 1])
        } else {
            0.
        };
        let cash_cost = (q - previous_qty).abs() * prices[i] * fee;
        assert_near(&[actual.gross_returns[i]], &[cash_pnl / capital]);
        assert_near(&[actual.cost_returns[i]], &[cash_cost / capital]);
        assert_near(
            &[actual.net_returns[i]],
            &[(cash_pnl - cash_cost) / capital],
        );
        assert_near(&[quantities[i], trades[i]], &[q, q - previous_qty]);
        previous_qty = q;
    }
    let twice = execution_ledger(&prices, &weights, fee, capital * 2., None).unwrap();
    assert_eq!(actual.net_returns, twice.net_returns);
    for (one, two) in quantities.iter().zip(twice.quantities.unwrap()) {
        assert_eq!(one * 2., two);
    }
}

#[test]
fn unchanged_weight_charges_drift_but_unchanged_price_does_not() {
    for side in [-1., 1.] {
        let actual =
            execution_ledger(&[100., 100., 110., 110.], &[0.5 * side; 4], 0.001, 1., None).unwrap();
        assert_near(&actual.turnover, &[0.5, 0., 0.05, 0.]);
        assert_near(&actual.gross_returns, &[0., 0., 0.05 * side, 0.]);
        assert_near(&actual.cost_returns, &[0.0005, 0., 0.00005, 0.]);
    }
}

#[test]
fn resizing_can_cost_less_than_absolute_weight_change() {
    let actual = execution_ledger(&[100., 80.], &[0.5, 0.3], 0.001, 1., None).unwrap();
    assert_near(&actual.turnover, &[0.5, 0.1]);
    assert_near(&actual.cost_returns, &[0.0005, 0.0001]);
    assert!(actual.cost_returns[1] < (0.5_f64 - 0.3).abs() * 0.001);
}

#[test]
fn missing_quotes_freeze_quantity_then_resume_across_price_gap() {
    let prices = [f64::NAN, 100., f64::NAN, 120., 120.];
    let weights = [0., 0.5, 0.5, 0.5, 0.];
    let observed = [false, true, false, true, true];
    let actual = execution_ledger(&prices, &weights, 0.001, 100., Some(&observed)).unwrap();
    assert_near(
        actual.quantities.as_ref().unwrap(),
        &[0., 0.5, 0.5, 5. / 12., 0.],
    );
    assert_near(&actual.gross_returns, &[0., 0., 0., 0.1, 0.]);
    assert_near(&actual.turnover, &[0., 0.5, 0., 0.1, 0.5]);
    let by_returns = executed_weight_ledger(
        &[0., 0., 0., 0.2, 0.],
        &weights,
        0.001,
        Some(&observed),
        None,
    )
    .unwrap();
    assert_near(&actual.net_returns, &by_returns.net_returns);
    assert!(by_returns.quantities.is_none());
    assert!(by_returns.trade_quantities.is_none());
    for (returns, target) in [([0., 0.1], [0.5, 0.5]), ([0., 0.], [0.5, 0.25])] {
        assert!(
            executed_weight_ledger(&returns, &target, 0.001, Some(&[true, false]), None)
                .unwrap_err()
                .contains("missing quotes")
        );
    }
    assert!(execution_ledger(&[f64::NAN], &[0.5], 0.001, 1., Some(&[false])).is_err());
    assert!(execution_ledger(&[100., 120.], &[0.5, 0.5], 0.001, 1., Some(&[true, false])).is_err());
}

#[test]
fn reverse_episodes_split_actual_exit_and_entry_quantity_costs() {
    let prices = [100., 100., 120., 90.];
    let weights = [0.5, -0.25, 0., 0.];
    let episodes = execution_episodes(
        &prices, &prices, &prices, &prices, &[1.; 4], &weights, 0.001,
    )
    .unwrap();
    assert_eq!(episodes.len(), 2);
    // Long: 0.5 units of capital opened at 100 and closed at 120.
    assert_near(&[episodes[0].weighted_net_ret], &[0.1 - 0.0005 - 0.0006]);
    // Short: 0.25 units opened at 120 and closed at 90.
    assert_near(
        &[episodes[1].weighted_net_ret],
        &[0.0625 - 0.00025 - 0.0001875],
    );
    let ledger = execution_net_returns(&prices, &weights, 0.001).unwrap();
    assert_near(
        &[ledger.iter().sum()],
        &[episodes.iter().map(|e| e.weighted_net_ret).sum()],
    );
}

#[test]
fn terminal_liquidation_is_explicit_and_interval_attribution_conserves_flows() {
    let open = [100., 110., 99., 120.];
    let weights = [0.5, 0.25, -0.75, 0.9];
    let retained = decision_execution_ledger(&open, &weights, 0.001, None).unwrap();
    assert_eq!(retained.executed_weights, [0., 0.5, 0.25, -0.75]);
    assert!(closed_execution_intervals(&retained).is_err());
    let closed = decision_execution_ledger(&open, &weights, 0.001, Some(118.)).unwrap();
    let intervals = decision_interval_ledger(
        &[0., 99. / 110. - 1., 120. / 99. - 1., 118. / 120. - 1.],
        &weights,
        0.001,
    )
    .unwrap();
    assert_eq!(closed.gross_returns, intervals.gross_returns);
    assert_eq!(closed.cost_returns, intervals.cost_returns);
    let forward = closed_execution_intervals(&closed).unwrap();
    assert_eq!(forward.net_returns.len(), weights.len());
    assert_near(
        &[forward.net_returns.iter().sum()],
        &[closed.net_returns.iter().sum()],
    );
    assert_near(
        &[forward.turnover.iter().sum()],
        &[closed.turnover.iter().sum()],
    );
    assert_near(&[closed.turnover[4]], &[0.75 * 118. / 120.]);
    // The last close-time decision never creates a same-close open position.
    assert_eq!(closed.executed_weights, [0., 0.5, 0.25, -0.75, 0.]);
}

#[test]
fn return_ledger_has_no_cumulative_synthetic_price_overflow() {
    let returns = vec![1.; 2048];
    let weights = vec![0.5; 2048];
    let ledger = executed_weight_ledger(&returns, &weights, 0.001, None, None).unwrap();
    assert_eq!(ledger.gross_returns[2047], 0.5);
    assert_eq!(ledger.turnover[2047], 0.5);
    assert_eq!(ledger.net_returns[2047], 0.4995);
}

#[test]
fn shared_input_validation_and_external_gross_standardization() {
    assert_near(
        &standardize_execution_returns(&[0.1, -0.2], &[0.01, 0.02]).unwrap(),
        &[0.09, -0.22],
    );
    assert!(standardize_execution_returns(&[0.], &[]).is_err());
    assert!(standardize_execution_returns(&[0.], &[-0.1]).is_err());
    assert!(standardize_execution_returns(&[f64::NAN], &[0.]).is_err());
    for capital in [0., -1., f64::INFINITY, f64::NAN] {
        assert!(execution_ledger(&[100.], &[0.], 0.001, capital, None).is_err());
    }
    for ret in [-1., -2., f64::INFINITY, f64::NAN] {
        assert!(executed_weight_ledger(&[ret], &[0.], 0.001, None, None).is_err());
    }
    assert!(executed_weight_ledger(&[0.], &[0.], 0.001, Some(&[]), None).is_err());
    assert!(executed_weight_ledger(&[0.], &[0.], 0.001, None, Some(&[])).is_err());
    assert!(decision_interval_ledger(&[0.1], &[0.], 0.001).is_err());
    assert!(decision_execution_ledger(&[], &[], 0.001, Some(100.)).is_err());
    assert!(
        execution_ledger(&[], &[], 0.001, 1., None)
            .unwrap()
            .net_returns
            .is_empty()
    );
}

#[test]
fn contract_roll_mask_strips_gap_return_and_charges_two_sided_rollover_friction() {
    let returns = [0., 0., 0.15, 0.02, 0.];
    let weights = [0., 0.5, 0.5, 0.5, 0.];
    let roll_mask = [false, false, true, false, false];
    let ledger = executed_weight_ledger(&returns, &weights, 0.001, None, Some(&roll_mask)).unwrap();
    assert_near(&ledger.gross_returns, &[0., 0., 0., 0.01, 0.]);
    assert_near(&ledger.turnover, &[0., 0.5, 1.0, 0.01, 0.5]);
    assert_near(&ledger.cost_returns, &[0., 0.0005, 0.001, 0.00001, 0.0005]);
    assert_near(
        &ledger.net_returns,
        &[0., -0.0005, -0.001, 0.00999, -0.0005],
    );
}

#[test]
fn general_execution_allows_leverage_while_decision_api_preserves_bounds() {
    let prices = [100., 120., 110., 108.];
    let weights = [0.5, 0.25, -0.75, 0.];
    let leveraged: Vec<_> = weights.iter().map(|w| w * 4.).collect();
    let base = execution_ledger(&prices, &weights, 0.001, 1., None).unwrap();
    let scaled = execution_ledger(&prices, &leveraged, 0.001, 1., None).unwrap();
    assert_near(
        &scaled.net_returns,
        &base.net_returns.iter().map(|x| x * 4.).collect::<Vec<_>>(),
    );
    assert_near(
        &scaled.turnover,
        &base.turnover.iter().map(|x| x * 4.).collect::<Vec<_>>(),
    );
    assert!(execution_net_returns(&prices, &leveraged, 0.001).is_err());
    assert!(decision_interval_ledger(&[0.; 4], &leveraged, 0.001).is_err());
}
