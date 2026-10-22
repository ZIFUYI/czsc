//! Shared fixed-initial-capital, quantity-based execution accounting and research episodes.
//! Targets rebalance at executable marks; fees charge traded notional after price drift.
//! Intraday extrema on the exit bar are excluded; unfinished episodes are not observations.
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction};

#[cfg_attr(feature = "python", gen_stub_pyclass)]
#[cfg_attr(
    feature = "python",
    pyclass(get_all, from_py_object, module = "czsc._native")
)]
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionEpisode {
    pub trigger_idx: usize,
    pub entry_idx: usize,
    pub exit_idx: usize,
    pub direction: i8,
    pub net_ret: f64,
    pub weighted_net_ret: f64,
    pub quality_w: f64,
    pub atr_pct: f64,
    pub mfe_atr: f64,
    pub mae_atr: f64,
}

/// Fixed-initial-capital execution accounting. Monetary flows are divided by the
/// initial capital; quantities are available only when actual prices are supplied.
#[cfg_attr(feature = "python", gen_stub_pyclass)]
#[cfg_attr(
    feature = "python",
    pyclass(get_all, from_py_object, module = "czsc._native")
)]
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionLedger {
    pub gross_returns: Vec<f64>,
    pub turnover: Vec<f64>,
    pub cost_returns: Vec<f64>,
    pub net_returns: Vec<f64>,
    pub pretrade_weights: Vec<f64>,
    pub executed_weights: Vec<f64>,
    pub quantities: Option<Vec<f64>>,
    pub trade_quantities: Option<Vec<f64>>,
}

/// Completed forward intervals with entry costs attributed to the start mark
/// and explicit terminal liquidation costs attributed to the last interval.
#[cfg_attr(feature = "python", gen_stub_pyclass)]
#[cfg_attr(
    feature = "python",
    pyclass(get_all, from_py_object, module = "czsc._native")
)]
#[derive(Clone, Debug, PartialEq)]
pub struct ClosedExecutionIntervals {
    pub net_returns: Vec<f64>,
    pub turnover: Vec<f64>,
}

/// Convert a closed boundary ledger to forward interval attribution.
pub fn closed_execution_intervals(
    ledger: &ExecutionLedger,
) -> Result<ClosedExecutionIntervals, String> {
    let n = ledger.net_returns.len();
    if n < 2 || ledger.executed_weights.last() != Some(&0.) {
        return Err(
            "closed intervals require at least two marks and zero terminal quantity".into(),
        );
    }
    if [
        ledger.gross_returns.len(),
        ledger.cost_returns.len(),
        ledger.turnover.len(),
        ledger.executed_weights.len(),
    ]
    .iter()
    .any(|m| *m != n)
    {
        return Err("execution input lengths must match".into());
    }
    let mut costs = ledger.cost_returns[..n - 1].to_vec();
    costs[n - 2] += ledger.cost_returns[n - 1];
    let net_returns = standardize_execution_returns(&ledger.gross_returns[1..], &costs)?;
    let mut turnover = ledger.turnover[..n - 1].to_vec();
    turnover[n - 2] += ledger.turnover[n - 1];
    Ok(ClosedExecutionIntervals {
        net_returns,
        turnover,
    })
}

fn validate_execution_inputs(
    n: usize,
    weights: &[f64],
    fee_rate: f64,
    tradable: Option<&[bool]>,
) -> Result<(), String> {
    if weights.len() != n || tradable.is_some_and(|mask| mask.len() != n) {
        return Err("execution input lengths must match".into());
    }
    if !fee_rate.is_finite() || fee_rate < 0. {
        return Err("fee_rate must be finite and nonnegative".into());
    }
    if weights.iter().any(|w| !w.is_finite()) {
        return Err("weights must be finite".into());
    }
    Ok(())
}

/// Subtract an independently calculated gross ledger's standardized execution costs.
/// Both inputs use the same fixed initial-capital denominator.
pub fn standardize_execution_returns(
    gross_returns: &[f64],
    cost_returns: &[f64],
) -> Result<Vec<f64>, String> {
    if gross_returns.len() != cost_returns.len() {
        return Err("execution input lengths must match".into());
    }
    if gross_returns.iter().any(|x| !x.is_finite())
        || cost_returns.iter().any(|x| !x.is_finite() || *x < 0.)
    {
        return Err(
            "gross returns must be finite; cost returns must be finite and nonnegative".into(),
        );
    }
    let net: Vec<_> = gross_returns
        .iter()
        .zip(cost_returns)
        .map(|(g, c)| g - c)
        .collect();
    if net.iter().any(|x| !x.is_finite()) {
        return Err("execution accounting overflow".into());
    }
    Ok(net)
}

/// Account targets already executed on the supplied price-return clock.
///
/// Every tradable mark rebalances to `weight * initial_capital / price`, even if
/// the target is unchanged. Fees charge actual traded notional, including price
/// drift. A false `tradable` element denotes a missing quote: return must be zero
/// and the target must remain unchanged, so quantity is frozen. The next observed
/// return spans the missing interval. Returns and turnover use fixed initial
/// capital, not current NAV. This entry point never reconstructs synthetic prices
/// or quantities and never inserts liquidation at either boundary.
pub fn executed_weight_ledger(
    price_returns: &[f64],
    weights: &[f64],
    fee_rate: f64,
    tradable: Option<&[bool]>,
    roll_mask: Option<&[bool]>,
) -> Result<ExecutionLedger, String> {
    let n = price_returns.len();
    validate_execution_inputs(n, weights, fee_rate, tradable)?;
    if roll_mask.is_some_and(|mask| mask.len() != n) {
        return Err("execution input lengths must match".into());
    }
    if price_returns
        .iter()
        .enumerate()
        .any(|(t, r)| !r.is_finite() || (!roll_mask.is_some_and(|m| m[t]) && *r <= -1.))
    {
        return Err("price_returns must be finite and greater than -1".into());
    }
    let mut ledger = ExecutionLedger {
        gross_returns: Vec::with_capacity(n),
        turnover: Vec::with_capacity(n),
        cost_returns: Vec::with_capacity(n),
        net_returns: Vec::with_capacity(n),
        pretrade_weights: Vec::with_capacity(n),
        executed_weights: Vec::with_capacity(n),
        quantities: None,
        trade_quantities: None,
    };
    let mut previous = 0.;
    for (t, (&price_return, &target)) in price_returns.iter().zip(weights).enumerate() {
        let can_trade = tradable.is_none_or(|mask| mask[t]);
        let is_roll = roll_mask.is_some_and(|mask| mask[t]);
        let eff_return = if is_roll { 0. } else { price_return };
        if !can_trade && (eff_return != 0. || target != previous) {
            return Err("missing quotes require zero return and unchanged executed weight".into());
        }
        let gross = previous * eff_return;
        let before = previous + gross;
        let roll_turnover = if can_trade && is_roll {
            2. * previous.abs()
        } else {
            0.
        };
        let turnover = if can_trade {
            (target - before).abs() + roll_turnover
        } else {
            0.
        };
        let cost = turnover * fee_rate;
        if [gross, before, turnover, cost]
            .iter()
            .any(|x| !x.is_finite())
        {
            return Err("execution accounting overflow".into());
        }
        ledger.gross_returns.push(gross);
        ledger.pretrade_weights.push(before);
        ledger.executed_weights.push(target);
        ledger.turnover.push(turnover);
        ledger.cost_returns.push(cost);
        previous = target;
    }
    ledger.net_returns =
        standardize_execution_returns(&ledger.gross_returns, &ledger.cost_returns)?;
    Ok(ledger)
}

/// Account already-executed targets using actual positive mark prices.
///
/// Target quantity is `weight * initial_capital / price`, with unit contract
/// multiplier and a fixed initial-capital basis. A false `tradable` mask denotes
/// a missing quote, allowing NaN or an unchanged carried mark. Quantity and target
/// must remain unchanged. The terminal quantity is retained unless the final
/// executable target explicitly equals zero.
pub fn execution_ledger(
    prices: &[f64],
    weights: &[f64],
    fee_rate: f64,
    initial_capital: f64,
    tradable: Option<&[bool]>,
) -> Result<ExecutionLedger, String> {
    validate_execution_inputs(prices.len(), weights, fee_rate, tradable)?;
    if !initial_capital.is_finite() || initial_capital <= 0. {
        return Err("initial_capital must be finite and positive".into());
    }
    let mut previous_price: Option<f64> = None;
    let mut returns = Vec::with_capacity(prices.len());
    let mut quantities = Vec::with_capacity(prices.len());
    let mut trade_quantities = Vec::with_capacity(prices.len());
    let mut previous_quantity = 0.;
    for (t, &price) in prices.iter().enumerate() {
        let can_trade = tradable.is_none_or(|mask| mask[t]);
        if price.is_nan() && !can_trade {
            returns.push(0.);
            quantities.push(previous_quantity);
            trade_quantities.push(0.);
            continue;
        }
        if !price.is_finite() || price <= 0. {
            return Err("prices must be finite and positive at observed marks".into());
        }
        returns.push(previous_price.map_or(0., |previous| price / previous - 1.));
        let quantity = if can_trade {
            weights[t] * initial_capital / price
        } else {
            previous_quantity
        };
        let trade_quantity = quantity - previous_quantity;
        if !quantity.is_finite() || !trade_quantity.is_finite() {
            return Err("execution accounting overflow".into());
        }
        quantities.push(quantity);
        trade_quantities.push(trade_quantity);
        previous_quantity = quantity;
        previous_price = Some(price);
    }
    let mut ledger = executed_weight_ledger(&returns, weights, fee_rate, tradable, None)?;
    ledger.quantities = Some(quantities);
    ledger.trade_quantities = Some(trade_quantities);
    Ok(ledger)
}

/// Close-time decisions filled at subsequent opens, with optional explicit final
/// close liquidation. A terminal close adds one observable execution mark.
pub fn decision_execution_ledger(
    open: &[f64],
    weights: &[f64],
    fee_rate: f64,
    terminal_close: Option<f64>,
) -> Result<ExecutionLedger, String> {
    validate_execution_inputs(open.len(), weights, fee_rate, None)?;
    if weights.iter().any(|w| w.abs() > 1.) {
        return Err("decision weights must be in [-1, 1]".into());
    }
    if open.is_empty() && terminal_close.is_some() {
        return Err("terminal_close requires a nonempty execution series".into());
    }
    let mut prices = open.to_vec();
    let mut filled: Vec<f64> = (0..weights.len())
        .map(|t| if t == 0 { 0. } else { weights[t - 1] })
        .collect();
    if let Some(close) = terminal_close {
        prices.push(close);
        filled.push(0.);
    }
    execution_ledger(&prices, &filled, fee_rate, 1., None)
}

/// Account precomputed forward holding intervals for close-time decisions.
/// The first interval must be zero (there is no first-bar executable decision).
/// The output has one extra boundary and an explicit terminal liquidation.
pub fn decision_interval_ledger(
    interval_returns: &[f64],
    weights: &[f64],
    fee_rate: f64,
) -> Result<ExecutionLedger, String> {
    validate_execution_inputs(interval_returns.len(), weights, fee_rate, None)?;
    if weights.iter().any(|w| w.abs() > 1.) {
        return Err("decision weights must be in [-1, 1]".into());
    }
    if interval_returns.first().is_some_and(|r| *r != 0.) {
        return Err("first decision interval return must be zero".into());
    }
    if interval_returns.is_empty() {
        return executed_weight_ledger(&[], &[], fee_rate, None, None);
    }
    let mut returns = Vec::with_capacity(weights.len() + 1);
    returns.push(0.);
    returns.extend_from_slice(interval_returns);
    let mut filled = Vec::with_capacity(weights.len() + 1);
    filled.push(0.);
    filled.extend_from_slice(&weights[..weights.len() - 1]);
    filled.push(0.);
    executed_weight_ledger(&returns, &filled, fee_rate, None, None)
}

/// Per-bar net ledger for close-time targets filled at subsequent observed opens,
/// retaining open positions at the right boundary without a fabricated exit fill.
pub fn execution_net_returns(
    open: &[f64],
    weights: &[f64],
    fee_rate: f64,
) -> Result<Vec<f64>, String> {
    Ok(decision_execution_ledger(open, weights, fee_rate, None)?.net_returns)
}

/// Account signed close-time targets at subsequent opens. No terminal liquidation is invented.
/// `net_ret * quality_w` equals the ledger contribution, including all resizing and both exit legs.
#[allow(clippy::too_many_arguments)]
pub fn execution_episodes(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr: &[f64],
    weights: &[f64],
    fee_rate: f64,
) -> Result<Vec<ExecutionEpisode>, String> {
    let n = open.len();
    if [high.len(), low.len(), close.len(), atr.len(), weights.len()]
        .iter()
        .any(|&m| m != n)
    {
        return Err("execution input lengths must match".into());
    }
    if !fee_rate.is_finite() || fee_rate < 0. {
        return Err("fee_rate must be finite and nonnegative".into());
    }
    if open
        .iter()
        .chain(high)
        .chain(low)
        .chain(close)
        .any(|p| !p.is_finite() || *p <= 0.)
        || weights.iter().any(|w| !w.is_finite() || w.abs() > 1.)
    {
        return Err("prices must be finite and positive; weights must be finite in [-1, 1]".into());
    }
    let ledger = decision_execution_ledger(open, weights, fee_rate, None)?;
    let mut result = Vec::new();
    let mut start: Option<usize> = None;
    let (mut pnl, mut mfe, mut mae) = (0., 0., 0.);
    let mut previous: f64 = 0.;
    for fill in 1..n {
        let target = weights[fill - 1];
        if let Some(entry) = start {
            pnl += ledger.gross_returns[fill];
            let direction = weights[entry - 1].signum();
            let unit = atr[entry - 1];
            if unit.is_finite() && unit > 0. {
                let move_at_open = direction * (open[fill] - open[entry]) / unit;
                mfe = f64::max(mfe, move_at_open);
                mae = f64::max(mae, -move_at_open);
            }
            if target == 0. || target.signum() != previous.signum() {
                pnl -= ledger.pretrade_weights[fill].abs() * fee_rate;
                let quality = weights[entry - 1].abs();
                if unit.is_finite() && unit > 0. {
                    result.push(ExecutionEpisode {
                        trigger_idx: entry - 1,
                        entry_idx: entry,
                        exit_idx: fill,
                        direction: direction as i8,
                        net_ret: pnl / quality,
                        weighted_net_ret: pnl,
                        quality_w: quality,
                        atr_pct: unit / open[entry],
                        mfe_atr: mfe,
                        mae_atr: mae,
                    });
                }
                start = None;
            } else {
                pnl -= ledger.cost_returns[fill];
            }
        }
        if start.is_none() && target != 0. {
            start = Some(fill);
            pnl = -target.abs() * fee_rate;
            mfe = 0.;
            mae = 0.;
        }
        if let Some(entry) = start {
            let unit = atr[entry - 1];
            if unit.is_finite() && unit > 0. {
                let direction = target.signum();
                let favorable = if direction > 0. {
                    high[fill]
                } else {
                    low[fill]
                };
                let adverse = if direction > 0. {
                    low[fill]
                } else {
                    high[fill]
                };
                mfe = f64::max(mfe, direction * (favorable - open[entry]) / unit);
                mae = f64::max(mae, -direction * (adverse - open[entry]) / unit);
            }
        }
        previous = target;
    }
    Ok(result)
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction]
#[pyfunction(name = "execution_episodes", signature = (open, high, low, close, atr, weights, fee_rate=0.0002))]
#[allow(clippy::too_many_arguments)]
fn py_execution_episodes(
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    atr: Vec<f64>,
    weights: Vec<f64>,
    fee_rate: f64,
) -> PyResult<Vec<ExecutionEpisode>> {
    execution_episodes(&open, &high, &low, &close, &atr, &weights, fee_rate)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction]
#[pyfunction(name = "execution_net_returns", signature = (open, weights, fee_rate=0.0002))]
fn py_execution_net_returns(
    open: Vec<f64>,
    weights: Vec<f64>,
    fee_rate: f64,
) -> PyResult<Vec<f64>> {
    execution_net_returns(&open, &weights, fee_rate)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction]
#[pyfunction(
    name = "executed_weight_ledger",
    signature = (price_returns, weights, fee_rate=0.0002, tradable=None, roll_mask=None)
)]
/// Account already executed finite target weights against mark-to-mark price returns.
/// Every tradable mark rebalances on fixed initial capital and charges actual traded
/// notional. Missing marks require zero return and unchanged weight. On contract roll
/// marks (`roll_mask=True`), non-trading price gaps are stripped to zero and open
/// inventory adds `2 * |previous|` rollover turnover. Results are initial-capital
/// fractions; quantity fields are None. No terminal exit is inserted.
fn py_executed_weight_ledger(
    price_returns: Vec<f64>,
    weights: Vec<f64>,
    fee_rate: f64,
    tradable: Option<Vec<bool>>,
    roll_mask: Option<Vec<bool>>,
) -> PyResult<ExecutionLedger> {
    executed_weight_ledger(
        &price_returns,
        &weights,
        fee_rate,
        tradable.as_deref(),
        roll_mask.as_deref(),
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction]
#[pyfunction(name = "execution_ledger", signature = (prices, weights, fee_rate=0.0002, initial_capital=1.0, tradable=None))]
/// Account already executed weights with q = weight * initial_capital / price.
/// Returns and fees use fixed initial capital; quantities use a unit multiplier.
/// Missing quotes require tradable=False and unchanged weight, allow NaN or an
/// unchanged carried price, and freeze quantity. Final inventory is retained.
fn py_execution_ledger(
    prices: Vec<f64>,
    weights: Vec<f64>,
    fee_rate: f64,
    initial_capital: f64,
    tradable: Option<Vec<bool>>,
) -> PyResult<ExecutionLedger> {
    execution_ledger(
        &prices,
        &weights,
        fee_rate,
        initial_capital,
        tradable.as_deref(),
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction]
#[pyfunction(name = "decision_execution_ledger", signature = (open, weights, fee_rate=0.0002, terminal_close=None))]
/// Fill close-time decisions in [-1, 1] at the next open using the shared quantity ledger.
/// With terminal_close=None, preserve terminal inventory. A supplied positive close
/// appends an explicit liquidation mark, so every result array has one extra item.
fn py_decision_execution_ledger(
    open: Vec<f64>,
    weights: Vec<f64>,
    fee_rate: f64,
    terminal_close: Option<f64>,
) -> PyResult<ExecutionLedger> {
    decision_execution_ledger(&open, &weights, fee_rate, terminal_close)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction]
#[pyfunction(name = "decision_interval_ledger", signature = (interval_returns, weights, fee_rate=0.0002))]
/// Account close-time decisions from forward holding intervals without synthetic prices.
/// The first interval must be zero. Nonempty inputs produce one extra boundary with
/// explicit terminal liquidation; decision weights must lie in [-1, 1].
fn py_decision_interval_ledger(
    interval_returns: Vec<f64>,
    weights: Vec<f64>,
    fee_rate: f64,
) -> PyResult<ExecutionLedger> {
    decision_interval_ledger(&interval_returns, &weights, fee_rate)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction]
#[pyfunction(name = "closed_execution_intervals")]
/// Attribute a closed boundary ledger to forward intervals without changing totals.
/// Entry costs belong to the starting mark and terminal liquidation costs to the final
/// interval. Requires at least two marks and an explicitly zero terminal position.
fn py_closed_execution_intervals(ledger: ExecutionLedger) -> PyResult<ClosedExecutionIntervals> {
    closed_execution_intervals(&ledger).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction]
#[pyfunction(name = "standardize_execution_returns")]
/// Subtract standardized quantity-based costs from an independently calculated gross ledger.
/// Inputs must be finite, equal-length initial-capital fractions; costs must be nonnegative.
fn py_standardize_execution_returns(
    gross_returns: Vec<f64>,
    cost_returns: Vec<f64>,
) -> PyResult<Vec<f64>> {
    standardize_execution_returns(&gross_returns, &cost_returns)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[cfg(feature = "python")]
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ExecutionEpisode>()?;
    m.add_class::<ExecutionLedger>()?;
    m.add_class::<ClosedExecutionIntervals>()?;
    m.add_function(wrap_pyfunction!(py_executed_weight_ledger, m)?)?;
    m.add_function(wrap_pyfunction!(py_execution_ledger, m)?)?;
    m.add_function(wrap_pyfunction!(py_decision_execution_ledger, m)?)?;
    m.add_function(wrap_pyfunction!(py_decision_interval_ledger, m)?)?;
    m.add_function(wrap_pyfunction!(py_closed_execution_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(py_standardize_execution_returns, m)?)?;
    m.add_function(wrap_pyfunction!(py_execution_episodes, m)?)?;
    m.add_function(wrap_pyfunction!(py_execution_net_returns, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moving_prices_match_independent_fill_ledger_and_scale_linearly() {
        let prices = [100., 110., 99., 120., 118., 125., 126.];
        let weights = [0.5, 0.25, -0.75, -0.25, 0., 0., 0.];
        let expected = [
            0.,
            -0.0005,
            0.5 * (99. / 110. - 1.) - (0.5 * 99. / 110. - 0.25) * 0.001,
            0.25 * (120. / 99. - 1.) - (0.25 * 120. / 99. + 0.75) * 0.001,
            -0.75 * (118. / 120. - 1.) - (0.75 * 118. / 120. - 0.25) * 0.001,
            -0.25 * (125. / 118. - 1.) - (0.25 * 125. / 118.) * 0.001,
            0.,
        ];
        let ledger = execution_net_returns(&prices, &weights, 0.001).unwrap();
        for (actual, expected) in ledger.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-15);
        }
        let episodes = execution_episodes(
            &prices, &prices, &prices, &prices, &[1.; 7], &weights, 0.001,
        )
        .unwrap();
        assert_eq!(episodes.len(), 2);
        assert!(
            (episodes.iter().map(|e| e.weighted_net_ret).sum::<f64>() - ledger.iter().sum::<f64>())
                .abs()
                < 1e-15
        );
        let half_weights: Vec<_> = weights.iter().map(|w| w * 0.5).collect();
        let half = execution_episodes(
            &prices,
            &prices,
            &prices,
            &prices,
            &[1.; 7],
            &half_weights,
            0.001,
        )
        .unwrap();
        for (full, half) in episodes.iter().zip(half) {
            assert_eq!(half.quality_w, full.quality_w * 0.5);
            assert!((half.weighted_net_ret - full.weighted_net_ret * 0.5).abs() < 1e-15);
            assert!((half.net_ret - full.net_ret).abs() < 1e-15);
        }
    }

    #[test]
    fn completed_episodes_and_ledger_are_prefix_causal() {
        let prices = [100., 110., 99., 120., 118., 125., 126.];
        let weights = [0.5, 0.25, -0.75, -0.25, 0., 0., 0.];
        let full = execution_episodes(
            &prices, &prices, &prices, &prices, &[1.; 7], &weights, 0.001,
        )
        .unwrap();
        let ledger = execution_net_returns(&prices, &weights, 0.001).unwrap();
        for (n, completed) in [(2, 0), (3, 0), (4, 1), (5, 1), (6, 2)] {
            let prefix = execution_episodes(
                &prices[..n],
                &prices[..n],
                &prices[..n],
                &prices[..n],
                &vec![1.; n],
                &weights[..n],
                0.001,
            )
            .unwrap();
            assert_eq!(prefix, full[..completed]);
            assert_eq!(
                execution_net_returns(&prices[..n], &weights[..n], 0.001).unwrap(),
                ledger[..n]
            );
        }
    }

    #[test]
    fn final_close_target_has_no_invented_fill() {
        let prices = [100., 110., 99., 120.];
        let before = [-0.5; 4];
        let final_exit = [-0.5, -0.5, -0.5, 0.];
        assert_eq!(
            execution_net_returns(&prices, &before, 0.001).unwrap(),
            execution_net_returns(&prices, &final_exit, 0.001).unwrap()
        );
        assert!(
            execution_episodes(
                &prices,
                &prices,
                &prices,
                &prices,
                &[1.; 4],
                &final_exit,
                0.001
            )
            .unwrap()
            .is_empty()
        );
    }

    #[test]
    fn execution_validation_matches_for_both_shared_apis() {
        assert!(execution_net_returns(&[], &[], 0.001).unwrap().is_empty());
        assert!(
            execution_episodes(&[], &[], &[], &[], &[], &[], 0.001)
                .unwrap()
                .is_empty()
        );
        let p = [100., 101.];
        assert!(
            execution_net_returns(&p, &[0.], 0.001)
                .unwrap_err()
                .contains("lengths")
        );
        assert!(
            execution_episodes(&p, &p, &p, &p, &[1.; 2], &[0.], 0.001)
                .unwrap_err()
                .contains("lengths")
        );
        for fee in [-0.1, f64::INFINITY, f64::NAN] {
            assert!(
                execution_net_returns(&p, &[0.; 2], fee)
                    .unwrap_err()
                    .contains("fee_rate")
            );
            assert!(
                execution_episodes(&p, &p, &p, &p, &[1.; 2], &[0.; 2], fee)
                    .unwrap_err()
                    .contains("fee_rate")
            );
        }
        for invalid in [1.1, f64::INFINITY, f64::NAN] {
            assert!(
                execution_net_returns(&p, &[0., invalid], 0.001)
                    .unwrap_err()
                    .contains("weights")
            );
            assert!(
                execution_episodes(&p, &p, &p, &p, &[1.; 2], &[0., invalid], 0.001)
                    .unwrap_err()
                    .contains("weights")
            );
        }
    }

    #[test]
    fn resize_reverse_and_close_costs_match_signed_ledger() {
        let price = vec![100.; 7];
        let episodes = execution_episodes(
            &price,
            &price,
            &price,
            &price,
            &[1.; 7],
            &[0.5, 0.25, -0.75, -0.25, 0., 0., 0.],
            0.001,
        )
        .unwrap();
        assert_eq!(episodes.len(), 2);
        assert!((episodes[0].weighted_net_ret + 0.001).abs() < 1e-12);
        assert!((episodes[1].weighted_net_ret + 0.0015).abs() < 1e-12);
        assert_eq!((episodes[0].entry_idx, episodes[0].exit_idx), (1, 3));
        assert_eq!((episodes[1].entry_idx, episodes[1].exit_idx), (3, 5));
        let returns =
            execution_net_returns(&price, &[0.5, 0.25, -0.75, -0.25, 0., 0., 0.], 0.001).unwrap();
        assert!(
            (returns.iter().sum::<f64>()
                - episodes.iter().map(|e| e.weighted_net_ret).sum::<f64>())
            .abs()
                < 1e-12
        );
    }
    #[test]
    fn unfinished_fold_and_post_exit_extrema_are_excluded() {
        let price = vec![100.; 4];
        let high = vec![100., 100., 1000., 100.];
        let episodes = execution_episodes(
            &price,
            &high,
            &price,
            &price,
            &[1.; 4],
            &[1., 0., 1., 0.],
            0.,
        )
        .unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].mfe_atr, 0.);
        assert!(
            execution_episodes(
                &price[..2],
                &price[..2],
                &price[..2],
                &price[..2],
                &[1.; 2],
                &[1., 0.],
                0.
            )
            .unwrap()
            .is_empty()
        );
    }
}
