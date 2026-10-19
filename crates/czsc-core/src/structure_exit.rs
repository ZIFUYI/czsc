//! Causal structural anchors and the shared Setup -> Confirmation -> Exit machine.
//!
//! Decisions use completed closes. A caller executes the resulting target at the next
//! tradable open; these are not claims about the unknowable order of intrabar prices.

use crate::analyze::{CZSC, resolve_max_bi_num, resolve_min_bi_len, utils::get_zs_seq};
use crate::objects::{bar::RawBar, direction::Direction, mark::Mark};
#[cfg(feature = "python")]
use pyo3::pyclass;
#[cfg(feature = "python")]
use pyo3_stub_gen::derive::gen_stub_pyclass;

/// The originating mean-reversion Setup and its structural stop use the same BI history.
pub const ZS_REVERSION_BI_WINDOW: usize = 20;

/// Structures known at this bar's close. Missing prices are NaN and missing IDs are -1.
/// IDs are the structure's endpoint timestamp in milliseconds; completed_bi_dir is
/// an event, not a persistent direction, and is emitted only for a newly completed BI.
/// The row's input bar is the confirmation time; the earlier endpoint ID is never
/// used as its availability time.
#[cfg_attr(feature = "python", gen_stub_pyclass)]
#[cfg_attr(
    feature = "python",
    pyclass(from_py_object, get_all, frozen, module = "czsc._native")
)]
#[derive(Clone, Debug)]
pub struct StructureSnapshot {
    pub asof_utc: String,
    pub fx_low: f64,
    pub fx_high: f64,
    pub fx_low_id: i64,
    pub fx_high_id: i64,
    pub bi_low: f64,
    pub bi_high: f64,
    pub bi_id: i64,
    pub bi_second_low: f64,
    pub bi_second_high: f64,
    pub bi_second_low_id: i64,
    pub bi_second_high_id: i64,
    pub zs_low: f64,
    pub zs_high: f64,
    pub zs_mid: f64,
    pub zs_zd: f64,
    pub zs_zg: f64,
    pub zs_id: i64,
    pub completed_bi_dir: i8,
    pub completed_bi_id: i64,
    pub completed_fx_dir: i8,
    pub completed_fx_id: i64,
}

impl Default for StructureSnapshot {
    fn default() -> Self {
        Self {
            asof_utc: String::new(),
            fx_low: f64::NAN,
            fx_high: f64::NAN,
            fx_low_id: -1,
            fx_high_id: -1,
            bi_low: f64::NAN,
            bi_high: f64::NAN,
            bi_id: -1,
            bi_second_low: f64::NAN,
            bi_second_high: f64::NAN,
            bi_second_low_id: -1,
            bi_second_high_id: -1,
            zs_low: f64::NAN,
            zs_high: f64::NAN,
            zs_mid: f64::NAN,
            zs_zd: f64::NAN,
            zs_zg: f64::NAN,
            zs_id: -1,
            completed_bi_dir: 0,
            completed_bi_id: -1,
            completed_fx_dir: 0,
            completed_fx_id: -1,
        }
    }
}

fn validate_bars(bars: &[RawBar]) -> Result<(), String> {
    for (i, b) in bars.iter().enumerate() {
        if [b.open, b.high, b.low, b.close]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
            || b.high < b.open.max(b.close)
            || b.low > b.open.min(b.close)
        {
            return Err(format!("invalid OHLC at index {i}"));
        }
        if i > 0 && (b.dt <= bars[i - 1].dt || b.freq != bars[0].freq || b.symbol != bars[0].symbol)
        {
            return Err(
                "bars must have strictly increasing timestamps and one symbol/frequency".into(),
            );
        }
    }
    Ok(())
}

/// Build one immutable, causal snapshot per input bar using the core CZSC analyzer.
/// Zero configuration values use the same environment-aware resolvers as CZSC signals.
pub fn structure_snapshots(
    bars: &[RawBar],
    min_bi_len: usize,
    max_bi_num: usize,
) -> Result<Vec<StructureSnapshot>, String> {
    let min_bi_len = resolve_min_bi_len(min_bi_len);
    let max_bi_num = resolve_max_bi_num(max_bi_num);
    if min_bi_len < 3 || max_bi_num < 3 {
        return Err("min_bi_len and max_bi_num must be at least 3".into());
    }
    validate_bars(bars)?;
    if bars.is_empty() {
        return Ok(vec![]);
    }
    let mut c = CZSC::new(vec![bars[0].clone()], max_bi_num, min_bi_len);
    let mut previous = StructureSnapshot::default();
    let mut latest_completed = -1;
    let mut latest_completed_fx = -1;
    let mut out = Vec::with_capacity(bars.len());
    for (i, bar) in bars.iter().enumerate() {
        if i > 0 {
            c.update_bar(bar.clone());
        }
        let mut row = previous.clone();
        row.asof_utc = bar.dt.to_rfc3339();
        row.completed_bi_dir = 0;
        row.completed_bi_id = -1;
        row.completed_fx_dir = 0;
        row.completed_fx_id = -1;
        let fx_list = c.get_fx_list();
        for fx in &fx_list {
            let id = fx.dt.timestamp_millis();
            if fx.mark == Mark::D && id > row.fx_low_id {
                row.fx_low = fx.low;
                row.fx_low_id = id;
            }
            if fx.mark == Mark::G && id > row.fx_high_id {
                row.fx_high = fx.high;
                row.fx_high_id = id;
            }
        }
        if let Some(last_fx) = fx_list.last() {
            let fx_id = last_fx.dt.timestamp_millis();
            if fx_id > latest_completed_fx {
                row.completed_fx_id = fx_id;
                row.completed_fx_dir = if last_fx.mark == Mark::D { 1 } else { -1 };
                latest_completed_fx = fx_id;
            }
        }
        let bis = c.get_finished_bis();
        if let Some(bi) = bis.last() {
            row.bi_low = bi.get_low();
            row.bi_high = bi.get_high();
            row.bi_id = bi.end_dt().timestamp_millis();
            if row.bi_id > latest_completed {
                row.completed_bi_id = row.bi_id;
                row.completed_bi_dir = if bi.direction == Direction::Up { 1 } else { -1 };
                latest_completed = row.bi_id;
            }
        }
        let mut bottoms: Vec<(f64, i64)> = Vec::with_capacity(8);
        let mut tops: Vec<(f64, i64)> = Vec::with_capacity(8);
        for bi in bis.iter().rev().take(6).rev() {
            let (l_id, h_id) = if bi.direction == Direction::Down {
                (
                    bi.end_dt().timestamp_millis(),
                    bi.start_dt().timestamp_millis(),
                )
            } else {
                (
                    bi.start_dt().timestamp_millis(),
                    bi.end_dt().timestamp_millis(),
                )
            };
            if bottoms.last().is_none_or(|(_, prev_id)| *prev_id != l_id) {
                bottoms.push((bi.get_low(), l_id));
            }
            if tops.last().is_none_or(|(_, prev_id)| *prev_id != h_id) {
                tops.push((bi.get_high(), h_id));
            }
        }
        if row.fx_low.is_finite() && row.fx_low_id > bottoms.last().map_or(-1, |(_, id)| *id) {
            bottoms.push((row.fx_low, row.fx_low_id));
        }
        if row.fx_high.is_finite() && row.fx_high_id > tops.last().map_or(-1, |(_, id)| *id) {
            tops.push((row.fx_high, row.fx_high_id));
        }
        if bottoms.len() >= 2 {
            let (l1, id1) = bottoms[bottoms.len() - 2];
            let (l2, id2) = bottoms[bottoms.len() - 1];
            let ((hi_l, hi_id), (lo_l, lo_id)) = if l2 >= l1 {
                ((l2, id2), (l1, id1))
            } else {
                ((l1, id1), (l2, id2))
            };
            if bar.close > hi_l {
                row.bi_second_low = hi_l;
                row.bi_second_low_id = hi_id;
            } else {
                row.bi_second_low = lo_l;
                row.bi_second_low_id = lo_id;
            }
        } else if let Some(&(l1, id1)) = bottoms.last() {
            row.bi_second_low = l1;
            row.bi_second_low_id = id1;
        }
        if tops.len() >= 2 {
            let (h1, id1) = tops[tops.len() - 2];
            let (h2, id2) = tops[tops.len() - 1];
            let ((lo_h, lo_id), (hi_h, hi_id)) = if h2 <= h1 {
                ((h2, id2), (h1, id1))
            } else {
                ((h1, id1), (h2, id2))
            };
            if bar.close < lo_h {
                row.bi_second_high = lo_h;
                row.bi_second_high_id = lo_id;
            } else {
                row.bi_second_high = hi_h;
                row.bi_second_high_id = hi_id;
            }
        } else if let Some(&(h1, id1)) = tops.last() {
            row.bi_second_high = h1;
            row.bi_second_high_id = id1;
        }
        let zs_bis = &bis[bis.len().saturating_sub(ZS_REVERSION_BI_WINDOW)..];
        if let Some(zs) = get_zs_seq(zs_bis).iter().rev().find(|z| z.is_valid()) {
            row.zs_low = zs.dd;
            row.zs_high = zs.gg;
            row.zs_mid = zs.zz;
            row.zs_zd = zs.zd;
            row.zs_zg = zs.zg;
            row.zs_id = zs.edt.timestamp_millis();
        }
        previous = row.clone();
        out.push(row);
    }
    Ok(out)
}

/// Side-specific stops and source structure IDs. NaN means the required structure is absent.
#[derive(Clone, Debug)]
pub struct StructureStopAnchors {
    pub long_stop: Vec<f64>,
    pub short_stop: Vec<f64>,
    pub long_id: Vec<i64>,
    pub short_id: Vec<i64>,
}

/// Dispatch the research stop plugins in the shared core.
/// `pos_zs_mid` refers to the mean-reversion family: its invalidation is outside
/// the originating ZS (dd/gg), not the midpoint which is its mean-reversion target.
/// `pos_second_ext` anchors to secondary completed extrema (`bi_second_low` / `bi_second_high`).
/// `pos_zs_reentry` combines 3rd-buy/3rd-sell ZS edge (`zs_zg` / `zs_zd`) reentry invalidation
/// with secondary extrema.
pub fn structure_stop_anchors(
    bars: &[RawBar],
    snapshots: &[StructureSnapshot],
    stop_plugin: &str,
    htf_ma: Option<&[f64]>,
    box_window: usize,
) -> Result<StructureStopAnchors, String> {
    validate_bars(bars)?;
    if snapshots.len() != bars.len() || htf_ma.is_some_and(|x| x.len() != bars.len()) {
        return Err("bars, snapshots and htf_ma must have equal lengths".into());
    }
    if bars
        .iter()
        .zip(snapshots)
        .any(|(bar, row)| row.asof_utc != bar.dt.to_rfc3339())
    {
        return Err("snapshot asof must match each input bar timestamp".into());
    }
    if box_window == 0 {
        return Err("box_window must be positive".into());
    }
    if ![
        "pos_fx_stop",
        "pos_structure_ext",
        "pos_second_ext",
        "pos_zs_mid",
        "pos_zs_reentry",
        "pos_flow_stop",
        "pos_box_stop",
        "pos_htf_ma_stop",
    ]
    .contains(&stop_plugin)
    {
        return Err(format!("unknown stop_plugin: {stop_plugin}"));
    }
    let mut out = StructureStopAnchors {
        long_stop: vec![],
        short_stop: vec![],
        long_id: vec![],
        short_id: vec![],
    };
    for (i, (bar, s)) in bars.iter().zip(snapshots).enumerate() {
        let id = bar.dt.timestamp_millis();
        let (long, short, lid, sid) = match stop_plugin {
            "pos_fx_stop" => (s.fx_low, s.fx_high, s.fx_low_id, s.fx_high_id),
            "pos_structure_ext" => (s.bi_low, s.bi_high, s.bi_id, s.bi_id),
            "pos_second_ext" => (
                s.bi_second_low,
                s.bi_second_high,
                s.bi_second_low_id,
                s.bi_second_high_id,
            ),
            "pos_zs_mid" => (s.zs_low, s.zs_high, s.zs_id, s.zs_id),
            "pos_zs_reentry" => {
                let (l_stop, l_id) = if s.zs_zg.is_finite() && bar.close > s.zs_zg {
                    if s.bi_second_low.is_finite()
                        && s.bi_second_low < bar.close
                        && s.bi_second_low >= s.zs_zg
                    {
                        (s.bi_second_low, s.bi_second_low_id)
                    } else {
                        (s.zs_zg, s.zs_id)
                    }
                } else if s.bi_second_low.is_finite() {
                    (s.bi_second_low, s.bi_second_low_id)
                } else {
                    (s.fx_low, s.fx_low_id)
                };
                let (s_stop, s_id) = if s.zs_zd.is_finite() && bar.close < s.zs_zd {
                    if s.bi_second_high.is_finite()
                        && s.bi_second_high > bar.close
                        && s.bi_second_high <= s.zs_zd
                    {
                        (s.bi_second_high, s.bi_second_high_id)
                    } else {
                        (s.zs_zd, s.zs_id)
                    }
                } else if s.bi_second_high.is_finite() {
                    (s.bi_second_high, s.bi_second_high_id)
                } else {
                    (s.fx_high, s.fx_high_id)
                };
                (l_stop, s_stop, l_id, s_id)
            }
            "pos_flow_stop" => (bar.low, bar.high, id, id),
            "pos_box_stop" => {
                if i + 1 < box_window {
                    (f64::NAN, f64::NAN, -1, -1)
                } else {
                    let window = &bars[i + 1 - box_window..=i];
                    (
                        window.iter().map(|b| b.low).fold(f64::INFINITY, f64::min),
                        window
                            .iter()
                            .map(|b| b.high)
                            .fold(f64::NEG_INFINITY, f64::max),
                        window[0].dt.timestamp_millis(),
                        window[0].dt.timestamp_millis(),
                    )
                }
            }
            "pos_htf_ma_stop" => {
                let ma = htf_ma.map_or(f64::NAN, |x| x[i]);
                (ma, ma, id, id)
            }
            _ => unreachable!(),
        };
        out.long_stop.push(long);
        out.short_stop.push(short);
        out.long_id.push(lid);
        out.short_id.push(sid);
    }
    Ok(out)
}

/// Shared controls. Trailing fractions must remain in the strategy SOP's 1.2--1.5% range.
#[derive(Clone, Debug)]
pub struct StructureFactorConfig {
    pub window: usize,
    pub lam0: f64,
    pub rho_min: f64,
    pub rho_max: f64,
    pub d_eps: f64,
    pub max_hold: usize,
    pub trailing: f64,
    pub trail_activation: f64,
    pub ratchet_stop: bool,
    pub min_hold: usize,
    pub cooldown: usize,
    pub track_signal: bool,
    pub zs_zg: Option<Vec<f64>>,
    pub zs_zd: Option<Vec<f64>>,
    pub completed_fx_dir: Option<Vec<i8>>,
    pub soft_exit: Option<Vec<bool>>,
}
impl Default for StructureFactorConfig {
    fn default() -> Self {
        Self {
            window: 6,
            lam0: 0.10,
            rho_min: 0.5,
            rho_max: 2.0,
            d_eps: 0.02,
            max_hold: 64,
            trailing: 0.013,
            trail_activation: 0.0,
            ratchet_stop: false,
            min_hold: 0,
            cooldown: 0,
            track_signal: false,
            zs_zg: None,
            zs_zd: None,
            completed_fx_dir: None,
            soft_exit: None,
        }
    }
}
impl StructureFactorConfig {
    fn validate(&self, allow_disabled_trailing: bool, n: usize) -> Result<(), String> {
        if self.window == 0
            || self.max_hold == 0
            || !self.lam0.is_finite()
            || self.lam0 < 0.0
            || !self.rho_min.is_finite()
            || !self.rho_max.is_finite()
            || self.rho_min <= 0.0
            || self.rho_max < self.rho_min
            || !self.d_eps.is_finite()
            || self.d_eps <= 0.0
            || self.d_eps >= 1.0
            || !self.trailing.is_finite()
            || !((0.012..=0.015).contains(&self.trailing)
                || allow_disabled_trailing && self.trailing == 0.0)
            || !self.trail_activation.is_finite()
            || !(0.0..=0.25).contains(&self.trail_activation)
        {
            return Err("invalid structure configuration: require positive windows, valid decay and trailing in [0.012, 0.015]".into());
        }
        if self.zs_zg.as_ref().is_some_and(|v| v.len() != n)
            || self.zs_zd.as_ref().is_some_and(|v| v.len() != n)
            || self.completed_fx_dir.as_ref().is_some_and(|v| v.len() != n)
            || self.soft_exit.as_ref().is_some_and(|v| v.len() != n)
        {
            return Err("optional structure arrays must match input length".into());
        }
        if self
            .completed_fx_dir
            .as_ref()
            .is_some_and(|v| v.iter().any(|d| !(-1..=1).contains(d)))
        {
            return Err("completed_fx_dir must be -1, 0 or 1".into());
        }
        Ok(())
    }
}

/// All vectors share a chronological close-time grid. NaN stops/ATR warmup are allowed;
/// absent anchors prevent setup, and missing/nonpositive ATR uses a neutral decay ratio.
#[derive(Clone, Debug)]
pub struct StructureFactorInput {
    pub setup_long: Vec<bool>,
    pub setup_short: Vec<bool>,
    pub setup_quality_long: Vec<f64>,
    pub setup_quality_short: Vec<f64>,
    pub confirm_long: Vec<bool>,
    pub confirm_short: Vec<bool>,
    pub confirm_quality_long: Vec<f64>,
    pub confirm_quality_short: Vec<f64>,
    pub close: Vec<f64>,
    pub long_stop: Vec<f64>,
    pub short_stop: Vec<f64>,
    pub completed_bi_dir: Vec<i8>,
    pub atr_fast: Vec<f64>,
    pub atr_slow: Vec<f64>,
}

/// Decision-time diagnostics. stop/setup_index retain the exited position's anchor
/// on its exit bar; direction/quality describe accepted entries only.
#[derive(Clone, Debug)]
pub struct StructureFactorResult {
    pub strength: Vec<f64>,
    pub trigger: Vec<bool>,
    pub direction: Vec<i8>,
    pub position_direction: Vec<i8>,
    pub quality: Vec<f64>,
    pub stop: Vec<f64>,
    pub setup_index: Vec<i64>,
    pub exit_reason: Vec<String>,
}
impl StructureFactorResult {
    fn new(n: usize) -> Self {
        Self {
            strength: vec![0.0; n],
            trigger: vec![false; n],
            direction: vec![0; n],
            position_direction: vec![0; n],
            quality: vec![0.0; n],
            stop: vec![f64::NAN; n],
            setup_index: vec![-1; n],
            exit_reason: vec![String::new(); n],
        }
    }
}

#[derive(Clone, Copy)]
struct Armed {
    index: usize,
    quality: f64,
    stop: f64,
}
#[derive(Clone, Copy)]
struct Holding {
    direction: i8,
    quality: f64,
    decay: f64,
    stop: f64,
    setup: usize,
    entry_close: f64,
    best_close: f64,
    held: usize,
}
fn valid_stop(side: i8, price: f64, stop: f64) -> bool {
    stop.is_finite() && stop > 0.0 && f64::from(side) * (price - stop) > 0.0
}

#[allow(clippy::too_many_arguments)]
fn advance(
    holding: &mut Option<Holding>,
    t: usize,
    close: f64,
    reverse: i8,
    fast: f64,
    slow: f64,
    current_stop_anchor: f64,
    cfg: &StructureFactorConfig,
    out: &mut StructureFactorResult,
) {
    if let Some(h) = holding {
        out.position_direction[t] = h.direction;
        out.stop[t] = h.stop;
        out.setup_index[t] = h.setup as i64;
        let peak_gain = if h.direction > 0 {
            (h.best_close - h.entry_close) / h.entry_close.max(1e-12)
        } else {
            (h.entry_close - h.best_close) / h.entry_close.max(1e-12)
        };
        let trail_active = cfg.trailing > 0.0
            && (cfg.trail_activation <= 0.0 || peak_gain >= cfg.trail_activation);
        let trailing_broken = trail_active
            && if h.direction > 0 {
                close <= h.best_close * (1.0 - cfg.trailing)
            } else {
                close >= h.best_close * (1.0 + cfg.trailing)
            };
        let fx_dir = cfg.completed_fx_dir.as_ref().map_or(0, |v| v[t]);
        let trailing_fx_broken = trail_active
            && cfg.trail_activation > 0.0
            && fx_dir == -h.direction
            && if h.direction > 0 {
                close <= h.best_close * (1.0 - 0.35 * cfg.trailing)
            } else {
                close >= h.best_close * (1.0 + 0.35 * cfg.trailing)
            };
        let zs_reentry_broken = if h.direction > 0 {
            cfg.zs_zg.as_ref().is_some_and(|zg| {
                let z = zg[t];
                z.is_finite() && z > 0.0 && h.best_close > z && close <= z
            })
        } else {
            cfg.zs_zd.as_ref().is_some_and(|zd| {
                let z = zd[t];
                z.is_finite() && z > 0.0 && h.best_close < z && close >= z
            })
        };
        let soft_broken = cfg
            .soft_exit
            .as_ref()
            .is_some_and(|se| se[t] && (h.held + 1) >= cfg.min_hold.max(1));
        let reason = if !valid_stop(h.direction, close, h.stop) {
            "structure_break"
        } else if zs_reentry_broken {
            "zs_reentry_break"
        } else if reverse == -h.direction {
            "reverse_bi"
        } else if trailing_broken {
            "trailing_stop"
        } else if trailing_fx_broken {
            "trailing_fx_exit"
        } else if soft_broken {
            "soft_exit"
        } else {
            let ratio = if fast.is_finite() && slow.is_finite() && fast > 0.0 && slow > 0.0 {
                fast / slow
            } else {
                1.0
            };
            h.decay *= (-cfg.lam0 * ratio.clamp(cfg.rho_min, cfg.rho_max)).exp();
            h.held += 1;
            if h.direction > 0 {
                h.best_close = h.best_close.max(close);
            } else {
                h.best_close = h.best_close.min(close);
            }
            if cfg.ratchet_stop && valid_stop(h.direction, close, current_stop_anchor) {
                if h.direction > 0 {
                    h.stop = h.stop.max(current_stop_anchor);
                } else {
                    h.stop = h.stop.min(current_stop_anchor);
                }
            }
            if h.held > cfg.max_hold {
                "max_hold"
            } else if h.held >= cfg.min_hold && h.decay < cfg.d_eps {
                "decay"
            } else {
                ""
            }
        };
        if !reason.is_empty() {
            out.exit_reason[t] = reason.into();
            *holding = None;
        }
    }
}

fn accept(
    holding: &mut Option<Holding>,
    side: i8,
    q: f64,
    anchor: Armed,
    t: usize,
    close: f64,
    out: &mut StructureFactorResult,
) {
    if q > 0.0
        && valid_stop(side, close, anchor.stop)
        && holding.is_none_or(|h| h.direction != side || q >= h.quality * h.decay)
    {
        *holding = Some(Holding {
            direction: side,
            quality: q.clamp(0.0, 1.0),
            decay: 1.0,
            stop: anchor.stop,
            setup: anchor.index,
            entry_close: close,
            best_close: close,
            held: 0,
        });
        out.trigger[t] = true;
        out.direction[t] = side;
        out.quality[t] = q.clamp(0.0, 1.0);
    }
}
fn record(holding: Option<Holding>, t: usize, out: &mut StructureFactorResult) {
    if let Some(h) = holding {
        out.position_direction[t] = h.direction;
        out.strength[t] = f64::from(h.direction) * h.quality * h.decay;
        out.stop[t] = h.stop;
        out.setup_index[t] = h.setup as i64;
    }
}

fn is_risk_exit(reason: &str) -> bool {
    matches!(
        reason,
        "structure_break"
            | "zs_reentry_break"
            | "reverse_bi"
            | "trailing_stop"
            | "trailing_fx_exit"
            | "soft_exit"
    )
}

/// Run the complete two-stage structural machine. The first valid Setup freezes its
/// structure until consumption, expiry or invalidation; repeated setups cannot move it.
/// A confirmation must occur on a strictly later close. Simultaneous equal-quality
/// opposite confirmations cancel rather than introducing a long-side tie preference.
pub fn evaluate_structure_factor(
    x: &StructureFactorInput,
    cfg: &StructureFactorConfig,
) -> Result<StructureFactorResult, String> {
    let n = x.close.len();
    cfg.validate(false, n)?;
    if [
        x.setup_long.len(),
        x.setup_short.len(),
        x.setup_quality_long.len(),
        x.setup_quality_short.len(),
        x.confirm_long.len(),
        x.confirm_short.len(),
        x.confirm_quality_long.len(),
        x.confirm_quality_short.len(),
        x.long_stop.len(),
        x.short_stop.len(),
        x.completed_bi_dir.len(),
        x.atr_fast.len(),
        x.atr_slow.len(),
    ]
    .iter()
    .any(|l| *l != n)
    {
        return Err("all structure-factor arrays must have equal lengths".into());
    }
    validate_prices_directions(&x.close, &x.completed_bi_dir)?;
    let mut out = StructureFactorResult::new(n);
    let mut armed: [Option<Armed>; 2] = [None, None];
    let mut holding: Option<Holding> = None;
    let mut cooldown_rem: usize = 0;
    for t in 0..n {
        cooldown_rem = cooldown_rem.saturating_sub(1);
        let cur_anchor = match holding {
            Some(h) if h.direction > 0 => x.long_stop[t],
            Some(h) if h.direction < 0 => x.short_stop[t],
            _ => f64::NAN,
        };
        advance(
            &mut holding,
            t,
            x.close[t],
            x.completed_bi_dir[t],
            x.atr_fast[t],
            x.atr_slow[t],
            cur_anchor,
            cfg,
            &mut out,
        );
        if !out.exit_reason[t].is_empty() && cfg.cooldown > 0 {
            cooldown_rem = cfg.cooldown;
        }
        let blocked_side = if is_risk_exit(out.exit_reason[t].as_str()) {
            out.position_direction[t]
        } else {
            0
        };
        let mut candidate: Option<(i8, f64, Armed)> = None;
        let mut tied = false;
        for (slot, side, setup, sq, confirm, cq, stop) in [
            (
                0,
                1,
                x.setup_long[t],
                x.setup_quality_long[t],
                x.confirm_long[t],
                x.confirm_quality_long[t],
                x.long_stop[t],
            ),
            (
                1,
                -1,
                x.setup_short[t],
                x.setup_quality_short[t],
                x.confirm_short[t],
                x.confirm_quality_short[t],
                x.short_stop[t],
            ),
        ] {
            if side == blocked_side || x.completed_bi_dir[t] == -side {
                armed[slot] = None;
                continue;
            }
            if armed[slot]
                .is_some_and(|a| t - a.index > cfg.window || !valid_stop(side, x.close[t], a.stop))
            {
                armed[slot] = None;
            }
            if cooldown_rem == 0
                && confirm
                && cq.is_finite()
                && let Some(a) = armed[slot].take()
            {
                let q = (a.quality * (0.6 + 0.4 * cq.clamp(0.0, 1.0))).clamp(0.0, 1.0);
                if q > 0.0 {
                    if candidate.is_some_and(|(_, old_q, _)| q == old_q) {
                        tied = true;
                    } else if candidate.is_none_or(|(_, old_q, _)| q > old_q) {
                        candidate = Some((side, q, a));
                        tied = false;
                    }
                }
            }
            if setup
                && armed[slot].is_none()
                && sq.is_finite()
                && sq > 0.0
                && valid_stop(side, x.close[t], stop)
            {
                armed[slot] = Some(Armed {
                    index: t,
                    quality: sq.clamp(0.0, 1.0),
                    stop,
                });
            }
        }
        if cooldown_rem == 0
            && !tied
            && let Some((side, q, a)) = candidate
        {
            accept(&mut holding, side, q, a, t, x.close[t], &mut out);
        }
        record(holding, t, &mut out);
    }
    Ok(out)
}

fn validate_prices_directions(close: &[f64], directions: &[i8]) -> Result<(), String> {
    if close.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || directions.iter().any(|v| !(-1..=1).contains(v))
    {
        return Err("close must be finite and positive; directions must be -1, 0 or 1".into());
    }
    Ok(())
}

/// Trigger-only compatibility entry. Setup refreshes the armed quality after checking
/// Confirmation; a same-bar pair cannot create its own trigger. Structural consumers
/// should use evaluate_structure_factor to freeze and validate structure anchors.
pub fn two_stage_trigger(
    setup: &[bool],
    setup_quality: &[f64],
    confirm: &[bool],
    confirm_quality: &[f64],
    window: usize,
) -> Result<(Vec<bool>, Vec<f64>), String> {
    let n = setup.len();
    if window == 0
        || [setup_quality.len(), confirm.len(), confirm_quality.len()]
            .iter()
            .any(|len| *len != n)
    {
        return Err("window must be positive and trigger arrays must have equal lengths".into());
    }
    let mut fire = vec![false; n];
    let mut quality = vec![0.0; n];
    let mut armed: Option<(usize, f64)> = None;
    for t in 0..n {
        if armed.is_some_and(|(index, _)| t - index > window) {
            armed = None;
        }
        if confirm[t]
            && confirm_quality[t].is_finite()
            && let Some((_, q)) = armed.take()
        {
            fire[t] = true;
            quality[t] = (q * (0.6 + 0.4 * confirm_quality[t])).clamp(0.01, 1.0);
        }
        if setup[t] && setup_quality[t].is_finite() {
            armed = Some((t, setup_quality[t]));
        }
    }
    Ok((fire, quality))
}

/// Compatibility and continuous-governance entry for already-confirmed triggers or weight streams.
/// Uses the exact same exit transition as evaluate_structure_factor.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_structure_decay(
    direction: &[i8],
    quality: &[f64],
    close: &[f64],
    stop_level: &[f64],
    reverse_dir: &[i8],
    atr_fast: &[f64],
    atr_slow: &[f64],
    cfg: &StructureFactorConfig,
) -> Result<StructureFactorResult, String> {
    let n = close.len();
    cfg.validate(true, n)?;
    if [
        direction.len(),
        quality.len(),
        stop_level.len(),
        reverse_dir.len(),
        atr_fast.len(),
        atr_slow.len(),
    ]
    .iter()
    .any(|l| *l != n)
    {
        return Err("all structure-decay arrays must have equal lengths".into());
    }
    validate_prices_directions(close, direction)?;
    validate_prices_directions(close, reverse_dir)?;
    let mut out = StructureFactorResult::new(n);
    let mut holding: Option<Holding> = None;
    let mut cooldown_rem: usize = 0;
    let mut latched_side: i8 = 0;
    for t in 0..n {
        cooldown_rem = cooldown_rem.saturating_sub(1);
        if cfg.track_signal {
            if direction[t] != latched_side {
                latched_side = 0;
            }
            if let Some(h) = holding
                && direction[t] == 0
            {
                out.position_direction[t] = h.direction;
                out.stop[t] = h.stop;
                out.setup_index[t] = h.setup as i64;
                out.exit_reason[t] = "signal_flat".into();
                holding = None;
                continue;
            }
        }
        advance(
            &mut holding,
            t,
            close[t],
            reverse_dir[t],
            atr_fast[t],
            atr_slow[t],
            stop_level[t],
            cfg,
            &mut out,
        );
        if !out.exit_reason[t].is_empty() && cfg.cooldown > 0 {
            cooldown_rem = cfg.cooldown;
        }
        let same_side_risk_exit =
            direction[t] == out.position_direction[t] && is_risk_exit(out.exit_reason[t].as_str());
        if cfg.track_signal {
            if !out.exit_reason[t].is_empty() {
                latched_side = out.position_direction[t];
            } else if let Some(mut h) = holding {
                if direction[t] == h.direction && quality[t].is_finite() {
                    h.quality = quality[t].clamp(0.0, 1.0);
                    holding = Some(h);
                } else if direction[t] == -h.direction && quality[t].is_finite() {
                    holding = None;
                    latched_side = 0;
                    cooldown_rem = 0;
                    accept(
                        &mut holding,
                        direction[t],
                        quality[t],
                        Armed {
                            index: t,
                            quality: quality[t],
                            stop: stop_level[t],
                        },
                        t,
                        close[t],
                        &mut out,
                    );
                }
            } else if direction[t] != 0 && quality[t].is_finite() {
                let fx_dir = cfg.completed_fx_dir.as_ref().map_or(0, |v| v[t]);
                if latched_side == direction[t]
                    && cooldown_rem == 0
                    && (reverse_dir[t] == direction[t] || fx_dir == direction[t])
                {
                    latched_side = 0;
                }
                if latched_side == 0 && cooldown_rem == 0 {
                    accept(
                        &mut holding,
                        direction[t],
                        quality[t],
                        Armed {
                            index: t,
                            quality: quality[t],
                            stop: stop_level[t],
                        },
                        t,
                        close[t],
                        &mut out,
                    );
                }
            }
        } else if direction[t] != 0
            && quality[t].is_finite()
            && !same_side_risk_exit
            && cooldown_rem == 0
            && out.exit_reason[t].is_empty()
        {
            accept(
                &mut holding,
                direction[t],
                quality[t],
                Armed {
                    index: t,
                    quality: quality[t],
                    stop: stop_level[t],
                },
                t,
                close[t],
                &mut out,
            );
        }
        record(holding, t, &mut out);
    }
    Ok(out)
}
