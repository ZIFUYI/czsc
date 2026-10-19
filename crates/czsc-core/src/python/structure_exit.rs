//! PyO3 type conversion only; validation, defaults and algorithms are shared core code.
use crate::objects::bar::RawBar;
use crate::structure_exit::{
    self as engine, StructureFactorConfig, StructureFactorInput, StructureFactorResult,
    StructureSnapshot,
};
use pyo3::{prelude::*, types::PyDict};
use pyo3_stub_gen::derive::gen_stub_pyfunction;

fn value_error(message: String) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message)
}

fn result_dict(py: Python<'_>, result: StructureFactorResult) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("strength", result.strength)?;
    d.set_item("trigger", result.trigger)?;
    d.set_item("direction", result.direction)?;
    d.set_item("position_direction", result.position_direction)?;
    d.set_item("quality", result.quality)?;
    d.set_item("stop", result.stop)?;
    d.set_item("setup_index", result.setup_index)?;
    d.set_item("exit_reason", result.exit_reason)?;
    Ok(d.unbind())
}

/// Structures known at each completed close; no future bars or direction-string inference.
#[gen_stub_pyfunction]
#[pyfunction(name="structure_snapshots", signature=(bars, min_bi_len=0, max_bi_num=0))]
fn structure_snapshots_py(
    bars: Vec<RawBar>,
    min_bi_len: usize,
    max_bi_num: usize,
) -> PyResult<Vec<StructureSnapshot>> {
    engine::structure_snapshots(&bars, min_bi_len, max_bi_num).map_err(value_error)
}

/// Dispatch FX, completed BI, ZS exterior, setup flow bar, rolling box, or causal HTF MA stops.
#[gen_stub_pyfunction]
#[pyfunction(name="structure_stop_anchors", signature=(bars, snapshots, stop_plugin, htf_ma=None, box_window=20))]
fn structure_stop_anchors_py(
    py: Python<'_>,
    bars: Vec<RawBar>,
    snapshots: Vec<StructureSnapshot>,
    stop_plugin: &str,
    htf_ma: Option<Vec<f64>>,
    box_window: usize,
) -> PyResult<Py<PyDict>> {
    let result = engine::structure_stop_anchors(
        &bars,
        &snapshots,
        stop_plugin,
        htf_ma.as_deref(),
        box_window,
    )
    .map_err(value_error)?;
    let d = PyDict::new(py);
    d.set_item("long_stop", result.long_stop)?;
    d.set_item("short_stop", result.short_stop)?;
    d.set_item("long_id", result.long_id)?;
    d.set_item("short_id", result.short_id)?;
    Ok(d.unbind())
}

/// Shared two-stage machine: freeze the Setup anchor, confirm on a later close,
/// exit on structural invalidation, opposite completed BI, trailing, decay or timeout.
/// Result targets are close-time decisions for execution at the next tradable open.
#[gen_stub_pyfunction]
#[pyfunction(name="evaluate_structure_factor", signature=(setup_long, setup_short, setup_quality_long, setup_quality_short,
    confirm_long, confirm_short, confirm_quality_long, confirm_quality_short, close, long_stop, short_stop,
    completed_bi_dir, atr_fast, atr_slow, window=6, lam0=0.10, rho_min=0.5, rho_max=2.0, d_eps=0.02, max_hold=64, trailing=0.013,
    trail_activation=0.0, ratchet_stop=false, min_hold=0, cooldown=0, zs_zg=None, zs_zd=None, completed_fx_dir=None, soft_exit=None,
    reentry_mode="none", reentry_breakout_buf=0.0, partial_exit_ratio=0.0, breakeven_buffer=0.0, runner_trail_mult=1.5))]
#[allow(clippy::too_many_arguments)]
fn evaluate_structure_factor_py(
    py: Python<'_>,
    setup_long: Vec<bool>,
    setup_short: Vec<bool>,
    setup_quality_long: Vec<f64>,
    setup_quality_short: Vec<f64>,
    confirm_long: Vec<bool>,
    confirm_short: Vec<bool>,
    confirm_quality_long: Vec<f64>,
    confirm_quality_short: Vec<f64>,
    close: Vec<f64>,
    long_stop: Vec<f64>,
    short_stop: Vec<f64>,
    completed_bi_dir: Vec<i8>,
    atr_fast: Vec<f64>,
    atr_slow: Vec<f64>,
    window: usize,
    lam0: f64,
    rho_min: f64,
    rho_max: f64,
    d_eps: f64,
    max_hold: usize,
    trailing: f64,
    trail_activation: f64,
    ratchet_stop: bool,
    min_hold: usize,
    cooldown: usize,
    zs_zg: Option<Vec<f64>>,
    zs_zd: Option<Vec<f64>>,
    completed_fx_dir: Option<Vec<i8>>,
    soft_exit: Option<Vec<bool>>,
    reentry_mode: &str,
    reentry_breakout_buf: f64,
    partial_exit_ratio: f64,
    breakeven_buffer: f64,
    runner_trail_mult: f64,
) -> PyResult<Py<PyDict>> {
    let input = StructureFactorInput {
        setup_long,
        setup_short,
        setup_quality_long,
        setup_quality_short,
        confirm_long,
        confirm_short,
        confirm_quality_long,
        confirm_quality_short,
        close,
        long_stop,
        short_stop,
        completed_bi_dir,
        atr_fast,
        atr_slow,
    };
    let cfg = StructureFactorConfig {
        window,
        lam0,
        rho_min,
        rho_max,
        d_eps,
        max_hold,
        trailing,
        trail_activation,
        ratchet_stop,
        min_hold,
        cooldown,
        track_signal: false,
        reentry_mode: reentry_mode.into(),
        reentry_breakout_buf,
        partial_exit_ratio,
        breakeven_buffer,
        runner_trail_mult,
        zs_zg,
        zs_zd,
        completed_fx_dir,
        soft_exit,
    };
    result_dict(
        py,
        engine::evaluate_structure_factor(&input, &cfg).map_err(value_error)?,
    )
}

/// Already-confirmed-trigger or continuous-weight entry using the identical shared structural exit transition.
#[gen_stub_pyfunction]
#[pyfunction(name="evaluate_structure_decay", signature=(direction, quality, close, stop_level, reverse_dir, atr_fast, atr_slow,
    lam0=0.10, rho_min=0.5, rho_max=2.0, d_eps=0.02, max_hold=64, trailing=0.0,
    trail_activation=0.0, ratchet_stop=false, min_hold=0, cooldown=0, track_signal=false,
    zs_zg=None, zs_zd=None, completed_fx_dir=None, soft_exit=None,
    reentry_mode="none", reentry_breakout_buf=0.0, partial_exit_ratio=0.0, breakeven_buffer=0.0, runner_trail_mult=1.5))]
#[allow(clippy::too_many_arguments)]
fn evaluate_structure_decay_py(
    py: Python<'_>,
    direction: Vec<i8>,
    quality: Vec<f64>,
    close: Vec<f64>,
    stop_level: Vec<f64>,
    reverse_dir: Vec<i8>,
    atr_fast: Vec<f64>,
    atr_slow: Vec<f64>,
    lam0: f64,
    rho_min: f64,
    rho_max: f64,
    d_eps: f64,
    max_hold: usize,
    trailing: f64,
    trail_activation: f64,
    ratchet_stop: bool,
    min_hold: usize,
    cooldown: usize,
    track_signal: bool,
    zs_zg: Option<Vec<f64>>,
    zs_zd: Option<Vec<f64>>,
    completed_fx_dir: Option<Vec<i8>>,
    soft_exit: Option<Vec<bool>>,
    reentry_mode: &str,
    reentry_breakout_buf: f64,
    partial_exit_ratio: f64,
    breakeven_buffer: f64,
    runner_trail_mult: f64,
) -> PyResult<Py<PyDict>> {
    let cfg = StructureFactorConfig {
        lam0,
        rho_min,
        rho_max,
        d_eps,
        max_hold,
        trailing,
        trail_activation,
        ratchet_stop,
        min_hold,
        cooldown,
        track_signal,
        reentry_mode: reentry_mode.into(),
        reentry_breakout_buf,
        partial_exit_ratio,
        breakeven_buffer,
        runner_trail_mult,
        zs_zg,
        zs_zd,
        completed_fx_dir,
        soft_exit,
        ..Default::default()
    };
    result_dict(
        py,
        engine::evaluate_structure_decay(
            &direction,
            &quality,
            &close,
            &stop_level,
            &reverse_dir,
            &atr_fast,
            &atr_slow,
            &cfg,
        )
        .map_err(value_error)?,
    )
}

/// Trigger-only compatibility entry with strict Setup-before-Confirmation ordering.
#[gen_stub_pyfunction]
#[pyfunction(name="two_stage_trigger", signature=(setup, setup_quality, confirm, confirm_quality, window))]
fn two_stage_trigger_py(
    setup: Vec<bool>,
    setup_quality: Vec<f64>,
    confirm: Vec<bool>,
    confirm_quality: Vec<f64>,
    window: usize,
) -> PyResult<(Vec<bool>, Vec<f64>)> {
    engine::two_stage_trigger(&setup, &setup_quality, &confirm, &confirm_quality, window)
        .map_err(value_error)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StructureSnapshot>()?;
    m.add_function(wrap_pyfunction!(structure_snapshots_py, m)?)?;
    m.add_function(wrap_pyfunction!(structure_stop_anchors_py, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_structure_factor_py, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_structure_decay_py, m)?)?;
    m.add_function(wrap_pyfunction!(two_stage_trigger_py, m)?)?;
    Ok(())
}
