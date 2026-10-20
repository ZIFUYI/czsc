// 结构强度信号（`strength_*`）：把缠论结构与规则信号的“质量”写入 `Signal.score`。
//
// 约定：`v1 ∈ {看多, 看空, 其他}` 表示方向 `d`；`score = round(100·Q)`，`Q ∈ (0, 1]`
// 为结构质量连续分（`其他` 时 score 固定为 0）。Python 端以 `±score/100` 解码为
// `[-1, 1]` 连续强度。
//
// 设计铁律：
// - 所有公式只使用价差 / ATR 比值、笔斜率与实体面积的绝对比值，保证价格镜像
//   （`p' = M - p`）后多空严格对称；
// - 只读取当前 `CZSC` 状态（≤ 当前 bar），天然零前视；
// - 只使用已完成笔（`bars_ubi < 5` 时排除仍可能延伸的最后一笔）。

use crate::params::ParamView;
use crate::types::TaCache;
use crate::utils::cxt::{fx_power_str, get_zs_seq, raw_bar_lower, raw_bar_upper};
use crate::utils::sig::{get_sub_elements, get_usize_param, make_kline_signal_scored};
use czsc_core::analyze::CZSC;
use czsc_core::analyze::utils::check_fx;
use czsc_core::objects::bar::RawBar;
use czsc_core::objects::bi::BI;
use czsc_core::objects::direction::Direction;
use czsc_core::objects::mark::Mark;
use czsc_core::objects::signal::Signal;
use czsc_signal_macros::signal;

const EPS: f64 = 1e-9;
const LONG: &str = "看多";
const SHORT: &str = "看空";
const NONE: &str = "其他";

fn true_range(cur: &RawBar, prev_close: f64) -> f64 {
    (cur.high - cur.low)
        .max((cur.high - prev_close).abs())
        .max((cur.low - prev_close).abs())
}

/// 最近 `n` 根 K 线的简单平均真实波幅（需要 `n + 1` 根 K 线）。
fn atr_sma(bars: &[RawBar], n: usize) -> Option<f64> {
    if n == 0 || bars.len() < n + 1 {
        return None;
    }
    let tail = &bars[bars.len() - n - 1..];
    let sum: f64 = tail.windows(2).map(|w| true_range(&w[1], w[0].close)).sum();
    Some(sum / n as f64)
}

fn q_to_score(q: f64) -> i32 {
    if !q.is_finite() {
        return 1;
    }
    (q.clamp(0.01, 1.0) * 100.0).round() as i32
}

fn emit(k1: &str, k2: &str, k3: &str, dir: &str, q: f64) -> Vec<Signal> {
    make_kline_signal_scored(k1, k2, k3, dir, q_to_score(q))
}

fn none(k1: &str, k2: &str, k3: &str) -> Vec<Signal> {
    make_kline_signal_scored(k1, k2, k3, NONE, 0)
}

/// `dt` 之后（不含）已经走过的 K 线根数。
fn bars_since(bars: &[RawBar], dt: chrono::DateTime<chrono::Utc>) -> usize {
    bars.iter().rev().take_while(|b| b.dt > dt).count()
}

/// 已完成笔：`bars_ubi < 5` 时，最后一笔仍可能延伸，需要排除。
fn finished_bis(c: &CZSC) -> &[BI] {
    let n = c.bi_list.len();
    if n == 0 {
        return &c.bi_list[..0];
    }
    if c.bars_ubi.len() < 5 {
        &c.bi_list[..n - 1]
    } else {
        &c.bi_list[..]
    }
}

fn body_area(bars: &[RawBar]) -> f64 {
    bars.iter().map(|b| (b.close - b.open).abs()).sum()
}

/// 分型极值K线的“反向影线占比”：底分型取最低价K线的下影线 / 振幅，顶分型取最高价K线的上影线 / 振幅。
///
/// 价格并列时两侧都取**最先出现**的极值K线：`Iterator::min_by` 并列返回第一个，
/// 而 `max_by` 并列返回最后一个，所以顶分型这一侧需要反向迭代，才能保证镜像对称。
fn extreme_shadow_ratio(mark: &Mark, raw: &[RawBar]) -> f64 {
    let ratio = |x: &RawBar, shadow: f64| shadow / (x.high - x.low).max(EPS);
    match mark {
        Mark::D => raw
            .iter()
            .min_by(|x, y| x.low.total_cmp(&y.low))
            .map(|x| ratio(x, raw_bar_lower(x))),
        Mark::G => raw
            .iter()
            .rev()
            .max_by(|x, y| x.high.total_cmp(&y.high))
            .map(|x| ratio(x, raw_bar_upper(x))),
    }
    .unwrap_or(0.0)
}

/// strength_fx：分型强度
///
/// 参数模板：`"{freq}_N{n}_FX强度V260601"`
///
/// 信号逻辑：
/// 1. 仅在“最新一根 K 线刚好完成分型第三根”时触发（事件型，不重复触发）；
/// 2. 底分型 `看多` / 顶分型 `看空`；
/// 3. `Q = clip(0.5·tanh(ΔP/ATR_n) + 0.3·pwr + 0.2·shadow, 0.01, 1)`：
///    `ΔP` 为第三根收盘突破分型中间K线的深度（底分型 `close - k2.high`，顶分型
///    `k2.low - close`，负值按 0），`pwr` 为 `强/中/弱 → 1.0/0.6/0.3`，
///    `shadow` 为分型极值K线的反向影线占比。
///
/// 信号列表示例：`Signal('15分钟_N14_FX强度V260601_看多_任意_任意_73')`
///
/// 参数说明：`n`：ATR 窗口，默认 `14`。
#[signal(
    category = "kline",
    name = "strength_fx",
    template = "{freq}_N{n}_FX强度V260601",
    opcode = "StrengthFx",
    param_kind = "StrengthFxV260601"
)]
pub fn strength_fx(c: &CZSC, params: &ParamView, _cache: &mut TaCache) -> Vec<Signal> {
    let n = get_usize_param(params, "n", 14).max(2);
    let k1 = c.freq.to_string();
    let k2 = format!("N{n}");
    let k3 = "FX强度V260601";

    let u = &c.bars_ubi;
    let (Some(last), true) = (c.bars_raw.last(), u.len() >= 3) else {
        return none(&k1, &k2, k3);
    };
    let (a, b, d) = (&u[u.len() - 3], &u[u.len() - 2], &u[u.len() - 1]);
    if d.elements.last().map(|x| x.dt) != Some(last.dt) {
        return none(&k1, &k2, k3);
    }
    let Some(fx) = check_fx(a, b, d) else {
        return none(&k1, &k2, k3);
    };
    let Some(atr) = atr_sma(&c.bars_raw, n).filter(|x| *x > EPS) else {
        return none(&k1, &k2, k3);
    };

    let (dir, delta) = match fx.mark {
        Mark::D => (LONG, last.close - b.high),
        Mark::G => (SHORT, b.low - last.close),
        #[allow(unreachable_patterns)]
        _ => return none(&k1, &k2, k3),
    };
    let shadow = extreme_shadow_ratio(&fx.mark, &b.elements);
    let pwr = match fx_power_str(&fx) {
        "强" => 1.0,
        "中" => 0.6,
        _ => 0.3,
    };
    let q = 0.5 * (delta.max(0.0) / atr).tanh() + 0.3 * pwr + 0.2 * shadow.clamp(0.0, 1.0);
    emit(&k1, &k2, k3, dir, q)
}

/// strength_bi_div：笔背驰强度
///
/// 参数模板：`"{freq}_N{n}_笔背驰强度V260601"`
///
/// 信号逻辑：
/// 1. 取最近已完成笔 `b0` 与前一同向笔 `b2`，且 `b0` 创出新极值（向下笔新低 / 向上笔新高）；
/// 2. `R_slope = |slope0| / |slope2|`，`R_area = Σ|close-open|(b0) / Σ|close-open|(b2)`；
/// 3. `min(R_slope, R_area) < 1` 才触发，`Q_div = clip(1 - min(R_slope, R_area), 0.1, 1.0)`；
/// 4. 向下笔 `看多`（底背驰），向上笔 `看空`（顶背驰）；仅在笔结束后 `n` 根 K 线内有效。
///
/// 参数说明：`n`：笔结束后的有效 K 线根数，默认 `10`。
#[signal(
    category = "kline",
    name = "strength_bi_div",
    template = "{freq}_N{n}_笔背驰强度V260601",
    opcode = "StrengthBiDiv",
    param_kind = "StrengthBiDivV260601"
)]
pub fn strength_bi_div(c: &CZSC, params: &ParamView, _cache: &mut TaCache) -> Vec<Signal> {
    let n = get_usize_param(params, "n", 10).max(1);
    let k1 = c.freq.to_string();
    let k2 = format!("N{n}");
    let k3 = "笔背驰强度V260601";

    let bis = finished_bis(c);
    if bis.len() < 3 {
        return none(&k1, &k2, k3);
    }
    let b0 = &bis[bis.len() - 1];
    let b2 = &bis[bis.len() - 3];
    if b0.direction != b2.direction || bars_since(&c.bars_raw, b0.fx_b.dt) > n {
        return none(&k1, &k2, k3);
    }
    let new_extreme = match b0.direction {
        Direction::Down => b0.get_low() < b2.get_low(),
        Direction::Up => b0.get_high() > b2.get_high(),
    };
    if !new_extreme {
        return none(&k1, &k2, k3);
    }
    let r_slope = b0.get_slope().abs() / b2.get_slope().abs().max(EPS);
    let r_area = body_area(&b0.get_raw_bars()) / body_area(&b2.get_raw_bars()).max(EPS);
    let m = r_slope.min(r_area);
    if m.is_nan() || m >= 1.0 {
        return none(&k1, &k2, k3);
    }
    let dir = if b0.direction == Direction::Down {
        LONG
    } else {
        SHORT
    };
    emit(&k1, &k2, k3, dir, (1.0 - m).clamp(0.1, 1.0))
}

/// strength_zs_revert：中枢偏离回归强度
///
/// 参数模板：`"{freq}_K{k}_中枢回归强度V260601"`
///
/// 信号逻辑：
/// 1. 用最近 20 根已完成笔重建中枢序列，取最新有效中枢（≥3 笔）；
/// 2. `Z = (P - M) / (0.5·(ZG - ZD) + ε)`，`M` 为中枢中轴，`P` 为最新收盘；
/// 3. 仅 `|Z| ≤ 1.2` 触发，输出 `-tanh(k·Z)`：`Z > 0`（偏上）`看空`，`Z < 0`（偏下）`看多`，
///    `Q = tanh(k·|Z|)`。
///
/// 参数说明：`k`：陡峭度的 10 倍整数，默认 `15`（即 `k = 1.5`）。
#[signal(
    category = "kline",
    name = "strength_zs_revert",
    template = "{freq}_K{k}_中枢回归强度V260601",
    opcode = "StrengthZsRevert",
    param_kind = "StrengthZsRevertV260601"
)]
pub fn strength_zs_revert(c: &CZSC, params: &ParamView, _cache: &mut TaCache) -> Vec<Signal> {
    let k = get_usize_param(params, "k", 15).max(1);
    let k1 = c.freq.to_string();
    let k2 = format!("K{k}");
    let k3 = "中枢回归强度V260601";

    let sub = get_sub_elements(
        finished_bis(c),
        1,
        czsc_core::structure_exit::ZS_REVERSION_BI_WINDOW,
    );
    let (Some(last), true) = (c.bars_raw.last(), sub.len() >= 3) else {
        return none(&k1, &k2, k3);
    };
    let zss = get_zs_seq(sub);
    let Some(zs) = zss.iter().rev().find(|z| z.is_valid() && z.bis.len() >= 3) else {
        return none(&k1, &k2, k3);
    };
    let z = (last.close - zs.zz) / (0.5 * (zs.zg - zs.zd) + EPS);
    if !z.is_finite() || z.abs() > 1.2 {
        return none(&k1, &k2, k3);
    }
    let q = (k as f64 / 10.0 * z.abs()).tanh();
    if q < 0.005 {
        return none(&k1, &k2, k3);
    }
    let dir = if z > 0.0 { SHORT } else { LONG };
    emit(&k1, &k2, k3, dir, q)
}

/// strength_zs_break：中枢脱离（三买/三卖）强度
///
/// 参数模板：`"{freq}_N{n}_中枢脱离强度V260601"`
///
/// 信号逻辑：
/// 1. 取最近 12 根已完成笔，最后一笔 `pullback`、倒数第二笔 `breakout`，其前序列重建中枢；
/// 2. 三买：`breakout` 向上且最高价 > ZG，`pullback` 向下且最低价 > ZG；三卖镜像（ZD）；
/// 3. `ext = 脱离幅度 / ATR14`，`margin = 回踩安全边际 / ATR14`，
///    `Q = tanh(ext) · clip(margin, 0.1, 1.0)`；
/// 4. 仅在回踩笔结束后 `n` 根 K 线内有效。
///
/// 参数说明：`n`：回踩笔结束后的有效 K 线根数，默认 `10`。
#[signal(
    category = "kline",
    name = "strength_zs_break",
    template = "{freq}_N{n}_中枢脱离强度V260601",
    opcode = "StrengthZsBreak",
    param_kind = "StrengthZsBreakV260601"
)]
pub fn strength_zs_break(c: &CZSC, params: &ParamView, _cache: &mut TaCache) -> Vec<Signal> {
    let n = get_usize_param(params, "n", 10).max(1);
    let k1 = c.freq.to_string();
    let k2 = format!("N{n}");
    let k3 = "中枢脱离强度V260601";

    let sub = get_sub_elements(finished_bis(c), 1, 12);
    if sub.len() < 5 {
        return none(&k1, &k2, k3);
    }
    let pullback = &sub[sub.len() - 1];
    let breakout = &sub[sub.len() - 2];
    if bars_since(&c.bars_raw, pullback.fx_b.dt) > n {
        return none(&k1, &k2, k3);
    }
    let zss = get_zs_seq(&sub[..sub.len() - 2]);
    let Some(zs) = zss.iter().rev().find(|z| z.is_valid() && z.bis.len() >= 3) else {
        return none(&k1, &k2, k3);
    };
    let Some(atr) = atr_sma(&c.bars_raw, 14).filter(|x| *x > EPS) else {
        return none(&k1, &k2, k3);
    };

    let (dir, ext, margin) = if pullback.direction == Direction::Down
        && breakout.direction == Direction::Up
        && breakout.get_high() > zs.zg
        && pullback.get_low() > zs.zg
    {
        (
            LONG,
            (breakout.get_high() - zs.zg) / atr,
            (pullback.get_low() - zs.zg) / atr,
        )
    } else if pullback.direction == Direction::Up
        && breakout.direction == Direction::Down
        && breakout.get_low() < zs.zd
        && pullback.get_high() < zs.zd
    {
        (
            SHORT,
            (zs.zd - breakout.get_low()) / atr,
            (zs.zd - pullback.get_high()) / atr,
        )
    } else {
        return none(&k1, &k2, k3);
    };
    emit(&k1, &k2, k3, dir, ext.tanh() * margin.clamp(0.1, 1.0))
}

/// strength_flow：量价资金流强度
///
/// 参数模板：`"{freq}_N{n}K{k}_资金流强度V260601"`
///
/// 信号逻辑：
/// 1. 取最近 `n` 根，`flow = Σ(CLV·vol) / Σvol`，`CLV = (2C - L - H) / (H - L)`；
/// 2. `cost = Σ(close·vol) / Σvol` 为成交量加权成本中枢，`dev = (C - cost) / ATR_n`；
/// 3. 方向取 `sign(flow)`，`Q = 0.6·tanh(k·|flow|) + 0.4·tanh(max(sign·dev, 0))`。
///
/// 参数说明：`n`：窗口，默认 `20`；`k`：陡峭度的 10 倍整数，默认 `20`（`k = 2.0`）。
#[signal(
    category = "kline",
    name = "strength_flow",
    template = "{freq}_N{n}K{k}_资金流强度V260601",
    opcode = "StrengthFlow",
    param_kind = "StrengthFlowV260601"
)]
pub fn strength_flow(c: &CZSC, params: &ParamView, _cache: &mut TaCache) -> Vec<Signal> {
    let n = get_usize_param(params, "n", 20).max(2);
    let k = get_usize_param(params, "k", 20).max(1);
    let k1 = c.freq.to_string();
    let k2 = format!("N{n}K{k}");
    let k3 = "资金流强度V260601";

    let Some(atr) = atr_sma(&c.bars_raw, n).filter(|x| *x > EPS) else {
        return none(&k1, &k2, k3);
    };
    let tail = &c.bars_raw[c.bars_raw.len() - n..];
    let vol_sum: f64 = tail.iter().map(|b| b.vol).sum();
    if vol_sum <= EPS {
        return none(&k1, &k2, k3);
    }
    let flow = tail
        .iter()
        .map(|b| {
            let rng = b.high - b.low;
            let clv = if rng > EPS {
                (2.0 * b.close - b.low - b.high) / rng
            } else {
                0.0
            };
            clv * b.vol
        })
        .sum::<f64>()
        / vol_sum;
    let cost = tail.iter().map(|b| b.close * b.vol).sum::<f64>() / vol_sum;
    let (dir, sign) = if flow > EPS {
        (LONG, 1.0)
    } else if flow < -EPS {
        (SHORT, -1.0)
    } else {
        return none(&k1, &k2, k3);
    };
    let last_close = tail[tail.len() - 1].close;
    let dev = sign * (last_close - cost) / atr;
    let q = 0.6 * (k as f64 / 10.0 * flow.abs()).tanh() + 0.4 * dev.max(0.0).tanh();
    emit(&k1, &k2, k3, dir, q)
}

/// strength_squeeze：波动率压缩后扩张强度
///
/// 参数模板：`"{freq}_N{n}M{m}TH{th}_压缩扩张强度V260601"`
///
/// 信号逻辑：
/// 1. `ATR_n` 在前一根的历史分位 `rank`（最近 `m` 根滚动 ATR 中不大于它的占比）
///    `≤ th%`，视为压缩；
/// 2. 当前 K 线实体 `|C - O| / ATR_prev ≥ 0.5` 视为扩张点火，方向取实体方向；
/// 3. `Q = (1 - rank) · tanh(|C - O| / ATR_prev)`。
///
/// 参数说明：`n`：ATR 窗口，默认 `14`；`m`：分位回看，默认 `60`；`th`：压缩分位阈值（百分数），默认 `30`。
#[signal(
    category = "kline",
    name = "strength_squeeze",
    template = "{freq}_N{n}M{m}TH{th}_压缩扩张强度V260601",
    opcode = "StrengthSqueeze",
    param_kind = "StrengthSqueezeV260601"
)]
pub fn strength_squeeze(c: &CZSC, params: &ParamView, _cache: &mut TaCache) -> Vec<Signal> {
    let n = get_usize_param(params, "n", 14).max(2);
    let m = get_usize_param(params, "m", 60).max(5);
    let th = get_usize_param(params, "th", 30).clamp(1, 100);
    let k1 = c.freq.to_string();
    let k2 = format!("N{n}M{m}TH{th}");
    let k3 = "压缩扩张强度V260601";

    let bars = &c.bars_raw;
    // 需要 m + n 根 TR（含当前根）+ 1 根前收盘
    if bars.len() < m + n + 1 {
        return none(&k1, &k2, k3);
    }
    let tail = &bars[bars.len() - (m + n + 1)..];
    let tr: Vec<f64> = tail
        .windows(2)
        .map(|w| true_range(&w[1], w[0].close))
        .collect();
    // atr[j] = tr[j+1-n ..= j] 的均值，j ∈ [n-1, m+n-1]，共 m+1 个，最后一个对应当前根
    let atr: Vec<f64> = (n - 1..tr.len())
        .map(|j| tr[j + 1 - n..=j].iter().sum::<f64>() / n as f64)
        .collect();
    let prev_window = &atr[..atr.len() - 1];
    let atr_prev = prev_window[prev_window.len() - 1];
    if atr_prev <= EPS {
        return none(&k1, &k2, k3);
    }
    let rank =
        prev_window.iter().filter(|x| **x <= atr_prev).count() as f64 / prev_window.len() as f64;
    if rank * 100.0 > th as f64 {
        return none(&k1, &k2, k3);
    }
    let last = &bars[bars.len() - 1];
    let body = (last.close - last.open).abs();
    let ratio = body / atr_prev;
    if ratio < 0.5 {
        return none(&k1, &k2, k3);
    }
    let dir = if last.close > last.open { LONG } else { SHORT };
    emit(&k1, &k2, k3, dir, (1.0 - rank) * ratio.tanh())
}

/// Legacy alias; preserves the historical signal key and value exactly.
#[signal(
    category = "kline",
    name = "strength_fx_V260601",
    template = "{freq}_N{n}_FX强度V260601",
    opcode = "StrengthFxV260601",
    param_kind = "StrengthFxV260601"
)]
pub fn strength_fx_v260601(c: &CZSC, params: &ParamView, cache: &mut TaCache) -> Vec<Signal> {
    strength_fx(c, params, cache)
}

/// Legacy alias; preserves the historical signal key and value exactly.
#[signal(
    category = "kline",
    name = "strength_bi_div_V260601",
    template = "{freq}_N{n}_笔背驰强度V260601",
    opcode = "StrengthBiDivV260601",
    param_kind = "StrengthBiDivV260601"
)]
pub fn strength_bi_div_v260601(c: &CZSC, params: &ParamView, cache: &mut TaCache) -> Vec<Signal> {
    strength_bi_div(c, params, cache)
}

/// Legacy alias; preserves the historical signal key and value exactly.
#[signal(
    category = "kline",
    name = "strength_zs_revert_V260601",
    template = "{freq}_K{k}_中枢回归强度V260601",
    opcode = "StrengthZsRevertV260601",
    param_kind = "StrengthZsRevertV260601"
)]
pub fn strength_zs_revert_v260601(
    c: &CZSC,
    params: &ParamView,
    cache: &mut TaCache,
) -> Vec<Signal> {
    strength_zs_revert(c, params, cache)
}

/// Legacy alias; preserves the historical signal key and value exactly.
#[signal(
    category = "kline",
    name = "strength_zs_break_V260601",
    template = "{freq}_N{n}_中枢脱离强度V260601",
    opcode = "StrengthZsBreakV260601",
    param_kind = "StrengthZsBreakV260601"
)]
pub fn strength_zs_break_v260601(c: &CZSC, params: &ParamView, cache: &mut TaCache) -> Vec<Signal> {
    strength_zs_break(c, params, cache)
}

/// Legacy alias; preserves the historical signal key and value exactly.
#[signal(
    category = "kline",
    name = "strength_flow_V260601",
    template = "{freq}_N{n}K{k}_资金流强度V260601",
    opcode = "StrengthFlowV260601",
    param_kind = "StrengthFlowV260601"
)]
pub fn strength_flow_v260601(c: &CZSC, params: &ParamView, cache: &mut TaCache) -> Vec<Signal> {
    strength_flow(c, params, cache)
}

/// Legacy alias; preserves the historical signal key and value exactly.
#[signal(
    category = "kline",
    name = "strength_squeeze_V260601",
    template = "{freq}_N{n}M{m}TH{th}_压缩扩张强度V260601",
    opcode = "StrengthSqueezeV260601",
    param_kind = "StrengthSqueezeV260601"
)]
pub fn strength_squeeze_v260601(c: &CZSC, params: &ParamView, cache: &mut TaCache) -> Vec<Signal> {
    strength_squeeze(c, params, cache)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};
    use czsc_core::objects::bar::RawBarBuilder;
    use czsc_core::objects::freq::Freq;
    use std::collections::HashMap;

    type Ohlcv = (f64, f64, f64, f64, f64);
    type SigFn = fn(&CZSC, &ParamView, &mut TaCache) -> Vec<Signal>;

    const FUNCS: [(&str, SigFn); 6] = [
        ("fx", strength_fx),
        ("bi_div", strength_bi_div),
        ("zs_revert", strength_zs_revert),
        ("zs_break", strength_zs_break),
        ("flow", strength_flow),
        ("squeeze", strength_squeeze),
    ];

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64) / ((1u64 << 31) as f64)
    }

    /// 确定性合成行情：随机游走 + 慢周期 + 间歇性波动率压缩/扩张。
    fn synth(n: usize, seed: u64) -> Vec<Ohlcv> {
        let mut s = seed;
        let mut price = 100.0;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let regime = if (i / 80) % 3 == 0 { 0.25 } else { 1.0 };
            let drift = 0.35 * (i as f64 / 37.0).sin();
            let open = price;
            let close = open + drift + (lcg(&mut s) - 0.5) * 2.4 * regime;
            let high = open.max(close) + lcg(&mut s) * 0.9 * regime;
            let low = open.min(close) - lcg(&mut s) * 0.9 * regime;
            let vol = 1000.0 * (0.5 + lcg(&mut s) * 1.5);
            out.push((open, high, low, close, vol));
            price = close;
        }
        out
    }

    fn mirror(data: &[Ohlcv]) -> Vec<Ohlcv> {
        let hi = data.iter().map(|x| x.1).fold(f64::MIN, f64::max);
        let lo = data.iter().map(|x| x.2).fold(f64::MAX, f64::min);
        let m = hi + lo;
        data.iter()
            .map(|&(o, h, l, c, v)| (m - o, m - l, m - h, m - c, v))
            .collect()
    }

    fn raw_bars(data: &[Ohlcv]) -> Vec<RawBar> {
        let dt0 = Utc.with_ymd_and_hms(2024, 1, 1, 9, 30, 0).unwrap();
        data.iter()
            .enumerate()
            .map(|(i, &(o, h, l, c, v))| {
                RawBarBuilder::default()
                    .symbol("TEST")
                    .dt(dt0 + Duration::minutes(15 * i as i64))
                    .freq(Freq::F15)
                    .id(i as i32)
                    .open(o)
                    .close(c)
                    .high(h)
                    .low(l)
                    .vol(v)
                    .amount(c * v)
                    .build()
                    .unwrap()
            })
            .collect()
    }

    /// 逐根推进，返回每个信号在每根 K 线上的 `(v1, score)`。
    fn run_incremental(data: &[Ohlcv]) -> Vec<Vec<(String, i32)>> {
        let bars = raw_bars(data);
        let warm = 60;
        let mut c = CZSC::new(bars[..warm].to_vec(), 50, 6);
        let mut out = Vec::new();
        for (i, bar) in bars.iter().enumerate().skip(warm) {
            if i > warm {
                c.update_bar(bar.clone());
            }
            out.push(eval_all(&c));
        }
        out
    }

    fn eval_all(c: &CZSC) -> Vec<(String, i32)> {
        let p = HashMap::new();
        let view = ParamView::new(&p);
        FUNCS
            .iter()
            .map(|(_, f)| {
                let mut cache = TaCache::default();
                let s = f(c, &view, &mut cache)[0].to_string();
                let parts: Vec<&str> = s.split('_').collect();
                (parts[3].to_string(), parts[6].parse::<i32>().unwrap())
            })
            .collect()
    }

    #[test]
    fn test_legacy_aliases_are_exactly_equivalent_on_every_prefix() {
        let aliases: [SigFn; 6] = [
            strength_fx_v260601,
            strength_bi_div_v260601,
            strength_zs_revert_v260601,
            strength_zs_break_v260601,
            strength_flow_v260601,
            strength_squeeze_v260601,
        ];
        let bars = raw_bars(&synth(600, 19));
        let params = HashMap::new();
        let view = ParamView::new(&params);
        let mut c = CZSC::new(bars[..60].to_vec(), 50, 6);
        for bar in &bars[60..] {
            c.update_bar(bar.clone());
            for ((_, canonical), legacy) in FUNCS.iter().zip(aliases) {
                let actual = canonical(&c, &view, &mut TaCache::default());
                let expected = legacy(&c, &view, &mut TaCache::default());
                assert_eq!(
                    actual.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                    expected.iter().map(|s| s.to_string()).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn test_scores_in_range_and_each_signal_fires() {
        let rows = run_incremental(&synth(4000, 7));
        let mut fired = [0usize; 6];
        let mut distinct: [std::collections::BTreeSet<i32>; 6] = Default::default();
        for row in &rows {
            for (j, (v1, score)) in row.iter().enumerate() {
                if v1 == NONE {
                    assert_eq!(*score, 0, "{} 无信号时 score 必须为 0", FUNCS[j].0);
                } else {
                    assert!(v1 == LONG || v1 == SHORT);
                    assert!((1..=100).contains(score), "{} score={}", FUNCS[j].0, score);
                    fired[j] += 1;
                    distinct[j].insert(*score);
                }
            }
        }
        for j in 0..6 {
            assert!(fired[j] > 0, "{} 在合成行情上从未触发", FUNCS[j].0);
            assert!(
                distinct[j].len() > 3,
                "{} 的 score 不应只有离散几档: {:?}",
                FUNCS[j].0,
                distinct[j]
            );
        }
    }

    #[test]
    fn test_truncation_causality() {
        let data = synth(1500, 11);
        let full = run_incremental(&data);
        let warm = 60;
        for t in (warm + 40..data.len()).step_by(53) {
            let bars = raw_bars(&data[..=t]);
            let c = CZSC::new(bars, 50, 6);
            assert_eq!(
                eval_all(&c),
                full[t - warm],
                "截断到 t={t} 的信号与全量不一致"
            );
        }
    }

    fn assert_mirror_symmetric(data: &[Ohlcv], min_checked: usize) {
        let a = run_incremental(data);
        let b = run_incremental(&mirror(data));
        let flip = |v: &str| match v {
            LONG => SHORT,
            SHORT => LONG,
            _ => NONE,
        };
        let mut checked = 0;
        for (t, (ra, rb)) in a.iter().zip(b.iter()).enumerate() {
            for (j, ((va, sa), (vb, sb))) in ra.iter().zip(rb.iter()).enumerate() {
                assert_eq!(flip(va), vb, "{} t={t}: 方向未镜像 {va}/{vb}", FUNCS[j].0);
                assert!(
                    (sa - sb).abs() <= 1,
                    "{} t={t}: score {sa} vs {sb}",
                    FUNCS[j].0
                );
                if va != NONE {
                    checked += 1;
                }
            }
        }
        assert!(checked > min_checked, "镜像检验覆盖的触发太少: {checked}");
    }

    #[test]
    fn test_price_mirror_symmetry() {
        assert_mirror_symmetric(&synth(3000, 23), 20);
    }

    /// 价格并列时的极值K线选择必须镜像对称：顶分型(取最高价K线的上影线) 与镜像后的底分型(最低价K线的下影线)
    /// 在并列极值上应选中同一根“最先出现”的K线。回归：`max_by` 并列返回最后一个、`min_by` 返回第一个，
    /// 曾导致顶/底分型影线占比不一致 (由 Python 层镜像对称测试暴露)。
    #[test]
    fn test_extreme_shadow_tie_break_is_mirror_symmetric() {
        // 两根最低价/最高价相同的K线，影线长度不同
        let data: Vec<Ohlcv> = vec![
            (10.0, 12.0, 8.0, 11.0, 1000.0), // 下影线 2，上影线 1
            (10.0, 12.0, 8.0, 9.0, 1000.0),  // 下影线 1，上影线 2
        ];
        let flipped = mirror(&data);
        let raw = raw_bars(&data);
        let raw_m = raw_bars(&flipped);
        // 底分型取最低价K线 -> 第一根 (下影线 2 / 振幅 4)；镜像后的顶分型必须取回同一根
        let d = extreme_shadow_ratio(&Mark::D, &raw);
        let g_mirror = extreme_shadow_ratio(&Mark::G, &raw_m);
        assert!(
            (d - 0.5).abs() < 1e-12,
            "底分型应取第一根并列最低价K线: {d}"
        );
        assert!(
            (d - g_mirror).abs() < 1e-12,
            "顶/底并列极值选择不对称: {d} vs {g_mirror}"
        );
        // 反方向同理
        let g = extreme_shadow_ratio(&Mark::G, &raw);
        let d_mirror = extreme_shadow_ratio(&Mark::D, &raw_m);
        assert!((g - d_mirror).abs() < 1e-12, "{g} vs {d_mirror}");
    }
}
