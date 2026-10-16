#!/usr/bin/env python
"""
6 战略簇 × 连续强度原子因子库 (Continuous-Strength Atomic Factors, Issue #3 Phase 1)

每个原子因子输出连续强度 S ∈ [-1, 1]，而非二值脉冲 {-1, 0, +1}：

    S_{i,t} = d_{i,t} · Q_{i,t} · D_{i,t}

- d ∈ {-1, 0, +1}：两阶段状态机 (Setup 势能/结构前提 → Confirmation 微观触发) 给出的方向；
- Q ∈ (0, 1]：结构质量连续分，来自 Rust `strength_*` 信号的 `Signal.score / 100`
  (FX 强度 / 笔背驰 Q_div / 中枢偏离 Z_zs / 量价资金流 / 波动率压缩扩张)，宪法第一条：核心算法在 Rust，Python 仅解码；
- D ∈ [0, 1]：动态结构衰减机，取代固定半衰期的指数衰减：
      D_t = D_{t-1} · exp(-λ0 · ρ_t),   ρ_t = clip(ATR_fast / ATR_slow, ρ_min, ρ_max)
  波动率放大 → 衰减加快；出现**反向笔确认**或收盘**跌破结构证伪点** (多头=分型最低价 / 空头=分型最高价) 时
  立即截断归零 (D = 0)。

多空 100% 对称：每个因子由同一函数 `build(ctx, side)` 以 `side ∈ {+1, -1}` 生成，所有价格比较都写成
`side · (价差)` 形式，镜像对称由构造保证 (见 tests/unit/test_six_cluster_atomic_factors.py)。

零前视：只使用 ≤ t 的滚动窗口 / Rust 增量信号 / 已收盘的大级别 K 线 (`merge_asof(direction="backward")`)。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from examples.alpha_mining_agent.hypothesis_bank import (
    ARCH_A,
    ARCH_B,
    ARCH_C,
    ARCH_D,
    ARCH_E,
    ARCH_F,
    STRENGTH_SIGNAL_KINDS,
    decode_strength,
    rust_sma,
)

BI_BASE_COL = "cz_D0BL9_V230228"  # cxt_bi_base：最新笔方向 (向上/向下)，翻转 = 反向笔确认
EPS = 1e-9


# ---------------------------------------------------------------------------
# 1. 跨品种截面强弱接口 (预留，兼容 Issue #6)
# ---------------------------------------------------------------------------
class CrossSectionProvider(Protocol):
    """跨品种截面强弱提供者：返回与 `df` 行对齐的、因果的截面强弱 ∈ [-1, 1] (正 = 相对强势)。"""

    def strength(self, df: pd.DataFrame) -> np.ndarray: ...


class NeutralCrossSection:
    """默认中性实现 (全 0)，使 F 簇在 Issue #6 落地前退化为纯多周期共振。"""

    def strength(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df), dtype=np.float64)


# ---------------------------------------------------------------------------
# 2. 因子上下文：一次性预计算所有因果特征
# ---------------------------------------------------------------------------
def _native_state(df: pd.DataFrame, suffix: str, part: int = 0) -> np.ndarray:
    """按列名后缀取 Rust 原生信号的第 `part` 段 (0=v1, 1=v2)；缺列返回全 '其他'。"""
    for col in df.columns:
        if isinstance(col, str) and col.startswith("cz_") and col.endswith(suffix):
            parts = df[col].astype(str).str.split("_")
            return parts.str[part].fillna("其他").to_numpy(dtype=object)
    return np.full(len(df), "其他", dtype=object)


@dataclass
class FactorContext:
    n: int
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray  # noqa: E741
    c: np.ndarray
    v: np.ndarray
    vma20: np.ndarray
    atr: np.ndarray
    atr_fast: np.ndarray
    atr_slow: np.ndarray
    ma20: np.ndarray
    kc_up: np.ndarray
    kc_dn: np.ndarray
    rng: np.ndarray
    in_squeeze: np.ndarray
    atr_rank: np.ndarray
    er20: np.ndarray
    bi_flip: np.ndarray  # +1: 最新笔翻转为向上；-1: 翻转为向下；0: 无
    strength: dict[str, tuple[np.ndarray, np.ndarray]]
    states: dict[str, np.ndarray] = field(default_factory=dict)
    htf_dir: np.ndarray | None = None
    htf_q: np.ndarray | None = None
    xs: np.ndarray | None = None

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        htf: pd.DataFrame | None = None,
        cross_section: CrossSectionProvider | None = None,
    ) -> FactorContext:
        n = len(df)
        f64 = lambda col, default=np.nan: (  # noqa: E731
            df[col].to_numpy(dtype=np.float64) if col in df.columns else np.full(n, default)
        )
        c = df["close"].to_numpy(dtype=np.float64)
        tr = df["tr"] if "tr" in df.columns else pd.Series(np.zeros(n), index=df.index)
        atr = np.nan_to_num(f64("atr14"), nan=0.0)
        atr_fast = np.nan_to_num(rust_sma(tr, 5).to_numpy(), nan=0.0)
        atr_slow = np.nan_to_num(rust_sma(tr, 20).to_numpy(), nan=0.0)

        net = pd.Series(c).diff(20).abs()
        path = pd.Series(c).diff().abs().rolling(20).sum()
        er20 = (net / path.replace(0.0, np.nan)).fillna(1.0).to_numpy()

        bi_v1 = (
            df[BI_BASE_COL].astype(str).str.split("_").str[0].to_numpy(dtype=object)
            if BI_BASE_COL in df.columns
            else np.full(n, "其他", dtype=object)
        )
        bi_state = np.where(bi_v1 == "向上", 1, np.where(bi_v1 == "向下", -1, 0)).astype(np.int8)
        bi_flip = np.zeros(n, dtype=np.int8)
        if n > 1:
            changed = (bi_state[1:] != bi_state[:-1]) & (bi_state[1:] != 0) & (bi_state[:-1] != 0)
            bi_flip[1:] = np.where(changed, bi_state[1:], 0)

        ctx = cls(
            n=n,
            o=df["open"].to_numpy(dtype=np.float64),
            h=df["high"].to_numpy(dtype=np.float64),
            l=df["low"].to_numpy(dtype=np.float64),
            c=c,
            v=df["volume"].to_numpy(dtype=np.float64),
            vma20=np.nan_to_num(f64("vol_ma20"), nan=0.0),
            atr=atr,
            atr_fast=atr_fast,
            atr_slow=atr_slow,
            ma20=f64("ma20"),  # 预热期保持 NaN (填 0 会在首个有效点制造巨大的假斜率)
            kc_up=f64("kc_up"),
            kc_dn=f64("kc_dn"),
            rng=f64("total_range", 0.0),
            in_squeeze=df["in_squeeze"].fillna(False).to_numpy(dtype=bool)
            if "in_squeeze" in df.columns
            else np.zeros(n, bool),
            atr_rank=np.nan_to_num(f64("atr_rank_60"), nan=1.0),
            er20=er20,
            bi_flip=bi_flip,
            strength={k: decode_strength(df, k) for k in STRENGTH_SIGNAL_KINDS},
        )
        ctx.states = {
            "ma_cls": _native_state(df, "分类V221203"),
            "td_v2": _native_state(df, "TD_BS辅助V221111", part=1),
            "clv": _native_state(df, "CLV多空V230605"),
            "ntmdk": _native_state(df, "NTMDK多空V230824"),
            "kcatr": _native_state(df, "KCATR多空V230823"),
            "fake_break": _native_state(df, "假突破"),
        }
        if htf is not None:
            ctx.htf_dir, ctx.htf_q = _align_htf_state(df, htf)
        ctx.xs = (cross_section or NeutralCrossSection()).strength(df)
        return ctx


def _align_htf_state(df_base: pd.DataFrame, df_htf: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    把大级别 (已挂载 Rust 信号的) K 线状态因果地映射到基准网格：
    大级别 K 线在其收盘时刻 `dt` 才生效，基准 bar 只看到 `dt_htf <= dt_base` 的最近一根 (backward asof)。
    """
    n = len(df_htf)
    bi = (
        df_htf[BI_BASE_COL].astype(str).str.split("_").str[0].to_numpy(dtype=object)
        if BI_BASE_COL in df_htf.columns
        else np.full(n, "其他", dtype=object)
    )
    hdir = np.where(bi == "向上", 1, np.where(bi == "向下", -1, 0)).astype(np.int8)
    bd, bq = decode_strength(df_htf, "zs_break")
    hq = np.clip(0.5 + 0.5 * np.where(bd == hdir, bq, 0.0), 0.5, 1.0)
    left = pd.DataFrame({"dt": pd.to_datetime(df_base["dt"]).to_numpy()})
    right = pd.DataFrame({"dt": pd.to_datetime(df_htf["dt"]).to_numpy(), "hdir": hdir, "hq": hq})
    merged = pd.merge_asof(left, right.sort_values("dt"), on="dt", direction="backward")
    return (
        merged["hdir"].fillna(0).to_numpy(dtype=np.int8),
        merged["hq"].fillna(0.0).to_numpy(dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# 3. 两阶段状态机 (Setup -> Confirmation) 与动态结构衰减机
# ---------------------------------------------------------------------------
def _shift(a: np.ndarray, k: int = 1, fill: float = np.nan) -> np.ndarray:
    out = np.full_like(a, fill, dtype=np.float64)
    if k < len(a):
        out[k:] = a[:-k]
    return out


def two_stage_trigger(
    setup: np.ndarray,
    setup_q: np.ndarray,
    confirm: np.ndarray,
    confirm_q: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    严禁单根 K 线静态拼接开仓：Confirmation 必须发生在**严格更早**的 Setup 之后、`window` 根之内。
    返回 (fire, q)：`q = setup_q · (0.6 + 0.4 · confirm_q)`，Setup 一经消费即解除武装。
    """
    n = len(setup)
    fire = np.zeros(n, dtype=bool)
    q = np.zeros(n, dtype=np.float64)
    armed_idx, armed_q = -1, 0.0
    for t in range(n):
        if armed_idx >= 0 and t - armed_idx > window:
            armed_idx = -1
        if confirm[t] and armed_idx >= 0:
            fire[t] = True
            q[t] = float(np.clip(armed_q * (0.6 + 0.4 * confirm_q[t]), 0.01, 1.0))
            armed_idx = -1
        if setup[t]:
            armed_idx, armed_q = t, float(setup_q[t])
    return fire, q


@dataclass(frozen=True)
class DecayConfig:
    lam0: float = 0.10  # 基础衰减率 (ρ=1 时半衰期 ≈ ln2/λ0 ≈ 7 根)
    rho_min: float = 0.5
    rho_max: float = 2.0
    d_eps: float = 0.02  # D 低于此值视为耗尽，清除状态
    max_hold: int = 64


def apply_dynamic_structure_decay(
    direction: np.ndarray,
    quality: np.ndarray,
    close: np.ndarray,
    stop_level: np.ndarray,
    reverse_dir: np.ndarray,
    atr_fast: np.ndarray,
    atr_slow: np.ndarray,
    cfg: DecayConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    动态结构衰减机 (替代固定 half_life 指数衰减)：返回 (S, trigger)，S = d·Q·D ∈ [-1, 1]。

    每根 K 线 t 依次：
    1. 若持有 d0：出现反向笔确认 (`reverse_dir[t] == -d0`) 或收盘跌破证伪点 → D = 0；否则
       `D *= exp(-λ0 · clip(ATR_fast/ATR_slow, ρ_min, ρ_max))`，超过 `max_hold` 或 `D < d_eps` 清空；
    2. 若有新触发 (`direction[t] != 0`)：反向、空仓或强度不弱于当前残余强度时刷新状态 (D=1)。
    """
    cfg = cfg or DecayConfig()
    n = len(direction)
    S = np.zeros(n, dtype=np.float64)
    trig = np.zeros(n, dtype=bool)
    d0, q0, dec, stop, held = 0, 0.0, 0.0, np.nan, 0
    for t in range(n):
        if d0 != 0:
            broke = (d0 > 0 and close[t] <= stop) or (d0 < 0 and close[t] >= stop)
            if reverse_dir[t] == -d0 or broke:
                dec = 0.0
            else:
                ratio = atr_fast[t] / atr_slow[t] if atr_slow[t] > EPS else 1.0
                rho = float(np.clip(ratio, cfg.rho_min, cfg.rho_max))
                dec *= float(np.exp(-cfg.lam0 * rho))
                held += 1
                if held > cfg.max_hold:
                    dec = 0.0
            if dec < cfg.d_eps:
                d0, q0, dec = 0, 0.0, 0.0
        if direction[t] != 0 and quality[t] > 0.0:
            cur = q0 * dec if d0 != 0 else 0.0
            if d0 == 0 or direction[t] != d0 or quality[t] >= cur:
                d0, q0, dec, stop, held = int(direction[t]), float(quality[t]), 1.0, float(stop_level[t]), 0
                trig[t] = True
        S[t] = d0 * q0 * dec
    return np.clip(S, -1.0, 1.0), trig


# ---------------------------------------------------------------------------
# 4. 六簇因子：`build(ctx, side) -> (setup, setup_q, confirm, confirm_q, window)`
# ---------------------------------------------------------------------------
def _adj(ctx: FactorContext, side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """side 视角价格：c_s/o_s 为同向收盘/开场，hi_s 为有利方向极值，lo_s 为不利方向极值。"""
    c_s, o_s = side * ctx.c, side * ctx.o
    hi_s = ctx.h if side > 0 else -ctx.l
    lo_s = ctx.l if side > 0 else -ctx.h
    return c_s, o_s, hi_s, lo_s


def _safe_atr(ctx: FactorContext) -> np.ndarray:
    return np.maximum(ctx.atr, EPS)


def _break_confirm(ctx: FactorContext, side: int) -> tuple[np.ndarray, np.ndarray]:
    """同向阳线收盘突破前一根有利极值；confirm_q = tanh(实体/ATR)。"""
    c_s, o_s, hi_s, _ = _adj(ctx, side)
    prev_hi = _shift(hi_s)
    body = c_s - o_s
    confirm = (c_s > prev_hi) & (body > 0) & (ctx.atr > 0)
    return confirm, np.tanh(np.maximum(body, 0.0) / _safe_atr(ctx))


def _dir_eq(ctx: FactorContext, kind: str, side: int) -> tuple[np.ndarray, np.ndarray]:
    d, q = ctx.strength[kind]
    return d == side, q


def _build_ac01(ctx, side):  # A: 三买/三卖脱离 + 突破确认
    setup, q = _dir_eq(ctx, "zs_break", side)
    confirm, cq = _break_confirm(ctx, side)
    return setup, q, confirm, cq, 6


def _build_ac02(ctx, side):  # A: 均线斜率动量 + 唐奇安突破
    label = "多头" if side > 0 else "空头"
    slope_s = side * (ctx.ma20 - _shift(ctx.ma20, 5)) / (5.0 * _safe_atr(ctx))
    slope_s = np.nan_to_num(slope_s, nan=0.0)
    # 原生 tas_ma_base 在收盘价恰等于均线时只归为“空头”(并列不对称)，这里再加严格不等式，使并列两侧都不触发
    setup = (ctx.states["ma_cls"] == label) & (slope_s > 0.05) & (side * (ctx.c - ctx.ma20) > 0)
    c_s, o_s, hi_s, _ = _adj(ctx, side)
    donchian = pd.Series(_shift(hi_s)).rolling(10).max().to_numpy()
    body = c_s - o_s
    confirm = (c_s > donchian) & (body > 0)
    return setup, np.clip(np.tanh(3.0 * slope_s), 0.05, 1.0), confirm, np.tanh(np.maximum(body, 0) / _safe_atr(ctx)), 5


def _build_bc01(ctx, side):  # B: 笔背驰 Q_div + 分型确认
    setup, q = _dir_eq(ctx, "bi_div", side)
    fx_ok, fx_q = _dir_eq(ctx, "fx", side)
    return setup, q, fx_ok & (fx_q >= 0.3), fx_q, 4


def _build_bc02(ctx, side):  # B: TD 九转极值 + 2B 假突破
    td_label = "TD底" if side > 0 else "TD顶"
    fb_label = "看多" if side > 0 else "看空"
    c_s, o_s, _, _ = _adj(ctx, side)
    depth = np.nan_to_num(np.maximum(side * (ctx.ma20 - ctx.c), 0.0) / (2.0 * _safe_atr(ctx)), nan=0.0)
    setup = ctx.states["td_v2"] == td_label
    confirm = ctx.states["fake_break"] == fb_label
    body = np.maximum(c_s - o_s, 0.0)
    return setup, np.clip(np.tanh(depth), 0.1, 1.0), confirm, np.tanh(body / _safe_atr(ctx)), 6


def _build_cc01(ctx, side):  # C: 中枢偏离回归 -tanh(kZ) + 低趋势效率 + 反转 K
    setup, q = _dir_eq(ctx, "zs_revert", side)
    setup = setup & (ctx.er20 < 0.4)
    c_s, o_s, _, _ = _adj(ctx, side)
    confirm = (c_s > o_s) & (c_s > _shift(c_s))
    return setup, q, confirm, np.tanh(np.maximum(c_s - o_s, 0.0) / _safe_atr(ctx)), 3


def _build_cc02(ctx, side):  # C: KC 轨道刺破 + 收回箱体
    c_s, o_s, _, lo_s = _adj(ctx, side)
    band_s = ctx.kc_dn if side > 0 else -ctx.kc_up
    pierce = np.nan_to_num(band_s - lo_s, nan=-1.0)
    setup = (pierce > 0) & (ctx.er20 < 0.5)
    confirm = (c_s > band_s) & (c_s > o_s)
    return (
        setup,
        np.clip(np.tanh(np.maximum(pierce, 0) / _safe_atr(ctx)), 0.1, 1.0),
        confirm,
        np.tanh(np.maximum(c_s - o_s, 0) / _safe_atr(ctx)),
        3,
    )


def _build_dc01(ctx, side):  # D: 量价资金流 + CLV 同向 + 放量阳线
    label = "看多" if side > 0 else "看空"
    flow_ok, q = _dir_eq(ctx, "flow", side)
    c_s, o_s, _, _ = _adj(ctx, side)
    vr = np.divide(ctx.v, ctx.vma20, out=np.ones(ctx.n), where=ctx.vma20 > EPS)
    confirm = (c_s > o_s) & (vr >= 1.0)
    return flow_ok & (ctx.states["clv"] == label), q, confirm, np.tanh(np.maximum(vr - 1.0, 0.0)), 4


def _build_dc02(ctx, side):  # D: NTMDK + KC-ATR 通道多空共振 + 放量突破
    label = "看多" if side > 0 else "看空"
    flow_ok, fq = _dir_eq(ctx, "flow", side)
    vr = np.divide(ctx.v, ctx.vma20, out=np.ones(ctx.n), where=ctx.vma20 > EPS)
    brk, _ = _break_confirm(ctx, side)
    setup = (ctx.states["ntmdk"] == label) & (ctx.states["kcatr"] == label)
    return setup, np.maximum(np.where(flow_ok, fq, 0.0), 0.3), brk & (vr >= 1.2), np.tanh(np.maximum(vr - 1.0, 0.0)), 5


def _build_ec01(ctx, side):  # E: KC/BOLL 挤压 -> 波动率扩张点火
    sq_recent = pd.Series(ctx.in_squeeze.astype(float)).rolling(8, min_periods=1).max().to_numpy() > 0
    ok, q = _dir_eq(ctx, "squeeze", side)
    return sq_recent & (ctx.atr_rank <= 0.35), np.clip(1.0 - ctx.atr_rank, 0.1, 1.0), ok, q, 8


def _build_ec02(ctx, side):  # E: NR7 窄幅 -> 突破 (缺口/实体动能)
    rng = pd.Series(ctx.rng)
    nr7 = (rng <= rng.rolling(7).min()).to_numpy()
    brk, bq = _break_confirm(ctx, side)
    c_s, o_s, _, _ = _adj(ctx, side)
    big = (c_s - o_s) >= 0.6 * _safe_atr(ctx)
    setup_q = np.clip(1.0 - ctx.rng / (2.0 * _safe_atr(ctx)), 0.1, 1.0)
    return nr7, setup_q, brk & big, bq, 2


def _build_fc01(ctx, side):  # F: 大级别结构锚定 + 小级别分型扳机 (× 截面强弱乘子)
    n = ctx.n
    hdir = ctx.htf_dir if ctx.htf_dir is not None else np.zeros(n, dtype=np.int8)
    hq = ctx.htf_q if ctx.htf_q is not None else np.zeros(n)
    xs = ctx.xs if ctx.xs is not None else np.zeros(n)
    fx_ok, fx_q = _dir_eq(ctx, "fx", side)
    mult = np.clip(1.0 + 0.5 * side * xs, 0.5, 1.5)  # 中性截面 = 1.0
    return hdir == side, np.clip(hq * mult, 0.0, 1.0), fx_ok & (fx_q >= 0.3), fx_q, 3


@dataclass(frozen=True)
class AtomicFactorDef:
    family: str  # 例如 "AC01"
    name: str
    archetype: str
    build: Callable
    eligible_tfs: tuple[str, ...]


_ALL_TFS = ("1d", "60m", "30m", "15m", "5m")
ATOMIC_FACTOR_DEFS: dict[str, AtomicFactorDef] = {
    d.family: d
    for d in (
        AtomicFactorDef("AC01", "三买卖中枢脱离安全边际突破确认", ARCH_A, _build_ac01, _ALL_TFS),
        AtomicFactorDef("AC02", "均线斜率动量唐奇安突破", ARCH_A, _build_ac02, _ALL_TFS),
        AtomicFactorDef("BC01", "笔背驰衰竭分型确认", ARCH_B, _build_bc01, _ALL_TFS),
        AtomicFactorDef("BC02", "TD九转极值2B假突破", ARCH_B, _build_bc02, _ALL_TFS),
        AtomicFactorDef("CC01", "中枢偏离回归", ARCH_C, _build_cc01, _ALL_TFS),
        AtomicFactorDef("CC02", "KC轨道刺破收回", ARCH_C, _build_cc02, _ALL_TFS),
        AtomicFactorDef("DC01", "量价资金流CLV共振", ARCH_D, _build_dc01, _ALL_TFS),
        AtomicFactorDef("DC02", "NTMDK-KCATR多空共振放量突破", ARCH_D, _build_dc02, _ALL_TFS),
        AtomicFactorDef("EC01", "挤压后波动率扩张点火", ARCH_E, _build_ec01, _ALL_TFS),
        AtomicFactorDef("EC02", "NR7窄幅突破", ARCH_E, _build_ec02, _ALL_TFS),
        AtomicFactorDef("FC01", "大级别结构锚定小级别分型扳机", ARCH_F, _build_fc01, ("60m", "30m", "15m", "5m")),
    )
}
ATOMIC_SIGNAL_IDS: tuple[str, ...] = tuple(f"{fam}{side}" for fam in ATOMIC_FACTOR_DEFS for side in ("L", "S"))


# ---------------------------------------------------------------------------
# 5. 组装：Setup/Confirmation -> Q -> 动态衰减 -> S ∈ [-1, 1]
# ---------------------------------------------------------------------------
def compute_family(
    ctx: FactorContext,
    family: str,
    sides: tuple[int, ...] = (1, -1),
    decay: DecayConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """计算某因子族在给定方向集合上的连续强度 S 与触发布尔数组。"""
    fdef = ATOMIC_FACTOR_DEFS[family]
    n = ctx.n
    direction = np.zeros(n, dtype=np.int8)
    quality = np.zeros(n, dtype=np.float64)
    for side in sides:
        setup, setup_q, confirm, confirm_q, window = fdef.build(ctx, side)
        fire, q = two_stage_trigger(
            np.asarray(setup, dtype=bool),
            np.nan_to_num(np.asarray(setup_q, dtype=np.float64)),
            np.asarray(confirm, dtype=bool),
            np.nan_to_num(np.asarray(confirm_q, dtype=np.float64)),
            window,
        )
        better = fire & (q > quality)
        direction[better] = side
        quality[better] = q[better]

    # 结构证伪点：多头 = 近 3 根最低价 (底分型低点)；空头 = 近 3 根最高价
    low3 = pd.Series(ctx.l).rolling(3, min_periods=1).min().to_numpy()
    high3 = pd.Series(ctx.h).rolling(3, min_periods=1).max().to_numpy()
    stop = np.where(direction > 0, low3, high3)
    return apply_dynamic_structure_decay(
        direction, quality, ctx.c, stop, ctx.bi_flip, ctx.atr_fast, ctx.atr_slow, decay
    )


TF_ORDER: tuple[str, ...] = ("5m", "15m", "30m", "60m", "1d")


class _AttrBox:
    """`df.attrs` 的共享引用容器：pandas 在切片/赋值时会 deepcopy attrs，这里让 deepcopy 共享同一引用。"""

    def __init__(self, value):
        self.value = value

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self


def set_factor_context(
    df: pd.DataFrame,
    htf: pd.DataFrame | None = None,
    cross_section: CrossSectionProvider | None = None,
) -> pd.DataFrame:
    """给周期 K 线挂载 F 簇所需的外部上下文 (大级别 K 线 / 截面强弱提供者)，原地修改并返回。"""
    if htf is not None:
        df.attrs["htf_df"] = _AttrBox(htf)
    if cross_section is not None:
        df.attrs["cross_section"] = _AttrBox(cross_section)
    return df


def link_higher_timeframes(tf_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    把每个周期的 K 线 DataFrame 链接到「下一个更高的可用周期」，供 F 簇多周期共振因子使用。
    大级别状态在 `compute_atomic_signal` 内按其收盘时刻因果对齐。原地修改并返回。
    """
    present = [tf for tf in TF_ORDER if tf in tf_dfs]
    for i, tf in enumerate(present[:-1]):
        set_factor_context(tf_dfs[tf], htf=tf_dfs[present[i + 1]])
    return tf_dfs


def _unbox(df: pd.DataFrame, key: str):
    box = df.attrs.get(key)
    return box.value if isinstance(box, _AttrBox) else box


def compute_atomic_signal(df: pd.DataFrame, sig_id: str, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    注册表入口 (签名与 `compute_signal_triggers_and_weights` 一致)：`sig_id = 因子族 + L/S`，
    返回 (triggers, weights)，weights = 单一方向的连续强度 S。大级别 K 线 / 截面强弱提供者通过
    `set_factor_context` / `link_higher_timeframes` 挂载 (均可缺省，F 簇缺省时无触发)。
    """
    del scale  # 阈值已由 ATR 归一，周期缩放不再需要
    family, side_char = sig_id[:-1], sig_id[-1]
    if family not in ATOMIC_FACTOR_DEFS or side_char not in ("L", "S"):
        raise KeyError(f"unknown atomic factor: {sig_id}")
    htf = _unbox(df, "htf_df")
    key = (len(df), hash(df["close"].to_numpy(dtype=np.float64).tobytes()), id(htf))
    cache = _unbox(df, "_atomic_ctx")
    if cache is None or cache[0] != key:
        ctx = FactorContext.from_df(df, htf, _unbox(df, "cross_section"))
        df.attrs["_atomic_ctx"] = _AttrBox((key, ctx))
    else:
        ctx = cache[1]
    s, trig = compute_family(ctx, family, sides=(1 if side_char == "L" else -1,))
    return trig, s
