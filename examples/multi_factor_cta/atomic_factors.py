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
# 1. 跨品种截面强弱接口与连续 Sigmoid 软空间门控 (Issue #6 Phase 3)
# ---------------------------------------------------------------------------
CS_FEATURE_COLS: tuple[str, ...] = (
    "cs_slope_rank",
    "cs_strength",
    "sector_leader_dir",
    "leader_lag_gap",
    "sector_spread_zscore",
)


class CrossSectionProvider(Protocol):
    """跨品种截面强弱提供者：返回与 `df` 行对齐的、因果的截面强弱 ∈ [-1, 1] (正 = 相对强势)。"""

    def strength(self, df: pd.DataFrame) -> np.ndarray: ...


class NeutralCrossSection:
    """默认中性实现 (全 0)，使 F 簇在未挂载外部多品种截面时退化为纯多周期/跨尺度共振。"""

    def strength(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df), dtype=np.float64)

    def rank(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), 0.5, dtype=np.float64)

    def features(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        n = len(df)
        return {
            "cs_slope_rank": np.full(n, 0.5, dtype=np.float64),
            "cs_strength": np.zeros(n, dtype=np.float64),
            "sector_leader_dir": np.zeros(n, dtype=np.float64),
            "leader_lag_gap": np.zeros(n, dtype=np.float64),
            "sector_spread_zscore": np.zeros(n, dtype=np.float64),
        }


class DataFrameCrossSectionProvider:
    """
    基于预计算截面特征 DataFrame 的因果截面强弱提供者 (实现 CrossSectionProvider 协议)：
    - 支持传入单周期截面 DataFrame 或 `{tf: cs_df}` 多周期字典；
    - 通过 `pd.merge_asof(..., on="dt", direction="backward")` 将 `cs_slope_rank`、`sector_leader_dir`、
      `leader_lag_gap`、`sector_spread_zscore` 零前视映射到目标品种任意周期 K 线。
    """

    def __init__(
        self,
        cs_source: pd.DataFrame | dict[str, pd.DataFrame],
        default_tf: str | None = None,
    ) -> None:
        if isinstance(cs_source, dict):
            self._tf_map: dict[str, pd.DataFrame] = {
                tf: self._normalize_source_df(sdf) for tf, sdf in cs_source.items() if sdf is not None
            }
            self._default_df: pd.DataFrame | None = (
                self._tf_map.get(default_tf)
                if default_tf and default_tf in self._tf_map
                else (next(iter(self._tf_map.values())) if self._tf_map else None)
            )
        else:
            norm_df = self._normalize_source_df(cs_source)
            self._tf_map = {}
            self._default_df = norm_df

    @staticmethod
    def _normalize_source_df(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["dt", *CS_FEATURE_COLS])
        keep = ["dt"] + [c for c in CS_FEATURE_COLS if c in df.columns]
        out = df[keep].copy()
        out["dt"] = pd.to_datetime(out["dt"])
        out = out.sort_values("dt").drop_duplicates(subset=["dt"], keep="last").reset_index(drop=True)
        n = len(out)
        if "cs_slope_rank" not in out.columns:
            if "cs_strength" in out.columns:
                out["cs_slope_rank"] = np.clip(0.5 * (out["cs_strength"].to_numpy(dtype=np.float64) + 1.0), 0.0, 1.0)
            else:
                out["cs_slope_rank"] = np.full(n, 0.5, dtype=np.float64)
        if "cs_strength" not in out.columns:
            out["cs_strength"] = np.clip(2.0 * out["cs_slope_rank"].to_numpy(dtype=np.float64) - 1.0, -1.0, 1.0)
        for col in ("sector_leader_dir", "leader_lag_gap", "sector_spread_zscore"):
            if col not in out.columns:
                out[col] = np.zeros(n, dtype=np.float64)
        return out

    def _select_source_df(self, target_df: pd.DataFrame) -> pd.DataFrame | None:
        if self._tf_map:
            for cand in self._tf_map.values():
                if len(cand) == len(target_df) and np.array_equal(
                    cand["dt"].to_numpy(), pd.to_datetime(target_df["dt"]).to_numpy()
                ):
                    return cand
        return self._default_df

    def features(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        n = len(df)
        if n == 0:
            return {col: np.zeros(0, dtype=np.float64) for col in CS_FEATURE_COLS}
        src = self._select_source_df(df)
        if src is None or src.empty:
            return NeutralCrossSection().features(df)

        tgt_dt = pd.to_datetime(df["dt"]).to_numpy()
        src_dt = src["dt"].to_numpy()
        if len(src) == n and np.array_equal(src_dt, tgt_dt):
            aligned = src
        else:
            left = pd.DataFrame({"dt": tgt_dt, "_ord": np.arange(n)})
            left_sorted = left.sort_values("dt")
            merged = pd.merge_asof(left_sorted, src, on="dt", direction="backward")
            aligned = merged.sort_values("_ord").reset_index(drop=True)

        rank_arr = np.clip(
            np.nan_to_num(aligned["cs_slope_rank"].to_numpy(dtype=np.float64), nan=0.5),
            0.0,
            1.0,
        )
        strength_arr = np.clip(2.0 * rank_arr - 1.0, -1.0, 1.0)
        leader_dir = np.clip(
            np.nan_to_num(aligned["sector_leader_dir"].to_numpy(dtype=np.float64), nan=0.0),
            -1.0,
            1.0,
        )
        lag_gap = np.clip(
            np.nan_to_num(aligned["leader_lag_gap"].to_numpy(dtype=np.float64), nan=0.0),
            -2.0,
            2.0,
        )
        spread_z = np.clip(
            np.nan_to_num(aligned["sector_spread_zscore"].to_numpy(dtype=np.float64), nan=0.0),
            -5.0,
            5.0,
        )
        return {
            "cs_slope_rank": rank_arr,
            "cs_strength": strength_arr,
            "sector_leader_dir": leader_dir,
            "leader_lag_gap": lag_gap,
            "sector_spread_zscore": spread_z,
        }

    def rank(self, df: pd.DataFrame) -> np.ndarray:
        return self.features(df)["cs_slope_rank"]

    def strength(self, df: pd.DataFrame) -> np.ndarray:
        return self.features(df)["cs_strength"]

    def cache_fingerprint(self) -> int:
        src = self._default_df
        if src is None or src.empty:
            return 0
        return hash(
            (
                len(src),
                src["cs_slope_rank"].to_numpy(dtype=np.float64).tobytes(),
                src["sector_leader_dir"].to_numpy(dtype=np.float64).tobytes(),
                src["sector_spread_zscore"].to_numpy(dtype=np.float64).tobytes(),
            )
        )


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -40.0, 40.0)))


@dataclass(frozen=True)
class SoftCrossSectionalGate:
    """
    连续可微 Sigmoid 软空间门控 (Soft Cross-Sectional Setup Gating)：
    将「截面强弱选战场 (Setup) + 时序规则扣扳机 (Confirmation)」形式化为连续可微门控乘子
    G_gate(R_{i,t}, ER_{i,t}, archetype, dir) ∈ [0, 1]，彻底消除 0.70 / 0.30 硬截断仓位跳变：
    - B_背驰衰竭与极值反转簇：
      * 空头摸顶 (dir == -1): g_{B,short}(R) = 1 / (1 + exp(k_B * (R - 0.70)))
      * 多头抄底 (dir == +1): g_{B,long}(R)  = 1 / (1 + exp(-k_B * (R - 0.30)))
    - C_中枢震荡与锚定回归簇：
      结合截面居中度 M = 1 - |2R - 1| 与时序低效度 1 - ER，在中庸震荡区保持高权重，在极端单边趋势区平滑阻尼逆势回归信号。
    - A_趋势突破与斜率动量簇：
      在截面同向强势区保持全开/增益，在截面弱势且低 ER 震荡区温和压降防假突破磨损。
    """

    k_b: float = 12.0
    b_short_center: float = 0.70
    b_long_center: float = 0.30
    k_c: float = 10.0
    c_short_center: float = 0.70
    c_long_center: float = 0.30
    k_a: float = 8.0
    a_min_floor: float = 0.35
    trigger_gate_floor: float = 0.25

    def gate_b_short(self, cs_rank: np.ndarray | float) -> np.ndarray:
        r = np.clip(np.nan_to_num(np.asarray(cs_rank, dtype=np.float64), nan=0.5), 0.0, 1.0)
        return _sigmoid(-self.k_b * (r - self.b_short_center))

    def gate_b_long(self, cs_rank: np.ndarray | float) -> np.ndarray:
        r = np.clip(np.nan_to_num(np.asarray(cs_rank, dtype=np.float64), nan=0.5), 0.0, 1.0)
        return _sigmoid(self.k_b * (r - self.b_long_center))

    def gate_multiplier(
        self,
        cs_rank: np.ndarray | float,
        er: np.ndarray | float | None = None,
        archetype: str = ARCH_B,
        direction: int = 1,
        *,
        dir: int | None = None,  # noqa: A002
    ) -> np.ndarray:
        if dir is not None:
            direction = int(dir)
        r = np.clip(np.nan_to_num(np.asarray(cs_rank, dtype=np.float64), nan=0.5), 0.0, 1.0)
        er_arr = (
            np.full_like(r, 0.35, dtype=np.float64)
            if er is None
            else np.clip(np.nan_to_num(np.asarray(er, dtype=np.float64), nan=0.35), 0.0, 1.0)
        )
        if er_arr.shape != r.shape:
            er_arr = np.broadcast_to(er_arr, r.shape).astype(np.float64)

        if archetype == ARCH_B or str(archetype).startswith("B"):
            if direction < 0:
                return self.gate_b_short(r)
            if direction > 0:
                return self.gate_b_long(r)
            g_sym = self.gate_b_long(r) * self.gate_b_short(r)
            norm = float(self.gate_b_long(0.5) * self.gate_b_short(0.5))
            return np.clip(g_sym / max(norm, EPS), 0.0, 1.0)

        if archetype == ARCH_C or str(archetype).startswith("C"):
            k_eff = self.k_c * (0.75 + 0.25 * er_arr)
            er_shift = 0.18 * (0.50 - er_arr)
            if direction < 0:
                return _sigmoid(-k_eff * (r - (self.c_short_center + er_shift)))
            if direction > 0:
                return _sigmoid(k_eff * (r - (self.c_long_center - er_shift)))
            # 无方向对称形式：结合居中度 M = 1 - sqrt((2R-1)^2 + eps) 与时序低效度 1 - ER
            m_smooth = 1.0 - np.sqrt((2.0 * r - 1.0) ** 2 + 1e-4) + 0.01
            return _sigmoid(self.k_c * (m_smooth - 0.35 - 0.25 * er_arr))

        if archetype == ARCH_A or str(archetype).startswith("A"):
            er_boost = 0.15 * (er_arr - 0.35)
            if direction > 0:
                sig = _sigmoid(self.k_a * (r - 0.45 + er_boost))
                return self.a_min_floor + (1.0 - self.a_min_floor) * sig
            if direction < 0:
                sig = _sigmoid(-self.k_a * (r - 0.55 - er_boost))
                return self.a_min_floor + (1.0 - self.a_min_floor) * sig
            m_smooth = 1.0 - np.sqrt((2.0 * r - 1.0) ** 2 + 1e-4)
            sig = _sigmoid(self.k_a * ((1.0 - m_smooth) + 0.5 * er_arr - 0.50))
            return self.a_min_floor + (1.0 - self.a_min_floor) * sig

        if archetype in (ARCH_D, ARCH_E) or str(archetype).startswith(("D", "E")):
            if direction > 0:
                return 0.65 + 0.35 * _sigmoid(6.0 * (r - 0.40))
            if direction < 0:
                return 0.65 + 0.35 * _sigmoid(-6.0 * (r - 0.60))
            return np.ones_like(r, dtype=np.float64)

        return np.ones_like(r, dtype=np.float64)

    __call__ = gate_multiplier

    def apply(
        self,
        weights: np.ndarray,
        cs_rank: np.ndarray | float,
        er: np.ndarray | float | None = None,
        archetype: str = ARCH_B,
    ) -> np.ndarray:
        """对双向连续持仓权重序列 `weights ∈ [-1, 1]` 按多空方向分别施加连续 Sigmoid 软门控。"""
        w = np.asarray(weights, dtype=np.float64)
        g_pos = self.gate_multiplier(cs_rank, er, archetype=archetype, direction=1)
        g_neg = self.gate_multiplier(cs_rank, er, archetype=archetype, direction=-1)
        return np.where(w >= 0.0, w * g_pos, w * g_neg)

    def apply_to_triggers_and_weights(
        self,
        triggers: np.ndarray,
        weights: np.ndarray | None,
        cs_rank: np.ndarray | float,
        er: np.ndarray | float | None,
        archetype: str,
        direction: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """在法医检验与信号生成中同步对触发脉冲与连续权重施加软门控。"""
        trig_bool = np.asarray(triggers, dtype=bool)
        g = self.gate_multiplier(cs_rank, er, archetype=archetype, direction=direction)
        if weights is not None:
            w_arr = np.asarray(weights, dtype=np.float64)
            gated_w = self.apply(w_arr, cs_rank, er, archetype=archetype)
        else:
            gated_w = trig_bool.astype(np.float64) * float(direction) * g
        gated_trig = trig_bool & (g >= self.trigger_gate_floor)
        return gated_trig, gated_w

    @staticmethod
    def extract_causal_er(df: pd.DataFrame, window: int = 20) -> np.ndarray:
        """从 K 线 DataFrame 提取因果 Kaufman 效率系数 ER(window) ∈ [0, 1]。"""
        if "er20" in df.columns and window == 20:
            return np.clip(np.nan_to_num(df["er20"].to_numpy(dtype=np.float64), nan=0.35), 0.0, 1.0)
        c = df["close"].to_numpy(dtype=np.float64)
        net = pd.Series(c).diff(window).abs()
        path = pd.Series(c).diff().abs().rolling(window, min_periods=3).sum()
        return np.clip((net / path.replace(0.0, np.nan)).fillna(0.35).to_numpy(dtype=np.float64), 0.0, 1.0)

    @staticmethod
    def extract_cs_rank(df: pd.DataFrame) -> np.ndarray | None:
        """从 df 列或 df.attrs['cross_section'] 中提取截面分位排名 R_{i,t} ∈ [0, 1]；未挂载则返回 None。"""
        if "cs_slope_rank" in df.columns:
            return np.clip(np.nan_to_num(df["cs_slope_rank"].to_numpy(dtype=np.float64), nan=0.5), 0.0, 1.0)
        prov = _unbox(df, "cross_section")
        if prov is not None and not isinstance(prov, NeutralCrossSection):
            if hasattr(prov, "rank"):
                return np.clip(np.nan_to_num(prov.rank(df), nan=0.5), 0.0, 1.0)
            s = np.nan_to_num(prov.strength(df), nan=0.0)
            return np.clip(0.5 * (s + 1.0), 0.0, 1.0)
        return None


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
    cs_rank: np.ndarray | None = None
    cs_strength_eff: np.ndarray | None = None
    sector_leader_dir: np.ndarray | None = None
    leader_lag_gap: np.ndarray | None = None
    sector_spread_zscore: np.ndarray | None = None
    has_external_cs: bool = False

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

        ma20 = f64("ma20")
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
            ma20=ma20,  # 预热期保持 NaN (填 0 会在首个有效点制造巨大的假斜率)
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

        # 解析跨品种因果截面特征 (支持外部 CrossSectionProvider 或 df 已挂载的截面特征列)
        has_cs_cols = any(col in df.columns for col in ("cs_slope_rank", "sector_leader_dir", "sector_spread_zscore"))
        if cross_section is None and has_cs_cols:
            cross_section = DataFrameCrossSectionProvider(df)

        is_real_cs = cross_section is not None and not isinstance(cross_section, NeutralCrossSection)
        ctx.has_external_cs = bool(is_real_cs)

        if is_real_cs and hasattr(cross_section, "features"):
            feats = cross_section.features(df)  # type: ignore[union-attr]
            ctx.cs_rank = feats["cs_slope_rank"]
            ctx.xs = feats["cs_strength"]
            ctx.cs_strength_eff = feats["cs_strength"]
            ctx.sector_leader_dir = feats["sector_leader_dir"]
            ctx.leader_lag_gap = feats["leader_lag_gap"]
            ctx.sector_spread_zscore = feats["sector_spread_zscore"]
        elif is_real_cs:
            xs_arr = np.clip(np.nan_to_num(cross_section.strength(df), nan=0.0), -1.0, 1.0)  # type: ignore[union-attr]
            ctx.xs = xs_arr
            ctx.cs_rank = np.clip(0.5 * (xs_arr + 1.0), 0.0, 1.0)
            ctx.cs_strength_eff = xs_arr
            ctx.sector_leader_dir = xs_arr
            ctx.leader_lag_gap = np.zeros(n, dtype=np.float64)
            ctx.sector_spread_zscore = -2.0 * xs_arr
        else:
            # 单品种未挂载外部截面时：FC01 保持中性 xs=0；FC02~FC04 使用严格因果且 100% 镜像对称的跨尺度相对强弱代理
            ctx.xs = np.zeros(n, dtype=np.float64)
            ctx.cs_rank = np.full(n, 0.5, dtype=np.float64)
            satr = np.maximum(atr, EPS)
            c_prev20 = np.full(n, np.nan, dtype=np.float64)
            c_prev15 = np.full(n, np.nan, dtype=np.float64)
            c_prev3 = np.full(n, np.nan, dtype=np.float64)
            if n > 20:
                c_prev20[20:] = c[:-20]
            if n > 15:
                c_prev15[15:] = c[:-15]
            if n > 3:
                c_prev3[3:] = c[:-3]
            ns20 = np.nan_to_num((c - c_prev20) / (20.0 * satr), nan=0.0)
            ns15 = np.nan_to_num((c - c_prev15) / (15.0 * satr), nan=0.0)
            ns3 = np.nan_to_num((c - c_prev3) / (3.0 * satr), nan=0.0)
            ctx.cs_strength_eff = np.tanh(3.5 * ns15)
            ctx.sector_leader_dir = np.tanh(3.5 * ns20)
            ctx.leader_lag_gap = np.tanh(3.5 * ns20) - np.tanh(2.5 * ns3)
            ctx.sector_spread_zscore = np.nan_to_num(np.clip((c - ma20) / (0.80 * satr), -5.0, 5.0), nan=0.0)
        return ctx


def _align_htf_state(df_base: pd.DataFrame, df_htf: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    把大级别 (已挂载 Rust 信号的) K 线状态因果地映射到基准网格：
    大级别 K 线在其收盘时刻 `dt` 才生效，基准 bar 只看到 `dt_htf <= dt_base` 的最近一根 (backward asof)。
    """
    if df_htf is None or len(df_htf) == 0 or len(df_base) == 0:
        return np.zeros(len(df_base), dtype=np.int8), np.zeros(len(df_base), dtype=np.float64)
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


def _build_fc01(ctx: FactorContext, side: int):
    """FC01: 大级别结构锚定 × 截面强弱 × 小级别分型扳机。"""
    n = ctx.n
    xs = ctx.xs if ctx.xs is not None else np.zeros(n, dtype=np.float64)
    if ctx.htf_dir is not None and np.any(ctx.htf_dir != 0):
        hdir = ctx.htf_dir
        hq = ctx.htf_q if ctx.htf_q is not None else np.full(n, 0.7, dtype=np.float64)
    else:
        # 当未显式挂载更高周期 htf 时，以本周期均线斜率与截面同向强弱作为大级别结构锚定
        slope_s = np.nan_to_num(side * (ctx.ma20 - _shift(ctx.ma20, 5)) / (5.0 * _safe_atr(ctx)), nan=0.0)
        anchor_ok = (side * (ctx.c - ctx.ma20) > 0.0) & (slope_s > 0.0)
        hdir = np.where(anchor_ok, side, 0).astype(np.int8)
        hq = np.clip(0.55 + 0.45 * np.tanh(3.0 * np.maximum(slope_s, 0.0)), 0.5, 1.0)
    fx_ok, fx_q = _dir_eq(ctx, "fx", side)
    mult = np.clip(1.0 + 0.5 * side * xs, 0.5, 1.5)  # 中性截面 = 1.0
    cs_ok = (side * xs) >= -0.45  # 过滤截面极度逆势区，中性 xs=0 时恒为 True
    setup = (hdir == side) & cs_ok
    return setup, np.clip(hq * mult, 0.0, 1.0), fx_ok & (fx_q >= 0.3), fx_q, 3


def _build_fc02(ctx: FactorContext, side: int):
    """FC02 (F01_板块龙头共振补涨/补跌, Leader-Follower Resonance)。"""
    n = ctx.n
    leader_dir = ctx.sector_leader_dir if ctx.sector_leader_dir is not None else np.zeros(n, dtype=np.float64)
    lag_gap = ctx.leader_lag_gap if ctx.leader_lag_gap is not None else np.zeros(n, dtype=np.float64)
    c_s, _, _, lo_s = _adj(ctx, side)
    defense_lo = pd.Series(_shift(lo_s)).rolling(10, min_periods=2).min().to_numpy()
    struct_hold = np.nan_to_num(c_s - defense_lo, nan=0.0) >= -0.5 * _safe_atr(ctx)

    leader_s = side * leader_dir
    gap_s = side * lag_gap
    setup = (leader_s > 0.30) & (gap_s > 0.0) & struct_hold
    setup_q = np.clip(np.tanh(1.2 * np.maximum(leader_s, 0.0) + 0.8 * np.maximum(gap_s, 0.0)), 0.10, 1.0)

    brk_ok, brk_q = _break_confirm(ctx, side)
    fx_ok, fx_q = _dir_eq(ctx, "fx", side)
    confirm = brk_ok | (fx_ok & (fx_q >= 0.30))
    confirm_q = np.maximum(brk_q, fx_q)
    return setup, setup_q, confirm, confirm_q, 5


def _build_fc03(ctx: FactorContext, side: int):
    """FC03 (F02_截面比价超跌/超涨背驰反包, Cross-Sectional Spread Reversion)。"""
    n = ctx.n
    spread_z = ctx.sector_spread_zscore if ctx.sector_spread_zscore is not None else np.zeros(n, dtype=np.float64)
    spread_stretch = -side * spread_z

    div_ok, div_q = _dir_eq(ctx, "bi_div", side)
    rev_ok, rev_q = _dir_eq(ctx, "zs_revert", side)
    exhaust_depth = np.nan_to_num(np.maximum(side * (ctx.ma20 - ctx.c), 0.0) / _safe_atr(ctx), nan=0.0)
    exhaust_ok = div_ok | rev_ok | (exhaust_depth > 0.45)

    setup = (spread_stretch > 1.50) & exhaust_ok
    ex_q = np.maximum(np.maximum(div_q, rev_q), np.tanh(0.6 * exhaust_depth))
    setup_q = np.clip(0.55 * np.tanh(0.65 * np.maximum(spread_stretch - 1.0, 0.0)) + 0.45 * ex_q, 0.10, 1.0)

    fx_ok, fx_q = _dir_eq(ctx, "fx", side)
    brk_ok, brk_q = _break_confirm(ctx, side)
    confirm = (fx_ok & (fx_q >= 0.25)) | brk_ok
    confirm_q = np.maximum(fx_q, brk_q)
    return setup, setup_q, confirm, confirm_q, 5


def _build_fc04(ctx: FactorContext, side: int):
    """FC04 (F03_截面TopK斜率×时序突破双共振, CS-TS Dual Momentum Breakout)。"""
    n = ctx.n
    cs_str = ctx.cs_strength_eff if ctx.cs_strength_eff is not None else np.zeros(n, dtype=np.float64)
    cs_s = side * cs_str

    zs_ok, zs_q = _dir_eq(ctx, "zs_break", side)
    ma_slope = np.nan_to_num(side * (ctx.ma20 - _shift(ctx.ma20, 5)) / (5.0 * _safe_atr(ctx)), nan=0.0)
    ma_struct_ok = (side * (ctx.c - ctx.ma20) > 0.0) & (ma_slope > 0.0)
    struct_ok = zs_ok | ma_struct_ok

    setup = (cs_s >= 0.40) & struct_ok
    struct_q = np.maximum(zs_q, np.tanh(3.0 * np.maximum(ma_slope, 0.0)))
    setup_q = np.clip(0.60 * np.clip(cs_s, 0.0, 1.0) + 0.40 * struct_q, 0.10, 1.0)

    c_s, o_s, hi_s, _ = _adj(ctx, side)
    donchian10 = pd.Series(_shift(hi_s)).rolling(10, min_periods=3).max().to_numpy()
    body = c_s - o_s
    donch_brk = (c_s > donchian10) & (body > 0.0) & (ctx.atr > 0.0)
    donch_q = np.tanh(np.maximum(body, 0.0) / _safe_atr(ctx))
    brk_ok, brk_q = _break_confirm(ctx, side)

    confirm = donch_brk | (zs_ok & brk_ok)
    confirm_q = np.maximum(donch_q, 0.5 * (zs_q + brk_q))
    return setup, setup_q, confirm, confirm_q, 5


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
        AtomicFactorDef("FC02", "F01_板块龙头共振补涨补跌", ARCH_F, _build_fc02, _ALL_TFS),
        AtomicFactorDef("FC03", "F02_截面比价超跌超涨背驰反包", ARCH_F, _build_fc03, _ALL_TFS),
        AtomicFactorDef("FC04", "F03_截面TopK斜率时序突破双共振", ARCH_F, _build_fc04, _ALL_TFS),
    )
}
F_CLUSTER_FAMILY_ALIASES: dict[str, str] = {
    "F01": "FC02",
    "F02": "FC03",
    "F03": "FC04",
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
    soft_gate: SoftCrossSectionalGate | bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """计算某因子族在给定方向集合上的连续强度 S 与触发布尔数组。"""
    resolved_family = F_CLUSTER_FAMILY_ALIASES.get(family, family)
    fdef = ATOMIC_FACTOR_DEFS[resolved_family]
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
    s, trig = apply_dynamic_structure_decay(
        direction, quality, ctx.c, stop, ctx.bi_flip, ctx.atr_fast, ctx.atr_slow, decay
    )

    if soft_gate is not None and soft_gate is not False and ctx.cs_rank is not None:
        gate_obj = soft_gate if isinstance(soft_gate, SoftCrossSectionalGate) else SoftCrossSectionalGate()
        s = gate_obj.apply(s, ctx.cs_rank, ctx.er20, archetype=fdef.archetype)
        if len(sides) == 1:
            g_dir = gate_obj.gate_multiplier(ctx.cs_rank, ctx.er20, archetype=fdef.archetype, direction=sides[0])
            trig = trig & (g_dir >= gate_obj.trigger_gate_floor)
    return s, trig


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
    soft_gate: SoftCrossSectionalGate | bool | None = None,
) -> pd.DataFrame:
    """给周期 K 线挂载 F 簇所需的外部上下文 (大级别 K 线 / 截面强弱提供者 / 软空间门控)，原地修改并返回。"""
    if htf is not None:
        df.attrs["htf_df"] = _AttrBox(htf)
    if cross_section is not None:
        df.attrs["cross_section"] = _AttrBox(cross_section)
    if soft_gate is not None:
        df.attrs["soft_gate"] = _AttrBox(soft_gate)
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


def compute_atomic_signal(
    df: pd.DataFrame,
    sig_id: str,
    scale: float = 1.0,
    decay: DecayConfig | None = None,
    soft_gate: SoftCrossSectionalGate | bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    注册表入口 (签名与 `compute_signal_triggers_and_weights` 一致)：`sig_id = 因子族 + L/S`，
    返回 (triggers, weights)，weights = 单一方向的连续强度 S。大级别 K 线 / 截面强弱提供者通过
    `set_factor_context` / `link_higher_timeframes` 挂载 (均可缺省，F 簇缺省时退化为跨尺度共振)。
    """
    del scale  # 阈值已由 ATR 归一，周期缩放不再需要
    raw_family, side_char = sig_id[:-1], sig_id[-1]
    family = F_CLUSTER_FAMILY_ALIASES.get(raw_family, raw_family)
    if family not in ATOMIC_FACTOR_DEFS or side_char not in ("L", "S"):
        raise KeyError(f"unknown atomic factor: {sig_id}")
    htf = _unbox(df, "htf_df")
    cs_prov = _unbox(df, "cross_section")
    cs_fp = (
        cs_prov.cache_fingerprint()
        if hasattr(cs_prov, "cache_fingerprint")
        else (
            hash(df["cs_slope_rank"].to_numpy(dtype=np.float64).tobytes())
            if "cs_slope_rank" in df.columns
            else id(cs_prov)
        )
    )
    key = (len(df), hash(df["close"].to_numpy(dtype=np.float64).tobytes()), id(htf), cs_fp)
    cache = _unbox(df, "_atomic_ctx")
    if cache is None or cache[0] != key:
        ctx = FactorContext.from_df(df, htf, cs_prov)
        df.attrs["_atomic_ctx"] = _AttrBox((key, ctx))
    else:
        ctx = cache[1]
    eff_gate = soft_gate if soft_gate is not None else _unbox(df, "soft_gate")
    s, trig = compute_family(
        ctx,
        family,
        sides=(1 if side_char == "L" else -1,),
        decay=decay,
        soft_gate=eff_gate,
    )
    return trig, s
