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
  波动率放大 → 衰减加快；出现反向完成笔、收盘突破冻结的结构证伪点或触及 1.3% 跟踪阈值时
  截断归零 (D = 0)，下一开盘执行。

Setup / Confirmation 与结构规则由同一函数 `build(ctx, side)` 以 `side ∈ {+1, -1}` 生成，
按方向价差实现镜像对称。跟踪退出在两侧应用相同的 1.3% 规则；仿射价格镜像不保持百分比回撤，
因此最终持仓不承诺仿射镜像后逐值反号（分别见镜像条件与原生跟踪阈值回归）。

零前视：只使用 ≤ t 的滚动窗口 / Rust 增量信号 / 已收盘的大级别 K 线 (`merge_asof(direction="backward")`)。
"""

from __future__ import annotations

import hashlib
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
    ARCHETYPE_RULERS,
    STRENGTH_SIGNAL_KINDS,
    decode_strength,
    research_bars,
    rust_sma,
)

BI_BASE_COL = "cz_D0BL9_V230228"  # 未完成笔方向仅用于趋势状态；退出读取原生完成笔事件
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
    "cs_available",
    "cs_active_count",
)


class CrossSectionProvider(Protocol):
    """跨品种截面强弱提供者：返回与 `df` 行对齐的、因果的截面强弱 ∈ [-1, 1] (正 = 相对强势)。"""

    def strength(self, df: pd.DataFrame) -> np.ndarray: ...


class NeutralCrossSection:
    """缺失截面占位；不满足跨资产因子 FC02~FC04 的上下文要求。"""

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
            "cs_available": np.zeros(n, dtype=bool),
            "cs_active_count": np.zeros(n, dtype=np.int64),
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
        max_quote_age: str | pd.Timedelta | None = None,
    ) -> None:
        # Fixed timeframe-based age, never inferred from future gaps in the source history.
        durations = {"5m": "5min", "15m": "15min", "30m": "30min", "60m": "60min", "1d": "1D"}
        self.max_quote_age = pd.Timedelta(max_quote_age or durations.get(default_tf, "60min"))
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
        complete = all(c in df.columns for c in CS_FEATURE_COLS[:5])
        if "cs_available" not in out:
            out["cs_available"] = complete
        if "cs_active_count" not in out:
            out["cs_active_count"] = np.where(out["cs_available"], 2, 0)
        for col in ("sector_leader_dir", "leader_lag_gap", "sector_spread_zscore"):
            if col not in out.columns:
                out[col] = np.zeros(n, dtype=np.float64)
        return out

    def _select_source_df(self, target_df: pd.DataFrame) -> pd.DataFrame | None:
        # Selecting a timeframe by full-history length/equality changes when future bars are appended.
        timeframe = target_df.attrs.get("factor_timeframe")
        return self._tf_map.get(timeframe, self._default_df)

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
            merged = pd.merge_asof(left_sorted, src, on="dt", direction="backward", tolerance=self.max_quote_age)
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
            "cs_available": aligned["cs_available"].eq(True).to_numpy(dtype=bool),
            "cs_active_count": aligned["cs_active_count"].fillna(0).to_numpy(dtype=np.int64),
        }

    def rank(self, df: pd.DataFrame) -> np.ndarray:
        return self.features(df)["cs_slope_rank"]

    def strength(self, df: pd.DataFrame) -> np.ndarray:
        return self.features(df)["cs_strength"]

    def cache_fingerprint(self) -> str:
        """Fingerprint every timeframe, timestamp and feature used by this provider."""
        frames = sorted(self._tf_map.items()) if self._tf_map else [("default", self._default_df)]
        digest = hashlib.sha256(str(self.max_quote_age).encode())
        digest.update(_frame_fingerprint(self._default_df).encode())
        for tf, src in frames:
            digest.update(tf.encode())
            digest.update(_frame_fingerprint(src).encode())
        return digest.hexdigest()


def _frame_fingerprint(df: pd.DataFrame | None) -> str:
    """Content key includes OHLCV, derived/native columns, timestamps and row order."""
    if df is None:
        return "none"
    digest = hashlib.sha256(repr(tuple((str(c), str(t)) for c, t in df.dtypes.items())).encode())
    digest.update(pd.util.hash_pandas_object(df, index=True).to_numpy(dtype=np.uint64).tobytes())
    return digest.hexdigest()


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
        """只缩放风险暴露，保留每个实际发生的触发事件，避免选择性删除低权重亏损样本。"""
        trig_bool = np.asarray(triggers, dtype=bool)
        g = self.gate_multiplier(cs_rank, er, archetype=archetype, direction=direction)
        if weights is not None:
            w_arr = np.asarray(weights, dtype=np.float64)
            gated_w = self.apply(w_arr, cs_rank, er, archetype=archetype)
        else:
            gated_w = trig_bool.astype(np.float64) * float(direction) * g
        return trig_bool.copy(), gated_w

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


def attach_causal_oi_features(df_tf: pd.DataFrame, df_5m: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
    """将 5m 真实持仓量 open_interest (或本周期 open_interest) 严格因果对齐至目标周期收盘时刻，提取无量纲量仓特征。"""
    n = len(df_tf)
    if df_5m is None or df_5m.empty or "open_interest" not in df_5m.columns:
        if "open_interest" in df_tf.columns:
            oi = df_tf["open_interest"].to_numpy(dtype=np.float64)
        else:
            return {
                "oi_avail": np.zeros(n, dtype=bool),
                "oi_ratio": np.ones(n, dtype=np.float64),
                "oi_chg_z": np.zeros(n, dtype=np.float64),
                "oi_flow3_z": np.zeros(n, dtype=np.float64),
            }
    else:
        left = pd.DataFrame({"dt": pd.to_datetime(df_tf["dt"]).to_numpy(), "_ord": np.arange(n)})
        right = pd.DataFrame(
            {
                "dt": pd.to_datetime(df_5m["dt"]).to_numpy(),
                "oi": df_5m["open_interest"].to_numpy(dtype=np.float64),
            }
        ).sort_values("dt")
        merged = pd.merge_asof(left.sort_values("dt"), right, on="dt", direction="backward")
        oi = merged.sort_values("_ord")["oi"].ffill().fillna(0.0).to_numpy(dtype=np.float64)

    s_oi = pd.Series(oi)
    oi_ma20 = s_oi.rolling(20, min_periods=3).mean().replace(0.0, np.nan).to_numpy(dtype=np.float64)
    oi_ratio = np.nan_to_num(oi / oi_ma20, nan=1.0)
    oi_diff = s_oi.diff().fillna(0.0)
    oi_std = oi_diff.rolling(20, min_periods=5).std().replace(0.0, np.nan).to_numpy(dtype=np.float64)
    oi_chg_z = np.clip(np.nan_to_num(oi_diff.to_numpy(dtype=np.float64) / oi_std, nan=0.0), -4.0, 4.0)
    oi_sum3 = oi_diff.rolling(3, min_periods=1).sum().to_numpy(dtype=np.float64)
    oi_flow3_z = np.clip(np.nan_to_num(oi_sum3 / (oi_std * np.sqrt(3.0)), nan=0.0), -4.0, 4.0)
    return {
        "oi_avail": oi > 0,
        "oi_ratio": np.clip(oi_ratio, 0.5, 2.0),
        "oi_chg_z": oi_chg_z,
        "oi_flow3_z": oi_flow3_z,
    }


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
    cs_available: np.ndarray | None = None
    structure_anchors: dict = field(default_factory=dict)
    structure_results: dict = field(default_factory=dict)
    fx_flip: np.ndarray | None = None
    zs_zg: np.ndarray | None = None
    zs_zd: np.ndarray | None = None
    snapshots: list = field(default_factory=list)
    oi_avail: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    oi_ratio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    oi_chg_z: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    oi_flow3_z: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

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

        import czsc

        # RawBar serialization is boundary glue; structure discovery and dispatch live in Rust.
        bars = research_bars(df, df.attrs.get("factor_timeframe", "30m"))
        snapshots = czsc.structure_snapshots(bars)
        bi_flip = np.asarray([snapshot.completed_bi_dir for snapshot in snapshots], dtype=np.int8)
        fx_flip = np.asarray([snapshot.completed_fx_dir for snapshot in snapshots], dtype=np.int8)
        zs_zg = np.asarray([snapshot.zs_zg for snapshot in snapshots], dtype=np.float64)
        zs_zd = np.asarray([snapshot.zs_zd for snapshot in snapshots], dtype=np.float64)
        htf_ma = None
        if htf is not None and not htf.empty:
            left = pd.DataFrame({"dt": pd.to_datetime(df["dt"])})
            right = pd.DataFrame({"dt": pd.to_datetime(htf["dt"]), "ma": htf["ma20"]})
            htf_ma = pd.merge_asof(left, right, on="dt", direction="backward")["ma"].to_numpy()
        anchors = {
            arch: czsc.structure_stop_anchors(bars, snapshots, ruler.stop_plugin, htf_ma=htf_ma)
            for arch, ruler in ARCHETYPE_RULERS.items()
        }
        anchors["pos_second_ext"] = czsc.structure_stop_anchors(bars, snapshots, "pos_second_ext", htf_ma=htf_ma)
        anchors["pos_zs_reentry"] = czsc.structure_stop_anchors(bars, snapshots, "pos_zs_reentry", htf_ma=htf_ma)

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
            fx_flip=fx_flip,
            zs_zg=zs_zg,
            zs_zd=zs_zd,
            snapshots=snapshots,
            structure_anchors=anchors,
            strength={k: decode_strength(df, k) for k in STRENGTH_SIGNAL_KINDS},
        )
        ctx.states = {
            "ma_cls": _native_state(df, "分类V221203"),
            "td_v2": _native_state(df, "TD_BS辅助V221111", part=1),
            "clv": _native_state(df, "CLV多空V230605"),
            "ntmdk": _native_state(df, "NTMDK多空V230824"),
            "kcatr": _native_state(df, "KCATR多空V230823"),
            "fake_break": _native_state(df, "假突破"),
            # Stage 2 新增：182 未开发信号普查筛出的高信噪比且 100% 严格价格镜像对称 (exact_mirror=True) 原生状态
            "ma_joint": _native_state(df, "均线粘合发散V260520"),
            "double_ma": _native_state(df, "双均线_BS辅助V240208"),
            "dif_layer": _native_state(df, "完全分类V241010"),
            "dif_zero": _native_state(df, "DIF靠近零轴T50_BS辅助V240612"),
            "macd_bc_ubi": _native_state(df, "MACD背驰_UBI观察V230804"),
            "macd_bc": _native_state(df, "柱子背驰_BS辅助V240307"),
            "bar_acc": _native_state(df, "加速V240428"),
            "polyfit": _native_state(df, "分类V240428"),
            "plr_long": _native_state(df, "M多头_盈亏比V240427"),
            "plr_short": _native_state(df, "M空头_盈亏比V240427"),
            "tnr": _native_state(df, "趋势V230630"),
            "zfzd": _native_state(df, "形态V241013"),
            "overlap": _native_state(df, "支撑压力V240612"),
            "bi_end_618": _native_state(df, "BE辅助V230618"),
            "zdy_dif": _native_state(df, "DIF远离V230528"),
            "macd_shrink": _native_state(df, "MACD缩柱_BS辅助V230519"),
            "xl_candle": _native_state(df, "形态V240412"),
            "xl_trend": _native_state(df, "突破信号V240623"),
            "xl_trend_v2": _native_state(df, "突破信号V240623", part=1),
            "sar": _native_state(df, "SAR_BS辅助V230325"),
            # Stage 3 新增：量仓微结构与双顺中线 (CVOLP / Pressure / 量柱 / 布林强弱 / 双顺中线)
            "cvolp": _native_state(df, "CVOLP动量变化率V230612"),
            "pressure": _native_state(df, "支撑压力V240222"),
            "vol_ts_v1": _native_state(df, "量柱V221216"),
            "vol_ts_v2": _native_state(df, "量柱V221216", part=1),
            "vol_gd": _native_state(df, "量柱V221218"),
            "boll_pwr_v1": _native_state(df, "强弱V221112"),
            "boll_pwr_v2": _native_state(df, "强弱V221112", part=1),
            "ds_mid_v1": _native_state(df, "双顺中线突破V260521"),
            "ds_mid_v2": _native_state(df, "双顺中线突破V260521", part=1),
        }
        oi_feats = attach_causal_oi_features(df, _unbox(df, "oi_5m"))
        ctx.oi_avail = oi_feats["oi_avail"]
        ctx.oi_ratio = oi_feats["oi_ratio"]
        ctx.oi_chg_z = oi_feats["oi_chg_z"]
        ctx.oi_flow3_z = oi_feats["oi_flow3_z"]
        if htf is not None:
            ctx.htf_dir, ctx.htf_q = _align_htf_state(df, htf)

        # 解析跨品种因果截面特征 (支持外部 CrossSectionProvider 或 df 已挂载的截面特征列)
        has_cs_cols = any(col in df.columns for col in ("cs_slope_rank", "sector_leader_dir", "sector_spread_zscore"))
        if cross_section is None and has_cs_cols:
            cross_section = DataFrameCrossSectionProvider(df)

        is_real_cs = cross_section is not None and not isinstance(cross_section, NeutralCrossSection)
        ctx.has_external_cs = bool(is_real_cs and hasattr(cross_section, "features"))

        if is_real_cs and hasattr(cross_section, "features"):
            feats = cross_section.features(df)  # type: ignore[union-attr]
            ctx.cs_rank = feats["cs_slope_rank"]
            ctx.xs = feats["cs_strength"]
            ctx.cs_strength_eff = feats["cs_strength"]
            ctx.sector_leader_dir = feats["sector_leader_dir"]
            ctx.leader_lag_gap = feats["leader_lag_gap"]
            ctx.sector_spread_zscore = feats["sector_spread_zscore"]
            ctx.cs_available = np.asarray(feats.get("cs_available", np.ones(n, dtype=bool)), dtype=bool)
        else:
            # A scalar relative-strength provider is insufficient evidence for sector leaders/spreads.
            ctx.xs = (
                np.clip(np.nan_to_num(cross_section.strength(df), nan=0.0), -1.0, 1.0)
                if is_real_cs
                else np.zeros(n, dtype=np.float64)
            )
            ctx.cs_rank = 0.5 * (ctx.xs + 1.0)
            ctx.cs_strength_eff = np.zeros(n, dtype=np.float64)
            ctx.sector_leader_dir = np.zeros(n, dtype=np.float64)
            ctx.leader_lag_gap = np.zeros(n, dtype=np.float64)
            ctx.sector_spread_zscore = np.zeros(n, dtype=np.float64)
            ctx.cs_available = np.zeros(n, dtype=bool)
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
    from czsc import two_stage_trigger as native_trigger

    fire, quality = native_trigger(setup, setup_q, confirm, confirm_q, window)
    return np.asarray(fire, dtype=bool), np.asarray(quality, dtype=np.float64)


@dataclass(frozen=True)
class DecayConfig:
    lam0: float = 0.10  # 基础衰减率 (ρ=1 时半衰期 ≈ ln2/λ0 ≈ 7 根)
    rho_min: float = 0.5
    rho_max: float = 2.0
    d_eps: float = 0.02  # D 低于此值视为耗尽，清除状态
    max_hold: int = 64
    trailing: float = 0.013
    trail_activation: float = 0.0
    ratchet_stop: bool = False
    min_hold: int = 0
    cooldown: int = 0
    track_signal: bool = False
    reentry_mode: str = "none"
    reentry_breakout_buf: float = 0.0
    partial_exit_ratio: float = 0.0
    breakeven_buffer: float = 0.0
    runner_trail_mult: float = 1.5
    use_zs_reentry: bool = False
    use_fx_exit: bool = False
    stop_plugin: str | None = None


def apply_dynamic_structure_decay(
    direction: np.ndarray,
    quality: np.ndarray,
    close: np.ndarray,
    stop_level: np.ndarray,
    reverse_dir: np.ndarray,
    atr_fast: np.ndarray,
    atr_slow: np.ndarray,
    cfg: DecayConfig | None = None,
    *,
    zs_zg: np.ndarray | None = None,
    zs_zd: np.ndarray | None = None,
    completed_fx_dir: np.ndarray | None = None,
    soft_exit: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    动态结构衰减机 (替代固定 half_life 指数衰减)：返回 (S, trigger)，S = d·Q·D ∈ [-1, 1]。

    每根 K 线 t 依次：
    1. 若持有 d0：出现反向笔确认 (`reverse_dir[t] == -d0`) 或收盘跌破证伪点 → D = 0；否则
       `D *= exp(-λ0 · clip(ATR_fast/ATR_slow, ρ_min, ρ_max))`，超过 `max_hold` 或 `D < d_eps` 清空；
    2. 若有新触发 (`direction[t] != 0`)：反向、空仓或强度不弱于当前残余强度时刷新状态 (D=1)。
    """
    from czsc import evaluate_structure_decay

    cfg = cfg or DecayConfig()
    pass_fx = cfg.use_fx_exit or (cfg.reentry_mode not in ("", "none"))
    result = evaluate_structure_decay(
        direction,
        quality,
        close,
        stop_level,
        reverse_dir,
        atr_fast,
        atr_slow,
        cfg.lam0,
        cfg.rho_min,
        cfg.rho_max,
        cfg.d_eps,
        cfg.max_hold,
        cfg.trailing,
        trail_activation=cfg.trail_activation,
        ratchet_stop=cfg.ratchet_stop,
        min_hold=cfg.min_hold,
        cooldown=cfg.cooldown,
        track_signal=cfg.track_signal,
        reentry_mode=cfg.reentry_mode,
        reentry_breakout_buf=cfg.reentry_breakout_buf,
        partial_exit_ratio=cfg.partial_exit_ratio,
        breakeven_buffer=cfg.breakeven_buffer,
        runner_trail_mult=cfg.runner_trail_mult,
        zs_zg=zs_zg if cfg.use_zs_reentry else None,
        zs_zd=zs_zd if cfg.use_zs_reentry else None,
        completed_fx_dir=completed_fx_dir if pass_fx else None,
        soft_exit=soft_exit,
    )
    return np.asarray(result["strength"]), np.asarray(result["trigger"], dtype=bool)


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


def _build_ac03(ctx, side):  # A: 多均线共振粘合发散/双均线蓄势 × K线二阶加速/通道连涨确认 (Stage 2)
    j_label = "看多" if side > 0 else "看空"
    dm_label = "多头" if side > 0 else "空头"
    acc_label = "上涨" if side > 0 else "下跌"
    xl_label = "做多" if side > 0 else "做空"
    xl_v2 = "连续2次上涨" if side > 0 else "连续2次下跌"
    slope_s = np.nan_to_num(side * (ctx.ma20 - _shift(ctx.ma20, 5)) / (5.0 * _safe_atr(ctx)), nan=0.0)
    setup = ((ctx.states["ma_joint"] == j_label) | (ctx.states["double_ma"] == dm_label)) & (
        side * (ctx.c - ctx.ma20) > 0
    )
    setup_q = np.clip(0.55 + 0.45 * np.tanh(2.5 * np.maximum(slope_s, 0.0)), 0.15, 1.0)
    c_s, o_s, _, _ = _adj(ctx, side)
    body = np.maximum(c_s - o_s, 0.0)
    confirm = ((ctx.states["bar_acc"] == acc_label) | (ctx.states["xl_trend"] == xl_label)) & (c_s > o_s)
    bonus = np.where(ctx.states["xl_trend_v2"] == xl_v2, 0.20, 0.0)
    confirm_q = np.clip(np.tanh(body / _safe_atr(ctx)) + bonus, 0.10, 1.0)
    return setup, setup_q, confirm, confirm_q, 6


def _build_ac04(ctx, side):  # A: 中枢重叠支撑压力 + SAR 同向蓄势 × 长蜡烛实体/二次拟合加速突破 (Stage 2)
    ov_label = "支撑" if side > 0 else "压力"
    sar_label = "多头" if side > 0 else "空头"
    candle_label = "看涨长蜡烛" if side > 0 else "看跌长蜡烛"
    poly_acc = "加速上涨" if side > 0 else "加速下跌"
    zs_ok, zs_q = _dir_eq(ctx, "zs_break", side)
    slope_s = np.nan_to_num(side * (ctx.ma20 - _shift(ctx.ma20, 5)) / (5.0 * _safe_atr(ctx)), nan=0.0)
    setup = (ctx.states["sar"] == sar_label) & ((ctx.states["overlap"] == ov_label) | zs_ok | (slope_s > 0.08))
    setup_q = np.clip(np.maximum(zs_q, 0.45 + 0.55 * np.tanh(2.5 * np.maximum(slope_s, 0.0))), 0.15, 1.0)
    brk, brk_q = _break_confirm(ctx, side)
    confirm = (ctx.states["xl_candle"] == candle_label) | ((ctx.states["polyfit"] == poly_acc) & brk)
    confirm_q = np.clip(np.maximum(brk_q, 0.55), 0.15, 1.0)
    return setup, setup_q, confirm, confirm_q, 5


def _build_ac05(ctx, side):  # A: 双顺中枢中线突破 + 布林强势与 CVOLP/增仓动量共振 (Stage 3)
    ds_label = "看多" if side > 0 else "看空"
    bp_label = "多头" if side > 0 else "空头"
    slope_s = np.nan_to_num(side * (ctx.ma20 - _shift(ctx.ma20, 5)) / (5.0 * _safe_atr(ctx)), nan=0.0)
    setup = (
        (ctx.states["ds_mid_v1"] == ds_label)
        & (ctx.states["boll_pwr_v1"] == bp_label)
        & np.isin(ctx.states["boll_pwr_v2"], ["强势", "超强"])
        & (ctx.er20 >= 0.22)
        & (slope_s > 0.03)
    )
    setup_q = np.clip(0.55 + 0.45 * np.tanh(2.5 * np.maximum(slope_s, 0.0)), 0.25, 1.0)
    brk, brk_q = _break_confirm(ctx, side)
    c_s, o_s, _, _ = _adj(ctx, side)
    vol_oi_push = (ctx.oi_flow3_z > 0.15) | ((ctx.states["cvolp"] == "看多") & (ctx.states["vol_ts_v1"] == "梯量"))
    confirm = brk & vol_oi_push & (c_s > o_s)
    confirm_q = np.clip(0.60 * brk_q + 0.40 * np.tanh(np.maximum(ctx.oi_ratio - 0.8, 0.2)), 0.20, 1.0)
    return setup, setup_q, confirm, confirm_q, 5


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


def _build_bc03(ctx, side):  # B: MACD 柱子背驰/UBI背驰 + 连续缩柱蓄势 × 缠论笔终点反转确认 (Stage 2)
    bc_label = "底背驰" if side > 0 else "顶背驰"
    ubi_label = "多头" if side > 0 else "空头"
    shrink_label = "空头连续缩柱" if side > 0 else "多头连续缩柱"
    be_label = "看多" if side > 0 else "看空"
    c_s, o_s, _, _ = _adj(ctx, side)
    depth = np.nan_to_num(np.maximum(side * (ctx.ma20 - ctx.c), 0.0) / (1.8 * _safe_atr(ctx)), nan=0.0)
    div_ok, div_q = _dir_eq(ctx, "bi_div", side)
    setup = (
        (ctx.states["macd_bc"] == bc_label)
        | (ctx.states["macd_bc_ubi"] == ubi_label)
        | ((ctx.states["macd_shrink"] == shrink_label) & (depth > 0.25))
    )
    setup_q = np.clip(np.maximum(div_q, 0.45 + 0.55 * np.tanh(depth)), 0.15, 1.0)
    fx_ok, fx_q = _dir_eq(ctx, "fx", side)
    body = np.maximum(c_s - o_s, 0.0)
    confirm = (ctx.states["bi_end_618"] == be_label) & ((c_s > o_s) | fx_ok)
    confirm_q = np.clip(np.maximum(fx_q, np.tanh(body / _safe_atr(ctx))), 0.15, 1.0)
    return setup, setup_q, confirm, confirm_q, 5


def _build_bc04(ctx, side):  # B: 持仓量急降砍仓/逼空出清与天量派发背驰反转 (Stage 3)
    bc_label = "底背驰" if side > 0 else "顶背驰"
    be_label = "看多" if side > 0 else "看空"
    div_ok, div_q = _dir_eq(ctx, "bi_div", side)
    fx_ok, fx_q = _dir_eq(ctx, "fx", side)
    depth = np.nan_to_num(np.maximum(side * (ctx.ma20 - ctx.c), 0.0) / (1.6 * _safe_atr(ctx)), nan=0.0)
    oi_exhaust = (
        (ctx.oi_flow3_z < -0.25)
        | (ctx.states["vol_gd"] == "高量柱")
        | ((ctx.states["cvolp"] == "看空") & (ctx.oi_chg_z < -0.10))
    )
    setup = ((div_ok | (ctx.states["macd_bc"] == bc_label)) & oi_exhaust) | (
        (depth > 1.25) & ((ctx.states["vol_gd"] == "高量柱") | (ctx.oi_flow3_z < -0.45))
    )
    setup_q = np.clip(np.maximum(div_q, 0.45 + 0.55 * np.tanh(depth)), 0.20, 1.0)
    c_s, o_s, _, _ = _adj(ctx, side)
    body = np.maximum(c_s - o_s, 0.0)
    confirm = ((fx_ok & (fx_q >= 0.30)) | (ctx.states["bi_end_618"] == be_label)) & (c_s > o_s)
    confirm_q = np.clip(np.maximum(fx_q, np.tanh(body / _safe_atr(ctx))), 0.20, 1.0)
    return setup, setup_q, confirm, confirm_q, 5


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


def _build_cc03(ctx, side):  # C: DIF 分层远离/靠近零轴蓄势 × 二阶多项式拟合减速拐点回归 (Stage 2)
    layer_opp = "空头远离" if side > 0 else "多头远离"
    zero_label = "买点" if side > 0 else "卖点"
    poly_decel = "减速下跌" if side > 0 else "减速上涨"
    c_s, o_s, _, _ = _adj(ctx, side)
    stretch = np.nan_to_num(np.maximum(side * (ctx.ma20 - ctx.c), 0.0) / (1.5 * _safe_atr(ctx)), nan=0.0)
    rev_ok, rev_q = _dir_eq(ctx, "zs_revert", side)
    setup = ((ctx.states["dif_layer"] == layer_opp) & (ctx.er20 < 0.48)) | (ctx.states["dif_zero"] == zero_label)
    setup_q = np.clip(np.maximum(rev_q, 0.40 + 0.60 * np.tanh(stretch)), 0.15, 1.0)
    body = np.maximum(c_s - o_s, 0.0)
    confirm = (ctx.states["polyfit"] == poly_decel) & (c_s > o_s) & (c_s > _shift(c_s))
    confirm_q = np.clip(np.tanh(body / _safe_atr(ctx)) + 0.15, 0.15, 1.0)
    return setup, setup_q, confirm, confirm_q, 4


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


def _build_dc03(ctx, side):  # D: PLR 分段线性回归盈亏比满足 + DIF 远离蓄势 × 放量突破脉冲 (Stage 2)
    plr_key = "plr_long" if side > 0 else "plr_short"
    dif_label = "多头远离" if side > 0 else "空头远离"
    clv_label = "看多" if side > 0 else "看空"
    flow_ok, fq = _dir_eq(ctx, "flow", side)
    vr = np.divide(ctx.v, ctx.vma20, out=np.ones(ctx.n), where=ctx.vma20 > EPS)
    brk, brk_q = _break_confirm(ctx, side)
    setup = (ctx.states[plr_key] == "满足") & ((ctx.states["zdy_dif"] == dif_label) | flow_ok)
    setup_q = np.clip(np.maximum(np.where(flow_ok, fq, 0.35), 0.45), 0.15, 1.0)
    confirm = brk & ((vr >= 1.10) | (ctx.states["clv"] == clv_label))
    confirm_q = np.clip(0.5 * brk_q + 0.5 * np.tanh(np.maximum(vr - 0.85, 0.0)), 0.15, 1.0)
    return setup, setup_q, confirm, confirm_q, 5


def _build_dc04(ctx, side):  # D: 筹码动量 CVOLP/黄金柱蓄势 + 持仓量增仓梯量突破共振 (Stage 3)
    ov_label = "支撑" if side > 0 else "压力"
    bp_label = "多头" if side > 0 else "空头"
    clv_label = "看多" if side > 0 else "看空"
    ts_v2_label = "价升" if side > 0 else "价跌"
    flow_ok, fq = _dir_eq(ctx, "flow", side)
    vr = np.divide(ctx.v, ctx.vma20, out=np.ones(ctx.n), where=ctx.vma20 > EPS)
    oi_acc = np.where(ctx.oi_avail, ctx.oi_flow3_z > 0.20, vr >= 1.05)
    vol_setup = (ctx.states["cvolp"] == "看多") & (np.isin(ctx.states["vol_gd"], ["高量黄金柱", "低量柱"]) | flow_ok)
    dir_setup = (ctx.states["overlap"] == ov_label) | (ctx.states["boll_pwr_v1"] == bp_label)
    setup = vol_setup & dir_setup & oi_acc
    setup_q = np.clip(
        0.50 * np.maximum(np.where(flow_ok, fq, 0.35), 0.40) + 0.50 * np.tanh(np.maximum(ctx.oi_flow3_z, 0.2)),
        0.20,
        1.0,
    )
    brk, brk_q = _break_confirm(ctx, side)
    ti_confirm = (ctx.states["vol_ts_v1"] == "梯量") & (ctx.states["vol_ts_v2"] == ts_v2_label)
    oi_push = np.where(ctx.oi_avail, ctx.oi_chg_z > 0.10, vr >= 1.15)
    confirm = brk & (ti_confirm | (oi_push & (vr >= 1.15) & (ctx.states["clv"] == clv_label)))
    confirm_q = np.clip(0.55 * np.maximum(brk_q, 0.50) + 0.45 * np.tanh(np.maximum(vr - 0.8, 0.1)), 0.20, 1.0)
    return setup, setup_q, confirm, confirm_q, 5


def _build_dc05(ctx, side):  # D: 支撑压力位缩量减仓洗盘出清 + 双顺中线/筹码回流脉冲 (Stage 3)
    pr_label = "支撑位" if side > 0 else "压力位"
    ov_label = "支撑" if side > 0 else "压力"
    ds_label = "看多" if side > 0 else "看空"
    clv_label = "看多" if side > 0 else "看空"
    flow_ok, fq = _dir_eq(ctx, "flow", side)
    vr = np.divide(ctx.v, ctx.vma20, out=np.ones(ctx.n), where=ctx.vma20 > EPS)
    at_level = (ctx.states["pressure"] == pr_label) | (ctx.states["overlap"] == ov_label)
    vol_dry = (ctx.states["vol_ts_v1"] == "缩量") | (ctx.states["vol_gd"] == "低量柱") | (vr < 0.82)
    oi_wash = (ctx.oi_flow3_z < -0.15) | (ctx.states["cvolp"] == "看空")
    setup = at_level & vol_dry & oi_wash
    wash_q = np.clip(0.55 + 0.45 * np.tanh(np.maximum(-ctx.oi_flow3_z, 0.2)), 0.25, 1.0)
    brk, brk_q = _break_confirm(ctx, side)
    c_s, o_s, _, _ = _adj(ctx, side)
    re_inflow = (ctx.oi_chg_z > 0.05) | (ctx.states["cvolp"] == "看多") | flow_ok | (ctx.states["clv"] == clv_label)
    confirm = ((ctx.states["ds_mid_v1"] == ds_label) & (c_s > o_s) & re_inflow) | (brk & re_inflow & (vr >= 0.95))
    confirm_q = np.clip(0.50 * np.maximum(brk_q, np.where(flow_ok, fq, 0.45)) + 0.50 * np.tanh(vr), 0.20, 1.0)
    return setup, wash_q, confirm, confirm_q, 5


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


def _build_ec03(ctx, side):  # E: TNR 趋势信噪比噪音减少 + 窄幅震荡压缩 × 二阶加速突破点火 (Stage 2)
    poly_acc = "加速上涨" if side > 0 else "加速下跌"
    xl_label = "做多" if side > 0 else "做空"
    sq_ok, sq_q = _dir_eq(ctx, "squeeze", side)
    brk, brk_q = _break_confirm(ctx, side)
    setup = (ctx.states["tnr"] == "噪音减少") & ((ctx.states["zfzd"] == "满足") | (ctx.atr_rank <= 0.45))
    setup_q = np.clip(0.55 * (1.0 - ctx.atr_rank) + 0.45 * np.clip(1.0 - ctx.er20, 0.2, 1.0), 0.15, 1.0)
    confirm = ((ctx.states["polyfit"] == poly_acc) & brk) | ((ctx.states["xl_trend"] == xl_label) & brk) | sq_ok
    confirm_q = np.clip(np.maximum(brk_q, sq_q), 0.15, 1.0)
    return setup, setup_q, confirm, confirm_q, 6


def _build_fc01(ctx: FactorContext, side: int):
    """FC01: 大级别结构锚定 × 截面强弱 × 小级别分型扳机。"""
    n = ctx.n
    xs = ctx.xs if ctx.xs is not None else np.zeros(n, dtype=np.float64)
    if ctx.htf_dir is not None:
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
    cs_ok = ((side * xs) >= 0.26) if ctx.has_external_cs else ((side * xs) >= -0.45)
    setup = (hdir == side) & cs_ok
    win = 4 if ctx.has_external_cs else 3
    return (
        setup,
        np.clip(hq * mult, 0.25 if ctx.has_external_cs else 0.0, 1.0),
        fx_ok & (fx_q >= 0.3),
        np.clip(fx_q, 0.25 if ctx.has_external_cs else 0.0, 1.0),
        win,
    )


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
    exhaust_ok = div_ok | rev_ok | (exhaust_depth > 0.50)

    setup = (spread_stretch > 1.28) & exhaust_ok
    ex_q = np.maximum(np.maximum(div_q, rev_q), np.tanh(0.6 * exhaust_depth))
    setup_q = np.clip(0.55 * np.tanh(0.65 * np.maximum(spread_stretch - 0.8, 0.0)) + 0.45 * ex_q, 0.15, 1.0)

    fx_ok, fx_q = _dir_eq(ctx, "fx", side)
    brk_ok, brk_q = _break_confirm(ctx, side)
    confirm = (fx_ok & (fx_q >= 0.35)) | brk_ok
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
    default_decay: DecayConfig | None = None


_ALL_TFS = ("1d", "60m", "30m", "15m", "5m")
_BREAKOUT_TRAIL_DECAY = DecayConfig(
    lam0=0.04,
    trailing=0.012,
    trail_activation=0.0020,
    use_fx_exit=True,
    ratchet_stop=True,
    max_hold=24,
)
_SQUEEZE_ACCEL_DECAY = DecayConfig(
    lam0=0.10,
    max_hold=64,
    trailing=0.012,
    ratchet_stop=True,
)
_HTF_RESONANCE_DECAY = DecayConfig(
    lam0=0.08,
    max_hold=56,
    trailing=0.012,
    use_fx_exit=True,
    ratchet_stop=True,
)
_SPREAD_REVERT_DECAY = DecayConfig(
    lam0=0.10,
    max_hold=48,
    trailing=0.012,
    ratchet_stop=True,
)
ATOMIC_FACTOR_DEFS: dict[str, AtomicFactorDef] = {
    d.family: d
    for d in (
        AtomicFactorDef("AC01", "三买卖中枢脱离安全边际突破确认", ARCH_A, _build_ac01, _ALL_TFS),
        AtomicFactorDef("AC02", "均线斜率动量唐奇安突破", ARCH_A, _build_ac02, _ALL_TFS, _BREAKOUT_TRAIL_DECAY),
        AtomicFactorDef("AC03", "多均线共振双均线蓄势与斜率加速突破", ARCH_A, _build_ac03, _ALL_TFS),
        AtomicFactorDef("AC04", "中枢重叠支撑压力与SAR长实体突破", ARCH_A, _build_ac04, _ALL_TFS),
        AtomicFactorDef(
            "AC05", "双顺中枢中线突破与布林强势量仓共振", ARCH_A, _build_ac05, _ALL_TFS, _BREAKOUT_TRAIL_DECAY
        ),
        AtomicFactorDef("BC01", "笔背驰衰竭分型确认", ARCH_B, _build_bc01, _ALL_TFS),
        AtomicFactorDef("BC02", "TD九转极值2B假突破", ARCH_B, _build_bc02, _ALL_TFS),
        AtomicFactorDef("BC03", "MACD背驰缩柱蓄势与笔终点反转确认", ARCH_B, _build_bc03, _ALL_TFS),
        AtomicFactorDef("BC04", "持仓量急降砍仓逼空出清与高量柱背驰反转", ARCH_B, _build_bc04, _ALL_TFS),
        AtomicFactorDef("CC01", "中枢偏离回归", ARCH_C, _build_cc01, _ALL_TFS),
        AtomicFactorDef("CC02", "KC轨道刺破收回", ARCH_C, _build_cc02, _ALL_TFS),
        AtomicFactorDef("CC03", "DIF分层远离超跌超涨与二阶拟合减速回归", ARCH_C, _build_cc03, _ALL_TFS),
        AtomicFactorDef("DC01", "量价资金流CLV共振", ARCH_D, _build_dc01, _ALL_TFS),
        AtomicFactorDef("DC02", "NTMDK-KCATR多空共振放量突破", ARCH_D, _build_dc02, _ALL_TFS),
        AtomicFactorDef("DC03", "PLR分段线性回归与DIF动量放量脉冲", ARCH_D, _build_dc03, _ALL_TFS),
        AtomicFactorDef("DC04", "CVOLP筹码扩张黄金柱蓄势与增仓梯量突破", ARCH_D, _build_dc04, _ALL_TFS),
        AtomicFactorDef("DC05", "支撑压力位缩量减仓洗盘与双顺中线筹码回流", ARCH_D, _build_dc05, _ALL_TFS),
        AtomicFactorDef("EC01", "挤压后波动率扩张点火", ARCH_E, _build_ec01, _ALL_TFS),
        AtomicFactorDef("EC02", "NR7窄幅突破", ARCH_E, _build_ec02, _ALL_TFS),
        AtomicFactorDef(
            "EC03", "TNR趋势信噪比降噪窄幅压缩与二阶加速点火", ARCH_E, _build_ec03, _ALL_TFS, _SQUEEZE_ACCEL_DECAY
        ),
        AtomicFactorDef(
            "FC01",
            "大级别结构锚定小级别分型扳机",
            ARCH_F,
            _build_fc01,
            ("60m", "30m", "15m", "5m"),
            _HTF_RESONANCE_DECAY,
        ),
        AtomicFactorDef("FC02", "F01_板块龙头共振补涨补跌", ARCH_F, _build_fc02, _ALL_TFS),
        AtomicFactorDef("FC03", "F02_截面比价超跌超涨背驰反包", ARCH_F, _build_fc03, _ALL_TFS, _SPREAD_REVERT_DECAY),
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
    if resolved_family in {"FC02", "FC03", "FC04"} and not ctx.has_external_cs:
        raise ValueError(f"missing_context: {resolved_family} requires real cross-asset rank/leader/spread features")
    from czsc import evaluate_structure_factor

    inputs = {}
    for side in (1, -1):
        setup, setup_q, confirm, confirm_q, window = fdef.build(ctx, side)
        setup, confirm = np.asarray(setup, dtype=bool), np.asarray(confirm, dtype=bool)
        if side not in sides:
            setup, confirm = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        if resolved_family in {"FC02", "FC03", "FC04"} and ctx.cs_available is not None:
            setup, confirm = setup & ctx.cs_available, confirm & ctx.cs_available
        inputs[side] = (setup, np.asarray(setup_q, dtype=np.float64), confirm, np.asarray(confirm_q, dtype=np.float64))
    long, short = inputs[1], inputs[-1]
    cfg = decay or fdef.default_decay or DecayConfig()
    plugin_key = cfg.stop_plugin if (cfg.stop_plugin and cfg.stop_plugin in ctx.structure_anchors) else fdef.archetype
    anchors = ctx.structure_anchors[plugin_key]
    pass_fx = cfg.use_fx_exit or (cfg.reentry_mode not in ("", "none"))
    result = evaluate_structure_factor(
        long[0],
        short[0],
        long[1],
        short[1],
        long[2],
        short[2],
        long[3],
        short[3],
        ctx.c,
        anchors["long_stop"],
        anchors["short_stop"],
        ctx.bi_flip,
        ctx.atr_fast,
        ctx.atr_slow,
        window,
        cfg.lam0,
        cfg.rho_min,
        cfg.rho_max,
        cfg.d_eps,
        cfg.max_hold,
        cfg.trailing,
        trail_activation=cfg.trail_activation,
        ratchet_stop=cfg.ratchet_stop,
        min_hold=cfg.min_hold,
        cooldown=cfg.cooldown,
        reentry_mode=cfg.reentry_mode,
        reentry_breakout_buf=cfg.reentry_breakout_buf,
        partial_exit_ratio=cfg.partial_exit_ratio,
        breakeven_buffer=cfg.breakeven_buffer,
        runner_trail_mult=cfg.runner_trail_mult,
        zs_zg=ctx.zs_zg if cfg.use_zs_reentry else None,
        zs_zd=ctx.zs_zd if cfg.use_zs_reentry else None,
        completed_fx_dir=ctx.fx_flip if pass_fx else None,
    )
    # Join IDs to native-frozen Setup indexes; no Python exit/state-machine algorithm.
    result["structure_id"] = [
        anchors["long_id" if d > 0 else "short_id"][i] if i >= 0 else -1
        for i, d in zip(result["setup_index"], result["position_direction"], strict=True)
    ]
    result["stop_plugin"] = cfg.stop_plugin or ARCHETYPE_RULERS[fdef.archetype].stop_plugin
    ctx.structure_results[(family, sides)] = result
    s, trig = np.asarray(result["strength"]), np.asarray(result["trigger"], dtype=bool)

    if soft_gate is not None and soft_gate is not False and ctx.cs_rank is not None:
        gate_obj = soft_gate if isinstance(soft_gate, SoftCrossSectionalGate) else SoftCrossSectionalGate()
        s = gate_obj.apply(s, ctx.cs_rank, ctx.er20, archetype=fdef.archetype)
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
    oi_5m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """给周期 K 线挂载所需的外部上下文 (大级别 K 线 / 截面强弱提供者 / 软空间门控 / 5m 持仓量)，原地修改并返回。"""
    if htf is not None:
        df.attrs["htf_df"] = _AttrBox(htf)
    if cross_section is not None:
        df.attrs["cross_section"] = _AttrBox(cross_section)
    if soft_gate is not None:
        df.attrs["soft_gate"] = _AttrBox(soft_gate)
    if oi_5m is not None:
        df.attrs["oi_5m"] = _AttrBox(oi_5m)
    return df


def link_higher_timeframes(tf_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    把每个周期的 K 线 DataFrame 链接到「下一个更高的可用周期」，供 F 簇多周期共振因子使用。
    大级别状态在 `compute_atomic_signal` 内按其收盘时刻因果对齐。原地修改并返回。
    """
    present = [tf for tf in TF_ORDER if tf in tf_dfs]
    oi_src = tf_dfs.get("5m") if ("5m" in tf_dfs and "open_interest" in tf_dfs["5m"].columns) else None
    for tf in present:
        tf_dfs[tf].attrs["factor_timeframe"] = tf
        if oi_src is not None and tf != "5m" and "open_interest" not in tf_dfs[tf].columns:
            set_factor_context(tf_dfs[tf], oi_5m=oi_src)
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
    `set_factor_context` / `link_higher_timeframes` 挂载；FC02~FC04 缺少真实截面时明确拒绝计算。
    soft_gate=False 明确禁止门控，None 使用挂载的门控；注册中心统一持有门控时应传 False。
    """
    del scale  # 阈值已由 ATR 归一，周期缩放不再需要
    raw_family, side_char = sig_id[:-1], sig_id[-1]
    family = F_CLUSTER_FAMILY_ALIASES.get(raw_family, raw_family)
    if family not in ATOMIC_FACTOR_DEFS or side_char not in ("L", "S"):
        raise KeyError(f"unknown atomic factor: {sig_id}")
    htf = _unbox(df, "htf_df")
    oi_5m = _unbox(df, "oi_5m")
    cs_prov = _unbox(df, "cross_section")
    cs_fp = cs_prov.cache_fingerprint() if hasattr(cs_prov, "cache_fingerprint") else None
    import os

    # Providers without an explicit version cannot be safely cached across evaluations.
    cacheable = cs_prov is None or cs_fp is not None
    key = (
        "atomic-context-v3-native-structures",
        _frame_fingerprint(df),
        _frame_fingerprint(htf),
        _frame_fingerprint(oi_5m),
        cs_fp,
        df.attrs.get("factor_timeframe"),
        tuple(
            os.environ.get(name)
            for name in ("CZSC_MIN_BI_LEN", "czsc_min_bi_len", "CZSC_MAX_BI_NUM", "czsc_max_bi_num")
        ),
    )
    cache = _unbox(df, "_atomic_ctx")
    if not cacheable or cache is None or cache[0] != key:
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
