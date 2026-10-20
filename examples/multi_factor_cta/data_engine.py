#!/usr/bin/env python
"""
第一层：跨资产统一数据底座与多周期无延迟对齐引擎 (Cross-Asset Data & Multi-Timeframe Alignment Engine)

核心职责：
1. 统一处理股指期货、日内 T+0 ETF、商品期货（含夜盘 21:00-02:30）的交易日归属 (`trading_date`)；
2. 处理期货主力合约换月跳空的比例后复权 (`adjust_roll_gaps`)，消除假突破失真；
3. 按品种交易时段内有效 K 线序列合成多级周期 (`5m / 15m / 30m / 60m / 1d`)；
4. 基于因果 ATR(14) 对所有价格空间特征进行无量纲化，并挂载 `czsc.generate_czsc_signals` 原生 Rust 信号；
5. 提供零延迟、无未来函数的跨周期阶梯持仓映射 (`map_tf_weight_to_base_grid`)。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from examples.alpha_mining_agent.hypothesis_bank import (
    TF_SCALE_MAP,
    attach_native_czsc_signals,
    enrich_kline_features,
)
from examples.multi_factor_cta.atomic_factors import (
    CS_FEATURE_COLS,
    DataFrameCrossSectionProvider,
    link_higher_timeframes,
    set_factor_context,
)


class AssetClass(str, Enum):
    EQUITY_REFERENCE = "EQUITY_REFERENCE"  # 权益指数/ETF参考行情；不宣称可直接执行双向期货交易
    INDEX_FUTURES = "INDEX_FUTURES"  # 股指期货 (IF / IH / IC / IM)
    ETF_T0 = "ETF_T0"  # 可日内 T+0 交易的宽基/行业 ETF
    COMMODITY_FUTURES = "COMMODITY_FUTURES"  # 商品期货主力连续 (含夜盘)


class ExecutionMode(str, Enum):
    LONG_SHORT = "LONG_SHORT"  # 双向多空隔夜/波段模式 [-1, +1]
    LONG_ONLY_SPOT = "LONG_ONLY_SPOT"  # 纯多头现货增强模式 [0, +1] (空头信号仅作减仓/清仓风控)
    INTRADAY_FLAT_1455 = "INTRADAY_FLAT_1455"  # 日内 T+0 强平模式 (14:45 禁新开, 14:55 全部平仓归零)


@dataclass(frozen=True)
class AssetSpec:
    """跨资产标的物理属性与制度边界配置"""

    symbol: str
    name: str
    asset_class: AssetClass
    sector: str  # 板块归属 (用于跨品种 Sector HRP 风险预算)
    base_tf: str  # 底层已收盘周期 ("5m" | "15m" | "30m" | "60m" | "1d")
    eligible_tfs: tuple[str, ...]  # 该品种参与合成的多周期列表
    fee_rate: float  # 单边综合交易成本 (手续费 + 冲击滑点)
    has_night_session: bool = False  # 是否包含夜盘 (21:00 起归属下一交易日)
    roll_gap_threshold_atr: float = 3.5  # 主力换月异常跳空后复权识别阈值
    execution_mode: ExecutionMode = ExecutionMode.LONG_SHORT
    data_path: Path | None = None
    adjust_rolls: bool = True  # 连续合约无换月标识时，研究可显式禁用启发式跳空调整


def assign_trading_date(dt_series: pd.Series, has_night_session: bool, calendar_path: Path | None = None) -> pd.Series:
    """
    计算标准交易日归属 (trading_date)：
    - 无夜盘品种 (ETF / 股指期货): 直接取自然日 dt.date
    - 含夜盘商品期货:
      1. 若提供交易所日历 `calendar.parquet`，严格将 >=18:00 的夜盘及次日凌晨 <=02:30 映射到紧邻的下一个真实交易日；
      2. 否则按周历规则：>=18:00 夜盘向后顺延 (周五夜盘 +3 天至下周一，其余 +1 天)，周六凌晨 00:00~02:30 (+2 天至下周一)。
    """
    dts = pd.to_datetime(dt_series)
    if not has_night_session:
        return dts.dt.date

    if calendar_path is not None and calendar_path.exists():
        cal = pd.read_parquet(calendar_path)
        trading_days = np.array(sorted(pd.to_datetime(cal.loc[cal["trading"], "date"]).dt.date.tolist()))
        raw_dates = dts.dt.date.values
        hours = dts.dt.hour.values
        idx_right = np.searchsorted(trading_days, raw_dates, side="right")
        idx_left = np.searchsorted(trading_days, raw_dates, side="left")
        idx = np.where(hours >= 18, idx_right, idx_left)
        idx = np.clip(idx, 0, len(trading_days) - 1)
        return pd.Series(trading_days[idx], index=dt_series.index)

    hours = dts.dt.hour
    weekdays = dts.dt.weekday
    shift_days = np.where(
        hours >= 18,
        np.where(weekdays == 4, 3, 1),
        np.where(weekdays == 5, 2, np.where(weekdays == 6, 1, 0)),
    )
    shifted = dts + pd.to_timedelta(shift_days, unit="D")
    return shifted.dt.date


def adjust_continuous_roll_gaps(df: pd.DataFrame, gap_atr_mult: float = 4.0) -> pd.DataFrame:
    """
    期货主力连续合约换月跳空平滑后复权：
    当交易日切换处出现超过 `gap_atr_mult * ATR` 的换月基差跳空时，
    计算累积比例复权因子 `adj_factor` 并对 OHLC 序列执行平滑复权，消除假突破失真。
    """
    out = df.copy()
    if out.empty:
        return out
    c = out["close"].values.astype(np.float64)
    o = out["open"].values.astype(np.float64)
    h = out["high"].values.astype(np.float64)
    low_arr = out["low"].values.astype(np.float64)

    tr = np.maximum(h - low_arr, np.abs(c - np.roll(c, 1)))
    tr[0] = h[0] - low_arr[0]
    # Gap identification uses only completed history; cold starts remain unadjusted.
    atr = pd.Series(tr).rolling(20, min_periods=1).mean().shift(1).values

    dates = out["date"].values
    day_switch = np.zeros(len(out), dtype=bool)
    day_switch[1:] = dates[1:] != dates[:-1]

    overnight_gap = np.zeros(len(out), dtype=np.float64)
    overnight_gap[1:] = np.abs(o[1:] - c[:-1])

    if "underlying_symbol" in out.columns:
        usym = out["underlying_symbol"].ffill()
        known_pair = usym.notna() & usym.shift(1).notna()
        roll_mask = (known_pair & usym.ne(usym.shift(1))).to_numpy(dtype=bool)
    else:
        roll_mask = day_switch & (overnight_gap > gap_atr_mult * np.maximum(atr, 1e-6))
    ratios = np.ones(len(out), dtype=np.float64)
    if np.any(roll_mask):
        idx_arr = np.where(roll_mask)[0]
        for idx in idx_arr:
            if c[idx - 1] > 0 and o[idx] > 0:
                ratios[idx] = o[idx] / c[idx - 1]

    cum_factor = np.cumprod(ratios)
    out["raw_close"] = c
    out["adj_factor"] = cum_factor
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].values.astype(np.float64) / cum_factor
    return out


def map_tf_weight_to_base_grid(
    df_source: pd.DataFrame,
    w_source: np.ndarray,
    df_target: pd.DataFrame,
) -> np.ndarray:
    """Hold the most recent source target whose timestamp is no later than the target bar.

    Both clocks must be unique and increasing. Source timestamps represent closed
    bars; missing target bars do not discard a previously observed source decision.
    """
    weights = np.asarray(w_source, dtype=np.float64)
    if weights.shape != (len(df_source),) or not np.isfinite(weights).all():
        raise ValueError("Source weights must be finite and aligned with source timestamps")
    target_dt = pd.DatetimeIndex(pd.to_datetime(df_target["dt"]))
    if target_dt.hasnans or target_dt.has_duplicates or not target_dt.is_monotonic_increasing:
        raise ValueError("Target timestamps must be valid, unique and increasing")
    if df_source.empty:
        return np.zeros(len(df_target), dtype=np.float64)
    source_dt = pd.DatetimeIndex(pd.to_datetime(df_source["dt"]))
    if source_dt.hasnans or source_dt.has_duplicates or not source_dt.is_monotonic_increasing:
        raise ValueError("Source timestamps must be valid, unique and increasing")
    if source_dt.tz != target_dt.tz:
        raise ValueError("Source and target timestamps must use the same timezone")
    if df_target.empty:
        return np.zeros(0, dtype=np.float64)
    if source_dt.equals(target_dt):
        return weights.copy()
    aligned = pd.merge_asof(
        pd.DataFrame({"dt": target_dt}),
        pd.DataFrame({"dt": source_dt, "weight": weights}),
        on="dt",
        direction="backward",
        allow_exact_matches=True,
    )
    return aligned["weight"].fillna(0.0).to_numpy(dtype=np.float64)


class CrossAssetDataEngine:
    """跨资产（股指期货 + ETF + 商品期货）统一数据加载与多周期特征工程引擎"""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_asset_multi_timeframes(self, spec: AssetSpec, attach_czsc: bool = True) -> dict[str, pd.DataFrame]:
        """加载单个品种的底层数据并合成其 eligible_tfs 多周期 K 线与特征"""
        tf_mins = {"5m": 5, "15m": 15, "30m": 30, "60m": 60, "1d": 240}
        if spec.base_tf not in tf_mins or any(tf not in tf_mins for tf in spec.eligible_tfs):
            raise ValueError("Unsupported base or research timeframe")
        if any(tf_mins[tf] < tf_mins[spec.base_tf] for tf in spec.eligible_tfs):
            raise ValueError("Research timeframes cannot be finer than the source bars")
        if spec.data_path is None or not spec.data_path.exists():
            raise FileNotFoundError(f"Data file not found for {spec.symbol}: {spec.data_path}")

        df_base = self._read_raw_dataframe(spec)
        cal_path = spec.data_path.parent / "calendar.parquet"
        df_base["date"] = assign_trading_date(df_base["dt"], spec.has_night_session, cal_path)
        df_base["bar_in_day"] = df_base.groupby("date", sort=False).cumcount()

        if spec.adjust_rolls and spec.asset_class in (AssetClass.INDEX_FUTURES, AssetClass.COMMODITY_FUTURES):
            df_base = adjust_continuous_roll_gaps(df_base, gap_atr_mult=spec.roll_gap_threshold_atr)

        base_mins = tf_mins[spec.base_tf]

        sym_cache_dir = (
            self.cache_dir
            if spec.symbol == "SZSE.159915"
            else self.cache_dir / spec.symbol.replace(".", "_").replace("@", "_")
        )
        sym_cache_dir.mkdir(parents=True, exist_ok=True)

        result: dict[str, pd.DataFrame] = {}
        for tf in spec.eligible_tfs:
            if tf == "1d":
                bars_per_day = max(1, tf_mins["1d"] // base_mins)
                tmp = df_base.copy()
                if len(tmp) > 0 and spec.base_tf != "1d":
                    last_date = tmp["date"].iloc[-1]
                    last_dt = pd.to_datetime(tmp["dt"].iloc[-1])
                    last_day_cnt = int((tmp["date"] == last_date).sum())
                    prev_days = tmp[tmp["date"] != last_date]
                    expected_day_bars = (
                        int(prev_days.groupby("date")["close"].count().median()) if len(prev_days) > 0 else bars_per_day
                    )
                    if (last_day_cnt < expected_day_bars and (last_dt.hour * 60 + last_dt.minute) < 15 * 60) or (
                        last_day_cnt < bars_per_day and expected_day_bars >= bars_per_day
                    ):
                        tmp = prev_days
                if len(tmp) > 0:
                    kdf = tmp.groupby("date", as_index=False, sort=False).agg(
                        dt=("dt", "last"),
                        open=("open", "first"),
                        high=("high", "max"),
                        low=("low", "min"),
                        close=("close", "last"),
                        volume=("volume", "sum"),
                    )
                    kdf["grp"] = 0
                else:
                    kdf = pd.DataFrame(columns=["date", "dt", "open", "high", "low", "close", "volume", "grp"])
            elif tf == spec.base_tf:
                kdf = df_base[["date", "dt", "open", "high", "low", "close", "volume"]].copy()
                kdf["grp"] = df_base["bar_in_day"]
            else:
                bars_per_grp = max(1, tf_mins[tf] // base_mins)
                tmp = df_base.copy()
                tmp["grp"] = tmp["bar_in_day"] // bars_per_grp
                grp_cnt = tmp.groupby(["date", "grp"], sort=False)["close"].transform("count")
                tmp = tmp[grp_cnt >= bars_per_grp]
                if len(tmp) > 0:
                    kdf = tmp.groupby(["date", "grp"], as_index=False, sort=False).agg(
                        dt=("dt", "last"),
                        open=("open", "first"),
                        high=("high", "max"),
                        low=("low", "min"),
                        close=("close", "last"),
                        volume=("volume", "sum"),
                    )
                else:
                    kdf = pd.DataFrame(columns=["date", "grp", "dt", "open", "high", "low", "close", "volume"])

            kdf = kdf.sort_values("dt").reset_index(drop=True)
            kdf["symbol"] = spec.symbol
            scale = TF_SCALE_MAP.get(tf, 1)
            kdf = enrich_kline_features(kdf, scale=scale)
            if attach_czsc and not kdf.empty:
                kdf = attach_native_czsc_signals(kdf, tf, sym_cache_dir)
            result[tf] = kdf

        return link_higher_timeframes(result)

    load_and_align_asset = load_asset_multi_timeframes

    @staticmethod
    def _read_raw_dataframe(spec: AssetSpec) -> pd.DataFrame:
        path = spec.data_path
        assert path is not None
        if path.suffix.lower() == ".csv":
            raw = pd.read_csv(path)
        else:
            warmup_path = path.parent / f"warmup_{path.name}"
            if warmup_path.exists():
                raw = pd.concat([pd.read_parquet(warmup_path), pd.read_parquet(path)], ignore_index=True)
            else:
                raw = pd.read_parquet(path)

        # 标准化列名映射
        col_map = {}
        for c in raw.columns:
            cl = c.lower()
            if cl in ("datetime", "dt", "time"):
                col_map[c] = "dt"
            elif cl.endswith(".open") or cl == "open":
                col_map[c] = "open"
            elif cl.endswith(".high") or cl == "high":
                col_map[c] = "high"
            elif cl.endswith(".low") or cl == "low":
                col_map[c] = "low"
            elif cl.endswith(".close") or cl == "close":
                col_map[c] = "close"
            elif cl.endswith(".volume") or cl in ("volume", "vol"):
                col_map[c] = "volume"

        df = raw.rename(columns=col_map)
        if np.issubdtype(df["dt"].dtype, np.integer) and df["dt"].iloc[0] > 1e15:
            df["dt"] = pd.to_datetime(df["dt"], unit="ns", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        else:
            df["dt"] = pd.to_datetime(df["dt"])

        keep_cols = ["dt", "open", "high", "low", "close", "volume"]
        if "underlying_symbol" in df.columns:
            keep_cols.append("underlying_symbol")
        df = (
            df[keep_cols]
            .dropna(subset=["open", "high", "low", "close"])
            .sort_values("dt")
            .drop_duplicates(subset=["dt"])
            .reset_index(drop=True)
        )
        # 统一截取 2020-01-01 之后的高质量区间
        df = df[df["dt"] >= pd.Timestamp("2020-01-01")].reset_index(drop=True)
        p0 = float(df["open"].iloc[0])
        if p0 > 20.0:
            norm_factor = p0 / 2.0
            for col in ("open", "high", "low", "close"):
                df[col] = df[col].astype(np.float64) / norm_factor
        return df

    @staticmethod
    def compute_causal_cross_sectional_features(
        asset_tf_dfs: Mapping[str, pd.DataFrame | Mapping[str, pd.DataFrame]],
        specs: Mapping[str, AssetSpec | str] | Sequence[AssetSpec] | None = None,
        tf: str = "60m",
        *,
        slope_window: int = 20,
        spread_window: int = 20,
        lag_window: int = 5,
        attach_to_dfs: bool = True,
        max_quote_age: str | pd.Timedelta | None = None,
        membership_masks: Mapping[str, pd.Series] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        零前视因果截面特征引擎 (Causal Cross-Sectional Feature Engine)：
        对齐全市场活跃品种在指定周期 `tf` 的已收盘时间戳面板 (严格 `merge_asof(..., direction="backward")`)，
        预计算并挂载：
        1. `cs_slope_rank` (R_{i,t} ∈ [0, 1]) 与中心化强度 `cs_strength` (X_{i,t} = 2 R_{i,t} - 1 ∈ [-1, 1])；
        2. `sector_leader_dir` (∈ [-1, 1]) 与相对龙头短期滞涨/滞跌差 `leader_lag_gap`；
        3. `sector_spread_zscore` (本品种相对于所属板块等权均值的滚动超额残差 Z-score)。
        `max_quote_age` 默认一个 tf，过期报价不进入截面；`membership_masks` 是以生效时间为索引的
        历史成员布尔序列。缺少板块元数据的品种独立分组，不假定它们属于同一板块。
        `cs_available` 表示当前有新鲜本品种报价及至少一个同板块成员，`cs_active_count` 报告有效资产数。
        """
        return CrossSectionalFeatureEngine(
            slope_window=slope_window,
            spread_window=spread_window,
            lag_window=lag_window,
            max_quote_age=max_quote_age,
        ).compute_and_attach(
            asset_tf_dfs=asset_tf_dfs,
            specs=specs,
            tf=tf,
            attach_to_dfs=attach_to_dfs,
            membership_masks=membership_masks,
        )


class CrossSectionalFeatureEngine:
    """
    跨品种零前视因果截面特征计算器：
    - 严禁任何跨期未来信息泄露及停牌/夜盘未收盘前视 (全部通过 backward merge_asof 仅读取 <= t 已收盘价格)；
    - 所有底层动量与残差均以单品种因果 ATR 无量纲标准化，严格满足全市场价格镜像翻转 P'_{i,t} = M_i - P_{i,t}
      下 `cs_slope_rank -> 1 - R_{i,t}`、`sector_leader_dir -> -sector_leader_dir`、
      `leader_lag_gap -> -leader_lag_gap`、`sector_spread_zscore -> -sector_spread_zscore` 100% 镜像对称。
    """

    def __init__(
        self,
        slope_window: int = 20,
        spread_window: int = 20,
        lag_window: int = 5,
        max_quote_age: str | pd.Timedelta | None = None,
    ) -> None:
        self.slope_window = int(max(2, slope_window))
        self.spread_window = int(max(3, spread_window))
        self.lag_window = int(max(1, lag_window))
        self.max_quote_age = pd.Timedelta(max_quote_age) if max_quote_age is not None else None

    @staticmethod
    def _resolve_sector_map(
        symbols: list[str],
        specs: Mapping[str, AssetSpec | str] | Sequence[AssetSpec] | None,
        raw_dfs: Mapping[str, pd.DataFrame],
    ) -> dict[str, str]:
        sector_map: dict[str, str] = {}
        if isinstance(specs, Mapping):
            for k, v in specs.items():
                sector_map[str(k)] = v.sector if isinstance(v, AssetSpec) else str(v)
        elif isinstance(specs, Sequence):
            for item in specs:
                if isinstance(item, AssetSpec):
                    sector_map[item.symbol] = item.sector
        for sym in symbols:
            if sym not in sector_map:
                df_s = raw_dfs[sym]
                if "sector" in df_s.columns and len(df_s) > 0:
                    sector_map[sym] = str(df_s["sector"].iloc[0])
                elif "sector" in df_s.attrs:
                    sector_map[sym] = str(df_s.attrs["sector"])
                else:
                    sector_map[sym] = f"unknown:{sym}"
        return sector_map

    @staticmethod
    def _causal_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
        c = df["close"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        low_arr = df["low"].to_numpy(dtype=np.float64)
        n = len(c)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        tr = np.empty(n, dtype=np.float64)
        tr[0] = max(h[0] - low_arr[0], 1e-9)
        if n > 1:
            prev_c = c[:-1]
            tr[1:] = np.maximum(
                h[1:] - low_arr[1:],
                np.maximum(np.abs(h[1:] - prev_c), np.abs(low_arr[1:] - prev_c)),
            )
        atr = pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy(dtype=np.float64)
        return np.maximum(atr, 1e-9)

    def _compute_single_asset_primitives(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        if n == 0:
            return pd.DataFrame(
                columns=["dt", "norm_slope", "short_slope", "cum_norm_price", "causal_vol", "causal_snr"]
            )
        c = df["close"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        low_arr = df["low"].to_numpy(dtype=np.float64)
        atr = self._causal_atr(df, period=14)
        idx = np.arange(n, dtype=int)

        # 1. 因果波动率标准化斜率 NormSlope_{i,t} = (P_{i,t} - P_{i,t-w}) / (w * ATR_{i,t})
        w_idx = np.maximum(0, idx - self.slope_window)
        w_span = np.maximum(1, idx - w_idx).astype(np.float64)
        norm_slope = np.zeros(n, dtype=np.float64)
        if n > 1:
            norm_slope[1:] = (c[1:] - c[w_idx[1:]]) / (w_span[1:] * atr[1:])

        # 2. 因果短期标准化斜率 ShortSlope_{i,t} ∈ [-1, 1]
        l_idx = np.maximum(0, idx - self.lag_window)
        l_span = np.maximum(1, idx - l_idx).astype(np.float64)
        short_raw = np.zeros(n, dtype=np.float64)
        if n > 1:
            short_raw[1:] = (c[1:] - c[l_idx[1:]]) / (l_span[1:] * atr[1:])
        short_slope = np.tanh(2.0 * short_raw)

        # 3. 因果 1-bar ATR 归一化累积价格路径 (用于计算零前视板块残差)
        step_norm = np.zeros(n, dtype=np.float64)
        if n > 1:
            step_norm[1:] = (c[1:] - c[:-1]) / atr[1:]
        cum_norm_price = np.cumsum(step_norm)

        # 4. 严格因果 (<= t-1) 的成交活跃度与趋势信噪比 (Efficiency Ratio × 波动稳健度)，镜像翻转完全不变量
        vol_col = "volume" if "volume" in df.columns else ("vol" if "vol" in df.columns else None)
        if vol_col is not None:
            raw_v = np.maximum(np.nan_to_num(df[vol_col].to_numpy(dtype=np.float64), nan=1.0), 0.0)
        else:
            raw_v = np.ones(n, dtype=np.float64)
        vol_win = max(20, 2 * self.slope_window)
        s_vol = pd.Series(raw_v)
        vol_short = (
            s_vol.rolling(self.slope_window, min_periods=1).mean().shift(1).fillna(1.0).to_numpy(dtype=np.float64)
        )
        vol_long = s_vol.rolling(2 * vol_win, min_periods=1).mean().shift(1).fillna(1.0).to_numpy(dtype=np.float64)
        vol_act = np.clip(vol_short / np.maximum(vol_long, 1e-6), 0.50, 2.00)
        causal_vol = np.maximum(vol_long, 1e-6) * vol_act

        abs_diff = np.zeros(n, dtype=np.float64)
        if n > 1:
            abs_diff[1:] = np.abs(c[1:] - c[:-1])
        path_w = pd.Series(abs_diff).rolling(self.slope_window, min_periods=1).sum().to_numpy(dtype=np.float64)
        net_w = np.abs(c - c[w_idx])
        er_curr = np.where(path_w > 1e-9, np.clip(net_w / np.maximum(path_w, 1e-9), 0.0, 1.0), 0.5)
        er_prev = pd.Series(er_curr).shift(1).fillna(0.5).to_numpy(dtype=np.float64)

        tr = np.empty(n, dtype=np.float64)
        tr[0] = max(h[0] - low_arr[0], 1e-9)
        if n > 1:
            prev_c = c[:-1]
            tr[1:] = np.maximum(
                h[1:] - low_arr[1:],
                np.maximum(np.abs(h[1:] - prev_c), np.abs(low_arr[1:] - prev_c)),
            )
        tr_std = (
            pd.Series(tr).rolling(self.slope_window, min_periods=2).std(ddof=0).fillna(0.0).to_numpy(dtype=np.float64)
        )
        cv_tr_prev = pd.Series(tr_std / np.maximum(atr, 1e-9)).shift(1).fillna(0.0).to_numpy(dtype=np.float64)
        spike_prev = pd.Series(tr / np.maximum(atr, 1e-9)).shift(1).fillna(1.0).to_numpy(dtype=np.float64)
        stab_prev = np.clip(
            (1.0 / (1.0 + 0.5 * cv_tr_prev)) * np.minimum(1.0, 2.0 / np.maximum(spike_prev, 1.0)),
            0.25,
            1.0,
        )
        causal_snr = np.clip((0.35 + 0.65 * er_prev) * stab_prev * vol_act, 0.15, 2.0)

        out = pd.DataFrame(
            {
                "dt": pd.to_datetime(df["dt"]).to_numpy(),
                "norm_slope": norm_slope,
                "short_slope": short_slope,
                "cum_norm_price": cum_norm_price,
                "causal_vol": causal_vol,
                "causal_snr": causal_snr,
            }
        )
        return out.sort_values("dt").drop_duplicates(subset=["dt"], keep="last").reset_index(drop=True)

    def compute_and_attach(
        self,
        asset_tf_dfs: Mapping[str, pd.DataFrame | Mapping[str, pd.DataFrame]],
        specs: Mapping[str, AssetSpec | str] | Sequence[AssetSpec] | None = None,
        tf: str = "60m",
        *,
        attach_to_dfs: bool = True,
        max_quote_age: str | pd.Timedelta | None = None,
        membership_masks: Mapping[str, pd.Series] | None = None,
    ) -> dict[str, pd.DataFrame]:
        # 提取各品种在目标周期 tf 上的 DataFrame
        target_dfs: dict[str, pd.DataFrame] = {}
        is_nested = False
        for sym, val in asset_tf_dfs.items():
            if isinstance(val, pd.DataFrame):
                target_dfs[sym] = val
            elif isinstance(val, Mapping):
                is_nested = True
                if tf in val and isinstance(val[tf], pd.DataFrame):
                    target_dfs[sym] = val[tf]

        durations = {"5m": "5min", "15m": "15min", "30m": "30min", "60m": "60min", "1d": "1D"}
        quote_age = (
            pd.Timedelta(max_quote_age)
            if max_quote_age is not None
            else (self.max_quote_age or pd.Timedelta(durations.get(tf, "60min")))
        )
        symbols = list(target_dfs.keys())
        if not symbols:
            return {}

        sector_map = self._resolve_sector_map(symbols, specs, target_dfs)
        primitives: dict[str, pd.DataFrame] = {
            sym: self._compute_single_asset_primitives(target_dfs[sym]) for sym in symbols
        }

        all_dts = [p["dt"].to_numpy() for p in primitives.values() if not p.empty]
        if not all_dts:
            return {sym: target_dfs[sym].copy() for sym in symbols}

        grid_dt = np.unique(np.concatenate(all_dts))
        grid_dt.sort()
        t_len = len(grid_dt)
        m_syms = len(symbols)
        grid_frame = pd.DataFrame({"dt": grid_dt})

        ns_mat = np.full((t_len, m_syms), np.nan, dtype=np.float64)
        ss_mat = np.full((t_len, m_syms), np.nan, dtype=np.float64)
        pn_mat = np.full((t_len, m_syms), np.nan, dtype=np.float64)
        vol_mat = np.full((t_len, m_syms), np.nan, dtype=np.float64)
        snr_mat = np.full((t_len, m_syms), np.nan, dtype=np.float64)

        for j, sym in enumerate(symbols):
            p_df = primitives[sym]
            if p_df.empty:
                continue
            if len(p_df) == t_len and np.array_equal(p_df["dt"].to_numpy(), grid_dt):
                ns_mat[:, j] = p_df["norm_slope"].to_numpy(dtype=np.float64)
                ss_mat[:, j] = p_df["short_slope"].to_numpy(dtype=np.float64)
                pn_mat[:, j] = p_df["cum_norm_price"].to_numpy(dtype=np.float64)
                vol_mat[:, j] = p_df["causal_vol"].to_numpy(dtype=np.float64)
                snr_mat[:, j] = p_df["causal_snr"].to_numpy(dtype=np.float64)
            else:
                merged = pd.merge_asof(grid_frame, p_df, on="dt", direction="backward", tolerance=quote_age)
                ns_mat[:, j] = merged["norm_slope"].to_numpy(dtype=np.float64)
                ss_mat[:, j] = merged["short_slope"].to_numpy(dtype=np.float64)
                pn_mat[:, j] = merged["cum_norm_price"].to_numpy(dtype=np.float64)
                vol_mat[:, j] = merged["causal_vol"].to_numpy(dtype=np.float64)
                snr_mat[:, j] = merged["causal_snr"].to_numpy(dtype=np.float64)

        # Historical membership is an as-of event series. Before its first observation the asset is inactive.
        for j, sym in enumerate(symbols):
            member = (membership_masks or {}).get(sym)
            if member is not None:
                membership = pd.DataFrame({"dt": pd.to_datetime(member.index), "active": member.to_numpy(dtype=bool)})
                membership = membership.sort_values("dt").drop_duplicates("dt", keep="last")
                active = pd.merge_asof(grid_frame, membership, on="dt", direction="backward")["active"]
                inactive = ~active.fillna(False).to_numpy(dtype=bool)
                ns_mat[inactive, j] = np.nan
                ss_mat[inactive, j] = np.nan
                pn_mat[inactive, j] = np.nan
                vol_mat[inactive, j] = np.nan
                snr_mat[inactive, j] = np.nan

        valid_mask = np.isfinite(ns_mat)
        valid_cnt = np.sum(valid_mask, axis=1, keepdims=True)

        # 构造全市场与板块内严格因果 (<= t-1) 的连续报价覆盖率 × 成交活跃度 × 趋势信噪比 (ER × 波动稳健度) 权重矩阵 W_{k, t-1}
        sector_to_indices: dict[str, list[int]] = {}
        for j, sym in enumerate(symbols):
            sector_to_indices.setdefault(sector_map[sym], []).append(j)

        cov_win = max(20, 2 * self.slope_window)
        cov_mat = (
            pd.DataFrame(valid_mask.astype(np.float64))
            .rolling(cov_win, min_periods=1)
            .mean()
            .shift(1)
            .fillna(1.0)
            .to_numpy(dtype=np.float64)
        )
        cov_factor = np.power(np.clip(cov_mat, 0.15, 1.0), 2.5)

        cs_weight_mat = np.zeros((t_len, m_syms), dtype=np.float64)
        for _sec, col_idxs in sector_to_indices.items():
            sec_vol = np.where(valid_mask[:, col_idxs], np.maximum(vol_mat[:, col_idxs], 1e-6), np.nan)
            any_sec_valid = np.any(valid_mask[:, col_idxs], axis=1)
            sec_med_vol = np.ones((t_len, 1), dtype=np.float64)
            if np.any(any_sec_valid):
                sec_med_vol[any_sec_valid] = np.nanmedian(sec_vol[any_sec_valid], axis=1, keepdims=True)
            sec_med_vol = np.where(np.isfinite(sec_med_vol) & (sec_med_vol > 1e-9), sec_med_vol, 1.0)
            liq_ratio = np.clip(np.power(np.nan_to_num(sec_vol / sec_med_vol, nan=1.0), 0.45), 0.15, 3.00)
            sec_snr = np.clip(np.nan_to_num(snr_mat[:, col_idxs], nan=0.5), 0.15, 2.0)
            sec_cov = cov_factor[:, col_idxs]
            cs_weight_mat[:, col_idxs] = np.where(valid_mask[:, col_idxs], sec_cov * liq_ratio * sec_snr, 0.0)

        # 1. 全市场因果加权横截面百分位排名 cs_slope_rank ∈ [0, 1] 与中心化强度 cs_strength ∈ [-1, 1]
        cs_rank_mat = np.full((t_len, m_syms), 0.5, dtype=np.float64)
        if m_syms > 1:
            for j in range(m_syms):
                v_j = valid_mask[:, j]
                ns_j = ns_mat[:, j : j + 1]
                other_mask = valid_mask.copy()
                other_mask[:, j] = False
                w_other = np.where(other_mask, cs_weight_mat, 0.0)
                w_sum = np.sum(w_other, axis=1)
                less = other_mask & (ns_mat < ns_j - 1e-12)
                tie = other_mask & (np.abs(ns_mat - ns_j) <= 1e-12)
                w_less = np.sum(np.where(less, w_other, 0.0) + 0.5 * np.where(tie, w_other, 0.0), axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    rk_j = np.where(v_j & (w_sum > 1e-12), w_less / np.maximum(w_sum, 1e-12), 0.5)
                cs_rank_mat[:, j] = np.clip(np.nan_to_num(rk_j, nan=0.5), 0.0, 1.0)
        cs_strength_mat = np.clip(2.0 * cs_rank_mat - 1.0, -1.0, 1.0)

        # 2. 按板块分组计算因果去噪龙头方向 sector_leader_dir ∈ [-1, 1]、滞涨差 leader_lag_gap 与链式无跳变板块基准路径
        leader_dir_mat = np.zeros((t_len, m_syms), dtype=np.float64)
        lag_gap_mat = np.zeros((t_len, m_syms), dtype=np.float64)
        sector_mean_pn_mat = np.zeros((t_len, m_syms), dtype=np.float64)
        peer_available_mat = np.zeros((t_len, m_syms), dtype=bool)

        for _sec, col_idxs in sector_to_indices.items():
            sec_pn = pn_mat[:, col_idxs]
            sec_valid_pn = np.isfinite(sec_pn)
            sec_w_pn = np.where(sec_valid_pn, cs_weight_mat[:, col_idxs], 0.0)
            sec_w_sum = np.sum(sec_w_pn, axis=1)
            sec_pn_ffill = pd.DataFrame(sec_pn).ffill().fillna(0.0).to_numpy(dtype=np.float64)
            sec_dpn = np.zeros_like(sec_pn_ffill)
            if t_len > 1:
                sec_dpn[1:] = sec_pn_ffill[1:] - sec_pn_ffill[:-1]
            sec_dpn = np.where(sec_valid_pn, sec_dpn, 0.0)
            sec_mean_dpn = np.sum(sec_dpn * sec_w_pn, axis=1) / np.maximum(sec_w_sum, 1e-12)
            sec_mean_pn = np.cumsum(sec_mean_dpn)

            for j in col_idxs:
                sector_mean_pn_mat[:, j] = sec_mean_pn
                peer_idxs = [k for k in col_idxs if k != j]
                if not peer_idxs:
                    continue
                peer_ns = ns_mat[:, peer_idxs]
                peer_ss = ss_mat[:, peer_idxs]
                peer_w = cs_weight_mat[:, peer_idxs]
                abs_ns = np.abs(peer_ns)
                valid_peer = np.isfinite(abs_ns) & (peer_w > 0.0)
                any_valid = np.any(valid_peer, axis=1) & np.isfinite(ns_mat[:, j])
                peer_available_mat[:, j] = any_valid

                # 采用因果活跃度 × 信噪比加权的 Top-K 龙头评分替代单 Bar 裸 max_abs，消除边缘小品种毛刺篡权
                leader_score = np.where(valid_peer, peer_w * abs_ns, -1.0)
                n_peers = len(peer_idxs)
                k_top = min(n_peers, max(1, min(2, len(col_idxs) // 4)))
                if k_top == 1 or n_peers == 1:
                    thresh = np.max(leader_score, axis=1, keepdims=True)
                else:
                    part = np.partition(leader_score, n_peers - k_top, axis=1)
                    thresh = np.maximum(part[:, n_peers - k_top : n_peers - k_top + 1], 0.0)
                is_top = valid_peer & (leader_score >= thresh - 1e-12)

                top_w = np.where(is_top, peer_w * (0.25 + abs_ns), 0.0)
                top_w_sum = np.maximum(np.sum(top_w, axis=1), 1e-12)

                leader_ns = np.sum(np.where(is_top, top_w * peer_ns, 0.0), axis=1) / top_w_sum
                leader_ss = np.sum(np.where(is_top, top_w * peer_ss, 0.0), axis=1) / top_w_sum

                leader_dir_j = np.where(any_valid, np.tanh(2.5 * leader_ns), 0.0)
                self_ss = np.nan_to_num(ss_mat[:, j], nan=0.0)
                lag_gap_j = np.where(any_valid, leader_ss - self_ss, 0.0)

                leader_dir_mat[:, j] = np.clip(leader_dir_j, -1.0, 1.0)
                lag_gap_mat[:, j] = np.clip(lag_gap_j, -2.0, 2.0)

        # 3. 映射回各品种自身 K 线时间戳并计算因果滚动残差 Z-score (sector_spread_zscore)
        result_dfs: dict[str, pd.DataFrame] = {}
        for j, sym in enumerate(symbols):
            base_df = target_dfs[sym]
            p_df = primitives[sym]
            panel_j = pd.DataFrame(
                {
                    "dt": grid_dt,
                    "cs_slope_rank": cs_rank_mat[:, j],
                    "cs_strength": cs_strength_mat[:, j],
                    "sector_leader_dir": leader_dir_mat[:, j],
                    "leader_lag_gap": lag_gap_mat[:, j],
                    "_sec_mean_pn": sector_mean_pn_mat[:, j],
                    "cs_available": peer_available_mat[:, j] & (valid_cnt[:, 0] > 1),
                    "cs_active_count": valid_cnt[:, 0],
                }
            )
            if len(p_df) == t_len and np.array_equal(p_df["dt"].to_numpy(), grid_dt):
                aligned_j = panel_j.copy()
                aligned_j["cum_norm_price"] = p_df["cum_norm_price"].to_numpy(dtype=np.float64)
            else:
                aligned_j = pd.merge_asof(p_df, panel_j, on="dt", direction="backward")

            excess_spread = aligned_j["cum_norm_price"].to_numpy(dtype=np.float64) - aligned_j["_sec_mean_pn"].to_numpy(
                dtype=np.float64
            )
            s_exc = pd.Series(excess_spread)
            roll_mean = s_exc.rolling(self.spread_window, min_periods=3).mean().to_numpy(dtype=np.float64)
            roll_std = s_exc.rolling(self.spread_window, min_periods=3).std(ddof=0).to_numpy(dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                z_raw = np.where(roll_std > 1e-8, (excess_spread - roll_mean) / roll_std, 0.0)
            spread_z = np.clip(np.nan_to_num(z_raw, nan=0.0), -5.0, 5.0)

            cs_only_df = pd.DataFrame(
                {
                    "dt": aligned_j["dt"].to_numpy(),
                    "cs_slope_rank": np.clip(
                        np.nan_to_num(aligned_j["cs_slope_rank"].to_numpy(dtype=np.float64), nan=0.5), 0.0, 1.0
                    ),
                    "cs_strength": np.clip(
                        np.nan_to_num(aligned_j["cs_strength"].to_numpy(dtype=np.float64), nan=0.0), -1.0, 1.0
                    ),
                    "sector_leader_dir": np.clip(
                        np.nan_to_num(aligned_j["sector_leader_dir"].to_numpy(dtype=np.float64), nan=0.0), -1.0, 1.0
                    ),
                    "leader_lag_gap": np.clip(
                        np.nan_to_num(aligned_j["leader_lag_gap"].to_numpy(dtype=np.float64), nan=0.0), -2.0, 2.0
                    ),
                    "sector_spread_zscore": np.where(aligned_j["cs_available"], spread_z, 0.0),
                    "cs_available": aligned_j["cs_available"].to_numpy(dtype=bool),
                    "cs_active_count": aligned_j["cs_active_count"].to_numpy(dtype=np.int64),
                }
            )
            provider = DataFrameCrossSectionProvider(cs_only_df, default_tf=tf, max_quote_age=quote_age)

            if attach_to_dfs:
                if is_nested and isinstance(asset_tf_dfs[sym], Mapping):
                    for _sub_tf, sub_df in asset_tf_dfs[sym].items():  # type: ignore[union-attr]
                        self._attach_cs_to_single_df(sub_df, provider)
                else:
                    self._attach_cs_to_single_df(base_df, provider)

            out_df = base_df.copy()
            self._attach_cs_to_single_df(out_df, provider)
            result_dfs[sym] = out_df

        return result_dfs

    @staticmethod
    def _attach_cs_to_single_df(df: pd.DataFrame, provider: DataFrameCrossSectionProvider) -> pd.DataFrame:
        feats = provider.features(df)
        for col in CS_FEATURE_COLS:
            df[col] = feats[col]
        set_factor_context(df, cross_section=provider)
        return df


def compute_causal_cross_sectional_features(
    asset_tf_dfs: Mapping[str, pd.DataFrame | Mapping[str, pd.DataFrame]],
    specs: Mapping[str, AssetSpec | str] | Sequence[AssetSpec] | None = None,
    tf: str = "60m",
    *,
    slope_window: int = 20,
    spread_window: int = 20,
    lag_window: int = 5,
    attach_to_dfs: bool = True,
    max_quote_age: str | pd.Timedelta | None = None,
    membership_masks: Mapping[str, pd.Series] | None = None,
) -> dict[str, pd.DataFrame]:
    """模块级便捷函数：计算并挂载跨品种零前视因果截面特征。"""
    return CrossAssetDataEngine.compute_causal_cross_sectional_features(
        asset_tf_dfs=asset_tf_dfs,
        specs=specs,
        tf=tf,
        slope_window=slope_window,
        spread_window=spread_window,
        lag_window=lag_window,
        attach_to_dfs=attach_to_dfs,
        max_quote_age=max_quote_age,
        membership_masks=membership_masks,
    )
