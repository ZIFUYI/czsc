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
from examples.multi_factor_cta.atomic_factors import link_higher_timeframes


class AssetClass(str, Enum):
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
    base_tf: str  # 该品种底层基准执行周期 ("5m" | "15m" | "30m")
    eligible_tfs: tuple[str, ...]  # 该品种参与合成的多周期列表
    fee_rate: float  # 单边综合交易成本 (手续费 + 冲击滑点)
    has_night_session: bool = False  # 是否包含夜盘 (21:00 起归属下一交易日)
    roll_gap_threshold_atr: float = 3.5  # 主力换月异常跳空后复权识别阈值
    execution_mode: ExecutionMode = ExecutionMode.LONG_SHORT
    data_path: Path | None = None


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
    c = out["close"].values.astype(np.float64)
    o = out["open"].values.astype(np.float64)
    h = out["high"].values.astype(np.float64)
    low_arr = out["low"].values.astype(np.float64)

    tr = np.maximum(h - low_arr, np.abs(c - np.roll(c, 1)))
    tr[0] = h[0] - low_arr[0]
    atr = pd.Series(tr).rolling(20, min_periods=5).mean().bfill().values

    dates = out["date"].values
    day_switch = np.zeros(len(out), dtype=bool)
    day_switch[1:] = dates[1:] != dates[:-1]

    overnight_gap = np.zeros(len(out), dtype=np.float64)
    overnight_gap[1:] = np.abs(o[1:] - c[:-1])

    if "underlying_symbol" in out.columns and out["underlying_symbol"].dropna().nunique() > 1:
        usym = out["underlying_symbol"].ffill().bfill().astype(str).values
        roll_mask = np.zeros(len(out), dtype=bool)
        roll_mask[1:] = usym[1:] != usym[:-1]
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
    """
    将任意高周期 (15m / 30m / 60m / 1d) 的规则持仓权重无未来函数、零延迟地映射到品种基准网格 (5m / 15m / 30m) 上：
    - 当源周期与目标网格相同时，1:1 原样返回；
    - 当源周期高于目标网格时，严格在源周期 K 线收盘时刻 `dt` 生效，并向后阶梯保持 (Step-Hold ffill)。
    """
    if len(df_source) == len(df_target) and np.array_equal(df_source["dt"].values, df_target["dt"].values):
        return w_source.astype(np.float64).copy()

    n_src = len(df_source)
    n_tgt = len(df_target)
    if n_src > 0 and n_tgt % n_src == 0:
        g = n_tgt // n_src
        if g > 1:
            out = np.zeros(n_tgt, dtype=np.float64)
            out[g - 1 : -1] = np.repeat(w_source[:-1], g)
            out[-1] = w_source[-1]
            return out

    s_src = pd.Series(w_source.astype(np.float64), index=df_source["dt"])
    return df_target["dt"].map(s_src).ffill().fillna(0.0).values


class CrossAssetDataEngine:
    """跨资产（股指期货 + ETF + 商品期货）统一数据加载与多周期特征工程引擎"""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_asset_multi_timeframes(self, spec: AssetSpec, attach_czsc: bool = True) -> dict[str, pd.DataFrame]:
        """加载单个品种的底层数据并合成其 eligible_tfs 多周期 K 线与特征"""
        if spec.data_path is None or not spec.data_path.exists():
            raise FileNotFoundError(f"Data file not found for {spec.symbol}: {spec.data_path}")

        df_base = self._read_raw_dataframe(spec)
        cal_path = spec.data_path.parent / "calendar.parquet"
        df_base["date"] = assign_trading_date(df_base["dt"], spec.has_night_session, cal_path)
        df_base["bar_in_day"] = df_base.groupby("date", sort=False).cumcount()

        if spec.asset_class in (AssetClass.INDEX_FUTURES, AssetClass.COMMODITY_FUTURES):
            df_base = adjust_continuous_roll_gaps(df_base, gap_atr_mult=spec.roll_gap_threshold_atr)

        base_mins = {"5m": 5, "15m": 15, "30m": 30, "60m": 60}[spec.base_tf]
        tf_mins = {"5m": 5, "15m": 15, "30m": 30, "60m": 60, "1d": 240}

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
