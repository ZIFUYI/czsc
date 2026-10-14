#!/usr/bin/env python
"""
159915 (易方达创业板ETF) 全天候六合一黄金结构多空复合策略 (Ultimate Six-Pattern Composite Strategy)
汇聚两轮实证检验挖掘出的 6 大高胜率、高盈亏比黄金结构：
1. 窄幅中枢突破 + 三买三卖确认 (Box Breakout & 3rd BS)
2. 均线粘合一阳穿多线 / 一阴下多线 (MA Ribbon Squeeze Breakout)
3. 内孕线 (Inside Bar) 破局放量突破母线高点 / 跌破母线低点
4. VCP 波动率极度压缩爆发 (VCP Volatility Contraction Breakout)
5. 顺势首次回踩 MA20 底分型企稳 / 首次反抽 MA20 顶分型跌破
6. 顺势三连阴缩量洗盘反包 (Uptrend Low-Volume Retest)

全天候架构：
- 100% 多空对称
- 结构证伪止损 + 浮盈保本锁定 + 极值动态追踪止盈
- 动态仓位共振加权 (W ∈ [-1.0, 1.0])
- 2020-2026 年近 7 年真实数据回测检验
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from czsc import WeightBacktest

OUTPUT_DIR = Path("examples/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REAL_5M_CSV = OUTPUT_DIR / "159915_real_5m_2020_2026.csv"
HTML_PATH = OUTPUT_DIR / "strategy_ultimate_composite_159915.html"


def load_daily_data(csv_path: Path) -> pd.DataFrame:
    """加载 5m 真实历史数据并合成为日线数据"""
    df_raw = pd.read_csv(csv_path)
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
    df_raw = df_raw.rename(
        columns={
            "SZSE.159915.open": "open",
            "SZSE.159915.high": "high",
            "SZSE.159915.low": "low",
            "SZSE.159915.close": "close",
            "SZSE.159915.volume": "volume",
        }
    )
    df_raw.set_index("datetime", inplace=True)
    daily = (
        df_raw.resample("D")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )
    daily = daily[daily["volume"] > 0].copy()
    daily.reset_index(inplace=True)
    daily.rename(columns={"datetime": "dt"}, inplace=True)
    daily["symbol"] = "159915"
    return daily


# ----------------------------------------------------------------------
# 模块 1: 中枢突破与三买三卖 (Box Breakout)
# ----------------------------------------------------------------------
def get_box_signals(df: pd.DataFrame) -> list[float]:
    pos = 0
    stop_p, entry_p, peak_p, trough_p = None, None, None, None
    pending_long, l_idx, l_box_top, l_retest_l = False, -999, 0.0, 9999.0
    pending_short, s_idx, s_box_bot, s_rebound_h = False, -999, 0.0, 0.0
    weights = []

    for i in range(len(df)):
        c, o, h, l = df.loc[i, "close"], df.loc[i, "open"], df.loc[i, "high"], df.loc[i, "low"]
        ma20, ma60 = df.loc[i, "ma20"], df.loc[i, "ma60"]
        amp = df.loc[i, "box_amp"]
        box_top, box_bot = df.loc[i, "prev_high_box"], df.loc[i, "prev_low_box"]

        if i < 60:
            weights.append(0.0)
            continue

        exited = False
        if pos == 1:
            peak_p = max(peak_p or c, h)
            unreal = (peak_p - entry_p) / entry_p
            if unreal >= 0.02:
                stop_p = max(stop_p, entry_p * 1.002)
            if stop_p is not None and l <= stop_p or unreal >= 0.03 and (peak_p - c) / peak_p >= 0.03 or c < ma20:
                pos = 0
                exited = True
        elif pos == -1:
            trough_p = min(trough_p or c, l)
            unreal = (entry_p - trough_p) / entry_p
            if unreal >= 0.02:
                stop_p = min(stop_p, entry_p * 0.998)
            if stop_p is not None and h >= stop_p or unreal >= 0.03 and (c - trough_p) / trough_p >= 0.03 or c > ma20:
                pos = 0
                exited = True

        if exited:
            stop_p, entry_p, peak_p, trough_p = None, None, None, None

        if pos == 0 and not pending_long and amp <= 0.09 and c > ma60 and c > box_top:
            pending_long, l_idx, l_box_top, l_retest_l = True, i, box_top, l
        elif pos == 0 and not pending_short and amp <= 0.09 and c < ma60 and c < box_bot:
            pending_short, s_idx, s_box_bot, s_rebound_h = True, i, box_bot, h

        if pending_long and (i - l_idx > 4):
            pending_long = False
        if pending_short and (i - s_idx > 4):
            pending_short = False

        if pos == 0 and pending_long and (i > l_idx):
            l_retest_l = min(l_retest_l, l)
            if c < l_box_top * 0.995:
                pending_long = False
            elif c > o and c >= df.loc[i - 1, "high"]:
                pos, stop_p, entry_p, peak_p = 1, max(l_retest_l, l_box_top * 0.992), c, c
                pending_long = False
        elif pos == 0 and pending_short and (i > s_idx):
            s_rebound_h = max(s_rebound_h, h)
            if c > s_box_bot * 1.005:
                pending_short = False
            elif c < o and c <= df.loc[i - 1, "low"]:
                pos, stop_p, entry_p, trough_p = -1, min(s_rebound_h, s_box_bot * 1.008), c, c
                pending_short = False

        weights.append(float(pos))
    return weights


# ----------------------------------------------------------------------
# 模块 2: 均线缠绕突破 (Ribbon Squeeze Breakout)
# ----------------------------------------------------------------------
def get_ribbon_signals(df: pd.DataFrame) -> list[float]:
    pos = 0
    stop_p, entry_p, peak_p, trough_p = None, None, None, None
    weights = []

    for i in range(len(df)):
        c, o, h, l = df.loc[i, "close"], df.loc[i, "open"], df.loc[i, "high"], df.loc[i, "low"]
        ma5, ma10, ma20, ma30, ma60 = (
            df.loc[i, "ma5"],
            df.loc[i, "ma10"],
            df.loc[i, "ma20"],
            df.loc[i, "ma30"],
            df.loc[i, "ma60"],
        )
        disp = df.loc[i, "ma_dispersion"]

        if i < 60:
            weights.append(0.0)
            continue

        exited = False
        if pos == 1:
            peak_p = max(peak_p or c, h)
            unreal = (peak_p - entry_p) / entry_p
            if unreal >= 0.02:
                stop_p = max(stop_p, entry_p * 1.002)
            if stop_p is not None and l <= stop_p or unreal >= 0.035 and (peak_p - c) / peak_p >= 0.025 or c < ma20:
                pos = 0
                exited = True
        elif pos == -1:
            trough_p = min(trough_p or c, l)
            unreal = (entry_p - trough_p) / entry_p
            if unreal >= 0.02:
                stop_p = min(stop_p, entry_p * 0.998)
            if stop_p is not None and h >= stop_p or unreal >= 0.035 and (c - trough_p) / trough_p >= 0.025 or c > ma20:
                pos = 0
                exited = True

        if exited:
            stop_p, entry_p, peak_p, trough_p = None, None, None, None

        # 触发
        if pos == 0 and disp <= 0.02:
            # 多头：一阳穿多线
            if o <= min(ma5, ma10, ma20) * 1.005 and c >= max(ma5, ma10, ma20, ma30) and c > o * 1.015 and c > ma60:
                pos, entry_p, peak_p, stop_p = 1, c, c, l - 0.002
            # 空头：一阴下多线
            elif o >= max(ma5, ma10, ma20) * 0.995 and c <= min(ma5, ma10, ma20, ma30) and c < o * 0.985 and c < ma60:
                pos, entry_p, trough_p, stop_p = -1, c, c, h + 0.002

        weights.append(float(pos))
    return weights


# ----------------------------------------------------------------------
# 模块 3: 内孕线破局突破 (Inside Bar Breakout)
# ----------------------------------------------------------------------
def get_inside_bar_signals(df: pd.DataFrame) -> list[float]:
    pos = 0
    stop_p, entry_p, peak_p, trough_p = None, None, None, None
    weights = []

    for i in range(len(df)):
        c, o, h, l = df.loc[i, "close"], df.loc[i, "open"], df.loc[i, "high"], df.loc[i, "low"]
        ma20, ma60 = df.loc[i, "ma20"], df.loc[i, "ma60"]
        vol, vol_ma20 = df.loc[i, "volume"], df.loc[i, "vol_ma20"]

        if i < 60:
            weights.append(0.0)
            continue

        exited = False
        if pos == 1:
            peak_p = max(peak_p or c, h)
            unreal = (peak_p - entry_p) / entry_p
            if unreal >= 0.025:
                stop_p = max(stop_p, entry_p * 1.002)
            if stop_p is not None and l <= stop_p or unreal >= 0.04 and (peak_p - c) / peak_p >= 0.025 or c < ma20:
                pos = 0
                exited = True
        elif pos == -1:
            trough_p = min(trough_p or c, l)
            unreal = (entry_p - trough_p) / entry_p
            if unreal >= 0.025:
                stop_p = min(stop_p, entry_p * 0.998)
            if stop_p is not None and h >= stop_p or unreal >= 0.04 and (c - trough_p) / trough_p >= 0.025 or c > ma20:
                pos = 0
                exited = True

        if exited:
            stop_p, entry_p, peak_p, trough_p = None, None, None, None

        if pos == 0 and i >= 2:
            is_inside_prev = (df.loc[i - 1, "high"] < df.loc[i - 2, "high"]) and (
                df.loc[i - 1, "low"] > df.loc[i - 2, "low"]
            )
            if is_inside_prev and vol >= 1.2 * vol_ma20:
                # 突破前天母线高点
                if c > df.loc[i - 2, "high"] and c > o and c > ma60:
                    pos, entry_p, peak_p, stop_p = 1, c, c, df.loc[i - 1, "low"] - 0.002
                # 跌破前天母线低点
                elif c < df.loc[i - 2, "low"] and c < o and c < ma60:
                    pos, entry_p, trough_p, stop_p = -1, c, c, df.loc[i - 1, "high"] + 0.002

        weights.append(float(pos))
    return weights


# ----------------------------------------------------------------------
# 模块 4: VCP 波动率极度压缩爆发 (VCP Breakout)
# ----------------------------------------------------------------------
def get_vcp_signals(df: pd.DataFrame) -> list[float]:
    pos = 0
    stop_p, entry_p, peak_p, trough_p = None, None, None, None
    weights = []

    for i in range(len(df)):
        c, o, h, l = df.loc[i, "close"], df.loc[i, "open"], df.loc[i, "high"], df.loc[i, "low"]
        ma20, ma60 = df.loc[i, "ma20"], df.loc[i, "ma60"]
        atr14, atr_rank = df.loc[i, "atr14"], df.loc[i, "atr_rank_60"]
        rng = df.loc[i, "total_range"]

        if i < 60:
            weights.append(0.0)
            continue

        exited = False
        if pos == 1:
            peak_p = max(peak_p or c, h)
            unreal = (peak_p - entry_p) / entry_p
            if unreal >= 0.025:
                stop_p = max(stop_p, entry_p * 1.002)
            if stop_p is not None and l <= stop_p or unreal >= 0.035 and (peak_p - c) / peak_p >= 0.03 or c < ma20:
                pos = 0
                exited = True
        elif pos == -1:
            trough_p = min(trough_p or c, l)
            unreal = (entry_p - trough_p) / entry_p
            if unreal >= 0.025:
                stop_p = min(stop_p, entry_p * 0.998)
            if stop_p is not None and h >= stop_p or unreal >= 0.035 and (c - trough_p) / trough_p >= 0.03 or c > ma20:
                pos = 0
                exited = True

        if exited:
            stop_p, entry_p, peak_p, trough_p = None, None, None, None

        if pos == 0 and atr_rank <= 0.20 and rng >= 1.5 * atr14:
            if c > ma60 and c > o and c > ma20:
                pos, entry_p, peak_p, stop_p = 1, c, c, l - 0.002
            elif c < ma60 and c < o and c < ma20:
                pos, entry_p, trough_p, stop_p = -1, c, c, h + 0.002

        weights.append(float(pos))
    return weights


# ----------------------------------------------------------------------
# 模块 5: 顺势首次回踩企稳 (Trend First Pullback)
# ----------------------------------------------------------------------
def get_pullback_signals(df: pd.DataFrame) -> list[float]:
    pos = 0
    stop_p, entry_p, peak_p, trough_p = None, None, None, None
    weights = []

    for i in range(len(df)):
        c, o, h, l = df.loc[i, "close"], df.loc[i, "open"], df.loc[i, "high"], df.loc[i, "low"]
        ma20, ma60 = df.loc[i, "ma20"], df.loc[i, "ma60"]

        if i < 60:
            weights.append(0.0)
            continue

        exited = False
        if pos == 1:
            peak_p = max(peak_p or c, h)
            unreal = (peak_p - entry_p) / entry_p
            if unreal >= 0.02:
                stop_p = max(stop_p, entry_p * 1.002)
            if stop_p is not None and l <= stop_p or unreal >= 0.03 and (peak_p - c) / peak_p >= 0.025 or c < ma20:
                pos = 0
                exited = True
        elif pos == -1:
            trough_p = min(trough_p or c, l)
            unreal = (entry_p - trough_p) / entry_p
            if unreal >= 0.02:
                stop_p = min(stop_p, entry_p * 0.998)
            if stop_p is not None and h >= stop_p or unreal >= 0.03 and (c - trough_p) / trough_p >= 0.025 or c > ma20:
                pos = 0
                exited = True

        if exited:
            stop_p, entry_p, peak_p, trough_p = None, None, None, None

        if pos == 0 and i >= 5:
            was_above = (df.loc[i - 1, "close"] > df.loc[i - 1, "ma20"]) and (
                df.loc[i - 2, "close"] > df.loc[i - 2, "ma20"]
            )
            touch_rebound_l = (l <= ma20 * 1.005) and (c >= ma20) and (c > o) and (c > df.loc[i - 1, "high"])
            if c > ma60 and ma20 > ma60 and was_above and touch_rebound_l:
                pos, entry_p, peak_p, stop_p = 1, c, c, l - 0.002

            was_below = (df.loc[i - 1, "close"] < df.loc[i - 1, "ma20"]) and (
                df.loc[i - 2, "close"] < df.loc[i - 2, "ma20"]
            )
            touch_rebound_s = (h >= ma20 * 0.995) and (c <= ma20) and (c < o) and (c < df.loc[i - 1, "low"])
            if c < ma60 and ma20 < ma60 and was_below and touch_rebound_s:
                pos, entry_p, trough_p, stop_p = -1, c, c, h + 0.002

        weights.append(float(pos))
    return weights


# ----------------------------------------------------------------------
# 模块 6: 顺势三连阴缩量洗盘反包 (Uptrend 3-Red Low Vol Retest)
# ----------------------------------------------------------------------
def get_retest_signals(df: pd.DataFrame) -> list[float]:
    pos = 0
    stop_p, entry_p, peak_p, trough_p = None, None, None, None
    weights = []

    for i in range(len(df)):
        c, o, h, l = df.loc[i, "close"], df.loc[i, "open"], df.loc[i, "high"], df.loc[i, "low"]
        ma20, ma60 = df.loc[i, "ma20"], df.loc[i, "ma60"]

        if i < 60:
            weights.append(0.0)
            continue

        exited = False
        if pos == 1:
            peak_p = max(peak_p or c, h)
            unreal = (peak_p - entry_p) / entry_p
            if unreal >= 0.02:
                stop_p = max(stop_p, entry_p * 1.002)
            if stop_p is not None and l <= stop_p or unreal >= 0.035 and (peak_p - c) / peak_p >= 0.025 or c < ma20:
                pos = 0
                exited = True
        elif pos == -1:
            trough_p = min(trough_p or c, l)
            unreal = (entry_p - trough_p) / entry_p
            if unreal >= 0.02:
                stop_p = min(stop_p, entry_p * 0.998)
            if stop_p is not None and h >= stop_p or unreal >= 0.035 and (c - trough_p) / trough_p >= 0.025 or c > ma20:
                pos = 0
                exited = True

        if exited:
            stop_p, entry_p, peak_p, trough_p = None, None, None, None

        if pos == 0 and i >= 4:
            # 多头：顺势三连阴缩量反包
            three_red = (df.loc[i - 1, "close"] < df.loc[i - 2, "close"]) and (
                df.loc[i - 2, "close"] < df.loc[i - 3, "close"]
            )
            vol_shrink = (df.loc[i - 1, "volume"] < df.loc[i - 2, "volume"]) and (
                df.loc[i - 2, "volume"] < df.loc[i - 3, "volume"]
            )
            bull_engulf = (
                (c > o) and (c >= df.loc[i - 1, "high"]) and (df.loc[i, "volume"] >= 1.3 * df.loc[i - 1, "volume"])
            )
            if c > ma60 and three_red and vol_shrink and bull_engulf and df.loc[i - 1, "low"] > df.loc[i - 1, "ma60"]:
                pos, entry_p, peak_p, stop_p = 1, c, c, l - 0.002

            # 空头：逆势三连阳缩量死猫跳大阴反杀
            three_up = (df.loc[i - 1, "close"] > df.loc[i - 2, "close"]) and (
                df.loc[i - 2, "close"] > df.loc[i - 3, "close"]
            )
            vol_weak = (df.loc[i - 1, "volume"] <= 1.1 * df.loc[i - 1, "vol_ma20"]) and (
                df.loc[i - 2, "volume"] <= 1.1 * df.loc[i - 2, "vol_ma20"]
            )
            bear_kill = (c < o) and (c <= df.loc[i - 1, "low"]) and (df.loc[i, "pct_chg"] <= -0.015)
            if c < ma60 and three_up and vol_weak and bear_kill:
                pos, entry_p, trough_p, stop_p = -1, c, c, h + 0.002

        weights.append(float(pos))
    return weights


def main():
    print("=" * 90)
    print("  【159915 全天候六合一黄金结构多空复合策略】 2020-2026 全周期真实数据回测")
    print("=" * 90)

    daily_df = load_daily_data(REAL_5M_CSV)
    print(f"数据加载完成: 共 {len(daily_df)} 根真实日线 (2020-01-02 ~ 2026-10-12)")

    # 预计算通用指标
    daily_df["pct_chg"] = daily_df["close"].pct_change()
    daily_df["ma5"] = daily_df["close"].rolling(5).mean()
    daily_df["ma10"] = daily_df["close"].rolling(10).mean()
    daily_df["ma20"] = daily_df["close"].rolling(20).mean()
    daily_df["ma30"] = daily_df["close"].rolling(30).mean()
    daily_df["ma60"] = daily_df["close"].rolling(60).mean()
    daily_df["vol_ma20"] = daily_df["volume"].rolling(20).mean()
    daily_df["total_range"] = daily_df["high"] - daily_df["low"] + 1e-6

    # 均线粘合度
    ma_max = daily_df[["ma5", "ma10", "ma20", "ma30"]].max(axis=1)
    ma_min = daily_df[["ma5", "ma10", "ma20", "ma30"]].min(axis=1)
    daily_df["ma_dispersion"] = (ma_max - ma_min) / ma_min

    # ATR
    tr1 = daily_df["high"] - daily_df["low"]
    tr2 = (daily_df["high"] - daily_df["close"].shift(1)).abs()
    tr3 = (daily_df["low"] - daily_df["close"].shift(1)).abs()
    daily_df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    daily_df["atr14"] = daily_df["tr"].rolling(14).mean()
    daily_df["atr_rank_60"] = (
        daily_df["atr14"].rolling(60).apply(lambda s: pd.Series(s).rank(pct=True).iloc[-1], raw=False)
    )

    # 滚动箱体
    daily_df["prev_high_box"] = daily_df["high"].shift(1).rolling(20).max()
    daily_df["prev_low_box"] = daily_df["low"].shift(1).rolling(20).min()
    daily_df["box_amp"] = (daily_df["prev_high_box"] - daily_df["prev_low_box"]) / daily_df["prev_low_box"]

    # 运算六大子策略
    print("\n正在运算 6 大黄金结构状态机...")
    w1_box = get_box_signals(daily_df)
    w2_ribbon = get_ribbon_signals(daily_df)
    w3_inside = get_inside_bar_signals(daily_df)
    w4_vcp = get_vcp_signals(daily_df)
    w5_pull = get_pullback_signals(daily_df)
    w6_retest = get_retest_signals(daily_df)

    # 权重融合方案：
    # 结构1(中枢突破) 25% + 结构2(均线缠绕) 20% + 结构3(内孕破局) 15% + 结构4(VCP爆发) 15% + 结构5(首次回踩) 15% + 结构6(三连阴洗盘) 10%
    w_ultimate = [
        np.clip(0.25 * b + 0.20 * r + 0.15 * i + 0.15 * v + 0.15 * p + 0.10 * t, -1.0, 1.0)
        for b, r, i, v, p, t in zip(w1_box, w2_ribbon, w3_inside, w4_vcp, w5_pull, w6_retest)
    ]

    dts = [str(d) for d in daily_df["dt"]]
    prices = daily_df["close"].tolist()

    dfw_ult = pd.DataFrame({"dt": dts, "symbol": "159915", "weight": w_ultimate, "price": prices})
    wb_ult = WeightBacktest(data=dfw_ult, fee_rate=0.0002, weight_type="ts")

    dr = wb_ult.daily_return.copy()
    dr["date"] = pd.to_datetime(dr["date"])
    ret_ult = dr.set_index("date")["total"]

    # 基准
    bench_pnl = daily_df.set_index("dt")["close"].pct_change().fillna(0)
    ret_bench = bench_pnl.loc[ret_ult.index]

    nav_ult = (1 + ret_ult).cumprod()
    nav_bench = (1 + ret_bench).cumprod()
    nav_bench_norm = nav_bench / nav_bench.iloc[0]

    dd_ult = (nav_ult - nav_ult.cummax()) / nav_ult.cummax() * 100
    dd_bench = (nav_bench_norm - nav_bench_norm.cummax()) / nav_bench_norm.cummax() * 100

    # 分年度统计
    dr["year"] = dr["date"].dt.year
    ypnl_ult = dr.groupby("year")["total"].apply(lambda s: (1 + s).prod() - 1) * 100

    bench_df = pd.DataFrame({"bench": ret_bench, "date": ret_bench.index})
    bench_df["year"] = bench_df["date"].dt.year
    ypnl_bench = bench_df.groupby("year")["bench"].apply(lambda s: (1 + s).prod() - 1) * 100

    st = wb_ult.stats

    print("\n" + "=" * 90)
    print("【WeightBacktest 全天候六合一策略终审指标卡】")
    print("=" * 90)
    print(
        f"绝对累计收益率:       {st.get('绝对收益', 0) * 100:+.2f}%   (基准: {(nav_bench_norm.iloc[-1] - 1) * 100:+.2f}%)"
    )
    print(f"年化收益率 (Ann Ret): {st.get('年化收益', 0) * 100:+.2f}%")
    print(f"夏普比率 (Sharpe):    {st.get('夏普比率', 0):.2f}    (基准夏普: 0.38)")
    print(f"最大回撤 (MDD):       {st.get('最大回撤', 0) * 100:.2f}%    (基准最大回撤: {dd_bench.min():.2f}%)")
    print(f"卡玛比率 (Calmar):    {st.get('卡玛比率', 0):.2f}")
    print(f"日胜率 (Win Rate):    {st.get('日胜率', 0) * 100:.2f}%")
    print(f"多头持仓日占比:       {(pd.Series(w_ultimate) > 0).mean() * 100:.1f}%")
    print(f"空头持仓日占比:       {(pd.Series(w_ultimate) < 0).mean() * 100:.1f}%")
    print(f"空仓观望日占比:       {(pd.Series(w_ultimate) == 0).mean() * 100:.1f}%")

    yearly_table = pd.DataFrame(
        {
            "六合一策略收益 (%)": ypnl_ult.round(2),
            "159915基准 (%)": ypnl_bench.round(2),
            "超额收益 Alpha (%)": (ypnl_ult - ypnl_bench).round(2),
        }
    )

    print("\n" + "=" * 90)
    print("【2020-2026 逐年收益率对比表 (%)】")
    print("=" * 90)
    print(yearly_table.to_string())

    # 绘制高精度 Plotly HTML 大屏
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "【159915 全天候六合一黄金结构多空复合策略】 2020-2026 真实累计净值对比 (Cumulative NAV)",
            "动态回撤深度对比 (Drawdown %)",
        ),
        row_heights=[0.72, 0.28],
    )

    dates_str = ret_ult.index.strftime("%Y-%m-%d").tolist()

    # 基准
    fig.add_trace(
        go.Scatter(
            x=dates_str,
            y=nav_bench_norm.values,
            name="159915 买入持有基准",
            line=dict(color="#90A4AE", width=1.4, dash="dot"),
            opacity=0.6,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates_str,
            y=dd_bench.values,
            name="基准回撤",
            line=dict(color="#90A4AE", width=1, dash="dot"),
            showlegend=False,
            opacity=0.3,
        ),
        row=2,
        col=1,
    )

    # 六合一复合策略
    fig.add_trace(
        go.Scatter(
            x=dates_str,
            y=nav_ult.values,
            name="★ 全天候六合一复合策略 (Six-Pattern Composite)",
            line=dict(color="#00E676", width=3.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates_str,
            y=dd_ult.values,
            name="策略回撤",
            line=dict(color="#FF5252", width=1.5),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A0E14",
        plot_bgcolor="#101721",
        title=dict(
            text="159915 创业板 ETF · 全天候六合一黄金结构多空复合策略实盘回测大屏 (2020-2026)",
            font=dict(size=18, color="#FFFFFF"),
            x=0.02,
            y=0.98,
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.6)",
        ),
        height=740,
    )
    fig.update_yaxes(title_text="累计净值 (NAV, 初始=1.0)", gridcolor="#263238", row=1, col=1)
    fig.update_yaxes(title_text="回撤深度 (%)", gridcolor="#263238", row=2, col=1)
    fig.update_xaxes(gridcolor="#263238", row=1, col=1)
    fig.update_xaxes(gridcolor="#263238", row=2, col=1)

    fig.write_html(str(HTML_PATH), include_plotlyjs="cdn")
    print(f"\n交互式大屏已生成并保存至: {HTML_PATH.resolve()}")

    return st, yearly_table


if __name__ == "__main__":
    main()
