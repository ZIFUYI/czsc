#!/usr/bin/env python
"""
159915 (创业板ETF) 法医级纯化版 · 多周期分形共振生产级策略
(Forensic-Purified Multi-Timeframe Fractal Resonance Strategy: 1d + 60m + 30m)

核心架构：
1. 严选通过样本内 (2020-2023) 法医级三重屏障检验 (Purged + Next-Open + 2BP + Bootstrap + FDR) 的【五大真金信号】：
   - [多头真金 1] S1L_中枢三买确认：窄幅箱体突破 + 4根K线内回踩不破箱顶 + 阳线过前高
   - [多头真金 2] S2L_均线粘合一阳穿线：MA5/10/20/30 极度收敛 + 单根长阳贯穿四线
   - [多头真金 3] S3L_内孕线放量突破：Inside Bar 收敛 + 1.2倍放量阳线突破母线高点
   - [空头真金 4] S1S_中枢三卖确认：窄幅箱体跌破 + 4根K线内反抽不破箱底 + 阴线破前低
   - [空头真金 5] S4S_VCP压缩空头爆发：ATR 处于近60期最低20%分位 + 1.5倍ATR长阴破位
2. 剔除所有被法医检验判定为伪显著/负安全垫的 7 个冗余信号（如简单回踩MA20、布林2B摸底摸顶等）。
3. 在【日线 (scale=1.0) + 60分钟 (scale=0.50) + 30分钟 (scale=0.35)】三周期上实行分形同构叠加，
   全面检验全周期 (2020-2026) 与纯样本外盲测期 (2024-2026 OOS) 的实盘净值表现。
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
HTML_PATH = OUTPUT_DIR / "strategy_forensic_multitimeframe_159915.html"
IS_END_DT = pd.Timestamp("2023-12-31 23:59:59")


def load_raw_5m(csv_path: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
    df_raw = (
        df_raw.rename(
            columns={
                "SZSE.159915.open": "open",
                "SZSE.159915.high": "high",
                "SZSE.159915.low": "low",
                "SZSE.159915.close": "close",
                "SZSE.159915.volume": "volume",
            }
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    df_raw["date"] = df_raw["datetime"].dt.date
    df_raw["bar_in_day"] = df_raw.groupby("date").cumcount()
    return df_raw


def build_kline(df_raw: pd.DataFrame, bars_per_group: int) -> pd.DataFrame:
    """
    按日内固定 5m 根数合成等长 K 线：
    - 48 根 = 日线 (每天 1 根)
    - 12 根 = 60分钟 (每天 4 根)
    - 6 根  = 30分钟 (每天 8 根)
    """
    df_tmp = df_raw.copy()
    df_tmp["grp"] = df_tmp["bar_in_day"] // bars_per_group
    df = df_tmp.groupby(["date", "grp"], as_index=False).agg(
        dt=("datetime", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    df["symbol"] = "159915"
    return df


def run_forensic_fractal_engine(df: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """
    在给定周期上同时计算：
    - weight_gold: 仅包含通过法医检验的 5 大真金信号 (S1L, S1S, S2L, S3L, S4S)
    - weight_rejected: 仅包含被法医淘汰的 7 个弱/伪信号 (S2S, S3S, S4L, S5L, S5S, S6L, S6S)
    - weight_all: 未经纯化的全信号混合版本
    """
    df = df.copy()
    df["pct_chg"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma30"] = df["close"].rolling(30).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["total_range"] = df["high"] - df["low"] + 1e-6

    std20 = df["close"].rolling(20).std()
    df["boll_up"] = df["ma20"] + 2.0 * std20
    df["boll_dn"] = df["ma20"] - 2.0 * std20

    ma_max = df[["ma5", "ma10", "ma20", "ma30"]].max(axis=1)
    ma_min = df[["ma5", "ma10", "ma20", "ma30"]].min(axis=1)
    df["ma_dispersion"] = (ma_max - ma_min) / ma_min

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = df["tr"].rolling(14).mean()
    df["atr_rank_60"] = df["atr14"].rolling(60).apply(lambda s: pd.Series(s).rank(pct=True).iloc[-1], raw=False)

    df["prev_high_box"] = df["high"].shift(1).rolling(20).max()
    df["prev_low_box"] = df["low"].shift(1).rolling(20).min()
    df["box_amp"] = (df["prev_high_box"] - df["prev_low_box"]) / df["prev_low_box"]

    box_amp_th = 0.09 * scale
    disp_th = 0.02 * scale
    be_th = 0.02 * scale
    act_th = 0.03 * scale
    trail_th = 0.03 * scale
    pierce_th = 0.015 * scale

    n = len(df)
    c_arr = df["close"].values
    o_arr = df["open"].values
    h_arr = df["high"].values
    l_arr = df["low"].values
    ma5_arr = df["ma5"].values
    ma10_arr = df["ma10"].values
    ma20_arr = df["ma20"].values
    ma30_arr = df["ma30"].values
    ma60_arr = df["ma60"].values
    vol_arr = df["volume"].values
    vma20_arr = df["vol_ma20"].values
    amp_arr = df["box_amp"].values
    btop_arr = df["prev_high_box"].values
    bbot_arr = df["prev_low_box"].values
    disp_arr = df["ma_dispersion"].values
    atr_arr = df["atr14"].values
    arank_arr = df["atr_rank_60"].values
    rng_arr = df["total_range"].values
    bup_arr = df["boll_up"].values
    bdn_arr = df["boll_dn"].values

    def run_single_signal_sm(sig_code: str) -> np.ndarray:
        """
        对单个特定信号运行带结构止损、保本提损、移动止盈与均线破位离场的独立状态机。
        """
        pos = 0
        stop_p = entry_p = peak_p = trough_p = None
        pend_l = pend_s = False
        l_idx = s_idx = -999
        l_btop = s_bbot = 0.0
        l_ret = 9999.0
        s_reb = 0.0
        out = np.zeros(n)

        is_mean_rev = sig_code in ("S6L", "S6S")

        for i in range(60, n):
            c, o, h, l = c_arr[i], o_arr[i], h_arr[i], l_arr[i]
            ma20, ma60 = ma20_arr[i], ma60_arr[i]

            exited = False
            if pos == 1:
                peak_p = max(peak_p or c, h)
                unreal = (peak_p - entry_p) / entry_p
                if unreal >= be_th:
                    stop_p = max(stop_p, entry_p * (1 + 0.002 * scale))
                if (
                    stop_p is not None
                    and l <= stop_p
                    or unreal >= act_th
                    and (peak_p - c) / peak_p >= trail_th
                    or not is_mean_rev
                    and c < ma20
                    or is_mean_rev
                    and c >= ma20
                ):
                    pos = 0
                    exited = True

            elif pos == -1:
                trough_p = min(trough_p or c, l)
                unreal = (entry_p - trough_p) / entry_p
                if unreal >= be_th:
                    stop_p = min(stop_p, entry_p * (1 - 0.002 * scale))
                if (
                    stop_p is not None
                    and h >= stop_p
                    or unreal >= act_th
                    and (c - trough_p) / trough_p >= trail_th
                    or not is_mean_rev
                    and c > ma20
                    or is_mean_rev
                    and c <= ma20
                ):
                    pos = 0
                    exited = True

            if exited:
                stop_p = entry_p = peak_p = trough_p = None

            # --- 信号触发逻辑 ---
            if sig_code == "S1L":
                if pos == 0 and not pend_l and amp_arr[i] <= box_amp_th and c > ma60 and c > btop_arr[i]:
                    pend_l, l_idx, l_btop, l_ret = True, i, btop_arr[i], l
                if pend_l and (i - l_idx > 4):
                    pend_l = False
                if pos == 0 and pend_l and i > l_idx:
                    l_ret = min(l_ret, l)
                    if c < l_btop * (1 - 0.005 * scale):
                        pend_l = False
                    elif c > o and c >= h_arr[i - 1]:
                        pos, stop_p, entry_p, peak_p, pend_l = 1, max(l_ret, l_btop * (1 - 0.008 * scale)), c, c, False

            elif sig_code == "S1S":
                if pos == 0 and not pend_s and amp_arr[i] <= box_amp_th and c < ma60 and c < bbot_arr[i]:
                    pend_s, s_idx, s_bbot, s_reb = True, i, bbot_arr[i], h
                if pend_s and (i - s_idx > 4):
                    pend_s = False
                if pos == 0 and pend_s and i > s_idx:
                    s_reb = max(s_reb, h)
                    if c > s_bbot * (1 + 0.005 * scale):
                        pend_s = False
                    elif c < o and c <= l_arr[i - 1]:
                        pos, stop_p, entry_p, trough_p, pend_s = (
                            -1,
                            min(s_reb, s_bbot * (1 + 0.008 * scale)),
                            c,
                            c,
                            False,
                        )

            elif sig_code == "S2L":
                if pos == 0 and disp_arr[i] <= disp_th:
                    m_min = min(ma5_arr[i], ma10_arr[i], ma20)
                    m_max = max(ma5_arr[i], ma10_arr[i], ma20, ma30_arr[i])
                    if o <= m_min * (1 + 0.005 * scale) and c >= m_max and c > o * (1 + pierce_th) and c > ma60:
                        pos, entry_p, peak_p, stop_p = 1, c, c, l - 0.002 * scale

            elif sig_code == "S2S":
                if pos == 0 and disp_arr[i] <= disp_th:
                    m_min = min(ma5_arr[i], ma10_arr[i], ma20)
                    m_max = max(ma5_arr[i], ma10_arr[i], ma20, ma30_arr[i])
                    if o >= m_max * (1 - 0.005 * scale) and c <= m_min and c < o * (1 - pierce_th) and c < ma60:
                        pos, entry_p, trough_p, stop_p = -1, c, c, h + 0.002 * scale

            elif sig_code == "S3L":
                if (
                    pos == 0
                    and h_arr[i - 1] < h_arr[i - 2]
                    and l_arr[i - 1] > l_arr[i - 2]
                    and vol_arr[i] >= 1.2 * vma20_arr[i]
                ):
                    if c > h_arr[i - 2] and c > o and c > ma60:
                        pos, entry_p, peak_p, stop_p = 1, c, c, l_arr[i - 1] - 0.002 * scale

            elif sig_code == "S3S":
                if (
                    pos == 0
                    and h_arr[i - 1] < h_arr[i - 2]
                    and l_arr[i - 1] > l_arr[i - 2]
                    and vol_arr[i] >= 1.2 * vma20_arr[i]
                ):
                    if c < l_arr[i - 2] and c < o and c < ma60:
                        pos, entry_p, trough_p, stop_p = -1, c, c, h_arr[i - 1] + 0.002 * scale

            elif sig_code == "S4L":
                if pos == 0 and arank_arr[i] <= 0.20 and rng_arr[i] >= 1.5 * atr_arr[i]:
                    if c > ma60 and c > o and c > ma20:
                        pos, entry_p, peak_p, stop_p = 1, c, c, l - 0.002 * scale

            elif sig_code == "S4S":
                if pos == 0 and arank_arr[i] <= 0.20 and rng_arr[i] >= 1.5 * atr_arr[i]:
                    if c < ma60 and c < o and c < ma20:
                        pos, entry_p, trough_p, stop_p = -1, c, c, h + 0.002 * scale

            elif sig_code == "S5L":
                if (
                    pos == 0
                    and c > ma60
                    and ma20 > ma60
                    and c_arr[i - 1] > ma20_arr[i - 1]
                    and c_arr[i - 2] > ma20_arr[i - 2]
                ):
                    if l <= ma20 * (1 + 0.005 * scale) and c >= ma20 and c > o and c > h_arr[i - 1]:
                        pos, entry_p, peak_p, stop_p = 1, c, c, l - 0.002 * scale

            elif sig_code == "S5S":
                if (
                    pos == 0
                    and c < ma60
                    and ma20 < ma60
                    and c_arr[i - 1] < ma20_arr[i - 1]
                    and c_arr[i - 2] < ma20_arr[i - 2]
                ):
                    if h >= ma20 * (1 - 0.005 * scale) and c <= ma20 and c < o and c < l_arr[i - 1]:
                        pos, entry_p, trough_p, stop_p = -1, c, c, h + 0.002 * scale

            elif sig_code == "S6L":
                if pos == 0 and amp_arr[i] <= box_amp_th * 1.3:
                    if (
                        l_arr[i - 1] < bdn_arr[i - 1]
                        and c > bdn_arr[i]
                        and c > o
                        and c >= h_arr[i - 1]
                        and c > ma60 * 0.97
                    ):
                        pos, entry_p, peak_p, stop_p = 1, c, c, min(l, l_arr[i - 1]) - 0.002 * scale

            elif sig_code == "S6S":
                if pos == 0 and amp_arr[i] <= box_amp_th * 1.3:
                    if (
                        h_arr[i - 1] > bup_arr[i - 1]
                        and c < bup_arr[i]
                        and c < o
                        and c <= l_arr[i - 1]
                        and c < ma60 * 1.03
                    ):
                        pos, entry_p, trough_p, stop_p = -1, c, c, max(h, h_arr[i - 1]) + 0.002 * scale

            out[i] = pos
        return out

    # 1. 法医认证五大真金信号 (S1L, S1S, S2L, S3L, S4S)
    w_s1l = run_single_signal_sm("S1L")
    w_s1s = run_single_signal_sm("S1S")
    w_s2l = run_single_signal_sm("S2L")
    w_s3l = run_single_signal_sm("S3L")
    w_s4s = run_single_signal_sm("S4S")

    # 2. 法医淘汰七个伪/弱信号 (S2S, S3S, S4L, S5L, S5S, S6L, S6S)
    w_s2s = run_single_signal_sm("S2S")
    w_s3s = run_single_signal_sm("S3S")
    w_s4l = run_single_signal_sm("S4L")
    w_s5l = run_single_signal_sm("S5L")
    w_s5s = run_single_signal_sm("S5S")
    w_s6l = run_single_signal_sm("S6L")
    w_s6s = run_single_signal_sm("S6S")

    # 五大真金信号合成权重（多空能量配平：多头 S1L/S2L/S3L，空头 S1S/S4S）
    df["weight_gold"] = np.clip(
        0.40 * w_s1l + 0.35 * w_s2l + 0.30 * w_s3l + 0.50 * w_s1s + 0.50 * w_s4s,
        -1.0,
        1.0,
    )

    # 被淘汰七信号合成权重
    df["weight_rejected"] = np.clip(
        0.25 * (w_s2s + w_s3s + w_s4l + w_s5l + w_s5s + w_s6l + w_s6s),
        -1.0,
        1.0,
    )

    # 未纯化全信号合成权重
    df["weight_all"] = np.clip(
        0.25 * (w_s1l + w_s1s)
        + 0.20 * (w_s2l + w_s2s)
        + 0.15 * (w_s3l + w_s3s)
        + 0.15 * (w_s4l + w_s4s)
        + 0.15 * (w_s5l + w_s5s)
        + 0.15 * (w_s6l + w_s6s),
        -1.0,
        1.0,
    )
    return df


def count_round_trips(w: np.ndarray) -> tuple[int, int, int]:
    total = long_c = short_c = 0
    in_p = False
    cur_d = 0
    for val in w:
        d = 1 if val > 0.01 else (-1 if val < -0.01 else 0)
        if not in_p and d != 0:
            in_p = True
            cur_d = d
            total += 1
            if d == 1:
                long_c += 1
            else:
                short_c += 1
        elif in_p and d != cur_d:
            if d != 0:
                cur_d = d
                total += 1
                if d == 1:
                    long_c += 1
                else:
                    short_c += 1
            else:
                in_p = False
    return total, long_c, short_c


def evaluate_portfolio(
    df_sub: pd.DataFrame, weight_col: str, label: str, years_span: float
) -> tuple[dict, pd.Series, pd.Series]:
    dfw = pd.DataFrame(
        {
            "dt": df_sub["dt"].astype(str),
            "symbol": "159915",
            "weight": df_sub[weight_col].values,
            "price": df_sub["close"].values,
        }
    )
    wb = WeightBacktest(data=dfw, fee_rate=0.0002, weight_type="ts")
    st = wb.stats
    tot_t, l_t, s_t = count_round_trips(df_sub[weight_col].values)
    active_pct = (df_sub[weight_col].abs() > 0.01).mean() * 100

    dr = wb.daily_return.copy()
    dr["date"] = pd.to_datetime(dr["date"])
    ret_s = dr.set_index("date")["total"]
    nav_s = (1 + ret_s).cumprod()
    dd_s = (nav_s - nav_s.cummax()) / nav_s.cummax() * 100

    row = {
        "策略版本": label,
        "总交易笔数": f"{tot_t} 笔 ({l_t}多/{s_t}空)",
        "年均频次": f"{tot_t / years_span:.1f} 笔/年",
        "在场率": f"{active_pct:.1f}%",
        "累计收益": f"{st.get('绝对收益', 0) * 100:+.2f}%",
        "年化收益": f"{st.get('年化收益', 0) * 100:+.2f}%",
        "夏普比率": f"{st.get('夏普比率', 0):.2f}",
        "最大回撤": f"{st.get('最大回撤', 0) * 100:.2f}%",
        "卡玛比率": f"{st.get('卡玛比率', 0):.2f}",
        "日胜率": f"{st.get('日胜率', 0) * 100:.1f}%",
    }
    return row, nav_s, dd_s


def main():
    print("=" * 115)
    print("  【159915 法医纯化版 · 多周期分形共振生产级策略 (1d + 60m + 30m)】")
    print("  仅保留通过样本内 (2020-2023) 法医级三重屏障检验的 5 大真金信号，剔除 7 个伪/弱信号")
    print("=" * 115)

    df_raw = load_raw_5m(REAL_5M_CSV)

    # 1. 在 1d / 60m / 30m 三个周期上分别运行法医纯化分形引擎
    df_1d = run_forensic_fractal_engine(build_kline(df_raw, 48), scale=1.0)
    df_60m = run_forensic_fractal_engine(build_kline(df_raw, 12), scale=0.50)
    df_30m = run_forensic_fractal_engine(build_kline(df_raw, 6), scale=0.35)

    # 2. 在 30m 主时间轴上对齐合成三周期分形权重（无未来函数：日线/60m 收盘信号向后映射）
    df_multi = df_30m[["dt", "date", "close"]].copy()
    for mode_suffix in ("gold", "all", "rejected"):
        col = f"weight_{mode_suffix}"
        m_1d = df_1d.set_index("dt")[col]
        m_60m = df_60m.set_index("dt")[col]

        w30 = df_30m[col].values
        w60 = df_multi["dt"].map(m_60m).ffill().fillna(0.0).values
        w1d = df_multi["dt"].map(m_1d).ffill().fillna(0.0).values

        df_multi[col] = np.clip(0.65 * w1d + 0.55 * w60 + 0.45 * w30, -1.0, 1.0)

    # 3. 全周期 (2020-2026, 6.77年) 对比评估
    full_rows = []
    nav_map = {}
    dd_map = {}

    for col, label in [
        ("weight_gold", "★ 法医纯化 5真金 · 三周期共振 (1d+60m+30m)"),
        ("weight_all", "未纯化 12全信号 · 三周期共振 (原版)"),
        ("weight_rejected", "✗ 被法医淘汰 7伪信号 · 三周期合成"),
    ]:
        r, nav_s, dd_s = evaluate_portfolio(df_multi, col, label, years_span=6.77)
        full_rows.append(r)
        nav_map[label] = nav_s
        dd_map[label] = dd_s

    # 同时展示法医纯化版在单周期 (1d / 60m / 30m) 的独立表现
    for d_single, single_label in [
        (df_1d, "  ├─ 法医纯化 5真金 · 纯日线层 (1d)"),
        (df_60m, "  ├─ 法医纯化 5真金 · 60分钟层 (60m)"),
        (df_30m, "  └─ 法医纯化 5真金 · 30分钟层 (30m)"),
    ]:
        r_s, _, _ = evaluate_portfolio(d_single, "weight_gold", single_label, years_span=6.77)
        full_rows.append(r_s)

    print("\n" + "=" * 120)
    print("【一、全周期 (2020-2026) 生产级绩效对比：法医纯化 5真金 vs 未纯化 12信号 vs 被淘汰 7信号】")
    print("=" * 120)
    print(pd.DataFrame(full_rows).to_string(index=False))

    # 4. 纯样本外盲测期 (2024-01 ~ 2026-10 OOS, 2.77年) 严格对比
    df_multi_oos = df_multi[df_multi["dt"] > IS_END_DT].reset_index(drop=True)
    oos_rows = []
    for col, label in [
        ("weight_gold", "★ 法医纯化 5真金 · 样本外盲测 (2024-2026 OOS)"),
        ("weight_all", "未纯化 12全信号 · 样本外盲测 (2024-2026 OOS)"),
        ("weight_rejected", "✗ 被法医淘汰 7伪信号 · 样本外盲测 (2024-2026 OOS)"),
    ]:
        r_oos, _, _ = evaluate_portfolio(df_multi_oos, col, label, years_span=2.77)
        oos_rows.append(r_oos)

    print("\n" + "=" * 120)
    print("【二、2024-2026 纯样本外盲测 (Out-of-Sample) 绩效对比：验证去伪存真的样本外超额增益】")
    print("=" * 120)
    print(pd.DataFrame(oos_rows).to_string(index=False))

    # 5. 逐年收益率对比表 (2020-2026)
    gold_nav = nav_map["★ 法医纯化 5真金 · 三周期共振 (1d+60m+30m)"]
    all_nav = nav_map["未纯化 12全信号 · 三周期共振 (原版)"]
    rej_nav = nav_map["✗ 被法医淘汰 7伪信号 · 三周期合成"]

    gold_ret = gold_nav.pct_change().fillna(0)
    all_ret = all_nav.pct_change().fillna(0)
    rej_ret = rej_nav.pct_change().fillna(0)
    bench_ret = df_1d.set_index(pd.to_datetime(df_1d["dt"].dt.date))["close"].pct_change().fillna(0)

    common_idx = gold_ret.index.intersection(all_ret.index).intersection(bench_ret.index)
    gold_ret = gold_ret.loc[common_idx]
    all_ret = all_ret.loc[common_idx]
    rej_ret = rej_ret.loc[common_idx]
    bench_ret = bench_ret.loc[common_idx]

    yearly_df = pd.DataFrame(
        {
            "★ 法医纯化5真金 (%)": (
                gold_ret.groupby(gold_ret.index.year).apply(lambda s: (1 + s).prod() - 1) * 100
            ).round(2),
            "未纯化12信号 (%)": (all_ret.groupby(all_ret.index.year).apply(lambda s: (1 + s).prod() - 1) * 100).round(
                2
            ),
            "✗ 被淘汰7伪信号 (%)": (
                rej_ret.groupby(rej_ret.index.year).apply(lambda s: (1 + s).prod() - 1) * 100
            ).round(2),
            "159915基准 (%)": (bench_ret.groupby(bench_ret.index.year).apply(lambda s: (1 + s).prod() - 1) * 100).round(
                2
            ),
        }
    )
    yearly_df["纯化版 vs 淘汰版净胜 (%)"] = (yearly_df["★ 法医纯化5真金 (%)"] - yearly_df["✗ 被淘汰7伪信号 (%)"]).round(
        2
    )

    print("\n" + "=" * 105)
    print("【三、2020-2026 逐年收益率对比表 (%) —— 2024-2026 为纯样本外盲测期】")
    print("=" * 105)
    print(yearly_df.to_string())

    # 6. 生成交互式 HTML 大屏
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "【159915 法医纯化版 · 三周期分形共振策略】全周期与样本外盲测净值对比 (2020-2026)",
            "动态回撤深度对比 (Drawdown %)",
        ),
        row_heights=[0.72, 0.28],
    )

    dates_str = common_idx.strftime("%Y-%m-%d").tolist()
    bench_nav = (1 + bench_ret).cumprod()
    bench_dd = (bench_nav - bench_nav.cummax()) / bench_nav.cummax() * 100

    fig.add_trace(
        go.Scatter(
            x=dates_str,
            y=bench_nav.values,
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
            y=bench_dd.values,
            name="基准回撤",
            line=dict(color="#90A4AE", width=1, dash="dot"),
            showlegend=False,
            opacity=0.3,
        ),
        row=2,
        col=1,
    )

    style_cfg = {
        "★ 法医纯化 5真金 · 三周期共振 (1d+60m+30m)": ("#00E676", 3.4, "solid"),
        "未纯化 12全信号 · 三周期共振 (原版)": ("#29B6F6", 2.0, "solid"),
        "✗ 被法医淘汰 7伪信号 · 三周期合成": ("#FF5252", 2.0, "dash"),
    }

    for label, (col_hex, width_v, dash_v) in style_cfg.items():
        nav_a = nav_map[label].loc[common_idx]
        dd_a = dd_map[label].loc[common_idx]
        fig.add_trace(
            go.Scatter(x=dates_str, y=nav_a.values, name=label, line=dict(color=col_hex, width=width_v, dash=dash_v)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates_str, y=dd_a.values, name=f"{label} 回撤", line=dict(color=col_hex, width=1.4, dash=dash_v)
            ),
            row=2,
            col=1,
        )

    # 标注样本内与样本外分界线
    fig.add_vline(x="2024-01-02", line_width=2, line_dash="dash", line_color="#FFD54F")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A0E14",
        plot_bgcolor="#101721",
        title=dict(
            text="159915 创业板 ETF · 法医纯化五真金三周期分形共振生产级策略 (左侧样本内 2020-2023 | 右侧样本外盲测 2024-2026)",
            font=dict(size=17, color="#FFFFFF"),
            x=0.02,
            y=0.98,
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0.6)"),
        height=740,
    )
    fig.update_yaxes(title_text="累计净值 (NAV, 初始=1.0)", gridcolor="#263238", row=1, col=1)
    fig.update_yaxes(title_text="回撤深度 (%)", gridcolor="#263238", row=2, col=1)
    fig.update_xaxes(gridcolor="#263238", row=1, col=1)
    fig.update_xaxes(gridcolor="#263238", row=2, col=1)

    fig.write_html(str(HTML_PATH), include_plotlyjs="cdn")
    print(f"\n生产级交互式回测大屏已保存至: {HTML_PATH.resolve()}")


if __name__ == "__main__":
    main()
