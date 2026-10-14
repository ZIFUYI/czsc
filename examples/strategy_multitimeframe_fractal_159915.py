#!/usr/bin/env python
"""
159915 (创业板ETF) 跨周期分形共振与高频扩充复合策略
(Multi-Timeframe Fractal & High-Frequency Expansion Strategy)

解决日线信号频次偏低 (年均 10.6 笔、空仓率 70%) 的四大核心工程方案：
1. 方案一：周期下沉与波动率平方根缩放 (Timeframe Downscaling: 60m / 30m / 15m)
2. 方案二：多周期分形同构叠加 (Fractal Self-Similarity: 日线底仓 + 60m波段 + 30m滚动)
3. 方案三：补充震荡市均值回归结构 (Bollinger 2B Reversal: 填补 70% 震荡垃圾时间)
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
HTML_PATH = OUTPUT_DIR / "strategy_multitimeframe_fractal_159915.html"


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
    - 3 根  = 15分钟 (每天 16 根)
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


def run_fractal_patterns(df: pd.DataFrame, scale: float = 1.0, include_mean_reversion: bool = True) -> pd.DataFrame:
    """
    在任意周期上运行分形黄金结构引擎 (参数通过波动率平方根法则 scale 自动归一化)
    包含 6 大顺势动量结构 + 1 个震荡市布林带 2B 极值反转结构
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

    # 布林带 (用于震荡市 2B 均值回归)
    std20 = df["close"].rolling(20).std()
    df["boll_up"] = df["ma20"] + 2.0 * std20
    df["boll_dn"] = df["ma20"] - 2.0 * std20

    # 均线粘合度
    ma_max = df[["ma5", "ma10", "ma20", "ma30"]].max(axis=1)
    ma_min = df[["ma5", "ma10", "ma20", "ma30"]].min(axis=1)
    df["ma_dispersion"] = (ma_max - ma_min) / ma_min

    # ATR
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = df["tr"].rolling(14).mean()
    df["atr_rank_60"] = df["atr14"].rolling(60).apply(lambda s: pd.Series(s).rank(pct=True).iloc[-1], raw=False)

    # 滚动箱体
    df["prev_high_box"] = df["high"].shift(1).rolling(20).max()
    df["prev_low_box"] = df["low"].shift(1).rolling(20).min()
    df["box_amp"] = (df["prev_high_box"] - df["prev_low_box"]) / df["prev_low_box"]

    # 自适应缩放参数
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
    pct_arr = df["pct_chg"].values
    bup_arr = df["boll_up"].values
    bdn_arr = df["boll_dn"].values

    def run_sm(mode: int) -> np.ndarray:
        pos = 0
        stop_p = entry_p = peak_p = trough_p = None
        pend_l = pend_s = False
        l_idx = s_idx = -999
        l_btop = s_bbot = 0.0
        l_ret = 9999.0
        s_reb = 0.0
        out = np.zeros(n)

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
                    or mode != 7
                    and c < ma20
                    or mode == 7
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
                    or mode != 7
                    and c > ma20
                    or mode == 7
                    and c <= ma20
                ):
                    pos = 0
                    exited = True

            if exited:
                stop_p = entry_p = peak_p = trough_p = None

            # 1. 窄幅中枢突破 + 三买三卖
            if mode == 1:
                if pos == 0 and not pend_l and amp_arr[i] <= box_amp_th and c > ma60 and c > btop_arr[i]:
                    pend_l, l_idx, l_btop, l_ret = True, i, btop_arr[i], l
                elif pos == 0 and not pend_s and amp_arr[i] <= box_amp_th and c < ma60 and c < bbot_arr[i]:
                    pend_s, s_idx, s_bbot, s_reb = True, i, bbot_arr[i], h
                if pend_l and (i - l_idx > 4):
                    pend_l = False
                if pend_s and (i - s_idx > 4):
                    pend_s = False
                if pos == 0 and pend_l and i > l_idx:
                    l_ret = min(l_ret, l)
                    if c < l_btop * (1 - 0.005 * scale):
                        pend_l = False
                    elif c > o and c >= h_arr[i - 1]:
                        pos, stop_p, entry_p, peak_p, pend_l = 1, max(l_ret, l_btop * (1 - 0.008 * scale)), c, c, False
                elif pos == 0 and pend_s and i > s_idx:
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

            # 2. 均线粘合一阳穿/一阴下多线
            elif mode == 2:
                if pos == 0 and disp_arr[i] <= disp_th:
                    m_min = min(ma5_arr[i], ma10_arr[i], ma20)
                    m_max = max(ma5_arr[i], ma10_arr[i], ma20, ma30_arr[i])
                    if o <= m_min * (1 + 0.005 * scale) and c >= m_max and c > o * (1 + pierce_th) and c > ma60:
                        pos, entry_p, peak_p, stop_p = 1, c, c, l - 0.002 * scale
                    elif o >= m_max * (1 - 0.005 * scale) and c <= m_min and c < o * (1 - pierce_th) and c < ma60:
                        pos, entry_p, trough_p, stop_p = -1, c, c, h + 0.002 * scale

            # 3. 内孕线破局
            elif mode == 3:
                if (
                    pos == 0
                    and h_arr[i - 1] < h_arr[i - 2]
                    and l_arr[i - 1] > l_arr[i - 2]
                    and vol_arr[i] >= 1.2 * vma20_arr[i]
                ):
                    if c > h_arr[i - 2] and c > o and c > ma60:
                        pos, entry_p, peak_p, stop_p = 1, c, c, l_arr[i - 1] - 0.002 * scale
                    elif c < l_arr[i - 2] and c < o and c < ma60:
                        pos, entry_p, trough_p, stop_p = -1, c, c, h_arr[i - 1] + 0.002 * scale

            # 4. VCP 波动率极度压缩爆发
            elif mode == 4:
                if pos == 0 and arank_arr[i] <= 0.20 and rng_arr[i] >= 1.5 * atr_arr[i]:
                    if c > ma60 and c > o and c > ma20:
                        pos, entry_p, peak_p, stop_p = 1, c, c, l - 0.002 * scale
                    elif c < ma60 and c < o and c < ma20:
                        pos, entry_p, trough_p, stop_p = -1, c, c, h + 0.002 * scale

            # 5. 顺势首次回踩 MA20 企稳
            elif mode == 5:
                if pos == 0:
                    if c > ma60 and ma20 > ma60 and c_arr[i - 1] > ma20_arr[i - 1] and c_arr[i - 2] > ma20_arr[i - 2]:
                        if l <= ma20 * (1 + 0.005 * scale) and c >= ma20 and c > o and c > h_arr[i - 1]:
                            pos, entry_p, peak_p, stop_p = 1, c, c, l - 0.002 * scale
                    elif c < ma60 and ma20 < ma60 and c_arr[i - 1] < ma20_arr[i - 1] and c_arr[i - 2] < ma20_arr[i - 2]:
                        if h >= ma20 * (1 - 0.005 * scale) and c <= ma20 and c < o and c < l_arr[i - 1]:
                            pos, entry_p, trough_p, stop_p = -1, c, c, h + 0.002 * scale

            # 6. 顺势三连阴缩量洗盘反包
            elif mode == 6:
                if pos == 0:
                    if (
                        c > ma60
                        and c_arr[i - 1] < c_arr[i - 2] < c_arr[i - 3]
                        and vol_arr[i - 1] < vol_arr[i - 2] < vol_arr[i - 3]
                    ):
                        if (
                            c > o
                            and c >= h_arr[i - 1]
                            and vol_arr[i] >= 1.3 * vol_arr[i - 1]
                            and l_arr[i - 1] > ma60_arr[i - 1]
                        ):
                            pos, entry_p, peak_p, stop_p = 1, c, c, l - 0.002 * scale
                    elif (
                        c < ma60
                        and c_arr[i - 1] > c_arr[i - 2] > c_arr[i - 3]
                        and vol_arr[i - 1] <= 1.1 * vma20_arr[i - 1]
                    ):
                        if c < o and c <= l_arr[i - 1] and pct_arr[i] <= -0.015 * scale:
                            pos, entry_p, trough_p, stop_p = -1, c, c, h + 0.002 * scale

            # 7. 震荡市布林带轨线 2B 假破反包 (填补震荡期空白)
            elif mode == 7 and include_mean_reversion:
                if pos == 0 and amp_arr[i] <= box_amp_th * 1.3:
                    # 昨日刺破下轨，今日阳线反包站回下轨之上且超前高
                    if (
                        l_arr[i - 1] < bdn_arr[i - 1]
                        and c > bdn_arr[i]
                        and c > o
                        and c >= h_arr[i - 1]
                        and c > ma60 * 0.97
                    ):
                        pos, entry_p, peak_p, stop_p = 1, c, c, min(l, l_arr[i - 1]) - 0.002 * scale
                    # 昨日刺破上轨，今日阴线反杀跌回上轨之下且破前低
                    elif (
                        h_arr[i - 1] > bup_arr[i - 1]
                        and c < bup_arr[i]
                        and c < o
                        and c <= l_arr[i - 1]
                        and c < ma60 * 1.03
                    ):
                        pos, entry_p, trough_p, stop_p = -1, c, c, max(h, h_arr[i - 1]) + 0.002 * scale

            out[i] = pos
        return out

    w1 = run_sm(1)
    w2 = run_sm(2)
    w3 = run_sm(3)
    w4 = run_sm(4)
    w5 = run_sm(5)
    w6 = run_sm(6)
    w7 = run_sm(7) if include_mean_reversion else np.zeros(n)

    w_comp = np.clip(0.25 * w1 + 0.20 * w2 + 0.15 * w3 + 0.15 * w4 + 0.15 * w5 + 0.10 * w6 + 0.20 * w7, -1.0, 1.0)
    df["weight"] = w_comp
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


def main():
    print("=" * 95)
    print("  【159915 跨周期分形共振与信号扩充策略】 (日线 vs 60m vs 30m vs 三周期分形复合)")
    print("=" * 95)

    df_raw = load_raw_5m(REAL_5M_CSV)

    # 1. 构建各周期 K 线并运行策略
    df_1d = run_fractal_patterns(build_kline(df_raw, 48), scale=1.0, include_mean_reversion=False)
    df_60m = run_fractal_patterns(build_kline(df_raw, 12), scale=0.50, include_mean_reversion=True)
    df_30m = run_fractal_patterns(build_kline(df_raw, 6), scale=0.35, include_mean_reversion=True)

    # 2. 构建【日线(35%) + 60m(35%) + 30m(30%) 三周期分形共振复合策略】
    # 将日线与60m的最新信号映射到 30m 时间轴上 (严禁未来函数：日线信号在收盘时产生，对齐时间戳)
    df_multi = df_30m[["dt", "date", "close"]].copy()
    map_1d = df_1d.set_index("dt")["weight"]
    map_60m = df_60m.set_index("dt")["weight"]

    df_multi["w_30m"] = df_30m["weight"].values
    df_multi["w_60m"] = df_multi["dt"].map(map_60m).ffill().fillna(0.0)
    df_multi["w_1d"] = df_multi["dt"].map(map_1d).ffill().fillna(0.0)

    # 当多周期方向一致或互补时，复合权重叠加 (等风险敞口配平放大后截断在 [-1, 1])
    df_multi["weight"] = np.clip(
        0.65 * df_multi["w_1d"] + 0.55 * df_multi["w_60m"] + 0.45 * df_multi["w_30m"],
        -1.0,
        1.0,
    )

    configs = [
        ("纯日线六合一 (原方案)", df_1d),
        ("60分钟分形七合一 (中频)", df_60m),
        ("30分钟分形七合一 (高频)", df_30m),
        ("★ 三周期分形共振复合 (1d+60m+30m)", df_multi),
    ]

    summary_rows = []
    nav_dict = {}
    dd_dict = {}

    for label, d_sub in configs:
        dfw = pd.DataFrame(
            {
                "dt": d_sub["dt"].astype(str),
                "symbol": "159915",
                "weight": d_sub["weight"].values,
                "price": d_sub["close"].values,
            }
        )
        wb = WeightBacktest(data=dfw, fee_rate=0.0002, weight_type="ts")
        st = wb.stats
        tot_t, l_t, s_t = count_round_trips(d_sub["weight"].values)
        active_pct = (d_sub["weight"].abs() > 0.01).mean() * 100

        dr = wb.daily_return.copy()
        dr["date"] = pd.to_datetime(dr["date"])
        ret_s = dr.set_index("date")["total"]
        nav_s = (1 + ret_s).cumprod()
        dd_s = (nav_s - nav_s.cummax()) / nav_s.cummax() * 100

        nav_dict[label] = nav_s
        dd_dict[label] = dd_s

        summary_rows.append(
            {
                "策略方案": label,
                "总交易笔数": f"{tot_t} 笔",
                "多/空笔数": f"{l_t}多 / {s_t}空",
                "年均频次": f"{tot_t / 6.77:.1f} 笔/年",
                "资金在场率": f"{active_pct:.1f}%",
                "累计收益": f"{st.get('绝对收益', 0) * 100:+.2f}%",
                "年化收益": f"{st.get('年化收益', 0) * 100:+.2f}%",
                "夏普比率": f"{st.get('夏普比率', 0):.2f}",
                "最大回撤": f"{st.get('最大回撤', 0) * 100:.2f}%",
                "卡玛比率": f"{st.get('卡玛比率', 0):.2f}",
            }
        )

    print("\n" + "=" * 110)
    print("【159915 跨周期扩充方案 交易频次与终审绩效对比表 (2020-2026)】")
    print("=" * 110)
    print(pd.DataFrame(summary_rows).to_string(index=False))

    # 打印三周期分形共振复合策略的逐年收益
    multi_nav = nav_dict["★ 三周期分形共振复合 (1d+60m+30m)"]
    multi_ret = multi_nav.pct_change().fillna(0)
    oned_nav = nav_dict["纯日线六合一 (原方案)"]
    oned_ret = oned_nav.pct_change().fillna(0)

    bench_ret = df_1d.set_index(pd.to_datetime(df_1d["dt"].dt.date))["close"].pct_change().fillna(0)
    common_idx = multi_ret.index.intersection(oned_ret.index).intersection(bench_ret.index)

    multi_ret = multi_ret.loc[common_idx]
    oned_ret = oned_ret.loc[common_idx]
    bench_ret = bench_ret.loc[common_idx]

    y_multi = multi_ret.groupby(multi_ret.index.year).apply(lambda s: (1 + s).prod() - 1) * 100
    y_oned = oned_ret.groupby(oned_ret.index.year).apply(lambda s: (1 + s).prod() - 1) * 100
    y_bench = bench_ret.groupby(bench_ret.index.year).apply(lambda s: (1 + s).prod() - 1) * 100

    yearly_df = pd.DataFrame(
        {
            "★ 三周期分形复合 (%)": y_multi.round(2),
            "纯日线原方案 (%)": y_oned.round(2),
            "159915基准 (%)": y_bench.round(2),
            "分形超额 Alpha (%)": (y_multi - y_bench).round(2),
        }
    )

    print("\n" + "=" * 95)
    print("【三周期分形共振复合策略 2020-2026 逐年收益率对比表 (%)】")
    print("=" * 95)
    print(yearly_df.to_string())

    # 生成可视化 HTML
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "【159915 跨周期分形共振扩充策略】 2020-2026 真实累计净值对比 (Cumulative NAV)",
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

    colors = {
        "纯日线六合一 (原方案)": ("#FFB300", 1.8),
        "60分钟分形七合一 (中频)": ("#00E5FF", 1.6),
        "30分钟分形七合一 (高频)": ("#AB47BC", 1.6),
        "★ 三周期分形共振复合 (1d+60m+30m)": ("#00E676", 3.5),
    }

    for label, (col, w_line) in colors.items():
        nav_aligned = nav_dict[label].loc[common_idx]
        dd_aligned = dd_dict[label].loc[common_idx]
        fig.add_trace(
            go.Scatter(x=dates_str, y=nav_aligned.values, name=label, line=dict(color=col, width=w_line)), row=1, col=1
        )
        if "★" in label or "纯日线" in label:
            fig.add_trace(
                go.Scatter(x=dates_str, y=dd_aligned.values, name=f"{label} 回撤", line=dict(color=col, width=1.4)),
                row=2,
                col=1,
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A0E14",
        plot_bgcolor="#101721",
        title=dict(
            text="159915 创业板 ETF · 跨周期分形共振复合策略实盘回测大屏 (2020-2026)",
            font=dict(size=18, color="#FFFFFF"),
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
    print(f"\n交互式大屏已保存至: {HTML_PATH.resolve()}")


if __name__ == "__main__":
    main()
