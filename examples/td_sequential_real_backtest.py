#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复刻 d:\\my-qlib\\strategies\\td_strategy.py
在天勤 100% 真实 159915 创业板 ETF (2020-2026, 78672根 5m K线) 上执行完整回测
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUTPUT_DIR = Path("examples/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REAL_CSV = OUTPUT_DIR / "159915_real_5m_2020_2026.csv"
HTML_PATH = OUTPUT_DIR / "td_sequential_real_nav_chart.html"


def load_real_data(csv_path: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)
    df = pd.DataFrame()
    df["dt"] = pd.to_datetime(df_raw["datetime"])
    df["open"] = df_raw["SZSE.159915.open"].astype(float)
    df["high"] = df_raw["SZSE.159915.high"].astype(float)
    df["low"] = df_raw["SZSE.159915.low"].astype(float)
    df["close"] = df_raw["SZSE.159915.close"].astype(float)
    df["vol"] = df_raw["SZSE.159915.volume"].astype(float)
    df = df.dropna().sort_values("dt").reset_index(drop=True)
    return df


def run_td_strategy(df: pd.DataFrame, n1=5, n2=3, n3=6, fee_rate=0.0002):
    """
    1:1 严格复刻 td_strategy.py 的 TDSequentialStrategy 状态机逻辑
    fee_rate: 单边万分之二 (双边万分之四，4 BP)
    """
    open_time = time(9, 30)
    no_open_time = time(14, 45)
    exit_time = time(14, 55)

    last_date = None
    last_ud = 0
    ud_run = 0

    buy_setup = False
    buy_count = 0
    buy_k1_close = None
    buy_count_low = float("inf")
    buy_count_high = float("-inf")

    sell_setup = False
    sell_count = 0
    sell_k1_close = None
    sell_count_high = float("-inf")

    pos = 0  # 1=多头, -1=空头, 0=空仓
    entry_price = 0.0
    entry_dt = None
    stop_price = None

    trades = []
    daily_records = []

    # 逐根 5 分钟 K 线驱动
    for i in range(len(df)):
        row = df.iloc[i]
        dt = row["dt"]
        c_now = row["close"]
        h_now = row["high"]
        l_now = row["low"]
        cur_date = dt.date()
        cur_time = dt.time()

        # 1. 每日开盘重置状态 (对应 _reset_daily_state)
        if last_date != cur_date:
            last_date = cur_date
            last_ud = 0
            ud_run = 0
            buy_setup = False
            buy_count = 0
            buy_k1_close = None
            buy_count_low = float("inf")
            buy_count_high = float("-inf")
            sell_setup = False
            sell_count = 0
            sell_k1_close = None
            sell_count_high = float("-inf")
            pos = 0
            entry_price = 0.0
            entry_dt = None
            stop_price = None

        # 2. 14:55 收盘强平 (对应收盘强平逻辑，日内不隔夜)
        if cur_time >= exit_time:
            if pos != 0:
                raw_pnl = (c_now - entry_price) / entry_price if pos == 1 else (entry_price - c_now) / entry_price
                net_pnl = raw_pnl - 2 * fee_rate
                trades.append({
                    "entry_dt": entry_dt,
                    "exit_dt": dt,
                    "dir": pos,
                    "entry_price": entry_price,
                    "exit_price": c_now,
                    "raw_pnl": raw_pnl,
                    "net_pnl": net_pnl,
                    "reason": "14:55收盘强平",
                })
                pos = 0
                stop_price = None
            daily_records.append({"dt": dt, "pos": 0, "price": c_now})
            continue

        # 3. 盘中动态止损检查 (对应止损价判断)
        if pos != 0 and stop_price is not None:
            if pos == 1 and l_now <= stop_price:
                # 多头触发止损
                raw_pnl = (stop_price - entry_price) / entry_price
                net_pnl = raw_pnl - 2 * fee_rate
                trades.append({
                    "entry_dt": entry_dt,
                    "exit_dt": dt,
                    "dir": 1,
                    "entry_price": entry_price,
                    "exit_price": stop_price,
                    "raw_pnl": raw_pnl,
                    "net_pnl": net_pnl,
                    "reason": "多头极值止损",
                })
                pos = 0
                stop_price = None
                daily_records.append({"dt": dt, "pos": 0, "price": c_now})
                continue
            elif pos == -1 and h_now >= stop_price:
                # 空头触发止损
                raw_pnl = (entry_price - stop_price) / entry_price
                net_pnl = raw_pnl - 2 * fee_rate
                trades.append({
                    "entry_dt": entry_dt,
                    "exit_dt": dt,
                    "dir": -1,
                    "entry_price": entry_price,
                    "exit_price": stop_price,
                    "raw_pnl": raw_pnl,
                    "net_pnl": net_pnl,
                    "reason": "空头极值止损",
                })
                pos = 0
                stop_price = None
                daily_records.append({"dt": dt, "pos": 0, "price": c_now})
                continue

        # 预热期
        if i < n1 or i < 2:
            daily_records.append({"dt": dt, "pos": pos, "price": c_now})
            continue

        # 4. 计算价格差分方向 ud
        c_lag = df["close"].iloc[i - n1]
        if c_now > c_lag:
            ud = 1
        elif c_now < c_lag:
            ud = -1
        else:
            ud = 0

        # 5. 更新连续段计数器 ud_run
        if ud == last_ud and ud != 0:
            ud_run += ud
        else:
            ud_run = ud
        last_ud = ud

        # 6. 检查启动 Setup (并行状态机)
        if ud_run == n2:
            sell_setup = True
            sell_count = 0
            sell_k1_close = None
            sell_count_high = float("-inf")
        elif ud_run == -n2:
            buy_setup = True
            buy_count = 0
            buy_k1_close = None
            buy_count_low = float("inf")
            buy_count_high = float("-inf")

        # 7. 计数逻辑 (三重微观确认)
        if buy_setup:
            condA = c_now >= df["high"].iloc[i - 2]
            condB = h_now > df["high"].iloc[i - 1]
            condC = (buy_k1_close is None) or (c_now > buy_k1_close)
            if condA and condB and condC:
                buy_count += 1
                if buy_count == 1:
                    buy_k1_close = c_now
                buy_count_low = min(buy_count_low, l_now)
                buy_count_high = max(buy_count_high, h_now)

        if sell_setup:
            condA = c_now <= df["low"].iloc[i - 2]
            condB = l_now < df["low"].iloc[i - 1]
            condC = (sell_k1_close is None) or (c_now < sell_k1_close)
            if condA and condB and condC:
                sell_count += 1
                if sell_count == 1:
                    sell_k1_close = c_now
                sell_count_high = max(sell_count_high, h_now)

        # 8. 信号触发 (开仓时间窗口: 09:30 ~ 14:45)
        in_open_window = open_time <= cur_time < no_open_time
        buy_ready = buy_count >= n3
        sell_ready = sell_count >= n3

        if in_open_window:
            # 买入做多 (买 Call)
            if buy_ready and pos <= 0:
                if pos < 0:
                    raw_pnl = (entry_price - c_now) / entry_price
                    net_pnl = raw_pnl - 2 * fee_rate
                    trades.append({
                        "entry_dt": entry_dt,
                        "exit_dt": dt,
                        "dir": -1,
                        "entry_price": entry_price,
                        "exit_price": c_now,
                        "raw_pnl": raw_pnl,
                        "net_pnl": net_pnl,
                        "reason": "反手平空做多",
                    })
                pos = 1
                entry_price = c_now
                entry_dt = dt
                stop_price = buy_count_low
                buy_setup = False
                buy_count = 0

            # 卖出做空 (买 Put)
            elif sell_ready and pos >= 0:
                if pos > 0:
                    raw_pnl = (c_now - entry_price) / entry_price
                    net_pnl = raw_pnl - 2 * fee_rate
                    trades.append({
                        "entry_dt": entry_dt,
                        "exit_dt": dt,
                        "dir": 1,
                        "entry_price": entry_price,
                        "exit_price": c_now,
                        "raw_pnl": raw_pnl,
                        "net_pnl": net_pnl,
                        "reason": "反手平多做空",
                    })
                pos = -1
                entry_price = c_now
                entry_dt = dt
                stop_price = sell_count_high
                sell_setup = False
                sell_count = 0

        daily_records.append({"dt": dt, "pos": pos, "price": c_now})

    return pd.DataFrame(trades), pd.DataFrame(daily_records)


def main():
    print("=" * 75)
    print("  加载天勤 100% 真实 159915 创业板 ETF 数据 (2020-2026, 78672 根 5m K线)...")
    print("=" * 75)

    fee_rate = 0.0002
    df_real = load_real_data(REAL_CSV)
    df_trades, df_holds = run_td_strategy(df_real, n1=5, n2=3, n3=6, fee_rate=fee_rate)

    print(f"数据行数: {len(df_real)} | 回测区间: {df_real['dt'].min()} ~ {df_real['dt'].max()}")
    print(f"总交易笔数: {len(df_trades)} 笔 (多头做多: {(df_trades['dir']==1).sum()} 笔, 空头做空: {(df_trades['dir']==-1).sum()} 笔)")

    # 计算交易指标
    pnl = df_trades["net_pnl"].values
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = len(wins) / len(pnl) * 100
    pl_ratio = wins.mean() / abs(losses.mean()) if len(losses) > 0 else np.nan

    print("\n" + "=" * 75)
    print("【实盘 TDSequential 策略在天勤 7年真实行情上的微观指标】")
    print("=" * 75)
    print(f"  实际胜率:          {win_rate:.2f}%")
    print(f"  盈亏比 (P/L Ratio): {pl_ratio:.2f}")
    print(f"  单笔平均净收益:    {pnl.mean()*10000:.2f} BP ({pnl.mean()*100:.3f}%)")
    print(f"  累计单利收益:      {pnl.sum()*100:.2f}%")

    # 构建每日收益序列 (严格扣除每笔换仓手续费)
    df_holds["date"] = df_holds["dt"].dt.strftime("%Y-%m-%d")
    df_holds["raw_ret"] = df_holds["price"].pct_change().fillna(0) * df_holds["pos"].shift(1).fillna(0)
    df_holds["fee"] = df_holds["pos"].diff().abs().fillna(0) * fee_rate
    df_holds["net_ret"] = df_holds["raw_ret"] - df_holds["fee"]

    # 每日累计收益 (扣费后)
    daily_pnl = df_holds.groupby("date")["net_ret"].sum()
    nav_strategy = (1 + daily_pnl).cumprod()

    # 基准 159915 现货
    bench_close = df_real.groupby(df_real["dt"].dt.strftime("%Y-%m-%d"))["close"].last()
    bench_ret = bench_close.pct_change().fillna(0)
    bench_nav = (1 + bench_ret).cumprod()
    bench_nav = bench_nav.reindex(nav_strategy.index).ffill()
    bench_nav = bench_nav / bench_nav.iloc[0]

    # 分年度表现
    yearly_df = pd.DataFrame({
        "TD多空策略(扣费后 %)": df_holds.groupby(df_holds["dt"].dt.year)["net_ret"].sum() * 100,
        "159915现货基准 (%)": bench_ret.groupby(pd.to_datetime(bench_ret.index).year).sum() * 100,
    })

    print("\n" + "=" * 75)
    print("【分年度收益统计表 (2020-2026, 严格扣费后)】")
    print("=" * 75)
    print(yearly_df.round(2).to_string())

    # 计算夏普与最大回撤
    ann_ret = (nav_strategy.iloc[-1] ** (252 / len(daily_pnl))) - 1
    ann_vol = daily_pnl.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else np.nan
    cummax = nav_strategy.cummax()
    dd = (nav_strategy - cummax) / cummax
    mdd = dd.min()

    print("\n" + "=" * 75)
    print("【7年全周期量化绩效总览】")
    print("=" * 75)
    print(f"  7年绝对累计收益:  {(nav_strategy.iloc[-1]-1)*100:.2f}%")
    print(f"  年化收益率:        {ann_ret*100:.2f}%")
    print(f"  夏普比率 (Sharpe): {sharpe:.2f}")
    print(f"  最大回撤 (MDD):    {mdd*100:.2f}%")
    print(f"  卡玛比率 (Calmar): {abs(ann_ret/mdd):.2f}")

    # 绘制专业交互图
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "【TD Sequential 策略实盘逻辑 · 天勤 100% 真实数据 2020-2026】累计净值走势",
            "动态回撤深度 (%)",
        ),
        row_heights=[0.72, 0.28],
    )

    dates = nav_strategy.index.tolist()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=bench_nav.values,
            name="159915 现货买入持有 (基准)",
            line=dict(color="#78909C", width=1.6, dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=nav_strategy.values,
            name="TD多空双向策略 (实盘逻辑)",
            line=dict(color="#00E676", width=2.8),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=(dd * 100).values,
            name="TD策略回撤",
            line=dict(color="#FF5252", width=1.2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A0E14",
        plot_bgcolor="#101721",
        title=dict(
            text="TD 序列实盘策略 · 天勤 100% 真实数据全周期回测大屏 (2020-2026)",
            font=dict(size=18, color="#FFFFFF"),
            x=0.02,
            y=0.98,
        ),
        hovermode="x unified",
        margin=dict(l=50, r=30, t=100, b=50),
        height=750,
    )

    fig.update_yaxes(title_text="累计净值 (NAV, 初始=1.0)", gridcolor="#263238", row=1, col=1)
    fig.update_yaxes(title_text="回撤幅度 (%)", gridcolor="#263238", row=2, col=1)
    fig.update_xaxes(gridcolor="#263238", row=1, col=1)
    fig.update_xaxes(gridcolor="#263238", row=2, col=1)

    fig.write_html(str(HTML_PATH), include_plotlyjs="cdn")
    print(f"\n交互式图表已成功保存至: {HTML_PATH.resolve()}")


if __name__ == "__main__":
    main()
