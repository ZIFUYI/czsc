"""强多头趋势过滤 + 15分钟一买策略 (中证1000 000852.XSHG)

策略核心思想：
- 大周期顺势过滤：日线 MA20 处于“多头且向上”（close >= MA20 且 MA20 斜率向上）；
- 小周期逆势进场：15分钟级别出现缠论标准第一类买点（底背驰一买，cxt_first_buy_V221126）；
- 动态平仓与风控：
  1. 15分钟级别出现缠论第一类卖点（一卖平多）；
  2. 日线级别趋势转弱走坏（MA20 斜率转头向下时主动离场）；
  3. 硬止损：200 BP (2.0%)；
  4. 最大持仓时限：5 个交易日（80 根 15 分钟 K 线超时强制平仓）；
  5. 交易冷静期：开仓后 2 小时内不重复开仓。

运行：
    uv run --no-sync python examples/signals_dev/backtest_trend_first_buy_000852_15m.py
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jqdatasdk as jq
from czsc import CzscStrategyBase, Event, Freq, Position, format_standard_kline

OUTPUT_DIR = ROOT / "examples" / "results" / "trend_first_buy_000852_15m"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "000852.XSHG"
START_DATE = "2021-01-01"
END_DATE = "2026-09-11"
BACKTEST_SDT = "2021-06-01"


class StrongTrendFirstBuyStrategy(CzscStrategyBase):
    """日线 MA20 多头向上 + 15分钟一买 联立策略"""

    @property
    def positions(self) -> list[Position]:
        # 1. 开多事件：必须同时满足（signals_all）日线多头向上 与 15分钟一买
        open_events = [
            Event.load(
                {
                    "name": "开多_日线MA20多头向上且15分一买",
                    "operate": "开多",
                    "signals_all": [
                        "日线_D1SMA#20_分类V221101_多头_向上_任意_0",
                        "15分钟_D1B_BUY1_一买_任意_任意_0",
                    ],
                }
            )
        ]

        # 2. 平多事件：15分一卖 或 日线趋势转弱向下
        exit_events = [
            Event.load(
                {
                    "name": "平多_15分一卖",
                    "operate": "平多",
                    "signals_all": ["15分钟_D1B_SELL1_一卖_任意_任意_0"],
                }
            ),
            Event.load(
                {
                    "name": "平多_日线MA20向下走坏",
                    "operate": "平多",
                    "signals_all": ["日线_D1SMA#20_分类V221101_任意_向下_任意_0"],
                }
            ),
        ]

        # 3. 仓位与风控规则
        pos = Position(
            name="日线多头向上_15分一买",
            symbol=self.symbol,
            opens=open_events,
            exits=exit_events,
            interval=3600 * 2,  # 开仓最小间隔 2 小时（秒）
            timeout=16 * 5,     # 最长持有 5 个交易日（80 根 15 分钟 K 线）
            stop_loss=200,      # 200 BP 止损（2.0%）
            t0=False,           # 非 T+0
        )
        return [pos]

    @property
    def signals_config(self) -> list[dict]:
        """显式定义多级别计算所需的信号函数配置"""
        return [
            {"name": "tas_ma_base_V221101", "freq": "日线", "di": 1, "timeperiod": 20, "ma_type": "SMA"},
            {"name": "cxt_first_buy_V221126", "freq": "15分钟", "di": 1},
            {"name": "cxt_first_sell_V221126", "freq": "15分钟", "di": 1},
        ]


def fetch_and_prepare_data(symbol: str = SYMBOL) -> tuple[pd.DataFrame, list]:
    """从 JQData 获取 15 分钟 K 线并格式化为 RawBar 列表"""
    jq.auth("15901878101", "w5qredw")
    print(f"正在从 JQData 拉取 {symbol} 15分钟 K 线 ({START_DATE} ~ {END_DATE})...")
    df = jq.get_price(symbol, start_date=START_DATE, end_date=END_DATE, frequency="15m")
    df = df.reset_index().rename(columns={"index": "dt", "volume": "vol", "money": "amount"})
    df["symbol"] = symbol
    print(f"成功获取 {len(df)} 根 15 分钟 K 线。")
    raw_bars = format_standard_kline(df, freq=Freq.F15)
    return df, raw_bars


def analyze_pairs(pairs: pd.DataFrame) -> dict:
    """分析开平交易对核心指标"""
    if pairs.empty:
        return {"总交易数": 0}

    ret_pct = pairs["盈亏比例"] / 100.0  # BP 转百分比
    wins = ret_pct[ret_pct > 0]
    losses = ret_pct[ret_pct <= 0]

    stats = {
        "总交易对数": len(pairs),
        "盈利笔数": len(wins),
        "亏损笔数": len(losses),
        "交易胜率": f"{(len(wins) / len(pairs)) * 100:.1f}%",
        "单笔均收益": f"{ret_pct.mean():+.2f}%",
        "累计收益": f"{ret_pct.sum():+.2f}%",
        "单笔最大盈利": f"{ret_pct.max():+.2f}%",
        "单笔最大亏损": f"{ret_pct.min():+.2f}%",
        "单笔均盈利": f"{wins.mean():+.2f}%" if len(wins) > 0 else "0.00%",
        "单笔均亏损": f"{losses.mean():+.2f}%" if len(losses) > 0 else "0.00%",
        "盈亏比": f"{abs(wins.mean() / losses.mean()):.2f}" if len(losses) > 0 and len(wins) > 0 else "N/A",
        "平均持仓天数": f"{pairs['持仓天数'].mean():.1f} 天",
    }
    return stats


def generate_interactive_chart(df_15m: pd.DataFrame, pairs: pd.DataFrame, holds: pd.DataFrame) -> Path:
    """生成包含买卖点标记与持仓净值的 Plotly 交互式 HTML 报告"""
    df_plot = df_15m.copy()
    df_plot["dt_str"] = pd.to_datetime(df_plot["dt"]).dt.strftime("%Y-%m-%d %H:%M")

    # 构建持仓净值曲线
    holds_plot = holds.copy()
    holds_plot["dt_str"] = pd.to_datetime(holds_plot["dt"]).dt.strftime("%Y-%m-%d %H:%M")
    df_merged = df_plot.merge(holds_plot[["dt_str", "pos"]], on="dt_str", how="left").fillna({"pos": 0.0})
    df_merged["ret"] = df_merged["close"].pct_change().fillna(0.0)
    df_merged["strat_ret"] = df_merged["pos"].shift(1).fillna(0.0) * df_merged["ret"]
    df_merged["equity"] = (1 + df_merged["strat_ret"]).cumprod()
    df_merged["bench_equity"] = (1 + df_merged["ret"]).cumprod()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"中证1000 ({SYMBOL}) 15分钟走势与强趋势一买开平仓点",
            "策略累计净值 vs 基准买入持有",
        ),
        row_heights=[0.65, 0.35],
    )

    # 1. 价格曲线
    fig.add_trace(
        go.Scatter(
            x=df_merged["dt_str"],
            y=df_merged["close"],
            mode="lines",
            name="中证1000收盘价",
            line={"color": "#666666", "width": 1.2},
        ),
        row=1,
        col=1,
    )

    # 2. 开仓点与平仓点标记
    if not pairs.empty:
        open_dts = pd.to_datetime(pairs["开仓时间"]).dt.strftime("%Y-%m-%d %H:%M")
        exit_dts = pd.to_datetime(pairs["平仓时间"]).dt.strftime("%Y-%m-%d %H:%M")

        fig.add_trace(
            go.Scatter(
                x=open_dts,
                y=pairs["开仓价格"],
                mode="markers",
                name="强多头一买入场 (开多)",
                marker={"symbol": "triangle-up", "size": 10, "color": "#e74c3c", "line": {"width": 1.5, "color": "darkred"}},
                text=[f"开仓价: {p:.2f}" for p in pairs["开仓价格"]],
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=exit_dts,
                y=pairs["平仓价格"],
                mode="markers",
                name="平多离场 (一卖/趋势转弱/止损)",
                marker={"symbol": "triangle-down", "size": 10, "color": "#2ecc71", "line": {"width": 1.5, "color": "darkgreen"}},
                text=[f"平仓价: {p:.2f}<br>盈亏: {r:.2f}BP" for p, r in zip(pairs["平仓价格"], pairs["盈亏比例"], strict=True)],
            ),
            row=1,
            col=1,
        )

    # 3. 净值曲线
    fig.add_trace(
        go.Scatter(
            x=df_merged["dt_str"],
            y=df_merged["bench_equity"],
            mode="lines",
            name="基准买入持有",
            line={"color": "#95a5a6", "dash": "dash", "width": 1.5},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_merged["dt_str"],
            y=df_merged["equity"],
            mode="lines",
            name="强多头一买策略净值",
            line={"color": "#e74c3c", "width": 2.0},
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="中证1000 日线MA20多头向上 + 15分钟一买 联立策略回测 (2021-2026)",
        template="plotly_white",
        height=850,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    html_path = OUTPUT_DIR / "trend_first_buy_backtest_report.html"
    fig.write_html(str(html_path))
    print(f"交互式 HTML 图表已保存至: {html_path}")
    return html_path


def main():
    df_15m, raw_bars = fetch_and_prepare_data(SYMBOL)

    print("\n正在启动策略回测...")
    strat = StrongTrendFirstBuyStrategy(symbol=SYMBOL)
    res = strat.backtest(raw_bars, sdt=BACKTEST_SDT)
    print("策略回测完成！")

    pairs = res.pairs_df()
    holds = res.holds_df()

    # 保存 CSV 产物
    pairs_csv = OUTPUT_DIR / "strategy_trade_pairs.csv"
    holds_csv = OUTPUT_DIR / "strategy_holds.csv"
    pairs.to_csv(pairs_csv, index=False, encoding="utf-8-sig")
    holds.to_csv(holds_csv, index=False, encoding="utf-8-sig")
    print(f"交易对明细已保存至: {pairs_csv}")
    print(f"持仓时序已保存至: {holds_csv}")

    # 统计核心指标
    stats = analyze_pairs(pairs)
    print("\n================ 策略整体回测表现 ================")
    for k, v in stats.items():
        print(f"  {k:12s}: {v}")
    print("==================================================")

    if not pairs.empty:
        pairs["ret_pct"] = pairs["盈亏比例"] / 100.0
        pairs["year"] = pd.to_datetime(pairs["开仓时间"]).dt.year
        print("\n================ 分年度回测明细 ================")
        yearly_rows = []
        for y, g in pairs.groupby("year"):
            cnt = len(g)
            win = (g["ret_pct"] > 0).mean() * 100
            mean_ret = g["ret_pct"].mean()
            cum_ret = g["ret_pct"].sum()
            yearly_rows.append({
                "年份": y,
                "交易次数": cnt,
                "胜率": f"{win:.1f}%",
                "单笔均收益": f"{mean_ret:+.2f}%",
                "累计收益": f"{cum_ret:+.2f}%",
            })
        df_yearly = pd.DataFrame(yearly_rows)
        print(df_yearly.to_string(index=False))
        print("==================================================")

    # 生成可视化图表
    html_path = generate_interactive_chart(df_15m, pairs, holds)
    print(f"\n[🔗 点击在浏览器中查看交互式图表](file://{html_path.resolve()})\n")


if __name__ == "__main__":
    main()
