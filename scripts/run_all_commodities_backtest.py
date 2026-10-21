# ruff: noqa: E402
"""中国商品期货全市场 56 品种趋势策略回测与资产组合分析。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.ipc as ipc  # noqa: E402

from czsc import CzscStrategyBase, Event, Position  # noqa: E402

# 中文品种名称映射表
SYMBOL_NAMES = {
    # 黑色板块
    "SHFE.rb": ("螺纹钢", "黑色建材"),
    "SHFE.hc": ("热轧卷板", "黑色建材"),
    "DCE.i": ("铁矿石", "黑色建材"),
    "DCE.j": ("焦炭", "黑色建材"),
    "DCE.jm": ("焦煤", "黑色建材"),
    "CZCE.SF": ("硅铁", "黑色建材"),
    "CZCE.SM": ("锰硅", "黑色建材"),
    "CZCE.FG": ("玻璃", "黑色建材"),
    # 有色贵金属
    "SHFE.cu": ("沪铜", "有色金属"),
    "SHFE.al": ("沪铝", "有色金属"),
    "SHFE.zn": ("沪锌", "有色金属"),
    "SHFE.ni": ("沪镍", "有色金属"),
    "SHFE.sn": ("沪锡", "有色金属"),
    "SHFE.ss": ("不锈钢", "有色金属"),
    "SHFE.ao": ("氧化铝", "有色金属"),
    "SHFE.au": ("沪金", "贵金属"),
    "SHFE.ag": ("沪银", "贵金属"),
    # 能化板块
    "INE.sc": ("原油", "能源化工"),
    "SHFE.fu": ("燃油", "能源化工"),
    "SHFE.bu": ("沥青", "能源化工"),
    "SHFE.ru": ("天然橡胶", "能源化工"),
    "INE.nr": ("20号胶", "能源化工"),
    "SHFE.br": ("丁二烯橡胶", "能源化工"),
    "SHFE.sp": ("纸浆", "能源化工"),
    "CZCE.TA": ("PTA", "能源化工"),
    "CZCE.MA": ("甲醇", "能源化工"),
    "CZCE.SA": ("纯碱", "能源化工"),
    "CZCE.UR": ("尿素", "能源化工"),
    "CZCE.PF": ("短纤", "能源化工"),
    "CZCE.PX": ("对二甲苯", "能源化工"),
    "CZCE.SH": ("烧碱", "能源化工"),
    "CZCE.PR": ("瓶片", "能源化工"),
    "CZCE.PL": ("丙烯", "能源化工"),
    "DCE.l": ("塑料(L)", "能源化工"),
    "DCE.pp": ("聚丙烯(PP)", "能源化工"),
    "DCE.v": ("PVC", "能源化工"),
    "DCE.eg": ("乙二醇", "能源化工"),
    "DCE.eb": ("苯乙烯", "能源化工"),
    "DCE.pg": ("液化气(LPG)", "能源化工"),
    "DCE.bz": ("纯苯", "能源化工"),
    # 农产品
    "DCE.m": ("豆粕", "农产品"),
    "DCE.a": ("豆一", "农产品"),
    "DCE.b": ("豆二", "农产品"),
    "DCE.y": ("豆油", "农产品"),
    "DCE.p": ("棕榈油", "农产品"),
    "DCE.c": ("玉米", "农产品"),
    "DCE.jd": ("鸡蛋", "农产品"),
    "DCE.lh": ("生猪", "农产品"),
    "CZCE.RM": ("菜粕", "农产品"),
    "CZCE.OI": ("菜油", "农产品"),
    "CZCE.SR": ("白糖", "农产品"),
    "CZCE.CF": ("棉花", "农产品"),
    "CZCE.AP": ("苹果", "农产品"),
    # 新能源板块
    "GFEX.si": ("工业硅", "新能源"),
    "GFEX.lc": ("碳酸锂", "新能源"),
    "GFEX.ps": ("多晶硅", "新能源"),
}


from scripts.cta_execution import equal_weight_execution, futures_daily_bars, signed_holds_execution
from scripts.cta_paths import research_data_dir, research_output_path


def _arrow_to_df(arrow_bytes: bytes) -> pd.DataFrame:
    with ipc.open_file(arrow_bytes) as reader:
        return reader.read_all().to_pandas()


class CommoditySlopeTrendStrategy(CzscStrategyBase):
    def __init__(self, symbol: str, n: int = 20, **kwargs):
        super().__init__(symbol=symbol, n=n, **kwargs)
        self.n = n

    @property
    def positions(self) -> list[Position]:
        n = self.n
        pos_long = Position(
            name="日线多头",
            symbol=self.symbol,
            opens=[
                Event.load({"operate": "开多", "signals_all": [f"日线_D1N{n}_斜率突破V261003_看多_向上突破_任意_0"]})
            ],
            exits=[Event.load({"operate": "平多", "signals_all": [f"日线_D1N{n}_斜率突破V261003_看空_任意_任意_0"]})],
            interval=0,
            timeout=0,
            stop_loss=0,
        )
        pos_short = Position(
            name="日线空头",
            symbol=self.symbol,
            opens=[
                Event.load({"operate": "开空", "signals_all": [f"日线_D1N{n}_斜率突破V261003_看空_向下跌破_任意_0"]})
            ],
            exits=[Event.load({"operate": "平空", "signals_all": [f"日线_D1N{n}_斜率突破V261003_看多_任意_任意_0"]})],
            interval=0,
            timeout=0,
            stop_loss=0,
        )
        return [pos_long, pos_short]


def backtest_symbol(file_path: Path, n: int = 20):
    raw_key = file_path.stem.replace("_15m", "")
    info = SYMBOL_NAMES.get(raw_key, (raw_key, "其他"))
    name_cn, sector = info[0], info[1]

    df_15m = pd.read_parquet(file_path)
    df_15m["dt"] = pd.to_datetime(df_15m["dt"]) + pd.Timedelta(minutes=15)
    df_15m["symbol"] = name_cn
    df_15m["vol"] = df_15m["volume"]
    df_15m["amount"] = df_15m["vol"] * df_15m["close"] * 10
    bars_d = futures_daily_bars(df_15m)

    if len(bars_d) < 100:
        return None, None

    strat = CommoditySlopeTrendStrategy(symbol=name_cn, n=n)
    res = strat.backtest(bars_d, sdt="2024-03-01")
    pairs = _arrow_to_df(res.pairs_arrow)
    holds = _arrow_to_df(res.holds_arrow)

    if holds.empty or "pos_name" not in holds.columns:
        print(
            f"跳过 {raw_key}：holds 为空或缺少 pos_name (columns={holds.columns.tolist() if not holds.empty else []})"
        )
        return None, None

    p_long = holds[holds["pos_name"] == "日线多头"].set_index("dt")
    p_short = holds[holds["pos_name"] == "日线空头"].set_index("dt")
    if p_long.empty:
        return None, None
    df_nav = signed_holds_execution(p_long, p_short, bars_d)
    df_nav["nav"] = 1.0 + df_nav["strat_ret"].cumsum()
    df_nav["bench_nav"] = df_nav["price"] / df_nav["price"].iloc[0]
    df_nav["peak"] = df_nav["nav"].clip(lower=1.0).cummax()
    df_nav["dd"] = (df_nav["nav"] - df_nav["peak"]) / df_nav["peak"]

    # 2. 统计交易指标
    if pairs.empty:
        summary = {
            "代码": raw_key,
            "品种": name_cn,
            "板块": sector,
            "交易笔数": 0,
            "胜率": "0.0%",
            "盈亏比": 0.0,
            "单笔均益": "0.00%",
            "累计收益": "0.00%",
            "最大回撤": "0.00%",
            "夏普比率": 0.0,
            "原始收益值": 0.0,
        }
    else:
        ret_series = pairs["盈亏比例"] / 10000.0
        win_trades = ret_series[ret_series > 0]
        loss_trades = ret_series[ret_series < 0]
        win_rate = len(win_trades) / len(ret_series) if len(ret_series) > 0 else 0.0
        avg_win = win_trades.mean() if not win_trades.empty else 0.0
        avg_loss = abs(loss_trades.mean()) if not loss_trades.empty else 0.0
        pnl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        total_pnl = float(df_nav["nav"].iloc[-1] - 1.0)
        max_dd = float(df_nav["dd"].min())
        # 计算日度收益率年化夏普
        day_rets = df_nav["strat_ret"].values
        sharpe = float(np.mean(day_rets) / np.std(day_rets) * np.sqrt(252)) if np.std(day_rets) > 1e-8 else 0.0

        summary = {
            "代码": raw_key,
            "品种": name_cn,
            "板块": sector,
            "交易笔数": len(pairs),
            "胜率": f"{win_rate:.1%}",
            "盈亏比": round(pnl_ratio, 2),
            "单笔均益": f"{ret_series.mean():+.2%}",
            "累计收益": f"{total_pnl:+.2%}",
            "最大回撤": f"{max_dd:.2%}",
            "夏普比率": round(sharpe, 2),
            "原始收益值": total_pnl,
        }

    summary["原始收益值"] = float(df_nav["nav"].iloc[-1] - 1.0)
    summary["累计收益"] = f"{summary['原始收益值']:+.2%}"
    summary["最大回撤"] = f"{float(df_nav['dd'].min()):.2%}"
    returns = df_nav["strat_ret"].to_numpy()
    summary["夏普比率"] = (
        round(float(returns.mean() / returns.std() * np.sqrt(252)), 2) if returns.std() > 1e-8 else 0.0
    )
    return summary, df_nav


def main(*, output_path=None):
    data_dir = research_data_dir()
    files = sorted(data_dir.glob("*.parquet"))
    print(f"正在全市场扫描回测 {len(files)} 个商品期货品种（SLOPE LENGTH = 20）...")

    summaries = []
    nav_dict = {}
    execution_inputs = []

    for f in files:
        s, df_nav = backtest_symbol(f, n=20)
        if s and df_nav is not None:
            summaries.append(s)
            raw_key = s["代码"]
            name_cn = s["品种"]
            execution_inputs.append({"symbol": raw_key, "daily_inputs": df_nav.reset_index()})
            nav_dict[f"{name_cn} ({raw_key})"] = {
                "sector": s["板块"],
                "dates": [d.strftime("%Y-%m-%d") for d in df_nav.index],
                "nav": [round(float(x), 4) for x in df_nav["nav"]],
                "bench_nav": [round(float(x), 4) for x in df_nav["bench_nav"]],
                "dd": [round(float(x) * 100, 2) for x in df_nav["dd"]],
                "pos": [int(x) for x in df_nav["pos"]],
                "total_return": round(s["原始收益值"] * 100, 2),
                "bench_return": round((float(df_nav["bench_nav"].iloc[-1]) - 1.0) * 100, 2),
                "max_dd": round(float(df_nav["dd"].min()) * 100, 2),
                "sharpe": s["夏普比率"],
                "win_rate": s["胜率"] + "（信号收盘配对）",
                "accounting": "fixed_initial_capital_next_open",
                "pair_statistics_basis": "native_position_close_signal_pairs_not_executable_fills",
            }

    # 计算 56 品种全市场等权组合 (Universe Equal-Weight Portfolio)
    # 取全部日期的并集 / 交集
    all_dates = sorted(set.intersection(*[set(v["dates"]) for v in nav_dict.values() if len(v["dates"]) >= 150]))
    portfolio = equal_weight_execution(execution_inputs, all_dates, "daily")["daily"].reindex(all_dates, fill_value=0.0)
    port_rets = portfolio.net.to_numpy()
    port_nav = 1.0 + np.cumsum(port_rets)

    port_peak = np.maximum.accumulate(np.maximum(port_nav, 1.0))
    port_dd = [(port_nav[j] - port_peak[j]) / port_peak[j] * 100 for j in range(len(port_nav))]
    port_sharpe = float(np.mean(port_rets) / np.std(port_rets) * np.sqrt(252)) if np.std(port_rets) > 1e-8 else 0.0

    port_summary = {
        "sector": "综合资产池",
        "dates": all_dates,
        "nav": [round(float(x), 4) for x in port_nav],
        "bench_nav": [1.0] * len(all_dates),
        "dd": [round(float(x), 2) for x in port_dd],
        "pos": [1] * len(all_dates),
        "total_return": round((port_nav[-1] - 1.0) * 100, 2),
        "bench_return": 0.0,
        "max_dd": round(min(port_dd), 2),
        "sharpe": round(port_sharpe, 2),
        "win_rate": "组合分散",
        "accounting": "fixed_initial_capital_next_open",
        "daily_returns": port_rets.tolist(),
        "gross_return_pct": float(portfolio.gross.sum() * 100),
        "fee_return_pct": float(portfolio.fee.sum() * 100),
    }

    # 插入到最前面
    nav_dict = {"【全市场 56 品种等权组合】": port_summary, **nav_dict}

    # 打印全市场排名
    df_sum = pd.DataFrame(summaries)
    print("\n" + "=" * 100)
    print(f" 全市场 56 个商品期货品种回测统计（样本数: {len(df_sum)}）")
    print("=" * 100)
    print(
        f"全市场等权组合累计收益: {port_summary['total_return']:+.2f}% | 最大回撤: {port_summary['max_dd']:.2f}% | 年化夏普: {port_sharpe:.2f}"
    )
    print("-" * 100)

    # 板块统计
    sector_perf = []
    for sector, sub in df_sum.groupby("板块"):
        rets = sub["原始收益值"]
        sector_perf.append(
            {
                "板块": sector,
                "品种数量": len(sub),
                "正收益占比": f"{(rets > 0).mean():.1%}",
                "平均累计收益": f"{rets.mean():+.2%}",
                "最高收益品种": f"{sub.loc[rets.idxmax(), '品种']} ({rets.max():+.2%})",
                "最大亏损品种": f"{sub.loc[rets.idxmin(), '品种']} ({rets.min():+.2%})",
            }
        )
    print(pd.DataFrame(sector_perf).to_string(index=False))
    print("-" * 100)

    print("\n【全市场累计收益前 10 强品种】")
    top10 = df_sum.sort_values("原始收益值", ascending=False).head(10)
    print(
        top10[["代码", "品种", "板块", "交易笔数", "胜率", "盈亏比", "累计收益", "最大回撤", "夏普比率"]].to_string(
            index=False
        )
    )

    print("\n【全市场累计收益后 5 弱品种】")
    bottom5 = df_sum.sort_values("原始收益值", ascending=True).head(5)
    print(
        bottom5[["代码", "品种", "板块", "交易笔数", "胜率", "盈亏比", "累计收益", "最大回撤", "夏普比率"]].to_string(
            index=False
        )
    )
    print("=" * 100)

    # 保存数据
    out_json = research_output_path("all_commodities_nav_data.json", output_path)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(nav_dict, f, ensure_ascii=False)
    print(f"全市场净值时序数据已导出: {out_json}")
    return nav_dict


if __name__ == "__main__":
    main()
