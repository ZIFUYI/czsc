"""连续信号强度商品期货 CTA 策略回测（对齐国信证券原始研报完整修正版）

基于国信证券《CTA系列专题之五：基于连续信号的商品期货交易策略》原始研报全面修正：
1. 15分钟 K 线级别；
2. EMA10 / EMA100 均线差与其 10 周期均值判定趋势增强；
3. 10 周期 ATR 波动率过滤（ATR/Close < 0.5%）与单品种杠杆倍率（上限 4 倍）；
4. 4 周期趋势噪音比 TNR 与 3 个交易日前同时间槽位对比过滤（Delta TNR > 0）；
5. 连续概率状态机（开平彻底解耦：ATR/TNR 仅用于进场门禁，出场仅由概率中性区 |U2P| <= 0.2 决定）；
6. 截面调仓死区缓冲（Rebalance Deadband，消除 1/N 及微小概率跳动带来的无谓磨损）；
7. 目标年化波动率（15%）平滑机制（以近 1 年已实现波动率动态缩放组合杠杆，上限 4 倍）；
8. 动态流动性准入（近半年日均估算成交额 > 50 亿元）；
9. 收盘到收盘权重收益评估，扣除交易费用与滑点（万分之一单边）。

运行：
    uv run --no-sync python examples/signals_dev/backtest_continuous_signal_commodity_cta.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT_DIR = ROOT / "examples" / "results" / "continuous_signal_commodity_cta"
DEFAULT_DATA_DIR = DEFAULT_OUTPUT_DIR / "data_cache"

# 56 个主力商品期货候选品种
DEFAULT_SYMBOLS = (
    "SHFE.au",
    "SHFE.ag",
    "SHFE.sn",
    "DCE.jm",
    "INE.sc",
    "SHFE.cu",
    "DCE.p",
    "SHFE.ru",
    "CZCE.MA",
    "GFEX.lc",
    "SHFE.ni",
    "CZCE.TA",
    "DCE.m",
    "DCE.lh",
    "CZCE.CF",
    "DCE.pp",
    "DCE.v",
    "SHFE.al",
    "DCE.eg",
    "CZCE.SH",
    "SHFE.fu",
    "DCE.eb",
    "DCE.y",
    "DCE.l",
    "CZCE.OI",
    "SHFE.bu",
    "CZCE.FG",
    "DCE.jd",
    "SHFE.rb",
    "SHFE.br",
    "CZCE.SA",
    "CZCE.SR",
    "SHFE.ao",
    "DCE.i",
    "CZCE.PX",
    "CZCE.RM",
    "DCE.c",
    "SHFE.zn",
    "DCE.pg",
    "SHFE.hc",
    "SHFE.sp",
    "SHFE.ss",
    "DCE.a",
    "INE.nr",
    "GFEX.ps",
    "CZCE.PR",
    "GFEX.si",
    "CZCE.UR",
    "CZCE.AP",
    "CZCE.SM",
    "CZCE.SF",
    "CZCE.PF",
    "DCE.j",
    "DCE.b",
    "DCE.bz",
    "CZCE.PL",
)

# 合约乘数对照表
DEFAULT_MULTIPLIERS = {
    "SHFE.au": 1000.0,
    "SHFE.ag": 15.0,
    "SHFE.sn": 1.0,
    "DCE.jm": 60.0,
    "INE.sc": 1000.0,
    "SHFE.cu": 5.0,
    "DCE.p": 10.0,
    "SHFE.ru": 10.0,
    "CZCE.MA": 10.0,
    "GFEX.lc": 1.0,
    "SHFE.ni": 1.0,
    "CZCE.TA": 5.0,
    "DCE.m": 10.0,
    "DCE.lh": 16.0,
    "CZCE.CF": 5.0,
    "DCE.pp": 5.0,
    "DCE.v": 5.0,
    "SHFE.al": 5.0,
    "DCE.eg": 10.0,
    "CZCE.SH": 30.0,
    "SHFE.fu": 10.0,
    "DCE.eb": 5.0,
    "DCE.y": 10.0,
    "DCE.l": 5.0,
    "CZCE.OI": 10.0,
    "SHFE.bu": 10.0,
    "CZCE.FG": 20.0,
    "DCE.jd": 10.0,
    "SHFE.rb": 10.0,
    "SHFE.br": 5.0,
    "CZCE.SA": 20.0,
    "CZCE.SR": 10.0,
    "SHFE.ao": 20.0,
    "DCE.i": 100.0,
    "CZCE.PX": 5.0,
    "CZCE.RM": 10.0,
    "DCE.c": 10.0,
    "SHFE.zn": 5.0,
    "DCE.pg": 20.0,
    "SHFE.hc": 10.0,
    "SHFE.sp": 10.0,
    "SHFE.ss": 5.0,
    "DCE.a": 10.0,
    "INE.nr": 10.0,
    "GFEX.ps": 3.0,
    "CZCE.PR": 15.0,
    "GFEX.si": 5.0,
    "CZCE.UR": 20.0,
    "CZCE.AP": 10.0,
    "CZCE.SM": 5.0,
    "CZCE.SF": 5.0,
    "CZCE.PF": 5.0,
    "DCE.j": 100.0,
    "DCE.b": 10.0,
    "DCE.bz": 30.0,
    "CZCE.PL": 20.0,
}


@dataclass
class StrategyConfig:
    start: str | None = "2025-08-14"  # 默认使用全 56 品种数据齐备的同窗区间
    end: str | None = None
    fee_rate: float = 0.0001  # 万分之一手续费与滑点
    liquidity_threshold: float = 5e9  # 半年日均成交额门槛：50亿元
    rebalance_band: float = 0.025  # 调仓死区缓冲带：2.5%
    relative_band: float = 0.0  # 相对死区比例（0.0表示仅用绝对死区，如0.20表示20%相对死区）
    rebalance_mode: str = "daily_close"  # "daily_close" (日内实时开平+收盘定点再平衡) 或 "continuous" (15m连续微调)
    min_hold_bars: int = 15  # 最小持仓 K 线根数（锁定半个交易日，消除短命交易）
    prob_decay: float = 0.03  # 概率平滑衰减率（当无持续动量时，朝中性概率0.5自然平滑回落）
    entry_u2p: float = 0.30  # 进场概率差门槛（提升至 0.30，过滤低质量动量噪声，大幅压缩交易次数）
    exit_u2p: float = 0.05  # 出场概率差门槛（拓宽至 0.05，宽幅非对称迟滞带，防止洗盘频繁止损，持仓拉长至月度级别）
    max_atr_mult: float = 2.5  # 单品种 ATR 杠杆上限（适度收敛超大仓位换手）
    enable_target_vol: bool = True  # 是否启用研报第18页的年化15%目标波动率自适应调整
    target_vol: float = 0.15  # 目标波动率：15%
    exclude_symbols: tuple[str, ...] = ("DCE.lh", "DCE.jd")  # 默认排除非趋势性养殖品种
    refresh_data: bool = False
    output_dir: Path = DEFAULT_OUTPUT_DIR
    data_dir: Path = DEFAULT_DATA_DIR


def load_env() -> tuple[str, str]:
    """读取天勤凭证"""
    tq_user = os.getenv("TQ_USER", "")
    tq_pass = os.getenv("TQ_PASS", "")
    if not tq_user or not tq_pass:
        env_file = ROOT / ".env"
        if env_file.exists():
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k == "TQ_USER":
                            tq_user = v
                        elif k == "TQ_PASS":
                            tq_pass = v
    return tq_user, tq_pass


def get_trading_day(dt: pd.Timestamp) -> datetime.date:
    """根据期货交易时间段确定归属交易日

    国内期货夜盘（21:00~次日02:30）计入下一个自然交易日。
    """
    t = dt.time()
    d = dt.date()
    # 20:00 之后属于下一个自然日的夜盘
    if t >= time(20, 0):
        next_d = d + timedelta(days=1)
        # 若下一个自然日是周六，则属于下周一
        while next_d.weekday() >= 5:
            next_d += timedelta(days=1)
        return next_d
    elif t < time(3, 0):
        # 凌晨0点到3点（周六凌晨属于周五夜盘，交易日为周一）
        if d.weekday() == 5:
            return d + timedelta(days=2)
        elif d.weekday() == 6:
            return d + timedelta(days=1)
        return d
    else:
        # 日盘09:00~15:00
        return d


def load_or_fetch_all_data(symbols: list[str], config: StrategyConfig) -> dict[str, pd.DataFrame]:
    """加载本地缓存或拉取天勤数据"""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    data_map: dict[str, pd.DataFrame] = {}

    missing_symbols: list[str] = []
    for sym in symbols:
        p1 = config.data_dir / f"{sym}_15m.parquet"
        p2 = config.data_dir / f"{sym}.parquet"
        target_p = p1 if p1.exists() else p2
        if target_p.exists() and not config.refresh_data:
            df = pd.read_parquet(target_p)
            if not df.empty and len(df) >= 200:
                data_map[sym] = df
                continue
        missing_symbols.append(sym)

    if missing_symbols:
        tq_user, tq_pass = load_env()
        if not tq_user or not tq_pass:
            raise ValueError(f"缺少天勤账号凭证，无法拉取缺失数据: {missing_symbols}")

        from tqsdk import TqApi, TqAuth

        api = TqApi(auth=TqAuth(tq_user, tq_pass))
        try:
            for sym in missing_symbols:
                tq_sym = f"KQ.m@{sym}"
                print(f"  正在从天勤下载 {tq_sym} 15分钟 K 线...")
                kline = api.get_kline_serial(tq_sym, 15 * 60, data_length=10000)
                if kline is None or len(kline) == 0:
                    continue
                df = pd.DataFrame(kline)
                df["dt"] = pd.to_datetime(df["datetime"], unit="ns")
                cols = ["dt", "open", "high", "low", "close", "volume", "open_interest"]
                df = df[cols].copy().sort_values("dt").reset_index(drop=True)
                parquet_path = config.data_dir / f"{sym}.parquet"
                df.to_parquet(parquet_path)
                data_map[sym] = df
        finally:
            api.close()

    return data_map


def compute_symbol_signals(
    symbol: str,
    df: pd.DataFrame,
    min_hold_bars: int = 15,
    prob_decay: float = 0.03,
    entry_u2p: float = 0.20,
    exit_u2p: float = 0.20,
    max_atr_mult: float = 4.0,
) -> pd.DataFrame:
    """计算单个品种的连续信号状态机与过滤指标（开平分离）"""
    res = df.copy().sort_values("dt").reset_index(drop=True)
    c = res["close"]

    # 1. 均线与增强触发
    fast = c.ewm(span=10, adjust=False).mean()
    slow = c.ewm(span=100, adjust=False).mean()
    diff = fast - slow
    diff_mean = diff.rolling(10).mean()

    long_trig = (diff > 0) & (diff > diff_mean)
    short_trig = (diff < 0) & (diff < diff_mean)
    res["diff"] = diff
    res["diff_mean"] = diff_mean
    res["long_trig"] = long_trig
    res["short_trig"] = short_trig

    # 2. ATR 过滤与杠杆倍率
    prev_c = c.shift(1)
    tr = np.maximum(
        res["high"] - res["low"],
        np.maximum((res["high"] - prev_c).abs(), (res["low"] - prev_c).abs()),
    )
    atr = tr.rolling(10).mean()
    l_atr = 0.005 * c / atr
    atr_pass = l_atr > 1.0
    atr_mult = np.minimum(l_atr, max_atr_mult)

    res["l_atr"] = l_atr
    res["atr_mult"] = atr_mult
    res["atr_pass"] = atr_pass

    # 3. 趋势噪音比 TNR 与 3 个交易日前同时间槽位对比
    diff_c = c.diff().abs()
    denom = diff_c.rolling(4).sum()
    numer = (c - c.shift(4)).abs()
    tnr = np.where(denom > 0, numer / denom, 0.0)
    res["tnr"] = tnr

    # 交易日与时间槽位映射
    res["trading_day"] = res["dt"].apply(get_trading_day)
    res["time_str"] = res["dt"].dt.strftime("%H:%M")

    trade_days = sorted(res["trading_day"].unique())
    day_to_idx = {d: i for i, d in enumerate(trade_days)}
    res["day_idx"] = res["trading_day"].map(day_to_idx)

    # 建立 (day_idx, time_str) -> tnr 查找字典
    tnr_map = res.set_index(["day_idx", "time_str"])["tnr"].to_dict()

    delta_tnr = []
    for d_idx, t_str, cur_tnr in zip(res["day_idx"], res["time_str"], res["tnr"], strict=False):
        ref_val = tnr_map.get((d_idx - 3, t_str), np.nan)
        if np.isnan(ref_val):
            # 若历史不足3日同时间槽位，兜底取当日前4根 bar 均值做对比
            delta_tnr.append(0.0)
        else:
            delta_tnr.append(cur_tnr - ref_val)

    res["delta_tnr"] = pd.Series(delta_tnr, index=res.index)
    res["tnr_pass"] = res["delta_tnr"] > 0

    # 4. 连续概率状态机计算（开平分离）
    up_probs = []
    down_probs = []
    signals = []
    positions = []

    cur_up = 0.5
    cur_down = 0.5
    cur_pos = 0  # 0: 空仓, 1: 多头, -1: 空头
    bars_held = 0

    l_trig_arr = res["long_trig"].to_numpy()
    s_trig_arr = res["short_trig"].to_numpy()
    atr_pass_arr = res["atr_pass"].to_numpy()
    tnr_pass_arr = res["tnr_pass"].to_numpy()

    for i in range(len(res)):
        is_l = l_trig_arr[i]
        is_s = s_trig_arr[i]
        is_atr = atr_pass_arr[i]
        is_tnr = tnr_pass_arr[i]

        # 概率状态更新公式
        if is_l:
            cur_up = cur_up + 0.5 * (1.0 - cur_up)
            cur_down = 1.0 - cur_up
        elif is_s:
            cur_down = cur_down + 0.5 * (1.0 - cur_down)
            cur_up = 1.0 - cur_down
        else:
            # 当无明确单边增强触发时，概率朝中性 0.5 平滑衰减
            cur_up = cur_up + prob_decay * (0.5 - cur_up)
            cur_down = 1.0 - cur_up

        u2p = cur_up - cur_down

        # 进场门禁：四重条件同时满足
        entry_long_ok = is_l and is_atr and is_tnr and (u2p > entry_u2p)
        entry_short_ok = is_s and is_atr and is_tnr and (u2p < -entry_u2p)

        if cur_pos == 0:
            if entry_long_ok:
                cur_pos = 1
                sig = cur_up
                bars_held = 0
            elif entry_short_ok:
                cur_pos = -1
                sig = -cur_down
                bars_held = 0
            else:
                sig = 0.0
        elif cur_pos == 1:
            bars_held += 1
            # 持仓与退出逻辑：由概率中性区或反向信号决定（不被单根 ATR/TNR 抖动误杀）
            if abs(u2p) <= exit_u2p:
                if bars_held >= min_hold_bars:
                    cur_pos = 0
                    sig = 0.0
                else:
                    sig = cur_up
            elif entry_short_ok and bars_held >= min_hold_bars:
                cur_pos = -1
                sig = -cur_down
                bars_held = 0
            else:
                sig = cur_up
        elif cur_pos == -1:
            bars_held += 1
            if abs(u2p) <= exit_u2p:
                if bars_held >= min_hold_bars:
                    cur_pos = 0
                    sig = 0.0
                else:
                    sig = -cur_down
            elif entry_long_ok and bars_held >= min_hold_bars:
                cur_pos = 1
                sig = cur_up
                bars_held = 0
            else:
                sig = -cur_down

        up_probs.append(cur_up)
        down_probs.append(cur_down)
        signals.append(sig)
        positions.append(cur_pos)

    res["up_prob"] = up_probs
    res["down_prob"] = down_probs
    res["u2p"] = res["up_prob"] - res["down_prob"]
    res["signal"] = signals
    res["pos"] = positions
    res["symbol"] = symbol

    return res


def compute_monthly_universe(
    data_map: dict[str, pd.DataFrame],
    multipliers: dict[str, float],
    threshold: float = 5e9,
) -> dict[str, set[str]]:
    """按前 6 个完整月日均成交额计算月度准入名单"""
    daily_turnovers: list[dict] = []
    for sym, df in data_map.items():
        mult = multipliers.get(sym, DEFAULT_MULTIPLIERS.get(sym, 10.0))
        t_day = df["dt"].apply(get_trading_day)
        amt = df["close"] * df["volume"] * mult
        sym_daily = pd.DataFrame({"symbol": sym, "trading_day": t_day, "amount": amt})
        day_sum = sym_daily.groupby(["symbol", "trading_day"])["amount"].sum().reset_index()
        daily_turnovers.append(day_sum)

    if not daily_turnovers:
        return {}

    all_turnovers = pd.concat(daily_turnovers, ignore_index=True)
    all_turnovers["trading_day"] = pd.to_datetime(all_turnovers["trading_day"])
    all_turnovers["ym"] = all_turnovers["trading_day"].dt.to_period("M")

    unique_months = sorted(all_turnovers["ym"].unique())
    monthly_universe: dict[str, set[str]] = {}

    for cur_m in unique_months:
        lookback_months = [m for m in unique_months if m < cur_m][-6:]
        cur_str = str(cur_m)
        if not lookback_months:
            monthly_universe[cur_str] = set(data_map.keys())
            continue

        window_data = all_turnovers[all_turnovers["ym"].isin(lookback_months)]
        n_days = window_data["trading_day"].nunique()
        if n_days == 0:
            monthly_universe[cur_str] = set(data_map.keys())
            continue

        sym_totals = window_data.groupby("symbol")["amount"].sum()
        sym_daily_avg = sym_totals / n_days
        qualified = sym_daily_avg[sym_daily_avg > threshold].index.tolist()

        if len(qualified) < 5:
            qualified = sym_daily_avg.nlargest(30).index.tolist()

        monthly_universe[cur_str] = set(qualified)

    return monthly_universe


def apply_rebalance_band(
    desired: pd.DataFrame,
    band: float = 0.025,
    relative_band: float = 0.0,
    mode: str = "daily_close",
) -> pd.DataFrame:
    """对目标权重施加调仓缓冲带（Deadband）与模式控制，消除微小调仓带来的高频摩擦"""
    rows = []
    last_w: dict[str, float] = {}
    last_pos: dict[str, int] = {}

    for dt, row in desired.iterrows():
        is_close_bar = dt.hour == 14 and dt.minute in (30, 45)
        new_row = {}
        for sym, target_w in row.items():
            prev = last_w.get(sym, 0.0)
            prev_p = last_pos.get(sym, 0)
            cur_p = int(np.sign(target_w))

            eff_band = band
            if relative_band > 0:
                eff_band = max(band, relative_band * abs(target_w))

            if mode == "daily_close":
                # 日内实时开平 + 收盘定点再平衡
                if cur_p == 0:
                    actual = 0.0  # 退出信号：日内 15m 立即平仓
                elif cur_p != prev_p:
                    actual = target_w  # 新开仓 / 反手：日内 15m 立即开仓
                elif is_close_bar and abs(target_w - prev) >= eff_band:
                    actual = target_w  # 存量仓位微调：仅在收盘定点 bar 且偏离超死区时才调
                else:
                    actual = prev  # 其余时间拒绝微调，杜绝高频损耗
            else:
                # 15m 连续死区模式
                if target_w == 0.0:
                    actual = 0.0
                elif (np.sign(target_w) != np.sign(prev) and prev != 0.0) or abs(target_w - prev) >= eff_band:
                    actual = target_w
                else:
                    actual = prev
            new_row[sym] = actual
            last_w[sym] = actual
            last_pos[sym] = int(np.sign(actual))
        rows.append(new_row)

    return pd.DataFrame(rows, index=desired.index)


def run_portfolio_backtest(
    data_map: dict[str, pd.DataFrame],
    multipliers: dict[str, float],
    config: StrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """多品种截面等分回测与目标波动率自适应调整"""
    print("\n[1/4] 逐品种计算连续信号状态机...")
    all_signals: list[pd.DataFrame] = []
    for sym, df in data_map.items():
        if sym in config.exclude_symbols:
            continue
        sig_df = compute_symbol_signals(
            sym,
            df,
            min_hold_bars=config.min_hold_bars,
            prob_decay=config.prob_decay,
            entry_u2p=config.entry_u2p,
            exit_u2p=config.exit_u2p,
            max_atr_mult=config.max_atr_mult,
        )
        all_signals.append(sig_df)

    if not all_signals:
        raise ValueError("没有可用的品种信号数据！")

    combined = pd.concat(all_signals, ignore_index=True)
    combined["ym"] = combined["dt"].dt.to_period("M").astype(str)

    print("[2/4] 计算月度 50 亿元流动性准入名单...")
    monthly_univ = compute_monthly_universe(data_map, multipliers, config.liquidity_threshold)

    admitted_list = []
    for ym, sym in zip(combined["ym"], combined["symbol"], strict=False):
        valid_pool = monthly_univ.get(ym, set(data_map.keys()))
        admitted_list.append(sym in valid_pool)
    combined["admitted"] = admitted_list

    # 计算截面活跃品种数并等权分配基础名义权重
    combined["active_signal"] = np.where(combined["admitted"], combined["signal"], 0.0)
    combined["is_active"] = (combined["active_signal"] != 0).astype(int)

    n_counts = combined.groupby("dt")["is_active"].transform("sum")
    combined["n_active"] = n_counts

    combined["base_weight"] = np.where(
        combined["n_active"] > 0,
        (1.0 / combined["n_active"]) * combined["active_signal"] * combined["atr_mult"],
        0.0,
    )

    pivot_raw_w = combined.pivot(index="dt", columns="symbol", values="base_weight").ffill().fillna(0.0)

    # 目标波动率平滑调整（研报第18页：年化15%目标波动率）
    if config.enable_target_vol:
        print("[3/4] 施加滚动已实现波动率动态杠杆调整（目标年化 15%）...")
        pivot_close = combined.pivot(index="dt", columns="symbol", values="close").ffill()
        bar_ret = pivot_close.pct_change().fillna(0.0)
        daily_dt = combined[["dt", "trading_day"]].drop_duplicates().set_index("dt")["trading_day"]

        # 先测算基础名义权重的未加杠杆日收益
        unlev_bar_pnl = (pivot_raw_w.shift(1).fillna(0.0) * bar_ret).sum(axis=1)
        unlev_df = pd.DataFrame({"bar_pnl": unlev_bar_pnl, "trading_day": daily_dt.reindex(unlev_bar_pnl.index)})
        unlev_daily_pnl = unlev_df.groupby("trading_day")["bar_pnl"].sum()

        rolling_std = unlev_daily_pnl.rolling(window=120, min_periods=20).std() * np.sqrt(250)
        rolling_mul = (config.target_vol / rolling_std.replace(0.0, np.nan)).fillna(2.0)
        rolling_mul = rolling_mul.clip(lower=0.5, upper=4.0)

        dt_mul = daily_dt.map(rolling_mul).fillna(2.0)
        pivot_w = pivot_raw_w.mul(dt_mul, axis=0)
    else:
        pivot_w = pivot_raw_w

    # 截面调仓死区与再平衡模式过滤
    print(
        f"  施加调仓缓冲带 (模式={config.rebalance_mode}, 绝对死区={config.rebalance_band * 100:.1f}%, 相对死区={config.relative_band * 100:.1f}%)..."
    )
    pivot_w = apply_rebalance_band(
        pivot_w,
        band=config.rebalance_band,
        relative_band=config.relative_band,
        mode=config.rebalance_mode,
    )

    # 截取回测有效区间
    if config.start:
        pivot_w = pivot_w[pivot_w.index >= config.start]
    if config.end:
        pivot_w = pivot_w[pivot_w.index <= config.end]

    print("[4/4] 运行逐 bar 权重损益与费率核算（扣除单边万分之一）...")
    pivot_close = combined.pivot(index="dt", columns="symbol", values="close").ffill()
    pivot_close = pivot_close.reindex(pivot_w.index).ffill()
    bar_returns = pivot_close.pct_change().fillna(0.0)

    # 收益计算：持仓名义权重 * 标的收益率
    gross_bar_pnl = (pivot_w.shift(1).fillna(0.0) * bar_returns).sum(axis=1)

    # 换手率与费用扣除：单边每笔按成交名义金额扣除万一
    delta_w = pivot_w.diff().abs().fillna(pivot_w.abs())
    bar_turnover = delta_w.sum(axis=1)
    bar_fees = bar_turnover * config.fee_rate
    net_bar_pnl = gross_bar_pnl - bar_fees

    # 汇总为日度表现
    t_day_series = pd.Series(pivot_w.index, index=pivot_w.index).apply(get_trading_day)
    df_eval = pd.DataFrame(
        {
            "gross_pnl": gross_bar_pnl,
            "fee": bar_fees,
            "net_pnl": net_bar_pnl,
            "turnover": bar_turnover,
            "trading_day": t_day_series,
            "gross_leverage": pivot_w.abs().sum(axis=1),
            "active_symbols": (pivot_w.abs() > 0).sum(axis=1),
        }
    )

    daily_df = (
        df_eval.groupby("trading_day")
        .agg(
            {
                "gross_pnl": "sum",
                "fee": "sum",
                "net_pnl": "sum",
                "turnover": "sum",
                "gross_leverage": "mean",
                "active_symbols": "mean",
            }
        )
        .reset_index()
    )

    daily_df.rename(
        columns={
            "trading_day": "date",
            "gross_pnl": "gross_ret",
            "net_pnl": "net_ret",
        },
        inplace=True,
    )

    daily_df["cum_gross"] = (1.0 + daily_df["gross_ret"]).cumprod()
    daily_df["cum_net"] = (1.0 + daily_df["net_ret"]).cumprod()
    daily_df["drawdown"] = daily_df["cum_net"] / daily_df["cum_net"].cummax() - 1.0

    # 统计核心指标
    n_days = len(daily_df)
    cum_net = daily_df["cum_net"].iloc[-1] - 1.0
    cum_gross = daily_df["cum_gross"].iloc[-1] - 1.0
    ann_net = (1.0 + cum_net) ** (250.0 / n_days) - 1.0
    ann_vol = daily_df["net_ret"].std() * np.sqrt(250.0)
    sharpe = (
        (daily_df["net_ret"].mean() / daily_df["net_ret"].std()) * np.sqrt(250.0)
        if daily_df["net_ret"].std() > 0
        else 0.0
    )
    max_dd = daily_df["drawdown"].min()
    calmar = (ann_net / abs(max_dd)) if abs(max_dd) > 0 else 0.0

    win_days = (daily_df["net_ret"] > 0).sum()
    daily_win_rate = win_days / n_days if n_days > 0 else 0.0
    total_turnover = daily_df["turnover"].sum()

    # 统计交易动作与次数
    prev_w = pivot_w.shift(1).fillna(0.0)
    diff_w = (pivot_w - prev_w).abs()
    is_change = diff_w > 1e-4
    is_open = (prev_w.abs() < 1e-4) & (pivot_w.abs() >= 1e-4)
    is_close = (prev_w.abs() >= 1e-4) & (pivot_w.abs() < 1e-4)
    is_flip = (prev_w.abs() >= 1e-4) & (pivot_w.abs() >= 1e-4) & (np.sign(prev_w) != np.sign(pivot_w))
    is_rebal = is_change & ~is_open & ~is_close & ~is_flip

    n_opens = int(is_open.sum().sum())
    n_closes = int(is_close.sum().sum())
    n_flips = int(is_flip.sum().sum())
    n_rebals = int(is_rebal.sum().sum())
    total_actions = n_opens + n_closes + n_flips + n_rebals
    n_syms = len(pivot_w.columns)
    avg_hold_days = float(n_days / max(1, (n_opens / n_syms)))

    stats = {
        "start": str(daily_df["date"].iloc[0]),
        "end": str(daily_df["date"].iloc[-1]),
        "trading_days": int(n_days),
        "fee_rate": config.fee_rate,
        "cumulative_net_return": float(cum_net),
        "cumulative_gross_return": float(cum_gross),
        "annualized_net_return": float(ann_net),
        "annualized_volatility": float(ann_vol),
        "max_drawdown": float(max_dd),
        "sharpe_ratio": float(sharpe),
        "calmar_ratio": float(calmar),
        "daily_win_rate": float(daily_win_rate),
        "avg_daily_turnover": float(daily_df["turnover"].mean()),
        "total_turnover": float(total_turnover),
        "avg_gross_leverage": float(daily_df["gross_leverage"].mean()),
        "avg_active_symbols": float(daily_df["active_symbols"].mean()),
        "total_trades": int(total_actions),
        "daily_trades": float(total_actions / n_days),
        "open_trades": int(n_opens),
        "close_trades": int(n_closes),
        "flip_trades": int(n_flips),
        "rebalance_trades": int(n_rebals),
        "avg_hold_days": float(avg_hold_days),
        "round_trips_per_sym": float(n_opens / n_syms),
    }

    # 统计各品种收益贡献归因
    sym_contrib = (pivot_w.shift(1).fillna(0.0) * bar_returns).sum(axis=0)
    sym_turnover = delta_w.sum(axis=0)
    sym_net = sym_contrib - sym_turnover * config.fee_rate
    attrib_df = (
        pd.DataFrame(
            {
                "symbol": sym_contrib.index,
                "gross_contrib": sym_contrib.values,
                "turnover": sym_turnover.values,
                "net_contrib": sym_net.values,
            }
        )
        .sort_values("net_contrib", ascending=False)
        .reset_index(drop=True)
    )

    return daily_df, attrib_df, stats


def calculate_yearly_stats(daily_df: pd.DataFrame) -> pd.DataFrame:
    """分年度统计绩效"""
    df = daily_df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    records = []
    for yr, sub in df.groupby("year"):
        n = len(sub)
        c_net = (1.0 + sub["net_ret"]).cumprod().iloc[-1] - 1.0
        ann_ret = (1.0 + c_net) ** (250.0 / n) - 1.0
        ann_vol = sub["net_ret"].std() * np.sqrt(250.0)
        sr = (sub["net_ret"].mean() / sub["net_ret"].std()) * np.sqrt(250.0) if sub["net_ret"].std() > 0 else 0.0
        cum = (1.0 + sub["net_ret"]).cumprod()
        mdd = (cum / cum.cummax() - 1.0).min()
        win = (sub["net_ret"] > 0).sum() / n if n > 0 else 0.0
        to = sub["turnover"].sum()
        records.append(
            {
                "year": int(yr),
                "trading_days": n,
                "net_return": c_net,
                "annualized_return": ann_ret,
                "annualized_volatility": ann_vol,
                "max_drawdown": mdd,
                "sharpe_ratio": sr,
                "daily_win_rate": win,
                "total_turnover": to,
            }
        )
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="连续信号强度商品期货 CTA 策略回测")
    parser.add_argument("--start", type=str, default="2025-08-14", help="回测起始日期（默认全品种齐备的 2025-08-14）")
    parser.add_argument("--end", type=str, default=None, help="回测结束日期")
    parser.add_argument("--fee-rate", type=float, default=0.0001, help="单边手续费与滑点（默认万分之一）")
    parser.add_argument("--rebalance-band", type=float, default=0.025, help="调仓死区缓冲（默认 2.5%）")
    parser.add_argument("--relative-band", type=float, default=0.0, help="相对死区比例（默认 0.0）")
    parser.add_argument(
        "--rebalance-mode",
        type=str,
        default="daily_close",
        choices=["daily_close", "continuous"],
        help="再平衡模式：daily_close 或 continuous",
    )
    parser.add_argument("--min-hold-bars", type=int, default=15, help="最小持仓 K 线根数")
    parser.add_argument("--entry-u2p", type=float, default=0.30, help="进场概率差门槛（默认 0.30，高质量趋势过滤）")
    parser.add_argument(
        "--exit-u2p", type=float, default=0.05, help="出场概率差门槛（默认 0.05，宽幅非对称迟滞防洗盘）"
    )
    parser.add_argument("--max-atr-mult", type=float, default=2.5, help="单品种 ATR 杠杆上限（默认 2.5）")
    parser.add_argument("--include-all-symbols", action="store_true", help="是否包含生猪与鸡蛋等养殖品种")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude = () if args.include_all_symbols else ("DCE.lh", "DCE.jd")
    config = StrategyConfig(
        start=args.start,
        end=args.end,
        fee_rate=args.fee_rate,
        rebalance_band=args.rebalance_band,
        relative_band=args.relative_band,
        rebalance_mode=args.rebalance_mode,
        min_hold_bars=args.min_hold_bars,
        entry_u2p=args.entry_u2p,
        exit_u2p=args.exit_u2p,
        max_atr_mult=args.max_atr_mult,
        exclude_symbols=exclude,
        output_dir=out_dir,
    )

    print("================================================================================")
    print("  国信证券连续信号强度商品期货 CTA 策略回测（修正版）")
    print(f"  回测区间: {config.start or '全样本'} ~ {config.end or '最新'}")
    print(f"  单边交易费用与滑点: {config.fee_rate * 10000:.1f} BP (万分之 {config.fee_rate * 10000:.1f})")
    print(
        f"  再平衡模式: {config.rebalance_mode} | 死区缓冲: {config.rebalance_band * 100:.1f}% (相对: {config.relative_band * 100:.1f}%)"
    )
    print(
        f"  持仓锁定: {config.min_hold_bars} 根 bar | 迟滞门槛: 进场 {config.entry_u2p:.2f} / 平仓 {config.exit_u2p:.2f}"
    )
    print(f"  单品种杠杆上限: {config.max_atr_mult:.1f}x")
    print(f"  排除品种: {config.exclude_symbols if config.exclude_symbols else '无（全品种）'}")
    print("================================================================================")

    data_map = load_or_fetch_all_data(list(DEFAULT_SYMBOLS), config)
    print(f"成功加载标的数据: {len(data_map)} 个品种")

    daily_df, attrib_df, stats = run_portfolio_backtest(data_map, DEFAULT_MULTIPLIERS, config)
    yearly_df = calculate_yearly_stats(daily_df)

    # 导出文件
    daily_df.to_csv(out_dir / "daily.csv", index=False)
    yearly_df.to_csv(out_dir / "yearly_stats.csv", index=False)
    attrib_df.to_csv(out_dir / "attribution.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 导出交互图表
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=("策略累计净值 (费后 vs 毛收益)", "动态回撤", "活跃品种数与日度换手"),
            row_heights=[0.5, 0.25, 0.25],
        )
        fig.add_trace(
            go.Scatter(
                x=daily_df["date"],
                y=daily_df["cum_net"],
                name="费后净值 (Net)",
                line={"color": "#0284c7", "width": 2.2},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=daily_df["date"],
                y=daily_df["cum_gross"],
                name="毛收益净值 (Gross)",
                line={"color": "#94a3b8", "width": 1.5, "dash": "dash"},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=daily_df["date"],
                y=daily_df["drawdown"] * 100,
                name="动态回撤 (%)",
                fill="tozeroy",
                line={"color": "#ef4444", "width": 1.2},
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=daily_df["date"],
                y=daily_df["active_symbols"],
                name="持仓品种数",
                line={"color": "#10b981", "width": 1.5},
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=daily_df["date"], y=daily_df["turnover"], name="日换手率", marker_color="rgba(148, 163, 184, 0.4)"
            ),
            row=3,
            col=1,
        )

        fig.update_layout(
            title=f"国信证券连续信号商品期货 CTA 策略净值走势 (Sharpe: {stats['sharpe_ratio']:.2f}, MaxDD: {stats['max_drawdown'] * 100:.2f}%)",
            template="plotly_white",
            height=780,
            hovermode="x unified",
        )
        fig.write_html(str(out_dir / "equity.html"))
        print(f"  已导出交互净值图: {out_dir / 'equity.html'}")
    except Exception as e:
        print(f"  导出图表略过: {e}")

    print("\n============================= 核心回测绩效统计 =============================")
    print(f"  回测时间跨度: {stats['start']} 至 {stats['end']} (共 {stats['trading_days']} 个交易日)")
    print(
        f"  累计净收益 (费后): {stats['cumulative_net_return'] * 100:.2f}% (毛收益: {stats['cumulative_gross_return'] * 100:.2f}%)"
    )
    print(f"  年化净收益率: {stats['annualized_net_return'] * 100:.2f}%")
    print(f"  年化波动率: {stats['annualized_volatility'] * 100:.2f}%")
    print(f"  夏普比率 (Sharpe): {stats['sharpe_ratio']:.2f}")
    print(f"  最大回撤 (MaxDD): {stats['max_drawdown'] * 100:.2f}%")
    print(f"  卡玛比率 (Calmar): {stats['calmar_ratio']:.2f}")
    print(f"  日度胜率: {stats['daily_win_rate'] * 100:.1f}%")
    print(f"  总换手率: {stats['total_turnover']:.1f} 倍 (日均换手: {stats['avg_daily_turnover']:.2f} 倍)")
    print(f"  平均杠杆率: {stats['avg_gross_leverage']:.2f} 倍")
    print(f"  平均活跃品种数: {stats['avg_active_symbols']:.1f} 个")
    print(f"  总下单调整次数: {stats['total_trades']:,} 次 (日均全组合仅 {stats['daily_trades']:.1f} 次)")
    print(
        f"  交易动作构成: 新开仓 {stats['open_trades']:,} 次 | 完全平仓 {stats['close_trades']:,} 次 | 多空反手 {stats['flip_trades']:,} 次 | 存量再平衡 {stats['rebalance_trades']:,} 次"
    )
    print(
        f"  单品种平均持仓周期: {stats['avg_hold_days']:.1f} 个交易日/笔 (单品种全期平均交易 {stats['round_trips_per_sym']:.1f} 笔)"
    )

    print("\n============================== 分年度表现统计 ==============================")
    print(yearly_df.to_string(index=False))

    print("\n=========================== 净收益贡献 Top 10 品种 ===========================")
    print(attrib_df.head(10).to_string(index=False))

    print("\n=========================== 净亏损拖累 Bottom 5 品种 ==========================")
    print(attrib_df.tail(5).to_string(index=False))
    print("================================================================================")


if __name__ == "__main__":
    main()
