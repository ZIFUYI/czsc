# ruff: noqa: E402
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.cta_execution import assign_trading_date, lagged_daily_volatility_multiplier, next_open_panel_ledger
from scripts.cta_execution import get_trading_day as get_trading_day
from scripts.cta_paths import research_data_dir, research_output_path

# 56 个主力商品期货合约乘数对照表
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
    "DCE.bz": 5.0,
    "CZCE.PL": 5.0,
}


@dataclass
class SlopeContinuousConfig:
    start: str = "2024-01-02"
    end: str = "2026-09-30"
    fee_rate: float = 0.0001  # 单边万分之一
    rebalance_band: float = 0.025  # 2.5% 调仓死区
    min_hold_bars: int = 15  # 最小持仓 K 线根数
    entry_u2p: float = 0.25  # 进场概率差门槛
    exit_u2p: float = 0.05  # 出场概率中性区门槛 (宽幅迟滞)
    prob_decay: float = 0.05  # 概率中性衰减率
    slope_window: int = 40  # 线性回归斜率窗口
    slope_ma_window: int = 10  # 斜率均线窗口
    max_atr_mult: float = 2.5  # 单品种 ATR 杠杆上限
    enable_target_vol: bool = True  # 启用目标年化波动率平滑
    target_vol: float = 0.15  # 目标年化波动率 15%


def compute_symbol_slope_signals(symbol: str, df_15m: pd.DataFrame, config: SlopeContinuousConfig) -> pd.DataFrame:
    """在 15m 高频数据上计算线性回归斜率与连续概率状态机"""
    res = df_15m.copy()
    c = res["close"]
    h = res["high"]
    low = res["low"]

    # 1. 计算 10 周期 ATR 与 单品种 ATR 杠杆倍率
    prev_c = c.shift(1).fillna(c.iloc[0])
    tr = np.maximum(h - low, np.maximum((h - prev_c).abs(), (low - prev_c).abs()))
    atr = tr.rolling(10, min_periods=3).mean()
    l_atr = 0.005 * c / np.maximum(atr, 0.0001)
    atr_mult = np.clip(l_atr, 0.5, config.max_atr_mult)
    res["atr"] = atr
    res["atr_mult"] = atr_mult
    res["atr_pass"] = l_atr > 0.8  # 波动率适度准入门禁

    # 2. 计算归一化线性回归斜率 (NormSlope = Slope / Close * 1000)
    w = config.slope_window
    closes = c.values
    slopes = np.zeros(len(closes))
    x_arr = np.arange(w)
    for i in range(w - 1, len(closes)):
        y_slice = closes[i - w + 1 : i + 1]
        slopes[i] = (np.polyfit(x_arr, y_slice, 1)[0] / closes[i]) * 1000.0
    res["slope"] = slopes
    res["slope_ma"] = pd.Series(slopes, index=res.index).rolling(config.slope_ma_window, min_periods=3).mean()

    # 趋势增强判定：斜率突破其均线且同向
    res["long_trig"] = (res["slope"] > res["slope_ma"]) & (res["slope"] > 0)
    res["short_trig"] = (res["slope"] < res["slope_ma"]) & (res["slope"] < 0)

    # 3. 连续概率状态机计算（开平分离与非对称迟滞）
    up_probs = []
    down_probs = []
    signals = []
    positions = []

    cur_up = 0.5
    cur_down = 0.5
    cur_pos = 0
    bars_held = 0

    l_trig_arr = res["long_trig"].to_numpy()
    s_trig_arr = res["short_trig"].to_numpy()
    atr_pass_arr = res["atr_pass"].to_numpy()

    for i in range(len(res)):
        is_l = l_trig_arr[i]
        is_s = s_trig_arr[i]
        is_atr = atr_pass_arr[i]

        # 概率状态更新公式
        if is_l:
            cur_up = cur_up + 0.5 * (1.0 - cur_up)
            cur_down = 1.0 - cur_up
        elif is_s:
            cur_down = cur_down + 0.5 * (1.0 - cur_down)
            cur_up = 1.0 - cur_down
        else:
            cur_up = cur_up + config.prob_decay * (0.5 - cur_up)
            cur_down = 1.0 - cur_up

        u2p = cur_up - cur_down

        # 进场门槛: 强趋势概率差 + 波动率过滤
        entry_long_ok = is_l and is_atr and (u2p > config.entry_u2p)
        entry_short_ok = is_s and is_atr and (u2p < -config.entry_u2p)

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
            # 宽幅出场门槛: 只有进入中性区 |u2p| <= 0.05 且持仓超过最小周期才出场
            if abs(u2p) <= config.exit_u2p:
                if bars_held >= config.min_hold_bars:
                    cur_pos = 0
                    sig = 0.0
                else:
                    sig = cur_up
            elif entry_short_ok and bars_held >= config.min_hold_bars:
                cur_pos = -1
                sig = -cur_down
                bars_held = 0
            else:
                sig = cur_up
        elif cur_pos == -1:
            bars_held += 1
            if abs(u2p) <= config.exit_u2p:
                if bars_held >= config.min_hold_bars:
                    cur_pos = 0
                    sig = 0.0
                else:
                    sig = -cur_down
            elif entry_long_ok and bars_held >= config.min_hold_bars:
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
    res["trading_day"] = assign_trading_date(res["dt"], True).astype(str)

    return res


def apply_rebalance_deadband(desired: pd.DataFrame, band: float = 0.025) -> pd.DataFrame:
    """日内 15m 实时开平 + 收盘定点微调 (死区 2.5%)"""
    rows = []
    last_w: dict[str, float] = {}
    last_pos: dict[str, int] = {}

    for dt, row in desired.iterrows():
        is_close_bar = (dt.hour == 14 and dt.minute in (30, 45)) or (dt.hour == 15 and dt.minute == 0)
        new_row = {}
        for sym, target_w in row.items():
            prev = last_w.get(sym, 0.0)
            prev_p = last_pos.get(sym, 0)
            cur_p = int(np.sign(target_w))

            if cur_p == 0:
                actual = 0.0  # 平仓立即执行
            elif cur_p != prev_p:
                actual = target_w  # 新开仓/反手立即执行
            elif is_close_bar and abs(target_w - prev) >= band:
                actual = target_w  # 存量仓位仅在收盘偏离超 2.5% 才调
            else:
                actual = prev  # 其余时间坚决不调仓，杜绝高频损耗

            new_row[sym] = actual
            last_w[sym] = actual
            last_pos[sym] = cur_p
        rows.append(new_row)

    return pd.DataFrame(rows, index=desired.index)


def run_slope_continuous_system(config: SlopeContinuousConfig, *, output_path: Path | None = None):
    data_dir = research_data_dir()
    files = sorted(data_dir.glob("*.parquet"))
    print(f"[1/4] 加载全市场 56 个品种并计算斜率连续信号，共 {len(files)} 个文件...")

    all_signals = []
    data_map = {}
    for f in files:
        sym = f.stem.replace("_15m", "")
        # 排除流动性差且历史过短的个别品种
        if sym in ("DCE.lh", "DCE.jd"):
            continue
        try:
            df = pd.read_parquet(f)
            df["dt"] = pd.to_datetime(df["dt"]) + pd.Timedelta(minutes=15)
            if len(df) < 500:
                continue
            data_map[sym] = df
            sig_df = compute_symbol_slope_signals(sym, df, config)
            all_signals.append(sig_df)
        except Exception:
            continue

    print(f"成功处理有效品种数: {len(all_signals)}")
    combined = pd.concat(all_signals, ignore_index=True)

    # 截面活跃品种等分基础权重 (1 / N_active)
    print("[2/4] 计算截面活跃品种动态等分权重 (1 / N_active * Signal * ATR_mult)...")
    combined["is_active"] = (combined["signal"] != 0.0).astype(int)
    n_counts = combined.groupby("dt")["is_active"].transform("sum")
    combined["n_active"] = n_counts

    combined["base_weight"] = np.where(
        combined["n_active"] > 0,
        (1.0 / combined["n_active"]) * combined["signal"] * combined["atr_mult"],
        0.0,
    )

    pivot_raw_w = combined.pivot(index="dt", columns="symbol", values="base_weight").ffill().fillna(0.0)

    # 目标年化波动率平滑调整 (15%)
    if config.enable_target_vol:
        print("[3/4] 施加宏观目标年化 15% 波动率动态杠杆调整...")
        unlevered = next_open_panel_ledger(combined, pivot_raw_w, 0.0, liquidate=False)
        dt_mul = lagged_daily_volatility_multiplier(unlevered["gross"].sum(axis=1), target_vol=config.target_vol)
        pivot_w = pivot_raw_w.mul(dt_mul, axis=0)
    else:
        pivot_w = pivot_raw_w

    # 施加调仓死区缓冲
    print(f"  施加调仓死区缓冲带 (Deadband = {config.rebalance_band * 100:.1f}%)...")
    pivot_w = apply_rebalance_deadband(pivot_w, band=config.rebalance_band)

    # 截取有效回测区间
    if config.start:
        pivot_w = pivot_w[pivot_w.index >= config.start]
    if config.end:
        pivot_w = pivot_w[pivot_w.index < pd.Timestamp(config.end).normalize() + pd.Timedelta(days=1)]

    print("[4/4] 运行逐 15m bar 损益与费率核算（扣除单边万分之一）...")
    ledger = next_open_panel_ledger(combined, pivot_w, config.fee_rate)
    gross_bar_pnl = ledger["gross"].sum(axis=1)
    bar_turnover = ledger["turnover"].sum(axis=1)
    bar_fees = ledger["fee"].sum(axis=1)
    net_bar_pnl = ledger["net"].sum(axis=1)

    t_day_series = assign_trading_date(pd.Series(pivot_w.index, index=pivot_w.index), True).astype(str)
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

    daily_df = df_eval.groupby("trading_day").agg(
        {
            "gross_pnl": "sum",
            "fee": "sum",
            "net_pnl": "sum",
            "turnover": "sum",
            "gross_leverage": "mean",
            "active_symbols": "mean",
        }
    )

    # 计算整体统计与分年度表现
    daily_rets = daily_df["net_pnl"].values
    nav = 1.0 + np.cumsum(daily_rets)
    peak = np.maximum.accumulate(np.maximum(nav, 1.0))
    dd = (nav - peak) / peak * 100.0

    tot_ret = (nav[-1] - 1.0) * 100.0
    ann_ret = tot_ret * (252.0 / len(daily_rets))
    max_dd = float(np.min(dd))
    sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)) if np.std(daily_rets) > 0 else 0
    calmar = abs(ann_ret / max_dd) if max_dd != 0 else 0
    ann_vol = float(np.std(daily_rets) * np.sqrt(252) * 100)
    win_rate = float(np.mean(daily_rets > 0) * 100)

    # 分年度统计
    daily_df["year"] = pd.to_datetime(daily_df.index).year
    yearly_stats = []
    for y, g in daily_df.groupby("year"):
        g_rets = g["net_pnl"].values
        g_nav = 1.0 + np.cumsum(g_rets)
        g_peak = np.maximum.accumulate(np.maximum(g_nav, 1.0))
        g_dd = (g_nav - g_peak) / g_peak * 100.0
        g_tot = (g_nav[-1] - 1.0) * 100.0
        g_ann = g_tot * (252.0 / len(g_rets))
        g_std = np.std(g_rets)
        g_sharpe = float(np.mean(g_rets) / g_std * np.sqrt(252)) if g_std > 0 else 0
        yearly_stats.append(
            {
                "year": int(y),
                "days": len(g),
                "net_return": round(float(g_tot), 2),
                "ann_return": round(float(g_ann), 2),
                "ann_vol": round(float(g_std * np.sqrt(252) * 100), 2),
                "max_dd": round(float(np.min(g_dd)), 2),
                "sharpe": round(float(g_sharpe), 2),
                "turnover": round(float(g["turnover"].sum()), 1),
                "win_rate": round(float(np.mean(g_rets > 0) * 100), 1),
            }
        )

    # 单品种收益归因
    sym_returns = ledger["net"].sum(axis=0)
    attribution = []
    for sym, r_val in sym_returns.items():
        attribution.append(
            {
                "symbol": sym,
                "contribution_pct": round(float(r_val * 100), 2),
            }
        )
    attribution.sort(key=lambda x: x["contribution_pct"], reverse=True)

    print("\n================== 连续斜率 CTA 策略全期实证结果 (费后) ==================")
    print(f"全期交易日数: {len(daily_df)} 天 ({daily_df.index[0]} 至 {daily_df.index[-1]})")
    print(f"累计净收益率: {tot_ret:+.2f}% | 年化净收益率: {ann_ret:+.2f}%")
    print(f"年化波动率: {ann_vol:.2f}% | 年化夏普比率: {sharpe:.2f}")
    print(f"最大动态回撤: {max_dd:.2f}% | 卡玛比率: {calmar:.2f}")
    print(f"全期总换手率: {daily_df['turnover'].sum():.1f} 倍 | 日均换手率: {daily_df['turnover'].mean():.2f} 倍")
    print(
        f"日均活跃品种数: {daily_df['active_symbols'].mean():.1f} 个 | 平均名义杠杆: {daily_df['gross_leverage'].mean():.2f} 倍"
    )

    print("\n================== 分年度表现剖析 ==================")
    for ys in yearly_stats:
        print(
            f"【{ys['year']}年】净收益: {ys['net_return']:+.2f}% (年化 {ys['ann_return']:+.2f}%) | 最大回撤: {ys['max_dd']:.2f}% | 夏普: {ys['sharpe']:.2f} | 换手: {ys['turnover']}倍"
        )

    out_data = {
        "dates": daily_df.index.tolist(),
        "accounting": "fixed_initial_capital_next_open",
        "daily_gross": daily_df["gross_pnl"].tolist(),
        "daily_fees": daily_df["fee"].tolist(),
        "daily_net": daily_df["net_pnl"].tolist(),
        "nav": [round(float(x), 4) for x in nav],
        "dd": [round(float(x), 2) for x in dd],
        "summary": {
            "tot_ret": round(float(tot_ret), 2),
            "gross_return_pct": float(daily_df["gross_pnl"].sum() * 100),
            "fee_return_pct": float(daily_df["fee"].sum() * 100),
            "ann_ret": round(float(ann_ret), 2),
            "ann_vol": round(float(ann_vol), 2),
            "max_dd": round(float(max_dd), 2),
            "sharpe": round(float(sharpe), 2),
            "calmar": round(float(calmar), 2),
            "win_rate": round(float(win_rate), 1),
            "total_turnover": round(float(daily_df["turnover"].sum()), 1),
            "avg_leverage": round(float(daily_df["gross_leverage"].mean()), 2),
            "avg_active_symbols": round(float(daily_df["active_symbols"].mean()), 1),
        },
        "yearly_stats": yearly_stats,
        "top_winners": attribution[:10],
        "top_losers": attribution[-5:],
    }

    out_path = research_output_path("slope_continuous_signal_cta_results.json", output_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)
    print(f"\n完整策略结果已保存至: {out_path}")

    return out_data


if __name__ == "__main__":
    cfg = SlopeContinuousConfig()
    run_slope_continuous_system(cfg)
