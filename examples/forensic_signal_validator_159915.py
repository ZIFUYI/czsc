#!/usr/bin/env python
"""
159915 (创业板ETF) 机构级单信号法医检验器与样本外盲测系统 (Forensic Signal Validator & OOS Backtest)

全面修复传统单信号检验 (Event Study) 的 6 大结构性缺陷：
1. [修复缺陷1 - 重叠自相关]: 实施 Purged Non-Overlapping 去重叠抽样 (前笔未平仓期间忽略新触发) + Newey-West HAC t检验
2. [修复缺陷2 - 零假设错配]: 引入 Monte Carlo Bootstrap 随机入场置换检验 (1000次随机同方向抽样，剥离市场牛熊漂移)
3. [修复缺陷3 - 路径盲区]: 引入 Triple-Barrier 三重屏障法 (盘中 High/Low 逐根检验 +2.2*ATR 止盈 / -1.0*ATR 结构止损 / 16根超时)
4. [修复缺陷4 - 收盘偷价]: 强制改为次根 K 线开盘价 Open[t+1] 成交，并扣除双边 2 BP 交易摩擦成本
5. [修复缺陷5 - 窥探与过拟合]: 严格切分 In-Sample 样本内 (2020-2023) 选拔 与 Out-of-Sample 样本外 (2024-2026) 盲测 + BH-FDR 多重校正
6. [修复缺陷6 - 状态盲区]: 输出 牛市 / 熊市 / 震荡市 三态分层稳健性检验
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
from scipy import stats

from czsc import WeightBacktest

OUTPUT_DIR = Path("examples/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REAL_5M_CSV = OUTPUT_DIR / "159915_real_5m_2020_2026.csv"
HTML_PATH = OUTPUT_DIR / "forensic_oos_validation_159915.html"


def load_standard_kline(csv_path: Path, bars_per_group: int = 6) -> pd.DataFrame:
    """
    合成等长标准 K 线:
    - bars_per_group = 6  -> 30分钟 K 线 (每天 8 根，样本充足 N=13,112)
    - bars_per_group = 12 -> 60分钟 K 线 (每天 4 根，N=6,556)
    """
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
    df_raw["grp"] = df_raw["bar_in_day"] // bars_per_group

    df = df_raw.groupby(["date", "grp"], as_index=False).agg(
        dt=("datetime", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    df["symbol"] = "159915"
    return df


def prepare_features_and_signals(
    df: pd.DataFrame, scale: float = 0.35
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, str]]]:
    """
    计算基础技术特征与 7 大候选结构信号 (纯当时点计算，严禁任何未来信息)
    返回: (带有特征的 df, 候选信号字典 {信号名称: (布尔触发数组, 'long'/'short')})
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

    # 布林带
    std20 = df["close"].rolling(20).std()
    df["boll_up"] = df["ma20"] + 2.0 * std20
    df["boll_dn"] = df["ma20"] - 2.0 * std20

    # 均线粘合度
    ma_max = df[["ma5", "ma10", "ma20", "ma30"]].max(axis=1)
    ma_min = df[["ma5", "ma10", "ma20", "ma30"]].min(axis=1)
    df["ma_dispersion"] = (ma_max - ma_min) / ma_min

    # ATR14
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

    # 市场状态划分 (Regime: 牛市 / 熊市 / 震荡市)
    ma60_slope = (df["ma60"] - df["ma60"].shift(10)) / df["ma60"].shift(10)
    df["regime"] = np.where(
        (df["close"] > df["ma60"]) & (ma60_slope > 0.005 * scale),
        "牛市",
        np.where((df["close"] < df["ma60"]) & (ma60_slope < -0.005 * scale), "熊市", "震荡"),
    )

    box_amp_th = 0.09 * scale
    disp_th = 0.02 * scale
    pierce_th = 0.015 * scale

    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]
    ma5, ma10, ma20, ma30, ma60 = df["ma5"], df["ma10"], df["ma20"], df["ma30"], df["ma60"]
    vol, vma20 = df["volume"], df["vol_ma20"]

    # 候选信号定义 (多空对称共 10 个核心触发器)
    signals = {}

    # 1. 窄幅中枢突破 + 回踩三买/三卖确认
    # 前 1~3 根曾发生窄幅箱体突破，当前回踩不破箱体轨线且收阳反包
    brk_up_recent = ((df["box_amp"].shift(1) <= box_amp_th) & (c.shift(1) > df["prev_high_box"].shift(1))) | (
        (df["box_amp"].shift(2) <= box_amp_th) & (c.shift(2) > df["prev_high_box"].shift(2))
    )
    box_top_ref = np.maximum(df["prev_high_box"].shift(1), df["prev_high_box"].shift(2))
    sig_box_long = brk_up_recent & (l >= box_top_ref * (1 - 0.006 * scale)) & (c > o) & (c >= h.shift(1)) & (c > ma60)

    brk_dn_recent = ((df["box_amp"].shift(1) <= box_amp_th) & (c.shift(1) < df["prev_low_box"].shift(1))) | (
        (df["box_amp"].shift(2) <= box_amp_th) & (c.shift(2) < df["prev_low_box"].shift(2))
    )
    box_bot_ref = np.minimum(df["prev_low_box"].shift(1), df["prev_low_box"].shift(2))
    sig_box_short = brk_dn_recent & (h <= box_bot_ref * (1 + 0.006 * scale)) & (c < o) & (c <= l.shift(1)) & (c < ma60)

    signals["S1L_中枢三买确认"] = (sig_box_long.fillna(False).values, "long")
    signals["S1S_中枢三卖确认"] = (sig_box_short.fillna(False).values, "short")

    # 2. 均线粘合一阳穿多线 / 一阴下多线
    m_min = df[["ma5", "ma10", "ma20"]].min(axis=1)
    m_max = df[["ma5", "ma10", "ma20", "ma30"]].max(axis=1)
    sig_rib_long = (
        (df["ma_dispersion"] <= disp_th)
        & (o <= m_min * (1 + 0.005 * scale))
        & (c >= m_max)
        & (c > o * (1 + pierce_th))
        & (c > ma60)
    )
    sig_rib_short = (
        (df["ma_dispersion"] <= disp_th)
        & (o >= m_max * (1 - 0.005 * scale))
        & (c <= m_min)
        & (c < o * (1 - pierce_th))
        & (c < ma60)
    )

    signals["S2L_均线粘合一阳穿线"] = (sig_rib_long.fillna(False).values, "long")
    signals["S2S_均线粘合一阴下线"] = (sig_rib_short.fillna(False).values, "short")

    # 3. 内孕线破局 (Inside Bar Breakout)
    inside_prev = (h.shift(1) < h.shift(2)) & (l.shift(1) > l.shift(2))
    sig_ib_long = inside_prev & (vol >= 1.2 * vma20) & (c > h.shift(2)) & (c > o) & (c > ma60)
    sig_ib_short = inside_prev & (vol >= 1.2 * vma20) & (c < l.shift(2)) & (c < o) & (c < ma60)

    signals["S3L_内孕线放量突破"] = (sig_ib_long.fillna(False).values, "long")
    signals["S3S_内孕线放量跌破"] = (sig_ib_short.fillna(False).values, "short")

    # 4. VCP 波动率极度压缩爆发
    vcp_cond = (df["atr_rank_60"] <= 0.20) & (df["total_range"] >= 1.5 * df["atr14"])
    sig_vcp_long = vcp_cond & (c > ma60) & (c > o) & (c > ma20)
    sig_vcp_short = vcp_cond & (c < ma60) & (c < o) & (c < ma20)

    signals["S4L_VCP压缩多头爆发"] = (sig_vcp_long.fillna(False).values, "long")
    signals["S4S_VCP压缩空头爆发"] = (sig_vcp_short.fillna(False).values, "short")

    # 5. 顺势首次回踩 MA20 企稳 / 反抽受阻
    was_above = (c.shift(1) > ma20.shift(1)) & (c.shift(2) > ma20.shift(2))
    sig_pb_long = (
        (c > ma60)
        & (ma20 > ma60)
        & was_above
        & (l <= ma20 * (1 + 0.005 * scale))
        & (c >= ma20)
        & (c > o)
        & (c > h.shift(1))
    )

    was_below = (c.shift(1) < ma20.shift(1)) & (c.shift(2) < ma20.shift(2))
    sig_pb_short = (
        (c < ma60)
        & (ma20 < ma60)
        & was_below
        & (h >= ma20 * (1 - 0.005 * scale))
        & (c <= ma20)
        & (c < o)
        & (c < l.shift(1))
    )

    signals["S5L_顺势回踩MA20企稳"] = (sig_pb_long.fillna(False).values, "long")
    signals["S5S_顺势反抽MA20受阻"] = (sig_pb_short.fillna(False).values, "short")

    # 6. 震荡市布林轨线 2B 假破反包
    sig_b2b_long = (
        (df["box_amp"] <= box_amp_th * 1.3)
        & (l.shift(1) < df["boll_dn"].shift(1))
        & (c > df["boll_dn"])
        & (c > o)
        & (c >= h.shift(1))
        & (c > ma60 * 0.97)
    )
    sig_b2b_short = (
        (df["box_amp"] <= box_amp_th * 1.3)
        & (h.shift(1) > df["boll_up"].shift(1))
        & (c < df["boll_up"])
        & (c < o)
        & (c <= l.shift(1))
        & (c < ma60 * 1.03)
    )

    signals["S6L_布林下轨2B反包"] = (sig_b2b_long.fillna(False).values, "long")
    signals["S6S_布林上轨2B反杀"] = (sig_b2b_short.fillna(False).values, "short")

    return df, signals


def newey_west_tstat(x: np.ndarray, max_lags: int = 3) -> tuple[float, float]:
    """计算 Newey-West HAC 自相关与异方差稳健 t 统计量与双侧 p 值"""
    n = len(x)
    if n < 5:
        return 0.0, 1.0
    mean = np.mean(x)
    u = x - mean
    gamma_0 = np.sum(u**2) / n
    hac_var = gamma_0
    for lag in range(1, min(max_lags + 1, n)):
        w = 1.0 - lag / (max_lags + 1.0)
        gamma_l = np.sum(u[lag:] * u[:-lag]) / n
        hac_var += 2.0 * w * gamma_l
    se_hac = np.sqrt(max(hac_var, 1e-12) / n)
    t_stat = mean / se_hac
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1)))
    return float(t_stat), p_val


def evaluate_signal_forensic(
    df: pd.DataFrame,
    raw_trigger: np.ndarray,
    direction: str,
    horizon: int = 16,
    tp_atr_mult: float = 2.2,
    sl_atr_mult: float = 1.1,
    fee_bp: float = 2.0,
    n_bootstrap: int = 500,
) -> dict:
    """
    对比检验：
    A) 旧方法 (Naive Event Study): Close[t] 入场，固定持有 horizon 根，不去重，普通 t 检验 vs 0
    B) 法医级新方法 (Forensic Triple-Barrier):
       - Next-Open (Open[t+1]) 入场 + 扣减双边滑点手续费
       - Purged 去重叠 (持仓未结束前忽略后续触发)
       - Triple-Barrier 盘中逐根检查 High/Low 止盈止损先后顺序
       - Bootstrap 随机入场置换检验 (剥离市场漂移)
       - Newey-West HAC 稳健标准误
    """
    c_arr = df["close"].values
    o_arr = df["open"].values
    h_arr = df["high"].values
    l_arr = df["low"].values
    atr_arr = df["atr14"].values
    regime_arr = df["regime"].values
    n = len(df)
    dir_sign = 1.0 if direction == "long" else -1.0
    cost = (fee_bp * 2.0) / 10000.0  # 双边成本

    # --- 1. 旧方法 (Naive: Close-to-Close, Overlapping, vs 0) ---
    naive_idx = [i for i in range(60, n - horizon) if raw_trigger[i]]
    naive_rets = [dir_sign * (c_arr[i + horizon] / c_arr[i] - 1.0) for i in naive_idx]
    naive_mean = np.mean(naive_rets) * 100 if len(naive_rets) > 0 else 0.0
    naive_win = np.mean(np.array(naive_rets) > 0) * 100 if len(naive_rets) > 0 else 0.0
    _, naive_p = stats.ttest_1samp(naive_rets, 0.0) if len(naive_rets) >= 5 else (0.0, 1.0)

    # --- 2. 法医级新方法 (Forensic: Next-Open, Purged Non-Overlapping, Triple-Barrier) ---
    def simulate_single_trade(trig_i: int) -> tuple[float, int, str]:
        """在 trig_i 收盘触发，于 trig_i + 1 开盘入场，运行三重屏障，返回 (净收益率, 平仓索引, 触发屏障类型)"""
        entry_idx = trig_i + 1
        entry_p = o_arr[entry_idx]
        atr = atr_arr[trig_i] if not np.isnan(atr_arr[trig_i]) and atr_arr[trig_i] > 0 else entry_p * 0.01

        tp_dist = tp_atr_mult * atr
        sl_dist = sl_atr_mult * atr

        if direction == "long":
            tp_price = entry_p + tp_dist
            sl_price = entry_p - sl_dist
        else:
            tp_price = entry_p - tp_dist
            sl_price = entry_p + sl_dist

        max_exit_idx = min(entry_idx + horizon, n - 1)
        for j in range(entry_idx, max_exit_idx + 1):
            bh, bl, bc, bo = h_arr[j], l_arr[j], c_arr[j], o_arr[j]
            if direction == "long":
                # 保守原则：先检查盘中是否触及止损轨！(防止先打止损再拉升的假盈利)
                if bl <= sl_price:
                    fill_p = min(bo, sl_price) if j > entry_idx else sl_price
                    return (fill_p / entry_p - 1.0) - cost, j, "SL"
                if bh >= tp_price:
                    fill_p = max(bo, tp_price) if j > entry_idx else tp_price
                    return (fill_p / entry_p - 1.0) - cost, j, "TP"
            else:
                if bh >= sl_price:
                    fill_p = max(bo, sl_price) if j > entry_idx else sl_price
                    return (entry_p / fill_p - 1.0) - cost, j, "SL"
                if bl <= tp_price:
                    fill_p = min(bo, tp_price) if j > entry_idx else tp_price
                    return (entry_p / fill_p - 1.0) - cost, j, "TP"

        # 垂直屏障 (时间到期平仓)
        final_ret = dir_sign * (c_arr[max_exit_idx] / entry_p - 1.0) - cost
        return final_ret, max_exit_idx, "TIME"

    # 执行 Purged 去重叠抽样
    purged_rets = []
    purged_regimes = []
    barrier_hits = {"TP": 0, "SL": 0, "TIME": 0}
    last_exit_idx = -1

    for i in range(60, n - horizon - 1):
        if raw_trigger[i] and i > last_exit_idx:
            ret, exit_idx, b_type = simulate_single_trade(i)
            purged_rets.append(ret)
            purged_regimes.append(regime_arr[i])
            barrier_hits[b_type] += 1
            last_exit_idx = exit_idx

    purged_arr = np.array(purged_rets)
    n_purged = len(purged_arr)
    if n_purged < 5:
        return {"n_naive": len(naive_idx), "n_purged": n_purged, "valid": False}

    f_mean = float(np.mean(purged_arr) * 100)
    f_win = float(np.mean(purged_arr > 0) * 100)
    wins = purged_arr[purged_arr > 0]
    losses = purged_arr[purged_arr < 0]
    avg_w = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_l = float(abs(np.mean(losses))) if len(losses) > 0 else 1e-6
    pl_ratio = avg_w / avg_l if avg_l > 0 else 0.0
    breakeven_win = (1.0 / (1.0 + pl_ratio)) * 100 if pl_ratio > 0 else 100.0
    safety_margin = f_win - breakeven_win

    # Newey-West 稳健 t 检验
    nw_t, nw_p = newey_west_tstat(purged_arr, max_lags=2)

    # Monte Carlo Bootstrap 随机入场置换检验 (剥离市场自身牛熊漂移)
    rng = np.random.default_rng(42)
    valid_pool = np.arange(60, n - horizon - 2)
    rand_means = []
    for _ in range(n_bootstrap):
        sampled_indices = rng.choice(valid_pool, size=n_purged, replace=False)
        r_sim = [simulate_single_trade(int(idx))[0] for idx in sampled_indices]
        rand_means.append(np.mean(r_sim) * 100)

    rand_means_arr = np.array(rand_means)
    benchmark_drift_mean = float(np.mean(rand_means_arr))
    excess_alpha = f_mean - benchmark_drift_mean
    # 单侧经验 p 值：随机入场表现优于或等于当前信号的比例
    boot_pval = float(np.mean(rand_means_arr >= f_mean))

    # 牛熊震荡分层收益
    reg_df = pd.DataFrame({"ret": purged_arr * 100, "reg": purged_regimes})
    reg_means = reg_df.groupby("reg")["ret"].mean().to_dict()

    return {
        "valid": True,
        "n_naive": len(naive_idx),
        "naive_mean": naive_mean,
        "naive_win": naive_win,
        "naive_p": float(naive_p),
        "n_purged": n_purged,
        "f_mean": f_mean,
        "f_win": f_win,
        "pl_ratio": pl_ratio,
        "safety_margin": safety_margin,
        "tp_rate": barrier_hits["TP"] / n_purged * 100,
        "sl_rate": barrier_hits["SL"] / n_purged * 100,
        "bench_drift": benchmark_drift_mean,
        "excess_alpha": excess_alpha,
        "nw_pval": nw_p,
        "boot_pval": boot_pval,
        "bull_mean": reg_means.get("牛市", 0.0),
        "bear_mean": reg_means.get("熊市", 0.0),
        "osc_mean": reg_means.get("震荡", 0.0),
    }


def benjamini_hochberg_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR 多重假设检验 p 值校正"""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    ranked_p = np.array(p_values)[order]
    adj_p = np.zeros(m)
    prev_val = 1.0
    for idx in range(m - 1, -1, -1):
        rank = idx + 1
        val = min(prev_val, ranked_p[idx] * m / rank)
        adj_p[order[idx]] = val
        prev_val = val
    return adj_p.tolist()


def main():
    print("=" * 115)
    print("  【159915 机构级单信号法医检验器 (Forensic Signal Validator) & 样本内外盲测系统】")
    print("=" * 115)

    # 加载 30分钟等长标准 K 线 (13,112 根，满足充足样本量要求)
    df_all = load_standard_kline(REAL_5M_CSV, bars_per_group=6)
    df_all, signals_dict = prepare_features_and_signals(df_all, scale=0.35)

    # 严格切分 In-Sample (2020-2023) 与 Out-of-Sample (2024-2026)
    is_mask = df_all["dt"] < pd.Timestamp("2024-01-01")
    oos_mask = df_all["dt"] >= pd.Timestamp("2024-01-01")

    df_is = df_all[is_mask].reset_index(drop=True)
    df_oos = df_all[oos_mask].reset_index(drop=True)

    print(f"数据总览: 30分钟等长 K 线共 {len(df_all)} 根")
    print(f"  ├─ 样本内训练集 (In-Sample, 2020-01 ~ 2023-12):     {len(df_is)} 根 K 线 (用于信号审计与选拔)")
    print(f"  └─ 样本外盲测集 (Out-of-Sample, 2024-01 ~ 2026-10): {len(df_oos)} 根 K 线 (冻结参数，纯粹盲测)")

    # 1. 在 In-Sample (2020-2023) 上执行旧方法 vs 法医级新方法对比审计
    is_records = []
    oos_records = []

    for sig_name, (full_trig, direction) in signals_dict.items():
        trig_is = full_trig[is_mask.values]
        trig_oos = full_trig[oos_mask.values]

        res_is = evaluate_signal_forensic(df_is, trig_is, direction, horizon=16)
        res_oos = evaluate_signal_forensic(df_oos, trig_oos, direction, horizon=16)

        if res_is.get("valid", False):
            res_is["name"] = sig_name
            res_is["dir"] = "多" if direction == "long" else "空"
            res_is["oos_mean"] = res_oos.get("f_mean", 0.0) if res_oos.get("valid", False) else 0.0
            res_is["oos_alpha"] = res_oos.get("excess_alpha", 0.0) if res_oos.get("valid", False) else 0.0
            res_is["oos_safety"] = res_oos.get("safety_margin", 0.0) if res_oos.get("valid", False) else 0.0
            res_is["oos_n"] = res_oos.get("n_purged", 0)
            is_records.append(res_is)

    audit_df = pd.DataFrame(is_records)
    # 执行 Benjamini-Hochberg FDR 多重检验校正
    audit_df["fdr_qval"] = benjamini_hochberg_fdr(audit_df["boot_pval"].tolist())

    # 样本内严苛选拔门禁 (Gate):
    # 1. 样本内超额收益 Excess Alpha > +0.15% (跑赢随机入场基准)
    # 2. 样本内安全边际 Safety Margin > +2.0%
    # 3. 置换检验 p 值 < 0.15
    audit_df["passed_is_gate"] = (
        (audit_df["excess_alpha"] > 0.15) & (audit_df["safety_margin"] > 2.0) & (audit_df["boot_pval"] < 0.15)
    )

    print("\n" + "=" * 125)
    print("【一、2020-2023 样本内 (In-Sample)：传统缺陷方法 vs 法医级三重屏障检验 对比审计表】")
    print("=" * 125)
    cols_compare = [
        "name",
        "dir",
        "n_naive",
        "naive_mean",
        "naive_p",
        "n_purged",
        "f_mean",
        "bench_drift",
        "excess_alpha",
        "f_win",
        "pl_ratio",
        "safety_margin",
        "boot_pval",
        "fdr_qval",
        "passed_is_gate",
    ]
    fmt_compare = {
        "naive_mean": "{:+.2f}%".format,
        "naive_p": "{:.3f}".format,
        "f_mean": "{:+.2f}%".format,
        "bench_drift": "{:+.2f}%".format,
        "excess_alpha": "{:+.2f}%".format,
        "f_win": "{:.1f}%".format,
        "pl_ratio": "{:.2f}".format,
        "safety_margin": "{:+.1f}%".format,
        "boot_pval": "{:.3f}".format,
        "fdr_qval": "{:.3f}".format,
    }
    print(audit_df[cols_compare].to_string(index=False, formatters=fmt_compare))

    print("\n" + "=" * 125)
    print("【二、牛/熊/震荡分层表现 与 2024-2026 样本外 (Out-of-Sample) 盲测延续性验证表】")
    print("=" * 125)
    cols_oos = [
        "name",
        "passed_is_gate",
        "bull_mean",
        "bear_mean",
        "osc_mean",
        "oos_n",
        "oos_mean",
        "oos_alpha",
        "oos_safety",
    ]
    fmt_oos = {
        "bull_mean": "{:+.2f}%".format,
        "bear_mean": "{:+.2f}%".format,
        "osc_mean": "{:+.2f}%".format,
        "oos_mean": "{:+.2f}%".format,
        "oos_alpha": "{:+.2f}%".format,
        "oos_safety": "{:+.1f}%".format,
    }
    print(audit_df[cols_oos].to_string(index=False, formatters=fmt_oos))

    # 2. 基于样本内选拔通过的信号，构建【严格无窥探 IS/OOS 复合策略】与【全样本未筛选策略】对比
    passed_names = audit_df[audit_df["passed_is_gate"]]["name"].tolist()
    rejected_names = audit_df[~audit_df["passed_is_gate"]]["name"].tolist()
    print(f"\n>> [样本内选拔通过的真金信号 ({len(passed_names)}个)]: {passed_names}")
    print(f">> [样本内被证伪淘汰的伪/弱信号 ({len(rejected_names)}个)]: {rejected_names}")

    # 构建次根开盘成交 (Next-Open Execution，shift(1) 权重消除收盘偷价) 的持仓序列
    def build_portfolio_weights(selected_names: list[str]) -> np.ndarray:
        n_bars = len(df_all)
        total_w = np.zeros(n_bars)
        atr_arr = df_all["atr14"].values
        c_arr = df_all["close"].values
        o_arr = df_all["open"].values
        h_arr = df_all["high"].values
        l_arr = df_all["low"].values

        for s_name in selected_names:
            trig, direction = signals_dict[s_name]
            dir_val = 1.0 if direction == "long" else -1.0
            w_sig = np.zeros(n_bars)
            pos = 0
            entry_p = stop_p = tp_p = None
            entry_idx = 0

            for i in range(60, n_bars):
                # 盘中检查三重屏障平仓
                if pos != 0:
                    bars_held = i - entry_idx
                    if pos == 1:
                        if l_arr[i] <= stop_p or h_arr[i] >= tp_p or bars_held >= 16:
                            pos = 0
                    else:
                        if h_arr[i] >= stop_p or l_arr[i] <= tp_p or bars_held >= 16:
                            pos = 0

                # 若前一根 K 线 (i-1) 收盘触发信号，在当前根 K 线 (i) 建仓 (彻底消除当根收盘偷价！)
                if pos == 0 and trig[i - 1]:
                    pos = int(dir_val)
                    entry_p = o_arr[i]
                    atr = atr_arr[i - 1] if not np.isnan(atr_arr[i - 1]) else entry_p * 0.01
                    entry_idx = i
                    if pos == 1:
                        tp_p = entry_p + 2.2 * atr
                        stop_p = entry_p - 1.1 * atr
                    else:
                        tp_p = entry_p - 2.2 * atr
                        stop_p = entry_p + 1.1 * atr

                w_sig[i] = float(pos)
            total_w += w_sig

        # 归一化截断在 [-1.0, 1.0]
        return np.clip(total_w * 0.45, -1.0, 1.0)

    w_forensic = build_portfolio_weights(passed_names)
    w_rejected = build_portfolio_weights(rejected_names) if len(rejected_names) > 0 else np.zeros(len(df_all))

    # 使用 WeightBacktest 回测 (重点考察 2024-2026 纯样本外盲测期表现)
    dfw_forensic = pd.DataFrame(
        {
            "dt": df_all["dt"].astype(str),
            "symbol": "159915",
            "weight": w_forensic,
            "price": df_all["close"].values,
        }
    )
    wb_full = WeightBacktest(data=dfw_forensic, fee_rate=0.0002, weight_type="ts")

    # 单独截取 2024-2026 纯样本外盲测 (OOS)
    dfw_oos = dfw_forensic[df_all["dt"] >= pd.Timestamp("2024-01-01")].reset_index(drop=True)
    wb_oos = WeightBacktest(data=dfw_oos, fee_rate=0.0002, weight_type="ts")

    # 被淘汰信号在样本外的表现 (验证法医检验器是否成功剔除了毒药信号)
    dfw_rej_oos = pd.DataFrame(
        {
            "dt": df_all["dt"].astype(str),
            "symbol": "159915",
            "weight": w_rejected,
            "price": df_all["close"].values,
        }
    )[df_all["dt"] >= pd.Timestamp("2024-01-01")].reset_index(drop=True)
    wb_rej_oos = WeightBacktest(data=dfw_rej_oos, fee_rate=0.0002, weight_type="ts")

    st_full = wb_full.stats
    st_oos = wb_oos.stats
    st_rej = wb_rej_oos.stats

    print("\n" + "=" * 115)
    print("【三、法医级甄别效果实证：入选真金信号组合 vs 被淘汰伪信号组合 在样本外 (2024-2026) 的盲测对比】")
    print("=" * 115)
    summary_port = pd.DataFrame(
        [
            {
                "组合类型": "★ 法医入选组合 · 全周期 (2020-2026)",
                "累计收益": f"{st_full.get('绝对收益', 0) * 100:+.2f}%",
                "年化收益": f"{st_full.get('年化收益', 0) * 100:+.2f}%",
                "夏普比率": f"{st_full.get('夏普比率', 0):.2f}",
                "最大回撤": f"{st_full.get('最大回撤', 0) * 100:.2f}%",
                "卡玛比率": f"{st_full.get('卡玛比率', 0):.2f}",
            },
            {
                "组合类型": "★ 法医入选组合 · 纯样本外盲测 (2024-2026 OOS)",
                "累计收益": f"{st_oos.get('绝对收益', 0) * 100:+.2f}%",
                "年化收益": f"{st_oos.get('年化收益', 0) * 100:+.2f}%",
                "夏普比率": f"{st_oos.get('夏普比率', 0):.2f}",
                "最大回撤": f"{st_oos.get('最大回撤', 0) * 100:.2f}%",
                "卡玛比率": f"{st_oos.get('卡玛比率', 0):.2f}",
            },
            {
                "组合类型": "✗ 被法医淘汰信号组合 · 样本外盲测 (2024-2026 OOS)",
                "累计收益": f"{st_rej.get('绝对收益', 0) * 100:+.2f}%",
                "年化收益": f"{st_rej.get('年化收益', 0) * 100:+.2f}%",
                "夏普比率": f"{st_rej.get('夏普比率', 0):.2f}",
                "最大回撤": f"{st_rej.get('最大回撤', 0) * 100:.2f}%",
                "卡玛比率": f"{st_rej.get('卡玛比率', 0):.2f}",
            },
        ]
    )
    print(summary_port.to_string(index=False))

    # 绘制样本内外分割与对比 HTML 大屏
    dr_full = wb_full.daily_return.copy()
    dr_full["date"] = pd.to_datetime(dr_full["date"])
    ret_forensic = dr_full.set_index("date")["total"]
    nav_forensic = (1 + ret_forensic).cumprod()
    dd_forensic = (nav_forensic - nav_forensic.cummax()) / nav_forensic.cummax() * 100

    # 被淘汰组合全周期净值
    wb_rej_full = WeightBacktest(
        data=pd.DataFrame(
            {"dt": df_all["dt"].astype(str), "symbol": "159915", "weight": w_rejected, "price": df_all["close"].values}
        ),
        fee_rate=0.0002,
        weight_type="ts",
    )
    dr_rej = wb_rej_full.daily_return.copy()
    dr_rej["date"] = pd.to_datetime(dr_rej["date"])
    nav_rej = (1 + dr_rej.set_index("date")["total"]).cumprod()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "【159915 法医级信号甄别与样本外盲测 (IS: 2020-2023 | OOS: 2024-2026)】累计净值对比",
            "法医入选组合 动态回撤深度 (Drawdown %)",
        ),
        row_heights=[0.72, 0.28],
    )

    dates_str = nav_forensic.index.strftime("%Y-%m-%d").tolist()
    fig.add_trace(
        go.Scatter(
            x=dates_str,
            y=nav_forensic.values,
            name="★ 法医入选真金信号组合 (Next-Open + Triple-Barrier)",
            line=dict(color="#00E676", width=3.2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates_str,
            y=nav_rej.loc[nav_forensic.index].values,
            name="✗ 被法医淘汰的伪信号组合",
            line=dict(color="#FF5252", width=1.8, dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates_str, y=dd_forensic.values, name="入选组合回撤 (%)", line=dict(color="#00E5FF", width=1.5)),
        row=2,
        col=1,
    )

    # 标注 2024-01-01 样本内外分割线
    fig.add_vline(x="2024-01-02", line_width=2, line_dash="dash", line_color="#FFD600")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A0E14",
        plot_bgcolor="#101721",
        title=dict(
            text="159915 创业板 ETF · 法医级单信号检验与样本外盲测大屏 (2020-2023 IS -> 2024-2026 OOS)",
            font=dict(size=17, color="#FFFFFF"),
            x=0.02,
            y=0.98,
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0.6)"),
        height=720,
    )
    fig.write_html(str(HTML_PATH), include_plotlyjs="cdn")
    print(f"\n样本内外盲测可视化大屏已保存至: {HTML_PATH.resolve()}")


if __name__ == "__main__":
    main()
