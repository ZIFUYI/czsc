#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多标的 (IM, IC, IF, 159915) 经典K线形态与结构自动化因子挖掘与前向收益实证引擎
(Multi-Asset Factor & Pattern Discovery Engine)

参照 159915 原型挖掘标准 (Round 1 + Round 2 全对称多空架构)：
系统性检验 20 大经典量价形态与动量博弈结构在 IM、IC、IF、159915 上的前向预测力 (Event Study):
支持多周期切换：--freq 1d (默认日线) 或 --freq 60min (60分钟高频CTA周期)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy import stats

DATA_BASE_DIR = Path(r"D:\my-qlib\data")
OUTPUT_DIR = Path("examples/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_ASSETS = [
    ("IM9999.CCFX", "IM中证1000期指主力连续"),
    ("IC9999.CCFX", "IC中证500期指主力连续"),
    ("IF9999.CCFX", "IF沪深300期指主力连续"),
    ("159915.XSHE", "159915创业板ETF"),
]


def load_asset_data(symbol: str, freq: str = "1d") -> pd.DataFrame:
    file_path = DATA_BASE_DIR / freq / f"{symbol}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"找不到本地数据文件: {file_path}")

    df = pd.read_parquet(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.rename(columns={"datetime": "dt"})
    df["symbol"] = symbol
    return df


def compute_forward_metrics(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"fwd_ret_{w}b"] = df["close"].shift(-w) / df["close"] - 1.0
        fwd_high = df["high"].iloc[::-1].rolling(w).max().iloc[::-1].shift(-w)
        fwd_low = df["low"].iloc[::-1].rolling(w).min().iloc[::-1].shift(-w)
        df[f"fwd_mfe_{w}b"] = fwd_high / df["close"] - 1.0
        df[f"fwd_mae_{w}b"] = fwd_low / df["close"] - 1.0
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 基础收益与均线族
    df["pct_chg"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma30"] = df["close"].rolling(30).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    # K线几何微观特征
    df["body"] = (df["close"] - df["open"]).abs()
    df["total_range"] = df["high"] - df["low"] + 1e-6
    df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

    # ATR 14
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = df["tr"].rolling(14).mean()
    df["atr_ratio"] = df["atr14"] / df["close"]

    # 历史极值位
    df["prev_high_20"] = df["high"].shift(1).rolling(20).max()
    df["prev_low_20"] = df["low"].shift(1).rolling(20).min()
    df["prev_high_10"] = df["high"].shift(1).rolling(10).max()
    df["prev_low_10"] = df["low"].shift(1).rolling(10).min()

    # 均线粘合度 (MA5, MA10, MA20, MA30 的极差比率)
    ma_max = df[["ma5", "ma10", "ma20", "ma30"]].max(axis=1)
    ma_min = df[["ma5", "ma10", "ma20", "ma30"]].min(axis=1)
    df["ma_dispersion"] = (ma_max - ma_min) / ma_min

    # MACD 指标 (12, 26, 9)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_diff"] = ema12 - ema26
    df["macd_dea"] = df["macd_diff"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (df["macd_diff"] - df["macd_dea"]) * 2

    # 乖离率 Bias-20
    df["bias20"] = (df["close"] - df["ma20"]) / df["ma20"]

    return df


def define_patterns(df: pd.DataFrame, freq: str = "1d") -> list[tuple[pd.Series, str, str]]:
    """
    定义 20 大经典形态与结构（多空对称设计）
    针对不同频率自动调节振幅/缺口容忍度
    """
    is_daily = freq == "1d"
    # 自适应门槛
    min_engulf_ret = 0.010 if is_daily else 0.005
    bias_limit = -0.050 if is_daily else -0.025
    gap_limit = 1.002 if is_daily else 1.001

    # ---------------- 结构 1: 2B 假跌破反转 (Long) ----------------
    mask_2b_long = (
        (df["low"] < df["prev_low_20"])
        & (df["close"] > df["prev_low_20"])
        & (df["close"] > df["open"])
    )

    # ---------------- 结构 2: 2B 假突破反转 (Short) ----------------
    mask_2b_short = (
        (df["high"] > df["prev_high_20"])
        & (df["close"] < df["prev_high_20"])
        & (df["close"] < df["open"])
    )

    # ---------------- 结构 3: W底次低点抬高突破 (Long / 二买) ----------------
    rolling_min_loc = df["low"].rolling(20).apply(lambda s: s.argmin(), raw=True)
    low_raised = (rolling_min_loc >= 5) & (
        df["low"].rolling(5).min() > df["low"].shift(5).rolling(15).min() * 1.002
    )
    mask_w_bottom = (
        low_raised
        & (df["close"] > df["open"])
        & (df["close"] > df["high"].shift(1))
        & (df["close"] > df["ma20"])
    )

    # ---------------- 结构 4: M顶次高点降低跌破 (Short / 二卖) ----------------
    rolling_max_loc = df["high"].rolling(20).apply(lambda s: s.argmax(), raw=True)
    high_lowered = (rolling_max_loc >= 5) & (
        df["high"].rolling(5).max() < df["high"].shift(5).rolling(15).max() * 0.998
    )
    mask_m_top = (
        high_lowered
        & (df["close"] < df["open"])
        & (df["close"] < df["low"].shift(1))
        & (df["close"] < df["ma20"])
    )

    # ---------------- 结构 5: VCP 波动率压缩爆发 (Long) ----------------
    atr_rank_60 = df["atr14"].rolling(60).apply(lambda s: pd.Series(s).rank(pct=True).iloc[-1], raw=False)
    mask_vcp_break = (
        (atr_rank_60 <= 0.25)
        & (df["total_range"] >= 1.35 * df["atr14"])
        & (df["close"] > df["open"])
        & (df["close"] > df["ma20"])
    )

    # ---------------- 结构 6: 均线顺势首次回踩MA20企稳 (Long) ----------------
    was_above_ma20 = (
        (df["close"].shift(1) > df["ma20"].shift(1))
        & (df["close"].shift(2) > df["ma20"].shift(2))
        & (df["close"].shift(3) > df["ma20"].shift(3))
    )
    touch_ma20 = (df["low"] <= df["ma20"] * 1.005) & (df["close"] >= df["ma20"])
    bull_engulf = (df["close"] > df["open"]) & (df["close"] > df["high"].shift(1))
    mask_pullback_ma20 = (
        (df["close"] > df["ma60"])
        & (df["ma20"] > df["ma60"])
        & was_above_ma20
        & touch_ma20
        & bull_engulf
    )

    # ---------------- 结构 7: TD 序列买入结构 9 (Long) ----------------
    td_buy = (df["close"] < df["close"].shift(4)).astype(int)
    td_buy_count = td_buy.groupby((~td_buy.astype(bool)).cumsum()).cumsum()
    mask_td9_buy = td_buy_count == 9

    # ---------------- 结构 8: 放量突破跳空缺口不补 (Long) ----------------
    has_gap_up = (
        (df["low"] > df["high"].shift(1) * gap_limit)
        & (df["close"] > df["open"])
        & (df["volume"] >= 1.20 * df["vol_ma20"])
    )

    # ---------------- 结构 9: 地量十字星后放量长阳反包 (Long) ----------------
    # 前一根地量十字星 (成交量 <= 0.70x MA20 且 实体/振幅 <= 0.40)，当根放量收阳反包
    yesterday_star = (df["volume"].shift(1) <= 0.70 * df["vol_ma20"].shift(1)) & (
        df["body"].shift(1) / df["total_range"].shift(1) <= 0.40
    )
    today_engulf = (
        (df["close"] > df["high"].shift(1))
        & (df["close"] > df["open"])
        & (df["volume"] >= 1.25 * df["vol_ma20"])
    )
    mask_star_engulf = yesterday_star & today_engulf

    # ---------------- 结构 10: 极端长下影锤子线 (Long) ----------------
    mask_hammer = (
        (df["lower_shadow"] / df["total_range"] >= 0.50)
        & (df["close"] > df["low"] + 0.50 * df["total_range"])
        & (df["close"] < df["ma10"])
    )

    # ---------------- 结构 11: 均线粘合一阳穿多线 (Long) ----------------
    ribbon_tight = df["ma_dispersion"] <= (0.022 if is_daily else 0.015)
    bull_pierce = (
        (df["open"] <= df[["ma5", "ma10", "ma20"]].min(axis=1) * 1.005)
        & (df["close"] >= df[["ma5", "ma10", "ma20", "ma30"]].max(axis=1))
        & (df["close"] > df["open"] * (1.010 if is_daily else 1.005))
    )
    mask_ribbon_long = ribbon_tight & bull_pierce & (df["close"] > df["ma60"])

    # ---------------- 结构 12: 均线粘合一阴下多线 (Short) ----------------
    bear_pierce = (
        (df["open"] >= df[["ma5", "ma10", "ma20"]].max(axis=1) * 0.995)
        & (df["close"] <= df[["ma5", "ma10", "ma20", "ma30"]].min(axis=1))
        & (df["close"] < df["open"] * (0.990 if is_daily else 0.995))
    )
    mask_ribbon_short = ribbon_tight & bear_pierce & (df["close"] < df["ma60"])

    # ---------------- 结构 13: 顺势三连阴缩量洗盘反包 (Long) ----------------
    three_red_bars = (df["close"].shift(1) < df["close"].shift(2)) & (df["close"].shift(2) < df["close"].shift(3))
    vol_shrink = (df["volume"].shift(1) < df["volume"].shift(2)) & (df["volume"].shift(2) < df["volume"].shift(3))
    bull_reversal = (
        (df["close"] > df["open"])
        & (df["close"] >= df["high"].shift(1))
        & (df["volume"] >= 1.20 * df["volume"].shift(1))
    )
    mask_pullback_3d = (
        (df["close"] > df["ma60"])
        & three_red_bars
        & vol_shrink
        & bull_reversal
        & (df["low"].shift(1) > df["ma60"].shift(1))
    )

    # ---------------- 结构 14: 逆势三连阳缩量死猫跳反杀 (Short) ----------------
    three_up_bars = (df["close"].shift(1) > df["close"].shift(2)) & (df["close"].shift(2) > df["close"].shift(3))
    vol_weak = (df["volume"].shift(1) <= 1.20 * df["vol_ma20"].shift(1)) & (
        df["volume"].shift(2) <= 1.20 * df["vol_ma20"].shift(2)
    )
    bear_reversal = (
        (df["close"] < df["open"])
        & (df["close"] <= df["low"].shift(1))
        & (df["pct_chg"] <= (-0.010 if is_daily else -0.005))
    )
    mask_bounce_trap = (df["close"] < df["ma60"]) & three_up_bars & vol_weak & bear_reversal

    # ---------------- 结构 15: 内孕线突破母线高点 (Long) ----------------
    inside_bar_prev = (df["high"].shift(1) < df["high"].shift(2)) & (df["low"].shift(1) > df["low"].shift(2))
    break_mother_bar = (
        (df["close"] > df["high"].shift(2)) & (df["close"] > df["open"]) & (df["volume"] >= 1.10 * df["vol_ma20"])
    )
    mask_inside_break = inside_bar_prev & break_mother_bar & (df["close"] > df["ma60"])

    # ---------------- 结构 16: 窒息地量后放量大阳启动 (Long) ----------------
    had_dry_vol = (df["volume"].shift(1) <= 0.70 * df["vol_ma20"].shift(1)) | (
        df["volume"].shift(2) <= 0.70 * df["vol_ma20"].shift(2)
    )
    vol_explosion = (
        (df["volume"] >= 1.65 * df["vol_ma20"])
        & (df["close"] > df["open"])
        & (df["total_range"] >= 1.25 * df["atr14"])
    )
    mask_dry_explosion = had_dry_vol & vol_explosion

    # ---------------- 结构 17: 高位放天量滞涨阴线出货 (Short) ----------------
    run_up_bars = (df["close"].shift(1) / df["close"].shift(20) - 1.0) >= (0.05 if is_daily else 0.025)
    churning_bar = (
        (df["volume"] >= 1.80 * df["vol_ma20"])
        & (df["close"] < df["open"])
        & (df["close"] <= df["low"] + 0.50 * (df["high"] - df["low"]))
    )
    mask_churning_top = run_up_bars & churning_bar

    # ---------------- 结构 18: 极端负乖离超跌反弹 (Long) ----------------
    extreme_bias = df["bias20"] <= bias_limit
    bias_rebound = (df["close"] > df["open"]) & (df["close"] > df["high"].shift(1))
    mask_extreme_oversold = extreme_bias & bias_rebound

    # ---------------- 结构 19: MACD 零轴上二次水上金叉 (Long) ----------------
    macd_cross_up = (df["macd_diff"].shift(1) <= df["macd_dea"].shift(1)) & (df["macd_diff"] > df["macd_dea"])
    macd_above_zero = (df["macd_diff"] > 0) & (df["macd_dea"] > 0)
    mask_macd_bounce = macd_cross_up & macd_above_zero & (df["close"] > df["ma60"])

    # ---------------- 结构 20: MACD 零轴下二次水下死叉 (Short) ----------------
    macd_cross_down = (df["macd_diff"].shift(1) >= df["macd_dea"].shift(1)) & (df["macd_diff"] < df["macd_dea"])
    macd_below_zero = (df["macd_diff"] < 0) & (df["macd_dea"] < 0)
    mask_macd_rejection = macd_cross_down & macd_below_zero & (df["close"] < df["ma60"])

    patterns = [
        (df.index >= 60, "全样本无条件基准 (Benchmark)", "long"),
        (mask_2b_long, "【结构01】2B 假跌破反转 (2B Fake Breakdown)", "long"),
        (mask_2b_short, "【结构02】2B 假突破反转 (2B Fake Breakout)", "short"),
        (mask_w_bottom, "【结构03】W底次低抬高突破 (W-Bottom/二买)", "long"),
        (mask_m_top, "【结构04】M顶次高降低跌破 (M-Top/二卖)", "short"),
        (mask_vcp_break, "【结构05】VCP 波动率极度压缩爆发", "long"),
        (mask_pullback_ma20, "【结构06】顺势首次回踩 MA20 企稳", "long"),
        (mask_td9_buy, "【结构07】TD 序列买入结构 9 (TD Buy 9)", "long"),
        (has_gap_up, "【结构08】放量跳空突破缺口不补 (Breakaway Gap)", "long"),
        (mask_star_engulf, "【结构09】地量十字星后放量长阳反包", "long"),
        (mask_hammer, "【结构10】短期超跌长下影锤子线 (Pinbar)", "long"),
        (mask_ribbon_long, "【结构11】均线粘合一阳穿多线 (Ribbon Long)", "long"),
        (mask_ribbon_short, "【结构12】均线粘合一阴下多线 (Ribbon Short)", "short"),
        (mask_pullback_3d, "【结构13】顺势三连阴缩量洗盘反包", "long"),
        (mask_bounce_trap, "【结构14】逆势三连阳缩量死猫跳反杀", "short"),
        (mask_inside_break, "【结构15】内孕线突破母线高点", "long"),
        (mask_dry_explosion, "【结构16】窒息地量后放量大阳启动", "long"),
        (mask_churning_top, "【结构17】高位放天量滞涨阴线出货", "short"),
        (mask_extreme_oversold, "【结构18】极端负乖离 (Bias<=-5%) 超跌反弹", "long"),
        (mask_macd_bounce, "【结构19】MACD 零轴上二次水上金叉", "long"),
        (mask_macd_rejection, "【结构20】MACD 零轴下二次水下死叉", "short"),
    ]
    return patterns


def evaluate_pattern(
    df: pd.DataFrame,
    mask: pd.Series,
    pattern_name: str,
    direction: str = "long",
    windows: list[int] = [1, 3, 5, 10, 20],
) -> dict:
    valid_mask = mask & (df.index >= 60)
    sub = df[valid_mask].copy()
    n = len(sub)
    if n == 0:
        return {"name": pattern_name, "count": 0, "direction": direction}

    res = {
        "name": pattern_name,
        "count": n,
        "direction": direction,
    }

    for w in windows:
        col = f"fwd_ret_{w}b"
        raw_rets = sub[col].dropna()
        if len(raw_rets) == 0:
            continue

        rets = raw_rets if direction == "long" else -raw_rets

        mean_r = rets.mean() * 100
        win_r = (rets > 0).mean() * 100
        pos = rets[rets > 0]
        neg = rets[rets < 0]
        avg_g = pos.mean() * 100 if len(pos) > 0 else 0
        avg_l = abs(neg.mean() * 100) if len(neg) > 0 else 0
        pl = (avg_g / avg_l) if avg_l > 0 else np.nan

        if len(rets) > 1 and rets.std() > 0:
            t_stat, p_val = stats.ttest_1samp(rets, 0.0)
        else:
            t_stat, p_val = np.nan, np.nan

        res[f"{w}b_mean"] = mean_r
        res[f"{w}b_win"] = win_r
        res[f"{w}b_pl"] = pl
        res[f"{w}b_tstat"] = t_stat
        res[f"{w}b_pval"] = p_val

        if direction == "long":
            res[f"{w}b_mfe"] = sub[f"fwd_mfe_{w}b"].dropna().mean() * 100
            res[f"{w}b_mae"] = sub[f"fwd_mae_{w}b"].dropna().mean() * 100
        else:
            res[f"{w}b_mfe"] = -sub[f"fwd_mae_{w}b"].dropna().mean() * 100
            res[f"{w}b_mae"] = -sub[f"fwd_mfe_{w}b"].dropna().mean() * 100

    return res


def run_mining_for_freq(freq: str = "1d"):
    windows = [1, 3, 5, 10, 20] if freq == "1d" else [1, 4, 8, 16, 32]
    core_w = 5 if freq == "1d" else 8  # 核心评价持有期 (日线5天，60m对应8根约2天)

    print("\n" + "=" * 115)
    print(f"  【IM · IC · IF · 159915 四大核心标的 · 20大经典K线形态因子挖掘】 · 频率: {freq}")
    print(f"  持有期评价窗口: {windows} 根 Bar (核心观测: {core_w} 根 Bar)")
    print("=" * 115)

    all_asset_results: dict[str, pd.DataFrame] = {}

    for symbol, name in TARGET_ASSETS:
        df_raw = load_asset_data(symbol, freq=freq)
        df_fwd = compute_forward_metrics(df_raw, windows=windows)
        df_feat = compute_features(df_fwd)
        patterns = define_patterns(df_feat, freq=freq)

        results = []
        for mask, p_name, direct in patterns:
            eval_res = evaluate_pattern(df_feat, mask, p_name, direct, windows=windows)
            results.append(eval_res)

        res_df = pd.DataFrame(results)
        res_df["symbol"] = symbol
        res_df["asset_name"] = name
        res_df["freq"] = freq
        all_asset_results[symbol] = res_df

        # 打印单标的结果表
        w1, w2, w3, w4 = windows[0], windows[1], core_w, windows[-1]
        cols_show = [
            "name",
            "count",
            "direction",
            f"{w1}b_mean",
            f"{w1}b_win",
            f"{w2}b_mean",
            f"{w2}b_win",
            f"{w3}b_mean",
            f"{w3}b_win",
            f"{w3}b_pl",
            f"{w3}b_mfe",
            f"{w3}b_mae",
            f"{w3}b_pval",
            f"{w4}b_mean",
            f"{w4}b_win",
        ]
        formatters = {
            f"{w1}b_mean": "{:+.2f}%".format,
            f"{w1}b_win": "{:.1f}%".format,
            f"{w2}b_mean": "{:+.2f}%".format,
            f"{w2}b_win": "{:.1f}%".format,
            f"{w3}b_mean": "{:+.2f}%".format,
            f"{w3}b_win": "{:.1f}%".format,
            f"{w3}b_pl": "{:.2f}".format,
            f"{w3}b_mfe": "{:+.2f}%".format,
            f"{w3}b_mae": "{:+.2f}%".format,
            f"{w3}b_pval": "{:.4f}".format,
            f"{w4}b_mean": "{:+.2f}%".format,
            f"{w4}b_win": "{:.1f}%".format,
        }

        print("-" * 115)
        print(f"【{name} ({freq}) 因子与形态前向统计实证表 (总K线条数 N = {len(df_raw)})】")
        print("-" * 115)
        # 兼容部分未触发的列
        valid_cols = [c for c in cols_show if c in res_df.columns]
        valid_formatters = {k: v for k, v in formatters.items() if k in valid_cols}
        print(res_df[valid_cols].to_string(index=False, formatters=valid_formatters))

    # 保存全量 CSV
    combined_df = pd.concat(all_asset_results.values(), ignore_index=True)
    combined_csv = OUTPUT_DIR / f"multi_asset_factor_discovery_{freq}.csv"
    combined_df.to_csv(combined_csv, index=False, encoding="utf-8-sig")

    # 构建跨标的横向对比矩阵
    bench_ret = {}
    for sym, _ in TARGET_ASSETS:
        sub = all_asset_results[sym]
        bench_ret[sym] = sub[sub["name"].str.contains("基准")][f"{core_w}b_mean"].values[0]

    pattern_names = [p[1] for p in patterns if "基准" not in p[1]]
    matrix_rows = []
    for p_name in pattern_names:
        row = {"因子结构名称": p_name}
        for sym, nick in [
            ("IM9999.CCFX", "IM(中证1000)"),
            ("IC9999.CCFX", "IC(中证500)"),
            ("IF9999.CCFX", "IF(沪深300)"),
            ("159915.XSHE", "159915(创业板)"),
        ]:
            sub = all_asset_results[sym]
            p_data = sub[sub["name"] == p_name]
            if len(p_data) == 0 or p_data["count"].values[0] == 0:
                row[f"{nick} (N/收益/胜率/α)"] = "N=0"
                continue

            cnt = p_data["count"].values[0]
            ret_val = p_data[f"{core_w}b_mean"].values[0]
            win_val = p_data[f"{core_w}b_win"].values[0]
            pval_val = p_data[f"{core_w}b_pval"].values[0]
            alpha_val = ret_val - bench_ret[sym]
            star = "***" if pval_val < 0.01 else ("**" if pval_val < 0.05 else ("*" if pval_val < 0.10 else ""))
            row[f"{nick} (N/收益/胜率/α)"] = f"N={cnt}|{ret_val:+.2f}%|{win_val:.0f}%|α:{alpha_val:+.2f}%{star}"

        matrix_rows.append(row)

    matrix_df = pd.DataFrame(matrix_rows)
    matrix_csv = OUTPUT_DIR / f"cross_asset_alpha_matrix_{freq}_{core_w}b.csv"
    matrix_df.to_csv(matrix_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 115)
    print(f"  【四大标的 (IM / IC / IF / 159915) 核心 {core_w} 根 Bar 持有期横向对比矩阵 · {freq}】")
    print("=" * 115)
    print(matrix_df.to_string(index=False))
    print("=" * 115)
    print(f"[OK] 结果已导出至: {combined_csv} 和 {matrix_csv}")

    return combined_df, matrix_df


def main():
    parser = argparse.ArgumentParser(description="多标的因子与形态自动化挖掘")
    parser.add_argument("--freq", type=str, default="1d", choices=["1d", "60min", "15min", "5min"], help="分析周期")
    args = parser.parse_args()

    run_mining_for_freq(freq=args.freq)


if __name__ == "__main__":
    main()
