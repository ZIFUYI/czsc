"""
159915 (创业板ETF) 第二轮深度K线形态与微观信号挖掘引擎 (Pattern Discovery Engine Round 2)
系统性检验 10 大全新深度量价与动量信号：
1. 均线粘合一阳穿三线 (MA Ribbon Squeeze & 1-Bar Breakout Long)
2. 均线粘合一阴下三线 (MA Ribbon Squeeze & 1-Bar Breakdown Short)
3. 顺势三阴缩量洗盘反包 (Uptrend 3-Red-Bar Low Volume Retest Long)
4. 逆势三阳缩量死猫跳大阴反杀 (Downtrend 3-Bounce Trap Short)
5. 内孕线 (Inside Bar) 破局爆发 (Inside Bar Expansion Breakout Long)
6. 窒息地量后放量长阳启动 (Volume Dry-Up with Sudden Volume Explosion Long)
7. 天量滞涨大阴出货 (High Volume Churning Distribution Short)
8. 极端负乖离超跌反弹 (Extreme Oversold Bias Mean Reversion Long)
9. MACD 零轴上顺势二次金叉 (MACD Zero-Line Bounce Long)
10. MACD 零轴下顺势二次死叉 (MACD Zero-Line Rejection Short)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = Path("examples/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REAL_5M_CSV = OUTPUT_DIR / "159915_real_5m_2020_2026.csv"


def load_daily_data(csv_path: Path) -> pd.DataFrame:
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


def compute_forward_metrics(df: pd.DataFrame, windows=[1, 3, 5, 10, 20]) -> pd.DataFrame:
    for w in windows:
        df[f"fwd_ret_{w}d"] = df["close"].shift(-w) / df["close"] - 1.0
        fwd_high = df["high"].iloc[::-1].rolling(w).max().iloc[::-1].shift(-w)
        fwd_low = df["low"].iloc[::-1].rolling(w).min().iloc[::-1].shift(-w)
        df[f"fwd_mfe_{w}d"] = fwd_high / df["close"] - 1.0
        df[f"fwd_mae_{w}d"] = fwd_low / df["close"] - 1.0
    return df


def evaluate_pattern(df: pd.DataFrame, mask: pd.Series, pattern_name: str, direction: str = "long") -> dict:
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

    for w in [1, 3, 5, 10, 20]:
        col = f"fwd_ret_{w}d"
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

        t_stat, p_val = stats.ttest_1samp(rets, 0.0)

        res[f"{w}d_mean"] = mean_r
        res[f"{w}d_win"] = win_r
        res[f"{w}d_pl"] = pl
        res[f"{w}d_pval"] = p_val

        if direction == "long":
            res[f"{w}d_mfe"] = sub[f"fwd_mfe_{w}d"].dropna().mean() * 100
            res[f"{w}d_mae"] = sub[f"fwd_mae_{w}d"].dropna().mean() * 100
        else:
            res[f"{w}d_mfe"] = -sub[f"fwd_mae_{w}d"].dropna().mean() * 100
            res[f"{w}d_mae"] = -sub[f"fwd_mfe_{w}d"].dropna().mean() * 100

    return res


def run_discovery_round2():
    print("=" * 90)
    print("  【159915 创业板ETF · 第二轮深度K线形态与微观信号挖掘引擎】 (2020-2026)")
    print("=" * 90)

    df = load_daily_data(REAL_5M_CSV)
    df = compute_forward_metrics(df)

    # 1. 均线族
    df["pct_chg"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma30"] = df["close"].rolling(30).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    # 均线粘合度 (MA5, MA10, MA20, MA30 的极差比率)
    ma_max = df[["ma5", "ma10", "ma20", "ma30"]].max(axis=1)
    ma_min = df[["ma5", "ma10", "ma20", "ma30"]].min(axis=1)
    df["ma_dispersion"] = (ma_max - ma_min) / ma_min

    # 2. MACD 指标 (12, 26, 9)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_diff"] = ema12 - ema26
    df["macd_dea"] = df["macd_diff"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (df["macd_diff"] - df["macd_dea"]) * 2

    # 3. 乖离率 Bias-20
    df["bias20"] = (df["close"] - df["ma20"]) / df["ma20"]

    # ----------------------------------------------------
    # 信号 1: 均线粘合一阳穿三线 (MA Ribbon Squeeze Breakout Long)
    # 均线粘合度 <= 1.8%，今日放量收阳线同时站上 MA5, MA10, MA20, MA30
    ribbon_tight = df["ma_dispersion"] <= 0.02
    bull_pierce = (
        (df["open"] <= df[["ma5", "ma10", "ma20"]].min(axis=1) * 1.005)
        & (df["close"] >= df[["ma5", "ma10", "ma20", "ma30"]].max(axis=1))
        & (df["close"] > df["open"] * 1.015)
    )
    mask_ribbon_long = ribbon_tight & bull_pierce & (df["close"] > df["ma60"])

    # 信号 2: 均线粘合一阴下三线 (MA Ribbon Squeeze Breakdown Short)
    bear_pierce = (
        (df["open"] >= df[["ma5", "ma10", "ma20"]].max(axis=1) * 0.995)
        & (df["close"] <= df[["ma5", "ma10", "ma20", "ma30"]].min(axis=1))
        & (df["close"] < df["open"] * 0.985)
    )
    mask_ribbon_short = ribbon_tight & bear_pierce & (df["close"] < df["ma60"])

    # 信号 3: 顺势三阴缩量洗盘反包 (Uptrend 3-Pullback Retest Long)
    # 处于上升趋势中 (C > MA60, MA20 > MA60)，前三天下跌 (c1<c2<c3)，成交量逐日递减，今日放量大阳反包
    three_red_days = (df["close"].shift(1) < df["close"].shift(2)) & (df["close"].shift(2) < df["close"].shift(3))
    vol_shrink_3d = (df["volume"].shift(1) < df["volume"].shift(2)) & (df["volume"].shift(2) < df["volume"].shift(3))
    bull_engulf_today = (
        (df["close"] > df["open"])
        & (df["close"] >= df["high"].shift(1))
        & (df["volume"] >= 1.3 * df["volume"].shift(1))
    )
    mask_pullback_3d = (
        (df["close"] > df["ma60"])
        & three_red_days
        & vol_shrink_3d
        & bull_engulf_today
        & (df["low"].shift(1) > df["ma60"].shift(1))
    )

    # 信号 4: 逆势三阳缩量死猫跳大阴反杀 (Downtrend 3-Bounce Trap Short)
    # 处于单边下行中 (C < MA60, MA20 < MA60)，前三天小幅反弹 (c1>c2>c3)，但缩量无力；今日大阴反杀跌破前一阳线低点
    three_up_days = (df["close"].shift(1) > df["close"].shift(2)) & (df["close"].shift(2) > df["close"].shift(3))
    vol_weak_3d = (df["volume"].shift(1) <= 1.1 * df["vol_ma20"].shift(1)) & (
        df["volume"].shift(2) <= 1.1 * df["vol_ma20"].shift(2)
    )
    bear_kill_today = (df["close"] < df["open"]) & (df["close"] <= df["low"].shift(1)) & (df["pct_chg"] <= -0.015)
    mask_bounce_trap = (df["close"] < df["ma60"]) & three_up_days & vol_weak_3d & bear_kill_today

    # 信号 5: 内孕线破局爆发 (Inside Bar Expansion Long)
    # 昨日完全内包于前日 (High1 < High2 且 Low1 > Low2)，今日放量向上突破前天母线高点
    inside_bar_yesterday = (df["high"].shift(1) < df["high"].shift(2)) & (df["low"].shift(1) > df["low"].shift(2))
    break_mother_bar = (
        (df["close"] > df["high"].shift(2)) & (df["close"] > df["open"]) & (df["volume"] >= 1.2 * df["vol_ma20"])
    )
    mask_inside_break = inside_bar_yesterday & break_mother_bar & (df["close"] > df["ma60"])

    # 信号 6: 窒息地量后放量长阳启动 (Volume Dry-Up with Explosion Long)
    # 近3天至少有1天地量 (成交量 < 0.65 * MA_Vol20)，今日突然成交量放大 >= 2.0 倍 MA20，涨幅 >= 2.2%
    had_dry_vol = (df["volume"].shift(1) <= 0.65 * df["vol_ma20"].shift(1)) | (
        df["volume"].shift(2) <= 0.65 * df["vol_ma20"].shift(2)
    )
    vol_explosion = (df["volume"] >= 2.0 * df["vol_ma20"]) & (df["pct_chg"] >= 0.022) & (df["close"] > df["open"])
    mask_dry_explosion = had_dry_vol & vol_explosion

    # 信号 7: 天量滞涨大阴出货 (High Volume Churning Distribution Short)
    # 近20天涨幅 >= 7%，今日高位振幅很大，成交量暴增至 >= 2.2 倍 MA20，但收盘收出实体阴线 (或冲高回落长上影)
    run_up_20d = (df["close"].shift(1) / df["close"].shift(20) - 1.0) >= 0.07
    churning_bar = (
        (df["volume"] >= 2.2 * df["vol_ma20"])
        & (df["close"] < df["open"])
        & (df["close"] <= df["low"] + 0.4 * (df["high"] - df["low"]))
    )
    mask_churning_top = run_up_20d & churning_bar

    # 信号 8: 极端负乖离超跌反弹 (Extreme Oversold Bias Long)
    # Bias-20 <= -5.5% (严重远离20日均线，空头动能耗尽)，今日收出反包阳线 (C > High[t-1])
    extreme_bias = df["bias20"] <= -0.055
    bias_rebound = (df["close"] > df["open"]) & (df["close"] > df["high"].shift(1))
    mask_extreme_oversold = extreme_bias & bias_rebound

    # 信号 9: MACD 零轴上二次水上金叉 (MACD Zero-Line Bounce Long)
    # DIFF > 0 且 DEA > 0 (水上)，DIFF 触碰 DEA 后的金叉 (昨日 diff <= dea，今日 diff > dea)
    macd_cross_up = (df["macd_diff"].shift(1) <= df["macd_dea"].shift(1)) & (df["macd_diff"] > df["macd_dea"])
    macd_above_zero = (df["macd_diff"] > 0) & (df["macd_dea"] > 0)
    mask_macd_bounce = macd_cross_up & macd_above_zero & (df["close"] > df["ma60"])

    # 信号 10: MACD 零轴下二次水下死叉 (MACD Zero-Line Rejection Short)
    # DIFF < 0 且 DEA < 0 (水下)，反弹受阻后死叉 (昨日 diff >= dea，今日 diff < dea)
    macd_cross_down = (df["macd_diff"].shift(1) >= df["macd_dea"].shift(1)) & (df["macd_diff"] < df["macd_dea"])
    macd_below_zero = (df["macd_diff"] < 0) & (df["macd_dea"] < 0)
    mask_macd_rejection = macd_cross_down & macd_below_zero & (df["close"] < df["ma60"])

    # 评估列表
    patterns_to_test = [
        (df.index >= 60, "基准: 全样本无条件基准 (All Days)", "long"),
        (mask_ribbon_long, "【信号1】均线粘合一阳穿多线 (Ribbon Long)", "long"),
        (mask_ribbon_short, "【信号2】均线粘合一阴下多线 (Ribbon Short)", "short"),
        (mask_pullback_3d, "【信号3】顺势三连阴缩量洗盘反包", "long"),
        (mask_bounce_trap, "【信号4】逆势三连阳缩量死猫跳反杀", "short"),
        (mask_inside_break, "【信号5】内孕线突破母线高点", "long"),
        (mask_dry_explosion, "【信号6】窒息地量后放量大阳启动", "long"),
        (mask_churning_top, "【信号7】高位放天量滞涨阴线出货", "short"),
        (mask_extreme_oversold, "【信号8】极端负乖离 (Bias<=-5.5%) 超跌反弹", "long"),
        (mask_macd_bounce, "【信号9】MACD 零轴上二次水上金叉", "long"),
        (mask_macd_rejection, "【信号10】MACD 零轴下二次水下死叉", "short"),
    ]

    results = []
    for m, name, direct in patterns_to_test:
        eval_res = evaluate_pattern(df, m, name, direct)
        results.append(eval_res)

    res_df = pd.DataFrame(results)

    print("\n" + "=" * 115)
    print("【159915 真实行情 第二轮深度信号前向预测力实证统计表 (2020-2026)】")
    print("=" * 115)

    cols = [
        "name",
        "count",
        "direction",
        "1d_mean",
        "1d_win",
        "3d_mean",
        "3d_win",
        "5d_mean",
        "5d_win",
        "5d_pl",
        "5d_mfe",
        "5d_mae",
        "10d_mean",
        "10d_win",
        "10d_pval",
        "20d_mean",
        "20d_win",
    ]

    formatters = {
        "1d_mean": "{:+.2f}%".format,
        "1d_win": "{:.1f}%".format,
        "3d_mean": "{:+.2f}%".format,
        "3d_win": "{:.1f}%".format,
        "5d_mean": "{:+.2f}%".format,
        "5d_win": "{:.1f}%".format,
        "5d_pl": "{:.2f}".format,
        "5d_mfe": "{:+.2f}%".format,
        "5d_mae": "{:+.2f}%".format,
        "10d_mean": "{:+.2f}%".format,
        "10d_win": "{:.1f}%".format,
        "10d_pval": "{:.4f}".format,
        "20d_mean": "{:+.2f}%".format,
        "20d_win": "{:.1f}%".format,
    }

    print(res_df[cols].to_string(index=False, formatters=formatters))
    print("=" * 115)

    return res_df


if __name__ == "__main__":
    run_discovery_round2()
