"""
159915 (创业板ETF) 经典K线形态与结构自动化挖掘与前向收益检验引擎 (Pattern Discovery Engine)
系统性检验 8 大经典 K 线几何与博弈结构：
1. 2B 假跌破反转 (2B Fake Breakdown Reversal)
2. 2B 假突破反转 (2B Fake Breakout Reversal)
3. W底次低点抬高突破颈线 (Higher Low W-Bottom / 缠论二买)
4. M顶次高点降低跌破颈线 (Lower High M-Top / 缠论二卖)
5. 极端波动率压缩爆发 (VCP Volatility Contraction Breakout)
6. 均线顺势首次回踩MA20企稳 (Trend First Pullback to MA20)
7. TD 序列 9 转买入衰竭 (TD Setup Buy 9)
8. 放量突破跳空缺口不补 (Breakaway Runaway Gap)
9. 极端长下影锤子线 (Extreme Pinbar Hammer)
10. 极度缩量地量十字星后长阳反包 (Ultra-Low Volume Doji with Bull Engulfing)
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

    # 以 5 天和 10 天作为核心评价窗口
    for w in [1, 3, 5, 10, 20]:
        col = f"fwd_ret_{w}d"
        raw_rets = sub[col].dropna()
        if len(raw_rets) == 0:
            continue

        # 若为空头方向，则收益反向计算
        rets = raw_rets if direction == "long" else -raw_rets

        mean_r = rets.mean() * 100
        win_r = (rets > 0).mean() * 100
        pos = rets[rets > 0]
        neg = rets[rets < 0]
        avg_g = pos.mean() * 100 if len(pos) > 0 else 0
        avg_l = abs(neg.mean() * 100) if len(neg) > 0 else 0
        pl = (avg_g / avg_l) if avg_l > 0 else np.nan

        # t 检验 vs 0
        t_stat, p_val = stats.ttest_1samp(rets, 0.0)

        res[f"{w}d_mean"] = mean_r
        res[f"{w}d_win"] = win_r
        res[f"{w}d_pl"] = pl
        res[f"{w}d_pval"] = p_val

        if direction == "long":
            res[f"{w}d_mfe"] = sub[f"fwd_mfe_{w}d"].dropna().mean() * 100
            res[f"{w}d_mae"] = sub[f"fwd_mae_{w}d"].dropna().mean() * 100
        else:
            # 空头方向：有利偏移是跌幅，不利偏移是涨幅
            res[f"{w}d_mfe"] = -sub[f"fwd_mae_{w}d"].dropna().mean() * 100
            res[f"{w}d_mae"] = -sub[f"fwd_mfe_{w}d"].dropna().mean() * 100

    return res


def run_pattern_discovery():
    print("=" * 85)
    print("  【159915 创业板ETF · K线形态与结构自动化挖掘引擎】 (2020-2026)")
    print("=" * 85)

    df = load_daily_data(REAL_5M_CSV)
    df = compute_forward_metrics(df)

    # 基础指标
    df["pct_chg"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    # K线微观特征
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

    # 过去 20 天最高最低（前移 1 天）
    df["prev_high_20"] = df["high"].shift(1).rolling(20).max()
    df["prev_low_20"] = df["low"].shift(1).rolling(20).min()
    df["prev_high_10"] = df["high"].shift(1).rolling(10).max()
    df["prev_low_10"] = df["low"].shift(1).rolling(10).min()

    # ----------------------------------------------------
    # 结构 1: 2B 假跌破反转 (2B Fake Breakdown Long)
    # 当日盘中跌破过去20日低点，但收盘价强力反包收在20日低点上方，且当日收阳
    mask_2b_long = (df["low"] < df["prev_low_20"]) & (df["close"] > df["prev_low_20"]) & (df["close"] > df["open"])

    # 结构 2: 2B 假突破反转 (2B Fake Breakout Short)
    # 当日盘中突破过去20日高点，但收盘价跌破20日高点下方，且当日收阴
    mask_2b_short = (df["high"] > df["prev_high_20"]) & (df["close"] < df["prev_high_20"]) & (df["close"] < df["open"])

    # 结构 3: W底次低点抬高突破 (Higher Low W-Bottom Long / 缠论二买)
    # 过去20日内出现过明显低点A，随后反弹，近期回踩低点B未破A (Low_B > Low_A)，且今日收阳突破近5日小高点
    # 量化判定：过去20日最低点在6~18天前发生；近5天的最低点高于该最低点（抬高）；今日放量收阳突破10日最高
    rolling_min_loc = df["low"].rolling(20).apply(lambda s: s.argmin(), raw=True)  # 距离当前的天数
    low_raised = (rolling_min_loc >= 5) & (df["low"].rolling(5).min() > df["low"].shift(5).rolling(15).min() * 1.005)
    mask_w_bottom = (
        low_raised & (df["close"] > df["open"]) & (df["close"] > df["high"].shift(1)) & (df["close"] > df["ma20"])
    )

    # 结构 4: M顶次高点降低跌破 (Lower High M-Top Short / 缠论二卖)
    rolling_max_loc = df["high"].rolling(20).apply(lambda s: s.argmax(), raw=True)
    high_lowered = (rolling_max_loc >= 5) & (
        df["high"].rolling(5).max() < df["high"].shift(5).rolling(15).max() * 0.995
    )
    mask_m_top = (
        high_lowered & (df["close"] < df["open"]) & (df["close"] < df["low"].shift(1)) & (df["close"] < df["ma20"])
    )

    # 结构 5: 极端波动率压缩爆发 (VCP Volatility Contraction Breakout)
    # ATR处于过去60天最低15%分位，且今日单根实体振幅扩张 > 1.8倍ATR，且收盘突破布林上轨
    atr_rank_60 = df["atr14"].rolling(60).apply(lambda s: pd.Series(s).rank(pct=True).iloc[-1], raw=False)
    mask_vcp_break = (
        (atr_rank_60 <= 0.20)
        & (df["total_range"] >= 1.5 * df["atr14"])
        & (df["close"] > df["open"])
        & (df["close"] > df["ma20"])
    )

    # 结构 6: 均线顺势首次回踩 MA20 企稳 (First Pullback to MA20 in Bull Trend)
    # 顺大势 (C > MA60, MA20 > MA60)，此前至少5天在MA20之上，今日盘中触及或微跌破MA20但收盘重新站上，收出底分型阳线
    was_above_ma20 = (
        (df["close"].shift(1) > df["ma20"].shift(1))
        & (df["close"].shift(2) > df["ma20"].shift(2))
        & (df["close"].shift(3) > df["ma20"].shift(3))
    )
    touch_ma20 = (df["low"] <= df["ma20"] * 1.005) & (df["close"] >= df["ma20"])
    bull_engulf = (df["close"] > df["open"]) & (df["close"] > df["high"].shift(1))
    mask_pullback_ma20 = (
        (df["close"] > df["ma60"]) & (df["ma20"] > df["ma60"]) & was_above_ma20 & touch_ma20 & bull_engulf
    )

    # 结构 7: TD 序列买入结构 9 (TD Buy Setup 9)
    # 连续 9 根 K 线收盘价低于 4 根前收盘价
    td_buy = (df["close"] < df["close"].shift(4)).astype(int)
    td_buy_count = td_buy.groupby((~td_buy.astype(bool)).cumsum()).cumsum()
    mask_td9_buy = td_buy_count == 9

    # 结构 8: 放量跳空突破缺口不补 (Breakaway Gap Long)
    # 今日最低价高于昨日最高价 (未补缺口)，缺口幅度 >= 0.5%，且成交量 >= 1.5倍MA20，且收阳
    has_gap_up = (
        (df["low"] > df["high"].shift(1) * 1.003) & (df["close"] > df["open"]) & (df["volume"] >= 1.3 * df["vol_ma20"])
    )

    # 结构 9: 极度缩量地量十字星后长阳反包 (Volume Star Engulfing Long)
    # 昨日成交量处于近20天极度地量 (成交量 < 0.6 * MA_Vol20 且星线)，今日放量长阳反包
    yesterday_star = (df["volume"].shift(1) <= 0.65 * df["vol_ma20"].shift(1)) & (
        df["body"].shift(1) / df["total_range"].shift(1) <= 0.35
    )
    today_engulf = (
        (df["pct_chg"] >= 0.018) & (df["close"] > df["high"].shift(1)) & (df["volume"] >= 1.5 * df["vol_ma20"])
    )
    mask_star_engulf = yesterday_star & today_engulf

    # 结构 10: 极端长下影锤子线 (Extreme Pinbar Hammer)
    # 下影线占全天振幅 60% 以上，实体在上部 35%，且处于短期回调低位 (C < MA10)
    mask_hammer = (
        (df["lower_shadow"] / df["total_range"] >= 0.60)
        & (df["close"] > df["low"] + 0.6 * df["total_range"])
        & (df["close"] < df["ma10"])
    )

    # 评估所有形态
    patterns_to_test = [
        (df.index >= 60, "基准: 全样本无条件基准 (All Days)", "long"),
        (mask_2b_long, "【结构1】2B 假跌破反转 (2B Fake Breakdown)", "long"),
        (mask_2b_short, "【结构2】2B 假突破反转 (2B Fake Breakout)", "short"),
        (mask_w_bottom, "【结构3】W底次低抬高突破 (Higher Low W-Bottom/二买)", "long"),
        (mask_m_top, "【结构4】M顶次高降低跌破 (Lower High M-Top/二卖)", "short"),
        (mask_vcp_break, "【结构5】VCP 波动率极度压缩爆发", "long"),
        (mask_pullback_ma20, "【结构6】顺势首次回踩 MA20 企稳", "long"),
        (mask_td9_buy, "【结构7】TD 序列买入结构 9 (TD Buy 9)", "long"),
        (has_gap_up, "【结构8】放量跳空突破缺口不补 (Breakaway Gap)", "long"),
        (mask_star_engulf, "【结构9】地量十字星后放量长阳反包", "long"),
        (mask_hammer, "【结构10】短期超跌长下影锤子线 (Pinbar)", "long"),
    ]

    results = []
    for m, name, direct in patterns_to_test:
        eval_res = evaluate_pattern(df, m, name, direct)
        results.append(eval_res)

    res_df = pd.DataFrame(results)

    print("\n" + "=" * 115)
    print("【159915 真实行情 10 大 K 线结构前向预测力实证统计表 (2020-2026)】")
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
    run_pattern_discovery()
