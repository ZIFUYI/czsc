"""
159915 (创业板ETF) 关键K线与K线结构实证研究 (Event Study)
包含：
1. 大阳线之后是否更容易上涨？（不同幅度、不同均线趋势环境、放量与否、实体比例）
2. 横盘区间突破之后是否容易产生一段趋势？（箱体突破、布林带挤压突破、真突破vs假突破）
3. 5分钟高频微观结构检验（日内动量 vs 均值回归）
"""

import numpy as np
import pandas as pd
from scipy import stats


def load_data():
    df_5m = pd.read_csv("examples/results/159915_real_5m_2020_2026.csv")
    df_5m["datetime"] = pd.to_datetime(df_5m["datetime"])
    df_5m = df_5m.rename(
        columns={
            "SZSE.159915.open": "open",
            "SZSE.159915.high": "high",
            "SZSE.159915.low": "low",
            "SZSE.159915.close": "close",
            "SZSE.159915.volume": "volume",
        }
    )
    df_5m.set_index("datetime", inplace=True)
    daily = (
        df_5m.resample("D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    daily = daily[daily["volume"] > 0].copy()
    daily.reset_index(inplace=True)
    return df_5m, daily


def compute_forward_metrics(df, forward_windows=[1, 2, 3, 5, 10, 20]):
    """计算各周期前向收益率以及 MFE (最大有利偏移) / MAE (最大不利偏移)"""
    for w in forward_windows:
        df[f"fwd_ret_{w}d"] = df["close"].shift(-w) / df["close"] - 1.0

        # MFE: 未来 w 天内的最高价相对当前收盘价的最大涨幅
        # MAE: 未来 w 天内的最低价相对当前收盘价的最大跌幅
        fwd_high = df["high"].iloc[::-1].rolling(w).max().iloc[::-1].shift(-w)
        fwd_low = df["low"].iloc[::-1].rolling(w).min().iloc[::-1].shift(-w)
        df[f"fwd_mfe_{w}d"] = fwd_high / df["close"] - 1.0
        df[f"fwd_mae_{w}d"] = fwd_low / df["close"] - 1.0
    return df


def analyze_event(df, condition_mask, event_name, benchmark_mask=None):
    """
    分析特定事件发生后的前向收益分布
    """
    event_df = df[condition_mask].copy()
    n = len(event_df)
    if n == 0:
        return {"event_name": event_name, "count": 0}

    results = {"event_name": event_name, "count": n}

    for w in [1, 3, 5, 10, 20]:
        col = f"fwd_ret_{w}d"
        valid = event_df[col].dropna()
        if len(valid) == 0:
            continue
        mean_ret = valid.mean() * 100
        median_ret = valid.median() * 100
        win_rate = (valid > 0).mean() * 100
        pos_rets = valid[valid > 0]
        neg_rets = valid[valid < 0]
        avg_gain = pos_rets.mean() * 100 if len(pos_rets) > 0 else 0
        avg_loss = abs(neg_rets.mean() * 100) if len(neg_rets) > 0 else 0
        profit_loss_ratio = (avg_gain / avg_loss) if avg_loss > 0 else np.nan

        mfe_col = f"fwd_mfe_{w}d"
        mae_col = f"fwd_mae_{w}d"
        avg_mfe = event_df[mfe_col].dropna().mean() * 100
        avg_mae = event_df[mae_col].dropna().mean() * 100

        # t-test vs 0
        t_stat, p_val = stats.ttest_1samp(valid, 0.0)

        results[f"{w}d_mean"] = mean_ret
        results[f"{w}d_median"] = median_ret
        results[f"{w}d_win_rate"] = win_rate
        results[f"{w}d_pl_ratio"] = profit_loss_ratio
        results[f"{w}d_mfe"] = avg_mfe
        results[f"{w}d_mae"] = avg_mae
        results[f"{w}d_pval"] = p_val

    return results


def run_study():
    print("================================================================================")
    print("正在加载 159915 真实历史数据 (2020-2026)...")
    df_5m, daily = load_data()
    print(
        f"成功加载日线数据：{len(daily)} 个交易日，时间跨度: {daily['datetime'].min().date()} 至 {daily['datetime'].max().date()}"
    )

    # 计算指标
    daily["pct_chg"] = daily["close"].pct_change()
    daily["body_pct"] = (daily["close"] - daily["open"]) / daily["open"]
    daily["range_pct"] = (daily["high"] - daily["low"]) / daily["open"]
    daily["upper_shadow"] = (daily["high"] - daily[["open", "close"]].max(axis=1)) / daily["open"]
    daily["lower_shadow"] = (daily[["open", "close"]].min(axis=1) - daily["low"]) / daily["open"]
    daily["body_ratio"] = (daily["close"] - daily["open"]).abs() / (daily["high"] - daily["low"] + 1e-6)

    # 均线
    daily["ma5"] = daily["close"].rolling(5).mean()
    daily["ma20"] = daily["close"].rolling(20).mean()
    daily["ma60"] = daily["close"].rolling(60).mean()
    daily["vol_ma20"] = daily["volume"].rolling(20).mean()

    # 布林带
    rolling_std = daily["close"].rolling(20).std()
    daily["boll_mid"] = daily["ma20"]
    daily["boll_up"] = daily["boll_mid"] + 2 * rolling_std
    daily["boll_down"] = daily["boll_mid"] - 2 * rolling_std
    daily["boll_width"] = (daily["boll_up"] - daily["boll_down"]) / daily["boll_mid"]
    # 过去60日布林带宽分位数
    daily["boll_width_rank"] = (
        daily["boll_width"].rolling(60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )

    # 计算横盘震荡指标：过去20日最高价与最低价振幅比率
    daily["high_20"] = daily["high"].rolling(20).max()
    daily["low_20"] = daily["low"].rolling(20).min()
    daily["box_amplitude_20"] = (daily["high_20"] - daily["low_20"]) / daily["low_20"]

    # 计算前向收益
    daily = compute_forward_metrics(daily)

    # 1. 基准表现（无条件基准）
    benchmark_res = analyze_event(daily, daily.index >= 60, "全样本无条件基准 (All Days)")

    # 2. 大阳线专题研究
    # 2.1 不同涨幅的大阳线
    big_bull_2 = analyze_event(daily, (daily["pct_chg"] >= 0.02) & (daily.index >= 60), "大阳线 (涨幅 >= 2.0%)")
    big_bull_3 = analyze_event(daily, (daily["pct_chg"] >= 0.03) & (daily.index >= 60), "特大阳线 (涨幅 >= 3.0%)")
    big_bull_4 = analyze_event(daily, (daily["pct_chg"] >= 0.04) & (daily.index >= 60), "暴涨大阳线 (涨幅 >= 4.0%)")

    # 2.2 大阳线的市场环境分层 (Trend Conditioning)
    bull_trend = (daily["close"] > daily["ma60"]) & (daily.index >= 60)
    bear_trend = (daily["close"] <= daily["ma60"]) & (daily.index >= 60)

    big_bull_in_uptrend = analyze_event(
        daily, (daily["pct_chg"] >= 0.02) & bull_trend, "顺势大阳线 (Close > MA60, 涨幅 >= 2%)"
    )
    big_bull_in_downtrend = analyze_event(
        daily, (daily["pct_chg"] >= 0.02) & bear_trend, "逆势/超跌大阳线 (Close <= MA60, 涨幅 >= 2%)"
    )

    # 2.3 大阳线的量能与形态分层
    vol_surge = daily["volume"] >= 1.8 * daily["vol_ma20"]
    vol_normal = daily["volume"] < 1.8 * daily["vol_ma20"]
    solid_body = daily["body_ratio"] >= 0.70  # 长实体
    shadow_body = daily["upper_shadow"] >= 0.01  # 带长上影线冲高回落

    big_bull_vol_surge = analyze_event(
        daily, (daily["pct_chg"] >= 0.02) & vol_surge & (daily.index >= 60), "放量大阳线 (成交量 >= 1.8倍MA20)"
    )
    big_bull_vol_normal = analyze_event(
        daily, (daily["pct_chg"] >= 0.02) & vol_normal & (daily.index >= 60), "温和量大阳线 (成交量 < 1.8倍MA20)"
    )
    big_bull_solid = analyze_event(
        daily, (daily["pct_chg"] >= 0.02) & solid_body & (daily.index >= 60), "光头/饱满实体大阳线 (实体比 >= 70%)"
    )
    big_bull_shadow = analyze_event(
        daily, (daily["pct_chg"] >= 0.02) & shadow_body & (daily.index >= 60), "冲高回落阳线 (上影线 >= 1%)"
    )

    # 3. 横盘区间突破专题研究
    # 3.1 20日窄幅箱体突破（横盘区间定义：过去20天最高最低振幅 < 8% 或 10%）
    # 当天收盘价突破前20天的最高价 (注意前20天不含当天，shift(1))
    daily["prev_high_20"] = daily["high"].shift(1).rolling(20).max()
    daily["prev_low_20"] = daily["low"].shift(1).rolling(20).min()
    daily["prev_box_amp_20"] = (daily["prev_high_20"] - daily["prev_low_20"]) / daily["prev_low_20"]

    # 箱体振幅 < 9% 的横盘整理
    is_box_tight = daily["prev_box_amp_20"] <= 0.09
    is_box_loose = daily["prev_box_amp_20"] > 0.09
    is_break_up = daily["close"] > daily["prev_high_20"]
    is_break_down = daily["close"] < daily["prev_low_20"]

    box_break_up = analyze_event(
        daily, is_break_up & is_box_tight & (daily.index >= 60), "窄幅横盘向上突破 (20日振幅<=9% + 破前高)"
    )
    box_break_down = analyze_event(
        daily, is_break_down & is_box_tight & (daily.index >= 60), "窄幅横盘向下跌破 (20日振幅<=9% + 破前低)"
    )
    loose_break_up = analyze_event(
        daily, is_break_up & is_box_loose & (daily.index >= 60), "宽幅震荡向上创新高 (20日振幅>9% + 破前高)"
    )

    # 3.2 布林带挤压突破 (Bollinger Band Squeeze Breakout)
    # 布林带处于过去60天最低25%分位数，且当天收盘价上穿布林带上轨
    is_squeeze = daily["boll_width_rank"] <= 0.25
    is_boll_break = daily["close"] > daily["boll_up"]
    boll_squeeze_break = analyze_event(
        daily, is_squeeze & is_boll_break & (daily.index >= 60), "布林带挤压向上突破 (带宽极度收敛+破上轨)"
    )

    # 3.3 突破的成交量特征
    box_break_vol_surge = analyze_event(
        daily, is_break_up & is_box_tight & vol_surge & (daily.index >= 60), "放量窄幅横盘突破 (放量>=1.8倍)"
    )
    box_break_vol_normal = analyze_event(
        daily, is_break_up & is_box_tight & vol_normal & (daily.index >= 60), "平量/缩量横盘突破"
    )

    # 3.4 真突破 vs 假突破（突破后3天的跟踪）：
    # 突破后如果未来3天收盘价没有跌破被突破的阻力位（prev_high_20），记为真突破；反之如果跌破，记为假突破/2B陷阱
    daily["fwd_close_min_3d"] = daily["close"].shift(-1).rolling(3).min().shift(-2)
    is_true_break = (
        (daily["fwd_close_min_3d"] >= daily["prev_high_20"]) & is_break_up & is_box_tight & (daily.index >= 60)
    )
    is_false_break = (
        (daily["fwd_close_min_3d"] < daily["prev_high_20"]) & is_break_up & is_box_tight & (daily.index >= 60)
    )

    true_break_res = analyze_event(daily, is_true_break, "真突破 (突破后3天回踩不破前高)")
    false_break_res = analyze_event(daily, is_false_break, "假突破 (突破后3天跌回原箱体内)")

    # 打印结果表
    summary_list = [
        benchmark_res,
        big_bull_2,
        big_bull_3,
        big_bull_4,
        big_bull_in_uptrend,
        big_bull_in_downtrend,
        big_bull_vol_surge,
        big_bull_vol_normal,
        big_bull_solid,
        big_bull_shadow,
        box_break_up,
        box_break_down,
        loose_break_up,
        boll_squeeze_break,
        box_break_vol_surge,
        box_break_vol_normal,
        true_break_res,
        false_break_res,
    ]

    res_df = pd.DataFrame(summary_list)
    print("\n" + "=" * 90)
    print("【159915 创业板ETF 关键K线与形态结构 事件前向收益统计表 (2020-2026)】")
    print("=" * 90)

    cols_to_show = [
        "event_name",
        "count",
        "1d_mean",
        "1d_win_rate",
        "3d_mean",
        "3d_win_rate",
        "5d_mean",
        "5d_win_rate",
        "5d_pl_ratio",
        "5d_mfe",
        "5d_mae",
        "10d_mean",
        "10d_win_rate",
        "20d_mean",
        "20d_win_rate",
    ]

    formatters = {
        "1d_mean": "{:.2f}%".format,
        "1d_win_rate": "{:.1f}%".format,
        "3d_mean": "{:.2f}%".format,
        "3d_win_rate": "{:.1f}%".format,
        "5d_mean": "{:.2f}%".format,
        "5d_win_rate": "{:.1f}%".format,
        "5d_pl_ratio": "{:.2f}".format,
        "5d_mfe": "{:.2f}%".format,
        "5d_mae": "{:.2f}%".format,
        "10d_mean": "{:.2f}%".format,
        "10d_win_rate": "{:.1f}%".format,
        "20d_mean": "{:.2f}%".format,
        "20d_win_rate": "{:.1f}%".format,
    }

    print(res_df[cols_to_show].to_string(index=False, formatters=formatters))
    print("=" * 90)

    # 4. 统计真突破和假突破的占比
    total_breaks = len(daily[is_break_up & is_box_tight & (daily.index >= 60)])
    true_cnt = len(daily[is_true_break])
    false_cnt = len(daily[is_false_break])
    print("\n【窄幅横盘突破的真实度统计】：")
    print(f"总突破样本: {total_breaks} 次")
    print(f"真突破 (站稳原高点之上): {true_cnt} 次，占比: {true_cnt / total_breaks * 100:.1f}%")
    print(f"假突破 (跌回原高点之下): {false_cnt} 次，占比: {false_cnt / total_breaks * 100:.1f}%")


if __name__ == "__main__":
    run_study()
