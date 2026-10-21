# ruff: noqa: E402
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json

import numpy as np
import pandas as pd

from scripts.cta_execution import futures_daily_bars, next_open_panel_ledger
from scripts.cta_paths import research_data_dir, research_output_path


def load_data_and_compute_slope_signals():
    data_dir = research_data_dir()
    files = sorted(data_dir.glob("*.parquet"))
    print(f"开始加载全市场数据并计算线性回归斜率连续信号，共 {len(files)} 个品种...")

    symbol_data = {}
    for f in files:
        sym = f.stem.replace("_15m", "")
        try:
            df = pd.read_parquet(f)
            df["dt"] = pd.to_datetime(df["dt"]) + pd.Timedelta(minutes=15)
            if len(df) < 300:
                continue
            df["symbol"] = sym
            df["vol"] = df["volume"]
            df["amount"] = df["vol"] * df["close"] * 10
            bars_d = futures_daily_bars(df)
            if len(bars_d) < 100:
                continue

            df_d = pd.DataFrame(
                [
                    {
                        "dt": b.dt.strftime("%Y-%m-%d"),
                        "open": b.open,
                        "close": b.close,
                        "high": b.high,
                        "low": b.low,
                    }
                    for b in bars_d
                ]
            ).set_index("dt")

            closes = df_d["close"].values
            highs = df_d["high"].values
            lows = df_d["low"].values

            # 1. 计算 10 周期 ATR 与 单品种 ATR 杠杆 (对齐连续信号策略: 0.005 * C / ATR)
            prev_c = df_d["close"].shift(1).fillna(df_d["close"].iloc[0]).values
            tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
            atr = pd.Series(tr, index=df_d.index).rolling(10, min_periods=3).mean().values
            l_atr = 0.005 * closes / np.maximum(atr, 0.001)
            atr_mult = np.clip(l_atr, 0.5, 2.5)  # 单品种杠杆上限 2.5
            df_d["atr"] = atr
            df_d["atr_mult"] = atr_mult

            # 2. 计算 20 周期归一化线性回归斜率
            n = 20
            slopes = np.zeros(len(closes))
            for i in range(n - 1, len(closes)):
                y = closes[i - n + 1 : i + 1]
                x = np.arange(n)
                slopes[i] = (np.polyfit(x, y, 1)[0] / closes[i]) * 100.0
            df_d["slope"] = slopes

            # 3. 连续概率状态机 / 连续信号强度映射:
            # 当 slope > 0.05 增强多头概率，slope < -0.05 增强空头概率
            # 采用连续信号策略的概率累积公式:
            cur_up = 0.5
            cur_down = 0.5
            cur_pos = 0
            signals = []
            positions = []
            prob_decay = 0.05

            for s in slopes:
                if s > 0.05:
                    cur_up = cur_up + 0.5 * (1.0 - cur_up)
                    cur_down = 1.0 - cur_up
                elif s < -0.05:
                    cur_down = cur_down + 0.5 * (1.0 - cur_down)
                    cur_up = 1.0 - cur_down
                else:
                    cur_up = cur_up + prob_decay * (0.5 - cur_up)
                    cur_down = 1.0 - cur_down

                u2p = cur_up - cur_down
                # 开平分离迟滞带: 进场门槛 |u2p| > 0.25, 退出门槛 |u2p| <= 0.05
                if cur_pos == 0:
                    if u2p > 0.25:
                        cur_pos = 1
                        sig = cur_up
                    elif u2p < -0.25:
                        cur_pos = -1
                        sig = -cur_down
                    else:
                        sig = 0.0
                elif cur_pos == 1:
                    if u2p <= 0.05:
                        cur_pos = 0
                        sig = 0.0
                    elif u2p < -0.25:
                        cur_pos = -1
                        sig = -cur_down
                    else:
                        sig = cur_up
                elif cur_pos == -1:
                    if u2p >= -0.05:
                        cur_pos = 0
                        sig = 0.0
                    elif u2p > 0.25:
                        cur_pos = 1
                        sig = cur_up
                    else:
                        sig = -cur_down

                signals.append(sig)
                positions.append(cur_pos)

            df_d["signal"] = signals
            df_d["pos"] = positions
            symbol_data[sym] = df_d

        except Exception:
            continue

    print(f"成功处理品种: {len(symbol_data)}")
    return symbol_data


def run_continuous_portfolio(
    symbol_data, enable_target_vol=True, target_vol=0.15, rebalance_band=0.025, fee_rate=0.0001
):
    # 构建所有日期的对齐表
    all_dates = sorted(set.union(*[set(df.index) for df in symbol_data.values()]))
    # 过滤早期不足 20 个品种的日期
    all_dates = [d for d in all_dates if sum([1 for df in symbol_data.values() if d in df.index]) >= 20]
    dates = all_dates[25:]  # 预热后
    print(f"回测评估区间: {len(dates)} 天 ({dates[0]} 至 {dates[-1]})")

    # 1. 计算每个日期的截面活跃品种数与基础名义权重
    # base_weight = (1.0 / n_active) * signal * atr_mult
    raw_weights_df = pd.DataFrame(index=dates, columns=list(symbol_data.keys()))
    close_df = pd.DataFrame(index=dates, columns=list(symbol_data.keys()))

    for d in dates:
        active_cnt = 0
        row_sigs = {}
        row_atr_mult = {}
        for sym, df in symbol_data.items():
            if d in df.index:
                close_df.loc[d, sym] = df.loc[d, "close"]
                sig = df.loc[d, "signal"]
                mult = df.loc[d, "atr_mult"]
                if sig != 0:
                    active_cnt += 1
                row_sigs[sym] = sig
                row_atr_mult[sym] = mult

        for sym in symbol_data:
            sig = row_sigs.get(sym, 0.0)
            mult = row_atr_mult.get(sym, 1.0)
            if active_cnt > 0 and sig != 0:
                raw_weights_df.loc[d, sym] = (1.0 / active_cnt) * sig * mult
            else:
                raw_weights_df.loc[d, sym] = 0.0

    raw_weights_df = raw_weights_df.astype(float).fillna(0.0)
    combined = pd.concat(
        [frame.reset_index().assign(symbol=symbol) for symbol, frame in symbol_data.items()], ignore_index=True
    )
    combined["dt"] = pd.to_datetime(combined.dt) + pd.Timedelta(hours=15)
    execution_index = pd.to_datetime(dates) + pd.Timedelta(hours=15)
    native_targets = raw_weights_df.set_axis(execution_index)

    # 2. 目标波动率平滑机制 (近 120 日已实现年化波动率动态缩放)
    if enable_target_vol:
        unlev_pnl = (
            next_open_panel_ledger(combined, native_targets, 0.0, liquidate=False)["gross"].sum(axis=1).set_axis(dates)
        )
        rolling_std = unlev_pnl.rolling(window=120, min_periods=20).std() * np.sqrt(252)
        rolling_mul = (target_vol / rolling_std.replace(0.0, np.nan)).fillna(2.0)
        rolling_mul = rolling_mul.clip(lower=0.5, upper=4.0)
        adjusted_weights_df = raw_weights_df.mul(rolling_mul, axis=0)
    else:
        adjusted_weights_df = raw_weights_df

    # 3. 施加调仓死区缓冲带 (Rebalance Deadband = 2.5%)
    final_weights = pd.DataFrame(index=dates, columns=adjusted_weights_df.columns)
    last_w = dict.fromkeys(adjusted_weights_df.columns, 0.0)

    for d in dates:
        row = adjusted_weights_df.loc[d]
        for sym in adjusted_weights_df.columns:
            target_w = row[sym]
            prev_w = last_w[sym]
            if target_w == 0.0:
                actual = 0.0
            elif np.sign(target_w) != np.sign(prev_w) and prev_w != 0.0 or abs(target_w - prev_w) >= rebalance_band:
                actual = target_w
            else:
                actual = prev_w
            final_weights.loc[d, sym] = actual
            last_w[sym] = actual

    final_weights = final_weights.astype(float)

    # 4. 损益与费率核算 (扣除单边万分之一)
    ledger = next_open_panel_ledger(combined, final_weights.set_axis(execution_index), fee_rate)
    gross_pnl = ledger["gross"].sum(axis=1).set_axis(dates)
    turnover = ledger["turnover"].sum(axis=1).set_axis(dates)
    fee = ledger["fee"].sum(axis=1).set_axis(dates)
    net_pnl = ledger["net"].sum(axis=1).set_axis(dates)

    nav = 1.0 + np.cumsum(net_pnl.values)
    peak = np.maximum.accumulate(np.maximum(nav, 1.0))
    dd = (nav - peak) / peak * 100.0
    tot_ret = (nav[-1] - 1.0) * 100.0
    ann_ret = tot_ret * (252.0 / len(dates))
    max_dd = float(np.min(dd))
    sharpe = float(np.mean(net_pnl) / np.std(net_pnl) * np.sqrt(252)) if np.std(net_pnl) > 0 else 0
    calmar = abs(ann_ret / max_dd) if max_dd != 0 else 0
    ann_vol = float(np.std(net_pnl) * np.sqrt(252) * 100)

    print("================== 斜率连续信号策略（全套研报仓位架构）绩效 ==================")
    print(f"回测区间: {dates[0]} 至 {dates[-1]} ({len(dates)} 天)")
    print(f"全期累计净收益: {tot_ret:+.2f}% | 年化净收益: {ann_ret:+.2f}%")
    print(f"年化波动率: {ann_vol:.2f}% | 年化夏普比率: {sharpe:.2f}")
    print(f"最大动态回撤: {max_dd:.2f}% | 卡玛比率: {calmar:.2f}")
    print(f"总换手率: {turnover.sum():.1f} 倍 | 日均换手率: {turnover.mean():.2f} 倍")
    print(
        f"平均名义杠杆: {final_weights.abs().sum(axis=1).mean():.2f} 倍 (平均保证金占用约 {final_weights.abs().sum(axis=1).mean() * 10:.1f}%)"
    )

    return {
        "accounting": "fixed_initial_capital_next_open",
        "daily_returns": net_pnl.tolist(),
        "gross_return_pct": float(gross_pnl.sum() * 100),
        "fee_return_pct": float(fee.sum() * 100),
        "dates": dates,
        "nav": [round(float(x), 4) for x in nav],
        "dd": [round(float(x), 2) for x in dd],
        "tot_ret": round(float(tot_ret), 2),
        "ann_ret": round(float(ann_ret), 2),
        "ann_vol": round(float(ann_vol), 2),
        "max_dd": round(float(max_dd), 2),
        "sharpe": round(float(sharpe), 2),
        "calmar": round(float(calmar), 2),
        "avg_leverage": round(float(final_weights.abs().sum(axis=1).mean()), 2),
    }


def main():
    sym_data = load_data_and_compute_slope_signals()
    res = run_continuous_portfolio(sym_data, enable_target_vol=True, target_vol=0.15)

    out_file = research_output_path("slope_with_continuous_sizing_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    print(f"结果已写入: {out_file}")


if __name__ == "__main__":
    main()
