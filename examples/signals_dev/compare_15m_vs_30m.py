"""15分钟 vs 30分钟 K线级别 CTA 实证对比分析"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/chenqiang/Desktop/czsc")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import czsc  # noqa: E402
from czsc import Freq  # noqa: E402
from examples.signals_dev.backtest_continuous_signal_commodity_cta import (  # noqa: E402
    DEFAULT_MULTIPLIERS,
    DEFAULT_SYMBOLS,
    StrategyConfig,
    compute_monthly_universe,
    compute_symbol_signals,
    load_or_fetch_all_data,
)


def resample_data_map_to_30m(
    data_map_15m: dict[str, pd.DataFrame], multipliers: dict[str, float]
) -> dict[str, pd.DataFrame]:
    """将全市场 15m 数据重采样为 30m 标准 K 线"""
    data_map_30m = {}
    for sym, df_15m in data_map_15m.items():
        mult = multipliers.get(sym, 10.0)
        df = df_15m.copy()
        df["symbol"] = sym
        df["vol"] = df["volume"]
        df["amount"] = df["close"] * df["volume"] * mult

        df_30 = czsc.resample_bars(df, target_freq=Freq.F30, base_freq=Freq.F15, raw_bars=False)
        df_30["volume"] = df_30["vol"]
        data_map_30m[sym] = df_30[["dt", "open", "high", "low", "close", "volume", "amount"]]
    return data_map_30m


def run_single_backtest(
    data_map: dict[str, pd.DataFrame],
    multipliers: dict[str, float],
    config: StrategyConfig,
    is_30m: bool = False,
) -> dict:
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

    combined = pd.concat(all_signals, ignore_index=True)
    combined["ym"] = combined["dt"].dt.to_period("M").astype(str)

    monthly_univ = compute_monthly_universe(data_map, multipliers, config.liquidity_threshold)
    admitted_list = [
        sym in monthly_univ.get(ym, set(data_map.keys()))
        for ym, sym in zip(combined["ym"], combined["symbol"], strict=False)
    ]
    combined["admitted"] = admitted_list

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

    pivot_close = combined.pivot(index="dt", columns="symbol", values="close").ffill()
    bar_ret = pivot_close.pct_change().fillna(0.0)
    daily_dt = combined[["dt", "trading_day"]].drop_duplicates().set_index("dt")["trading_day"]

    unlev_bar_pnl = (pivot_raw_w.shift(1).fillna(0.0) * bar_ret).sum(axis=1)
    unlev_df = pd.DataFrame({"bar_pnl": unlev_bar_pnl, "trading_day": daily_dt.reindex(unlev_bar_pnl.index)})
    unlev_daily_pnl = unlev_df.groupby("trading_day")["bar_pnl"].sum()

    rolling_std = unlev_daily_pnl.rolling(window=120, min_periods=20).std() * np.sqrt(250)
    rolling_mul = (config.target_vol / rolling_std.replace(0.0, np.nan)).fillna(2.0).clip(lower=0.5, upper=4.0)
    dt_mul = daily_dt.map(rolling_mul).fillna(2.0)
    pivot_w = pivot_raw_w.mul(dt_mul, axis=0)

    # 调仓死区与再平衡模式
    rows = []
    last_w: dict[str, float] = {}
    last_pos: dict[str, int] = {}

    for dt, row in pivot_w.iterrows():
        if is_30m:
            is_close_bar = (dt.hour == 14 and dt.minute in (30, 45, 0)) or (dt.hour == 15 and dt.minute == 0)
        else:
            is_close_bar = dt.hour == 14 and dt.minute in (30, 45)
        new_row = {}
        for sym, target_w in row.items():
            prev = last_w.get(sym, 0.0)
            prev_p = last_pos.get(sym, 0)
            cur_p = int(np.sign(target_w))
            eff_band = config.rebalance_band

            if cur_p == 0:
                actual = 0.0
            elif cur_p != prev_p or is_close_bar and abs(target_w - prev) >= eff_band:
                actual = target_w
            else:
                actual = prev
            new_row[sym] = actual
            last_w[sym] = actual
            last_pos[sym] = int(np.sign(actual))
        rows.append(new_row)

    w_df = pd.DataFrame(rows, index=pivot_w.index)
    w_df = w_df[(w_df.index >= config.start) & (w_df.index <= config.end)]

    # 统计交易次数
    prev_w = w_df.shift(1).fillna(0.0)
    diff_w = (w_df - prev_w).abs()
    is_change = diff_w > 1e-4
    is_open = (prev_w.abs() < 1e-4) & (w_df.abs() >= 1e-4)
    is_close = (prev_w.abs() >= 1e-4) & (w_df.abs() < 1e-4)
    is_flip = (prev_w.abs() >= 1e-4) & (w_df.abs() >= 1e-4) & (np.sign(prev_w) != np.sign(w_df))
    is_rebal = is_change & ~is_open & ~is_close & ~is_flip

    n_opens = int(is_open.sum().sum())
    n_closes = int(is_close.sum().sum())
    n_flips = int(is_flip.sum().sum())
    n_rebals = int(is_rebal.sum().sum())
    total_actions = n_opens + n_closes + n_flips + n_rebals
    n_syms = len(w_df.columns)

    # 收益与换手
    delta_w = w_df.diff().abs().fillna(w_df.abs())
    bar_to = delta_w.sum(axis=1)
    bar_pnl = (w_df.shift(1).fillna(0.0) * bar_ret.reindex(w_df.index).fillna(0.0)).sum(axis=1)
    bar_fees = bar_to * config.fee_rate
    net_pnl = bar_pnl - bar_fees

    df_res = pd.DataFrame(
        {
            "net": net_pnl,
            "gross": bar_pnl,
            "to": bar_to,
            "tday": daily_dt.reindex(w_df.index),
        }
    )
    day_res = df_res.groupby("tday").sum()
    n_days = len(day_res)
    c_net = (1.0 + day_res["net"]).cumprod().iloc[-1] - 1.0
    c_gross = (1.0 + day_res["gross"]).cumprod().iloc[-1] - 1.0
    ann_net = (1.0 + c_net) ** (250.0 / n_days) - 1.0
    ann_vol = day_res["net"].std() * np.sqrt(250.0)
    sr = (day_res["net"].mean() / day_res["net"].std()) * np.sqrt(250.0)
    cum = (1.0 + day_res["net"]).cumprod()
    mdd = (cum / cum.cummax() - 1.0).min()
    total_to = day_res["to"].sum()
    avg_hold_days = float(n_days / max(1, (n_opens / n_syms)))

    # 分年统计
    day_res["year"] = pd.to_datetime(day_res.index).year
    yearly = {}
    for yr, sub in day_res.groupby("year"):
        y_net = (1.0 + sub["net"]).cumprod().iloc[-1] - 1.0
        y_sr = (sub["net"].mean() / sub["net"].std()) * np.sqrt(250.0) if sub["net"].std() > 0 else 0
        yearly[int(yr)] = {"net": y_net * 100, "sharpe": y_sr}

    return {
        "c_net": c_net * 100,
        "c_gross": c_gross * 100,
        "ann_net": ann_net * 100,
        "ann_vol": ann_vol * 100,
        "sharpe": sr,
        "max_dd": mdd * 100,
        "total_turnover": total_to,
        "daily_turnover": total_to / n_days,
        "total_actions": total_actions,
        "daily_actions": total_actions / n_days,
        "n_opens": n_opens,
        "avg_hold_days": avg_hold_days,
        "yearly": yearly,
    }


def main():
    print("================================================================================")
    print("  15分钟 vs 30分钟 K线级别 CTA 策略多维度实证测算 (2024-01-01 ~ 2026-09-30)")
    print("================================================================================")

    data_map_15m = load_or_fetch_all_data(list(DEFAULT_SYMBOLS), StrategyConfig())
    print("\n正在将全市场 56 个品种 15m K线重采样为 30m K线...")
    data_map_30m = resample_data_map_to_30m(data_map_15m, DEFAULT_MULTIPLIERS)
    print("重采样完成！")

    configs = [
        (
            "15m 当前最优基准 (进0.30/出0.05, 锁15b)",
            data_map_15m,
            StrategyConfig(start="2024-01-01", end="2026-09-30", entry_u2p=0.30, exit_u2p=0.05, min_hold_bars=15),
            False,
        ),
        (
            "30m 直接平移 (进0.30/出0.05, 锁8b~半天)",
            data_map_30m,
            StrategyConfig(start="2024-01-01", end="2026-09-30", entry_u2p=0.30, exit_u2p=0.05, min_hold_bars=8),
            True,
        ),
        (
            "30m 锁定15b (进0.30/出0.05, 锁15b~1天)",
            data_map_30m,
            StrategyConfig(start="2024-01-01", end="2026-09-30", entry_u2p=0.30, exit_u2p=0.05, min_hold_bars=15),
            True,
        ),
        (
            "30m 适度放宽 (进0.25/出0.10, 锁8b~半天)",
            data_map_30m,
            StrategyConfig(start="2024-01-01", end="2026-09-30", entry_u2p=0.25, exit_u2p=0.10, min_hold_bars=8),
            True,
        ),
    ]

    results = []
    for label, dmap, cfg, is_30m in configs:
        print(f"\n>>> 正在运行测算: {label} ...")
        res = run_single_backtest(dmap, DEFAULT_MULTIPLIERS, cfg, is_30m=is_30m)
        res["name"] = label
        results.append(res)

    print("\n=============================== 15m vs 30m 综合测算对比表 ===============================")
    rows = []
    for r in results:
        rows.append(
            {
                "方案": r["name"],
                "累计净收益": f"{r['c_net']:+.2f}%",
                "年化净收益": f"{r['ann_net']:+.2f}%",
                "夏普比率": f"{r['sharpe']:.2f}",
                "最大回撤": f"{r['max_dd']:.2f}%",
                "总换手率": f"{r['total_turnover']:.1f}",
                "日均换手": f"{r['daily_turnover']:.2f}",
                "下单次数": f"{r['total_actions']:,}",
                "平均持仓天数": f"{r['avg_hold_days']:.1f}天",
                "2024收益": f"{r['yearly'][2024]['net']:+.1f}%",
                "2025收益": f"{r['yearly'][2025]['net']:+.1f}%",
                "2026收益": f"{r['yearly'][2026]['net']:+.1f}%",
            }
        )
    df_comp = pd.DataFrame(rows)
    print(df_comp.to_string(index=False))
    print("==========================================================================================")


if __name__ == "__main__":
    main()
