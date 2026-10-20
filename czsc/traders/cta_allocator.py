"""CTA 连续信号与仓位调度引擎（Continuous CTA Position Allocator）。

实现国信证券金工 CTA 研报及连续信号体系中的五大核心仓位支柱：
1. 概率状态机与非对称迟滞带（ContinuousSignalStateMachine）：动量强度映射到概率空间，进出场阈值解耦；
2. 动态活跃池等分（1 / N_active）：消除静态等权资金闲置，资金向活跃趋势倾斜；
3. 单品种 ATR 风险平价杠杆（ATR Risk Parity）：高波动品种自动降权，低波动品种适度放大；
4. 调仓死区缓冲与定点再平衡（Deadband & Rebalance Filter）：压降 80% 无谓高频换手；
5. 组合级目标波动率平滑（Target Volatility）：自适应调节整体名义杠杆，平滑资金曲线。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CtaAllocatorConfig:
    """CTA 连续信号与仓位分配器配置参数。

    :param fee_rate: 单边费率（含交易手续费与滑点），默认万分之一 (0.0001)
    :param rebalance_band: 调仓死区缓冲绝对阈值，默认 2.5% (0.025)
    :param min_hold_bars: 最小持仓 K 线根数，用于过滤短命假突破，默认 15 根
    :param entry_u2p: 进场概率差阈值 |U2P|，默认 0.25
    :param exit_u2p: 出场概率差中性阈值 |U2P|，默认 0.05（宽幅迟滞防洗盘）
    :param prob_decay: 无动量推动时的自然概率衰减率，默认 0.05
    :param max_atr_mult: 单品种 ATR 杠杆放大上限，默认 2.5 倍
    :param target_risk_pct: ATR 风险平价基准风险敞口比例，默认 0.5% (0.005)
    :param enable_target_vol: 是否启用宏观组合目标波动率自适应平滑，默认 True
    :param target_vol: 目标年化波动率，默认 15% (0.15)
    :param vol_window: 波动率滚动跟踪窗口（交易日数），默认 120 天
    :param vol_min_periods: 波动率计算最小样本数，默认 20 天
    :param vol_bounds: 波动率杠杆乘数上下限约束，默认 (0.5, 4.0)
    """

    fee_rate: float = 0.0001
    rebalance_band: float = 0.025
    min_hold_bars: int = 15
    entry_u2p: float = 0.25
    exit_u2p: float = 0.05
    prob_decay: float = 0.05
    max_atr_mult: float = 2.5
    target_risk_pct: float = 0.005
    enable_target_vol: bool = True
    target_vol: float = 0.15
    vol_window: int = 120
    vol_min_periods: int = 20
    vol_bounds: tuple[float, float] = (0.5, 4.0)


class ContinuousSignalStateMachine:
    """连续信号概率演化状态机。

    将离散的技术指标增强触发（long_trig / short_trig）转化为连续的上涨/下跌概率，
    并通过非对称迟滞带（Hysteresis Band）和最小持仓锁定（min_hold_bars）输出连续交易信号。
    """

    def __init__(
        self,
        entry_u2p: float = 0.25,
        exit_u2p: float = 0.05,
        prob_decay: float = 0.05,
        min_hold_bars: int = 15,
    ) -> None:
        self.entry_u2p = entry_u2p
        self.exit_u2p = exit_u2p
        self.prob_decay = prob_decay
        self.min_hold_bars = min_hold_bars

        # 内部演化状态
        self.cur_up: float = 0.5
        self.cur_down: float = 0.5
        self.cur_pos: int = 0  # 0: 空仓, 1: 多头, -1: 空头
        self.bars_held: int = 0

    def reset(self) -> None:
        """重置状态机为初始中性状态。"""
        self.cur_up = 0.5
        self.cur_down = 0.5
        self.cur_pos = 0
        self.bars_held = 0

    def update(
        self,
        long_trig: bool,
        short_trig: bool,
        filter_pass: bool = True,
    ) -> tuple[float, int, float]:
        """更新单个时步的状态并输出信号。

        :param long_trig: 是否触发多头动量增强
        :param short_trig: 是否触发空头动量增强
        :param filter_pass: 过滤准入门禁（如波动率门禁）是否通过
        :return: (signal, position, u2p)
            - signal: 连续信号强度 [-1.0, 1.0]
            - position: 离散持仓状态 {-1, 0, 1}
            - u2p: 概率偏离差 (cur_up - cur_down)
        """
        # 1. 概率状态平滑演化
        if long_trig:
            self.cur_up = self.cur_up + 0.5 * (1.0 - self.cur_up)
            self.cur_down = 1.0 - self.cur_up
        elif short_trig:
            self.cur_down = self.cur_down + 0.5 * (1.0 - self.cur_down)
            self.cur_up = 1.0 - self.cur_down
        else:
            # 无持续单边驱动时，向 0.5 中性基准自然平滑衰减
            self.cur_up = self.cur_up + self.prob_decay * (0.5 - self.cur_up)
            self.cur_down = 1.0 - self.cur_up

        u2p = self.cur_up - self.cur_down

        # 2. 开平仓分离与迟滞带判断
        entry_long_ok = long_trig and filter_pass and (u2p > self.entry_u2p)
        entry_short_ok = short_trig and filter_pass and (u2p < -self.entry_u2p)

        if self.cur_pos == 0:
            if entry_long_ok:
                self.cur_pos = 1
                sig = self.cur_up
                self.bars_held = 0
            elif entry_short_ok:
                self.cur_pos = -1
                sig = -self.cur_down
                self.bars_held = 0
            else:
                sig = 0.0
        elif self.cur_pos == 1:
            self.bars_held += 1
            if abs(u2p) <= self.exit_u2p:
                if self.bars_held >= self.min_hold_bars:
                    self.cur_pos = 0
                    sig = 0.0
                else:
                    sig = self.cur_up
            elif entry_short_ok and self.bars_held >= self.min_hold_bars:
                self.cur_pos = -1
                sig = -self.cur_down
                self.bars_held = 0
            else:
                sig = self.cur_up
        elif self.cur_pos == -1:
            self.bars_held += 1
            if abs(u2p) <= self.exit_u2p:
                if self.bars_held >= self.min_hold_bars:
                    self.cur_pos = 0
                    sig = 0.0
                else:
                    sig = -self.cur_down
            elif entry_long_ok and self.bars_held >= self.min_hold_bars:
                self.cur_pos = 1
                sig = self.cur_up
                self.bars_held = 0
            else:
                sig = -self.cur_down
        else:
            sig = 0.0

        return sig, self.cur_pos, u2p

    def batch_process(
        self,
        df: pd.DataFrame,
        long_col: str = "long_trig",
        short_col: str = "short_trig",
        filter_col: str | None = None,
    ) -> pd.DataFrame:
        """批量处理单品种的时序数据。

        :param df: 包含触发列的数据帧
        :param long_col: 多头触发布尔列名
        :param short_col: 空头触发布尔列名
        :param filter_col: 过滤条件布尔列名（为 None 时默认全部通过）
        :return: 扩展了 ['up_prob', 'down_prob', 'u2p', 'signal', 'pos'] 的数据帧
        """
        self.reset()
        res = df.copy()
        long_arr = res[long_col].to_numpy()
        short_arr = res[short_col].to_numpy()
        filter_arr = (
            res[filter_col].to_numpy() if filter_col and filter_col in res.columns else np.ones(len(res), dtype=bool)
        )

        signals = np.zeros(len(res))
        positions = np.zeros(len(res), dtype=int)
        u2ps = np.zeros(len(res))
        up_probs = np.zeros(len(res))
        down_probs = np.zeros(len(res))

        for i in range(len(res)):
            sig, pos, u2p_val = self.update(bool(long_arr[i]), bool(short_arr[i]), bool(filter_arr[i]))
            signals[i] = sig
            positions[i] = pos
            u2ps[i] = u2p_val
            up_probs[i] = self.cur_up
            down_probs[i] = self.cur_down

        res["up_prob"] = up_probs
        res["down_prob"] = down_probs
        res["u2p"] = u2ps
        res["signal"] = signals
        res["pos"] = positions
        return res


class ContinuousCtaAllocator:
    """CTA 连续仓位分配与执行引擎。

    包含 ATR 风险平价杠杆、动态活跃池等分、目标波动率平滑、调仓死区缓冲和逐 Bar 回测结算。
    """

    def __init__(self, config: CtaAllocatorConfig | None = None) -> None:
        self.config = config or CtaAllocatorConfig()

    @staticmethod
    def compute_atr_multipliers(
        df: pd.DataFrame,
        atr_window: int = 10,
        max_mult: float = 2.5,
        target_risk_pct: float = 0.005,
    ) -> tuple[pd.Series, pd.Series]:
        """计算单个品种的 10 周期 ATR 与风险平价杠杆倍率。

        :param df: 必须包含 ['high', 'low', 'close'] 列的数据帧
        :param atr_window: ATR 平滑周期，默认 10
        :param max_mult: 杠杆上限，默认 2.5
        :param target_risk_pct: 风险预算比例，默认 0.005 (0.5%)
        :return: (atr_series, atr_multipliers_series)
        """
        close_s = df["close"]
        high_s = df["high"]
        low_s = df["low"]
        prev_c = close_s.shift(1).bfill()
        tr = np.maximum(high_s - low_s, np.maximum((high_s - prev_c).abs(), (low_s - prev_c).abs()))
        atr = tr.rolling(atr_window, min_periods=max(3, atr_window // 3)).mean().bfill()

        # 逆波动率杠杆公式: ATR_mult = clip(target_risk_pct * Close / ATR, 0.5, max_mult)
        raw_mult = target_risk_pct * close_s / np.maximum(atr, 1e-6)
        atr_mult = raw_mult.clip(lower=0.5, upper=max_mult)
        return atr, atr_mult

    @staticmethod
    def allocate_active_weights(
        signals: pd.DataFrame,
        atr_multipliers: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """截面动态活跃池等分权重计算 (1 / N_active * Signal * ATR_mult)。

        :param signals: 截面信号矩阵，index 为时间，columns 为品种代码，值为 [-1, 1]
        :param atr_multipliers: 截面 ATR 杠杆矩阵，与 signals 同形状；若为 None 则默认 1.0
        :return: 截面基础目标权重矩阵 base_weights
        """
        # 判断活跃状态
        is_active = (signals.abs() > 1e-5).astype(float)
        n_active = is_active.sum(axis=1)

        # 基础等分乘数: 1 / N_active
        inv_n = np.where(n_active > 0, 1.0 / n_active, 0.0)

        # 结合信号与 ATR 杠杆
        raw_w = signals * atr_multipliers if atr_multipliers is not None else signals
        base_weights = raw_w.mul(inv_n, axis=0).fillna(0.0)
        return base_weights

    def apply_target_volatility(
        self,
        base_weights: pd.DataFrame,
        closes: pd.DataFrame,
        trading_days: pd.Series | None = None,
    ) -> pd.DataFrame:
        """根据组合近 120 日已实现波动率，按 15% 目标波动率动态自适应缩放全组合。

        :param base_weights: 基础权重矩阵 (index 为 Bar 时间 dt)
        :param closes: 价格矩阵 (index 为 Bar 时间 dt，对齐 base_weights)
        :param trading_days: 对应每个 Bar 的归属交易日序列 (Series，index 为 dt)
        :return: 施加目标波动率调整后的权重矩阵
        """
        if not self.config.enable_target_vol:
            return base_weights

        bar_returns = closes.pct_change().fillna(0.0)
        unlev_bar_pnl = (base_weights.shift(1).fillna(0.0) * bar_returns).sum(axis=1)

        if trading_days is None:
            # 默认按自然日期聚合
            t_day_series = pd.Series(base_weights.index.date.astype(str), index=base_weights.index)
        else:
            t_day_series = trading_days

        pnl_df = pd.DataFrame({"bar_pnl": unlev_bar_pnl, "t_day": t_day_series})
        daily_pnl = pnl_df.groupby("t_day")["bar_pnl"].sum()

        rolling_std = daily_pnl.rolling(
            window=self.config.vol_window,
            min_periods=self.config.vol_min_periods,
        ).std() * np.sqrt(250)

        low_b, high_b = self.config.vol_bounds
        rolling_mul = (
            (self.config.target_vol / rolling_std.replace(0.0, np.nan)).fillna(2.0).clip(lower=low_b, upper=high_b)
        )

        dt_mul = t_day_series.map(rolling_mul).fillna(2.0)
        scaled_weights = base_weights.mul(dt_mul, axis=0)
        return scaled_weights

    def apply_rebalance_deadband(
        self,
        desired_weights: pd.DataFrame,
        close_bar_mask: pd.Series | None = None,
    ) -> pd.DataFrame:
        r"""施加 2.5% 调仓死区缓冲与日内/收盘执行隔离。

        规则：
        1. 绝对平仓（目标为 0）：立即执行；
        2. 新开仓或反手：立即执行；
        3. 存量持仓微调：仅在收盘时段（close_bar_mask 为 True）且绝对偏离 $|\Delta w| \ge band$ 时才执行；
        4. 其余时间保持原仓位不变，坚决不微调。

        :param desired_weights: 目标期望权重矩阵
        :param close_bar_mask: 布尔序列，标记哪些 Bar 为收盘再平衡时段；若为 None 则默认所有 Bar 均可调仓
        :return: 经过死区过滤后的实际执行权重矩阵
        """
        band = self.config.rebalance_band
        rows: list[dict[str, float]] = []
        last_w: dict[str, float] = {}
        last_pos: dict[str, int] = {}

        for dt, row in desired_weights.iterrows():
            is_close = True if close_bar_mask is None else bool(close_bar_mask.get(dt, False))
            new_row = {}
            for sym, target_w in row.items():
                prev_w = last_w.get(sym, 0.0)
                prev_p = last_pos.get(sym, 0)
                cur_p = int(np.sign(target_w))

                if cur_p == 0:
                    actual = 0.0  # 平仓立即执行
                elif cur_p != prev_p:
                    actual = target_w  # 开仓/反手立即执行
                elif is_close and abs(target_w - prev_w) >= band:
                    actual = target_w  # 存量仓位仅在收盘定点且超死区才调
                else:
                    actual = prev_w  # 保持原仓位，杜绝高频损耗

                new_row[sym] = actual
                last_w[sym] = actual
                last_pos[sym] = cur_p
            rows.append(new_row)

        return pd.DataFrame(rows, index=desired_weights.index)

    def evaluate_backtest(
        self,
        actual_weights: pd.DataFrame,
        closes: pd.DataFrame,
        trading_days: pd.Series | None = None,
    ) -> dict[str, Any]:
        """对实际持仓权重执行逐 Bar 费后回测结算与多维度绩效统计。

        :param actual_weights: 实际执行权重矩阵
        :param closes: 对应时间的价格矩阵
        :param trading_days: 交易日序列
        :return: 包含全期汇总、分年度表现和品种归因的结构化字典
        """
        bar_returns = closes.reindex(actual_weights.index).pct_change().fillna(0.0)
        gross_bar_pnl = (actual_weights.shift(1).fillna(0.0) * bar_returns).sum(axis=1)

        delta_w = actual_weights.diff().abs().fillna(actual_weights.abs())
        bar_turnover = delta_w.sum(axis=1)
        bar_fees = bar_turnover * self.config.fee_rate
        net_bar_pnl = gross_bar_pnl - bar_fees

        if trading_days is None:
            t_day_series = pd.Series(actual_weights.index.date.astype(str), index=actual_weights.index)
        else:
            t_day_series = trading_days.reindex(actual_weights.index)

        df_eval = pd.DataFrame(
            {
                "gross_pnl": gross_bar_pnl,
                "fee": bar_fees,
                "net_pnl": net_bar_pnl,
                "turnover": bar_turnover,
                "trading_day": t_day_series,
                "gross_leverage": actual_weights.abs().sum(axis=1),
                "active_symbols": (actual_weights.abs() > 1e-5).sum(axis=1),
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

        daily_rets = daily_df["net_pnl"].to_numpy()
        nav = np.cumprod(1.0 + daily_rets)
        peak = np.maximum.accumulate(nav)
        dd = (nav - peak) / peak * 100.0

        tot_ret = float((nav[-1] - 1.0) * 100.0)
        ann_ret = float(tot_ret * (252.0 / len(daily_rets))) if len(daily_rets) > 0 else 0.0
        max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0
        std_ret = float(np.std(daily_rets))
        sharpe = float(np.mean(daily_rets) / std_ret * np.sqrt(252)) if std_ret > 1e-8 else 0.0
        calmar = abs(ann_ret / max_dd) if max_dd < 0 else 0.0
        ann_vol = float(std_ret * np.sqrt(252) * 100.0)
        win_rate = float(np.mean(daily_rets > 0) * 100.0)

        # 分年度统计
        daily_df["year"] = pd.to_datetime(daily_df.index).year
        yearly_stats = []
        for y, g in daily_df.groupby("year"):
            g_rets = g["net_pnl"].to_numpy()
            g_nav = np.cumprod(1.0 + g_rets)
            g_peak = np.maximum.accumulate(g_nav)
            g_dd = (g_nav - g_peak) / g_peak * 100.0
            g_tot = float((g_nav[-1] - 1.0) * 100.0)
            g_ann = float(g_tot * (252.0 / len(g_rets))) if len(g_rets) > 0 else 0.0
            g_std = float(np.std(g_rets))
            g_sharpe = float(np.mean(g_rets) / g_std * np.sqrt(252)) if g_std > 1e-8 else 0.0
            yearly_stats.append(
                {
                    "year": int(y),
                    "days": len(g),
                    "net_return": round(g_tot, 2),
                    "ann_return": round(g_ann, 2),
                    "ann_vol": round(float(g_std * np.sqrt(252) * 100.0), 2),
                    "max_dd": round(float(np.min(g_dd)), 2),
                    "sharpe": round(g_sharpe, 2),
                    "turnover": round(float(g["turnover"].sum()), 1),
                    "win_rate": round(float(np.mean(g_rets > 0) * 100.0), 1),
                }
            )

        # 单品种收益贡献
        sym_returns = (actual_weights.shift(1).fillna(0.0) * bar_returns).sum(axis=0)
        attribution = []
        for sym, r_val in sym_returns.items():
            attribution.append(
                {
                    "symbol": str(sym),
                    "contribution_pct": round(float(r_val * 100.0), 2),
                }
            )
        attribution.sort(key=lambda x: x["contribution_pct"], reverse=True)

        return {
            "dates": daily_df.index.tolist(),
            "nav": [round(float(x), 4) for x in nav],
            "dd": [round(float(x), 2) for x in dd],
            "summary": {
                "tot_ret": round(tot_ret, 2),
                "ann_ret": round(ann_ret, 2),
                "ann_vol": round(ann_vol, 2),
                "max_dd": round(max_dd, 2),
                "sharpe": round(sharpe, 2),
                "calmar": round(calmar, 2),
                "win_rate": round(win_rate, 2),
                "total_turnover": round(float(daily_df["turnover"].sum()), 1),
                "avg_leverage": round(float(daily_df["gross_leverage"].mean()), 2),
                "avg_active_symbols": round(float(daily_df["active_symbols"].mean()), 1),
            },
            "yearly_stats": yearly_stats,
            "top_winners": attribution[:10],
            "top_losers": attribution[-5:],
        }
