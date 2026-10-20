"""Unit tests for Continuous CTA Position Allocator (czsc.traders.cta_allocator)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from czsc.mock import generate_symbol_kines
from czsc.traders import (
    ContinuousCtaAllocator,
    ContinuousSignalStateMachine,
    CtaAllocatorConfig,
)


def test_cta_allocator_config_defaults():
    config = CtaAllocatorConfig()
    assert config.fee_rate == 0.0001
    assert config.rebalance_band == 0.025
    assert config.min_hold_bars == 15
    assert config.entry_u2p == 0.25
    assert config.exit_u2p == 0.05
    assert config.max_atr_mult == 2.5
    assert config.target_vol == 0.15


def test_continuous_signal_state_machine_transitions():
    sm = ContinuousSignalStateMachine(
        entry_u2p=0.25,
        exit_u2p=0.05,
        prob_decay=0.05,
        min_hold_bars=3,
    )

    # Initial state
    assert sm.cur_pos == 0
    assert sm.cur_up == 0.5
    assert sm.cur_down == 0.5

    # 1. 触发多头增强
    sig, pos, u2p = sm.update(long_trig=True, short_trig=False, filter_pass=True)
    # cur_up = 0.5 + 0.5*(1-0.5) = 0.75, u2p = 0.75 - 0.25 = 0.50 > 0.25
    assert pos == 1
    assert sig == 0.75
    assert u2p == 0.50

    # 2. 中性环境自然衰减，但因 min_hold_bars=3 强制锁仓，不应平仓
    sig2, pos2, u2p2 = sm.update(long_trig=False, short_trig=False, filter_pass=True)
    assert pos2 == 1
    assert u2p2 < 0.50

    sig3, pos3, u2p3 = sm.update(long_trig=False, short_trig=False, filter_pass=True)
    assert pos3 == 1

    # 持续衰减直到 u2p <= exit_u2p (0.05)
    for _ in range(60):
        sig_n, pos_n, u2p_n = sm.update(long_trig=False, short_trig=False, filter_pass=True)
        if pos_n == 0:
            break

    assert pos_n == 0
    assert abs(u2p_n) <= 0.05


def test_state_machine_batch_process():
    df = pd.DataFrame(
        {
            "long_trig": [True, True, False, False, False],
            "short_trig": [False, False, False, True, False],
            "filter_pass": [True, True, True, True, True],
        }
    )
    sm = ContinuousSignalStateMachine(entry_u2p=0.25, exit_u2p=0.05, min_hold_bars=1)
    res = sm.batch_process(df, long_col="long_trig", short_col="short_trig", filter_col="filter_pass")

    expected_cols = ["up_prob", "down_prob", "u2p", "signal", "pos"]
    for col in expected_cols:
        assert col in res.columns
    assert len(res) == len(df)
    assert res["pos"].iloc[0] == 1


def test_atr_multipliers():
    df = pd.DataFrame(
        {
            "close": [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 107.0, 106.0, 108.0],
            "high": [101.0, 103.0, 102.0, 104.0, 106.0, 105.0, 107.0, 108.0, 107.0, 109.0],
            "low": [99.0, 101.0, 100.0, 102.0, 104.0, 103.0, 105.0, 106.0, 105.0, 107.0],
        }
    )
    atr, atr_mult = ContinuousCtaAllocator.compute_atr_multipliers(df, atr_window=5, max_mult=2.5)
    assert len(atr) == len(df)
    assert len(atr_mult) == len(df)
    assert (atr_mult >= 0.5).all()
    assert (atr_mult <= 2.5).all()


def test_active_weight_allocation():
    # 模拟截面 3 个品种，2 个时步
    dates = pd.date_range("2024-01-01", periods=2, freq="15min")
    signals = pd.DataFrame(
        {
            "A": [0.8, 0.0],
            "B": [-0.6, -0.7],
            "C": [0.0, 0.5],
        },
        index=dates,
    )
    atr_mults = pd.DataFrame(
        {
            "A": [1.0, 1.0],
            "B": [1.5, 1.0],
            "C": [1.0, 2.0],
        },
        index=dates,
    )

    weights = ContinuousCtaAllocator.allocate_active_weights(signals, atr_mults)
    # 在第 0 时步：活跃品种为 A 和 B (N=2)
    # A 权重 = 1/2 * 0.8 * 1.0 = 0.4
    # B 权重 = 1/2 * (-0.6) * 1.5 = -0.45
    # C 权重 = 0.0
    assert np.isclose(weights.loc[dates[0], "A"], 0.4)
    assert np.isclose(weights.loc[dates[0], "B"], -0.45)
    assert np.isclose(weights.loc[dates[0], "C"], 0.0)


def test_rebalance_deadband():
    allocator = ContinuousCtaAllocator(CtaAllocatorConfig(rebalance_band=0.03))
    dates = pd.date_range("2024-01-01 14:00", periods=4, freq="15min")
    desired = pd.DataFrame(
        {
            "A": [0.20, 0.22, 0.25, 0.0],  # 0.20开仓 -> 0.22 (变化0.02<0.03死区) -> 0.25 (超死区调仓) -> 0.0立即平仓
        },
        index=dates,
    )
    # 假设所有 bar 均为允许调仓 bar
    close_mask = pd.Series([True, True, True, True], index=dates)

    actual = allocator.apply_rebalance_deadband(desired, close_bar_mask=close_mask)
    assert actual.loc[dates[0], "A"] == 0.20
    assert actual.loc[dates[1], "A"] == 0.20  # 死区缓冲过滤
    assert actual.loc[dates[2], "A"] == 0.25  # 超过死区，调仓
    assert actual.loc[dates[3], "A"] == 0.0  # 平仓立即执行


def test_evaluate_backtest_end_to_end():
    # 使用 czsc.mock 生成两个品种的模拟数据
    df_a = generate_symbol_kines("SYM_A", "15分钟", "20240101", "20240115")
    df_b = generate_symbol_kines("SYM_B", "15分钟", "20240101", "20240115")

    pvt_close = pd.DataFrame(
        {
            "SYM_A": df_a.set_index("dt")["close"],
            "SYM_B": df_b.set_index("dt")["close"],
        }
    ).dropna()

    # 构造假定权重
    pvt_w = pd.DataFrame(0.1, index=pvt_close.index, columns=pvt_close.columns)

    allocator = ContinuousCtaAllocator(CtaAllocatorConfig(fee_rate=0.0001))
    t_days = pd.Series(pvt_close.index.date.astype(str), index=pvt_close.index)

    result = allocator.evaluate_backtest(pvt_w, pvt_close, trading_days=t_days)

    assert "dates" in result
    assert "nav" in result
    assert "summary" in result
    assert "yearly_stats" in result
    assert "top_winners" in result
    assert "top_losers" in result

    summary = result["summary"]
    assert "tot_ret" in summary
    assert "sharpe" in summary
    assert "max_dd" in summary
    assert "calmar" in summary
    assert isinstance(summary["tot_ret"], float)
