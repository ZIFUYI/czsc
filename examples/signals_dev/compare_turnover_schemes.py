"""多种降换手方案横向对比测试

对比 7 种不同方案在 56 个商品期货全市场组合下的表现：
1. Baseline（基准）：15m 连续调仓，Deadband=2.5%，MinHold=4b
2. Scheme 1（收盘定点再平衡）：日内实时开平 + 14:45 收盘定点再平衡
3. Scheme 2（迟滞区间）：非对称进出门槛（进场 |U2P|>0.25，平仓 |U2P|<=0.10）
4. Scheme 3（持仓锁定期）：MinHold=20b（锁定约半个至1个交易日）
5. Scheme 4（相对死区）：RelativeBand=25%（相对变动 25% 且绝对大于 2.5% 才调）
6. Scheme 5（杠杆上限收窄）：单品种 ATR 杠杆上限从 4.0x 压降至 2.0x
7. Scheme 6（推荐综合优化方案）：收盘定点再平衡 + 迟滞带(0.25/0.15) + MinHold=15b + 杠杆上限 2.5x
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.signals_dev.backtest_continuous_signal_commodity_cta import (  # noqa: E402
    DEFAULT_MULTIPLIERS,
    DEFAULT_SYMBOLS,
    StrategyConfig,
    load_or_fetch_all_data,
    run_portfolio_backtest,
)


def main():
    print("================================================================================")
    print("  开始多方案降换手对比测试（全市场 56 个商品期货 15 分钟连续 CTA）")
    print("  测试区间: 2024-01-01 至 2026-09-30 (666个交易日) | 单边费率: 万分之一 (1.0 BP)")
    print("================================================================================")

    data_map = load_or_fetch_all_data(list(DEFAULT_SYMBOLS), StrategyConfig())

    schemes = [
        (
            "基准方案 (15m连续死区2.5%, MinHold=4b)",
            {
                "rebalance_mode": "continuous",
                "rebalance_band": 0.025,
                "relative_band": 0.0,
                "min_hold_bars": 4,
                "entry_u2p": 0.20,
                "exit_u2p": 0.20,
                "max_atr_mult": 4.0,
            },
        ),
        (
            "方案1: 日内实时开平 + 收盘定点再平衡",
            {
                "rebalance_mode": "daily_close",
                "rebalance_band": 0.025,
                "relative_band": 0.0,
                "min_hold_bars": 4,
                "entry_u2p": 0.20,
                "exit_u2p": 0.20,
                "max_atr_mult": 4.0,
            },
        ),
        (
            "方案2: 迟滞区间 (进场0.25 / 平仓0.10)",
            {
                "rebalance_mode": "continuous",
                "rebalance_band": 0.025,
                "relative_band": 0.0,
                "min_hold_bars": 4,
                "entry_u2p": 0.25,
                "exit_u2p": 0.10,
                "max_atr_mult": 4.0,
            },
        ),
        (
            "方案3: 延长最小持仓 (MinHold=20b 约半天)",
            {
                "rebalance_mode": "continuous",
                "rebalance_band": 0.025,
                "relative_band": 0.0,
                "min_hold_bars": 20,
                "entry_u2p": 0.20,
                "exit_u2p": 0.20,
                "max_atr_mult": 4.0,
            },
        ),
        (
            "方案4: 相对死区 (RelativeBand=25%)",
            {
                "rebalance_mode": "continuous",
                "rebalance_band": 0.025,
                "relative_band": 0.25,
                "min_hold_bars": 4,
                "entry_u2p": 0.20,
                "exit_u2p": 0.20,
                "max_atr_mult": 4.0,
            },
        ),
        (
            "方案5: 杠杆上限收紧 (Max ATR Mult=2.0x)",
            {
                "rebalance_mode": "continuous",
                "rebalance_band": 0.025,
                "relative_band": 0.0,
                "min_hold_bars": 4,
                "entry_u2p": 0.20,
                "exit_u2p": 0.20,
                "max_atr_mult": 2.0,
            },
        ),
        (
            "方案6 (推荐综合方案: 组合拳优化)",
            {
                "rebalance_mode": "daily_close",
                "rebalance_band": 0.025,
                "relative_band": 0.0,
                "min_hold_bars": 15,
                "entry_u2p": 0.25,
                "exit_u2p": 0.15,
                "max_atr_mult": 2.5,
            },
        ),
    ]

    results = []
    base_turnover = None

    for name, params in schemes:
        print(f"\n>>> 正在测算: {name} ...")
        cfg = StrategyConfig(
            start="2024-01-01",
            end="2026-09-30",
            exclude_symbols=("DCE.lh", "DCE.jd"),
            **params,
        )
        daily_df, _, stats = run_portfolio_backtest(data_map, DEFAULT_MULTIPLIERS, cfg)

        to = stats["total_turnover"]
        if base_turnover is None:
            base_turnover = to
        to_drop = (1.0 - to / base_turnover) * 100.0

        results.append(
            {
                "方案名称": name,
                "累计毛收益": f"{stats['cumulative_gross_return'] * 100:.2f}%",
                "累计净收益": f"{stats['cumulative_net_return'] * 100:.2f}%",
                "年化净收益": f"{stats['annualized_net_return'] * 100:.2f}%",
                "夏普比率": f"{stats['sharpe_ratio']:.2f}",
                "最大回撤": f"{stats['max_drawdown'] * 100:.2f}%",
                "卡玛比率": f"{stats['calmar_ratio']:.2f}",
                "总换手率": f"{stats['total_turnover']:.1f}",
                "日均换手": f"{stats['avg_daily_turnover']:.2f}",
                "换手降幅": f"{to_drop:+.1f}%" if to_drop != 0 else "--",
            }
        )

    res_df = pd.DataFrame(results)
    print("\n================================ 多方案测试汇总对比表 ================================")
    print(res_df.to_string(index=False))
    print("======================================================================================")

    res_df.to_csv(
        ROOT / "examples" / "results" / "continuous_signal_commodity_cta" / "turnover_schemes_comparison.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
