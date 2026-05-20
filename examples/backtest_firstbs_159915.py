# -*- coding: utf-8 -*-
"""
一买/一卖信号回测脚本
- 标的：159915（创业板ETF，深交所）
- 频率：5分钟
- 数据源：聚宽 JQData（HTTP API，支持任意日期范围）
- 信号：cxt_first_buy_V221126（缠论结构一买）/ cxt_first_sell_V221126（缠论结构一卖）
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 解决 lz4/clickhouse 在 Python 3.14 下的兼容问题
from unittest.mock import MagicMock
for mod in [
    'clickhouse_connect',
    'clickhouse_connect.driver',
    'clickhouse_connect.driver.client',
    'clickhouse_connect.driver.httpclient',
    'clickhouse_connect.driver.compression',
]:
    sys.modules[mod] = MagicMock()

import pandas as pd
from rs_czsc import Event, Position
from czsc.strategies import CzscStrategyBase
from dotenv import load_dotenv
from pathlib import Path
from loguru import logger

# ----- 1. 加载环境变量（聚宽账号密码）-----
load_dotenv(Path(__file__).parent.parent / '.env')
JQ_USER = os.getenv('JQDATA_USERNAME')
JQ_PASS = os.getenv('JQDATA_PASSWORD')

# 初始化聚宽 token（保存到 ~/jq.token）
from czsc.connectors.jq_connector import set_token
set_token(JQ_USER, JQ_PASS)
logger.info(f"聚宽账号：{JQ_USER}，token 已初始化")

# ----- 2. 定义回测结果输出目录 -----
results_path = Path(__file__).parent / 'results' / 'firstbs_159915_5m_jq'
results_path.mkdir(parents=True, exist_ok=True)
logger.add(results_path / "backtest.log", rotation="1 week", encoding="utf-8")



# ----- 3. 定义一买/一卖持仓策略工厂函数 -----

def create_first_buy_sell(symbol, base_freq='5分钟'):
    """
    一买开多 + 一卖平多/开空 的持仓策略

    信号说明：
    - cxt_first_buy_V221126:   基于缠论多笔结构背驰的一买信号（向下结构末端）
    - cxt_first_sell_V221126:  基于缠论多笔结构背驰的一卖信号（向上结构末端）

    信号Key格式：
    - 一买：'{freq}_D1B_BUY1_一买_{N}笔_任意_0'
    - 一卖：'{freq}_D1B_SELL1_一卖_{N}笔_任意_0'
    """
    f = base_freq

    # ---- 开仓条件 ----
    opens = [
        {
            "operate": "开多",
            "signals_all": [],
            "signals_any": [
                # 只要出现任意笔数的一买信号即可触发开多
                f"{f}_D1B_BUY1_一买_5笔_任意_0",
                f"{f}_D1B_BUY1_一买_7笔_任意_0",
                f"{f}_D1B_BUY1_一买_9笔_任意_0",
                f"{f}_D1B_BUY1_一买_11笔_任意_0",
                f"{f}_D1B_BUY1_一买_13笔_任意_0",
            ],
            "signals_not": [],
            "factors": [],
        },
    ]

    # ---- 平仓条件 ----
    exits = [
        {
            "operate": "平多",
            "signals_all": [],
            "signals_any": [
                # 出现任意笔数的一卖信号即平多仓
                f"{f}_D1B_SELL1_一卖_5笔_任意_0",
                f"{f}_D1B_SELL1_一卖_7笔_任意_0",
                f"{f}_D1B_SELL1_一卖_9笔_任意_0",
                f"{f}_D1B_SELL1_一卖_11笔_任意_0",
                f"{f}_D1B_SELL1_一卖_13笔_任意_0",
            ],
            "signals_not": [],
            "factors": [],
        },
    ]

    pos = Position(
        name=f"{f}一买一卖策略",
        symbol=symbol,
        opens=[Event.load(x) for x in opens],
        exits=[Event.load(x) for x in exits],
        interval=0,    # 不强制最短持仓时间
        timeout=10,    # 最多持有 10 根K线（5分钟×10 = 约50分钟）
        stop_loss=300, # 止损 3%（300 BP）
    )
    return pos


# ----- 4. 封装策略类 -----
class FirstBuySellStrategy(CzscStrategyBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def positions(self):
        return [
            create_first_buy_sell(self.symbol, base_freq='5分钟'),
        ]


# ----- 5. 定义读取 K 线的函数（聚宽数据源）-----
def read_bars(symbol, freq, sdt, edt, fq='前复权'):
    """通过聚宽 HTTP API 拉取 K 线，返回标准 RawBar 列表

    :param symbol: 聚宽标的码，如 '159915.XSHE'
    :param freq: 中文周期，如 '5分钟'
    :param sdt: 开始日期，如 '20200101'
    :param edt: 结束日期，如 '20260222'
    :param fq: 复权类型，'前复权' 或 '不复权'
    """
    from czsc.connectors.jq_connector import get_kline_period, freq_cn2jq
    jq_freq = freq_cn2jq[freq]
    fq_flag = (fq == '前复权')
    logger.info(f"拉取 {symbol} {freq} K线（聚宽），时间范围：{sdt} ~ {edt}")
    bars = get_kline_period(symbol, start_date=sdt, end_date=edt, freq=jq_freq, fq=fq_flag)
    logger.info(f"拉取完成，共 {len(bars)} 根K线，范围：{bars[0].dt} ~ {bars[-1].dt}")
    return bars


# ----- 6. 主程序：拉数据 → Replay 回测 -----
if __name__ == '__main__':
    # 聚宽深交所 ETF 代号格式：159915.XSHE
    symbol = '159915.XSHE'  # 创业板 ETF

    tactic = FirstBuySellStrategy(symbol=symbol)
    logger.info(f"策略名称：{tactic.__class__.__name__}")
    logger.info(f"基础K线周期：{tactic.base_freq}")
    logger.info(f"信号配置列表：{tactic.signals_config}")

    # 拉取 5 分钟 K 线（聚宽支持任意日期范围，无根数限制）
    bars = read_bars(symbol, freq='5分钟', sdt='20200101', edt='20260222')

    if not bars:
        logger.error("K线数据为空，请检查账号或网络连接")
        sys.exit(1)

    logger.info(f"K线数量：{len(bars)}，日期范围：{bars[0].dt} ~ {bars[-1].dt}")

    # 使用 init_trader 执行回测（不依赖 Position.evaluate()，规避 rs_czsc 兼容问题）
    # sdt 设置为第一根K线日期，让所有K线都参与信号生成
    trader = tactic.init_trader(bars, sdt=str(bars[0].dt.date()))

    logger.info("回测完成！")
    total_trades = sum(len(p.pairs) for p in trader.positions)
    logger.info(f"共触发 {total_trades} 次完整交易（开仓+平仓配对）")

    # 打印每个持仓策略的统计摘要
    for pos in trader.positions:
        pairs = pos.pairs
        if pairs:
            df_pairs = pd.DataFrame(pairs)
            logger.info(f"pairs 字段：{list(df_pairs.columns)}")   # 调试：打印实际字段名
            # rs_czsc Rust 版：字段名为中文
            profit_col = '盈亏比例' if '盈亏比例' in df_pairs.columns else 'profit_rate'
            open_col = '开仓时间' if '开仓时间' in df_pairs.columns else 'open_dt'
            close_col = '平仓时间' if '平仓时间' in df_pairs.columns else 'close_dt'

            win_rate = (df_pairs[profit_col] > 0).mean()
            avg_profit = df_pairs[profit_col].mean()
            total_profit = df_pairs[profit_col].sum()
            logger.info(
                f"\n{'='*50}\n"
                f"策略：{pos.name}\n"
                f"  交易次数：{len(df_pairs)}\n"
                f"  胜率：{win_rate:.2%}\n"
                f"  平均单笔收益（BP）：{avg_profit:.1f}\n"
                f"  总收益（BP）：{total_profit:.1f}\n"
                f"  最近5笔：\n{df_pairs[[open_col, close_col, profit_col]].tail().to_string(index=False)}\n"
                f"{'='*50}"
            )

            # 保存完整交易记录到 CSV
            out_file = results_path / f"{pos.name}_pairs.csv"
            df_pairs.to_csv(out_file, index=False, encoding='utf-8-sig')
            logger.info(f"完整交易记录已保存到：{out_file}")
        else:
            logger.warning(f"策略 {pos.name} 没有触发任何交易，请检查信号配置")
