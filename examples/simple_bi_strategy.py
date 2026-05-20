#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的缠论笔策略示例

策略逻辑：
1. 当出现向上笔且笔的力度较强时，做多
2. 当出现向下笔且笔的力度较强时，做空
3. 使用固定止损和止盈

作者：CZSC
日期：2026-01-08
"""
import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from czsc import CZSC, Direction
from czsc.connectors.jq_connector import set_token, get_kline

# 加载环境变量
load_dotenv()

# 设置聚宽Token
JQDATA_USERNAME = os.getenv('JQDATA_USERNAME')
JQDATA_PASSWORD = os.getenv('JQDATA_PASSWORD')
set_token(jq_mob=JQDATA_USERNAME, jq_pwd=JQDATA_PASSWORD)


class SimpleBiStrategy:
    """简单的笔策略"""
    
    def __init__(self, symbol, min_bi_len=7, stop_loss_pct=0.03, take_profit_pct=0.05):
        """
        初始化策略
        
        :param symbol: 股票代码
        :param min_bi_len: 最小笔长度（K线数量）
        :param stop_loss_pct: 止损比例
        :param take_profit_pct: 止盈比例
        """
        self.symbol = symbol
        self.min_bi_len = min_bi_len
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        self.position = 0  # 持仓：1=多头，-1=空头，0=空仓
        self.entry_price = 0  # 开仓价格
        self.trades = []  # 交易记录
        
    def check_bi_strength(self, bi):
        """
        检查笔的力度
        
        力度定义：笔的价格变化幅度 / 笔的K线数量
        """
        if not bi:
            return 0
        
        price_change = abs(bi.fx_b.fx - bi.fx_a.fx)
        bi_len = len(bi.fxs)
        
        if bi_len == 0:
            return 0
            
        strength = price_change / bi.fx_a.fx  # 相对变化幅度
        return strength
    
    def generate_signal(self, czsc):
        """
        生成交易信号
        
        :param czsc: CZSC对象
        :return: 信号 (1=做多, -1=做空, 0=无信号)
        """
        if len(czsc.bi_list) < 3:
            return 0
        
        # 获取最近的笔
        last_bi = czsc.bi_list[-1]
        prev_bi = czsc.bi_list[-2]
        
        # 检查笔的力度
        last_strength = self.check_bi_strength(last_bi)
        
        # 笔的长度要求
        if len(last_bi.fxs) < self.min_bi_len:
            return 0
        
        # 向上笔且力度较强，做多信号
        if last_bi.direction == Direction.Up and last_strength > 0.03:
            # 确认前一笔是向下的（形成底分型）
            if prev_bi.direction == Direction.Down:
                return 1
        
        # 向下笔且力度较强，做空信号
        if last_bi.direction == Direction.Down and last_strength > 0.03:
            # 确认前一笔是向上的（形成顶分型）
            if prev_bi.direction == Direction.Up:
                return -1
        
        return 0
    
    def check_exit_signal(self, current_price):
        """
        检查是否需要平仓
        
        :param current_price: 当前价格
        :return: True=需要平仓, False=继续持有
        """
        if self.position == 0:
            return False
        
        # 计算盈亏比例
        if self.position == 1:  # 多头
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # 空头
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        # 止损
        if pnl_pct <= -self.stop_loss_pct:
            return True
        
        # 止盈
        if pnl_pct >= self.take_profit_pct:
            return True
        
        return False
    
    def execute_trade(self, dt, price, signal, reason=""):
        """
        执行交易
        
        :param dt: 交易时间
        :param price: 交易价格
        :param signal: 交易信号
        :param reason: 交易原因
        """
        trade = {
            'datetime': dt,
            'price': price,
            'signal': signal,
            'position': self.position,
            'reason': reason,
            'pnl': 0,
            'pnl_pct': 0
        }
        
        # 平仓
        if self.position != 0 and signal == 0:
            if self.position == 1:
                pnl_pct = (price - self.entry_price) / self.entry_price
            else:
                pnl_pct = (self.entry_price - price) / self.entry_price
            
            trade['pnl_pct'] = pnl_pct
            trade['pnl'] = pnl_pct * 10000  # 假设每次交易1万元
            self.position = 0
            self.entry_price = 0
        
        # 开仓
        elif self.position == 0 and signal != 0:
            self.position = signal
            self.entry_price = price
        
        self.trades.append(trade)
    
    def backtest(self, bars):
        """
        回测策略
        
        :param bars: K线数据列表
        """
        print(f"\n{'='*60}")
        print(f"开始回测: {self.symbol}")
        print(f"回测区间: {bars[0].dt} 至 {bars[-1].dt}")
        print(f"K线数量: {len(bars)}")
        print(f"{'='*60}\n")
        
        # 创建CZSC对象
        czsc = CZSC(bars[:100])  # 先用前100根K线初始化
        
        # 逐根K线更新
        for bar in bars[100:]:
            czsc.update(bar)
            
            current_price = bar.close
            
            # 检查是否需要平仓
            if self.position != 0:
                if self.check_exit_signal(current_price):
                    reason = "止损" if (current_price - self.entry_price) / self.entry_price * self.position < 0 else "止盈"
                    self.execute_trade(bar.dt, current_price, 0, reason)
                    continue
            
            # 生成交易信号
            signal = self.generate_signal(czsc)
            
            # 执行交易
            if signal != 0 and self.position == 0:
                reason = "做多信号" if signal == 1 else "做空信号"
                self.execute_trade(bar.dt, current_price, signal, reason)
        
        # 如果最后还有持仓，强制平仓
        if self.position != 0:
            self.execute_trade(bars[-1].dt, bars[-1].close, 0, "强制平仓")
        
        # 输出回测结果
        self.print_results()
    
    def print_results(self):
        """打印回测结果"""
        if not self.trades:
            print("没有交易记录")
            return
        
        df = pd.DataFrame(self.trades)
        
        # 只统计平仓交易
        closed_trades = df[df['signal'] == 0]
        
        if len(closed_trades) == 0:
            print("没有完成的交易")
            return
        
        total_trades = len(closed_trades)
        win_trades = len(closed_trades[closed_trades['pnl'] > 0])
        lose_trades = len(closed_trades[closed_trades['pnl'] < 0])
        
        total_pnl = closed_trades['pnl'].sum()
        total_pnl_pct = closed_trades['pnl_pct'].sum()
        
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"回测结果统计")
        print(f"{'='*60}")
        print(f"总交易次数: {total_trades}")
        print(f"盈利次数: {win_trades}")
        print(f"亏损次数: {lose_trades}")
        print(f"胜率: {win_rate:.2%}")
        print(f"总收益: {total_pnl:.2f} 元")
        print(f"总收益率: {total_pnl_pct:.2%}")
        print(f"平均每笔收益: {avg_pnl:.2f} 元")
        print(f"{'='*60}\n")
        
        # 显示交易明细
        print("交易明细（最近10笔）:")
        print(closed_trades[['datetime', 'price', 'reason', 'pnl', 'pnl_pct']].tail(10).to_string(index=False))
        print()


def main():
    """主函数"""
    # 获取数据
    print("正在获取数据...")
    bars = get_kline(
        symbol="000001.XSHE",  # 平安银行
        freq="D",
        start_date="2023-01-01",
        end_date="2024-12-31",
    )
    print(f"获取到 {len(bars)} 条K线数据\n")
    
    # 创建策略
    strategy = SimpleBiStrategy(
        symbol="000001.XSHE",
        min_bi_len=7,           # 最小笔长度
        stop_loss_pct=0.05,     # 5%止损
        take_profit_pct=0.10,   # 10%止盈
    )
    
    # 运行回测
    strategy.backtest(bars)


if __name__ == "__main__":
    main()
