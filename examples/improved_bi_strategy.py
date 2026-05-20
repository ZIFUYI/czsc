#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的缠论笔策略示例

策略逻辑：
1. 使用笔的背驰判断：当笔创新高/新低但力度减弱时，反向开仓
2. 结合中枢：在中枢震荡后的突破方向开仓
3. 动态止损：使用前一笔的极值作为止损位

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


class ImprovedBiStrategy:
    """改进的笔策略"""
    
    def __init__(self, symbol, min_bi_count=5, take_profit_pct=0.08):
        """
        初始化策略
        
        :param symbol: 股票代码
        :param min_bi_count: 最少笔数量要求
        :param take_profit_pct: 止盈比例
        """
        self.symbol = symbol
        self.min_bi_count = min_bi_count
        self.take_profit_pct = take_profit_pct
        
        self.position = 0  # 持仓：1=多头，-1=空头，0=空仓
        self.entry_price = 0  # 开仓价格
        self.stop_loss_price = 0  # 止损价格
        self.trades = []  # 交易记录
        
    def calculate_bi_strength(self, bi):
        """
        计算笔的力度
        
        力度 = 价格变化幅度 / 时间跨度
        """
        if not bi or not bi.fx_a or not bi.fx_b:
            return 0
        
        price_change = abs(bi.fx_b.fx - bi.fx_a.fx)
        time_span = len(bi.fxs)
        
        if time_span == 0:
            return 0
        
        # 归一化力度
        strength = (price_change / bi.fx_a.fx) / time_span * 100
        return strength
    
    def check_divergence(self, czsc):
        """
        检查背驰
        
        背驰定义：价格创新高/新低，但力度减弱
        """
        if len(czsc.bi_list) < 3:
            return 0
        
        last_bi = czsc.bi_list[-1]
        prev_same_direction_bi = None
        
        # 找到前一个同方向的笔
        for bi in reversed(czsc.bi_list[:-1]):
            if bi.direction == last_bi.direction:
                prev_same_direction_bi = bi
                break
        
        if not prev_same_direction_bi:
            return 0
        
        last_strength = self.calculate_bi_strength(last_bi)
        prev_strength = self.calculate_bi_strength(prev_same_direction_bi)
        
        # 向上笔背驰：价格创新高但力度减弱
        if last_bi.direction == Direction.Up:
            if last_bi.fx_b.fx > prev_same_direction_bi.fx_b.fx:
                if last_strength < prev_strength * 0.7:  # 力度减弱30%以上
                    return -1  # 做空信号
        
        # 向下笔背驰：价格创新低但力度减弱
        if last_bi.direction == Direction.Down:
            if last_bi.fx_b.fx < prev_same_direction_bi.fx_b.fx:
                if last_strength < prev_strength * 0.7:
                    return 1  # 做多信号
        
        return 0
    
    def check_trend_following(self, czsc):
        """
        趋势跟随信号
        
        当连续出现同方向的笔时，跟随趋势
        """
        if len(czsc.bi_list) < 4:
            return 0
        
        recent_bis = czsc.bi_list[-4:]
        
        # 统计最近4笔的方向
        up_count = sum(1 for bi in recent_bis if bi.direction == Direction.Up)
        down_count = sum(1 for bi in recent_bis if bi.direction == Direction.Down)
        
        # 计算最近笔的平均力度
        last_bi = czsc.bi_list[-1]
        last_strength = self.calculate_bi_strength(last_bi)
        
        # 强势上涨趋势
        if up_count >= 3 and last_bi.direction == Direction.Up and last_strength > 0.5:
            return 1
        
        # 强势下跌趋势
        if down_count >= 3 and last_bi.direction == Direction.Down and last_strength > 0.5:
            return -1
        
        return 0
    
    def generate_signal(self, czsc):
        """
        生成交易信号
        
        :param czsc: CZSC对象
        :return: 信号 (1=做多, -1=做空, 0=无信号)
        """
        if len(czsc.bi_list) < self.min_bi_count:
            return 0
        
        # 优先检查背驰信号（逆势）
        divergence_signal = self.check_divergence(czsc)
        if divergence_signal != 0:
            return divergence_signal
        
        # 其次检查趋势跟随信号
        trend_signal = self.check_trend_following(czsc)
        if trend_signal != 0:
            return trend_signal
        
        return 0
    
    def update_stop_loss(self, czsc):
        """
        更新止损价格
        
        使用前一笔的极值作为止损位
        """
        if len(czsc.bi_list) < 2:
            return
        
        prev_bi = czsc.bi_list[-2]
        
        if self.position == 1:  # 多头，止损设在前一笔的低点
            if prev_bi.direction == Direction.Down:
                self.stop_loss_price = prev_bi.fx_b.fx
        elif self.position == -1:  # 空头，止损设在前一笔的高点
            if prev_bi.direction == Direction.Up:
                self.stop_loss_price = prev_bi.fx_b.fx
    
    def check_exit_signal(self, current_price):
        """
        检查是否需要平仓
        
        :param current_price: 当前价格
        :return: True=需要平仓, False=继续持有
        """
        if self.position == 0:
            return False, ""
        
        # 计算盈亏比例
        if self.position == 1:  # 多头
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            # 止损：跌破止损位
            if current_price < self.stop_loss_price:
                return True, "止损"
        else:  # 空头
            pnl_pct = (self.entry_price - current_price) / self.entry_price
            # 止损：涨破止损位
            if current_price > self.stop_loss_price:
                return True, "止损"
        
        # 止盈
        if pnl_pct >= self.take_profit_pct:
            return True, "止盈"
        
        return False, ""
    
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
            self.stop_loss_price = 0
        
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
                should_exit, exit_reason = self.check_exit_signal(current_price)
                if should_exit:
                    self.execute_trade(bar.dt, current_price, 0, exit_reason)
                    continue
                
                # 更新止损位
                self.update_stop_loss(czsc)
            
            # 生成交易信号
            if self.position == 0:  # 只在空仓时才开新仓
                signal = self.generate_signal(czsc)
                
                # 执行交易
                if signal != 0:
                    reason = "背驰做多" if signal == 1 else "背驰做空"
                    self.execute_trade(bar.dt, current_price, signal, reason)
                    # 设置初始止损位
                    self.update_stop_loss(czsc)
        
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
        
        # 最大回撤
        cumulative_pnl = closed_trades['pnl'].cumsum()
        max_drawdown = (cumulative_pnl.cummax() - cumulative_pnl).max()
        
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
        print(f"最大回撤: {max_drawdown:.2f} 元")
        
        if win_trades > 0:
            avg_win = closed_trades[closed_trades['pnl'] > 0]['pnl'].mean()
            print(f"平均盈利: {avg_win:.2f} 元")
        
        if lose_trades > 0:
            avg_loss = closed_trades[closed_trades['pnl'] < 0]['pnl'].mean()
            print(f"平均亏损: {avg_loss:.2f} 元")
        
        print(f"{'='*60}\n")
        
        # 显示交易明细
        print("交易明细:")
        print(closed_trades[['datetime', 'price', 'reason', 'pnl', 'pnl_pct']].to_string(index=False))
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
    strategy = ImprovedBiStrategy(
        symbol="000001.XSHE",
        min_bi_count=5,         # 最少5笔
        take_profit_pct=0.08,   # 8%止盈
    )
    
    # 运行回测
    strategy.backtest(bars)
    
    print("\n策略说明:")
    print("1. 背驰信号：当价格创新高/新低但力度减弱时，反向开仓")
    print("2. 趋势跟随：当连续出现同方向的强势笔时，顺势开仓")
    print("3. 动态止损：使用前一笔的极值作为止损位")
    print("4. 固定止盈：达到8%收益时止盈")


if __name__ == "__main__":
    main()
