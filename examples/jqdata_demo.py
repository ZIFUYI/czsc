#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
聚宽数据源使用示例
"""
import os
from dotenv import load_dotenv
from czsc.connectors.jq_connector import set_token, get_kline, get_query_count
from czsc import CZSC, Freq

# 加载环境变量
load_dotenv()

# 设置聚宽Token
JQDATA_USERNAME = os.getenv('JQDATA_USERNAME')
JQDATA_PASSWORD = os.getenv('JQDATA_PASSWORD')
set_token(jq_mob=JQDATA_USERNAME, jq_pwd=JQDATA_PASSWORD)


def example1_get_daily_kline():
    """示例1：获取日线数据"""
    print("\n" + "="*50)
    print("示例1：获取日线数据")
    print("="*50)
    
    bars = get_kline(
        symbol="000001.XSHE",  # 平安银行
        freq="D",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    
    print(f"获取到 {len(bars)} 条日线数据")
    print(f"第一条: {bars[0].dt} - 收盘价: {bars[0].close}")
    print(f"最后一条: {bars[-1].dt} - 收盘价: {bars[-1].close}")
    return bars


def example2_get_minute_kline():
    """示例2：获取分钟线数据"""
    print("\n" + "="*50)
    print("示例2：获取5分钟线数据")
    print("="*50)
    
    bars = get_kline(
        symbol="000001.XSHE",
        freq="5min",
        start_date="2024-12-01",
        end_date="2024-12-31",
    )
    
    print(f"获取到 {len(bars)} 条5分钟线数据")
    if bars:
        print(f"第一条: {bars[0].dt} - 收盘价: {bars[0].close}")
        print(f"最后一条: {bars[-1].dt} - 收盘价: {bars[-1].close}")
    return bars


def example3_czsc_analysis():
    """示例3：使用CZSC进行缠论分析"""
    print("\n" + "="*50)
    print("示例3：缠论分析")
    print("="*50)
    
    # 获取日线数据
    bars = get_kline(
        symbol="000001.XSHE",
        freq="D",
        start_date="2023-01-01",
        end_date="2024-12-31",
    )
    
    # 创建CZSC对象进行分析
    czsc = CZSC(bars)
    
    print(f"标的: {czsc.symbol}")
    print(f"周期: {czsc.freq}")
    print(f"K线数量: {len(czsc.bars_raw)}")
    print(f"分型数量: {len(czsc.fx_list)}")
    print(f"笔数量: {len(czsc.bi_list)}")
    
    if czsc.bi_list:
        last_bi = czsc.bi_list[-1]
        print(f"\n最后一笔:")
        print(f"  方向: {last_bi.direction}")
        print(f"  起点: {last_bi.fx_a.dt} - {last_bi.fx_a.fx}")
        print(f"  终点: {last_bi.fx_b.dt} - {last_bi.fx_b.fx}")
    
    return czsc


def example4_check_query_count():
    """示例4：查询剩余查询次数"""
    print("\n" + "="*50)
    print("示例4：查询剩余次数")
    print("="*50)
    
    count = get_query_count()
    print(f"剩余查询次数: {count}")
    return count


if __name__ == "__main__":
    print("聚宽数据源使用示例")
    print("="*50)
    
    # 运行示例
    example1_get_daily_kline()
    example2_get_minute_kline()
    example3_czsc_analysis()
    example4_check_query_count()
    
    print("\n" + "="*50)
    print("所有示例运行完成！")
    print("="*50)
