#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
聚宽数据源配置和测试脚本
"""
import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 聚宽账号配置
JQDATA_USERNAME = os.getenv("JQDATA_USERNAME")
JQDATA_PASSWORD = os.getenv("JQDATA_PASSWORD")

print("聚宽账号配置:")
print(f"用户名: {'已设置' if JQDATA_USERNAME else '未设置'}")
print(f"密码: {'已设置' if JQDATA_PASSWORD else '未设置'}")
print("-" * 50)


def ensure_credentials():
    """确认聚宽账号密码已经通过环境变量配置。"""
    if JQDATA_USERNAME and JQDATA_PASSWORD:
        return True

    print("请先在本地 .env 或系统环境变量中设置 JQDATA_USERNAME 和 JQDATA_PASSWORD")
    return False


def test_jqdata_connection():
    """测试聚宽数据连接"""
    from czsc.connectors.jq_connector import set_token, get_kline

    if not ensure_credentials():
        return False

    print("\n正在设置聚宽Token...")
    set_token(jq_mob=JQDATA_USERNAME, jq_pwd=JQDATA_PASSWORD)
    print("Token设置成功！")

    print("\n正在测试数据获取...")
    # 测试获取平安银行日线数据
    try:
        bars = get_kline(
            symbol="000001.XSHE",  # 平安银行
            freq="D",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        print(f"成功获取数据，共 {len(bars)} 条K线")
        print(f"\n最新一条数据:")
        print(f"  时间: {bars[-1].dt}")
        print(f"  开盘: {bars[-1].open}")
        print(f"  收盘: {bars[-1].close}")
        print(f"  最高: {bars[-1].high}")
        print(f"  最低: {bars[-1].low}")
        print(f"  成交量: {bars[-1].vol}")
        return True
    except Exception as e:
        print(f"数据获取失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("聚宽数据源配置测试")
    print("=" * 50)
    
    success = test_jqdata_connection()
    
    if success:
        print("\n" + "=" * 50)
        print("聚宽数据源配置成功！")
        print("=" * 50)
        print("\n你现在可以使用以下方式获取数据:")
        print("""
	from czsc.connectors.jq_connector import get_kline

	# 获取日线数据
	bars = get_kline(
	    symbol="000001.XSHE",  # 股票代码
	    freq="D",              # 日线
	    start_date="2024-01-01",
	    end_date="2024-12-31",
	)

	# 获取分钟线数据
	bars = get_kline(
	    symbol="000001.XSHE",
	    freq="5min",           # 5分钟线
	    start_date="2024-12-01",
	    end_date="2024-12-31",
	)
        """)
    else:
        print("\n" + "=" * 50)
        print("配置失败，请检查账号密码或网络连接")
        print("=" * 50)
        sys.exit(1)
