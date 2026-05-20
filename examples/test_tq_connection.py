# -*- coding: utf-8 -*-
"""
天勤数据源连通性测试
- 验证账号密码是否正确
- 验证能否成功拉取 A 股（ETF）日线 K 线数据
"""
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（从项目根目录读取）
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

TQ_USER = os.getenv('TQ_USER')
TQ_PASS = os.getenv('TQ_PASS')

print(f"天勤账号：{TQ_USER}")
print(f"开始测试连接...")

# 直接使用 tqsdk 测试，不走 czsc 顶层包（避免 lz4/clickhouse 兼容冲突）
import pandas as pd
from datetime import datetime, timedelta
from tqsdk import TqApi, TqAuth

try:
    api = TqApi(auth=TqAuth(user_name=TQ_USER, password=TQ_PASS), web_gui=False)
    # 拉取上证 50 ETF 的日线数据
    df = api.get_kline_serial('SSE.510050', duration_seconds=86400, data_length=100)
    df = df.dropna(subset=['close'])
    api.close()

    print(f"\n✅ 连接成功！拉取到 {len(df)} 根日线 K 线")
    print(f"   最新一根K线收盘价：{df['close'].iloc[-1]:.4f}")
    print(f"\n天勤账号配置完成，可以正常使用！")

except Exception as e:
    print(f"\n❌ 连接失败，错误信息：{e}")

