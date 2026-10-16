#!/usr/bin/env python
"""
下载扩展的 10 大商品期货 15m K 线数据与真实换月主力映射序列:
1. 沪锡 (SN)
2. 沪铝 (AL)
3. 焦煤 (JM)
4. 纯碱 (SA)
5. 天然橡胶 (RU)
6. 甲醇 (MA)
7. 燃料油 (FU)
8. 棕榈油 (P)
9. 生猪 (LH)
10. 白糖 (SR)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
COMMODITY_DATA_DIR = ROOT_DIR / "examples" / "results" / "commodities_5m"
COMMODITY_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 加载 .env 凭据
env_file = ROOT_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))

import jqdatasdk as jq

u, p = os.getenv("JQDATA_USERNAME"), os.getenv("JQDATA_PASSWORD")
if not u or not p:
    raise RuntimeError("未在 .env 中找到 JQDATA_USERNAME / JQDATA_PASSWORD")

jq.auth(u, p)
print(">>> JQData 认证成功，开始下载 10 大新商品 15m 数据...")

TARGETS = [
    ("sn", "SN9999.XSGE", "SN", "BASE_METALS", "沪锡"),
    ("al", "AL9999.XSGE", "AL", "BASE_METALS", "沪铝"),
    ("jm", "JM9999.XDCE", "JM", "FERROUS",     "焦煤"),
    ("sa", "SA9999.XZCE", "SA", "FERROUS",     "纯碱"),
    ("ru", "RU9999.XSGE", "RU", "ENERGY_CHEM", "天然橡胶"),
    ("ma", "MA9999.XZCE", "MA", "ENERGY_CHEM", "甲醇"),
    ("fu", "FU9999.XSGE", "FU", "ENERGY_CHEM", "燃料油"),
    ("p",  "P9999.XDCE",  "P",  "AGRICULTURE", "棕榈油"),
    ("lh", "LH9999.XDCE", "LH", "AGRICULTURE", "生猪"),
    ("sr", "SR9999.XZCE", "SR", "AGRICULTURE", "白糖"),
]

for sid, code, dom_code, sector, name in TARGETS:
    p15 = COMMODITY_DATA_DIR / f"{sid}_15m_2020_2026.parquet"
    if p15.exists():
        print(f"[{sid.upper():4s} - {name}] 目标文件已存在 ({p15.name})，跳过。")
        continue

    print(f"\n>>> 正在抓取 [{sid.upper():4s} - {name} ({code})] 15m K线 (2020-01-01 至 2026-10-15)...")
    try:
        df = jq.get_price(
            code,
            start_date="2020-01-01",
            end_date="2026-10-15 16:00:00",
            frequency="15m",
            fields=["open", "high", "low", "close", "volume", "money"],
        )
        if df is None or df.empty:
            print(f"警告: {code} 返回空数据")
            continue

        df = df.dropna(subset=["close"]).reset_index().rename(columns={"index": "dt", "money": "amount"})
        df["dt"] = pd.to_datetime(df["dt"])
        df["symbol"] = code

        # 获取历史主力合约映射序列
        min_date_str = str(df["dt"].min().date())
        dom = jq.get_dominant_future(dom_code, min_date_str, "2026-10-15")
        dom_df = dom.reset_index()
        dom_df.columns = ["trade_date", "underlying_symbol"]
        dom_df["trade_date"] = pd.to_datetime(dom_df["trade_date"]).dt.date

        # 处理期货夜盘换日对齐 (晚 18:00 之后对齐至下一个交易日，周五晚/周六凌晨顺延至周一)
        hrs = df["dt"].dt.hour
        wks = df["dt"].dt.weekday
        sh = pd.Series(0, index=df.index)
        sh = sh.mask(hrs >= 18, 1).mask((hrs >= 18) & (wks == 4), 3).mask(wks == 5, 2).mask(wks == 6, 1)
        df["trade_date"] = (df["dt"] + pd.to_timedelta(sh, unit="D")).dt.date

        df = df.merge(dom_df, on="trade_date", how="left").drop(columns=["trade_date"])
        df["underlying_symbol"] = df["underlying_symbol"].ffill().bfill()

        df.to_parquet(p15, index=False)
        dmin, dmax = df["dt"].min(), df["dt"].max()
        nc = df["underlying_symbol"].nunique()
        print(f"成功保存 {p15.name}: 共 {len(df)} 行 15m K线 ({dmin} ~ {dmax})，覆盖 {nc} 个换月主力合约")
    except Exception as e:
        print(f"抓取 {code} 失败: {e}")

print("\n>>> 10 大新商品数据获取全部完毕！")
