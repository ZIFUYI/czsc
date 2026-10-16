#!/usr/bin/env python
"""
下载第四批扩容的 18 大主流高流动性商品期货 15m K线数据:
能源化工 (6): 塑料 (L), PVC (V), 液化气 (PG), 沥青 (BU), 低硫燃油 (LU), 尿素 (UR)
农产品软商 (7): 菜粕 (RM), 鸡蛋 (JD), 豆一 (A), 玉米 (C), 苹果 (AP), 红枣 (CJ), 花生 (PK)
金属与特钢 (3): 氧化铝 (AO), 不锈钢 (SS), 纸浆 (SP - 轻工造纸)
黑色铁合金 (2): 锰硅 (SM), 硅铁 (SF)
使商品全景池正式拓展至 50 大主力品种！
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

env_file = ROOT_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))

import jqdatasdk as jq

u, p = os.getenv("JQDATA_USERNAME"), os.getenv("JQDATA_PASSWORD")
jq.auth(u, p)
print(">>> JQData 认证成功，开始下载第四批 18 大新商品 15m 数据 (至 50 品种全市场超级池)...")

TARGETS = [
    # 能源化工 (6)
    ("l",  "L9999.XDCE",  "L",  "ENERGY_CHEM", "塑料"),
    ("v",  "V9999.XDCE",  "V",  "ENERGY_CHEM", "PVC"),
    ("pg", "PG9999.XDCE", "PG", "ENERGY_CHEM", "液化气"),
    ("bu", "BU9999.XSGE", "BU", "ENERGY_CHEM", "沥青"),
    ("lu", "LU9999.XINE", "LU", "ENERGY_CHEM", "低硫燃油"),
    ("ur", "UR9999.XZCE", "UR", "ENERGY_CHEM", "尿素"),
    # 农产品与软商品 (7)
    ("rm", "RM9999.XZCE", "RM", "AGRICULTURE", "菜粕"),
    ("jd", "JD9999.XDCE", "JD", "AGRICULTURE", "鸡蛋"),
    ("a",  "A9999.XDCE",  "A",  "AGRICULTURE", "豆一"),
    ("c",  "C9999.XDCE",  "C",  "AGRICULTURE", "玉米"),
    ("ap", "AP9999.XZCE", "AP", "AGRICULTURE", "苹果"),
    ("cj", "CJ9999.XZCE", "CJ", "AGRICULTURE", "红枣"),
    ("pk", "PK9999.XZCE", "PK", "AGRICULTURE", "花生"),
    # 轻工造纸 / 特钢 / 有色 (3)
    ("sp", "SP9999.XSGE", "SP", "ENERGY_CHEM", "纸浆"),
    ("ss", "SS9999.XSGE", "SS", "BASE_METALS",  "不锈钢"),
    ("ao", "AO9999.XSGE", "AO", "BASE_METALS",  "氧化铝"),
    # 黑色铁合金 (2)
    ("sm", "SM9999.XZCE", "SM", "FERROUS",      "锰硅"),
    ("sf", "SF9999.XZCE", "SF", "FERROUS",      "硅铁"),
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

        min_date_str = str(df["dt"].min().date())
        dom = jq.get_dominant_future(dom_code, min_date_str, "2026-10-15")
        dom_df = dom.reset_index()
        dom_df.columns = ["trade_date", "underlying_symbol"]
        dom_df["trade_date"] = pd.to_datetime(dom_df["trade_date"]).dt.date

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
        print(f"成功保存 {p15.name}: 共 {len(df)} 根 15m K线 ({dmin} ~ {dmax})，覆盖 {nc} 个主力合约")
    except Exception as e:
        print(f"抓取 {code} 失败: {e}")

print("\n>>> 第四批 18 大商品数据下载全部完成！已成功集齐 50 大主力商品品种！")
