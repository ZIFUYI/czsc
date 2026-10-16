#!/usr/bin/env python
"""
下载第二批扩容的 12 大主流活跃商品期货 15m K线数据:
1. 沪锌 (ZN)
2. 沪镍 (NI)
3. 热卷 (HC)
4. 焦炭 (J)
5. 玻璃 (FG)
6. 苯乙烯 (EB)
7. 乙二醇 (EG)
8. 聚丙烯 (PP)
9. 豆油 (Y)
10. 菜油 (OI)
11. 棉花 (CF)
12. 工业硅 (SI)
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
print(">>> JQData 认证成功，开始下载第二批 12 大新商品 15m 数据...")

TARGETS = [
    ("zn", "ZN9999.XSGE", "ZN", "BASE_METALS", "沪锌"),
    ("ni", "NI9999.XSGE", "NI", "BASE_METALS", "沪镍"),
    ("hc", "HC9999.XSGE", "HC", "FERROUS",     "热卷"),
    ("j",  "J9999.XDCE",  "J",  "FERROUS",     "焦炭"),
    ("fg", "FG9999.XZCE", "FG", "FERROUS",     "玻璃"),
    ("eb", "EB9999.XDCE", "EB", "ENERGY_CHEM", "苯乙烯"),
    ("eg", "EG9999.XDCE", "EG", "ENERGY_CHEM", "乙二醇"),
    ("pp", "PP9999.XDCE", "PP", "ENERGY_CHEM", "聚丙烯"),
    ("y",  "Y9999.XDCE",  "Y",  "AGRICULTURE", "豆油"),
    ("oi", "OI9999.XZCE", "OI", "AGRICULTURE", "菜油"),
    ("cf", "CF9999.XZCE", "CF", "AGRICULTURE", "棉花"),
    ("si", "SI9999.GFEX", "SI", "NEW_ENERGY",  "工业硅"),
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
        print(f"成功保存 {p15.name}: 共 {len(df)} 行 15m K线 ({dmin} ~ {dmax})，覆盖 {nc} 个换月主力合约")
    except Exception as e:
        print(f"抓取 {code} 失败: {e}")

print("\n>>> 第二批 12 大新商品数据获取全部完毕！")
