import os
import sys
from pathlib import Path
import jqdatasdk as jq
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))

u, p = os.getenv("JQDATA_USERNAME"), os.getenv("JQDATA_PASSWORD")
jq.auth(u, p)

# 标准机构 CTA 七大商品板块划分
SECTOR_MAP = {
    "贵金属 (PRECIOUS)": [
        ("AG", "AG9999.XSGE", "沪银", "上期所"),
        ("AU", "AU9999.XSGE", "沪金", "上期所"),
        ("PT", "PT9999.GFEX", "铂金", "广期所(待/新)"),
        ("PD", "PD9999.GFEX", "钯金", "广期所(待/新)"),
    ],
    "有色金属 (BASE_METALS)": [
        ("CU", "CU9999.XSGE", "沪铜", "上期所"),
        ("AL", "AL9999.XSGE", "沪铝", "上期所"),
        ("ZN", "ZN9999.XSGE", "沪锌", "上期所"),
        ("PB", "PB9999.XSGE", "沪铅", "上期所"),
        ("NI", "NI9999.XSGE", "沪镍", "上期所"),
        ("SN", "SN9999.XSGE", "沪锡", "上期所"),
        ("AO", "AO9999.XSGE", "氧化铝", "上期所"),
        ("BC", "BC9999.XINE", "国际铜", "能源中心"),
    ],
    "黑色建材 (FERROUS)": [
        ("RB", "RB9999.XSGE", "螺纹钢", "上期所"),
        ("HC", "HC9999.XSGE", "热轧卷板", "上期所"),
        ("I",  "I9999.XDCE",  "铁矿石", "大商所"),
        ("J",  "J9999.XDCE",  "焦炭",   "大商所"),
        ("JM", "JM9999.XDCE", "焦煤",   "大商所"),
        ("FG", "FG9999.XZCE", "玻璃",   "郑商所"),
        ("SA", "SA9999.XZCE", "纯碱",   "郑商所"),
        ("SS", "SS9999.XSGE", "不锈钢", "上期所"),
        ("SM", "SM9999.XZCE", "锰硅",   "郑商所"),
        ("SF", "SF9999.XZCE", "硅铁",   "郑商所"),
    ],
    "能源化工 (ENERGY_CHEM)": [
        ("SC", "SC9999.XINE", "原油",   "能源中心"),
        ("FU", "FU9999.XSGE", "燃料油", "上期所"),
        ("LU", "LU9999.XINE", "低硫燃油", "能源中心"),
        ("BU", "BU9999.XSGE", "沥青",   "上期所"),
        ("TA", "TA9999.XZCE", "PTA",    "郑商所"),
        ("MA", "MA9999.XZCE", "甲醇",   "郑商所"),
        ("PP", "PP9999.XDCE", "聚丙烯", "大商所"),
        ("L",  "L9999.XDCE",  "塑料LLDPE", "大商所"),
        ("V",  "V9999.XDCE",  "PVC聚氯乙烯", "大商所"),
        ("RU", "RU9999.XSGE", "天然橡胶", "上期所"),
        ("NR", "NR9999.XINE", "20号胶", "能源中心"),
        ("BR", "BR9999.XSGE", "丁二烯胶", "上期所"),
        ("EB", "EB9999.XDCE", "苯乙烯", "大商所"),
        ("EG", "EG9999.XDCE", "乙二醇", "大商所"),
        ("UR", "UR9999.XZCE", "尿素",   "郑商所"),
        ("PG", "PG9999.XDCE", "LPG液化气", "大商所"),
        ("PX", "PX9999.XZCE", "对二甲苯", "郑商所"),
        ("SH", "SH9999.XZCE", "烧碱",   "郑商所"),
        ("SP", "SP9999.XSGE", "纸浆",   "上期所"),
    ],
    "农产品与软商品 (AGRICULTURE)": [
        ("M",  "M9999.XDCE",  "豆粕",   "大商所"),
        ("Y",  "Y9999.XDCE",  "豆油",   "大商所"),
        ("P",  "P9999.XDCE",  "棕榈油", "大商所"),
        ("OI", "OI9999.XZCE", "菜油",   "郑商所"),
        ("RM", "RM9999.XZCE", "菜粕",   "郑商所"),
        ("C",  "C9999.XDCE",  "玉米",   "大商所"),
        ("CS", "CS9999.XDCE", "玉米淀粉", "大商所"),
        ("A",  "A9999.XDCE",  "黄大豆1号", "大商所"),
        ("B",  "B9999.XDCE",  "黄大豆2号", "大商所"),
        ("CF", "CF9999.XZCE", "棉花",   "郑商所"),
        ("SR", "SR9999.XZCE", "白糖",   "郑商所"),
        ("AP", "AP9999.XZCE", "苹果",   "郑商所"),
        ("CJ", "CJ9999.XZCE", "红枣",   "郑商所"),
        ("LH", "LH9999.XDCE", "生猪",   "大商所"),
        ("JD", "JD9999.XDCE", "鸡蛋",   "大商所"),
        ("PK", "PK9999.XZCE", "花生",   "郑商所"),
    ],
    "新能源与新兴产业 (NEW_ENERGY)": [
        ("LC", "LC9999.GFEX", "碳酸锂", "广期所"),
        ("SI", "SI9999.GFEX", "工业硅", "广期所"),
        ("PS", "PS9999.GFEX", "多晶硅", "广期所"),
    ],
    "航运航线 (SHIPPING)": [
        ("EC", "EC9999.XINE", "集运欧线", "能源中心"),
    ],
}

# 检查当前已回测并缓存 15m 数据的品种
cached_files = {p.name.split("_15m")[0].lower() for p in Path("examples/results/commodities_5m").glob("*_15m*.parquet")}

print("=========================================================================================================")
print("  国内商品期货主流品种图谱 (七大板块 · 机构级 CTA 候选池)")
print("=========================================================================================================")

all_rows = []
for sec_name, items in SECTOR_MAP.items():
    print(f"\n【{sec_name}】")
    for code, sym, name, exch in items:
        is_covered = code.lower() in cached_files
        status_tag = "[已覆盖并回测]" if is_covered else "[可新增导入]"
        # 查一下近 1 年日均成交额 (亿元) 判断流动性
        try:
            p_bars = jq.get_bars(sym, count=60, unit="1d", fields=["money"])
            avg_money = float(p_bars["money"].mean() / 1e8) if p_bars is not None and not p_bars.empty else 0.0
            liq_str = f"近60日均成交 {avg_money:5.1f} 亿元"
        except Exception:
            liq_str = "流动性待查"
        all_rows.append({
            "板块": sec_name.split()[0],
            "代码": code,
            "名称": name,
            "主力代码": sym,
            "交易所": exch,
            "当前状态": status_tag,
            "流动性": liq_str,
        })
        print(f"  {code:4s} | {name:8s} ({sym:13s}) | {exch:6s} | {status_tag:12s} | {liq_str}")

df_all = pd.DataFrame(all_rows)
