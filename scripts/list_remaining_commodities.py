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

all_sec = jq.get_all_securities(types=["futures"])
all_sec["symbol"] = all_sec.index
dom = all_sec[all_sec["symbol"].str.contains("9999")].copy()
dom["code"] = dom["symbol"].apply(lambda x: x.split("9999")[0].lower())
financial = ["if", "ic", "im", "ih", "ts", "tf", "t", "tl"]
comm = dom[~dom["code"].isin(financial)].copy()

# 已覆盖的 32 个品种
covered = {
    "ag", "au", "cu", "al", "sn", "zn", "ni", "rb", "i", "jm", "j", "sa", "hc", "fg",
    "sc", "ta", "ma", "ru", "fu", "eb", "eg", "pp", "m", "p", "y", "oi", "lh", "sr", "cf",
    "lc", "ps", "si"
}

remaining = comm[~comm["code"].isin(covered)].copy()

rows = []
for idx, r in remaining.iterrows():
    sym = r["symbol"]
    code = r["code"].upper()
    name = r["display_name"]
    s_dt = str(r["start_date"])[:10]
    e_dt = str(r["end_date"])[:10]
    try:
        b = jq.get_bars(sym, count=30, unit="1d", fields=["date", "close", "volume"])
        if b is not None and not b.empty:
            last_close = float(b["close"][-1])
            avg_vol = float(b["volume"].mean())
            last_trade = str(b["date"][-1])[:10]
            # 粗略估算单日成交额 (亿元)
            avg_m = (avg_vol * last_close * 10) / 1e8  # 考虑乘数，粗算估值
        else:
            avg_m = 0.0
            last_trade = "无成交"
    except Exception:
        avg_m = 0.0
        last_trade = "接口异常"
    rows.append({
        "代码": code,
        "名称": name,
        "主力代码": sym,
        "上市时间": s_dt,
        "近60日均成交额(亿元)": round(avg_m, 1),
    })

rem_df = pd.DataFrame(rows).sort_values("近60日均成交额(亿元)", ascending=False)
print(f"当前已覆盖品种数: {len(covered)} 个")
print(f"剩余未覆盖品种总数: {len(rem_df)} 个\n")
print(rem_df.to_string(index=False))
