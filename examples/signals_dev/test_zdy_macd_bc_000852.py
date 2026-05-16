# -*- coding: utf-8 -*-
"""Test zdy_macd_bc_V230422 on CSI 1000 index data.

The script uses JoinQuant JQData SDK for 000852.XSHG and scans the MACD area
divergence signal with the default threshold T50.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Avoid optional ClickHouse/lz4 import issues when importing czsc top-level modules.
for mod in [
    "clickhouse_connect",
    "clickhouse_connect.driver",
    "clickhouse_connect.driver.client",
    "clickhouse_connect.driver.httpclient",
    "clickhouse_connect.driver.compression",
]:
    sys.modules.setdefault(mod, MagicMock())

from czsc.core import CZSC, RawBar
from czsc.core import Freq
from czsc.signals.zdy import zdy_macd_bc_V230422


def load_jq_credentials():
    """Load JQData credentials from .env or environment."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            import os

            os.environ.setdefault(key, value)


def read_jq_sdk_bars(
    freq: str,
    czsc_freq: Freq,
    sdt: str = "20240101",
    edt: str = "20260430",
) -> list[RawBar]:
    """Read 000852.XSHG bars from JoinQuant SDK."""
    import os
    import jqdatasdk as jq

    load_jq_credentials()
    user = os.getenv("JQDATA_USERNAME")
    password = os.getenv("JQDATA_PASSWORD")
    if not user or not password:
        raise RuntimeError("请先设置 JQDATA_USERNAME / JQDATA_PASSWORD")

    jq.auth(user, password)
    df = jq.get_price(
        "000852.XSHG",
        start_date=pd.to_datetime(sdt).strftime("%Y-%m-%d"),
        end_date=pd.to_datetime(edt).strftime("%Y-%m-%d"),
        frequency=freq,
        fields=["open", "close", "high", "low", "volume", "money"],
        fq=None,
    )
    if df is None or df.empty:
        return []

    df = df.sort_index()
    bars = []
    for dt, row in df.iterrows():
        bars.append(
            RawBar(
                symbol="000852.XSHG",
                dt=pd.to_datetime(dt),
                id=len(bars),
                freq=czsc_freq,
                open=round(float(row["open"]), 4),
                close=round(float(row["close"]), 4),
                high=round(float(row["high"]), 4),
                low=round(float(row["low"]), 4),
                vol=int(float(row["volume"])),
                amount=int(float(row["money"])),
            )
        )
    return bars


def scan_macd_area_bc(label: str, bars: list[RawBar], th: int = 50, init_n: int = 200) -> list[dict]:
    """Scan zdy_macd_bc_V230422 on rolling CZSC bars."""
    print(f"\n=== {label} bars={len(bars)} ===")
    if not bars:
        print("No bars returned")
        return []

    print(f"range: {bars[0].dt} -> {bars[-1].dt}")
    if len(bars) <= init_n + 20:
        print(f"Skip: bars <= init_n + 20 ({init_n + 20})")
        return []

    c = CZSC(bars[:init_n])
    events = []
    key = None
    for bar in bars[init_n:]:
        c.update(bar)
        signal = zdy_macd_bc_V230422(c, di=1, th=th)
        if key is None:
            key = next(iter(signal.keys()))

        value = next(iter(signal.values()))
        if "其他" in value:
            continue

        events.append(
            {
                "symbol": bar.symbol,
                "freq": bar.freq.value,
                "dt": bar.dt,
                "close": bar.close,
                "signal": f"{key}_{value}",
                "value": value,
            }
        )

    print(f"signal key: {key}")
    print(f"events: {len(events)}")
    if events:
        print("counts:", dict(Counter(x["value"] for x in events)))
        print(pd.DataFrame(events).tail(10).to_string(index=False))
    return events


def main():
    th = 50
    jobs = [
        ("日线 T50", "daily", Freq.D, 200),
        ("15分钟 T50", "15m", Freq.F15, 300),
        ("30分钟 T50", "30m", Freq.F30, 240),
        ("60分钟 T50", "60m", Freq.F60, 160),
    ]

    all_events = []
    for label, freq, czsc_freq, init_n in jobs:
        bars = read_jq_sdk_bars(freq=freq, czsc_freq=czsc_freq)
        all_events.extend(scan_macd_area_bc(label, bars, th=th, init_n=init_n))

    out_dir = Path(__file__).resolve().parents[1] / "results" / "macd_area_bc_000852_jq_sdk"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "zdy_macd_bc_events.csv"
    pd.DataFrame(all_events).to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved events to: {out_file}")


if __name__ == "__main__":
    main()
