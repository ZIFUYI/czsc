"""批量下载 56 个主力商品期货从 2024-01-01 到 2026-10-01 的 15分钟 K 线数据"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqsdk import TqApi, TqAuth
from tqsdk.tools import DataDownloader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.signals_dev.backtest_continuous_signal_commodity_cta import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_SYMBOLS,
    load_env,
)


def main():
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = DEFAULT_DATA_DIR / "tmp_csv"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tq_user, tq_pass = load_env()
    if not tq_user or not tq_pass:
        raise ValueError("请在 .env 中配置 TQ_USER 和 TQ_PASS")

    # 检查哪些品种缺少 2024 年初数据
    to_download = []
    for sym in DEFAULT_SYMBOLS:
        p = DEFAULT_DATA_DIR / f"{sym}_15m.parquet"
        if not p.exists():
            to_download.append(sym)
        else:
            try:
                df = pd.read_parquet(p)
                first_d = str(df["dt"].min())[:10]
                if first_d > "2024-01-05":
                    to_download.append(sym)
            except Exception:
                to_download.append(sym)

    print(f"共有 {len(to_download)} / {len(DEFAULT_SYMBOLS)} 个品种需要拉取 2024-01-01 至今的完整数据")
    if not to_download:
        print("所有品种历史数据已完全齐备！")
        return

    api = TqApi(auth=TqAuth(tq_user, tq_pass))
    try:
        start_dt = datetime(2024, 1, 1)
        end_dt = datetime(2026, 10, 1)

        total = len(to_download)
        for idx, sym in enumerate(to_download, 1):
            t0 = time.time()
            tq_sym = f"KQ.m@{sym}"
            csv_file = tmp_dir / f"{sym}.csv"

            print(f"[{idx}/{total}] 正在同步 {tq_sym} (2024-01-01 ~ 2026-10-01) ...", end="", flush=True)
            try:
                kd = DataDownloader(
                    api,
                    symbol_list=tq_sym,
                    dur_sec=15 * 60,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    csv_file_name=str(csv_file),
                )
                while not kd.is_finished():
                    api.wait_update()

                if csv_file.exists():
                    df = pd.read_csv(csv_file)
                    if not df.empty:
                        df["dt"] = pd.to_datetime(df["datetime"])
                        prefix = f"{tq_sym}."
                        df["open"] = df[f"{prefix}open"]
                        df["high"] = df[f"{prefix}high"]
                        df["low"] = df[f"{prefix}low"]
                        df["close"] = df[f"{prefix}close"]
                        df["volume"] = df[f"{prefix}volume"]
                        oi_col = f"{prefix}close_oi" if f"{prefix}close_oi" in df.columns else f"{prefix}open_oi"
                        df["open_interest"] = df[oi_col] if oi_col in df.columns else 0.0

                        cols = ["dt", "open", "high", "low", "close", "volume", "open_interest"]
                        df = df[cols].sort_values("dt").reset_index(drop=True)

                        out_parquet = DEFAULT_DATA_DIR / f"{sym}_15m.parquet"
                        df.to_parquet(out_parquet)
                        print(f" 完成 ({len(df)} 根 K 线, 耗时 {time.time() - t0:.1f}s)")
                    else:
                        print(" 空数据")
                    csv_file.unlink(missing_ok=True)
            except Exception as e:
                print(f" 异常: {e}")
                csv_file.unlink(missing_ok=True)

    finally:
        api.close()
        tmp_dir.rmdir()

    print("\n✅ 全部 56 个品种 20240101-20260930 历史数据同步完成！")


if __name__ == "__main__":
    main()
