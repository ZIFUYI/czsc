"""Validate tas_ma_joint_V260520 on 000852.XSHG 15-minute bars.

The script compares the native Rust signal against an independent pandas
translation of the TongDaXin formula:

    MA5/10/20/40/60/120/250, HH/LL, A1/A2/B1/B2, BARSLAST

JQData intraday ``end_date`` is exclusive in practice, so the fetch end date is
shifted by one calendar day while the validation window remains inclusive.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SIGNAL_NAME = "tas_ma_joint_V260520"
MA_SEQ = "5#10#20#40#60#120#250"


def load_jq_credentials() -> None:
    """Load JQData credentials from .env without printing secret values."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_jq_15m(symbol: str, sdt: str, edt: str) -> pd.DataFrame:
    """Read 15-minute bars from JQData; ``edt`` is inclusive at script level."""
    import jqdatasdk as jq

    load_jq_credentials()
    user = os.getenv("JQDATA_USERNAME")
    password = os.getenv("JQDATA_PASSWORD")
    if not user or not password:
        raise RuntimeError("请先设置 JQDATA_USERNAME / JQDATA_PASSWORD")

    jq.auth(user, password)
    end_date = (pd.to_datetime(edt) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = jq.get_price(
        symbol,
        start_date=pd.to_datetime(sdt).strftime("%Y-%m-%d"),
        end_date=end_date,
        frequency="15m",
        fields=["open", "close", "high", "low", "volume", "money"],
        fq=None,
    )
    if df is None or df.empty:
        raise RuntimeError(f"JQData returned no data: {symbol} {sdt} -> {edt}")

    df = df.sort_index().rename(columns={"volume": "vol", "money": "amount"})
    df.index = pd.to_datetime(df.index)
    return df


def barslast(cond: pd.Series) -> pd.Series:
    """TongDaXin BARSLAST: bars since the latest true value; inf if never true."""
    last_true: int | None = None
    out: list[float] = []
    for i, value in enumerate(cond.fillna(False).astype(bool).tolist()):
        if value:
            last_true = i
        out.append(np.inf if last_true is None else float(i - last_true))
    return pd.Series(out, index=cond.index)


def reference_signal(df: pd.DataFrame, ma_seq: str = MA_SEQ) -> pd.DataFrame:
    """Independent pandas reference implementation of the TongDaXin formula."""
    periods = [int(x) for x in ma_seq.split("#")]
    close = df["close"].astype(float)
    ma = {n: close.rolling(n, min_periods=n).mean() for n in periods}
    ma_frame = pd.DataFrame({f"MA{n}": ma[n] for n in periods}, index=df.index)

    hh = ma_frame.max(axis=1, skipna=False)
    ll = ma_frame.min(axis=1, skipna=False)
    a1 = (close > hh) & (ma[60] > ma[120])
    a2 = (ma[60] < ma[120]) | (close < ll)
    b1 = (close < ll) & (ma[60] < ma[120])
    b2 = (ma[60] > ma[120]) | (close > hh)

    signal = pd.Series("其他", index=df.index, dtype=object)
    signal[barslast(a1) < barslast(a2)] = "看多"
    signal[(signal == "其他") & (barslast(b1) < barslast(b2))] = "看空"

    out = df[["open", "close", "high", "low", "vol", "amount"]].copy()
    out["expected"] = signal
    out["A1"] = a1
    out["A2"] = a2
    out["B1"] = b1
    out["B2"] = b2
    return out


def native_signal_available() -> bool:
    """Return whether the current installed extension contains the target signal."""
    try:
        from czsc._native.signals import list_signal_names

        return SIGNAL_NAME in list_signal_names("tas")
    except Exception:
        return False


def compare_native(df: pd.DataFrame, expected: pd.Series, sdt: str, edt: str) -> pd.Series:
    """Run native signal incrementally and return v1 values for the validation window."""
    from czsc import CZSC, Freq, RawBar
    from czsc._native.signals import call_signal

    bars: list[RawBar] = []
    for i, (dt, row) in enumerate(df.iterrows()):
        bars.append(
            RawBar(
                symbol="000852.XSHG",
                dt=dt,
                id=i,
                freq=Freq.F15,
                open=round(float(row["open"]), 4),
                close=round(float(row["close"]), 4),
                high=round(float(row["high"]), 4),
                low=round(float(row["low"]), 4),
                vol=int(float(row["vol"])),
                amount=int(float(row["amount"])),
            )
        )

    start_dt = pd.to_datetime(sdt)
    end_dt = pd.to_datetime(edt) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    start_idx = next(i for i, bar in enumerate(bars) if start_dt <= pd.Timestamp(bar.dt).tz_localize(None) <= end_dt)
    c = CZSC(bars[:start_idx])

    actual: dict[pd.Timestamp, str] = {}
    for bar in bars[start_idx:]:
        dt = pd.Timestamp(bar.dt).tz_localize(None)
        c.update(bar)
        if dt < start_dt or dt > end_dt:
            continue
        sig = call_signal(SIGNAL_NAME, c, {"di": 1, "ma_seq": MA_SEQ})[0]
        actual[dt] = sig.value.split("_")[0]

    return pd.Series(actual).reindex(expected.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="000852.XSHG")
    parser.add_argument("--sdt", default="20260101")
    parser.add_argument("--edt", default="20260520")
    parser.add_argument("--warmup-sdt", default="20251101")
    parser.add_argument("--require-native", action="store_true", help="fail if native signal is not available")
    args = parser.parse_args()

    df = read_jq_15m(args.symbol, args.warmup_sdt, args.edt)
    ref = reference_signal(df)

    start_dt = pd.to_datetime(args.sdt)
    end_dt = pd.to_datetime(args.edt) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    sample = ref.loc[(ref.index >= start_dt) & (ref.index <= end_dt)].copy()

    out_dir = ROOT / "examples" / "results" / "ma_joint_000852_jq_sdk"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"ma_joint_15m_accuracy_{args.sdt}_{args.edt}.csv"

    native_ok = native_signal_available()
    if native_ok:
        sample["actual"] = compare_native(df, sample["expected"], args.sdt, args.edt)
        sample["match"] = sample["expected"] == sample["actual"]
    else:
        sample["actual"] = pd.NA
        sample["match"] = pd.NA
        if args.require_native:
            raise RuntimeError(
                f"{SIGNAL_NAME} is not available in current czsc._native; rebuild the Rust extension first"
            )

    sample.to_csv(out_file, index_label="dt", encoding="utf-8-sig")

    print(f"symbol: {args.symbol}")
    print(f"fetch range: {df.index[0]} -> {df.index[-1]} rows={len(df)}")
    print(f"validation range: {args.sdt} -> {args.edt} rows={len(sample)}")
    print(f"expected counts: {sample['expected'].value_counts(dropna=False).to_dict()}")
    print(f"native available: {native_ok}")
    if native_ok:
        mismatches = sample[~sample["match"].fillna(False)]
        print(f"mismatches: {len(mismatches)}")
        if not mismatches.empty:
            print(mismatches[["close", "expected", "actual"]].head(20).to_string())
            raise SystemExit(1)
    else:
        print("native comparison skipped: rebuild _native before running with --require-native")
    print(f"saved: {out_file}")


if __name__ == "__main__":
    main()
