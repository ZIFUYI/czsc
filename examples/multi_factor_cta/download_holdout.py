"""Download unopened CTA holdout data; inspect coverage, never strategy performance.

The parent process enforces a wall-clock timeout on all SDK calls, including
authentication. Credentials stay inside the worker and SDK console output is
discarded. Continuous-contract bars remain reference-only research prices.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from examples.multi_factor_cta.audit_protocol import file_sha256, validate_output_directory

ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_SOURCE = ROOT / "examples/results/intraday_cta_ablation_20261017/run_full_60_commodities_study.py"
DEFAULT_OUTPUT = ROOT / "examples/results/cta_research/holdout_data"
EXCHANGE_MAP = {"XSGE": "SHFE", "XDCE": "DCE", "XZCE": "CZCE", "XINE": "INE", "GFEX": "GFEX"}


def read_tq_universe(path: Path = UNIVERSE_SOURCE) -> list[dict[str, Any]]:
    """Read only the static universe literal, without executing a research entrypoint."""
    tree = ast.parse(path.read_text())
    rows = None
    for node in tree.body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(target, ast.Name) and target.id == "RAW_60_META" for target in targets):
            rows = ast.literal_eval(node.value)
            break
    if rows is None:
        raise ValueError("Frozen universe RAW_60_META was not found")
    result = []
    for aid, symbol, sector, name, night, fee in rows:
        exchange = EXCHANGE_MAP[symbol.rsplit(".", 1)[1]]
        product = aid.upper() if exchange == "CZCE" else aid.lower()
        result.append(
            {
                "id": aid,
                "symbol": symbol,
                "tq_symbol": f"KQ.m@{exchange}.{product}",
                "sector": sector,
                "name": name,
                "has_night_session": night,
                "fee_rate": fee,
            }
        )
    return result


def normalize_tq_bars(
    raw: pd.DataFrame, *, duration_seconds: int, start: str, end: str, observed_at: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert UTC epoch bar starts to completed Shanghai bars and audit OHLCV.

    Requested end and current observation time are both inclusive availability
    bounds. A bar closing exactly at either bound is complete and retained.
    """
    if duration_seconds <= 0:
        raise ValueError("Bar duration must be positive")
    frame = raw.copy()
    start_dt, end_dt, observed = pd.Timestamp(start), pd.Timestamp(end), pd.Timestamp(observed_at)
    starts = pd.to_datetime(frame["datetime"], unit="ns", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    frame["dt"] = starts + pd.Timedelta(seconds=duration_seconds)
    in_window = (frame.dt >= start_dt) & (frame.dt <= end_dt) & (frame.dt <= observed)
    raw_window = frame.loc[in_window].copy()
    prices = raw_window[["open", "high", "low", "close"]].to_numpy(dtype=float)
    valid = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    valid &= (raw_window.high >= raw_window[["open", "close"]].max(axis=1)).to_numpy()
    valid &= (raw_window.low <= raw_window[["open", "close"]].min(axis=1)).to_numpy()
    valid &= (raw_window.high >= raw_window.low).to_numpy()
    valid &= np.isfinite(raw_window.volume.to_numpy(dtype=float)) & (raw_window.volume.to_numpy(dtype=float) >= 0)
    cleaned = raw_window.loc[valid].sort_values("dt")
    duplicates = int(cleaned.dt.duplicated().sum())
    cleaned = cleaned.drop_duplicates("dt", keep="last")
    if "open_interest" not in cleaned and "close_oi" in cleaned:
        cleaned["open_interest"] = cleaned["close_oi"]
    columns = ["dt", "open", "high", "low", "close", "volume"]
    if "open_interest" in cleaned:
        columns.append("open_interest")
    cleaned = cleaned[columns].reset_index(drop=True)
    counts = cleaned.groupby(cleaned.dt.dt.strftime("%Y-%m-%d")).size()
    summary = {
        "downloaded_rows": len(raw),
        "requested_window_rows": len(raw_window),
        "rows": len(cleaned),
        "invalid_ohlcv_rows": int((~valid).sum()),
        "duplicate_timestamps_removed": duplicates,
        "start": str(cleaned.dt.min()) if len(cleaned) else None,
        "end": str(cleaned.dt.max()) if len(cleaned) else None,
        "rows_by_calendar_date": {str(day): int(count) for day, count in counts.items()},
        "timestamp_semantics": "UTC epoch start -> Asia/Shanghai + duration; completed-bar availability",
        "duration_seconds": duration_seconds,
        "execution_scope": "continuous_reference_only",
    }
    return cleaned, summary


def _credentials() -> tuple[str, str]:
    values = {key: os.environ.get(key, "") for key in ("TQ_USER", "TQ_PASS")}
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                if key.strip() in values and not values[key.strip()]:
                    values[key.strip()] = value.strip().strip("\"'")
    if not all(values.values()):
        raise ValueError("TQ_USER and TQ_PASS must be configured")
    return values["TQ_USER"], values["TQ_PASS"]


def _safe_error(error: Exception, secrets: tuple[str, ...]) -> dict[str, str]:
    message = str(error)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[redacted]")
    message = re.sub(r"https?://\S+", "[service URL]", message)
    message = re.sub(r"(?i)(password|token|secret|authorization)\s*[:=]\s*[^\s,;]+", r"\1=[redacted]", message)
    return {"error_type": type(error).__name__, "message": message[:500]}


def _save_manifest(output: Path, manifest: dict[str, Any]) -> None:
    target = output / "download_manifest.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    temporary.replace(target)


def download_worker(args: argparse.Namespace) -> int:
    """Run read-only SDK requests; persist raw observations and coverage after each asset."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    universe = read_tq_universe()
    if args.symbols:
        chosen = set(args.symbols.split(","))
        universe = [row for row in universe if row["id"] in chosen]
    manifest: dict[str, Any] = {
        "status": "downloading",
        "requested_start": args.start,
        "requested_end": args.end,
        "holdout_performance_computed": False,
        "execution_scope": "continuous_reference_only",
        "source_sha256": file_sha256(__file__),
        "universe_source_sha256": file_sha256(UNIVERSE_SOURCE),
        "universe": universe,
        "assets": {},
        "mapping": {},
    }
    existing_path = output / "download_manifest.json"
    if existing_path.exists():
        if not args.resume:
            raise ValueError("Existing holdout download requires --resume or a fresh output directory")
        previous = json.loads(existing_path.read_text())
        if any(previous.get(key) != manifest[key] for key in ("requested_start", "requested_end", "universe")):
            raise ValueError("Resume requires the same frozen date window and universe")
        manifest = previous
        manifest["status"] = "downloading"
    manifest.setdefault("attempts", []).append({"source_sha256": file_sha256(__file__)})
    _save_manifest(output, manifest)
    api, secrets = None, ()
    try:
        user, password = _credentials()
        secrets = (user, password)
        from tqsdk import TqApi, TqAuth

        api = TqApi(auth=TqAuth(user, password), web_gui=False, disable_print=True)
        try:
            map_path = output / "historical_main_contract_mapping.parquet"
            if manifest["mapping"].get("status") == "downloaded":
                if file_sha256(map_path) != manifest["mapping"]["sha256"]:
                    raise ValueError("Downloaded mapping changed; refusing silent replacement")
            else:
                mapping = api.query_his_cont_quotes([row["tq_symbol"] for row in universe], n=args.mapping_days)
                mapping["date"] = pd.to_datetime(mapping["date"])
                mapping = mapping.loc[(mapping.date >= "2024-01-01") & (mapping.date <= pd.Timestamp(args.end))]
                mapping.to_parquet(map_path, index=False)
                manifest["mapping"] = {
                    "status": "downloaded",
                    "rows": len(mapping),
                    "path": str(map_path),
                    "sha256": file_sha256(map_path),
                    "start": str(mapping.date.min()),
                    "end": str(mapping.date.max()),
                    "meaning": "Historical underlying identifiers only; no individual-contract fills simulated",
                }
        except Exception as error:
            manifest["mapping"] = {"status": "failed", **_safe_error(error, secrets)}
        _save_manifest(output, manifest)
        for row in universe:
            asset = row["id"]
            try:
                if manifest["assets"].get(asset, {}).get("status") == "downloaded":
                    for frame in manifest["assets"][asset]["frames"].values():
                        if (
                            file_sha256(frame["path"]) != frame["sha256"]
                            or file_sha256(frame["raw_path"]) != frame["raw_sha256"]
                        ):
                            raise ValueError("Downloaded bars changed; refusing silent replacement")
                    continue
                frames = {}
                for duration, suffix in ((300, "5m"), (900, "15m")):
                    raw = api.get_kline_serial(row["tq_symbol"], duration_seconds=duration, data_length=10000).copy()
                    observed = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
                    cleaned, summary = normalize_tq_bars(
                        raw, duration_seconds=duration, start=args.start, end=args.end, observed_at=str(observed)
                    )
                    raw_dt = pd.to_datetime(raw.datetime, unit="ns", utc=True).dt.tz_convert("Asia/Shanghai")
                    raw_dt = raw_dt.dt.tz_localize(None) + pd.Timedelta(seconds=duration)
                    selected_raw = raw.loc[(raw_dt >= args.start) & (raw_dt <= args.end) & (raw_dt <= observed)]
                    raw_path = output / f"{asset}_{suffix}_raw.parquet"
                    derived_path = output / f"{asset}_{suffix}.parquet"
                    selected_raw.to_parquet(raw_path, index=False)
                    cleaned.to_parquet(derived_path, index=False)
                    frames[suffix] = {
                        **summary,
                        "raw_path": str(raw_path),
                        "raw_sha256": file_sha256(raw_path),
                        "path": str(derived_path),
                        "sha256": file_sha256(derived_path),
                    }
                status = "downloaded" if all(frame["rows"] > 0 for frame in frames.values()) else "empty"
                manifest["assets"][asset] = {"status": status, "frames": frames}
            except Exception as error:
                manifest["assets"][asset] = {"status": "failed", **_safe_error(error, secrets)}
            _save_manifest(output, manifest)
        manifest["status"] = (
            "complete" if all(v["status"] == "downloaded" for v in manifest["assets"].values()) else "partial"
        )
    except Exception as error:
        manifest["status"] = "failed"
        manifest["connection_error"] = _safe_error(error, secrets)
    finally:
        _save_manifest(output, manifest)
        if api is not None:
            api.close()
    return 0 if manifest["status"] == "complete" else 1


def main() -> None:
    """Launch a bounded worker; print only a credential-free completion summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="2026-10-01 00:00:00")
    parser.add_argument("--end", default="2026-10-20 23:59:59")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--mapping-days", type=int, default=800)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--resume", action="store_true", help="Verify and preserve completed files after a timeout")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    readonly_roots = [
        ROOT / "examples/results/intraday_cta_ablation_20261017",
        ROOT / "examples/results/research_rerun_20261017",
        ROOT / "examples/results/continuous_signal_commodity_cta",
    ]
    validate_output_directory(args.output, readonly_roots)
    if args.worker:
        logging.disable(logging.CRITICAL)
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            code = download_worker(args)
        raise SystemExit(code)
    args.output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "examples.multi_factor_cta.download_holdout",
        "--worker",
        "--output",
        str(args.output),
        "--start",
        args.start,
        "--end",
        args.end,
        "--mapping-days",
        str(args.mapping_days),
        "--symbols",
        args.symbols,
    ]
    if args.resume:
        command.append("--resume")
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=args.timeout_seconds,
            check=False,
        )
        status = {"worker_returncode": result.returncode}
    except subprocess.TimeoutExpired:
        status = {"status": "timeout", "timeout_seconds": args.timeout_seconds}
        manifest_path = args.output / "download_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest.update(status)
        _save_manifest(args.output, manifest)
    status["manifest"] = str((args.output / "download_manifest.json").resolve())
    print(json.dumps(status))


if __name__ == "__main__":
    main()
