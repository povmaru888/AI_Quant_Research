"""Build monthly training panels with bounded bulk reads and safe checkpoints.

Usage:
    python tools/build_panels.py [--config config.yaml]
        [--start 2019-01] [--end 2023-12] [--out data/panels]
        [--force] [--inference-only] [--profile-json reports/panel-profile.json]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import pickle
import sys
import time
import traceback
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runtime.dotenv import load_dotenv  # noqa: E402
from runtime.panel_data import (  # noqa: E402
    PANEL_FORMAT_VERSION,
    PreparedPanelData,
    build_fingerprint,
    canonical_panel_hash,
    source_fingerprint,
)
from services.factor_v4_eligibility_service import filter_factor_v4_universe  # noqa: E402
from services.feature_preprocess_service import preprocess_features  # noqa: E402
from services.feature_service import (  # noqa: E402
    calculate_raw_features,
    is_factor_v4_feature_version,
    uses_pit_market_values,
)
from services.label_service import build_labels  # noqa: E402
from services.pit_service import build_pit_snapshot  # noqa: E402
from settings import load_settings  # noqa: E402


def _atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _peak_working_set_bytes() -> int | None:
    if sys.platform != "win32":
        try:
            with open("/proc/self/status", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError):
            return None
        return None
    try:
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_Counters),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        if get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
    except Exception:  # noqa: BLE001 - profiling must not break panel builds.
        return None
    return None


def _lock(out: Path) -> Path:
    path = out / ".build_panels.lock"
    record = json.dumps({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()})
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"panel build already locked: {path}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(record)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _read_valid_panel(
    path: Path,
    feature_version: str,
    build_fp: str,
    source_fp: str,
    expected_signal_date: str | None = None,
    expected_purpose: str = "training",
) -> bool:
    try:
        with path.open("rb") as handle:
            panel = pickle.load(handle)
    except Exception:  # noqa: BLE001 - unreadable cache is rebuilt.
        return False
    if not isinstance(panel, dict):
        return False
    if (
        panel.get("panel_format_version") != PANEL_FORMAT_VERSION
        or panel.get("feature_version") != feature_version
        or panel.get("build_fingerprint") != build_fp
        or panel.get("source_fingerprint") != source_fp
        or not isinstance(panel.get("frame"), pd.DataFrame)
        or not isinstance(panel.get("labels"), pd.Series)
    ):
        return False
    if expected_signal_date is not None and panel.get("signal_date") != expected_signal_date:
        return False
    # Old panels predate this field and are training panels by definition.
    if panel.get("panel_purpose", "training") != expected_purpose:
        return False
    try:
        return panel.get("content_hash") == canonical_panel_hash(panel)
    except Exception:  # noqa: BLE001
        return False


def _write_panel(path: Path, panel: dict) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as handle:
        pickle.dump(panel, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _build_month(
    data: PreparedPanelData,
    signal_day: str,
    run_id: str,
    settings,
    *,
    require_labels: bool = True,
) -> dict:
    as_of = date.fromisoformat(signal_day)
    inputs = data.month_inputs(signal_day)
    universe = inputs["universe"]
    if not universe.included_ids:
        raise RuntimeError("empty universe")
    snapshot = build_pit_snapshot(
        universe,
        as_of,
        inputs["financials"],
        inputs["institutional"],
        inputs["prices"],
        market_values=(
            inputs["market_values"]
            if uses_pit_market_values(settings.features.feature_version)
            else None
        ),
    )
    eligibility = None
    if is_factor_v4_feature_version(settings.features.feature_version):
        # Historical training panels must be complete at t+20 before label
        # ranking.  The latest inference panel deliberately omits this gate:
        # future prices do not exist yet and are not model inputs.
        labelable = (
            set(data.labelable_ids(signal_day, list(universe.included_ids)))
            if require_labels
            else None
        )
        universe, eligibility = filter_factor_v4_universe(
            universe,
            inputs["stocks"],
            inputs["feature_prices"],
            inputs["financials"],
            as_of,
            settings,
            labelable_ids=labelable,
        )
        if not universe.included_ids:
            raise RuntimeError(
                f"empty factor_v4 eligible universe: {eligibility['excluded_counts']}"
            )
        keep = set(universe.included_ids)
        snapshot = snapshot.loc[snapshot["stock_id"].astype(str).isin(keep)].reset_index(drop=True)
    raw = calculate_raw_features(
        snapshot,
        inputs["feature_prices"],
        as_of,
        inputs["financials"],
        inputs["institutional"],
    )
    features = preprocess_features(raw, settings, run_id, as_of)
    if require_labels:
        labelable = data.labelable_ids(signal_day, list(universe.included_ids))
        labels = build_labels(inputs["labels_prices"], labelable, as_of, settings)
        if labels.empty:
            raise RuntimeError("no labels")
    else:
        labels = pd.Series(
            dtype="int64",
            index=pd.Index([], name="stock_id", dtype="object"),
            name=signal_day,
        )
    feature_ids = features.frame["stock_id"].astype(str).tolist()
    universe_ids = list(universe.included_ids)
    if feature_ids != universe_ids:
        raise RuntimeError("feature and eligible universe membership/order differ")
    if require_labels and labels.index.astype(str).tolist() != universe_ids:
        raise RuntimeError("label and eligible universe membership/order differ")
    if is_factor_v4_feature_version(settings.features.feature_version):
        minimum = max(30, settings.portfolio.top_n * 2)
        if len(universe_ids) < minimum:
            raise RuntimeError(
                f"factor_v4 eligible universe {len(universe_ids)} "
                f"is below release minimum {minimum}"
            )
    panel = {
        "signal_date": signal_day,
        "universe": list(universe.included_ids),
        "feature_columns": list(features.feature_columns),
        "feature_version": features.feature_version,
        "panel_purpose": "training" if require_labels else "inference",
        "feature_coverage": {
            column: rate
            for column, rate in features.coverage.items()
            if not column.endswith("__missing")
        },
        "frame": features.frame,
        "labels": labels,
    }
    if eligibility is not None:
        panel["eligibility"] = eligibility
    return panel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build monthly panels.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", default="2019-01")
    parser.add_argument("--end", default="2023-12")
    parser.add_argument("--out", default="data/panels")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help=(
            "Build feature-only panels without the future t+20 label gate. "
            "Use only for frozen-model inference, never for training."
        ),
    )
    parser.add_argument("--profile-json", default=None)
    parser.add_argument(
        "--schedule",
        choices=("month_end", "midmonth"),
        default="month_end",
        help="Signal schedule; midmonth uses the last trading day on or before the 15th.",
    )
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"cannot load config: {exc}", file=sys.stderr)
        return 2
    if len(args.start) != 7 or len(args.end) != 7 or args.start > args.end:
        print("--start and --end must be ordered YYYY-MM values", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock_path: Path | None = None
    profile: dict = {
        "start": args.start,
        "end": args.end,
        "feature_version": settings.features.feature_version,
        "schedule": args.schedule,
        "panel_purpose": "inference" if args.inference_only else "training",
        "stages_seconds": {},
        "months": {},
        "rows_loaded": {},
    }
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    try:
        lock_path = _lock(out)
        build_fp = build_fingerprint(settings)
        source_fp = source_fingerprint(settings)
        run_id = f"panels-{args.start}-{args.end}"
        data = PreparedPanelData(settings, args.start, args.end, run_id)
        try:
            months = (
                data.month_ends()
                if args.schedule == "month_end"
                else data.midmonth_signal_dates()
            )
            if not months:
                print(f"no {args.schedule} signal dates in range", file=sys.stderr)
                return 1
            profile["months_requested"] = len(months)
            print(f"{len(months)} months: {months[0]}..{months[-1]}", flush=True)
            pending = []
            for signal_day in months:
                dest = out / f"{signal_day[:7]}.pkl"
                if not args.force and _read_valid_panel(
                    dest,
                    settings.features.feature_version,
                    build_fp,
                    source_fp,
                    expected_signal_date=signal_day,
                    expected_purpose=("inference" if args.inference_only else "training"),
                ):
                    continue
                pending.append(signal_day)
            profile["months_cache_hit"] = len(months) - len(pending)
            profile["months_pending"] = len(pending)
            if not pending:
                print(
                    f"cache valid: {len(months)}/{len(months)}; no large source reads", flush=True
                )
                profile["no_op"] = True
                return 0

            # Reduce the requested range before loading data so a one-month
            # repair does not pay for all previously cached months.
            data.start = pending[0][:7]
            data.end = pending[-1][:7]
            stage_times = data.prepare(pending)
            profile["stages_seconds"].update(stage_times)
            profile["rows_loaded"] = dict(data.rows_loaded)
            if source_fingerprint(settings) != source_fp:
                raise RuntimeError(
                    "source data changed during preparation; rerun after sync completes"
                )
            done = len(months) - len(pending)
            failed: list[tuple[str, str]] = []
            for signal_day in pending:
                month_started = time.perf_counter()
                try:
                    panel = _build_month(
                        data,
                        signal_day,
                        run_id,
                        settings,
                        require_labels=not args.inference_only,
                    )
                    panel["panel_format_version"] = PANEL_FORMAT_VERSION
                    panel["build_fingerprint"] = build_fp
                    panel["source_fingerprint"] = source_fp
                    panel["content_hash"] = canonical_panel_hash(panel)
                    _write_panel(out / f"{signal_day[:7]}.pkl", panel)
                except Exception as exc:  # noqa: BLE001 - isolate months and resume.
                    failed.append((signal_day, f"{type(exc).__name__}: {exc}"))
                    traceback.print_exc()
                    profile["months"][signal_day] = {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "seconds": time.perf_counter() - month_started,
                    }
                    print(f"[{signal_day}] FAILED: {exc}", file=sys.stderr)
                    continue
                done += 1
                elapsed = time.perf_counter() - month_started
                profile["months"][signal_day] = {
                    "status": "succeeded",
                    "n_universe": len(panel["universe"]),
                    "n_labels": len(panel["labels"]),
                    "seconds": elapsed,
                }
                print(
                    f"[{done}/{len(months)}] {signal_day} "
                    f"n={len(panel['universe'])} {elapsed:.2f}s",
                    flush=True,
                )
            if not failed and source_fingerprint(settings) != source_fp:
                failed.append(("source", "source data changed while panels were being built"))
                print(
                    "source data changed while panels were being built; rerun after sync completes",
                    file=sys.stderr,
                )
            if not failed:
                manifest_path = out / ".panel_manifest.json"
                try:
                    previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    previous_manifest = {}
                merged_months = sorted(set(previous_manifest.get("months", [])) | set(months))
                month_sources = dict(previous_manifest.get("month_sources", {}))
                for source_metadata in month_sources.values():
                    if isinstance(source_metadata, dict):
                        source_metadata.setdefault("panel_purpose", "training")
                previous_build_fp = previous_manifest.get("build_fingerprint")
                previous_source_fp = previous_manifest.get("source_fingerprint")
                if previous_build_fp and previous_source_fp:
                    for old_month in previous_manifest.get("months", []):
                        month_sources.setdefault(
                            old_month,
                            {
                                "build_fingerprint": previous_build_fp,
                                "source_fingerprint": previous_source_fp,
                                "panel_purpose": "training",
                            },
                        )
                for built_month in months:
                    month_sources[built_month] = {
                        "build_fingerprint": build_fp,
                        "source_fingerprint": source_fp,
                        "panel_purpose": (
                            "inference" if args.inference_only else "training"
                        ),
                    }
                _atomic_json(
                    manifest_path,
                    {
                        "panel_format_version": PANEL_FORMAT_VERSION,
                        "build_fingerprint": build_fp,
                        "source_fingerprint": source_fp,
                        "schedule": args.schedule,
                        "latest_panel_purpose": (
                            "inference" if args.inference_only else "training"
                        ),
                        "months": merged_months,
                        "month_sources": month_sources,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            print(f"done {done}/{len(months)} failed={failed}")
            return 0 if not failed else 1
        finally:
            data.close()
    except Exception as exc:  # noqa: BLE001
        print(f"panel build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if lock_path is not None:
            try:
                lock_path.unlink()
            except OSError:
                pass
        profile["wall_seconds"] = time.perf_counter() - wall_started
        profile["cpu_seconds"] = time.process_time() - cpu_started
        profile["peak_working_set_bytes"] = _peak_working_set_bytes()
        if args.profile_json:
            try:
                _atomic_json(Path(args.profile_json), profile)
            except OSError as exc:
                print(f"cannot write profile JSON: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
