#!/usr/bin/env python3
"""Download/cache EEGDash NM000228 recordings for local BIDS-style use.

The EEGDash API exposes lazy recording objects. Constructing ``NM000228`` only
queries metadata; accessing ``recording.raw`` downloads the raw file and BIDS
sidecars into the cache directory. This script performs that access for every
selected recording so the training dataset can later scan the cache on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


BAD_FILE_MARKERS = (
    "Bad BDF file provided",
    "file does not start with",
    "invalid literal for int()",
)


def _description_dict(description: Any) -> dict[str, Any]:
    if description is None:
        return {}
    if hasattr(description, "to_dict"):
        return description.to_dict()
    if isinstance(description, dict):
        return description
    return {}


def _parse_query(args: argparse.Namespace) -> dict[str, Any] | None:
    query: dict[str, Any] = {}
    if args.query:
        loaded = json.loads(args.query)
        if not isinstance(loaded, dict):
            raise ValueError("--query must decode to a JSON object")
        query.update(loaded)
    if args.subject:
        query["subject"] = {"$in": args.subject}
    if args.task:
        query["task"] = {"$in": args.task}
    return query or None


def _sidecar_status(raw_path: Path) -> dict[str, bool]:
    stem = raw_path.name.rsplit("_eeg.", 1)[0]
    return {
        "events": raw_path.with_name(f"{stem}_events.tsv").exists(),
        "channels": raw_path.with_name(f"{stem}_channels.tsv").exists(),
        "eeg_json": raw_path.with_name(f"{stem}_eeg.json").exists(),
    }


def _read_recording(recording: Any) -> tuple[Any, Path]:
    raw = recording.raw
    if raw is None:
        raise RuntimeError("recording.raw returned None")
    return raw, Path(recording.filecache)


def _should_retry_bad_file(exc: Exception) -> bool:
    message = repr(exc)
    return any(marker in message for marker in BAD_FILE_MARKERS)


def _remove_cached_recording_file(recording: Any) -> Path | None:
    raw_path = Path(recording.filecache)
    if raw_path.exists():
        raw_path.unlink()
        return raw_path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default="datasets/eegdash/data",
        help="EEGDash cache parent. NM000228 is cached below this directory.",
    )
    parser.add_argument(
        "--subject",
        action="append",
        help="Restrict to one subject. Repeat for multiple subjects. Default: all.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="Restrict to one task, for example delong. Repeat for multiple tasks. Default: all.",
    )
    parser.add_argument(
        "--query",
        help='Additional EEGDash query JSON, for example \'{"task": {"$in": ["delong"]}}\'.',
    )
    parser.add_argument("--limit", type=int, help="Download at most this many recordings.")
    parser.add_argument("--dry-run", action="store_true", help="Only list selected recordings.")
    parser.add_argument("--on-error", default="warn", choices=["raise", "warn", "ignore"])
    args = parser.parse_args()

    try:
        from eegdash import EEGDash  # noqa: F401
        from eegdash.dataset import NM000228
    except ImportError as exc:
        print(
            "Missing dependency 'eegdash'. Install it in the active environment, "
            "or run scripts/download_eegdash_docker.sh.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    query = _parse_query(args)

    print(f"cache_dir: {cache_dir}")
    print(f"dataset: NM000228")
    print(f"query: {query or '{}'}")

    dataset = NM000228(cache_dir=str(cache_dir), query=query, on_error=args.on_error)
    recordings = list(dataset.datasets)
    if args.limit is not None:
        recordings = recordings[: args.limit]

    print(f"metadata records: {len(dataset.records)}")
    print(f"selected recordings: {len(recordings)}")
    print(f"data_dir: {dataset.data_dir}")

    if args.dry_run:
        for idx, recording in enumerate(recordings):
            desc = _description_dict(recording.description)
            print(
                f"[{idx + 1:04d}/{len(recordings):04d}] "
                f"subject={desc.get('subject', '?')} "
                f"task={desc.get('task', '?')} "
                f"run={desc.get('run', '')} "
                f"file={recording.filecache}"
            )
        return 0

    started = time.time()
    downloaded = 0
    failed: list[dict[str, str]] = []
    missing_sidecars: list[dict[str, Any]] = []

    for idx, recording in enumerate(recordings):
        desc = _description_dict(recording.description)
        label = (
            f"subject={desc.get('subject', '?')} "
            f"task={desc.get('task', '?')} "
            f"run={desc.get('run', '')}"
        )
        print(f"[{idx + 1:04d}/{len(recordings):04d}] downloading {label}", flush=True)
        try:
            try:
                raw, raw_path = _read_recording(recording)
            except Exception as exc:
                if not _should_retry_bad_file(exc):
                    raise
                removed_path = _remove_cached_recording_file(recording)
                if removed_path is not None:
                    print(f"    removed corrupt cached file: {removed_path}", flush=True)
                print("    retrying download once", flush=True)
                raw, raw_path = _read_recording(recording)

            sidecars = _sidecar_status(raw_path)
            if not all(sidecars.values()):
                missing_sidecars.append({
                    "file": str(raw_path),
                    "sidecars": sidecars,
                })
            downloaded += 1
            print(
                f"    ok: {raw_path} "
                f"sfreq={raw.info.get('sfreq')} channels={len(raw.ch_names)}",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - depends on remote service/files.
            failed.append({"recording": label, "error": repr(exc)})
            print(f"    failed: {exc!r}", file=sys.stderr, flush=True)
            if args.on_error == "raise":
                raise

    elapsed = time.time() - started
    summary = {
        "dataset": "NM000228",
        "cache_dir": str(cache_dir),
        "data_dir": str(dataset.data_dir),
        "selected_recordings": len(recordings),
        "downloaded_recordings": downloaded,
        "failed_recordings": failed,
        "missing_sidecars": missing_sidecars,
        "elapsed_seconds": round(elapsed, 1),
    }
    print(json.dumps(summary, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
