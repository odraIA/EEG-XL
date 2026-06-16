#!/usr/bin/env python3
"""Inspect all listening intervals inside ds007808 listeningcovert events.tsv files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="/workspace/datasets/OpenNeuroEEG_ds007808",
    )
    parser.add_argument("--sessions", nargs="*", default=None)
    return parser.parse_args()


def merge_seconds(intervals):
    intervals = sorted((start, end) for start, end in intervals if end > start)
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    session_filter = set(args.sessions or [])

    totals = Counter()
    files = []

    for path in sorted(root.rglob("*task-listeningcovert*_events.tsv")):
        if session_filter and not any(session in path.parts for session in session_filter):
            continue

        events = pd.read_csv(path, sep="\t")
        trial_types = events["trial_type"].astype(str).str.strip().str.lower()
        listening = events.loc[trial_types.eq("listening")]
        covert = events.loc[trial_types.eq("covert")]

        intervals = []
        raw_duration = 0.0
        for _, row in listening.iterrows():
            try:
                onset = float(row["onset"])
                duration = float(row["duration"])
            except (TypeError, ValueError):
                continue
            if duration <= 0:
                continue
            intervals.append((onset, onset + duration))
            raw_duration += duration

        merged = merge_seconds(intervals)
        union_duration = sum(end - start for start, end in merged)

        item = {
            "file": str(path),
            "listening_rows": int(len(listening)),
            "covert_rows": int(len(covert)),
            "listening_duration_sum_seconds": raw_duration,
            "listening_union_seconds": union_duration,
            "merged_intervals": len(merged),
        }
        files.append(item)
        totals["files"] += 1
        totals["listening_rows"] += len(listening)
        totals["covert_rows"] += len(covert)
        totals["listening_duration_sum_seconds"] += raw_duration
        totals["listening_union_seconds"] += union_duration

    output = {
        "data_root": str(root),
        "sessions": sorted(session_filter) if session_filter else "all",
        "policy": {
            "included": "trial_type == listening",
            "excluded": "trial_type != listening, including covert",
            "interval": "[onset, onset + duration]",
        },
        "totals": dict(totals),
        "files": files,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
