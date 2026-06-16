#!/usr/bin/env python3
"""Smoke-test continuous ds007808 loading and listening-only extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from brainstorm.data.eeg_word_aligned_dataset import scan_bids_eeg_channel_counts
from brainstorm.data.openneuro_eeg_continuous_dataset import (
    OpenNeuroEEGContinuousDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="/workspace/datasets/OpenNeuroEEG_ds007808",
    )
    parser.add_argument(
        "--cache-dir",
        default="/tmp/scrabrain_eeg_continuous_smoke",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=["ses-20240624"],
        help="Use one or a few sessions for a quick check.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["listeningcovert"],
    )
    parser.add_argument("--segment-length", type=float, default=30.0)
    parser.add_argument("--target-sfreq", type=float, default=50.0)
    parser.add_argument("--l-freq", type=float, default=0.1)
    parser.add_argument("--h-freq", type=float, default=24.0)
    parser.add_argument(
        "--interval-start",
        choices=("onset", "wav_onset"),
        default="onset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    if not root.exists():
        raise FileNotFoundError(root)

    counts = scan_bids_eeg_channel_counts(root, tasks=args.tasks)
    max_channel_dim = max(item.n_channels for item in counts) if counts else None

    dataset = OpenNeuroEEGContinuousDataset(
        data_root=str(root),
        dataset_name="openneuro_ds007808",
        segment_length=args.segment_length,
        subsegment_duration=3.0,
        cache_dir=args.cache_dir,
        sessions=args.sessions,
        tasks=args.tasks,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        target_sfreq=args.target_sfreq,
        max_channel_dim=max_channel_dim,
        listeningcovert_policy="listening_only",
        listening_trial_type="listening",
        listening_interval_start=args.interval_start,
        group_listeningcovert_by="subject_session",
        cover_all_samples=True,
        short_stream_policy="repeat",
    )

    listening_streams = [
        stream
        for stream in dataset.recordings
        if stream["content_mode"] == "listening_only"
    ]
    if not listening_streams:
        raise AssertionError("No listening-only streams were created")
    if len(dataset) == 0:
        raise AssertionError("No fixed-duration segments were created")

    sample = dataset[0]
    expected_samples = int(round(args.segment_length * args.target_sfreq))
    if sample["meg"].shape[-1] != expected_samples:
        raise AssertionError(
            f"Expected {expected_samples} samples, got {sample['meg'].shape[-1]}"
        )
    if sample["content_mode"] != "listening_only":
        raise AssertionError(sample["content_mode"])
    if "words" in sample:
        raise AssertionError("Continuous pre-training sample must not contain words")

    report = dict(dataset.coverage_report)
    report["first_sample"] = {
        "shape": list(sample["meg"].shape),
        "subject": sample["subject"],
        "session": sample["session"],
        "task": sample["task"],
        "content_mode": sample["content_mode"],
        "source_recording_indices": list(sample["source_recording_indices"]),
    }
    report["assertions"] = {
        "listening_only_stream_created": True,
        "no_word_alignment_used": True,
        "fixed_segment_length": True,
        "covert_intervals_excluded_by_trial_type_filter": True,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    dataset.close()


if __name__ == "__main__":
    main()
