#!/usr/bin/env python3
"""Smoke-test EEG word-aligned datasets without running training."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

MISSING_DEPENDENCY: str | None = None

try:
    import h5py
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    h5py = None
    np = None
    MISSING_DEPENDENCY = exc.name

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from brainstorm.data.eeg_word_aligned_dataset import (
        EEG_SENSOR_TYPE_ID,
        EEGDashWordAlignedDataset,
        OpenNeuroEEGWordAlignedDataset,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    EEG_SENSOR_TYPE_ID = 2
    EEGDashWordAlignedDataset = None
    OpenNeuroEEGWordAlignedDataset = None
    MISSING_DEPENDENCY = MISSING_DEPENDENCY or exc.name


REQUIRED_SAMPLE_KEYS = {
    "meg",
    "words",
    "subsegment_boundaries",
    "sensor_xyzdir",
    "sensor_types",
    "sensor_mask",
    "subject",
    "session",
    "task",
    "run",
    "dataset_name",
    "task_mode",
    "start_time",
    "end_time",
}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _write_cache(dataset: OpenNeuroEEGWordAlignedDataset, n_channels: int = 4, n_samples: int = 64) -> None:
    for rec_idx, rec in enumerate(dataset.recordings):
        cache_path = Path(rec["cache_path"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = np.linspace(0, 1, n_channels * n_samples, dtype=np.float32).reshape(n_channels, n_samples)
        data += rec_idx
        sensor_xyzdir = np.zeros((n_channels, 6), dtype=np.float32)
        sensor_xyzdir[:, 0] = np.linspace(-0.2, 0.2, n_channels, dtype=np.float32)
        sensor_xyzdir[:, 1] = np.linspace(0.2, -0.2, n_channels, dtype=np.float32)
        sensor_types = np.full(n_channels, EEG_SENSOR_TYPE_ID, dtype=np.int64)

        with h5py.File(cache_path, "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("sensor_xyzdir", data=sensor_xyzdir)
            f.create_dataset("sensor_types", data=sensor_types)
            f.create_dataset("channel_names", data=[f"EEG{idx}".encode("utf-8") for idx in range(n_channels)])
            f.attrs["sample_freq"] = float(dataset.target_sfreq)
            f.attrs["n_channels"] = n_channels
            f.attrs["n_samples"] = n_samples
            f.attrs["dataset_name"] = dataset.dataset_name
            f.attrs["task_mode"] = dataset.task_mode
            f.attrs["tokenizer_name"] = dataset.tokenizer_name


def _assert_sample(sample: dict[str, Any], *, dataset_name: str, task_mode: str, words: list[str]) -> None:
    missing = sorted(REQUIRED_SAMPLE_KEYS - set(sample))
    if missing:
        raise AssertionError(f"Sample is missing required keys: {missing}")
    if sample["dataset_name"] != dataset_name:
        raise AssertionError(f"Expected dataset_name={dataset_name}, got {sample['dataset_name']}")
    if sample["task_mode"] != task_mode:
        raise AssertionError(f"Expected task_mode={task_mode}, got {sample['task_mode']}")
    if list(sample["words"]) != words:
        raise AssertionError(f"Expected words={words}, got {sample['words']}")
    if int(sample["sensor_types"][0]) != EEG_SENSOR_TYPE_ID:
        raise AssertionError(f"Expected EEG sensor type id {EEG_SENSOR_TYPE_ID}")
    if sample["meg"].shape[0] != sample["sensor_mask"].shape[0]:
        raise AssertionError("sensor_mask length does not match signal channel count")
    if not bool(sample["sensor_xyzdir"].isfinite().all()):
        raise AssertionError("sensor_xyzdir contains non-finite values")


def _create_textgrid_ds004408(root: Path) -> None:
    raw = root / "sub-01" / "eeg" / "sub-01_task-listening_run-01_eeg.vhdr"
    events = root / "sub-01" / "eeg" / "sub-01_task-listening_run-01_events.tsv"
    textgrid = root / "stimuli" / "audio01.TextGrid"
    _touch(raw)
    _write_text(events, "onset\tduration\ttrial_type\n0.75\t1.00\taudio\n")
    _write_text(
        textgrid,
        "\n".join(
            [
                'xmin = 0.0',
                'xmax = 0.5',
                'text = "hello"',
                'xmin = 0.5',
                'xmax = 1.0',
                'text = "world"',
            ]
        ),
    )


def _create_ds007808(root: Path) -> None:
    listening_raw = root / "sub-01" / "eeg" / "sub-01_task-listening_run-01_eeg.edf"
    listening_events = root / "sub-01" / "eeg" / "sub-01_task-listening_run-01_events.tsv"
    covert_raw = root / "sub-01" / "eeg" / "sub-01_task-listeningcovert_run-01_eeg.edf"
    covert_events = root / "sub-01" / "eeg" / "sub-01_task-listeningcovert_run-01_events.tsv"
    speechopen_raw = root / "sub-01" / "eeg" / "sub-01_task-speechopen_run-01_eeg.edf"
    speechopen_events = root / "sub-01" / "eeg" / "sub-01_task-speechopen_run-01_events.tsv"

    for path in (listening_raw, covert_raw, speechopen_raw):
        _touch(path)
    _write_text(listening_events, "onset\tduration\tword\n0.75\t0.5\talpha\n1.25\t0.5\tbeta\n")
    _write_text(
        covert_events,
        "\n".join(
            [
                "onset\tduration\tphase\tword",
                "0.75\t0.5\tlisten\tcovert",
                "1.25\t0.5\tlisten\theard",
                "1.75\t0.5\tspeak\tignored",
            ]
        )
        + "\n",
    )
    _write_text(speechopen_events, "onset\tduration\tword\n0.75\t0.5\tmust_skip\n")


def smoke_openneuro(work_dir: Path) -> dict[str, Any]:
    ds004408_root = work_dir / "OpenNeuro_ds004408"
    ds007808_root = work_dir / "OpenNeuro_ds007808"
    cache_root = work_dir / "cache"
    _create_textgrid_ds004408(ds004408_root)
    _create_ds007808(ds007808_root)

    ds004408 = OpenNeuroEEGWordAlignedDataset(
        data_root=str(ds004408_root),
        dataset_name="openneuro_ds004408",
        task_mode="listening",
        tasks=["listening"],
        words_per_segment=2,
        subsegment_duration=1.0,
        target_sfreq=10,
        max_channel_dim=4,
        cache_dir=str(cache_root),
    )
    _write_cache(ds004408)
    ds004408_sample = ds004408[0]
    _assert_sample(
        ds004408_sample,
        dataset_name="openneuro_ds004408",
        task_mode="listening",
        words=["hello", "world"],
    )

    ds007808 = OpenNeuroEEGWordAlignedDataset(
        data_root=str(ds007808_root),
        dataset_name="openneuro_ds007808",
        task_mode="listening",
        tasks=["listening", "listeningcovert", "speechopen"],
        words_per_segment=2,
        subsegment_duration=1.0,
        target_sfreq=10,
        max_channel_dim=4,
        cache_dir=str(cache_root),
    )
    _write_cache(ds007808)
    words_by_task = {ds007808[idx]["task"]: list(ds007808[idx]["words"]) for idx in range(len(ds007808))}
    if "speechopen" in words_by_task:
        raise AssertionError("task-speechopen should be excluded from ds007808 listening smoke")
    if words_by_task.get("listening") != ["alpha", "beta"]:
        raise AssertionError(f"Unexpected ds007808 listening words: {words_by_task}")
    if words_by_task.get("listeningcovert") != ["covert", "heard"]:
        raise AssertionError(f"Unexpected ds007808 listeningcovert words: {words_by_task}")

    return {
        "ds004408": {
            "segments": len(ds004408),
            "words": ds004408_sample["words"],
            "shape": tuple(ds004408_sample["meg"].shape),
        },
        "ds007808": {
            "segments": len(ds007808),
            "words_by_task": words_by_task,
            "tasks": sorted(words_by_task),
        },
    }


def smoke_eegdash(eegdash_root: Path, work_dir: Path) -> dict[str, Any]:
    dataset = EEGDashWordAlignedDataset(
        data_root=str(eegdash_root),
        tasks=["delong"],
        words_per_segment=2,
        max_channel_dim=128,
        target_sfreq=25,
        cache_dir=str(work_dir / "eegdash_cache"),
        tokenizer_name="biocodec",
    )
    sample = dataset[0]
    _assert_sample(sample, dataset_name="eegdash", task_mode="reading", words=list(sample["words"]))
    return {
        "segments": len(dataset),
        "first_words": sample["words"],
        "shape": tuple(sample["meg"].shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="/tmp/scrabrain_eeg_word_dataset_smoke")
    parser.add_argument("--eegdash-root", default="datasets/eegdash/data/nm000228")
    parser.add_argument("--skip-eegdash", action="store_true")
    parser.add_argument("--skip-synthetic-openneuro", action="store_true")
    args = parser.parse_args()

    if MISSING_DEPENDENCY:
        print(json.dumps({
            "skipped": True,
            "reason": f"missing optional dependency: {MISSING_DEPENDENCY}",
        }, indent=2))
        return

    run_dir = Path(args.work_dir) / f"run_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=False)

    results: dict[str, Any] = {"work_dir": str(run_dir)}
    if not args.skip_synthetic_openneuro:
        results["synthetic_openneuro"] = smoke_openneuro(run_dir / "openneuro")

    eegdash_root = Path(args.eegdash_root)
    if not args.skip_eegdash:
        if eegdash_root.exists():
            results["eegdash"] = smoke_eegdash(eegdash_root, run_dir / "eegdash")
        else:
            results["eegdash"] = {
                "skipped": True,
                "reason": f"eegdash root not found: {eegdash_root}",
            }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
