"""Multi-dataset wrapper for EEG word-aligned datasets.

This mirrors the role of ``MultiMEGDataset`` but keeps the name EEG-specific.
Samples are not renamed: the underlying EEG datasets still return the signal
under the ``meg`` key because the current model/data pipeline expects that key.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import Dataset


class MultiEEGDataset(Dataset):
    """Combine several EEG datasets while preserving recording-level indexing.

    Each underlying dataset must expose:
    - ``segment_index``: list of ``(recording_idx, segment_idx)`` tuples
    - ``recordings``: recording metadata list
    - ``__getitem__`` returning a dict with at least ``meg``, ``sensor_xyzdir``,
      ``sensor_types`` and ``sensor_mask``.

    The combined dataset adds:
    - ``dataset_idx``
    - ``dataset_name``
    - ``modality = "eeg"``
    """

    def __init__(
        self,
        datasets: List[Dataset],
        dataset_names: Optional[List[str]] = None,
    ) -> None:
        if not datasets:
            raise ValueError("MultiEEGDataset requires at least one dataset")

        self.datasets = datasets
        self.dataset_names = dataset_names or [f"dataset_{i}" for i in range(len(datasets))]
        if len(self.datasets) != len(self.dataset_names):
            raise ValueError(
                f"Number of datasets ({len(self.datasets)}) must match number of "
                f"dataset names ({len(self.dataset_names)})"
            )

        self.segment_index: List[Tuple[int, int]] = []
        self.segment_to_dataset: List[int] = []
        self.cumulative_recordings: List[int] = [0]
        self._local_lookup: List[Dict[Tuple[int, int], int]] = []

        recording_offset = 0
        for dataset_idx, dataset in enumerate(self.datasets):
            if not hasattr(dataset, "segment_index"):
                raise ValueError(f"Dataset {dataset_idx} has no segment_index attribute")
            if not hasattr(dataset, "recordings"):
                raise ValueError(f"Dataset {dataset_idx} has no recordings attribute")

            local_lookup: Dict[Tuple[int, int], int] = {}
            for local_idx, (rec_idx, seg_idx) in enumerate(dataset.segment_index):
                rec_idx = int(rec_idx)
                seg_idx = int(seg_idx)
                local_lookup[(rec_idx, seg_idx)] = local_idx
                self.segment_index.append((rec_idx + recording_offset, seg_idx))
                self.segment_to_dataset.append(dataset_idx)

            self._local_lookup.append(local_lookup)
            recording_offset += len(dataset.recordings)
            self.cumulative_recordings.append(recording_offset)

    def __len__(self) -> int:
        return len(self.segment_index)

    def _get_dataset_local_idx(self, dataset_idx: int, rec_idx_local: int, seg_idx: int) -> int:
        try:
            return self._local_lookup[dataset_idx][(rec_idx_local, seg_idx)]
        except KeyError as exc:
            raise ValueError(
                f"Could not find segment (rec={rec_idx_local}, seg={seg_idx}) "
                f"inside dataset {self.dataset_names[dataset_idx]}"
            ) from exc

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_idx_adjusted, seg_idx = self.segment_index[idx]
        dataset_idx = self.segment_to_dataset[idx]
        recording_offset = self.cumulative_recordings[dataset_idx]
        rec_idx_local = rec_idx_adjusted - recording_offset

        dataset_local_idx = self._get_dataset_local_idx(dataset_idx, rec_idx_local, seg_idx)
        sample = dict(self.datasets[dataset_idx][dataset_local_idx])
        sample["dataset_idx"] = dataset_idx
        sample["dataset_name"] = self.dataset_names[dataset_idx]
        sample["modality"] = "eeg"
        return sample

    def get_segment_words(self, idx: int) -> List[str]:
        dataset_idx, local_idx = self._dataset_and_local_index(idx)
        dataset = self.datasets[dataset_idx]
        if hasattr(dataset, "get_segment_words"):
            return list(dataset.get_segment_words(local_idx))
        return list(dataset[local_idx]["words"])

    def get_segment_metadata(self, idx: int) -> Dict[str, Any]:
        dataset_idx, local_idx = self._dataset_and_local_index(idx)
        dataset = self.datasets[dataset_idx]
        if hasattr(dataset, "get_segment_metadata"):
            meta = dict(dataset.get_segment_metadata(local_idx))
        else:
            sample = dataset[local_idx]
            meta = {
                "dataset_name": sample.get("dataset_name", self.dataset_names[dataset_idx]),
                "subject": sample.get("subject", ""),
                "session": sample.get("session", ""),
                "task": sample.get("task", ""),
                "run": sample.get("run", ""),
            }
        meta["dataset_idx"] = dataset_idx
        meta["dataset_name"] = self.dataset_names[dataset_idx]
        return meta

    def get_split_group(self, idx: int, group_kind: str = "auto") -> str:
        dataset_idx, local_idx = self._dataset_and_local_index(idx)
        dataset = self.datasets[dataset_idx]
        if hasattr(dataset, "get_split_group"):
            return str(dataset.get_split_group(local_idx, group_kind))
        meta = self.get_segment_metadata(idx)
        if group_kind == "subject":
            return f"{meta['dataset_name']}:{meta.get('subject', '')}"
        if group_kind == "session":
            return f"{meta['dataset_name']}:{meta.get('subject', '')}:{meta.get('session', '')}"
        return (
            f"{meta['dataset_name']}:{meta.get('subject', '')}:"
            f"{meta.get('session', '')}:{meta.get('task', '')}:{meta.get('run', '')}"
        )

    def _dataset_and_local_index(self, idx: int) -> Tuple[int, int]:
        rec_idx_adjusted, seg_idx = self.segment_index[idx]
        dataset_idx = self.segment_to_dataset[idx]
        rec_idx_local = rec_idx_adjusted - self.cumulative_recordings[dataset_idx]
        local_idx = self._get_dataset_local_idx(dataset_idx, rec_idx_local, seg_idx)
        return dataset_idx, local_idx

    def close(self) -> None:
        for dataset in self.datasets:
            close = getattr(dataset, "close", None)
            if callable(close):
                close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["MultiEEGDataset"]
