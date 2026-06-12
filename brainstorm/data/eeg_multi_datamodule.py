"""PyTorch Lightning DataModule for OpenNeuro EEG word-aligned datasets.

This is the EEG equivalent of ``MultiMEGDataModule``. It is deliberately
conservative: no validation split is guessed. Every dataset config must specify
``val_subjects`` or ``val_sessions``.

Example
-------
>>> dm = MultiEEGDataModule(
...     datasets_config=[
...         {
...             "type": "openneuro_ds004408",
...             "data_root": "/workspace/datasets/OpenNeuroEEG_ds004408",
...             "tasks": ["listening"],
...             "val_subjects": ["sub-019"],
...         },
...         {
...             "type": "openneuro_ds007808",
...             "data_root": "/workspace/datasets/OpenNeuroEEG_ds007808",
...             "tasks": ["listening", "listeningcovert"],
...             "val_subjects": ["sub-03"],
...         },
...     ],
...     batch_size=8,
... )
>>> dm.setup("fit")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from .eeg_multi_dataset import MultiEEGDataset
from .openneuroEEG_ds004408_word_aligned_dataset import (
    OpenNeuroEEGDs004408WordAlignedDataset,
    DEFAULT_TASKS as DS004408_DEFAULT_TASKS,
)
from .openneuroEEG_ds007808_word_aligned_dataset import (
    OpenNeuroEEGDs007808WordAlignedDataset,
    DEFAULT_TASKS as DS007808_DEFAULT_TASKS,
)
from .eeg_word_aligned_dataset import scan_bids_eeg_channel_counts
from .subsampled_dataset import SubsampledRecordingDataset


_DATASET_ALIASES = {
    "ds004408": "openneuro_ds004408",
    "openneuro_ds004408": "openneuro_ds004408",
    "openneuroeeg_ds004408": "openneuro_ds004408",
    "openneuroEEG_ds004408": "openneuro_ds004408",
    "ds007808": "openneuro_ds007808",
    "openneuro_ds007808": "openneuro_ds007808",
    "openneuroeeg_ds007808": "openneuro_ds007808",
    "openneuroEEG_ds007808": "openneuro_ds007808",
}


def _as_list(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if values is None:
        return None
    return [str(value) for value in values]


def _norm_id(value: str, prefix: str) -> str:
    text = str(value)
    return text if text.startswith(prefix) else f"{prefix}{text}"


def _strip_prefix(value: str, prefix: str) -> str:
    text = str(value)
    return text[len(prefix):] if text.startswith(prefix) else text


def _sorted_dirs(root: Path, pattern: str) -> List[str]:
    return sorted(path.name for path in root.glob(pattern) if path.is_dir())


def _discover_subjects(data_root: str) -> List[str]:
    return _sorted_dirs(Path(data_root), "sub-*")


def _discover_sessions(data_root: str, subjects: Optional[Sequence[str]] = None) -> List[str]:
    root = Path(data_root)
    subject_names = list(subjects) if subjects is not None else _discover_subjects(data_root)
    sessions = set()
    for subject in subject_names:
        for session_dir in (root / _norm_id(subject, "sub-")).glob("ses-*"):
            if session_dir.is_dir():
                sessions.add(session_dir.name)
    return sorted(sessions)


def _exclude(values: Optional[Sequence[str]], excluded: Sequence[str], prefix: str) -> Optional[List[str]]:
    if values is None:
        return None
    excluded_norm = {_strip_prefix(_norm_id(item, prefix), prefix).lower() for item in excluded}
    kept = []
    for value in values:
        full = _norm_id(value, prefix)
        if _strip_prefix(full, prefix).lower() not in excluded_norm:
            kept.append(full)
    return kept


class MultiEEGDataModule(pl.LightningDataModule):
    """DataModule for multiple OpenNeuro EEG word-aligned datasets.

    Required per-dataset config keys:
    - ``type``: one of ``openneuro_ds004408`` or ``openneuro_ds007808``
    - ``data_root``: local dataset root
    - ``val_subjects`` or ``val_sessions``: explicit validation split

    Optional config keys:
    - ``subjects`` / ``sessions`` / ``tasks``
    - ``validation_only``
    - ``allow_missing_word_alignment``
    - ``tokenizer_name``
    """

    def __init__(
        self,
        datasets_config: List[Dict[str, Any]],
        segment_length: float = 150.0,
        subsegment_duration: float = 3.0,
        words_per_segment: int = 50,
        window_onset_offset: float = -0.5,
        cache_dir: str = "./data/cache/eeg",
        l_freq: float = 0.1,
        h_freq: float = 40.0,
        target_sfreq: float = 50.0,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        use_recording_sampler: bool = True,
        sampler_seed: int = 42,
        debug_mode: bool = False,
        max_channel_dim: Optional[int] = None,
        infer_max_channel_dim: bool = True,
        recording_subsample_prop: Optional[float] = None,
        allow_missing_word_alignment: bool = False,
        tokenizer_name: str = "biocodec",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.datasets_config = datasets_config
        self.segment_length = segment_length
        self.subsegment_duration = subsegment_duration
        self.words_per_segment = words_per_segment
        self.window_onset_offset = window_onset_offset
        self.cache_dir = cache_dir
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.target_sfreq = target_sfreq
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.use_recording_sampler = use_recording_sampler
        self.sampler_seed = sampler_seed
        self.debug_mode = debug_mode
        self.max_channel_dim = max_channel_dim
        self.infer_max_channel_dim = infer_max_channel_dim
        self.recording_subsample_prop = recording_subsample_prop
        self.allow_missing_word_alignment = allow_missing_word_alignment
        self.tokenizer_name = tokenizer_name

        if recording_subsample_prop is not None and not (0.0 < recording_subsample_prop <= 1.0):
            raise ValueError(
                f"recording_subsample_prop must be in (0.0, 1.0], got {recording_subsample_prop}"
            )

        self.train_dataset = None
        self.val_dataset = None

    def _canonical_type(self, dataset_type: str) -> str:
        key = str(dataset_type)
        return _DATASET_ALIASES.get(key, _DATASET_ALIASES.get(key.lower(), key))

    def _dataset_class_and_defaults(self, dataset_type: str):
        canonical = self._canonical_type(dataset_type)
        if canonical == "openneuro_ds004408":
            return OpenNeuroEEGDs004408WordAlignedDataset, DS004408_DEFAULT_TASKS
        if canonical == "openneuro_ds007808":
            return OpenNeuroEEGDs007808WordAlignedDataset, DS007808_DEFAULT_TASKS
        raise ValueError(
            f"Unknown EEG dataset type: {dataset_type}. Expected openneuro_ds004408 or openneuro_ds007808."
        )

    def _infer_max_channel_dim(self) -> Optional[int]:
        counts = []
        for config in self.datasets_config:
            _, default_tasks = self._dataset_class_and_defaults(config["type"])
            tasks = config.get("tasks", default_tasks)
            counts.extend(scan_bids_eeg_channel_counts(config["data_root"], tasks=tasks))

        if not counts:
            return None
        return max(item.n_channels for item in counts)

    def _resolve_max_channel_dim(self) -> Optional[int]:
        if self.max_channel_dim is not None:
            return self.max_channel_dim
        if not self.infer_max_channel_dim:
            return None
        inferred = self._infer_max_channel_dim()
        if inferred is None:
            print(
                "Could not infer max_channel_dim from materialized EEG headers/channels.tsv. "
                "Set max_channel_dim manually if datasets have different channel counts."
            )
        else:
            print(f"Inferred EEG max_channel_dim: {inferred}")
        return inferred

    def _split_filters(
        self,
        config: Dict[str, Any],
        split: str,
    ) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        data_root = config["data_root"]
        subjects = _as_list(config.get("subjects"))
        sessions = _as_list(config.get("sessions"))
        val_subjects = _as_list(config.get("val_subjects"))
        val_sessions = _as_list(config.get("val_sessions"))

        if val_subjects and val_sessions:
            raise ValueError(
                f"Dataset {config['type']} defines both val_subjects and val_sessions. "
                "Use only one split axis to avoid ambiguous leakage."
            )
        if not val_subjects and not val_sessions:
            raise ValueError(
                f"Missing validation split for {config['type']}. Add val_subjects=[...] "
                "or val_sessions=[...] to the dataset config. The provided .txt summaries do not "
                "define train/validation/test splits, so this is intentionally not guessed."
            )

        if val_subjects:
            val_subjects = [_norm_id(item, "sub-") for item in val_subjects]
            if subjects is None:
                subjects = _discover_subjects(data_root)
            subjects = [_norm_id(item, "sub-") for item in subjects]
            if split == "val":
                return val_subjects, sessions
            return _exclude(subjects, val_subjects, "sub-"), sessions

        assert val_sessions is not None
        val_sessions = [_norm_id(item, "ses-") for item in val_sessions]
        if sessions is None:
            sessions = _discover_sessions(data_root, subjects=subjects)
        sessions = [_norm_id(item, "ses-") for item in sessions]
        if split == "val":
            return subjects, val_sessions
        return subjects, _exclude(sessions, val_sessions, "ses-")

    def _create_dataset(self, config: Dict[str, Any], split: str, max_channel_dim: Optional[int]):
        dataset_cls, default_tasks = self._dataset_class_and_defaults(config["type"])
        subjects, sessions = self._split_filters(config, split=split)
        tasks = config.get("tasks", default_tasks)

        if self.debug_mode:
            if subjects:
                subjects = subjects[:1]
            if sessions:
                sessions = sessions[:1]

        return dataset_cls(
            data_root=config["data_root"],
            segment_length=config.get("segment_length", self.segment_length),
            subsegment_duration=config.get("subsegment_duration", self.subsegment_duration),
            words_per_segment=config.get("words_per_segment", self.words_per_segment),
            window_onset_offset=config.get("window_onset_offset", self.window_onset_offset),
            cache_dir=config.get("cache_dir", self.cache_dir),
            subjects=subjects,
            sessions=sessions,
            tasks=tasks,
            l_freq=config.get("l_freq", self.l_freq),
            h_freq=config.get("h_freq", self.h_freq),
            target_sfreq=config.get("target_sfreq", self.target_sfreq),
            channel_filter=config.get("channel_filter", None),
            max_channel_dim=max_channel_dim,
            baseline_duration=config.get("baseline_duration", 0.5),
            clip_range=config.get("clip_range", (-5, 5)),
            tokenizer_name=config.get("tokenizer_name", self.tokenizer_name),
            allow_missing_word_alignment=config.get(
                "allow_missing_word_alignment",
                self.allow_missing_word_alignment,
            ),
        )

    def _subsample_by_recordings(self, dataset, proportion: float, seed: int):
        rng = np.random.RandomState(seed)
        recording_ids = sorted(set(rec_idx for rec_idx, _ in dataset.segment_index))
        n_keep = max(1, int(round(len(recording_ids) * proportion)))
        keep_recordings = set(rng.choice(recording_ids, size=n_keep, replace=False).tolist())
        indices = [idx for idx, (rec_idx, _) in enumerate(dataset.segment_index) if rec_idx in keep_recordings]
        return SubsampledRecordingDataset(dataset, indices)

    @staticmethod
    def collate_fn(batch):
        eeg = torch.stack([item["meg"] for item in batch])
        sensor_xyzdir = torch.stack([item["sensor_xyzdir"] for item in batch])
        sensor_types = torch.stack([item["sensor_types"] for item in batch])
        sensor_mask = torch.stack([item["sensor_mask"] for item in batch])
        dataset_ids = torch.tensor([item["dataset_idx"] for item in batch], dtype=torch.long)
        return eeg, sensor_xyzdir, sensor_types, sensor_mask, dataset_ids

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit", "validate", "test", "predict"):
            return

        max_channel_dim = self._resolve_max_channel_dim()

        if stage in (None, "fit"):
            train_datasets = []
            train_names = []
            for config in self.datasets_config:
                if config.get("validation_only", False):
                    continue
                dataset = self._create_dataset(config, split="train", max_channel_dim=max_channel_dim)
                train_datasets.append(dataset)
                train_names.append(self._canonical_type(config["type"]))
                print(f"Training {config['type']}: {len(dataset)} segments")

            if not train_datasets:
                raise ValueError("No training datasets configured. Remove validation_only=True from at least one config.")

            self.train_dataset = MultiEEGDataset(train_datasets, train_names)
            if self.recording_subsample_prop is not None:
                self.train_dataset = self._subsample_by_recordings(
                    self.train_dataset,
                    proportion=self.recording_subsample_prop,
                    seed=self.sampler_seed,
                )
            print(f"Total EEG training segments: {len(self.train_dataset)}")

        if stage in (None, "fit", "validate", "test"):
            val_datasets = []
            val_names = []
            for config in self.datasets_config:
                dataset = self._create_dataset(config, split="val", max_channel_dim=max_channel_dim)
                val_datasets.append(dataset)
                val_names.append(self._canonical_type(config["type"]))
                print(f"Validation {config['type']}: {len(dataset)} segments")

            self.val_dataset = MultiEEGDataset(val_datasets, val_names)

            if self.debug_mode:
                total = len(self.val_dataset)
                subset_size = max(1, int(0.1 * total))
                rng = np.random.RandomState(self.sampler_seed)
                subset_indices = sorted(rng.choice(total, size=subset_size, replace=False))
                self.val_dataset = Subset(self.val_dataset, subset_indices)
                print(f"Debug mode: using {subset_size}/{total} validation segments")

            print(f"Total EEG validation segments: {len(self.val_dataset)}")

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before train_dataloader().")

        if self.use_recording_sampler:
            from .samplers import RecordingShuffleSampler
            sampler = RecordingShuffleSampler(self.train_dataset, seed=self.sampler_seed)
            shuffle = None
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') or setup('validate') before val_dataloader().")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

    def get_dataset_name_mapping(self) -> Dict[int, str]:
        dataset = self.val_dataset.dataset if isinstance(self.val_dataset, Subset) else self.val_dataset
        if dataset is None or not hasattr(dataset, "dataset_names"):
            return {}
        return {idx: name for idx, name in enumerate(dataset.dataset_names)}

    def teardown(self, stage: Optional[str] = None) -> None:
        for dataset in (self.train_dataset, self.val_dataset):
            base = dataset.dataset if isinstance(dataset, Subset) else dataset
            close = getattr(base, "close", None)
            if callable(close):
                close()


__all__ = ["MultiEEGDataModule"]
