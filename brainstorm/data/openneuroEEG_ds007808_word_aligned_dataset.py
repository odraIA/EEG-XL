"""OpenNeuro ds007808 EEG word-aligned dataset.

This module is a dataset-specific wrapper around the generic
``OpenNeuroEEGWordAlignedDataset`` already present in ``brainstorm.data``.

Grounded dataset facts from the provided dataset summary:
- OpenNeuro id: ds007808
- Raw EEG format: EDF ``*_eeg.edf``
- Tasks detected in the summary: listening, listeningcovert, speechopen
- ``speechopen`` is not included in the default task list here because it is a
  speech-production/open-speech task, while this wrapper is for listening EEG.
- Subjects in the summary: sub-01, sub-02, sub-03

No train/validation split is hard-coded here because the provided summary does
not define one. Choose it explicitly in ``eeg_multi_datamodule.py`` config.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from .eeg_word_aligned_dataset import OpenNeuroEEGWordAlignedDataset


DATASET_ID = "ds007808"
DATASET_NAME = "openneuroEEG_ds007808"
TASK_MODE = "listening"
DEFAULT_TASKS: List[str] = ["listening", "listeningcovert"]
EXCLUDED_TASKS: List[str] = ["speechopen"]

# Intentionally blank: the provided .txt summary does not define a validation split.
DEFAULT_VAL_SUBJECTS: List[str] = []
DEFAULT_VAL_SESSIONS: List[str] = []


class OpenNeuroEEGDs007808WordAlignedDataset(OpenNeuroEEGWordAlignedDataset):
    """Dataset-specific wrapper for OpenNeuro ds007808.

    By default, this wrapper uses only ``listening`` and ``listeningcovert``.
    Pass ``tasks=[...]`` explicitly if you want a narrower selection.
    """

    def __init__(
        self,
        data_root: str,
        segment_length: float = 150.0,
        subsegment_duration: float = 3.0,
        words_per_segment: int = 50,
        window_onset_offset: float = -0.5,
        cache_dir: str = "./data/cache/eeg",
        subjects: Optional[Sequence[str]] = None,
        sessions: Optional[Sequence[str]] = None,
        tasks: Optional[Sequence[str]] = None,
        l_freq: float = 0.1,
        h_freq: float = 40.0,
        target_sfreq: float = 50.0,
        channel_filter=None,
        max_channel_dim: Optional[int] = None,
        baseline_duration: float = 0.5,
        clip_range: tuple = (-5, 5),
        tokenizer_name: str = "biocodec",
        allow_missing_word_alignment: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data_root=data_root,
            dataset_name=DATASET_NAME,
            task_mode=TASK_MODE,
            segment_length=segment_length,
            subsegment_duration=subsegment_duration,
            words_per_segment=words_per_segment,
            window_onset_offset=window_onset_offset,
            cache_dir=cache_dir,
            subjects=list(subjects) if subjects is not None else None,
            sessions=list(sessions) if sessions is not None else None,
            tasks=list(tasks) if tasks is not None else list(DEFAULT_TASKS),
            l_freq=l_freq,
            h_freq=h_freq,
            target_sfreq=target_sfreq,
            channel_filter=channel_filter,
            max_channel_dim=max_channel_dim,
            baseline_duration=baseline_duration,
            clip_range=clip_range,
            tokenizer_name=tokenizer_name,
            allow_missing_word_alignment=allow_missing_word_alignment,
            **kwargs,
        )


# Compatibility aliases for configs/imports that use the dataset id literally.
OpenNeuroEEG_ds007808_WordAlignedDataset = OpenNeuroEEGDs007808WordAlignedDataset
OpenNeuroEEGDS007808WordAlignedDataset = OpenNeuroEEGDs007808WordAlignedDataset


__all__ = [
    "DATASET_ID",
    "DATASET_NAME",
    "TASK_MODE",
    "DEFAULT_TASKS",
    "EXCLUDED_TASKS",
    "DEFAULT_VAL_SUBJECTS",
    "DEFAULT_VAL_SESSIONS",
    "OpenNeuroEEGDs007808WordAlignedDataset",
    "OpenNeuroEEG_ds007808_WordAlignedDataset",
    "OpenNeuroEEGDS007808WordAlignedDataset",
]
