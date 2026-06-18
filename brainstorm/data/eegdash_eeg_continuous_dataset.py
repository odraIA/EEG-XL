"""Continuous EEGDash NM000228 dataset for MEG-XL-style pre-training.

EEGDash materializes NM000228 as a BIDS-like tree of BioSemi BDF recordings and
sidecars. This adapter reuses the shared continuous EEG preprocessing and
segmentation pipeline while adding dataset-specific defaults and standard
10-20/10-05 sensor positions for the heterogeneous acquisition sites.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

import mne

from .eeg_word_aligned_dataset import EEGChannelCount, scan_bids_eeg_channel_counts
from .openneuro_eeg_continuous_dataset import OpenNeuroEEGContinuousDataset


EEGDASH_DATASET_ID = "nm000228"
EEGDASH_DEFAULT_TASK = "delong"


def _resolve_eegdash_root(data_root: str | Path) -> Path:
    """Resolve either the NM000228 directory or its EEGDash cache parent."""

    root = Path(data_root)
    if (root / EEGDASH_DATASET_ID).is_dir():
        return root / EEGDASH_DATASET_ID
    return root


def scan_eegdash_eeg_channel_counts(
    data_root: str | Path,
    tasks: Optional[Sequence[str]] = None,
) -> List[EEGChannelCount]:
    """Return genuine EEG channel counts for materialized NM000228 BDF files."""

    root = _resolve_eegdash_root(data_root)
    if tasks is None:
        tasks = [EEGDASH_DEFAULT_TASK]
    return scan_bids_eeg_channel_counts(root, tasks=tasks)


class EEGDashEEGContinuousDataset(OpenNeuroEEGContinuousDataset):
    """Continuous loader for the EEGDash ``NM000228`` reading dataset.

    NM000228 combines recordings from multiple laboratories, so channel counts
    and original sampling rates vary between subjects. The shared parent class
    reads the BIDS ``channels.tsv`` files, retains only channels typed as EEG,
    filters and resamples each recording, caches it as HDF5, and returns fixed
    duration windows under the MEG-compatible ``"meg"`` key.
    """

    def __init__(
        self,
        data_root: str | Path,
        *args: Any,
        dataset_name: str = "eegdash",
        tasks: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        if tasks is None:
            tasks = [EEGDASH_DEFAULT_TASK]
        super().__init__(
            data_root=str(_resolve_eegdash_root(data_root)),
            dataset_name=dataset_name,
            *args,
            tasks=tasks,
            **kwargs,
        )

    @staticmethod
    def _apply_eeg_montage(raw: mne.io.BaseRaw, raw_path: Path) -> None:
        """Attach standard positions to the site-specific 10-20/10-05 labels."""

        montage = mne.channels.make_standard_montage("standard_1005")
        montage_names = {name.casefold() for name in montage.ch_names}
        matched = sum(name.casefold() in montage_names for name in raw.ch_names)

        if matched:
            raw.set_montage(
                montage,
                match_case=False,
                match_alias=True,
                on_missing="ignore",
            )
            print(
                f"Applied standard_1005 montage to {raw_path.name}: "
                f"{matched}/{len(raw.ch_names)} EEG labels matched"
            )
            return

        OpenNeuroEEGContinuousDataset._apply_eeg_montage(raw, raw_path)


__all__ = [
    "EEGDASH_DATASET_ID",
    "EEGDASH_DEFAULT_TASK",
    "EEGDashEEGContinuousDataset",
    "scan_eegdash_eeg_channel_counts",
]
