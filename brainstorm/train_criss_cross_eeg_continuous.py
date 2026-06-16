"""Run the existing EEG trainer with the continuous EEG DataModule.

The original training script imports ``brainstorm.data.eeg_multi_datamodule``.
This small compatibility entrypoint swaps that module's exported class before
importing the trainer, avoiding a duplicated 600-line training script.
"""

from __future__ import annotations

import brainstorm.data.eeg_multi_datamodule as legacy_datamodule
from brainstorm.data.eeg_continuous_multi_datamodule import MultiEEGDataModule

legacy_datamodule.MultiEEGDataModule = MultiEEGDataModule

from brainstorm.train_criss_cross_eeg_multi import main  # noqa: E402


if __name__ == "__main__":
    main()
