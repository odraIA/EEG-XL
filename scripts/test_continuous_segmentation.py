#!/usr/bin/env python3
"""Small dependency-light tests for continuous EEG segmentation helpers."""

from brainstorm.data.openneuro_eeg_continuous_dataset import (
    merge_intervals,
    segment_starts_cover_all,
)


def main() -> None:
    assert merge_intervals([(0, 10), (8, 20), (30, 40)]) == [(0, 20), (30, 40)]
    assert merge_intervals([(0, 10), (12, 20)], max_gap_samples=2) == [(0, 20)]

    assert segment_starts_cover_all(100, 100) == [0]
    assert segment_starts_cover_all(80, 100) == [0]
    assert segment_starts_cover_all(250, 100) == [0, 100, 150]
    assert segment_starts_cover_all(300, 100) == [0, 100, 200]

    print("Continuous EEG segmentation helper tests passed.")


if __name__ == "__main__":
    main()
