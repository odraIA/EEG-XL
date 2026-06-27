"""Validated Alice evaluator entry point."""

from __future__ import annotations

import sys
from typing import Any, Dict

import torch
from omegaconf import DictConfig, open_dict

from brainstorm import evaluate_criss_cross_word_classification_alice_reported as base
from brainstorm import optimized_word_finetuning as optimized
from brainstorm.data.alice_eeg_word_aligned_dataset_missing_first_fix import (
    AliceEEGWordAlignedDatasetMissingFirstFix,
)

_BASE_STEP = optimized._optimized_step
_BASE_TRAIN = optimized.optimized_train_and_evaluate
base.AliceEEGWordAlignedDataset = AliceEEGWordAlignedDatasetMissingFirstFix


def _require_finite(name: str, value: torch.Tensor) -> None:
    bad = ~torch.isfinite(value)
    if bad.any():
        raise FloatingPointError(
            f"Non-finite values in {name}: {int(bad.sum().item())} / {value.numel()} "
            f"for shape {tuple(value.shape)}"
        )


def _checked_step(
    batch: Dict[str, Any],
    criss_cross_model,
    word_mlp,
    vocab_embeddings_device,
    criterion,
    device,
    downsample_ratio,
    mixed_precision,
    amp_dtype,
    features_only_forward,
):
    _require_finite("batch.meg", batch["meg"])
    _require_finite("batch.sensor_xyzdir", batch["sensor_xyzdir"])
    _require_finite("vocab_embeddings", vocab_embeddings_device)
    loss, predictions, targets = _BASE_STEP(
        batch,
        criss_cross_model,
        word_mlp,
        vocab_embeddings_device,
        criterion,
        device,
        downsample_ratio,
        mixed_precision,
        amp_dtype,
        features_only_forward,
    )
    _require_finite("predictions", predictions)
    _require_finite("targets", targets)
    _require_finite("loss", loss.reshape(1))
    return loss, predictions, targets


def _train_with_primary_top50(
    criss_cross_model,
    word_mlp,
    train_loader,
    val_loader,
    test_loader,
    vocab_embeddings,
    cfg: DictConfig,
    device,
    downsample_ratio,
):
    original = [int(value) for value in cfg.evaluation.retrieval_set_sizes]
    primary = int(cfg.evaluation.get("primary_retrieval_size", 50))
    if primary not in original:
        raise ValueError(
            f"Primary retrieval size {primary} is not in requested sizes {original}"
        )
    reordered = [value for value in original if value != primary] + [primary]
    with open_dict(cfg):
        cfg.evaluation.retrieval_set_sizes = reordered
    try:
        return _BASE_TRAIN(
            criss_cross_model,
            word_mlp,
            train_loader,
            val_loader,
            test_loader,
            vocab_embeddings,
            cfg,
            device,
            downsample_ratio,
        )
    finally:
        with open_dict(cfg):
            cfg.evaluation.retrieval_set_sizes = original


def _has_override(name: str) -> bool:
    prefix = f"{name}="
    return any(argument.startswith(prefix) for argument in sys.argv[1:])


def main():
    if not _has_override("data.eeg_sensor_type"):
        sys.argv.append("data.eeg_sensor_type=eeg")
    if not _has_override("evaluation.primary_retrieval_size"):
        sys.argv.append("+evaluation.primary_retrieval_size=50")
    optimized._optimized_step = _checked_step
    optimized.optimized_train_and_evaluate = _train_with_primary_top50
    return base.main()


if __name__ == "__main__":
    main()
