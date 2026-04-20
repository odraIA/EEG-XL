#!/usr/bin/env python3
"""
Precomputa imágenes MEG en disco para entrenar sin CWT/augmentación on-the-fly.

Salida esperada en --output_dir:
  - train_images.npy, train_labels.npy
  - validation_images.npy, validation_labels.npy
  - test_images.npy, test_labels.npy
  - train_source_idx.npy, train_is_augmented.npy
  - manifest.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

from meg_transfer_learning_libribrain import (
    AUG_AMPLITUDE_JITTER_PROB,
    AUG_AMPLITUDE_JITTER_RANGE,
    AUG_CHANNEL_DROPOUT_FRAC,
    AUG_CHANNEL_DROPOUT_PROB,
    AUG_TEMPORAL_SHIFT_FRAC,
    AUG_TEMPORAL_SHIFT_PROB,
    LibriBrainConfig,
    MEGPreprocessor,
    MEGToImage,
    apply_signal_augmentation,
    load_libribrain,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precomputar y guardar imágenes MEG para entrenamiento."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="phoneme",
        choices=["speech", "phoneme"],
        help="Tarea LibriBrain a precomputar.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./libribrain_data",
        help="Ruta raíz del dataset LibriBrain para pnpl.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./precomputed_images",
        help="Directorio de salida donde se guardan los .npy y manifest.json.",
    )
    parser.add_argument(
        "--n_freqs",
        type=int,
        default=96,
        help="Número de bins de frecuencia CWT (igual que entrenamiento).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Tamaño final de imagen (img_size x img_size).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16"],
        default="float32",
        help="Tipo de dato para guardar imágenes.",
    )
    parser.add_argument(
        "--train_augmented_copies",
        type=int,
        default=1,
        help=(
            "Cuántas versiones aumentadas guardar por sample de train. "
            "Las augmentaciones se aplican en señal antes de CWT."
        ),
    )
    parser.add_argument(
        "--skip_train_original",
        action="store_true",
        default=False,
        help="Si se activa, no guarda la versión no-augmentada en train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducir augmentaciones precomputadas.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Sobrescribir archivos existentes en output_dir.",
    )
    return parser.parse_args()


def _label_to_int(label) -> int:
    if hasattr(label, "item"):
        return int(label.item())
    return int(label)


def _prepare_output_dir(output_dir: Path, overwrite: bool):
    expected = [
        "train_images.npy",
        "train_labels.npy",
        "train_source_idx.npy",
        "train_is_augmented.npy",
        "validation_images.npy",
        "validation_labels.npy",
        "test_images.npy",
        "test_labels.npy",
        "manifest.json",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = [output_dir / name for name in expected if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{output_dir} ya contiene archivos precomputados. "
            "Usa --overwrite para reemplazarlos."
        )
    for path in existing:
        path.unlink()


def precompute_split(
    split_name: str,
    pnpl_dataset,
    preprocessor: MEGPreprocessor,
    img_converter: MEGToImage,
    output_dir: Path,
    image_dtype: np.dtype,
    rng: np.random.Generator,
    augmented_copies: int = 0,
    include_original: bool = True,
    save_provenance: bool = False,
) -> Dict[str, int]:
    n_base = len(pnpl_dataset)
    variants_per_sample = int(include_original) + augmented_copies
    if variants_per_sample <= 0:
        raise ValueError(f"{split_name}: variants_per_sample debe ser > 0")

    n_total = n_base * variants_per_sample

    images_path = output_dir / f"{split_name}_images.npy"
    labels_path = output_dir / f"{split_name}_labels.npy"
    source_idx_path = output_dir / f"{split_name}_source_idx.npy"
    is_aug_path = output_dir / f"{split_name}_is_augmented.npy"

    images_mm = open_memmap(
        images_path,
        mode="w+",
        dtype=image_dtype,
        shape=(n_total, 3, img_converter.img_size, img_converter.img_size),
    )
    labels_mm = open_memmap(labels_path, mode="w+", dtype=np.int64, shape=(n_total,))

    source_mm = None
    is_aug_mm = None
    if save_provenance:
        source_mm = open_memmap(source_idx_path, mode="w+", dtype=np.int64, shape=(n_total,))
        is_aug_mm = open_memmap(is_aug_path, mode="w+", dtype=np.uint8, shape=(n_total,))

    write_idx = 0
    for sample_idx in tqdm(range(n_base), desc=f"Precompute {split_name}", unit="sample"):
        sample = pnpl_dataset[sample_idx]
        epoch, label = sample[0], sample[1]
        label = _label_to_int(label)

        epoch = np.asarray(epoch, dtype=np.float32)
        epoch = preprocessor(epoch)

        if include_original:
            image = img_converter(epoch)
            images_mm[write_idx] = image.astype(image_dtype, copy=False)
            labels_mm[write_idx] = label
            if save_provenance:
                source_mm[write_idx] = sample_idx
                is_aug_mm[write_idx] = 0
            write_idx += 1

        for _ in range(augmented_copies):
            aug_epoch = apply_signal_augmentation(epoch, rng=rng)
            image = img_converter(aug_epoch)
            images_mm[write_idx] = image.astype(image_dtype, copy=False)
            labels_mm[write_idx] = label
            if save_provenance:
                source_mm[write_idx] = sample_idx
                is_aug_mm[write_idx] = 1
            write_idx += 1

    if write_idx != n_total:
        raise RuntimeError(f"{split_name}: escritos {write_idx}, esperados {n_total}")

    del images_mm
    del labels_mm
    if source_mm is not None:
        del source_mm
    if is_aug_mm is not None:
        del is_aug_mm

    return {
        "num_original_samples": n_base,
        "num_samples": n_total,
        "augmented_copies_per_sample": augmented_copies,
        "includes_original": int(include_original),
    }


def main():
    args = parse_args()

    if args.train_augmented_copies < 0:
        raise ValueError("--train_augmented_copies debe ser >= 0")
    if args.skip_train_original and args.train_augmented_copies == 0:
        raise ValueError("Debes guardar al menos una variante de train.")

    output_dir = Path(args.output_dir).resolve()
    _prepare_output_dir(output_dir, overwrite=args.overwrite)

    print("\n" + "=" * 72)
    print(" PRECOMPUTE DE IMÁGENES MEG (CWT + AUGMENTACIÓN EN SEÑAL)")
    print("=" * 72)
    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] Data path: {args.data_path}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] n_freqs={args.n_freqs} | img_size={args.img_size} | dtype={args.dtype}")
    print(
        "[INFO] Train variants: "
        f"original={'no' if args.skip_train_original else 'sí'}, "
        f"augmented_copies={args.train_augmented_copies}"
    )

    train_pnpl, _, _ = load_libribrain(LibriBrainConfig(args.data_path, args.task, "train"))
    val_pnpl, _, _ = load_libribrain(LibriBrainConfig(args.data_path, args.task, "validation"))
    test_pnpl, _, _ = load_libribrain(LibriBrainConfig(args.data_path, args.task, "test"))

    preprocessor = MEGPreprocessor(
        use_instance_norm=True,
        baseline_samples=None,
        clip_std=5.0,
    )
    img_converter = MEGToImage(
        sfreq=250.0,
        n_freqs=args.n_freqs,
        f_min=1.0,
        f_max=125.0,
        img_size=args.img_size,
        wavelet="cmor1.5-1.0",
        projection="pca",
    )

    rng = np.random.default_rng(args.seed)
    image_dtype = np.float32 if args.dtype == "float32" else np.float16

    split_stats: Dict[str, Dict[str, int]] = {}
    split_stats["train"] = precompute_split(
        split_name="train",
        pnpl_dataset=train_pnpl,
        preprocessor=preprocessor,
        img_converter=img_converter,
        output_dir=output_dir,
        image_dtype=image_dtype,
        rng=rng,
        augmented_copies=args.train_augmented_copies,
        include_original=not args.skip_train_original,
        save_provenance=True,
    )
    split_stats["validation"] = precompute_split(
        split_name="validation",
        pnpl_dataset=val_pnpl,
        preprocessor=preprocessor,
        img_converter=img_converter,
        output_dir=output_dir,
        image_dtype=image_dtype,
        rng=rng,
        augmented_copies=0,
        include_original=True,
        save_provenance=False,
    )
    split_stats["test"] = precompute_split(
        split_name="test",
        pnpl_dataset=test_pnpl,
        preprocessor=preprocessor,
        img_converter=img_converter,
        output_dir=output_dir,
        image_dtype=image_dtype,
        rng=rng,
        augmented_copies=0,
        include_original=True,
        save_provenance=False,
    )

    manifest = {
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "task": args.task,
        "data_path": str(Path(args.data_path).resolve()),
        "output_dir": str(output_dir),
        "n_freqs": args.n_freqs,
        "img_size": args.img_size,
        "dtype": args.dtype,
        "seed": args.seed,
        "augmentation": {
            "temporal_shift_prob": AUG_TEMPORAL_SHIFT_PROB,
            "temporal_shift_frac": AUG_TEMPORAL_SHIFT_FRAC,
            "amplitude_jitter_prob": AUG_AMPLITUDE_JITTER_PROB,
            "amplitude_jitter_range": AUG_AMPLITUDE_JITTER_RANGE,
            "channel_dropout_prob": AUG_CHANNEL_DROPOUT_PROB,
            "channel_dropout_frac": AUG_CHANNEL_DROPOUT_FRAC,
            "train_augmented_copies": args.train_augmented_copies,
            "include_train_original": int(not args.skip_train_original),
        },
        "splits": split_stats,
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n[OK] Precompute finalizado.")
    print(f"[OK] Manifest: {manifest_path}")
    for split, info in split_stats.items():
        print(f"  - {split}: {info['num_samples']} imágenes")


if __name__ == "__main__":
    main()
