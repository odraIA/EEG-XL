from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pywt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


@dataclass
class AugmentationConfig:
    temporal_shift: bool = False
    amplitude_jitter: bool = False
    frequency_masking: bool = False
    temporal_shift_frac: float = 0.10
    amplitude_jitter_range: float = 0.05
    freq_mask_max_frac: float = 0.15


@dataclass
class TFImageConfig:
    sfreq: float = 250.0
    n_freqs: int = 96
    f_min: float = 1.0
    f_max: float = 125.0
    wavelet: str = "cmor1.5-1.0"
    img_size: int = 224
    tf_variant: str = "full_band_tf"  # full_band_tf | low_freq_biased_tf
    low_freq_bias_strength: float = 0.8


class TFImageGenerator:
    def __init__(self, config: TFImageConfig):
        self.config = config
        self.frequencies = np.logspace(
            np.log10(config.f_min), np.log10(config.f_max), config.n_freqs
        )
        self.scales = pywt.frequency2scale(config.wavelet, self.frequencies / config.sfreq)

    def compute_scalograms(self, epoch: np.ndarray) -> np.ndarray:
        n_channels, t = epoch.shape
        scalograms = np.zeros((n_channels, self.config.n_freqs, t), dtype=np.float32)
        for ch in range(n_channels):
            coeffs, _ = pywt.cwt(epoch[ch], self.scales, self.config.wavelet)
            scalograms[ch] = np.abs(coeffs).astype(np.float32)

        mu = scalograms.mean(axis=(0, 2), keepdims=True)
        std = scalograms.std(axis=(0, 2), keepdims=True) + 1e-8
        scalograms = (scalograms - mu) / std
        return scalograms

    def apply_tf_variant(self, scalograms: np.ndarray) -> np.ndarray:
        if self.config.tf_variant == "full_band_tf":
            return scalograms
        if self.config.tf_variant == "low_freq_biased_tf":
            freq_rank = np.linspace(0.0, 1.0, self.config.n_freqs, dtype=np.float32)
            weights = 1.0 + self.config.low_freq_bias_strength * (1.0 - freq_rank)
            return scalograms * weights[None, :, None]
        raise ValueError(f"tf_variant desconocido: {self.config.tf_variant}")

    @staticmethod
    def apply_frequency_masking(
        scalograms: np.ndarray,
        rng: np.random.Generator,
        max_frac: float,
    ) -> np.ndarray:
        out = np.array(scalograms, copy=True)
        n_freqs = out.shape[1]
        max_width = max(1, int(n_freqs * max_frac))
        width = int(rng.integers(1, max_width + 1))
        start = int(rng.integers(0, max(1, n_freqs - width + 1)))
        out[:, start:start + width, :] = 0.0
        return out

    def project_pca3(self, scalograms: np.ndarray, pca_components: np.ndarray) -> np.ndarray:
        c, f, t = scalograms.shape
        flat = scalograms.reshape(c, -1)
        projected = pca_components @ flat
        return projected.reshape(3, f, t).astype(np.float32)

    @staticmethod
    def project_current_image_projection(scalograms: np.ndarray) -> np.ndarray:
        # Mantiene una ruta simple/compatible con el pipeline actual: 3 canales directos.
        return np.array(scalograms[:3], dtype=np.float32, copy=True)

    def resize_and_imagenet_normalize(self, maps_3: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(maps_3).unsqueeze(0)
        x = F.interpolate(
            x,
            size=(self.config.img_size, self.config.img_size),
            mode="bilinear",
            align_corners=False,
        )
        x = x.squeeze(0).numpy().astype(np.float32)
        x_min = x.min(axis=(1, 2), keepdims=True)
        x_max = x.max(axis=(1, 2), keepdims=True)
        x = (x - x_min) / (x_max - x_min + 1e-8)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x.astype(np.float32)


class MEGTFDataset(Dataset):
    def __init__(
        self,
        pnpl_dataset,
        preprocessor,
        generator: TFImageGenerator,
        projection_type: str,
        split: str,
        window_seconds: float,
        augment_cfg: AugmentationConfig,
        pca_components: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.dataset = pnpl_dataset
        self.preprocessor = preprocessor
        self.generator = generator
        self.projection_type = projection_type
        self.split = split
        self.window_seconds = float(window_seconds)
        self.augment_cfg = augment_cfg
        self.pca_components = pca_components
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def _to_scalar_label(self, raw_label) -> int:
        arr = np.asarray(raw_label)
        if arr.size == 1:
            return int(arr.reshape(-1)[0])
        return int(arr.astype(np.float32).mean() >= 0.5)

    def _window_epoch(self, epoch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        target_len = max(1, int(round(self.window_seconds * self.generator.config.sfreq)))
        if epoch.shape[1] <= target_len:
            return epoch
        if self.split == "train":
            start = int(rng.integers(0, epoch.shape[1] - target_len + 1))
        else:
            start = (epoch.shape[1] - target_len) // 2
        return epoch[:, start:start + target_len]

    def _augment_signal(self, epoch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = np.array(epoch, copy=True)
        if self.augment_cfg.temporal_shift:
            t = out.shape[1]
            max_shift = max(1, int(self.augment_cfg.temporal_shift_frac * t))
            shift = int(rng.integers(-max_shift, max_shift + 1))
            out = np.roll(out, shift, axis=1)
        if self.augment_cfg.amplitude_jitter:
            jitter = 1.0 + float(
                rng.uniform(-self.augment_cfg.amplitude_jitter_range, self.augment_cfg.amplitude_jitter_range)
            )
            out *= jitter
        return out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]
        epoch, raw_label = sample[0], sample[1]
        label = self._to_scalar_label(raw_label)

        epoch = np.asarray(epoch, dtype=np.float32)
        epoch = self.preprocessor(epoch)

        rng = np.random.default_rng(self.seed + idx)
        epoch = self._window_epoch(epoch, rng)
        if self.split == "train":
            epoch = self._augment_signal(epoch, rng)

        scalograms = self.generator.compute_scalograms(epoch)
        scalograms = self.generator.apply_tf_variant(scalograms)

        if self.split == "train" and self.augment_cfg.frequency_masking:
            scalograms = self.generator.apply_frequency_masking(
                scalograms,
                rng=rng,
                max_frac=self.augment_cfg.freq_mask_max_frac,
            )

        if self.projection_type == "learnable_1x1_projection":
            x = torch.from_numpy(scalograms.astype(np.float32))
        elif self.projection_type == "pca3_projection":
            if self.pca_components is None:
                raise ValueError("pca_components es obligatorio para pca3_projection")
            maps_3 = self.generator.project_pca3(scalograms, self.pca_components)
            image = self.generator.resize_and_imagenet_normalize(maps_3)
            x = torch.from_numpy(image.astype(np.float32))
        elif self.projection_type == "current_image_projection":
            maps_3 = self.generator.project_current_image_projection(scalograms)
            image = self.generator.resize_and_imagenet_normalize(maps_3)
            x = torch.from_numpy(image.astype(np.float32))
        else:
            raise ValueError(f"projection_type desconocido: {self.projection_type}")

        return x, torch.tensor(label, dtype=torch.long)


def fit_pca3_components(
    pnpl_train_dataset,
    preprocessor,
    generator: TFImageGenerator,
    window_seconds: float,
    max_samples: int = 32,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = min(len(pnpl_train_dataset), int(max_samples))
    idxs = rng.choice(len(pnpl_train_dataset), size=n, replace=False)

    n_channels = None
    total_rows = 0
    sum_x = None
    sum_xx = None

    for idx in idxs:
        sample = pnpl_train_dataset[int(idx)]
        epoch = np.asarray(sample[0], dtype=np.float32)
        epoch = preprocessor(epoch)

        target_len = max(1, int(round(window_seconds * generator.config.sfreq)))
        if epoch.shape[1] > target_len:
            start = int(rng.integers(0, epoch.shape[1] - target_len + 1))
            epoch = epoch[:, start:start + target_len]

        scalograms = generator.compute_scalograms(epoch)
        scalograms = generator.apply_tf_variant(scalograms)

        c, f, t = scalograms.shape
        if n_channels is None:
            n_channels = c
            sum_x = np.zeros((c,), dtype=np.float64)
            sum_xx = np.zeros((c, c), dtype=np.float64)

        x = scalograms.reshape(c, -1).T.astype(np.float64)
        total_rows += x.shape[0]
        sum_x += x.sum(axis=0)
        sum_xx += x.T @ x

    if total_rows == 0:
        raise RuntimeError("No se pudieron reunir muestras para PCA-3")

    mean = sum_x / total_rows
    cov = (sum_xx / total_rows) - np.outer(mean, mean)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    comps = eigvecs[:, order[:3]].T.astype(np.float32)
    return comps


def extract_binary_labels_fast(pnpl_dataset) -> np.ndarray:
    samples = getattr(pnpl_dataset, "samples", None)
    if not samples:
        labels = []
        for i in range(len(pnpl_dataset)):
            raw = pnpl_dataset[i][1]
            arr = np.asarray(raw)
            labels.append(int(arr.reshape(-1)[0] if arr.size == 1 else (arr.astype(np.float32).mean() >= 0.5)))
        return np.asarray(labels, dtype=np.int64)

    labels = []
    for sample in samples:
        raw = sample[5] if len(sample) > 5 else None
        if raw is None:
            continue
        arr = np.asarray(raw)
        labels.append(int(arr.reshape(-1)[0] if arr.size == 1 else (arr.astype(np.float32).mean() >= 0.5)))
    return np.asarray(labels, dtype=np.int64)
