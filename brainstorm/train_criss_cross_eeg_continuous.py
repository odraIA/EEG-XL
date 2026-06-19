"""Continuous EEG training with the original MEG-XL model.

This entrypoint keeps the shared EEG training utilities, logging, checkpoints,
and model, but uses the continuity-aware DataModule. It intentionally follows
the original MEG-XL preprocessing order: filter at the source sampling rate and
then resample. Therefore it does not reject ``h_freq=40`` with a final sampling
rate of 50 Hz; MNE handles anti-alias filtering during resampling.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from brainstorm.data.eeg_continuous_masked_datamodule import MultiEEGDataModule
from brainstorm.models.criss_cross_transformer import CrissCrossTransformerModule
from brainstorm.neuro_tokenizers.factory import load_neuro_tokenizer
from brainstorm.train_criss_cross_eeg_multi import (
    MetricsFileCallback,
    SamplerVerificationCallback,
    install_tee,
    load_partial_checkpoint,
    resolve_save_dir,
    write_config_snapshot,
    write_final_results,
)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train_criss_cross_eeg_multi_continuous",
)
def main(cfg: DictConfig):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    save_dir = resolve_save_dir(cfg)
    log_file = install_tee(save_dir)
    config_snapshot = write_config_snapshot(save_dir, cfg)

    checkpoint_dir = Path(str(cfg.checkpoint.get("save_dir", "./checkpoints"))) / str(
        cfg.logging.get("experiment_name", "eeg_continuous_training")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback: Optional[ModelCheckpoint] = None
    metrics_callback = MetricsFileCallback(save_dir)
    tokenizer = None
    datamodule: Optional[MultiEEGDataModule] = None
    checkpoint_load_report = {"requested": False, "loaded": False}
    wandb_logger = None
    status = "failed"
    error_text: Optional[str] = None

    try:
        print("\n" + "=" * 80)
        print("CONTINUOUS EEG MEG-XL TRAINING")
        print("=" * 80)
        print("\n=== Configuration ===")
        print(OmegaConf.to_yaml(cfg))

        if float(cfg.data.target_sfreq) != float(cfg.model.sampling_rate):
            raise ValueError(
                f"data.target_sfreq ({cfg.data.target_sfreq}) must match "
                f"model.sampling_rate ({cfg.model.sampling_rate})."
            )

        print("✓ Config validation passed")
        print(
            "✓ MEG-XL preprocessing order: "
            "source-rate filtering followed by resampling"
        )

        torch.set_float32_matmul_precision("high")
        if hasattr(cfg, "seed"):
            pl.seed_everything(int(cfg.seed), workers=True)
            print(f"✓ Random seed set to {cfg.seed}")

        print("\n" + "=" * 80)
        print("SETTING UP CONTINUOUS EEG DATA")
        print("=" * 80)
        tokenizer_name = str(cfg.model.get("tokenizer_name", "biocodec"))
        datamodule = MultiEEGDataModule(
            datasets_config=OmegaConf.to_container(
                cfg.datasets_config,
                resolve=True,
            ),
            segment_length=float(cfg.data.segment_length),
            subsegment_duration=float(
                cfg.data.get("subsegment_duration", 3.0)
            ),
            words_per_segment=int(cfg.data.get("words_per_segment", 50)),
            window_onset_offset=float(
                cfg.data.get("window_onset_offset", -0.5)
            ),
            cache_dir=str(cfg.data.cache_dir),
            l_freq=float(cfg.data.l_freq),
            h_freq=float(cfg.data.h_freq),
            target_sfreq=float(cfg.data.target_sfreq),
            batch_size=int(cfg.training.batch_size),
            num_workers=int(cfg.training.num_workers),
            pin_memory=bool(cfg.training.pin_memory),
            persistent_workers=bool(cfg.training.persistent_workers),
            use_recording_sampler=bool(cfg.training.use_recording_sampler),
            sampler_seed=int(cfg.training.sampler_seed),
            debug_mode=bool(cfg.data.get("debug_mode", False)),
            max_channel_dim=cfg.data.get("max_channel_dim", None),
            infer_max_channel_dim=bool(
                cfg.data.get("infer_max_channel_dim", True)
            ),
            recording_subsample_prop=cfg.data.get(
                "recording_subsample_prop",
                None,
            ),
            allow_missing_word_alignment=bool(
                cfg.data.get("allow_missing_word_alignment", False)
            ),
            tokenizer_name=tokenizer_name,
        )
        datamodule.setup("fit")

        num_epochs = cfg.training.get("num_epochs", None)
        max_steps = cfg.training.get("max_steps", None)
        if num_epochs is not None and max_steps is not None:
            raise ValueError(
                "Set only one of training.num_epochs or training.max_steps."
            )
        if num_epochs is None and max_steps is None:
            raise ValueError(
                "Set either training.num_epochs or training.max_steps."
            )

        steps_per_epoch = len(datamodule.train_dataloader())
        training_steps = (
            int(max_steps)
            if max_steps is not None
            else int(num_epochs) * steps_per_epoch
        )
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {training_steps}")

        print("\n" + "=" * 80)
        print("LOADING TOKENIZER")
        print("=" * 80)
        tokenizer_checkpoint = cfg.model.get(
            "tokenizer_checkpoint",
            cfg.model.get("tokenizer_ckpt", None),
        )
        print(f"Tokenizer name: {tokenizer_name}")
        print(f"Tokenizer checkpoint: {tokenizer_checkpoint}")
        tokenizer = load_neuro_tokenizer(
            tokenizer_name=tokenizer_name,
            checkpoint_path=tokenizer_checkpoint,
            device="cpu",
        )
        print("✓ Tokenizer loaded")
        print(f"  RVQ levels: {tokenizer.n_q}")
        print(f"  Codebook size: {tokenizer.vocab_size}")
        print(f"  Downsample ratio: {tokenizer.downsample_ratio}")

        print("\n" + "=" * 80)
        print("CREATING MEG-XL MODEL")
        print("=" * 80)
        model = CrissCrossTransformerModule(
            tokenizer=tokenizer,
            latent_dim=int(cfg.model.latent_dim),
            num_layers=int(cfg.model.num_layers),
            num_heads=int(cfg.model.num_heads),
            vocab_size=int(cfg.model.vocab_size),
            learning_rate=float(cfg.training.learning_rate),
            warmup_steps=int(cfg.training.warmup_steps),
            training_steps=training_steps,
            mask_duration=float(cfg.model.get("mask_duration", 3.0)),
            num_subsegments_to_mask=int(
                cfg.model.get("num_subsegments_to_mask", 20)
            ),
            sampling_rate=int(cfg.model.sampling_rate),
            fourier_pos_dim=int(cfg.model.get("fourier_pos_dim", 250)),
            num_sensor_types=int(cfg.model.get("num_sensor_types", 3)),
        )

        if bool(cfg.model.get("use_gradient_checkpointing", False)):
            model.enable_gradient_checkpointing()

        train_from_scratch = bool(
            cfg.model.get("train_from_scratch", True)
        )
        init_checkpoint = (
            cfg.model.get("promoted_checkpoint", None)
            if bool(cfg.model.get("use_promoted_checkpoint", False))
            else cfg.model.get("criss_cross_checkpoint", None)
        )
        if train_from_scratch:
            checkpoint_load_report = {
                "requested": False,
                "loaded": False,
                "reason": "model.train_from_scratch=true",
            }
            print("✓ Training from scratch")
        else:
            checkpoint_load_report = load_partial_checkpoint(
                model,
                init_checkpoint,
            )
            print("Checkpoint load report:")
            print(json.dumps(checkpoint_load_report, indent=2)[:5000])

        print(
            f"Total parameters: "
            f"{sum(parameter.numel() for parameter in model.parameters()):,}"
        )
        print(
            f"Trainable parameters: "
            f"{sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}"
        )

        loggers = []
        if str(cfg.logging.get("wandb_project", "")):
            wandb_logger = WandbLogger(
                project=cfg.logging.wandb_project,
                entity=cfg.logging.get("wandb_entity", None),
                name=cfg.logging.experiment_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                save_dir=str(save_dir),
            )
            loggers.append(wandb_logger)
            print(
                f"✓ WandB logger: project="
                f"{cfg.logging.wandb_project}"
            )

        csv_logger = CSVLogger(
            save_dir=str(save_dir),
            name="lightning_csv",
        )
        loggers.append(csv_logger)
        print(f"✓ Local CSV logger: {save_dir / 'lightning_csv'}")

        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            SamplerVerificationCallback(),
            metrics_callback,
        ]
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="checkpoint-{epoch:02d}-{step:06d}",
            monitor=str(cfg.checkpoint.get("monitor", "val/loss")),
            mode=str(cfg.checkpoint.get("mode", "min")),
            every_n_train_steps=cfg.checkpoint.get(
                "every_n_train_steps",
                None,
            ),
            save_top_k=int(cfg.checkpoint.get("save_top_k", 1)),
            save_last=bool(cfg.checkpoint.get("save_last", True)),
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

        trainer_kwargs = {
            "accelerator": cfg.trainer.accelerator,
            "devices": cfg.trainer.devices,
            "precision": cfg.trainer.precision,
            "callbacks": callbacks,
            "logger": loggers,
            "gradient_clip_val": float(cfg.training.gradient_clip_val),
            "log_every_n_steps": int(cfg.logging.log_every_n_steps),
            "accumulate_grad_batches": int(
                cfg.trainer.accumulate_grad_batches
            ),
            "val_check_interval": cfg.trainer.val_check_interval,
            "deterministic": "warn" if hasattr(cfg, "seed") else False,
        }
        if cfg.trainer.get("strategy", None) is not None:
            trainer_kwargs["strategy"] = cfg.trainer.strategy
        if num_epochs is not None:
            trainer_kwargs["max_epochs"] = int(num_epochs)
        else:
            trainer_kwargs["max_steps"] = int(max_steps)

        trainer = pl.Trainer(**trainer_kwargs)

        ckpt_path = None
        if bool(cfg.checkpoint.get("resume", False)):
            ckpt_path = cfg.checkpoint.get("resume_path", None)
            if not ckpt_path:
                raise ValueError(
                    "checkpoint.resume=true but "
                    "checkpoint.resume_path is empty"
                )

        print("\n" + "=" * 80)
        print("STARTING CONTINUOUS EEG TRAINING")
        print("=" * 80)
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )
        status = "completed"

    except KeyboardInterrupt:
        status = "interrupted"
        error_text = "Training interrupted by user."
        print("\nTRAINING INTERRUPTED")
    except Exception:
        status = "failed"
        error_text = traceback.format_exc()
        print(error_text)
        raise
    finally:
        try:
            write_final_results(
                save_dir=save_dir,
                checkpoint_dir=checkpoint_dir,
                cfg=cfg,
                status=status,
                error=error_text,
                checkpoint_callback=checkpoint_callback,
                checkpoint_load_report=checkpoint_load_report,
                config_snapshot=config_snapshot,
                metrics_callback=metrics_callback,
                tokenizer=tokenizer if tokenizer is not None else object(),
                datamodule=datamodule,
            )
        finally:
            if datamodule is not None:
                datamodule.teardown("fit")
            if wandb_logger is not None:
                wandb_logger.experiment.finish()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


if __name__ == "__main__":
    main()
