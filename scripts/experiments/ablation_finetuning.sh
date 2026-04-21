#!/usr/bin/env bash
set -euo pipefail
python run_speech_image_experiments.py \
  --experiment ablation_finetuning \
  --data_path ./libribrain_data \
  --output_dir ./results/speech_image_experiments \
  --epochs 20 \
  --stage1_epochs 6 \
  --batch_size 32 \
  --num_workers 4 \
  --seeds 42,43,44 \
  --tf_variant full_band_tf
