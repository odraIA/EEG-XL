#!/usr/bin/env bash
set -euo pipefail
python run_speech_image_experiments.py \
  --experiment baseline_image_resnet18 \
  --data_path ./libribrain_data \
  --output_dir ./results/speech_image_experiments \
  --epochs 20 \
  --stage1_epochs 6 \
  --batch_size 32 \
  --num_workers 4 \
  --seeds 42,43,44 \
  --tf_variant low_freq_biased_tf
