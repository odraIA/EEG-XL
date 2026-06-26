#!/usr/bin/env bash
set -Eeuo pipefail

# Sequential three-way Weissbart word-aligned fine-tuning comparison on one GPU.
#
# Order:
#   1. Randomly initialized CrissCross architecture.
#   2. EEG model originally trained from scratch (reading -> listening).
#   3. EEG model initialized from pretrained MEG-XL (reading -> listening).
#
# Each container performs fine-tuning and, at the end, reloads its best validation
# checkpoint and evaluates it on the Weissbart test split. The script then writes
# a CSV and Markdown comparison of the three final test evaluations.
#
# Run from the repository root:
#   bash scripts/run_weissbart_three_way_finetuning.sh
#
# Useful overrides:
#   GPU=0 WANDB_MODE=offline TRAIN_PCT=1.0 NUM_EPOCHS=50 \
#     bash scripts/run_weissbart_three_way_finetuning.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

WEISSBART_ROOT="${WEISSBART_ROOT:-./datasets/WeissbartEEG}"
BIOCODEC_CHECKPOINT="${BIOCODEC_CHECKPOINT:-./brainstorm/neuro_tokenizers/biocodec_ckpt.pt}"
MEGXL_ARCH_CHECKPOINT="${MEGXL_ARCH_CHECKPOINT:-./checkpoints/baseline/meg-xl-med.ckpt}"

SCRATCH_EEG_CHECKPOINT="${SCRATCH_EEG_CHECKPOINT:-./checkpoints/eeg_full_band_reading_then_listening_compare/20260624_182700/eeg_full_band_0p1_50_fixed50_50hz_biocodec_from_scratch_listening_seed42/checkpoint_best.pt}"
PRETRAINED_EEG_CHECKPOINT="${PRETRAINED_EEG_CHECKPOINT:-./checkpoints/eeg_full_band_reading_then_listening_compare/20260624_182700/eeg_full_band_0p1_50_fixed50_50hz_biocodec_pretrained_listening_seed42/checkpoint_best.pt}"

TRAIN_PCT="${TRAIN_PCT:-1.0}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"

RESULTS_ROOT="${RESULTS_ROOT:-./results/weissbart_three_way/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-./logs/word_classification_weissbart_eeg/$RUN_ID}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints/word_classification_weissbart_eeg/$RUN_ID}"
HYDRA_ROOT="${HYDRA_ROOT:-./logs/hydra/weissbart_three_way/$RUN_ID}"

mkdir -p "$RESULTS_ROOT" "$LOG_ROOT" "$CHECKPOINT_ROOT" "$HYDRA_ROOT"

require_file() {
  local path="$1"
  local description="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: Missing $description: $path" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  local description="$2"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: Missing $description: $path" >&2
    exit 2
  fi
}

require_dir "$WEISSBART_ROOT" "Weissbart dataset directory"
require_file "$BIOCODEC_CHECKPOINT" "BioCodec checkpoint"
require_file "$MEGXL_ARCH_CHECKPOINT" "MEG-XL architecture checkpoint"
require_file "$SCRATCH_EEG_CHECKPOINT" "EEG checkpoint originally trained from scratch"
require_file "$PRETRAINED_EEG_CHECKPOINT" "EEG checkpoint initialized from pretrained MEG-XL"

if [[ "$BUILD_IMAGE" == "1" || "$BUILD_IMAGE" == "true" ]]; then
  echo "Building Docker image..."
  docker compose build eval_eeg_listening
fi

CURRENT_EXPERIMENT=""
trap 'status=$?; echo "ERROR: experiment ${CURRENT_EXPERIMENT:-unknown} failed with exit code $status" >&2; exit "$status"' ERR

run_experiment() {
  local label="$1"
  local train_from_scratch="$2"
  local initialization_checkpoint="$3"

  local experiment_name="weissbart_${label}_${RUN_ID}"
  local container_name="weissbart_${label}_${RUN_ID}"
  local save_dir="$LOG_ROOT/$label"
  local checkpoint_dir="$CHECKPOINT_ROOT/$label"
  local hydra_dir="$HYDRA_ROOT/$label"

  CURRENT_EXPERIMENT="$experiment_name"
  mkdir -p "$save_dir" "$checkpoint_dir" "$hydra_dir"

  # Remove only a stale container with the exact generated name. Successful runs
  # use --rm, so normally there is nothing to remove here.
  docker rm -f "$container_name" >/dev/null 2>&1 || true

  echo
  echo "================================================================================"
  echo "START: $experiment_name"
  echo "GPU: $GPU"
  echo "Initialization checkpoint: $initialization_checkpoint"
  echo "Random initialization: $train_from_scratch"
  echo "Fine-tuning data: Weissbart word-aligned listening EEG"
  echo "Final evaluation: best validation checkpoint on the test split"
  echo "================================================================================"

  env \
    EEG_GPU="$GPU" \
    WANDB_MODE="$WANDB_MODE" \
    docker compose run --rm --no-deps \
      --name "$container_name" \
      -e "NVIDIA_VISIBLE_DEVICES=$GPU" \
      -e "WANDB_MODE=$WANDB_MODE" \
      eval_eeg_listening \
      uv run --no-sync python -m brainstorm.evaluate_criss_cross_word_classification_weissbart \
        --config-name=eval_criss_cross_word_classification_weissbart_eeg \
        "model.train_from_scratch=$train_from_scratch" \
        model.use_promoted_checkpoint=false \
        model.promoted_checkpoint=null \
        "model.criss_cross_checkpoint=$initialization_checkpoint" \
        model.tokenizer_name=biocodec \
        "model.tokenizer_checkpoint=$BIOCODEC_CHECKPOINT" \
        "data.root=$WEISSBART_ROOT" \
        "data.train_pct=$TRAIN_PCT" \
        "training.num_epochs=$NUM_EPOCHS" \
        "training.batch_size=$BATCH_SIZE" \
        "training.num_workers=$NUM_WORKERS" \
        'evaluation.retrieval_set_sizes=[250,50]' \
        evaluation.k=10 \
        "logging.experiment_name=$experiment_name" \
        "logging.save_dir=$save_dir" \
        "logging.checkpoint_dir=$checkpoint_dir" \
        "hydra.run.dir=$hydra_dir"

  local final_results="$save_dir/final_results.json"
  local best_checkpoint="$checkpoint_dir/checkpoint_best.pt"
  require_file "$final_results" "final test results for $experiment_name"
  require_file "$best_checkpoint" "best fine-tuned checkpoint for $experiment_name"

  echo "COMPLETED: $experiment_name"
  echo "Results: $final_results"
  echo "Best checkpoint: $best_checkpoint"
}

# The order below is intentional and matches the requested comparison.
run_experiment "random_init" true "$MEGXL_ARCH_CHECKPOINT"
run_experiment "eeg_from_scratch" false "$SCRATCH_EEG_CHECKPOINT"
run_experiment "eeg_pretrained" false "$PRETRAINED_EEG_CHECKPOINT"

CURRENT_EXPERIMENT="summary"

SUMMARY_CSV="$RESULTS_ROOT/weissbart_three_way_test_metrics.csv"
SUMMARY_MD="$RESULTS_ROOT/weissbart_three_way_test_metrics.md"

python3 - "$LOG_ROOT" "$CHECKPOINT_ROOT" "$SUMMARY_CSV" "$SUMMARY_MD" <<'PY'
import csv
import json
import sys
from pathlib import Path

log_root = Path(sys.argv[1])
checkpoint_root = Path(sys.argv[2])
csv_path = Path(sys.argv[3])
md_path = Path(sys.argv[4])

ordered_runs = [
    ("random_init", "Arquitectura aleatoria"),
    ("eeg_from_scratch", "Checkpoint EEG entrenado desde cero"),
    ("eeg_pretrained", "Checkpoint EEG inicializado desde MEG-XL"),
]

metric_keys = [
    "balanced_top10_accuracy_retrieval50",
    "top10_accuracy_retrieval50",
    "balanced_top10_accuracy_retrieval250",
    "top10_accuracy_retrieval250",
    "n_samples_retrieval50",
    "n_samples_retrieval250",
    "loss",
    "mean_cosine_similarity",
]

rows = []
for label, description in ordered_runs:
    result_path = log_root / label / "final_results.json"
    with result_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload.get("test_metrics", {})
    row = {
        "order": len(rows) + 1,
        "run": label,
        "initialization": description,
        "experiment_name": payload.get("experiment_name", ""),
        "checkpoint_best": str(checkpoint_root / label / "checkpoint_best.pt"),
        "final_results": str(result_path),
    }
    for key in metric_keys:
        row[key] = metrics.get(key, "")
    rows.append(row)

csv_path.parent.mkdir(parents=True, exist_ok=True)
fieldnames = list(rows[0].keys())
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

md_columns = [
    "order",
    "initialization",
    "balanced_top10_accuracy_retrieval50",
    "top10_accuracy_retrieval50",
    "balanced_top10_accuracy_retrieval250",
    "top10_accuracy_retrieval250",
    "loss",
    "mean_cosine_similarity",
]

def render(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)

with md_path.open("w", encoding="utf-8") as handle:
    handle.write("# Weissbart three-way final test comparison\n\n")
    handle.write("| " + " | ".join(md_columns) + " |\n")
    handle.write("| " + " | ".join(["---"] * len(md_columns)) + " |\n")
    for row in rows:
        handle.write("| " + " | ".join(render(row[column]) for column in md_columns) + " |\n")

print("\nFinal comparison:")
for row in rows:
    print(
        f"{row['order']}. {row['initialization']}: "
        f"balanced top-10@50={render(row['balanced_top10_accuracy_retrieval50'])}, "
        f"top-10@50={render(row['top10_accuracy_retrieval50'])}, "
        f"balanced top-10@250={render(row['balanced_top10_accuracy_retrieval250'])}"
    )
print(f"CSV: {csv_path}")
print(f"Markdown: {md_path}")
PY

echo
echo "================================================================================"
echo "ALL THREE WEISSBART RUNS COMPLETED"
echo "Run ID: $RUN_ID"
echo "Comparison CSV: $SUMMARY_CSV"
echo "Comparison Markdown: $SUMMARY_MD"
echo "================================================================================"
