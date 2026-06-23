#!/usr/bin/env bash
set -uo pipefail

# Run exactly two continuous-listening experiments:
#   1) MEG-XL pretrained initialization
#   2) random initialization
#
# Both use full-band 0.1-50 Hz, filter-then-resample to 50 Hz and BioCodec.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

TRAIN_COMPOSE="${EEG_COMPOSE_FILE:-${ROOT_DIR}/docker-compose.eeg-reading-listening.yml}"
PREPROCESS_COMPOSE="${EEG_PREPROCESS_COMPOSE_FILE:-${ROOT_DIR}/docker-compose.eeg-preprocess.yml}"
TRAIN_SERVICE="${EEG_TRAIN_SERVICE:-eeg_train_reading_listening}"
PREPROCESS_SERVICE="${EEG_PREPROCESS_SERVICE:-eeg_preprocess}"
PYTHON_MODULE="${EEG_TRAIN_MODULE:-brainstorm.train_criss_cross_eeg_continuous}"
CONFIG_NAME="${EEG_LISTENING_CONFIG:-train_criss_cross_eeg_listening_continuous}"

SEED="${EEG_SEED:-42}"
BATCH_SIZE="${EEG_BATCH_SIZE:-4}"
NUM_WORKERS="${EEG_NUM_WORKERS:-6}"
VAL_CHECK_INTERVAL="${EEG_VAL_CHECK_INTERVAL:-500}"
CHECKPOINT_INTERVAL="${EEG_CHECKPOINT_EVERY_N_TRAIN_STEPS:-5000}"
WANDB_MODE="${WANDB_MODE:-offline}"
CACHE_DIR="${EEG_CACHE_DIR:-./data/cache/eeg_preprocessed}"
CRISS_CROSS_CHECKPOINT="${CRISS_CROSS_CHECKPOINT:-./checkpoints/baseline/meg-xl-med.ckpt}"

read -r -a GPUS <<< "${EEG_GPUS:-0 1}"
if [[ ${#GPUS[@]} -lt 2 ]]; then
  echo 'EEG_GPUS must contain two GPU ids, for example EEG_GPUS="0 1".' >&2
  exit 2
fi

GPU_PRETRAINED="${GPUS[0]}"
GPU_SCRATCH="${GPUS[1]}"
STAMP="${EEG_SWEEP_STAMP:-$(date +%Y%m%d_%H%M%S)}"

PRETRAINED_EXP="eeg_full_band_0p1_50_fixed50_50hz_biocodec_pretrained_listening_seed${SEED}"
SCRATCH_EXP="eeg_full_band_0p1_50_fixed50_50hz_biocodec_from_scratch_listening_seed${SEED}"

SWEEP_ROOT="${EEG_SWEEP_ROOT:-results/eeg_full_band_listening_compare/${STAMP}}"
TRAIN_LOG_ROOT="${EEG_TRAIN_LOG_ROOT:-./logs/eeg_full_band_listening_compare/${STAMP}}"
CHECKPOINT_ROOT="${EEG_CHECKPOINT_ROOT:-./checkpoints/eeg_full_band_listening_compare/${STAMP}}"
STAGING_CACHE="${EEG_STAGING_CACHE:-./data/cache/eeg_full_band_listening_compare_staging/${STAMP}}"

mkdir -p "$SWEEP_ROOT" "$TRAIN_LOG_ROOT" "$CHECKPOINT_ROOT" "$STAGING_CACHE"
RUNS_FILE="${SWEEP_ROOT}/runs.tsv"
printf 'experiment\tgpu\tinitialization\tstatus\texit_code\tlog\n' > "$RUNS_FILE"

sanitize() {
  printf '%s' "$1" | tr -c '[:alnum:]_.-' '_'
}

gpu_busy() {
  nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$'
}

if [[ "${EEG_ALLOW_BUSY_GPUS:-false}" != "true" ]]; then
  for gpu in "$GPU_PRETRAINED" "$GPU_SCRATCH"; do
    if gpu_busy "$gpu"; then
      echo "GPU ${gpu} is already busy. Stop the current sweep or choose two free GPUs." >&2
      echo 'Use EEG_ALLOW_BUSY_GPUS=true only if sharing the GPU is intentional.' >&2
      exit 2
    fi
  done
fi

cat > "${SWEEP_ROOT}/metadata.txt" <<EOF
Started: $(date -Iseconds)
Pretrained experiment: ${PRETRAINED_EXP}
From-scratch experiment: ${SCRATCH_EXP}
GPUs: ${GPU_PRETRAINED} ${GPU_SCRATCH}
Batch size: ${BATCH_SIZE}
Band: 0.1-50 Hz
Target sampling rate: 50 Hz
Tokenizer: BioCodec
Shared cache: ${CACHE_DIR}
Training logs: ${TRAIN_LOG_ROOT}
Checkpoints: ${CHECKPOINT_ROOT}
EOF

echo "Building Docker services..."
docker compose -f "$PREPROCESS_COMPOSE" build "$PREPROCESS_SERVICE" || exit $?
docker compose -f "$TRAIN_COMPOSE" build "$TRAIN_SERVICE" || exit $?

echo "Preparing the shared full-band listening cache once..."
PREPROCESS_LOG="${SWEEP_ROOT}/preprocessing.log"
docker compose -f "$PREPROCESS_COMPOSE" run --rm --no-deps \
  "$PREPROCESS_SERVICE" \
  uv run --no-sync python scripts/preprocess_eeg_reading_listening.py \
    --config-name "$CONFIG_NAME" \
    --target-sfreq 50 \
    --l-freq 0.1 \
    --h-freq 50 \
    --cache-dir "$STAGING_CACHE" \
    --main-cache-dir "$CACHE_DIR" \
  > "$PREPROCESS_LOG" 2>&1 || {
    echo "Preprocessing failed: ${PREPROCESS_LOG}" >&2
    exit 1
  }

tokenizer_overrides=(
  'model.tokenizer_name=biocodec'
  'model.tokenizer_variant=default'
  'model.tokenizer_checkpoint=./brainstorm/neuro_tokenizers/biocodec_ckpt.pt'
  'model.tokenizer_ckpt=./brainstorm/neuro_tokenizers/biocodec_ckpt.pt'
  'model.tokenizer_config_path=null'
  'model.vocab_size=256'
  'model.num_quantizers=6'
  'model.num_quantizers_used=6'
  'model.tokenizer_downsample_ratio=12'
  'model.overlap_ratio=0.0'
)

run_training() {
  local gpu="$1" experiment="$2" initialization="$3"
  local log_path="${SWEEP_ROOT}/${experiment}.log"
  local container_name
  local -a init_overrides command

  container_name="$(sanitize "scrabrain_${experiment}")"

  if [[ "$initialization" == "pretrained" ]]; then
    init_overrides=(
      'model.train_from_scratch=false'
      'model.use_promoted_checkpoint=false'
      "model.criss_cross_checkpoint=${CRISS_CROSS_CHECKPOINT}"
    )
  else
    init_overrides=(
      'model.train_from_scratch=true'
      'model.use_promoted_checkpoint=false'
    )
  fi

  command=(
    uv run --no-sync python -m "$PYTHON_MODULE"
    --config-name "$CONFIG_NAME"
    'data.target_sfreq=50.0'
    'model.sampling_rate=50'
    'data.l_freq=0.1'
    'data.h_freq=50.0'
    "data.cache_dir=${CACHE_DIR}"
    "training.batch_size=${BATCH_SIZE}"
    "training.num_workers=${NUM_WORKERS}"
    'training.persistent_workers=true'
    'trainer.devices=1'
    'trainer.strategy=auto'
    "trainer.val_check_interval=${VAL_CHECK_INTERVAL}"
    "checkpoint.every_n_train_steps=${CHECKPOINT_INTERVAL}"
    "logging.experiment_name=${experiment}"
    "logging.save_dir=${TRAIN_LOG_ROOT}"
    "checkpoint.save_dir=${CHECKPOINT_ROOT}"
    "seed=${SEED}"
    "${tokenizer_overrides[@]}"
    "${init_overrides[@]}"
  )

  local quoted status
  printf -v quoted '%q ' "${command[@]}"

  echo "[GPU ${gpu}] Starting ${experiment}"
  EEG_GPU="$gpu" EEG_GPU_COUNT=1 WANDB_MODE="$WANDB_MODE" \
    docker compose -f "$TRAIN_COMPOSE" run --rm --no-deps \
      --name "$container_name" \
      -e "WANDB_MODE=${WANDB_MODE}" \
      "$TRAIN_SERVICE" bash -lc "$quoted" \
      > "$log_path" 2>&1
  status=$?

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$experiment" "$gpu" "$initialization" \
    "$([[ $status -eq 0 ]] && echo OK || echo FAILED)" \
    "$status" "$log_path" >> "$RUNS_FILE"
  return "$status"
}

run_training "$GPU_PRETRAINED" "$PRETRAINED_EXP" pretrained &
PID_PRETRAINED=$!
run_training "$GPU_SCRATCH" "$SCRATCH_EXP" from_scratch &
PID_SCRATCH=$!

failed=0
wait "$PID_PRETRAINED" || failed=1
wait "$PID_SCRATCH" || failed=1

cat > "${SWEEP_ROOT}/final_results.txt" <<EOF
Finished: $(date -Iseconds)
Run table: ${RUNS_FILE}
Preprocessing log: ${PREPROCESS_LOG}
Training logs: ${TRAIN_LOG_ROOT}
Checkpoints: ${CHECKPOINT_ROOT}
EOF

cat "${SWEEP_ROOT}/final_results.txt"
exit "$failed"
