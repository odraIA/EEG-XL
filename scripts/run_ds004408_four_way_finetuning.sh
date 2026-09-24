#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare ds004408 once and compare four initializations in parallel:
#   1. random CrissCross initialization;
#   2. EEG curriculum trained from scratch, using EEG embedding row 2;
#   3. MEG-XL -> EEG curriculum, using dedicated EEG embedding row 2;
#   4. MEG-XL -> EEG curriculum, reusing MEG embedding row 1 for EEG.
#
# ds004408 always remains physical EEG sensor type 2. The per-run embedding id
# only selects the sensor_type_layer row used by that checkpoint.
#
# By default, GPUs that look idle at launch are detected automatically and two
# fine-tunings are assigned to each GPU. With two idle GPUs, all four runs start
# concurrently. Use GPU_LIST=0,1 to select devices explicitly, or
# JOBS_PER_GPU=1 to use only one process per GPU.
#
# Repeated seeds: SEEDS=42,43,44 (or `bash run_ds004408_four_way_finetuning.sh
# 42 43 44`) fine-tunes every model once per seed. The seed is passed as the
# Hydra `seed=` override, so it controls weight init, data order and the hashed
# sentence split; all four models share the same split within a seed. Jobs are
# ordered seed-major, so each batch finishes complete seed replicates first.
# The comparison report aggregates runs per model (mean/std/SEM) and runs
# pairwise Welch tests when every model has at least two seeds.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/run_ds004408_four_way_finetuning.sh"
cd "$ROOT_DIR"

GPU_LIST="${GPU_LIST:-auto}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
FREE_GPU_MAX_MEMORY_MIB="${FREE_GPU_MAX_MEMORY_MIB:-1024}"
FREE_GPU_MAX_UTILIZATION="${FREE_GPU_MAX_UTILIZATION:-10}"
OMP_NUM_THREADS_PER_JOB="${OMP_NUM_THREADS_PER_JOB:-4}"
WANDB_MODE="${WANDB_MODE:-offline}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-42}"
if (($# > 0)); then
  SEEDS="$*"
fi

DS004408_ROOT="${DS004408_ROOT:-./datasets/OpenNeuroEEG_ds004408}"
BIOCODEC_CHECKPOINT="${BIOCODEC_CHECKPOINT:-./brainstorm/neuro_tokenizers/biocodec_ckpt.pt}"
MEGXL_ARCH_CHECKPOINT="${MEGXL_ARCH_CHECKPOINT:-./checkpoints/baseline/meg-xl-med.ckpt}"
CURRICULUM_ROOT="${CURRICULUM_ROOT:-./checkpoints/eeg_language_curriculum_three_models/20260629_004853}"
FROM_SCRATCH_EEG_CHECKPOINT="${FROM_SCRATCH_EEG_CHECKPOINT:-$CURRICULUM_ROOT/eeg_curriculum_from_scratch_language_seed42/checkpoint_best.pt}"
MEGXL_EEG2_CHECKPOINT="${MEGXL_EEG2_CHECKPOINT:-$CURRICULUM_ROOT/eeg_curriculum_megxl_eeg2_language_seed42/checkpoint_best.pt}"
MEGXL_EEG1_CHECKPOINT="${MEGXL_EEG1_CHECKPOINT:-$CURRICULUM_ROOT/eeg_curriculum_megxl_eeg1_language_seed42/checkpoint_best.pt}"

TRAIN_PCT="${TRAIN_PCT:-1.0}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREPARE_WORD_ALIGNED="${PREPARE_WORD_ALIGNED:-1}"
WARM_WORD_ALIGNED_CACHE="${WARM_WORD_ALIGNED_CACHE:-1}"

RESULTS_ROOT="${RESULTS_ROOT:-./results/ds004408_four_way/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-./logs/word_classification_ds004408_four_way/$RUN_ID}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints/word_classification_ds004408_four_way/$RUN_ID}"
HYDRA_ROOT="${HYDRA_ROOT:-./logs/hydra/ds004408_four_way/$RUN_ID}"
WORD_ALIGNED_OUTPUT="${WORD_ALIGNED_OUTPUT:-$RESULTS_ROOT/word_aligned}"
WORD_ALIGNED_CACHE="${WORD_ALIGNED_CACHE:-./data/cache/ds004408_word_aligned_v2}"
MASTER_LOG="${MASTER_LOG:-$ROOT_DIR/logs/ds004408_four_way_${RUN_ID}.log}"
PID_FILE="${PID_FILE:-$ROOT_DIR/ds004408_four_way_${RUN_ID}.pid}"
CURRENT_CONTAINERS_FILE="$RESULTS_ROOT/current_containers.tsv"
STATUS_FILE="$RESULTS_ROOT/runs.tsv"

JOB_LABELS=(random_init eeg_from_scratch megxl_eeg2 megxl_eeg1)
JOB_TRAIN_FROM_SCRATCH=(true false false false)
JOB_CHECKPOINTS=(
  "$MEGXL_ARCH_CHECKPOINT"
  "$FROM_SCRATCH_EEG_CHECKPOINT"
  "$MEGXL_EEG2_CHECKPOINT"
  "$MEGXL_EEG1_CHECKPOINT"
)
JOB_EMBEDDING_IDS=(2 2 2 1)

declare -a SEED_LIST=()
declare -a AVAILABLE_GPUS=()
declare -a GPU_SLOTS=()
declare -a COMPLETED_RUN_SPECS=()
declare -A LABEL_BY_RUN=()
declare -A SEED_BY_RUN=()
declare -A CONTAINER_BY_RUN=()
declare -A GPU_BY_RUN=()
declare -A EMBEDDING_BY_RUN=()
declare -A ORDER_BY_RUN=()
declare -A STARTED_BY_RUN=()
declare -A SAVE_DIR_BY_RUN=()
declare -A CHECKPOINT_DIR_BY_RUN=()
declare -A LOG_PID_BY_RUN=()

mkdir -p "$RESULTS_ROOT" "$LOG_ROOT" "$CHECKPOINT_ROOT" "$HYDRA_ROOT" \
  "$WORD_ALIGNED_OUTPUT" "$(dirname "$MASTER_LOG")"

require_file() {
  [[ -f "$1" ]] || { echo "ERROR: Missing $2: $1" >&2; exit 2; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "ERROR: Missing $2: $1" >&2; exit 2; }
}

trim() {
  local value="$*"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

parse_seeds() {
  local seed
  local -A seen=()
  for seed in ${SEEDS//,/ }; do
    [[ "$seed" =~ ^[0-9]+$ ]] || {
      echo "ERROR: Invalid seed '$seed' in SEEDS=$SEEDS" >&2
      exit 2
    }
    [[ -z "${seen[$seed]:-}" ]] || {
      echo "ERROR: Duplicate seed '$seed' in SEEDS=$SEEDS" >&2
      exit 2
    }
    seen["$seed"]=1
    SEED_LIST+=("$seed")
  done

  ((${#SEED_LIST[@]} > 0)) || {
    echo "ERROR: SEEDS must contain at least one integer seed" >&2
    exit 2
  }
  echo "Seeds: ${SEED_LIST[*]}"
}

resolve_available_gpus() {
  local gpu index memory_used utilization round
  if [[ "$GPU_LIST" == "auto" ]]; then
    while IFS=',' read -r index memory_used utilization; do
      index="$(trim "$index")"
      memory_used="$(trim "$memory_used")"
      utilization="$(trim "$utilization")"
      if (( memory_used <= FREE_GPU_MAX_MEMORY_MIB && utilization <= FREE_GPU_MAX_UTILIZATION )); then
        AVAILABLE_GPUS+=("$index")
      else
        echo "Skipping busy GPU $index: memory=${memory_used} MiB, utilization=${utilization}%"
      fi
    done < <(
      nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits
    )
  else
    IFS=',' read -r -a requested_gpus <<< "$GPU_LIST"
    for gpu in "${requested_gpus[@]}"; do
      gpu="$(trim "$gpu")"
      [[ -n "$gpu" ]] || continue
      nvidia-smi -i "$gpu" >/dev/null 2>&1 || {
        echo "ERROR: Invalid or unavailable GPU index: $gpu" >&2
        exit 2
      }
      AVAILABLE_GPUS+=("$gpu")
    done
  fi

  ((${#AVAILABLE_GPUS[@]} > 0)) || {
    echo "ERROR: No free GPUs found." >&2
    echo "Set GPU_LIST=0,1 explicitly or adjust FREE_GPU_MAX_MEMORY_MIB/FREE_GPU_MAX_UTILIZATION." >&2
    exit 2
  }

  for ((round = 0; round < JOBS_PER_GPU; round++)); do
    for gpu in "${AVAILABLE_GPUS[@]}"; do
      GPU_SLOTS+=("$gpu")
    done
  done

  ((${#GPU_SLOTS[@]} > 0)) || {
    echo "ERROR: JOBS_PER_GPU must be at least 1" >&2
    exit 2
  }

  echo "Available GPUs: ${AVAILABLE_GPUS[*]}"
  echo "Concurrent slots: ${#GPU_SLOTS[@]} (${JOBS_PER_GPU} job(s) per GPU)"
}

preflight() {
  command -v docker >/dev/null 2>&1 || {
    echo "ERROR: docker is not available" >&2
    exit 2
  }
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "ERROR: nvidia-smi is not available" >&2
    exit 2
  }
  [[ "$JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: JOBS_PER_GPU must be a positive integer" >&2
    exit 2
  }
  [[ "$FREE_GPU_MAX_MEMORY_MIB" =~ ^[0-9]+$ ]] || {
    echo "ERROR: FREE_GPU_MAX_MEMORY_MIB must be a non-negative integer" >&2
    exit 2
  }
  [[ "$FREE_GPU_MAX_UTILIZATION" =~ ^[0-9]+$ ]] || {
    echo "ERROR: FREE_GPU_MAX_UTILIZATION must be a non-negative integer" >&2
    exit 2
  }

  require_dir "$DS004408_ROOT" "ds004408 dataset directory"
  compgen -G "$DS004408_ROOT/sub-*/eeg/*_eeg.vhdr" >/dev/null || {
    echo "ERROR: No ds004408 BrainVision files under $DS004408_ROOT" >&2
    exit 2
  }
  compgen -G "$DS004408_ROOT/stimuli/*.TextGrid" >/dev/null || {
    echo "ERROR: No materialized TextGrids. Run scripts/clone_openneuro_ds004408.sh" >&2
    exit 2
  }
  require_file "$BIOCODEC_CHECKPOINT" "BioCodec checkpoint"
  require_file "$MEGXL_ARCH_CHECKPOINT" "MEG-XL architecture checkpoint"
  require_file "$FROM_SCRATCH_EEG_CHECKPOINT" "curriculum checkpoint from scratch"
  require_file "$MEGXL_EEG2_CHECKPOINT" "curriculum checkpoint MEG-XL/eeg2"
  require_file "$MEGXL_EEG1_CHECKPOINT" "curriculum checkpoint MEG-XL/eeg1"

  parse_seeds
  resolve_available_gpus
}
preflight

if [[ "${DS004408_FOUR_WAY_WORKER:-0}" != "1" ]]; then
  nohup env DS004408_FOUR_WAY_WORKER=1 RUN_ID="$RUN_ID" MASTER_LOG="$MASTER_LOG" \
    PID_FILE="$PID_FILE" GPU_LIST="$(IFS=,; echo "${AVAILABLE_GPUS[*]}")" \
    SEEDS="$(IFS=,; echo "${SEED_LIST[*]}")" \
    JOBS_PER_GPU="$JOBS_PER_GPU" OMP_NUM_THREADS_PER_JOB="$OMP_NUM_THREADS_PER_JOB" \
    FREE_GPU_MAX_MEMORY_MIB="$FREE_GPU_MAX_MEMORY_MIB" \
    FREE_GPU_MAX_UTILIZATION="$FREE_GPU_MAX_UTILIZATION" \
    WANDB_MODE="$WANDB_MODE" BUILD_IMAGE="$BUILD_IMAGE" \
    DS004408_ROOT="$DS004408_ROOT" BIOCODEC_CHECKPOINT="$BIOCODEC_CHECKPOINT" \
    MEGXL_ARCH_CHECKPOINT="$MEGXL_ARCH_CHECKPOINT" CURRICULUM_ROOT="$CURRICULUM_ROOT" \
    FROM_SCRATCH_EEG_CHECKPOINT="$FROM_SCRATCH_EEG_CHECKPOINT" \
    MEGXL_EEG2_CHECKPOINT="$MEGXL_EEG2_CHECKPOINT" \
    MEGXL_EEG1_CHECKPOINT="$MEGXL_EEG1_CHECKPOINT" TRAIN_PCT="$TRAIN_PCT" \
    NUM_EPOCHS="$NUM_EPOCHS" BATCH_SIZE="$BATCH_SIZE" NUM_WORKERS="$NUM_WORKERS" \
    PREPARE_WORD_ALIGNED="$PREPARE_WORD_ALIGNED" \
    WARM_WORD_ALIGNED_CACHE="$WARM_WORD_ALIGNED_CACHE" RESULTS_ROOT="$RESULTS_ROOT" \
    LOG_ROOT="$LOG_ROOT" CHECKPOINT_ROOT="$CHECKPOINT_ROOT" HYDRA_ROOT="$HYDRA_ROOT" \
    WORD_ALIGNED_OUTPUT="$WORD_ALIGNED_OUTPUT" WORD_ALIGNED_CACHE="$WORD_ALIGNED_CACHE" \
    bash "$SCRIPT_PATH" >> "$MASTER_LOG" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  echo "$RUN_ID" > "$ROOT_DIR/ds004408_four_way.latest"
  echo "ds004408 four-way pipeline launched. PID: $(cat "$PID_FILE")"
  echo "GPUs: ${AVAILABLE_GPUS[*]} | jobs per GPU: $JOBS_PER_GPU"
  echo "Seeds: ${SEED_LIST[*]} | fine-tunings: $(( ${#JOB_LABELS[@]} * ${#SEED_LIST[@]} ))"
  echo "Log: $MASTER_LOG"
  echo "Results: $RESULTS_ROOT"
  exit 0
fi

echo $$ > "$PID_FILE"
printf 'order\tlabel\tseed\tembedding_id\tgpu\tcontainer\tstatus\texit_code\tstarted_at\tfinished_at\n' > "$STATUS_FILE"
printf 'run\tgpu\tcontainer\n' > "$CURRENT_CONTAINERS_FILE"

if [[ "$BUILD_IMAGE" == "1" || "$BUILD_IMAGE" == "true" ]]; then
  docker compose build eval_eeg_listening
fi

on_error() {
  local status=$?
  echo "ERROR: ds004408 four-way pipeline failed ($status)" >&2
  if [[ -s "$CURRENT_CONTAINERS_FILE" ]]; then
    echo "Containers launched in this run:" >&2
    tail -n +2 "$CURRENT_CONTAINERS_FILE" >&2 || true
  fi
  exit "$status"
}
trap on_error ERR

append_status() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$STATUS_FILE"
}

prepare_word_aligned() {
  [[ "$PREPARE_WORD_ALIGNED" == "1" || "$PREPARE_WORD_ALIGNED" == "true" ]] || return 0
  local started finished warm_flag="--no-warm-cache"
  local prep_gpu="${AVAILABLE_GPUS[0]}"
  started="$(date --iso-8601=seconds)"
  [[ "$WARM_WORD_ALIGNED_CACHE" == "1" || "$WARM_WORD_ALIGNED_CACHE" == "true" ]] && warm_flag="--warm-cache"

  echo "Preparing ds004408 word alignment once on GPU $prep_gpu"
  env EEG_GPU="$prep_gpu" WANDB_MODE="$WANDB_MODE" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS_PER_JOB" \
    docker compose run --rm --no-deps \
      -e "NVIDIA_VISIBLE_DEVICES=$prep_gpu" \
      -e "WANDB_MODE=$WANDB_MODE" \
      -e "OMP_NUM_THREADS=$OMP_NUM_THREADS_PER_JOB" \
      eval_eeg_listening \
      uv run --no-sync python scripts/prepare_ds004408_word_aligned.py \
        --root "$DS004408_ROOT" --cache-dir "$WORD_ALIGNED_CACHE" \
        --output-dir "$WORD_ALIGNED_OUTPUT" --eeg-sensor-type eeg \
        --montage-name biosemi128 --drop-bad-channels "$warm_flag"

  require_file "$WORD_ALIGNED_OUTPUT/summary.json" "word-aligned summary"
  require_file "$WORD_ALIGNED_OUTPUT/word_aligned_manifest.csv" "word-aligned manifest"
  require_file "$WORD_ALIGNED_OUTPUT/alignment_report.json" "alignment report"
  finished="$(date --iso-8601=seconds)"
  append_status 0 prepare_word_aligned - 2 "$prep_gpu" docker-compose-run-rm COMPLETED 0 "$started" "$finished"
}

launch_experiment() {
  local order="$1" label="$2" seed="$3" train_from_scratch="$4" init_checkpoint="$5"
  local embedding_id="$6" gpu="$7"
  local run_key="${label}_seed${seed}"
  local experiment="ds004408_${run_key}_${RUN_ID}" container="ds004408_${run_key}_${RUN_ID}"
  local save_dir="$LOG_ROOT/$label/seed$seed" checkpoint_dir="$CHECKPOINT_ROOT/$label/seed$seed"
  local hydra_dir="$HYDRA_ROOT/$label/seed$seed" started

  if [[ "$embedding_id" != "1" && "$embedding_id" != "2" ]]; then
    echo "ERROR: Unsupported EEG embedding id for $label: $embedding_id" >&2
    exit 2
  fi

  mkdir -p "$save_dir" "$checkpoint_dir" "$hydra_dir"
  if docker container inspect "$container" >/dev/null 2>&1; then
    [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != "true" ]] || {
      echo "ERROR: container already running: $container" >&2
      exit 3
    }
    docker rm "$container" >/dev/null
  fi

  started="$(date --iso-8601=seconds)"
  echo "START $experiment | GPU=$gpu | seed=$seed | physical_sensor=eeg:2 | embedding=$embedding_id | checkpoint=$init_checkpoint"

  env EEG_GPU="$gpu" WANDB_MODE="$WANDB_MODE" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS_PER_JOB" \
    docker compose run -d --no-deps --name "$container" \
      -e "NVIDIA_VISIBLE_DEVICES=$gpu" \
      -e "WANDB_MODE=$WANDB_MODE" \
      -e "OMP_NUM_THREADS=$OMP_NUM_THREADS_PER_JOB" \
      -e "EEG_SENSOR_EMBEDDING_TYPE_ID=$embedding_id" \
      eval_eeg_listening \
      uv run --no-sync python -m scripts.evaluate_ds004408_word_classification \
        --config-name=ds004408_word_finetuning \
        "seed=$seed" \
        "model.train_from_scratch=$train_from_scratch" model.use_promoted_checkpoint=false \
        model.promoted_checkpoint=null "model.criss_cross_checkpoint=$init_checkpoint" \
        "model.eeg_sensor_embedding_type_id=$embedding_id" \
        model.tokenizer_name=biocodec "model.tokenizer_checkpoint=$BIOCODEC_CHECKPOINT" \
        "data.root=$DS004408_ROOT" "data.cache_dir=$WORD_ALIGNED_CACHE" \
        data.eeg_sensor_type=eeg data.montage_name=biosemi128 data.drop_bad_channels=true \
        "data.train_pct=$TRAIN_PCT" "training.num_epochs=$NUM_EPOCHS" \
        "training.batch_size=$BATCH_SIZE" "training.num_workers=$NUM_WORKERS" \
        'evaluation.retrieval_set_sizes=[50,250]' evaluation.k=10 \
        "logging.experiment_name=$experiment" "logging.save_dir=$save_dir" \
        "logging.checkpoint_dir=$checkpoint_dir" "hydra.run.dir=$hydra_dir"

  LABEL_BY_RUN["$run_key"]="$label"
  SEED_BY_RUN["$run_key"]="$seed"
  CONTAINER_BY_RUN["$run_key"]="$container"
  GPU_BY_RUN["$run_key"]="$gpu"
  EMBEDDING_BY_RUN["$run_key"]="$embedding_id"
  ORDER_BY_RUN["$run_key"]="$order"
  STARTED_BY_RUN["$run_key"]="$started"
  SAVE_DIR_BY_RUN["$run_key"]="$save_dir"
  CHECKPOINT_DIR_BY_RUN["$run_key"]="$checkpoint_dir"
  printf '%s\t%s\t%s\n' "$run_key" "$gpu" "$container" >> "$CURRENT_CONTAINERS_FILE"

  docker logs --follow "$container" 2>&1 | sed -u "s/^/[$run_key|gpu$gpu] /" &
  LOG_PID_BY_RUN["$run_key"]=$!
}

validate_experiment_outputs() {
  local run_key="$1"
  local save_dir="${SAVE_DIR_BY_RUN[$run_key]}"
  local checkpoint_dir="${CHECKPOINT_DIR_BY_RUN[$run_key]}"
  local missing=0 path description

  while IFS='|' read -r path description; do
    if [[ ! -f "$path" ]]; then
      echo "ERROR: Missing $description for $run_key: $path" >&2
      missing=1
    fi
  done <<EOF
$save_dir/final_results.json|final results
$save_dir/paper_test_metrics.csv|paper metrics
$save_dir/paper_report_manifest.json|report manifest
$checkpoint_dir/checkpoint_best.pt|best checkpoint
EOF

  ((missing == 0))
}

wait_experiment() {
  local run_key="$1"
  local label="${LABEL_BY_RUN[$run_key]}"
  local seed="${SEED_BY_RUN[$run_key]}"
  local container="${CONTAINER_BY_RUN[$run_key]}"
  local gpu="${GPU_BY_RUN[$run_key]}"
  local embedding_id="${EMBEDDING_BY_RUN[$run_key]}"
  local order="${ORDER_BY_RUN[$run_key]}"
  local started="${STARTED_BY_RUN[$run_key]}"
  local exit_code finished status

  exit_code="$(docker wait "$container")"
  wait "${LOG_PID_BY_RUN[$run_key]}" || true
  finished="$(date --iso-8601=seconds)"

  if [[ "$exit_code" == "0" ]] && validate_experiment_outputs "$run_key"; then
    status="COMPLETED"
    docker rm "$container" >/dev/null
    COMPLETED_RUN_SPECS+=("$label=${SAVE_DIR_BY_RUN[$run_key]}")
  else
    status="FAILED"
    echo "ERROR: $container exited with $exit_code; keeping it for inspection" >&2
  fi

  append_status "$order" "$label" "$seed" "$embedding_id" "$gpu" "$container" \
    "$status" "$exit_code" "$started" "$finished"

  [[ "$status" == "COMPLETED" ]]
}

run_all_experiments() {
  local model_count="${#JOB_LABELS[@]}"
  local total_jobs=$((model_count * ${#SEED_LIST[@]}))
  local slot_count="${#GPU_SLOTS[@]}"
  local batch_start job_index slot_index model_index seed label gpu run_key
  local -a batch_runs=()
  local failures=0

  echo "Launching $total_jobs experiments ($model_count models x ${#SEED_LIST[@]} seeds)" \
    "with up to $slot_count concurrent containers"

  # Seed-major order: job_index = seed_index * model_count + model_index.
  for ((batch_start = 0; batch_start < total_jobs; batch_start += slot_count)); do
    batch_runs=()
    for ((slot_index = 0; slot_index < slot_count; slot_index++)); do
      job_index=$((batch_start + slot_index))
      ((job_index < total_jobs)) || break

      model_index=$((job_index % model_count))
      seed="${SEED_LIST[$((job_index / model_count))]}"
      label="${JOB_LABELS[$model_index]}"
      gpu="${GPU_SLOTS[$slot_index]}"
      launch_experiment \
        "$((job_index + 1))" \
        "$label" \
        "$seed" \
        "${JOB_TRAIN_FROM_SCRATCH[$model_index]}" \
        "${JOB_CHECKPOINTS[$model_index]}" \
        "${JOB_EMBEDDING_IDS[$model_index]}" \
        "$gpu"
      batch_runs+=("${label}_seed${seed}")
    done

    echo "Batch running: ${batch_runs[*]}"
    for run_key in "${batch_runs[@]}"; do
      if ! wait_experiment "$run_key"; then
        failures=$((failures + 1))
      fi
    done
  done

  ((failures == 0)) || {
    echo "ERROR: $failures fine-tuning job(s) failed" >&2
    return 1
  }
}

generate_comparison_report() {
  local report_gpu="${AVAILABLE_GPUS[0]}"
  local spec
  local -a run_args=()
  for spec in "${COMPLETED_RUN_SPECS[@]}"; do
    run_args+=(--run "$spec")
  done
  echo "Generating combined four-way report over ${#COMPLETED_RUN_SPECS[@]} runs (seeds: ${SEED_LIST[*]})"

  env EEG_GPU="$report_gpu" WANDB_MODE="$WANDB_MODE" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS_PER_JOB" \
    docker compose run --rm --no-deps \
      -e "NVIDIA_VISIBLE_DEVICES=$report_gpu" \
      -e "WANDB_MODE=$WANDB_MODE" \
      -e "OMP_NUM_THREADS=$OMP_NUM_THREADS_PER_JOB" \
      eval_eeg_listening \
      uv run --no-sync python -m brainstorm.megxl_test_reporting compare \
        "${run_args[@]}" \
        --output-dir "$RESULTS_ROOT" \
        --retrieval-sizes 50 250 \
        --top-k 10

  require_file "$RESULTS_ROOT/weissbart_three_way_test_metrics.csv" "combined long metrics"
  cp "$RESULTS_ROOT/weissbart_three_way_test_metrics.csv" \
    "$RESULTS_ROOT/ds004408_four_way_test_metrics.csv"
  require_file "$RESULTS_ROOT/megxl_paper_metrics_summary.csv" "aggregate metrics"
  require_file "$RESULTS_ROOT/megxl_pairwise_welch_tests.csv" "Welch tests"
  require_file "$RESULTS_ROOT/megxl_paper_report_manifest.json" "combined report"
}

prepare_word_aligned
run_all_experiments
generate_comparison_report

rm -f "$CURRENT_CONTAINERS_FILE"
echo "$RUN_ID" > "$RESULTS_ROOT/COMPLETED"
rm -f "$PID_FILE"

echo "ds004408 four-way fine-tuning completed."
echo "GPUs used: ${AVAILABLE_GPUS[*]}"
echo "Seeds: ${SEED_LIST[*]}"
echo "Word alignment: $WORD_ALIGNED_OUTPUT/summary.json"
echo "Metrics: $RESULTS_ROOT/ds004408_four_way_test_metrics.csv"
echo "Results: $RESULTS_ROOT"
