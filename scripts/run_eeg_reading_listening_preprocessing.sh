#!/usr/bin/env bash
set -uo pipefail

# CPU-only cache preparation for the reading -> listening EEG sweep.
# Preprocessing is independent of tokenizer, initialization and batch size, so
# only 6 signal profiles x 2 task families are required.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

COMPOSE_FILE="${EEG_PREPROCESS_COMPOSE_FILE:-${ROOT_DIR}/docker-compose.eeg-preprocess.yml}"
SERVICE="${EEG_PREPROCESS_SERVICE:-eeg_preprocess}"
MAIN_CACHE="${EEG_CACHE_DIR:-./data/cache/eeg_preprocessed}"
STAGING_CACHE="${EEG_PREPROCESS_STAGING_CACHE:-./data/cache/eeg_preprocessed_staging}"
WORKERS="${EEG_PREPROCESS_WORKERS:-1}"
OMP_THREADS="${EEG_PREPROCESS_OMP_THREADS:-4}"
CONTINUE_ON_ERROR="${EEG_PREPROCESS_CONTINUE_ON_ERROR:-true}"
STAMP="${EEG_PREPROCESS_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${EEG_PREPROCESS_LOG_ROOT:-logs/eeg_reading_listening_preprocessing/${STAMP}}"

if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EEG_PREPROCESS_WORKERS must be a positive integer, got: $WORKERS" >&2
  exit 2
fi

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

mkdir -p "$MAIN_CACHE" "$STAGING_CACHE" "$LOG_ROOT"
printf 'task\tstatus\tlog\n' > "${LOG_ROOT}/results.tsv"

# Reverse sweep order prioritizes profiles that an already-running sweep is
# less likely to have reached. Existing main-cache files are seeded and skipped.
PROFILES=(
  "full_band_0p1_50_nyquist|128|0.1|50"
  "low_gamma_30_45_nyquist|100|30|45"
  "full_band_0p1_50_fixed50|50|0.1|50"
  "low_gamma_30_45_fixed50|50|30|45"
  "beta_13_24_fixed50|50|13|24"
  "alpha_8_12_fixed50|50|8|12"
)

TASKS=()
for profile_spec in "${PROFILES[@]}"; do
  IFS='|' read -r label sfreq low high <<< "$profile_spec"
  TASKS+=("${label}_reading|train_criss_cross_eeg_reading_continuous|${sfreq}|${low}|${high}")
  TASKS+=("${label}_listening|train_criss_cross_eeg_listening_continuous|${sfreq}|${low}|${high}")
done

cat > "${LOG_ROOT}/metadata.txt" <<EOF
Started: $(date -Iseconds)
Main cache: ${MAIN_CACHE}
Staging cache: ${STAGING_CACHE}
Workers: ${WORKERS}
OMP threads per worker: ${OMP_THREADS}
Tasks: ${#TASKS[@]}
EOF

echo "Building CPU-only preprocessing service ${SERVICE}..."
docker compose -f "$COMPOSE_FILE" build "$SERVICE" || exit $?

run_task() {
  local spec="$1"
  local task_label config_name sfreq low high log_path status
  IFS='|' read -r task_label config_name sfreq low high <<< "$spec"
  log_path="${LOG_ROOT}/${task_label}.log"

  echo "[$(date -Iseconds)] Starting ${task_label}"
  EEG_PREPROCESS_OMP_THREADS="$OMP_THREADS" \
    docker compose -f "$COMPOSE_FILE" run \
      --rm --no-deps \
      -e "OMP_NUM_THREADS=${OMP_THREADS}" \
      "$SERVICE" \
      uv run --no-sync python scripts/preprocess_eeg_reading_listening.py \
        --config-name "$config_name" \
        --target-sfreq "$sfreq" \
        --l-freq "$low" \
        --h-freq "$high" \
        --cache-dir "$STAGING_CACHE" \
        --main-cache-dir "$MAIN_CACHE" \
      > "$log_path" 2>&1
  status=$?

  if [[ $status -eq 0 ]]; then
    printf '%s\tOK\t%s\n' "$task_label" "$log_path" >> "${LOG_ROOT}/results.tsv"
    echo "[$(date -Iseconds)] Completed ${task_label}"
  else
    printf '%s\tFAILED(%s)\t%s\n' "$task_label" "$status" "$log_path" >> "${LOG_ROOT}/results.tsv"
    echo "[$(date -Iseconds)] FAILED ${task_label}; see ${log_path}" >&2
  fi
  return "$status"
}

active_pids=()
active_labels=()
failed=0

wait_oldest() {
  local pid="${active_pids[0]}"
  local label="${active_labels[0]}"
  if ! wait "$pid"; then
    failed=1
    if ! is_true "$CONTINUE_ON_ERROR"; then
      echo "Stopping after failure in ${label}." >&2
      return 1
    fi
  fi
  active_pids=("${active_pids[@]:1}")
  active_labels=("${active_labels[@]:1}")
}

for task_spec in "${TASKS[@]}"; do
  task_label="${task_spec%%|*}"
  run_task "$task_spec" &
  active_pids+=("$!")
  active_labels+=("$task_label")

  if (( ${#active_pids[@]} >= WORKERS )); then
    wait_oldest || break
  fi
done

while (( ${#active_pids[@]} > 0 )); do
  wait_oldest || break
done

ok_count="$(awk -F '\t' 'NR > 1 && $2 == "OK" {count++} END {print count+0}' "${LOG_ROOT}/results.tsv")"
failed_count="$(awk -F '\t' 'NR > 1 && $2 ~ /^FAILED/ {count++} END {print count+0}' "${LOG_ROOT}/results.tsv")"

cat > "${LOG_ROOT}/final_results.txt" <<EOF
EEG preprocessing results
=========================
Finished: $(date -Iseconds)
Tasks completed: ${ok_count}/${#TASKS[@]}
Tasks failed: ${failed_count}
Main cache: ${MAIN_CACHE}
Staging cache: ${STAGING_CACHE}
Task table: ${LOG_ROOT}/results.tsv
Per-task logs: ${LOG_ROOT}/*.log
EOF

cat "${LOG_ROOT}/final_results.txt"
exit "$failed"
