#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash run_libribrain_evals.sh [options]

Launches the two LibriBrain MEG-XL evaluation containers in detached mode:
  - eval_libribrain on EVAL_GPU
  - eval_libribrain_linear_probe on LINEAR_PROBE_GPU
  - monitor on MONITOR_PORT

Options:
  --no-build             Do not run docker compose build first.
  --no-monitor           Do not launch the monitor service.
  --monitor-only         Launch only the monitor service.
  --eval-only            Launch only eval_libribrain.
  --linear-probe-only    Launch only eval_libribrain_linear_probe.
  --logs                 Follow logs after launching.
  -h, --help             Show this help.

Environment overrides:
  EVAL_GPU=0
  LINEAR_PROBE_GPU=1
  MONITOR_PORT=8080
  DATASETS_DIR=./datasets
  LIBRIBRAIN_ROOT=./datasets/libribrain
  CHECKPOINTS_DIR=./checkpoints
  CRISS_CROSS_CHECKPOINT=./checkpoints/baseline/meg-xl-med.ckpt
  WANDB_MODE=offline
USAGE
}

build=1
follow_logs=0
services=(eval_libribrain eval_libribrain_linear_probe)
launch_monitor=1
validate_eval_inputs=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build)
      build=0
      shift
      ;;
    --no-monitor)
      launch_monitor=0
      shift
      ;;
    --monitor-only)
      services=(monitor)
      launch_monitor=0
      validate_eval_inputs=0
      shift
      ;;
    --eval-only)
      services=(eval_libribrain)
      shift
      ;;
    --linear-probe-only)
      services=(eval_libribrain_linear_probe)
      shift
      ;;
    --logs)
      follow_logs=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export EVAL_GPU="${EVAL_GPU:-0}"
export LINEAR_PROBE_GPU="${LINEAR_PROBE_GPU:-1}"
export MONITOR_PORT="${MONITOR_PORT:-8080}"
export DATASETS_DIR="${DATASETS_DIR:-./datasets}"
export LIBRIBRAIN_ROOT="${LIBRIBRAIN_ROOT:-./datasets/libribrain}"
export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"
export CRISS_CROSS_CHECKPOINT="${CRISS_CROSS_CHECKPOINT:-./checkpoints/baseline/meg-xl-med.ckpt}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p logs data/cache embeddings_cache hf_cache wandb "$CHECKPOINTS_DIR"

if [[ "$launch_monitor" -eq 1 ]]; then
  services+=(monitor)
fi

if [[ "$validate_eval_inputs" -eq 1 ]]; then
  checkpoint_host_path="$CRISS_CROSS_CHECKPOINT"
  case "$CRISS_CROSS_CHECKPOINT" in
    ./checkpoints/*)
      checkpoint_host_path="${CHECKPOINTS_DIR%/}/${CRISS_CROSS_CHECKPOINT#./checkpoints/}"
      ;;
    /workspace/checkpoints/*)
      checkpoint_host_path="${CHECKPOINTS_DIR%/}/${CRISS_CROSS_CHECKPOINT#/workspace/checkpoints/}"
      ;;
  esac

  if [[ ! -e "$checkpoint_host_path" ]]; then
    echo "Checkpoint not found on host: $checkpoint_host_path" >&2
    echo "Set CHECKPOINTS_DIR=/host/path/to/checkpoints and CRISS_CROSS_CHECKPOINT=./checkpoints/<file>.ckpt." >&2
    exit 1
  fi

  if [[ ! -d "$DATASETS_DIR" ]]; then
    echo "Dataset mount directory not found: $DATASETS_DIR" >&2
    echo "Set DATASETS_DIR=/path/to/datasets so it contains libribrain/." >&2
    exit 1
  fi
fi

if [[ "$build" -eq 1 ]]; then
  docker compose build "${services[@]}"
fi

docker compose up -d "${services[@]}"
docker compose ps "${services[@]}"

echo
echo "Containers launched detached. They keep running after this terminal closes."
if [[ " ${services[*]} " == *" monitor "* ]]; then
  echo "Monitor: http://localhost:${MONITOR_PORT}"
fi
echo "Logs:"
echo "  docker compose logs -f ${services[*]}"
echo "Stop:"
echo "  docker compose stop ${services[*]}"

if [[ "$follow_logs" -eq 1 ]]; then
  docker compose logs -f "${services[@]}"
fi
