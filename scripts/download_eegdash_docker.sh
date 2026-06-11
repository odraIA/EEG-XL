#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEST_DIR="${EEGDASH_DEST_DIR:-${REPO_ROOT}/datasets/eegdash/data}"
IMAGE="${EEGDASH_DOCKER_IMAGE:-scrabrain-megxl:latest}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
DOCKER_CMD=("${DOCKER_BIN}")

if [[ "${DOCKER_USE_SUDO:-0}" == "1" ]]; then
  DOCKER_CMD=(sudo "${DOCKER_BIN}")
fi

mkdir -p "${DEST_DIR}"
DEST_ABS="$(cd "${DEST_DIR}" && pwd)"

echo "Downloading EEGDash NM000228 into ${DEST_ABS}"
echo "Expected training root after download: ${DEST_ABS}/nm000228"

exec "${DOCKER_CMD[@]}" run --rm \
  --user "$(id -u):$(id -g)" \
  -v "${REPO_ROOT}:/workspace" \
  -v "${DEST_ABS}:/data" \
  -w /workspace \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail
    python_bin=python
    if ! python -c "import eegdash" >/dev/null 2>&1; then
      python -m venv /tmp/eegdash-venv
      /tmp/eegdash-venv/bin/python -m pip install --quiet --upgrade pip
      /tmp/eegdash-venv/bin/python -m pip install --quiet eegdash
      python_bin=/tmp/eegdash-venv/bin/python
    fi
    exec "${python_bin}" scripts/download_eegdash_nm000228.py --cache-dir /data "$@"
  ' bash "$@"
