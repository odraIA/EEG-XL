#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_URL="${OPENNEURO_DS004408_REPO_URL:-https://github.com/OpenNeuroDatasets/ds004408.git}"
DEST_DIR="${OPENNEURO_DS004408_DEST_DIR:-${REPO_ROOT}/datasets/OpenNeuroEEG_ds004408}"
GIT_BIN="${GIT_BIN:-git}"
FETCH_TEXTGRIDS="${OPENNEURO_DS004408_FETCH_TEXTGRIDS:-1}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
DOCKER_CMD=("${DOCKER_BIN}")

if [[ "${DOCKER_USE_SUDO:-0}" == "1" ]]; then
  DOCKER_CMD=(sudo "${DOCKER_BIN}")
fi

fetch_textgrids_with_docker() {
  if ! command -v "${DOCKER_BIN}" >/dev/null 2>&1; then
    echo "Docker is not available; could not fetch ds004408 TextGrid stimuli without git-annex." >&2
    return 1
  fi

  mkdir -p "${DEST_DIR}"
  local dest_abs
  dest_abs="$(cd "${DEST_DIR}" && pwd)"
  echo "Fetching ds004408 TextGrid stimuli with Docker/AWS CLI into ${dest_abs}"
  "${DOCKER_CMD[@]}" run --rm \
    --user "$(id -u):$(id -g)" \
    -v "${dest_abs}:/data" \
    amazon/aws-cli \
    s3 sync --no-sign-request "s3://openneuro.org/ds004408" /data \
      --exclude "*" \
      --include "stimuli/*.TextGrid"
}

fetch_textgrids() {
  if [[ "${FETCH_TEXTGRIDS}" != "1" ]]; then
    return
  fi
  if ! "${GIT_BIN}" -C "${DEST_DIR}" annex version >/dev/null 2>&1; then
    echo "git-annex is not available; falling back to Docker/AWS CLI for TextGrid stimuli." >&2
    fetch_textgrids_with_docker
    return
  fi

  echo "Materializing ds004408 TextGrid stimuli in ${DEST_DIR}"
  "${GIT_BIN}" -C "${DEST_DIR}" annex get -- 'stimuli/*.TextGrid'
}

if [[ -d "${DEST_DIR}/.git" ]]; then
  echo "Updating existing repository in ${DEST_DIR}"
  "${GIT_BIN}" -C "${DEST_DIR}" fetch --prune
  "${GIT_BIN}" -C "${DEST_DIR}" pull --ff-only
  fetch_textgrids
  exit 0
fi

if [[ -e "${DEST_DIR}" ]]; then
  echo "Destination exists but is not a Git repository: ${DEST_DIR}" >&2
  exit 2
fi

mkdir -p "$(dirname "${DEST_DIR}")"

echo "Cloning ${REPO_URL} into ${DEST_DIR}"
"${GIT_BIN}" clone "${REPO_URL}" "${DEST_DIR}"
fetch_textgrids
