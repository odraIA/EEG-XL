#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_URL="${OPENNEURO_DS004408_REPO_URL:-https://github.com/OpenNeuroDatasets/ds004408.git}"
DEST_DIR="${OPENNEURO_DS004408_DEST_DIR:-${REPO_ROOT}/datasets/OpenNeuroEEG_ds004408}"
GIT_BIN="${GIT_BIN:-git}"

if [[ -d "${DEST_DIR}/.git" ]]; then
  echo "Updating existing repository in ${DEST_DIR}"
  "${GIT_BIN}" -C "${DEST_DIR}" fetch --prune
  exec "${GIT_BIN}" -C "${DEST_DIR}" pull --ff-only
fi

if [[ -e "${DEST_DIR}" ]]; then
  echo "Destination exists but is not a Git repository: ${DEST_DIR}" >&2
  exit 2
fi

mkdir -p "$(dirname "${DEST_DIR}")"

echo "Cloning ${REPO_URL} into ${DEST_DIR}"
exec "${GIT_BIN}" clone "${REPO_URL}" "${DEST_DIR}"
