#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PATH="${REPO_ROOT}/results/geefetch_tests/.venv"
DEFAULT_CONFIG="${REPO_ROOT}/results/geefetch_tests/config_alphaearth_embeddings_2023_local.yaml"

usage() {
  cat <<USAGE
Usage: ${0##*/} [-c CONFIG]

Ensure the local GeeFetch virtual environment is active, create the target data directory described in the
config, and launch GeeFetch for the AlphaEarth embeddings. Defaults to the 2023 local config if none provided.

Options:
  -c CONFIG   Path to the GeeFetch YAML configuration to use (default: ${DEFAULT_CONFIG})
USAGE
}

CONFIG_PATH="${DEFAULT_CONFIG}"
while getopts ":c:h" opt; do
  case "${opt}" in
    c)
      CONFIG_PATH="${OPTARG}"
      ;;
    h)
      usage
      exit 0
      ;;
    :)
      echo "Missing argument for -${OPTARG}" >&2
      usage
      exit 1
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "Virtual environment not found at ${VENV_PATH}." >&2
  echo "Create it first with: /usr/local/opt/python@3.13/bin/python3.13 -m venv ${VENV_PATH}" >&2
  exit 1
fi

PYTHON_BIN="${VENV_PATH}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found in venv: ${PYTHON_BIN}" >&2
  exit 1
fi

DATA_DIR="$(${PYTHON_BIN} - "${CONFIG_PATH}" <<'PY'
import sys
import yaml
from pathlib import Path

cfg_path = Path(sys.argv[1])
with cfg_path.open() as f:
    cfg = yaml.safe_load(f)

data_dir = cfg.get('data_dir')
if not data_dir:
    raise SystemExit('`data_dir` missing from config')

print(data_dir)
PY
)"

mkdir -p "${DATA_DIR}"

# Optional logs directory alongside results for convenience
mkdir -p "${REPO_ROOT}/results/geefetch_tests/logs"

# Activate the virtual environment and run GeeFetch
source "${VENV_PATH}/bin/activate"

# Ensure compatible geedim version is present to avoid PatchedBaseImage regression
pip install 'geedim<2' --quiet

geefetch custom alphaearth_embeddings -c "${CONFIG_PATH}"
