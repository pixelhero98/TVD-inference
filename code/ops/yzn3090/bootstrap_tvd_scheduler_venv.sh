#!/bin/bash
set -euo pipefail

home_root="$(cd ~ && pwd)"
repo_root="${1:-${home_root}/work/TVD-Scheduler}"
venv_dir="${2:-${repo_root}/.venv}"

if [[ ! -d "${repo_root}" ]]; then
  echo "Missing TVD-Scheduler repo at ${repo_root}" >&2
  exit 1
fi

python3 -m venv "${venv_dir}"
source "${venv_dir}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${repo_root}/requirements.txt"