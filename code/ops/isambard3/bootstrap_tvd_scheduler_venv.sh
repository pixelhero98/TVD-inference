#!/bin/bash
set -euo pipefail

module load cray-python/3.11.7

projectdir="${PROJECTDIR:-/projects/b35z}"
project_root="${projectdir}/tvd-scheduler"
repo_root="${project_root}/TVD-Scheduler"
code_root="${repo_root}/code"
venv_dir="${project_root}/.venv"

if [[ ! -d "${code_root}" ]]; then
  echo "Missing TVD-Scheduler code root at ${code_root}" >&2
  exit 1
fi

python3 -m venv "${venv_dir}"
source "${venv_dir}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${repo_root}/requirements.txt"
