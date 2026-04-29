#!/bin/bash
set -euo pipefail

module load cray-python/3.11.7

projectdir="${PROJECTDIR:-/projects/b35z}"
project_root="${projectdir}/tvd-scheduler"
repo_root="${project_root}/TVD-Scheduler"
code_root="${repo_root}/code"
venv_dir="${project_root}/.venv"
matrix_root="${repo_root}/TVD-result/results/backbone_matrix"

if [[ ! -f "${venv_dir}/bin/activate" ]]; then
  echo "Missing virtual environment at ${venv_dir}. Run code/ops/isambard3/bootstrap_tvd_scheduler_venv.sh first." >&2
  exit 1
fi

source "${venv_dir}/bin/activate"
cd "${code_root}"
budget_steps="${OTFLOW_BACKBONE_BUDGET_STEPS:-20000}"
manifest_out="${OTFLOW_BACKBONE_MANIFEST:-${matrix_root}/backbone_manifest.json}"

python fm_backbone_readiness_audit.py \
  --matrix_root "${matrix_root}" \
  --manifest_path "${manifest_out}" \
  --dataset_root "${repo_root}/paper_datasets" \
  --sleep_edf_path "${repo_root}/data/sleep_edf_3ch_100hz_stage_conditioned.npz" \
  --budget_steps "${budget_steps}"
