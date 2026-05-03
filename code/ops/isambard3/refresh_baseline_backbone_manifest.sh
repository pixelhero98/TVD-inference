
#!/bin/bash
set -euo pipefail

module load cray-python/3.11.7 2>/dev/null || true

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${REPO_ROOT:-$(cd "${script_dir}/../../.." && pwd)}"
output_root="${OUTPUT_ROOT:-${repo_root}/outputs}"
code_root="${repo_root}/code"
venv_dir="${VENV_DIR:-${repo_root}/.venv}"
matrix_root="${BACKBONE_MATRIX_ROOT:-${output_root}/backbone_matrix}"
manifest_out="${BACKBONE_MANIFEST:-${matrix_root}/backbone_manifest.json}"

if [[ ! -f "${venv_dir}/bin/activate" ]]; then
  echo "Missing virtual environment at ${venv_dir}. Run code/ops/isambard3/bootstrap_env.sh first." >&2
  exit 1
fi

source "${venv_dir}/bin/activate"
cd "${code_root}"
budget_steps="${OTFLOW_BACKBONE_BUDGET_STEPS:-20000}"

python fm_backbone_readiness_audit.py \
  --matrix_root "${matrix_root}" \
  --manifest_path "${manifest_out}" \
  --dataset_root "${repo_root}/paper_datasets" \
  --sleep_edf_path "${repo_root}/data/sleep_edf_3ch_100hz_stage_conditioned.npz" \
  --budget_steps "${budget_steps}"
