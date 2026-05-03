
#!/bin/bash
set -euo pipefail

module load cray-python/3.11.7 2>/dev/null || true

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${REPO_ROOT:-$(cd "${script_dir}/../../.." && pwd)}"
output_root="${OUTPUT_ROOT:-${repo_root}/outputs}"
code_root="${repo_root}/code"
venv_dir="${VENV_DIR:-${repo_root}/.venv}"
manifest_path="${BACKBONE_MANIFEST:-${output_root}/backbone_matrix/backbone_manifest.json}"
out_base="${OTFLOW_OUT_BASE:-${output_root}/experiments/otflow_lob_baseline_backbone_sweep}"
scratch_base="${SCRATCHDIR:-${TMPDIR:-/tmp}}"

datasets=("cryptos" "es_mbp_10" "sleep_edf")
train_steps_values=(4000 8000 12000 16000)

task_index="${OTFLOW_SWEEP_TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
num_datasets="${#datasets[@]}"
num_budgets="${#train_steps_values[@]}"
num_tasks=$((num_datasets * num_budgets))
if [[ "${task_index}" -lt 0 || "${task_index}" -ge "${num_tasks}" ]]; then
  echo "Invalid sweep task index: ${task_index}" >&2
  exit 2
fi

budget_index=$((task_index / num_datasets))
dataset_index=$((task_index % num_datasets))
dataset="${datasets[${dataset_index}]}"
train_steps="${OTFLOW_TRAIN_STEPS:-${train_steps_values[${budget_index}]}}"
budget_label="$((train_steps / 1000))k"
scratch_job_root="${scratch_base}/diffusion-flow-inference/${SLURM_JOB_ID:-manual}-lob-${budget_label}-${dataset}"
out_root="${OTFLOW_OUT_ROOT:-${out_base}/${budget_label}/${dataset}}"

mkdir -p "${scratch_job_root}" "${out_root}"

if [[ ! -f "${venv_dir}/bin/activate" ]]; then
  echo "Missing virtual environment at ${venv_dir}. Run code/ops/isambard3/bootstrap_env.sh first." >&2
  exit 1
fi
if [[ ! -f "${manifest_path}" ]]; then
  echo "Missing backbone manifest at ${manifest_path}" >&2
  exit 1
fi

source "${venv_dir}/bin/activate"
export TMPDIR="${scratch_job_root}"
export PYTHONUNBUFFERED=1
cd "${code_root}"

solver_names="${OTFLOW_SOLVER_NAMES:-euler,heun,midpoint_rk2,dpmpp2m}"
target_nfe_values="${OTFLOW_TARGET_NFE_VALUES:-10,12,16}"
scheduler_names="${OTFLOW_SCHEDULER_NAMES:-uniform,ays,gits,ots}"
seeds="${OTFLOW_SEEDS:-0,1,2}"
num_eval_samples="${OTFLOW_NUM_EVAL_SAMPLES:-25}"
eval_windows_val="${OTFLOW_EVAL_WINDOWS_VAL:-0}"
eval_windows_test="${OTFLOW_EVAL_WINDOWS_TEST:-0}"

echo "TASK_INDEX=${task_index}"
echo "DATASET=${dataset}"
echo "TRAIN_STEPS=${train_steps}"
echo "OUT_ROOT=${out_root}"
echo "MANIFEST=${manifest_path}"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi

python diffusion_flow_time_reparameterization.py \
  --allow_execute \
  --out_root "${out_root}" \
  --dataset_root "${repo_root}/paper_datasets" \
  --backbone_manifest "${manifest_path}" \
  --otflow_train_steps "${train_steps}" \
  --forecast_datasets "" \
  --lob_datasets "${dataset}" \
  --cryptos_path "${repo_root}/data/cryptos_binance_spot_monthly_1s_l10.npz" \
  --es_path "${repo_root}/data/es_mbp_10.npz" \
  --sleep_edf_path "${repo_root}/data/sleep_edf_3ch_100hz_stage_conditioned.npz" \
  --solver_names "${solver_names}" \
  --target_nfe_values "${target_nfe_values}" \
  --baseline_scheduler_names "${scheduler_names}" \
  --seeds "${seeds}" \
  --num_eval_samples "${num_eval_samples}" \
  --eval_windows_val "${eval_windows_val}" \
  --eval_windows_test "${eval_windows_test}"
