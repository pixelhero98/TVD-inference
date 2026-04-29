#!/bin/bash
set -euo pipefail

module load cray-python/3.11.7

projectdir="${PROJECTDIR:-/projects/b35z}"
scratchdir="${SCRATCHDIR:-/scratch/b35z/$(id -un)}"
project_root="${projectdir}/tvd-scheduler"
repo_root="${project_root}/TVD-Scheduler"
code_root="${repo_root}/code"
venv_dir="${project_root}/.venv"
matrix_root="${repo_root}/TVD-result/results/backbone_matrix"
manifest_path="${OTFLOW_BACKBONE_MANIFEST:-${matrix_root}/backbone_manifest_lob_4k_16k.json}"
out_base="${OTFLOW_OUT_BASE:-${repo_root}/TVD-result/experiments/results_otflow_lob_baseline_backbone_sweep}"

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
scratch_job_root="${scratchdir}/tvd-scheduler/${SLURM_JOB_ID:-manual}-lob-${budget_label}-${dataset}"
out_root="${OTFLOW_OUT_ROOT:-${out_base}/${budget_label}/${dataset}}"

mkdir -p "${scratch_job_root}" "${out_root}"

if [[ ! -f "${venv_dir}/bin/activate" ]]; then
  echo "Missing virtual environment at ${venv_dir}. Run code/ops/isambard3/bootstrap_tvd_scheduler_venv.sh first." >&2
  exit 1
fi
if [[ ! -f "${manifest_path}" ]]; then
  echo "Missing backbone manifest at ${manifest_path}. Refresh the 4k-16k LOB manifest first." >&2
  exit 1
fi

source "${venv_dir}/bin/activate"
export TMPDIR="${scratch_job_root}"
export PYTHONUNBUFFERED=1
cd "${code_root}"

solver_names="${OTFLOW_SOLVER_NAMES:-euler,heun,midpoint_rk2,dpmpp2m}"
target_nfe_values="${OTFLOW_TARGET_NFE_VALUES:-10,12,16}"
baseline_scheduler_names="${OTFLOW_BASELINE_SCHEDULER_NAMES:-uniform,ays,gits,ots}"
seeds="${OTFLOW_SEEDS:-0,1,2}"
num_eval_samples="${OTFLOW_NUM_EVAL_SAMPLES:-25}"
eval_windows_val="${OTFLOW_EVAL_WINDOWS_VAL:-0}"
eval_windows_test="${OTFLOW_EVAL_WINDOWS_TEST:-0}"

echo "TASK_INDEX=${task_index}"
echo "DATASET=${dataset}"
echo "TRAIN_STEPS=${train_steps}"
echo "BUDGET_LABEL=${budget_label}"
echo "OUT_ROOT=${out_root}"
echo "MANIFEST=${manifest_path}"
echo "SOLVERS=${solver_names}"
echo "NFES=${target_nfe_values}"
echo "BASELINES=${baseline_scheduler_names}"
echo "SEEDS=${seeds}"
nvidia-smi

python diffusion_flow_time_reparameterization.py \
  --baseline_only \
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
  --baseline_scheduler_names "${baseline_scheduler_names}" \
  --seeds "${seeds}" \
  --num_eval_samples "${num_eval_samples}" \
  --eval_windows_val "${eval_windows_val}" \
  --eval_windows_test "${eval_windows_test}"
