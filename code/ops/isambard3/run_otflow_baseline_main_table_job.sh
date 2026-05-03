
#!/bin/bash
set -euo pipefail

module load cray-python/3.11.7 2>/dev/null || true

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${REPO_ROOT:-$(cd "${script_dir}/../../.." && pwd)}"
output_root="${OUTPUT_ROOT:-${repo_root}/outputs}"
code_root="${repo_root}/code"
venv_dir="${VENV_DIR:-${repo_root}/.venv}"
manifest_path="${BACKBONE_MANIFEST:-${output_root}/backbone_matrix/backbone_manifest.json}"
out_base="${OTFLOW_OUT_BASE:-${output_root}/experiments/otflow_baseline_main_table_20k}"
scratch_base="${SCRATCHDIR:-${TMPDIR:-/tmp}}"

datasets=(
  "forecast_extrapolation:wind_farms_wo_missing"
  "forecast_extrapolation:san_francisco_traffic"
  "forecast_extrapolation:london_smart_meters_wo_missing"
  "forecast_extrapolation:electricity"
  "forecast_extrapolation:solar_energy_10m"
  "lob_conditional_generation:cryptos"
  "lob_conditional_generation:es_mbp_10"
  "lob_conditional_generation:sleep_edf"
)

dataset_index="${OTFLOW_DATASET_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
if [[ "${dataset_index}" -lt 0 || "${dataset_index}" -ge "${#datasets[@]}" ]]; then
  echo "Invalid dataset index: ${dataset_index}" >&2
  exit 2
fi

entry="${datasets[${dataset_index}]}"
family="${entry%%:*}"
dataset="${entry#*:}"
scratch_job_root="${scratch_base}/diffusion-flow-inference/${SLURM_JOB_ID:-manual}-${dataset}"
out_root="${OTFLOW_OUT_ROOT:-${out_base}/${dataset}}"

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

forecast_datasets=""
lob_datasets=""
if [[ "${family}" == "forecast_extrapolation" ]]; then
  forecast_datasets="${dataset}"
else
  lob_datasets="${dataset}"
fi

solver_names="${OTFLOW_SOLVER_NAMES:-euler,heun,midpoint_rk2,dpmpp2m}"
target_nfe_values="${OTFLOW_TARGET_NFE_VALUES:-10,12,16}"
scheduler_names="${OTFLOW_SCHEDULER_NAMES:-uniform,late_power_3,ays,gits,ots}"
seeds="${OTFLOW_SEEDS:-0,1,2}"
num_eval_samples="${OTFLOW_NUM_EVAL_SAMPLES:-25}"
eval_windows_val="${OTFLOW_EVAL_WINDOWS_VAL:-0}"
eval_windows_test="${OTFLOW_EVAL_WINDOWS_TEST:-0}"

echo "DATASET_INDEX=${dataset_index}"
echo "FAMILY=${family}"
echo "DATASET=${dataset}"
echo "OUT_ROOT=${out_root}"
echo "MANIFEST=${manifest_path}"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi

python diffusion_flow_time_reparameterization.py \
  --allow_execute \
  --out_root "${out_root}" \
  --dataset_root "${repo_root}/paper_datasets" \
  --backbone_manifest "${manifest_path}" \
  --otflow_train_steps 20000 \
  --forecast_datasets "${forecast_datasets}" \
  --lob_datasets "${lob_datasets}" \
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
