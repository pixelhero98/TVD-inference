#!/bin/bash
set -euo pipefail

module load cray-python/3.11.7

: "${BENCHMARK_FAMILY:?BENCHMARK_FAMILY is required.}"
: "${DATASET_KEY:?DATASET_KEY is required.}"
: "${TRAIN_STEPS:?TRAIN_STEPS is required.}"

projectdir="${PROJECTDIR:-/lfs1i3/projects/b35z}"
scratchdir="${SCRATCHDIR:-/lfs1i3/scratch/b35z/$(id -un)}"

project_root="${projectdir}/tvd-scheduler"
repo_root="${project_root}/TVD-Scheduler"
venv_dir="${project_root}/.venv"
matrix_root="${repo_root}/results/backbone_matrix"
scratch_job_root="${scratchdir}/tvd-scheduler/${SLURM_JOB_ID:-manual}"

mkdir -p "${scratch_job_root}"
mkdir -p "${matrix_root}"

if [[ ! -f "${venv_dir}/bin/activate" ]]; then
  echo "Missing virtual environment at ${venv_dir}. Run ops/isambard3/bootstrap_tvd_scheduler_venv.sh first." >&2
  exit 1
fi

case "${BENCHMARK_FAMILY}" in
  forecast_extrapolation)
    forecast_datasets="${DATASET_KEY}"
    lob_datasets=""
    ;;
  lob_conditional_generation)
    forecast_datasets=""
    lob_datasets="${DATASET_KEY}"
    ;;
  *)
    echo "Unsupported BENCHMARK_FAMILY=${BENCHMARK_FAMILY}" >&2
    exit 1
    ;;
esac

source "${venv_dir}/bin/activate"
export TMPDIR="${scratch_job_root}"
export PYTHONUNBUFFERED=1

cd "${repo_root}"

run_matrix_job() {
  python fm_backbone_matrix.py \
    --allow_execute \
    --matrix_root "${matrix_root}" \
    --manifest_path "${matrix_root}/backbone_manifest.json" \
    --dataset_root "${repo_root}/paper_datasets" \
    --sleep_edf_path "${repo_root}/data/sleep_edf_3ch_100hz_stage_conditioned.npz" \
    --forecast_datasets "${forecast_datasets}" \
    --lob_datasets "${lob_datasets}" \
    --budget_steps "${TRAIN_STEPS}" \
    "$@"
}

if [[ "${BENCHMARK_FAMILY}" == "lob_conditional_generation" && "${DATASET_KEY}" == "sleep_edf" && "${TRAIN_STEPS}" == "4000" ]]; then
  set +e
  run_matrix_job
  first_status=$?
  set -e
  if [[ ${first_status} -ne 0 ]]; then
    echo "Sleep-EDF 4k smoke failed with status ${first_status}; retrying once with physical batch size 1 and --no_resume." >&2
    export OTFLOW_SLEEP_EDF_PHYSICAL_BATCH_SIZE=1
    run_matrix_job --no_resume
  fi
  exit 0
fi

run_matrix_job
