
#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${REPO_ROOT:-$(cd "${script_dir}/../../.." && pwd)}"
output_root="${OUTPUT_ROOT:-${repo_root}/outputs}"
code_root="${repo_root}/code"
log_root="${output_root}/experiments/slurm"
mkdir -p "${log_root}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=dfi-baseline-smoke
#SBATCH --partition=${SLURM_PARTITION:-hopper}
#SBATCH --nodes=1
#SBATCH --gpus=${SLURM_GPUS:-1}
#SBATCH --mem=${SLURM_MEM:-128G}
#SBATCH --time=${SLURM_TIME:-24:00:00}

#SBATCH --output=${log_root}/%x.%A_%a.out

export REPO_ROOT="${repo_root}"
export OUTPUT_ROOT="${output_root}"
export OTFLOW_DATASET_INDEX=3
export OTFLOW_SOLVER_NAMES=euler
export OTFLOW_TARGET_NFE_VALUES=10
export OTFLOW_SCHEDULER_NAMES=uniform,late_power_3,ays,gits,ots
export OTFLOW_SEEDS=0
export OTFLOW_NUM_EVAL_SAMPLES=2
export OTFLOW_OUT_ROOT="${output_root}/experiments/otflow_baseline_smoke"

bash "${code_root}/ops/isambard3/run_otflow_baseline_main_table_job.sh"
EOF
