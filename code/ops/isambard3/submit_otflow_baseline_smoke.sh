#!/bin/bash
set -euo pipefail

projectdir="${PROJECTDIR:-/projects/b35z}"
repo_root="${projectdir}/tvd-scheduler/TVD-Scheduler"
code_root="${repo_root}/code"
log_root="${repo_root}/TVD-result/experiments/slurm"
mkdir -p "${log_root}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=dftr-baseline-smoke
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=${log_root}/%x.%j.out

export OTFLOW_DATASET_INDEX=3
export OTFLOW_SOLVER_NAMES=euler
export OTFLOW_TARGET_NFE_VALUES=10
export OTFLOW_BASELINE_SCHEDULER_NAMES=uniform,late_power_3,ays,gits,ots
export OTFLOW_SEEDS=0
export OTFLOW_NUM_EVAL_SAMPLES=2
export OTFLOW_OUT_ROOT="${repo_root}/TVD-result/experiments/results_otflow_baseline_schedule_fix_smoke"
bash "${code_root}/ops/isambard3/run_otflow_baseline_main_table_job.sh"
EOF
