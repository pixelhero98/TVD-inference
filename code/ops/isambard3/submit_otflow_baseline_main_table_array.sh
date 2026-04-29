#!/bin/bash
set -euo pipefail

projectdir="${PROJECTDIR:-/projects/b35z}"
repo_root="${projectdir}/tvd-scheduler/TVD-Scheduler"
code_root="${repo_root}/code"
log_root="${repo_root}/TVD-result/experiments/slurm"
mkdir -p "${log_root}"

dependency_directive=""
if [[ -n "${SBATCH_DEPENDENCY:-}" ]]; then
  dependency_directive="#SBATCH --dependency=${SBATCH_DEPENDENCY}"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=dftr-baseline-main
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-7%2
#SBATCH --output=${log_root}/%x.%A_%a.out
${dependency_directive}

bash "${code_root}/ops/isambard3/run_otflow_baseline_main_table_job.sh"
EOF
