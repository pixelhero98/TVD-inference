#!/bin/bash
set -euo pipefail

projectdir="${PROJECTDIR:-/projects/b35z}"
repo_root="${projectdir}/tvd-scheduler/TVD-Scheduler"
code_root="${repo_root}/code"
log_root="${repo_root}/TVD-result/experiments/slurm"
mkdir -p "${log_root}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=dftr-baseline-preflight
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=${log_root}/%x.%j.out

bash "${code_root}/ops/isambard3/run_otflow_baseline_preflight.sh"
EOF
