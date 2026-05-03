
#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${REPO_ROOT:-$(cd "${script_dir}/../../.." && pwd)}"
output_root="${OUTPUT_ROOT:-${repo_root}/outputs}"
code_root="${repo_root}/code"
log_root="${output_root}/experiments/slurm"
mkdir -p "${log_root}"

dependency_directive=""
if [[ -n "${SBATCH_DEPENDENCY:-}" ]]; then
  dependency_directive="#SBATCH --dependency=${SBATCH_DEPENDENCY}"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=dfi-baseline-main
#SBATCH --partition=${SLURM_PARTITION:-hopper}
#SBATCH --nodes=1
#SBATCH --gpus=${SLURM_GPUS:-1}
#SBATCH --mem=${SLURM_MEM:-128G}
#SBATCH --time=${SLURM_TIME:-24:00:00}
#SBATCH --array=0-7%2
#SBATCH --output=${log_root}/%x.%A_%a.out
${dependency_directive}

export REPO_ROOT="${repo_root}"
export OUTPUT_ROOT="${output_root}"
bash "${code_root}/ops/isambard3/run_otflow_baseline_main_table_job.sh"
EOF
