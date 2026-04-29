#!/bin/bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <benchmark_family> <dataset_key> <train_steps>" >&2
  exit 1
fi

benchmark_family="$1"
dataset_key="$2"
train_steps="$3"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
walltime="12:00:00"
memory="64G"

case "${dataset_key}" in
  london_smart_meters_wo_missing)
    walltime="24:00:00"
    ;;
  sleep_edf)
    walltime="24:00:00"
    memory="128G"
    ;;
  cryptos|es_mbp_10)
    memory="128G"
    ;;
esac

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=fm-${dataset_key}-${train_steps}
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=${memory}
#SBATCH --time=${walltime}
#SBATCH --output=%x.%j.out

export BENCHMARK_FAMILY="${benchmark_family}"
export DATASET_KEY="${dataset_key}"
export TRAIN_STEPS="${train_steps}"

bash "${repo_root}/ops/isambard3/run_fm_backbone_job.sh"
EOF
