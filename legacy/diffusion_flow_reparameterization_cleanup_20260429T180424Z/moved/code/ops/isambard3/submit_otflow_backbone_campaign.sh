#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <smoke|wave_a|sleep_smoke|wave_b>" >&2
  exit 1
fi

stage="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
submit_job="${repo_root}/ops/isambard3/submit_fm_backbone_job.sh"

submit() {
  local benchmark_family="$1"
  local dataset_key="$2"
  local train_steps="$3"
  bash "${submit_job}" "${benchmark_family}" "${dataset_key}" "${train_steps}"
}

case "${stage}" in
  smoke)
    submit forecast_extrapolation san_francisco_traffic 4000
    submit lob_conditional_generation cryptos 4000
    ;;
  wave_a)
    for dataset in wind_farms_wo_missing san_francisco_traffic london_smart_meters_wo_missing electricity solar_energy_10m; do
      for train_steps in 8000 12000 16000; do
        submit forecast_extrapolation "${dataset}" "${train_steps}"
      done
    done
    ;;
  sleep_smoke)
    submit lob_conditional_generation sleep_edf 4000
    ;;
  wave_b)
    for dataset in wind_farms_wo_missing london_smart_meters_wo_missing electricity solar_energy_10m; do
      submit forecast_extrapolation "${dataset}" 4000
    done
    for train_steps in 8000 12000 16000; do
      submit lob_conditional_generation cryptos "${train_steps}"
    done
    for train_steps in 4000 8000 12000 16000; do
      submit lob_conditional_generation es_mbp_10 "${train_steps}"
    done
    for train_steps in 4000 8000 12000 16000 20000; do
      submit lob_conditional_generation sleep_edf "${train_steps}"
    done
    ;;
  *)
    echo "Unsupported stage=${stage}. Use smoke, wave_a, sleep_smoke, or wave_b." >&2
    exit 1
    ;;
esac
