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
manifest_path="${matrix_root}/backbone_manifest.json"
scratch_job_root="${scratchdir}/tvd-scheduler/${SLURM_JOB_ID:-manual}-baseline-preflight"

mkdir -p "${scratch_job_root}"

if [[ ! -f "${venv_dir}/bin/activate" ]]; then
  echo "Missing virtual environment at ${venv_dir}. Run code/ops/isambard3/bootstrap_tvd_scheduler_venv.sh first." >&2
  exit 1
fi
if [[ ! -d "${code_root}" ]]; then
  echo "Missing code root at ${code_root}" >&2
  exit 1
fi

source "${venv_dir}/bin/activate"
export TMPDIR="${scratch_job_root}"
export PYTHONUNBUFFERED=1
cd "${code_root}"

echo "PROJECTDIR=${projectdir}"
echo "SCRATCHDIR=${scratchdir}"
echo "REPO_ROOT=${repo_root}"
echo "CODE_ROOT=${code_root}"
echo "VENV_DIR=${venv_dir}"
nvidia-smi

python - <<PY
import importlib
import json
from pathlib import Path

modules = ("numpy", "torch", "pyedflib")
results = {}
for name in modules:
    try:
        mod = importlib.import_module(name)
        results[name] = {"ok": True, "version": getattr(mod, "__version__", "unknown")}
    except Exception as exc:
        results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
print({"imports": results})

import torch
print({"torch_cuda_available": bool(torch.cuda.is_available())})

repo_root = Path("${repo_root}")
required_paths = [
    repo_root / "data" / "cryptos_binance_spot_monthly_1s_l10.npz",
    repo_root / "data" / "es_mbp_10.npz",
    repo_root / "data" / "sleep_edf_3ch_100hz_stage_conditioned.npz",
    repo_root / "paper_datasets" / "monash" / "wind_farms_wo_missing" / "manifest.json",
    repo_root / "paper_datasets" / "monash" / "san_francisco_traffic" / "manifest.json",
    repo_root / "paper_datasets" / "monash" / "london_smart_meters_wo_missing" / "manifest.json",
    repo_root / "paper_datasets" / "monash" / "electricity" / "manifest.json",
    repo_root / "paper_datasets" / "monash" / "solar_energy_10m" / "manifest.json",
]
missing = [str(path) for path in required_paths if not path.exists()]
print({"missing_required_paths": missing})
if missing:
    raise SystemExit(2)

manifest_path = Path("${manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
ready_20k = [
    item for item in manifest.get("artifacts", [])
    if int(item.get("train_steps", -1)) == 20000 and item.get("status") == "ready"
]
print({
    "manifest_path": str(manifest_path),
    "artifact_count": manifest.get("artifact_count"),
    "ready_count": manifest.get("ready_count"),
    "missing_count": manifest.get("missing_count"),
    "ready_20k_count": len(ready_20k),
})
if len(ready_20k) != 8:
    raise SystemExit(3)
for item in ready_20k:
    checkpoint_path = Path(str(item["checkpoint_path"]))
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint from manifest: {checkpoint_path}")
PY
