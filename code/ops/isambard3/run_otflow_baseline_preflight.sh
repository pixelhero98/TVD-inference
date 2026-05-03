
#!/bin/bash
set -euo pipefail

module load cray-python/3.11.7 2>/dev/null || true

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${REPO_ROOT:-$(cd "${script_dir}/../../.." && pwd)}"
output_root="${OUTPUT_ROOT:-${repo_root}/outputs}"
code_root="${repo_root}/code"
venv_dir="${VENV_DIR:-${repo_root}/.venv}"
matrix_root="${BACKBONE_MATRIX_ROOT:-${output_root}/backbone_matrix}"
manifest_path="${BACKBONE_MANIFEST:-${matrix_root}/backbone_manifest.json}"
scratch_base="${SCRATCHDIR:-${TMPDIR:-/tmp}}"
scratch_job_root="${scratch_base}/diffusion-flow-inference/${SLURM_JOB_ID:-manual}-baseline-preflight"

mkdir -p "${scratch_job_root}"

if [[ ! -f "${venv_dir}/bin/activate" ]]; then
  echo "Missing virtual environment at ${venv_dir}. Run code/ops/isambard3/bootstrap_env.sh first." >&2
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

echo "REPO_ROOT=${repo_root}"
echo "OUTPUT_ROOT=${output_root}"
echo "CODE_ROOT=${code_root}"
echo "VENV_DIR=${venv_dir}"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi

python - <<PY
import importlib
import json
from pathlib import Path

modules = ("numpy", "torch", "scipy", "pyedflib")
results = {}
for name in modules:
    try:
        mod = importlib.import_module(name)
        results[name] = {"ok": True, "version": getattr(mod, "__version__", "unknown")}
    except Exception as exc:
        results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
print({"imports": results})

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
ready = [item for item in manifest.get("artifacts", []) if item.get("status") == "ready"]
missing_ckpts = [item.get("checkpoint_path") for item in ready if not Path(str(item.get("checkpoint_path"))).exists()]
print({"manifest_path": str(manifest_path), "ready_count": len(ready), "missing_checkpoint_paths": missing_ckpts[:5]})
if int(manifest.get("ready_count", 0)) != 40 or int(manifest.get("missing_count", -1)) != 0 or missing_ckpts:
    raise SystemExit(3)
PY
