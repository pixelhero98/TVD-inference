
#!/bin/bash
set -euo pipefail

module load cray-python/3.11.7 2>/dev/null || true

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${REPO_ROOT:-$(cd "${script_dir}/../../.." && pwd)}"
venv_dir="${VENV_DIR:-${repo_root}/.venv}"

if [[ ! -d "${repo_root}/code" ]]; then
  echo "Missing code root at ${repo_root}/code" >&2
  exit 1
fi

python3 -m venv "${venv_dir}"
source "${venv_dir}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${repo_root}/requirements.txt"
