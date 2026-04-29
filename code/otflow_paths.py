from __future__ import annotations

import os
from pathlib import Path


def code_root() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return code_root().parent


def project_data_root() -> Path:
    return project_root() / "data"


def project_paper_dataset_root() -> Path:
    return project_root() / "paper_datasets"


def project_results_root() -> Path:
    return project_root() / "results"


def project_checkpoint_import_root() -> Path:
    return project_root() / "checkpoints" / "imported_otflow_schedule_12k"


def project_medical_staging_root() -> Path:
    raw = str(os.environ.get("OTFLOW_MEDICAL_STAGING_ROOT", "") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path("/home/yzn/work/medical data")


def default_cryptos_data_path() -> str:
    return str(project_data_root() / "cryptos_binance_spot_monthly_1s_l10.npz")


def default_es_mbp_10_data_path() -> str:
    return str(project_data_root() / "es_mbp_10.npz")


def default_sleep_edf_data_path() -> str:
    return str(project_data_root() / "sleep_edf_3ch_100hz_stage_conditioned.npz")
