
# Diffusion-Flow-Inference

Diffusion-Flow-Inference evaluates optimized diffusion schedules after mapping them onto normalized flow time for fixed OTFlow backbones. The active schedules are `uniform`, `late_power_3`, `ays`, `gits`, and `ots`; the transferred diffusion schedules are `ays`, `gits`, and `ots`.

## Active Code Path

- `code/diffusion_flow_time_reparameterization.py`: fixed-schedule evaluation entrypoint.
- `code/diffusion_flow_schedules.py`: schedule construction for uniform, late-power-3, AYS, GITS, and OTS.
- `code/otflow_evaluation_support.py`: checkpoint loading, dataset split resolution, solver mappings, and metric helpers.
- `code/otflow_paper_registry.py`: method, schedule, and solver registry.
- `code/otflow_model.py`, `code/conditioning.py`, `code/config.py`, `code/modules.py`, and `code/otflow_train_val.py`: OTFlow backbone model and training/evaluation utilities.

## Data, Outputs, And Backbones

This repository is source-only. Local runs may use `data/`, `paper_datasets/`, `outputs/`, and a virtual environment such as `.venv/`, but those large or machine-local directories are intentionally ignored and are not part of the public source tree.

Generated outputs default to:

```text
outputs/
```

The default backbone manifest path is:

```text
outputs/backbone_matrix/backbone_manifest.json
```

If you have a prepared local backbone matrix, it should report 40 ready checkpoint artifacts and 0 missing artifacts. The public smoke tests do not require those private/local artifacts.

## Environment

Pip:

```bash
python -m pip install -r requirements.txt
```

Conda:

```bash
conda env create -f environment.conda.yml
```

Raw medical dataset preparation requires `OTFLOW_MEDICAL_STAGING_ROOT` to point at the local staging directory. Prepared dataset evaluation uses the processed files in `data/`.

## CPU Smoke Checks

Run the public, artifact-independent smoke tests from `code/` with your active Python environment:

```bash
cd code
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest -q test_backbone_matrix test_otflow_paper_prep test_ptg_observed_gain_figure test_adaptive_solver_matched_nfe_study test_hardness_mismatch_figure
```

If you have local datasets and backbone artifacts, dry-run prep from either the repository root or `code/` accepts the same project-relative manifest path:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python code/diffusion_flow_time_reparameterization.py --forecast_datasets '' --lob_datasets '' --backbone_manifest outputs/backbone_matrix/backbone_manifest.json
```
