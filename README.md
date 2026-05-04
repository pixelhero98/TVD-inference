
# Diffusion-Flow-Inference

Diffusion-Flow-Inference evaluates optimized diffusion schedules after mapping them onto normalized flow time for fixed OTFlow backbones. The active schedules are `uniform`, `late_power_3`, `ays`, `gits`, and `ots`; the transferred diffusion schedules are `ays`, `gits`, and `ots`.

## Package Layout

- `diffusion_flow_inference.datasets`: dataset settings, loaders, audits, and experiment plans.
- `diffusion_flow_inference.backbones.settings`: backbone configs, modules, model definitions, and baseline architectures.
- `diffusion_flow_inference.backbones.training`: backbone training, benchmarking, readiness audits, and manifest registry.
- `diffusion_flow_inference.schedules`: diffusion-to-flow schedule construction and schedule registry.
- `diffusion_flow_inference.solvers`: solver runtime settings.
- `diffusion_flow_inference.evaluation`: fixed-schedule evaluation, metric support, and paper table helpers.
- `diffusion_flow_inference.diagnostics`: PTG, adaptive-solver, hardness, and signal-trace diagnostics.

## Data, Outputs, And Backbones

This repository is source-only. Local runs may use `data/`, `paper_datasets/`, `outputs/`, and a virtual environment such as `.venv/`, but those large or machine-local directories are intentionally ignored and are not part of the public source tree.

The processed experiment datasets can be kept as an external zip instead of checked into Git. The local bundle prepared for the experiments is:

- File: `diffusion_flow_processed_datasets.zip`
- Size: 2,908,513,477 bytes (2.71 GiB)
- SHA256: `17e31b4c0c313d977c14bfba4c781f6d10812d605b91266cfb612c499f3356ed`
- Contents: processed `data/` files plus Monash `paper_datasets/` sources/manifests; raw Monash download zips are excluded.

This bundle is too large for a GitHub source PR and should be distributed through an external artifact store rather than committed to the repository.

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
python -m pip install -e .
```

Conda:

```bash
conda env create -f environment.conda.yml
```

Raw medical dataset preparation requires `OTFLOW_MEDICAL_STAGING_ROOT` to point at the local staging directory. Prepared dataset evaluation uses the processed files in `data/`.

## CPU Smoke Checks

Run the public, artifact-independent smoke tests with your active Python environment:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -q
```

If you have local datasets and backbone artifacts, dry-run prep accepts the project-relative manifest path:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m diffusion_flow_inference.evaluation.diffusion_flow_time_reparameterization --forecast_datasets '' --lob_datasets '' --backbone_manifest outputs/backbone_matrix/backbone_manifest.json
```

To evaluate from an external processed dataset bundle, pass the zip path and let the runner extract it under `outputs/` when local processed inputs are missing:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m diffusion_flow_inference.evaluation.diffusion_flow_time_reparameterization \
  --dataset_bundle_zip /path/to/diffusion_flow_processed_datasets.zip \
  --dataset_bundle_mode auto \
  --dataset_bundle_extract_root outputs/dataset_bundles/extracted \
  --backbone_manifest outputs/backbone_matrix/backbone_manifest.json \
  --allow_execute
```
