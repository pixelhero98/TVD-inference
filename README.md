# Diffusion-Flow Time Reparameterization

This project now studies transfer/conversion of optimized diffusion schedules into flow-time schedules for OTFlow backbones. The active paper position is **diffusion-flow time reparameterization**: keep the velocity-field/context-encoder backbone fixed, map diffusion schedules onto normalized flow time, and measure whether those schedules improve deterministic and adaptive solver behavior.

`TVD-result` is intentionally still used as a stable storage path during this cleanup. It should be read as legacy storage naming, not as the active paper method.

## Active Code Path

- `code/diffusion_flow_time_reparameterization.py`: active fixed-schedule evaluation entrypoint.
- `code/diffusion_flow_schedules.py`: uniform, late-power-3, AYS, GITS, and OTS schedule construction plus fixed-schedule LOB evaluation support.
- `code/otflow_evaluation_support.py`: checkpoint loading, split/window resolution, deterministic forecast evaluation, solver mappings, and metric helpers.
- `code/otflow_paper_registry.py`: active method registry with `diffusion_flow_time_reparameterization`.
- `code/otflow_model.py`, `code/conditioning.py`, `code/config.py`, `code/modules.py`, datasets, and `code/otflow_train_val.py`: active backbone training core for the velocity-field network and context encoder.

## Schedules And Solvers

Active deterministic schedules:

- `uniform`
- `late_power_3`
- `ays`
- `gits`
- `ots`

Transferred optimized diffusion schedules are exactly `ays`, `gits`, and `ots`. `uniform` and `late_power_3` are deterministic baselines and should not be used as transfer candidates for PTG or adaptive target selection.

Active deterministic solvers:

- `euler`
- `heun`
- `midpoint_rk2`
- `dpmpp2m`

Adaptive solver studies remain in `code/build_adaptive_solver_matched_nfe_study.py` for `rk45_adaptive` and `dopri5_adaptive` matched-NFE evaluation.

## Native Hardness And PTG

Native hardness is now the **info-growth** trace, exposed as `info_growth_hardness_by_step` by `code/otflow_signal_traces.py`. PTG and observed-gain tooling should treat info-growth as the paper-facing trace. Local-defect PTG remains available only as a diagnostic variant.

Use `code/build_ptg_observed_gain_figure.py` for PTG-vs-observed-integration-gain analysis. Use `code/build_hardness_mismatch_figure.py` for the neutral native info-growth trace and schedule-node visualization.

## Checkpoints

The active checkpoint matrix remains at:

`TVD-result/results/backbone_matrix/backbone_manifest.json`

The current manifest is expected to report 40 ready artifacts and 0 missing artifacts, covering the 4k-20k extrapolation and conditional-generation checkpoint matrix.

## Legacy Archive

Canonical TVD code, the old root `plan.md`, TVD-only support utilities, old imported 12k schedule checkpoints, retired result folders, and non-core extras were moved under:

`legacy/diffusion_flow_reparameterization_cleanup_20260429T180424Z/`

The archive manifest at `legacy/diffusion_flow_reparameterization_cleanup_20260429T180424Z/archive_manifest.json` records original paths, archive paths, sizes, and checksums. Do not delete archived artifacts unless a later cleanup explicitly supersedes this archive.

## Isambard Jobs

Active Isambard fixed-schedule jobs now call `diffusion_flow_time_reparameterization.py`. The broken FM backbone Slurm submission path that referenced the missing `fm_backbone_matrix.py` was archived; use `code/fm_backbone_readiness_audit.py` and the active backbone manifest for readiness checks.

## Tests

Focused tests:

```bash
cd /home/yzn/work/TVD-Scheduler/code
../.venv/bin/python -m unittest -q test_backbone_matrix test_otflow_paper_prep test_ptg_observed_gain_figure test_adaptive_solver_matched_nfe_study
```

Full discovery:

```bash
cd /home/yzn/work/TVD-Scheduler
.venv/bin/python -m unittest discover -s code -p "test_*.py"
```

## Open Points

- The physical directory names `TVD-Scheduler` and `TVD-result` are preserved for path stability. A later migration can rename storage roots after all scripts and external jobs stop depending on those paths.
- OTS currently uses the existing VP-time optimizer mapping. If the paper needs a different published OTS convention, update `code/diffusion_flow_schedules.py` and the registry together.
