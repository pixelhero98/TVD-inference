# OTFlow Schedule Paper Path

This paper-facing path treats the trained model as a fixed `OTFlow` backbone and studies only inference-schedule and solver design.

## Scope

- Backbone: `OTFlow`
- Benchmark family A: Monash extrapolation
- Benchmark family B: LOB conditional generation
- Main deployable method: `TVD-p schedule`
- Legacy reproduction method: `legacy_hybrid_signal`
- Published mapped baselines: `AYS`, `GITS`, `OTS`
- Main fixed-grid ODE solvers: `Euler`, `Heun / RK2`, `Midpoint RK2`, `Dopri5`, `RK45`
- Solver-transfer only: `DPM++2M`

The canonical experiment plan lives at the project root in `../plan.md`.

## Frozen Backbone Preset

The paper runners fix the tested backbone to:

- `levels=10`
- `history_len=256`
- `ctx_encoder="hybrid"`
- `ctx_causal=True`
- `ctx_local_kernel=7`
- `ctx_pool_scales="8,32"`
- `use_time_features=True`

Architecture/loss-upgrade knobs are held at zero in this path. The contribution is schedule design, not model redesign. For timestamped datasets, the context encoder receives two generic time channels per history step:

- `log_gap`: log-scaled elapsed gap from the previous observation
- `elapsed_time`: normalized cumulative elapsed time within the sampled history window

This prep pass does not launch training or evaluation jobs. It only prepares dataset adapters, baseline registries, runner plumbing, and table scaffolding.

## Entry Point

Use `otflow_schedule_paper.py` for paper-facing prep and benchmark orchestration.

Example:

```bash
python otflow_schedule_paper.py \
  --dataset_root ./paper_datasets \
  --out_root ./results_otflow_paper_prep
```

## Outputs

The prep suite writes:

- `paper_prep_summary.json`
- `forecast_extrapolation_prep_summary.json`
- `lob_conditional_generation_prep_summary.json`
- `rollout_length_review.json`

It also prepares table layout stubs for:

- Monash extrapolation main tables: `CRPS`, `MSE`
- LOB conditional-generation pilot tables: `score_main`, `latency_ms_per_sample`, `realized_nfe`
