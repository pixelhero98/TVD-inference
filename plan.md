# OTFlow Paper Prep Plan

This document is the authoritative tracker for the next OTFlow paper-prep pass.

## Thesis

The paper studies a fixed `OTFlow` backbone and asks how inference schedules and solver choices change performance at matched realized NFE.

This pass is **prep only**:

- allowed: dataset preparation, baseline integration, runner/framework plumbing, summary/table scaffolding, dry-run validation
- not allowed: training runs, evaluation sweeps, pilot jobs, full benchmark execution

## Benchmark Family A: Monash Extrapolation

Datasets:

- `wind_farms_wo_missing`
- `san_francisco_traffic`
- `london_smart_meters_wo_missing`
- `electricity`
- `solar_energy_10m`

Policy:

- use the official forecast horizon from prepared dataset metadata
- use single-tail-holdout validation
- main metrics: `CRPS`, `MSE`
- appendix metric: `R^2` only if stable

## Benchmark Family B: LOB Conditional Generation

Datasets:

- `cryptos`
- `es_mbp_10`

Policy:

- keep the current history-conditioned OTFlow backbone fixed
- main schedule method: `hybrid_signal`
- main ablation: raw `disagreement`
- main table metrics: `score_main`, `unconditional_w1`, `conditional_w1`
- appendix: full conditional-generation diagnostics

## Baseline Matrix

Main NFE:

- `10`
- `12`
- `16`

Appendix NFE:

- `6`
- `8`
- `20`
- `24`

Schedules:

- `uniform`
- `late-power-2`
- `late-power-3`
- `early-biased-control`
- `AYS`
- `GITS`
- `OTS`
- `hybrid_signal`
- `disagreement`

Solvers:

- `Euler`
- `Heun / RK2`
- `Midpoint RK2`
- `DPM++2M`

Scope rules:

- `AYS`, `GITS`, and `OTS` are mapped fixed schedules and are Euler-only in the main matrix
- solver baselines are compared separately on uniform/simple fixed grids
- `ATSS` and the wider DPM-Solver family are deferred

## Implementation Checklist

- add paper-only Monash dataset adapter scaffolding
- add paper schedule registry
- add paper solver registry
- add dry-run forecast extrapolation runner
- add dry-run LOB conditional-generation runner
- add joint paper suite runner
- add table schema builders for forecast and LOB results
- add external schedule source catalog
- add tests for registries, manifests, dry-run summaries, and table layouts

## Blocked / Deferred

- actual Monash dataset download and cache generation
- imported numeric knot values for `AYS`, `GITS`, and `OTS`
- OTFlow runtime integration for `Midpoint RK2`
- benchmark execution
- multi-seed training/evaluation

## Run Staging Checklist

1. finish prep pass and dry-run validation
2. verify dataset manifests for all five Monash datasets
3. import and validate published schedule knots
4. run one-seed pilot
5. freeze matrix
6. run three-seed main experiments
