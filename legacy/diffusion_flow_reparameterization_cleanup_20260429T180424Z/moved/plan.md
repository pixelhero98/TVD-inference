# OTFlow Paper Plan

This is the canonical paper-v1 plan for the local `TVD-Scheduler` checkout.

## Paper-v1 Scope

- Main-text datasets are locked to the 8 active OTFlow benchmarks:
  - `wind_farms_wo_missing`
  - `san_francisco_traffic`
  - `london_smart_meters_wo_missing`
  - `electricity`
  - `solar_energy_10m`
  - `cryptos`
  - `sleep_edf`
  - `es_mbp_10`
- Medical forecast note:
  - `long_term_headered_ECG_records` is dropped from the active campaign after repeated pre-step OOM on Isambard3 and remains historical debugging evidence only.
- Appendix TVD method:
  - canonical `TVD` with information-growth hardness, `r_*` scaling, validation-selected `delta` from `{0.02, 0.05, 0.10, 0.20}`, no density ceiling, and continuous inverse-CDF interpolation
  - TVD is excluded from the main table and reported only in appendix / diagnostics unless a later completed controlled study changes this policy.
- Required solver set for the main baseline matrix and appendix TVD sensitivity:
  - `euler`
  - `heun`
  - `midpoint_rk2`
  - `dpmpp2m`
- Required scheduler set for the baseline-only main paper matrix:
  - `uniform`
  - `late_power_3`
  - `AYS`
  - `GITS`
  - `OTS`
- Archived legacy TVD evidence, not an active scheduler path:
  - `tvdp_unified`
  - `tvdp_interpolated`
  - thermo / ceiling / no-`r_*` ablations as historical evidence only
- Legacy / appendix non-main schedules:
  - `FlowTS power sampling`
- Appendix adaptive solver identities:
  - `dopri5_adaptive` (`Dopri5 (adaptive)`)
  - `rk45_adaptive` (`RK45 (adaptive)`)
- Main NFEs:
  - `10`
  - `12`
  - `16`
- Appendix extra NFEs, deferred until after the main study locks:
  - `6`
  - `8`
  - `20`
  - `24`

## Locked Dataset Setup

All paper-v1 experiments keep horizon-wise non-AR rollout:

\[
\text{future\_block\_len} = \text{experiment\_horizon}
\]

| Dataset | Family | History Length | Horizon | Future Block Length | Axis |
|---|---|---:|---:|---:|---|
| `wind_farms_wo_missing` | Forecast | 1440 | 1440 | 1440 | physical time |
| `san_francisco_traffic` | Forecast | 336 | 168 | 168 | physical time |
| `london_smart_meters_wo_missing` | Forecast | 672 | 336 | 336 | physical time |
| `electricity` | Forecast | 336 | 168 | 168 | physical time |
| `solar_energy_10m` | Forecast | 1008 | 1008 | 1008 | physical time |
| `cryptos` | LOB | 256 | 200 | 200 | event count |
| `es_mbp_10` | LOB | 256 | 200 | 200 | event count |
| `sleep_edf` | LOB | 12000 | 3000 | 3000 | physical time |

## Reporting Policy

### Forecast Extrapolation

- Main metrics:
  - `relative_crps_gain_vs_uniform`
  - `MASE`
  - average tied rank per scheduler for each main metric
- Appendix table metrics:
  - raw `CRPS`
  - raw `MSE`

### Conditional Generation

- Main metrics:
  - `relative_score_gain_vs_uniform`
  - `conditional_w1` (`C-W1`)
  - `tstr_macro_f1` (`TSTR Macro F1`)
  - average tied rank per scheduler for each main metric
- Paper-facing conditional-generation runs should default to the full metric bundle; `score_main_only` is only for pilot / debug passes.
- Appendix table metrics:
  - `score_main`
  - `unconditional_w1`
  - `conditional_w1`
  - `tstr_macro_f1`
- Appendix / audit-only metrics:
  - raw `disc_auc`
  - `u_l1`
  - `c_l1`
  - per-stat W1/L1 breakdowns
  - threshold metadata
  - sleep-specific counts when present

## Ranking Policy

- Compute ranks per matched paper cell across schedulers with average tied ranks.
- Main-table ranks are computed across baseline schedulers only: `uniform`, `late_power_3`, `AYS`, `GITS`, and `OTS`.
- Appendix TVD ranks must be reported separately from the baseline-only main table.
- Matching key:
  `dataset x solver x target_nfe x backbone x train_steps x checkpoint x split x experiment_scope`
- Higher is better:
  - `relative_crps_gain_vs_uniform`
  - `relative_score_gain_vs_uniform`
  - `tstr_macro_f1`
- Lower is better:
  - `MASE`
  - `conditional_w1`
  - `score_main`
- Report mean rank by averaging the per-cell ranks, never by ranking global mean metrics.

## Execution Order

1. Normalize the imported Isambard OTFlow checkpoints into the canonical matrix root under `TVD-result/results/backbone_matrix/otflow/...`.
2. Lock the active backbone inventory to the strict 40-artifact campaign:
   - Forecast:
     - `san_francisco_traffic`: `4k,8k,12k,16k,20k`
     - `london_smart_meters_wo_missing`: `4k,8k,12k,16k,20k`
     - `electricity`: `4k,8k,12k,16k,20k`
     - `solar_energy_10m`: `4k,8k,12k,16k,20k`
     - `wind_farms_wo_missing`: `4k,8k,12k,16k,20k`
   - Conditional generation:
     - `cryptos`: `4k,8k,12k,16k,20k`
     - `es_mbp_10`: `4k,8k,12k,16k,20k`
     - `sleep_edf`: `4k,8k,12k,16k,20k`
3. Finish the remaining strict-grid OTFlow backbones after normalization:
   - `forecast_extrapolation / electricity / 4k`
   - `forecast_extrapolation / solar_energy_10m / 4k`
   - `forecast_extrapolation / wind_farms_wo_missing / 4k`
   - `lob_conditional_generation / cryptos / 8k`
   - `lob_conditional_generation / cryptos / 12k`
   - `lob_conditional_generation / cryptos / 16k`
   - `lob_conditional_generation / es_mbp_10 / 4k`
   - `lob_conditional_generation / es_mbp_10 / 8k`
   - `lob_conditional_generation / es_mbp_10 / 12k`
   - `lob_conditional_generation / es_mbp_10 / 16k`
   - `lob_conditional_generation / sleep_edf / 8k`
   - `lob_conditional_generation / sleep_edf / 12k`
   - `lob_conditional_generation / sleep_edf / 16k`
   - `lob_conditional_generation / sleep_edf / 20k`
4. Keep canonical `TVD` appendix-only and archive the older `tvdp_*`, thermo, ceiling, and no-`r_*` families outside active code/result paths.
5. Run the required baseline-only main table at `10,12,16` with:
   - solvers: `euler`, `heun`, `midpoint_rk2`, `dpmpp2m`
   - baselines: `uniform`, `late_power_3`, `AYS`, `GITS`, `OTS`
6. Run appendix canonical `TVD` validation at `10,12,16`, select one `delta` per dataset, then run the locked canonical `TVD` appendix matrix with the same 4 fixed-grid solvers.
7. Run appendix canonical `TVD` delta sensitivity on the full 8-dataset paper scope:
   - forecast: `wind_farms_wo_missing`, `san_francisco_traffic`, `london_smart_meters_wo_missing`, `electricity`, `solar_energy_10m`
   - conditional generation: `cryptos`, `sleep_edf`, `es_mbp_10`
   - solvers: `euler`, `heun`, `midpoint_rk2`, `dpmpp2m`
   - NFEs: `10`, `12`, `16`
   - deltas: `0.02`, `0.05`, `0.10`, `0.20`
8. Add representative qualitative extrapolation figures for the forecast family:
   - datasets: `solar_energy_10m`, `electricity`
   - shared representative cell: `heun`, `NFE=12`
   - compare the best matched baseline schedulers in the same `dataset x solver x NFE` cell
9. Run the appendix adaptive-solver efficiency study:
   - datasets: `solar_energy_10m`, `electricity`, `cryptos`
   - adaptive solver ids: `dopri5_adaptive`, `rk45_adaptive`
   - paper-facing labels: `Dopri5 (adaptive)`, `RK45 (adaptive)`
   - schedule policy: internal adaptivity only, with no external TVD or baseline schedule grid
   - target policy: match the best fixed-step canonical `TVD` result for the same dataset / split / backbone context
   - tolerance sweep: `1e-2`, `1e-3`, `1e-4`
   - report: achieved metric, realized NFE, inference latency, smallest realized NFE that matches the fixed-step TVD target (if any), `nfe_delta_vs_fixed_tvd`, and `latency_delta_vs_fixed_tvd`
10. Run the deferred appendix NFE sweep afterward.

## Current Priority

Next remote execution: conditional-generation baseline-only main table for `cryptos`, `es_mbp_10`, `sleep_edf`, NFEs `10,12,16`, solvers `euler,heun,midpoint_rk2,dpmpp2m`, seeds `0,1,2`.
