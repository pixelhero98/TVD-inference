# TVD-Sampler

This workspace isolates the **sampler-side** exploration from the main LoBiFlows project and from the regularization experiments.

The working hypothesis is:
- a frozen LoBiFlow model may already contain useful local error information in its rollout dynamics
- that information may be extractable from **temporal velocity disagreement**
- a sampler can use that signal to spend deterministic solver budget more efficiently than uniform allocation

In short: this folder is for testing whether a disagreement signal can improve inference **without retraining the generator**.

## Aim

The current sampler research question is:

> Can disagreement-guided deterministic refinement beat plain Euler at matched field-evaluation budget?

The specific signal being tested is the cosine disagreement between consecutive rollout velocities:
- `d_n = 1 - cos(v_n, EMA(v))`

The current actuator being tested is:
- rollback + two half-Euler substeps

The benchmark policy is:
- compare methods at matched **total field-evaluation budgets**
- not matched macro-step counts

## Current Story

What has already been established:

1. Adaptive noise is not the right correction.
- The earlier stochastic corrector pilot was negative on `cryptos`.
- Adaptive noise was better than fixed noise, but pure Euler remained best.
- Interpretation: the signal is detecting **numerical stress**, not a need for stochastic exploration.

2. The disagreement signal itself is useful.
- In Stage A of the deterministic refinement pilot, raw disagreement correlated strongly with a cheap oracle local-error proxy.
- Normalized z-score disagreement was much weaker.
- Interpretation: the detector has signal, but the first controller transform was wrong.

3. The pressure is concentrated late in the rollout.
- Per-step disagreement rises sharply near the end of the solver trajectory.
- That means any controller has to be compared against a trivial but strong baseline: **always refine the late steps**.

These findings motivated the current adjustment:
- drop z-score triggering
- use **raw disagreement with step-wise percentile gating**
- benchmark against **fixed late-step deterministic refinement**

## Current Stage

Finished experiment:
- `results/results_adaptive_deterministic_refinement_followup_cryptos_seed0_pctgate_20260326`

Completed outputs:
- `model_seed0.pt`
- `trace_calibration.json`
- `oracle_error_analysis.json`
- `stage_b_summary.json`
- Stage A and Stage B plots

What Stage B compared:
- `Euler`
- `Heun`
- `adaptive percentile-gated half-step refinement`
- `fixed late-step half-step refinement`

Budget families:
- `8`
- `16`

Percentile triggers:
- `0.85`
- `0.90`
- `0.95`

Fixed late-step baselines:
- last actionable step
- last two actionable steps

## Current Findings

Stage A:
- raw disagreement tracked the oracle local-error proxy strongly
  - budget `8`: Pearson `0.632`, Spearman `0.851`
  - budget `16`: Pearson `0.655`, Spearman `0.908`
- normalized disagreement was much weaker
- conclusion: raw disagreement is the useful detector; z-score normalization was not

Stage B:
- budget `8`
  - `Euler` was best: `score_main = 0.9072`, `conditional_w1 = 2.3306`
  - best adaptive percentile-gated variant (`p90`) was worse: `0.9561`
  - fixed late-step refinement was also worse: `0.9894`
  - `Heun` was much worse: `1.4341`
  - diagnosis: `euler_already_robust`
- budget `16`
  - best adaptive percentile-gated variant (`p85`) was best overall:
    - `score_main = 0.9238`
    - `conditional_w1 = 2.4479`
  - `Euler` baseline:
    - `score_main = 0.9573`
    - `conditional_w1 = 2.4594`
  - fixed late-step refinement was worse: `0.9822`
  - `Heun` was worse: `1.2682`
  - diagnosis: `signal_validated`

Important nuance:
- the adaptive controller remained sparse
- realized adaptive field-eval usage stayed well below the nominal budget families
- that means the trigger is informative, but the current percentile grid does not yet spend the intended budget aggressively

## Decision Logic

The next interpretation will be:

1. If adaptive percentile-gated refinement beats both Euler and fixed late-step refinement:
- the disagreement signal is doing real sample-specific budget allocation

2. If fixed late-step refinement matches or beats adaptive refinement:
- the signal is mainly telling us that late steps are hard
- useful, but weaker than a true sample-specific controller

3. If Heun beats Euler but adaptive refinement does not:
- the main benefit is just "use a better deterministic solver"

4. If none of them beat Euler:
- the current low-NFE Euler baseline is already robust on this setup

## Workspace Layout

- `code/`: minimal LoBiFlow code required for sampler experiments
- `results/`: sampler pilot outputs and logs
- `data/`: symlink to the crypto NPZ used by the pilots

Included code:
- `adaptive_deterministic_refinement_followup.py`
- `adaptive_noise_sampler_followup.py`
- `benchmark_lobiflow_suite.py`
- `experiment_common.py`
- `lob_train_val.py`
- `lob_model.py`
- `lob_datasets.py`
- `lob_utils.py`
- `lob_baselines.py`
- `baselines.py`
- `conditioning.py`
- `config.py`
- `test_lobiflow.py`

Python environment used on the remote machine:
- `/home/yzn/work/uv_torch_project/.venv/bin/python`

## Example Run

```bash
cd /home/yzn/work/TVD-Sampler/code
/home/yzn/work/uv_torch_project/.venv/bin/python adaptive_deterministic_refinement_followup.py \
  --cryptos_path /home/yzn/work/TVD-Sampler/data/cryptos_binance_spot_monthly_1s_l10.npz \
  --out_root /home/yzn/work/TVD-Sampler/results/my_run \
  --device cuda \
  --steps 12000 \
  --eval_horizon 200 \
  --eval_windows_val 30 \
  --eval_windows_test 30 \
  --budget_families 8,16 \
  --percentile_values 0.85,0.90,0.95 \
  --fixed_last_k_values 1,2
```

## Next Steps

Immediate:
- confirm the budget-`16` adaptive result on more seeds
- retune the percentile/controller settings so adaptive variants spend budget closer to the target families
- add per-variant progress logging and partial result writes to future pilots

After that:
- if the budget-`16` result holds, test transfer to `es_mbp_10`
- if it does not hold, keep the detector as a useful diagnostic and stop the operational controller branch

## Notes

- The finished percentile-gated run should be kept here as an independent sampler result bundle.
- The original LoBiFlows project was left unchanged to avoid breaking existing experiments.
