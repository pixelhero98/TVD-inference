# TSFlow Backbone Lane

This note defines how `TSFlow` is used in the `TVD-Scheduler` project.

## Scope

- `TSFlow` is a forecast-only backbone lane
- It is intended for the 5 forecast extrapolation datasets only
- It is not in scope for LOB adaptation in the current phase

## Source Policy

- Official repo: [marcelkollovieh/TSFlow](https://github.com/marcelkollovieh/TSFlow)
- Use TSFlow as an external companion checkout
- Do not vendor TSFlow source into the `TVD-Scheduler` repo or the checkpoint handoff zip

## Upstream Training Entry Point

The upstream README exposes the training entry point:

```bash
python bin/train_model.py -c configs_local/train_conditional.yaml
```

The public config examples also expose `num_steps` and `solver`, which is enough for backbone training, but fixed-grid scheduler integration is still a separate follow-up task.

## Current Role

- Use TSFlow first as a second SOTA forecasting backbone
- Train it on separate machines in parallel with OTFlow scheduler jobs
- Report it as a backbone-comparison lane, not yet as a TVD scheduler lane

## Deferred Work

Before any fixed-grid TVD evaluation is attempted on TSFlow, add a dedicated adapter that:

- emits the same experiment summary schema used by OTFlow runs
- records explicit `backbone_name`
- aligns forecast metrics with the OTFlow reporting path
- adds scheduler-grid support only if fixed-grid evaluation is actually needed
