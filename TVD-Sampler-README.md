# TVD-Sampler

## Hybrid Temporal Velocity Disagreement

The current implementation uses a hybrid temporal velocity disagreement signal computed from the rollout velocity field of a frozen `OTFlow` model.

At solver step `i`, let:

- `v_i` be the current flattened velocity field
- `m_i` be an exponential moving average of past velocities

The EMA reference is updated as:

`m_{i+1} = β m_i + (1 − β) v_i`

with the first step initialized from the first observed velocity.

The base temporal velocity disagreement is:

`d_i = 1 − cos(v_i, m_i) = 1 − <v_i, m_i> / (||v_i|| ||m_i|| + ε)`

The residual magnitude is:

`r_i = ||v_i − m_i||_2`

The current method uses the hybrid signal:

`h_i = r_i · d_i`

## How It Works

The signal compares the current solver velocity to a temporally smoothed reference direction. It is large when the current step is both:

- directionally inconsistent with the recent rollout trend
- far from that recent trend in magnitude

So the signal is not only asking whether the velocity turns, but also whether the turn happens with substantial geometric displacement.

In the current paper path, this hybrid temporal velocity disagreement is used **offline**:

1. Run validation rollouts with the frozen OTFlow backbone.
2. Record `h_i` at each solver step.
3. Average the per-step signal across validation windows.
4. Convert the averaged signal trace into a fixed nonuniform inference grid.
5. Reuse that fixed grid at test time for all samples.

This means the method uses temporal velocity disagreement as a schedule-design statistic, not as an online per-sample trigger.
