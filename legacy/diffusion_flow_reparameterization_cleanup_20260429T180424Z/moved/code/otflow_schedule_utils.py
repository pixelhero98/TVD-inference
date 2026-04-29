from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from adaptive_deterministic_refinement_followup import _collect_rollout_diagnostics, _metric_bundle
from adaptive_noise_sampler_followup import _apply_sample_overrides, _restore_sample_overrides
from otflow_signal_traces import CANONICAL_INFO_GROWTH_TRACE_KEY, compute_canonical_info_growth_hardness_numpy
from otflow_train_val import eval_many_windows


def _resolved_reference_time_grid(
    density: Sequence[float],
    reference_time_grid: Optional[Sequence[float]],
) -> np.ndarray:
    density_arr = np.asarray(density, dtype=np.float64)
    if density_arr.ndim != 1:
        raise ValueError("density must be a one-dimensional sequence.")
    if density_arr.size == 0:
        return np.asarray([0.0, 1.0], dtype=np.float64)
    if reference_time_grid is None:
        return np.linspace(0.0, 1.0, int(density_arr.size) + 1, dtype=np.float64)
    grid_arr = np.asarray(reference_time_grid, dtype=np.float64)
    if grid_arr.ndim != 1 or grid_arr.size != int(density_arr.size) + 1:
        raise ValueError(
            "reference_time_grid must have length len(density) + 1 "
            f"({int(density_arr.size) + 1}), got {int(grid_arr.size)}."
        )
    if abs(float(grid_arr[0])) > 1e-8 or abs(float(grid_arr[-1]) - 1.0) > 1e-8:
        raise ValueError("reference_time_grid must start at 0.0 and end at 1.0.")
    if np.any(np.diff(grid_arr) <= 0.0):
        raise ValueError("reference_time_grid must be strictly increasing.")
    return grid_arr


def interpolated_equal_mass_grid(
    density: Sequence[float],
    n_steps: int,
    *,
    reference_time_grid: Optional[Sequence[float]] = None,
) -> Tuple[float, ...]:
    density_arr = np.asarray(density, dtype=np.float64)
    n_steps = int(n_steps)
    if density_arr.size == 0:
        return tuple(float(step_idx) / float(max(n_steps, 1)) for step_idx in range(max(n_steps, 1) + 1))
    reference_grid = _resolved_reference_time_grid(density_arr, reference_time_grid)
    cumulative = np.concatenate([[0.0], np.cumsum(density_arr)])
    total = float(cumulative[-1])
    grid = [0.0]
    for step_idx in range(1, n_steps):
        target = total * float(step_idx) / float(n_steps)
        bucket = int(np.searchsorted(cumulative, target, side="right") - 1)
        bucket = min(max(bucket, 0), int(density_arr.size) - 1)
        bucket_left_mass = float(cumulative[bucket])
        bucket_mass = float(density_arr[bucket])
        frac = 0.0 if bucket_mass <= 0.0 else (target - bucket_left_mass) / bucket_mass
        frac = min(max(frac, 0.0), 1.0)
        left_time = float(reference_grid[bucket])
        right_time = float(reference_grid[bucket + 1])
        grid.append(left_time + frac * (right_time - left_time))
    grid.append(1.0)
    for idx in range(1, len(grid)):
        if not grid[idx] > grid[idx - 1]:
            grid[idx] = min(1.0, grid[idx - 1] + 1e-6)
    grid[-1] = 1.0
    return tuple(float(x) for x in grid)


def _clean_nonnegative_signal_values(values: Sequence[float]) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float64)
    fill_value = float(np.nanmean(signal)) if np.any(np.isfinite(signal)) else 0.0
    signal = np.nan_to_num(signal, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return np.clip(signal, 0.0, None)


def canonical_normalized_hardness(values: Sequence[float]) -> np.ndarray:
    hardness = _clean_nonnegative_signal_values(values)
    mean_hardness = max(float(np.mean(hardness)), 1e-8)
    return np.clip(hardness / mean_hardness, 0.0, None)


def _normalize_probability(values: Sequence[float], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence.")
    arr = np.clip(arr, 0.0, None)
    total = float(np.sum(arr))
    if total <= 1e-12:
        raise ValueError(f"{name} must have positive total mass.")
    return arr / total


def _project_mass_shares_to_bounds(
    mass_shares: Sequence[float],
    *,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
) -> np.ndarray:
    shares = _normalize_probability(mass_shares, name="mass_shares")
    lower = np.asarray(lower_bounds, dtype=np.float64)
    upper = np.asarray(upper_bounds, dtype=np.float64)
    if shares.shape != lower.shape or shares.shape != upper.shape:
        raise ValueError("mass_shares, lower_bounds, and upper_bounds must have the same shape.")
    if np.any(lower < -1e-12):
        raise ValueError("lower_bounds must be non-negative.")
    if np.any(upper < lower - 1e-12):
        raise ValueError("upper_bounds must be greater than or equal to lower_bounds.")
    lower = np.clip(lower, 0.0, None)
    upper = np.maximum(upper, lower)
    lower_total = float(np.sum(lower))
    upper_total = float(np.sum(upper))
    if lower_total > 1.0 + 1e-10:
        raise ValueError("Lower mass bounds are infeasible because they sum above 1.")
    if upper_total < 1.0 - 1e-10:
        raise ValueError("Upper mass bounds are infeasible because they sum below 1.")

    projected = np.clip(shares, lower, upper)
    for _ in range(max(10, 4 * int(projected.size))):
        residual = 1.0 - float(np.sum(projected))
        if abs(residual) <= 1e-12:
            break
        if residual > 0.0:
            capacity = np.clip(upper - projected, 0.0, None)
            capacity_total = float(np.sum(capacity))
            if capacity_total <= 1e-12:
                break
            projected = np.minimum(upper, projected + residual * capacity / capacity_total)
        else:
            removable = np.clip(projected - lower, 0.0, None)
            removable_total = float(np.sum(removable))
            if removable_total <= 1e-12:
                break
            projected = np.maximum(lower, projected + residual * removable / removable_total)

    residual = 1.0 - float(np.sum(projected))
    if abs(residual) > 1e-9:
        raise ValueError("Unable to project interval masses onto feasible bounds.")
    if abs(residual) > 1e-12:
        if residual > 0.0:
            capacity = np.clip(upper - projected, 0.0, None)
            idx = int(np.argmax(capacity))
            projected[idx] = min(float(upper[idx]), float(projected[idx] + residual))
        else:
            removable = np.clip(projected - lower, 0.0, None)
            idx = int(np.argmax(removable))
            projected[idx] = max(float(lower[idx]), float(projected[idx] + residual))
    return np.clip(projected, 0.0, None)


def _apply_interval_mass_bounds(
    mass_shares: Sequence[float],
    interval_widths: Sequence[float],
    *,
    mass_floor_multiplier: Optional[float] = None,
    mass_cap_multiplier: Optional[float] = None,
) -> Dict[str, Any]:
    floor_multiplier = None if mass_floor_multiplier is None else float(mass_floor_multiplier)
    cap_multiplier = None if mass_cap_multiplier is None else float(mass_cap_multiplier)
    if floor_multiplier is not None and floor_multiplier < 0.0:
        raise ValueError(f"mass_floor_multiplier must be non-negative, got {mass_floor_multiplier}")
    if cap_multiplier is not None and cap_multiplier < 0.0:
        raise ValueError(f"mass_cap_multiplier must be non-negative, got {mass_cap_multiplier}")
    shares = _normalize_probability(mass_shares, name="mass_shares")
    widths = _normalize_probability(interval_widths, name="interval_widths")
    if shares.shape != widths.shape:
        raise ValueError("mass_shares and interval_widths must have the same shape.")
    lower = np.zeros_like(shares)
    upper = np.ones_like(shares)
    if floor_multiplier is not None and floor_multiplier > 0.0:
        lower = floor_multiplier * widths
    if cap_multiplier is not None and cap_multiplier > 0.0:
        upper = cap_multiplier * widths
    floor_hit_mask = shares < lower - 1e-12
    cap_hit_mask = shares > upper + 1e-12
    bounded = _project_mass_shares_to_bounds(
        shares,
        lower_bounds=lower,
        upper_bounds=upper,
    )
    return {
        "interval_masses": bounded,
        "mass_floor_hit_count": int(np.sum(floor_hit_mask)),
        "mass_floor_deficit_share": float(np.sum(np.clip(lower - shares, 0.0, None))),
        "mass_floor_min_share_after": float(np.min(bounded)) if bounded.size else None,
        "mass_cap_hit_count": int(np.sum(cap_hit_mask)),
        "mass_cap_overflow_share": float(np.sum(np.clip(shares - upper, 0.0, None))),
        "mass_cap_max_share_after": float(np.max(bounded)) if bounded.size else None,
    }


def _apply_interval_mass_cap(
    mass_shares: Sequence[float],
    interval_widths: Sequence[float],
    *,
    mass_cap_multiplier: float,
) -> Dict[str, Any]:
    return _apply_interval_mass_bounds(
        mass_shares,
        interval_widths,
        mass_cap_multiplier=float(mass_cap_multiplier),
    )


def canonical_interval_masses(
    values: Sequence[float],
    *,
    delta: float,
    solver_order: float,
    reference_time_grid: Sequence[float],
    uniform_blend: float = 0.0,
    gibbs_temperature: float = 1.0,
    mass_floor_multiplier: Optional[float] = None,
    mass_cap_multiplier: Optional[float] = None,
    hardness_tilt_gamma: float = 0.0,
) -> Dict[str, Any]:
    blend = float(uniform_blend)
    if blend < 0.0 or blend > 1.0:
        raise ValueError(f"uniform_blend must lie in [0, 1], got {uniform_blend}")
    temperature = float(gibbs_temperature)
    if temperature <= 0.0:
        raise ValueError(f"gibbs_temperature must be positive, got {gibbs_temperature}")
    tilt_gamma = float(hardness_tilt_gamma)
    if not np.isfinite(tilt_gamma):
        raise ValueError(f"hardness_tilt_gamma must be finite, got {hardness_tilt_gamma}")
    floor_multiplier = None if mass_floor_multiplier is None else float(mass_floor_multiplier)
    if floor_multiplier is not None and floor_multiplier < 0.0:
        raise ValueError(f"mass_floor_multiplier must be non-negative, got {mass_floor_multiplier}")
    cap_multiplier = None if mass_cap_multiplier is None else float(mass_cap_multiplier)
    if cap_multiplier is not None and cap_multiplier < 0.0:
        raise ValueError(f"mass_cap_multiplier must be non-negative, got {mass_cap_multiplier}")
    reference_grid = _resolved_reference_time_grid(values, reference_time_grid)
    interval_widths = np.diff(reference_grid)
    normalized = canonical_normalized_hardness(values)
    interval_midpoints = 0.5 * (reference_grid[:-1] + reference_grid[1:])
    tilt_weights = np.exp(tilt_gamma * (interval_midpoints - 0.5))
    tilted_normalized = tilt_weights * normalized
    potential = np.log(np.clip(float(delta) + tilted_normalized, 1e-12, None))
    raw_interval_masses = np.exp(potential / (float(solver_order) * temperature)) * interval_widths
    interval_masses = raw_interval_masses.copy()
    canonical_mass_share_total = max(float(np.sum(raw_interval_masses)), 1e-12)
    canonical_mass_shares = raw_interval_masses / canonical_mass_share_total
    if blend > 0.0:
        interval_masses = (1.0 - blend) * canonical_mass_shares + blend * interval_widths
    constraint_payload = {
        "mass_floor_hit_count": 0,
        "mass_floor_deficit_share": 0.0,
        "mass_floor_min_share_after": None,
        "mass_cap_hit_count": 0,
        "mass_cap_overflow_share": 0.0,
        "mass_cap_max_share_after": None,
    }
    if (
        floor_multiplier is not None
        and floor_multiplier > 0.0
        or cap_multiplier is not None
        and cap_multiplier > 0.0
    ):
        pre_bound_shares = interval_masses / max(float(np.sum(interval_masses)), 1e-12)
        constraint_payload = _apply_interval_mass_bounds(
            pre_bound_shares,
            interval_widths,
            mass_floor_multiplier=floor_multiplier,
            mass_cap_multiplier=cap_multiplier,
        )
        interval_masses = constraint_payload["interval_masses"]
    return {
        "normalized_hardness": normalized,
        "energy_potential": potential,
        "interval_widths": interval_widths,
        "raw_interval_masses": raw_interval_masses,
        "interval_masses": interval_masses,
        "canonical_interval_mass_shares": canonical_mass_shares,
        "reference_time_grid": reference_grid,
        "interval_midpoints": interval_midpoints,
        "hardness_tilt_gamma": tilt_gamma,
        "hardness_tilt_weights": tilt_weights,
        "tilted_normalized_hardness": tilted_normalized,
        "uniform_blend": blend,
        "gibbs_temperature": temperature,
        "mass_floor_multiplier": floor_multiplier,
        "mass_cap_multiplier": cap_multiplier,
        **constraint_payload,
    }


def _mean_step_trace_from_rows(calibration: Mapping[str, Any], row_key: str) -> Optional[np.ndarray]:
    rows = calibration.get("rows")
    if not isinstance(rows, Sequence) or not rows:
        return None
    macro_steps = int(calibration.get("macro_steps", 0))
    if macro_steps <= 0:
        return None
    buckets = [[] for _ in range(macro_steps)]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        step_idx = row.get("step_index")
        value = row.get(row_key)
        if step_idx is None or value is None:
            continue
        step_idx = int(step_idx)
        if 0 <= step_idx < macro_steps:
            buckets[step_idx].append(float(value))
    if any(not values for values in buckets):
        return None
    return np.asarray([float(np.mean(values)) for values in buckets], dtype=np.float64)


def _resolved_step_trace(
    calibration: Mapping[str, Any],
    *,
    by_step_key: str,
    row_key: str,
) -> Optional[np.ndarray]:
    values = calibration.get(by_step_key)
    if values is not None:
        return _clean_nonnegative_signal_values(values)
    from_rows = _mean_step_trace_from_rows(calibration, row_key)
    if from_rows is None:
        return None
    return _clean_nonnegative_signal_values(from_rows)


def _resolved_canonical_trace_values(
    calibration: Mapping[str, Any],
    *,
    signal_trace_key: str,
    r_star_multiplier: float,
) -> Dict[str, Any]:
    base_values = _clean_nonnegative_signal_values(calibration[str(signal_trace_key)])
    base_r_star = float(calibration.get("r_star", 1.0))
    multiplier = float(r_star_multiplier)
    if multiplier <= 0.0:
        raise ValueError(f"r_star_multiplier must be positive, got {r_star_multiplier}")
    if abs(multiplier - 1.0) <= 1e-12:
        return {
            "trace_values": base_values,
            "effective_r_star": base_r_star,
            "signal_recomputed": False,
        }
    if str(signal_trace_key) != CANONICAL_INFO_GROWTH_TRACE_KEY:
        raise ValueError("r_star_multiplier is only supported for the canonical info-growth signal trace.")
    disagreement = _resolved_step_trace(
        calibration,
        by_step_key="disagreement_by_step",
        row_key="disagreement",
    )
    residual = _resolved_step_trace(
        calibration,
        by_step_key="residual_norm_by_step",
        row_key="residual_norm",
    )
    if disagreement is None or residual is None:
        raise ValueError(
            "r_star_multiplier requires disagreement_by_step and residual_norm_by_step traces "
            "or calibration rows with disagreement/residual_norm fields."
        )
    effective_r_star = base_r_star * multiplier
    recomputed = compute_canonical_info_growth_hardness_numpy(
        residual,
        disagreement,
        r_star=float(effective_r_star),
    )
    return {
        "trace_values": recomputed,
        "effective_r_star": float(effective_r_star),
        "signal_recomputed": True,
    }


def schedule_shape_statistics(
    interval_masses: Sequence[float],
    time_grid: Sequence[float],
    *,
    top_k: int = 3,
) -> Dict[str, Optional[float]]:
    masses = _clean_nonnegative_signal_values(interval_masses)
    if masses.size == 0:
        return {
            "interval_mass_top1_share": None,
            "interval_mass_top3_share": None,
            "interval_mass_max_min_ratio": None,
            "runtime_grid_q25": None,
            "runtime_grid_q50": None,
            "runtime_grid_q75": None,
        }
    total_mass = max(float(np.sum(masses)), 1e-12)
    mass_shares = masses / total_mass
    sorted_shares = np.sort(mass_shares)[::-1]
    positive = mass_shares[mass_shares > 0.0]
    max_min_ratio = None
    if positive.size > 0:
        max_min_ratio = float(np.max(positive) / max(float(np.min(positive)), 1e-12))
    grid = np.asarray(time_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        runtime_q25 = runtime_q50 = runtime_q75 = None
    else:
        positions = np.linspace(0.0, 1.0, int(grid.size), dtype=np.float64)
        runtime_q25, runtime_q50, runtime_q75 = [
            float(x) for x in np.interp(np.asarray([0.25, 0.50, 0.75], dtype=np.float64), positions, grid)
        ]
    return {
        "interval_mass_top1_share": float(sorted_shares[0]),
        "interval_mass_top3_share": float(np.sum(sorted_shares[: min(int(top_k), int(sorted_shares.size))])),
        "interval_mass_max_min_ratio": max_min_ratio,
        "runtime_grid_q25": runtime_q25,
        "runtime_grid_q50": runtime_q50,
        "runtime_grid_q75": runtime_q75,
    }


def _schedule_details_from_density(
    density: Sequence[float],
    *,
    macro_steps: int,
    reference_time_grid: Optional[Sequence[float]],
    reference_time_alignment: str,
) -> Dict[str, Any]:
    density_arr = np.asarray(density, dtype=np.float64)
    reference_grid = _resolved_reference_time_grid(density_arr, reference_time_grid)
    runtime_grid = interpolated_equal_mass_grid(
        density_arr,
        int(macro_steps),
        reference_time_grid=reference_grid,
    )
    return {
        "time_grid": [float(x) for x in runtime_grid],
        "quantization_mode": "interpolated",
        "reference_macro_steps": int(reference_grid.size) - 1,
        "reference_time_grid": [float(x) for x in reference_grid.tolist()],
        "reference_time_alignment": str(reference_time_alignment),
    }


def canonical_tvd_schedule_details(
    calibration: Mapping[str, Any],
    *,
    macro_steps: int,
    delta: float,
    solver_order: float,
    signal_trace_key: str,
    uniform_blend: float = 0.0,
    gibbs_temperature: float = 1.0,
    reference_macro_factor: Optional[float] = None,
    r_star_multiplier: float = 1.0,
    mass_floor_multiplier: Optional[float] = None,
    mass_cap_multiplier: Optional[float] = None,
    grid_uniform_blend: float = 0.0,
    hardness_tilt_gamma: float = 0.0,
) -> Dict[str, Any]:
    grid_blend = float(grid_uniform_blend)
    if grid_blend < 0.0 or grid_blend > 1.0:
        raise ValueError(f"grid_uniform_blend must lie in [0, 1], got {grid_uniform_blend}")
    trace_payload = _resolved_canonical_trace_values(
        calibration,
        signal_trace_key=str(signal_trace_key),
        r_star_multiplier=float(r_star_multiplier),
    )
    mass_payload = canonical_interval_masses(
        trace_payload["trace_values"],
        delta=float(delta),
        solver_order=float(solver_order),
        reference_time_grid=calibration.get("reference_time_grid"),
        uniform_blend=float(uniform_blend),
        gibbs_temperature=float(gibbs_temperature),
        mass_floor_multiplier=mass_floor_multiplier,
        mass_cap_multiplier=mass_cap_multiplier,
        hardness_tilt_gamma=float(hardness_tilt_gamma),
    )
    details = _schedule_details_from_density(
        mass_payload["interval_masses"],
        macro_steps=int(macro_steps),
        reference_time_grid=mass_payload["reference_time_grid"],
        reference_time_alignment=str(calibration.get("reference_time_alignment", "left_endpoint")),
    )
    if grid_blend > 0.0:
        tvd_grid = np.asarray(details["time_grid"], dtype=np.float64)
        uniform_grid = np.linspace(0.0, 1.0, int(tvd_grid.size), dtype=np.float64)
        blended_grid = (1.0 - grid_blend) * tvd_grid + grid_blend * uniform_grid
        blended_grid[0] = 0.0
        blended_grid[-1] = 1.0
        details["time_grid"] = [float(x) for x in blended_grid.tolist()]
        details["quantization_mode"] = "interpolated_grid_blend"
    details["density_family"] = "canonical_tvd"
    details["normalized_hardness"] = [
        float(x) for x in np.asarray(mass_payload["normalized_hardness"], dtype=np.float64).tolist()
    ]
    details["energy_potential"] = [
        float(x) for x in np.asarray(mass_payload["energy_potential"], dtype=np.float64).tolist()
    ]
    details["interval_widths"] = [
        float(x) for x in np.asarray(mass_payload["interval_widths"], dtype=np.float64).tolist()
    ]
    details["raw_interval_masses"] = [
        float(x) for x in np.asarray(mass_payload["raw_interval_masses"], dtype=np.float64).tolist()
    ]
    details["interval_masses"] = [
        float(x) for x in np.asarray(mass_payload["interval_masses"], dtype=np.float64).tolist()
    ]
    details["canonical_interval_mass_shares"] = [
        float(x) for x in np.asarray(mass_payload["canonical_interval_mass_shares"], dtype=np.float64).tolist()
    ]
    details["interval_midpoints"] = [
        float(x) for x in np.asarray(mass_payload["interval_midpoints"], dtype=np.float64).tolist()
    ]
    details["hardness_tilt_gamma"] = float(mass_payload["hardness_tilt_gamma"])
    details["hardness_tilt_weights"] = [
        float(x) for x in np.asarray(mass_payload["hardness_tilt_weights"], dtype=np.float64).tolist()
    ]
    details["tilted_normalized_hardness"] = [
        float(x) for x in np.asarray(mass_payload["tilted_normalized_hardness"], dtype=np.float64).tolist()
    ]
    details["uniform_blend"] = float(uniform_blend)
    details["gibbs_temperature"] = float(gibbs_temperature)
    details["mass_floor_multiplier"] = None if mass_floor_multiplier is None else float(mass_floor_multiplier)
    details["mass_floor_hit_count"] = int(mass_payload["mass_floor_hit_count"])
    details["mass_floor_deficit_share"] = (
        None if mass_payload["mass_floor_deficit_share"] is None else float(mass_payload["mass_floor_deficit_share"])
    )
    details["mass_floor_min_share_after"] = (
        None if mass_payload["mass_floor_min_share_after"] is None else float(mass_payload["mass_floor_min_share_after"])
    )
    details["mass_cap_multiplier"] = None if mass_cap_multiplier is None else float(mass_cap_multiplier)
    details["mass_cap_hit_count"] = int(mass_payload["mass_cap_hit_count"])
    details["mass_cap_overflow_share"] = (
        None if mass_payload["mass_cap_overflow_share"] is None else float(mass_payload["mass_cap_overflow_share"])
    )
    details["mass_cap_max_share_after"] = (
        None if mass_payload["mass_cap_max_share_after"] is None else float(mass_payload["mass_cap_max_share_after"])
    )
    details["reference_macro_factor"] = (
        None if reference_macro_factor is None else float(reference_macro_factor)
    )
    details["r_star_multiplier"] = float(r_star_multiplier)
    details["grid_uniform_blend"] = float(grid_blend)
    details["signal_recomputed"] = bool(trace_payload["signal_recomputed"])
    details["r_star"] = float(trace_payload["effective_r_star"])
    details.update(
        schedule_shape_statistics(
            details["interval_masses"],
            details["time_grid"],
            top_k=3,
        )
    )
    return details


def signal_validation_spearman(calibration: Mapping[str, Any], signal_key: str) -> Optional[float]:
    corr_map = dict(calibration.get("signal_correlations_vs_oracle", {}))
    stats = corr_map.get(str(signal_key))
    if stats is None:
        return None
    spearman = stats.get("spearman")
    if spearman is None:
        return None
    return float(spearman)


def run_fixed_schedule_variant(
    *,
    model,
    ds,
    cfg,
    eval_horizon: int,
    eval_windows: int,
    grid_spec: Mapping[str, Any],
    chosen_t0s: Sequence[int],
    generation_seed_base: int,
    metrics_seed: int,
    score_main_only: bool,
) -> Dict[str, Any]:
    solver_name = str(grid_spec["solver_name"])
    time_grid = tuple(float(x) for x in grid_spec["time_grid"])
    backup = _apply_sample_overrides(model, cfg, solver=solver_name, time_grid=time_grid)
    try:
        t0 = time.time()
        result = eval_many_windows(
            ds,
            model,
            cfg,
            horizon=int(eval_horizon),
            nfe=int(grid_spec["nfe"]),
            n_windows=int(eval_windows),
            seed=int(metrics_seed),
            horizons_eval=[int(eval_horizon)],
            chosen_t0s=chosen_t0s,
            generation_seed_base=int(generation_seed_base),
            metrics_seed=int(metrics_seed),
            main_metrics_only=bool(score_main_only),
        )
        eval_seconds = float(time.time() - t0)
        diag = _collect_rollout_diagnostics(
            model,
            ds,
            cfg,
            horizon=int(eval_horizon),
            macro_steps=int(grid_spec["nfe"]),
            n_windows=int(eval_windows),
            seed=int(metrics_seed),
            solver=solver_name,
            chosen_t0s=chosen_t0s,
            generation_seed_base=int(generation_seed_base),
        )
    finally:
        _restore_sample_overrides(model, cfg, backup)

    row = {
        "grid_name": str(grid_spec["grid_name"]),
        "grid_kind": str(grid_spec["grid_kind"]),
        "selection_group": str(grid_spec["selection_group"]),
        "comparison_role": None if grid_spec.get("comparison_role") is None else str(grid_spec["comparison_role"]),
        "solver_name": str(solver_name),
        "nfe": int(grid_spec["nfe"]),
        "power": None if grid_spec.get("power") is None else float(grid_spec["power"]),
        "piecewise_early_frac": None
        if grid_spec.get("piecewise_early_frac") is None
        else float(grid_spec["piecewise_early_frac"]),
        "signal_validation_spearman": None
        if grid_spec.get("signal_validation_spearman") is None
        else float(grid_spec["signal_validation_spearman"]),
        "time_grid": [float(x) for x in time_grid],
        "target_total_field_evals": int(grid_spec["nfe"]),
        "solver_override": str(solver_name),
        "eval_seconds": eval_seconds,
        "mean_field_evals_per_step": float(diag["mean_field_evals_per_step"]),
        "mean_total_field_evals_per_rollout": float(diag["mean_total_field_evals_per_rollout"]),
        "trigger_rate": float(diag["trigger_rate"]),
        "diag": diag,
        "evaluation_protocol": {
            "chosen_t0s": [int(t0) for t0 in result["meta"]["chosen_t0s"]],
            "generation_seed_base": None
            if result["meta"]["generation_seed_base"] is None
            else int(result["meta"]["generation_seed_base"]),
            "metrics_seed": int(result["meta"]["metrics_seed"]),
            "main_metrics_only": bool(result["meta"].get("main_metrics_only", False)),
        },
        "score_main_only": bool(result["meta"].get("main_metrics_only", False)),
    }
    row.update(_metric_bundle(result))
    return row


__all__ = [
    "canonical_interval_masses",
    "canonical_normalized_hardness",
    "canonical_tvd_schedule_details",
    "interpolated_equal_mass_grid",
    "run_fixed_schedule_variant",
    "signal_validation_spearman",
]
