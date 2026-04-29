from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from adaptive_deterministic_refinement_followup import _collect_calibration
from adaptive_noise_sampler_followup import _choose_valid_windows
from otflow_canonical_only_support import (
    ALL_SOLVER_ORDER,
    CANONICAL_TVD_SCHEDULER_KEY,
    DEFAULT_FORECAST_DATASETS,
    DEFAULT_LOB_DATASETS,
    DEFAULT_SHARED_BACKBONE_ROOT,
    DEFAULT_SIGNAL_TRACE_KEY,
    FORECAST_FAMILY,
    LOCKED_TEST_PHASE,
    LOB_FAMILY,
    SOLVER_RUNTIME_NAMES,
    UNIFORM_SCHEDULER_KEY,
    VALIDATION_PHASE,
    collect_forecast_calibration,
    evaluate_forecast_schedule,
    load_forecast_checkpoint_splits,
    load_lob_checkpoint_splits,
    parse_csv,
    parse_float_csv,
    parse_forecast_datasets,
    parse_int_csv,
    parse_lob_datasets,
    resolve_reference_macro_steps,
    resolved_eval_horizon,
    resolved_eval_windows,
    save_json,
    safe_spearman,
    selection_metric_for_family,
    solver_experiment_scope,
    solver_macro_steps,
    solver_order_p,
    validate_execution_preflight,
)
from otflow_paper_registry import BASELINE_SCHEDULE_KEYS, build_schedule_grid, schedule_display_name, schedule_time_alignment
from otflow_paper_tables import augment_rows_with_relative_metrics
from otflow_paths import project_paper_dataset_root, project_root
from otflow_schedule_utils import canonical_tvd_schedule_details, run_fixed_schedule_variant, signal_validation_spearman
from otflow_signal_traces import (
    EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_TRACE_KEY,
    compute_no_rstar_info_growth_hardness_numpy,
)

RUNNER_SIGNATURE_VERSION = "canonical_only_study_v1"
DEFAULT_OUT_ROOT = project_root() / "TVD-result" / "experiments" / "results_otflow_canonical_only_study"
DEFAULT_CANONICAL_DELTA_VALUES = (0.02, 0.05, 0.10, 0.20)
DEFAULT_TARGET_NFE_VALUES = (10, 12, 16)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_BASELINE_SCHEDULERS = BASELINE_SCHEDULE_KEYS
DEFAULT_FIXED_SOLVER_NAMES = ("euler", "heun", "midpoint_rk2")
DEFAULT_CANONICAL_SIGNAL_VARIANT = "canonical"
DEFAULT_TVD_RECOVERY_SIGNAL_VARIANTS = ("canonical", "no_rstar")
DEFAULT_TVD_RECOVERY_WIND_DATASET = "wind_farms_wo_missing"
DEFAULT_TVD_RECOVERY_SAN_FRANCISCO_DATASET = "san_francisco_traffic"
DEFAULT_TVD_RECOVERY_STAGE_B_BLEND_VALUES = (0.15, 0.25)
DEFAULT_TVD_RECOVERY_STAGE_B_TEMPERATURE_VALUES = (1.25, 1.5)
DEFAULT_MATCHED_TVD_UNIFORM_FORECAST_DATASETS = (
    DEFAULT_TVD_RECOVERY_WIND_DATASET,
    DEFAULT_TVD_RECOVERY_SAN_FRANCISCO_DATASET,
)
DEFAULT_MATCHED_TVD_UNIFORM_SOLVER_NAMES = ("euler", "heun", "midpoint_rk2", "dpmpp2m")
DEFAULT_MATCHED_TVD_UNIFORM_TRACE_SAMPLES = 4
DEFAULT_MATCHED_TVD_UNIFORM_BLEND_VALUES = (0.25, 0.50)
DEFAULT_MATCHED_TVD_UNIFORM_TEMPERATURE_VALUES = (1.50, 2.00)
DEFAULT_MATCHED_TVD_UNIFORM_CAP_MULTIPLIERS = (1.25, 1.50, 2.00)
DEFAULT_MATCHED_TVD_UNIFORM_MASS_BANDS = ((0.50, 1.50), (0.75, 1.50))
DEFAULT_MATCHED_TVD_UNIFORM_GRID_BLEND_VALUES = (0.25, 0.50)
DEFAULT_MATCHED_TVD_UNIFORM_NO_RSTAR_BLEND_VALUES = (0.25, 0.50)
DEFAULT_MATCHED_TVD_UNIFORM_CRPS_TIE_TOLERANCE = 1e-3
DEFAULT_FORECAST_SELECTED_DELTAS: Dict[str, float] = {
    "wind_farms_wo_missing": 0.05,
    "san_francisco_traffic": 0.02,
    "london_smart_meters_wo_missing": 0.02,
    "electricity": 0.05,
    "solar_energy_10m": 0.10,
}

ROW_RECORD_FIELDS: Tuple[str, ...] = (
    "benchmark_family",
    "split_phase",
    "seed",
    "dataset",
    "checkpoint_id",
    "checkpoint_path",
    "backbone_name",
    "train_steps",
    "train_budget_label",
    "target_nfe",
    "runtime_nfe",
    "solver_key",
    "solver_name",
    "scheduler_key",
    "scheduler_variant_key",
    "scheduler_variant_name",
    "schedule_name",
    "row_signature",
    "signal_trace_key",
    "signal_validation_spearman",
    "canonical_delta",
    "r_star",
    "uniform_blend",
    "gibbs_temperature",
    "reference_macro_factor",
    "r_star_multiplier",
    "mass_floor_multiplier",
    "mass_floor_hit_count",
    "mass_floor_deficit_share",
    "mass_floor_min_share_after",
    "mass_cap_multiplier",
    "mass_cap_hit_count",
    "mass_cap_overflow_share",
    "mass_cap_max_share_after",
    "grid_uniform_blend",
    "hardness_tilt_gamma",
    "interval_mass_top1_share",
    "interval_mass_top3_share",
    "interval_mass_max_min_ratio",
    "runtime_grid_q25",
    "runtime_grid_q50",
    "runtime_grid_q75",
    "reference_macro_steps",
    "reference_time_alignment",
    "calibration_trace_samples",
    "paper_duplicate_count",
    "experiment_scope",
    "selection_metric",
    "selection_metric_value",
    "crps",
    "mse",
    "mase",
    "score_main",
    "conditional_w1",
    "tstr_macro_f1",
    "relative_crps_gain_vs_uniform",
    "relative_mase_gain_vs_uniform",
    "relative_score_gain_vs_uniform",
    "realized_nfe",
    "latency_ms_per_sample",
    "num_eval_samples",
    "eval_examples",
    "eval_windows",
    "row_status",
)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(cast):
        return None
    return cast


def _scheduler_variant_key_from_row(row: Mapping[str, Any]) -> str:
    value = row.get("scheduler_variant_key")
    if value is None:
        value = row.get("scheduler_key")
    return str(value)


def _row_signature(
    *,
    scheduler_key: str,
    signal_trace_key: Optional[str],
    canonical_delta: Optional[float],
    reference_macro_steps: int,
    calibration_trace_samples: int = 1,
    mass_floor_multiplier: Optional[float] = None,
    mass_cap_multiplier: Optional[float] = None,
    grid_uniform_blend: Optional[float] = None,
    hardness_tilt_gamma: Optional[float] = None,
    scheduler_variant_tag: Optional[str] = None,
) -> str:
    signal_token = "none" if signal_trace_key is None else str(signal_trace_key).strip()
    delta_token = "none" if canonical_delta is None else f"{float(canonical_delta):.8f}"
    signature = (
        f"{RUNNER_SIGNATURE_VERSION}:"
        f"{str(scheduler_key).strip()}:"
        f"{signal_token}:"
        f"{delta_token}:"
        f"{int(reference_macro_steps)}"
    )
    if scheduler_variant_tag:
        signature += f":{str(scheduler_variant_tag).strip()}"
    if _is_canonical_tvd_scheduler(str(scheduler_key)) and int(calibration_trace_samples) != 1:
        signature += f":calib{int(calibration_trace_samples)}"
    floor_value = _optional_float(mass_floor_multiplier)
    if _is_canonical_tvd_scheduler(str(scheduler_key)) and floor_value is not None and float(floor_value) > 0.0:
        signature += f":floor{_format_variant_token(float(floor_value))}"
    cap_value = _optional_float(mass_cap_multiplier)
    if _is_canonical_tvd_scheduler(str(scheduler_key)) and cap_value is not None and float(cap_value) > 0.0:
        signature += f":cap{_format_variant_token(float(cap_value))}"
    grid_blend_value = _optional_float(grid_uniform_blend)
    if (
        _is_canonical_tvd_scheduler(str(scheduler_key))
        and grid_blend_value is not None
        and float(grid_blend_value) > 0.0
    ):
        signature += f":gridblend{_format_variant_token(float(grid_blend_value))}"
    tilt_gamma_value = _optional_float(hardness_tilt_gamma)
    if (
        _is_canonical_tvd_scheduler(str(scheduler_key))
        and tilt_gamma_value is not None
        and abs(float(tilt_gamma_value)) > 1e-12
    ):
        signature += f":tiltgamma{_format_variant_token(float(tilt_gamma_value))}"
    return signature


def _scheduler_variant_tag_from_case(case: Mapping[str, Any]) -> Optional[str]:
    scheduler_key = str(case["scheduler_key"])
    variant_key = str(case.get("scheduler_variant_key") or scheduler_key)
    return None if variant_key == scheduler_key else variant_key


def _scheduler_variant_key_from_case(case: Mapping[str, Any]) -> str:
    return str(case.get("scheduler_variant_key") or case["scheduler_key"])


def _scheduler_variant_name_from_case(case: Mapping[str, Any]) -> str:
    if case.get("scheduler_variant_name") is not None:
        return str(case["scheduler_variant_name"])
    scheduler_key = str(case["scheduler_key"])
    if _is_canonical_tvd_scheduler(scheduler_key):
        return "Canonical TVD"
    return str(schedule_display_name(scheduler_key))


def _resolved_case_float(
    cli_args: argparse.Namespace,
    case: Mapping[str, Any],
    *,
    field_name: str,
    default_value: float,
) -> float:
    value = case.get(field_name)
    if value is None:
        value = getattr(cli_args, field_name, default_value)
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ValueError(f"{field_name} must be finite, got {value}")
    return resolved


def _resolved_case_positive_int(
    cli_args: argparse.Namespace,
    case: Mapping[str, Any],
    *,
    field_name: str,
    default_value: int,
) -> int:
    value = case.get(field_name)
    if value is None:
        value = getattr(cli_args, field_name, default_value)
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}")
    return resolved


def _resolve_scheduler_case(
    cli_args: argparse.Namespace,
    case: Mapping[str, Any],
    *,
    runtime_nfe: int,
) -> Dict[str, Any]:
    scheduler_key = str(case["scheduler_key"])
    reference_macro_factor = _resolved_case_float(
        cli_args,
        case,
        field_name="reference_macro_factor",
        default_value=4.0,
    )
    reference_macro_steps = resolve_reference_macro_steps(
        int(cli_args.reference_macro_steps),
        int(runtime_nfe),
        reference_macro_factor=float(reference_macro_factor),
    )
    resolved = dict(case)
    resolved["scheduler_key"] = scheduler_key
    resolved["scheduler_variant_key"] = _scheduler_variant_key_from_case(case)
    resolved["scheduler_variant_name"] = _scheduler_variant_name_from_case(case)
    resolved["signal_trace_key"] = str(case.get("signal_trace_key") or _scheduler_signal_key(scheduler_key))
    resolved["uniform_blend"] = _resolved_case_float(
        cli_args,
        case,
        field_name="uniform_blend",
        default_value=0.0,
    )
    resolved["gibbs_temperature"] = _resolved_case_float(
        cli_args,
        case,
        field_name="gibbs_temperature",
        default_value=1.0,
    )
    resolved["reference_macro_factor"] = float(reference_macro_factor)
    resolved["r_star_multiplier"] = _resolved_case_float(
        cli_args,
        case,
        field_name="r_star_multiplier",
        default_value=1.0,
    )
    resolved["mass_floor_multiplier"] = (
        _resolved_case_float(
            cli_args,
            case,
            field_name="mass_floor_multiplier",
            default_value=0.0,
        )
        if _is_canonical_tvd_scheduler(scheduler_key)
        else 0.0
    )
    if float(resolved["mass_floor_multiplier"]) < 0.0:
        raise ValueError(f"mass_floor_multiplier must be non-negative, got {resolved['mass_floor_multiplier']}")
    resolved["mass_cap_multiplier"] = (
        _resolved_case_float(
            cli_args,
            case,
            field_name="mass_cap_multiplier",
            default_value=0.0,
        )
        if _is_canonical_tvd_scheduler(scheduler_key)
        else 0.0
    )
    if float(resolved["mass_cap_multiplier"]) < 0.0:
        raise ValueError(f"mass_cap_multiplier must be non-negative, got {resolved['mass_cap_multiplier']}")
    if (
        float(resolved["mass_floor_multiplier"]) > 0.0
        and float(resolved["mass_cap_multiplier"]) > 0.0
        and float(resolved["mass_floor_multiplier"]) > float(resolved["mass_cap_multiplier"])
    ):
        raise ValueError(
            "mass_floor_multiplier must be <= mass_cap_multiplier when both are enabled, "
            f"got floor={resolved['mass_floor_multiplier']} cap={resolved['mass_cap_multiplier']}"
        )
    resolved["grid_uniform_blend"] = (
        _resolved_case_float(
            cli_args,
            case,
            field_name="grid_uniform_blend",
            default_value=0.0,
        )
        if _is_canonical_tvd_scheduler(scheduler_key)
        else 0.0
    )
    if float(resolved["grid_uniform_blend"]) < 0.0 or float(resolved["grid_uniform_blend"]) > 1.0:
        raise ValueError(f"grid_uniform_blend must lie in [0, 1], got {resolved['grid_uniform_blend']}")
    resolved["hardness_tilt_gamma"] = (
        _resolved_case_float(
            cli_args,
            case,
            field_name="hardness_tilt_gamma",
            default_value=0.0,
        )
        if _is_canonical_tvd_scheduler(scheduler_key)
        else 0.0
    )
    resolved["calibration_trace_samples"] = (
        _resolved_case_positive_int(
            cli_args,
            case,
            field_name="calibration_trace_samples",
            default_value=1,
        )
        if _is_canonical_tvd_scheduler(scheduler_key)
        else 1
    )
    resolved["reference_macro_steps"] = int(reference_macro_steps)
    resolved["scheduler_variant_tag"] = _scheduler_variant_tag_from_case(resolved)
    return resolved


def _is_canonical_tvd_scheduler(scheduler_key: str) -> bool:
    return str(scheduler_key).strip().lower() == CANONICAL_TVD_SCHEDULER_KEY


def _scheduler_signal_key(scheduler_key: str) -> Optional[str]:
    return DEFAULT_SIGNAL_TRACE_KEY if _is_canonical_tvd_scheduler(str(scheduler_key)) else None


def _scheduler_grid_kind(scheduler_key: str) -> str:
    key = str(scheduler_key).strip().lower()
    if key == UNIFORM_SCHEDULER_KEY:
        return "uniform"
    if key == CANONICAL_TVD_SCHEDULER_KEY:
        return "canonical_tvd"
    return "fixed_baseline"


def _scheduler_comparison_role(scheduler_key: str) -> str:
    key = str(scheduler_key).strip().lower()
    if key == UNIFORM_SCHEDULER_KEY:
        return "baseline_uniform"
    if key == CANONICAL_TVD_SCHEDULER_KEY:
        return "paper_method"
    return "baseline"


def _parse_baseline_scheduler_names(text: str) -> List[str]:
    names = [name.strip().lower() for name in parse_csv(str(text))]
    if not names:
        raise ValueError("At least one baseline scheduler must be selected.")
    unsupported = [name for name in names if build_schedule_grid(name, 4) is None]
    if unsupported:
        raise ValueError(f"Unsupported fixed baseline scheduler(s): {unsupported}")
    return names


def _ordered_scheduler_keys_from_rows(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    present = {str(row.get("scheduler_key")) for row in rows if row.get("scheduler_key") is not None}
    preferred = list(DEFAULT_BASELINE_SCHEDULERS) + [CANONICAL_TVD_SCHEDULER_KEY]
    ordered = [key for key in preferred if key in present]
    ordered.extend(sorted(key for key in present if key not in set(ordered)))
    return ordered


def _row_key(row: Mapping[str, Any]) -> Tuple[str, str, int, str, int, str, str, str]:
    return (
        str(row["benchmark_family"]),
        str(row["split_phase"]),
        int(row["seed"]),
        str(row["dataset"]),
        int(row["target_nfe"]),
        str(row["solver_key"]),
        str(row["scheduler_key"]),
        str(row["row_signature"]),
    )


def _write_row_csv(csv_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ROW_RECORD_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in ROW_RECORD_FIELDS})


def _load_rows(jsonl_path: Path) -> Dict[Tuple[str, str, int, str, int, str, str, str], Dict[str, Any]]:
    rows: Dict[Tuple[str, str, int, str, int, str, str, str], Dict[str, Any]] = {}
    if not jsonl_path.exists():
        return rows
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            payload = json.loads(line)
            rows[_row_key(payload)] = payload
    return rows


def _init_row_recorder(out_root: Path, cli_args: argparse.Namespace) -> Dict[str, Any]:
    jsonl_path = out_root / str(getattr(cli_args, "row_jsonl_name", "rows.jsonl"))
    csv_path = out_root / str(getattr(cli_args, "row_csv_name", "rows.csv"))
    rows_by_key = _load_rows(jsonl_path) if bool(getattr(cli_args, "resume", True)) else {}
    row_order = list(rows_by_key.keys())
    _write_row_csv(csv_path, [rows_by_key[item_key] for item_key in row_order])
    return {
        "jsonl_path": jsonl_path,
        "csv_path": csv_path,
        "rows_by_key": rows_by_key,
        "row_order": row_order,
    }


def _append_row_record(row_recorder: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    item_key = _row_key(row)
    rows_by_key = row_recorder["rows_by_key"]
    row_order = row_recorder["row_order"]
    rows_by_key[item_key] = dict(row)
    if item_key not in row_order:
        row_order.append(item_key)
    jsonl_path = Path(row_recorder["jsonl_path"])
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for ordered_key in row_order:
            fh.write(json.dumps(rows_by_key[ordered_key], sort_keys=True) + "\n")
    _write_row_csv(Path(row_recorder["csv_path"]), [rows_by_key[ordered_key] for ordered_key in row_order])


def _existing_complete_row(
    row_recorder: Mapping[str, Any],
    *,
    benchmark_family: str,
    split_phase: str,
    seed: int,
    dataset: str,
    target_nfe: int,
    solver_key: str,
    scheduler_key: str,
    row_signature: str,
) -> Optional[Dict[str, Any]]:
    item_key = (
        str(benchmark_family),
        str(split_phase),
        int(seed),
        str(dataset),
        int(target_nfe),
        str(solver_key),
        str(scheduler_key),
        str(row_signature),
    )
    row = row_recorder["rows_by_key"].get(item_key)
    if row is None or str(row.get("row_status", "")) != "complete":
        return None
    return dict(row)


def _pending_scheduler_cases(
    row_recorder: Mapping[str, Any],
    *,
    benchmark_family: str,
    split_phase: str,
    seed: int,
    dataset: str,
    target_nfe: int,
    solver_key: str,
    scheduler_cases: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    existing_rows: List[Dict[str, Any]] = []
    pending_cases: List[Dict[str, Any]] = []
    for case in scheduler_cases:
        scheduler_key = str(case["scheduler_key"])
        canonical_delta = case.get("canonical_delta")
        reference_macro_steps = int(case["reference_macro_steps"])
        signal_trace_key = case.get("signal_trace_key", _scheduler_signal_key(scheduler_key))
        normalized_delta = None if canonical_delta is None else float(canonical_delta)
        calibration_trace_samples = int(case.get("calibration_trace_samples", 1))
        mass_floor_multiplier = _optional_float(case.get("mass_floor_multiplier"))
        mass_cap_multiplier = _optional_float(case.get("mass_cap_multiplier"))
        grid_uniform_blend = _optional_float(case.get("grid_uniform_blend"))
        hardness_tilt_gamma = _optional_float(case.get("hardness_tilt_gamma"))
        row_signature = _row_signature(
            scheduler_key=scheduler_key,
            signal_trace_key=None if signal_trace_key is None else str(signal_trace_key),
            canonical_delta=normalized_delta,
            reference_macro_steps=int(reference_macro_steps),
            calibration_trace_samples=int(calibration_trace_samples),
            mass_floor_multiplier=mass_floor_multiplier,
            mass_cap_multiplier=mass_cap_multiplier,
            grid_uniform_blend=grid_uniform_blend,
            hardness_tilt_gamma=hardness_tilt_gamma,
            scheduler_variant_tag=case.get("scheduler_variant_tag"),
        )
        existing = _existing_complete_row(
            row_recorder,
            benchmark_family=str(benchmark_family),
            split_phase=str(split_phase),
            seed=int(seed),
            dataset=str(dataset),
            target_nfe=int(target_nfe),
            solver_key=str(solver_key),
            scheduler_key=scheduler_key,
            row_signature=row_signature,
        )
        if existing is not None:
            existing_rows.append(existing)
            continue
        pending_case = dict(case)
        pending_case["scheduler_key"] = scheduler_key
        pending_case["canonical_delta"] = normalized_delta
        pending_case["calibration_trace_samples"] = int(calibration_trace_samples)
        pending_case["mass_floor_multiplier"] = 0.0 if mass_floor_multiplier is None else float(mass_floor_multiplier)
        pending_case["mass_cap_multiplier"] = 0.0 if mass_cap_multiplier is None else float(mass_cap_multiplier)
        pending_case["grid_uniform_blend"] = 0.0 if grid_uniform_blend is None else float(grid_uniform_blend)
        pending_case["hardness_tilt_gamma"] = 0.0 if hardness_tilt_gamma is None else float(hardness_tilt_gamma)
        pending_case["row_signature"] = row_signature
        pending_cases.append(pending_case)
    return existing_rows, pending_cases


def _build_row(
    *,
    benchmark_family: str,
    split_phase: str,
    seed: int,
    dataset: str,
    checkpoint: Mapping[str, Any],
    target_nfe: int,
    runtime_nfe: int,
    solver_key: str,
    scheduler_key: str,
    scheduler_variant_key: str,
    scheduler_variant_name: str,
    signal_trace_key: Optional[str],
    signal_validation: Optional[float],
    canonical_delta: Optional[float],
    reference_macro_steps: int,
    details: Mapping[str, Any],
    metrics: Mapping[str, Any],
    calibration_trace_samples: int = 1,
    scheduler_variant_tag: Optional[str] = None,
) -> Dict[str, Any]:
    selection_metric = selection_metric_for_family(str(benchmark_family))
    selection_metric_value = _optional_float(metrics.get(selection_metric))
    schedule_name = str(scheduler_variant_name)
    latency_ms = _optional_float(metrics.get("latency_ms_per_sample"))
    if latency_ms is None:
        latency_ms = _optional_float(metrics.get("efficiency_ms_per_sample"))
    return {
        "benchmark_family": str(benchmark_family),
        "split_phase": str(split_phase),
        "seed": int(seed),
        "dataset": str(dataset),
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
        "checkpoint_path": str(checkpoint["checkpoint_path"]),
        "backbone_name": str(checkpoint["backbone_name"]),
        "train_steps": int(checkpoint["train_steps"]),
        "train_budget_label": str(checkpoint["train_budget_label"]),
        "target_nfe": int(target_nfe),
        "runtime_nfe": int(runtime_nfe),
        "solver_key": str(solver_key),
        "solver_name": str(SOLVER_RUNTIME_NAMES[str(solver_key)]),
        "scheduler_key": str(scheduler_key),
        "scheduler_variant_key": str(scheduler_variant_key),
        "scheduler_variant_name": str(scheduler_variant_name),
        "schedule_name": str(schedule_name),
        "row_signature": _row_signature(
            scheduler_key=str(scheduler_key),
            signal_trace_key=signal_trace_key,
            canonical_delta=canonical_delta,
            reference_macro_steps=int(reference_macro_steps),
            calibration_trace_samples=int(calibration_trace_samples),
            mass_floor_multiplier=details.get("mass_floor_multiplier"),
            mass_cap_multiplier=details.get("mass_cap_multiplier"),
            grid_uniform_blend=details.get("grid_uniform_blend"),
            hardness_tilt_gamma=details.get("hardness_tilt_gamma"),
            scheduler_variant_tag=scheduler_variant_tag,
        ),
        "signal_trace_key": None if signal_trace_key is None else str(signal_trace_key),
        "signal_validation_spearman": _optional_float(signal_validation),
        "canonical_delta": _optional_float(canonical_delta),
        "r_star": _optional_float(details.get("r_star")),
        "uniform_blend": _optional_float(details.get("uniform_blend")),
        "gibbs_temperature": _optional_float(details.get("gibbs_temperature")),
        "reference_macro_factor": _optional_float(details.get("reference_macro_factor")),
        "r_star_multiplier": _optional_float(details.get("r_star_multiplier")),
        "mass_floor_multiplier": _optional_float(details.get("mass_floor_multiplier")),
        "mass_floor_hit_count": (
            None if details.get("mass_floor_hit_count") is None else int(details.get("mass_floor_hit_count"))
        ),
        "mass_floor_deficit_share": _optional_float(details.get("mass_floor_deficit_share")),
        "mass_floor_min_share_after": _optional_float(details.get("mass_floor_min_share_after")),
        "mass_cap_multiplier": _optional_float(details.get("mass_cap_multiplier")),
        "mass_cap_hit_count": (
            None if details.get("mass_cap_hit_count") is None else int(details.get("mass_cap_hit_count"))
        ),
        "mass_cap_overflow_share": _optional_float(details.get("mass_cap_overflow_share")),
        "mass_cap_max_share_after": _optional_float(details.get("mass_cap_max_share_after")),
        "grid_uniform_blend": _optional_float(details.get("grid_uniform_blend")),
        "hardness_tilt_gamma": _optional_float(details.get("hardness_tilt_gamma")),
        "interval_mass_top1_share": _optional_float(details.get("interval_mass_top1_share")),
        "interval_mass_top3_share": _optional_float(details.get("interval_mass_top3_share")),
        "interval_mass_max_min_ratio": _optional_float(details.get("interval_mass_max_min_ratio")),
        "runtime_grid_q25": _optional_float(details.get("runtime_grid_q25")),
        "runtime_grid_q50": _optional_float(details.get("runtime_grid_q50")),
        "runtime_grid_q75": _optional_float(details.get("runtime_grid_q75")),
        "reference_macro_steps": int(reference_macro_steps),
        "reference_time_alignment": str(details.get("reference_time_alignment", "left_endpoint")),
        "calibration_trace_samples": int(calibration_trace_samples)
        if _is_canonical_tvd_scheduler(str(scheduler_key))
        else None,
        "paper_duplicate_count": int(details.get("paper_duplicate_count", 0)),
        "experiment_scope": str(solver_experiment_scope(str(solver_key))),
        "selection_metric": str(selection_metric),
        "selection_metric_value": _optional_float(selection_metric_value),
        "crps": _optional_float(metrics.get("crps")),
        "mse": _optional_float(metrics.get("mse")),
        "mase": _optional_float(metrics.get("mase")),
        "score_main": _optional_float(metrics.get("score_main")),
        "conditional_w1": _optional_float(metrics.get("conditional_w1")),
        "tstr_macro_f1": _optional_float(metrics.get("tstr_macro_f1")),
        "relative_crps_gain_vs_uniform": _optional_float(metrics.get("relative_crps_gain_vs_uniform")),
        "relative_mase_gain_vs_uniform": _optional_float(metrics.get("relative_mase_gain_vs_uniform")),
        "relative_score_gain_vs_uniform": _optional_float(metrics.get("relative_score_gain_vs_uniform")),
        "realized_nfe": int(metrics["realized_nfe"]),
        "latency_ms_per_sample": _optional_float(latency_ms),
        "num_eval_samples": None if metrics.get("num_eval_samples") is None else int(metrics["num_eval_samples"]),
        "eval_examples": None if metrics.get("eval_examples") is None else int(metrics["eval_examples"]),
        "eval_windows": None if metrics.get("eval_windows") is None else int(metrics["eval_windows"]),
        "row_status": "complete",
    }


def _candidate_rows_by_phase(
    rows: Iterable[Mapping[str, Any]],
    split_phase: str,
    *,
    solver_names: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    allowed_solvers = set(ALL_SOLVER_ORDER if solver_names is None else solver_names)
    return [
        dict(row)
        for row in rows
        if str(row.get("split_phase")) == str(split_phase) and str(row.get("row_status", "complete")) == "complete"
        and str(row.get("solver_key")) in allowed_solvers
    ]


def _mean(values: Sequence[float]) -> Optional[float]:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(arr.mean())


def _std(values: Sequence[float]) -> Optional[float]:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(arr.std(ddof=0))


def _safe_relative_gain(value: Any, baseline_value: Any) -> Optional[float]:
    scored = _optional_float(value)
    baseline = _optional_float(baseline_value)
    if scored is None or baseline is None or abs(float(baseline)) <= 1e-12:
        return None
    return float((float(baseline) - float(scored)) / float(baseline))


def _augment_rows_with_relative_metrics_extended(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    baseline_rows: Dict[Tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("scheduler_key")) == UNIFORM_SCHEDULER_KEY:
            baseline_rows[_ranking_match_key(row)] = row
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        baseline = baseline_rows.get(_ranking_match_key(row))
        payload["relative_crps_gain_vs_uniform"] = _optional_float(row.get("relative_crps_gain_vs_uniform"))
        payload["relative_mase_gain_vs_uniform"] = _optional_float(row.get("relative_mase_gain_vs_uniform"))
        payload["relative_score_gain_vs_uniform"] = _optional_float(row.get("relative_score_gain_vs_uniform"))
        family = str(row.get("benchmark_family"))
        if baseline is not None and family == FORECAST_FAMILY:
            payload["relative_crps_gain_vs_uniform"] = _safe_relative_gain(
                row.get("crps"),
                baseline.get("crps"),
            )
            payload["relative_mase_gain_vs_uniform"] = _safe_relative_gain(
                row.get("mase"),
                baseline.get("mase"),
            )
        if baseline is not None and family == LOB_FAMILY:
            payload["relative_score_gain_vs_uniform"] = _safe_relative_gain(
                row.get("score_main"),
                baseline.get("score_main"),
            )
        enriched.append(payload)
    return enriched


def _recomputed_signal_validation_spearman(
    calibration: Mapping[str, Any],
    *,
    signal_trace_key: str,
    r_star_multiplier: float,
) -> Optional[float]:
    if abs(float(r_star_multiplier) - 1.0) <= 1e-12:
        return signal_validation_spearman(calibration, signal_trace_key)
    rows = calibration.get("rows")
    if not isinstance(rows, Sequence) or not rows:
        return signal_validation_spearman(calibration, signal_trace_key)
    base_r_star = _optional_float(calibration.get("base_r_star"))
    if base_r_star is None:
        base_r_star = _optional_float(calibration.get("r_star"))
    if base_r_star is None:
        return signal_validation_spearman(calibration, signal_trace_key)
    effective_r_star = float(base_r_star) * float(r_star_multiplier)
    if effective_r_star <= 0.0:
        raise ValueError(f"r_star_multiplier must keep r_star positive, got {r_star_multiplier}")
    signal_values: List[float] = []
    oracle_values: List[float] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if int(row.get("step_index", 0)) <= 0:
            continue
        disagreement = _optional_float(row.get("disagreement"))
        residual = _optional_float(row.get("residual_norm"))
        oracle = _optional_float(row.get("oracle_local_error"))
        if disagreement is None or residual is None or oracle is None:
            continue
        signal_values.append(float(disagreement) * float(np.log1p(max(float(residual), 0.0) / effective_r_star)))
        oracle_values.append(float(oracle))
    if not signal_values or len(signal_values) != len(oracle_values):
        return signal_validation_spearman(calibration, signal_trace_key)
    return _optional_float(safe_spearman(signal_values, oracle_values))


def _group_pending_cases_by_reference_macro_steps(
    pending_cases: Sequence[Mapping[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for case in pending_cases:
        grouped.setdefault(int(case["reference_macro_steps"]), []).append(dict(case))
    return grouped


def _mean_rank_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_delta: Dict[float, Dict[str, List[float]]] = {}
    grouped: Dict[Tuple[str, str, int], List[Mapping[str, Any]]] = {}
    if not rows:
        return {"selected_delta": None, "candidates": []}
    metric_key = selection_metric_for_family(str(rows[0]["benchmark_family"]))
    for row in rows:
        key = (str(row["dataset"]), str(row["solver_key"]), int(row["target_nfe"]))
        grouped.setdefault(key, []).append(row)
    for cell_rows in grouped.values():
        scored = []
        for row in cell_rows:
            metric_value = _optional_float(row.get(metric_key))
            if metric_value is None:
                continue
            scored.append((float(row["canonical_delta"]), float(metric_value)))
        if not scored:
            continue
        ordered = sorted(scored, key=lambda item: (item[1], item[0]))
        ranks: Dict[float, float] = {}
        start = 0
        while start < len(ordered):
            end = start
            while end + 1 < len(ordered) and abs(float(ordered[end + 1][1]) - float(ordered[start][1])) <= 1e-12:
                end += 1
            avg_rank = 0.5 * (float(start + 1) + float(end + 1))
            for idx in range(start, end + 1):
                ranks[float(ordered[idx][0])] = float(avg_rank)
            start = end + 1
        for delta_value, metric_value in scored:
            payload = by_delta.setdefault(float(delta_value), {"ranks": [], "metrics": []})
            payload["ranks"].append(float(ranks[float(delta_value)]))
            payload["metrics"].append(float(metric_value))
    candidates = []
    for delta_value, payload in sorted(by_delta.items()):
        mean_rank = _mean(payload["ranks"])
        mean_metric = _mean(payload["metrics"])
        if mean_rank is None or mean_metric is None:
            continue
        candidates.append(
            {
                "delta": float(delta_value),
                "matched_cells": int(len(payload["ranks"])),
                "mean_tied_rank": float(mean_rank),
                "mean_metric": float(mean_metric),
            }
        )
    selected = None
    if candidates:
        selected = min(
            candidates,
            key=lambda item: (float(item["mean_tied_rank"]), float(item["mean_metric"]), float(item["delta"])),
        )
    return {
        "selected_delta": None if selected is None else float(selected["delta"]),
        "metric_key": str(metric_key),
        "candidates": candidates,
    }


def _select_dataset_deltas(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    datasets: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        datasets.setdefault(str(row["dataset"]), []).append(row)
    payload: Dict[str, Any] = {"datasets": {}, "selected_deltas": {}}
    for dataset, dataset_rows in sorted(datasets.items()):
        summary = _mean_rank_summary(dataset_rows)
        payload["datasets"][str(dataset)] = {
            "benchmark_family": str(dataset_rows[0]["benchmark_family"]),
            **summary,
        }
        payload["selected_deltas"][str(dataset)] = summary["selected_delta"]
    return payload


def _ranking_match_key_from_values(
    *,
    benchmark_family: Any,
    split_phase: Any,
    seed: Any,
    dataset: Any,
    backbone_name: Any,
    train_steps: Any,
    train_budget_label: Any,
    checkpoint_id: Any,
    target_nfe: Any,
    solver_key: Any,
    experiment_scope: Any,
) -> Tuple[Any, ...]:
    return (
        benchmark_family,
        split_phase,
        seed,
        dataset,
        backbone_name,
        train_steps,
        train_budget_label,
        checkpoint_id,
        target_nfe,
        solver_key,
        experiment_scope,
    )


def _ranking_match_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return _ranking_match_key_from_values(
        benchmark_family=row.get("benchmark_family"),
        split_phase=row.get("split_phase"),
        seed=row.get("seed"),
        dataset=row.get("dataset"),
        backbone_name=row.get("backbone_name"),
        train_steps=row.get("train_steps"),
        train_budget_label=row.get("train_budget_label"),
        checkpoint_id=row.get("checkpoint_id"),
        target_nfe=row.get("target_nfe"),
        solver_key=row.get("solver_key"),
        experiment_scope=row.get("experiment_scope"),
    )


def _metric_rank_specs(family: str) -> Tuple[Tuple[str, bool], ...]:
    if str(family) == FORECAST_FAMILY:
        return (
            ("relative_crps_gain_vs_uniform", True),
            ("mase", False),
        )
    if str(family) == LOB_FAMILY:
        return (
            ("relative_score_gain_vs_uniform", True),
            ("conditional_w1", False),
            ("tstr_macro_f1", True),
        )
    raise ValueError(f"Unsupported benchmark family for ranking: {family}")


def _rank_summary_by_scheduler(rows: Sequence[Mapping[str, Any]], family: str) -> Dict[str, Dict[str, Optional[float]]]:
    summary: Dict[str, Dict[str, List[float]]] = {}
    for metric_key, higher_is_better in _metric_rank_specs(str(family)):
        grouped: Dict[Tuple[Any, ...], List[Tuple[str, float]]] = {}
        for row in rows:
            metric_value = _optional_float(row.get(metric_key))
            if metric_value is None:
                continue
            grouped.setdefault(_ranking_match_key(row), []).append((str(row["scheduler_key"]), float(metric_value)))
        for cell_rows in grouped.values():
            ordered = sorted(
                cell_rows,
                key=lambda item: (-float(item[1]) if higher_is_better else float(item[1]), str(item[0])),
            )
            start = 0
            while start < len(ordered):
                end = start
                while end + 1 < len(ordered) and abs(float(ordered[end + 1][1]) - float(ordered[start][1])) <= 1e-12:
                    end += 1
                avg_rank = 0.5 * (float(start + 1) + float(end + 1))
                for idx in range(start, end + 1):
                    scheduler_key = str(ordered[idx][0])
                    summary.setdefault(scheduler_key, {}).setdefault(
                        f"average_tied_rank_{metric_key}",
                        [],
                    ).append(float(avg_rank))
                start = end + 1
    return {
        scheduler_key: {metric_key: _mean(values) for metric_key, values in metrics.items()}
        for scheduler_key, metrics in summary.items()
    }


def _aggregate_seed_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    augmented = _augment_rows_with_relative_metrics_extended(rows)
    scheduler_keys = _ordered_scheduler_keys_from_rows(augmented)
    seed_summary: Dict[str, Any] = {"families": {}}
    for family in (FORECAST_FAMILY, LOB_FAMILY):
        family_rows = [row for row in augmented if str(row.get("benchmark_family")) == str(family)]
        rank_summary = _rank_summary_by_scheduler(family_rows, str(family)) if family_rows else {}
        family_block: Dict[str, Any] = {}
        for scheduler_key in scheduler_keys:
            scheduler_rows = [row for row in family_rows if str(row.get("scheduler_key")) == str(scheduler_key)]
            metrics: Dict[str, Any] = {"row_count": int(len(scheduler_rows))}
            if family == FORECAST_FAMILY:
                metrics["relative_crps_gain_vs_uniform"] = _mean(
                    [float(row["relative_crps_gain_vs_uniform"]) for row in scheduler_rows if row.get("relative_crps_gain_vs_uniform") is not None]
                )
                metrics["relative_mase_gain_vs_uniform"] = _mean(
                    [float(row["relative_mase_gain_vs_uniform"]) for row in scheduler_rows if row.get("relative_mase_gain_vs_uniform") is not None]
                )
                metrics["mase"] = _mean([float(row["mase"]) for row in scheduler_rows if row.get("mase") is not None])
            else:
                metrics["relative_score_gain_vs_uniform"] = _mean(
                    [float(row["relative_score_gain_vs_uniform"]) for row in scheduler_rows if row.get("relative_score_gain_vs_uniform") is not None]
                )
                metrics["conditional_w1"] = _mean(
                    [float(row["conditional_w1"]) for row in scheduler_rows if row.get("conditional_w1") is not None]
                )
                metrics["tstr_macro_f1"] = _mean(
                    [float(row["tstr_macro_f1"]) for row in scheduler_rows if row.get("tstr_macro_f1") is not None]
                )
            metrics.update(rank_summary.get(str(scheduler_key), {}))
            family_block[str(scheduler_key)] = metrics
        seed_summary["families"][str(family)] = family_block
    return {"rows": augmented, "summary": seed_summary}


def _aggregate_main_table(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scheduler_keys = _ordered_scheduler_keys_from_rows(rows)
    by_seed: Dict[int, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    seed_summaries: Dict[int, Dict[str, Any]] = {}
    for seed, seed_rows in sorted(by_seed.items()):
        seed_summaries[int(seed)] = _aggregate_seed_rows(seed_rows)["summary"]

    main_summary: Dict[str, Any] = {"seed_summaries": seed_summaries, "families": {}}
    for family in (FORECAST_FAMILY, LOB_FAMILY):
        family_block: Dict[str, Any] = {}
        for scheduler_key in scheduler_keys:
            metric_values: Dict[str, List[float]] = {}
            for seed in sorted(seed_summaries):
                scheduler_summary = seed_summaries[seed]["families"][str(family)].get(str(scheduler_key), {})
                for metric_key, value in scheduler_summary.items():
                    if metric_key == "row_count" or value is None:
                        continue
                    metric_values.setdefault(str(metric_key), []).append(float(value))
            family_block[str(scheduler_key)] = {
                metric_key: {"mean": _mean(values), "std": _std(values)}
                for metric_key, values in sorted(metric_values.items())
            }
        main_summary["families"][str(family)] = family_block
    return main_summary


def _parse_dataset_float_map(text: str) -> Dict[str, float]:
    assignments: Dict[str, float] = {}
    raw = str(text or "").strip()
    if not raw:
        return assignments
    for token in parse_csv(raw):
        if "=" not in token:
            raise ValueError(
                "Dataset float assignments must use dataset=value tokens, "
                f"got {token!r}."
            )
        dataset, raw_value = token.split("=", 1)
        assignments[str(dataset).strip()] = float(raw_value)
    return assignments


def _resolved_forecast_selected_deltas(cli_args: argparse.Namespace) -> Dict[str, float]:
    json_path = str(getattr(cli_args, "forecast_selected_deltas_json", "") or "").strip()
    if json_path:
        payload = json.loads(Path(json_path).expanduser().resolve().read_text(encoding="utf-8"))
        selected = payload.get("selected_deltas", payload)
        return {str(dataset): float(value) for dataset, value in dict(selected).items()}
    parsed = _parse_dataset_float_map(str(getattr(cli_args, "forecast_selected_deltas", "") or ""))
    if parsed:
        return parsed
    return dict(DEFAULT_FORECAST_SELECTED_DELTAS)


def _format_variant_token(value: float) -> str:
    token = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return token.replace("-", "m").replace(".", "p")


def _parse_ablation_signal_variants(text: str) -> List[str]:
    names = [name.strip().lower() for name in parse_csv(str(text))]
    if not names:
        names = list(DEFAULT_TVD_RECOVERY_SIGNAL_VARIANTS)
    deduped: List[str] = []
    for name in names:
        if name not in DEFAULT_TVD_RECOVERY_SIGNAL_VARIANTS:
            raise ValueError(
                "Unsupported ablation signal variant "
                f"{name!r}; expected one of {list(DEFAULT_TVD_RECOVERY_SIGNAL_VARIANTS)}."
            )
        if name not in deduped:
            deduped.append(name)
    return deduped


def _signal_trace_key_for_variant(signal_variant: str) -> str:
    variant = str(signal_variant).strip().lower()
    if variant == "canonical":
        return DEFAULT_SIGNAL_TRACE_KEY
    if variant == "no_rstar":
        return EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_TRACE_KEY
    raise ValueError(f"Unsupported TVD signal variant {signal_variant!r}")


def _ensure_signal_trace_available(
    calibration: Mapping[str, Any],
    *,
    signal_trace_key: str,
) -> Dict[str, Any]:
    key = str(signal_trace_key)
    payload = calibration if isinstance(calibration, dict) else dict(calibration)
    if key in payload:
        return payload
    if key != EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_TRACE_KEY:
        raise KeyError(f"Calibration payload is missing requested signal trace {signal_trace_key!r}")

    residual_trace = payload.get("residual_norm_by_step")
    disagreement_trace = payload.get("disagreement_by_step")
    if residual_trace is None or disagreement_trace is None:
        raise KeyError(
            "Cannot synthesize no-rstar info-growth trace without residual_norm_by_step and disagreement_by_step."
        )
    no_rstar_trace = compute_no_rstar_info_growth_hardness_numpy(
        residual_trace,
        disagreement_trace,
    )
    payload[key] = [float(x) for x in np.asarray(no_rstar_trace, dtype=np.float64).tolist()]

    corr_map = dict(payload.get("signal_correlations_vs_oracle", {}))
    if key not in corr_map:
        rows = payload.get("rows")
        signal_values: List[float] = []
        oracle_values: List[float] = []
        if isinstance(rows, Sequence):
            for row in rows:
                if not isinstance(row, Mapping) or int(row.get("step_index", 0)) <= 0:
                    continue
                disagreement = _optional_float(row.get("disagreement"))
                residual = _optional_float(row.get("residual_norm"))
                oracle = _optional_float(row.get("oracle_local_error"))
                if disagreement is None or residual is None or oracle is None:
                    continue
                signal_values.append(float(disagreement) * float(np.log1p(max(float(residual), 0.0))))
                oracle_values.append(float(oracle))
        corr_map[key] = {"spearman": _optional_float(safe_spearman(signal_values, oracle_values))}
        payload["signal_correlations_vs_oracle"] = corr_map
    return payload


def _canonical_only_tvd_case(*, canonical_delta: float, signal_variant: str) -> Dict[str, Any]:
    variant = str(signal_variant).strip().lower()
    if variant == DEFAULT_CANONICAL_SIGNAL_VARIANT:
        return _build_tvd_variant_case(
            scheduler_variant_key=CANONICAL_TVD_SCHEDULER_KEY,
            scheduler_variant_name="TVD",
            canonical_delta=float(canonical_delta),
            signal_variant=variant,
        )
    if variant == "no_rstar":
        return _build_tvd_variant_case(
            scheduler_variant_key="tvd_no_rstar",
            scheduler_variant_name="TVD no-r*",
            canonical_delta=float(canonical_delta),
            signal_variant=variant,
        )
    raise ValueError(f"Unsupported canonical signal variant {signal_variant!r}")


def _resolved_recovery_solver_names(cli_args: argparse.Namespace) -> List[str]:
    requested = [name.strip().lower() for name in parse_csv(str(cli_args.ablation_solver_names))]
    expected = set(DEFAULT_FIXED_SOLVER_NAMES)
    if set(requested) != expected or len(requested) != len(DEFAULT_FIXED_SOLVER_NAMES):
        raise ValueError(
            "tvd_only_recovery requires ablation_solver_names to be exactly "
            f"{','.join(DEFAULT_FIXED_SOLVER_NAMES)}"
        )
    return list(DEFAULT_FIXED_SOLVER_NAMES)


def _selected_forecast_delta(selected_deltas: Mapping[str, Any], dataset: str) -> float:
    if str(dataset) not in selected_deltas:
        raise ValueError(f"Missing selected forecast delta for dataset={dataset}")
    return float(selected_deltas[str(dataset)])


def _build_tvd_variant_case(
    *,
    scheduler_variant_key: str,
    scheduler_variant_name: str,
    canonical_delta: float,
    signal_variant: str,
    uniform_blend: float = 0.0,
    gibbs_temperature: float = 1.0,
    reference_macro_factor: float = 4.0,
    r_star_multiplier: float = 1.0,
) -> Dict[str, Any]:
    return {
        "scheduler_key": CANONICAL_TVD_SCHEDULER_KEY,
        "scheduler_variant_key": str(scheduler_variant_key),
        "scheduler_variant_name": str(scheduler_variant_name),
        "canonical_delta": float(canonical_delta),
        "signal_trace_key": _signal_trace_key_for_variant(signal_variant),
        "uniform_blend": float(uniform_blend),
        "gibbs_temperature": float(gibbs_temperature),
        "reference_macro_factor": float(reference_macro_factor),
        "r_star_multiplier": float(r_star_multiplier),
    }


def _tvd_only_recovery_stage_a_cases(
    cli_args: argparse.Namespace,
    *,
    canonical_delta: float,
) -> List[Dict[str, Any]]:
    variants = _parse_ablation_signal_variants(str(cli_args.ablation_signal_variants))
    if set(variants) != set(DEFAULT_TVD_RECOVERY_SIGNAL_VARIANTS) or len(variants) != len(DEFAULT_TVD_RECOVERY_SIGNAL_VARIANTS):
        raise ValueError(
            "tvd_only_recovery Stage A requires ablation_signal_variants=canonical,no_rstar"
        )
    return [
        _build_tvd_variant_case(
            scheduler_variant_key="tvd_canonical",
            scheduler_variant_name="Canonical TVD",
            canonical_delta=float(canonical_delta),
            signal_variant="canonical",
        ),
        _build_tvd_variant_case(
            scheduler_variant_key="tvd_no_rstar",
            scheduler_variant_name="TVD no-r*",
            canonical_delta=float(canonical_delta),
            signal_variant="no_rstar",
        ),
    ]


def _tvd_only_recovery_stage_b_cases(
    *,
    canonical_delta: float,
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = [
        _build_tvd_variant_case(
            scheduler_variant_key="tvd_no_rstar",
            scheduler_variant_name="TVD no-r*",
            canonical_delta=float(canonical_delta),
            signal_variant="no_rstar",
            reference_macro_factor=4.0,
        )
    ]
    for blend in DEFAULT_TVD_RECOVERY_STAGE_B_BLEND_VALUES:
        token = _format_variant_token(float(blend))
        cases.append(
            _build_tvd_variant_case(
                scheduler_variant_key=f"tvd_no_rstar_blend_{token}",
                scheduler_variant_name=f"TVD no-r* blend {float(blend):.2f}",
                canonical_delta=float(canonical_delta),
                signal_variant="no_rstar",
                uniform_blend=float(blend),
                reference_macro_factor=4.0,
            )
        )
    for temperature in DEFAULT_TVD_RECOVERY_STAGE_B_TEMPERATURE_VALUES:
        token = _format_variant_token(float(temperature))
        cases.append(
            _build_tvd_variant_case(
                scheduler_variant_key=f"tvd_no_rstar_temp_{token}",
                scheduler_variant_name=f"TVD no-r* temp {float(temperature):.2f}",
                canonical_delta=float(canonical_delta),
                signal_variant="no_rstar",
                gibbs_temperature=float(temperature),
                reference_macro_factor=4.0,
            )
        )
    return cases


def _load_external_uniform_comparator_rows(csv_path: Path) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    rows_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    if not csv_path.exists():
        raise FileNotFoundError(f"Comparator rows CSV not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            if not raw:
                continue
            if str(raw.get("benchmark_family")) != FORECAST_FAMILY:
                continue
            if str(raw.get("split_phase")) != LOCKED_TEST_PHASE:
                continue
            if str(raw.get("scheduler_key")) != UNIFORM_SCHEDULER_KEY:
                continue
            if str(raw.get("row_status", "complete")) != "complete":
                continue
            row = {
                "benchmark_family": str(raw["benchmark_family"]),
                "split_phase": str(raw["split_phase"]),
                "seed": int(raw["seed"]),
                "dataset": str(raw["dataset"]),
                "backbone_name": str(raw["backbone_name"]),
                "train_steps": int(raw["train_steps"]),
                "train_budget_label": str(raw["train_budget_label"]),
                "checkpoint_id": str(raw["checkpoint_id"]),
                "target_nfe": int(raw["target_nfe"]),
                "solver_key": str(raw["solver_key"]),
                "experiment_scope": str(raw.get("experiment_scope") or solver_experiment_scope(str(raw["solver_key"]))),
                "scheduler_key": UNIFORM_SCHEDULER_KEY,
                "crps": _optional_float(raw.get("crps")),
                "mase": _optional_float(raw.get("mase")),
            }
            item_key = _ranking_match_key(row)
            if item_key in rows_by_key:
                raise ValueError(f"Duplicate external uniform comparator row for key={item_key}")
            rows_by_key[item_key] = row
    if not rows_by_key:
        raise ValueError(f"No locked-test uniform forecast rows found in comparator CSV: {csv_path}")
    return rows_by_key


def _comparator_match_key_for_context(
    *,
    benchmark_family: str,
    split_phase: str,
    seed: int,
    dataset: str,
    checkpoint: Mapping[str, Any],
    target_nfe: int,
    solver_key: str,
) -> Tuple[Any, ...]:
    return _ranking_match_key_from_values(
        benchmark_family=str(benchmark_family),
        split_phase=str(split_phase),
        seed=int(seed),
        dataset=str(dataset),
        backbone_name=str(checkpoint["backbone_name"]),
        train_steps=int(checkpoint["train_steps"]),
        train_budget_label=str(checkpoint["train_budget_label"]),
        checkpoint_id=str(checkpoint["checkpoint_id"]),
        target_nfe=int(target_nfe),
        solver_key=str(solver_key),
        experiment_scope=str(solver_experiment_scope(str(solver_key))),
    )


def _with_external_uniform_metrics(
    row: Mapping[str, Any],
    comparator_rows_by_key: Mapping[Tuple[Any, ...], Mapping[str, Any]],
) -> Dict[str, Any]:
    payload = dict(row)
    comparator = comparator_rows_by_key.get(_ranking_match_key(row))
    if comparator is None:
        return payload
    payload["relative_crps_gain_vs_uniform"] = _safe_relative_gain(
        row.get("crps"),
        comparator.get("crps"),
    )
    payload["relative_mase_gain_vs_uniform"] = _safe_relative_gain(
        row.get("mase"),
        comparator.get("mase"),
    )
    return payload


def _enrich_rows_with_external_uniform_metrics(
    rows: Sequence[Mapping[str, Any]],
    comparator_rows_by_key: Mapping[Tuple[Any, ...], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        if (
            str(row.get("benchmark_family")) == FORECAST_FAMILY
            and str(row.get("split_phase")) == LOCKED_TEST_PHASE
            and str(row.get("scheduler_key")) != UNIFORM_SCHEDULER_KEY
        ):
            payload = _with_external_uniform_metrics(payload, comparator_rows_by_key)
        enriched.append(payload)
    return enriched


def _cloned_cli_args(cli_args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    cloned = argparse.Namespace(**vars(cli_args))
    for key, value in updates.items():
        setattr(cloned, str(key), value)
    return cloned


def _forecast_ablation_variant_cases(
    cli_args: argparse.Namespace,
    *,
    canonical_delta: float,
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = [
        {
            "scheduler_key": UNIFORM_SCHEDULER_KEY,
            "scheduler_variant_key": UNIFORM_SCHEDULER_KEY,
            "scheduler_variant_name": "Uniform",
            "canonical_delta": None,
        },
        {
            "scheduler_key": CANONICAL_TVD_SCHEDULER_KEY,
            "scheduler_variant_key": "tvd_canonical",
            "scheduler_variant_name": "Canonical TVD",
            "canonical_delta": float(canonical_delta),
        },
    ]
    for blend in parse_float_csv(str(cli_args.ablation_uniform_blend_values)):
        if abs(float(blend) - float(cli_args.uniform_blend)) <= 1e-12:
            continue
        token = _format_variant_token(float(blend))
        cases.append(
            {
                "scheduler_key": CANONICAL_TVD_SCHEDULER_KEY,
                "scheduler_variant_key": f"tvd_uniform_blend_{token}",
                "scheduler_variant_name": f"TVD blend {float(blend):.2f}",
                "canonical_delta": float(canonical_delta),
                "uniform_blend": float(blend),
            }
        )
    for temperature in parse_float_csv(str(cli_args.ablation_gibbs_temperature_values)):
        if abs(float(temperature) - float(cli_args.gibbs_temperature)) <= 1e-12:
            continue
        token = _format_variant_token(float(temperature))
        cases.append(
            {
                "scheduler_key": CANONICAL_TVD_SCHEDULER_KEY,
                "scheduler_variant_key": f"tvd_gibbs_temperature_{token}",
                "scheduler_variant_name": f"TVD temp {float(temperature):.2f}",
                "canonical_delta": float(canonical_delta),
                "gibbs_temperature": float(temperature),
            }
        )
    for factor in parse_float_csv(str(cli_args.ablation_reference_macro_factors)):
        if abs(float(factor) - float(cli_args.reference_macro_factor)) <= 1e-12:
            continue
        token = _format_variant_token(float(factor))
        cases.append(
            {
                "scheduler_key": CANONICAL_TVD_SCHEDULER_KEY,
                "scheduler_variant_key": f"tvd_reference_macro_factor_{token}",
                "scheduler_variant_name": f"TVD ref factor {float(factor):.2f}",
                "canonical_delta": float(canonical_delta),
                "reference_macro_factor": float(factor),
            }
        )
    for multiplier in parse_float_csv(str(cli_args.ablation_r_star_multipliers)):
        if abs(float(multiplier) - float(cli_args.r_star_multiplier)) <= 1e-12:
            continue
        token = _format_variant_token(float(multiplier))
        cases.append(
            {
                "scheduler_key": CANONICAL_TVD_SCHEDULER_KEY,
                "scheduler_variant_key": f"tvd_r_star_multiplier_{token}",
                "scheduler_variant_name": f"TVD r* x{float(multiplier):.2f}",
                "canonical_delta": float(canonical_delta),
                "r_star_multiplier": float(multiplier),
            }
        )
    for gamma in parse_float_csv(str(cli_args.ablation_hardness_tilt_gamma_values)):
        if abs(float(gamma) - float(cli_args.hardness_tilt_gamma)) <= 1e-12:
            continue
        token = _format_variant_token(float(gamma))
        cases.append(
            {
                "scheduler_key": CANONICAL_TVD_SCHEDULER_KEY,
                "scheduler_variant_key": f"tvd_late_tilt_gamma_{token}",
                "scheduler_variant_name": f"TVD late tilt gamma {float(gamma):.2f}",
                "canonical_delta": float(canonical_delta),
                "hardness_tilt_gamma": float(gamma),
            }
        )
    deduped: Dict[str, Dict[str, Any]] = {}
    for case in cases:
        deduped[str(case["scheduler_variant_key"])] = dict(case)
    return [deduped[key] for key in sorted(deduped, key=lambda item: (item != UNIFORM_SCHEDULER_KEY, item))]


def _forecast_scheduler_ablation_cases_by_dataset(
    cli_args: argparse.Namespace,
) -> Dict[str, List[Dict[str, Any]]]:
    selected_deltas = _resolved_forecast_selected_deltas(cli_args)
    cases_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for dataset in parse_forecast_datasets(str(cli_args.forecast_datasets)):
        if dataset not in selected_deltas:
            raise ValueError(f"Missing selected forecast delta for dataset={dataset}")
        cases_by_dataset[str(dataset)] = _forecast_ablation_variant_cases(
            cli_args,
            canonical_delta=float(selected_deltas[str(dataset)]),
        )
    return cases_by_dataset


def _matched_recovery_forecast_datasets(cli_args: argparse.Namespace) -> List[str]:
    datasets = parse_csv(str(cli_args.matched_recovery_forecast_datasets))
    if not datasets:
        raise ValueError("matched_tvd_uniform_recovery requires at least one forecast dataset.")
    return [str(dataset) for dataset in datasets]


def _matched_recovery_solver_names(cli_args: argparse.Namespace) -> List[str]:
    solver_names = parse_csv(str(cli_args.matched_recovery_solver_names))
    if not solver_names:
        raise ValueError("matched_tvd_uniform_recovery requires at least one solver.")
    unsupported = [name for name in solver_names if name not in ALL_SOLVER_ORDER]
    if unsupported:
        raise ValueError(f"Unsupported matched recovery solver(s): {unsupported}")
    return list(solver_names)


def _matched_recovery_variant_key_filter(cli_args: argparse.Namespace) -> Optional[set[str]]:
    requested = {str(key) for key in parse_csv(str(getattr(cli_args, "matched_recovery_variant_keys", "")))}
    return requested or None


def _matched_tvd_uniform_variant_cases(
    *,
    canonical_delta: float,
    calibration_trace_samples: int,
) -> List[Dict[str, Any]]:
    trace_samples = int(calibration_trace_samples)
    prefix = f"tvd_mc{trace_samples}"
    cases: List[Dict[str, Any]] = [
        {
            "scheduler_key": CANONICAL_TVD_SCHEDULER_KEY,
            "scheduler_variant_key": f"{prefix}_canonical",
            "scheduler_variant_name": f"TVD MC{trace_samples} canonical",
            "canonical_delta": float(canonical_delta),
            "calibration_trace_samples": int(trace_samples),
        }
    ]
    for multiplier in DEFAULT_MATCHED_TVD_UNIFORM_CAP_MULTIPLIERS:
        token = _format_variant_token(float(multiplier))
        cases.append(
            {
                "scheduler_key": CANONICAL_TVD_SCHEDULER_KEY,
                "scheduler_variant_key": f"{prefix}_cap_{token}",
                "scheduler_variant_name": f"TVD MC{trace_samples} cap {float(multiplier):.2f}",
                "canonical_delta": float(canonical_delta),
                "mass_cap_multiplier": float(multiplier),
                "calibration_trace_samples": int(trace_samples),
            }
        )
    for floor_multiplier, cap_multiplier in DEFAULT_MATCHED_TVD_UNIFORM_MASS_BANDS:
        floor_token = _format_variant_token(float(floor_multiplier))
        cap_token = _format_variant_token(float(cap_multiplier))
        cases.append(
            {
                "scheduler_key": CANONICAL_TVD_SCHEDULER_KEY,
                "scheduler_variant_key": f"{prefix}_band_floor{floor_token}_cap{cap_token}",
                "scheduler_variant_name": (
                    f"TVD MC{trace_samples} band floor {float(floor_multiplier):.2f} "
                    f"cap {float(cap_multiplier):.2f}"
                ),
                "canonical_delta": float(canonical_delta),
                "mass_floor_multiplier": float(floor_multiplier),
                "mass_cap_multiplier": float(cap_multiplier),
                "calibration_trace_samples": int(trace_samples),
            }
        )
    for blend in DEFAULT_MATCHED_TVD_UNIFORM_BLEND_VALUES:
        token = _format_variant_token(float(blend))
        cases.append(
            {
                "scheduler_key": CANONICAL_TVD_SCHEDULER_KEY,
                "scheduler_variant_key": f"{prefix}_blend_{token}",
                "scheduler_variant_name": f"TVD MC{trace_samples} blend {float(blend):.2f}",
                "canonical_delta": float(canonical_delta),
                "uniform_blend": float(blend),
                "calibration_trace_samples": int(trace_samples),
            }
        )
    for grid_blend in DEFAULT_MATCHED_TVD_UNIFORM_GRID_BLEND_VALUES:
        token = _format_variant_token(float(grid_blend))
        cases.append(
            {
                "scheduler_key": CANONICAL_TVD_SCHEDULER_KEY,
                "scheduler_variant_key": f"{prefix}_gridblend_{token}",
                "scheduler_variant_name": f"TVD MC{trace_samples} grid blend {float(grid_blend):.2f}",
                "canonical_delta": float(canonical_delta),
                "grid_uniform_blend": float(grid_blend),
                "calibration_trace_samples": int(trace_samples),
            }
        )
    for blend in DEFAULT_MATCHED_TVD_UNIFORM_NO_RSTAR_BLEND_VALUES:
        token = _format_variant_token(float(blend))
        cases.append(
            {
                "scheduler_key": CANONICAL_TVD_SCHEDULER_KEY,
                "scheduler_variant_key": f"{prefix}_no_rstar_blend_{token}",
                "scheduler_variant_name": f"TVD MC{trace_samples} no-rstar blend {float(blend):.2f}",
                "signal_trace_key": EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_TRACE_KEY,
                "canonical_delta": float(canonical_delta),
                "uniform_blend": float(blend),
                "calibration_trace_samples": int(trace_samples),
            }
        )
    for temperature in DEFAULT_MATCHED_TVD_UNIFORM_TEMPERATURE_VALUES:
        token = _format_variant_token(float(temperature))
        cases.append(
            {
                "scheduler_key": CANONICAL_TVD_SCHEDULER_KEY,
                "scheduler_variant_key": f"{prefix}_temp_{token}",
                "scheduler_variant_name": f"TVD MC{trace_samples} temp {float(temperature):.2f}",
                "canonical_delta": float(canonical_delta),
                "gibbs_temperature": float(temperature),
                "calibration_trace_samples": int(trace_samples),
            }
        )
    return cases


def _matched_tvd_uniform_validation_cases_by_dataset(
    cli_args: argparse.Namespace,
) -> Dict[str, List[Dict[str, Any]]]:
    trace_samples = int(cli_args.matched_recovery_calibration_trace_samples)
    variant_key_filter = _matched_recovery_variant_key_filter(cli_args)
    cases_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for dataset in _matched_recovery_forecast_datasets(cli_args):
        cases: List[Dict[str, Any]] = [
            {
                "scheduler_key": UNIFORM_SCHEDULER_KEY,
                "scheduler_variant_key": UNIFORM_SCHEDULER_KEY,
                "scheduler_variant_name": "Uniform",
                "canonical_delta": None,
            }
        ]
        for delta_value in parse_float_csv(str(cli_args.canonical_delta_values)):
            for case in _matched_tvd_uniform_variant_cases(
                canonical_delta=float(delta_value),
                calibration_trace_samples=int(trace_samples),
            ):
                if variant_key_filter is not None and str(case["scheduler_variant_key"]) not in variant_key_filter:
                    continue
                cases.append(case)
        cases_by_dataset[str(dataset)] = cases
    return cases_by_dataset


def _matched_tvd_uniform_variant_preference(candidate: Mapping[str, Any]) -> Tuple[float, float, float, float, float, int]:
    floor = _optional_float(candidate.get("mass_floor_multiplier"))
    cap = _optional_float(candidate.get("mass_cap_multiplier"))
    grid_blend = _optional_float(candidate.get("grid_uniform_blend"))
    blend = _optional_float(candidate.get("uniform_blend"))
    temperature = _optional_float(candidate.get("gibbs_temperature"))
    variant_key = str(candidate.get("scheduler_variant_key", ""))
    return (
        float("inf") if cap is None or float(cap) <= 0.0 else float(cap),
        -(0.0 if floor is None else float(floor)),
        -(0.0 if grid_blend is None else float(grid_blend)),
        -(0.0 if blend is None else float(blend)),
        -(1.0 if temperature is None else float(temperature)),
        0 if variant_key.endswith("_canonical") else 1,
    )


def _select_matched_tvd_uniform_candidates(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    augmented = _augment_rows_with_relative_metrics_extended(rows)
    datasets: Dict[str, List[Mapping[str, Any]]] = {}
    for row in augmented:
        if str(row.get("benchmark_family")) != FORECAST_FAMILY:
            continue
        if str(row.get("split_phase")) != VALIDATION_PHASE:
            continue
        datasets.setdefault(str(row["dataset"]), []).append(row)

    payload: Dict[str, Any] = {"datasets": {}, "selected_cases": {}}
    for dataset, dataset_rows in sorted(datasets.items()):
        grouped: Dict[Tuple[str, float], List[Mapping[str, Any]]] = {}
        for row in dataset_rows:
            if not _is_canonical_tvd_scheduler(str(row.get("scheduler_key"))):
                continue
            delta = _optional_float(row.get("canonical_delta"))
            if delta is None:
                continue
            grouped.setdefault((_scheduler_variant_key_from_row(row), float(delta)), []).append(row)

        candidates: List[Dict[str, Any]] = []
        for (variant_key, delta), candidate_rows in sorted(grouped.items()):
            crps_values = [
                float(row["relative_crps_gain_vs_uniform"])
                for row in candidate_rows
                if row.get("relative_crps_gain_vs_uniform") is not None
            ]
            mase_values = [
                float(row["relative_mase_gain_vs_uniform"])
                for row in candidate_rows
                if row.get("relative_mase_gain_vs_uniform") is not None
            ]
            mean_crps = _mean(crps_values)
            mean_mase = _mean(mase_values)
            mean_q50 = _mean(
                [
                    float(row["runtime_grid_q50"])
                    for row in candidate_rows
                    if row.get("runtime_grid_q50") is not None
                ]
            )
            first = dict(candidate_rows[0])
            failed_reasons: List[str] = []
            if mean_crps is None or float(mean_crps) < 0.0:
                failed_reasons.append("negative_mean_relative_crps_gain_vs_uniform")
            if mean_mase is None or float(mean_mase) < 0.0:
                failed_reasons.append("negative_mean_relative_mase_gain_vs_uniform")
            if (
                str(dataset) == DEFAULT_TVD_RECOVERY_WIND_DATASET
                and mean_q50 is not None
                and float(mean_q50) < 0.25
                and mean_mase is not None
                and float(mean_mase) < 0.0
            ):
                failed_reasons.append("wind_front_loaded_negative_mase")
            candidates.append(
                {
                    "scheduler_key": CANONICAL_TVD_SCHEDULER_KEY,
                    "scheduler_variant_key": str(variant_key),
                    "scheduler_variant_name": str(first.get("scheduler_variant_name") or variant_key),
                    "signal_trace_key": str(first.get("signal_trace_key") or DEFAULT_SIGNAL_TRACE_KEY),
                    "canonical_delta": float(delta),
                    "uniform_blend": _optional_float(first.get("uniform_blend")),
                    "gibbs_temperature": _optional_float(first.get("gibbs_temperature")),
                    "reference_macro_factor": _optional_float(first.get("reference_macro_factor")),
                    "r_star_multiplier": _optional_float(first.get("r_star_multiplier")),
                    "mass_floor_multiplier": _optional_float(first.get("mass_floor_multiplier")),
                    "mean_mass_floor_hit_count": _mean(
                        [
                            float(row["mass_floor_hit_count"])
                            for row in candidate_rows
                            if row.get("mass_floor_hit_count") is not None
                        ]
                    ),
                    "mean_mass_floor_deficit_share": _mean(
                        [
                            float(row["mass_floor_deficit_share"])
                            for row in candidate_rows
                            if row.get("mass_floor_deficit_share") is not None
                        ]
                    ),
                    "mass_cap_multiplier": _optional_float(first.get("mass_cap_multiplier")),
                    "mean_mass_cap_hit_count": _mean(
                        [
                            float(row["mass_cap_hit_count"])
                            for row in candidate_rows
                            if row.get("mass_cap_hit_count") is not None
                        ]
                    ),
                    "mean_mass_cap_overflow_share": _mean(
                        [
                            float(row["mass_cap_overflow_share"])
                            for row in candidate_rows
                            if row.get("mass_cap_overflow_share") is not None
                        ]
                    ),
                    "grid_uniform_blend": _optional_float(first.get("grid_uniform_blend")),
                    "calibration_trace_samples": int(first.get("calibration_trace_samples") or 1),
                    "matched_cells": int(len(candidate_rows)),
                    "mean_relative_crps_gain_vs_uniform": mean_crps,
                    "mean_relative_mase_gain_vs_uniform": mean_mase,
                    "mean_runtime_grid_q50": mean_q50,
                    "rejected": bool(failed_reasons),
                    "rejection_reasons": failed_reasons,
                }
            )
        eligible = [candidate for candidate in candidates if not bool(candidate["rejected"])]
        selected = None
        if eligible:
            best_crps = max(float(candidate["mean_relative_crps_gain_vs_uniform"]) for candidate in eligible)
            within_tolerance = [
                candidate
                for candidate in eligible
                if best_crps - float(candidate["mean_relative_crps_gain_vs_uniform"])
                <= DEFAULT_MATCHED_TVD_UNIFORM_CRPS_TIE_TOLERANCE
            ]
            selected = min(
                within_tolerance,
                key=lambda candidate: (*_matched_tvd_uniform_variant_preference(candidate), float(candidate["canonical_delta"])),
            )
        payload["datasets"][str(dataset)] = {
            "selection_metric": "relative_crps_gain_vs_uniform",
            "guardrails": {
                "mean_relative_crps_gain_vs_uniform_must_be_non_negative": True,
                "mean_relative_mase_gain_vs_uniform_must_be_non_negative": True,
                "wind_front_loaded_negative_mase_rejected": True,
                "cap_tie_tolerance_relative_crps": float(DEFAULT_MATCHED_TVD_UNIFORM_CRPS_TIE_TOLERANCE),
                "tie_breaker_prefers_more_uniform_band_or_grid_shape": True,
            },
            "candidates": candidates,
            "selected_candidate": selected,
        }
        if selected is not None:
            payload["selected_cases"][str(dataset)] = {
                key: selected[key]
                for key in (
                    "scheduler_key",
                    "scheduler_variant_key",
                    "scheduler_variant_name",
                    "signal_trace_key",
                    "canonical_delta",
                    "uniform_blend",
                    "gibbs_temperature",
                    "reference_macro_factor",
                    "r_star_multiplier",
                    "mass_floor_multiplier",
                    "mass_cap_multiplier",
                    "grid_uniform_blend",
                    "calibration_trace_samples",
                )
                if selected.get(key) is not None
            }
    return payload


def _matched_tvd_uniform_locked_cases_by_dataset(
    cli_args: argparse.Namespace,
    selection: Mapping[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    cases_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    selected_cases = dict(selection.get("selected_cases", {}))
    for dataset in _matched_recovery_forecast_datasets(cli_args):
        cases: List[Dict[str, Any]] = [
            {
                "scheduler_key": UNIFORM_SCHEDULER_KEY,
                "scheduler_variant_key": UNIFORM_SCHEDULER_KEY,
                "scheduler_variant_name": "Uniform",
                "canonical_delta": None,
            }
        ]
        if dataset in selected_cases:
            cases.append(dict(selected_cases[str(dataset)]))
        cases_by_dataset[str(dataset)] = cases
    return cases_by_dataset


def _summarize_paired_rows(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in paired_rows:
        grouped.setdefault(str(row[group_key]), []).append(row)
    summary: Dict[str, Dict[str, Any]] = {}
    for key, rows in sorted(grouped.items()):
        summary[str(key)] = {
            "matched_cells": int(len(rows)),
            "mean_relative_crps_gain_vs_uniform": _mean(
                [float(row["relative_crps_gain_vs_uniform"]) for row in rows if row.get("relative_crps_gain_vs_uniform") is not None]
            ),
            "mean_relative_mase_gain_vs_uniform": _mean(
                [float(row["relative_mase_gain_vs_uniform"]) for row in rows if row.get("relative_mase_gain_vs_uniform") is not None]
            ),
            "mean_signal_validation_spearman": _mean(
                [float(row["signal_validation_spearman"]) for row in rows if row.get("signal_validation_spearman") is not None]
            ),
            "mean_interval_mass_top3_share": _mean(
                [float(row["interval_mass_top3_share"]) for row in rows if row.get("interval_mass_top3_share") is not None]
            ),
            "mean_interval_mass_max_min_ratio": _mean(
                [float(row["interval_mass_max_min_ratio"]) for row in rows if row.get("interval_mass_max_min_ratio") is not None]
            ),
        }
    return summary


def _diagnose_forecast_locked_test_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    augmented = _augment_rows_with_relative_metrics_extended(rows)
    paired_rows: List[Dict[str, Any]] = []
    for row in augmented:
        if str(row.get("benchmark_family")) != FORECAST_FAMILY:
            continue
        if str(row.get("split_phase")) != LOCKED_TEST_PHASE:
            continue
        if str(row.get("row_status", "complete")) != "complete":
            continue
        if not _is_canonical_tvd_scheduler(str(row.get("scheduler_key"))):
            continue
        if row.get("relative_crps_gain_vs_uniform") is None:
            continue
        paired_rows.append(
            {
                "dataset": str(row["dataset"]),
                "seed": int(row["seed"]),
                "solver_key": str(row["solver_key"]),
                "target_nfe": int(row["target_nfe"]),
                "scheduler_variant_key": _scheduler_variant_key_from_row(row),
                "canonical_delta": _optional_float(row.get("canonical_delta")),
                "relative_crps_gain_vs_uniform": _optional_float(row.get("relative_crps_gain_vs_uniform")),
                "relative_mase_gain_vs_uniform": _optional_float(row.get("relative_mase_gain_vs_uniform")),
                "r_star": _optional_float(row.get("r_star")),
                "signal_validation_spearman": _optional_float(row.get("signal_validation_spearman")),
                "interval_mass_top1_share": _optional_float(row.get("interval_mass_top1_share")),
                "interval_mass_top3_share": _optional_float(row.get("interval_mass_top3_share")),
                "interval_mass_max_min_ratio": _optional_float(row.get("interval_mass_max_min_ratio")),
                "runtime_grid_q25": _optional_float(row.get("runtime_grid_q25")),
                "runtime_grid_q50": _optional_float(row.get("runtime_grid_q50")),
                "runtime_grid_q75": _optional_float(row.get("runtime_grid_q75")),
            }
        )
    paired_rows = sorted(
        paired_rows,
        key=lambda row: (
            str(row["dataset"]),
            int(row["seed"]),
            str(row["solver_key"]),
            int(row["target_nfe"]),
            str(row["scheduler_variant_key"]),
        ),
    )
    by_dataset = _summarize_paired_rows(paired_rows, group_key="dataset")
    by_solver = _summarize_paired_rows(paired_rows, group_key="solver_key")
    by_variant = _summarize_paired_rows(paired_rows, group_key="scheduler_variant_key")
    dominant_negative_mase_dataset = None
    dataset_candidates = [
        (dataset, payload["mean_relative_mase_gain_vs_uniform"])
        for dataset, payload in by_dataset.items()
        if payload.get("mean_relative_mase_gain_vs_uniform") is not None
    ]
    if dataset_candidates:
        dominant_negative_mase_dataset = min(dataset_candidates, key=lambda item: float(item[1]))[0]
    return {
        "matched_pair_count": int(len(paired_rows)),
        "paired_rows": paired_rows,
        "by_dataset": by_dataset,
        "by_solver": by_solver,
        "by_variant": by_variant,
        "dominant_negative_mase_dataset": dominant_negative_mase_dataset,
    }


def _select_promoted_forecast_variant(summary: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    variants = dict(summary.get("variants", {}))
    baseline = variants.get("tvd_canonical")
    baseline_crps = None if baseline is None else _optional_float(baseline.get("mean_relative_crps_gain_vs_uniform"))
    candidates: List[Dict[str, Any]] = []
    for variant_key, payload in variants.items():
        if variant_key in {"uniform", "tvd_canonical"}:
            continue
        mean_crps = _optional_float(payload.get("mean_relative_crps_gain_vs_uniform"))
        if mean_crps is None:
            continue
        if baseline_crps is not None and float(mean_crps) <= float(baseline_crps) + 1e-12:
            continue
        wind_stats = dict(payload.get("by_dataset", {}).get("wind_farms_wo_missing", {}))
        sf_stats = dict(payload.get("by_dataset", {}).get("san_francisco_traffic", {}))
        wind_mase = _optional_float(wind_stats.get("mean_relative_mase_gain_vs_uniform"))
        sf_crps = _optional_float(sf_stats.get("mean_relative_crps_gain_vs_uniform"))
        sf_mase = _optional_float(sf_stats.get("mean_relative_mase_gain_vs_uniform"))
        if wind_mase is None or float(wind_mase) < 0.0:
            continue
        if sf_crps is None or float(sf_crps) < 0.0:
            continue
        if sf_mase is None or float(sf_mase) < 0.0:
            continue
        candidates.append(
            {
                "scheduler_variant_key": str(variant_key),
                "mean_relative_crps_gain_vs_uniform": float(mean_crps),
                "wind_mean_relative_mase_gain_vs_uniform": float(wind_mase),
                "san_francisco_mean_relative_crps_gain_vs_uniform": float(sf_crps),
                "san_francisco_mean_relative_mase_gain_vs_uniform": float(sf_mase),
            }
        )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            float(item["mean_relative_crps_gain_vs_uniform"]),
            float(item["wind_mean_relative_mase_gain_vs_uniform"]),
            -len(str(item["scheduler_variant_key"])),
        ),
    )


def _summarize_forecast_ablation(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    diagnosis = _diagnose_forecast_locked_test_rows(rows)
    variants: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in diagnosis["paired_rows"]:
        grouped.setdefault(str(row["scheduler_variant_key"]), []).append(row)
    for variant_key, variant_rows in sorted(grouped.items()):
        variants[str(variant_key)] = {
            "matched_cells": int(len(variant_rows)),
            "mean_relative_crps_gain_vs_uniform": _mean(
                [float(row["relative_crps_gain_vs_uniform"]) for row in variant_rows if row.get("relative_crps_gain_vs_uniform") is not None]
            ),
            "mean_relative_mase_gain_vs_uniform": _mean(
                [float(row["relative_mase_gain_vs_uniform"]) for row in variant_rows if row.get("relative_mase_gain_vs_uniform") is not None]
            ),
            "by_dataset": _summarize_paired_rows(variant_rows, group_key="dataset"),
        }
    promotion = _select_promoted_forecast_variant({"variants": variants})
    return {
        "matched_pair_count": int(diagnosis["matched_pair_count"]),
        "variants": variants,
        "promotion_candidate": promotion,
        "promotion_criteria": {
            "mean_relative_crps_gain_vs_uniform_must_improve": True,
            "wind_mean_relative_mase_gain_vs_uniform_must_be_non_negative": True,
            "san_francisco_relative_crps_gain_vs_uniform_must_be_non_negative": True,
            "san_francisco_relative_mase_gain_vs_uniform_must_be_non_negative": True,
        },
        "dominant_negative_mase_dataset": diagnosis.get("dominant_negative_mase_dataset"),
    }


def _balanced_dual_gain(
    relative_crps_gain_vs_uniform: Any,
    relative_mase_gain_vs_uniform: Any,
) -> Optional[float]:
    crps = _optional_float(relative_crps_gain_vs_uniform)
    mase = _optional_float(relative_mase_gain_vs_uniform)
    if crps is None or mase is None:
        return None
    return float(min(crps, mase))


def _nonnegative_dual_gain(payload: Mapping[str, Any]) -> bool:
    crps = _optional_float(payload.get("mean_relative_crps_gain_vs_uniform"))
    mase = _optional_float(payload.get("mean_relative_mase_gain_vs_uniform"))
    return crps is not None and mase is not None and float(crps) >= 0.0 and float(mase) >= 0.0


def _summarize_tvd_only_recovery_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    diagnosis = _diagnose_forecast_locked_test_rows(rows)
    variants: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in diagnosis["paired_rows"]:
        grouped.setdefault(str(row["scheduler_variant_key"]), []).append(row)
    for variant_key, variant_rows in sorted(grouped.items()):
        mean_crps = _mean(
            [float(row["relative_crps_gain_vs_uniform"]) for row in variant_rows if row.get("relative_crps_gain_vs_uniform") is not None]
        )
        mean_mase = _mean(
            [float(row["relative_mase_gain_vs_uniform"]) for row in variant_rows if row.get("relative_mase_gain_vs_uniform") is not None]
        )
        variants[str(variant_key)] = {
            "matched_cells": int(len(variant_rows)),
            "mean_relative_crps_gain_vs_uniform": mean_crps,
            "mean_relative_mase_gain_vs_uniform": mean_mase,
            "balance_both_score": _balanced_dual_gain(mean_crps, mean_mase),
            "both_metrics_non_negative": bool(
                mean_crps is not None and mean_mase is not None and float(mean_crps) >= 0.0 and float(mean_mase) >= 0.0
            ),
            "by_dataset": _summarize_paired_rows(variant_rows, group_key="dataset"),
            "by_solver": _summarize_paired_rows(variant_rows, group_key="solver_key"),
            "by_target_nfe": _summarize_paired_rows(variant_rows, group_key="target_nfe"),
        }
    return {
        "matched_pair_count": int(diagnosis["matched_pair_count"]),
        "paired_rows": diagnosis["paired_rows"],
        "variants": variants,
        "by_dataset": diagnosis["by_dataset"],
        "by_solver": diagnosis["by_solver"],
        "dominant_negative_mase_dataset": diagnosis.get("dominant_negative_mase_dataset"),
    }


def _select_tvd_only_recovery_stage_a_candidate(summary: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    variants = dict(summary.get("variants", {}))
    baseline = dict(variants.get("tvd_canonical", {}))
    baseline_crps = _optional_float(baseline.get("mean_relative_crps_gain_vs_uniform"))
    baseline_mase = _optional_float(baseline.get("mean_relative_mase_gain_vs_uniform"))
    candidates: List[Dict[str, Any]] = []
    for variant_key, payload in variants.items():
        if str(variant_key) == "tvd_canonical":
            continue
        mean_crps = _optional_float(payload.get("mean_relative_crps_gain_vs_uniform"))
        mean_mase = _optional_float(payload.get("mean_relative_mase_gain_vs_uniform"))
        balance_score = _balanced_dual_gain(mean_crps, mean_mase)
        if mean_crps is None or mean_mase is None or balance_score is None:
            continue
        if float(mean_crps) < 0.0 or float(mean_mase) < 0.0:
            continue
        if baseline_crps is None or baseline_mase is None:
            continue
        if float(mean_crps) <= float(baseline_crps) + 1e-12:
            continue
        if float(mean_mase) <= float(baseline_mase) + 1e-12:
            continue
        candidates.append(
            {
                "scheduler_variant_key": str(variant_key),
                "matched_cells": int(payload.get("matched_cells", 0)),
                "mean_relative_crps_gain_vs_uniform": float(mean_crps),
                "mean_relative_mase_gain_vs_uniform": float(mean_mase),
                "balance_both_score": float(balance_score),
            }
        )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            float(item["balance_both_score"]),
            float(item["mean_relative_crps_gain_vs_uniform"]),
            float(item["mean_relative_mase_gain_vs_uniform"]),
            -len(str(item["scheduler_variant_key"])),
        ),
    )


def _select_tvd_only_recovery_stage_b_candidate(summary: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    variants = dict(summary.get("variants", {}))
    candidates: List[Dict[str, Any]] = []
    for variant_key, payload in variants.items():
        mean_crps = _optional_float(payload.get("mean_relative_crps_gain_vs_uniform"))
        mean_mase = _optional_float(payload.get("mean_relative_mase_gain_vs_uniform"))
        balance_score = _balanced_dual_gain(mean_crps, mean_mase)
        if mean_crps is None or mean_mase is None or balance_score is None:
            continue
        if float(mean_crps) < 0.0 or float(mean_mase) < 0.0:
            continue
        candidates.append(
            {
                "scheduler_variant_key": str(variant_key),
                "matched_cells": int(payload.get("matched_cells", 0)),
                "mean_relative_crps_gain_vs_uniform": float(mean_crps),
                "mean_relative_mase_gain_vs_uniform": float(mean_mase),
                "balance_both_score": float(balance_score),
            }
        )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            float(item["balance_both_score"]),
            float(item["mean_relative_crps_gain_vs_uniform"]),
            float(item["mean_relative_mase_gain_vs_uniform"]),
            -len(str(item["scheduler_variant_key"])),
        ),
    )


def _prep_summary(cli_args: argparse.Namespace) -> Dict[str, Any]:
    forecast_datasets = parse_forecast_datasets(str(cli_args.forecast_datasets))
    lob_datasets = parse_lob_datasets(str(cli_args.lob_datasets))
    solver_names = parse_csv(str(cli_args.solver_names))
    target_nfe_values = parse_int_csv(str(cli_args.target_nfe_values))
    seeds = parse_int_csv(str(cli_args.seeds))
    canonical_delta_values = parse_float_csv(str(cli_args.canonical_delta_values))
    canonical_signal_variant = str(cli_args.canonical_signal_variant).strip().lower()
    canonical_signal_trace_key = _signal_trace_key_for_variant(canonical_signal_variant)
    baseline_scheduler_names = _parse_baseline_scheduler_names(str(cli_args.baseline_scheduler_names))
    dataset_count = int(len(forecast_datasets) + len(lob_datasets))
    baseline_only = bool(getattr(cli_args, "baseline_only", False))
    tvd_only_recovery = bool(getattr(cli_args, "tvd_only_recovery", False))
    forecast_scheduler_ablation = bool(getattr(cli_args, "forecast_scheduler_ablation", False))
    matched_tvd_uniform_recovery = bool(getattr(cli_args, "matched_tvd_uniform_recovery", False))
    if matched_tvd_uniform_recovery:
        matched_datasets = _matched_recovery_forecast_datasets(cli_args)
        matched_solver_names = _matched_recovery_solver_names(cli_args)
        trace_samples = int(cli_args.matched_recovery_calibration_trace_samples)
        variant_key_filter = _matched_recovery_variant_key_filter(cli_args)
        variant_keys = [
            UNIFORM_SCHEDULER_KEY,
            *[
                str(case["scheduler_variant_key"])
                for case in _matched_tvd_uniform_variant_cases(
                    canonical_delta=float(canonical_delta_values[0]),
                    calibration_trace_samples=int(trace_samples),
                )
                if variant_key_filter is None or str(case["scheduler_variant_key"]) in variant_key_filter
            ],
        ]
        validation_cases_per_cell = int(1 + len(canonical_delta_values) * (len(variant_keys) - 1))
        validation_cells = int(len(matched_datasets) * len(matched_solver_names) * len(target_nfe_values))
        locked_cells = int(len(matched_datasets) * len(seeds) * len(matched_solver_names) * len(target_nfe_values))
        return {
            "runner_mode": "matched_tvd_uniform_recovery",
            "forecast_datasets": list(matched_datasets),
            "lob_datasets": [],
            "solver_names": list(matched_solver_names),
            "target_nfe_values": [int(x) for x in target_nfe_values],
            "validation_seed": 0,
            "locked_test_seeds": [int(x) for x in seeds],
            "canonical_delta_values": [float(x) for x in canonical_delta_values],
            "num_eval_samples": int(cli_args.num_eval_samples),
            "calibration_trace_samples": int(trace_samples),
            "mass_cap_multipliers": [float(x) for x in DEFAULT_MATCHED_TVD_UNIFORM_CAP_MULTIPLIERS],
            "mass_bands": [
                {"floor": float(floor), "cap": float(cap)}
                for floor, cap in DEFAULT_MATCHED_TVD_UNIFORM_MASS_BANDS
            ],
            "grid_uniform_blend_values": [float(x) for x in DEFAULT_MATCHED_TVD_UNIFORM_GRID_BLEND_VALUES],
            "no_rstar_mass_blend_values": [float(x) for x in DEFAULT_MATCHED_TVD_UNIFORM_NO_RSTAR_BLEND_VALUES],
            "matched_recovery_variant_keys_filter": None if variant_key_filter is None else sorted(variant_key_filter),
            "scheduler_variant_keys": variant_keys,
            "validation_expected_rows": int(validation_cells * validation_cases_per_cell),
            "locked_test_expected_rows_max": int(locked_cells * 2),
            "selection_rule": {
                "objective": "maximize mean paired relative_crps_gain_vs_uniform",
                "guardrails": [
                    "mean relative CRPS gain must be non-negative",
                    "mean relative MASE gain must be non-negative",
                    "wind rows with runtime_grid_q50 < 0.25 and negative MASE gain are rejected",
                ],
                "tie_breaker": "within CRPS tolerance, prefer more uniform shapes: smaller cap, higher floor, higher grid blend, higher mass blend, higher temperature, then canonical",
            },
        }
    if tvd_only_recovery:
        recovery_solver_names = _resolved_recovery_solver_names(cli_args)
        selected_deltas = _resolved_forecast_selected_deltas(cli_args)
        wind_delta = _selected_forecast_delta(selected_deltas, DEFAULT_TVD_RECOVERY_WIND_DATASET)
        stage_a_cases = _tvd_only_recovery_stage_a_cases(cli_args, canonical_delta=float(wind_delta))
        stage_b_cases = _tvd_only_recovery_stage_b_cases(canonical_delta=float(wind_delta))
        base_cells_stage_a = int(len(recovery_solver_names) * len(target_nfe_values))
        return {
            "runner_mode": "tvd_only_recovery",
            "comparator_rows_csv": str(getattr(cli_args, "comparator_rows_csv", "") or ""),
            "note": "Post-hoc test recovery only; do not treat as fair final evidence.",
            "selected_deltas": {
                DEFAULT_TVD_RECOVERY_WIND_DATASET: float(wind_delta),
                DEFAULT_TVD_RECOVERY_SAN_FRANCISCO_DATASET: _selected_forecast_delta(
                    selected_deltas,
                    DEFAULT_TVD_RECOVERY_SAN_FRANCISCO_DATASET,
                ),
            },
            "stage_a": {
                "datasets": [DEFAULT_TVD_RECOVERY_WIND_DATASET],
                "seeds": [0],
                "solver_names": list(recovery_solver_names),
                "target_nfe_values": [int(x) for x in target_nfe_values],
                "scheduler_variant_keys": [str(case["scheduler_variant_key"]) for case in stage_a_cases],
                "uniform_comparator_expected_rows": int(base_cells_stage_a),
                "tvd_expected_rows_total": int(base_cells_stage_a * len(stage_a_cases)),
            },
            "stage_b": {
                "datasets": [DEFAULT_TVD_RECOVERY_WIND_DATASET],
                "seeds": [0],
                "solver_names": list(recovery_solver_names),
                "target_nfe_values": [int(x) for x in target_nfe_values],
                "scheduler_variant_keys": [str(case["scheduler_variant_key"]) for case in stage_b_cases],
                "uniform_comparator_expected_rows": int(base_cells_stage_a),
                "tvd_expected_rows_total": int(base_cells_stage_a * len(stage_b_cases)),
                "reference_macro_factor": 4.0,
            },
            "stage_c": {
                "datasets": [
                    DEFAULT_TVD_RECOVERY_WIND_DATASET,
                    DEFAULT_TVD_RECOVERY_SAN_FRANCISCO_DATASET,
                ],
                "seed_plan": {
                    DEFAULT_TVD_RECOVERY_WIND_DATASET: [1, 2],
                    DEFAULT_TVD_RECOVERY_SAN_FRANCISCO_DATASET: [0],
                },
                "solver_names": list(recovery_solver_names),
                "target_nfe_values": [int(x) for x in target_nfe_values],
                "uniform_comparator_expected_rows": int(3 * len(recovery_solver_names) * len(target_nfe_values)),
                "tvd_expected_rows_total": int(3 * len(recovery_solver_names) * len(target_nfe_values)),
            },
        }
    if forecast_scheduler_ablation:
        ablation_solver_names = parse_csv(str(cli_args.ablation_solver_names))
        cases_by_dataset = _forecast_scheduler_ablation_cases_by_dataset(cli_args)
        variant_keys = [
            str(case["scheduler_variant_key"])
            for case in cases_by_dataset[str(forecast_datasets[0])]
        ] if forecast_datasets else []
        ablation_base_cells = int(len(forecast_datasets) * len(ablation_solver_names) * len(target_nfe_values))
        return {
            "runner_mode": "forecast_scheduler_ablation",
            "forecast_datasets": list(forecast_datasets),
            "solver_names": list(ablation_solver_names),
            "target_nfe_values": [int(x) for x in target_nfe_values],
            "ablation_seed": int(cli_args.ablation_seed),
            "selected_deltas": _resolved_forecast_selected_deltas(cli_args),
            "scheduler_variant_keys": variant_keys,
            "ablation_base_cells": int(ablation_base_cells),
            "locked_test_expected_rows_total": int(
                sum(len(cases_by_dataset[dataset]) for dataset in forecast_datasets) * len(ablation_solver_names) * len(target_nfe_values)
            ),
            "default_scheduler_knobs": {
                "uniform_blend": float(cli_args.uniform_blend),
                "gibbs_temperature": float(cli_args.gibbs_temperature),
                "reference_macro_factor": float(cli_args.reference_macro_factor),
                "r_star_multiplier": float(cli_args.r_star_multiplier),
                "mass_floor_multiplier": float(cli_args.mass_floor_multiplier),
                "mass_cap_multiplier": float(cli_args.mass_cap_multiplier),
                "grid_uniform_blend": float(cli_args.grid_uniform_blend),
                "hardness_tilt_gamma": float(cli_args.hardness_tilt_gamma),
                "calibration_trace_samples": int(cli_args.calibration_trace_samples),
            },
        }
    validation_expected = 0 if baseline_only else dataset_count * len(solver_names) * len(target_nfe_values) * len(canonical_delta_values)
    locked_scheduler_keys = baseline_scheduler_names if baseline_only else [UNIFORM_SCHEDULER_KEY, CANONICAL_TVD_SCHEDULER_KEY]
    locked_per_seed = dataset_count * len(solver_names) * len(target_nfe_values) * len(locked_scheduler_keys)
    return {
        "runner_mode": "baseline_only" if baseline_only else "canonical_only",
        "scheduler_keys": list(locked_scheduler_keys),
        "baseline_scheduler_keys": list(baseline_scheduler_names),
        "canonical_signal_variant": canonical_signal_variant,
        "signal_trace_key": canonical_signal_trace_key,
        "canonical_delta_values": [float(x) for x in canonical_delta_values],
        "forecast_datasets": list(forecast_datasets),
        "lob_datasets": list(lob_datasets),
        "solver_names": list(solver_names),
        "target_nfe_values": [int(x) for x in target_nfe_values],
        "seeds": [int(x) for x in seeds],
        "otflow_train_steps": int(cli_args.otflow_train_steps),
        "default_scheduler_knobs": {
            "uniform_blend": float(cli_args.uniform_blend),
            "gibbs_temperature": float(cli_args.gibbs_temperature),
            "reference_macro_factor": float(cli_args.reference_macro_factor),
            "r_star_multiplier": float(cli_args.r_star_multiplier),
            "mass_floor_multiplier": float(cli_args.mass_floor_multiplier),
            "mass_cap_multiplier": float(cli_args.mass_cap_multiplier),
            "grid_uniform_blend": float(cli_args.grid_uniform_blend),
            "hardness_tilt_gamma": float(cli_args.hardness_tilt_gamma),
            "calibration_trace_samples": int(cli_args.calibration_trace_samples),
        },
        "validation_expected_rows": int(validation_expected),
        "locked_test_expected_rows_per_seed": int(locked_per_seed),
        "locked_test_expected_rows_total": int(locked_per_seed * len(seeds)),
    }


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run the canonical TVD study or the fixed-baseline-only matrix.")
    ap.add_argument("--out_root", type=str, default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--dataset_root", type=str, default=str(project_paper_dataset_root()))
    ap.add_argument("--shared_backbone_root", type=str, default=str(DEFAULT_SHARED_BACKBONE_ROOT))
    ap.add_argument("--backbone_manifest", type=str, default="")
    ap.add_argument("--otflow_train_steps", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--forecast_datasets", type=str, default=",".join(DEFAULT_FORECAST_DATASETS))
    ap.add_argument("--lob_datasets", type=str, default=",".join(DEFAULT_LOB_DATASETS))
    ap.add_argument("--cryptos_path", type=str, default="")
    ap.add_argument("--es_path", type=str, default="")
    ap.add_argument("--sleep_edf_path", type=str, default="")
    ap.add_argument("--solver_names", type=str, default=",".join(ALL_SOLVER_ORDER))
    ap.add_argument("--target_nfe_values", type=str, default=",".join(str(x) for x in DEFAULT_TARGET_NFE_VALUES))
    ap.add_argument("--baseline_scheduler_names", type=str, default=",".join(DEFAULT_BASELINE_SCHEDULERS))
    ap.add_argument(
        "--canonical_delta_values",
        type=str,
        default=",".join(f"{float(x):.2f}" for x in DEFAULT_CANONICAL_DELTA_VALUES),
    )
    ap.add_argument(
        "--canonical_signal_variant",
        type=str,
        choices=list(DEFAULT_TVD_RECOVERY_SIGNAL_VARIANTS),
        default=DEFAULT_CANONICAL_SIGNAL_VARIANT,
    )
    ap.add_argument("--uniform_blend", type=float, default=0.0)
    ap.add_argument("--gibbs_temperature", type=float, default=1.0)
    ap.add_argument("--reference_macro_factor", type=float, default=4.0)
    ap.add_argument("--r_star_multiplier", type=float, default=1.0)
    ap.add_argument("--mass_floor_multiplier", type=float, default=0.0)
    ap.add_argument("--mass_cap_multiplier", type=float, default=0.0)
    ap.add_argument("--grid_uniform_blend", type=float, default=0.0)
    ap.add_argument("--hardness_tilt_gamma", type=float, default=0.0)
    ap.add_argument("--seeds", type=str, default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--reference_macro_steps", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dataset_seed", type=int, default=0)
    ap.add_argument("--num_eval_samples", type=int, default=5)
    ap.add_argument("--calibration_trace_samples", type=int, default=1)
    ap.add_argument("--eval_horizon", type=int, default=0)
    ap.add_argument("--eval_windows_val", type=int, default=0)
    ap.add_argument("--eval_windows_test", type=int, default=0)
    ap.add_argument("--sigma_eps", type=float, default=1e-6)
    ap.add_argument("--hidden_dim", type=int, default=160)
    ap.add_argument("--fu_net_layers", type=int, default=3)
    ap.add_argument("--fu_net_heads", type=int, default=4)
    ap.add_argument("--rollout_mode", type=str, default="non_ar")
    ap.add_argument("--future_block_len", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--row_jsonl_name", type=str, default="rows.jsonl")
    ap.add_argument("--row_csv_name", type=str, default="rows.csv")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no_resume", dest="resume", action="store_false")
    ap.add_argument("--baseline_only", action="store_true", default=False)
    ap.add_argument("--diagnose_locked_forecast_only", action="store_true", default=False)
    ap.add_argument("--forecast_scheduler_ablation", action="store_true", default=False)
    ap.add_argument("--tvd_only_recovery", action="store_true", default=False)
    ap.add_argument("--matched_tvd_uniform_recovery", action="store_true", default=False)
    ap.add_argument("--comparator_rows_csv", type=str, default="")
    ap.add_argument("--ablation_seed", type=int, default=0)
    ap.add_argument("--ablation_solver_names", type=str, default=",".join(DEFAULT_FIXED_SOLVER_NAMES))
    ap.add_argument("--ablation_signal_variants", type=str, default=",".join(DEFAULT_TVD_RECOVERY_SIGNAL_VARIANTS))
    ap.add_argument("--forecast_selected_deltas", type=str, default="")
    ap.add_argument("--forecast_selected_deltas_json", type=str, default="")
    ap.add_argument("--ablation_uniform_blend_values", type=str, default="0.15,0.25")
    ap.add_argument("--ablation_gibbs_temperature_values", type=str, default="1.25,1.5")
    ap.add_argument("--ablation_reference_macro_factors", type=str, default="4,8")
    ap.add_argument("--ablation_r_star_multipliers", type=str, default="1.0,1.5")
    ap.add_argument("--ablation_hardness_tilt_gamma_values", type=str, default="")
    ap.add_argument(
        "--matched_recovery_forecast_datasets",
        type=str,
        default=",".join(DEFAULT_MATCHED_TVD_UNIFORM_FORECAST_DATASETS),
    )
    ap.add_argument(
        "--matched_recovery_solver_names",
        type=str,
        default=",".join(DEFAULT_MATCHED_TVD_UNIFORM_SOLVER_NAMES),
    )
    ap.add_argument(
        "--matched_recovery_calibration_trace_samples",
        type=int,
        default=DEFAULT_MATCHED_TVD_UNIFORM_TRACE_SAMPLES,
    )
    ap.add_argument("--matched_recovery_variant_keys", type=str, default="")
    ap.add_argument("--allow_execute", action="store_true", default=False)
    return ap


def _realized_nfe_for_solver(solver_key: str, runtime_nfe: int) -> int:
    multiplier = {
        "euler": 1,
        "heun": 2,
        "midpoint_rk2": 2,
        "dpmpp2m": 1,
    }[str(solver_key)]
    return int(runtime_nfe) * int(multiplier)


def _run_forecast_phase(
    cli_args: argparse.Namespace,
    *,
    row_recorder: Mapping[str, Any],
    split_phase: str,
    seeds: Sequence[int],
    scheduler_cases_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    external_uniform_rows_by_key: Optional[Mapping[Tuple[Any, ...], Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    dataset_root = Path(str(cli_args.dataset_root)).resolve()
    shared_backbone_root = Path(str(cli_args.shared_backbone_root)).resolve()
    device = torch.device(str(cli_args.device))
    dataset_cache: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for dataset_idx, dataset in enumerate(parse_forecast_datasets(str(cli_args.forecast_datasets))):
        if dataset not in dataset_cache:
            dataset_cache[dataset] = load_forecast_checkpoint_splits(
                cli_args=cli_args,
                dataset_root=dataset_root,
                shared_backbone_root=shared_backbone_root,
                dataset=dataset,
                device=device,
            )
        checkpoint = dataset_cache[dataset]
        model = checkpoint["model"]
        cfg = checkpoint["cfg"]
        splits = checkpoint["splits"]
        eval_ds = splits["val"] if str(split_phase) == VALIDATION_PHASE else splits["test"]
        calibration_ds = splits["val"]

        for seed in seeds:
            for target_nfe_idx, target_nfe in enumerate(parse_int_csv(str(cli_args.target_nfe_values))):
                for solver_idx, solver_key in enumerate(parse_csv(str(cli_args.solver_names))):
                    runtime_nfe = solver_macro_steps(str(solver_key), int(target_nfe))
                    scheduler_cases = [
                        _resolve_scheduler_case(
                            cli_args,
                            case,
                            runtime_nfe=int(runtime_nfe),
                        )
                        for case in scheduler_cases_by_dataset[str(dataset)]
                    ]
                    existing_rows, pending_cases = _pending_scheduler_cases(
                        row_recorder,
                        benchmark_family=FORECAST_FAMILY,
                        split_phase=str(split_phase),
                        seed=int(seed),
                        dataset=str(dataset),
                        target_nfe=int(target_nfe),
                        solver_key=str(solver_key),
                        scheduler_cases=scheduler_cases,
                    )
                    rows.extend(existing_rows)
                    cell_uniform_metrics: Optional[Mapping[str, Any]] = None
                    for existing_row in existing_rows:
                        if str(existing_row.get("scheduler_key")) == UNIFORM_SCHEDULER_KEY:
                            cell_uniform_metrics = existing_row
                    if not pending_cases:
                        continue
                    if external_uniform_rows_by_key is not None:
                        comparator_key = _comparator_match_key_for_context(
                            benchmark_family=FORECAST_FAMILY,
                            split_phase=str(split_phase),
                            seed=int(seed),
                            dataset=str(dataset),
                            checkpoint=checkpoint,
                            target_nfe=int(target_nfe),
                            solver_key=str(solver_key),
                        )
                        if comparator_key not in external_uniform_rows_by_key:
                            raise ValueError(
                                "Missing external uniform comparator row for "
                                f"dataset={dataset}, seed={seed}, solver={solver_key}, target_nfe={target_nfe}, "
                                f"checkpoint_id={checkpoint['checkpoint_id']}"
                            )
                    calibration_seed = int(seed) + 10_000 * dataset_idx + 100 * target_nfe_idx + solver_idx
                    for reference_macro_steps, grouped_cases in sorted(
                        _group_pending_cases_by_reference_macro_steps(pending_cases).items()
                    ):
                        needs_canonical = any(
                            _is_canonical_tvd_scheduler(str(case["scheduler_key"]))
                            for case in grouped_cases
                        )
                        calibration = None
                        if needs_canonical:
                            calibration_trace_samples = max(
                                int(case.get("calibration_trace_samples", 1))
                                for case in grouped_cases
                                if _is_canonical_tvd_scheduler(str(case["scheduler_key"]))
                            )
                            calibration = collect_forecast_calibration(
                                model,
                                calibration_ds,
                                cfg,
                                macro_steps=int(reference_macro_steps),
                                solver_name=str(SOLVER_RUNTIME_NAMES[str(solver_key)]),
                                seed=int(calibration_seed),
                                calibration_trace_samples=int(calibration_trace_samples),
                            )

                        for case in grouped_cases:
                            scheduler_key = str(case["scheduler_key"])
                            scheduler_variant_key = str(case["scheduler_variant_key"])
                            scheduler_variant_name = str(case["scheduler_variant_name"])
                            canonical_delta = case.get("canonical_delta")

                            if not _is_canonical_tvd_scheduler(scheduler_key):
                                fixed_grid = build_schedule_grid(scheduler_key, int(runtime_nfe))
                                if fixed_grid is None:
                                    raise ValueError(f"Unable to build fixed grid for scheduler={scheduler_key}")
                                time_grid = tuple(float(x) for x in fixed_grid)
                                details = {
                                    "time_grid": [float(x) for x in time_grid],
                                    "reference_time_alignment": schedule_time_alignment(scheduler_key),
                                    "paper_duplicate_count": 0,
                                    "reference_macro_factor": case.get("reference_macro_factor"),
                                    "uniform_blend": case.get("uniform_blend"),
                                    "gibbs_temperature": case.get("gibbs_temperature"),
                                    "r_star_multiplier": case.get("r_star_multiplier"),
                                    "mass_floor_multiplier": case.get("mass_floor_multiplier"),
                                    "mass_cap_multiplier": case.get("mass_cap_multiplier"),
                                    "grid_uniform_blend": case.get("grid_uniform_blend"),
                                    "hardness_tilt_gamma": case.get("hardness_tilt_gamma"),
                                }
                                signal_key = None
                                signal_score = None
                            else:
                                assert calibration is not None
                                signal_key = str(case["signal_trace_key"])
                                calibration = _ensure_signal_trace_available(
                                    calibration,
                                    signal_trace_key=signal_key,
                                )
                                details = canonical_tvd_schedule_details(
                                    calibration,
                                    macro_steps=int(runtime_nfe),
                                    delta=float(canonical_delta),
                                    solver_order=float(solver_order_p(str(solver_key))),
                                    signal_trace_key=signal_key,
                                    uniform_blend=float(case["uniform_blend"]),
                                    gibbs_temperature=float(case["gibbs_temperature"]),
                                    reference_macro_factor=float(case["reference_macro_factor"]),
                                    r_star_multiplier=float(case["r_star_multiplier"]),
                                    mass_floor_multiplier=(
                                        None
                                        if float(case.get("mass_floor_multiplier", 0.0)) <= 0.0
                                        else float(case["mass_floor_multiplier"])
                                    ),
                                    mass_cap_multiplier=(
                                        None
                                        if float(case.get("mass_cap_multiplier", 0.0)) <= 0.0
                                        else float(case["mass_cap_multiplier"])
                                    ),
                                    grid_uniform_blend=float(case.get("grid_uniform_blend", 0.0)),
                                    hardness_tilt_gamma=float(case.get("hardness_tilt_gamma", 0.0)),
                                )
                                details["calibration_trace_samples"] = int(calibration.get("calibration_trace_samples", 1))
                                time_grid = tuple(float(x) for x in details["time_grid"])
                                signal_score = _recomputed_signal_validation_spearman(
                                    calibration,
                                    signal_trace_key=signal_key,
                                    r_star_multiplier=float(case["r_star_multiplier"]),
                                )

                            eval_seed = int(seed) + 100_000 * dataset_idx + 1_000 * target_nfe_idx + solver_idx
                            metrics = evaluate_forecast_schedule(
                                model,
                                eval_ds,
                                cfg,
                                solver_name=str(SOLVER_RUNTIME_NAMES[str(solver_key)]),
                                runtime_nfe=int(runtime_nfe),
                                time_grid=time_grid,
                                num_eval_samples=int(cli_args.num_eval_samples),
                                seed=int(eval_seed),
                            )
                            if external_uniform_rows_by_key is not None:
                                comparator_key = _comparator_match_key_for_context(
                                    benchmark_family=FORECAST_FAMILY,
                                    split_phase=str(split_phase),
                                    seed=int(seed),
                                    dataset=str(dataset),
                                    checkpoint=checkpoint,
                                    target_nfe=int(target_nfe),
                                    solver_key=str(solver_key),
                                )
                                comparator = external_uniform_rows_by_key.get(comparator_key)
                                if comparator is None:
                                    raise ValueError(
                                        "Missing external uniform comparator row for "
                                        f"dataset={dataset}, seed={seed}, solver={solver_key}, target_nfe={target_nfe}, "
                                        f"checkpoint_id={checkpoint['checkpoint_id']}"
                                    )
                                metrics = dict(metrics)
                                metrics["relative_crps_gain_vs_uniform"] = _safe_relative_gain(
                                    metrics.get("crps"),
                                    comparator.get("crps"),
                                )
                                metrics["relative_mase_gain_vs_uniform"] = _safe_relative_gain(
                                    metrics.get("mase"),
                                    comparator.get("mase"),
                                )
                            elif str(scheduler_key) != UNIFORM_SCHEDULER_KEY and cell_uniform_metrics is not None:
                                metrics = dict(metrics)
                                metrics["relative_crps_gain_vs_uniform"] = _safe_relative_gain(
                                    metrics.get("crps"),
                                    cell_uniform_metrics.get("crps"),
                                )
                                metrics["relative_mase_gain_vs_uniform"] = _safe_relative_gain(
                                    metrics.get("mase"),
                                    cell_uniform_metrics.get("mase"),
                                )
                            row = _build_row(
                                benchmark_family=FORECAST_FAMILY,
                                split_phase=str(split_phase),
                                seed=int(seed),
                                dataset=str(dataset),
                                checkpoint=checkpoint,
                                target_nfe=int(target_nfe),
                                runtime_nfe=int(runtime_nfe),
                                solver_key=str(solver_key),
                                scheduler_key=str(scheduler_key),
                                scheduler_variant_key=scheduler_variant_key,
                                scheduler_variant_name=scheduler_variant_name,
                                signal_trace_key=signal_key,
                                signal_validation=signal_score,
                                canonical_delta=None if canonical_delta is None else float(canonical_delta),
                                reference_macro_steps=int(reference_macro_steps),
                                details=details,
                                metrics=metrics,
                                calibration_trace_samples=int(case.get("calibration_trace_samples", 1)),
                                scheduler_variant_tag=case.get("scheduler_variant_tag"),
                            )
                            _append_row_record(row_recorder, row)
                            rows.append(row)
                            if str(scheduler_key) == UNIFORM_SCHEDULER_KEY:
                                cell_uniform_metrics = row
    return rows


def _run_lob_phase(
    cli_args: argparse.Namespace,
    *,
    row_recorder: Mapping[str, Any],
    split_phase: str,
    seeds: Sequence[int],
    scheduler_cases_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    shared_backbone_root = Path(str(cli_args.shared_backbone_root)).resolve()
    device = torch.device(str(cli_args.device))
    dataset_cache: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for dataset_idx, dataset in enumerate(parse_lob_datasets(str(cli_args.lob_datasets))):
        if dataset not in dataset_cache:
            dataset_cache[dataset] = load_lob_checkpoint_splits(
                cli_args=cli_args,
                shared_backbone_root=shared_backbone_root,
                dataset=dataset,
                device=device,
            )
        checkpoint = dataset_cache[dataset]
        model = checkpoint["model"]
        cfg = checkpoint["cfg"]
        splits = checkpoint["splits"]
        calibration_ds = splits["val"]
        eval_ds = splits["val"] if str(split_phase) == VALIDATION_PHASE else splits["test"]
        eval_horizon = resolved_eval_horizon(cli_args, str(dataset))
        val_windows = resolved_eval_windows(cli_args, str(dataset), "val")
        eval_windows = resolved_eval_windows(cli_args, str(dataset), "val" if str(split_phase) == VALIDATION_PHASE else "test")

        for seed in seeds:
            chosen_eval_t0s = np.asarray(
                _choose_valid_windows(
                    eval_ds,
                    horizon=int(eval_horizon),
                    n_windows=int(eval_windows),
                    seed=int(seed) + 1_000 * dataset_idx,
                ),
                dtype=np.int64,
            )
            chosen_val_t0s = np.asarray(
                _choose_valid_windows(
                    calibration_ds,
                    horizon=int(eval_horizon),
                    n_windows=int(val_windows),
                    seed=int(seed) + 10_000 * dataset_idx,
                ),
                dtype=np.int64,
            )

            for target_nfe_idx, target_nfe in enumerate(parse_int_csv(str(cli_args.target_nfe_values))):
                for solver_idx, solver_key in enumerate(parse_csv(str(cli_args.solver_names))):
                    runtime_nfe = solver_macro_steps(str(solver_key), int(target_nfe))
                    scheduler_cases = [
                        _resolve_scheduler_case(
                            cli_args,
                            case,
                            runtime_nfe=int(runtime_nfe),
                        )
                        for case in scheduler_cases_by_dataset[str(dataset)]
                    ]
                    existing_rows, pending_cases = _pending_scheduler_cases(
                        row_recorder,
                        benchmark_family=LOB_FAMILY,
                        split_phase=str(split_phase),
                        seed=int(seed),
                        dataset=str(dataset),
                        target_nfe=int(target_nfe),
                        solver_key=str(solver_key),
                        scheduler_cases=scheduler_cases,
                    )
                    rows.extend(existing_rows)
                    if not pending_cases:
                        continue
                    calibration_seed = int(seed) + 100_000 * dataset_idx + 100 * target_nfe_idx + solver_idx
                    for reference_macro_steps, grouped_cases in sorted(
                        _group_pending_cases_by_reference_macro_steps(pending_cases).items()
                    ):
                        needs_canonical = any(
                            _is_canonical_tvd_scheduler(str(case["scheduler_key"]))
                            for case in grouped_cases
                        )
                        calibration = None
                        if needs_canonical:
                            calibration = _collect_calibration(
                                model,
                                calibration_ds,
                                cfg,
                                horizon=int(eval_horizon),
                                macro_steps=int(reference_macro_steps),
                                n_windows=int(len(chosen_val_t0s)),
                                seed=int(calibration_seed),
                                sigma_eps=float(cli_args.sigma_eps),
                                solver=str(SOLVER_RUNTIME_NAMES[str(solver_key)]),
                                chosen_t0s=chosen_val_t0s,
                                generation_seed_base=int(calibration_seed),
                            )

                        for case in grouped_cases:
                            scheduler_key = str(case["scheduler_key"])
                            scheduler_variant_key = str(case["scheduler_variant_key"])
                            scheduler_variant_name = str(case["scheduler_variant_name"])
                            canonical_delta = case.get("canonical_delta")

                            if not _is_canonical_tvd_scheduler(scheduler_key):
                                fixed_grid = build_schedule_grid(scheduler_key, int(runtime_nfe))
                                if fixed_grid is None:
                                    raise ValueError(f"Unable to build fixed grid for scheduler={scheduler_key}")
                                details = {
                                    "time_grid": [float(x) for x in fixed_grid],
                                    "reference_time_alignment": schedule_time_alignment(scheduler_key),
                                    "paper_duplicate_count": 0,
                                    "reference_macro_factor": case.get("reference_macro_factor"),
                                    "uniform_blend": case.get("uniform_blend"),
                                    "gibbs_temperature": case.get("gibbs_temperature"),
                                    "r_star_multiplier": case.get("r_star_multiplier"),
                                    "mass_floor_multiplier": case.get("mass_floor_multiplier"),
                                    "mass_cap_multiplier": case.get("mass_cap_multiplier"),
                                    "grid_uniform_blend": case.get("grid_uniform_blend"),
                                    "hardness_tilt_gamma": case.get("hardness_tilt_gamma"),
                                }
                                time_grid = tuple(float(x) for x in fixed_grid)
                                signal_key = None
                                signal_score = None
                            else:
                                assert calibration is not None
                                signal_key = str(case["signal_trace_key"])
                                calibration = _ensure_signal_trace_available(
                                    calibration,
                                    signal_trace_key=signal_key,
                                )
                                details = canonical_tvd_schedule_details(
                                    calibration,
                                    macro_steps=int(runtime_nfe),
                                    delta=float(canonical_delta),
                                    solver_order=float(solver_order_p(str(solver_key))),
                                    signal_trace_key=signal_key,
                                    uniform_blend=float(case["uniform_blend"]),
                                    gibbs_temperature=float(case["gibbs_temperature"]),
                                    reference_macro_factor=float(case["reference_macro_factor"]),
                                    r_star_multiplier=float(case["r_star_multiplier"]),
                                    mass_floor_multiplier=(
                                        None
                                        if float(case.get("mass_floor_multiplier", 0.0)) <= 0.0
                                        else float(case["mass_floor_multiplier"])
                                    ),
                                    mass_cap_multiplier=(
                                        None
                                        if float(case.get("mass_cap_multiplier", 0.0)) <= 0.0
                                        else float(case["mass_cap_multiplier"])
                                    ),
                                    grid_uniform_blend=float(case.get("grid_uniform_blend", 0.0)),
                                    hardness_tilt_gamma=float(case.get("hardness_tilt_gamma", 0.0)),
                                )
                                time_grid = tuple(float(x) for x in details["time_grid"])
                                signal_score = _recomputed_signal_validation_spearman(
                                    calibration,
                                    signal_trace_key=signal_key,
                                    r_star_multiplier=float(case["r_star_multiplier"]),
                                )

                            grid_spec = {
                                "grid_name": str(scheduler_variant_key),
                                "grid_kind": _scheduler_grid_kind(scheduler_key),
                                "selection_group": str(scheduler_variant_key),
                                "comparison_role": _scheduler_comparison_role(scheduler_key),
                                "solver_name": str(SOLVER_RUNTIME_NAMES[str(solver_key)]),
                                "nfe": int(runtime_nfe),
                                "time_grid": [float(x) for x in time_grid],
                            }
                            metrics_seed = int(seed) + 1_000_000 * dataset_idx + 10_000 * target_nfe_idx + solver_idx
                            result_row = run_fixed_schedule_variant(
                                model=model,
                                ds=eval_ds,
                                cfg=cfg,
                                eval_horizon=int(eval_horizon),
                                eval_windows=int(len(chosen_eval_t0s)),
                                grid_spec=grid_spec,
                                chosen_t0s=chosen_eval_t0s,
                                generation_seed_base=int(metrics_seed),
                                metrics_seed=int(metrics_seed),
                                score_main_only=False,
                            )
                            metrics = {
                                "score_main": result_row.get("score_main"),
                                "conditional_w1": result_row.get("conditional_w1"),
                                "tstr_macro_f1": result_row.get("tstr_macro_f1"),
                                "efficiency_ms_per_sample": result_row.get("efficiency_ms_per_sample"),
                                "eval_windows": int(len(chosen_eval_t0s)),
                                "realized_nfe": _realized_nfe_for_solver(str(solver_key), int(runtime_nfe)),
                            }
                            row = _build_row(
                                benchmark_family=LOB_FAMILY,
                                split_phase=str(split_phase),
                                seed=int(seed),
                                dataset=str(dataset),
                                checkpoint=checkpoint,
                                target_nfe=int(target_nfe),
                                runtime_nfe=int(runtime_nfe),
                                solver_key=str(solver_key),
                                scheduler_key=str(scheduler_key),
                                scheduler_variant_key=scheduler_variant_key,
                                scheduler_variant_name=scheduler_variant_name,
                                signal_trace_key=signal_key,
                                signal_validation=signal_score,
                                canonical_delta=None if canonical_delta is None else float(canonical_delta),
                                reference_macro_steps=int(reference_macro_steps),
                                details=details,
                                metrics=metrics,
                                scheduler_variant_tag=case.get("scheduler_variant_tag"),
                            )
                            _append_row_record(row_recorder, row)
                            rows.append(row)
    return rows


def _write_forecast_diagnosis_artifacts(out_root: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    diagnosis = _diagnose_forecast_locked_test_rows(rows)
    save_json(dict(diagnosis), str(out_root / "forecast_locked_test_scheduler_shape_diagnosis.json"))
    if any(str(row.get("scheduler_variant_key") or row.get("scheduler_key")) != str(row.get("scheduler_key")) for row in rows):
        ablation_summary = _summarize_forecast_ablation(rows)
        save_json(dict(ablation_summary), str(out_root / "forecast_scheduler_ablation_summary.json"))
        diagnosis["forecast_scheduler_ablation_summary"] = dict(ablation_summary)
    return diagnosis


def _run_diagnose_locked_forecast_only(out_root: Path, cli_args: argparse.Namespace) -> Dict[str, Any]:
    jsonl_path = out_root / str(getattr(cli_args, "row_jsonl_name", "rows.jsonl"))
    rows = list(_load_rows(jsonl_path).values())
    locked_rows = _candidate_rows_by_phase(rows, LOCKED_TEST_PHASE)
    diagnosis = _write_forecast_diagnosis_artifacts(out_root, locked_rows)
    payload = {
        "runner_mode": "diagnose_locked_forecast_only",
        "row_count": int(len(rows)),
        "locked_row_count": int(len(locked_rows)),
        "matched_pair_count": int(diagnosis.get("matched_pair_count", 0)),
    }
    save_json(dict(payload), str(out_root / "combined_summary.json"))
    return payload


def _run_forecast_scheduler_ablation(
    cli_args: argparse.Namespace,
    *,
    out_root: Path,
    prep_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    validate_execution_preflight(cli_args)
    row_recorder = _init_row_recorder(out_root, cli_args)
    ablation_args = argparse.Namespace(**vars(cli_args))
    ablation_args.solver_names = str(cli_args.ablation_solver_names)
    scheduler_cases_by_dataset = _forecast_scheduler_ablation_cases_by_dataset(cli_args)
    _run_forecast_phase(
        ablation_args,
        row_recorder=row_recorder,
        split_phase=LOCKED_TEST_PHASE,
        seeds=(int(cli_args.ablation_seed),),
        scheduler_cases_by_dataset=scheduler_cases_by_dataset,
    )
    locked_rows = _candidate_rows_by_phase(
        row_recorder["rows_by_key"].values(),
        LOCKED_TEST_PHASE,
        solver_names=parse_csv(str(cli_args.ablation_solver_names)),
    )
    diagnosis = _write_forecast_diagnosis_artifacts(out_root, locked_rows)
    summary = _summarize_forecast_ablation(locked_rows)
    combined = {
        "prep": dict(prep_payload),
        "forecast_locked_test_scheduler_shape_diagnosis": dict(diagnosis),
        "forecast_scheduler_ablation_summary": dict(summary),
    }
    save_json(dict(combined), str(out_root / "combined_summary.json"))
    return combined


def _run_matched_tvd_uniform_recovery(
    cli_args: argparse.Namespace,
    *,
    out_root: Path,
    prep_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    matched_datasets = _matched_recovery_forecast_datasets(cli_args)
    matched_solver_names = _matched_recovery_solver_names(cli_args)
    matched_args = _cloned_cli_args(
        cli_args,
        forecast_datasets=",".join(matched_datasets),
        lob_datasets="",
        solver_names=",".join(matched_solver_names),
        calibration_trace_samples=int(cli_args.matched_recovery_calibration_trace_samples),
    )
    validate_execution_preflight(matched_args)
    row_recorder = _init_row_recorder(out_root, matched_args)
    validation_cases_by_dataset = _matched_tvd_uniform_validation_cases_by_dataset(matched_args)
    _run_forecast_phase(
        matched_args,
        row_recorder=row_recorder,
        split_phase=VALIDATION_PHASE,
        seeds=(0,),
        scheduler_cases_by_dataset=validation_cases_by_dataset,
    )
    validation_rows = _candidate_rows_by_phase(
        row_recorder["rows_by_key"].values(),
        VALIDATION_PHASE,
        solver_names=matched_solver_names,
    )
    validation_selection = _select_matched_tvd_uniform_candidates(validation_rows)
    save_json(dict(validation_selection), str(out_root / "matched_tvd_uniform_validation_selection.json"))

    locked_cases_by_dataset = _matched_tvd_uniform_locked_cases_by_dataset(matched_args, validation_selection)
    locked_seeds = parse_int_csv(str(cli_args.seeds))
    _run_forecast_phase(
        matched_args,
        row_recorder=row_recorder,
        split_phase=LOCKED_TEST_PHASE,
        seeds=locked_seeds,
        scheduler_cases_by_dataset=locked_cases_by_dataset,
    )
    locked_rows = _candidate_rows_by_phase(
        row_recorder["rows_by_key"].values(),
        LOCKED_TEST_PHASE,
        solver_names=matched_solver_names,
    )
    locked_diagnosis = _write_forecast_diagnosis_artifacts(out_root, locked_rows)
    combined: Dict[str, Any] = {
        "prep": dict(prep_payload),
        "validation_selection": dict(validation_selection),
        "locked_scheduler_cases_by_dataset": {
            str(dataset): [dict(case) for case in cases]
            for dataset, cases in locked_cases_by_dataset.items()
        },
        "forecast_locked_test_scheduler_shape_diagnosis": dict(locked_diagnosis),
    }
    save_json(dict(combined), str(out_root / "combined_summary.json"))
    return combined


def _run_tvd_only_recovery(
    cli_args: argparse.Namespace,
    *,
    out_root: Path,
    prep_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    comparator_csv = str(getattr(cli_args, "comparator_rows_csv", "") or "").strip()
    if not comparator_csv:
        raise ValueError("--tvd_only_recovery requires --comparator_rows_csv")
    comparator_rows_by_key = _load_external_uniform_comparator_rows(Path(comparator_csv).expanduser().resolve())
    recovery_solver_names = _resolved_recovery_solver_names(cli_args)
    selected_deltas = _resolved_forecast_selected_deltas(cli_args)
    wind_delta = _selected_forecast_delta(selected_deltas, DEFAULT_TVD_RECOVERY_WIND_DATASET)
    san_francisco_delta = _selected_forecast_delta(selected_deltas, DEFAULT_TVD_RECOVERY_SAN_FRANCISCO_DATASET)

    preflight_args = _cloned_cli_args(
        cli_args,
        forecast_datasets=",".join(
            (
                DEFAULT_TVD_RECOVERY_WIND_DATASET,
                DEFAULT_TVD_RECOVERY_SAN_FRANCISCO_DATASET,
            )
        ),
        lob_datasets="",
        solver_names=",".join(recovery_solver_names),
    )
    validate_execution_preflight(preflight_args)

    stage_a_root = out_root / "stage_a"
    stage_a_row_recorder = _init_row_recorder(stage_a_root, cli_args)
    stage_a_cases = _tvd_only_recovery_stage_a_cases(cli_args, canonical_delta=float(wind_delta))
    stage_a_args = _cloned_cli_args(
        cli_args,
        forecast_datasets=DEFAULT_TVD_RECOVERY_WIND_DATASET,
        lob_datasets="",
        solver_names=",".join(recovery_solver_names),
    )
    _run_forecast_phase(
        stage_a_args,
        row_recorder=stage_a_row_recorder,
        split_phase=LOCKED_TEST_PHASE,
        seeds=(0,),
        scheduler_cases_by_dataset={DEFAULT_TVD_RECOVERY_WIND_DATASET: stage_a_cases},
        external_uniform_rows_by_key=comparator_rows_by_key,
    )
    stage_a_rows = _candidate_rows_by_phase(
        stage_a_row_recorder["rows_by_key"].values(),
        LOCKED_TEST_PHASE,
        solver_names=recovery_solver_names,
    )
    stage_a_rows = _enrich_rows_with_external_uniform_metrics(stage_a_rows, comparator_rows_by_key)
    stage_a_summary = _summarize_tvd_only_recovery_rows(stage_a_rows)
    stage_a_summary["stage_name"] = "stage_a"
    stage_a_summary["comparator_rows_csv"] = str(Path(comparator_csv).expanduser().resolve())
    stage_a_summary["promotion_candidate"] = _select_tvd_only_recovery_stage_a_candidate(stage_a_summary)
    stage_a_summary["promotion_criteria"] = {
        "must_improve_both_mean_relative_crps_gain_vs_uniform_and_mean_relative_mase_gain_vs_uniform_over_tvd_canonical": True,
        "mean_relative_crps_gain_vs_uniform_must_be_non_negative": True,
        "mean_relative_mase_gain_vs_uniform_must_be_non_negative": True,
        "ranking_metric": "min(mean_relative_crps_gain_vs_uniform, mean_relative_mase_gain_vs_uniform)",
    }
    save_json(dict(stage_a_summary), str(stage_a_root / "combined_summary.json"))

    combined: Dict[str, Any] = {
        "prep": dict(prep_payload),
        "comparator_rows_csv": str(Path(comparator_csv).expanduser().resolve()),
        "note": "Post-hoc test recovery only; do not treat as fair final evidence.",
        "stage_a": dict(stage_a_summary),
    }
    stage_a_candidate = stage_a_summary.get("promotion_candidate")
    if stage_a_candidate is None:
        combined["stage_b"] = {
            "skipped": True,
            "reason": "Stage A no-rstar candidate did not beat canonical TVD on both wind metrics while staying non-negative.",
        }
        combined["stage_c"] = {
            "skipped": True,
            "reason": "Stage B was not reached.",
        }
        save_json(dict(combined), str(out_root / "combined_summary.json"))
        return combined

    stage_b_root = out_root / "stage_b"
    stage_b_row_recorder = _init_row_recorder(stage_b_root, cli_args)
    stage_b_cases = _tvd_only_recovery_stage_b_cases(canonical_delta=float(wind_delta))
    stage_b_args = _cloned_cli_args(
        cli_args,
        forecast_datasets=DEFAULT_TVD_RECOVERY_WIND_DATASET,
        lob_datasets="",
        solver_names=",".join(recovery_solver_names),
    )
    _run_forecast_phase(
        stage_b_args,
        row_recorder=stage_b_row_recorder,
        split_phase=LOCKED_TEST_PHASE,
        seeds=(0,),
        scheduler_cases_by_dataset={DEFAULT_TVD_RECOVERY_WIND_DATASET: stage_b_cases},
        external_uniform_rows_by_key=comparator_rows_by_key,
    )
    stage_b_rows = _candidate_rows_by_phase(
        stage_b_row_recorder["rows_by_key"].values(),
        LOCKED_TEST_PHASE,
        solver_names=recovery_solver_names,
    )
    stage_b_rows = _enrich_rows_with_external_uniform_metrics(stage_b_rows, comparator_rows_by_key)
    stage_b_summary = _summarize_tvd_only_recovery_rows(stage_b_rows)
    stage_b_summary["stage_name"] = "stage_b"
    stage_b_summary["comparator_rows_csv"] = str(Path(comparator_csv).expanduser().resolve())
    stage_b_summary["promotion_candidate"] = _select_tvd_only_recovery_stage_b_candidate(stage_b_summary)
    stage_b_summary["promotion_criteria"] = {
        "mean_relative_crps_gain_vs_uniform_must_be_non_negative": True,
        "mean_relative_mase_gain_vs_uniform_must_be_non_negative": True,
        "ranking_metric": "min(mean_relative_crps_gain_vs_uniform, mean_relative_mase_gain_vs_uniform)",
    }
    save_json(dict(stage_b_summary), str(stage_b_root / "combined_summary.json"))
    combined["stage_b"] = dict(stage_b_summary)

    stage_b_candidate = stage_b_summary.get("promotion_candidate")
    if stage_b_candidate is None:
        combined["stage_c"] = {
            "skipped": True,
            "reason": "No Stage B no-rstar variant kept both wind metrics non-negative.",
        }
        save_json(dict(combined), str(out_root / "combined_summary.json"))
        return combined

    promoted_variant_key = str(stage_b_candidate["scheduler_variant_key"])
    case_by_variant_key = {
        str(case["scheduler_variant_key"]): dict(case)
        for case in stage_b_cases
    }
    if promoted_variant_key not in case_by_variant_key:
        raise KeyError(f"Missing promoted Stage B variant case for key={promoted_variant_key}")
    promoted_case_template = dict(case_by_variant_key[promoted_variant_key])

    stage_c_root = out_root / "stage_c"
    stage_c_row_recorder = _init_row_recorder(stage_c_root, cli_args)
    wind_stage_c_case = dict(promoted_case_template)
    wind_stage_c_case["canonical_delta"] = float(wind_delta)
    san_francisco_stage_c_case = dict(promoted_case_template)
    san_francisco_stage_c_case["canonical_delta"] = float(san_francisco_delta)
    stage_c_wind_args = _cloned_cli_args(
        cli_args,
        forecast_datasets=DEFAULT_TVD_RECOVERY_WIND_DATASET,
        lob_datasets="",
        solver_names=",".join(recovery_solver_names),
    )
    stage_c_sf_args = _cloned_cli_args(
        cli_args,
        forecast_datasets=DEFAULT_TVD_RECOVERY_SAN_FRANCISCO_DATASET,
        lob_datasets="",
        solver_names=",".join(recovery_solver_names),
    )
    _run_forecast_phase(
        stage_c_wind_args,
        row_recorder=stage_c_row_recorder,
        split_phase=LOCKED_TEST_PHASE,
        seeds=(1, 2),
        scheduler_cases_by_dataset={DEFAULT_TVD_RECOVERY_WIND_DATASET: [wind_stage_c_case]},
        external_uniform_rows_by_key=comparator_rows_by_key,
    )
    _run_forecast_phase(
        stage_c_sf_args,
        row_recorder=stage_c_row_recorder,
        split_phase=LOCKED_TEST_PHASE,
        seeds=(0,),
        scheduler_cases_by_dataset={DEFAULT_TVD_RECOVERY_SAN_FRANCISCO_DATASET: [san_francisco_stage_c_case]},
        external_uniform_rows_by_key=comparator_rows_by_key,
    )
    stage_c_rows = _candidate_rows_by_phase(
        stage_c_row_recorder["rows_by_key"].values(),
        LOCKED_TEST_PHASE,
        solver_names=recovery_solver_names,
    )
    stage_c_rows = _enrich_rows_with_external_uniform_metrics(stage_c_rows, comparator_rows_by_key)
    stage_c_summary = _summarize_tvd_only_recovery_rows(stage_c_rows)
    stage_c_summary["stage_name"] = "stage_c"
    stage_c_summary["comparator_rows_csv"] = str(Path(comparator_csv).expanduser().resolve())
    stage_c_summary["promoted_variant_key"] = promoted_variant_key
    stage_c_summary["seed_plan"] = {
        DEFAULT_TVD_RECOVERY_WIND_DATASET: [1, 2],
        DEFAULT_TVD_RECOVERY_SAN_FRANCISCO_DATASET: [0],
    }
    save_json(dict(stage_c_summary), str(stage_c_root / "combined_summary.json"))
    combined["stage_c"] = dict(stage_c_summary)

    save_json(dict(combined), str(out_root / "combined_summary.json"))
    return combined


def run_canonical_only_study(cli_args: argparse.Namespace) -> Dict[str, Any]:
    out_root = Path(str(cli_args.out_root)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    exclusive_modes = [
        bool(getattr(cli_args, "tvd_only_recovery", False)),
        bool(getattr(cli_args, "forecast_scheduler_ablation", False)),
        bool(getattr(cli_args, "matched_tvd_uniform_recovery", False)),
        bool(getattr(cli_args, "baseline_only", False)),
    ]
    if sum(1 for enabled in exclusive_modes if enabled) > 1:
        raise ValueError(
            "Choose at most one of --baseline_only, --forecast_scheduler_ablation, "
            "--tvd_only_recovery, or --matched_tvd_uniform_recovery."
        )
    prep_payload = _prep_summary(cli_args)
    if bool(getattr(cli_args, "diagnose_locked_forecast_only", False)):
        return _run_diagnose_locked_forecast_only(out_root, cli_args)
    if not bool(cli_args.allow_execute):
        save_json(dict(prep_payload), str(out_root / "combined_summary.json"))
        return dict(prep_payload)
    if bool(getattr(cli_args, "matched_tvd_uniform_recovery", False)):
        return _run_matched_tvd_uniform_recovery(
            cli_args,
            out_root=out_root,
            prep_payload=prep_payload,
        )
    if bool(getattr(cli_args, "tvd_only_recovery", False)):
        return _run_tvd_only_recovery(
            cli_args,
            out_root=out_root,
            prep_payload=prep_payload,
        )
    if bool(getattr(cli_args, "forecast_scheduler_ablation", False)):
        return _run_forecast_scheduler_ablation(
            cli_args,
            out_root=out_root,
            prep_payload=prep_payload,
        )

    validate_execution_preflight(cli_args)
    row_recorder = _init_row_recorder(out_root, cli_args)
    locked_seeds = parse_int_csv(str(cli_args.seeds))
    all_datasets = list(parse_forecast_datasets(str(cli_args.forecast_datasets))) + list(
        parse_lob_datasets(str(cli_args.lob_datasets))
    )

    if bool(getattr(cli_args, "baseline_only", False)):
        baseline_scheduler_names = _parse_baseline_scheduler_names(str(cli_args.baseline_scheduler_names))
        delta_selection = {
            "baseline_only": True,
            "selected_deltas": {},
            "datasets": {},
        }
        save_json(dict(delta_selection), str(out_root / "delta_selection_summary.json"))
        locked_scheduler_cases = {
            dataset: [{"scheduler_key": scheduler_key, "canonical_delta": None} for scheduler_key in baseline_scheduler_names]
            for dataset in all_datasets
        }
    else:
        canonical_signal_variant = str(cli_args.canonical_signal_variant).strip().lower()
        validation_scheduler_cases = {
            dataset: [
                _canonical_only_tvd_case(
                    canonical_delta=float(delta_value),
                    signal_variant=canonical_signal_variant,
                )
                for delta_value in parse_float_csv(str(cli_args.canonical_delta_values))
            ]
            for dataset in all_datasets
        }

        _run_forecast_phase(
            cli_args,
            row_recorder=row_recorder,
            split_phase=VALIDATION_PHASE,
            seeds=(0,),
            scheduler_cases_by_dataset={
                dataset: validation_scheduler_cases[dataset]
                for dataset in parse_forecast_datasets(str(cli_args.forecast_datasets))
            },
        )
        _run_lob_phase(
            cli_args,
            row_recorder=row_recorder,
            split_phase=VALIDATION_PHASE,
            seeds=(0,),
            scheduler_cases_by_dataset={
                dataset: validation_scheduler_cases[dataset]
                for dataset in parse_lob_datasets(str(cli_args.lob_datasets))
            },
        )

        validation_rows = _candidate_rows_by_phase(row_recorder["rows_by_key"].values(), VALIDATION_PHASE)
        delta_selection = _select_dataset_deltas(validation_rows)
        save_json(dict(delta_selection), str(out_root / "delta_selection_summary.json"))

        selected_deltas = dict(delta_selection["selected_deltas"])
        locked_scheduler_cases = {
            dataset: [
                {"scheduler_key": UNIFORM_SCHEDULER_KEY, "canonical_delta": None},
                _canonical_only_tvd_case(
                    canonical_delta=float(selected_deltas[str(dataset)]),
                    signal_variant=canonical_signal_variant,
                ),
            ]
            for dataset in all_datasets
        }

    _run_forecast_phase(
        cli_args,
        row_recorder=row_recorder,
        split_phase=LOCKED_TEST_PHASE,
        seeds=locked_seeds,
        scheduler_cases_by_dataset={
            dataset: locked_scheduler_cases[dataset] for dataset in parse_forecast_datasets(str(cli_args.forecast_datasets))
        },
    )
    _run_lob_phase(
        cli_args,
        row_recorder=row_recorder,
        split_phase=LOCKED_TEST_PHASE,
        seeds=locked_seeds,
        scheduler_cases_by_dataset={
            dataset: locked_scheduler_cases[dataset] for dataset in parse_lob_datasets(str(cli_args.lob_datasets))
        },
    )

    locked_rows = _candidate_rows_by_phase(row_recorder["rows_by_key"].values(), LOCKED_TEST_PHASE)
    main_table_summary = _aggregate_main_table(locked_rows)
    seed_summaries = main_table_summary.pop("seed_summaries")
    save_json({"seed_summaries": seed_summaries}, str(out_root / "locked_test_seed_summary.json"))
    save_json(dict(main_table_summary), str(out_root / "main_table_summary.json"))
    diagnosis = _write_forecast_diagnosis_artifacts(out_root, locked_rows)

    combined = {
        "prep": dict(prep_payload),
        "delta_selection_summary": dict(delta_selection),
        "locked_test_seed_summary": {"seed_summaries": seed_summaries},
        "main_table_summary": dict(main_table_summary),
        "forecast_locked_test_scheduler_shape_diagnosis": dict(diagnosis),
    }
    save_json(dict(combined), str(out_root / "combined_summary.json"))
    return combined


def main() -> None:
    cli_args = build_argparser().parse_args()
    run_canonical_only_study(cli_args)


if __name__ == "__main__":
    main()
