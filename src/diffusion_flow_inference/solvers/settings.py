from __future__ import annotations

from typing import Dict, Tuple

ALL_SOLVER_ORDER: Tuple[str, ...] = ("euler", "heun", "midpoint_rk2", "dpmpp2m")
SOLVER_RUNTIME_NAMES: Dict[str, str] = {
    "euler": "euler",
    "heun": "heun",
    "midpoint_rk2": "midpoint_rk2",
    "dpmpp2m": "dpmpp2m",
}


def solver_eval_multiplier(solver_key: str) -> int:
    key = str(solver_key)
    if key in {"euler", "dpmpp2m"}:
        return 1
    if key in {"heun", "midpoint_rk2"}:
        return 2
    raise ValueError(f"Unsupported solver_key={solver_key}")


def solver_macro_steps(solver_key: str, target_nfe: int) -> int:
    if str(solver_key) in {"heun", "midpoint_rk2"}:
        if int(target_nfe) % 2 != 0:
            raise ValueError(f"{solver_key} requires an even target_nfe, got {target_nfe}")
        return int(target_nfe) // 2
    multiplier = int(solver_eval_multiplier(str(solver_key)))
    if multiplier == 1:
        return int(target_nfe)
    return max(1, int(round(float(target_nfe) / float(multiplier))))


def solver_experiment_scope(solver_key: str) -> str:
    return "solver_transfer" if str(solver_key) == "dpmpp2m" else "main"


def solver_order_p(solver_key: str) -> float:
    key = str(solver_key)
    if key == "euler":
        return 1.0
    if key in {"heun", "midpoint_rk2", "dpmpp2m"}:
        return 2.0
    raise ValueError(f"Unsupported solver order mapping for {solver_key}")


def resolve_reference_macro_steps(
    requested_macro_steps: int,
    runtime_nfe: int,
    *,
    reference_macro_factor: float = 4.0,
) -> int:
    requested = int(requested_macro_steps)
    if requested > 0:
        return requested
    factor = float(reference_macro_factor)
    if factor <= 0.0:
        raise ValueError(f"reference_macro_factor must be positive, got {reference_macro_factor}")
    return max(32, int(round(factor * int(runtime_nfe))))
