from __future__ import annotations

from typing import Sequence

REPLACEMENT_ENTRYPOINT = "otflow_canonical_only_study.py"
CANONICAL_ONLY_SUMMARY = (
    "uniform baseline + canonical TVD only "
    "(info-growth hardness, r_* scaling, no density ceiling, continuous inverse-CDF interpolation)"
)


def retired_study_message(legacy_name: str) -> str:
    return (
        f"{legacy_name} is retired. Use `{REPLACEMENT_ENTRYPOINT}` instead for "
        f"{CANONICAL_ONLY_SUMMARY}."
    )


def retired_study_main(legacy_name: str, argv: Sequence[str] | None = None) -> int:
    del argv
    raise SystemExit(retired_study_message(legacy_name))
