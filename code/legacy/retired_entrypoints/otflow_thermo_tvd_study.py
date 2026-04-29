from __future__ import annotations

import sys
from typing import Sequence

from otflow_retired_study import retired_study_main


def main(argv: Sequence[str] | None = None) -> int:
    return retired_study_main("otflow_thermo_tvd_study.py", argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
