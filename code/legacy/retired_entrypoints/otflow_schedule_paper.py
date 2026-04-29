#!/usr/bin/env python3
"""Paper-facing entrypoint for OTFlow paper prep and benchmark scaffolding."""

from __future__ import annotations

from otflow_paper_suite import build_argparser, run_suite


def main() -> None:
    run_suite(build_argparser().parse_args())


if __name__ == "__main__":
    main()
