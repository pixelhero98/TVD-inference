#!/usr/bin/env python3
"""Copy completed OTFlow result runs into the current project results tree."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

from otflow_paths import project_results_root


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Copy completed OTFlow result runs into this project's results directory.")
    ap.add_argument("--source_root", type=str, default="/home/yzn/work/TVD-Sampler/results")
    ap.add_argument("--dest_root", type=str, default=str(project_results_root()))
    ap.add_argument("--run_names", type=str, required=True, help="Comma-separated result directory basenames to copy.")
    ap.add_argument("--copy_logs", dest="copy_logs", action="store_true", default=True)
    ap.add_argument("--no_copy_logs", dest="copy_logs", action="store_false")
    ap.add_argument("--allow_missing_logs", dest="allow_missing_logs", action="store_true", default=True)
    ap.add_argument("--strict_logs", dest="allow_missing_logs", action="store_false")
    return ap


def _parse_names(text: str) -> List[str]:
    names = [part.strip() for part in str(text).split(",") if part.strip()]
    if not names:
        raise ValueError("At least one run name is required.")
    return names


def _copy_one(source_root: Path, dest_root: Path, run_name: str, *, copy_logs: bool, allow_missing_logs: bool) -> Dict[str, object]:
    source_dir = source_root / run_name
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source result directory: {source_dir}")
    dest_dir = dest_root / run_name
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
    copied_log = False
    if copy_logs:
        source_log = source_root / f"{run_name}.log"
        if source_log.exists():
            shutil.copy2(source_log, dest_root / source_log.name)
            copied_log = True
        elif not allow_missing_logs:
            raise FileNotFoundError(f"Missing log file for run: {source_log}")
    return {
        "run_name": run_name,
        "source_dir": str(source_dir),
        "dest_dir": str(dest_dir),
        "copied_log": bool(copy_logs and copied_log),
    }


def run_transfer(cli_args: argparse.Namespace) -> Dict[str, object]:
    source_root = Path(str(cli_args.source_root)).expanduser().resolve()
    dest_root = Path(str(cli_args.dest_root)).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    names = _parse_names(str(cli_args.run_names))
    copied = [
        _copy_one(
            source_root,
            dest_root,
            name,
            copy_logs=bool(cli_args.copy_logs),
            allow_missing_logs=bool(cli_args.allow_missing_logs),
        )
        for name in names
    ]
    payload = {
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "copied_runs": copied,
    }
    (dest_root / "transfer_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    run_transfer(build_argparser().parse_args())


if __name__ == "__main__":
    main()
