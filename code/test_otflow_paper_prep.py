from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import diffusion_flow_time_reparameterization as runner
from diffusion_flow_schedules import build_schedule_grid
from otflow_paper_registry import (
    BASELINE_SCHEDULE_KEYS,
    METHOD_KEY,
    TRANSFER_SCHEDULE_KEYS,
    paper_registry_snapshot,
    paper_schedule_specs,
)
from otflow_signal_traces import NATIVE_INFO_GROWTH_TRACE_KEY, NATIVE_SIGNAL_TRACE_KEYS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DiffusionFlowPaperPrepTests(unittest.TestCase):
    def test_registry_exposes_diffusion_flow_method_not_tvd(self) -> None:
        snapshot = paper_registry_snapshot()
        self.assertEqual(METHOD_KEY, "diffusion_flow_time_reparameterization")
        self.assertEqual(snapshot["paper_method"], "diffusion_flow_time_reparameterization")
        self.assertFalse(any(spec.comparison_role == "paper_method" and spec.key == "tvd" for spec in paper_schedule_specs()))

    def test_schedule_sets_are_exact(self) -> None:
        self.assertEqual(BASELINE_SCHEDULE_KEYS, ("uniform", "late_power_3", "ays", "gits", "ots"))
        self.assertEqual(TRANSFER_SCHEDULE_KEYS, ("ays", "gits", "ots"))

    def test_active_schedule_grids_have_endpoints(self) -> None:
        for key in BASELINE_SCHEDULE_KEYS:
            grid = build_schedule_grid(key, 4)
            self.assertIsNotNone(grid, key)
            self.assertEqual(len(grid), 5)
            self.assertAlmostEqual(grid[0], 0.0)
            self.assertAlmostEqual(grid[-1], 1.0)
            self.assertTrue(all(right > left for left, right in zip(grid, grid[1:])), key)

    def test_native_hardness_trace_is_info_growth(self) -> None:
        self.assertEqual(NATIVE_INFO_GROWTH_TRACE_KEY, "info_growth_hardness_by_step")
        self.assertIn("info_growth_hardness_by_step", NATIVE_SIGNAL_TRACE_KEYS)

    def test_runner_dry_run_writes_combined_summary(self) -> None:
        manifest = PROJECT_ROOT / "TVD-result" / "results" / "backbone_matrix" / "backbone_manifest.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--lob_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                ]
            )
            payload = runner.run_diffusion_flow_time_reparameterization(args)
            summary = json.loads((Path(tmpdir) / "combined_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["runner_mode"], "diffusion_flow_time_reparameterization")
        self.assertEqual(summary["method_key"], "diffusion_flow_time_reparameterization")
        self.assertEqual(summary["transfer_schedule_keys"], ["ays", "gits", "ots"])

    def test_active_scripts_use_new_entrypoint(self) -> None:
        scripts = [
            PROJECT_ROOT / "code" / "ops" / "isambard3" / "run_otflow_baseline_main_table_job.sh",
            PROJECT_ROOT / "code" / "ops" / "isambard3" / "run_otflow_lob_baseline_backbone_sweep_job.sh",
        ]
        for script in scripts:
            text = script.read_text(encoding="utf-8")
            self.assertIn("diffusion_flow_time_reparameterization.py", text)
            self.assertNotIn("otflow_canonical_only_study.py", text)

    def test_no_active_imports_from_archived_canonical_modules(self) -> None:
        offenders = []
        for path in (PROJECT_ROOT / "code").rglob("*.py"):
            if "__pycache__" in path.parts or "legacy" in path.parts:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if (stripped.startswith("import ") or stripped.startswith("from ")) and "otflow_canonical_only" in stripped:
                    offenders.append(str(path.relative_to(PROJECT_ROOT)))
                    break
        self.assertEqual(offenders, [])

    def test_backbone_manifest_keeps_40_ready_artifacts(self) -> None:
        manifest = PROJECT_ROOT / "TVD-result" / "results" / "backbone_matrix" / "backbone_manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(int(payload.get("ready_count", 0)), 40)
        self.assertEqual(int(payload.get("missing_count", 0)), 0)


if __name__ == "__main__":
    unittest.main()
