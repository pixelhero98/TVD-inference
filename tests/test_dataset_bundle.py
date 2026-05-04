from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import diffusion_flow_inference.evaluation.diffusion_flow_time_reparameterization as runner
from diffusion_flow_inference.datasets.bundles import (
    dataset_bundle_manifest,
    extract_processed_dataset_bundle,
)


class DatasetBundleTests(unittest.TestCase):
    def _write_bundle(self, root: Path) -> Path:
        bundle = root / "processed_datasets.zip"
        with ZipFile(bundle, "w") as zf:
            zf.writestr("data/cryptos_binance_spot_monthly_1s_l10.npz", b"cryptos")
            zf.writestr("data/es_mbp_10.npz", b"es")
            zf.writestr("data/sleep_edf_3ch_100hz_stage_conditioned.npz", b"sleep")
            zf.writestr("data/sleep_edf_3ch_100hz_stage_conditioned.json", "{}")
            zf.writestr("paper_datasets/monash/electricity/manifest.json", "{}")
            zf.writestr("paper_datasets/monash/electricity/audit.json", "{}")
            zf.writestr("paper_datasets/monash/electricity/source/electricity_hourly_dataset.tsf", "@data\nseries:1,2,3")
        return bundle

    def test_extract_processed_dataset_bundle_materializes_expected_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bundle = self._write_bundle(tmp)
            result = extract_processed_dataset_bundle(bundle, tmp / "extract")
            self.assertTrue((Path(result["data_root"]) / "cryptos_binance_spot_monthly_1s_l10.npz").exists())
            self.assertTrue((Path(result["paper_dataset_root"]) / "monash" / "electricity" / "source" / "electricity_hourly_dataset.tsf").exists())
            manifest = dataset_bundle_manifest(bundle)
        self.assertEqual(manifest["file_count"], 7)
        self.assertTrue(manifest["contains_data_root"])
        self.assertTrue(manifest["contains_paper_dataset_root"])

    def test_extract_processed_dataset_bundle_rejects_unsafe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bundle = tmp / "bad.zip"
            with ZipFile(bundle, "w") as zf:
                zf.writestr("../escape.txt", "bad")
            with self.assertRaises(ValueError):
                extract_processed_dataset_bundle(bundle, tmp / "extract")

    def test_runner_dry_run_can_use_processed_dataset_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bundle = self._write_bundle(tmp)
            out_root = tmp / "out"
            extract_root = tmp / "bundle_extract"
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    str(out_root),
                    "--forecast_datasets",
                    "electricity",
                    "--lob_datasets",
                    "cryptos,es_mbp_10,sleep_edf",
                    "--dataset_bundle_zip",
                    str(bundle),
                    "--dataset_bundle_extract_root",
                    str(extract_root),
                    "--dataset_bundle_mode",
                    "extract",
                ]
            )
            payload = runner.run_diffusion_flow_time_reparameterization(args)
            summary = json.loads((out_root / "combined_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["runner_mode"], "diffusion_flow_time_reparameterization")
        self.assertEqual(summary["dataset_bundle"]["file_count"], 7)
        self.assertEqual(Path(args.dataset_root), extract_root / "paper_datasets")
        self.assertEqual(Path(args.cryptos_path), extract_root / "data" / "cryptos_binance_spot_monthly_1s_l10.npz")
        self.assertEqual(Path(args.es_path), extract_root / "data" / "es_mbp_10.npz")
        self.assertEqual(Path(args.sleep_edf_path), extract_root / "data" / "sleep_edf_3ch_100hz_stage_conditioned.npz")


if __name__ == "__main__":
    unittest.main()
