from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import adaptive_deterministic_refinement_followup as adrf
import otflow_canonical_only_study as canonical_only
import otflow_canonical_only_support as canonical_support
from otflow_schedule_utils import canonical_interval_masses, canonical_tvd_schedule_details
from otflow_signal_traces import (
    CANONICAL_INFO_GROWTH_ROW_KEY,
    CANONICAL_INFO_GROWTH_TRACE_KEY,
    CANONICAL_SIGNAL_TRACE_KEYS,
    EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_TRACE_KEY,
    MODEL_SIGNAL_SPECS,
    compute_canonical_info_growth_hardness_numpy,
    compute_no_rstar_info_growth_hardness_numpy,
)


class CanonicalSignalTraceTests(unittest.TestCase):
    def test_canonical_info_growth_formula_exactness(self) -> None:
        residual = np.asarray([0.1, 0.5, 1.0], dtype=np.float64)
        disagreement = np.asarray([0.2, 0.5, 1.0], dtype=np.float64)
        r_star = 0.4
        actual = compute_canonical_info_growth_hardness_numpy(residual, disagreement, r_star=r_star)
        expected = disagreement * np.log1p(residual / r_star)
        self.assertTrue(np.allclose(actual, expected, atol=1e-12, rtol=1e-12))

    def test_no_rstar_info_growth_formula_exactness(self) -> None:
        residual = np.asarray([0.1, 0.5, 1.0], dtype=np.float64)
        disagreement = np.asarray([0.2, 0.5, 1.0], dtype=np.float64)
        actual = compute_no_rstar_info_growth_hardness_numpy(residual, disagreement)
        expected = disagreement * np.log1p(residual)
        self.assertTrue(np.allclose(actual, expected, atol=1e-12, rtol=1e-12))

    def test_no_rstar_trace_is_not_exposed(self) -> None:
        self.assertIn(CANONICAL_INFO_GROWTH_TRACE_KEY, CANONICAL_SIGNAL_TRACE_KEYS)
        self.assertNotIn(EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_TRACE_KEY, CANONICAL_SIGNAL_TRACE_KEYS)

    def test_step_arrays_include_only_canonical_info_growth_trace(self) -> None:
        rows = []
        for step_idx in range(3):
            row = {
                "step_index": int(step_idx),
                "disagreement": 0.1 * float(step_idx + 1),
                "oracle_local_error": 0.2 * float(step_idx + 1),
                CANONICAL_INFO_GROWTH_ROW_KEY: 0.3 * float(step_idx + 1),
            }
            for row_key, _ in MODEL_SIGNAL_SPECS:
                row[row_key] = 0.5 * float(step_idx + 1)
            rows.append(row)
        payload = adrf._step_arrays(rows, 3)
        self.assertIn(CANONICAL_INFO_GROWTH_TRACE_KEY, payload)
        self.assertNotIn("info_growth_hardness_no_rstar_by_step", payload)

    def test_rollout_context_alignment_uses_runtime_feature_gap(self) -> None:
        x_hist = torch.zeros(1, 5, 42)
        block = torch.ones(1, 2, 39)
        future_context = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)

        aligned = adrf._append_rollout_context_features(
            block,
            x_hist=x_hist,
            future_context_seq=future_context,
            cursor=1,
            take=2,
        )

        self.assertEqual(tuple(aligned.shape), (1, 2, 42))
        self.assertTrue(torch.equal(aligned[..., -3:], future_context[:, 1:3, :3]))

        too_wide = torch.ones(1, 2, 44)
        cropped = adrf._append_rollout_context_features(
            too_wide,
            x_hist=x_hist,
            future_context_seq=future_context,
            cursor=0,
            take=2,
        )
        self.assertEqual(tuple(cropped.shape), (1, 2, 42))

        narrow_block = torch.ones(1, 2, 41)
        narrow_context = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
        padded = adrf._append_rollout_context_features(
            narrow_block,
            x_hist=x_hist,
            future_context_seq=narrow_context,
            cursor=2,
            take=2,
        )
        self.assertEqual(tuple(padded.shape), (1, 2, 42))
        self.assertTrue(torch.equal(padded[..., -1:], narrow_context[:, 2:4, :1]))
        torch.cat([x_hist, padded], dim=1)

        missing_context = adrf._append_rollout_context_features(
            narrow_block,
            x_hist=x_hist,
            future_context_seq=None,
            cursor=0,
            take=2,
        )
        self.assertEqual(tuple(missing_context.shape), (1, 2, 42))
        self.assertTrue(torch.equal(missing_context[..., -1:], torch.zeros(1, 2, 1)))
        torch.cat([x_hist, missing_context], dim=1)

    def test_forecast_calibration_emits_canonical_and_no_rstar_traces(self) -> None:
        class _DummyDataset:
            def __len__(self) -> int:
                return 2

            def __getitem__(self, index: int):
                hist = torch.full((4, 3), float(index + 1))
                return hist, None, None, None

        class _DummyModel:
            def sample_future_trace(self, hist, *, steps: int, solver: str, oracle_local_error: bool):
                del hist, solver, oracle_local_error
                grid = torch.linspace(0.0, 1.0, int(steps) + 1, dtype=torch.float32)
                disagreement = torch.tensor([[0.0, 0.4, 0.8, 1.2]], dtype=torch.float32)
                residual = torch.tensor([[0.0, 0.5, 1.0, 1.5]], dtype=torch.float32)
                oracle = torch.tensor([[0.0, 0.3, 0.6, 0.9]], dtype=torch.float32)
                return None, {
                    "time_grid": grid,
                    "disagreement": disagreement,
                    "residual_norm": residual,
                    "oracle_local_error": oracle,
                }

        cfg = SimpleNamespace(train=SimpleNamespace(device="cpu"))
        calibration = canonical_support.collect_forecast_calibration(
            _DummyModel(),
            _DummyDataset(),
            cfg,
            macro_steps=3,
            solver_name="euler",
            seed=0,
        )
        self.assertIn(CANONICAL_INFO_GROWTH_TRACE_KEY, calibration)
        self.assertIn(EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_TRACE_KEY, calibration)
        self.assertEqual(len(calibration[CANONICAL_INFO_GROWTH_TRACE_KEY]), 4)
        self.assertEqual(len(calibration[EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_TRACE_KEY]), 4)
        self.assertIn(EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_TRACE_KEY, calibration["signal_correlations_vs_oracle"])

    def test_forecast_calibration_trace_samples_one_preserves_default_schedule_signal(self) -> None:
        class _DummyDataset:
            def __len__(self) -> int:
                return 2

            def __getitem__(self, index: int):
                hist = torch.full((4, 3), float(index + 1))
                return hist, None, None, None

        class _SeededDummyModel:
            def sample_future_trace(self, hist, *, steps: int, solver: str, oracle_local_error: bool):
                del hist, solver, oracle_local_error
                offset = float(torch.initial_seed() % 997) / 997.0
                grid = torch.linspace(0.0, 1.0, int(steps) + 1, dtype=torch.float32)
                base = torch.tensor([[0.0, 0.4, 0.8, 1.2]], dtype=torch.float32)
                return None, {
                    "time_grid": grid,
                    "disagreement": base + offset,
                    "residual_norm": base + 0.5 + offset,
                    "oracle_local_error": base + 0.3 + offset,
                }

        cfg = SimpleNamespace(train=SimpleNamespace(device="cpu"))
        default = canonical_support.collect_forecast_calibration(
            _SeededDummyModel(),
            _DummyDataset(),
            cfg,
            macro_steps=3,
            solver_name="euler",
            seed=123,
        )
        explicit = canonical_support.collect_forecast_calibration(
            _SeededDummyModel(),
            _DummyDataset(),
            cfg,
            macro_steps=3,
            solver_name="euler",
            seed=123,
            calibration_trace_samples=1,
        )
        self.assertEqual(default["calibration_trace_samples"], 1)
        self.assertTrue(np.allclose(default[CANONICAL_INFO_GROWTH_TRACE_KEY], explicit[CANONICAL_INFO_GROWTH_TRACE_KEY]))

    def test_forecast_calibration_trace_samples_four_is_deterministic_and_averaged(self) -> None:
        class _DummyDataset:
            def __len__(self) -> int:
                return 2

            def __getitem__(self, index: int):
                hist = torch.full((4, 3), float(index + 1))
                return hist, None, None, None

        class _SeededDummyModel:
            def sample_future_trace(self, hist, *, steps: int, solver: str, oracle_local_error: bool):
                del hist, solver, oracle_local_error
                offset = float(torch.initial_seed() % 997) / 997.0
                grid = torch.linspace(0.0, 1.0, int(steps) + 1, dtype=torch.float32)
                base = torch.tensor([[0.0, 0.4, 0.8, 1.2]], dtype=torch.float32)
                return None, {
                    "time_grid": grid,
                    "disagreement": base + offset,
                    "residual_norm": base + 0.5 + offset,
                    "oracle_local_error": base + 0.3 + offset,
                }

        cfg = SimpleNamespace(train=SimpleNamespace(device="cpu"))
        single = canonical_support.collect_forecast_calibration(
            _SeededDummyModel(),
            _DummyDataset(),
            cfg,
            macro_steps=3,
            solver_name="euler",
            seed=123,
            calibration_trace_samples=1,
        )
        first = canonical_support.collect_forecast_calibration(
            _SeededDummyModel(),
            _DummyDataset(),
            cfg,
            macro_steps=3,
            solver_name="euler",
            seed=123,
            calibration_trace_samples=4,
        )
        second = canonical_support.collect_forecast_calibration(
            _SeededDummyModel(),
            _DummyDataset(),
            cfg,
            macro_steps=3,
            solver_name="euler",
            seed=123,
            calibration_trace_samples=4,
        )
        self.assertEqual(first["calibration_trace_samples"], 4)
        self.assertEqual(first["n_windows"], 2)
        self.assertEqual(len(first["rows"]), 8)
        self.assertTrue(np.allclose(first[CANONICAL_INFO_GROWTH_TRACE_KEY], second[CANONICAL_INFO_GROWTH_TRACE_KEY]))
        self.assertFalse(np.allclose(single[CANONICAL_INFO_GROWTH_TRACE_KEY], first[CANONICAL_INFO_GROWTH_TRACE_KEY]))


class CanonicalScheduleUtilsTests(unittest.TestCase):
    def test_canonical_interval_masses_use_log_hardness_gibbs_allocation(self) -> None:
        hardness = np.asarray([0.0, 1.0, 3.0], dtype=np.float64)
        reference_time_grid = [0.0, 0.2, 0.7, 1.0]
        payload = canonical_interval_masses(
            hardness,
            delta=0.10,
            solver_order=2.0,
            reference_time_grid=reference_time_grid,
        )
        normalized = hardness / np.mean(hardness)
        expected_potential = np.log(0.10 + normalized)
        expected_masses = np.exp(expected_potential / 2.0) * np.diff(reference_time_grid)
        self.assertTrue(np.allclose(payload["energy_potential"], expected_potential, atol=1e-12, rtol=1e-12))
        self.assertTrue(np.allclose(payload["interval_masses"], expected_masses, atol=1e-12, rtol=1e-12))

    def test_mass_cap_disabled_preserves_current_interval_masses(self) -> None:
        hardness = np.asarray([0.0, 1.0, 3.0], dtype=np.float64)
        reference_time_grid = [0.0, 0.2, 0.7, 1.0]
        baseline = canonical_interval_masses(
            hardness,
            delta=0.10,
            solver_order=2.0,
            reference_time_grid=reference_time_grid,
        )
        disabled_zero = canonical_interval_masses(
            hardness,
            delta=0.10,
            solver_order=2.0,
            reference_time_grid=reference_time_grid,
            mass_cap_multiplier=0.0,
        )
        disabled_none = canonical_interval_masses(
            hardness,
            delta=0.10,
            solver_order=2.0,
            reference_time_grid=reference_time_grid,
            mass_cap_multiplier=None,
        )
        self.assertTrue(np.allclose(baseline["interval_masses"], disabled_zero["interval_masses"], atol=1e-12, rtol=1e-12))
        self.assertTrue(np.allclose(baseline["interval_masses"], disabled_none["interval_masses"], atol=1e-12, rtol=1e-12))
        self.assertEqual(disabled_zero["mass_cap_hit_count"], 0)

    def test_zero_hardness_tilt_preserves_current_interval_masses(self) -> None:
        hardness = np.asarray([0.2, 1.0, 3.0, 0.7], dtype=np.float64)
        reference_time_grid = [0.0, 0.1, 0.4, 0.8, 1.0]
        baseline = canonical_interval_masses(
            hardness,
            delta=0.10,
            solver_order=2.0,
            reference_time_grid=reference_time_grid,
        )
        tilted_zero = canonical_interval_masses(
            hardness,
            delta=0.10,
            solver_order=2.0,
            reference_time_grid=reference_time_grid,
            hardness_tilt_gamma=0.0,
        )
        self.assertTrue(np.allclose(baseline["interval_masses"], tilted_zero["interval_masses"], atol=1e-12, rtol=1e-12))
        self.assertTrue(np.allclose(baseline["energy_potential"], tilted_zero["energy_potential"], atol=1e-12, rtol=1e-12))
        self.assertTrue(np.allclose(tilted_zero["hardness_tilt_weights"], np.ones(4), atol=1e-12, rtol=1e-12))

    def test_positive_hardness_tilt_weights_are_larger_late_than_early(self) -> None:
        payload = canonical_interval_masses(
            np.ones(4, dtype=np.float64),
            delta=0.10,
            solver_order=1.0,
            reference_time_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
            hardness_tilt_gamma=1.0,
        )
        weights = np.asarray(payload["hardness_tilt_weights"], dtype=np.float64)
        self.assertLess(float(weights[0]), 1.0)
        self.assertGreater(float(weights[-1]), 1.0)
        self.assertTrue(all(float(a) < float(b) for a, b in zip(weights, weights[1:])))

    def test_mass_cap_projection_preserves_mass_and_respects_caps(self) -> None:
        hardness = np.asarray([100.0, 0.1, 0.1, 0.1], dtype=np.float64)
        reference_time_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        payload = canonical_interval_masses(
            hardness,
            delta=0.02,
            solver_order=1.0,
            reference_time_grid=reference_time_grid,
            mass_cap_multiplier=1.5,
        )
        masses = np.asarray(payload["interval_masses"], dtype=np.float64)
        widths = np.diff(np.asarray(reference_time_grid, dtype=np.float64))
        self.assertTrue(np.all(masses >= 0.0))
        self.assertAlmostEqual(float(np.sum(masses)), 1.0, places=12)
        self.assertTrue(np.all(masses <= 1.5 * widths + 1e-12))
        self.assertGreater(int(payload["mass_cap_hit_count"]), 0)
        self.assertGreater(float(payload["mass_cap_overflow_share"]), 0.0)

    def test_mass_cap_one_is_uniform_diagnostic(self) -> None:
        hardness = np.asarray([100.0, 0.1, 0.1, 0.1], dtype=np.float64)
        reference_time_grid = [0.0, 0.2, 0.4, 0.7, 1.0]
        payload = canonical_interval_masses(
            hardness,
            delta=0.02,
            solver_order=1.0,
            reference_time_grid=reference_time_grid,
            mass_cap_multiplier=1.0,
        )
        self.assertTrue(np.allclose(payload["interval_masses"], np.diff(reference_time_grid), atol=1e-12, rtol=1e-12))

    def test_mass_band_projection_preserves_mass_and_respects_bounds(self) -> None:
        hardness = np.asarray([100.0, 0.1, 0.1, 0.1], dtype=np.float64)
        reference_time_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        payload = canonical_interval_masses(
            hardness,
            delta=0.02,
            solver_order=1.0,
            reference_time_grid=reference_time_grid,
            mass_floor_multiplier=0.5,
            mass_cap_multiplier=1.5,
        )
        masses = np.asarray(payload["interval_masses"], dtype=np.float64)
        widths = np.diff(np.asarray(reference_time_grid, dtype=np.float64))
        self.assertAlmostEqual(float(np.sum(masses)), 1.0, places=12)
        self.assertTrue(np.all(masses >= 0.5 * widths - 1e-12))
        self.assertTrue(np.all(masses <= 1.5 * widths + 1e-12))
        self.assertGreater(int(payload["mass_floor_hit_count"]), 0)
        self.assertGreater(float(payload["mass_floor_deficit_share"]), 0.0)
        self.assertGreater(int(payload["mass_cap_hit_count"]), 0)

    def test_canonical_schedule_details_are_strictly_increasing(self) -> None:
        calibration = {
            CANONICAL_INFO_GROWTH_TRACE_KEY: [0.2, 0.5, 1.0, 0.6],
            "disagreement_by_step": [0.2, 0.5, 1.0, 0.6],
            "residual_norm_by_step": [0.4, 0.7, 1.2, 0.8],
            "reference_time_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "reference_time_alignment": "left_endpoint",
            "r_star": 0.8,
        }
        details = canonical_tvd_schedule_details(
            calibration,
            macro_steps=3,
            delta=0.10,
            solver_order=2.0,
            signal_trace_key=CANONICAL_INFO_GROWTH_TRACE_KEY,
        )
        time_grid = details["time_grid"]
        self.assertEqual(time_grid[0], 0.0)
        self.assertEqual(time_grid[-1], 1.0)
        self.assertTrue(all(time_grid[idx] < time_grid[idx + 1] for idx in range(len(time_grid) - 1)))
        self.assertEqual(details["quantization_mode"], "interpolated")
        self.assertNotIn("paper_discrete_reference", details)
        self.assertNotIn("paper_guided_interpolated_reference", details)

    def test_canonical_schedule_default_knobs_preserve_current_behavior(self) -> None:
        calibration = {
            CANONICAL_INFO_GROWTH_TRACE_KEY: [0.2, 0.5, 1.0, 0.6],
            "disagreement_by_step": [0.2, 0.5, 1.0, 0.6],
            "residual_norm_by_step": [0.4, 0.7, 1.2, 0.8],
            "reference_time_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "reference_time_alignment": "left_endpoint",
            "r_star": 0.8,
        }
        baseline = canonical_tvd_schedule_details(
            calibration,
            macro_steps=3,
            delta=0.10,
            solver_order=2.0,
            signal_trace_key=CANONICAL_INFO_GROWTH_TRACE_KEY,
        )
        explicit = canonical_tvd_schedule_details(
            calibration,
            macro_steps=3,
            delta=0.10,
            solver_order=2.0,
            signal_trace_key=CANONICAL_INFO_GROWTH_TRACE_KEY,
            uniform_blend=0.0,
            gibbs_temperature=1.0,
            reference_macro_factor=4.0,
            r_star_multiplier=1.0,
            mass_floor_multiplier=0.0,
            mass_cap_multiplier=0.0,
            grid_uniform_blend=0.0,
            hardness_tilt_gamma=0.0,
        )
        self.assertTrue(np.allclose(baseline["interval_masses"], explicit["interval_masses"], atol=1e-12, rtol=1e-12))
        self.assertTrue(np.allclose(baseline["time_grid"], explicit["time_grid"], atol=1e-12, rtol=1e-12))
        self.assertAlmostEqual(float(explicit["r_star"]), 0.8, places=12)
        self.assertTrue(np.allclose(explicit["hardness_tilt_weights"], np.ones(4), atol=1e-12, rtol=1e-12))

    def test_zero_hardness_tilt_preserves_current_time_grid(self) -> None:
        calibration = {
            CANONICAL_INFO_GROWTH_TRACE_KEY: [0.2, 0.5, 1.0, 0.6],
            "disagreement_by_step": [0.2, 0.5, 1.0, 0.6],
            "residual_norm_by_step": [0.4, 0.7, 1.2, 0.8],
            "reference_time_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "reference_time_alignment": "left_endpoint",
            "r_star": 0.8,
        }
        baseline = canonical_tvd_schedule_details(
            calibration,
            macro_steps=3,
            delta=0.10,
            solver_order=2.0,
            signal_trace_key=CANONICAL_INFO_GROWTH_TRACE_KEY,
        )
        explicit_zero = canonical_tvd_schedule_details(
            calibration,
            macro_steps=3,
            delta=0.10,
            solver_order=2.0,
            signal_trace_key=CANONICAL_INFO_GROWTH_TRACE_KEY,
            hardness_tilt_gamma=0.0,
        )
        self.assertTrue(np.allclose(baseline["interval_masses"], explicit_zero["interval_masses"], atol=1e-12, rtol=1e-12))
        self.assertTrue(np.allclose(baseline["time_grid"], explicit_zero["time_grid"], atol=1e-12, rtol=1e-12))

    def test_grid_uniform_blend_interpolates_runtime_grid_only(self) -> None:
        calibration = {
            CANONICAL_INFO_GROWTH_TRACE_KEY: [0.2, 0.5, 1.0, 0.6],
            "disagreement_by_step": [0.2, 0.5, 1.0, 0.6],
            "residual_norm_by_step": [0.4, 0.7, 1.2, 0.8],
            "reference_time_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "reference_time_alignment": "left_endpoint",
            "r_star": 0.8,
        }
        baseline = canonical_tvd_schedule_details(
            calibration,
            macro_steps=4,
            delta=0.10,
            solver_order=2.0,
            signal_trace_key=CANONICAL_INFO_GROWTH_TRACE_KEY,
        )
        blended = canonical_tvd_schedule_details(
            calibration,
            macro_steps=4,
            delta=0.10,
            solver_order=2.0,
            signal_trace_key=CANONICAL_INFO_GROWTH_TRACE_KEY,
            grid_uniform_blend=0.5,
        )
        expected_grid = 0.5 * np.asarray(baseline["time_grid"], dtype=np.float64) + 0.5 * np.linspace(0.0, 1.0, 5)
        self.assertTrue(np.allclose(blended["interval_masses"], baseline["interval_masses"], atol=1e-12, rtol=1e-12))
        self.assertTrue(np.allclose(blended["time_grid"], expected_grid, atol=1e-12, rtol=1e-12))
        self.assertEqual(blended["quantization_mode"], "interpolated_grid_blend")

    def test_blended_and_temperature_scaled_schedule_remains_positive_and_monotone(self) -> None:
        calibration = {
            CANONICAL_INFO_GROWTH_TRACE_KEY: [0.2, 0.5, 1.0, 0.6],
            "disagreement_by_step": [0.2, 0.5, 1.0, 0.6],
            "residual_norm_by_step": [0.4, 0.7, 1.2, 0.8],
            "reference_time_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "reference_time_alignment": "left_endpoint",
            "r_star": 0.8,
        }
        details = canonical_tvd_schedule_details(
            calibration,
            macro_steps=3,
            delta=0.10,
            solver_order=2.0,
            signal_trace_key=CANONICAL_INFO_GROWTH_TRACE_KEY,
            uniform_blend=0.25,
            gibbs_temperature=1.5,
            reference_macro_factor=8.0,
            r_star_multiplier=1.5,
        )
        self.assertTrue(all(float(value) >= 0.0 for value in details["interval_masses"]))
        self.assertTrue(all(float(b) > float(a) for a, b in zip(details["time_grid"], details["time_grid"][1:])))
        self.assertGreater(float(details["r_star"]), 0.8)
        self.assertIsNotNone(details["interval_mass_top3_share"])

    def test_canonical_schedule_rejects_legacy_density_ceiling_argument(self) -> None:
        calibration = {
            CANONICAL_INFO_GROWTH_TRACE_KEY: [0.2, 0.5, 1.0, 0.6],
            "reference_time_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "reference_time_alignment": "left_endpoint",
        }
        with self.assertRaises(TypeError):
            canonical_tvd_schedule_details(
                calibration,
                macro_steps=3,
                delta=0.10,
                solver_order=2.0,
                signal_trace_key=CANONICAL_INFO_GROWTH_TRACE_KEY,
                density_ceiling="runtime_cap",
            )


class CanonicalOnlyStudyTests(unittest.TestCase):
    def test_delta_selection_prefers_best_mean_rank_then_metric_then_smaller_delta(self) -> None:
        rows = [
            {
                "benchmark_family": canonical_only.FORECAST_FAMILY,
                "dataset": "electricity",
                "solver_key": "euler",
                "target_nfe": 10,
                "canonical_delta": 0.02,
                "crps": 1.0,
            },
            {
                "benchmark_family": canonical_only.FORECAST_FAMILY,
                "dataset": "electricity",
                "solver_key": "euler",
                "target_nfe": 10,
                "canonical_delta": 0.05,
                "crps": 2.0,
            },
            {
                "benchmark_family": canonical_only.FORECAST_FAMILY,
                "dataset": "electricity",
                "solver_key": "heun",
                "target_nfe": 12,
                "canonical_delta": 0.02,
                "crps": 2.0,
            },
            {
                "benchmark_family": canonical_only.FORECAST_FAMILY,
                "dataset": "electricity",
                "solver_key": "heun",
                "target_nfe": 12,
                "canonical_delta": 0.05,
                "crps": 1.0,
            },
        ]
        summary = canonical_only._mean_rank_summary(rows)
        self.assertAlmostEqual(float(summary["selected_delta"]), 0.02, places=8)

    def test_locked_test_aggregation_uses_uniform_as_same_seed_baseline(self) -> None:
        rows = [
            {
                "benchmark_family": canonical_only.FORECAST_FAMILY,
                "split_phase": canonical_only.LOCKED_TEST_PHASE,
                "seed": 0,
                "dataset": "electricity",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
                "checkpoint_id": "ckpt-f",
                "target_nfe": 10,
                "solver_key": "euler",
                "experiment_scope": "main",
                "scheduler_key": "uniform",
                "mase": 2.0,
                "crps": 2.0,
            },
            {
                "benchmark_family": canonical_only.FORECAST_FAMILY,
                "split_phase": canonical_only.LOCKED_TEST_PHASE,
                "seed": 0,
                "dataset": "electricity",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
                "checkpoint_id": "ckpt-f",
                "target_nfe": 10,
                "solver_key": "euler",
                "experiment_scope": "main",
                "scheduler_key": "tvd",
                "mase": 1.5,
                "crps": 1.0,
            },
            {
                "benchmark_family": canonical_only.FORECAST_FAMILY,
                "split_phase": canonical_only.LOCKED_TEST_PHASE,
                "seed": 1,
                "dataset": "electricity",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
                "checkpoint_id": "ckpt-f",
                "target_nfe": 10,
                "solver_key": "euler",
                "experiment_scope": "main",
                "scheduler_key": "uniform",
                "mase": 2.5,
                "crps": 4.0,
            },
            {
                "benchmark_family": canonical_only.FORECAST_FAMILY,
                "split_phase": canonical_only.LOCKED_TEST_PHASE,
                "seed": 1,
                "dataset": "electricity",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
                "checkpoint_id": "ckpt-f",
                "target_nfe": 10,
                "solver_key": "euler",
                "experiment_scope": "main",
                "scheduler_key": "tvd",
                "mase": 2.0,
                "crps": 2.0,
            },
            {
                "benchmark_family": canonical_only.LOB_FAMILY,
                "split_phase": canonical_only.LOCKED_TEST_PHASE,
                "seed": 0,
                "dataset": "cryptos",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
                "checkpoint_id": "ckpt-l",
                "target_nfe": 10,
                "solver_key": "euler",
                "experiment_scope": "main",
                "scheduler_key": "uniform",
                "score_main": 1.0,
                "conditional_w1": 0.8,
                "tstr_macro_f1": 0.55,
            },
            {
                "benchmark_family": canonical_only.LOB_FAMILY,
                "split_phase": canonical_only.LOCKED_TEST_PHASE,
                "seed": 0,
                "dataset": "cryptos",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
                "checkpoint_id": "ckpt-l",
                "target_nfe": 10,
                "solver_key": "euler",
                "experiment_scope": "main",
                "scheduler_key": "tvd",
                "score_main": 0.5,
                "conditional_w1": 0.6,
                "tstr_macro_f1": 0.65,
            },
        ]
        summary = canonical_only._aggregate_main_table(rows)
        forecast_uniform = summary["families"][canonical_only.FORECAST_FAMILY]["uniform"]
        forecast_tvd = summary["families"][canonical_only.FORECAST_FAMILY]["tvd"]
        self.assertAlmostEqual(float(forecast_tvd["relative_crps_gain_vs_uniform"]["mean"]), 0.5, places=8)
        self.assertAlmostEqual(float(forecast_tvd["relative_mase_gain_vs_uniform"]["mean"]), 0.225, places=8)
        self.assertAlmostEqual(float(forecast_uniform["relative_crps_gain_vs_uniform"]["mean"]), 0.0, places=8)
        lob_tvd = summary["families"][canonical_only.LOB_FAMILY]["tvd"]
        self.assertAlmostEqual(float(lob_tvd["relative_score_gain_vs_uniform"]["mean"]), 0.5, places=8)

    def test_row_recorder_resumes_without_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cli_args = SimpleNamespace(row_jsonl_name="rows.jsonl", row_csv_name="rows.csv", resume=True)
            recorder = canonical_only._init_row_recorder(Path(tmpdir), cli_args)
            row = {
                "benchmark_family": canonical_only.FORECAST_FAMILY,
                "split_phase": canonical_only.LOCKED_TEST_PHASE,
                "seed": 0,
                "dataset": "electricity",
                "target_nfe": 10,
                "solver_key": "euler",
                "scheduler_key": "tvd",
                "row_signature": "sig",
                "row_status": "complete",
            }
            canonical_only._append_row_record(recorder, row)
            canonical_only._append_row_record(recorder, row)
            resumed = canonical_only._init_row_recorder(Path(tmpdir), cli_args)
            self.assertEqual(len(resumed["rows_by_key"]), 1)

    def test_pending_scheduler_cases_skip_complete_group_before_calibration(self) -> None:
        common = {
            "benchmark_family": canonical_only.FORECAST_FAMILY,
            "split_phase": canonical_only.VALIDATION_PHASE,
            "seed": 0,
            "dataset": "electricity",
            "target_nfe": 10,
            "solver_key": "euler",
            "scheduler_key": "tvd",
            "row_status": "complete",
        }
        scheduler_cases = [
            {
                "scheduler_key": "tvd",
                "scheduler_variant_key": "tvd",
                "scheduler_variant_name": "Canonical TVD",
                "canonical_delta": 0.02,
                "reference_macro_steps": 16,
                "scheduler_variant_tag": None,
            },
            {
                "scheduler_key": "tvd",
                "scheduler_variant_key": "tvd",
                "scheduler_variant_name": "Canonical TVD",
                "canonical_delta": 0.05,
                "reference_macro_steps": 16,
                "scheduler_variant_tag": None,
            },
        ]
        rows_by_key = {}
        for delta in (0.02, 0.05):
            row_signature = canonical_only._row_signature(
                scheduler_key="tvd",
                signal_trace_key=canonical_only.DEFAULT_SIGNAL_TRACE_KEY,
                canonical_delta=delta,
                reference_macro_steps=16,
            )
            row = dict(common)
            row["canonical_delta"] = delta
            row["row_signature"] = row_signature
            rows_by_key[
                (
                    canonical_only.FORECAST_FAMILY,
                    canonical_only.VALIDATION_PHASE,
                    0,
                    "electricity",
                    10,
                    "euler",
                    "tvd",
                    row_signature,
                )
            ] = row
        recorder = {"rows_by_key": rows_by_key}
        existing_rows, pending_cases = canonical_only._pending_scheduler_cases(
            recorder,
            benchmark_family=canonical_only.FORECAST_FAMILY,
            split_phase=canonical_only.VALIDATION_PHASE,
            seed=0,
            dataset="electricity",
            target_nfe=10,
            solver_key="euler",
            scheduler_cases=scheduler_cases,
        )
        self.assertEqual(len(existing_rows), 2)
        self.assertEqual(pending_cases, [])

        first_key = next(
            key
            for key, row in rows_by_key.items()
            if abs(float(row["canonical_delta"]) - 0.02) <= 1e-12
        )
        partial_recorder = {"rows_by_key": {first_key: rows_by_key[first_key]}}
        existing_rows, pending_cases = canonical_only._pending_scheduler_cases(
            partial_recorder,
            benchmark_family=canonical_only.FORECAST_FAMILY,
            split_phase=canonical_only.VALIDATION_PHASE,
            seed=0,
            dataset="electricity",
            target_nfe=10,
            solver_key="euler",
            scheduler_cases=scheduler_cases,
        )
        self.assertEqual(len(existing_rows), 1)
        self.assertEqual([case["canonical_delta"] for case in pending_cases], [0.05])

    def test_reference_macro_factor_four_matches_current_default(self) -> None:
        self.assertEqual(
            canonical_only.resolve_reference_macro_steps(0, 10),
            canonical_only.resolve_reference_macro_steps(0, 10, reference_macro_factor=4.0),
        )

    def test_paired_forecast_diagnosis_identifies_wind_as_negative_mase_slice(self) -> None:
        rows = [
            {
                "benchmark_family": canonical_only.FORECAST_FAMILY,
                "split_phase": canonical_only.LOCKED_TEST_PHASE,
                "row_status": "complete",
                "seed": 0,
                "dataset": "wind_farms_wo_missing",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
                "checkpoint_id": "ckpt",
                "target_nfe": 10,
                "solver_key": "heun",
                "experiment_scope": "main",
                "scheduler_key": "uniform",
                "mase": 1.0,
                "crps": 1.0,
            },
            {
                "benchmark_family": canonical_only.FORECAST_FAMILY,
                "split_phase": canonical_only.LOCKED_TEST_PHASE,
                "row_status": "complete",
                "seed": 0,
                "dataset": "wind_farms_wo_missing",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
                "checkpoint_id": "ckpt",
                "target_nfe": 10,
                "solver_key": "heun",
                "experiment_scope": "main",
                "scheduler_key": "tvd",
                "scheduler_variant_key": "tvd_canonical",
                "mase": 1.2,
                "crps": 0.98,
                "signal_validation_spearman": 0.5,
            },
            {
                "benchmark_family": canonical_only.FORECAST_FAMILY,
                "split_phase": canonical_only.LOCKED_TEST_PHASE,
                "row_status": "complete",
                "seed": 0,
                "dataset": "san_francisco_traffic",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
                "checkpoint_id": "ckpt",
                "target_nfe": 10,
                "solver_key": "euler",
                "experiment_scope": "main",
                "scheduler_key": "uniform",
                "mase": 1.0,
                "crps": 1.0,
            },
            {
                "benchmark_family": canonical_only.FORECAST_FAMILY,
                "split_phase": canonical_only.LOCKED_TEST_PHASE,
                "row_status": "complete",
                "seed": 0,
                "dataset": "san_francisco_traffic",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
                "checkpoint_id": "ckpt",
                "target_nfe": 10,
                "solver_key": "euler",
                "experiment_scope": "main",
                "scheduler_key": "tvd",
                "scheduler_variant_key": "tvd_canonical",
                "mase": 0.9,
                "crps": 0.95,
                "signal_validation_spearman": 0.4,
            },
        ]
        diagnosis = canonical_only._diagnose_forecast_locked_test_rows(rows)
        self.assertEqual(diagnosis["dominant_negative_mase_dataset"], "wind_farms_wo_missing")

    def test_promotion_requires_crps_improvement_and_non_negative_wind_mase(self) -> None:
        summary = {
            "variants": {
                "tvd_canonical": {
                    "mean_relative_crps_gain_vs_uniform": 0.01,
                },
                "tvd_uniform_blend_0p15": {
                    "mean_relative_crps_gain_vs_uniform": 0.03,
                    "by_dataset": {
                        "wind_farms_wo_missing": {"mean_relative_mase_gain_vs_uniform": 0.01},
                        "san_francisco_traffic": {
                            "mean_relative_crps_gain_vs_uniform": 0.02,
                            "mean_relative_mase_gain_vs_uniform": 0.01,
                        },
                    },
                },
                "tvd_gibbs_temperature_1p25": {
                    "mean_relative_crps_gain_vs_uniform": 0.04,
                    "by_dataset": {
                        "wind_farms_wo_missing": {"mean_relative_mase_gain_vs_uniform": -0.01},
                        "san_francisco_traffic": {
                            "mean_relative_crps_gain_vs_uniform": 0.03,
                            "mean_relative_mase_gain_vs_uniform": 0.01,
                        },
                    },
                },
            }
        }
        promoted = canonical_only._select_promoted_forecast_variant(summary)
        self.assertIsNotNone(promoted)
        assert promoted is not None
        self.assertEqual(promoted["scheduler_variant_key"], "tvd_uniform_blend_0p15")

    def test_matched_selection_preserves_no_rstar_signal_key(self) -> None:
        common = {
            "benchmark_family": canonical_only.FORECAST_FAMILY,
            "split_phase": canonical_only.VALIDATION_PHASE,
            "seed": 0,
            "dataset": "wind_farms_wo_missing",
            "backbone_name": "otflow",
            "train_steps": 20000,
            "train_budget_label": "20k",
            "checkpoint_id": "ckpt",
            "target_nfe": 10,
            "solver_key": "euler",
            "experiment_scope": "main",
            "row_status": "complete",
        }
        rows = [
            {
                **common,
                "scheduler_key": canonical_only.UNIFORM_SCHEDULER_KEY,
                "scheduler_variant_key": canonical_only.UNIFORM_SCHEDULER_KEY,
                "crps": 10.0,
                "mase": 2.0,
            },
            {
                **common,
                "scheduler_key": canonical_only.CANONICAL_TVD_SCHEDULER_KEY,
                "scheduler_variant_key": "tvd_mc4_no_rstar_blend_0p25",
                "scheduler_variant_name": "TVD MC4 no-rstar blend 0.25",
                "signal_trace_key": EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_TRACE_KEY,
                "canonical_delta": 0.05,
                "uniform_blend": 0.25,
                "calibration_trace_samples": 4,
                "runtime_grid_q50": 0.40,
                "crps": 9.0,
                "mase": 1.9,
            },
        ]
        selection = canonical_only._select_matched_tvd_uniform_candidates(rows)
        selected = selection["selected_cases"]["wind_farms_wo_missing"]
        self.assertEqual(selected["scheduler_variant_key"], "tvd_mc4_no_rstar_blend_0p25")
        self.assertEqual(selected["signal_trace_key"], EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_TRACE_KEY)

    def test_external_uniform_comparator_enrichment_uses_same_seed_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            comparator_csv = Path(tmpdir) / "rows.csv"
            comparator_csv.write_text(
                "\n".join(
                    [
                        "benchmark_family,split_phase,seed,dataset,backbone_name,train_steps,train_budget_label,checkpoint_id,target_nfe,solver_key,experiment_scope,scheduler_key,crps,mase,row_status",
                        "forecast_extrapolation,locked_test,0,wind_farms_wo_missing,otflow,20000,20k,ckpt,10,euler,main,uniform,2.0,4.0,complete",
                        "forecast_extrapolation,locked_test,1,wind_farms_wo_missing,otflow,20000,20k,ckpt,10,euler,main,uniform,5.0,10.0,complete",
                        "forecast_extrapolation,locked_test,0,wind_farms_wo_missing,otflow,20000,20k,ckpt,10,euler,main,tvd,1.0,1.0,complete",
                    ]
                ),
                encoding="utf-8",
            )
            comparators = canonical_only._load_external_uniform_comparator_rows(comparator_csv)
            rows = [
                {
                    "benchmark_family": canonical_only.FORECAST_FAMILY,
                    "split_phase": canonical_only.LOCKED_TEST_PHASE,
                    "seed": 1,
                    "dataset": "wind_farms_wo_missing",
                    "backbone_name": "otflow",
                    "train_steps": 20000,
                    "train_budget_label": "20k",
                    "checkpoint_id": "ckpt",
                    "target_nfe": 10,
                    "solver_key": "euler",
                    "experiment_scope": "main",
                    "scheduler_key": "tvd",
                    "crps": 4.0,
                    "mase": 5.0,
                }
            ]
            enriched = canonical_only._enrich_rows_with_external_uniform_metrics(rows, comparators)
            self.assertAlmostEqual(float(enriched[0]["relative_crps_gain_vs_uniform"]), 0.2, places=8)
            self.assertAlmostEqual(float(enriched[0]["relative_mase_gain_vs_uniform"]), 0.5, places=8)

    def test_run_forecast_phase_fails_fast_when_external_uniform_comparator_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cli_args = canonical_only.build_argparser().parse_args(
                [
                    "--forecast_datasets",
                    "wind_farms_wo_missing",
                    "--lob_datasets",
                    "",
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "10",
                ]
            )
            recorder = canonical_only._init_row_recorder(Path(tmpdir), cli_args)
            checkpoint = {
                "model": object(),
                "cfg": SimpleNamespace(train=SimpleNamespace(device="cpu")),
                "splits": {"val": object(), "test": object()},
                "checkpoint_id": "ckpt",
                "checkpoint_path": "/tmp/model.pt",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
            }
            with mock.patch.object(canonical_only, "load_forecast_checkpoint_splits", return_value=checkpoint):
                with self.assertRaisesRegex(ValueError, "Missing external uniform comparator row"):
                    canonical_only._run_forecast_phase(
                        cli_args,
                        row_recorder=recorder,
                        split_phase=canonical_only.LOCKED_TEST_PHASE,
                        seeds=(0,),
                        scheduler_cases_by_dataset={
                            "wind_farms_wo_missing": [
                                {
                                    "scheduler_key": "tvd",
                                    "scheduler_variant_key": "tvd_canonical",
                                    "scheduler_variant_name": "Canonical TVD",
                                    "canonical_delta": 0.05,
                                }
                            ]
                        },
                        external_uniform_rows_by_key={},
                    )

    def test_run_lob_phase_uses_case_signal_trace_key_for_canonical_tvd(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cli_args = canonical_only.build_argparser().parse_args(
                [
                    "--forecast_datasets",
                    "",
                    "--lob_datasets",
                    "cryptos",
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "10",
                    "--device",
                    "cpu",
                ]
            )
            recorder = canonical_only._init_row_recorder(Path(tmpdir), cli_args)
            checkpoint = {
                "model": object(),
                "cfg": SimpleNamespace(train=SimpleNamespace(device="cpu")),
                "splits": {"val": object(), "test": object()},
                "checkpoint_id": "ckpt",
                "checkpoint_path": "/tmp/model.pt",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
            }
            mocked_details = {
                "time_grid": [0.0, 1.0],
                "reference_time_alignment": "runtime_learned",
                "paper_duplicate_count": 0,
                "reference_macro_factor": 4.0,
                "uniform_blend": 0.0,
                "gibbs_temperature": 1.0,
                "r_star_multiplier": 1.0,
                "mass_floor_multiplier": 0.0,
                "mass_cap_multiplier": 0.0,
                "grid_uniform_blend": 0.0,
            }
            mocked_metrics = {
                "score_main": 0.7,
                "conditional_w1": 1.2,
                "tstr_macro_f1": 0.34,
                "latency_ms_per_sample": 5.0,
            }
            calibration_payload = {
                "residual_norm_by_step": [0.0, 0.3],
                "disagreement_by_step": [0.0, 0.4],
                "rows": [
                    {"step_index": 0, "residual_norm": 0.0, "disagreement": 0.0, "oracle_local_error": 0.0},
                    {"step_index": 1, "residual_norm": 0.3, "disagreement": 0.4, "oracle_local_error": 0.2},
                ],
                "signal_correlations_vs_oracle": {},
            }
            case = {
                "scheduler_key": canonical_only.CANONICAL_TVD_SCHEDULER_KEY,
                "scheduler_variant_key": "tvd_no_rstar",
                "scheduler_variant_name": "TVD no-r*",
                "canonical_delta": 0.05,
                "signal_trace_key": EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_TRACE_KEY,
            }
            with mock.patch.object(canonical_only, "load_lob_checkpoint_splits", return_value=checkpoint):
                with mock.patch.object(
                    canonical_only,
                    "resolved_eval_horizon",
                    return_value=8,
                ):
                    with mock.patch.object(
                        canonical_only,
                        "resolved_eval_windows",
                        return_value=1,
                    ):
                        with mock.patch.object(
                            canonical_only,
                            "_choose_valid_windows",
                            return_value=np.asarray([0], dtype=np.int64),
                        ):
                            with mock.patch.object(
                                canonical_only,
                                "_collect_calibration",
                                return_value=calibration_payload,
                            ):
                                with mock.patch.object(
                                    canonical_only,
                                    "canonical_tvd_schedule_details",
                                    return_value=mocked_details,
                                ) as details_mock:
                                    with mock.patch.object(
                                        canonical_only,
                                        "_recomputed_signal_validation_spearman",
                                        return_value=0.25,
                                    ) as spearman_mock:
                                        with mock.patch.object(
                                            canonical_only,
                                            "run_fixed_schedule_variant",
                                            return_value=mocked_metrics,
                                        ):
                                            rows = canonical_only._run_lob_phase(
                                                cli_args,
                                                row_recorder=recorder,
                                                split_phase=canonical_only.LOCKED_TEST_PHASE,
                                                seeds=(0,),
                                                scheduler_cases_by_dataset={"cryptos": [case]},
                                            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["scheduler_variant_key"], "tvd_no_rstar")
            self.assertEqual(rows[0]["signal_trace_key"], EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_TRACE_KEY)
            self.assertEqual(
                details_mock.call_args.kwargs["signal_trace_key"],
                EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_TRACE_KEY,
            )
            self.assertIn(
                EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_TRACE_KEY,
                details_mock.call_args.args[0],
            )
            self.assertEqual(
                spearman_mock.call_args.kwargs["signal_trace_key"],
                EXPERIMENTAL_NO_RSTAR_INFO_GROWTH_TRACE_KEY,
            )

    def test_stage_a_promotion_requires_both_metric_improvements_and_non_negative_values(self) -> None:
        summary = {
            "variants": {
                "tvd_canonical": {
                    "mean_relative_crps_gain_vs_uniform": 0.01,
                    "mean_relative_mase_gain_vs_uniform": -0.02,
                },
                "tvd_no_rstar": {
                    "mean_relative_crps_gain_vs_uniform": 0.03,
                    "mean_relative_mase_gain_vs_uniform": 0.01,
                },
            }
        }
        promoted = canonical_only._select_tvd_only_recovery_stage_a_candidate(summary)
        self.assertIsNotNone(promoted)
        assert promoted is not None
        self.assertEqual(promoted["scheduler_variant_key"], "tvd_no_rstar")

        summary["variants"]["tvd_no_rstar"]["mean_relative_mase_gain_vs_uniform"] = -0.01
        self.assertIsNone(canonical_only._select_tvd_only_recovery_stage_a_candidate(summary))

    def test_stage_b_promotion_uses_balance_both_ranking(self) -> None:
        summary = {
            "variants": {
                "tvd_no_rstar": {
                    "mean_relative_crps_gain_vs_uniform": 0.03,
                    "mean_relative_mase_gain_vs_uniform": 0.03,
                },
                "tvd_no_rstar_blend_0p15": {
                    "mean_relative_crps_gain_vs_uniform": 0.06,
                    "mean_relative_mase_gain_vs_uniform": 0.01,
                },
                "tvd_no_rstar_temp_1p25": {
                    "mean_relative_crps_gain_vs_uniform": 0.05,
                    "mean_relative_mase_gain_vs_uniform": 0.04,
                },
            }
        }
        promoted = canonical_only._select_tvd_only_recovery_stage_b_candidate(summary)
        self.assertIsNotNone(promoted)
        assert promoted is not None
        self.assertEqual(promoted["scheduler_variant_key"], "tvd_no_rstar_temp_1p25")

    def test_candidate_rows_can_ignore_retired_fixed_grid_solvers(self) -> None:
        rows = [
            {
                "split_phase": canonical_only.VALIDATION_PHASE,
                "row_status": "complete",
                "solver_key": "euler",
            },
            {
                "split_phase": canonical_only.VALIDATION_PHASE,
                "row_status": "complete",
                "solver_key": "dopri5",
            },
            {
                "split_phase": canonical_only.VALIDATION_PHASE,
                "row_status": "complete",
                "solver_key": "rk45",
            },
        ]
        filtered = canonical_only._candidate_rows_by_phase(
            rows,
            canonical_only.VALIDATION_PHASE,
            solver_names=("euler", "heun", "midpoint_rk2", "dpmpp2m"),
        )
        self.assertEqual([row["solver_key"] for row in filtered], ["euler"])


if __name__ == "__main__":
    unittest.main()
