from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "admission.py"


def load_module():
    spec = importlib.util.spec_from_file_location("phase1_admission_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AdmissionThresholdTests(unittest.TestCase):
    def test_admission_threshold_is_strictly_less_than_one_e_minus_twelve(self):
        """Changing `< 1e-12` to `<= 1e-12` must fail this test."""
        module = load_module()
        accepted = pd.DataFrame(
            {
                "openml_id": np.arange(1, 479),
                "abs_diff_vs_preflight": np.zeros(478),
                "n_threads": np.ones(478, dtype=int),
            }
        )
        accepted.loc[477, "abs_diff_vs_preflight"] = np.nextafter(1e-12, 0.0)
        module.validate_admission(accepted)
        rejected = accepted.copy()
        rejected.loc[477, "abs_diff_vs_preflight"] = 1e-12
        with self.assertRaisesRegex(RuntimeError, "214 admission failed"):
            module.validate_admission(rejected)

    def test_admission_output_has_registered_columns_plus_preflight_difference(self):
        """Dropping a registered preflight column must fail this test."""
        module = load_module()
        expected = [
            "openml_id",
            "lr_auc",
            "abs_diff_vs_phase0c_reference",
            "tie_units",
            "cpu_model",
            "n_threads",
            "abs_diff_vs_preflight",
        ]
        self.assertEqual(module.ADMISSION_COLUMNS, expected)


class HostAssignmentTests(unittest.TestCase):
    def test_largest_first_greedy_assignment_uses_deterministic_tie_breaks(self):
        """Changing sort order or tie-breaking host selection must fail this test."""
        module = load_module()
        costs = pd.DataFrame(
            {
                "openml_id": [40, 10, 30, 20],
                "estimated_cpu_seconds": [7.0, 10.0, 8.0, 9.0],
            }
        )
        got = module.build_host_assignment(costs, ["686", "214"])
        expected = pd.DataFrame(
            {
                "openml_id": [10, 20, 30, 40],
                "host": ["214", "686", "686", "214"],
            }
        )
        pd.testing.assert_frame_equal(got.reset_index(drop=True), expected)

    def test_cost_estimate_includes_m1_and_target_only_m2_m4(self):
        """Omitting M2 or M4 from target-region assignment cost must fail this test."""
        module = load_module()
        timing_rows = []
        for model in module.CPU_MODELS:
            for stage in ("outer", "inner"):
                for n_train, p_train in ((80, 10), (160, 20), (320, 40)):
                    timing_rows.append(
                        {
                            "model": model,
                            "stage": stage,
                            "n_train": n_train,
                            "p_train": p_train,
                            "fit_seconds": 1.0,
                            "predict_seconds": 0.0,
                        }
                    )
        timing = pd.DataFrame(timing_rows)
        pool = pd.DataFrame(
            {
                "openml_id": [1, 2],
                "n": [2000, 2000],
                "p_used": [20, 20],
                "minority_rate": [0.3, 0.3],
                "auc_lr": [0.90, 0.70],
            }
        )
        eligible_sizes = {1: [100, 250, 500, 1000], 2: [100, 250, 500, 1000]}
        got = module.estimate_dataset_cpu_work(pool, timing, eligible_sizes)
        self.assertEqual(list(got.columns), ["openml_id", "estimated_cpu_seconds"])
        cost = dict(zip(got.openml_id, got.estimated_cpu_seconds, strict=True))
        self.assertGreater(cost[1], 0.0)
        self.assertGreater(cost[2], cost[1])

    def test_assignment_csv_schema_stays_exact(self):
        """Adding explanatory columns to host_assignment.csv must fail this test."""
        module = load_module()
        costs = pd.DataFrame(
            {"openml_id": [1, 2], "estimated_cpu_seconds": [2.0, 1.0]}
        )
        result = module.build_host_assignment(costs, ["686", "214"])
        self.assertEqual(list(result.columns), ["openml_id", "host"])

    def test_generate_host_assignment_reads_registered_inputs_and_writes_478_rows(self):
        """Breaking the registered-root integration must fail this test."""
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "final_pool").mkdir()
            (root / "phase0c_outputs").mkdir()
            (root / "cleaned_data").mkdir()
            pool = pd.DataFrame(
                {
                    "openml_id": np.arange(1, 479),
                    "n": np.full(478, 10),
                    "p_used": np.full(478, 4),
                    "minority_rate": np.full(478, 0.3),
                    "auc_lr": np.full(478, 0.9),
                    "excluded_reason": [""] * 478,
                }
            )
            pool.to_csv(root / "final_pool" / "datasets_stratifiers.csv", index=False)
            timing_rows = []
            for model in module.CPU_MODELS:
                for stage in ("outer", "inner"):
                    for n_train, p_train in ((8, 4), (16, 8), (32, 16)):
                        timing_rows.append(
                            {
                                "model": model,
                                "stage": stage,
                                "n_train": n_train,
                                "p_train": p_train,
                                "fit_seconds": 1.0,
                                "predict_seconds": 0.0,
                            }
                        )
            pd.DataFrame(timing_rows).to_csv(
                root / "phase0c_outputs" / "pilot_timing.csv", index=False
            )
            folds = pd.DataFrame({"fold": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]})
            for openml_id in range(1, 479):
                folds.to_parquet(root / "cleaned_data" / f"{openml_id}.parquet")
            output = root / "host_assignment.csv"
            result = module.generate_host_assignment(root, output)
            self.assertEqual(len(result), 478)
            self.assertEqual(result["openml_id"].nunique(), 478)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
