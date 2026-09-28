from __future__ import annotations

import importlib
import unittest

import numpy as np


class ToyDataContractTests(unittest.TestCase):
    def test_toy_data_shape_levels_and_missingness_are_exact(self):
        """Changing the registered toy data or its missingness must fail this test."""
        smoke = importlib.import_module("phase1_driver.smoke")
        features, labels, categorical = smoke.make_toy_data()
        self.assertEqual(features.shape, (600, 12))
        self.assertEqual(labels.shape, (600,))
        self.assertEqual(categorical, ["cat3", "cat5"])
        self.assertEqual(features["cat3"].dropna().nunique(), 3)
        self.assertEqual(features["cat5"].dropna().nunique(), 5)
        missing = features.isna().sum()
        self.assertEqual(set(missing[missing.gt(0)].index), {"x0", "x1", "cat3"})
        self.assertTrue((missing[["x0", "x1", "cat3"]] == 30).all())

    def test_disperse_noise_variant_adds_exactly_fifteen_numeric_columns(self):
        """Using a rewritten generator or wrong injection arm must fail this test."""
        smoke = importlib.import_module("phase1_driver.smoke")
        features, labels, categorical = smoke.make_toy_data()
        augmented = smoke.make_smoke_variant(features, labels, "dispn_0.85_r0")
        self.assertEqual(augmented.shape, (600, 27))
        self.assertEqual(categorical, ["cat3", "cat5"])
        injected = [column for column in augmented.columns if column.startswith("inj_")]
        self.assertEqual(len(injected), 15)
        self.assertFalse(augmented[injected].isna().any().any())

    def test_outer_test_has_missing_training_level_and_unseen_level(self):
        smoke = importlib.import_module("phase1_driver.smoke")
        features, labels, _categorical = smoke.make_toy_data()
        train, _y_train, test, _y_test = smoke.edge_case_split(features, labels)
        self.assertIn("a2", set(train["cat3"]))
        self.assertNotIn("a2", set(test["cat3"]))
        self.assertIn("unseen_outer_level", set(test["cat3"]))


class SmokeTaskContractTests(unittest.TestCase):
    def test_smoke_runs_twenty_model_variant_tasks_without_metrics(self):
        """Skipping a model/variant or logging metrics/predictions must fail this test."""
        smoke = importlib.import_module("phase1_driver.smoke")
        tasks = smoke.smoke_tasks()
        self.assertEqual(len(tasks), 20)
        self.assertEqual({task.variant for task in tasks}, {"raw", "dispn_0.85_r0"})
        self.assertEqual({task.model for task in tasks}, set(smoke.ALL_MODELS))
        allowed = {
            "variant", "model", "status", "train_shape", "test_shape",
            "categorical_positions", "categorical_parameter", "nan_free_input",
            "boundary_audit_count", "xgb_equivalence", "device", "determinism",
            "unseen_level_handling", "output_shape", "failure_reason", "network_guard",
            "driver_manifest_sha256", "loaded_parameter_sha256",
        }
        self.assertEqual(set(smoke.SmokeRecord.__dataclass_fields__), allowed)
        forbidden_fragments = ("auc", "accuracy", "metric", "score", "prediction", "proba", "seconds")
        for field in allowed:
            self.assertFalse(any(fragment in field.lower() for fragment in forbidden_fragments))

    def test_smoke_evidence_fields_exist(self):
        smoke = importlib.import_module("phase1_driver.smoke")
        fields = smoke.SmokeRecord.__dataclass_fields__
        self.assertIn("driver_manifest_sha256", fields)
        self.assertIn("loaded_parameter_sha256", fields)

    def test_smoke_input_audit_rejects_nan_before_model_fit(self):
        """Allowing EBM or foundation-model arrays with NaN must fail this test."""
        smoke = importlib.import_module("phase1_driver.smoke")
        with self.assertRaisesRegex(RuntimeError, "NaN reached model boundary"):
            smoke.audit_model_input(np.array([[1.0, np.nan]]))


if __name__ == "__main__":
    unittest.main()
