from __future__ import annotations

import importlib
import random
import unittest

import numpy as np
import pandas as pd


class TabICLInputContractTests(unittest.TestCase):
    @staticmethod
    def frames():
        train = pd.DataFrame({
            "n": [1.0, np.nan, 3.0, 4.0],
            "cat": ["b", None, "a", "b"],
        })
        test = pd.DataFrame({
            "n": [np.nan, 6.0],
            "cat": ["unseen", "a"],
        })
        return train, test

    def test_tabicl_uses_same_ordinal_transform_then_category_dtype(self):
        data = importlib.import_module("phase1_driver.data")
        train, test = self.frames()
        ordinal = data.prepare_ordinal_views(train, test, ["cat"])
        tabicl = data.prepare_tabicl_views(train, test, ["cat"])
        self.assertEqual(tabicl.categorical_positions, ordinal.categorical_positions)
        np.testing.assert_allclose(
            tabicl.train.astype(float).to_numpy(), ordinal.train, rtol=0, atol=0
        )
        np.testing.assert_allclose(
            tabicl.test.astype(float).to_numpy(), ordinal.test, rtol=0, atol=0
        )
        self.assertEqual(str(tabicl.train.dtypes.iloc[0]), "category")
        self.assertEqual(float(tabicl.test.iloc[0, 0]), -1.0)
        self.assertFalse(tabicl.train.isna().any().any())
        self.assertFalse(tabicl.test.isna().any().any())


class RetryAndProbabilityContractTests(unittest.TestCase):
    def test_probability_validation_is_finite_and_closed_unit_interval(self):
        execution = importlib.import_module("phase1_driver.execution")
        np.testing.assert_array_equal(
            execution.validate_probability(np.array([0.0, 0.25, 1.0]), 3),
            np.array([0.0, 0.25, 1.0]),
        )
        for invalid in (
            np.array([0.1, np.nan]),
            np.array([-0.1, 0.5]),
            np.array([0.5, 1.1]),
            np.array([[0.5], [0.6]]),
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid probability"):
                execution.validate_probability(invalid, 2)

    def test_retry_validates_inside_try_and_clears_first_failure_on_success(self):
        execution = importlib.import_module("phase1_driver.execution")
        attempts = iter([np.array([np.nan, 0.2]), np.array([0.3, 0.7])])
        result, failure, attempt_count, error_type = execution.run_with_retry(
            lambda: next(attempts), expected_rows=2
        )
        np.testing.assert_allclose(result, [0.3, 0.7])
        self.assertEqual(failure, "")
        self.assertEqual(attempt_count, 2)
        self.assertEqual(error_type, "")

    def test_out_of_memory_is_retried_and_recorded_as_model_failure(self):
        execution = importlib.import_module("phase1_driver.execution")
        calls = 0

        def fail():
            nonlocal calls
            calls += 1
            raise RuntimeError("CUDA out of memory while allocating")

        result, failure, attempt_count, error_type = execution.run_with_retry(
            fail, expected_rows=2
        )
        self.assertIsNone(result)
        self.assertEqual(calls, 2)
        self.assertEqual(attempt_count, 2)
        self.assertEqual(error_type, "out_of_memory")
        self.assertIn("attempt 2", failure)

    def test_destroyed_cuda_context_is_process_fatal(self):
        execution = importlib.import_module("phase1_driver.execution")
        with self.assertRaisesRegex(execution.InfrastructureFailure, "context is destroyed"):
            execution.run_with_retry(
                lambda: (_ for _ in ()).throw(RuntimeError("CUDA context is destroyed")),
                expected_rows=2,
            )

    def test_weight_attestation_failure_is_nonretryable_scheduler_fatal(self):
        execution = importlib.import_module("phase1_driver.execution")
        gpu_models = importlib.import_module("phase1_driver.gpu_models")
        calls = []
        def operation():
            calls.append(1)
            raise gpu_models.WeightAttestationFailure("loaded parameter hash mismatch")
        with self.assertRaises(execution.InfrastructureFailure) as caught:
            execution._run_model_with_retry(operation, 2, 2)
        self.assertEqual(caught.exception.error_type, "weight_attestation_fatal")
        self.assertEqual(len(calls), 1)

    def test_cpu_seed_function_resets_python_and_numpy(self):
        execution = importlib.import_module("phase1_driver.execution")
        execution.seed_cpu_process()
        first = (random.random(), float(np.random.random()))
        execution.seed_cpu_process()
        second = (random.random(), float(np.random.random()))
        self.assertEqual(first, second)


class FTTransformerBatchContractTests(unittest.TestCase):
    def test_forward_slices_cover_all_rows_and_never_exceed_1024(self):
        gpu = importlib.import_module("phase1_driver.gpu_models")
        slices = gpu.forward_slices(2501, 1024)
        lengths = [stop - start for start, stop in slices]
        self.assertEqual(lengths, [1024, 1024, 453])
        self.assertEqual(slices[0], (0, 1024))
        self.assertEqual(slices[-1], (2048, 2501))

    def test_sigmoid_is_computed_in_float64(self):
        gpu = importlib.import_module("phase1_driver.gpu_models")
        got = gpu.sigmoid_float64(np.array([-100.0, 0.0, 100.0], dtype=np.float32))
        self.assertEqual(got.dtype, np.float64)
        self.assertTrue(np.isfinite(got).all())
        self.assertTrue(((got >= 0.0) & (got <= 1.0)).all())


class XGBoostD1ContractTests(unittest.TestCase):
    def test_d1_is_bitwise_registered_without_unseen_and_handles_unseen(self):
        smoke = importlib.import_module("phase1_driver.smoke")
        cpu_models = importlib.import_module("phase1_driver.cpu_models")
        root = smoke._registered_root()
        pilot = cpu_models._load(
            root / "pipeline_code" / "phase0c" / "phase1_pilot.py",
            "phase1_pilot_d1_unit_test",
        )
        features, labels, categorical = smoke.make_toy_data()
        train, y_train, test, _y_test = smoke.edge_case_split(features, labels)
        no_unseen = test.copy()
        no_unseen.loc[no_unseen["cat3"].eq("unseen_outer_level"), "cat3"] = "a0"
        registered, _fit_seconds, _predict_seconds = pilot.cpu_fit_predict(
            "xgb", train, y_train, no_unseen, categorical
        )
        d1, unseen_cells = cpu_models.xgb_d1_fit_predict(
            pilot, train, y_train, no_unseen, categorical
        )
        self.assertEqual(unseen_cells, 0)
        self.assertTrue(np.array_equal(registered, d1))
        levels = sorted(train["cat3"].dropna().unique().tolist())
        middle = levels[len(levels) // 2]
        middle_missing = no_unseen.copy()
        middle_missing.loc[middle_missing["cat3"].eq(middle), "cat3"] = levels[0]
        self.assertNotIn(middle, middle_missing["cat3"].dropna().unique())
        registered_middle, *_ = pilot.cpu_fit_predict(
            "xgb", train, y_train, middle_missing, categorical
        )
        d1_middle, middle_unseen = cpu_models.xgb_d1_fit_predict(
            pilot, train, y_train, middle_missing, categorical
        )
        self.assertEqual(middle_unseen, 0)
        self.assertTrue(np.array_equal(registered_middle, d1_middle))
        unseen_prediction, unseen_cells = cpu_models.xgb_d1_fit_predict(
            pilot, train, y_train, test, categorical
        )
        self.assertGreater(unseen_cells, 0)
        self.assertEqual(unseen_prediction.shape, (len(test),))
        self.assertTrue(np.isfinite(unseen_prediction).all())


if __name__ == "__main__":
    unittest.main()
