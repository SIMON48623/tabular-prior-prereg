from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class PlanningContractTests(unittest.TestCase):
    def test_registered_module_order_and_m2_variant_groups_are_exact(self):
        """Reordering modules or losing an injection arm/repetition must fail this test."""
        config = importlib.import_module("phase1_driver.config")
        self.assertEqual(
            config.EXECUTION_STAGES,
            ("M1", "M6", "M2_PRIORITY_R0", "M2_PRIORITY_R1", "M2_PRIORITY_R2", "M4", "M2_REMAINDER", "M5"),
        )
        priority = config.m2_priority_variants()
        remainder = config.m2_remainder_variants()
        self.assertEqual(len(priority), 24)
        self.assertEqual(len(remainder), 15)
        self.assertEqual(priority[0], "conc_0.65_r0")
        self.assertEqual(priority[4], "dispn_0.65_r0")
        self.assertEqual(priority[-1], "dispn_0.95_r2")
        self.assertEqual(remainder[0], "noise_r0")
        self.assertEqual(len(set(priority + remainder)), 39)

    def test_output_schemas_match_registered_brief(self):
        """Changing any registered output column must fail this test."""
        config = importlib.import_module("phase1_driver.config")
        self.assertEqual(
            config.PREDICTION_COLUMNS,
            ("openml_id", "variant", "fold", "row_index", "y", "model", "p"),
        )
        self.assertEqual(
            config.SELECTION_COLUMNS,
            (
                "id", "variant", "fold", "model",
                "inner_auroc_1", "inner_auroc_2", "inner_auroc_3", "inner_auroc_4", "inner_auroc_5",
                "inner_auroc_mean", "inner_status", "selected",
            ),
        )
        self.assertEqual(
            config.METRIC_COLUMNS,
            (
                "id", "variant", "fold", "model", "auroc", "logloss", "brier",
                "n_test", "n_pos_test", "n_train", "fit_seconds", "host", "cpu_model",
                "gpu_model", "n_threads", "status", "failure_reason",
            ),
        )


class PreprocessingContractTests(unittest.TestCase):
    @staticmethod
    def sample_frames():
        train = pd.DataFrame(
            {
                "n0": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
                "n1": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0],
                "c3": ["a", "b", None, "a", "c", "b"],
                "c5": ["u", "v", "w", None, "x", "y"],
            }
        )
        test = pd.DataFrame(
            {"n0": [np.nan, 7.0], "n1": [2.0, np.nan], "c3": ["z", None], "c5": ["u", "z"]}
        )
        return train, test

    def test_ordinal_views_impute_before_encoding_and_report_category_positions(self):
        """Leaving NaNs or misreporting category indices must fail this test."""
        data = importlib.import_module("phase1_driver.data")
        train, test = self.sample_frames()
        prepared = data.prepare_ordinal_views(train, test, ["c3", "c5"])
        self.assertEqual(prepared.categorical_positions, [0, 1])
        self.assertFalse(np.isnan(prepared.train).any())
        self.assertFalse(np.isnan(prepared.test).any())
        self.assertEqual(prepared.train.shape, (6, 4))

    def test_ftt_views_use_lr_column_transform_and_all_continuous_features(self):
        """Treating FTT categories as embeddings or omitting LR-style transforms must fail this test."""
        data = importlib.import_module("phase1_driver.data")
        train, test = self.sample_frames()
        prepared = data.prepare_ftt_views(train, test, ["c3", "c5"])
        self.assertEqual(prepared.cat_cardinalities, [])
        self.assertFalse(np.isnan(prepared.train).any())
        self.assertFalse(np.isnan(prepared.test).any())
        self.assertEqual(prepared.train.shape[0], 6)
        self.assertGreater(prepared.train.shape[1], 4)
        means = prepared.train.mean(axis=0)
        self.assertTrue(np.all(np.isfinite(means)))


class FTTTrainingContractTests(unittest.TestCase):
    def test_epoch_batches_are_deterministic_shuffled_complete_and_at_most_256(self):
        """Removing shuffling, dropping rows, or exceeding batch size must fail this test."""
        gpu = importlib.import_module("phase1_driver.gpu_models")
        first = gpu.epoch_batches(601, epoch=0)
        repeat = gpu.epoch_batches(601, epoch=0)
        second_epoch = gpu.epoch_batches(601, epoch=1)
        self.assertEqual([x.tolist() for x in first], [x.tolist() for x in repeat])
        self.assertNotEqual([x.tolist() for x in first], [x.tolist() for x in second_epoch])
        self.assertEqual(sorted(np.concatenate(first).tolist()), list(range(601)))
        self.assertLessEqual(max(len(batch) for batch in first), 256)

    def test_gpu_determinism_environment_is_set_before_torch_use(self):
        """Removing the registered CUBLAS deterministic setting must fail this test."""
        config = importlib.import_module("phase1_driver.config")
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        settings = config.gpu_determinism_record()
        self.assertEqual(settings["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertTrue(settings["cudnn_deterministic"])
        self.assertFalse(settings["cudnn_benchmark"])
        self.assertTrue(settings["deterministic_algorithms"])


class CheckpointContractTests(unittest.TestCase):
    def test_checkpoint_identity_rejects_a_different_driver_or_registration(self):
        """Reusing a checkpoint from another registration or driver must fail this test."""
        checkpoints = importlib.import_module("phase1_driver.checkpoints")
        identity = checkpoints.CheckpointIdentity(
            registration_sha256="a" * 64,
            driver_sha256="b" * 64,
            host_assignment_sha256="c" * 64,
            task_key="M1|3|raw|0|lr",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            checkpoints.write_checkpoint(path, identity, {"status": "ok"})
            self.assertEqual(checkpoints.read_checkpoint(path, identity)["status"], "ok")
            changed = checkpoints.CheckpointIdentity(
                registration_sha256="a" * 64,
                driver_sha256="d" * 64,
                host_assignment_sha256="c" * 64,
                task_key="M1|3|raw|0|lr",
            )
            with self.assertRaisesRegex(RuntimeError, "checkpoint identity mismatch"):
                checkpoints.read_checkpoint(path, changed)

    def test_old_driver_checkpoint_requires_explicit_task_scoped_migration(self):
        checkpoints = importlib.import_module("phase1_driver.checkpoints")
        old = checkpoints.CheckpointIdentity("a" * 64, "b" * 64, "c" * 64, "unit")
        new = checkpoints.CheckpointIdentity("a" * 64, "d" * 64, "c" * 64, "unit")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            checkpoints.write_checkpoint(path, old, {"status": "ok"})
            with self.assertRaisesRegex(RuntimeError, "not approved for migration"):
                checkpoints.read_checkpoint_with_migration(path, new, [])
            payload, record = checkpoints.read_checkpoint_with_migration(path, new, [{
                "old_driver_sha256": "b" * 64,
                "new_driver_sha256": "d" * 64,
                "task_keys": ["unit"],
                "approved_by": "commissioning-party",
            }])
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(record["task_key"], "unit")


class FrozenDriverSurfaceTests(unittest.TestCase):
    def test_execution_planner_aggregator_and_guarded_cli_import(self):
        """Removing any post-confirmation driver surface must fail this test."""
        execution = importlib.import_module("phase1_driver.execution")
        importlib.import_module("phase1_driver.planner")
        importlib.import_module("phase1_driver.aggregate")
        cli = importlib.import_module("phase1_driver.cli")
        unit = execution.WorkUnit("M1", "3", "raw", 0, "cpu", "686")
        self.assertEqual(unit.key, "M1|3|raw|0|cpu")
        with tempfile.TemporaryDirectory() as temporary:
            authorization = Path(temporary) / "authorization.txt"
            with self.assertRaisesRegex(RuntimeError, "locked"):
                cli._authorized(authorization)
            authorization.write_text("PHASE1_AUTHORIZED\n", encoding="utf-8")
            cli._authorized(authorization)


if __name__ == "__main__":
    unittest.main()
