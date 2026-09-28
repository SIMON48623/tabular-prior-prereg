from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path


class ColumnarCheckpointTests(unittest.TestCase):
    def test_prediction_rows_round_trip_as_columnar_arrays(self):
        checkpoints = importlib.import_module("phase1_driver.checkpoints")
        rows = [
            {"openml_id": 3, "variant": "raw", "fold": 0, "row_index": 8, "y": 1, "model": "lr", "p": 0.7},
            {"openml_id": 3, "variant": "raw", "fold": 0, "row_index": 9, "y": 0, "model": "lr", "p": 0.2},
        ]
        columnar = checkpoints.rows_to_columnar(rows)
        self.assertEqual(set(columnar), {"openml_id", "variant", "fold", "row_index", "y", "model", "p"})
        self.assertEqual(columnar["row_index"], [8, 9])
        self.assertEqual(checkpoints.columnar_to_rows(columnar), rows)


class CompletePlanAggregationTests(unittest.TestCase):
    def identity(self, task_key: str):
        checkpoints = importlib.import_module("phase1_driver.checkpoints")
        return checkpoints.CheckpointIdentity("a" * 64, "b" * 64, "c" * 64, task_key)

    @staticmethod
    def payload(unit, row_index: int, model: str):
        checkpoints = importlib.import_module("phase1_driver.checkpoints")
        row = {"openml_id": int(unit.identifier), "variant": unit.variant, "fold": unit.fold,
               "row_index": row_index, "y": 1, "model": model, "p": 0.5}
        metric = {
            "id": unit.identifier, "variant": unit.variant, "fold": unit.fold,
            "model": model, "status": "ok", "n_test": 1,
        }
        return {
            "unit": asdict(unit),
            "metrics": [metric],
            "predictions": checkpoints.rows_to_columnar([row]),
            "selection": [], "ftt_log": [],
            "log": {"elapsed_seconds": 1.0, "status": "ok", "failure_reasons": {},
                    "s_undefined": False, "inner_failures": {}},
        }

    def test_complete_loader_requires_exact_identity_and_one_checkpoint_per_plan_unit(self):
        checkpoints = importlib.import_module("phase1_driver.checkpoints")
        aggregate = importlib.import_module("phase1_driver.aggregate")
        execution = importlib.import_module("phase1_driver.execution")
        units = [
            execution.WorkUnit("M1", "3", "raw", 0, "cpu", "686"),
            execution.WorkUnit("M1", "3", "raw", 0, "gpu", "4090"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for unit, model in zip(units, ("lr", "tabpfn35"), strict=True):
                path = checkpoints.checkpoint_path(root, unit)
                checkpoints.write_checkpoint(path, self.identity(unit.key), self.payload(unit, 8, model))
            loaded = aggregate.load_complete_checkpoints(root, units, self.identity)
            self.assertEqual(len(loaded), 2)
            checkpoints.checkpoint_path(root, units[1]).unlink()
            with self.assertRaisesRegex(RuntimeError, "missing checkpoint"):
                aggregate.load_complete_checkpoints(root, units, self.identity)

    def test_duplicate_metric_and_prediction_keys_are_rejected(self):
        aggregate = importlib.import_module("phase1_driver.aggregate")
        execution = importlib.import_module("phase1_driver.execution")
        unit = execution.WorkUnit("M1", "3", "raw", 0, "cpu", "686")
        payload = self.payload(unit, 8, "lr")
        payload["metrics"].append(dict(payload["metrics"][0]))
        with self.assertRaisesRegex(RuntimeError, "duplicate metric key"):
            aggregate.validate_payload_uniqueness([payload])
        payload = self.payload(unit, 8, "lr")
        for key in payload["predictions"]:
            payload["predictions"][key].append(payload["predictions"][key][0])
        with self.assertRaisesRegex(RuntimeError, "duplicate prediction key"):
            aggregate.validate_payload_uniqueness([payload])

    def test_plan_groups_are_streamed_by_module_and_dataset(self):
        aggregate = importlib.import_module("phase1_driver.aggregate")
        execution = importlib.import_module("phase1_driver.execution")
        units = [
            execution.WorkUnit("M1", "3", "raw", 0, "cpu", "686"),
            execution.WorkUnit("M1", "3", "raw", 0, "gpu", "4090"),
            execution.WorkUnit("M1", "15", "raw", 0, "cpu", "214"),
            execution.WorkUnit("M2", "31", "noise_r0", 0, "cpu", "686"),
        ]
        groups = list(aggregate.iter_plan_groups(units))
        self.assertEqual([(key, len(group)) for key, group in groups], [
            (("M1", "3"), 2), (("M1", "15"), 1), (("M2", "31"), 1)
        ])

    def test_success_predictions_equal_n_test_and_failed_models_have_no_rows(self):
        aggregate = importlib.import_module("phase1_driver.aggregate")
        execution = importlib.import_module("phase1_driver.execution")
        unit = execution.WorkUnit("M1", "3", "raw", 0, "cpu", "686")
        payload = self.payload(unit, 8, "lr")
        aggregate.validate_prediction_counts([payload])
        payload["metrics"][0]["n_test"] = 2
        with self.assertRaisesRegex(RuntimeError, "prediction row count"):
            aggregate.validate_prediction_counts([payload])
        payload = self.payload(unit, 8, "lr")
        payload["metrics"][0]["status"] = "failed"
        with self.assertRaisesRegex(RuntimeError, "failed model has prediction rows"):
            aggregate.validate_prediction_counts([payload])

    def test_single_module_aggregation_ignores_other_module_checkpoints(self):
        checkpoints = importlib.import_module("phase1_driver.checkpoints")
        aggregate = importlib.import_module("phase1_driver.aggregate")
        execution = importlib.import_module("phase1_driver.execution")
        units = [
            execution.WorkUnit("M1", "3", "raw", 0, "cpu", "686"),
            execution.WorkUnit("M2", "3", "noise_r0", 0, "cpu", "686"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_root = root / "checkpoints"
            output_root = root / "output"
            for unit in units:
                payload = {
                    "unit": asdict(unit), "metrics": [], "predictions": {},
                    "selection": [], "ftt_log": [],
                    "environment": {"host": "host686"},
                    "log": {
                        "task_key": unit.key, "status": "ok",
                        "elapsed_seconds": 0.0, "failure_reasons": {},
                        "s_undefined": False, "inner_failures": {},
                    },
                }
                checkpoints.write_checkpoint(
                    checkpoints.checkpoint_path(checkpoint_root, unit),
                    self.identity(unit.key), payload,
                )
            aggregate.aggregate_complete(
                checkpoint_root, output_root, units, self.identity,
                modules={"M1"}, migrations=[],
            )
            lines = (output_root / "run_log.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertIn(units[0].key, lines[0])

    def test_two_m2_schedules_aggregate_with_environment_history_per_host(self):
        """Catches rejecting a valid second M2 dispatch from the same machine."""
        checkpoints = importlib.import_module("phase1_driver.checkpoints")
        aggregate = importlib.import_module("phase1_driver.aggregate")
        execution = importlib.import_module("phase1_driver.execution")
        units = [
            execution.WorkUnit("M2", "3", "conc_0.65_r0", 0, "cpu", "686"),
            execution.WorkUnit("M2", "3", "noise_r0", 0, "cpu", "686"),
        ]
        environments = [
            {"host": "machine-686", "python": "3.11.9", "packages": {"pandas": "2.3.3"}},
            {"host": "machine-686", "python": "3.11.9", "packages": {"pandas": "2.3.3"},
             "kernel": "same-host-later-boot"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_root = root / "checkpoints"
            output_root = root / "output"
            for unit, environment in zip(units, environments, strict=True):
                payload = {
                    "unit": asdict(unit), "metrics": [], "predictions": {},
                    "selection": [], "ftt_log": [], "environment": environment,
                    "log": {"task_key": unit.key, "status": "ok", "elapsed_seconds": 0.0,
                            "failure_reasons": {}, "s_undefined": False,
                            "inner_failures": {}, "fatal_attempts": []},
                }
                checkpoints.write_checkpoint(
                    checkpoints.checkpoint_path(checkpoint_root, unit),
                    self.identity(unit.key), payload,
                )
            aggregate.aggregate_complete(
                checkpoint_root, output_root, units, self.identity,
                modules={"M2"}, migrations=[],
            )
            document = json.loads((output_root / "environment_run.json").read_text(encoding="utf-8"))
            self.assertEqual(document["hosts"]["machine-686"], environments)

    def test_multi_module_aggregation_keeps_approved_migration_and_environment_lists(self):
        """Catches migration metadata or another module collapsing host history."""
        checkpoints = importlib.import_module("phase1_driver.checkpoints")
        aggregate = importlib.import_module("phase1_driver.aggregate")
        execution = importlib.import_module("phase1_driver.execution")
        units = [
            execution.WorkUnit("M1", "3", "raw", 0, "cpu", "686"),
            execution.WorkUnit("M4", "3", "n100_r0", 0, "cpu", "686"),
        ]
        old_driver = "d" * 64
        migration = [{"old_driver_sha256": old_driver,
                      "new_driver_sha256": "b" * 64,
                      "task_keys": [units[0].key], "approved_by": "commissioner"}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_root = root / "checkpoints"
            output_root = root / "output"
            for index, unit in enumerate(units):
                payload = {
                    "unit": asdict(unit), "metrics": [], "predictions": {},
                    "selection": [], "ftt_log": [],
                    "environment": {"host": "machine-686", "record": index},
                    "log": {"task_key": unit.key, "status": "ok", "elapsed_seconds": 0.0,
                            "failure_reasons": {}, "s_undefined": False,
                            "inner_failures": {}, "fatal_attempts": []},
                }
                identity = self.identity(unit.key)
                if index == 0:
                    identity = checkpoints.CheckpointIdentity("a" * 64, old_driver, "c" * 64, unit.key)
                checkpoints.write_checkpoint(checkpoints.checkpoint_path(checkpoint_root, unit), identity, payload)
            aggregate.aggregate_complete(
                checkpoint_root, output_root, units, self.identity,
                modules={"M1", "M4"}, migrations=migration,
            )
            lines = (output_root / "run_log.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(any("checkpoint_migration" in line for line in lines))
            document = json.loads((output_root / "environment_run.json").read_text(encoding="utf-8"))
            self.assertEqual(len(document["hosts"]["machine-686"]), 2)


class ReportingContractTests(unittest.TestCase):
    def test_run_log_and_environment_record_include_required_fields(self):
        reporting = importlib.import_module("phase1_driver.reporting")
        log = {
            "task_key": "M1|3|raw|0|cpu", "elapsed_seconds": 1.2, "status": "ok",
            "failure_reasons": {}, "s_undefined": False,
            "inner_failures": {"xgb": "inner fold 2 failed"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reporting.append_run_log(root / "run_log.txt", log)
            saved = json.loads((root / "run_log.txt").read_text(encoding="utf-8"))
            self.assertEqual(saved["inner_failures"]["xgb"], "inner fold 2 failed")
            environment = {
                "host": "4090", "lock_sha256": "a" * 64,
                "weights": {"tabpfn35": "b" * 64},
                "determinism": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            }
            reporting.write_environment_record(root / "environment_run.json", environment)
            second = {**environment, "dispatch_note": "second registered stage"}
            reporting.write_environment_record(root / "environment_run.json", second)
            got = json.loads((root / "environment_run.json").read_text(encoding="utf-8"))
            self.assertEqual(len(got["hosts"]["4090"]), 2)
            self.assertEqual(got["hosts"]["4090"][0]["weights"]["tabpfn35"], "b" * 64)


if __name__ == "__main__":
    unittest.main()
