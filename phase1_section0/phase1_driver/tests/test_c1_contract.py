from __future__ import annotations

import hashlib
import importlib
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class C1DriverManifestTests(unittest.TestCase):
    def test_real_prefixed_manifest_is_resolved_from_manifest_directory(self):
        runtime = importlib.import_module("phase1_driver.runtime")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            driver = root / "phase1_driver"
            driver.mkdir()
            readme = driver / "README.md"
            module = driver / "runtime.py"
            readme.write_text("driver\n", encoding="utf-8")
            module.write_text("runtime\n", encoding="utf-8")
            manifest = root / "phase1_driver_sha256.txt"
            manifest.write_text(
                f"{digest(readme)}  phase1_driver/README.md\n"
                f"{digest(module)}  phase1_driver/runtime.py\n",
                encoding="utf-8",
            )
            self.assertEqual(runtime.verify_driver_manifest(driver), digest(manifest))


class C1GroupedAggregationTests(unittest.TestCase):
    @staticmethod
    def identity(task_key: str):
        checkpoints = importlib.import_module("phase1_driver.checkpoints")
        return checkpoints.CheckpointIdentity("a" * 64, "b" * 64, "c" * 64, task_key)

    @staticmethod
    def payload(unit, model: str) -> dict:
        checkpoints = importlib.import_module("phase1_driver.checkpoints")
        if unit.module == "M6":
            predictions = {}
            metrics = []
        else:
            row = {
                "openml_id": int(unit.identifier), "variant": unit.variant,
                "fold": unit.fold, "row_index": 0, "y": 0,
                "model": model, "p": 0.5,
            }
            predictions = checkpoints.rows_to_columnar([row])
            metrics = [{
                "id": unit.identifier, "variant": unit.variant,
                "fold": unit.fold, "model": model, "status": "ok", "n_test": 1,
            }]
        return {
            "unit": asdict(unit), "metrics": metrics, "predictions": predictions,
            "selection": [], "ftt_log": [], "marginal": {"feature": 0.5},
            "environment": {"host": f"host-{unit.host}"},
            "log": {
                "task_key": unit.key, "status": "ok", "elapsed_seconds": 0.0,
                "failure_reasons": {}, "s_undefined": False, "inner_failures": {},
            },
        }

    def test_each_requested_group_has_independent_coverage_and_delivery(self):
        aggregate = importlib.import_module("phase1_driver.aggregate")
        checkpoints = importlib.import_module("phase1_driver.checkpoints")
        execution = importlib.import_module("phase1_driver.execution")
        units = [
            execution.WorkUnit("M1", "3", "raw", 0, "cpu", "686"),
            execution.WorkUnit("M1", "3", "raw", 0, "gpu", "4090"),
            execution.WorkUnit("M1", "3", "raw", 0, "ftt", "4090"),
            execution.WorkUnit("M6", "3", "raw", -1, "none", "686"),
        ]
        models = ("lr", "tabpfn35", "ftt", "none")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_root = root / "checkpoints"
            for unit, model in zip(units, models, strict=True):
                checkpoints.write_checkpoint(
                    checkpoints.checkpoint_path(checkpoint_root, unit),
                    self.identity(unit.key), self.payload(unit, model),
                )
            expected_logs = {"cpu": 2, "gpu": 1, "ftt": 1}
            for group, expected in expected_logs.items():
                output = root / f"output-{group}"
                aggregate.aggregate_complete(
                    checkpoint_root, output, units, self.identity,
                    modules={"M1", "M6"}, groups={group}, migrations=[],
                )
                logs = (output / "run_log.txt").read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(logs), expected)
            self.assertTrue((root / "output-cpu" / "marginal_train" / "3.json").is_file())
            self.assertFalse((root / "output-gpu" / "marginal_train" / "3.json").exists())


class C1SchedulerTests(unittest.TestCase):
    def test_gpu_host_is_machine_derived_but_not_registry_bound_and_model_is_strict(self):
        scheduler = importlib.import_module("phase1_driver.scheduler")
        runtime = importlib.import_module("phase1_driver.runtime")
        execution = importlib.import_module("phase1_driver.execution")
        with patch.object(scheduler, "machine_host_label", return_value="replacement-4090-host"), \
             patch.object(scheduler, "resolve_registered_host", side_effect=AssertionError):
            role, host = scheduler._resolve_scheduler_host("gpu", Path("unused.json"))
        self.assertEqual((role, host), ("4090", "replacement-4090-host"))
        unit = execution.WorkUnit("M1", "3", "raw", 0, "gpu", "4090")
        runtime.validate_hardware(
            unit, role, host, "irrelevant", "NVIDIA GeForce RTX 4090",
        )
        with self.assertRaisesRegex(RuntimeError, "RTX 4090"):
            runtime.validate_hardware(unit, role, host, "irrelevant", "NVIDIA A100")

    def test_verify_only_runs_startup_checks_without_authorization_or_dispatch(self):
        scheduler = importlib.import_module("phase1_driver.scheduler")
        execution = importlib.import_module("phase1_driver.execution")
        unit = execution.WorkUnit("M1", "3", "raw", 0, "cpu", "686")
        argv = [
            "--registered-root", "/registered", "--host-assignment", "/assignment.csv",
            "--checkpoint-root", "/checkpoints", "--runtime-root", "/runtime",
            "--group", "cpu", "--workers", "32", "--stages", "M1", "--verify-only",
        ]
        required_environment = {
            "PYTHONHASHSEED": "13", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        with patch.dict(os.environ, required_environment, clear=False), \
             patch.object(scheduler, "verify_registered_inputs", return_value={"a": "b"}), \
             patch.object(scheduler, "verify_driver_manifest", return_value="d" * 64), \
             patch.object(scheduler, "ensure_cleaned_data_manifest", return_value={"x": "y"}), \
             patch.object(scheduler, "build_plan", return_value=[unit]), \
             patch.object(scheduler, "_resolve_scheduler_host", return_value=("686", "cpu-host")), \
             patch.object(scheduler, "cpu_model_name", return_value="Xeon Platinum 8352V"), \
             patch.object(scheduler, "_identity_factory", return_value=lambda key: key), \
             patch.object(scheduler, "_authorized", side_effect=AssertionError("authorization used")), \
             patch.object(scheduler, "_run_pending_units", side_effect=AssertionError("dispatch used")):
            scheduler.main(argv)


if __name__ == "__main__":
    unittest.main()
