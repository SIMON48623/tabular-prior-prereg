from __future__ import annotations

import hashlib
import importlib
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RuntimeAttestationTests(unittest.TestCase):
    def make_registered_root(self, base: Path) -> tuple[Path, dict[str, Path]]:
        root = base / "registered"
        (root / "pipeline_code" / "phase0b").mkdir(parents=True)
        (root / "code").mkdir()
        (root / "phase0c_outputs").mkdir()
        (root / "frozen").mkdir()
        (root / "final_pool").mkdir()
        pipeline = root / "pipeline_code" / "phase0b" / "phase0b_experiment.py"
        generator = root / "code" / "generators.py"
        lock = root / "phase0c_outputs" / "environment_phase1.lock.txt"
        pipeline.write_text("registered pipeline\n", encoding="utf-8")
        generator.write_text("registered generator\n", encoding="utf-8")
        lock.write_text("registered lock\n", encoding="utf-8")
        calibration = root / "code" / "synth_calibration.json"
        stratifier = root / "frozen" / "stratifier_v2.csv"
        final_pool = root / "final_pool" / "datasets_stratifiers.csv"
        calibration.write_text("{}\n", encoding="utf-8")
        stratifier.write_text("openml_id,target_region\n3,1\n", encoding="utf-8")
        final_pool.write_text("openml_id,excluded_reason\n3,\n", encoding="utf-8")
        weights = {}
        checkpoints = {}
        for key, filename in (
            ("tabpfn35", "v35.safetensors"),
            ("tabpfn2_b", "v2.ckpt"),
            ("tabicl2", "icl.ckpt"),
        ):
            path = base / filename
            path.write_bytes((key * 5).encode())
            weights[key] = path
            checkpoints[key] = {"path": str(path), "filename": filename, "sha256": digest(path), "bytes": path.stat().st_size}
        environment = root / "phase0c_outputs" / "environment_phase1.json"
        environment.write_text(json.dumps({"checkpoints": checkpoints}), encoding="utf-8")
        manifest_entries = [
            pipeline, generator, calibration, stratifier, final_pool, lock, environment,
        ]
        (root / "MANIFEST.sha256").write_text(
            "".join(f"{digest(path)}  {path.relative_to(root).as_posix()}\n" for path in manifest_entries),
            encoding="utf-8",
        )
        return root, weights

    def make_driver(self, base: Path) -> Path:
        driver = base / "phase1_driver"
        driver.mkdir()
        (driver / "runtime.py").write_text("candidate\n", encoding="utf-8")
        (base / "phase1_driver_sha256.txt").write_text(
            f"{digest(driver / 'runtime.py')}  phase1_driver/runtime.py\n", encoding="utf-8"
        )
        return driver

    def test_registered_and_driver_hashes_are_derived_from_files(self):
        runtime = importlib.import_module("phase1_driver.runtime")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, _ = self.make_registered_root(base)
            driver = self.make_driver(base)
            record = runtime.verify_registered_inputs(root)
            self.assertIn("code/generators.py", record)
            self.assertIn("phase0c_outputs/environment_phase1.lock.txt", record)
            runtime.verify_driver_manifest(driver)
            (driver / "runtime.py").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "driver hash mismatch"):
                runtime.verify_driver_manifest(driver)
            (root / "code" / "generators.py").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "registered hash mismatch"):
                runtime.verify_registered_inputs(root)

    def test_weight_paths_are_explicit_and_each_hash_is_checked(self):
        runtime = importlib.import_module("phase1_driver.runtime")
        with tempfile.TemporaryDirectory() as temporary:
            root, weights = self.make_registered_root(Path(temporary))
            got = runtime.registered_checkpoint_paths(root)
            self.assertEqual(got["tabpfn35"].path, weights["tabpfn35"].resolve())
            self.assertEqual(got["tabpfn2"].path, weights["tabpfn2_b"].resolve())
            self.assertEqual(got["tabicl2"].path, weights["tabicl2"].resolve())
            weights["tabicl2"].write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "weight hash mismatch"):
                runtime.registered_checkpoint_paths(root)

    def test_cleaned_data_ledger_is_created_once_and_rechecked(self):
        runtime = importlib.import_module("phase1_driver.runtime")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "registered"
            cleaned = root / "cleaned_data"
            cleaned.mkdir(parents=True)
            (cleaned / "3.parquet").write_bytes(b"parquet")
            (cleaned / "3.schema.json").write_text("{}\n", encoding="utf-8")
            runtime_root = base / "runtime"
            with self.assertRaisesRegex(RuntimeError, "aggregate hash mismatch"):
                runtime.ensure_cleaned_data_manifest(root, runtime_root)
            manifest_hash = runtime.sha256_file(runtime_root / "cleaned_data_sha256.txt")
            with patch.object(runtime, "CLEANED_DATA_MANIFEST_SHA256", manifest_hash):
                first = runtime.ensure_cleaned_data_manifest(root, runtime_root)
                second = runtime.ensure_cleaned_data_manifest(root, runtime_root)
            self.assertEqual(len(first), 2)
            self.assertEqual(first, second)
            (cleaned / "3.schema.json").write_text("changed\n", encoding="utf-8")
            with patch.object(runtime, "CLEANED_DATA_MANIFEST_SHA256", manifest_hash):
                with self.assertRaisesRegex(RuntimeError, "cleaned_data hash mismatch"):
                    runtime.ensure_cleaned_data_manifest(root, runtime_root)

    def test_cleaned_data_manifest_hash_is_frozen_to_three_host_consensus(self):
        runtime = importlib.import_module("phase1_driver.runtime")
        self.assertEqual(
            runtime.CLEANED_DATA_MANIFEST_SHA256,
            "a807db2390d5a85a9b77f1286e6ab7d1aa1a1f7faaf2396f3be8459220e0ce68",
        )

    def test_foundation_constructor_kwargs_contain_explicit_registered_path(self):
        runtime = importlib.import_module("phase1_driver.runtime")
        gpu_models = importlib.import_module("phase1_driver.gpu_models")
        with tempfile.TemporaryDirectory() as temporary:
            root, _ = self.make_registered_root(Path(temporary))
            checkpoints = runtime.registered_checkpoint_paths(root)
            for model in ("tabpfn35", "tabpfn2", "tabicl2"):
                kwargs = gpu_models.foundation_constructor_kwargs(model, checkpoints, [0, 1])
                self.assertEqual(Path(kwargs["model_path"]), checkpoints[model].path)
                self.assertFalse(kwargs.get("allow_auto_download", False))

    def test_constructor_path_alone_is_not_accepted_as_loaded_weight_attestation(self):
        """Catches mistaking a constructor argument for proof of loaded parameters."""
        gpu_models = importlib.import_module("phase1_driver.gpu_models")
        class Estimator:
            def __init__(self, path):
                self.path = path
            def get_params(self, deep=False):
                return {"model_path": str(self.path)}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weight.ckpt"
            path.write_bytes(b"weight")
            with self.assertRaisesRegex(RuntimeError, "loaded torch parameters"):
                gpu_models.loaded_parameter_sha256(Estimator(path))

    def test_missing_or_mismatched_loaded_parameters_raise_attestation_failure(self):
        gpu_models = importlib.import_module("phase1_driver.gpu_models")
        runtime = importlib.import_module("phase1_driver.runtime")
        class Empty:
            pass
        checkpoint = runtime.RegisteredCheckpoint(
            "tabpfn35", Path("/registered/model.ckpt"), "0" * 64, 1,
        )
        with self.assertRaises(gpu_models.WeightAttestationFailure):
            gpu_models.attest_loaded_parameters("tabpfn35", Empty(), checkpoint)

        class Tensor:
            def detach(self): return self
            def cpu(self): return self
            def contiguous(self): return self
            def numpy(self): return np.asarray([1.0], dtype=np.float32)
        class Loaded:
            def state_dict(self): return {"weight": Tensor()}
        with self.assertRaises(gpu_models.WeightAttestationFailure):
            gpu_models.attest_loaded_parameters("tabpfn35", Loaded(), checkpoint)

    def test_load_only_probe_wraps_loading_exception_as_attestation_failure(self):
        gpu_models = importlib.import_module("phase1_driver.gpu_models")
        with patch.object(
            gpu_models, "_construct_foundation_model",
            side_effect=RuntimeError("checkpoint load failed"),
        ):
            with self.assertRaises(gpu_models.WeightAttestationFailure):
                gpu_models.load_only_parameter_hashes({}, object())

    def test_loaded_parameter_hash_is_deterministic_and_changes_with_parameters(self):
        """Catches hashing constructor metadata rather than the loaded tensors."""
        gpu_models = importlib.import_module("phase1_driver.gpu_models")
        import numpy as np
        class Tensor:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=np.float32)
            def detach(self):
                return self
            def cpu(self):
                return self
            def contiguous(self):
                return self
            def numpy(self):
                return self.values
        class Module:
            def __init__(self, bias):
                self.weight = Tensor([[1.0, 2.0]])
                self.bias = Tensor([bias])
            def state_dict(self):
                return {"weight": self.weight, "bias": self.bias}
        first = Module(3.0)
        second = Module(3.0)
        digest = gpu_models.loaded_parameter_sha256(first)
        self.assertEqual(len(digest), 64)
        self.assertEqual(gpu_models.loaded_parameter_sha256(second), digest)
        class Wrapper:
            def __init__(self, name, module):
                setattr(self, name, module)
        self.assertEqual(
            gpu_models.loaded_parameter_sha256(Wrapper("before_fit", first)),
            gpu_models.loaded_parameter_sha256(Wrapper("after_fit_cache", second)),
        )
        second.bias = Tensor([4.0])
        self.assertNotEqual(gpu_models.loaded_parameter_sha256(second), digest)

    def test_gpu_cleanup_guard_runs_after_exception(self):
        """Catches CUDA cache cleanup occurring only on successful prediction."""
        gpu_models = importlib.import_module("phase1_driver.gpu_models")
        calls = []
        class Cuda:
            @staticmethod
            def empty_cache():
                calls.append("empty_cache")
        class FakeTorch:
            cuda = Cuda()
        with patch.object(gpu_models.gc, "collect", side_effect=lambda: calls.append("gc")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with gpu_models.gpu_cleanup_guard(FakeTorch()):
                    raise RuntimeError("injected")
        self.assertEqual(calls, ["gc", "empty_cache"])

    def test_offline_environment_and_socket_guard_block_network(self):
        runtime = importlib.import_module("phase1_driver.runtime")
        runtime.set_offline_environment()
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            self.assertEqual(os.environ[name], "1")
        with runtime.offline_socket_guard():
            with self.assertRaisesRegex(RuntimeError, "network disabled"):
                socket.create_connection(("example.com", 80))

    def test_hardware_and_plan_membership_are_strict(self):
        runtime = importlib.import_module("phase1_driver.runtime")
        execution = importlib.import_module("phase1_driver.execution")
        unit = execution.WorkUnit("M1", "3", "raw", 0, "cpu", "686")
        runtime.validate_hardware(unit, "686", "host686", "Intel(R) Xeon(R) Platinum 8352V CPU", "")
        with self.assertRaisesRegex(RuntimeError, "machine-derived host"):
            runtime.validate_hardware(unit, "214", "host214", "Intel(R) Xeon(R) Platinum 8352V CPU", "")
        with self.assertRaisesRegex(RuntimeError, "8352V"):
            runtime.validate_hardware(unit, "686", "host686", "AMD EPYC", "")
        gpu = execution.WorkUnit("M1", "3", "raw", 0, "gpu", "4090")
        runtime.validate_hardware(gpu, "4090", "host4090", "irrelevant", "NVIDIA GeForce RTX 4090")
        with self.assertRaisesRegex(RuntimeError, "RTX 4090"):
            runtime.validate_hardware(gpu, "4090", "host4090", "irrelevant", "NVIDIA A100")
        runtime.assert_unit_in_plan(unit, [unit])
        with self.assertRaisesRegex(RuntimeError, "not in verified plan"):
            runtime.assert_unit_in_plan(unit, [])

    def test_scheduler_contracts_set_environment_before_python_and_worker_counts(self):
        driver = Path(__file__).resolve().parents[1]
        for filename, workers in (
            ("schedule_cpu_686.sh", "32"),
            ("schedule_cpu_214.sh", "32"),
            ("schedule_gpu_4090.sh", "1"),
        ):
            text = (driver / filename).read_text(encoding="utf-8")
            self.assertIn("export PYTHONHASHSEED=13", text)
            self.assertNotIn("PHASE1_HOST_LABEL", text)
            self.assertIn(f"WORKERS={workers}", text)
            self.assertIn("export HF_HUB_OFFLINE=1", text)
            self.assertIn("export TRANSFORMERS_OFFLINE=1", text)
            self.assertIn("export HF_DATASETS_OFFLINE=1", text)
            self.assertLess(text.index("export PYTHONHASHSEED=13"), text.index("python"))
            self.assertLess(text.index('[ "$OMP_NUM_THREADS" = 1 ]'), text.index("python"))
        self.assertTrue((driver / "scheduler.py").is_file())
        self.assertTrue((driver / "host_registry.json").is_file())

    def test_process_environment_must_be_supplied_before_interpreter_start(self):
        runtime = importlib.import_module("phase1_driver.runtime")
        names = ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        saved = {name: os.environ.get(name) for name in names}
        try:
            os.environ["PYTHONHASHSEED"] = "12"
            for name in names[1:]:
                os.environ[name] = "1"
            with self.assertRaisesRegex(RuntimeError, "PYTHONHASHSEED"):
                runtime.assert_process_environment()
            os.environ["PYTHONHASHSEED"] = "13"
            runtime.assert_process_environment()
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_runtime_identity_hashes_are_derived_not_operator_supplied(self):
        runtime = importlib.import_module("phase1_driver.runtime")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, _ = self.make_registered_root(base)
            driver = self.make_driver(base)
            assignment = base / "host_assignment.csv"
            assignment.write_text("openml_id,host\n3,686\n", encoding="utf-8")
            identity = runtime.derive_runtime_identity(root, driver, assignment, "unit")
            self.assertEqual(identity.registration_sha256, digest(root / "MANIFEST.sha256"))
            self.assertEqual(identity.driver_sha256, digest(base / "phase1_driver_sha256.txt"))
            self.assertEqual(identity.host_assignment_sha256, digest(assignment))


if __name__ == "__main__":
    unittest.main()
