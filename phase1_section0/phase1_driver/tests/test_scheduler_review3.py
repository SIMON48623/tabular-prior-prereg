from __future__ import annotations

import importlib
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def supervised_test_operation(message):
    behavior = message["behavior"]
    attempt = int(message["attempt_number"])
    if behavior == "crash_once" and attempt == 1:
        os._exit(86)
    if behavior == "timeout_once" and attempt == 1:
        time.sleep(1.0)
    if behavior == "timeout_then_slow_success":
        time.sleep(1.0 if attempt == 1 else 0.2)
    if behavior == "always_crash":
        os._exit(87)
    if behavior == "model_failure":
        return {"status": "completed_with_failures", "task_key": message["key"]}
    if behavior == "weight_attestation_fatal":
        return {
            "status": "fatal", "error_type": "weight_attestation_fatal",
            "reason": "loaded parameter hash mismatch",
        }
    return {
        "status": "ok", "task_key": message["key"], "pid": os.getpid(),
        "seen_fatal_attempts": list(message.get("fatal_attempts", [])),
    }


class SchedulerEnvironmentTests(unittest.TestCase):
    def test_checkpoint_environment_excludes_dispatch_stage_arguments(self):
        """Catches embedding --stages, which makes valid split runs conflict."""
        scheduler = importlib.import_module("phase1_driver.scheduler")
        class Weight:
            sha256 = "a" * 64
            path = "/registered/weight.ckpt"
        saved = {name: os.environ.get(name) for name in (
            "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
        )}
        try:
            os.environ.update({
                "PYTHONHASHSEED": "13", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            })
            record = scheduler._environment_record(
                "machine-686", "686", "Xeon 8352V", "", "", {"a": "b"}, {"c": "d"},
                "e" * 64, {"tabpfn35": Weight()},
            )
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self.assertNotIn("stages", record)

    def test_unregistered_m2_shortcut_is_rejected_but_registered_m2_stage_is_allowed(self):
        """Catches bypassing the registered M2/M4 dispatch order with --stages M2."""
        scheduler = importlib.import_module("phase1_driver.scheduler")
        common = [
            "--registered-root", "/registered", "--host-assignment", "/assignment.csv",
            "--checkpoint-root", "/checkpoints", "--runtime-root", "/runtime",
            "--authorization-file", "/authorized", "--group", "cpu", "--workers", "32",
            "--stages",
        ]
        with self.assertRaises(SystemExit):
            scheduler.parse_args(common + ["M2"])
        parsed = scheduler.parse_args(common + ["M2_PRIORITY_R0"])
        self.assertEqual(parsed.stages, ["M2_PRIORITY_R0"])


class TimeoutContractTests(unittest.TestCase):
    def test_timeout_is_ten_times_projection_and_preflight_ratio_with_one_hour_floor(self):
        """Catches missing the multiplier, EBM ratio, or minimum deadline."""
        timeouts = importlib.import_module("phase1_driver.timeouts")
        self.assertEqual(timeouts.timeout_from_projection_seconds(10.0, 2.765933), 3600.0)
        self.assertAlmostEqual(
            timeouts.timeout_from_projection_seconds(1000.0, 2.765933),
            27659.33,
        )


class WorkerSupervisorTests(unittest.TestCase):
    def _run(self, behaviors, timeout=0.2, recycle_after=200,
             retry_timeout=None):
        supervisor = importlib.import_module("phase1_driver.worker_supervisor")
        tasks = [
            supervisor.SupervisedTask(
                key=f"unit-{number}", payload={
                    "key": f"unit-{number}", "behavior": behavior,
                    "fatal_attempts": [],
                },
                timeout_seconds=timeout,
                retry_timeout_seconds=retry_timeout,
            )
            for number, behavior in enumerate(behaviors)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            return supervisor.run_supervised_tasks(
                tasks, supervised_test_operation, worker_count=1,
                recycle_after=recycle_after, poll_interval=0.01,
                event_directory=Path(temporary),
            )

    def test_crashed_worker_is_attributed_retried_and_rebuilt(self):
        """Catches a BrokenProcessPool aborting the scheduler or losing the unit."""
        report = self._run(["crash_once", "ok"])
        self.assertEqual([row["status"] for row in report.results], ["ok", "ok"])
        crash = [row for row in report.fatal_events if row["error_type"] == "worker_crash"]
        self.assertEqual(len(crash), 1)
        self.assertEqual(crash[0]["task_key"], "unit-0")
        self.assertEqual(report.results[0]["seen_fatal_attempts"], crash)
        self.assertGreaterEqual(report.worker_starts, 2)

    def test_timed_out_worker_is_killed_retried_and_rebuilt(self):
        """Catches a hung unit occupying a slot forever."""
        report = self._run(["timeout_once", "ok"], timeout=0.1)
        self.assertEqual([row["status"] for row in report.results], ["ok", "ok"])
        timeout = [row for row in report.fatal_events if row["error_type"] == "timeout"]
        self.assertEqual(len(timeout), 1)
        self.assertEqual(timeout[0]["task_key"], "unit-0")
        self.assertGreaterEqual(report.worker_starts, 2)

    def test_fault_injection_can_restore_production_deadline_for_retry(self):
        """Catches the artificial first-attempt deadline killing a valid retry."""
        report = self._run(
            ["timeout_then_slow_success"], timeout=0.1, retry_timeout=1.0,
        )
        self.assertEqual(report.results[0]["status"], "ok")
        self.assertEqual(
            [row["error_type"] for row in report.fatal_events], ["timeout"],
        )

    def test_second_fatal_attempt_yields_failed_unit_and_scheduler_continues(self):
        """Catches infinite retry or aborting the following unit."""
        report = self._run(["always_crash", "ok"])
        self.assertEqual(report.results[0]["status"], "failed")
        self.assertEqual(len(report.results[0]["fatal_attempts"]), 2)
        self.assertEqual(report.results[1]["status"], "ok")

    def test_model_failure_does_not_count_as_unit_failure_streak(self):
        """Catches §7 model failures triggering the twenty-unit stop."""
        report = self._run(["model_failure"] * 25)
        self.assertEqual(len(report.results), 25)
        self.assertEqual(report.maximum_consecutive_unit_failures, 0)

    def test_weight_attestation_fatal_stops_scheduler_without_terminal_result(self):
        """Catches retrying the machine-fatal weight proof or writing a checkpoint."""
        supervisor = importlib.import_module("phase1_driver.worker_supervisor")
        task = supervisor.SupervisedTask(
            key="unit-attestation", payload={
                "key": "unit-attestation", "behavior": "weight_attestation_fatal",
                "fatal_attempts": [],
            }, timeout_seconds=10.0,
        )
        fatal_events, terminal_results = [], []
        with self.assertRaises(supervisor.SchedulerFatalError):
            supervisor.run_supervised_tasks(
                [task], supervised_test_operation, worker_count=1,
                poll_interval=0.01, on_fatal=lambda _task, event: fatal_events.append(event),
                on_result=lambda _task, result: terminal_results.append(result),
            )
        self.assertEqual(len(fatal_events), 1)
        self.assertEqual(fatal_events[0]["error_type"], "weight_attestation_fatal")
        self.assertEqual(terminal_results, [])

    def test_worker_is_recycled_after_configured_completed_unit_count(self):
        """Catches unbounded worker lifetime and retained library state."""
        report = self._run(["ok", "ok", "ok"], recycle_after=2)
        pids = [row["pid"] for row in report.results]
        self.assertEqual(pids[0], pids[1])
        self.assertNotEqual(pids[1], pids[2])
        self.assertEqual(report.recycled_workers, 1)


class FatalCheckpointContractTests(unittest.TestCase):
    def _registered_root(self, base: Path) -> Path:
        root = base / "registered"
        cleaned = root / "cleaned_data"
        cleaned.mkdir(parents=True)
        pd.DataFrame({
            "x": list(range(10)), "y": [0, 1] * 5,
            "fold": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        }).to_parquet(cleaned / "3.parquet", index=False)
        (cleaned / "3.schema.json").write_text(
            '{"columns":{"x":{"type":"numeric"}}}\n', encoding="utf-8"
        )
        return root

    def test_terminal_fatal_checkpoint_has_failed_rows_for_every_owned_model(self):
        """Catches schema-empty fatal checkpoints that cannot satisfy §5.3 aggregation."""
        execution = importlib.import_module("phase1_driver.execution")
        config = importlib.import_module("phase1_driver.config")
        environment = {"host": "machine", "cpu_model": "8352V", "gpu_model": "RTX 4090"}
        attempts = [
            {"attempt": 1, "error_type": "worker_crash", "reason": "exit 9"},
            {"attempt": 2, "error_type": "timeout", "reason": "deadline"},
        ]
        cases = (
            (execution.WorkUnit("M1", "3", "raw", 0, "cpu", "686"), config.CPU_MODELS, True),
            (execution.WorkUnit("M1", "3", "raw", 0, "gpu", "4090"), config.FOUNDATION_MODELS, True),
            (execution.WorkUnit("M1", "3", "raw", 0, "ftt", "4090"), ("ftt",), False),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = self._registered_root(Path(temporary))
            for unit, models, needs_selection in cases:
                payload = execution.failed_unit_payload(
                    root, unit, "deadline", "timeout", environment, attempts,
                )
                self.assertEqual([row["model"] for row in payload["metrics"]], list(models))
                self.assertTrue(all(row["status"] == "failed" for row in payload["metrics"]))
                self.assertTrue(all(row["n_train"] == 8 and row["n_test"] == 2 for row in payload["metrics"]))
                self.assertEqual(payload["predictions"], {})
                if needs_selection:
                    self.assertEqual([row["model"] for row in payload["selection"]], list(models))
                    self.assertTrue(all(row["inner_status"] == "failed" for row in payload["selection"]))
                else:
                    self.assertEqual(payload["selection"], [])
                self.assertEqual(payload["log"]["fatal_attempts"], attempts)

    def test_failure_log_and_fatal_ledger_fsync_their_directories(self):
        """Catches rename/data durability without directory-entry durability."""
        scheduler = importlib.import_module("phase1_driver.scheduler")
        execution = importlib.import_module("phase1_driver.execution")
        unit = execution.WorkUnit("M1", "3", "raw", 0, "cpu", "686")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(scheduler.os, "fsync", wraps=os.fsync) as fsync:
                scheduler._append_jsonl(root / "logs" / "failure.jsonl", {"status": "failed"})
                append_calls = fsync.call_count
                scheduler._save_fatal_attempts(
                    root, unit, [{"attempt": 1, "error_type": "timeout", "reason": "deadline"}],
                )
                ledger_calls = fsync.call_count - append_calls
            self.assertGreaterEqual(append_calls, 2)
            self.assertGreaterEqual(ledger_calls, 2)


class DryRunFaultPlanTests(unittest.TestCase):
    def test_fault_plan_selects_distinct_units_for_model_crash_and_timeout(self):
        """Catches a report claiming injections that were never attached to pool units."""
        dry_run = importlib.import_module("phase1_driver.dry_run")
        execution = importlib.import_module("phase1_driver.execution")
        plan = [
            execution.WorkUnit("M1", "3", "raw", 0, "cpu", "686"),
            execution.WorkUnit("M2", "3", "noise_r0", 0, "gpu", "4090"),
            execution.WorkUnit("M4", "3", "n100_r0", 0, "ftt", "4090"),
        ]
        controls = dry_run.build_fault_controls(plan)
        self.assertEqual(set(controls), {unit.key for unit in plan})
        self.assertEqual(
            {control["fault"] for control in controls.values()},
            {"model_failure", "worker_crash", "timeout"},
        )
        model = next(control for control in controls.values() if control["fault"] == "model_failure")
        self.assertEqual(model["model"], "lr")


if __name__ == "__main__":
    unittest.main()
