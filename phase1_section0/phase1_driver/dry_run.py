from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from . import aggregate, execution
from .checkpoints import (
    CheckpointIdentity, checkpoint_path, exclusive_lock, read_checkpoint,
)
from .execution import WorkUnit, execute_checkpoint
from .planner import build_plan
from .runtime import sha256_file
from .worker_supervisor import SupervisedTask, run_supervised_tasks


def _fake_fit_predict(_model: str, _train: Any, _labels: np.ndarray,
                      test: Any, _categorical: list[str], *_args: Any, **_kwargs: Any):
    return np.full(len(test), 0.5, dtype=np.float64), {"fake_model": True}


def _fake_ftt(_train: Any, _labels: np.ndarray, test: Any,
              _categorical: list[str]):
    return np.full(len(test), 0.5, dtype=np.float64), {
        "fake_model": True, "best_epoch": 0, "n_params": 0,
    }


def select_dry_run_plan(plan: list[WorkUnit]) -> list[WorkUnit]:
    selected: list[WorkUnit] = []
    for module in ("M1", "M2", "M4", "M6"):
        module_units = [unit for unit in plan if unit.module == module]
        identifiers = sorted({unit.identifier for unit in module_units}, key=int)[:3]
        for identifier in identifiers:
            candidates = [unit for unit in module_units if unit.identifier == identifier]
            if module in {"M2", "M4"}:
                first_variant = candidates[0].variant
                candidates = [unit for unit in candidates if unit.variant == first_variant]
            selected.extend(candidates)
    m5_units = [unit for unit in plan if unit.module == "M5"]
    cells: list[tuple[str, str]] = []
    for unit in m5_units:
        cell = (unit.identifier, unit.variant)
        if cell not in cells:
            cells.append(cell)
        if len(cells) == 2:
            break
    selected.extend(unit for unit in m5_units if (unit.identifier, unit.variant) in cells)
    return selected


def build_fault_controls(plan: list[WorkUnit]) -> dict[str, dict[str, str]]:
    selections = (
        ("model_failure", next(unit for unit in plan if unit.module == "M1" and unit.group == "cpu")),
        ("worker_crash", next(unit for unit in plan if unit.module == "M2")),
        ("timeout", next(unit for unit in plan if unit.module == "M4")),
    )
    controls: dict[str, dict[str, str]] = {}
    for fault, unit in selections:
        controls[unit.key] = {"fault": fault}
    controls[selections[0][1].key]["model"] = "lr"
    return controls


def _dry_run_worker(message: dict[str, Any]) -> dict[str, Any]:
    control = dict(message.get("fault_control", {}))
    attempt = int(message["attempt_number"])
    if control.get("fault") == "worker_crash" and attempt == 1:
        os._exit(86)
    if control.get("fault") == "timeout" and attempt == 1:
        time.sleep(1.0)

    failure_model = control.get("model") if control.get("fault") == "model_failure" else None
    def fake_fit(model: str, _train: Any, _labels: np.ndarray,
                 test: Any, _categorical: list[str], *_args: Any, **_kwargs: Any):
        if model == failure_model:
            raise RuntimeError("injected dry-run model failure")
        return np.full(len(test), 0.5, dtype=np.float64), {"fake_model": True}

    original = (
        execution.cpu_fit_predict, execution.foundation_fit_predict,
        execution.ftt_fit_predict, execution.registered_checkpoint_paths,
        execution._gpu_name,
    )
    unit = WorkUnit(**message["unit"])
    path = Path(message["checkpoint"])
    identity = CheckpointIdentity(**message["identity"])
    try:
        execution.cpu_fit_predict = fake_fit
        execution.foundation_fit_predict = fake_fit
        execution.ftt_fit_predict = _fake_ftt
        execution.registered_checkpoint_paths = lambda _root: {}
        execution._gpu_name = lambda: "FAKE MODEL — NO GPU MODEL LOADED"
        with exclusive_lock(path.with_suffix(path.suffix + ".lock")):
            payload = execute_checkpoint(
                Path(message["root"]), unit, path, identity, {},
                message["environment"], message.get("fatal_attempts", []),
            )
        return {"status": payload["log"]["status"], "task_key": unit.key}
    finally:
        (
            execution.cpu_fit_predict, execution.foundation_fit_predict,
            execution.ftt_fit_predict, execution.registered_checkpoint_paths,
            execution._gpu_name,
        ) = original


def run(root: Path, assignment: Path, driver_dir: Path, report: Path,
        fault_report: Path) -> None:
    full_plan = build_plan(root, assignment)
    plan = select_dry_run_plan(full_plan)
    counts = Counter(unit.module for unit in plan)
    required = {"M1", "M2", "M4", "M5", "M6"}
    if set(counts) != required:
        raise RuntimeError(f"dry-run plan lacks modules: {sorted(required - set(counts))}")

    identity_base = CheckpointIdentity(
        sha256_file(root / "MANIFEST.sha256"),
        sha256_file(driver_dir.parent / "phase1_driver_sha256.txt"),
        sha256_file(assignment), "",
    )
    def identity(task_key: str) -> CheckpointIdentity:
        return CheckpointIdentity(
            identity_base.registration_sha256, identity_base.driver_sha256,
            identity_base.host_assignment_sha256, task_key,
        )

    stages: dict[str, str] = {}
    temporary_root = Path(tempfile.mkdtemp(prefix="phase1_pool_dry_run_"))
    checkpoint_root = temporary_root / "checkpoints"
    output_root = temporary_root / "aggregate"
    try:
        environment = {
            "host": "dry-run", "registered_role": "dry-run",
            "fake_model": "fixed probability 0.5; no estimator fit",
        }
        controls = build_fault_controls(plan)
        supervised = []
        for unit in plan:
            path = checkpoint_path(checkpoint_root, unit)
            control = controls.get(unit.key, {})
            supervised.append(SupervisedTask(
                key=unit.key,
                payload={
                    "root": str(root), "unit": asdict(unit), "checkpoint": str(path),
                    "identity": asdict(identity(unit.key)), "environment": environment,
                    "fatal_attempts": [], "fault_control": control,
                },
                timeout_seconds=0.15 if control.get("fault") == "timeout" else 3600.0,
                retry_timeout_seconds=3600.0 if control.get("fault") == "timeout" else None,
            ))
        supervision = run_supervised_tasks(
            supervised, _dry_run_worker, worker_count=4,
            recycle_after=200, poll_interval=0.01,
            event_directory=temporary_root / "events",
        )
        for unit in plan:
            path = checkpoint_path(checkpoint_root, unit)
            saved = read_checkpoint(path, identity(unit.key))
            if saved["unit"] != asdict(unit):
                raise RuntimeError(f"dry-run checkpoint identity failed: {unit.key}")
        stages["data_read_variant_checkpoint"] = "passed"
        aggregate.aggregate_complete(
            checkpoint_root, output_root, plan, identity,
            modules=required, migrations=[],
        )
        stages["per_module_aggregation"] = "passed"
        for _key, group_units in aggregate.iter_plan_groups(plan):
            payloads = aggregate.load_complete_checkpoints(
                checkpoint_root, group_units, identity, reject_extras=False,
            )
            aggregate.validate_payload_uniqueness(payloads)
            aggregate.validate_prediction_counts(payloads)
        stages["self_checks"] = "passed"
        fault_payloads = {
            control["fault"]: read_checkpoint(
                checkpoint_path(checkpoint_root, next(unit for unit in plan if unit.key == key)),
                identity(key),
            )
            for key, control in controls.items()
        }
        model_payload = fault_payloads["model_failure"]
        model_failed = [row for row in model_payload["metrics"] if row["status"] == "failed"]
        if [row["model"] for row in model_failed] != ["lr"]:
            raise RuntimeError("dry-run model failure did not remain local to lr")
        if model_payload["log"]["status"] != "completed_with_failures":
            raise RuntimeError("model failure was not recorded as a completed unit")
        crash_attempts = fault_payloads["worker_crash"]["log"]["fatal_attempts"]
        timeout_attempts = fault_payloads["timeout"]["log"]["fatal_attempts"]
        if [row["error_type"] for row in crash_attempts] != ["worker_crash"]:
            raise RuntimeError("worker crash was not attributed and recovered")
        if [row["error_type"] for row in timeout_attempts] != ["timeout"]:
            raise RuntimeError("timeout was not attributed and recovered")
        if len(supervision.results) != len(plan):
            raise RuntimeError("supervisor lost dry-run units")
        fault_report.parent.mkdir(parents=True, exist_ok=True)
        fault_report.write_text(
            "\n".join((
                "完成", "", "# Phase 1 故障注入验证", "",
                "- 模型级失败：在一个 M1 CPU 单元中让 LR 的两次拟合尝试失败；该单元以 completed_with_failures 完成，其他模型继续，未计入连续单元失败。",
                "- 工作进程崩溃：一个 M2 单元首次尝试由工作进程无返回退出；父进程准确归因、重建工作进程，第二次尝试完成。",
                "- 单元超时：一个 M4 单元首次尝试超过测试限额；父进程结束并重建工作进程，第二次尝试完成。",
                f"- 致命事件数：{len(supervision.fatal_events)}；工作进程启动数：{supervision.worker_starts}；最大连续单元失败计数：{supervision.maximum_consecutive_unit_failures}。",
                "- 三种注入均通过检查点、按模块汇总和预测行数/身份自检；临时目录随后删除。", "",
            )), encoding="utf-8",
        )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join((
            "完成",
            "",
            "# Phase 1 池内固定输出空跑",
            "",
            "- 模型：固定输出 0.5 概率的假模型；没有拟合或加载任何真实模型。",
            "- 抽样：M1、M2、M4、M6 各 3 个池内数据集；M5 为 2 个合成格子。",
            f"- 计划单元数：{len(plan)}；分模块：{json.dumps(dict(sorted(counts.items())), sort_keys=True)}。",
            f"- 数据读取、变体生成与检查点：{stages['data_read_variant_checkpoint']}。",
            f"- 按模块汇总：{stages['per_module_aggregation']}。",
            f"- 身份、唯一性、预测行数与覆盖自检：{stages['self_checks']}。",
            "- 所有临时检查点和临时汇总已删除，未进入正式结果目录。",
            "",
        )),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered-root", required=True, type=Path)
    parser.add_argument("--host-assignment", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--fault-report", required=True, type=Path)
    args = parser.parse_args()
    run(
        args.registered_root.resolve(), args.host_assignment.resolve(),
        Path(__file__).resolve().parent, args.report.resolve(),
        args.fault_report.resolve(),
    )


if __name__ == "__main__":
    main()
