from __future__ import annotations

import json
import multiprocessing
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class ConsecutiveUnitFailures(RuntimeError):
    pass


class SchedulerFatalError(RuntimeError):
    pass


@dataclass(frozen=True)
class SupervisedTask:
    key: str
    payload: dict[str, Any]
    timeout_seconds: float
    retry_timeout_seconds: float | None = None


@dataclass
class SupervisionReport:
    results: list[dict[str, Any]]
    fatal_events: list[dict[str, Any]]
    worker_starts: int
    recycled_workers: int
    maximum_consecutive_unit_failures: int


@dataclass
class _Slot:
    process: Any
    connection: Any
    active: tuple[int, SupervisedTask] | None = None
    started_at: float | None = None
    dispatched_at: float = 0.0
    active_timeout_seconds: float = 0.0
    completed: int = 0


def _worker_loop(connection: Any, operation: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    try:
        while True:
            message = connection.recv()
            if message is None:
                return
            connection.send({"control": "started"})
            try:
                result = operation(message)
            except BaseException as exc:
                result = {
                    "status": "fatal", "error_type": "worker_exception",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            connection.send(result)
    finally:
        connection.close()


def _append_event(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_supervised_tasks(
    tasks: list[SupervisedTask],
    operation: Callable[[dict[str, Any]], dict[str, Any]],
    worker_count: int,
    *,
    recycle_after: int = 200,
    poll_interval: float = 0.1,
    event_directory: Path | None = None,
    initial_fatal_attempts: dict[str, list[dict[str, Any]]] | None = None,
    on_fatal: Callable[[SupervisedTask, dict[str, Any]], None] | None = None,
    on_result: Callable[[SupervisedTask, dict[str, Any]], None] | None = None,
) -> SupervisionReport:
    if worker_count <= 0 or recycle_after <= 0:
        raise ValueError("worker_count and recycle_after must be positive")
    context = multiprocessing.get_context("spawn")
    queued: deque[tuple[int, SupervisedTask]] = deque(enumerate(tasks))
    attempts = {
        task.key: list((initial_fatal_attempts or {}).get(task.key, []))
        for task in tasks
    }
    result_by_index: dict[int, dict[str, Any]] = {}
    fatal_events: list[dict[str, Any]] = []
    worker_starts = 0
    recycled_workers = 0
    consecutive_failures = 0
    maximum_failures = 0

    def start_slot() -> _Slot:
        nonlocal worker_starts
        parent, child = context.Pipe(duplex=True)
        process = context.Process(target=_worker_loop, args=(child, operation))
        process.start()
        child.close()
        worker_starts += 1
        return _Slot(process=process, connection=parent)

    def stop_slot(slot: _Slot, terminate: bool = False) -> None:
        if slot.process.is_alive():
            if terminate:
                slot.process.terminate()
            else:
                try:
                    slot.connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
        slot.process.join(timeout=5.0)
        if slot.process.is_alive():
            slot.process.kill()
            slot.process.join(timeout=5.0)
        slot.connection.close()

    def dispatch(slot: _Slot, indexed_task: tuple[int, SupervisedTask]) -> None:
        index, task = indexed_task
        fatal_attempt_count = len(attempts[task.key])
        message = dict(task.payload)
        message["attempt_number"] = fatal_attempt_count + 1
        if "fatal_attempts" in message:
            message["fatal_attempts"] = list(attempts[task.key])
        slot.connection.send(message)
        slot.active = (index, task)
        slot.started_at = None
        slot.dispatched_at = time.monotonic()
        slot.active_timeout_seconds = (
            task.retry_timeout_seconds
            if fatal_attempt_count and task.retry_timeout_seconds is not None
            else task.timeout_seconds
        )

    def fatal(slot: _Slot, error_type: str, reason: str) -> _Slot:
        nonlocal consecutive_failures, maximum_failures
        assert slot.active is not None
        index, task = slot.active
        event = {
            "task_key": task.key, "attempt": len(attempts[task.key]) + 1,
            "error_type": error_type, "reason": reason, "time_unix": time.time(),
        }
        attempts[task.key].append(event)
        fatal_events.append(event)
        if event_directory is not None:
            _append_event(event_directory / "fatal_events.jsonl", event)
        if on_fatal is not None:
            on_fatal(task, event)
        consecutive_failures += 1
        maximum_failures = max(maximum_failures, consecutive_failures)
        stop_slot(slot, terminate=True)
        replacement = start_slot()
        if len(attempts[task.key]) < 2:
            queued.appendleft((index, task))
        else:
            result = {
                "status": "failed", "task_key": task.key,
                "error_type": error_type, "reason": reason,
                "fatal_attempts": list(attempts[task.key]),
            }
            result_by_index[index] = result
            if on_result is not None:
                on_result(task, result)
        if consecutive_failures >= 20:
            raise ConsecutiveUnitFailures(
                f"stage stopped after 20 consecutive unit failures; last={task.key}"
            )
        return replacement

    slots = [start_slot() for _ in range(min(worker_count, max(1, len(tasks))))]
    try:
        while queued or any(slot.active is not None for slot in slots):
            for slot in slots:
                if slot.active is None and queued:
                    dispatch(slot, queued.popleft())
            progressed = False
            for number, slot in enumerate(list(slots)):
                if slot.active is None:
                    continue
                if slot.connection.poll():
                    progressed = True
                    try:
                        result = slot.connection.recv()
                    except (EOFError, OSError) as exc:
                        slots[number] = fatal(slot, "worker_crash", f"worker pipe closed: {exc}")
                        continue
                    if result.get("control") == "started":
                        slot.started_at = time.monotonic()
                        continue
                    if result.get("status") == "fatal":
                        error_type = str(result.get("error_type", "process_fatal"))
                        reason = str(result.get("reason", "worker returned fatal status"))
                        if error_type == "weight_attestation_fatal":
                            index, task = slot.active
                            event = {
                                "task_key": task.key,
                                "attempt": len(attempts[task.key]) + 1,
                                "error_type": error_type,
                                "reason": reason,
                                "time_unix": time.time(),
                            }
                            attempts[task.key].append(event)
                            fatal_events.append(event)
                            if event_directory is not None:
                                _append_event(event_directory / "fatal_events.jsonl", event)
                            if on_fatal is not None:
                                on_fatal(task, event)
                            raise SchedulerFatalError(
                                f"scheduler stopped by {error_type} for {task.key}: {reason}"
                            )
                        slots[number] = fatal(
                            slot, error_type, reason,
                        )
                        continue
                    index, task = slot.active
                    result_by_index[index] = dict(result)
                    if on_result is not None:
                        on_result(task, dict(result))
                    # A model-level failure is a completed unit under Section 7.
                    if result.get("status") in {"failed", "scheduler_error"}:
                        consecutive_failures += 1
                        maximum_failures = max(maximum_failures, consecutive_failures)
                    else:
                        consecutive_failures = 0
                    slot.active = None
                    slot.completed += 1
                    if consecutive_failures >= 20:
                        raise ConsecutiveUnitFailures(
                            f"stage stopped after 20 consecutive failed units; last={task.key}"
                        )
                    if slot.completed >= recycle_after:
                        stop_slot(slot)
                        slots[number] = start_slot()
                        recycled_workers += 1
                elif not slot.process.is_alive():
                    progressed = True
                    code = slot.process.exitcode
                    slots[number] = fatal(
                        slot, "worker_crash", f"worker exited without result; exitcode={code}",
                    )
                elif slot.started_at is not None and time.monotonic() - slot.started_at > slot.active_timeout_seconds:
                    progressed = True
                    slots[number] = fatal(
                        slot, "timeout",
                        f"unit exceeded {slot.active_timeout_seconds:.6f} seconds",
                    )
                elif slot.started_at is None and time.monotonic() - slot.dispatched_at > 300.0:
                    progressed = True
                    slots[number] = fatal(
                        slot, "worker_start_timeout",
                        "worker did not acknowledge unit start within 300 seconds",
                    )
            if not progressed:
                time.sleep(poll_interval)
    finally:
        for slot in slots:
            stop_slot(slot, terminate=slot.active is not None)
    return SupervisionReport(
        results=[result_by_index[index] for index in range(len(tasks))],
        fatal_events=fatal_events, worker_starts=worker_starts,
        recycled_workers=recycled_workers,
        maximum_consecutive_unit_failures=maximum_failures,
    )
