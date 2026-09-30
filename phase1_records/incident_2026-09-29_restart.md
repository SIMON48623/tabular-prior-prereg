# Execution record: CPU scheduler restart on 2026-09-29

This is a record of an operational incident. It is not a deviation from the plan: no registered
analysis, model or setting changed, and no model output was altered. Times are UTC.

## What happened

1. **Stop for D3 (about 11:45).** To write the D3 records, the two CPU schedulers (hosts 214 and
   686) were stopped by sending SIGTERM to the scheduler process.
   - The frozen supervisor has no signal handling. Its worker processes, started with the
     `spawn` method, did not exit with the scheduler. They became orphans (parent PID 1), kept
     computing their current units and kept holding those units' checkpoint locks.
   - The check before restarting inspected only descendants of the scheduler process, so it
     missed the orphans.
2. **Records written.** The D3 records were written as described in D3, with the counts given
   there. Every one was read back without error.
3. **Restart (214 at 11:51:40, 686 at 11:57:15).** The first units dispatched were the ones the
   orphans still held.
   - Each attempt failed with `RuntimeError: lock already held`.
   - After 20 consecutive failures, each scheduler stopped under its registered rule.
   - The restarted schedulers wrote failed checkpoints for units that reached two attempts.
   - Meanwhile, orphans that finished their units wrote normal checkpoints.
4. **Diagnosis (about 12:15).** The 40 new failure events were all lock errors on the units that
   were running at the stop (`evidence/d3_restart_diag_214_redacted.zip`,
   `evidence/d3_restart_diag_686_redacted.zip`).
5. **Full stop (about 12:25).**
   - All processes of the run were found system-wide, including orphans.
   - They were terminated: SIGTERM to the scheduler, then SIGKILL to the rest.
   - Checks: zero related processes, zero open checkpoint locks, and no `/proc/locks` entry on a
     checkpoint file.
6. **Clean-up.** Every original was archived on the same file system, and hashes were taken
   before and after.
   - **214:**
     - Seven checkpoints whose failure came from the lock error were archived, so these units
       run again: M1 995 folds 0–4, M1 920 fold 3, and M2 1065 `conc_0.95_r0` fold 2.
     - Six checkpoints completed by orphan workers were kept.
     - Nineteen fatal-attempt files had 20 lock-error entries between them, and those entries were
       removed. Twelve files became empty and were archived. Seven were rewritten keeping each
       unit's earlier, genuine time-out entry: M1 995 folds 0–4, M1 920 fold 3, and M2 1065
       `conc_0.95_r0` fold 2.
   - **686:**
     - No checkpoint was archived.
     - Nineteen checkpoints completed by orphan workers were kept.
     - Twenty fatal-attempt files held only lock-error entries. All were emptied and archived.
   - `scheduler_failures.jsonl` was not modified on either host. It keeps the 40 lock-error events
     as history.
   - A first attempt to archive to another file system failed on the first file (cross-device
     rename), and nothing was moved. The archive was then placed on the checkpoints' file system.
7. **Restarts.**
   - At 12:43, 214 was restarted and then stopped again after 15 minutes. The stop came from an
     observation rule that wrongly required a completed unit within 15 minutes. The run itself was
     healthy: 32 workers, 32 unit locks and no new failures.
   - A verify-only run on 686 then failed because a reconnected shell resolved a Python without
     the Phase 1 environment.
   - Both schedulers were restarted at 13:13:04 (214) and 13:13:05 (686), with the complete
     environment recorded from the original scheduler processes, the original command lines and
     the original working directories.
   - The health check at about 15.5 minutes passed on both hosts: scheduler alive, 32 workers,
     32 worker-held unit locks, no new failure events and no traceback
     (`evidence/health_15min_214.json`, `evidence/health_15min_686.json`,
     `evidence/restart_round_summary.md`).
   - Within two hours, both hosts had written new checkpoints with status `ok` and the frozen
     driver's manifest hash.

## Effect

- No model output was altered. Checkpoints are written atomically.
- The units completed by orphan workers ran the frozen code in the original environment, so their
  results are valid.
- The CPU instances did not dispatch new units for about 1.4 hours. The GPU instance was not
  touched.

## Rule adopted

From now on, stopping a scheduler means:
- terminating every process of the run, found system-wide, including orphaned workers; and
- confirming that no process holds a checkpoint lock before any restart.

Every restart uses the environment recorded from the original scheduler.
