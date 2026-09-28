from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


class FakeGenerator:
    SYN_N_TEST = 8

    @staticmethod
    def synth_draw(_gen, _k, _rho, _auc, size, _rep, _split):
        rows = int(size)
        labels = np.resize(np.array([0, 1], dtype=int), rows)
        return np.zeros((rows, 3), dtype=float), labels, {}, []


class ExecuteCheckpointMatrixTests(unittest.TestCase):
    def test_every_module_group_combination_completes_with_fake_models(self):
        execution = importlib.import_module("phase1_driver.execution")
        checkpoints = importlib.import_module("phase1_driver.checkpoints")
        train = pd.DataFrame({"x": np.arange(12, dtype=float)})
        test = pd.DataFrame({"x": np.arange(4, dtype=float)})
        pool_views = (
            3, train, np.resize([0, 1], 12), np.arange(12),
            test, np.array([0, 1, 0, 1]), np.arange(4), [],
        )

        def fake_model(_model, _train, _labels, test_frame, *_args, **_kwargs):
            return np.full(len(test_frame), 0.5), {"fake": True}

        def fake_ftt(_train, _labels, test_frame, _categorical):
            return np.full(len(test_frame), 0.5), {"best_epoch": 0, "n_params": 0}

        combinations = (
            ("M1", "cpu"), ("M1", "gpu"), ("M1", "ftt"),
            ("M2", "cpu"), ("M2", "gpu"),
            ("M4", "cpu"), ("M4", "gpu"),
            ("M5", "cpu"), ("M5", "gpu"), ("M6", "none"),
        )
        with tempfile.TemporaryDirectory() as temporary, patch.multiple(
            execution,
            _pool_views=lambda *_args, **_kwargs: pool_views,
            _inner_scores_cpu=lambda *_args, **_kwargs: (
                {model: [0.5] * 5 for model in execution.config.CPU_MODELS}, {}
            ),
            _inner_scores_foundation=lambda *_args, **_kwargs: (
                {model: [0.5] * 5 for model in execution.config.FOUNDATION_MODELS}, {}
            ),
            cpu_fit_predict=fake_model,
            foundation_fit_predict=fake_model,
            ftt_fit_predict=fake_ftt,
            registered_checkpoint_paths=lambda _root: {},
            _generator=lambda _root: FakeGenerator(),
            run_m6=lambda _root, _identifier: {"0": {"x": 0.5}},
            _gpu_name=lambda: "FAKE RTX 4090",
        ):
            root = Path(temporary)
            for number, (module, group) in enumerate(combinations):
                identifier = "g|2|0.0|0.65" if module == "M5" else "3"
                variant = "n8_r0" if module in {"M4", "M5"} else (
                    "dispn_0.85_r0" if module == "M2" else "raw"
                )
                fold = -1 if module == "M6" else 0
                host = "4090" if group in {"gpu", "ftt"} else "686"
                unit = execution.WorkUnit(module, identifier, variant, fold, group, host)
                identity = checkpoints.CheckpointIdentity(
                    "a" * 64, "b" * 64, "c" * 64, unit.key
                )
                path = root / f"checkpoint_{number}.json"
                payload = execution.execute_checkpoint(
                    root, unit, path, identity, {}, {"host": "fake"}, []
                )
                self.assertEqual(payload["unit"]["module"], module)
                self.assertEqual(payload["unit"]["group"], group)
                self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
