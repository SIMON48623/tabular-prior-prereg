from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class FakePhase0B:
    def __init__(self):
        self.prepared = False
        self.scored = 0

    def prepare_feature_frame(self, frame, categorical_map):
        self.prepared = True
        return frame.astype(float), list(frame.columns), []

    def single_feature_percentile_scores(self, train, test, categorical):
        self.scored += 1
        ordered = np.sort(train.to_numpy(dtype=float))
        return (
            np.searchsorted(ordered, train, side="right") / len(ordered),
            np.searchsorted(ordered, test, side="right") / len(ordered),
        )


class ModelFreeCheckTests(unittest.TestCase):
    def test_marginal_recalculation_uses_prepare_then_registered_folds(self):
        checks = __import__("phase1_driver.model_free_checks", fromlist=["x"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cleaned_data").mkdir()
            frame = pd.DataFrame({
                "n": np.arange(20, dtype=float),
                "y": np.tile([0, 1], 10),
                "fold": np.repeat(np.arange(5), 4),
            })
            frame.to_parquet(root / "cleaned_data" / "3.parquet", index=False)
            (root / "cleaned_data" / "3.schema.json").write_text(
                json.dumps({
                    "columns": {"n": {"type": "numeric"}},
                    "label": {
                        "positive_phase0b_encoded_class": 1,
                        "counts_before_flip": [10, 10],
                    },
                }), encoding="utf-8"
            )
            phase0b = FakePhase0B()
            got = checks.recompute_dataset_marginals(root, 3, phase0b)
            self.assertTrue(phase0b.prepared)
            self.assertEqual(phase0b.scored, 5)
            self.assertEqual(set(got), {"n"})

    def test_target_region_is_read_from_frozen_stratifier_and_exactly_238(self):
        checks = __import__("phase1_driver.model_free_checks", fromlist=["x"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "frozen").mkdir()
            pd.DataFrame({
                "openml_id": range(1, 240),
                "target_region": [True] * 238 + [False],
            }).to_csv(root / "frozen" / "stratifier_v2.csv", index=False)
            identifiers = checks.target_region_ids(root)
            self.assertEqual(len(identifiers), 238)
            self.assertNotIn(239, identifiers)

    def test_concentrated_signal_position_uses_frozen_seed_formula(self):
        checks = __import__("phase1_driver.model_free_checks", fromlist=["x"])

        class Generator:
            INJECT_K = 8

            @staticmethod
            def seed_for(*parts):
                digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
                return int.from_bytes(digest[:4], "little")

        first = checks.concentrated_signal_position(Generator, 31, 2)
        second = checks.concentrated_signal_position(Generator, 31, 2)
        self.assertEqual(first, second)
        self.assertIn(first, range(8))


if __name__ == "__main__":
    unittest.main()
