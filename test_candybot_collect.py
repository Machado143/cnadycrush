import types
import unittest
from unittest.mock import patch

import numpy as np

import candybot


class CollectTests(unittest.TestCase):
    def setUp(self):
        candybot.STOP = False

    def tearDown(self):
        candybot.STOP = False

    def _base_args(self, all_cells=False):
        return types.SimpleNamespace(
            seconds=300,
            interval=0.0,
            dedupe=0.1,
            delay=0,
            all_cells=all_cells,
        )

    def test_collect_default_saves_only_unknown(self):
        args = self._base_args(all_cells=False)
        saved = []

        crops = [
            np.full((12, 12, 3), 10, dtype=np.uint8),
            np.full((12, 12, 3), 220, dtype=np.uint8),
        ]

        def fake_cell_img(_img, _rows, _cols, _r, c):
            return crops[c].copy()

        with patch.object(candybot, "load_cfg", return_value={"rows": 1, "cols": 2, "board": {}}), \
             patch.object(candybot, "load_model", return_value={"X": np.zeros((1, 2304), np.float32), "y": np.array(["normal_azul"]), "thr": 1.0}), \
             patch.object(candybot, "wait_stable", return_value=np.zeros((24, 48, 3), np.uint8)), \
             patch.object(candybot, "cell_img", side_effect=fake_cell_img), \
             patch.object(candybot, "predict", return_value=["normal_azul", "?"]), \
             patch.object(candybot, "_save", side_effect=lambda crop, label, tag: saved.append((crop.copy(), label, tag))), \
             patch.object(candybot.keyboard, "add_hotkey", return_value=None), \
             patch.object(candybot, "countdown", return_value=None), \
             patch.object(candybot.time, "sleep", return_value=None), \
             patch.object(candybot.time, "time", side_effect=[0.0, 0.1, 0.2, 301.0]):
            candybot.cmd_collect(args)

        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0][1], "unlabeled")
        self.assertEqual(saved[0][2], "c")

    def test_collect_all_cells_saves_known_and_unknown_with_dedupe(self):
        args = self._base_args(all_cells=True)
        saved = []

        crops = [
            np.full((12, 12, 3), 40, dtype=np.uint8),
            np.full((12, 12, 3), 180, dtype=np.uint8),
        ]

        def fake_cell_img(_img, _rows, _cols, _r, c):
            return crops[c].copy()

        with patch.object(candybot, "load_cfg", return_value={"rows": 1, "cols": 2, "board": {}}), \
             patch.object(candybot, "load_model", return_value={"X": np.zeros((1, 2304), np.float32), "y": np.array(["normal_azul"]), "thr": 1.0}), \
             patch.object(candybot, "wait_stable", return_value=np.zeros((24, 48, 3), np.uint8)), \
             patch.object(candybot, "cell_img", side_effect=fake_cell_img), \
             patch.object(candybot, "predict", return_value=["normal_azul", "normal_roxo"]), \
             patch.object(candybot, "_save", side_effect=lambda crop, label, tag: saved.append((crop.copy(), label, tag))), \
             patch.object(candybot.keyboard, "add_hotkey", return_value=None), \
             patch.object(candybot, "countdown", return_value=None), \
             patch.object(candybot.time, "sleep", return_value=None), \
             patch.object(candybot.time, "time", side_effect=[0.0, 0.1, 0.2, 301.0]):
            candybot.cmd_collect(args)

        self.assertEqual(len(saved), 2)
        self.assertTrue(all(label == "unlabeled" for _, label, _ in saved))


if __name__ == "__main__":
    unittest.main()
