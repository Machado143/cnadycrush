import types
import unittest
from pathlib import Path
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


class PredictTests(unittest.TestCase):
    def test_rare_class_wins_with_default_k(self):
        rng = np.random.default_rng(0)

        def normal():
            im = np.full((40, 40, 3), (200, 100, 30), np.int16) + rng.integers(-4, 5, (40, 40, 3))
            return np.clip(im, 0, 255).astype(np.uint8)

        def special(shift=0):
            im = np.full((40, 40, 3), (200, 100, 30), np.int16)
            for x in range(6 + shift, 36, 8):
                im[:, x:x + 3] = 255
            return np.clip(im + rng.integers(-4, 5, (40, 40, 3)), 0, 255).astype(np.uint8)

        X = [candybot.feat(normal()) for _ in range(30)] + [candybot.feat(special())]
        y = ["normal_azul"] * 30 + ["listrado_vert_azul"]
        model = {"X": np.array(X, np.float32), "y": np.array(y), "thr": 1e9}
        self.assertEqual(candybot.predict(model, [special(1)]), ["listrado_vert_azul"])
        model["thr"] = 0.01
        self.assertEqual(candybot.predict(model, [special(1)]), ["?"])


class ClusterTests(unittest.TestCase):
    def _crop(self, rng, color, stripe=False):
        im = np.full((40, 40, 3), color, np.int16)
        if stripe:
            for x in range(6, 36, 8):
                im[:, x:x + 3] = 255
        return np.clip(im + rng.integers(-3, 4, (40, 40, 3)), 0, 255).astype(np.uint8)

    def test_cluster_feats_separates_types(self):
        rng = np.random.default_rng(1)
        crops = [self._crop(rng, (200, 100, 30)) for _ in range(6)] + \
                [self._crop(rng, (30, 100, 200)) for _ in range(4)] + \
                [self._crop(rng, (200, 100, 30), stripe=True) for _ in range(2)]
        F = np.array([candybot.feat(c) for c in crops])
        sizes = sorted(len(g) for g in candybot.cluster_feats(F, 6.0))
        self.assertEqual(sizes, [2, 4, 6])

    def test_pick_spread_returns_unique_and_limited(self):
        F = np.random.default_rng(2).normal(size=(10, 5)).astype(np.float32)
        ch = candybot.pick_spread(F, list(range(10)), 4)
        self.assertEqual(len(ch), 4)
        self.assertEqual(len(set(ch)), 4)
        self.assertEqual(ch[0], 0)

    def test_cmd_cluster_moves_files(self):
        import tempfile
        from pathlib import Path
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "dataset"
            (data / "unlabeled").mkdir(parents=True)
            specs = [((200, 100, 30), False, 10), ((30, 100, 200), False, 6), ((200, 100, 30), True, 3)]
            for gi, (col, st, n) in enumerate(specs):
                for i in range(n):
                    candybot.cv2.imwrite(str(data / "unlabeled" / f"g{gi}_{i}.png"), self._crop(rng, col, st))
            keys = iter(["n", "5", "n", "2", "v", "5"])  # grupos por tamanho: 10, 6, 3
            args = types.SimpleNamespace(dist=6.0, keep=3)
            with patch.object(candybot, "DATA", data), \
                 patch.object(candybot, "ask", side_effect=lambda *a, **k: next(keys)), \
                 patch.object(candybot.cv2, "destroyAllWindows", return_value=None):
                candybot.cmd_cluster(args)
            self.assertEqual(len(list((data / "normal_azul").glob("*.png"))), 3)
            self.assertEqual(len(list((data / "normal_laranja").glob("*.png"))), 3)
            self.assertEqual(len(list((data / "listrado_vert_azul").glob("*.png"))), 3)
            self.assertEqual(len(list((data / "descartados").glob("*.png"))), 7 + 3)
            self.assertEqual(len(list((data / "unlabeled").glob("*.png"))), 0)


class VerifyTests(unittest.TestCase):
    def _crop(self, rng, color, stripe=False):
        im = np.full((40, 40, 3), color, np.int16)
        if stripe:
            for x in range(6, 36, 8):
                im[:, x:x + 3] = 255
        return np.clip(im + rng.integers(-3, 4, (40, 40, 3)), 0, 255).astype(np.uint8)

    def _build(self, tmp):
        from pathlib import Path
        rng = np.random.default_rng(5)
        data = Path(tmp) / "dataset"
        for lab, col, n in (("normal_azul", (200, 100, 30), 12), ("normal_verde", (60, 190, 40), 12)):
            (data / lab).mkdir(parents=True)
            for i in range(n):
                candybot.cv2.imwrite(str(data / lab / f"ok_{i}.png"), self._crop(rng, col))
        # erro plantado: doce verde dentro de normal_azul
        candybot.cv2.imwrite(str(data / "normal_azul" / "ERRADO.png"), self._crop(rng, (60, 190, 40)))
        (data / "unlabeled").mkdir()
        return data

    def test_audit_flags_planted_mislabel_only(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            data = self._build(tmp)
            with patch.object(candybot, "DATA", data):
                items, flags, hues = candybot.audit_dataset()
            bad = sorted(items[i][0].name for i, f in flags.items() if f)
            self.assertEqual(bad, ["ERRADO.png"])
            self.assertIn("normal_azul", hues)

    def test_cmd_verify_writes_sheet_and_moves(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            data = self._build(tmp)
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                args = types.SimpleNamespace(only="normal_", hue_tol=20.0, cols=10, show=False, move=True)
                with patch.object(candybot, "DATA", data):
                    candybot.cmd_verify(args)
                self.assertTrue((Path(tmp) / "verificacao" / "normal_azul.png").exists())
            finally:
                os.chdir(cwd)
            self.assertTrue((data / "unlabeled" / "ERRADO.png").exists())
            self.assertFalse((data / "normal_azul" / "ERRADO.png").exists())


if __name__ == "__main__":
    unittest.main()