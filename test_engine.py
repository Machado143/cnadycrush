import unittest

import engine as e


def grid(rows):
    """Texto -> células. Letra minúscula = normal; maiúscula: V=listrado vert, H=horiz, E=embrulhado
    (cor 'a'); '#' bloqueador; '.' inativa; '*' bomba."""
    out = []
    for line in rows:
        row = []
        for ch in line.split():
            if ch == "#":
                row.append(e.BLOCK)
            elif ch == ".":
                row.append(e.INACTIVE)
            elif ch == "*":
                row.append((None, e.B))
            elif ch == "V":
                row.append(("a", e.SV))
            elif ch == "H":
                row.append(("a", e.SH))
            elif ch == "E":
                row.append(("a", e.W))
            else:
                row.append((ch, e.N))
        out.append(row)
    return out


class EngineTests(unittest.TestCase):
    def test_simple_match3(self):
        g = grid(["a a b a",
                  "c d a d",
                  "d c d c"])
        self.assertIn(((0, 2), (1, 2)), [(a, b) for _, a, b in e.rank_moves(g)] +
                      [(b, a) for _, a, b in e.rank_moves(g)])

    def test_no_move_returns_none(self):
        g = grid(["a b c", "d e f", "g h i"])
        self.assertIsNone(e.best_move(g))

    def test_invalid_swap_returns_none(self):
        g = grid(["a b c", "d e f", "g h i"])
        self.assertIsNone(e.simulate(g, (0, 0), (0, 1)))

    def test_four_in_row_creates_striped_and_scores_more(self):
        three = grid(["a a b a a",
                      "c d a d c",
                      "d c d c d"])
        s4 = e.simulate(three, (0, 2), (1, 2))   # a a a a a -> 5 em linha => bomba
        self.assertIsNotNone(s4)
        four = grid(["a a b a",
                     "c d a d",
                     "d c d c"])
        s4b = e.simulate(four, (0, 2), (1, 2))   # a a a a => listrado
        self.assertGreater(s4b, 4)
        self.assertGreater(s4, s4b)

    def test_creation_kinds(self):
        runs = [("h", [(0, 0), (0, 1), (0, 2), (0, 3)])]
        gp = e.merge_runs(runs)[0]
        self.assertEqual(e.creation(gp, ())[1], e.SV)
        runs = [("v", [(0, 0), (1, 0), (2, 0), (3, 0)])]
        self.assertEqual(e.creation(e.merge_runs(runs)[0], ())[1], e.SH)
        runs = [("h", [(2, 0), (2, 1), (2, 2)]), ("v", [(0, 0), (1, 0), (2, 0)])]
        self.assertEqual(e.creation(e.merge_runs(runs)[0], ())[1], e.W)
        runs = [("h", [(0, j) for j in range(5)])]
        self.assertEqual(e.creation(e.merge_runs(runs)[0], ())[1], e.B)

    def test_striped_clears_line(self):
        g = grid(["b c d",
                  "c b c",
                  "V d b"])
        # cria match de 'a'? não: apenas verifica efeito direto via expand
        cleared, _ = e.expand(g, {(2, 0)})
        self.assertEqual(cleared, {(0, 0), (1, 0), (2, 0)})  # SV limpa a coluna

    def test_wrapped_clears_3x3(self):
        g = grid(["b c d b",
                  "c E c d",
                  "d b d c"])
        cleared, _ = e.expand(g, {(1, 1)})
        self.assertEqual(len(cleared), 9)

    def test_gravity_and_cascade(self):
        g = grid(["a b c",
                  "a b c",
                  "d d a",
                  "b a b"])
        # trocar (2,2)<->(3,2)? mover 'a' de (3,1) para (2,... ) -> testa cascata determinística
        g2 = [row[:] for row in g]
        e.gravity(g2)
        self.assertEqual(g2, g)  # sem vazios, nada muda
        g2[3][0] = e.CLEAR
        e.gravity(g2)
        self.assertEqual(g2[0][0], e.CLEAR)
        self.assertEqual(g2[1][0], ("a", e.N))
        self.assertEqual(g2[3][0], ("d", e.N))

    def test_gravity_respects_blockers(self):
        g = grid(["a b", "# c", "d e"])
        g[2][0] = e.CLEAR
        e.gravity(g)
        self.assertEqual(g[1][0], e.BLOCK)
        self.assertEqual(g[2][0], e.CLEAR)  # bloqueador impede a queda

    def test_bomb_with_normal_clears_color(self):
        g = grid(["a * b a",
                  "b c d c",
                  "a b a d"])
        s = e.simulate(g, (0, 1), (0, 0))  # bomba + 'a' -> limpa todos os 'a'
        self.assertIsNotNone(s)
        self.assertGreaterEqual(s, 4)  # 4 'a' + bomba

    def test_bomb_swap_valid_without_match(self):
        g = grid(["* b", "c d"])
        self.assertIsNotNone(e.simulate(g, (0, 0), (0, 1)))

    def test_two_striped_swap(self):
        g = grid(["b c d",
                  "c V H",
                  "d b c"])
        s = e.simulate(g, (1, 1), (1, 2))
        self.assertIsNotNone(s)
        self.assertGreaterEqual(s, 5)

    def test_unknown_and_blocker_not_swappable(self):
        g = grid(["a #", "b c"])
        g[1][1] = e.UNK
        self.assertIsNone(e.simulate(g, (0, 0), (0, 1)))
        self.assertIsNone(e.simulate(g, (1, 0), (1, 1)))

    def test_parse_label(self):
        self.assertEqual(e.parse_label("normal_azul"), ("azul", e.N))
        self.assertEqual(e.parse_label("listrado_vert_azul"), ("azul", e.SV))
        self.assertEqual(e.parse_label("listrado_horiz_roxo"), ("roxo", e.SH))
        self.assertEqual(e.parse_label("embrulhado_verde"), ("verde", e.W))
        self.assertEqual(e.parse_label("peixe_azul"), ("azul", e.F))
        self.assertEqual(e.parse_label("bomba"), (None, e.B))
        self.assertEqual(e.parse_label("?"), e.UNK)

    def test_special_creation_scores_above_plain_match(self):
        plain = grid(["a a b d",
                      "c d a c",
                      "d c d c"])
        four = grid(["a a b a",
                     "c d a d",
                     "d c d c"])
        s_plain = e.simulate(plain, (0, 2), (1, 2))   # 3 em linha
        s_four = e.simulate(four, (0, 2), (1, 2))     # 4 em linha -> listrado
        self.assertGreater(s_four, s_plain)

    def test_goal_color_changes_choice(self):
        g = grid(["a a b d e e f g",
                  "c d a c h g e h",
                  "d c d c g h g h"])
        wa = e.with_goal("a")
        we = e.with_goal("e")
        a1, b1 = e.best_move(g, wa)
        a2, b2 = e.best_move(g, we)
        self.assertEqual({a1, b1}, {(0, 2), (1, 2)})
        self.assertEqual({a2, b2}, {(0, 6), (1, 6)})

    def test_goal_info_counts_target_candies(self):
        g = grid(["a a b d", "c d a c", "d c d c"])
        info = {}
        e.simulate(g, (0, 2), (1, 2), e.with_goal("a"), info)
        self.assertEqual(info["goal"], 3)
        info = {}
        e.simulate(g, (0, 2), (1, 2), e.with_goal("z"), info)
        self.assertEqual(info["goal"], 0)

    def test_runs_on_8x8_fast(self):
        import random
        import time
        random.seed(1)
        g = [[(random.choice("abcd"), e.N) for _ in range(8)] for _ in range(8)]
        t0 = time.time()
        e.rank_moves(g)
        self.assertLess(time.time() - t0, 1.0)


if __name__ == "__main__":
    unittest.main()
