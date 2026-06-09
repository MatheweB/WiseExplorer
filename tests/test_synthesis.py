"""Tests for the concept-invention engine (wise_explorer.synthesis).

The whole engine is built from one combinator: ``fold(op, domain, body)``. These tests
exercise (a) folds as evaluable/printable programs, (b) perspective-relativisation,
(c) that threats/forks are discovered as folds over discovered groups, and (d) the
end-to-end round loop. The expensive round-loop is run once in a module-scoped fixture,
bounded to ``max_size=5`` (the nim-sum of three heaps is a size-2 fold).
"""
import itertools
import time

import numpy as np
import pytest

from wise_explorer import synthesis as S


def _xor_data():
    """Synthetic Nim: 3 heaps 0..3, WIN iff the xor (nim-sum) is 0."""
    B = np.array(list(itertools.product(range(4), repeat=3)), dtype=np.int64)
    xor = B[:, 0] ^ B[:, 1] ^ B[:, 2]
    V = np.where(xor == 0, 0.95, 0.05)
    return B, V


@pytest.fixture(scope="module")
def xor_invention():
    B, V = _xor_data()
    res = S.invent_from_boards(B, V, max_rounds=4, max_size=5, cap=2000)
    return B, V, res


class TestFoldAST:
    def test_cell_group_walks_through_ops(self):
        e = S.BinOp("⊕", S.Cell(0), S.BinOp("+", S.Cell(2), S.Cell(0)))
        assert S._cell_group(e) == (0, 2)

    def test_cell_group_sees_through_named(self):
        inner = S.BinOp("&", S.Cell(3), S.Cell(5))
        named = S.Named(inner, np.zeros(1, dtype=np.int64))
        assert S._cell_group(named) == (3, 5)

    def test_cell_group_sees_through_a_group_fold(self):
        fold = S.Fold("max", S.GroupDomain([(0, 1, 2), (3, 4, 5)]),
                      S.Elem(0, S.GroupDomain.names))
        assert S._cell_group(fold) == (0, 1, 2, 3, 4, 5)

    def test_fold_over_cells_is_a_reduction(self):
        fold = S.Fold("⊕", S.CellDomain((0, 1, 2)), S.Elem(0, S.CellDomain.names))
        B = np.array([[1, 2, 3], [3, 3, 3]], dtype=np.int64)
        assert list(fold.eval(B)) == [1 ^ 2 ^ 3, 3 ^ 3 ^ 3]
        assert str(fold) == "fold(⊕, cells, cell)"


class TestInvention:
    def test_invents_the_xor_separator(self, xor_invention):
        B, _, res = xor_invention
        xor0 = ((B[:, 0] ^ B[:, 1] ^ B[:, 2]) == 0)
        assert any(np.array_equal(c.mask, xor0) for c in res.concepts), \
            "did not invent a concept equivalent to nim-sum == 0"

    def test_produces_win_and_loss_rules(self, xor_invention):
        _, _, res = xor_invention
        verdicts = {r.verdict for r in res.rules}
        assert "WIN" in verdicts and "LOSS" in verdicts

    def test_round_loop_stops(self, xor_invention):
        _, _, res = xor_invention
        assert res.stopped_after == 1            # nim-sum explains everything; later rounds add nothing
        assert any(not r.kept for r in res.rounds)

    def test_concept_runs_on_unseen_board(self, xor_invention):
        _, _, res = xor_invention
        assert isinstance(bool(res.concepts[0].holds(np.array([0, 0, 0]))), bool)

    def test_too_little_data_is_graceful(self):
        B = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
        res = S.invent_from_boards(B, np.array([1.0, 0.0]), max_rounds=2, max_size=5)
        assert res.rounds == [] or res.stopped_after == 0


class TestGroupCounting:
    def test_counts_played_and_empty_at_face_value(self):
        # the board is never recoded — counts compare to the move (== m) and to empty (== 0);
        # a piece that is neither (the '3') keeps its value and is simply not counted.
        B = np.array([[1, 1, 0, 2], [3, 1, 0, 2]], dtype=np.int64)
        T = S.GroupDomain([(0, 1, 2)]).tensor(B, np.array([1, 1]))   # both boards just played token 1
        assert T[0, 0].tolist() == [2, 1]    # line (1,1,0): played(==1)=2, empty=1
        assert T[1, 0].tolist() == [1, 1]    # line (3,1,0): played=1, empty=1; the 3 is neither

    def test_supports_are_atomic_cell_regions_only(self):
        line = S.Concept(S.BinOp("&", S.Cell(0), S.BinOp("&", S.Cell(1), S.Cell(2))),
                         "=", 0, np.zeros(4, dtype=bool), 5)
        assert S._supports([line]) == [(0, 1, 2)]        # an atomic cell region → folded over
        # a combination reuses an earlier concept (a Named block) — a union, skipped
        combo = S.Concept(S.BinOp("|", S.Named(line.expr, np.zeros(4, dtype=np.int64)), S.Cell(3)),
                          "=", 0, np.zeros(4, dtype=bool), 3)
        assert S._supports([combo]) == []
        # the nim-sum is a whole-board fold, not an atomic cell region — skipped
        nimsum = S.Concept(S.Fold("⊕", S.CellDomain((0, 1, 2, 3)), S.Elem(0, S.CellDomain.names)),
                           "=", 0, np.zeros(4, dtype=bool), 2)
        assert S._supports([nimsum]) == []


class TestDiscoveredThreats:
    def test_residual_is_zero_under_a_perfect_fit(self):
        V = np.array([1.0, 1.0, 0.0, 0.0])
        c = S.Concept(S.Cell(0), "=", 1, np.array([True, True, False, False]), 1)
        rules = [S.Rule([(c, True)], "WIN", 2, 1.0, 0.0),
                 S.Rule([(c, False)], "LOSS", 2, 0.0, 0.0)]
        assert np.allclose(S._residual(rules, V), 0.0)

    def test_group_fold_candidates_are_folds_over_groups(self):
        rng = np.random.RandomState(0)
        B = rng.randint(0, 3, size=(80, 9)).astype(np.int64)     # 9 cells, tokens {0,1,2}
        supports = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
        m = np.ones(80, dtype=np.int64)                          # everyone just played token 1
        target = rng.rand(80) - 0.5
        cands = S._group_fold_candidates(supports, B, target, min_leaf=5, m=m)
        assert cands, "expected at least one group-fold candidate"
        for c in cands:
            assert isinstance(c.expr, S.Fold) and isinstance(c.expr.domain, S.GroupDomain)
            assert c.expr.op in ("max", "+")
            assert str(c.expr).startswith("fold(") and "groups" in str(c.expr)
        assert S._group_fold_candidates([], B, target, 5, m) == []   # no discovered groups → nothing

    def test_group_fold_is_printable_and_holds(self):
        body = S.BinOp("min", S.Elem(0, S.GroupDomain.names),
                       S.BinOp("+", S.Elem(1, S.GroupDomain.names), S.Elem(1, S.GroupDomain.names)))
        fold = S.Fold("max", S.GroupDomain([(0, 1, 2)]), body)
        s = str(fold)
        assert s.startswith("fold(max, groups,") and "played" in s and "empty" in s
        assert fold.size == 1 + body.size
        B = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0],
                      [1, 2, 2, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        m = np.array([1, 1])
        vec = fold.eval(B, m)
        assert vec.shape == (2,) and vec.dtype == np.int64
        c = S.Concept(fold, "=", 1, vec == 1, fold.size)
        assert isinstance(bool(c.holds(B[0], 1)), bool)             # unseen board + its move


class TestPerformance:
    def test_bounded_search_is_fast(self):
        """Regression guard: a small instance must finish in seconds, not minutes."""
        B, V = _xor_data()
        t0 = time.perf_counter()
        S.invent_from_boards(B, V, max_rounds=2, max_size=5, cap=2000)
        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, f"bounded invent took {elapsed:.1f}s (expected < 30s)"
