"""Tests for the persistent, incremental concept library (the live value signal)."""
import itertools
import sqlite3

import numpy as np

from wise_explorer import synthesis as S
from wise_explorer.memory.concept_library import ConceptLibrary


def _xor_data():
    """Synthetic Nim: 3 heaps 0..3, WIN iff the nim-sum is 0 (cell-only, so move is irrelevant)."""
    B = np.array(list(itertools.product(range(4), repeat=3)), dtype=np.int64)
    xor = B[:, 0] ^ B[:, 1] ^ B[:, 2]
    V = np.where(xor == 0, 0.95, 0.05)
    return B, V, np.zeros(len(B), dtype=np.int64)


class TestSerialization:
    def test_programs_round_trip(self):
        progs = [
            S.Cell(2), S.Lit(1), S.BinOp("⊕", S.Cell(0), S.Cell(1)),
            S.Fold("⊕", S.CellDomain((0, 1, 2)), S.Elem(0, S.CellDomain.names)),
            S.Fold("max", S.GroupDomain([(0, 1, 2)]),
                   S.BinOp("min", S.Elem(0, S.GroupDomain.names), S.Elem(1, S.GroupDomain.names))),
        ]
        B = np.array([[1, 2, 3], [0, 1, 2]], dtype=np.int64); m = np.array([1, 1])
        for e in progs:
            e2 = S.expr_from_dict(S.expr_to_dict(e))
            assert str(e2) == str(e)
            assert np.array_equal(e.eval(B, m), e2.eval(B, m))


class TestConceptLibrary:
    def test_grow_persist_reload(self):
        B, V, M = _xor_data()
        conn = sqlite3.connect(":memory:")
        lib = ConceptLibrary(conn, read_only=False)
        assert lib.refresh(B, V, M, max_size=5) >= 1 and lib.rules   # discovered the nim-sum
        win = lib.value_for(np.array([0, 0, 0]))                  # nim-sum 0
        loss = lib.value_for(np.array([1, 0, 0]))                 # nim-sum ≠ 0
        assert win is not None and loss is not None and win > loss
        # a read-only reopen on the same DB (a worker) reloads the same value model
        reloaded = ConceptLibrary(conn, read_only=True)
        assert len(reloaded.rules) == len(lib.rules)
        assert reloaded.value_for(np.array([0, 0, 0])) == win

    def test_regrow_on_same_data_is_stable(self):
        B, V, M = _xor_data()
        lib = ConceptLibrary(sqlite3.connect(":memory:"))
        n1 = lib.refresh(B, V, M, max_size=5)
        n2 = lib.refresh(B, V, M, max_size=5)                     # no novel structure → no re-search
        assert n1 == n2

    def test_empty_library_is_inert(self):
        lib = ConceptLibrary(sqlite3.connect(":memory:"))
        assert lib.value_for(np.array([1, 2, 3])) is None         # nothing discovered → no signal
