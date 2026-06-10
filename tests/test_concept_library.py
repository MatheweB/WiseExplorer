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


def _xor_wave():
    """The xor data shaped as one training wave: per-transition keys, boards, scores."""
    B, V, _ = _xor_data()
    boards = {f"t{i}": B[i] for i in range(len(B))}
    trans = {(f"f{i}", f"t{i}"): ((1, 0, 0), float(V[i])) for i in range(len(B))}
    return list(trans.keys()), boards, trans


class TestConceptLibrary:
    def test_rebuild_persist_reload(self):
        B, V, M = _xor_data()
        conn = sqlite3.connect(":memory:")
        lib = ConceptLibrary(conn, read_only=False)
        assert lib.rebuild(B, V, M, max_size=5) >= 1 and lib.rules   # discovered the nim-sum
        win = lib.value_for(np.array([0, 0, 0]))                  # nim-sum 0
        loss = lib.value_for(np.array([1, 0, 0]))                 # nim-sum ≠ 0
        assert win is not None and loss is not None and win > loss
        # a read-only reopen on the same DB (a worker) reloads the same value model
        reloaded = ConceptLibrary(conn, read_only=True)
        assert len(reloaded.rules) == len(lib.rules)
        assert reloaded.value_for(np.array([0, 0, 0])) == win

    def test_regrow_on_same_wave_is_stable(self):
        keys, boards, trans = _xor_wave()
        lib = ConceptLibrary(sqlite3.connect(":memory:"))
        n1 = lib.grow(keys, boards, trans, max_size=5)            # cold start → one search
        assert n1 >= 1 and lib.rules
        n2 = lib.grow(keys, boards, trans, max_size=5)            # same data → no re-search
        assert n1 == n2

    def test_tiny_wave_on_resumed_db_keeps_rules(self):
        # a resumed run whose first wave holds <8 boards must not wipe the persisted model
        B, V, M = _xor_data()
        conn = sqlite3.connect(":memory:")
        ConceptLibrary(conn).rebuild(B, V, M, max_size=5)
        resumed = ConceptLibrary(conn)                            # reload: live table starts empty
        keys, boards, trans = _xor_wave()
        resumed.grow(keys[:2], boards, trans)                     # 2 boards < 8
        assert resumed.rules and ConceptLibrary(conn, read_only=True).rules

    def test_seed_from_carries_programs_only(self):
        B, V, M = _xor_data()
        src = sqlite3.connect(":memory:")
        ConceptLibrary(src).rebuild(B, V, M, max_size=5)
        dst = ConceptLibrary(sqlite3.connect(":memory:"))
        assert dst.seed_from(src) >= 1
        assert dst.kept and not dst.rules                         # structure transfers, worth doesn't
        assert dst.value_for(np.array([0, 0, 0])) is None         # inert until the first grow fits it

    def test_empty_library_is_inert(self):
        lib = ConceptLibrary(sqlite3.connect(":memory:"))
        assert lib.value_for(np.array([1, 2, 3])) is None         # nothing discovered → no signal


class TestBoardTableReservoir:
    @staticmethod
    def _offer(table, lo, hi):
        """Offer boards [i, i, i] for i in [lo, hi) as one wave."""
        boards = {f"t{i}": np.array([i, i, i]) for i in range(lo, hi)}
        trans = {(f"f{i}", f"t{i}"): ((1, 0, 0), 0.5) for i in range(lo, hi)}
        table.update(list(trans.keys()), boards, trans)

    def test_under_cap_keeps_every_board(self):
        t = S.BoardTable(cap=50)
        self._offer(t, 0, 30)
        assert len(t) == 30

    def test_over_cap_is_bounded_and_consistent(self):
        t = S.BoardTable(cap=20)
        self._offer(t, 0, 500)
        assert len(t) == 20                                        # bounded
        B, V, M = t.arrays()
        assert len(B) == len(V) == len(M) == 20                    # parallel lists stay in step
        for bkey, idx in t._row.items():                           # index ↔ rows never desync
            assert t._cells[idx].astype(np.int64).tobytes() == bkey

    def test_same_arrivals_same_sample(self):
        a, b = S.BoardTable(cap=20), S.BoardTable(cap=20)
        self._offer(a, 0, 500); self._offer(b, 0, 500)
        assert np.array_equal(a.arrays()[0], b.arrays()[0])        # seeded draw → reproducible
