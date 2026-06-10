"""Tests for the persistent concept library and the value loop it powers."""
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

    def test_tiny_rebuild_keeps_the_loaded_model(self):
        # a rebuild over <8 boards (a near-empty resumed run) must not wipe the persisted model
        B, V, M = _xor_data()
        conn = sqlite3.connect(":memory:")
        ConceptLibrary(conn).rebuild(B, V, M, max_size=5)
        resumed = ConceptLibrary(conn)
        resumed.rebuild(B[:2], V[:2], M[:2], max_size=5)          # 2 boards < 8
        assert resumed.rules and ConceptLibrary(conn, read_only=True).rules

    def test_seed_from_carries_programs_only(self):
        B, V, M = _xor_data()
        src = sqlite3.connect(":memory:")
        ConceptLibrary(src).rebuild(B, V, M, max_size=5)
        dst = ConceptLibrary(sqlite3.connect(":memory:"))
        assert dst.seed_from(src) >= 1
        assert dst.kept and not dst.rules                         # structure transfers, worth doesn't
        assert dst.value_for(np.array([0, 0, 0])) is None         # inert until a rebuild fits it

    def test_empty_library_is_inert(self):
        lib = ConceptLibrary(sqlite3.connect(":memory:"))
        assert lib.value_for(np.array([1, 2, 3])) is None         # nothing discovered → no signal

    def test_summary_survives_reload_with_verdicts(self):
        B, V, M = _xor_data()
        conn = sqlite3.connect(":memory:")
        ConceptLibrary(conn).rebuild(B, V, M, max_size=5)
        text = ConceptLibrary(conn, read_only=True).summary()     # a worker's view
        assert "K₁ = 0" in text and "[WIN ]" in text and "├─ yes" in text
        assert "KEY" in text and "fold(⊕, board, cell)" in text


class TestBoundedRebuild:
    def test_rebuild_subsamples_past_the_cap(self, monkeypatch):
        # discovery's data view is bounded: past the cap, rebuild fits a uniform sample
        monkeypatch.setattr(S, "CAP", 32)
        B, V, M = _xor_data()                                      # 64 rows > 32
        lib = ConceptLibrary(sqlite3.connect(":memory:"))
        assert lib.rebuild(B, V, M, max_size=5) >= 1               # nim-sum still found in the sample
        assert lib.value_for(np.array([0, 0, 0])) > lib.value_for(np.array([1, 0, 0]))


class TestValueLoop:
    """The loop's two pieces: batched pricing, and library-completed Bellman backups."""

    def test_values_for_matches_value_for(self):
        B, V, M = _xor_data()
        lib = ConceptLibrary(sqlite3.connect(":memory:"))
        lib.rebuild(B, V, M, max_size=5)
        for row, m, got in zip(B, M, lib.values_for(B, M)):
            assert got == lib.value_for(row, int(m))

    def test_values_for_empty_library_is_all_nan(self):
        lib = ConceptLibrary(sqlite3.connect(":memory:"))
        assert np.isnan(lib.values_for(np.zeros((3, 3), dtype=np.int64))).all()

    def test_complete_values_heals_an_unplayed_refutation(self, tmp_path):
        """Evidence-only backups call [0,1,2] safe because the one reply anyone PLAYED
        loses — the winning reply [0,1,1] was never visited. The library (which knows
        the nim-sum) prices it, and the completed backup flips the verdict."""
        from wise_explorer.core.hashing import hash_board
        from wise_explorer.games.nim import Nim
        from wise_explorer.memory import TransitionMemory

        mem = TransitionMemory(tmp_path / "nim3.db")
        B, V, M = _xor_data()
        mem.concept_library.rebuild(B, V, M, max_size=5)           # the library knows nim-sum

        boards = [np.array(b, dtype=np.int8) for b in ([0, 2, 2], [0, 1, 2], [0, 0, 2], [0, 0, 0])]
        h22, h12, h02, h00 = (hash_board(b) for b in boards)
        cur = mem.conn.cursor()
        cur.executemany(
            "INSERT INTO boards (board_hash, board_data, board_rows, board_cols) VALUES (?,?,?,?)",
            [(hash_board(b), b.reshape(1, -1).tobytes(), 1, 3) for b in boards])
        cur.executemany(
            "INSERT INTO transitions (from_hash, to_hash, wins, ties, losses) VALUES (?,?,?,?,?)",
            [(h22, h12, 0, 0, 8), (h12, h02, 0, 0, 8), (h02, h00, 8, 0, 0)])
        mem.conn.commit()

        mem.solve_graph()
        assert mem.get_propagated_score(h22, h12) > 0.8            # the blind spot: "[0,1,2] is safe"
        priced = mem.complete_values(Nim(3))
        assert priced >= 4                                         # the never-played replies got prices
        assert mem.get_propagated_score(h22, h12) < 0.1            # healed: the refutation now counts
        assert mem.get_propagated_score(h02, h00) > 0.9            # sound values stay sound
        mem.close()

    def test_complete_values_is_inert_without_rules(self, tmp_path):
        from wise_explorer.games.nim import Nim
        from wise_explorer.memory import TransitionMemory
        mem = TransitionMemory(tmp_path / "empty.db")
        assert mem.complete_values(Nim(3)) == 0                    # no concepts → nothing to lend
        mem.close()
