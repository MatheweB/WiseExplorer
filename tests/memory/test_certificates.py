"""Frontier proofs, proof-pinned completion, and proof-licensed deletion."""

import numpy as np
import pytest

import wise_explorer.memory as Memory
from wise_explorer.games.nim import Nim
from wise_explorer.games.game_state import GameState
from wise_explorer.core.hashing import hash_board


def nim_sum(b) -> int:
    return int(np.bitwise_xor.reduce(np.asarray(b).astype(np.int64)))


@pytest.fixture
def trained_nim2(tmp_path):
    """A 2-pile Nim memory with every position recorded by hand."""
    mem = Memory.for_game(Nim(n=2), base_dir=str(tmp_path))
    game = Nim(n=2)
    # record one playout per legal move from every position so every board
    # and transition exists; counts are irrelevant to the proofs under test
    import itertools
    boards, transitions = {}, {}
    for tup in itertools.product(range(2), range(3)):
        b = np.array(tup, dtype=np.int8)
        if b.sum() == 0:
            continue
        g = Nim(n=2)
        g.set_state(GameState(b.copy(), current_player=1))
        fh = hash_board(g.get_state().board)
        boards[fh] = g.get_state().board
        for mv in g.valid_moves():
            c = g.deep_clone()
            c.apply_move(mv, validated=True)
            th = hash_board(c.get_state().board)
            boards[th] = c.get_state().board
            transitions[(fh, th)] = [1.0, 0.0, 0.0]
    mem._store_boards({h: (np.asarray(b, dtype=np.int8).reshape(1, -1).tobytes(), 1, len(np.asarray(b).ravel()))
                       for h, b in boards.items()})
    mem._commit(transitions)
    yield mem
    mem.close()


class TestFrontier:
    def test_proves_all_values_exactly(self, trained_nim2):
        n = trained_nim2.frontier_certify(Nim(n=2))
        assert n > 0
        boards = trained_nim2._load_boards()
        for h, v in trained_nim2.certified_values.items():
            b = np.asarray(boards[h]).ravel()
            if b.sum() == 0:
                continue
            expected = 1.0 if nim_sum(b) == 0 else 0.0
            assert v == expected, f"{b}: proved {v}, minimax {expected}"

    def test_proofs_persist_and_cache_invalidates(self, trained_nim2):
        trained_nim2.frontier_certify(Nim(n=2))
        first = dict(trained_nim2.certified_values)
        assert trained_nim2.frontier_certify(Nim(n=2)) == 0   # nothing new
        assert trained_nim2.certified_values == first

    def test_proofs_ignore_a_wrong_library(self, trained_nim2):
        """Certification must not depend on the theory's opinions."""
        trained_nim2.frontier_certify(Nim(n=2))
        clean = dict(trained_nim2.certified_values)
        for r in trained_nim2.concept_library.rules:
            r.avg = 1.0 - r.avg
        trained_nim2.conn.execute("DELETE FROM certificates")
        trained_nim2._certified_cache = None
        trained_nim2.frontier_certify(Nim(n=2))
        assert trained_nim2.certified_values == clean


class TestCollapse:
    def test_deletes_only_proof_consistent_rows(self, trained_nim2):
        trained_nim2.frontier_certify(Nim(n=2))
        certs = trained_nim2.certified_values
        # mark half the rows proof-consistent, half not
        rows = trained_nim2.conn.execute(
            "SELECT from_hash, to_hash FROM transitions").fetchall()
        consistent = 0
        for i, (f, t) in enumerate(rows):
            if i % 2 == 0 and t in certs:
                ps, consistent = certs[t], consistent + 1
            else:
                ps = 0.5 if t not in certs or abs(certs[t] - 0.5) > 0.25 else None
            trained_nim2.conn.execute(
                "UPDATE transitions SET propagated_score=? WHERE from_hash=? AND to_hash=?",
                (ps, f, t))
        trained_nim2.conn.commit()

        deleted = trained_nim2.collapse_proven()
        after = trained_nim2.conn.execute(
            "SELECT COUNT(*) FROM transitions").fetchone()[0]
        assert deleted == consistent > 0
        assert after == len(rows) - consistent
        # survivors all violate the band (or carry no value yet)
        for t, ps in trained_nim2.conn.execute(
                "SELECT to_hash, propagated_score FROM transitions"):
            assert ps is None or t not in certs or abs(ps - certs[t]) > 0.25

    def test_no_certificates_no_deletion(self, trained_nim2):
        assert trained_nim2.collapse_proven() == 0


class TestEvaluateMoves:
    def test_proven_values_reach_selection(self, trained_nim2):
        trained_nim2.frontier_certify(Nim(n=2))
        g = Nim(n=2)
        g.set_state(GameState(np.array([1, 2], dtype=np.int8), current_player=1))
        ev = trained_nim2.evaluate_moves(g, list(g.valid_moves()))
        assert ev.proven                         # certified boards are visible
        # the winning move (to [1,1], nim-sum 0) must be proven 1.0
        winning = {mk: v for mk, v in ev.proven.items() if v == 1.0}
        assert winning


class TestRefreshIfStale:
    """A read-only handle (a pool worker) must pick up concepts/certificates the
    writer commits after the handle was opened — so steering uses the live theory."""

    def test_worker_picks_up_writer_concepts(self, tmp_path):
        import wise_explorer.memory as Memory
        from wise_explorer.games.nim import Nim
        main = Memory.for_game(Nim(n=2), base_dir=str(tmp_path))
        worker = Memory.open_readonly(str(main.db_path))
        assert len(worker.concept_library.kept) == 0

        # writer commits a concept + a certificate
        main.concept_library.conn.execute(
            "INSERT INTO concepts (id, expr_json, op, const, size) VALUES (0, '{}', '=', 0, 1)")
        main.conn.execute("INSERT INTO certificates VALUES ('h', 1.0)")
        main.conn.commit()

        worker.concept_library._load = lambda: setattr(worker.concept_library, "kept", ["x"])
        assert len(worker.concept_library.kept) == 0      # stale until refreshed
        worker.refresh_if_stale()
        assert worker.concept_library.kept == ["x"]       # reloaded on DB change
        assert worker.certified_values == {"h": 1.0}      # certificate cache cleared + reloaded
        main.close(); worker.close()

    def test_no_reload_when_unchanged(self, tmp_path):
        import wise_explorer.memory as Memory
        from wise_explorer.games.nim import Nim
        main = Memory.for_game(Nim(n=2), base_dir=str(tmp_path))
        worker = Memory.open_readonly(str(main.db_path))
        worker.refresh_if_stale()                         # sync versions
        calls = []
        worker.concept_library._load = lambda: calls.append(1)
        worker.refresh_if_stale()                         # nothing committed since
        assert calls == []                                # cheap: no reload
        main.close(); worker.close()
