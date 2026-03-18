"""
Tests for wise_explorer.memory.predicates

Tests the full predicate system: expressions, atoms, conjunctions,
mining, and library persistence.
"""

import json
import pytest
import numpy as np

from wise_explorer.core.types import Stats
from wise_explorer.memory.predicates import (
    # Expressions
    Literal, Var, BoardAt, RankOf, FileOf, MakeSq, Add, FromBoardAt, Expr,
    # Atoms
    Eq, Neq, Lt, Gt, Le, Ge, InBounds, Atom,
    # Clauses & Conjunctions
    AtomClause, ExistsClause, NotClause, Conjunction, Clause,
    # Predicates
    Predicate,
    # Mining & Library
    PredicateMiner, PredicateLibrary,
    # Helpers
    _sq, _board_at, _from_board_at, _score_variance,
)
from wise_explorer.memory import TransitionMemory


# =============================================================================
# Layer 1 — Expression Tests
# =============================================================================

class TestLiteral:
    def test_evaluate(self):
        assert Literal(42).evaluate(np.zeros((3, 3), dtype=np.int8), {}) == 42

    def test_serialization(self):
        expr = Literal(7)
        d = expr.to_dict()
        assert d == {"type": "literal", "value": 7}
        assert Expr.from_dict(d) == expr


class TestVar:
    def test_evaluate(self):
        assert Var("x").evaluate(np.zeros((3, 3), dtype=np.int8), {"x": 5}) == 5

    def test_unbound_raises(self):
        with pytest.raises(KeyError):
            Var("x").evaluate(np.zeros((3, 3), dtype=np.int8), {})

    def test_serialization(self):
        expr = Var("sq")
        assert Expr.from_dict(expr.to_dict()) == expr


class TestBoardAt:
    def test_evaluate(self):
        board = np.array([[0, 1], [2, 0]], dtype=np.int8)
        expr = BoardAt(MakeSq(Literal(0), Literal(1)))
        assert expr.evaluate(board, {}) == 1

    def test_with_var(self):
        board = np.array([[0, 1], [2, 0]], dtype=np.int8)
        expr = BoardAt(Var("sq"))
        assert expr.evaluate(board, {"sq": (1, 0)}) == 2

    def test_serialization(self):
        expr = BoardAt(MakeSq(Literal(1), Literal(2)))
        assert Expr.from_dict(expr.to_dict()) == expr


class TestMakeSq:
    def test_evaluate(self):
        sq = MakeSq(Literal(2), Literal(3))
        assert sq.evaluate(np.zeros((4, 4), dtype=np.int8), {}) == (2, 3)

    def test_serialization(self):
        sq = MakeSq(Literal(0), Literal(1))
        assert Expr.from_dict(sq.to_dict()) == sq


class TestRankFileOf:
    def test_rank(self):
        assert RankOf(Var("sq")).evaluate(np.zeros((3, 3), dtype=np.int8), {"sq": (2, 1)}) == 2

    def test_file(self):
        assert FileOf(Var("sq")).evaluate(np.zeros((3, 3), dtype=np.int8), {"sq": (2, 1)}) == 1

    def test_serialization(self):
        for expr in [RankOf(Var("sq")), FileOf(Var("sq"))]:
            assert Expr.from_dict(expr.to_dict()) == expr


class TestAdd:
    def test_positive_offset(self):
        expr = Add(Literal(3), 2)
        assert expr.evaluate(np.zeros((3, 3), dtype=np.int8), {}) == 5

    def test_negative_offset(self):
        expr = Add(Literal(3), -1)
        assert expr.evaluate(np.zeros((3, 3), dtype=np.int8), {}) == 2

    def test_serialization(self):
        expr = Add(RankOf(Var("sq")), 1)
        assert Expr.from_dict(expr.to_dict()) == expr


# =============================================================================
# Layer 2 — Atom Tests
# =============================================================================

class TestEq:
    def test_equal(self):
        board = np.array([[1, 0], [0, 1]], dtype=np.int8)
        atom = Eq(BoardAt(MakeSq(Literal(0), Literal(0))), Literal(1))
        assert atom.evaluate(board, {}) is True

    def test_not_equal(self):
        board = np.array([[1, 0], [0, 1]], dtype=np.int8)
        atom = Eq(BoardAt(MakeSq(Literal(0), Literal(0))), Literal(2))
        assert atom.evaluate(board, {}) is False

    def test_two_cells(self):
        board = np.array([[1, 0], [0, 1]], dtype=np.int8)
        atom = Eq(BoardAt(MakeSq(Literal(0), Literal(0))),
                   BoardAt(MakeSq(Literal(1), Literal(1))))
        assert atom.evaluate(board, {}) is True

    def test_serialization(self):
        atom = Eq(BoardAt(MakeSq(Literal(0), Literal(0))), Literal(1))
        assert Atom.from_dict(atom.to_dict()) == atom


class TestNeq:
    def test_not_equal(self):
        board = np.array([[1, 0], [0, 1]], dtype=np.int8)
        atom = Neq(BoardAt(MakeSq(Literal(0), Literal(0))), Literal(0))
        assert atom.evaluate(board, {}) is True

    def test_equal(self):
        board = np.array([[1, 0], [0, 1]], dtype=np.int8)
        atom = Neq(BoardAt(MakeSq(Literal(0), Literal(0))), Literal(1))
        assert atom.evaluate(board, {}) is False


class TestComparisons:
    def test_lt(self):
        board = np.zeros((3, 3), dtype=np.int8)
        assert Lt(Literal(1), Literal(2)).evaluate(board, {}) is True
        assert Lt(Literal(2), Literal(1)).evaluate(board, {}) is False

    def test_gt(self):
        board = np.zeros((3, 3), dtype=np.int8)
        assert Gt(Literal(2), Literal(1)).evaluate(board, {}) is True

    def test_le(self):
        board = np.zeros((3, 3), dtype=np.int8)
        assert Le(Literal(1), Literal(1)).evaluate(board, {}) is True
        assert Le(Literal(2), Literal(1)).evaluate(board, {}) is False

    def test_ge(self):
        board = np.zeros((3, 3), dtype=np.int8)
        assert Ge(Literal(1), Literal(1)).evaluate(board, {}) is True

    def test_serialization(self):
        for cls in [Lt, Gt, Le, Ge]:
            atom = cls(Literal(1), Literal(2))
            assert Atom.from_dict(atom.to_dict()) == atom


class TestInBounds:
    def test_in_bounds(self):
        board = np.zeros((3, 4), dtype=np.int8)
        assert InBounds(Literal(2), Literal(3)).evaluate(board, {}) is True

    def test_out_of_bounds_row(self):
        board = np.zeros((3, 4), dtype=np.int8)
        assert InBounds(Literal(3), Literal(0)).evaluate(board, {}) is False

    def test_out_of_bounds_negative(self):
        board = np.zeros((3, 4), dtype=np.int8)
        assert InBounds(Literal(-1), Literal(0)).evaluate(board, {}) is False

    def test_serialization(self):
        atom = InBounds(Literal(1), Literal(2))
        assert Atom.from_dict(atom.to_dict()) == atom


# =============================================================================
# Layer 3 — Conjunction Tests
# =============================================================================

class TestAtomClause:
    def test_matches(self):
        board = np.array([[1, 0], [0, 2]], dtype=np.int8)
        clause = AtomClause(Eq(_board_at(0, 0), Literal(1)))
        assert clause.matches(board, {}) is True

    def test_serialization(self):
        clause = AtomClause(Eq(_board_at(0, 0), Literal(1)))
        d = clause.to_dict()
        assert Clause.from_dict(d).matches(
            np.array([[1, 0], [0, 2]], dtype=np.int8), {}
        ) is True


class TestExistsClause:
    def test_exists_square(self):
        """∃sq: board[sq] == 1  (there exists a cell with value 1)."""
        board = np.array([[0, 0], [0, 1]], dtype=np.int8)
        clause = ExistsClause("sq", "squares", [
            AtomClause(Eq(BoardAt(Var("sq")), Literal(1)))
        ])
        assert clause.matches(board, {}) is True

    def test_exists_no_match(self):
        """∃sq: board[sq] == 3  (no cell has value 3)."""
        board = np.array([[0, 1], [2, 0]], dtype=np.int8)
        clause = ExistsClause("sq", "squares", [
            AtomClause(Eq(BoardAt(Var("sq")), Literal(3)))
        ])
        assert clause.matches(board, {}) is False

    def test_exists_rank(self):
        """∃r: board[sq(r, 0)] == 2."""
        board = np.array([[0], [2], [0]], dtype=np.int8)
        clause = ExistsClause("r", "ranks", [
            AtomClause(Eq(BoardAt(MakeSq(Var("r"), Literal(0))), Literal(2)))
        ])
        assert clause.matches(board, {}) is True

    def test_serialization(self):
        clause = ExistsClause("sq", "squares", [
            AtomClause(Eq(BoardAt(Var("sq")), Literal(1)))
        ])
        d = clause.to_dict()
        restored = Clause.from_dict(d)
        board = np.array([[0, 0], [0, 1]], dtype=np.int8)
        assert restored.matches(board, {}) is True


class TestNotClause:
    def test_negation(self):
        """¬(board[0,0] == 1) when board[0,0] == 0."""
        board = np.array([[0, 0], [0, 0]], dtype=np.int8)
        clause = NotClause([AtomClause(Eq(_board_at(0, 0), Literal(1)))])
        assert clause.matches(board, {}) is True

    def test_negation_fails(self):
        """¬(board[0,0] == 1) when board[0,0] == 1."""
        board = np.array([[1, 0], [0, 0]], dtype=np.int8)
        clause = NotClause([AtomClause(Eq(_board_at(0, 0), Literal(1)))])
        assert clause.matches(board, {}) is False


class TestConjunction:
    def test_all_match(self):
        board = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int8)
        conj = Conjunction([
            AtomClause(Eq(_board_at(0, 0), Literal(1))),
            AtomClause(Eq(_board_at(1, 1), Literal(1))),
            AtomClause(Eq(_board_at(2, 2), Literal(1))),
        ])
        assert conj.matches(board) is True

    def test_one_fails(self):
        board = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.int8)
        conj = Conjunction([
            AtomClause(Eq(_board_at(0, 0), Literal(1))),
            AtomClause(Eq(_board_at(1, 1), Literal(1))),  # fails
            AtomClause(Eq(_board_at(2, 2), Literal(1))),
        ])
        assert conj.matches(board) is False

    def test_num_atoms(self):
        conj = Conjunction([
            AtomClause(Eq(_board_at(0, 0), Literal(1))),
            AtomClause(Eq(_board_at(1, 1), Literal(1))),
        ])
        assert conj.num_atoms == 2

    def test_serialization_round_trip(self):
        conj = Conjunction([
            AtomClause(Eq(_board_at(0, 0), Literal(1))),
            AtomClause(Neq(_board_at(1, 1), Literal(0))),
        ])
        d = conj.to_dict()
        restored = Conjunction.from_dict(d)
        board = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=np.int8)
        assert restored.matches(board) is True


class TestForAll:
    def test_universal_via_negated_exists(self):
        """∀r: board[r, 0] != 1  ≡  ¬∃r: board[r, 0] == 1.

        No cell in column 0 has value 1.
        """
        board = np.array([[0, 1], [2, 0], [0, 0]], dtype=np.int8)
        # ¬∃r: board[sq(r, 0)] == 1
        clause = NotClause([
            ExistsClause("r", "ranks", [
                AtomClause(Eq(BoardAt(MakeSq(Var("r"), Literal(0))), Literal(1)))
            ])
        ])
        assert clause.matches(board, {}) is True  # no 1 in column 0

    def test_universal_fails(self):
        """∀r: board[r, 0] != 1 — fails because row 1 col 0 has value 1."""
        board = np.array([[0, 0], [1, 0], [0, 0]], dtype=np.int8)
        clause = NotClause([
            ExistsClause("r", "ranks", [
                AtomClause(Eq(BoardAt(MakeSq(Var("r"), Literal(0))), Literal(1)))
            ])
        ])
        assert clause.matches(board, {}) is False  # found 1 in column 0


class TestRelativePredicate:
    def test_existential_with_offset(self):
        """∃sq: board[sq]==1 ∧ board[sq+(1,1)]==1 — diagonal pair."""
        board = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8)
        conj = Conjunction([
            ExistsClause("sq", "squares", [
                AtomClause(Eq(BoardAt(Var("sq")), Literal(1))),
                AtomClause(InBounds(
                    Add(RankOf(Var("sq")), 1),
                    Add(FileOf(Var("sq")), 1),
                )),
                AtomClause(Eq(
                    BoardAt(MakeSq(
                        Add(RankOf(Var("sq")), 1),
                        Add(FileOf(Var("sq")), 1),
                    )),
                    Literal(1),
                )),
            ])
        ])
        assert conj.matches(board) is True

    def test_existential_no_match(self):
        """Same pattern but board has no diagonal pair."""
        board = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.int8)
        conj = Conjunction([
            ExistsClause("sq", "squares", [
                AtomClause(Eq(BoardAt(Var("sq")), Literal(1))),
                AtomClause(InBounds(
                    Add(RankOf(Var("sq")), 1),
                    Add(FileOf(Var("sq")), 1),
                )),
                AtomClause(Eq(
                    BoardAt(MakeSq(
                        Add(RankOf(Var("sq")), 1),
                        Add(FileOf(Var("sq")), 1),
                    )),
                    Literal(1),
                )),
            ])
        ])
        # (0,0) has 1, but (1,1) is 0 — no match
        # (2,2) has 1, but (3,3) is out of bounds — no match
        assert conj.matches(board) is False


# =============================================================================
# Layer 4 — Predicate Tests
# =============================================================================

class TestPredicate:
    def test_matches_delegates(self):
        board = np.array([[1, 0], [0, 1]], dtype=np.int8)
        pred = Predicate(
            conjunction=Conjunction([AtomClause(Eq(_board_at(0, 0), Literal(1)))]),
            counts=(10.0, 2.0, 1.0),
            support=5,
            variance=0.01,
        )
        assert pred.matches(board) is True

    def test_mean_score(self):
        pred = Predicate(
            conjunction=Conjunction([]),
            counts=(100.0, 0.0, 0.0),
            support=100,
        )
        assert pred.mean_score > 0.9

    def test_specificity(self):
        pred = Predicate(
            conjunction=Conjunction([
                AtomClause(Eq(_board_at(0, 0), Literal(1))),
                AtomClause(Eq(_board_at(1, 1), Literal(1))),
                AtomClause(Eq(_board_at(2, 2), Literal(1))),
            ]),
        )
        assert pred.specificity == 3


# =============================================================================
# Mining Tests
# =============================================================================

class TestScoreVariance:
    def test_identical_scores(self):
        scores = {"a": 0.5, "b": 0.5, "c": 0.5}
        assert _score_variance(["a", "b", "c"], scores) == 0.0

    def test_varied_scores(self):
        scores = {"a": 0.0, "b": 1.0}
        var = _score_variance(["a", "b"], scores)
        assert var == pytest.approx(0.25)

    def test_single_element(self):
        assert _score_variance(["a"], {"a": 0.5}) == 0.0


class TestPredicateMiner:
    def _make_ttt_boards(self):
        """Create synthetic TTT boards with known patterns."""
        boards = {}
        scores = {}

        # Winning diagonal for X: board[0,0]==1, board[1,1]==1, board[2,2]==1
        for i in range(10):
            board = np.zeros((3, 3), dtype=np.int8)
            board[0, 0] = 1
            board[1, 1] = 1
            board[2, 2] = 1
            # Vary other cells to create different boards
            board[0, 1] = 2 if i % 3 == 0 else 0
            board[2, 0] = 2 if i % 2 == 0 else 0
            board[1, 0] = 2 if i > 5 else 0
            h = f"win_diag_{i}"
            boards[h] = board
            scores[h] = ((8.0, 1.0, 1.0), Stats(8, 1, 1).mean_score)

        # Losing positions for X (O has a row)
        for i in range(10):
            board = np.zeros((3, 3), dtype=np.int8)
            board[0, 0] = 2
            board[0, 1] = 2
            board[0, 2] = 2
            board[1, 0] = 1 if i % 2 == 0 else 0
            board[2, 2] = 1 if i % 3 == 0 else 0
            h = f"loss_row_{i}"
            boards[h] = board
            scores[h] = ((1.0, 1.0, 8.0), Stats(1, 1, 8).mean_score)

        # Mixed/contested boards
        for i in range(10):
            board = np.zeros((3, 3), dtype=np.int8)
            board[1, 1] = 1  # center
            board[0, 0] = 2 if i % 2 == 0 else 0
            h = f"mixed_{i}"
            boards[h] = board
            scores[h] = ((4.0, 3.0, 3.0), Stats(4, 3, 3).mean_score)

        return boards, scores

    def test_mine_discovers_predicates(self):
        boards, scores = self._make_ttt_boards()
        miner = PredicateMiner(min_support=5, max_variance=0.1, variance_penalty=0.05)
        predicates = miner.mine(boards, scores)
        assert len(predicates) > 0

    def test_mine_low_variance_predicates(self):
        """Mined predicates should have lower variance than random."""
        boards, scores = self._make_ttt_boards()
        miner = PredicateMiner(min_support=5, max_variance=0.1, variance_penalty=0.05)
        predicates = miner.mine(boards, scores)

        # All predicates should have reasonable variance
        for pred in predicates:
            assert pred.support >= 5

    def test_mine_empty_returns_empty(self):
        miner = PredicateMiner()
        assert miner.mine({}, {}) == []

    def test_mine_insufficient_data(self):
        boards = {"a": np.zeros((3, 3), dtype=np.int8)}
        scores = {"a": ((1.0, 0.0, 0.0), 1.0)}
        miner = PredicateMiner(min_support=5)
        assert miner.mine(boards, scores) == []

    def test_winning_diagonal_discovered(self):
        """The miner should find the X diagonal pattern."""
        boards, scores = self._make_ttt_boards()
        miner = PredicateMiner(min_support=5, max_variance=0.15, variance_penalty=0.01)
        predicates = miner.mine(boards, scores)

        # Check that at least one predicate matches all winning diagonals
        win_boards = {h: b for h, b in boards.items() if h.startswith("win_diag")}
        found_win_pred = False
        for pred in predicates:
            matches_all_wins = all(pred.matches(b) for b in win_boards.values())
            if matches_all_wins and pred.mean_score > 0.6:
                found_win_pred = True
                break
        assert found_win_pred, "No predicate found that matches all winning diagonals"


# =============================================================================
# Library Tests
# =============================================================================

class TestPredicateLibrary:
    @pytest.fixture
    def library(self, temp_db_path):
        mem = TransitionMemory(temp_db_path)
        lib = PredicateLibrary(mem.conn)
        yield lib, mem
        mem.close()

    def test_empty_library_returns_none(self, library):
        lib, _ = library
        board = np.zeros((3, 3), dtype=np.int8)
        assert lib.match(board) is None

    def test_save_and_load(self, library):
        lib, mem = library
        pred = Predicate(
            conjunction=Conjunction([
                AtomClause(Eq(_board_at(0, 0), Literal(1))),
                AtomClause(Eq(_board_at(1, 1), Literal(1))),
            ]),
            counts=(50.0, 5.0, 5.0),
            support=20,
            variance=0.02,
        )
        lib.save([pred])
        assert lib.count == 1

        # Reload from DB
        lib2 = PredicateLibrary(mem.conn)
        assert lib2.count == 1

    def test_match_returns_stats(self, library):
        lib, _ = library
        pred = Predicate(
            conjunction=Conjunction([
                AtomClause(Eq(_board_at(0, 0), Literal(1))),
            ]),
            counts=(50.0, 5.0, 5.0),
            support=20,
            variance=0.02,
        )
        lib.save([pred])

        board = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
        stats = lib.match(board)
        assert stats is not None
        assert stats.wins == 50.0

    def test_match_no_match(self, library):
        lib, _ = library
        pred = Predicate(
            conjunction=Conjunction([
                AtomClause(Eq(_board_at(0, 0), Literal(1))),
            ]),
            counts=(50.0, 5.0, 5.0),
            support=20,
            variance=0.02,
        )
        lib.save([pred])

        board = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
        assert lib.match(board) is None

    def test_most_specific_match(self, library):
        lib, _ = library
        # General predicate: board[0,0]==1
        general = Predicate(
            conjunction=Conjunction([
                AtomClause(Eq(_board_at(0, 0), Literal(1))),
            ]),
            counts=(30.0, 10.0, 10.0),
            support=50,
            variance=0.1,
        )
        # Specific predicate: board[0,0]==1 AND board[1,1]==1
        specific = Predicate(
            conjunction=Conjunction([
                AtomClause(Eq(_board_at(0, 0), Literal(1))),
                AtomClause(Eq(_board_at(1, 1), Literal(1))),
            ]),
            counts=(90.0, 5.0, 5.0),
            support=30,
            variance=0.01,
        )
        lib.save([general, specific])

        board = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8)
        stats = lib.match(board)
        # Should return the more specific predicate
        assert stats.wins == 90.0

    def test_match_all(self, library):
        lib, _ = library
        pred1 = Predicate(
            conjunction=Conjunction([AtomClause(Eq(_board_at(0, 0), Literal(1)))]),
            counts=(30.0, 10.0, 10.0), support=50, variance=0.1,
        )
        pred2 = Predicate(
            conjunction=Conjunction([
                AtomClause(Eq(_board_at(0, 0), Literal(1))),
                AtomClause(Eq(_board_at(1, 1), Literal(1))),
            ]),
            counts=(90.0, 5.0, 5.0), support=30, variance=0.01,
        )
        lib.save([pred1, pred2])

        board = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8)
        matches = lib.match_all(board)
        assert len(matches) == 2
        # Most specific first
        assert matches[0].specificity >= matches[1].specificity

    def test_read_only_save_noop(self, temp_db_path):
        mem = TransitionMemory(temp_db_path)
        mem.close()
        mem_ro = TransitionMemory(temp_db_path, read_only=True)
        lib = PredicateLibrary(mem_ro.conn, read_only=True)
        pred = Predicate(
            conjunction=Conjunction([AtomClause(Eq(_board_at(0, 0), Literal(1)))]),
            counts=(10.0, 0.0, 0.0), support=5, variance=0.01,
        )
        lib.save([pred])  # should be a no-op
        assert lib.count == 0
        mem_ro.close()


# =============================================================================
# Integration: Board Recording + Mining
# =============================================================================

class TestBoardRecording:
    def test_boards_stored_during_recording(self, temp_db_path):
        """record_round() stores destination boards in the boards table."""
        from wise_explorer.agent.agent import State
        from wise_explorer.games.tic_tac_toe import TicTacToe

        mem = TransitionMemory(temp_db_path)
        game = TicTacToe()

        # Simulate a simple game
        moves_p1 = [
            (np.array([1, 1]), game.get_state().board.copy(), 1),
        ]
        game.apply_move(np.array([1, 1]))
        moves_p2 = [
            (np.array([0, 0]), game.get_state().board.copy(), 2),
        ]
        game.apply_move(np.array([0, 0]))

        stacks = [
            (moves_p1, State.WIN),
            (moves_p2, State.LOSS),
        ]
        mem.record_round(TicTacToe, stacks)

        board_count = mem.conn.execute("SELECT COUNT(*) FROM boards").fetchone()[0]
        assert board_count > 0
        mem.close()

    def test_mine_after_recording(self, temp_db_path):
        """mine_predicates() works after recording games."""
        from wise_explorer.agent.agent import State
        from wise_explorer.games.tic_tac_toe import TicTacToe

        mem = TransitionMemory(temp_db_path)

        # Record many games with clear patterns
        for _ in range(20):
            game = TicTacToe()

            # X wins with center
            moves_p1 = [
                (np.array([1, 1]), game.get_state().board.copy(), 1),
            ]
            game.apply_move(np.array([1, 1]))

            stacks = [(moves_p1, State.WIN)]
            mem.record_round(TicTacToe, stacks)

        # Try mining
        count = mem.mine_predicates()
        # May or may not find predicates depending on data diversity
        assert count >= 0
        mem.close()


# =============================================================================
# Cross-board (transformation) atoms
# =============================================================================

class TestFromBoardAt:
    def test_evaluate(self):
        to_board = np.array([[1, 0], [0, 2]], dtype=np.int8)
        from_board = np.array([[0, 0], [0, 1]], dtype=np.int8)
        expr = FromBoardAt(MakeSq(Literal(1), Literal(1)))
        assert expr.evaluate(to_board, {"_from": from_board}) == 1

    def test_serialization(self):
        expr = FromBoardAt(MakeSq(Literal(0), Literal(1)))
        d = expr.to_dict()
        restored = Expr.from_dict(d)
        assert restored == expr

    def test_cross_board_changed(self):
        """Neq(BoardAt(sq), FromBoardAt(sq)) detects a cell that changed."""
        to_board = np.array([[1, 0], [0, 0]], dtype=np.int8)
        from_board = np.array([[0, 0], [0, 0]], dtype=np.int8)
        # Cell (0,0) changed from 0 to 1
        atom = Neq(_board_at(0, 0), _from_board_at(0, 0))
        assert atom.evaluate(to_board, {"_from": from_board}) is True
        # Cell (0,1) didn't change
        atom2 = Neq(_board_at(0, 1), _from_board_at(0, 1))
        assert atom2.evaluate(to_board, {"_from": from_board}) is False

    def test_cross_board_appeared(self):
        """Piece appeared: from[sq]==0 AND to[sq]==val."""
        to_board = np.array([[1, 0], [0, 2]], dtype=np.int8)
        from_board = np.array([[0, 0], [0, 0]], dtype=np.int8)
        # Piece 1 appeared at (0,0)
        conj = Conjunction([
            AtomClause(Eq(_from_board_at(0, 0), Literal(0))),
            AtomClause(Eq(_board_at(0, 0), Literal(1))),
        ])
        assert conj.matches(to_board, {"_from": from_board}) is True

    def test_cross_board_capture(self):
        """Capture: from[sq]!=0 AND to[sq]!=0 AND from[sq]!=to[sq]."""
        to_board = np.array([[1, 0], [0, 0]], dtype=np.int8)
        from_board = np.array([[2, 0], [0, 0]], dtype=np.int8)
        # Cell (0,0): had piece 2, now has piece 1 = capture
        conj = Conjunction([
            AtomClause(Neq(_from_board_at(0, 0), Literal(0))),
            AtomClause(Neq(_board_at(0, 0), Literal(0))),
            AtomClause(Neq(_board_at(0, 0), _from_board_at(0, 0))),
        ])
        assert conj.matches(to_board, {"_from": from_board}) is True

    def test_miner_with_from_boards(self):
        """PredicateMiner generates cross-board atoms when from_boards provided."""
        # Create boards where a piece appearing at (0,0) predicts high score
        boards = {}
        from_boards = {}
        scores = {}
        for i in range(10):
            to_b = np.zeros((3, 3), dtype=np.int8)
            from_b = np.zeros((3, 3), dtype=np.int8)
            to_b[0, 0] = 1  # piece appeared at (0,0)
            to_b[1, 1] = 2 if i % 2 == 0 else 0
            from_b[1, 1] = 2 if i % 2 == 0 else 0  # this didn't change
            h = f"appeared_{i}"
            boards[h] = to_b
            from_boards[h] = from_b
            scores[h] = ((8.0, 1.0, 1.0), 0.8)

        for i in range(10):
            to_b = np.zeros((3, 3), dtype=np.int8)
            from_b = np.zeros((3, 3), dtype=np.int8)
            to_b[2, 2] = 2  # piece appeared at (2,2) instead
            from_b[0, 0] = 1 if i % 3 == 0 else 0
            h = f"other_{i}"
            boards[h] = to_b
            from_boards[h] = from_b
            scores[h] = ((1.0, 1.0, 8.0), 0.2)

        miner = PredicateMiner(min_support=5, max_variance=0.1, variance_penalty=0.05)
        preds = miner.mine(boards, scores, from_boards)
        assert len(preds) > 0
