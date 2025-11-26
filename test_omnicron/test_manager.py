import os
import tempfile
import numpy as np
import pytest

from omnicron.manager import GameMemory
from omnicron.manager import Play
from agent.agent import State
from games.game_state import GameState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyGameState(GameState):
    def __init__(self, board, current_player):
        self.board = np.array(board)
        self.current_player = current_player

def gs(board, player=1):
    return DummyGameState(board, player)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_mem():
    with tempfile.TemporaryDirectory() as base:
        db_path = os.path.join(base, "memory.db")
        mem = GameMemory(base_dir=base, db_path=db_path)
        yield mem


# ---------------------------------------------------------------------------
# Basic DB / Write Tests
# ---------------------------------------------------------------------------

def test_write_creates_row(temp_mem):
    g = temp_mem
    s = gs([1, 2, 3], 1)

    g.write("test_game", s, State.WIN, np.array([9]))

    assert Play.select().count() == 1


def test_write_increments_count(temp_mem):
    g = temp_mem
    s = gs([1, 1, 1], 1)
    move = np.array([5])

    g.write("g", s, State.WIN, move)
    g.write("g", s, State.WIN, move)

    row = Play.get()
    assert row.count == 2


def test_write_creates_distinct_rows_for_different_players(temp_mem):
    g = temp_mem
    board = [0, 0, 0]

    g.write("g", gs(board, 1), State.WIN, np.array([1]))
    g.write("g", gs(board, 2), State.WIN, np.array([1]))

    assert Play.select().count() == 2


# ---------------------------------------------------------------------------
# Snapshot Tests
# ---------------------------------------------------------------------------

def test_snapshot_creates_file(temp_mem):
    g = temp_mem
    s = gs([1, 2, 3], 1)

    g.write("g", s, State.WIN, np.array([77]))

    snap_dir = os.path.join(g.base_dir, "g")
    snap_path = os.path.join(snap_dir, "plays.sqlite")

    assert not os.path.exists(snap_path)

    g.snapshot("g")

    assert os.path.exists(snap_path)
    assert os.path.getsize(snap_path) > 0


# ---------------------------------------------------------------------------
# get_best_move() Tests (updated for certainty scoring)
# ---------------------------------------------------------------------------

def test_get_best_move_returns_none_if_no_rows(temp_mem):
    g = temp_mem
    assert g.get_best_move("g", gs([0, 0])) is None


def test_best_move_prefers_higher_certainty_over_raw_counts(temp_mem):
    """
    If a move has 1000 losses and 100 wins (certainty = 91% loss),
    while another move has a single tie and nothing else (certainty = 100%),
    the tie move must win.
    """
    g = temp_mem
    s = gs([7, 7, 7], 1)

    # Move A: many losses + some wins
    for _ in range(1000):
        g.write("g", s, State.LOSS, np.array([111]))
    for _ in range(100):
        g.write("g", s, State.WIN, np.array([111]))

    # Move B: perfect tie outcome (certainty = 1.0)
    g.write("g", s, State.TIE, np.array([222]))

    best = g.get_best_move("g", s)
    assert np.array_equal(best, np.array([222]))


def test_best_move_prefers_win_over_tie_when_certainty_equal(temp_mem):
    """
    If two moves have 100% certainty but one is WIN and one is TIE,
    the WIN move should have higher utility.
    """
    g = temp_mem
    s = gs([9, 9, 9], 1)

    g.write("g", s, State.TIE, np.array([10]))  # certainty = 1, utility = 0.2
    g.write("g", s, State.WIN, np.array([20]))  # certainty = 1, utility = 1.0

    best = g.get_best_move("g", s)
    assert np.array_equal(best, np.array([20]))


def test_best_move_respects_current_player(temp_mem):
    g = temp_mem
    board = [3, 3, 3]

    g.write("g", gs(board, 1), State.WIN, np.array([50]))
    g.write("g", gs(board, 2), State.WIN, np.array([99]))

    best = g.get_best_move("g", gs(board, 1))
    assert np.array_equal(best, np.array([50]))


def test_best_move_deserializes_array_correctly(temp_mem):
    g = temp_mem
    s = gs([4, 4, 4], 1)

    original = np.array([123, 456], dtype=np.int64)
    g.write("g", s, State.WIN, original)

    best = g.get_best_move("g", s)
    assert isinstance(best, np.ndarray)
    assert best.dtype == original.dtype
    assert np.array_equal(best, original)


def test_multiple_outcomes_for_same_move_are_combined(temp_mem):
    """
    WIN + TIE + NEUTRAL + LOSS rows for the same move_hash
    must aggregate into a single probability distribution.
    """
    g = temp_mem
    s = gs([8, 8, 8], 1)

    # Same move, mixed outcomes
    g.write("g", s, State.WIN, np.array([1]))
    g.write("g", s, State.LOSS, np.array([1]))
    g.write("g", s, State.TIE, np.array([1]))
    g.write("g", s, State.NEUTRAL, np.array([1]))

    # Another move with pure win
    g.write("g", s, State.WIN, np.array([2]))

    best = g.get_best_move("g", s)
    assert np.array_equal(best, np.array([2]))
