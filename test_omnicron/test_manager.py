import sys, os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import numpy as np
import pytest

from omnicron.manager import GameMemory
from agent.agent import State
from games.game_state import GameState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyGameState(GameState):
    """Minimal valid GameState substitute. Only 'board' + 'current_player' needed."""

    def __init__(self, board, current_player):
        self.board = board
        self.current_player = current_player


def gs(board, player=1):
    return DummyGameState(np.array(board), player)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_mem():
    """Creates a temporary DB + folder for each test."""
    with tempfile.TemporaryDirectory() as base:
        db = os.path.join(base, "memory.db")
        yield GameMemory(base_dir=base, db_path=db)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


def test_table_created_on_write(temp_mem):
    g = temp_mem
    g.write("test_game", gs([1, 2, 3]), State.WIN, np.array([9]))
    assert "plays_test_game" in g.known


def test_write_increments_count(temp_mem):
    g = temp_mem
    state = gs([1, 1, 1], 1)

    g.write("g", state, State.WIN, np.array([5]))
    g.write("g", state, State.WIN, np.array([5]))  # same board/move/outcome/player

    row = g.con.execute("SELECT count FROM plays_g LIMIT 1").fetchone()
    assert row[0] == 2


def test_write_creates_distinct_rows_for_different_players(temp_mem):
    g = temp_mem
    b = [0, 0, 0]

    g.write("g", gs(b, 1), State.WIN, np.array([1]))
    g.write("g", gs(b, 2), State.WIN, np.array([1]))

    rows = g.con.execute("SELECT COUNT(*) FROM plays_g").fetchone()[0]
    assert rows == 2  # player matters


# ---------------------------------------------------------------------------
# get_best_move scoring & selection
# ---------------------------------------------------------------------------


def test_get_best_move_returns_none_if_no_rows(temp_mem):
    g = temp_mem
    assert g.get_best_move("g", gs([0, 0])) is None


def test_best_move_prefers_non_loss_over_certain_loss(temp_mem):
    """
    Move A: LOSS but seen often (count=5). (Priority Group 0)
    Move B: WIN but seen once (count=1). (Priority Group 1)

    Logic: All non-LOSS outcomes (B) are strictly prioritized over LOSS outcomes (A),
    regardless of count difference. B must be selected.
    """
    g = temp_mem
    board = [7, 7, 7]
    s = gs(board, 1)

    # Move A: LOSS (outcome=LOSS) but count = 5
    for _ in range(5):
        g.write("g", s, State.LOSS, np.array([111]))

    # Move B: WIN (outcome=WIN) but count = 1
    g.write("g", s, State.WIN, np.array([222]))

    best = g.get_best_move("g", s)
    assert np.array_equal(best, np.array([222]))


def test_best_move_prefers_better_outcome_when_certainty_equal(temp_mem):
    """
    Test tertiary sort: count is equal, outcome decides (WIN > TIE).
    """
    g = temp_mem
    board = [9, 9, 9]
    s = gs(board, 1)

    # count = 1 each.
    g.write("g", s, State.TIE, np.array([10]))  # outcome=3
    g.write("g", s, State.WIN, np.array([20]))  # outcome=4

    best = g.get_best_move("g", s)
    assert np.array_equal(best, np.array([20]))


def test_best_move_respects_current_player(temp_mem):
    g = temp_mem
    b = [3, 3, 3]

    g.write("g", gs(b, 1), State.WIN, np.array([50]))
    g.write("g", gs(b, 2), State.WIN, np.array([99]))

    best = g.get_best_move("g", gs(b, 1))
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
