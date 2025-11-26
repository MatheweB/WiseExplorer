import pytest
import numpy as np

from games.tic_tac_toe import TicTacToe
from games.game_state import GameState
from agent.agent import State


# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
def test_init():
    game = TicTacToe()
    assert game.winner is None
    assert game.current_player() == 1
    assert game.state.board.shape == (3, 3)
    assert np.all(game.state.board == None)


# ------------------------------------------------------------
# Move and turn mechanics
# ------------------------------------------------------------
def test_valid_moves_initial():
    game = TicTacToe()
    moves = game.valid_moves()
    assert moves.shape == (9, 2)
    # All moves refer to empty cells
    for r, c in moves:
        assert game.state.board[r, c] is None


def test_apply_move_and_turn_switch():
    game = TicTacToe()

    game.apply_move(np.array([1, 1]))
    assert game.state.board[1, 1] == 1
    assert game.current_player() == 2

    game.apply_move(np.array([0, 0]))
    assert game.state.board[0, 0] == 2
    assert game.current_player() == 1


# ------------------------------------------------------------
# Invalid moves (core reliability for multi-agent usage)
# ------------------------------------------------------------
def test_overwrite_move_invalid():
    game = TicTacToe()
    game.apply_move(np.array([0, 0]))

    # Trying to play in same spot again should raise
    with pytest.raises(Exception):
        game.apply_move(np.array([0, 0]))


def test_invalid_coordinates():
    game = TicTacToe()

    with pytest.raises(Exception):
        game.apply_move(np.array([-1, 0]))

    with pytest.raises(Exception):
        game.apply_move(np.array([3, 1]))


# ------------------------------------------------------------
# Win detection (fundamental game logic)
# ------------------------------------------------------------
def test_row_win():
    game = TicTacToe()

    game.apply_move(np.array([0, 0]))  # P1
    game.apply_move(np.array([1, 0]))  # P2
    game.apply_move(np.array([0, 1]))  # P1
    game.apply_move(np.array([1, 1]))  # P2
    game.apply_move(np.array([0, 2]))  # P1 wins

    assert game.winner == 1
    assert game.is_over()
    assert game.get_result(1) == State.WIN
    assert game.get_result(2) == State.LOSS


def test_diagonal_win():
    game = TicTacToe()

    game.apply_move(np.array([0, 0]))  # P1
    game.apply_move(np.array([0, 1]))  # P2
    game.apply_move(np.array([1, 1]))  # P1
    game.apply_move(np.array([0, 2]))  # P2
    game.apply_move(np.array([2, 2]))  # P1 diag win

    assert game.winner == 1
    assert game.is_over()


# ------------------------------------------------------------
# Tie detection (no overwrites, no wins)
# ------------------------------------------------------------
def test_tie_game():
    game = TicTacToe()

    # A true tie game (no row/col/diag wins for either player)
    moves = [
        (0,0),(0,1),(0,2),
        (1,1),(1,0),(1,2),
        (2,1),(2,0),(2,2),
    ]

    for r, c in moves:
        game.apply_move(np.array([r, c]))

    assert game.winner is None
    assert game.is_over() is True



# ------------------------------------------------------------
# Cloning (critical for agent simulations)
# ------------------------------------------------------------
def test_clone_shallow():
    game = TicTacToe()
    clone = game.clone()

    assert clone is not game
    assert clone.state is game.state  # shallow clone shares state reference


def test_clone_deep():
    game = TicTacToe()
    deep = game.deep_clone()

    assert deep is not game
    assert deep.state is not game.state
    assert np.array_equal(deep.state.board, game.state.board)


# ------------------------------------------------------------
# set_state (agents will use this A LOT)
# ------------------------------------------------------------
def test_set_state():
    game = TicTacToe()
    new_board = np.array([[1, None, None],
                        [None, 2, None],
                        [None, None, None]])
    new_state = GameState(new_board, current_player=2)

    game.set_state(new_state)

    assert game.state is new_state
    assert game.current_player() == 2
    assert np.array_equal(game.state.board, new_board)
