import pytest
import numpy as np

from games.game_state import GameState


def test_game_state_init():
    board = np.array([[1, 2], [3, 4]])
    state = GameState(board=board, current_player=1)

    assert np.array_equal(state.board, board)
    assert state.current_player == 1


def test_game_state_clone_independent_board():
    board = np.array([[0, 1], [1, 0]])
    state = GameState(board=board, current_player=0)

    cloned = state.clone()

    # Contents equal
    assert np.array_equal(cloned.board, board)

    # But not same reference (deep copy)
    assert cloned.board is not state.board


def test_game_state_clone_current_player_copied():
    state = GameState(board=np.zeros((2, 2)), current_player=1)
    cloned = state.clone()

    assert cloned.current_player == state.current_player


def test_game_state_clone_mutation_does_not_affect_original():
    board = np.array([[9, 9], [9, 9]])
    state = GameState(board=board, current_player=1)
    cloned = state.clone()

    # Mutate clone
    cloned.board[0, 0] = 0

    # Original must remain unchanged
    assert state.board[0, 0] == 9
    assert cloned.board[0, 0] == 0
