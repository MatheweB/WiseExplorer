"""
Tests for wise_explorer.games.tic_tac_toe

Tests TicTacToe game implementation.
"""

import numpy as np
import pytest

from wise_explorer.agent.agent import State
from wise_explorer.games.game_state import GameState
from wise_explorer.games.tic_tac_toe import TicTacToe


@pytest.fixture
def game() -> TicTacToe:
    """Fresh TicTacToe game."""
    return TicTacToe()


class TestInitialization:
    """Initial game state tests."""

    def test_initial_state(self, game: TicTacToe):
        """Game starts with empty board, player 1, not over."""
        assert np.all(game.get_state().board == 0)
        assert game.current_player() == 1
        assert game.winner == 0
        assert game.is_over() is False

    def test_metadata(self, game: TicTacToe):
        """Game metadata is correct."""
        assert game.game_id() == "tic_tac_toe"
        assert game.num_players() == 2


class TestValidMoves:
    """Valid moves tests."""

    def test_initial_nine_moves(self, game: TicTacToe):
        """All 9 cells are valid initially."""
        moves = game.valid_moves()
        assert len(moves) == 9

    def test_moves_decrease_after_play(self, game: TicTacToe):
        """Valid moves decrease as game progresses."""
        game.apply_move(np.array([0, 0]))
        assert len(game.valid_moves()) == 8

    def test_occupied_not_in_moves(self, game: TicTacToe):
        """Occupied cells are not valid moves."""
        game.apply_move(np.array([1, 1]))
        positions = [(m[0], m[1]) for m in game.valid_moves()]
        assert (1, 1) not in positions


class TestApplyMove:
    """Move application tests."""

    def test_places_correct_marker(self, game: TicTacToe):
        """Move places current player's marker."""
        game.apply_move(np.array([0, 0]))
        assert game.get_state().board[0, 0] == 1

    def test_switches_player(self, game: TicTacToe):
        """Move switches current player."""
        game.apply_move(np.array([0, 0]))
        assert game.current_player() == 2
        game.apply_move(np.array([1, 1]))
        assert game.current_player() == 1

    def test_occupied_raises(self, game: TicTacToe):
        """Moving to occupied cell raises ValueError."""
        game.apply_move(np.array([0, 0]))
        with pytest.raises(ValueError):
            game.apply_move(np.array([0, 0]))


class TestWinDetection:
    """Win detection tests."""

    @pytest.mark.parametrize("winning_cells", [
        [(0, 0), (0, 1), (0, 2)],  # Top row
        [(1, 0), (1, 1), (1, 2)],  # Middle row
        [(2, 0), (2, 1), (2, 2)],  # Bottom row
        [(0, 0), (1, 0), (2, 0)],  # Left column
        [(0, 1), (1, 1), (2, 1)],  # Middle column
        [(0, 2), (1, 2), (2, 2)],  # Right column
        [(0, 0), (1, 1), (2, 2)],  # Main diagonal
        [(0, 2), (1, 1), (2, 0)],  # Anti-diagonal
    ])
    def test_all_win_lines(self, game: TicTacToe, winning_cells):
        """All 8 win lines are detected."""
        # Player 1 plays winning cells, Player 2 plays elsewhere
        p2_cells = [(r, c) for r in range(3) for c in range(3) 
                    if (r, c) not in winning_cells]
        
        for i, (r, c) in enumerate(winning_cells[:2]):
            game.apply_move(np.array([r, c]))  # P1
            game.apply_move(np.array(p2_cells[i]))  # P2
        
        # P1's winning move
        r, c = winning_cells[2]
        game.apply_move(np.array([r, c]))
        
        assert game.is_over()
        assert game.winner == 1

    def test_player_2_can_win(self, game: TicTacToe):
        """Player 2 can win."""
        # P1: (0,0), P2: (2,0), P1: (0,1), P2: (2,1), P1: (1,1), P2: (2,2)
        moves = [(0, 0), (2, 0), (0, 1), (2, 1), (1, 1), (2, 2)]
        for r, c in moves:
            game.apply_move(np.array([r, c]))
        
        assert game.winner == 2


class TestTieDetection:
    """Tie detection tests."""

    def test_full_board_no_winner_is_tie(self, game: TicTacToe):
        """Full board without winner is tie."""
        # X O X / X X O / O X O (tie game)
        moves = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (1, 1), (2, 2), (2, 1)]
        for r, c in moves:
            game.apply_move(np.array([r, c]))
        
        assert game.is_over()
        assert game.winner == 0


class TestGetResult:
    """Result reporting tests."""

    def test_result_for_winner(self, game: TicTacToe):
        """Winner gets WIN, loser gets LOSS."""
        # Quick P1 win on top row
        moves = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]
        for r, c in moves:
            game.apply_move(np.array([r, c]))
        
        assert game.get_result(1) == State.WIN
        assert game.get_result(2) == State.LOSS

    def test_result_for_tie(self, game: TicTacToe):
        """Both players get TIE on tie game."""
        moves = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (1, 1), (2, 2), (2, 1)]
        for r, c in moves:
            game.apply_move(np.array([r, c]))
        
        assert game.get_result(1) == State.TIE
        assert game.get_result(2) == State.TIE

    def test_result_neutral_during_game(self, game: TicTacToe):
        """In-progress game returns NEUTRAL."""
        game.apply_move(np.array([0, 0]))
        assert game.get_result(1) == State.NEUTRAL
        assert game.get_result(2) == State.NEUTRAL


class TestCloning:
    """Game cloning tests."""

    def test_deep_clone_independent(self, game: TicTacToe):
        """Deep clone has independent state."""
        game.apply_move(np.array([0, 0]))
        clone = game.deep_clone()
        
        clone.apply_move(np.array([1, 1]))
        
        assert game.get_state().board[1, 1] == 0  # Original unchanged
        assert clone.get_state().board[1, 1] == 2  # Clone changed


class TestSetState:
    """State setting tests."""

    def test_set_state_recomputes_winner(self, game: TicTacToe):
        """set_state recomputes winner from board."""
        winning_board = np.array([[1, 1, 1], [2, 2, 0], [0, 0, 0]], dtype=np.int8)
        game.set_state(GameState(winning_board, current_player=2))
        
        assert game.winner == 1
        assert game.is_over()