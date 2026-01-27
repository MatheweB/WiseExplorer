"""
Tests for wise_explorer.games.minichess

Tests MiniChess game implementation.
"""

import numpy as np
import pytest

from wise_explorer.agent.agent import State
from wise_explorer.games.game_state import GameState
from wise_explorer.games.minichess import MiniChess, EMPTY, PAWN, CASTLE, KING, QUEEN


@pytest.fixture
def game() -> MiniChess:
    """Fresh MiniChess game."""
    return MiniChess()


class TestInitialization:
    """Initial game state tests."""

    def test_board_dimensions(self, game: MiniChess):
        """Board is 6x4."""
        assert game.get_state().board.shape == (6, 4)

    def test_initial_player(self, game: MiniChess):
        """Player 1 starts."""
        assert game.current_player() == 1

    def test_player1_setup(self, game: MiniChess):
        """Player 1 pieces are correctly placed."""
        board = game.get_state().board
        assert board[0, 0] == CASTLE
        assert board[0, 1] == KING
        assert board[0, 2] == QUEEN
        assert board[0, 3] == CASTLE
        assert np.all(board[1] == PAWN)

    def test_player2_setup(self, game: MiniChess):
        """Player 2 pieces are negative and correctly placed."""
        board = game.get_state().board
        assert board[5, 0] == -CASTLE
        assert board[5, 1] == -KING
        assert board[5, 2] == -QUEEN
        assert board[5, 3] == -CASTLE
        assert np.all(board[4] == -PAWN)

    def test_middle_empty(self, game: MiniChess):
        """Middle rows are empty."""
        board = game.get_state().board
        assert np.all(board[2:4] == 0)

    def test_metadata(self, game: MiniChess):
        """Game metadata is correct."""
        assert game.game_id() == "mini_chess"
        assert game.num_players() == 2


class TestValidMoves:
    """Valid moves tests."""

    def test_initial_moves_exist(self, game: MiniChess):
        """Valid moves exist initially."""
        moves = game.valid_moves()
        assert len(moves) > 0

    def test_moves_format(self, game: MiniChess):
        """Moves are [from_r, from_c, to_r, to_c]."""
        moves = game.valid_moves()
        for move in moves:
            assert len(move) == 4

    def test_initial_pawn_moves(self, game: MiniChess):
        """All 4 P1 pawns can move forward initially."""
        moves = game.valid_moves()
        pawn_moves = [m for m in moves if m[0] == 1 and m[2] == 2]
        assert len(pawn_moves) == 4


class TestPawnMovement:
    """Pawn movement tests."""

    def test_pawn_moves_forward(self, game: MiniChess):
        """Pawn moves forward to empty square."""
        game.apply_move(np.array([1, 0, 2, 0]))
        board = game.get_state().board
        assert board[1, 0] == EMPTY
        assert board[2, 0] == PAWN

    def test_pawn_diagonal_capture(self, game: MiniChess):
        """Pawn captures diagonally."""
        # Setup: P1 pawn to (2,1), P2 pawn to (3,0), then P2 can capture
        game.apply_move(np.array([1, 1, 2, 1]))  # P1 pawn
        game.apply_move(np.array([4, 0, 3, 0]))  # P2 pawn
        game.apply_move(np.array([1, 2, 2, 2]))  # P1 (filler)
        
        # P2 pawn at (3,0) should capture P1 pawn at (2,1)
        moves = game.valid_moves()
        capture = [m for m in moves if tuple(m) == (3, 0, 2, 1)]
        assert len(capture) == 1


class TestPieceMovement:
    """General piece movement tests."""

    def test_castle_slides(self, game: MiniChess):
        """Castle (rook) slides orthogonally."""
        # Move pawn out of way
        game.apply_move(np.array([1, 0, 2, 0]))
        game.apply_move(np.array([4, 0, 3, 0]))  # P2
        
        # Castle at (0,0) should be able to move to (1,0)
        moves = game.valid_moves()
        castle_moves = [m for m in moves if m[0] == 0 and m[1] == 0]
        assert any(m[2] == 1 and m[3] == 0 for m in castle_moves)

    def test_king_moves_one_step(self, game: MiniChess):
        """King moves only one step."""
        # Move pawn to clear path
        game.apply_move(np.array([1, 1, 2, 1]))
        game.apply_move(np.array([4, 0, 3, 0]))  # P2
        
        # King at (0,1) should move to (1,1)
        moves = game.valid_moves()
        king_moves = [m for m in moves if m[0] == 0 and m[1] == 1]
        
        for m in king_moves:
            assert abs(m[2] - m[0]) <= 1 and abs(m[3] - m[1]) <= 1


class TestCaptures:
    """Capture mechanics tests."""

    def test_cannot_capture_own_piece(self, game: MiniChess):
        """Cannot capture own pieces."""
        moves = game.valid_moves()
        board = game.get_state().board
        
        for move in moves:
            target = board[move[2], move[3]]
            assert target <= 0  # P1's turn: can only capture P2 (negative) or empty


class TestGameEnding:
    """Game ending tests."""

    def test_king_capture_ends_game(self):
        """Capturing the king ends the game."""
        game = MiniChess()
        
        # Setup: Kings in capture range
        board = np.zeros((6, 4), dtype=np.int8)
        board[0, 1] = KING
        board[1, 1] = -KING
        
        game.set_state(GameState(board, current_player=1))
        game.winner = 0
        
        game.apply_move(np.array([0, 1, 1, 1]))  # King captures king
        
        assert game.is_over()
        assert game.winner == 1

    def test_max_moves_ends_game(self, game: MiniChess):
        """Game ends at MAX_MOVES."""
        game.move_count = MiniChess.MAX_MOVES
        assert game.is_over()


class TestGetResult:
    """Result reporting tests."""

    def test_result_after_king_capture(self):
        """Correct results after king capture."""
        game = MiniChess()
        
        board = np.zeros((6, 4), dtype=np.int8)
        board[2, 2] = QUEEN
        board[2, 3] = KING
        board[5, 2] = -KING
        
        game.set_state(GameState(board, current_player=1))
        game.apply_move(np.array([2, 2, 5, 2]))  # Queen captures king
        
        assert game.get_result(1) == State.WIN
        assert game.get_result(2) == State.LOSS

    def test_result_at_max_moves(self, game: MiniChess):
        """TIE when max moves reached."""
        game.move_count = MiniChess.MAX_MOVES
        assert game.get_result(1) == State.TIE
        assert game.get_result(2) == State.TIE


class TestCloning:
    """Game cloning tests."""

    def test_deep_clone_independent(self, game: MiniChess):
        """Deep clone is independent."""
        clone = game.deep_clone()
        game.apply_move(np.array([1, 0, 2, 0]))
        
        assert clone.get_state().board[2, 0] == EMPTY
        assert game.get_state().board[2, 0] == PAWN