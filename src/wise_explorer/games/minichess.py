"""
Mini Chess implementation - optimized.

Board encoding (int8):
    0 = empty
    Positive = Player 1: 1=Pawn, 2=Castle, 3=King, 4=Queen
    Negative = Player 2: -1=Pawn, -2=Castle, -3=King, -4=Queen

This allows fast player checks: piece > 0 → P1, piece < 0 → P2
"""

from __future__ import annotations

import numpy as np

from wise_explorer.agent.agent import State
from wise_explorer.games.game_base import GameBase
from wise_explorer.games.game_state import GameState


# Piece type constants
EMPTY = 0
PAWN = 1
CASTLE = 2
KING = 3
QUEEN = 4

# Direction vectors (dr, dc)
ORTHOGONAL = ((0, 1), (0, -1), (1, 0), (-1, 0))
DIAGONAL = ((1, 1), (1, -1), (-1, 1), (-1, -1))
ALL_DIRS = ORTHOGONAL + DIAGONAL
KING_DIRS = ALL_DIRS  # King moves 1 step in any direction

CELL_STRINGS = {
    0: "  ",
    1: "P1", -1: "P2",
    2: "C1", -2: "C2",
    3: "K1", -3: "K2",
    4: "Q1", -4: "Q2",
}


class MiniChess(GameBase):
    """Optimized Mini Chess on 6x4 board."""

    __slots__ = ('state', 'winner', 'move_count')

    ROWS = 6
    COLS = 4
    MAX_MOVES = 100

    def __init__(self):
        self.state = GameState(self._initial_board(), current_player=1)
        self.winner = 0
        self.move_count = 0

    @staticmethod
    def _initial_board() -> np.ndarray:
        """Create starting position."""
        board = np.zeros((6, 4), dtype=np.int8)
        # Player 1 (positive): top rows
        board[0] = [CASTLE, KING, QUEEN, CASTLE]
        board[1] = [PAWN, PAWN, PAWN, PAWN]
        # Player 2 (negative): bottom rows
        board[4] = [-PAWN, -PAWN, -PAWN, -PAWN]
        board[5] = [-CASTLE, -KING, -QUEEN, -CASTLE]
        return board

    def game_id(self) -> str:
        return "mini_chess"

    def get_cell_strings(self) -> dict[int, str]:
        return CELL_STRINGS

    def num_players(self) -> int:
        return 2

    def clone(self) -> "MiniChess":
        g = MiniChess.__new__(MiniChess)
        g.state = self.state
        g.winner = self.winner
        g.move_count = self.move_count
        return g

    def deep_clone(self) -> "MiniChess":
        g = MiniChess.__new__(MiniChess)
        g.state = self.state.copy()
        g.winner = self.winner
        g.move_count = self.move_count
        return g

    def get_state(self) -> GameState:
        return self.state

    def set_state(self, game_state: GameState) -> None:
        self.state = game_state

    def current_player(self) -> int:
        return self.state.current_player

    def valid_moves(self) -> np.ndarray:
        """
        Return all legal moves as Nx4 int32 array.
        Each row: [from_r, from_c, to_r, to_c]
        """
        board = self.state.board
        player = self.state.current_player
        is_p1 = player == 1
        
        moves = []
        
        for r in range(self.ROWS):
            for c in range(self.COLS):
                piece = board[r, c]
                
                # Check ownership: P1 has positive pieces, P2 has negative
                if (is_p1 and piece > 0) or (not is_p1 and piece < 0):
                    piece_type = abs(piece)
                    
                    if piece_type == PAWN:
                        self._add_pawn_moves(moves, board, r, c, is_p1)
                    elif piece_type == CASTLE:
                        self._add_line_moves(moves, board, r, c, is_p1, ORTHOGONAL)
                    elif piece_type == KING:
                        self._add_step_moves(moves, board, r, c, is_p1, KING_DIRS)
                    elif piece_type == QUEEN:
                        self._add_line_moves(moves, board, r, c, is_p1, ALL_DIRS)

        if not moves:
            return np.zeros((0, 4), dtype=np.int32)
        
        return np.array(moves, dtype=np.int32)

    def _add_pawn_moves(self, moves: list, board: np.ndarray, r: int, c: int, is_p1: bool):
        """Add pawn moves: forward 1, capture diagonal."""
        dr = 1 if is_p1 else -1
        nr = r + dr
        
        if 0 <= nr < self.ROWS:
            # Forward move (must be empty)
            if board[nr, c] == 0:
                moves.append((r, c, nr, c))
            
            # Diagonal captures
            for dc in (-1, 1):
                nc = c + dc
                if 0 <= nc < self.COLS:
                    target = board[nr, nc]
                    # Can capture enemy piece (opposite sign, non-zero)
                    if (is_p1 and target < 0) or (not is_p1 and target > 0):
                        moves.append((r, c, nr, nc))

    def _add_step_moves(self, moves: list, board: np.ndarray, r: int, c: int, 
                        is_p1: bool, directions: tuple):
        """Add single-step moves (King)."""
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.ROWS and 0 <= nc < self.COLS:
                target = board[nr, nc]
                # Can move to empty or capture enemy
                if target == 0 or (is_p1 and target < 0) or (not is_p1 and target > 0):
                    moves.append((r, c, nr, nc))

    def _add_line_moves(self, moves: list, board: np.ndarray, r: int, c: int,
                        is_p1: bool, directions: tuple):
        """Add sliding moves (Castle, Queen)."""
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            while 0 <= nr < self.ROWS and 0 <= nc < self.COLS:
                target = board[nr, nc]
                
                if target == 0:
                    moves.append((r, c, nr, nc))
                    nr += dr
                    nc += dc
                elif (is_p1 and target < 0) or (not is_p1 and target > 0):
                    moves.append((r, c, nr, nc))
                    break
                else:
                    break

    def is_valid_move(self, move: np.ndarray | list | tuple) -> bool:
        """
        Check whether `move` is a legal move for the current player.

        Uses per-piece movement rules directly — O(board_dim) for sliding
        pieces, O(1) for pawns and king. Does not regenerate the full
        move list.
        """
        try:
            fr, fc, tr, tc = int(move[0]), int(move[1]), int(move[2]), int(move[3])
        except Exception:
            return False

        # Bounds + identity check
        if not (0 <= fr < self.ROWS and 0 <= fc < self.COLS
                and 0 <= tr < self.ROWS and 0 <= tc < self.COLS):
            return False
        if fr == tr and fc == tc:
            return False

        board = self.state.board
        piece = int(board[fr, fc])
        if piece == 0:
            return False

        # Ownership
        is_p1 = self.state.current_player == 1
        if (is_p1 and piece < 0) or (not is_p1 and piece > 0):
            return False

        # Target must be empty or enemy
        target = int(board[tr, tc])
        if target != 0:
            if (is_p1 and target > 0) or (not is_p1 and target < 0):
                return False

        piece_type = abs(piece)
        dr = tr - fr
        dc = tc - fc

        if piece_type == PAWN:
            fwd = 1 if is_p1 else -1
            if dr != fwd:
                return False
            if dc == 0:
                return target == 0       # Forward: must be empty
            if abs(dc) == 1:
                return target != 0       # Diagonal: must capture
            return False

        if piece_type == KING:
            return abs(dr) <= 1 and abs(dc) <= 1

        # Sliding pieces (CASTLE, QUEEN) — check direction + clear path
        if piece_type == CASTLE:
            if dr != 0 and dc != 0:
                return False             # Orthogonal only
        elif piece_type == QUEEN:
            if dr != 0 and dc != 0 and abs(dr) != abs(dc):
                return False             # Must be straight line

        # Walk the path — every intermediate square must be empty
        step_r = (1 if dr > 0 else -1) if dr != 0 else 0
        step_c = (1 if dc > 0 else -1) if dc != 0 else 0
        cr, cc = fr + step_r, fc + step_c
        while cr != tr or cc != tc:
            if board[cr, cc] != 0:
                return False
            cr += step_r
            cc += step_c

        return True

    def apply_move(self, move: np.ndarray | list | tuple, *, validated: bool = False) -> None:
        """Apply move: [from_r, from_c, to_r, to_c].

        Args:
            move: The move to apply.
            validated:  If True, skip validation (caller guarantees legality).
                        Use when move was already selected from valid_moves().

        Raises:
            RuntimeError: if the game is already over.
            ValueError: if move is invalid for current player.
        """
        if self.is_over():
            raise RuntimeError("Cannot apply move: the game is already over.")

        fr, fc, tr, tc = int(move[0]), int(move[1]), int(move[2]), int(move[3])

        if not validated and not self.is_valid_move((fr, fc, tr, tc)):
            raise ValueError(
                f"Invalid move {(fr, fc, tr, tc)} for player {self.state.current_player}"
            )

        board = self.state.board
        target = board[tr, tc]

        # Check if capturing King
        if abs(target) == KING:
            self.winner = self.state.current_player

        # Execute move
        board[tr, tc] = board[fr, fc]
        board[fr, fc] = 0

        self.move_count += 1
        self.state.current_player = 3 - self.state.current_player

    def is_over(self) -> bool:
        if self.winner != 0:
            return True
        if self.move_count >= self.MAX_MOVES:
            return True
        # Check if both kings exist
        board = self.state.board
        return not (np.any(board == KING) and np.any(board == -KING))

    def get_result(self, agent_id: int) -> State:
        if self.winner == agent_id:
            return State.WIN
        if self.winner != 0:
            return State.LOSS
        
        if self.move_count >= self.MAX_MOVES:
            return State.TIE
        
        # Check for king capture
        board = self.state.board
        my_king = KING if agent_id == 1 else -KING
        opp_king = -KING if agent_id == 1 else KING
        
        if not np.any(board == my_king):
            return State.LOSS
        if not np.any(board == opp_king):
            return State.WIN
        
        return State.NEUTRAL

    def state_string(self) -> str:
        """Pretty-print the board."""        
        board = self.state.board
        lines = ["╭────┬────┬────┬────╮"]
        
        for i in range(self.ROWS):
            row_strs = [f"{CELL_STRINGS[board[i, j]]:^2}" for j in range(self.COLS)]
            lines.append("│ " + " │ ".join(row_strs) + " │")
            if i < self.ROWS - 1:
                lines.append("├────┼────┼────┼────┤")
        
        lines.append("╰────┴────┴────┴────╯")
        lines.append(f"\nPlayer: {self.state.current_player}  Moves: {self.move_count}")
        
        return "\n".join(lines)
