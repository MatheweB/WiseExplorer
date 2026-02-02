"""
TicTacToe game implementation - optimized.

Uses int8 board:
    0 = empty
    1 = player 1 (X)
    2 = player 2 (O)
"""

from __future__ import annotations

import numpy as np

from wise_explorer.agent.agent import State
from wise_explorer.games.game_base import GameBase
from wise_explorer.games.game_state import GameState

# Cell strings: each cell value maps to its display string
CELL_STRINGS = {0: " ", 1: "X", 2: "O"}

# Pre-computed winning lines (indices into flattened 3x3 board)
_WIN_LINES = np.array([
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
    [0, 4, 8], [2, 4, 6],             # diagonals
], dtype=np.int8)


class TicTacToe(GameBase):
    """Optimized TicTacToe implementation."""

    __slots__ = ('state', 'winner')

    def __init__(self):
        self.state = GameState(np.zeros((3, 3), dtype=np.int8), current_player=1)
        self.winner = 0  # 0=none, 1=player1, 2=player2

    def get_cell_strings(self) -> dict[int, str]:
        return CELL_STRINGS

    def game_id(self) -> str:
        return "tic_tac_toe"

    def num_players(self) -> int:
        return 2

    def clone(self) -> "TicTacToe":
        g = TicTacToe.__new__(TicTacToe)
        g.state = self.state
        g.winner = self.winner
        return g

    def deep_clone(self) -> "TicTacToe":
        g = TicTacToe.__new__(TicTacToe)
        g.state = self.state.copy()
        g.winner = self.winner
        return g

    def get_state(self) -> GameState:
        return self.state

    def set_state(self, game_state: GameState) -> None:
        self.state = game_state
        # Recompute winner from state
        self.winner = self._compute_winner()

    def current_player(self) -> int:
        return self.state.current_player

    def valid_moves(self) -> np.ndarray:
        """Return empty cell positions as array of shape (N, 2)."""
        return np.argwhere(self.state.board == 0)

    def apply_move(self, move: np.ndarray, *, validated: bool = False) -> None:
        r, c = int(move[0]), int(move[1])

        if not validated and self.state.board[r, c] != 0:
            raise ValueError(f"Cell ({r},{c}) is occupied")

        player = self.state.current_player
        self.state.board[r, c] = player

        # Check winner using flattened view
        flat = self.state.board.ravel()
        for line in _WIN_LINES:
            if flat[line[0]] == player and flat[line[1]] == player and flat[line[2]] == player:
                self.winner = player
                break

        self.state.current_player = 3 - player  # Toggle 1↔2

    def is_over(self) -> bool:
        return self.winner != 0 or not np.any(self.state.board == 0)

    def get_result(self, agent_id: int) -> State:
        if self.winner == agent_id:
            return State.WIN
        if self.winner != 0:
            return State.LOSS
        if not np.any(self.state.board == 0):
            return State.TIE
        return State.NEUTRAL

    def _compute_winner(self) -> int:
        """Recompute winner from current board state."""
        flat = self.state.board.ravel()
        for line in _WIN_LINES:
            v = flat[line[0]]
            if v != 0 and flat[line[1]] == v and flat[line[2]] == v:
                return int(v)
        return 0

    def state_string(self) -> str:
        board = self.state.board
        lines = ["╭───┬───┬───╮"]
        for i in range(3):
            row = "│ " + " │ ".join(CELL_STRINGS[board[i, j]] for j in range(3)) + " │"
            lines.append(row)
            if i < 2:
                lines.append("├───┼───┼───┤")
        lines.append("╰───┴───┴───╯")
        return "\n".join(lines)
