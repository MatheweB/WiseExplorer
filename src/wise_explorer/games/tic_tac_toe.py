"""
TicTacToe game implementation - optimized.

Uses int8 board:
    0 = empty
    1 = player 1 (X)
    2 = player 2 (O)
"""

from __future__ import annotations

from typing import List

import numpy as np

from wise_explorer.agent.agent import State
from wise_explorer.games.game_base import GameBase
from wise_explorer.games.game_state import GameState


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

    def valid_moves(self) -> List[np.ndarray]:
        """Return empty cell positions as array of [row, col]."""
        return np.argwhere(self.state.board == 0)

    def apply_move(self, move: np.ndarray) -> None:
        r, c = int(move[0]), int(move[1])
        
        if self.state.board[r, c] != 0:
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
                return v
        return 0

    def state_string(self) -> str:
        symbols = {0: " ", 1: "X", 2: "O"}
        board = self.state.board
        
        lines = ["╭───┬───┬───╮"]
        for i in range(3):
            row = "│ " + " │ ".join(symbols[board[i, j]] for j in range(3)) + " │"
            lines.append(row)
            if i < 2:
                lines.append("├───┼───┼───┤")
        lines.append("╰───┴───┴───╯")
        
        return "\n".join(lines)