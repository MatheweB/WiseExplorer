"""
Nim game implementation.

Two players take turns removing objects from heaps.
The player who takes the last object WINS (normal play).
Board size n gives staircase heaps [1, 2, 3, ..., n].

Uses int8 board (1D array of heap sizes):
    0 = empty heap
    positive = objects remaining
"""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray

from wise_explorer.agent.agent import State
from wise_explorer.games.game_base import GameBase
from wise_explorer.games.game_state import GameState

CELL_STRINGS = {0: " ", 1: "|"}


class Nim(GameBase):
    """Nim with staircase heaps [1, 2, ..., n]."""

    __slots__ = ('state', 'winner', 'n')

    def __init__(self, n: int = 4):
        self.n = n
        self.state = GameState(self._initial_heaps(n), current_player=1)
        self.winner = 0  # 0=none, 1=player1 won, 2=player2 won

    @staticmethod
    def _initial_heaps(n: int) -> np.ndarray:
        return np.arange(1, n + 1, dtype=np.int8)

    def get_cell_strings(self) -> dict[int, str]:
        return CELL_STRINGS

    def move_str(self, move: NDArray) -> str:
        heap_idx, num_remove = int(move[0]), int(move[1])
        return f"H{heap_idx + 1}×{num_remove}"

    def game_id(self) -> str:
        return "nim"

    def num_players(self) -> int:
        return 2

    def clone(self) -> "Nim":
        g = Nim.__new__(Nim)
        g.n = self.n
        g.state = self.state
        g.winner = self.winner
        return g

    def deep_clone(self) -> "Nim":
        g = Nim.__new__(Nim)
        g.n = self.n
        g.state = self.state.copy()
        g.winner = self.winner
        return g

    def get_state(self) -> GameState:
        return self.state

    def set_state(self, game_state: GameState) -> None:
        self.state = game_state
        self.n = len(game_state.board)
        self.winner = self._compute_winner()

    def current_player(self) -> int:
        return self.state.current_player

    def valid_moves(self) -> List[NDArray]:
        """Return all [heap_index, num_to_remove] moves."""
        moves = []
        for i, count in enumerate(self.state.board):
            for take in range(1, int(count) + 1):
                moves.append(np.array([i, take], dtype=np.int8))
        return moves

    def apply_move(self, move: NDArray, *, validated: bool = False) -> None:
        heap_idx, num_remove = int(move[0]), int(move[1])

        if not validated:
            if heap_idx < 0 or heap_idx >= self.n:
                raise ValueError(f"Invalid heap index: {heap_idx}")
            if num_remove < 1 or num_remove > self.state.board[heap_idx]:
                raise ValueError(
                    f"Cannot remove {num_remove} from heap {heap_idx} "
                    f"(has {self.state.board[heap_idx]})"
                )

        player = self.state.current_player
        self.state.board[heap_idx] -= num_remove

        # Normal play: last player to move wins
        if np.all(self.state.board == 0):
            self.winner = player

        self.state.current_player = 3 - player  # Toggle 1↔2

    def is_over(self) -> bool:
        return self.winner != 0

    def get_result(self, agent_id: int) -> State:
        if self.winner == agent_id:
            return State.WIN
        if self.winner != 0:
            return State.LOSS
        return State.NEUTRAL

    def _compute_winner(self) -> int:
        """Recompute winner from state. Used after set_state."""
        if np.all(self.state.board == 0):
            # All heaps empty — the player whose turn it is NOT must have
            # taken the last object, so the OTHER player won.
            return 3 - self.state.current_player
        return 0

    def state_string(self) -> str:
        heaps = self.state.board
        nim_sum = int(np.bitwise_xor.reduce(heaps))
        player = self.state.current_player

        lines = [f"Nim (n={self.n})  Player {player}'s turn"]
        lines.append("─" * (self.n + 12))
        for i, count in enumerate(heaps):
            bar = "│" * int(count)
            lines.append(f"  Heap {i + 1}: {bar} ({int(count)})")
        lines.append("─" * (self.n + 12))

        if nim_sum != 0:
            lines.append(f"  Nim-sum: {nim_sum} → Player {player} wins with perfect play")
        else:
            lines.append(f"  Nim-sum: 0 → Player {3 - player} wins with perfect play")

        return "\n".join(lines)
