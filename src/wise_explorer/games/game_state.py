"""
GameState - immutable game state container.

Optimized for fast copying and hashing.
"""

from __future__ import annotations

import numpy as np


class GameState:
    """
    Lightweight game state container.
    
    Uses int8 board for fast copy/hash:
        0 = empty
        1 = player 1's piece (or piece type for chess)
        2 = player 2's piece
        etc.
    """
    __slots__ = ('board', 'current_player')

    def __init__(self, board: np.ndarray, current_player: int):
        self.board = board
        self.current_player = current_player

    def copy(self) -> "GameState":
        """Fast copy - board.copy() is optimized for contiguous int arrays."""
        return GameState(self.board.copy(), self.current_player)