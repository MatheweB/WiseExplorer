from typing import Dict, Any, Optional
from copy import deepcopy
import numpy as np


class GameState:
    """
    Represents the full state of a board game at a given moment.
    This includes:
        - board (np.ndarray)
        - current_player (int)
        - meta (dict): Game-specific extras (castling rights, en passant, etc.)
    """

    def __init__(
        self,
        board: np.ndarray,
        current_player: int,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.board = board
        self.current_player = current_player
        self.meta = meta if meta is not None else {}

    def copy(self) -> "GameState":
        """Return a deep copy of this state."""
        return GameState(
            board=self.board.copy(),
            current_player=self.current_player,
            meta=deepcopy(self.meta) if self.meta else {},
        )

    # Keep clone() as alias for backward compatibility
    def clone(self) -> "GameState":
        return self.copy()
