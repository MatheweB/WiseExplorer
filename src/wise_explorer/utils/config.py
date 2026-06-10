"""
Configuration and game registry.
"""

import math
from pathlib import Path
from typing import Optional

import numpy as np

from wise_explorer.games import TicTacToe, MiniChess, Nim
from wise_explorer.games.game_state import GameState
from wise_explorer.simulation import DEFAULT_WORKER_COUNT


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).parent.parent  # src/wise_explorer/
# Trained memories live in the *working directory*, not inside the package —
# visible, per-project, and safe for non-editable installs.
DATA_DIR = Path.cwd() / "data"
MEMORY_DIR = DATA_DIR / "memory"


# ---------------------------------------------------------------------------
# Game Registry
# ---------------------------------------------------------------------------

GAMES = {
    "tic_tac_toe": TicTacToe,
    "minichess": MiniChess,
    "nim": Nim,
}

# Initial states use int8 encoding (0 = empty)
INITIAL_STATES = {
    "tic_tac_toe": GameState(
        np.zeros((3, 3), dtype=np.int8),
        current_player=1,
    ),
    "minichess": GameState(
        MiniChess._initial_board(),  # Static method
        current_player=1,
    ),
    "nim": GameState(
        Nim._initial_heaps(4),
        current_player=1,
    ),
}


# ---------------------------------------------------------------------------
# Default Settings
# ---------------------------------------------------------------------------

def _round_epochs_for_clean_split(epochs: int, num_players: int) -> int:
    """Round epochs up to ensure clean division in training phases."""
    divisor = num_players + 1
    return math.ceil(epochs / divisor) * divisor


class Config:
    """Training configuration with sensible defaults."""

    def __init__(
        self,
        game_name: str = "tic_tac_toe",
        epochs: int = 100,
        turn_depth: int = 40,
        num_workers: int = DEFAULT_WORKER_COUNT,
        gamma: float = 1.0,
        max_ply: Optional[int] = None,
    ):
        self.game_name = game_name
        self.turn_depth = turn_depth
        self.num_workers = num_workers
        self.gamma = gamma
        self.max_ply = max_ply

        # Derive dependent values
        game_class = GAMES[game_name]
        num_players = game_class().num_players()

        self.epochs = _round_epochs_for_clean_split(epochs, num_players)
        self.num_agents = num_workers
        self.simulations = self.epochs * self.num_agents


# Default configuration
DEFAULT_CONFIG = Config()