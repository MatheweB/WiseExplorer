"""
Game registry and paths.
"""

from pathlib import Path

import numpy as np

from wise_explorer.games import TicTacToe, MiniChess, Nim
from wise_explorer.games.game_state import GameState


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

# Max turns per simulated game, by game. The cap bounds cyclic games;
# acyclic games end on their own well before it.
TURN_DEPTHS = {
    "tic_tac_toe": 20,
    "minichess": 80,
    "nim": 60,
}


def default_turn_depth(game_id: str) -> int:
    return TURN_DEPTHS.get(game_id, 40)


# Self-play games run from the current position before each AI move during
# `play` — local learning. Light enough to be near-instant per move.
PONDER = {
    "tic_tac_toe": 100,
    "minichess": 200,
    "nim": 150,
}


def default_ponder(game_id: str) -> int:
    return PONDER.get(game_id, 100)
