"""
Job data structures for parallel simulation.

Defines the input (GameJob) and output (JobResult) types used
by worker processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wise_explorer.agent.agent import State
    from wise_explorer.games.game_base import GameBase


@dataclass
class MoveRecord:
    """A single move with its context."""
    move: np.ndarray
    board_before: np.ndarray
    player: int


@dataclass(frozen=True)
class GameJob:
    """
    Self-contained job for a worker process.
    
    Contains everything needed to run a single game simulation
    without requiring shared state.
    """
    game: "GameBase"
    player_map: Dict[int, int]  # player_id -> agent_index
    max_turns: int
    prune_players: Set[int]  # Player IDs that should play worst moves


@dataclass
class JobResult:
    """
    Result from a completed game simulation.
    
    Contains all moves made and outcomes for each player,
    ready to be committed to memory.
    """
    moves: Dict[int, List[MoveRecord]]
    outcomes: Dict[int, "State"]
    player_map: Dict[int, int]
    game_class: type