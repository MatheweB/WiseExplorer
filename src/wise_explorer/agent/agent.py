"""
Agent and game-result types for self-play simulation.
"""

from dataclasses import dataclass
from enum import Enum, auto


class State(Enum):
    WIN = auto()
    TIE = auto()
    LOSS = auto()
    NEUTRAL = auto()


@dataclass
class Agent:
    """One member of a player's self-play population.

    A swarm of agents per player is a *sampler*: each agent is one independent
    play-through, and the swarm size sets how many samples a position receives.
    Move choice is decided globally from the accumulated statistics (see
    ``wise_explorer.selection``), so an agent carries only the role it plays.
    """

    player_id: int = 0  # the role this agent fills (e.g. 1 or 2)
