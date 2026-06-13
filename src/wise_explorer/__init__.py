"""
Wise Explorer — zero-knowledge self-play that learns readable rules.

Quick start:
    from wise_explorer import train, play
    from wise_explorer.memory import for_game
    from wise_explorer.games import TicTacToe

    game, memory = TicTacToe(), for_game(TicTacToe())
    train(memory, game, games=2000)
    play(memory, game, human_players=[1])

Modules:
    core       - Fundamental types (Stats) and hashing
    memory     - GameMemory: stored transitions, the value loop, certificates
    selection  - Move selection (the evidence ladder)
    simulation - Parallel self-play and training orchestration
"""

from wise_explorer.api import (
    train,
    play,
    start_simulations,
    GameMemory,
    select_move,
    SimulationRunner,
    DEFAULT_WORKER_COUNT,
)
from wise_explorer.core import Stats

__version__ = "2.0.0"

__all__ = [
    "train",
    "play",
    "start_simulations",
    "GameMemory",
    "select_move",
    "SimulationRunner",
    "DEFAULT_WORKER_COUNT",
    "Stats",
]
