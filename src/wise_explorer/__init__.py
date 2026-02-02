"""
Game AI - Pattern-based learning for game AI using Monte Carlo simulations.

This package provides tools for training game-playing agents through
parallel simulations with Bayes factor clustering.

Quick Start:
    from game_ai import start_simulations, GameMemory
    
    game = YourGame()
    memory = GameMemory.for_game(game)
    swarms = {1: [Agent() for _ in range(20)], 2: [Agent() for _ in range(20)]}
    start_simulations(swarms, game, turn_depth=20, simulations=200, memory=memory)

Modules:
    core       - Fundamental types (Stats), hashing, Bayes factor statistics
    memory     - GameMemory database for storing and retrieving learned moves
    selection  - Move selection strategies for training and inference
    simulation - Parallel game execution and training orchestration
    debug      - Visualization tools for development
"""

from wise_explorer.api import (
    start_simulations,
    GameMemory,
    select_move,
    select_move_for_training,
    SimulationRunner,
    DEFAULT_WORKER_COUNT,
)

from wise_explorer.core import Stats

__version__ = "1.0.0"

__all__ = [
    # Main API
    "start_simulations",
    "GameMemory",
    "select_move",
    "select_move_for_training",
    "SimulationRunner",
    "DEFAULT_WORKER_COUNT",
    # Types
    "Stats",
]