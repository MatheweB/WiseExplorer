"""
Factory functions for creating games and agent swarms.
"""

from typing import Dict, List

from wise_explorer.agent.agent import Agent
from wise_explorer.games.game_base import GameBase
from wise_explorer.games.game_state import GameState
from wise_explorer.utils.config import GAMES, INITIAL_STATES


def create_agent_swarms(
    players: List[int],
    agents_per_player: int,
) -> Dict[int, List[Agent]]:
    """
    Create a swarm of agents for each player.

    Args:
        players: List of player IDs (e.g., [1, 2])
        agents_per_player: Number of agents per player

    Returns:
        Dict mapping player ID to list of agents
    """
    swarms: Dict[int, List[Agent]] = {}

    for pid in players:
        swarm = [Agent() for _ in range(agents_per_player)]
        for agent in swarm:
            agent.player_id = pid
        swarms[pid] = swarm

    return swarms


def create_game(game_name: str) -> GameBase:
    """
    Create a game instance with its initial state.

    Args:
        game_name: Key from GAMES registry (e.g., "tic_tac_toe")

    Returns:
        Configured game instance
    """
    if game_name not in GAMES:
        available = ", ".join(GAMES.keys())
        raise ValueError(f"Unknown game: {game_name}. Available: {available}")

    game_class = GAMES[game_name]
    initial_state: GameState = INITIAL_STATES[game_name]

    game = game_class()
    game.set_state(initial_state.copy())

    return game