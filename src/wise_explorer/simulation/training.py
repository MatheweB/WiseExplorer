"""
Training orchestration for game AI.

Alternates between:
- Prune phase: One player plays worst moves (explores weaknesses)
- Exploit phase: All players play best moves (reinforces strengths)
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wise_explorer.agent.agent import Agent
    from wise_explorer.games.game_base import GameBase
    from wise_explorer.simulation.runner import SimulationRunner


def run_training(
    runner: SimulationRunner,
    swarms: dict[int, list[Agent]],
    game: GameBase,
    simulations: int,
    turn_depth: int,
) -> int:
    """
    Run training with 50/50 Prune vs Exploit split.
    
    Returns total transitions recorded.
    """
    player_ids = sorted(swarms.keys())
    num_players = len(player_ids)
    
    if simulations <= 0 or num_players == 0:
        return 0

    total = 0
    prune_sims = simulations // 2
    exploit_sims = simulations - prune_sims

    # Prune phase - each player takes turns being "pruned"
    if prune_sims > 0:
        sims_per_player = prune_sims // num_players
        remainder = prune_sims % num_players
        
        for i, pid in enumerate(player_ids):
            batch = sims_per_player + (1 if i < remainder else 0)
            if batch > 0:
                total += runner.run_batch(
                    swarms, game,
                    num_sims=batch,
                    max_turns=turn_depth,
                    prune_players={pid},
                )

    # Exploit phase
    if exploit_sims > 0:
        total += runner.run_batch(
            swarms, game,
            num_sims=exploit_sims,
            max_turns=turn_depth,
            prune_players=set(),
        )

    # Discover from the converged values, then let the library heal the value
    # graph's blind spots (the value loop)
    runner.memory.grow_concepts(game=game)

    return total
