"""
Training orchestration for game AI.

Alternates between:
- Prune phase: One player plays worst moves (explores weaknesses)
- Exploit phase: All players play best moves (reinforces strengths)
"""

from __future__ import annotations
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from wise_explorer.agent.agent import Agent
    from wise_explorer.games.game_base import GameBase
    from wise_explorer.simulation.runner import SimulationRunner


def run_training(
    runner: "SimulationRunner",
    swarms: Dict[int, List["Agent"]],
    game: "GameBase",
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

    return total


def run_training_interleaved(
    runner: "SimulationRunner",
    swarms: Dict[int, List["Agent"]],
    game: "GameBase",
    simulations: int,
    turn_depth: int,
    phase_size: int = 50,
) -> int:
    """
    Interleaved training: rapidly alternate prune/exploit phases.
    
    Better learning dynamics by mixing exploration/exploitation throughout
    rather than in two big blocks.
    
    Phase order (2-player example):
        prune_p1 → prune_p2 → exploit → prune_p1 → prune_p2 → exploit → ...
    
    Args:
        phase_size: Simulations per phase before switching.
                    Smaller = more interleaved, larger = longer stretches.
    
    Returns:
        Total transitions recorded.
    """
    player_ids = sorted(swarms.keys())
    num_players = len(player_ids)
    
    if simulations <= 0 or num_players == 0:
        return 0

    total = 0
    remaining = simulations
    phase = 0  # 0..num_players-1 = prune each player, num_players = exploit
    
    while remaining > 0:
        batch = min(phase_size, remaining)
        
        if phase < num_players:
            prune = {player_ids[phase]}
        else:
            prune = set()
        
        total += runner.run_batch(
            swarms, game,
            num_sims=batch,
            max_turns=turn_depth,
            prune_players=prune,
        )
        
        remaining -= batch
        phase = (phase + 1) % (num_players + 1)

    return total