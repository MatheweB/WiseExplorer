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
    final_cycle: bool = True,
) -> int:
    """
    Run training with a 50/50 prune/exploit split.

    ``final_cycle=False`` skips the closing value-loop cycle; callers that
    train in many small bursts (pondering, chunked progress reporting) rely on
    the in-run doubling cadence and run one cycle themselves at the end.

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

    if final_cycle:
        # One closing value-loop cycle: solve, complete, fit, prove, forget.
        # Any in-flight mid-run cycle joins first so writes never interleave.
        runner.join_wheel()
        runner.memory.grow_concepts(game=game)

    return total
