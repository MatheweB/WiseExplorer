"""
Utilities for exploring move space in a two-player game.

A *change* flag on each agent tells us whether that agent has won
(`change == False`) or lost (`change == True`) the last round.
"""

from __future__ import annotations

import random

from agent.agent import Agent
from games.game_base import GameBase
from omnicron.manager import GameMemory


def _set_move(
    agent: Agent, game: GameBase, memory: GameMemory, is_prune_stage: bool
) -> None:
    """
    Update ``agent.core_move`` according to the current search stage.

    Parameters
    ----------
    agent:
        The agent whose move is being updated.
    is_prune_stage:
        * ``True`` - we are in the *pruning* (bad-path) phase:
        We bias towards picking the known worst move. 
        If the worst move has a high score, we have a higher probability of choosing random over it.
        If the worst move has a low score, we have a higher probability of choosing it over random.
        If there is no worst move, we choose random.

        * ``False`` - we are in the *exploration* (good-path) phase:
        We bias towards picking the best know move. 
        If the best move has a high score, we have a higher probability of choosing it over random.
        If the best move has a low score, we have a higher probability of choosing random over it.
        If there is no best move, we choose random.

    Notes
    -----
    The helper works for both normal agents and “anti-agents”; it
    simply calls the agent's public API.
    """
    valid = game.valid_moves()
    
    if is_prune_stage:
        result = memory.get_worst_move_with_score(game)
        if result is not None:
            move, score = result
            exploit_prob = (score + 1) / 2
            # We expect worst move to be low, so picking is more likely
            if random.random() > exploit_prob:
                agent.core_move = move
            else:
                agent.core_move = random.choice(valid)
        else:
            agent.core_move = random.choice(valid)

    else:
        result = memory.get_best_move_with_score(game)
        if result is not None:
            move, score = result
            exploit_prob = (score + 1) / 2
            # We expect best move to be high, so picking is more likely
            if random.random() < exploit_prob:
                agent.core_move = move
            else:
                agent.core_move = random.choice(valid)
        else:
            agent.core_move = random.choice(valid)


def update_agent(
    agent: Agent,
    game: GameBase,
    omnicron: GameMemory,
    is_prune_stage: bool = True,
) -> None:
    """
    Update the core moves of all agents based on the current search stage.

    Parameters
    ----------
    agent : Agent
        The agent in the current game state.
    game : GameBase
        The game instance; needed for generating random valid moves.
    is_prune_stage : bool, optional
        ``True`` if we are currently exploring bad paths (pruning).
        Defaults to ``True``.

    This function mutates the agents in place - no value is returned.
    """
    # Treat every agent uniformly: the helper handles the logic
    _set_move(agent, game, omnicron, is_prune_stage)
