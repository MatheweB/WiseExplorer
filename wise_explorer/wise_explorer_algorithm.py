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
    agent: Agent, game: GameBase, omnicron: GameMemory, is_prune_stage: bool
) -> None:
    """
    Update ``agent.core_move`` according to the current search stage.

    Parameters
    ----------
    agent:
        The agent whose move is being updated.
    is_prune_stage:
        * ``True`` - we are in the *pruning* (bad-path) phase:
        keep the losing agent on its current move and give a
        random move to the winning one.
        * ``False`` - we are in the *exploration* (good-path) phase:
        try a random move if the agent won, otherwise keep the best
        move found so far.

    Notes
    -----
    The helper works for both normal agents and “anti-agents”; it
    simply calls the agent's public API.
    """
    if is_prune_stage:
        # Bad‑path: keep losers, shuffle winners
        if not agent.change:  # agent won
            agent.core_move = random.choice(game.valid_moves())
        else:
            agent.core_move = random.choice(game.valid_moves()) 
            # new_move = omnicron.get_worst_move(game.game_id(), game.get_state().clone())
            # if new_move is not None:
            #     agent.core_move = new_move
            # else:
            #     agent.core_move = random.choice(game.valid_moves()) 
    else:
        # Good‑path: explore winners, exploit losers
        if not agent.change:  # agent won
            agent.core_move = random.choice(game.valid_moves())
        else:  # agent lost
            new_move = omnicron.get_best_move(game.game_id(), game.get_state().clone(), debug_move=False)
            if new_move is not None:
                agent.core_move = new_move
            else:
                print("move is none!")
                agent.core_move = random.choice(game.valid_moves())


def update_agents(
    agent: Agent,
    anti_agent: Agent,
    game: GameBase,
    omnicron: GameMemory,
    is_prune_stage: bool = True,
) -> None:
    """
    Update the core moves of all agents based on the current search stage.

    Parameters
    ----------
    agent : Agent
        The “friendly” agent in the current game state.
    anti_agent : Agent
        The “opponent” agent in the current game state.
    game : GameBase
        The game instance; needed for generating random valid moves.
    is_prune_stage : bool, optional
        ``True`` if we are currently exploring bad paths (pruning).
        Defaults to ``True``.

    This function mutates the agents in place - no value is returned.
    """
    # Treat every agent uniformly: the helper handles the logic
    _set_move(agent, game, omnicron, is_prune_stage)

    # Anti‑agents behave the same way as normal agents
    _set_move(anti_agent, game, omnicron, is_prune_stage)
