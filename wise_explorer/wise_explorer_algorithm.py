"""
Utilities for exploring move space in a two-player game.

Strategy is determined by global phase (symmetric play):
  - Prune phase: ALL agents bias toward worst moves (worst vs worst)
  - Exploit phase: ALL agents bias toward best moves (best vs best)

Two-stage selection:
  1. Try to pick best/worst with probability = score (or 1-score for prune)
     - Decisive moves (score=1.0 or 0.0) are ALWAYS picked
     - Uncertain moves have proportional chance to fall through
  2. Fallback: Weighted sample over ALL moves by DIRECT score
     - Uses direct stats (not anchor-pooled) for honest local exploration
     - Unexplored moves get min(known scores) as prior:
       * Exploit: low weight → conservative, stick with proven moves
       * Prune: high weight (1-min) → aggressive, unknowns might be bad!
     - Linear weights match Stage 1 semantics
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

from agent.agent import Agent
from games.game_base import GameBase
from omnicron.manager import GameMemory


def _weighted_sample(
    moves_with_scores: List[Tuple[np.ndarray, float]],
    is_prune: bool,
) -> np.ndarray:
    """
    Sample a move weighted by score.
    
    Exploit: weight = score → favors high scores
    Prune: weight = 1-score → favors low scores
    
    Linear weights match Stage 1's probability semantics:
    score 0.8 is 4x more likely than score 0.2 (0.8/0.2 = 4)
    """
    if not moves_with_scores:
        raise ValueError("No moves to sample from")
    
    moves = [m for m, s in moves_with_scores]
    scores = [s for m, s in moves_with_scores]
    
    if is_prune:
        weights = [1.0 - s for s in scores]
    else:
        weights = scores
    
    return random.choices(moves, weights=weights, k=1)[0]


def _set_move(
    agent: Agent, game: GameBase, memory: GameMemory, is_prune: bool
) -> None:
    """
    Update ``agent.core_move`` based on global phase.

    Two-stage selection:
      1. Try best/worst with probability based on score
         - Score 1.0 (exploit) or 0.0 (prune) → always picked (decisive)
         - Lower confidence → proportional chance to fall through
      2. Fallback: weighted sample over ALL moves by direct score
         - Uses direct stats for honest local exploration
         - Unexplored get min(known) prior: conservative exploit, aggressive prune
    
    Parameters
    ----------
    agent:
        The agent whose move is being updated.
    is_prune:
        * ``True`` - Prune phase: bias toward worst moves
        * ``False`` - Exploit phase: bias toward best moves
    """
    valid_moves = game.valid_moves()
    
    # Stage 1: Try to pick best/worst with probability = score
    if is_prune:
        result = memory.get_worst_move_with_score(game, valid_moves, deterministic=False)
        if result is not None:
            move, score = result
            # Want LOW scores in prune mode
            # Score 0.0 → always pick (1.0 - 0.0 = 1.0)
            # Score 0.8 → 20% chance to pick
            pick_prob = 1.0 - score
            if random.random() < pick_prob:
                agent.core_move = move
                return
    else:
        result = memory.get_best_move_with_score(game, valid_moves, deterministic=False)
        if result is not None:
            move, score = result
            # Want HIGH scores in exploit mode
            # Score 1.0 → always pick
            # Score 0.3 → 30% chance to pick
            pick_prob = score
            if random.random() < pick_prob:
                agent.core_move = move
                return

    # Stage 2: Smart fallback - weighted sample by DIRECT scores
    # Uses direct stats (not anchor-pooled) for honest local exploration
    moves_with_scores = memory.get_all_moves_with_scores(game, valid_moves)
    
    if moves_with_scores:
        agent.core_move = _weighted_sample(moves_with_scores, is_prune)
    else:
        # Ultimate fallback (shouldn't happen if valid_moves is non-empty)
        agent.core_move = random.choice(valid_moves)


def update_agent(
    agent: Agent,
    game: GameBase,
    memory: GameMemory,
    is_prune: bool,
) -> None:
    """
    Update the agent's move based on global phase.

    Parameters
    ----------
    agent:
        The agent to update.
    game:
        Current game state.
    memory:
        Game memory database.
    is_prune:
        True = prune phase (all play worst), False = exploit phase (all play best).

    This function mutates the agent in place.
    """
    _set_move(agent, game, memory, is_prune)
