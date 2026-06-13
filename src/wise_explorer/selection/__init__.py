"""
Move selection.

Competitive play ranks each move by the evidence ladder, strongest first:

    proven value  >  concept value  >  direct statistics

A proven value is the game's own verdict for the resulting board (a
certificate); a concept value is the invented theory's price; statistics are
the raw outcome counts. Missing rungs read as the neutral 0.5, so a proven
loss (0.0) ranks below an unknown board and a proven win (1.0) above
everything unproven.

Training play is uncertainty-driven sampling over the counts alone, tilted by
certificates: moves onto proven boards are damped (the game already confirmed
them), moves onto sharp-but-unproven claims are boosted (evidence is wanted
there). The tilt chooses where games go, never which move is good — the
counts stay independent of the theory they train.
"""

from __future__ import annotations

import os
import random
from math import ceil as _math_ceil, log as _math_log
from typing import TYPE_CHECKING

import numpy as np

from wise_explorer.core.types import Stats
from wise_explorer.selection import training

if TYPE_CHECKING:
    from wise_explorer.games.game_base import GameBase
    from wise_explorer.memory.game_memory import GameMemory

# WISE_STEERING=0 reduces training drive to plain statistical uncertainty.
STEERING = os.environ.get("WISE_STEERING", "1") != "0"


def _effective_score(stats: Stats) -> float:
    """Exact win ratio when the evidence is unanimous and significant,
    Bayesian mean otherwise. (Inlined for speed — called once per move.)"""
    w, t, l = stats.wins, stats.ties, stats.losses
    total = w + t + l
    n_types = (w > 0) + (t > 0) + (l > 0)
    if n_types == 1 and total >= max(3, _math_ceil(_math_log(0.05) / _math_log(0.5))):
        return (w * 1.0 + t * 0.5) / total
    return ((w + 1) * 1.0 + (t + 1) * 0.5) / (total + 3)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def select_move(
    game: GameBase,
    memory: GameMemory,
    is_prune: bool = False,
    debug: bool = False,
) -> np.ndarray:
    """Pick the best (or, for prune, worst) move by the evidence ladder."""
    valid_moves = game.valid_moves()
    if len(valid_moves) == 0:
        raise ValueError("No valid moves")

    ev = memory.evaluate_moves(game, valid_moves)
    if not ev.moves:
        return np.asarray(random.choice(valid_moves))

    def key(move, stats):
        mk = tuple(move)
        proven = ev.proven.get(mk)
        concept = ev.concept_scores.get(mk)
        return (proven if proven is not None else 0.5,
                concept if concept is not None else 0.5,
                _effective_score(stats))

    pick = min if is_prune else max
    selected = pick(ev.moves, key=lambda ms: key(*ms))[0]

    if debug:
        from wise_explorer.debug.viz import debug_move_selection
        debug_move_selection(memory, game, valid_moves, selected)
    return np.asarray(selected)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def select_move_for_training(
    game: GameBase,
    memory: GameMemory,
    is_prune: bool,
    debug: bool = False,
) -> np.ndarray:
    """Sample a move from the uncertainty-weighted distribution.

    A move's exploration drive is its total remaining uncertainty: statistical
    noise and theory–evidence disagreement, combined in quadrature; zero once
    the board is proven. The drive is spent on whichever side of the value the
    phase is pinning down (see training.move_weight). Disagreement is
    direction-blind — the theory can pull attention toward boards where it is
    informative and untested, never toward boards it merely favors.
    """
    valid_moves = game.valid_moves()
    ev = memory.evaluate_moves(game, valid_moves)
    if not ev.moves:
        return np.asarray(random.choice(valid_moves))

    def drive(mk, stats):
        se = stats.std_error
        if not STEERING:
            return se
        if mk in ev.proven:
            return 0.0                       # a proof leaves nothing to learn
        c = ev.concept_scores.get(mk)
        if c is None:
            return se
        gap = c - stats.mean_score
        return (se * se + gap * gap) ** 0.5

    moves, weights = [], []
    for move, stats in ev.moves:
        moves.append(move)
        weights.append(training.move_weight(stats, is_prune,
                                            drive=drive(tuple(move), stats)))

    if sum(weights) <= 1e-12:
        # no uncertainty left to spend — spread uniformly
        selected = random.choice(moves)
    else:
        selected = random.choices(moves, weights=weights, k=1)[0]

    if debug:
        from wise_explorer.debug.viz import debug_move_selection
        debug_move_selection(memory, game, valid_moves, selected)
    return np.asarray(selected)


__all__ = [
    "select_move",
    "select_move_for_training",
]
