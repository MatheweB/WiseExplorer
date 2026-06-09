"""
Move-selection weighting for training.

Training samples moves from an uncertainty-weighted value distribution.
``move_weight`` gives one move's weight; the sampling itself lives in
``select_move_for_training`` in the package root.
"""

from __future__ import annotations

from wise_explorer.core.types import Stats


def move_weight(stats: Stats, is_prune: bool) -> float:
    """Training weight for a single move.

    A move's *exploration drive* is its uncertainty (``Stats.std_error``). We
    spend that drive on whichever side of the value we are still pinning down:

    - exploit phase (``is_prune=False``): ``std_error * mean_score``
      (promising moves we are not yet sure of)
    - prune phase (``is_prune=True``):    ``std_error * (1 - mean_score)``
      (unpromising moves we are not yet sure of)

    The two weights are mirror images and sum to ``std_error``. Because the
    weight scales with ``std_error``, sampling a move shrinks its weight
    (self-correcting); on the unexplored frontier every move shares the maximal
    ``std_error``, so weights tie and sampling spreads ~uniformly.
    """
    drive = stats.std_error
    score = stats.mean_score
    return drive * (1.0 - score) if is_prune else drive * score
