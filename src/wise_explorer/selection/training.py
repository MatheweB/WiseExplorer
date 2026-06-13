"""
Move-selection weighting for training.

Training samples moves from an uncertainty-weighted value distribution.
``move_weight`` gives one move's weight; the sampling itself lives in
``select_move_for_training`` in the package root.
"""

from __future__ import annotations

from wise_explorer.core.types import Stats


def move_weight(stats: Stats, is_prune: bool, drive: float | None = None) -> float:
    """Training weight for a single move.

    A move's *exploration drive* defaults to its uncertainty
    (``Stats.std_error``); callers may pass a wider estimate (e.g. including
    theory–evidence disagreement). The drive is spent on whichever side of the
    value the phase is pinning down:

    - exploit phase (``is_prune=False``): ``drive * mean_score``
      (promising moves we are not yet sure of)
    - prune phase (``is_prune=True``):    ``drive * (1 - mean_score)``
      (unpromising moves we are not yet sure of)

    The two weights are mirror images and sum to the drive. Sampling a move
    shrinks its uncertainty and hence its weight (self-correcting); on the
    unexplored frontier all moves tie and sampling spreads ~uniformly.
    """
    if drive is None:
        drive = stats.std_error
    score = stats.mean_score
    return drive * (1.0 - score) if is_prune else drive * score
