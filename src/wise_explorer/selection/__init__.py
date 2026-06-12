"""
Selection module - move selection strategies for training and inference.

Provides the main entry points:
- select_move(): For competitive play (deterministic)
- select_move_for_training(): For training (probabilistic)
"""

from __future__ import annotations

import os
import random
from math import ceil as _math_ceil, log as _math_log
from typing import TYPE_CHECKING

import numpy as np

from wise_explorer.core.types import Stats
from wise_explorer.selection import training

# Experimental (docs/certificate-aware-exploration.md): training-time selection
# steers by the theory's verified epistemic state. Mode 1 damps moves that land on
# game-certified boards (memory.certified_hashes). Mode 2 additionally boosts moves
# landing on sharp-but-untested boards — the claims that actively need evidence.
# Multipliers are relative within the candidate set, so positions whose every move
# is certified renormalize to normal play. Competitive selection is untouched.
try:
    CERT_AWARE = int(os.environ.get("WISE_CERT_AWARE", "0") or "0")
except ValueError:
    CERT_AWARE = 0
CERT_DAMP = 0.05
CERT_BOOST = 2.0
CERT_SHARP = 0.3

if TYPE_CHECKING:
    from wise_explorer.games.game_base import GameBase
    from wise_explorer.memory.game_memory import GameMemory


# ---------------------------------------------------------------------------
# Decisive scoring helper
# ---------------------------------------------------------------------------

def _effective_score(stats: Stats, base_rate: float = 0.5) -> float:
    """Return exact ratio if evidence is unanimously significant,
    otherwise Bayesian mean.

    Inlines is_decisive() from core/types.py and Stats.utility/mean_score
    to avoid function call overhead (2.15x faster). Called ~137K times
    per 200 chess games — Python dispatch cost matters at this frequency.
    """
    # --- Inlined from is_decisive() in core/types.py ---
    w, t, l = stats.wins, stats.ties, stats.losses
    total = w + t + l
    n_types = (w > 0) + (t > 0) + (l > 0)
    if n_types == 1 and total > 0:
        p = max(min(base_rate, 0.95), 0.5)
        threshold = max(3, _math_ceil(_math_log(0.05) / _math_log(p)))
        if total >= threshold:
            # --- Inlined from Stats.utility in core/types.py ---
            return (w * 1.0 + t * 0.5) / total
    # --- Inlined from Stats.mean_score / Stats._moments in core/types.py ---
    ww, tt, ll = w + 1, t + 1, l + 1
    n = ww + tt + ll
    return (ww * 1.0 + tt * 0.5) / n


def _fast_var(values: list) -> float:
    """Variance for small lists. 7-13x faster than numpy for <20 elements."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((x - mean) ** 2 for x in values) / n


def _rank_signals(
    bell_scores: dict[tuple, float | None],
    anchor_scores: dict[tuple, float],
    solo_scores: dict[tuple, float],
    concept_scores: dict[tuple, float | None] | None = None,
) -> tuple[str, ...]:
    """Rank signals by variance — most informative first.

    Compares variance of bell, anchor, solo, and concept scores across
    all moves. Returns signal names ordered by descending variance.
    """
    bells = [b for b in bell_scores.values() if b is not None]
    anchors = list(anchor_scores.values())
    solos = list(solo_scores.values())

    variances = {
        "bell": _fast_var(bells),
        "anchor": _fast_var(anchors),
        "solo": _fast_var(solos),
    }

    if concept_scores:
        concepts = [c for c in concept_scores.values() if c is not None]
        variances["concept"] = _fast_var(concepts)

    ranked = sorted(variances, key=variances.get, reverse=True)  # type: ignore
    return tuple(ranked)


def _score_move(
    bell: float | None,
    anchor_score: float,
    solo_score: float,
    signal_order: tuple[str, ...],
    concept_score: float | None = None,
) -> tuple[float, ...]:
    """Score a move as a tuple based on signal ranking.

    signal_order is ranked by variance (most informative first).
    The move is scored using the most informative signal first,
    with remaining signals as tiebreakers.
    """
    b = bell if bell is not None else anchor_score
    c = concept_score if concept_score is not None else anchor_score
    values = {"bell": b, "anchor": anchor_score, "solo": solo_score, "concept": c}
    return tuple(values[s] for s in signal_order)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def select_move(
    game: GameBase,
    memory: GameMemory,
    is_prune: bool = False,
    debug: bool = False,
) -> np.ndarray:
    """
    Select move for inference (competitive play).

    Uses pick-best-signal: compares variance of bell vs mean scores
    across all moves and ranks by whichever signal is more informative.
    The other signal serves as tiebreaker.

    - Nim-like (bell varies): ranks by (bell, anchor, solo)
    - TTT-like (bell flat):  ranks by (anchor, bell, solo)

    Returns:
        Selected move as numpy array
    """
    valid_moves = game.valid_moves()
    if len(valid_moves) == 0:
        raise ValueError("No valid moves")

    pick_best = not is_prune

    evaluation = memory.evaluate_moves(game, valid_moves)
    anchors_with_moves = evaluation.anchors_with_moves
    anchor_stats = evaluation.anchor_stats

    if not anchors_with_moves:
        return np.asarray(random.choice(valid_moves))

    # Bell and concept scores are pre-collected by evaluate_moves (no re-cloning)
    bell_scores = evaluation.bell_scores
    concept_scores = evaluation.concept_scores

    # Collect anchor and solo scores per move for signal comparison
    # Uses decisive scoring: exact ratio when unanimously significant
    anchor_scores: dict[tuple, float] = {}
    solo_scores: dict[tuple, float] = {}
    for aid, moves in anchors_with_moves.items():
        a_stats = anchor_stats.get(aid)
        a_score = _effective_score(a_stats) if a_stats else 0.5
        for move, stats in moves:
            mk = tuple(move)
            anchor_scores[mk] = a_score
            solo_scores[mk] = _effective_score(stats, a_score)

    # Competitive play is *reliability*-first, not *discrimination*-first: rank
    # the game-theoretic Bellman value primary, with the remaining signals
    # (variance-ordered) as tiebreakers. Pure variance arbitration would defer to
    # whichever signal spreads the moves most — but exploration noise inflates the
    # spread of the raw W/T/L signals (solo/anchor), so they'd override a correct
    # Bellman value. This self-adapts: when Bellman discriminates it decides; when
    # it's flat (ties) or missing, the tiebreakers take over. Measured on TTT vs
    # minimax ground truth: 96%→99.6% optimal at convergence, 86%→88% early.
    # (Training keeps pure variance arbitration — it wants discriminative
    # exploration breadth, which is what makes Bellman converge in the first place.)
    signal_order = ("bell",) + tuple(
        s for s in _rank_signals(bell_scores, anchor_scores, solo_scores, concept_scores)
        if s != "bell"
    )

    # Score each move using the ranked signal order
    move_scored: list[tuple[np.ndarray, tuple[float, ...]]] = []
    for aid, moves in anchors_with_moves.items():
        a_stats = anchor_stats.get(aid)
        a_score = _effective_score(a_stats) if a_stats else 0.5
        for move, stats in moves:
            mk = tuple(move)
            bell = bell_scores.get(mk)
            concept = concept_scores.get(mk) if concept_scores else None
            key = _score_move(bell, a_score, _effective_score(stats, a_score), signal_order, concept)
            move_scored.append((move, key))

    if pick_best:
        selected_move = max(move_scored, key=lambda m: m[1])[0]
    else:
        selected_move = min(move_scored, key=lambda m: m[1])[0]

    if debug:
        from wise_explorer.debug.viz import debug_move_selection
        debug_move_selection(memory, game, valid_moves, selected_move)

    return np.asarray(selected_move)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def select_move_for_training(
    game: GameBase,
    memory: GameMemory,
    is_prune: bool,
    debug: bool = False,
) -> np.ndarray:
    """
    Select a move for training by sampling an uncertainty-weighted value
    distribution.

    A move's *exploration drive* is its uncertainty (``Stats.std_error``). That
    drive is spent on whichever side of the value we are still trying to pin
    down: the exploit phase samples each move in proportion to
    ``std_error * score`` (promising moves we are not yet sure of), the prune
    phase in proportion to ``std_error * (1 - score)`` (unpromising moves we are
    not yet sure of). The two weights are mirror images and sum to
    ``std_error``.

    This is symmetric, parameter-free, and self-correcting: sampling a move
    shrinks its ``std_error``, so its weight falls and the search spreads on its
    own. On the unexplored frontier every move shares the same (maximal)
    ``std_error``, so the weights tie and the draw is ~uniform — which is what
    yields broad coverage with no tuning knob. Verified to match argmax-uncertainty
    selection on Tic-Tac-Toe and Nim against minimax ground truth (coverage
    ~0.80 / ~0.92, optimal ~88%).

    Returns:
        Selected move as numpy array
    """
    valid_moves = game.valid_moves()
    evaluation = memory.evaluate_moves(game, valid_moves)
    anchors_with_moves = evaluation.anchors_with_moves

    if not anchors_with_moves:
        return np.asarray(random.choice(valid_moves))

    # Flatten to per-move (move, stats) — every valid move appears exactly once.
    moves: list[np.ndarray] = []
    weights: list[float] = []
    for anchor_moves in anchors_with_moves.values():
        for move, stats in anchor_moves:
            moves.append(move)
            weights.append(training.move_weight(stats, is_prune))

    if CERT_AWARE:
        certs = getattr(memory, "certified_hashes", set())
        if certs or CERT_AWARE >= 2:
            to_hash = {tuple(mv): h for mv, h, _ in
                       memory._compute_move_hashes(game, valid_moves)}

            def _mult(mk):
                if to_hash.get(mk) in certs:
                    return CERT_DAMP                    # proven: skip
                if CERT_AWARE >= 2:
                    L = evaluation.concept_scores.get(mk)
                    if L is not None and abs(L - 0.5) >= CERT_SHARP:
                        return CERT_BOOST               # claimed: needs testing
                return 1.0                              # guessed: as today
            weights = [w * _mult(tuple(m)) for m, w in zip(moves, weights)]

    if sum(weights) <= 1e-12:
        # No uncertainty signal to act on (unexplored frontier or fully
        # resolved) — fall back to uniform so coverage still spreads.
        selected_move = random.choice(moves)
    else:
        selected_move = random.choices(moves, weights=weights, k=1)[0]

    if debug:
        from wise_explorer.debug.viz import debug_move_selection
        debug_move_selection(memory, game, valid_moves, selected_move)

    return np.asarray(selected_move)


__all__ = [
    "select_move",
    "select_move_for_training",
]
