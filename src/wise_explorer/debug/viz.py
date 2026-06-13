"""Debug display for move selection."""

from __future__ import annotations

import numpy as np


def debug_move_selection(memory, game, valid_moves, selected) -> None:
    """Print each candidate move's evidence ladder: proven, concept, stats."""
    ev = memory.evaluate_moves(game, valid_moves)
    sel = tuple(np.asarray(selected).tolist())
    print(f"{'move':<16} {'proven':>7} {'concept':>8} {'W/T/L':>12} {'score':>6}")
    for move, stats in ev.moves:
        mk = tuple(move)
        proven = ev.proven.get(mk)
        concept = ev.concept_scores.get(mk)
        p = "—" if proven is None else f"{proven:.2f}"
        c = "—" if concept is None else f"{concept:.2f}"
        wtl = f"{stats.wins:.0f}/{stats.ties:.0f}/{stats.losses:.0f}"
        mark = " ←" if tuple(np.asarray(move).tolist()) == sel else ""
        print(f"{str(mk):<16} {p:>7} {c:>8} {wtl:>12} {stats.mean_score:>6.2f}{mark}")
