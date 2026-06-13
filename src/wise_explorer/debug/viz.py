"""Debug displays for move selection."""

from __future__ import annotations

import numpy as np


def _key(move) -> tuple:
    return tuple(np.asarray(move).tolist())


def debug_move_selection(memory, game, valid_moves, selected) -> None:
    """Full per-candidate dump: every move's proven / concept / stats values."""
    ev = memory.evaluate_moves(game, valid_moves)
    sel = _key(selected)
    print(f"{'move':<16} {'proven':>7} {'concept':>8} {'W/T/L':>12} {'score':>6}")
    for move, stats in ev.moves:
        mk = tuple(move)
        proven = ev.proven.get(mk)
        concept = ev.concept_scores.get(mk)
        p = "—" if proven is None else f"{proven:.2f}"
        c = "—" if concept is None else f"{concept:.2f}"
        wtl = f"{stats.wins:.0f}/{stats.ties:.0f}/{stats.losses:.0f}"
        mark = " ←" if _key(move) == sel else ""
        print(f"{str(mk):<16} {p:>7} {c:>8} {wtl:>12} {stats.mean_score:>6.2f}{mark}")


def explain_move(memory, game, chosen) -> None:
    """Why the AI chose this move, by the evidence ladder.

    Names the rung that decided — a game-proven certificate, the invented
    concept's value, or raw statistics — and lists the runners-up.
    """
    ev = memory.evaluate_moves(game, list(game.valid_moves()))
    sel = _key(chosen)

    def rung(mk):
        if mk in ev.proven:
            v = ev.proven[mk]
            verdict = "WIN" if v > 0.6 else "LOSS" if v < 0.4 else "DRAW"
            return f"proven {verdict} (certificate, value {v:.2f})"
        c = ev.concept_scores.get(mk)
        if c is not None and abs(c - 0.5) >= 0.1:
            return f"concept value {c:.2f}"
        return "statistics"

    chosen_stats = next((s for m, s in ev.moves if _key(m) == sel), None)
    print(f"  decided by: {rung(sel)}", end="")
    if chosen_stats is not None and chosen_stats.total:
        print(f"  ·  seen {chosen_stats.total:.0f}× "
              f"({chosen_stats.mean_score:.2f})", end="")
    print()

    def rank(item):
        m, s = item
        mk = tuple(m)
        return (ev.proven.get(mk, 0.5),
                ev.concept_scores.get(mk) if ev.concept_scores.get(mk) is not None else 0.5,
                s.mean_score)

    others = sorted((it for it in ev.moves if _key(it[0]) != sel),
                    key=rank, reverse=True)[:3]
    for m, s in others:
        mk = tuple(m)
        p = ev.proven.get(mk)
        c = ev.concept_scores.get(mk)
        bits = []
        if p is not None:
            bits.append(f"proven {p:.2f}")
        if c is not None:
            bits.append(f"concept {c:.2f}")
        bits.append(f"stats {s.mean_score:.2f}")
        print(f"    alt {str(list(np.asarray(m).tolist())):<14} {' · '.join(bits)}")
