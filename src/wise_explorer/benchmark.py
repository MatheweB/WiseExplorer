"""
Oracle benchmarks: how optimal is the AI's play, measured against a perfect
solver. Available for the games with a known oracle — Nim (the nim-sum) and
Tic-Tac-Toe (minimax). Returns an optimal-move rate over reachable positions.
"""

from __future__ import annotations

import itertools
import random
from functools import lru_cache

import numpy as np

from wise_explorer.games.game_state import GameState
from wise_explorer.games.nim import Nim
from wise_explorer.games.tic_tac_toe import TicTacToe
from wise_explorer.selection import select_move

_SAMPLE_OVER = 4000          # exhaustive below this many positions, else sampled
_SAMPLE_N = 400


# ---------------------------------------------------------------------------
# Nim — the nim-sum oracle
# ---------------------------------------------------------------------------

def _nim_sum(board) -> int:
    return int(np.bitwise_xor.reduce(np.asarray(board).astype(np.int64)))


def nim_optimal_rate(memory, piles: int) -> tuple[int, int]:
    """Over winning positions (nim-sum ≠ 0), the rate at which the AI moves to a
    nim-sum-0 position — the unique optimal class. Exhaustive when small."""
    space = 1
    for i in range(piles):
        space *= i + 2
    opt = total = 0

    def check(b):
        g = Nim(n=piles)
        g.set_state(GameState(b.copy(), current_player=1))
        mv = select_move(g, memory)
        nb = b.copy()
        nb[int(mv[0])] -= int(mv[1])
        return _nim_sum(nb) == 0

    if space <= _SAMPLE_OVER:
        for tup in itertools.product(*(range(i + 2) for i in range(piles))):
            b = np.array(tup, dtype=np.int8)
            if b.sum() == 0 or _nim_sum(b) == 0:
                continue                                # losing/terminal: no winning move
            total += 1
            opt += check(b)
    else:
        rng = random.Random(7)
        while total < _SAMPLE_N:
            b = np.array([rng.randint(0, i + 1) for i in range(piles)], dtype=np.int8)
            if b.sum() == 0 or _nim_sum(b) == 0:
                continue
            total += 1
            opt += check(b)
    return opt, total


# ---------------------------------------------------------------------------
# Tic-Tac-Toe — the minimax oracle
# ---------------------------------------------------------------------------

_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
          (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def _ttt_winner(flat) -> int:
    for a, b, c in _LINES:
        if flat[a] and flat[a] == flat[b] == flat[c]:
            return flat[a]
    return 0


@lru_cache(maxsize=None)
def _minimax(flat: tuple, player: int) -> int:
    """Value for the player to move: +1 win, 0 draw, -1 loss."""
    if _ttt_winner(flat):
        return -1                                       # the previous mover just won
    if 0 not in flat:
        return 0
    best = -2
    for i in range(9):
        if flat[i] == 0:
            best = max(best, -_minimax(flat[:i] + (player,) + flat[i + 1:], 3 - player))
            if best == 1:
                break
    return best


def _optimal_cells(flat: tuple, player: int) -> set[int]:
    vals = {i: -_minimax(flat[:i] + (player,) + flat[i + 1:], 3 - player)
            for i in range(9) if flat[i] == 0}
    best = max(vals.values())
    return {i for i, v in vals.items() if v == best}


def ttt_optimal_rate(memory) -> tuple[int, int]:
    """Over every reachable non-terminal position, the rate at which the AI's
    move achieves the minimax-optimal value."""
    seen: set[tuple] = set()
    stack = [((0,) * 9, 1)]
    positions: list[tuple[tuple, int]] = []
    while stack:
        flat, player = stack.pop()
        if (flat, player) in seen:
            continue
        seen.add((flat, player))
        if _ttt_winner(flat) or 0 not in flat:
            continue
        positions.append((flat, player))
        for i in range(9):
            if flat[i] == 0:
                stack.append((flat[:i] + (player,) + flat[i + 1:], 3 - player))

    opt = 0
    for flat, player in positions:
        g = TicTacToe()
        g.set_state(GameState(np.array(flat, dtype=np.int8).reshape(3, 3),
                              current_player=player))
        mv = select_move(g, memory)
        if int(mv[0]) * 3 + int(mv[1]) in _optimal_cells(flat, player):
            opt += 1
    return opt, len(positions)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def optimal_rate(memory, game) -> tuple[int, int, str] | None:
    """(optimal, total, description) against the game's oracle, or None if the
    game has no oracle here."""
    gid = game.game_id()
    if gid.startswith("nim"):
        piles = int(game.get_state().board.size)
        o, t = nim_optimal_rate(memory, piles)
        return o, t, "winning positions played to a nim-sum-0 reply"
    if gid == "tic_tac_toe":
        o, t = ttt_optimal_rate(memory)
        return o, t, "reachable positions played to a minimax-optimal move"
    return None
