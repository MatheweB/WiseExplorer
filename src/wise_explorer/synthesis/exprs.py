"""The feature-program algebra — the evaluable AST a concept is built from, the
operator tables, and (de)serialisation. Depends only on numpy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ───────────────────────────── feature programs (a tiny evaluable AST) ─────────
# Each program reads a board (a 1-D int vector of cells) and returns an int. They
# stay symbolic, so a discovered concept both *runs on unseen boards* and *renders*
# to a readable formula.

_OPS = {
    "⊕": np.bitwise_xor, "&": np.bitwise_and, "|": np.bitwise_or,
    "+": np.add, "∣·∣": lambda a, b: np.abs(a - b),
    "max": np.maximum, "min": np.minimum,
}

# How to say an op out loud — symbols a reader won't recognise become words.
_WORD = {"⊕": "xor", "&": "and", "|": "or", "∣·∣": "dist"}


class Expr:
    # every program reads the board ``B`` and, optionally, ``m`` = the token the last
    # move placed (only group-folds use ``m``; everything else ignores it).
    size: int = 1
    def eval(self, B: np.ndarray, m=None) -> np.ndarray:  # B: (N, cells) → (N,)
        raise NotImplementedError
    def __str__(self) -> str:
        raise NotImplementedError


class Cell(Expr):
    def __init__(self, i: int):
        self.i = i
        self.size = 1
    def eval(self, B, m=None): return B[:, self.i]
    def __str__(self): return f"c{self.i}"


class Lit(Expr):
    def __init__(self, v: int):
        self.v = v
        self.size = 1
    def eval(self, B, m=None): return np.full(B.shape[0], self.v, dtype=np.int64)
    def __str__(self): return str(self.v)


class BinOp(Expr):
    def __init__(self, op: str, a: Expr, b: Expr):
        self.op, self.a, self.b = op, a, b
        self.size = 1 + a.size + b.size
    def eval(self, B, m=None):
        return _OPS[self.op](self.a.eval(B, m), self.b.eval(B, m)).astype(np.int64)
    def __str__(self): return f"({self.a} {_WORD.get(self.op, self.op)} {self.b})"


class Named(Expr):
    """A promoted concept reused as a single building block (its formula is kept
    for rendering, but it costs size 1 to compose with — that is what 'reuse' buys)."""
    def __init__(self, inner: Expr, vec: np.ndarray):
        self.inner, self._vec = inner, vec
        self.size = 1
    def eval(self, B, m=None):
        if B.shape[0] == len(self._vec):
            return self._vec
        return self.inner.eval(B, m)            # re-derive for unseen boards
    def __str__(self): return str(self.inner)


# The monoids you can fold with — this is the whole justification for the op set:
# each is an associative reduction with an identity element.
_FOLD = {
    "⊕": (np.bitwise_xor, 0), "+": (np.add, 0), "|": (np.bitwise_or, 0),
    "&": (np.bitwise_and, -1),                          # -1 = all-ones, the int64 & identity
    "max": (np.maximum, np.iinfo(np.int64).min), "min": (np.minimum, np.iinfo(np.int64).max),
}


class Elem(Expr):
    """The item a fold is looking at right now — specifically its feature number ``j``. A
    cell offers one feature (its value, ``cell``); a line offers two (``played`` / ``empty``
    counts). Only ever evaluated inside a Fold, which feeds it the items one at a time as a
    flat (rows, width) table — never a board. ``names[j]`` is for rendering only."""
    def __init__(self, j: int, names: tuple[str, ...]):
        self.j, self.names, self.size = j, names, 1
    def eval(self, E, m=None): return E[:, self.j]
    def __str__(self): return self.names[self.j]


class CellDomain:
    """A fold domain whose elements are cells; each element exposes one feature —
    its token value. (The move ``m`` is irrelevant here — cell arithmetic ignores it.)"""
    names = ("cell",)
    def __init__(self, cells: tuple[int, ...]): self.cells = tuple(cells)
    def tensor(self, B, m=None): return B[:, list(self.cells)][:, :, None].astype(np.int64)
    def __str__(self): return "cells"


class BoardDomain:
    """A fold domain over the WHOLE board, width-free: its elements are *all* the cells of
    whatever board it is handed, not a frozen list. So the nim-sum ``fold(⊕, board, cell)``
    discovered at 4 piles is the *identical program* at 8 piles — it reads every heap, not the
    first four — and it serialises with no width, which is what lets a concept transfer across
    scales unchanged. (Compare ``CellDomain``, whose cell list is fixed at discovery time.)"""
    names = ("cell",)
    def tensor(self, B, m=None): return B[:, :, None].astype(np.int64)
    def __str__(self): return "board"


class GroupDomain:
    """A fold domain whose elements are groups of cells the search already discovered (the
    cell-supports of kept concepts — e.g. the lines round 1 found). Each group shows two
    counts, read at face value against the move: how many cells hold the *played* token
    (``cell == m``) and how many are *empty*. The board is never recoded — a piece that is
    neither the played token nor empty keeps its own value and just isn't counted — so
    piece types are never flattened. (``played`` + ``empty`` pin down a fixed-length line.)"""
    names = ("played", "empty")
    def __init__(self, groups): self.groups = tuple(tuple(g) for g in groups)
    def tensor(self, B, m):
        T = np.empty((B.shape[0], len(self.groups), 2), dtype=np.int64)
        for gi, g in enumerate(self.groups):
            cols = B[:, list(g)]
            T[:, gi, 0] = (cols == m[:, None]).sum(1)        # played: cells holding the moved token
            T[:, gi, 1] = (cols == 0).sum(1)                 # empty
        return T
    def __str__(self): return "groups"


class Fold(Expr):
    """``fold(op, domain, body)`` — reduce ``body`` over every element of ``domain``
    with the monoid ``op``. The one combinator everything is built from: the nim-sum is
    a fold over cells, a threat is a fold over groups. ``m`` (the just-moved token) is
    used only by group-folds. It recomputes from the board, so it runs on unseen boards
    and renders to a readable formula."""
    def __init__(self, op: str, domain, body: Expr):
        self.op, self.domain, self.body = op, domain, body
        self.size = 1 + body.size
    def eval(self, B, m=None):
        T = self.domain.tensor(B, m)                         # (N, n_items, n_features)
        N, K, W = T.shape
        per = self.body.eval(T.reshape(N * K, W)).reshape(N, K)
        fn, ident = _FOLD[self.op]
        return fn.reduce(per, axis=1, initial=ident).astype(np.int64)
    def __str__(self): return f"fold({self.op}, {self.domain}, {self.body})"


# ───────────────────────────── a discovered concept (a boolean feature) ────────

@dataclass
class Concept:
    expr: Expr            # the integer feature program
    op: str               # "=" or ">"
    const: int            # threshold
    mask: np.ndarray      # boolean: where it holds, over the training boards
    size: int             # description size (symbols)

    def __str__(self) -> str:
        return f"{self.expr} {self.op} {self.const}"

    def holds(self, board: np.ndarray, m=None) -> bool:
        mv = None if m is None else np.asarray([m])
        v = int(self.expr.eval(np.asarray(board).reshape(1, -1), mv)[0])
        return v == self.const if self.op == "=" else v > self.const


# ───────────────────────────── MDL helpers ─────────────────────────────────────

# How a node's values are priced, in bits. No buckets and no thresholds: each value
# contributes *fractional* mass to the two outcome anchors it sits between — {0, ½, 1},
# the game's own utility scale — and the node costs the Shannon entropy of those masses.
# (Measured 2026-06 against hard LOSS/DRAW/WIN cuts at 0.40/0.60: same nim-sum, same
# zero-shot transfer, slightly better TTT play, half the duplicate threat spellings, less
# junk at scale, faster — and nothing hand-placed. The cuts were deleted.
# Also measured and rejected: weighting each board's say by its evidence — it bloated the
# converged library and worsened play; junk fits noise in the values themselves, which no
# row-weighting can repair. One row, one vote.)


def expr_to_dict(e: Expr) -> dict:
    """Serialise a feature program to a plain dict (for SQLite). The program is what's
    board-order-independent and reusable; masks/value-vectors are never stored."""
    if isinstance(e, Cell):
        return {"t": "Cell", "i": e.i}
    if isinstance(e, Lit):
        return {"t": "Lit", "v": e.v}
    if isinstance(e, BinOp):
        return {"t": "BinOp", "op": e.op, "a": expr_to_dict(e.a), "b": expr_to_dict(e.b)}
    if isinstance(e, Named):
        return {"t": "Named", "inner": expr_to_dict(e.inner)}
    if isinstance(e, Elem):
        return {"t": "Elem", "j": e.j, "names": list(e.names)}
    if isinstance(e, Fold):
        if isinstance(e.domain, BoardDomain):
            dom = {"t": "BoardDomain"}                          # width-free → the program transfers
        elif isinstance(e.domain, CellDomain):
            dom = {"t": "CellDomain", "cells": list(e.domain.cells)}
        else:
            dom = {"t": "GroupDomain", "groups": [list(g) for g in e.domain.groups]}
        return {"t": "Fold", "op": e.op, "domain": dom, "body": expr_to_dict(e.body)}
    raise TypeError(f"cannot serialise {type(e).__name__}")


def expr_from_dict(d: dict) -> Expr:
    """Rebuild a feature program from :func:`expr_to_dict`. A reloaded program runs on any
    board (via ``eval``), which is how a discovered concept reaches the worker processes."""
    t = d["t"]
    if t == "Cell":
        return Cell(d["i"])
    if t == "Lit":
        return Lit(d["v"])
    if t == "BinOp":
        return BinOp(d["op"], expr_from_dict(d["a"]), expr_from_dict(d["b"]))
    if t == "Named":
        return Named(expr_from_dict(d["inner"]), np.empty(0, dtype=np.int64))
    if t == "Elem":
        return Elem(d["j"], tuple(d["names"]))
    if t == "Fold":
        dd = d["domain"]
        if dd["t"] == "BoardDomain":
            dom = BoardDomain()
        elif dd["t"] == "CellDomain":
            dom = CellDomain(tuple(dd["cells"]))
        else:
            dom = GroupDomain([tuple(g) for g in dd["groups"]])
        return Fold(d["op"], dom, expr_from_dict(d["body"]))
    raise ValueError(f"unknown expr tag {t!r}")


