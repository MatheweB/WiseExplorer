"""Concept invention via MDL-guided program synthesis, with reuse.

The discovery engine: it *searches for board features to build* out of generic
primitives (cell reads, arithmetic / bitwise ops, one fold combinator), scores
them by how much they compress the win/loss data (MDL), reuses its own
discoveries to reach richer concepts, and stops a round once it no longer pays
for itself. It runs live during training (:func:`grow`, a wave at a time) and
once more over the converged values at the end (:func:`invent_from_boards`).

Public entry point: ``invent(memory, game_id)`` → :class:`InventionResult`.
See ``docs/concept-invention.md`` and the ``invent`` CLI verb.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    def __init__(self, i: int): self.i = i; self.size = 1
    def eval(self, B, m=None): return B[:, self.i]
    def __str__(self): return f"c{self.i}"


class Lit(Expr):
    def __init__(self, v: int): self.v = v; self.size = 1
    def eval(self, B, m=None): return np.full(B.shape[0], self.v, dtype=np.int64)
    def __str__(self): return str(self.v)


class BinOp(Expr):
    def __init__(self, op: str, a: Expr, b: Expr):
        self.op, self.a, self.b = op, a, b; self.size = 1 + a.size + b.size
    def eval(self, B, m=None):
        return _OPS[self.op](self.a.eval(B, m), self.b.eval(B, m)).astype(np.int64)
    def __str__(self): return f"({self.a} {_WORD.get(self.op, self.op)} {self.b})"


class Named(Expr):
    """A promoted concept reused as a single building block (its formula is kept
    for rendering, but it costs size 1 to compose with — that is what 'reuse' buys)."""
    def __init__(self, inner: Expr, vec: np.ndarray):
        self.inner, self._vec = inner, vec; self.size = 1
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
    def __init__(self, j: int, names: Tuple[str, ...]):
        self.j, self.names, self.size = j, names, 1
    def eval(self, E, m=None): return E[:, self.j]
    def __str__(self): return self.names[self.j]


class CellDomain:
    """A fold domain whose elements are cells; each element exposes one feature —
    its token value. (The move ``m`` is irrelevant here — cell arithmetic ignores it.)"""
    names = ("cell",)
    def __init__(self, cells: Tuple[int, ...]): self.cells = tuple(cells)
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

def _soft_counts(v: np.ndarray) -> np.ndarray:
    """Each value's fractional LOSS/DRAW/WIN mass — linear interpolation between the two
    outcome anchors it sits between (V=0.59 ⇒ 0.82 draw + 0.18 win)."""
    ml = np.maximum(0.0, 1.0 - 2.0 * v)                  # 1 at V=0, 0 from V=½ up
    mw = np.maximum(0.0, 2.0 * v - 1.0)                  # 0 up to V=½, 1 at V=1
    return np.array([ml.sum(), len(v) - ml.sum() - mw.sum(), mw.sum()])


def _bits(v: np.ndarray) -> float:
    """Bits to code a node's values as outcome mixtures: n · H(soft LOSS/DRAW/WIN masses).
    Zero iff the node is pure (all its mass on one anchor); continuous in between."""
    n = len(v)
    if n == 0:
        return 0.0
    p = _soft_counts(v) / n
    return float(-sum(pk * math.log2(pk) for pk in p if pk > 0)) * n


# ───────────────────────────── bottom-up synthesis (obs-equivalence) ───────────

def _synthesize(terminals: List[Expr], B: np.ndarray, max_size: int, cap: Optional[int]):
    """Return {value_bytes: Expr} — one smallest program per distinct behaviour."""
    seen: Dict[bytes, Expr] = {}
    by_size: Dict[int, List[Tuple[Expr, np.ndarray]]] = {s: [] for s in range(1, max_size + 1)}

    def add(e: Expr):
        vec = e.eval(B).astype(np.int64)
        key = vec.tobytes()
        if key not in seen:
            seen[key] = e
            by_size[e.size].append((e, vec))

    for t in terminals:
        add(t)
    for s in range(2, max_size + 1):
        for op in _OPS:
            for i in range(1, s - 1):
                j = s - 1 - i
                for ea, va in by_size[i]:
                    for eb, vb in by_size[j]:
                        if cap and len(by_size[s]) >= cap:
                            break
                        vec = _OPS[op](va, vb).astype(np.int64)
                        key = vec.tobytes()
                        if key not in seen:
                            e = BinOp(op, ea, eb)
                            seen[key] = e
                            by_size[s].append((e, vec))
    return seen


# ───────────────────────────── concept selection (MDL, per round) ──────────────

def _candidate_concepts(seen, B, V, min_leaf) -> List[Concept]:
    """Derive boolean atoms (feature == c) and keep the best per yes/no partition.

    The variance-reduction gain of every (program, value) split is computed from *grouped
    sums* in one vectorised pass per program — far cheaper than calling numpy ``.var()``
    once per split, which is almost all per-call overhead on these small arrays."""
    N = len(V)
    V2 = V * V
    S = float(V.sum()); SS = float(V2.sum())
    total_var = SS / N - (S / N) ** 2
    best: Dict[bytes, Tuple[float, Concept]] = {}
    # The errstate is hoisted to wrap the whole loop: empty value-groups produce 0/0
    # (clamped below), and re-entering the context once per candidate is pure overhead.
    with np.errstate(invalid="ignore", divide="ignore"):
        for vec_bytes, expr in seen.items():
            vec = np.frombuffer(vec_bytes, dtype=np.int64)
            # Group V by the program's output value with a single bincount per stat — no
            # per-vector sort. Shift to a non-negative offset so the value IS the bin index;
            # empty bins (gaps in the value range) fall out via the min_leaf filter below.
            lo = int(vec.min()); span = int(vec.max()) - lo + 1
            # bincount is O(N + span) and allocates span-length arrays, so direct value-indexing
            # stays linear while span is within a small multiple of the data (the 4 is slack,
            # the +64 keeps tiny N dense); a wider range pays np.unique's sort instead.
            if span <= 4 * N + 64:                                  # dense: index by value directly
                idx = vec - lo
                n1 = np.bincount(idx, minlength=span).astype(np.float64)   # size of each value-group
                s1 = np.bincount(idx, weights=V, minlength=span)           # its Σ V
                ss1 = np.bincount(idx, weights=V2, minlength=span)         # its Σ V²
                base = lo
            else:                                                  # pathologically wide range: fall back
                vals, inv = np.unique(vec, return_inverse=True); K = len(vals)
                n1 = np.bincount(inv, minlength=K).astype(np.float64)
                s1 = np.bincount(inv, weights=V, minlength=K)
                ss1 = np.bincount(inv, weights=V2, minlength=K)
                base = vals                                        # value of bin k is vals[k]
            n0 = N - n1; s0 = S - s1; ss0 = SS - ss1                # the complement (value != c)
            var1 = np.maximum(ss1 / n1 - (s1 / n1) ** 2, 0.0)       # clamp float roundoff at 0
            var0 = np.maximum(ss0 / n0 - (s0 / n0) ** 2, 0.0)
            gain = total_var - (n1 * var1 + n0 * var0) / N
            keep = (n1 >= min_leaf) & (n0 >= min_leaf) & (gain > 0)
            for k in np.nonzero(keep)[0]:
                c = (base + int(k)) if np.isscalar(base) else int(base[k])
                mask = vec == c; sig = mask.tobytes()
                if sig not in best or expr.size < best[sig][1].size:
                    best[sig] = (float(gain[k]), Concept(expr, "=", c, mask, expr.size))
    # round the gain in the tiebreak so equivalent splits prefer the SIMPLER concept
    # deterministically (float roundoff in the grouped sums must not flip the order)
    return [c for _, c in sorted(best.values(), key=lambda x: (-round(x[0], 9), x[1].size))]


# ───────────────────────────── structural reuse (counting + threats) ───────────

def _cell_group(e: Expr) -> Tuple[int, ...]:
    """The distinct cells a concept's program reads (its support), seen through
    Named (reuse) and Fold (the fold's domain)."""
    out: set = set()
    def walk(x: Expr):
        if isinstance(x, Cell):
            out.add(x.i)
        elif isinstance(x, BinOp):
            walk(x.a); walk(x.b)
        elif isinstance(x, Named):
            walk(x.inner)
        elif isinstance(x, Fold):
            if isinstance(x.domain, CellDomain):
                out.update(x.domain.cells)
            elif isinstance(x.domain, GroupDomain):
                out.update(i for g in x.domain.groups for i in g)
            # BoardDomain folds the whole board — no fixed support, so they contribute
            # nothing here (and are correctly excluded from _supports / the group layer).
    walk(e)
    return tuple(sorted(out))


def _is_atomic(e: Expr) -> bool:
    """True iff the program reads cells directly — pure Cell/Lit arithmetic, no reused
    concept (Named) and no fold. That marks a region the search found *as one thing* (a
    line, a box, whatever) rather than a union glued from earlier concepts or an aggregate.
    No size or shape rule — any coherent cell region qualifies."""
    if isinstance(e, BinOp):
        return _is_atomic(e.a) and _is_atomic(e.b)
    return isinstance(e, (Cell, Lit))


def _supports(kept: List[Concept]) -> List[Tuple[int, ...]]:
    """The groups to fold over next are the cell-supports of the *atomic* concepts the
    search has kept — the regions it found directly from cells. Unions glued from earlier
    concepts and whole-board folds are skipped, so every fold counts over one coherent
    region; combining regions is the rule tree's job, not a single muddy fold's."""
    out, seen = [], set()
    for c in kept:
        if not _is_atomic(c.expr):
            continue
        g = _cell_group(c.expr)
        if len(g) >= 2 and g not in seen:
            seen.add(g); out.append(g)
    return out


def _board_fold_terminals() -> List[Expr]:
    """Seed each round-1 search with the whole-board fold under every monoid — e.g.
    fold(⊕, board, cell) is the nim-sum. The domain is width-free (:class:`BoardDomain`), so a
    fold kept here is the same program at any board width and transfers across scales unchanged.
    Observational-equivalence keeps whichever of these (or a cheaper composition) actually
    predicts the value; the rest are dropped."""
    board = BoardDomain()
    return [Fold(op, board, Elem(0, BoardDomain.names)) for op in _FOLD]


def _residual(rules: List["Rule"], V: np.ndarray) -> np.ndarray:
    """V minus what the current rule set already predicts (each board gets its leaf's
    mean). Scoring new candidates against this — rather than against V — rewards what
    the model still *fails* to explain, so a novel concept (the threat) beats one that
    merely re-states what the kept concepts (the win-lines) already capture."""
    pred = np.zeros_like(V)
    for r in rules:
        leaf = np.ones(len(V), dtype=bool)
        for con, sense in r.path:
            leaf &= con.mask if sense else ~con.mask
        pred[leaf] = r.avg
    return V - pred


def _group_fold_candidates(supports, B: np.ndarray, target: np.ndarray, min_leaf: int, m) -> List[Concept]:
    """Discover threats and forks over the discovered groups (``supports`` — the cells the
    kept concepts read). Search per-group bodies over the (played, empty) counts with the
    same synthesiser, then fold each body over the groups two ways: ``max`` (∃ a group
    where the body fires — a threat) and ``+`` (count the groups where it fires — a fork).
    One discovered body, two monoids. Each fold is thresholded and scored against the
    residual, so a novel concept beats one that merely restates the win-lines."""
    if not supports:
        return []
    dom = GroupDomain(supports)
    T = dom.tensor(B, m); N, K, F = T.shape
    flat = T.reshape(N * K, F)
    bodies = _synthesize([Elem(j, dom.names) for j in range(F)], flat, max_size=5, cap=None)
    scored = []
    for body in bodies.values():
        per = body.eval(flat).reshape(N, K)
        for op in ("max", "+"):                          # ∃ a group | count the groups
            fn, ident = _FOLD[op]
            vec = fn.reduce(per, axis=1, initial=ident).astype(np.int64)
            fold = Fold(op, dom, body)
            for c in np.unique(vec):                      # outer threshold also discovered
                mask = vec == c; n1 = int(mask.sum())
                if n1 < min_leaf or N - n1 < min_leaf:
                    continue
                gain = target.var() - (n1/N*target[mask].var() + (N-n1)/N*target[~mask].var())
                if gain > 0:
                    scored.append((gain, fold.size, Concept(fold, "=", int(c), mask, fold.size)))
    scored.sort(key=lambda x: (-x[0], x[1]))              # best residual gain, then smallest
    return [c for _, _, c in scored[:6]]                  # a residual-ranked handful


# ───────────────────────────── rules over invented concepts ────────────────────

@dataclass
class Rule:
    path: List[Tuple[Concept, bool]]
    verdict: str
    n: int
    avg: float
    resid: float
    def render(self) -> str:
        if not self.path:
            return "(everything)"
        return " AND ".join((str(c) if t else f"¬[{c}]") for c, t in self.path)


def _verdict(v: np.ndarray) -> str:
    return ["LOSS", "DRAW", "WIN"][int(_soft_counts(v).argmax())]   # heaviest outcome mass


def _build_rules(concepts: List[Concept], V: np.ndarray, min_leaf: int, max_depth: int):
    rules: List[Rule] = []
    split_cost = math.log2(max(len(concepts), 2)) + 2.0   # MDL: a split must beat its own description

    def grow(idx, path):
        v = V[idx]; n = len(idx)
        here = _bits(v)
        verd = _verdict(v)
        if n < 2 * min_leaf or here < 1e-6 or len(path) >= max_depth:
            rules.append(Rule(list(path), verd, n, float(v.mean()), here)); return
        best = None
        for con in concepts:
            m = con.mask[idx]; nl = int(m.sum())
            if nl < min_leaf or n - nl < min_leaf:
                continue
            g = here - (_bits(v[m]) + _bits(v[~m]))
            if best is None or g > best[0]:
                best = (g, con, m)
        if best is None or best[0] <= split_cost:
            rules.append(Rule(list(path), verd, n, float(v.mean()), here)); return
        _, con, m = best
        grow(idx[m], path + [(con, True)]); grow(idx[~m], path + [(con, False)])

    grow(np.arange(len(V)), [])
    return rules


def _model_bits(rules: List[Rule], n_atoms: int) -> float:
    a = math.log2(max(n_atoms, 2))
    return sum(len(r.path) * a + math.log2(3) for r in rules)


def _fit(kept: List[Concept], B, V, M, min_leaf) -> Tuple[List[Rule], float, float]:
    """Fit the existing library to this data: re-derive each concept's mask on these boards,
    rebuild the rule tree over them, and report ``(rules, resid, model)`` — how well what we
    already know explains what we now see."""
    for c in kept:
        v = c.expr.eval(B, M)
        c.mask = (v == c.const) if c.op == "=" else (v > c.const)
    rules = _build_rules(kept, V, min_leaf, max_depth=6)
    return rules, sum(r.resid for r in rules), _model_bits(rules, max(len(kept), 2))


# ───────────────────────────── the reuse loop with MDL round-stop ──────────────

@dataclass
class RoundInfo:
    number: int
    new_concepts: List[Concept]
    rules: List[Rule]
    residual: float
    data_saved: float
    cost: float
    kept: bool


@dataclass
class InventionResult:
    rounds: List[RoundInfo]
    concepts: List[Concept]        # everything kept across rounds
    rules: List[Rule]              # final rule set (last paying round)
    n_boards: int
    baseline_bits: float
    @property
    def stopped_after(self) -> int:
        paid = [r.number for r in self.rounds if r.kept]
        return paid[-1] if paid else 0


_BITS_PER_SYMBOL = math.log2(12)


def _invent_round(kept, prior_rules, resid, model, B, V, M, min_leaf, base, max_size, cap):
    """One round of invention: reuse ``kept`` as size-1 building blocks, search for new
    concepts, and keep what the rule tree actually uses iff it pays for itself in bits.
    Returns (new_used, rules, resid, model, data_saved, cost, paid); on an empty round it
    returns ([], prior_rules, resid, model, 0, 0, False)."""
    POOL = 40                                          # candidates offered to the rule tree
    if kept:
        library = list(base) + [Named(c.expr, c.expr.eval(B, M)) for c in kept]
        extra_atoms = (_group_fold_candidates(_supports(kept), B, _residual(prior_rules, V), min_leaf, M)
                       if M is not None else [])
    else:
        library = list(base) + _board_fold_terminals(); extra_atoms = []
    seen = _synthesize(library, B, max_size, cap)
    cands = _candidate_concepts(seen, B, V, min_leaf)
    have = {c.mask.tobytes() for c in kept}
    pool_new = [c for c in (extra_atoms + cands) if c.mask.tobytes() not in have][:POOL]
    if not pool_new:
        return [], prior_rules, resid, model, 0.0, 0.0, False
    rules = _build_rules(kept + pool_new, V, min_leaf, max_depth=6)
    used, used_keys = [], set()                        # charge only for what the tree uses
    for r in rules:
        for con, _ in r.path:
            key = con.mask.tobytes()
            if key not in used_keys:
                used_keys.add(key); used.append(con)
    new_used = [c for c in used if c.mask.tobytes() not in have]
    if not new_used:
        return [], prior_rules, resid, model, 0.0, 0.0, False
    new_resid = sum(r.resid for r in rules)
    new_model = _model_bits(rules, max(len(used), 2))
    data_saved = resid - new_resid
    cost = sum(c.size for c in new_used) * _BITS_PER_SYMBOL + max(new_model - model, 0.0)
    return new_used, rules, new_resid, new_model, data_saved, cost, data_saved > cost


def invent_from_boards(B: np.ndarray, V: np.ndarray, M: Optional[np.ndarray] = None, *,
                       max_rounds: int = 4, max_size: Optional[int] = None, cap="auto",
                       seed: Optional[List["Concept"]] = None) -> InventionResult:
    """Run the multi-round concept-invention loop on boards B with values V.

    ``M`` is the just-moved token per board (read from each board's transition); the
    group layer reads the board relative to it. Omit it only for move-free synthetic
    boards — they simply skip the group layer. ``max_size`` and ``cap`` bound the
    bottom-up search; left at their defaults they follow a board-width heuristic. Callers
    that know the target program is small (e.g. tests) can pass a tighter ``max_size``.

    ``seed`` is a library to start from — known concepts (e.g. a nim-sum transferred from a
    smaller scale, or the live library mid-training) that are carried in as building blocks and
    never re-derived: their masks are re-evaluated on *these* boards and the rule tree is built
    over them, so a sufficient seed yields a valid model even when no new concept is added.
    """
    N, n_cells = B.shape
    if max_size is None:
        max_size = 7 if n_cells <= 5 else 5     # reach: narrow boards may still need size-7 programs
    if cap == "auto":
        cap = 6000                              # but ALWAYS bound the search (cap=None was the explosion)
    min_leaf = max(3, N // 200)

    # base building blocks: cell reads + the small integer literals on the board
    base: List[Expr] = [Cell(i) for i in range(n_cells)]
    base += [Lit(v) for v in sorted(set(int(x) for x in np.unique(B)) | {0, 1})]

    rounds: List[RoundInfo] = []
    baseline = _bits(V)
    # carry the seed in: fit it to THESE boards and start the model from it
    kept: List[Concept] = list(seed) if seed else []
    if kept:
        final_rules, resid, model = _fit(kept, B, V, M, min_leaf)
    else:
        final_rules, resid, model = [], baseline, 0.0

    for k in range(1, max_rounds + 1):
        # Round 1 folds over the whole board (seeding e.g. the nim-sum); later rounds reuse
        # what was kept and fold over the groups it discovered.
        new_used, rep_rules, rep_resid, new_model, data_saved, cost, paid = _invent_round(
            kept, final_rules, resid, model, B, V, M, min_leaf, base, max_size, cap)
        rounds.append(RoundInfo(k, new_used, rep_rules, rep_resid, data_saved, cost, paid))
        if not paid:
            break
        kept = kept + new_used; resid = rep_resid; model = new_model; final_rules = rep_rules

    return InventionResult(rounds, kept, final_rules, N, baseline)


# ───────────────────────────── the bounded data view ───────────────────────────

# Discovery runs over at most this many boards. It is a compute budget (a fit is linear
# in the table, a search is table × programs — thousands keeps both interactive), not a
# tuned knob: any cap comfortably above the rule tree's ``min_leaf`` floor yields the
# same discoveries, because a concept is a *program* — a regularity visible in any fair
# sample, not a statistic that needs every board. Small games come in under it whole;
# bigger ones are uniformly subsampled by :meth:`ConceptLibrary.rebuild`.
# (A per-wave reservoir + live refit over this budget was built, benched, and deleted:
# training-time selection never reads the concept signal, so the live fit's only consumer
# was the value loop's heal — where a noisy mid-wave refit could transiently collapse the
# tree and poison one healing pass. Discovery now happens only where completed values
# exist: at the loop's boundaries.)
CAP = 6000


# ───────────────────────────── (de)serialisation for persistence ───────────────

def expr_to_dict(e: Expr) -> dict:
    """Serialise a feature program to a plain dict (for SQLite). The program is what's
    board-order-independent and reusable; masks/value-vectors are never stored."""
    if isinstance(e, Cell):  return {"t": "Cell", "i": e.i}
    if isinstance(e, Lit):   return {"t": "Lit", "v": e.v}
    if isinstance(e, BinOp): return {"t": "BinOp", "op": e.op, "a": expr_to_dict(e.a), "b": expr_to_dict(e.b)}
    if isinstance(e, Named): return {"t": "Named", "inner": expr_to_dict(e.inner)}
    if isinstance(e, Elem):  return {"t": "Elem", "j": e.j, "names": list(e.names)}
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
    if t == "Cell":  return Cell(d["i"])
    if t == "Lit":   return Lit(d["v"])
    if t == "BinOp": return BinOp(d["op"], expr_from_dict(d["a"]), expr_from_dict(d["b"]))
    if t == "Named": return Named(expr_from_dict(d["inner"]), np.empty(0, dtype=np.int64))
    if t == "Elem":  return Elem(d["j"], tuple(d["names"]))
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


def _boards_values(memory) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull (after-board, value, just-moved token) from the stored transitions. The move
    is read straight from the before→after diff — the token the mover just placed (the
    new non-empty value at a changed cell); boards whose before-board isn't stored get 0."""
    boards, trans = memory._build_trans_scores()
    bv: Dict[tuple, float] = {}
    bm: Dict[tuple, int] = {}
    canon: Dict[tuple, str] = {}                                 # board → its smallest incoming from_hash
    for (fh, th), (counts, score) in trans.items():
        if th not in boards:
            continue
        after = np.asarray(boards[th]).ravel()
        key = tuple(int(x) for x in after)
        # A board is reached by many transitions whose scores all converge to its Bellman
        # value; pre-convergence they can differ. Read the value from the smallest-from_hash
        # incoming transition so the result is independent of dict iteration order.
        if key in canon and fh >= canon[key]:
            continue
        canon[key] = fh
        bv[key] = float(score)
        placed = []
        if fh in boards:
            before = np.asarray(boards[fh]).ravel()
            placed = after[(before != after) & (after != 0)]    # what the move put down
        bm[key] = int(placed[0]) if len(placed) else 0
    keys = list(bv.keys())
    if not keys:
        return np.zeros((0, 0), dtype=np.int64), np.zeros(0), np.zeros(0, dtype=np.int64)
    B = np.array(keys, dtype=np.int64)
    V = np.array([bv[k] for k in keys], dtype=float)
    M = np.array([bm.get(k, 0) for k in keys], dtype=np.int64)   # the move that reached each board
    return B, V, M


def invent(memory, game_id: Optional[str] = None, **kw) -> InventionResult:
    """Invent concepts from a trained memory's stored transitions."""
    B, V, M = _boards_values(memory)
    if len(B) < 8:
        return InventionResult([], [], [], len(B), 0.0)
    return invent_from_boards(B, V, M, **kw)


def meaning(c: Concept) -> Optional[str]:
    """A plain-English reading of a fold concept, side by side with its formula.

    Derived, never asserted: a fold body over g-cell lines has only the (played, empty)
    pairs with played + empty ≤ g as possible inputs, so we enumerate the body over all of
    them and say which line-states the threshold picks out ("them" = g − played − empty).
    Returns ``None`` for concepts that are already readable (plain cell arithmetic)."""
    e = c.expr.inner if isinstance(c.expr, Named) else c.expr
    if not isinstance(e, Fold) or c.op != "=":
        return None
    if isinstance(e.domain, BoardDomain):
        word = _WORD.get(e.op, e.op)
        body = "every cell" if isinstance(e.body, Elem) else f"{e.body} over every cell"
        return f"the {word} of {body} is {c.const}"
    if not isinstance(e.domain, GroupDomain):
        return None
    sizes = {len(g) for g in e.domain.groups}
    if len(sizes) != 1:
        return None                                          # mixed-size groups: no single table
    g = sizes.pop()
    pairs = [(p, emp) for p in range(g + 1) for emp in range(g + 1 - p)]
    vals = e.body.eval(np.array(pairs, dtype=np.int64))
    by_val: Dict[int, List[str]] = {}
    for (p, emp), v in zip(pairs, vals):
        by_val.setdefault(int(v), []).append(f"you {p} · empty {emp} · them {g - p - emp}")
    # the gloss speaks the program's own vocabulary: the formula says "groups" (the regions
    # the search discovered — lines in Tic-Tac-Toe, whatever they are elsewhere), so the
    # reading says "group" — no game-specific words are ever introduced
    if e.op in ("max", "min") and c.const in by_val:
        hits = by_val[c.const]
        if e.op == "max":
            # whichever reading is shorter: the states the top group may be, or — when the
            # threshold sits low — the states no group is allowed to reach above it
            above = [d for v, ds in by_val.items() if v > c.const for d in ds]
            if above and len(above) < len(hits):
                return f"no group is {' or '.join(above)}"
            return f"the top-scoring group: {' or '.join(hits)}"
        return f"the lowest-scoring group: {' or '.join(hits)}"
    if e.op == "+":
        scores = "; ".join(f"{d} → {v}" for v in sorted(by_val) if v != 0 for d in by_val[v])
        return f"group scores ({scores}) sum to {c.const}"
    return None


def _fold_groups(concepts) -> List[Tuple[int, ...]]:
    """The distinct cell-regions the group-folds among ``concepts`` walk, in first-seen
    order — so a reading like "the top-scoring group" can be grounded in actual cells."""
    out, seen = [], set()
    for c in concepts:
        e = c.expr.inner if isinstance(c.expr, Named) else c.expr
        if isinstance(e, Fold) and isinstance(e.domain, GroupDomain):
            for g in e.domain.groups:
                if g not in seen:
                    seen.add(g); out.append(g)
    return out


def _groups_line(concepts) -> Optional[str]:
    regions = _fold_groups(concepts)
    if not regions:
        return None
    cells = " ".join("(" + "·".join(f"c{i}" for i in g) + ")" for g in regions)
    return f"groups = the board regions it discovered: {cells}"


def render(res: InventionResult, label: str = "") -> str:
    """Human-readable report of an invention run. Every fold concept is shown side by side
    with its derived plain-English reading (⟺ lines), so the raw output is never lost."""
    out: List[str] = []
    out.append("")
    out.append(f"══ CONCEPT INVENTION{(' — ' + label.upper()) if label else ''} ══"
               f"   ({res.n_boards} boards · baseline {res.baseline_bits:,.0f} bits to explain)")
    if not res.rounds:
        out.append("  (not enough data to invent from)"); return "\n".join(out)
    out.append("")
    for r in res.rounds:
        if r.kept:
            out.append(f"ROUND {r.number}  ✓ pays — saved {r.data_saved:,.0f} bits  vs  {r.cost:,.0f} cost   "
                       f"({len(r.new_concepts)} concept{'s' if len(r.new_concepts) != 1 else ''} invented)")
            for c in r.new_concepts:
                out.append(f"        + {c}")
                m = meaning(c)
                if m:
                    out.append(f"          ⟺ {m}")
        else:
            out.append(f"ROUND {r.number}  ✗ stop — saved {r.data_saved:,.0f} bits  vs  {r.cost:,.0f} cost   "
                       f"(nothing new pays for itself)")
    out.append("")
    out.append(f"→ stopped after round {res.stopped_after};  {len(res.concepts)} concept(s) kept.")
    out.append("")
    out.append("RULES it builds from the invented concepts:")
    for rule in sorted(res.rules, key=lambda r: -r.avg):
        out.append(f"   [{rule.verdict:<4}] n={rule.n:<5} avg={rule.avg:.2f}   {rule.render()}")
    # the key: every fold used by a rule, with its derived reading — formulas above stay verbatim
    used, seen = [], set()
    for rule in res.rules:
        for con, _ in rule.path:
            key = str(con)
            if key not in seen and meaning(con):
                seen.add(key); used.append(con)
    if used:
        out.append("")
        out.append("KEY — each fold above, in plain terms (derived from the program, not asserted):")
        gl = _groups_line(used)
        if gl:
            out.append(f"   {gl}")
        for con in used:
            out.append(f"   {con}")
            out.append(f"      ⟺ {meaning(con)}")
    return "\n".join(out)
