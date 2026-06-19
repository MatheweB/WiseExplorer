"""The discovery engine — MDL scoring, the bottom-up search, structural reuse,
rule-tree building, the multi-round invent loop, and the DB entry point."""
from __future__ import annotations

import math

from dataclasses import dataclass

import numpy as np

from wise_explorer.synthesis.exprs import (
    Expr, Cell, Lit, BinOp, UnaryOp, Named, Elem, Fold,
    CellDomain, BoardDomain, GroupDomain, IncidenceDomain, Concept, _OPS, _UNARY, _FOLD,
)


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

def _synthesize(terminals: list[Expr], B: np.ndarray, max_size: int, cap: int | None):
    """Return {value_bytes: Expr} — one smallest program per distinct behaviour."""
    seen: dict[bytes, Expr] = {}
    by_size: dict[int, list[tuple[Expr, np.ndarray]]] = {s: [] for s in range(1, max_size + 1)}

    def add(e: Expr):
        vec = e.eval(B).astype(np.int64)
        key = vec.tobytes()
        if key not in seen:
            seen[key] = e
            by_size[e.size].append((e, vec))

    for t in terminals:
        add(t)
    for s in range(2, max_size + 1):
        for u in _UNARY:                                  # sgn / abs of a smaller program
            for ea, va in by_size[s - 1]:
                if cap and len(by_size[s]) >= cap:
                    break
                vec = _UNARY[u](va).astype(np.int64)
                key = vec.tobytes()
                if key not in seen:
                    e = UnaryOp(u, ea)
                    seen[key] = e
                    by_size[s].append((e, vec))
        for op in _OPS:                                   # every _OP is commutative, so only
            for i in range(1, s - 1):                      # enumerate i ≤ j — (j,i) would just
                j = s - 1 - i                              # repeat (i,j) with the operands swapped
                if i > j:
                    continue
                left = by_size[i]
                for ai, (ea, va) in enumerate(left):
                    rest = by_size[j][ai:] if i == j else by_size[j]   # same size ⇒ upper triangle
                    for eb, vb in rest:
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

def _candidate_concepts(seen, B, V, min_leaf) -> list[Concept]:
    """Derive boolean atoms (feature == c) and keep the best per yes/no partition.

    The variance-reduction gain of every (program, value) split is computed from *grouped
    sums* in one vectorised pass per program — far cheaper than calling numpy ``.var()``
    once per split, which is almost all per-call overhead on these small arrays. (A fully
    batched flat-bincount version was built and benched: bit-identical results, 0.9× —
    the time lives in admitting kept candidates, not in the per-program numpy calls — so
    the simpler per-program form stays.)

    Ranks by variance, NOT the tree's exact `_bits` entropy gain. Making the two consistent
    was measured and reverted: the exact criterion fires `gain>0` on nearly every split
    (~15k candidates) and ranks pure-tiny-group isolators top, starving the group-fold layer
    — TTT discovery collapsed (0 concepts at 2k; cryptic xor/dist composites by 5k). Variance
    is the better *prefilter*; acceptance still uses `_bits`."""
    N = len(V)
    V2 = V * V
    S = float(V.sum())
    SS = float(V2.sum())
    total_var = SS / N - (S / N) ** 2
    best: dict[bytes, tuple[float, Concept]] = {}
    # The errstate is hoisted to wrap the whole loop: empty value-groups produce 0/0
    # (clamped below), and re-entering the context once per candidate is pure overhead.
    with np.errstate(invalid="ignore", divide="ignore"):
        for vec_bytes, expr in seen.items():
            vec = np.frombuffer(vec_bytes, dtype=np.int64)
            # Group V by the program's output value with a single bincount per stat — no
            # per-vector sort. Shift to a non-negative offset so the value IS the bin index;
            # empty bins (gaps in the value range) fall out via the min_leaf filter below.
            lo = int(vec.min())
            span = int(vec.max()) - lo + 1
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
                vals, inv = np.unique(vec, return_inverse=True)
                K = len(vals)
                n1 = np.bincount(inv, minlength=K).astype(np.float64)
                s1 = np.bincount(inv, weights=V, minlength=K)
                ss1 = np.bincount(inv, weights=V2, minlength=K)
                base = vals                                        # value of bin k is vals[k]
            n0 = N - n1
            s0 = S - s1
            ss0 = SS - ss1                # the complement (value != c)
            var1 = np.maximum(ss1 / n1 - (s1 / n1) ** 2, 0.0)       # clamp float roundoff at 0
            var0 = np.maximum(ss0 / n0 - (s0 / n0) ** 2, 0.0)
            gain = total_var - (n1 * var1 + n0 * var0) / N
            keep = (n1 >= min_leaf) & (n0 >= min_leaf) & (gain > 0)
            for k in np.nonzero(keep)[0]:
                c = (base + int(k)) if np.isscalar(base) else int(base[k])
                mask = vec == c
                sig = mask.tobytes()
                if sig not in best or expr.size < best[sig][1].size:
                    best[sig] = (float(gain[k]), Concept(expr, "=", c, mask, expr.size))
    # round the gain in the tiebreak so equivalent splits prefer the SIMPLER concept
    # deterministically (float roundoff in the grouped sums must not flip the order)
    return [c for _, c in sorted(best.values(), key=lambda x: (-round(x[0], 9), x[1].size))]


# ───────────────────────────── structural reuse (counting + threats) ───────────

def _cell_group(e: Expr) -> tuple[int, ...]:
    """The distinct cells a concept's program reads (its support), seen through
    Named (reuse) and Fold (the fold's domain)."""
    out: set = set()
    def walk(x: Expr):
        if isinstance(x, Cell):
            out.add(x.i)
        elif isinstance(x, BinOp):
            walk(x.a)
            walk(x.b)
        elif isinstance(x, UnaryOp):
            walk(x.a)
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
    if isinstance(e, UnaryOp):
        return _is_atomic(e.a)
    return isinstance(e, (Cell, Lit))


def _supports(kept: list[Concept]) -> list[tuple[int, ...]]:
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
            seen.add(g)
            out.append(g)
    return out


def _board_fold_terminals() -> list[Expr]:
    """Seed each round-1 search with the whole-board fold under every monoid — e.g.
    fold(⊕, board, cell) is the nim-sum. The domain is width-free (:class:`BoardDomain`), so a
    fold kept here is the same program at any board width and transfers across scales unchanged.
    Observational-equivalence keeps whichever of these (or a cheaper composition) actually
    predicts the value; the rest are dropped."""
    board = BoardDomain()
    return [Fold(op, board, Elem(0, BoardDomain.names)) for op in _FOLD]


def _residual(rules: list[Rule], V: np.ndarray) -> np.ndarray:
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


def _fold_search(dom, B, target, min_leaf, m, cap=None) -> list[tuple]:
    """Search bodies over a fold domain's per-element features and fold each two ways: ``max``
    (∃ an element where the body fires) and ``+`` (count the elements where it fires). Each fold
    is thresholded and scored against the residual. Returns ``(gain, size, Concept)`` triples."""
    T = dom.tensor(B, m)
    N, K, F = T.shape
    flat = T.reshape(N * K, F)
    bodies = _synthesize([Elem(j, dom.names) for j in range(F)], flat, max_size=5, cap=cap)
    out = []
    for body in bodies.values():
        per = body.eval(flat).reshape(N, K)
        for op in ("max", "+"):
            fn, ident = _FOLD[op]
            vec = fn.reduce(per, axis=1, initial=ident).astype(np.int64)
            fold = Fold(op, dom, body)
            for c in np.unique(vec):                      # outer threshold also discovered
                mask = vec == c
                n1 = int(mask.sum())
                if n1 < min_leaf or N - n1 < min_leaf:
                    continue
                gain = target.var() - (n1/N*target[mask].var() + (N-n1)/N*target[~mask].var())
                if gain > 0:
                    out.append((gain, fold.size, Concept(fold, "=", int(c), mask, fold.size)))
    return out


def _group_fold_candidates(supports, B: np.ndarray, target: np.ndarray, min_leaf: int, m) -> list[Concept]:
    """Discover threats and forks over the discovered groups (``supports`` — the cells the kept
    concepts read). Two fold views compete on residual gain: GroupDomain folds over the lines
    themselves (their (played, empty) counts — threats and threat-COUNTS), and IncidenceDomain
    folds over the cells the lines share (how many lines each cell completes — the true FORK,
    two distinct completion cells, which a flat line-count cannot tell from one shared cell).
    The MDL gate keeps whichever a fold actually pays for."""
    if not supports:
        return []
    scored = _fold_search(GroupDomain(supports), B, target, min_leaf, m) \
        + _fold_search(IncidenceDomain(supports), B, target, min_leaf, m, cap=CAP)
    scored.sort(key=lambda x: (-x[0], x[1]))              # best residual gain, then smallest
    return [c for _, _, c in scored[:6]]                  # a residual-ranked handful


# ───────────────────────────── rules over invented concepts ────────────────────

@dataclass
class Rule:
    path: list[tuple[Concept, bool]]
    verdict: str
    n: int
    avg: float
    resid: float
    mix: tuple = (0.0, 0.0, 0.0)             # fraction of the leaf's boards per outcome, each
                                             # classed to its nearest anchor on the COMPLETED
                                             # values (cert-pinned backup) — NOT raw counts
    def render(self) -> str:
        if not self.path:
            return "(everything)"
        return " AND ".join((str(c) if t else f"¬[{c}]") for c, t in self.path)


def _verdict(v: np.ndarray) -> str:
    return ["LOSS", "DRAW", "WIN"][int(_soft_counts(v).argmax())]   # heaviest outcome mass


def _verdicts(v: np.ndarray) -> np.ndarray:
    """Each value's outcome class (0=LOSS, 1=DRAW, 2=WIN) by the same anchors as ``_soft_counts``
    — add an outcome anchor here and both the leaf labels and earned forgetting extend with it."""
    ml = np.maximum(0.0, 1.0 - 2.0 * v)
    mw = np.maximum(0.0, 2.0 * v - 1.0)
    return np.argmax(np.stack([ml, 1.0 - ml - mw, mw]), axis=0)


def _build_rules(concepts: list[Concept], V: np.ndarray, min_leaf: int):
    rules: list[Rule] = []
    split_cost = math.log2(max(len(concepts), 2)) + 2.0   # MDL: a split must beat its own description

    def grow(idx, path):
        v = V[idx]
        n = len(idx)
        here = _bits(v)
        verd = _verdict(v)
        mix = tuple(np.bincount(_verdicts(v), minlength=3) / n)   # fraction of boards per outcome,
        #                          each board classed to its nearest anchor — a pure leaf is all-one
        # a node whose total bits don't exceed the cost of naming one split can never
        # pay — this single derived test is the whole leaf condition (it also bounds the
        # depth: every split must pay >= split_cost out of a finite bit budget)
        if here <= split_cost:
            rules.append(Rule(list(path), verd, n, float(v.mean()), here, mix))
            return
        best = None
        for con in concepts:
            m = con.mask[idx]
            nl = int(m.sum())
            if nl < min_leaf or n - nl < min_leaf:
                continue
            g = here - (_bits(v[m]) + _bits(v[~m]))
            if best is None or g > best[0]:
                best = (g, con, m)
        if best is None or best[0] <= split_cost:
            rules.append(Rule(list(path), verd, n, float(v.mean()), here, mix))
            return
        _, con, m = best
        grow(idx[m], path + [(con, True)])
        grow(idx[~m], path + [(con, False)])

    grow(np.arange(len(V)), [])
    return rules


def _model_bits(rules: list[Rule], n_atoms: int) -> float:
    a = math.log2(max(n_atoms, 2))
    return sum(len(r.path) * a + math.log2(3) for r in rules)


def _min_leaf(cap: int = None) -> int:
    """The rule tree's leaf floor: the smallest value-group whose own bits could ever cover a
    split's description cost. Derived from the candidate budget, not tuned (cap=6000 → 10)."""
    return math.ceil((math.log2(cap or CAP) + 2) / math.log2(3))


def _fit(kept: list[Concept], B, V, M, min_leaf) -> tuple[list[Rule], float, float]:
    """Fit the existing library to this data: re-derive each concept's mask on these boards,
    rebuild the rule tree over them, and report ``(rules, resid, model)`` — how well what we
    already know explains what we now see."""
    for c in kept:
        v = c.expr.eval(B, M)
        c.mask = (v == c.const) if c.op == "=" else (v > c.const)
    rules = _build_rules(kept, V, min_leaf)
    return rules, sum(r.resid for r in rules), _model_bits(rules, max(len(kept), 2))


# ───────────────────────────── the reuse loop with MDL round-stop ──────────────

@dataclass
class RoundInfo:
    number: int
    new_concepts: list[Concept]
    rules: list[Rule]
    residual: float
    data_saved: float
    cost: float
    kept: bool


@dataclass
class InventionResult:
    rounds: list[RoundInfo]
    concepts: list[Concept]        # everything kept across rounds
    rules: list[Rule]              # final rule set (last paying round)
    n_boards: int
    baseline_bits: float
    tokens: tuple[int, ...] = ()   # nonzero cell values seen — lets glosses enumerate honestly
    shape: tuple[int, int] | None = None  # native board shape — lets regions earn place-names
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
        library = list(base) + _board_fold_terminals()
        extra_atoms = []
    seen = _synthesize(library, B, max_size, cap)
    cands = _candidate_concepts(seen, B, V, min_leaf)
    have = {c.mask.tobytes() for c in kept}
    seen_masks = set(have)
    pool_new = []                                          # dedup within the pool too, not just
    for c in extra_atoms + cands:                          # vs kept — so POOL slots aren't wasted
        key = c.mask.tobytes()                             # on two equivalent new atoms
        if key in seen_masks:
            continue
        seen_masks.add(key)
        pool_new.append(c)
        if len(pool_new) >= POOL:
            break
    if not pool_new:
        return [], prior_rules, resid, model, 0.0, 0.0, False
    rules = _build_rules(kept + pool_new, V, min_leaf)
    used, used_keys = [], set()                        # charge only for what the tree uses
    for r in rules:
        for con, _ in r.path:
            key = con.mask.tobytes()
            if key not in used_keys:
                used_keys.add(key)
                used.append(con)
    new_used = [c for c in used if c.mask.tobytes() not in have]
    if not new_used:
        return [], prior_rules, resid, model, 0.0, 0.0, False
    new_resid = sum(r.resid for r in rules)
    new_model = _model_bits(rules, max(len(used), 2))
    data_saved = resid - new_resid
    cost = sum(c.size for c in new_used) * _BITS_PER_SYMBOL + max(new_model - model, 0.0)
    return new_used, rules, new_resid, new_model, data_saved, cost, data_saved > cost


def invent_from_boards(B: np.ndarray, V: np.ndarray, M: np.ndarray | None = None, *,
                       max_rounds: int = 32, max_size: int | None = None, cap="auto",
                       seed: list[Concept] | None = None) -> InventionResult:
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
        cap = CAP                               # but ALWAYS bound the search (cap=None was the explosion)
    min_leaf = _min_leaf(cap)   # smallest value-group worth naming (derived from the budget)

    # base building blocks: cell reads + the small integer literals on the board
    base: list[Expr] = [Cell(i) for i in range(n_cells)]
    base += [Lit(v) for v in sorted(set(int(x) for x in np.unique(B)) | {0, 1})]

    rounds: list[RoundInfo] = []
    baseline = _bits(V)
    # carry the seed in: fit it to THESE boards and start the model from it
    kept: list[Concept] = list(seed) if seed else []
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
        kept = kept + new_used
        resid = rep_resid
        model = new_model
        final_rules = rep_rules

    return InventionResult(rounds, kept, final_rules, N, baseline,
                           tuple(int(t) for t in np.unique(B) if t != 0))


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

# The smallest table any fit may run on. Below this even a single leaf's average is a
# coin flip — a statistical floor, not a tuned knob; shared by discovery, the library's
# rebuild guard, and the runner's first wheel turn.
MIN_BOARDS = 8


# ───────────────────────────── (de)serialisation for persistence ───────────────


def _boards_values(memory, boards=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull (after-board, value, just-moved token) from the stored transitions. The move
    is read straight from the before→after diff — the token the mover just placed (the
    new non-empty value at a changed cell); boards whose before-board isn't stored get 0.
    ``boards`` skips the reload when the caller already holds the boards table."""
    boards, trans = memory._build_trans_scores(boards)
    bv: dict[tuple, float] = {}
    bm: dict[tuple, int] = {}
    canon: dict[tuple, str] = {}                                 # board → its smallest incoming from_hash
    for (fh, th), (_counts, score) in trans.items():
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


def invent(memory, game_id: str | None = None, **kw) -> InventionResult:
    """Invent concepts from a trained memory's stored transitions."""
    B, V, M = _boards_values(memory)
    if len(B) < MIN_BOARDS:
        return InventionResult([], [], [], len(B), 0.0)
    res = invent_from_boards(B, V, M, **kw)
    res.shape = _board_shape(memory)
    return res


def _board_shape(memory) -> tuple[int, int] | None:
    row = memory.conn.execute(
        "SELECT board_rows, board_cols FROM boards LIMIT 1").fetchone()
    return (int(row[0]), int(row[1])) if row else None


