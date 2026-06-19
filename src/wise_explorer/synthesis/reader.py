"""The human-readable renderer — turns the discovered rule tree and concepts into
named, one-floor-deep formulas with derived ⟺ readings."""
from __future__ import annotations

import itertools

import numpy as np

from wise_explorer.synthesis.exprs import (
    Expr, Cell, Lit, BinOp, UnaryOp, Named, Elem, Fold, CellDomain, BoardDomain,
    GroupDomain, IncidenceDomain, Concept, _OPS, _WORD, _FOLD,
)
from wise_explorer.synthesis.engine import Rule, InventionResult


def meaning(c: Concept, tokens: tuple[int, ...] = (), brief: bool = False,
            names: dict[str, str] | None = None) -> str | None:
    """A plain-English reading of a concept, side by side with its formula.

    Derived, never asserted: a fold body over g-cell lines has only the (played, empty)
    pairs with played + empty ≤ g as possible inputs, so we enumerate the body over all of
    them and say which line-states the threshold picks out ("them" = g − played − empty).
    Cell-lattice chains (``and``/``or`` over cells) are likewise enumerated — but over the
    board's token alphabet, so they gloss only when ``tokens`` is given. Returns ``None``
    where no honest reading is derivable."""
    e = c.expr.inner if isinstance(c.expr, Named) else c.expr
    if not isinstance(e, Fold):
        return _lattice_meaning(e, c.op, c.const, tokens, names)
    if isinstance(e.domain, BoardDomain):
        word = _WORD.get(e.op, e.op)
        body = "every cell" if isinstance(e.body, Elem) else f"{e.body} over every cell"
        rel = "is" if c.op == "=" else "exceeds"
        return f"the {word} of {body} {rel} {c.const}"
    if isinstance(e.domain, IncidenceDomain):
        return _incidence_meaning(e, c.op, c.const)
    if not isinstance(e.domain, GroupDomain):
        return None
    sizes = {len(g) for g in e.domain.groups}
    if len(sizes) != 1:
        return None                                          # mixed-size groups: no single table
    g = sizes.pop()
    pairs = [(p, emp) for p in range(g + 1) for emp in range(g + 1 - p)]
    vals = e.body.eval(np.array(pairs, dtype=np.int64))
    by_val: dict[int, list[str]] = {}
    for (p, emp), v in zip(pairs, vals):
        by_val.setdefault(int(v), []).append(f"you {p} · empty {emp} · them {g - p - emp}")
    # the gloss speaks the program's own vocabulary: the formula says "groups" (the regions
    # the search discovered — lines in Tic-Tac-Toe, whatever they are elsewhere), so the
    # reading says "group" — no game-specific words are ever introduced
    if c.op == ">":
        above = [d for v, ds in sorted(by_val.items()) if v > c.const for d in ds]
        if not above:
            return None
        if e.op == "max":
            return f"some group is {' or '.join(above)}"
        if e.op == "min":
            return f"every group is {' or '.join(above)}"
        return None
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
        if set(by_val) <= {0, 1}:                            # 0/1 body: the fold is a count
            what = " or ".join(by_val.get(1, []))
            return f"how many groups are {what} — the count is {c.const}"
        if brief:
            return None                                      # a table, not a sentence — KEY only
        scores = "; ".join(f"{d} → {v}" for v in sorted(by_val) if v != 0 for d in by_val[v])
        return f"group scores ({scores}) sum to {c.const}"
    return None


def _body_features(e: Expr) -> set:
    """Which incidence features (0=empty, 1=played_short, 2=other_short) the body actually reads."""
    out = set()
    def walk(x):
        if isinstance(x, Elem):
            out.add(x.j)
        elif isinstance(x, BinOp):
            walk(x.a); walk(x.b)
        elif isinstance(x, UnaryOp):
            walk(x.a)
    walk(e)
    return out


# the raw incidence features (after ``empty``), each a count of incident groups in one line-state
_INC_SIDE = {1: ("you", 0), 2: ("you", 1), 3: ("you", 2), 4: ("they", 1), 5: ("they", 2)}


def _incidence_meaning(e: Fold, op: str, const: int) -> str | None:
    """Reading for a fold over the cell×group incidence. The body reads raw per-cell counts — how
    many incident groups sit at each line-state — so, like the group reading, we state the
    structure and let the reader draw the tactic: a cell that is empty AND shares a group where you
    already hold 2 is a winning square; counting those (a `+` fold) is a fork. The legible forms are
    counting empty cells, and counting/finding empty cells that share a group at one line-state;
    arbitrary arithmetic on the counts gets no sentence (None — the formula speaks)."""
    feats = _body_features(e.body)
    if feats == {0}:                                       # the body reads only emptiness
        if e.op == "+":
            return f"how many cells are empty — the count is {const}"
        return "some cell is empty" if e.op == "max" and const >= 1 else None
    counts = sorted(feats - {0})
    if 0 not in feats or not counts or any(c not in _INC_SIDE for c in counts):
        return None                                        # not "emptiness + line-states" → no sentence
    # the legible form: an empty cell that shares a group at one of these line-states. Confirm the
    # body fires EXACTLY when the cell is empty AND at least one of the read counts is non-zero.
    grid = [(emp, combo) for emp in (0, 1)
            for combo in itertools.product(range(3), repeat=len(counts))]
    arr = np.zeros((len(grid), 6), dtype=np.int64)
    for r, (emp, combo) in enumerate(grid):
        arr[r, 0] = emp
        for c, n in zip(counts, combo):
            arr[r, c] = n
    vals = [int(v) for v in e.body.eval(arr)]
    if [v > 0 for v in vals] != [bool(emp and any(combo)) for emp, combo in grid]:
        return None
    where = "a group where " + " or ".join(f"{s} hold {k}" for s, k in (_INC_SIDE[c] for c in counts))
    if e.op == "max":
        return f"no empty cell shares {where}" if const == 0 else f"some empty cell shares {where}"
    if e.op == "+" and set(vals) <= {0, 1}:
        return f"how many empty cells share {where} — the count is {const}"
    return None


# ───────────────────────────── derived readings for cell-lattice chains ────────

def _chain(e: Expr, op: str) -> list[Expr]:
    """Flatten a BinOp chain of one associative op into its operand list."""
    if isinstance(e, BinOp) and e.op == op:
        return _chain(e.a, op) + _chain(e.b, op)
    return [e]


def _strip(e: Expr) -> Expr:
    return e.inner if isinstance(e, Named) else e


def _chain_cells(e: Expr, op: str) -> list[int] | None:
    """The cell indices of a pure ``op``-chain over cell reads, else None."""
    cells = []
    for part in _chain(_strip(e), op):
        part = _strip(part)
        if not isinstance(part, Cell):
            return None
        cells.append(part.i)
    return None if len(cells) < 2 else cells


def _uniform_chain(e: Expr, tokens: tuple[int, ...]) -> list[int] | None:
    """If ``e`` is an &-chain over cells whose value — enumerated over the actual token
    alphabet — is nonzero exactly when all its cells hold the same nonzero token, return
    the cells. This is the derivation behind "all one player's"; with overlapping bit
    patterns (chess piece types) the check fails and no claim is made."""
    cells = _chain_cells(e, "&")
    if not cells or not tokens or len(tokens) ** len(cells) > 4096:
        return None
    toks = np.array([0, *tokens], dtype=np.int64)
    combos = np.stack(np.meshgrid(*[toks] * len(cells), indexing="ij"), -1).reshape(-1, len(cells))
    B = np.zeros((len(combos), max(cells) + 1), dtype=np.int64)
    B[:, cells] = combos
    vals = _strip(e).eval(B)
    uniform = (combos == combos[:, :1]).all(1) & (combos[:, 0] != 0)
    return cells if bool(((vals != 0) == uniform).all()) else None


def _cells_str(cells) -> str:
    return "·".join(f"c{i}" for i in cells)


def _xor_pair(e: Expr, tokens) -> str | None:
    """"A and B carry equal values" for an ⊕ of two uniform chains."""
    e = _strip(e)
    if isinstance(e, BinOp) and e.op == "⊕":
        a, b = _uniform_chain(e.a, tokens), _uniform_chain(e.b, tokens)
        if a and b:
            return f"({_cells_str(a)}) and ({_cells_str(b)}) carry equal values"
    return None


def _lattice_meaning(e: Expr, op: str, const: int, tokens: tuple[int, ...],
                     names: dict[str, str] | None = None) -> str | None:
    """Readings for and/or/xor chains over cells — the region concepts and their
    combinations. Everything here is an identity of the lattice ops (an |-chain is 0 iff
    every part is; an ⊕-pair is 0 iff the parts are equal) or an enumeration over the
    token alphabet — derived, never asserted."""
    if op != "=" or const != 0:
        return None
    cells = _uniform_chain(e, tokens)
    if cells:
        return f"{_cells_str(cells)} are not all one player's"
    pair = _xor_pair(e, tokens)
    if pair:
        return pair
    if not (isinstance(_strip(e), BinOp) and _strip(e).op == "|"):
        return None
    plain, clauses = [], []
    for part in _chain(_strip(e), "|"):
        cells = _uniform_chain(part, tokens)
        if cells:
            plain.append(f"({_cells_str(cells)})")
            continue
        pair = _xor_pair(part, tokens)
        if pair:
            clauses.append(pair)
            continue
        if names is not None:
            h = names.get(str(part))                     # a promoted floor below
            if h:
                clauses.append(f"{h} is 0")
                continue
        return None
    out = f"no group among {' '.join(plain)} is all one player's" if plain else ""
    if clauses:
        out += (", and " if out else "") + "; ".join(clauses)
    return out or None


def _fold_groups(concepts) -> list[tuple[int, ...]]:
    """The distinct cell-regions the group-folds among ``concepts`` walk, in first-seen
    order — so a reading like "the top-scoring group" can be grounded in actual cells."""
    out, seen = [], set()
    for c in concepts:
        e = c.expr.inner if isinstance(c.expr, Named) else c.expr
        if isinstance(e, Fold) and isinstance(e.domain, GroupDomain):
            for g in e.domain.groups:
                if g not in seen:
                    seen.add(g)
                    out.append(g)
    return out


# ───────────────────────────── the pretty reader: names all the way up ─────────
# A promoted concept costs size 1 to *compose* with — printing it expanded throws that
# compression away and is what buried the TTT rules under parentheses. The reader mirrors
# the search instead: every program gets a handle (K₁, K₂ …), every definition is written
# one floor deep (in the handles of the floor below), and the rule tree prints each split
# once. ``expand=True`` recovers the fully-expanded formulas.

_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _handles(concepts: list[Concept]) -> dict[str, str]:
    """Program-text → handle, in library (round) order. Keyed by the *full* formula text so
    a Named reuse deep inside a later concept finds the earlier program's name. Handles
    name programs, not thresholds: two concepts sharing a program share its K."""
    names: dict[str, str] = {}
    for c in concepts:
        key = str(_strip(c.expr))
        if key not in names:
            names[key] = f"K{str(len(names) + 1).translate(_SUB)}"
    return names


def _pretty(e: Expr, names: dict[str, str]) -> str:
    """Render a program one floor deep: any subprogram with a handle prints as the handle;
    associative chains print flat (associativity is what makes them folds)."""
    s = str(e)
    if s in names:
        return names[s]
    if isinstance(e, Named):
        return _pretty(e.inner, names)
    if isinstance(e, BinOp):
        w = _WORD.get(e.op, e.op)
        if e.op in _FOLD:                                   # the associative ops
            return "(" + f" {w} ".join(_pretty(p, names) for p in _chain(e, e.op)) + ")"
        return f"({_pretty(e.a, names)} {w} {_pretty(e.b, names)})"
    if isinstance(e, UnaryOp):
        return f"{e.op}({_pretty(e.a, names)})"
    if isinstance(e, Fold):
        return f"fold({e.op}, {e.domain}, {_pretty(e.body, names)})"
    return s


def _region_geometry(g: tuple[int, ...], shape) -> str | None:
    """A derived place-name for a cell region, when the board's stored shape supports one
    (same row, same column, or a contiguous diagonal). Anything else — and any 1-D board —
    stays a plain cell list; geometry is read from data, never assumed."""
    if not shape or shape[0] < 2 or shape[1] < 2 or len(g) < 2:
        return None
    C = shape[1]
    pts = sorted((i // C, i % C) for i in g)
    dr = [b[0] - a[0] for a, b in zip(pts, pts[1:])]
    dc = [b[1] - a[1] for a, b in zip(pts, pts[1:])]
    if all(d == 0 for d in dr) and all(d == 1 for d in dc):
        return f"row {pts[0][0]}"
    if all(d == 1 for d in dr) and all(d == 0 for d in dc):
        return f"column {pts[0][1]}"
    if all(d == 1 for d in dr) and all(d == 1 for d in dc):
        return "↘ diagonal"
    if all(d == 1 for d in dr) and all(d == -1 for d in dc):
        return "↗ diagonal"
    return None


def _used_programs(rules: list[Rule], concepts: list[Concept]) -> list[Concept]:
    """One concept per program the rules rely on, in library order — the programs the
    rule paths test, plus (transitively) every promoted program inside their bodies."""
    by_text = {}
    for c in concepts:
        by_text.setdefault(str(_strip(c.expr)), c)
    needed: list[str] = []

    def need(text):
        if text in by_text and text not in needed:
            needed.append(text)
            walk(_strip(by_text[text].expr), top=True)

    def walk(e, top=False):
        if not top and str(e) in by_text:
            need(str(e))
            return
        if isinstance(e, Named):
            walk(e.inner)
        elif isinstance(e, BinOp):
            walk(e.a)
            walk(e.b)
        elif isinstance(e, Fold):
            walk(e.body)

    for r in rules:
        for con, _ in r.path:
            need(str(_strip(con.expr)))
    order = {str(_strip(c.expr)): i for i, c in enumerate(concepts)}
    return [by_text[t] for t in sorted(needed, key=lambda t: order.get(t, 1 << 30))]


def closure_concepts(rules: list[Rule], concepts: list[Concept]) -> list[Concept]:
    """The rule tree's compositional dependency closure: every concept the rules split on, plus
    every concept composed (transitively, through a ``Named`` reuse) into them. Reachability over
    the concept graph — the rule tree is the root set, "built-from" the edge. Returns ALL threshold
    variants sharing a reachable program (``_used_programs`` keeps one per program for rendering;
    forgetting must keep every variant the tree actually tests), in library order. NB this follows
    only the COMPOSITIONAL edge — a fold's dependence on the *regions* it folds over is invisible
    here (the fold bakes the cells in), so a caller forgetting by this alone must separately pin the
    region-defining (atomic) concepts, or the group layer can starve."""
    reachable = {str(_strip(c.expr)) for c in _used_programs(rules, concepts)}
    return [c for c in concepts if str(_strip(c.expr)) in reachable]


def _tree_lines(rules: list[Rule], cond) -> list[str]:
    """Print the rule tree as a tree — each split once, leaves on their branches. The
    leaves carry root-to-leaf paths, so the tree is rebuilt from their shared prefixes.
    ``cond(concept)`` renders one split line."""
    out: list[str] = []

    def leaf(r):
        if not r.verdict:                                # a pre-verdict DB: values only
            return f"value {r.avg:.2f}"
        if sum(r.mix) <= 0:                              # a pre-mix DB: label only
            return f"[{r.verdict:<4}] n={r.n:<6} avg={r.avg:.2f}"
        if sum(x > 1e-9 for x in r.mix) == 1:            # one outcome — a genuine theorem
            return f"[{r.verdict:<4}] n={r.n:<6} avg={r.avg:.2f}  (pure)"
        l, d, w = (round(100 * x) for x in r.mix)        # concepts can't separate these — no
        return (f"[mixed] n={r.n:<6} avg={r.avg:.2f}  ·  "   # verdict claimed; the proof settles them
                f"L/D/W {l}/{d}/{w}%")

    def rec(rs, depth, pre):
        con = rs[0].path[depth][0]
        for line in cond(con):
            out.append(f"{pre}{line}")
        yes = [r for r in rs if r.path[depth][1]]
        no = [r for r in rs if not r.path[depth][1]]
        for tag, sub, last in (("yes", yes, False), ("no", no, True)):
            branch = f"{pre}{'└─' if last else '├─'} {tag}"
            if len(sub) == 1 and len(sub[0].path) == depth + 1:
                out.append(f"{branch:<{len(pre) + 6}} → {leaf(sub[0])}")
            else:
                out.append(branch)
                rec(sub, depth + 1, pre + ("   " if last else "│  "))

    if not rules:
        return ["(no rules — no concept explained the data)"]
    if len(rules) == 1 and not rules[0].path:
        return [f"→ {leaf(rules[0])}   (no split pays)"]
    rec(list(rules), 0, "")
    return out


def render(res: InventionResult, label: str = "", expand: bool = False) -> str:
    """Human-readable report of an invention run. By default every program prints one
    floor deep under a K-handle, rules print as one tree, and the KEY defines each floor
    in the floor below's names — with derived ⟺ readings throughout. ``expand=True``
    prints every formula fully spelled out instead."""
    out: list[str] = []
    out.append("")
    out.append(f"══ CONCEPT INVENTION{(' — ' + label.upper()) if label else ''} ══"
               f"   ({res.n_boards} boards · baseline {res.baseline_bits:,.0f} bits to explain)")
    if not res.rounds:
        out.append("  (not enough data to invent from)")
        return "\n".join(out)
    names = {} if expand else _handles(res.concepts)
    defined: set = set()                                # handles whose definition is printed
    toks = res.tokens

    def show(c: Concept) -> str:
        if expand:
            return str(c)
        return f"{_pretty(c.expr, names)} {c.op} {c.const}"

    def define(c: Concept) -> str:
        # the definition site expands its own name one floor (never to raw cells)
        key = str(_strip(c.expr))
        below = {k: v for k, v in names.items() if k != key}
        return _pretty(_strip(c.expr), below)

    out.append("")
    for r in res.rounds:
        if r.kept:
            out.append(f"ROUND {r.number}  ✓ pays — saved {r.data_saved:,.0f} bits  vs  {r.cost:,.0f} cost   "
                       f"({len(r.new_concepts)} concept{'s' if len(r.new_concepts) != 1 else ''} invented)")
            for c in r.new_concepts:
                if expand:
                    out.append(f"        + {c}")
                else:
                    h = names[str(_strip(c.expr))]
                    if h in defined:                    # same program, new threshold
                        out.append(f"        + {h} {c.op} {c.const}")
                    else:
                        defined.add(h)
                        out.append(f"        + {h} {c.op} {c.const}      {h} = {define(c)}")
                m = meaning(c, toks, names=None if expand else names)
                if m:
                    out.append(f"          ⟺ {m}")
        else:
            out.append(f"ROUND {r.number}  ✗ stop — saved {r.data_saved:,.0f} bits  vs  {r.cost:,.0f} cost   "
                       f"(nothing new pays for itself)")
    out.append("")
    out.append(f"→ stopped after round {res.stopped_after};  {len(res.concepts)} concept(s) kept.")
    out.append("")
    if expand:
        out.append("RULES it builds from the invented concepts:")
        for rule in sorted(res.rules, key=lambda r: -r.avg):
            out.append(f"   [{rule.verdict:<4}] n={rule.n:<5} avg={rule.avg:.2f}   {rule.render()}")
    else:
        out.append("RULES — one tree, each split shown once:")

        def cond(con):
            lines = [show(con)]
            m = meaning(con, toks, brief=True, names=names)
            if m:
                lines[0] += f"   ⟺ {m}"
            return lines

        out.extend("   " + l for l in _tree_lines(res.rules, cond))
    out.extend(_key_lines(res.rules, res.concepts, names, toks,
                          shape=res.shape, expand=expand))
    return "\n".join(out)


def _key_lines(rules, concepts, names, toks, shape=None, expand=False) -> list[str]:
    """The KEY: the discovered regions (named, with derived geometry where the board's
    shape supports it), then one definition per program the rules use — each written one
    floor deep, with its derived readings."""
    used = _used_programs(rules, concepts)
    if not used:
        return []
    out = [""]
    out.append("KEY — every name above, derived from the program (never hand-labeled):")
    regions = _fold_groups(used)
    if regions:
        rnames = {g: f"g{str(i + 1).translate(_SUB)}" for i, g in enumerate(regions)}
        out.append("   groups — discovered as cell-sets"
                   + ("; geometry derived from the board's shape:" if shape else ":"))
        for g, nm in rnames.items():
            geo = _region_geometry(g, shape)
            out.append(f"     {nm} = ({_cells_str(g)})" + (f"   {geo}" if geo else ""))
    if expand:
        # expanded mode: the old flat key — every used concept fully spelled out
        for c in used:
            m = meaning(c, toks)
            out.append(f"   {c}")
            if m:
                out.append(f"      ⟺ {m}")
        return out
    # collect each program's thresholds actually tested by the rules
    tests: dict[str, list[Concept]] = {}
    seen_tests = set()
    for r in rules:
        for con, _ in r.path:
            key = str(_strip(con.expr))
            if (key, con.op, con.const) not in seen_tests:
                seen_tests.add((key, con.op, con.const))
                tests.setdefault(key, []).append(con)
    for c in used:
        key = str(_strip(c.expr))
        below = {k: v for k, v in names.items() if k != key}
        out.append(f"   {names[key]} = {_pretty(_strip(c.expr), below)}")
        for t in tests.get(key, [c]):
            m = meaning(t, toks, names=below)
            if m:
                out.append(f"        [{t.op} {t.const}]  ⟺ {m}")
    return out
