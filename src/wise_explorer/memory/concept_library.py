"""A persistent library of invented concepts, used as a value signal.

Holds the concepts the synthesiser has discovered (the nim-sum, the lines, threats…) and the
rule tree over them, and turns *any* board into a value through that tree — including boards
never seen in training, which is the whole point: a discovered rule like the nim-sum values
every position, not just the visited ones.

It persists the discovered programs (not their board-dependent masks) to SQLite, so the
separate, read-only worker processes that generate self-play can reload exactly the same
concepts. Discovery has one venue: :meth:`rebuild`, called at each training-cycle boundary
by the value loop (docs/value-loop.md) over the completed values — between boundaries the
library simply *is* the last considered fit. (A per-wave live refit existed and was deleted:
nothing reads the concept signal during training, and a refit over still-drifting values
could transiently collapse the tree right where the loop's healing pass would read it.)
An empty library is inert (every value is ``None``), so its selection signal has no effect
until concepts have actually been discovered.
"""
import json
import sqlite3

import numpy as np

from wise_explorer import synthesis as S

_SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts (
    id INTEGER PRIMARY KEY, expr_json TEXT NOT NULL, op TEXT NOT NULL,
    const INTEGER NOT NULL, size INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS concept_rules (
    id INTEGER PRIMARY KEY, path_json TEXT NOT NULL, avg REAL NOT NULL,
    verdict TEXT DEFAULT '', n INTEGER DEFAULT 0
);
"""


class ConceptLibrary:
    """The discovered concepts + their rule tree, as a persisted board → value map."""

    def __init__(self, conn: sqlite3.Connection, read_only: bool = False) -> None:
        self.conn = conn
        self.read_only = read_only
        self.kept: list[S.Concept] = []     # the concepts discovered so far (carried forward)
        self.rules: list[S.Rule] = []       # the value model: rule paths with leaf values
        if not read_only:
            self.conn.executescript(_SCHEMA)
            for col in ("verdict TEXT DEFAULT ''", "n INTEGER DEFAULT 0"):
                try:                                     # migrate pre-verdict DBs in place
                    self.conn.execute(f"ALTER TABLE concept_rules ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass                                 # column already there
        self._load()

    # ── discovery (the value loop's distillation beat) ──────────────────────────
    def rebuild(self, B: np.ndarray, V: np.ndarray, M: np.ndarray,
                max_size=None, cap="auto") -> int:
        """Re-run discovery over the given boards and values — called at each training-cycle
        boundary, on the loop's completed values. The search is seeded with the current
        library: knowledge carries forward (a transferred concept survives even when this
        game's data alone couldn't re-derive it), and the fit decides what the rules actually
        use. A sufficient seed self-limits — the MDL gate finds nothing left that pays.
        Beyond the data budget (:data:`synthesis.CAP`), discovery runs over a uniform sample:
        a concept is a program, visible in any fair sample."""
        if self.read_only:
            return len(self.kept)
        if len(B) > S.CAP:
            keep = np.random.default_rng(0).choice(len(B), S.CAP, replace=False)
            B, V, M = B[keep], V[keep], M[keep]
        if len(B) >= S.MIN_BOARDS:
            res = S.invent_from_boards(B, V, M, max_size=max_size, cap=cap, seed=self.kept)
            self.kept, self.rules = res.concepts, res.rules
            self._forget()
        self.save()
        return len(self.kept)

    def _forget(self) -> None:
        """Drop orphan composites — concepts the value model neither tests nor is built from —
        while NEVER dropping an atomic (region-defining) concept. Keep = the rule tree's
        compositional closure (the splits + their ``Named`` build-chain) ∪ every atomic concept.
        Reachability GC over the concept graph that also pins the foundation: regions (hence the
        group/fork layer, seeded from atomic supports) can't be starved, so unlike the old
        drop-if-not-a-split this can't freeze discovery — that one freed the build chain AND the
        regions its splits stood on. An orphan changes no value and is re-derived if later needed.
        Guarded by a non-empty tree, so a freshly-seeded library (rules=[]) is never swept."""
        if not self.rules:
            return
        live = {id(c) for c in S.closure_concepts(self.rules, self.kept)}
        self.kept = [c for c in self.kept if id(c) in live or S._is_atomic(c.expr)]

    def summary(self, expand: bool = False) -> str:
        """What training discovered: the rule tree (each split shown once, with derived
        ⟺ readings) and a KEY defining every named program one floor deep. ``expand=True``
        prints flat rules with fully-spelled-out formulas instead."""
        if not self.kept:
            return "No concepts discovered yet (the data may not support any)."
        lines = [f"Discovered {len(self.kept)} concept{'s' if len(self.kept) != 1 else ''}, "
                 f"{len(self.rules)} rule{'s' if len(self.rules) != 1 else ''}:"]
        toks, shape = self._alphabet()
        if expand:
            for r in self.rules:
                lines.append(f"  {r.render()}  →  {r.avg:.2f}")
        else:
            names = S._handles(self.kept)

            def cond(con):
                text = f"{S._pretty(con.expr, names)} {con.op} {con.const}"
                m = S.meaning(con, toks, brief=True, names=names)
                return [text + (f"   ⟺ {m}" if m else "")]

            rules = [r for r in self.rules]
            lines.extend("  " + l for l in S._tree_lines(rules, cond))
            lines.append("  (each leaf value is what the library lends a board that lands there)")
        lines.extend(S._key_lines(self.rules, self.kept,
                                  {} if expand else S._handles(self.kept),
                                  toks, shape=shape, expand=expand))
        return "\n".join(lines)

    def _alphabet(self):
        """The observed nonzero tokens and the native board shape, from a bounded sample
        of stored boards — what derived glosses and region geometry may honestly use."""
        try:
            rows = self.conn.execute(
                "SELECT board_data, board_rows, board_cols FROM boards LIMIT 64").fetchall()
        except Exception:
            return (), None
        if not rows:
            return (), None
        toks = sorted({int(t) for bd, _r, _c in rows
                       for t in np.frombuffer(bd, dtype=np.int8) if t != 0})
        return tuple(toks), (int(rows[0][1]), int(rows[0][2]))

    # ── the signal (everywhere, including read-only workers) ────────────────────
    def value_for(self, board: np.ndarray, m: int | None = None) -> float | None:
        """The library's value for a board: the leaf its rule-path lands in. ``None`` when the
        library is empty or no rule matches. ``m`` is the just-played token (used only by
        move-relative concepts; cell-only ones ignore it)."""
        v = self.values_for(np.asarray(board).reshape(1, -1),
                            None if m is None else np.array([m]))[0]
        return None if np.isnan(v) else float(v)

    def values_for(self, B: np.ndarray, M: np.ndarray | None = None) -> np.ndarray:
        """One value per row of ``B`` (NaN where no rule matches), via a single batched
        rule-walk. ``M`` gives each row's just-played token. This is what lets the value
        loop price thousands of never-visited boards in one pass."""
        out = np.full(len(B), np.nan)
        if not self.rules or not len(B):
            return out
        unmatched = np.ones(len(B), dtype=bool)
        for r in self.rules:
            hit = np.ones(len(B), dtype=bool)
            for con, sense in r.path:
                v = con.expr.eval(B, M)
                holds = (v == con.const) if con.op == "=" else (v > con.const)
                hit &= (holds == sense)
            out[unmatched & hit] = r.avg
            unmatched &= ~hit
        return out

    # ── persistence (programs only — masks are board-order-dependent) ───────────
    def save(self) -> None:
        idx = {id(c): i for i, c in enumerate(self.kept)}
        cur = self.conn.cursor()
        cur.execute("DELETE FROM concepts")
        cur.execute("DELETE FROM concept_rules")
        cur.executemany(
            "INSERT INTO concepts (id, expr_json, op, const, size) VALUES (?,?,?,?,?)",
            [(i, json.dumps(S.expr_to_dict(c.expr)), c.op, int(c.const), int(c.size))
             for i, c in enumerate(self.kept)],
        )
        cur.executemany(
            "INSERT INTO concept_rules (id, path_json, avg, verdict, n) VALUES (?,?,?,?,?)",
            [(ri, json.dumps([[idx[id(con)], bool(s)] for con, s in r.path if id(con) in idx]),
              float(r.avg), r.verdict, int(r.n)) for ri, r in enumerate(self.rules)],
        )
        self.conn.commit()

    @staticmethod
    def _concepts_from(conn: sqlite3.Connection) -> dict:
        """Read the concept *programs* (maskless) from a connection's ``concepts`` table → {id:
        Concept}. The program is board-order-independent and width-free where it can be (a
        BoardDomain fold), so the same rows seed a fresh library at a different scale."""
        out = {}
        try:
            rows = conn.execute(
                "SELECT id, expr_json, op, const, size FROM concepts ORDER BY id").fetchall()
        except sqlite3.OperationalError:
            return out                              # tables not created yet (fresh / racing worker)
        for cid, ej, op, const, size in rows:
            try:                                    # fail-safe: skip a bad/old row, never crash
                out[cid] = S.Concept(S.expr_from_dict(json.loads(ej)), op, int(const),
                                     np.empty(0, dtype=np.int64), int(size))
            except Exception:
                continue
        return out

    def seed_from(self, conn: sqlite3.Connection) -> int:
        """Seed this library with the concepts another game/scale already discovered. Carries
        the *programs* in as building blocks — their masks are re-derived over the local
        boards, and the rule tree (the *worth* of each concept) is rebuilt natively, so only
        the transferable structure crosses over. A width-free nim-sum learned at 4 piles thus
        arrives ready to explain 8 piles. Any local rules are cleared, so the library stays
        inert (``value_for`` is ``None``) until the first rebuild fits the seed."""
        concepts = self._concepts_from(conn)
        self.kept = [concepts[k] for k in sorted(concepts)]
        self.rules = []
        return len(self.kept)

    def _load(self) -> None:
        concepts = self._concepts_from(self.conn)
        self.kept = [concepts[k] for k in sorted(concepts)]
        rules = []
        try:
            rows = self.conn.execute(
                "SELECT path_json, avg, verdict, n FROM concept_rules ORDER BY id").fetchall()
        except sqlite3.OperationalError:                 # a pre-verdict DB opened read-only
            rows = [(pj, avg, "", 0) for pj, avg in self.conn.execute(
                "SELECT path_json, avg FROM concept_rules ORDER BY id").fetchall()]
        for pj, avg, verdict, n in rows:
            try:
                path = [(concepts[cid], bool(s)) for cid, s in json.loads(pj) if cid in concepts]
            except Exception:
                continue
            rules.append(S.Rule(path, verdict or "", int(n or 0), float(avg), 0.0))
        self.rules = rules
