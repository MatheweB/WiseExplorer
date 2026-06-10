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
from typing import List, Optional

import numpy as np

from wise_explorer import synthesis as S

_SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts (
    id INTEGER PRIMARY KEY, expr_json TEXT NOT NULL, op TEXT NOT NULL,
    const INTEGER NOT NULL, size INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS concept_rules (
    id INTEGER PRIMARY KEY, path_json TEXT NOT NULL, avg REAL NOT NULL
);
"""


class ConceptLibrary:
    """The discovered concepts + their rule tree, as a persisted board → value map."""

    def __init__(self, conn: sqlite3.Connection, read_only: bool = False) -> None:
        self.conn = conn
        self.read_only = read_only
        self.kept: List[S.Concept] = []     # the concepts discovered so far (carried forward)
        self.rules: List[S.Rule] = []       # the value model: rule paths with leaf values
        if not read_only:
            self.conn.executescript(_SCHEMA)
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
        if len(B) >= 8:
            res = S.invent_from_boards(B, V, M, max_size=max_size, cap=cap, seed=self.kept)
            self.kept, self.rules = res.concepts, res.rules
        self.save()
        return len(self.kept)

    def summary(self) -> str:
        """A terse 'what training discovered' line: the rules verbatim, then a key giving
        each fold's derived plain-English reading alongside its formula."""
        if not self.kept:
            return "No concepts discovered yet (the data may not support any)."
        lines = [f"Discovered {len(self.kept)} concept{'s' if len(self.kept) != 1 else ''}, "
                 f"{len(self.rules)} rule{'s' if len(self.rules) != 1 else ''}:"]
        for r in self.rules:
            lines.append(f"  {r.render()}  →  {r.avg:.2f}")
        used, seen = [], set()
        for r in self.rules:
            for con, _ in r.path:
                key = str(con)
                if key not in seen:
                    seen.add(key); used.append(con)
        gl = S._groups_line(used)
        if gl:
            lines.append(f"  {gl}")
        for con in used:
            gloss = S.meaning(con)
            if gloss:
                lines.append(f"  where {con}  ⟺  {gloss}")
        return "\n".join(lines)

    # ── the signal (everywhere, including read-only workers) ────────────────────
    def value_for(self, board: np.ndarray, m: Optional[int] = None) -> Optional[float]:
        """The library's value for a board: the leaf its rule-path lands in. ``None`` when the
        library is empty or no rule matches. ``m`` is the just-played token (used only by
        move-relative concepts; cell-only ones ignore it)."""
        v = self.values_for(np.asarray(board).reshape(1, -1),
                            None if m is None else np.array([m]))[0]
        return None if np.isnan(v) else float(v)

    def values_for(self, B: np.ndarray, M: Optional[np.ndarray] = None) -> np.ndarray:
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
            "INSERT INTO concept_rules (id, path_json, avg) VALUES (?,?,?)",
            [(ri, json.dumps([[idx[id(con)], bool(s)] for con, s in r.path if id(con) in idx]),
              float(r.avg)) for ri, r in enumerate(self.rules)],
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
        for _rid, pj, avg in self.conn.execute(
                "SELECT id, path_json, avg FROM concept_rules ORDER BY id").fetchall():
            try:
                path = [(concepts[cid], bool(s)) for cid, s in json.loads(pj) if cid in concepts]
            except Exception:
                continue
            rules.append(S.Rule(path, "", 0, float(avg), 0.0))
        self.rules = rules
