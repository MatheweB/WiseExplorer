"""A persistent, growing library of invented concepts, used as a live value signal.

Holds the concepts the synthesiser has discovered (the nim-sum, the lines, threats…) and the
rule tree over them, and turns *any* board into a value through that tree — including boards
never seen in training, which is the whole point: a discovered rule like the nim-sum values
every position, not just the visited ones.

It persists the discovered programs (not their board-dependent masks) to SQLite, so the
separate, read-only worker processes that generate self-play can reload exactly the same
concepts. It grows **incrementally** via :func:`synthesis.grow_once` — re-evaluating what it
already kept and only searching when the residual rises — so it never re-creates the wheel.
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
        self.resid_frac: Optional[float] = None  # fraction of data left unexplained (grow trigger)
        if not read_only:
            self.conn.executescript(_SCHEMA)
        self._load()

    # ── growth (main process, on the training cadence) ──────────────────────────
    def refresh(self, B: np.ndarray, V: np.ndarray, M: np.ndarray,
                max_size=None, cap="auto") -> int:
        """Incrementally grow the library from the current transitions and persist it. Cheap
        when the library already explains the data; searches only when the residual rises.
        ``max_size`` bounds the search (left at its default it follows the board-width
        heuristic; callers/tests can pass a tighter bound)."""
        self.kept, self.rules, self.resid_frac = S.grow_once(
            self.kept, B, V, M, self.resid_frac, max_size, cap)
        if not self.read_only:
            self.save()
        return len(self.kept)

    # ── the signal (everywhere, including read-only workers) ────────────────────
    def value_for(self, board: np.ndarray, m: Optional[int] = None) -> Optional[float]:
        """The library's value for a board: the leaf its rule-path lands in. ``None`` when the
        library is empty or no rule matches. ``m`` is the just-played token (used only by
        move-relative concepts; cell-only ones ignore it)."""
        for r in self.rules:
            if all(c.holds(board, m) == sense for c, sense in r.path):
                return r.avg
        return None

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

    def _load(self) -> None:
        try:
            rows = self.conn.execute(
                "SELECT id, expr_json, op, const, size FROM concepts ORDER BY id").fetchall()
        except sqlite3.OperationalError:
            return                                  # tables not created yet (fresh / racing worker)
        concepts = {}
        for cid, ej, op, const, size in rows:
            try:                                    # fail-safe: skip a bad/old row, never crash
                concepts[cid] = S.Concept(S.expr_from_dict(json.loads(ej)), op, int(const),
                                          np.empty(0, dtype=np.int64), int(size))
            except Exception:
                continue
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
