"""A persistent, growing library of invented concepts, used as a live value signal.

Holds the concepts the synthesiser has discovered (the nim-sum, the lines, threats…) and the
rule tree over them, and turns *any* board into a value through that tree — including boards
never seen in training, which is the whole point: a discovered rule like the nim-sum values
every position, not just the visited ones.

It persists the discovered programs (not their board-dependent masks) to SQLite, so the
separate, read-only worker processes that generate self-play can reload exactly the same
concepts. During training it grows per wave via :func:`synthesis.grow` — folding only the
transitions the wave touched into a live :class:`synthesis.BoardTable`, refitting what it
already kept, and searching only when the library stops explaining the data — and at the end
of training :meth:`rebuild` re-derives everything over the converged values. An empty library
is inert (every value is ``None``), so its selection signal has no effect until concepts have
actually been discovered.
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
        self.floor: Optional[float] = None  # lowest unexplained fraction achieved (search trigger)
        self.table = S.BoardTable()         # the live board table, grown a wave at a time
        self._searched_n = None             # table size at the last search (bounds the search cadence)
        if not read_only:
            self.conn.executescript(_SCHEMA)
        self._load()

    # ── per-wave growth (main process, local to the wave) ───────────────────────
    def grow(self, wave_keys, boards, trans_scores, max_size=None, cap="auto") -> int:
        """Grow from only the transitions this wave touched: fold them into the board table,
        refit the library, and search only when :func:`synthesis.grow` says the library
        stopped explaining the data. Never rescans history, so per-wave cost tracks what
        changed, not how many games have run. :meth:`rebuild` still does the authoritative
        end-of-training pass."""
        if self.read_only or not wave_keys:
            return len(self.kept)
        self.table.update(wave_keys, boards, trans_scores)
        if len(self.table) < 8:
            return len(self.kept)            # too little data yet — leave the loaded model alone
        self.kept, self.rules, self.floor, self._searched_n = S.grow(
            self.table, self.kept, self.floor, self._searched_n, max_size, cap)
        self.save()
        return len(self.kept)

    # ── authoritative full rebuild (end of training, over converged values) ─────
    def rebuild(self, B: np.ndarray, V: np.ndarray, M: np.ndarray,
                max_size=None, cap="auto") -> int:
        """Re-run discovery over the converged table. Run once at the end of training so the
        persisted rules reflect the converged values — the per-wave path is a fast live
        approximation; this is the considered fit. The search is seeded with the current
        library: knowledge carries forward (a transferred concept survives even when this
        game's data alone couldn't re-derive it), and the fit decides what the rules actually
        use. When the table outgrows the per-wave budget (``table.cap``), discovery runs over
        a uniform sample of it instead: a concept is a program, visible in any fair sample."""
        if self.read_only:
            return len(self.kept)
        if len(B) > self.table.cap:
            keep = np.random.default_rng(0).choice(len(B), self.table.cap, replace=False)
            B, V, M = B[keep], V[keep], M[keep]
        if len(B) >= 8:
            res = S.invent_from_boards(B, V, M, max_size=max_size, cap=cap, seed=self.kept)
            self.kept, self.rules = res.concepts, res.rules
            self.floor = (sum(r.resid for r in res.rules) / res.baseline_bits
                          if res.baseline_bits > 0 else 0.0)
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
        the *programs* in as building blocks — their masks are re-derived on the first wave
        over the local boards, and the rule tree (the *worth* of each concept) is rebuilt
        natively, so only the transferable structure crosses over. A width-free nim-sum learned
        at 4 piles thus arrives ready to explain 8 piles. Any local rules are cleared, so the
        library stays inert (``value_for`` is ``None``) until the first grow fits the seed."""
        concepts = self._concepts_from(conn)
        self.kept = [concepts[k] for k in sorted(concepts)]
        self.rules = []
        self.floor = None
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
