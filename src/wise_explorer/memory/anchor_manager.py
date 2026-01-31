"""
Anchor clustering manager for game memory.

Anchors group statistically similar units together, allowing pooled
statistics for faster convergence via Bayes factor clustering.

Generic over key type - works with both TransitionMemory and MarkovMemory.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from wise_explorer.core.bayes import compatible, similarity

if TYPE_CHECKING:
    from wise_explorer.memory.game_memory import GameMemory

Counts = Tuple[int, int, int]


@dataclass
class Anchor:
    """Lightweight anchor data holder."""
    counts: Counts
    repr_key: str

    @property
    def total(self) -> int:
        return sum(self.counts)

    def add(self, delta: Counts) -> None:
        w, t, l = self.counts
        dw, dt, dl = delta
        self.counts = (w + dw, t + dt, l + dl)

    def without(self, other: Counts) -> Counts:
        """Return counts with other subtracted (for self-exclusion checks)."""
        return (self.counts[0] - other[0], self.counts[1] - other[1], self.counts[2] - other[2])


def _sub_counts(a: Counts, b: Counts) -> Counts:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _neg_counts(c: Counts) -> Counts:
    return (-c[0], -c[1], -c[2])


class AnchorManager:
    """Manages anchor clustering with O(1) incremental updates."""

    def __init__(self, memory: "GameMemory"):
        self._mem = memory
        self._dirty = True

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._mem.conn

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_details(self) -> List[dict]:
        """Get detailed information about all anchors."""
        main_table = self._mem.main_table
        rows = self._conn.execute(
            f"""SELECT a.anchor_id, a.repr_key, a.wins, a.ties, a.losses,
                       COUNT(t.anchor_id)
                FROM anchors a
                LEFT JOIN {main_table} t ON a.anchor_id = t.anchor_id
                GROUP BY a.anchor_id
                ORDER BY a.wins + a.ties + a.losses DESC"""
        ).fetchall()

        return [{
            "anchor_id": aid,
            "repr_key": repr_key,
            "wins": w, "ties": t, "losses": l,
            "total": (total := w + t + l),
            "members": members,
            "distribution": (w / total, t / total, l / total) if total else (0, 0, 0),
        } for aid, repr_key, w, t, l, members in rows]

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def ensure_initialized(self) -> None:
        """Ensure anchors are initialized if data exists."""
        if not self._dirty:
            return
        self._dirty = False

        if self._mem.read_only:
            return

        anchor_count = self._conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
        if anchor_count == 0:
            has_data = self._conn.execute(
                f"SELECT COUNT(*) FROM {self._mem.main_table}"
            ).fetchone()[0] > 0
            if has_data:
                self.rebuild()

    # -------------------------------------------------------------------------
    # Incremental Update
    # -------------------------------------------------------------------------

    def update(self, keys: List, deltas: Dict, cur: sqlite3.Cursor) -> None:
        """Update anchor assignments and stats incrementally."""
        if not keys:
            return

        self.ensure_initialized()

        # Gather current stats for changed keys
        changed = {}
        for key in keys:
            stats = self._mem._get_stats_by_key(key)
            if stats.total > 0:
                changed[key] = {
                    "counts": stats.as_tuple(),
                    "delta": deltas.get(key, (0, 0, 0)),
                }

        if not changed:
            return

        anchors = self._load_anchors(cur)
        max_id = max(anchors.keys(), default=-1)
        existing = self._mem._batch_get_anchor_ids(list(changed.keys()), cur)

        for key, data in changed.items():
            old_aid = existing.get(key)
            counts, delta = data["counts"], data["delta"]
            old_stats = _sub_counts(counts, delta)

            # Check if still compatible with current anchor (excluding self)
            if old_aid is not None and old_aid in anchors:
                anchor_without_self = anchors[old_aid].without(old_stats)
                if sum(anchor_without_self) > 0 and compatible(counts, anchor_without_self):
                    self._update_anchor_stats(old_aid, delta, anchors, cur)
                    continue

            # Find or create new anchor
            new_aid = self._find_compatible_anchor(counts, anchors, old_aid, old_stats)
            if new_aid is None:
                max_id += 1
                new_aid = max_id
                repr_key = self._mem._key_to_repr(key)
                cur.execute("INSERT INTO anchors VALUES (?,?,0,0,0)", (new_aid, repr_key))
                anchors[new_aid] = Anchor((0, 0, 0), repr_key)

            # Update membership
            if old_aid is None:
                self._update_anchor_stats(new_aid, counts, anchors, cur)
            else:
                self._update_anchor_stats(old_aid, _neg_counts(old_stats), anchors, cur)
                self._update_anchor_stats(new_aid, counts, anchors, cur)

            self._mem._set_anchor_id(key, new_aid, cur)

        # Cleanup empty anchors
        for aid in [a for a, anc in anchors.items() if anc.total <= 0]:
            cur.execute("DELETE FROM anchors WHERE anchor_id=?", (aid,))
            del anchors[aid]

    def _update_anchor_stats(self, aid: int, delta: Counts, anchors: Dict[int, Anchor], cur: sqlite3.Cursor) -> None:
        """Update anchor stats in DB and cache."""
        if delta == (0, 0, 0):
            return
        cur.execute(
            "UPDATE anchors SET wins=wins+?, ties=ties+?, losses=losses+? WHERE anchor_id=?",
            (*delta, aid)
        )
        if aid in anchors:
            anchors[aid].add(delta)

    def _load_anchors(self, cur: sqlite3.Cursor) -> Dict[int, Anchor]:
        """Load all anchors from database."""
        return {
            aid: Anchor((w, t, l), repr_key)
            for aid, repr_key, w, t, l in cur.execute(
                "SELECT anchor_id, repr_key, wins, ties, losses FROM anchors"
            )
        }

    def _find_compatible_anchor(
        self,
        counts: Counts,
        anchors: Dict[int, Anchor],
        exclude_aid: Optional[int] = None,
        exclude_stats: Optional[Counts] = None
    ) -> Optional[int]:
        """Find most similar compatible anchor."""
        best_aid, best_sim = None, -1.0

        for aid, anchor in anchors.items():
            if anchor.total == 0:
                continue

            comparison = anchor.without(exclude_stats) if aid == exclude_aid and exclude_stats else anchor.counts
            if sum(comparison) <= 0:
                continue

            if compatible(counts, comparison):
                sim = similarity(counts, comparison)
                if sim > best_sim:
                    best_sim, best_aid = sim, aid

        return best_aid

    # -------------------------------------------------------------------------
    # Consolidation
    # -------------------------------------------------------------------------

    def consolidate(self) -> int:
        """Merge anchors that have become statistically compatible."""
        if self._mem.read_only:
            return 0

        self._conn.commit()
        cur = self._conn.cursor()
        cur.execute("BEGIN IMMEDIATE")

        try:
            anchors = self._load_anchors(cur)
            initial_count = len(anchors)

            merged = True
            while merged:
                merged = False
                active = [aid for aid, a in anchors.items() if a.total > 0]

                for i, aid1 in enumerate(active):
                    if aid1 not in anchors:
                        continue
                    for aid2 in active[i + 1:]:
                        if aid2 not in anchors:
                            continue
                        if compatible(anchors[aid1].counts, anchors[aid2].counts):
                            self._merge_anchors(aid1, aid2, anchors, cur)
                            merged = True
                            break
                    if merged:
                        break

            cur.execute("COMMIT")
            return initial_count - len(anchors)
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def _merge_anchors(self, aid1: int, aid2: int, anchors: Dict[int, Anchor], cur: sqlite3.Cursor) -> None:
        """Merge two anchors, keeping the larger one."""
        survivor, absorbed = (aid1, aid2) if anchors[aid1].total >= anchors[aid2].total else (aid2, aid1)

        cur.execute(
            f"UPDATE {self._mem.main_table} SET anchor_id=? WHERE anchor_id=?",
            (survivor, absorbed)
        )
        self._update_anchor_stats(survivor, anchors[absorbed].counts, anchors, cur)
        cur.execute("DELETE FROM anchors WHERE anchor_id=?", (absorbed,))
        del anchors[absorbed]

    # -------------------------------------------------------------------------
    # Full Rebuild
    # -------------------------------------------------------------------------

    def rebuild(self) -> int:
        """Full rebuild of anchor clustering."""
        if self._mem.read_only:
            raise RuntimeError("Cannot rebuild anchors in read-only mode")

        units = self._mem._collect_units()
        if not units:
            return 0

        def entropy(counts: Counts) -> float:
            total = sum(counts)
            return -sum((c / total) * math.log(c / total + 1e-12) for c in counts)

        units.sort(key=lambda u: entropy(u[1]))
        anchors, membership = self._cluster_units(units)

        self._conn.commit()
        cur = self._conn.cursor()
        cur.execute("BEGIN IMMEDIATE")

        try:
            self._write_anchors(anchors, membership, cur)
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

        return len(anchors)

    def _cluster_units(self, units: List[Tuple]) -> Tuple[List[Anchor], Dict]:
        """Cluster units into anchors."""
        anchors: List[Anchor] = []
        membership = {}

        for key, counts in units:
            best_idx = None
            best_sim = -1.0
            for i, anchor in enumerate(anchors):
                if compatible(counts, anchor.counts):
                    sim = similarity(counts, anchor.counts)
                    if sim > best_sim:
                        best_sim, best_idx = sim, i

            if best_idx is not None:
                anchors[best_idx].add(counts)
                membership[key] = best_idx
            else:
                membership[key] = len(anchors)
                anchors.append(Anchor(counts, self._mem._key_to_repr(key)))

        return anchors, membership

    def _write_anchors(self, anchors: List[Anchor], membership: Dict, cur: sqlite3.Cursor) -> None:
        """Write anchors to database and update main table references."""
        cur.execute("DELETE FROM anchors")
        cur.executemany(
            "INSERT INTO anchors VALUES (?,?,?,?,?)",
            [(i, a.repr_key, *a.counts) for i, a in enumerate(anchors)]
        )
        self._mem._write_anchor_ids(membership, cur)