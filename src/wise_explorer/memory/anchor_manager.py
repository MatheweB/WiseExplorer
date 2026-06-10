"""
Anchor clustering manager for game memory.

Anchors group statistically similar units together, allowing pooled
statistics for faster convergence via Bayes factor clustering.

Fully decoupled from GameMemory — receives data as parameters and
returns results. The caller (GameMemory) orchestrates fetch/apply.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable

from wise_explorer.core.bayes import compatible, similarity
from wise_explorer.core.types import Counts


@dataclass
class Anchor:
    """Lightweight anchor data holder."""
    counts: Counts
    repr_key: str

    @property
    def total(self) -> float:
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
    """Manages anchor clustering with O(1) incremental updates.

    Takes conn, main_table, and read_only directly.
    Methods receive pre-fetched data and return results for the caller to apply.
    """

    def __init__(self, conn: sqlite3.Connection, main_table: str, read_only: bool = False):
        self._conn = conn
        self._main_table = main_table
        self._read_only = read_only
        self._dirty = True

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_details(self) -> list[dict]:
        """Get detailed information about all anchors."""
        rows = self._conn.execute(
            f"""SELECT a.anchor_id, a.repr_key, a.wins, a.ties, a.losses,
                       COUNT(t.anchor_id)
                FROM anchors a
                LEFT JOIN {self._main_table} t ON a.anchor_id = t.anchor_id
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

    def needs_initialization(self) -> bool:
        """Check if anchors need to be built from existing data.

        Returns True at most once (on first call when data exists but
        no anchors do). Caller is responsible for triggering rebuild.
        """
        if not self._dirty:
            return False
        self._dirty = False

        if self._read_only:
            return False

        anchor_count = self._conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
        if anchor_count == 0:
            has_data = self._conn.execute(
                f"SELECT COUNT(*) FROM {self._main_table}"
            ).fetchone()[0] > 0
            return has_data
        return False

    # -------------------------------------------------------------------------
    # Incremental Update
    # -------------------------------------------------------------------------

    def update(
        self,
        changed_stats: dict[Any, Counts],
        deltas: dict[Any, Counts],
        existing_aids: dict[Any, int | None],
        key_to_repr: Callable[[Any], str],
        cur: sqlite3.Cursor,
    ) -> tuple[dict[Any, int], int]:
        """
        Update anchor assignments and stats incrementally.

        Args:
            changed_stats: key -> current (w,t,l) for each changed unit
            deltas: key -> delta (dw,dt,dl) from this commit
            existing_aids: key -> current anchor_id (or None)
            key_to_repr: converts a key to a display string
            cur: database cursor for writes

        Returns:
            (assignments, swap_count) where assignments maps key -> new_aid
            for units that were reassigned. Caller writes these to the main table.
        """
        if not changed_stats:
            return {}, 0

        anchors = self._load_anchors(cur)
        max_id = max(anchors.keys(), default=-1)

        assignments: dict[Any, int] = {}
        swaps = 0
        for key, counts in changed_stats.items():
            old_aid = existing_aids.get(key)
            delta = deltas.get(key, (0.0, 0.0, 0.0))
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
                repr_key = key_to_repr(key)
                cur.execute("INSERT INTO anchors VALUES (?,?,0.0,0.0,0.0)", (new_aid, repr_key))
                anchors[new_aid] = Anchor((0.0, 0.0, 0.0), repr_key)

            # Update membership
            if old_aid is None:
                # New unit — first assignment, not a swap
                self._update_anchor_stats(new_aid, counts, anchors, cur)
            else:
                # SWAP: unit moved from old_aid to new_aid
                # This is actual learning — beliefs about this move changed
                swaps += 1
                self._update_anchor_stats(old_aid, _neg_counts(old_stats), anchors, cur)
                self._update_anchor_stats(new_aid, counts, anchors, cur)

            assignments[key] = new_aid

        # Cleanup empty anchors
        for aid in [a for a, anc in anchors.items() if anc.total <= 0]:
            cur.execute("DELETE FROM anchors WHERE anchor_id=?", (aid,))
            del anchors[aid]

        return assignments, swaps

    def _update_anchor_stats(self, aid: int, delta: Counts, anchors: dict[int, Anchor], cur: sqlite3.Cursor) -> None:
        """Update anchor stats in DB and cache."""
        if delta == (0, 0, 0):
            return
        cur.execute(
            "UPDATE anchors SET wins=wins+?, ties=ties+?, losses=losses+? WHERE anchor_id=?",
            (*delta, aid)
        )
        if aid in anchors:
            anchors[aid].add(delta)

    def _load_anchors(self, cur: sqlite3.Cursor) -> dict[int, Anchor]:
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
        anchors: dict[int, Anchor],
        exclude_aid: int | None = None,
        exclude_stats: Counts | None = None
    ) -> int | None:
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
        if self._read_only:
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

    def _merge_anchors(self, aid1: int, aid2: int, anchors: dict[int, Anchor], cur: sqlite3.Cursor) -> None:
        """Merge two anchors, keeping the larger one."""
        survivor, absorbed = (aid1, aid2) if anchors[aid1].total >= anchors[aid2].total else (aid2, aid1)

        cur.execute(
            f"UPDATE {self._main_table} SET anchor_id=? WHERE anchor_id=?",
            (survivor, absorbed)
        )
        self._update_anchor_stats(survivor, anchors[absorbed].counts, anchors, cur)
        cur.execute("DELETE FROM anchors WHERE anchor_id=?", (absorbed,))
        del anchors[absorbed]

    # -------------------------------------------------------------------------
    # Full Rebuild
    # -------------------------------------------------------------------------

    def rebuild(
        self,
        units: list[tuple[Any, Counts]],
        key_to_repr: Callable[[Any], str],
        cur: sqlite3.Cursor,
    ) -> tuple[int, dict[Any, int]]:
        """
        Full rebuild of anchor clustering.

        Args:
            units: list of (key, counts) tuples to cluster
            key_to_repr: converts a key to a display string
            cur: database cursor (caller manages transaction)

        Returns:
            (num_anchors, membership) where membership maps key -> anchor_index.
            Caller writes membership to the main table.
        """
        def entropy(counts: Counts) -> float:
            total = sum(counts)
            return -sum((c / total) * math.log(c / total + 1e-12) for c in counts)

        units.sort(key=lambda u: entropy(u[1]))
        anchor_list, membership = self._cluster_units(units, key_to_repr)

        cur.execute("DELETE FROM anchors")
        cur.executemany(
            "INSERT INTO anchors VALUES (?,?,?,?,?)",
            [(i, a.repr_key, *a.counts) for i, a in enumerate(anchor_list)]
        )

        return len(anchor_list), membership

    def _cluster_units(
        self,
        units: list[tuple[Any, Counts]],
        key_to_repr: Callable[[Any], str],
    ) -> tuple[list[Anchor], dict[Any, int]]:
        """Cluster units into anchors."""
        anchors: list[Anchor] = []
        membership: dict[Any, int] = {}

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
                anchors.append(Anchor(counts, key_to_repr(key)))

        return anchors, membership
