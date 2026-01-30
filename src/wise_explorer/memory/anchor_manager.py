"""
Anchor clustering manager for game memory.

Anchors group statistically similar moves together, allowing pooled
statistics for faster convergence via Bayes factor clustering.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from wise_explorer.core.types import Stats
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
    
    @property
    def distribution(self) -> Tuple[float, float, float]:
        w, t, l = self.counts
        total = self.total
        return (w/total, t/total, l/total) if t > 0 else (0.0, 0.0, 0.0)
    
    def add(self, delta: Counts) -> None:
        w, t, l = self.counts
        dw, dt, dl = delta
        self.counts = (w + dw, t + dt, l + dl)


    def subtract(self, delta: Counts) -> None:
        w, t, l = self.counts
        dw, dt, dl = delta
        self.counts = (w - dw, t - dt, l - dl)


    def without(self, other: Counts) -> Counts:
        """Return counts with other subtracted (for self-exclusion checks)."""
        w, t, l = self.counts
        ow, ot, ol = other
        return (w - ow, t - ot, l - ol)


def _sub_counts(a: Counts, b: Counts) -> Counts:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


class AnchorManager:
    """Manages anchor clustering with O(1) incremental updates."""

    def __init__(self, memory: "GameMemory"):
        self._mem = memory
        self._dirty = True

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._mem.conn

    @property
    def _markov(self) -> bool:
        return self._mem.markov

    @property
    def _read_only(self) -> bool:
        return self._mem.read_only

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_anchor_stats(self, scoring_key: str) -> Stats:
        """Get pooled statistics from the anchor cluster."""
        row = self._conn.execute(
            """SELECT a.wins, a.ties, a.losses 
                FROM scoring_anchors sa 
                JOIN anchors a ON sa.anchor_id = a.anchor_id 
                WHERE sa.scoring_key = ?""",
            (scoring_key,)
        ).fetchone()
        return Stats(*row) if row else self._mem.get_unit_stats(scoring_key)

    def get_anchor_id(self, scoring_key: str) -> Optional[int]:
        """Get the anchor ID for a scoring key."""
        row = self._conn.execute(
            "SELECT anchor_id FROM scoring_anchors WHERE scoring_key=?",
            (scoring_key,)
        ).fetchone()
        return row[0] if row else None

    def get_details(self) -> List[dict]:
        """Get detailed information about all anchors."""
        rows = self._conn.execute(
            """SELECT a.anchor_id, a.repr_key, a.wins, a.ties, a.losses, 
                    COUNT(sa.scoring_key)
                FROM anchors a 
                LEFT JOIN scoring_anchors sa ON a.anchor_id = sa.anchor_id 
                GROUP BY a.anchor_id 
                ORDER BY a.wins + a.ties + a.losses DESC"""
        ).fetchall()
        
        results = []
        for aid, repr_key, w, t, l, members in rows:
            total = w + t + l
            results.append({
                "anchor_id": aid,
                "repr_key": repr_key,
                "wins": w, "ties": t, "losses": l,
                "total": total,
                "members": members,
                "distribution": (w/total, t/total, l/total) if total else (0, 0, 0),
                "markov_mode": self._markov,
            })
        return results

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def ensure_initialized(self) -> None:
        """Ensure anchors are initialized if data exists."""
        if not self._dirty:
            return
        self._dirty = False
        
        if self._read_only:
            return
        
        anchor_count = self._conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
        if anchor_count == 0:
            table = "state_values" if self._markov else "transitions"
            has_data = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] > 0
            if has_data:
                self.rebuild()

    # -------------------------------------------------------------------------
    # Incremental Update
    # -------------------------------------------------------------------------

    def update(self, scoring_keys: List[str], deltas: Dict[str, Counts], cur: sqlite3.Cursor) -> None:
        """
        Update anchor assignments and stats incrementally.
        
        Caller is responsible for transaction management.
        """
        if not scoring_keys:
            return
        
        self.ensure_initialized()
        
        # Gather current stats for changed keys
        changed = {}
        for key in scoring_keys:
            stats = self._mem.get_unit_stats(key)
            if stats.total > 0:
                changed[key] = {
                    "counts": tuple(stats),
                    "delta": deltas.get(key, (0, 0, 0)),
                }
        
        if not changed:
            return

        anchors = self._load_anchors(cur)
        max_id = max(anchors.keys(), default=-1)

        # Batch fetch existing assignments
        placeholders = ','.join('?' * len(changed))
        existing = dict(cur.execute(
            f"SELECT scoring_key, anchor_id FROM scoring_anchors WHERE scoring_key IN ({placeholders})",
            list(changed.keys())
        ).fetchall())

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
                cur.execute("INSERT INTO anchors VALUES (?,?,0,0,0)", (new_aid, key))
                anchors[new_aid] = Anchor((0, 0, 0), key)

            # Update membership
            if old_aid is None:
                cur.execute("INSERT INTO scoring_anchors VALUES (?,?)", (key, new_aid))
                self._update_anchor_stats(new_aid, counts, anchors, cur)
            else:
                cur.execute("UPDATE scoring_anchors SET anchor_id=? WHERE scoring_key=?", (new_aid, key))
                self._update_anchor_stats(old_aid, _sub_counts((0,0,0), old_stats), anchors, cur)
                self._update_anchor_stats(new_aid, counts, anchors, cur)
            
            self._update_denormalized_id(key, new_aid, cur)
        
        # Cleanup empty anchors
        for aid in [a for a, anc in anchors.items() if anc.total <= 0]:
            cur.execute("DELETE FROM anchors WHERE anchor_id = ?", (aid,))
            cur.execute("DELETE FROM scoring_anchors WHERE anchor_id = ?", (aid,))
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

    def _update_denormalized_id(self, scoring_key: str, anchor_id: int, cur: sqlite3.Cursor) -> None:
        """
        Update denormalized anchor_id in main table.
        
        In Markov mode: scoring_key is the state_hash, update state_values
        In transition mode: scoring_key is from|to, update transitions
        """
        if self._markov:
            cur.execute(
                "UPDATE state_values SET anchor_id=? WHERE state_hash=?",
                (anchor_id, scoring_key)
            )
        else:
            cur.execute(
                "UPDATE transitions SET anchor_id=? WHERE scoring_key=?",
                (anchor_id, scoring_key)
            )

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
        """Find most similar compatible anchor, excluding self if specified."""
        best_aid, best_sim = None, -1.0
        
        for aid, anchor in anchors.items():
            if anchor.total == 0:
                continue
            
            # Self-exclusion: subtract own stats when checking old anchor
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
        
        self._conn.commit()  # Close any implicit transaction
        cur = self._conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        
        try:
            anchors = self._load_anchors(cur)
            initial_count = len(anchors)
            
            # Keep merging until no compatible pairs remain
            merged = True
            while merged:
                merged = False
                active = [aid for aid, a in anchors.items() if a.total > 0]
                
                for i, aid1 in enumerate(active):
                    if aid1 not in anchors:
                        continue
                    for aid2 in active[i+1:]:
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
        
        # Reassign memberships in scoring_anchors
        cur.execute("UPDATE scoring_anchors SET anchor_id=? WHERE anchor_id=?", (survivor, absorbed))
        
        # Update denormalized IDs in the appropriate table
        table = "state_values" if self._markov else "transitions"
        cur.execute(f"UPDATE {table} SET anchor_id=? WHERE anchor_id=?", (survivor, absorbed))
        
        # Merge stats
        self._update_anchor_stats(survivor, anchors[absorbed].counts, anchors, cur)
        
        # Remove absorbed
        cur.execute("DELETE FROM anchors WHERE anchor_id=?", (absorbed,))
        del anchors[absorbed]

    # -------------------------------------------------------------------------
    # Full Rebuild
    # -------------------------------------------------------------------------

    def rebuild(self) -> int:
        """Full rebuild of anchor clustering."""
        if self._read_only:
            raise RuntimeError("Cannot rebuild anchors in read-only mode")
        
        units = self._collect_units()
        if not units:
            return 0

        # Sort by entropy (low entropy = more certain = cluster first)
        units.sort(key=lambda u: u[2])
        anchors, membership = self._cluster_units(units)
        
        self._conn.commit()  # Close any implicit transaction
        cur = self._conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        
        try:
            self._write_anchors(anchors, membership, cur)
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
        
        return len(anchors)

    def _collect_units(self) -> List[Tuple[str, Counts, float]]:
        """
        Collect all scoring units as (key, counts, entropy) tuples.
        
        In Markov mode: scoring units are states from state_values
        In transition mode: scoring units are transitions from transitions table
        """
        if self._markov:
            query = "SELECT state_hash, wins, ties, losses FROM state_values WHERE wins+ties+losses > 0"
            rows = [(h, (w, t, l)) for h, w, t, l in self._conn.execute(query)]
        else:
            query = "SELECT from_hash, to_hash, wins, ties, losses FROM transitions WHERE wins+ties+losses > 0"
            rows = [(f"{fh}|{th}", (w, t, l)) for fh, th, w, t, l in self._conn.execute(query)]
        
        def entropy(counts: Counts) -> float:
            total = sum(counts)
            return -sum((c/total) * math.log(c/total + 1e-12) for c in counts)
        
        return [(key, counts, entropy(counts)) for key, counts in rows]

    def _cluster_units(self, units: List[Tuple[str, Counts, float]]) -> Tuple[List[Anchor], Dict[str, int]]:
        """Cluster units into anchors."""
        anchors: List[Anchor] = []
        membership: Dict[str, int] = {}
        
        for key, counts, _ in units:
            best_idx = self._find_best_cluster(counts, anchors)
            
            if best_idx is not None:
                anchors[best_idx].add(counts)
                membership[key] = best_idx
            else:
                membership[key] = len(anchors)
                anchors.append(Anchor(counts, key))
        
        return anchors, membership

    def _find_best_cluster(self, counts: Counts, anchors: List[Anchor]) -> Optional[int]:
        """Find index of most similar compatible anchor."""
        best_idx, best_sim = None, -1.0
        for i, anchor in enumerate(anchors):
            if compatible(counts, anchor.counts):
                sim = similarity(counts, anchor.counts)
                if sim > best_sim:
                    best_sim, best_idx = sim, i
        return best_idx

    def _write_anchors(self, anchors: List[Anchor], membership: Dict[str, int], cur: sqlite3.Cursor) -> None:
        """Write anchors to database."""
        cur.execute("DELETE FROM anchors")
        cur.execute("DELETE FROM scoring_anchors")
        
        cur.executemany(
            "INSERT INTO anchors VALUES (?,?,?,?,?)",
            [(i, a.repr_key, *a.counts) for i, a in enumerate(anchors)]
        )
        
        cur.executemany(
            "INSERT INTO scoring_anchors VALUES (?,?)",
            list(membership.items())
        )
        
        # Update denormalized anchor_ids in the appropriate table
        table, col = ("state_values", "state_hash") if self._markov else ("transitions", "scoring_key")
        cur.executemany(
            f"UPDATE {table} SET anchor_id=? WHERE {col}=?",
            [(aid, key) for key, aid in membership.items()]
        )