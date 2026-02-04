"""
Transition-based memory: learns (from_hash, to_hash) pairs.

Path-dependent - captures context of how you reached a state.
More precise but requires more data to converge.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from wise_explorer.core.types import Stats
from wise_explorer.memory.game_memory import GameMemory
from wise_explorer.memory.schema import SCHEMA_TRANSITIONS

Counts = Tuple[int, int, int]


class TransitionMemory(GameMemory):
    """Transition-based memory implementation."""

    main_table = "transitions"
    is_markov = False

    def _schema(self) -> str:
        return SCHEMA_TRANSITIONS

    def get_move_stats(self, from_hash: str, to_hash: str) -> Stats:
        """Get stats for a specific transition."""
        row = self.conn.execute(
            "SELECT wins, ties, losses FROM transitions WHERE from_hash=? AND to_hash=?",
            (from_hash, to_hash)
        ).fetchone()
        return Stats(*row) if row else Stats()

    def _get_stats_by_key(self, key: Tuple[str, str]) -> Stats:
        return self.get_move_stats(key[0], key[1])

    def _cache_key(self, from_hash: str, to_hash: str) -> Tuple[str, str]:
        return (from_hash, to_hash)

    def _fetch_anchor_id(self, from_hash: str, to_hash: str) -> Optional[int]:
        row = self.conn.execute(
            "SELECT anchor_id FROM transitions WHERE from_hash=? AND to_hash=?",
            (from_hash, to_hash)
        ).fetchone()
        return row[0] if row else None

    def _batch_get_anchor_ids(self, keys: List[Tuple[str, str]], cur: sqlite3.Cursor) -> Dict[Tuple[str, str], Optional[int]]:
        result = {}
        for from_hash, to_hash in keys:
            row = cur.execute(
                "SELECT anchor_id FROM transitions WHERE from_hash=? AND to_hash=?",
                (from_hash, to_hash)
            ).fetchone()
            result[(from_hash, to_hash)] = row[0] if row else None
        return result

    def _set_anchor_id(self, key: Tuple[str, str], anchor_id: int, cur: sqlite3.Cursor) -> None:
        cur.execute(
            "UPDATE transitions SET anchor_id=? WHERE from_hash=? AND to_hash=?",
            (anchor_id, key[0], key[1])
        )

    def _key_to_repr(self, key: Tuple[str, str]) -> str:
        return f"{key[0][:8]}→{key[1][:8]}"

    def _collect_units(self) -> List[Tuple[Tuple[str, str], Counts]]:
        rows = self.conn.execute(
            "SELECT from_hash, to_hash, wins, ties, losses FROM transitions WHERE wins+ties+losses > 0"
        ).fetchall()
        return [((fh, th), (w, t, l)) for fh, th, w, t, l in rows]

    def _write_anchor_ids(self, membership: Dict[Tuple[str, str], int], cur: sqlite3.Cursor) -> None:
        cur.executemany(
            "UPDATE transitions SET anchor_id=? WHERE from_hash=? AND to_hash=?",
            [(aid, key[0], key[1]) for key, aid in membership.items()]
        )

    def _commit_outcomes(self, transitions: Dict[Tuple[str, str], List[int]], cur: sqlite3.Cursor) -> Tuple[List, Dict]:
        """Commit outcomes and return keys/deltas for anchor manager."""
        cur.executemany(
            """INSERT INTO transitions (from_hash, to_hash, wins, ties, losses)
            VALUES (?,?,?,?,?)
            ON CONFLICT(from_hash, to_hash) DO UPDATE SET
                wins = wins + excluded.wins,
                ties = ties + excluded.ties,
                losses = losses + excluded.losses""",
            [(fh, th, int(c[0]), int(c[1]), int(c[2])) for (fh, th), c in transitions.items()],
        )

        keys = list(transitions.keys())
        deltas = {k: (int(c[0]), int(c[1]), int(c[2])) for k, c in transitions.items()}
        return keys, deltas

    def _get_mode_specific_info(self) -> Dict[str, Any]:
        trans = self.conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
        samples = self.conn.execute("SELECT COALESCE(SUM(wins+ties+losses), 0) FROM transitions").fetchone()[0]
        from_states = self.conn.execute("SELECT COUNT(DISTINCT from_hash) FROM transitions").fetchone()[0]
        to_states = self.conn.execute("SELECT COUNT(DISTINCT to_hash) FROM transitions").fetchone()[0]
        return {
            "mode": "transition",
            "transitions": trans,
            "from_states": from_states,
            "to_states": to_states,
            "total_samples": samples,
        }

    # -------------------------------------------------------------------------
    # Transition-Specific Methods
    # -------------------------------------------------------------------------

    def get_transitions_from(self, from_hash: str) -> Dict[str, Stats]:
        """Get all transitions from a given state."""
        rows = self.conn.execute(
            "SELECT to_hash, wins, ties, losses FROM transitions WHERE from_hash=?",
            (from_hash,)
        ).fetchall()
        return {r[0]: Stats(r[1], r[2], r[3]) for r in rows}