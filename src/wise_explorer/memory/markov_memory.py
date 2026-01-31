"""
State-based memory: learns state_hash values only.

Path-independent - only cares about destination state.
Faster convergence but loses contextual information.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from wise_explorer.core.types import Stats
from wise_explorer.memory.game_memory import GameMemory
from wise_explorer.memory.schema import SCHEMA_MARKOV

Counts = Tuple[int, int, int]


class MarkovMemory(GameMemory):
    """State-based (Markov) memory implementation."""

    main_table = "states"
    is_markov = True

    def _schema(self) -> str:
        return SCHEMA_MARKOV

    def get_move_stats(self, from_hash: str, to_hash: str) -> Stats:
        """Get stats for the destination state (from_hash is ignored)."""
        return self.get_state_stats(to_hash)

    def get_state_stats(self, state_hash: str) -> Stats:
        """Get stats for a state."""
        row = self.conn.execute(
            "SELECT wins, ties, losses FROM states WHERE state_hash=?",
            (state_hash,)
        ).fetchone()
        return Stats(*row) if row else Stats()

    def _get_stats_by_key(self, key: str) -> Stats:
        return self.get_state_stats(key)

    def _cache_key(self, from_hash: str, to_hash: str) -> str:
        return to_hash

    def _fetch_anchor_id(self, from_hash: str, to_hash: str) -> Optional[int]:
        row = self.conn.execute(
            "SELECT anchor_id FROM states WHERE state_hash=?",
            (to_hash,)
        ).fetchone()
        return row[0] if row else None

    def _batch_get_anchor_ids(self, keys: List[str], cur: sqlite3.Cursor) -> Dict[str, Optional[int]]:
        if not keys:
            return {}
        placeholders = ','.join('?' * len(keys))
        rows = cur.execute(
            f"SELECT state_hash, anchor_id FROM states WHERE state_hash IN ({placeholders})",
            keys
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def _set_anchor_id(self, key: str, anchor_id: int, cur: sqlite3.Cursor) -> None:
        cur.execute(
            "UPDATE states SET anchor_id=? WHERE state_hash=?",
            (anchor_id, key)
        )

    def _key_to_repr(self, key: str) -> str:
        return key[:16]

    def _collect_units(self) -> List[Tuple[str, Counts]]:
        rows = self.conn.execute(
            "SELECT state_hash, wins, ties, losses FROM states WHERE wins+ties+losses > 0"
        ).fetchall()
        return [(h, (w, t, l)) for h, w, t, l in rows]

    def _write_anchor_ids(self, membership: Dict[str, int], cur: sqlite3.Cursor) -> None:
        cur.executemany(
            "UPDATE states SET anchor_id=? WHERE state_hash=?",
            [(aid, key) for key, aid in membership.items()]
        )

    def _commit_outcomes(self, transitions: Dict[Tuple[str, str], List[int]], cur: sqlite3.Cursor) -> Tuple[List, Dict]:
        # Aggregate by destination state
        state_updates: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
        for (_, to_hash), counts in transitions.items():
            state_updates[to_hash][0] += int(counts[0])
            state_updates[to_hash][1] += int(counts[1])
            state_updates[to_hash][2] += int(counts[2])

        cur.executemany(
            """INSERT INTO states (state_hash, wins, ties, losses)
            VALUES (?,?,?,?)
            ON CONFLICT DO UPDATE SET
                wins = wins + excluded.wins,
                ties = ties + excluded.ties,
                losses = losses + excluded.losses""",
            [(s, c[0], c[1], c[2]) for s, c in state_updates.items()],
        )

        keys = list(state_updates.keys())
        deltas = {k: (v[0], v[1], v[2]) for k, v in state_updates.items()}
        return keys, deltas

    def _get_mode_specific_info(self) -> Dict[str, Any]:
        states = self.conn.execute("SELECT COUNT(*) FROM states").fetchone()[0]
        samples = self.conn.execute("SELECT COALESCE(SUM(wins+ties+losses), 0) FROM states").fetchone()[0]
        return {
            "mode": "markov",
            "unique_states": states,
            "total_samples": samples,
        }