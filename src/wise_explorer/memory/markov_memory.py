"""
State-based (Markov) memory: learns state_hash values only.

Assumes the Markov property — a state's value depends only on the
current position, not the path taken to reach it:  V(s) = f(s).
Faster convergence but loses contextual information.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Any

from wise_explorer.core.types import Stats
from wise_explorer.memory.game_memory import GameMemory
from wise_explorer.memory.schema import SCHEMA_MARKOV


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

    def batch_stats(self, from_hash: str, to_hashes: list[str]) -> dict[str, Stats]:
        """Stats for the given destination states, one query."""
        if not to_hashes:
            return {}
        placeholders = ",".join("?" * len(to_hashes))
        rows = self.conn.execute(
            f"SELECT state_hash, wins, ties, losses FROM states "
            f"WHERE state_hash IN ({placeholders})",
            to_hashes,
        ).fetchall()
        return {r[0]: Stats(r[1], r[2], r[3]) for r in rows}

    def _commit_outcomes(self, transitions: dict[tuple[str, str], list[float]], cur: sqlite3.Cursor) -> None:
        # aggregate by destination state
        state_updates: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        for (_, to_hash), counts in transitions.items():
            state_updates[to_hash][0] += counts[0]
            state_updates[to_hash][1] += counts[1]
            state_updates[to_hash][2] += counts[2]

        cur.executemany(
            """INSERT INTO states (state_hash, wins, ties, losses)
            VALUES (?,?,?,?)
            ON CONFLICT DO UPDATE SET
                wins = wins + excluded.wins,
                ties = ties + excluded.ties,
                losses = losses + excluded.losses""",
            [(s, c[0], c[1], c[2]) for s, c in state_updates.items()],
        )

    def _get_mode_specific_info(self) -> dict[str, Any]:
        states = self.conn.execute("SELECT COUNT(*) FROM states").fetchone()[0]
        samples = self.conn.execute("SELECT COALESCE(SUM(wins+ties+losses), 0) FROM states").fetchone()[0]
        return {
            "mode": "markov",
            "unique_states": states,
            "total_samples": samples,
        }
