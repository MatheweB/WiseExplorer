"""
Transition-based (non-Markov) memory: learns (from_hash, to_hash) pairs.

Does NOT assume the Markov property — the same destination reached
via different paths can carry different values:  V(s) = f(s_prev, s).
More precise but requires more data to converge.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from wise_explorer.core.types import Stats, Counts
from wise_explorer.memory.game_memory import GameMemory
from wise_explorer.memory.schema import SCHEMA_TRANSITIONS


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

    def get_stats_by_key(self, key: Tuple[str, str]) -> Stats:
        return self.get_move_stats(key[0], key[1])

    def _cache_key(self, from_hash: str, to_hash: str) -> Tuple[str, str]:
        return (from_hash, to_hash)

    def _fetch_anchor_id(self, from_hash: str, to_hash: str) -> Optional[int]:
        row = self.conn.execute(
            "SELECT anchor_id FROM transitions WHERE from_hash=? AND to_hash=?",
            (from_hash, to_hash)
        ).fetchone()
        return row[0] if row else None

    def batch_get_anchor_ids(self, keys: List[Tuple[str, str]], cur: sqlite3.Cursor) -> Dict[Tuple[str, str], Optional[int]]:
        result = {}
        for from_hash, to_hash in keys:
            row = cur.execute(
                "SELECT anchor_id FROM transitions WHERE from_hash=? AND to_hash=?",
                (from_hash, to_hash)
            ).fetchone()
            result[(from_hash, to_hash)] = row[0] if row else None
        return result

    def set_anchor_id(self, key: Tuple[str, str], anchor_id: int, cur: sqlite3.Cursor) -> None:
        cur.execute(
            "UPDATE transitions SET anchor_id=? WHERE from_hash=? AND to_hash=?",
            (anchor_id, key[0], key[1])
        )

    def key_to_repr(self, key: Tuple[str, str]) -> str:
        return f"{key[0][:8]}→{key[1][:8]}"

    def collect_units(self) -> List[Tuple[Tuple[str, str], Counts]]:
        rows = self.conn.execute(
            "SELECT from_hash, to_hash, wins, ties, losses FROM transitions WHERE wins+ties+losses > 0"
        ).fetchall()
        return [((fh, th), (w, t, l)) for fh, th, w, t, l in rows]

    def write_anchor_ids(self, membership: Dict[Tuple[str, str], int], cur: sqlite3.Cursor) -> None:
        cur.executemany(
            "UPDATE transitions SET anchor_id=? WHERE from_hash=? AND to_hash=?",
            [(aid, key[0], key[1]) for key, aid in membership.items()]
        )

    def _commit_outcomes(self, transitions: Dict[Tuple[str, str], List[float]], cur: sqlite3.Cursor) -> Tuple[List, Dict]:
        """Commit outcomes and return keys/deltas for anchor manager."""
        cur.executemany(
            """INSERT INTO transitions (from_hash, to_hash, wins, ties, losses)
            VALUES (?,?,?,?,?)
            ON CONFLICT(from_hash, to_hash) DO UPDATE SET
                wins = wins + excluded.wins,
                ties = ties + excluded.ties,
                losses = losses + excluded.losses""",
            [(fh, th, c[0], c[1], c[2]) for (fh, th), c in transitions.items()],
        )

        keys = list(transitions.keys())
        deltas = {k: (c[0], c[1], c[2]) for k, c in transitions.items()}
        return keys, deltas

    def _propagate_bellman(self, trajectory_keys: List[List[Tuple[str, str]]]) -> None:
        """Run Bellman backward sweep along each played trajectory."""
        for stack_keys in trajectory_keys:
            self.propagate_bellman(stack_keys)

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

    # -------------------------------------------------------------------------
    # Bellman Propagation
    # -------------------------------------------------------------------------

    def get_propagated_score(self, from_hash: str, to_hash: str) -> Optional[float]:
        """Get the propagated minimax score for a transition, or None if not computed."""
        row = self.conn.execute(
            "SELECT propagated_score FROM transitions WHERE from_hash=? AND to_hash=?",
            (from_hash, to_hash)
        ).fetchone()
        if row and row[0] is not None:
            return row[0]
        return None

    def _best_estimate(self, from_hash: str, to_hash: str) -> float:
        """Return propagated_score if available, else mean_score (Eq. 3 in design doc)."""
        row = self.conn.execute(
            "SELECT propagated_score, wins, ties, losses FROM transitions WHERE from_hash=? AND to_hash=?",
            (from_hash, to_hash)
        ).fetchone()
        if row is None:
            return Stats().mean_score  # Bayesian prior ~0.5
        if row[0] is not None:
            return row[0]
        return Stats(row[1], row[2], row[3]).mean_score

    def propagate_bellman(self, trajectory_keys: List[Tuple[str, str]]) -> None:
        """
        Backward Bellman sweep along a played trajectory.

        For each transition (from_hash, to_hash) processed deepest-first:
          - If to_hash has no children in DB: P = mean_score(from, to)
          - Else: P = 1 - max(V̂(to_hash, child)) over all children

        This is the core of amortized Bellman propagation (Section 3 of design doc).
        """
        if not trajectory_keys:
            return

        cur = self.conn.cursor()
        for from_hash, to_hash in reversed(trajectory_keys):
            children = self.conn.execute(
                "SELECT to_hash, propagated_score, wins, ties, losses "
                "FROM transitions WHERE from_hash=?",
                (to_hash,)
            ).fetchall()

            if not children:
                # Terminal: use empirical mean score
                row = self.conn.execute(
                    "SELECT wins, ties, losses FROM transitions WHERE from_hash=? AND to_hash=?",
                    (from_hash, to_hash)
                ).fetchone()
                prop = Stats(*row).mean_score if row else Stats().mean_score
            else:
                # Non-terminal: negamax over children
                child_scores = []
                for child_to, prop_score, w, t, l in children:
                    if prop_score is not None:
                        child_scores.append(prop_score)
                    else:
                        child_scores.append(Stats(w, t, l).mean_score)
                prop = 1.0 - max(child_scores)

            cur.execute(
                "UPDATE transitions SET propagated_score=? WHERE from_hash=? AND to_hash=?",
                (prop, from_hash, to_hash)
            )

        self.conn.commit()
