"""
Transition-based (non-Markov) memory: learns (from_hash, to_hash) pairs.

Does NOT assume the Markov property — the same destination reached
via different paths can carry different values:  V(s) = f(s_prev, s).
More precise but requires more data to converge.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from collections import defaultdict

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

    def _get_transition_from_hashes(self) -> Dict[str, str]:
        """Get the from_hash with the most samples for each to_hash."""
        rows = self.conn.execute(
            "SELECT to_hash, from_hash, wins+ties+losses as total "
            "FROM transitions WHERE total > 0 "
            "ORDER BY to_hash, total DESC"
        ).fetchall()
        result = {}
        for to_hash, from_hash, _ in rows:
            if to_hash not in result:
                result[to_hash] = from_hash
        return result

    def _get_destination_bellman_scores(self) -> Dict[str, float]:
        """Average Bellman propagated score per destination board hash."""
        rows = self.conn.execute(
            "SELECT to_hash, AVG(propagated_score) FROM transitions "
            "WHERE propagated_score IS NOT NULL GROUP BY to_hash"
        ).fetchall()
        return {h: score for h, score in rows}

    def _aggregate_destination_scores(self) -> Dict[str, Counts]:
        """Aggregate scores per destination board hash across all transitions."""
        rows = self.conn.execute(
            "SELECT to_hash, SUM(wins), SUM(ties), SUM(losses) "
            "FROM transitions GROUP BY to_hash HAVING SUM(wins+ties+losses) > 0"
        ).fetchall()
        return {h: (w, t, l) for h, w, t, l in rows}

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

    def _record_cross_scores(self, cross_scores: Dict) -> None:
        """Write accumulated cross-scores to the database."""
        if not cross_scores:
            return
        cur = self.conn.cursor()
        cur.executemany(
            """INSERT INTO cross_scores (from_hash, to_hash, observer_role, score_sum, score_count)
            VALUES (?,?,?,?,?)
            ON CONFLICT(from_hash, to_hash, observer_role) DO UPDATE SET
                score_sum = score_sum + excluded.score_sum,
                score_count = score_count + excluded.score_count""",
            [(fh, th, obs, vals[0], vals[1])
             for (fh, th, obs), vals in cross_scores.items()],
        )
        self.conn.commit()

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

    def batch_get_moves_from(self, from_hash: str) -> Dict[str, Tuple[Stats, Optional[int], Optional[float]]]:
        """Fetch stats, anchor_id, and bell score for ALL transitions from a position.

        Returns {to_hash: (Stats, anchor_id, propagated_score)} in a single query.
        Replaces 3 individual queries per move (get_move_stats + get_anchor_id + get_propagated_score).
        """
        rows = self.conn.execute(
            "SELECT to_hash, wins, ties, losses, anchor_id, propagated_score "
            "FROM transitions WHERE from_hash=?",
            (from_hash,)
        ).fetchall()
        return {
            r[0]: (Stats(r[1], r[2], r[3]), r[4], r[5])
            for r in rows
        }

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

    def _compute_alpha(self, child_from: str, child_to: str) -> float:
        """
        Compute the alignment factor α for a child transition.

        Uses the excess formula: α = max(0, μ_cross + μ_mover - 1)
        where μ_cross is the average observer score and μ_mover is
        the empirical mean score of the child transition.

        Returns 0.0 (adversarial default) when cross-score data is
        insufficient, recovering standard zero-sum minimax.
        """
        # Get observer cross-scores (averaged across all observer roles)
        rows = self.conn.execute(
            "SELECT score_sum, score_count FROM cross_scores "
            "WHERE from_hash=? AND to_hash=?",
            (child_from, child_to)
        ).fetchall()

        if not rows:
            return 0.0

        total_sum = sum(r[0] for r in rows)
        total_count = sum(r[1] for r in rows)
        if total_count < 1.0:
            return 0.0

        mu_cross = total_sum / total_count

        # Get mover's empirical mean score
        row = self.conn.execute(
            "SELECT wins, ties, losses FROM transitions "
            "WHERE from_hash=? AND to_hash=?",
            (child_from, child_to)
        ).fetchone()
        if row is None:
            return 0.0
        mu_mover = Stats(*row).mean_score

        return max(0.0, mu_cross + mu_mover - 1.0)

    def get_alpha(self, from_hash: str, to_hash: str) -> Optional[float]:
        """
        Get the α that would be used when propagating a transition.

        Returns the α from the best child of to_hash, or None if
        to_hash has no children (terminal state).
        """
        children = self.conn.execute(
            "SELECT to_hash, propagated_score, wins, ties, losses "
            "FROM transitions WHERE from_hash=?",
            (to_hash,)
        ).fetchall()

        if not children:
            return None

        best_score = -1.0
        best_child_to = None
        for child_to, prop_score, w, t, l in children:
            v = prop_score if prop_score is not None else Stats(w, t, l).mean_score
            if v > best_score:
                best_score = v
                best_child_to = child_to

        if best_child_to is None:
            return None

        return self._compute_alpha(to_hash, best_child_to)

    def _bellman_update(self, from_hash: str, to_hash: str, cur) -> float:
        """Compute and store the Bellman value for a single transition.

        Returns the computed propagated_score.
        """
        children = self.conn.execute(
            "SELECT to_hash, propagated_score, wins, ties, losses "
            "FROM transitions WHERE from_hash=?",
            (to_hash,)
        ).fetchall()

        if not children:
            row = self.conn.execute(
                "SELECT wins, ties, losses FROM transitions "
                "WHERE from_hash=? AND to_hash=?",
                (from_hash, to_hash)
            ).fetchone()
            prop = Stats(*row).mean_score if row else Stats().mean_score
        else:
            best_score = -1.0
            best_child_to = None
            for child_to, prop_score, w, t, l in children:
                v = prop_score if prop_score is not None else Stats(w, t, l).mean_score
                if v > best_score:
                    best_score = v
                    best_child_to = child_to

            v_next = best_score
            alpha = self._compute_alpha(to_hash, best_child_to)
            prop = alpha * v_next + (1.0 - alpha) * (1.0 - v_next)

        cur.execute(
            "UPDATE transitions SET propagated_score=? "
            "WHERE from_hash=? AND to_hash=?",
            (prop, from_hash, to_hash),
        )
        return prop

    def propagate_bellman(self, trajectory_keys: List[Tuple[str, str]]) -> None:
        """
        Backward Bellman sweep along a played trajectory.

        Walk the trajectory deepest-first, updating each transition
        based on its to_hash's children (all known moves from that board).

        When α = 0 (no cross-score data or adversarial game), this
        recovers the standard 1 − max minimax formula.
        """
        if not trajectory_keys:
            return

        cur = self.conn.cursor()

        for from_hash, to_hash in reversed(trajectory_keys):
            self._bellman_update(from_hash, to_hash, cur)

        self.conn.commit()

    def solve_graph(self, epsilon: float = 1e-6, max_iters: int = 200) -> int:
        """Full value iteration on the stored game graph.

        Unlike propagate_bellman (which walks one trajectory at a time),
        this propagates values across ALL edges simultaneously until
        convergence. Produces globally-consistent minimax values.

        Uses topological ordering when possible (acyclic games like Nim/TTT)
        for single-pass convergence. Falls back to iterative for cyclic graphs.

        Returns the number of iterations to convergence.
        """
        rows = self.conn.execute(
            "SELECT from_hash, to_hash, wins, ties, losses "
            "FROM transitions WHERE wins+ties+losses > 0"
        ).fetchall()
        if not rows:
            return 0

        # Build adjacency and stats caches
        children: Dict[str, List[Tuple[str, Stats]]] = defaultdict(list)
        all_boards: set = set()
        stats_cache: Dict[Tuple[str, str], Stats] = {}

        for fh, th, w, t, l in rows:
            s = Stats(w, t, l)
            children[fh].append((th, s))
            stats_cache[(fh, th)] = s
            all_boards.add(fh)
            all_boards.add(th)

        # Load cross-scores for alpha computation
        cross_cache: Dict[Tuple[str, str], float] = {}
        try:
            cross_agg: Dict[Tuple[str, str], List[float]] = defaultdict(
                lambda: [0.0, 0.0]
            )
            for fh, th, ss, sc in self.conn.execute(
                "SELECT from_hash, to_hash, score_sum, score_count "
                "FROM cross_scores WHERE score_count > 0"
            ).fetchall():
                agg = cross_agg[(fh, th)]
                agg[0] += ss
                agg[1] += sc
            for key, (ss, sc) in cross_agg.items():
                cross_cache[key] = ss / sc
        except Exception:
            pass

        def compute_alpha(parent: str, child: str) -> float:
            mu_cross = cross_cache.get((parent, child))
            if mu_cross is None:
                return 0.0
            st = stats_cache.get((parent, child))
            if st is None:
                return 0.0
            return max(0.0, mu_cross + st.mean_score - 1.0)

        # Terminal boards: appear as to_hash but have no outgoing edges
        terminal = all_boards - set(children.keys())

        # Initialize V[board]
        V: Dict[str, float] = {}
        for b in all_boards:
            if b in terminal:
                incoming = [s for fh in children for th, s in children[fh] if th == b]
                V[b] = sum(s.mean_score for s in incoming) / len(incoming) if incoming else 0.5
            else:
                V[b] = 0.5

        # Topological sort (Kahn's algorithm)
        in_degree: Dict[str, int] = defaultdict(int)
        for parent, kids in children.items():
            for child, _ in kids:
                in_degree[child] += 1

        queue = [b for b in children if in_degree.get(b, 0) == 0]
        topo_order: List[str] = []
        while queue:
            b = queue.pop()
            topo_order.append(b)
            for child, _ in children.get(b, []):
                in_degree[child] -= 1
                if in_degree[child] == 0 and child in children:
                    queue.append(child)

        is_acyclic = len(topo_order) == len(children)

        def best_child_value(board: str):
            best_v, best_c = -1.0, None
            for child, _ in children.get(board, []):
                cv = V.get(child, 0.5)
                if cv > best_v:
                    best_v, best_c = cv, child
            return best_v, best_c

        if is_acyclic:
            for b in reversed(topo_order):
                if not children.get(b):
                    continue
                best_v, best_c = best_child_value(b)
                alpha = compute_alpha(b, best_c) if best_c else 0.0
                V[b] = alpha * best_v + (1.0 - alpha) * (1.0 - best_v)
            n_iters = 1
        else:
            non_terminal = [b for b in children if b not in terminal]
            n_iters = max_iters
            for iteration in range(1, max_iters + 1):
                max_delta = 0.0
                for b in non_terminal:
                    best_v, best_c = best_child_value(b)
                    alpha = compute_alpha(b, best_c) if best_c else 0.0
                    new_v = alpha * best_v + (1.0 - alpha) * (1.0 - best_v)
                    max_delta = max(max_delta, abs(new_v - V[b]))
                    V[b] = new_v
                if max_delta < epsilon:
                    n_iters = iteration
                    break

        # Write back propagated_score for every transition
        cur = self.conn.cursor()
        for fh, th, w, t, l in rows:
            kids = children.get(th, [])
            if not kids:
                prop = Stats(w, t, l).mean_score
            else:
                best_v, best_c = best_child_value(th)
                alpha = compute_alpha(th, best_c) if best_c else 0.0
                prop = alpha * best_v + (1.0 - alpha) * (1.0 - best_v)
            cur.execute(
                "UPDATE transitions SET propagated_score=? "
                "WHERE from_hash=? AND to_hash=?",
                (prop, fh, th),
            )
        self.conn.commit()
        return n_iters
