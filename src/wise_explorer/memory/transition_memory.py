"""
Transition-based (non-Markov) memory: learns (from_hash, to_hash) pairs.

Does NOT assume the Markov property — the same destination reached
via different paths can carry different values:  V(s) = f(s_prev, s).
More precise but requires more data to converge.
"""

from __future__ import annotations

import sqlite3
from typing import Any


import numpy as np

from wise_explorer.core.hashing import hash_board
from wise_explorer.core.types import Stats, Counts, OUTCOME_SCORE
from wise_explorer.games.game_state import GameState
from wise_explorer.memory.game_memory import GameMemory
from wise_explorer.memory.schema import SCHEMA_TRANSITIONS


def _replies_chunk(args):
    """Pool worker for :meth:`TransitionMemory.reply_graph`: enumerate one chunk of
    boards' legal replies. Pure game work — all bookkeeping stays in the parent."""
    game, shape, known, chunk = args
    seats = range(1, game.num_players() + 1)
    out = []
    for i, raw in chunk:
        board = raw.reshape(shape)
        probe = game.deep_clone()
        probe.set_state(GameState(board.copy(), current_player=1))
        if probe.is_over():
            out.append((i, max(OUTCOME_SCORE[probe.get_result(p)] for p in seats), None))
            continue
        seen: set = set()
        edges = []
        for p in seats:
            seat_game = game.deep_clone()
            seat_game.set_state(GameState(board.copy(), current_player=p))
            for mv in seat_game.valid_moves():
                child_game = seat_game.deep_clone()
                child_game.apply_move(mv, validated=True)
                child = child_game.get_state().board
                ch = hash_board(child)
                if ch in seen:
                    continue
                seen.add(ch)
                edges.append((ch, None if ch in known
                              else np.asarray(child).ravel().astype(np.int64)))
        out.append((i, None, edges))
    return out


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

    def _commit_outcomes(self, transitions: dict[tuple[str, str], list[float]], cur: sqlite3.Cursor) -> None:
        cur.executemany(
            """INSERT INTO transitions (from_hash, to_hash, wins, ties, losses)
            VALUES (?,?,?,?,?)
            ON CONFLICT(from_hash, to_hash) DO UPDATE SET
                wins = wins + excluded.wins,
                ties = ties + excluded.ties,
                losses = losses + excluded.losses""",
            [(fh, th, c[0], c[1], c[2]) for (fh, th), c in transitions.items()],
        )

    def _record_cross_scores(self, cross_scores: dict) -> None:
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

    def _propagate_bellman(self, trajectory_keys: list[list[tuple[str, str]]]) -> None:
        """Run Bellman backward sweep along each played trajectory."""
        for stack_keys in trajectory_keys:
            self.propagate_bellman(stack_keys)

    def _get_mode_specific_info(self) -> dict[str, Any]:
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

    def get_transitions_from(self, from_hash: str) -> dict[str, Stats]:
        """Get all transitions from a given state."""
        rows = self.conn.execute(
            "SELECT to_hash, wins, ties, losses FROM transitions WHERE from_hash=?",
            (from_hash,)
        ).fetchall()
        return {r[0]: Stats(r[1], r[2], r[3]) for r in rows}

    # -------------------------------------------------------------------------
    # Bellman Propagation
    # -------------------------------------------------------------------------

    def get_propagated_score(self, from_hash: str, to_hash: str) -> float | None:
        """Get the propagated minimax score for a transition, or None if not computed."""
        row = self.conn.execute(
            "SELECT propagated_score FROM transitions WHERE from_hash=? AND to_hash=?",
            (from_hash, to_hash)
        ).fetchone()
        if row and row[0] is not None:
            return row[0]
        return None

    def batch_stats(self, from_hash: str, to_hashes: list[str]) -> dict[str, Stats]:
        """All known transitions out of a position: {to_hash: stats}, one query."""
        rows = self.conn.execute(
            "SELECT to_hash, wins, ties, losses FROM transitions WHERE from_hash=?",
            (from_hash,)
        ).fetchall()
        return {r[0]: Stats(r[1], r[2], r[3]) for r in rows}

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

    def get_alpha(self, from_hash: str, to_hash: str) -> float | None:
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

    def propagate_bellman(self, trajectory_keys: list[tuple[str, str]]) -> None:
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

    def _solve_graph_arrays(self, rows, epsilon: float, max_iters: int) -> int:
        """The value iteration as array work: boards are indexed, each board's stored
        children form one ``reduceat`` segment, terminals hold the mean of their incoming
        evidence, and ``V(b) = α·best + (1−α)·(1−best)`` iterates to the fixpoint. α is
        the alignment factor of the *best* edge (``max(0, μ_cross + μ_mover − 1)``, zero
        where no cross data exists); among tied-best children the MOST
        ADVERSARIAL alignment (min α) applies — solved positions share exact values, so
        ties are real, and resolving them by float noise made results order-dependent.
        ~80× the old dict-loop on a 77k-board cyclic graph."""
        fhs = [r[0] for r in rows]
        ths = [r[1] for r in rows]
        index: dict[str, int] = {}
        for h in fhs:
            index.setdefault(h, len(index))
        for h in ths:
            index.setdefault(h, len(index))
        n = len(index)
        parent = np.array([index[h] for h in fhs], dtype=np.int64)
        child = np.array([index[h] for h in ths], dtype=np.int64)
        mean = np.array([Stats(w, t, l).mean_score for _, _, w, t, l in rows])

        # per-edge cross-player mean (NaN where unobserved → α contribution is 0)
        cross = np.full(len(rows), np.nan)
        agg: dict[tuple[str, str], list[float]] = {}
        for fh, th, ss, sc in self.conn.execute(
                "SELECT from_hash, to_hash, score_sum, score_count "
                "FROM cross_scores WHERE score_count > 0").fetchall():
            a = agg.setdefault((fh, th), [0.0, 0.0])
            a[0] += ss
            a[1] += sc
        if agg:
            for i, key in enumerate(zip(fhs, ths)):
                if key in agg:
                    ss, sc = agg[key]
                    cross[i] = ss / sc
        edge_alpha = np.maximum(0.0, np.nan_to_num(cross, nan=-1.0) + mean - 1.0)

        # terminals (no outgoing edges) hold the mean of their incoming evidence
        has_kids = np.zeros(n, dtype=bool)
        has_kids[parent] = True
        inc_sum = np.zeros(n)
        inc_cnt = np.zeros(n)
        np.add.at(inc_sum, child, mean)
        np.add.at(inc_cnt, child, 1.0)
        V = np.full(n, 0.5)
        terminal = ~has_kids & (inc_cnt > 0)
        V[terminal] = inc_sum[terminal] / inc_cnt[terminal]

        order = np.argsort(parent, kind="stable")
        child_s, alpha_s = child[order], edge_alpha[order]
        starts = np.searchsorted(parent[order], np.arange(n))
        pos = np.arange(len(order))
        seg = np.searchsorted(starts, pos, side="right") - 1     # row → its parent segment
        kid_starts = starts[has_kids]

        def backup(V):
            cv = V[child_s]
            best = np.full(n, 0.5)
            best[has_kids] = np.maximum.reduceat(cv, kid_starts)
            # several replies can tie as best; the blend takes the MOST ADVERSARIAL
            # reading among them (min α) — ties are real (solved positions share exact
            # values) and resolving them by float noise made results order-dependent
            tie_a = np.where(cv == best[seg], alpha_s, np.inf)
            a = np.zeros(n)
            a[has_kids] = np.minimum.reduceat(tie_a, kid_starts)
            return np.where(has_kids, a * best + (1.0 - a) * (1.0 - best), V)

        n_iters = max_iters
        for it in range(1, max_iters + 1):
            newV = backup(V)
            delta = float(np.max(np.abs(newV - V))) if n else 0.0
            V = newV
            if delta < epsilon:
                n_iters = it
                break

        # propagated_score per edge: the value of landing on ``to`` (terminals keep
        # their own edge's observed mean, as in the dict-loop oracle)
        prop = np.where(has_kids[child], V[child], mean)
        self.conn.cursor().executemany(
            "UPDATE transitions SET propagated_score=? WHERE from_hash=? AND to_hash=?",
            [(float(p), f, t) for p, f, t in zip(prop, fhs, ths)],
        )
        self.conn.commit()
        return n_iters

    def solve_graph(self, epsilon: float = 1e-6, max_iters: int = 200) -> int:
        """Full value iteration on the stored game graph.

        Unlike propagate_bellman (which walks one trajectory at a time),
        this propagates values across ALL edges simultaneously until
        convergence. Produces globally-consistent minimax values.

        Solved as a vectorized fixpoint (the same array form :meth:`complete_values`
        uses; ~80× the old dict-loop on a 77k-board cyclic graph). The backup is
        ``V(b) = α·best + (1−α)·(1−best)`` with α — the alignment factor — read from
        the best edge's cross-player data, zero where none exists. Among tied-best
        children the most adversarial alignment (min α) applies: solved positions
        share exact values, so ties are real, and resolving them by float noise made
        results order-dependent. Returns the number of iterations to convergence.
        """
        rows = self.conn.execute(
            "SELECT from_hash, to_hash, wins, ties, losses "
            "FROM transitions WHERE wins+ties+losses > 0"
        ).fetchall()
        if not rows:
            return 0
        return self._solve_graph_arrays(rows, epsilon, max_iters)

    def reply_graph(self, game) -> dict | None:
        """Enumerate every stored board's full legal reply set — the structural half of
        :meth:`complete_values`. A pure function of (stored boards, game rules), and the
        loop never adds boards or transitions between its beats, so one boundary builds
        this once and hands it to both healing passes.

        Replies are enumerated for every seat and merged: exact for games whose moves
        don't depend on whose turn it is (Nim), a conservative superset otherwise.
        The enumeration is pure per-board game work, so when a runner has lent its
        worker pool (``self.pool``) the boards are chunked across it — same rows,
        same order, ~workers× faster on move-generation-heavy games."""
        boards = self._load_boards()
        if not boards:
            return None
        hashes = list(boards)
        index = {h: i for i, h in enumerate(hashes)}
        n = len(hashes)
        shape = game.get_state().board.shape            # stored boards are 2-D-normalized

        items = [(i, boards[h]) for i, h in enumerate(hashes)]
        pool = getattr(self, "pool", None)
        if pool is not None:
            chunks = max(1, len(pool._pool) * 4)        # a few tasks per worker
            step = -(-len(items) // chunks)
            known_set = set(hashes)
            results = pool.map(_replies_chunk,
                               [(game, shape, known_set, items[a:a + step])
                                for a in range(0, len(items), step)])
            per_board = [r for chunk in results for r in chunk]
        else:
            per_board = _replies_chunk((game, shape, set(hashes), items))

        V0 = np.full(n, 0.5)
        fixed = np.zeros(n, dtype=bool)                 # terminals hold their value
        parents: list[int] = []                         # one row per (board, legal reply)
        known: list[int] = []                           # reply's row index, or -1
        novel_of: list[int] = []                        # candidate row of each novel reply
        novel_boards: list[np.ndarray] = []
        novel_parents: list[np.ndarray] = []

        for i, terminal_v, edges in per_board:
            if terminal_v is not None:
                # the mover who LANDED here gets the best seat's outcome (they just moved)
                V0[i] = terminal_v
                fixed[i] = True
                continue
            for ch, novel_board in edges:
                parents.append(i)
                known.append(index.get(ch, -1))
                if novel_board is not None:
                    novel_of.append(len(known) - 1)
                    novel_boards.append(novel_board)
                    novel_parents.append(np.asarray(boards[hashes[i]]).ravel().astype(np.int64))

        # rows were appended board by board, so ``parents`` is already sorted — reduceat
        # segment starts come straight from searchsorted
        parents_a = np.array(parents, dtype=np.int64)
        known_a = np.array(known, dtype=np.int64)
        starts = np.searchsorted(parents_a, np.arange(n))
        ends = np.append(starts[1:], len(parents_a))

        if novel_boards:
            NB = np.stack(novel_boards)
            NP = np.stack(novel_parents)
            # the token each move placed — the new non-empty value at a changed cell —
            # in one pass over all novel replies (vectorized _placed_token)
            placed = (NP != NB) & (NB != 0)
            first = placed.argmax(1)
            novel_m = np.where(placed.any(1), NB[np.arange(len(NB)), first], 0)
        else:
            NB = np.zeros((0, 0), dtype=np.int64)
            novel_m = np.zeros(0, dtype=np.int64)

        return {
            "boards": boards, "index": index, "V0": V0,
            "known": known_a, "starts": starts,
            "has_kids": (starts < ends) & ~fixed,
            "novel_rows": np.array(novel_of, dtype=np.int64),
            "novel_boards": NB, "novel_m": novel_m,
            "edges": self.conn.execute("SELECT from_hash, to_hash FROM transitions").fetchall(),
        }

    def complete_values(self, game, graph: dict | None = None) -> int:
        """Complete the value graph with the concept library — the value loop's healing step.

        :meth:`solve_graph` can only take its max over replies somebody has *played*, so
        a position with an unexplored refutation looks safe until someone stumbles into
        it. This pass re-runs the same backup with the max over **all legal replies**,
        pricing the never-played ones with the library (no opinion → ignored, exactly as
        they are today). Call it after ``solve_graph``: evidence re-derives every value
        from raw counts first, then the library fills only the gaps — see
        ``grow_concepts`` for the full loop ordering and docs/value-loop.md for why.

        ``graph`` is a prebuilt :meth:`reply_graph`; passing it lets a boundary's two
        healing passes share one enumeration. Zero-sum backup (α = 0). Returns how many
        never-played replies were priced.
        """
        lib = self.concept_library
        if not lib.rules:
            return 0                                    # nothing known → nothing to lend
        if graph is None:
            graph = self.reply_graph(game)
        if graph is None:
            return 0
        index, known = graph["index"], graph["known"]
        starts, has_kids = graph["starts"], graph["has_kids"]
        n = len(graph["V0"])

        # price the never-played replies with the *current* rules (they change between heals)
        consts = np.full(len(known), np.nan)
        priced = 0
        if len(graph["novel_boards"]):
            prices = lib.values_for(graph["novel_boards"], graph["novel_m"])
            consts[graph["novel_rows"]] = prices
            priced = int(np.count_nonzero(~np.isnan(prices)))

        # the same backup as solve_graph — V(b) = 1 − max over replies — but the max now
        # ranges over every legal reply, library-priced where unvisited
        V = graph["V0"].copy()
        for _ in range(200):
            cand = np.where(known >= 0, V[np.maximum(known, 0)], consts)
            cand = np.where(np.isnan(cand), -np.inf, cand)
            best = np.full(n, -np.inf)
            if has_kids.any():
                best[has_kids] = np.maximum.reduceat(cand, starts[has_kids])
            newV = np.where(has_kids & np.isfinite(best), 1.0 - best, V)
            if np.allclose(newV, V, atol=1e-9):
                V = newV
                break
            V = newV

        self.conn.cursor().executemany(
            "UPDATE transitions SET propagated_score=? WHERE from_hash=? AND to_hash=?",
            [(float(V[index[t]]), f, t) for f, t in graph["edges"] if t in index],
        )
        self.conn.commit()
        return priced
