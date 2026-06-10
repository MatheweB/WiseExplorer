"""
Base class for game memory implementations.

Provides shared infrastructure for caching, anchor management,
move evaluation, and recording. Subclasses implement the
mode-specific storage and retrieval logic.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple, TYPE_CHECKING

import numpy as np

from wise_explorer.core.types import Stats, Counts, OUTCOME_INDEX, OUTCOME_SCORE, is_decisive
from wise_explorer.core.hashing import hash_board
from wise_explorer.core.bayes import compatible
from wise_explorer.memory.anchor_manager import AnchorManager
from wise_explorer.memory.concept_library import ConceptLibrary

if TYPE_CHECKING:
    from wise_explorer.games.game_base import GameBase
UNEXPLORED_ANCHOR_ID = -999


def _placed_token(from_board: np.ndarray, to_board: np.ndarray) -> int:
    """The token the move just placed — the new non-empty value at a changed cell. This is
    the only perspective the concept layer needs (move-relative concepts read it; cell-only
    ones ignore it). 0 when it can't be recovered."""
    a = np.asarray(from_board).ravel()
    b = np.asarray(to_board).ravel()
    if a.shape != b.shape:
        return 0
    placed = b[(a != b) & (b != 0)]
    return int(placed[0]) if len(placed) else 0


class MoveEvaluation(NamedTuple):
    """Result of evaluate_moves: moves grouped by anchor with anchor stats.

    Also carries bell and concept scores so selection doesn't need to
    re-clone games and re-query the DB for the same information.
    """
    anchors_with_moves: dict[int, list[tuple[np.ndarray, Stats]]]
    anchor_stats: dict[int, Stats]
    bell_scores: dict[tuple, float | None]  # move_key -> propagated_score
    concept_scores: dict[tuple, float | None]  # move_key -> invented-concept value


class GameMemory(ABC):
    """Abstract base for game memory implementations."""

    main_table: str  # Subclasses define: "transitions" or "states"
    is_markov: bool  # Subclasses define: False for Transition, True for Markov

    def __init__(self, db_path: str | Path, read_only: bool = False):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.read_only = read_only
        self._closed = False

        self._anchor_stats_cache: dict[int, Stats] = {}
        self._anchor_id_cache: dict[Any, int | None] = {}

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA cache_size=-65536")
        self.conn.execute("PRAGMA synchronous=NORMAL")

        if not read_only:
            self.conn.executescript(self._schema())
            self.conn.commit()

        self.anchors = AnchorManager(self.conn, self.main_table, self.read_only)
        self.concept_library = ConceptLibrary(self.conn, self.read_only)   # invented concepts, persisted

    # -------------------------------------------------------------------------
    # Abstract Methods (subclasses must implement)
    # -------------------------------------------------------------------------

    @abstractmethod
    def _schema(self) -> str:
        """Return the SQL schema for this memory type."""
        pass

    @abstractmethod
    def get_move_stats(self, from_hash: str, to_hash: str) -> Stats:
        """Get stats for evaluating a move from one state to another."""
        pass

    @abstractmethod
    def get_stats_by_key(self, key) -> Stats:
        """Get stats by native key type (for anchor manager)."""
        pass

    @abstractmethod
    def _cache_key(self, from_hash: str, to_hash: str):
        """Return the cache key for anchor ID lookups."""
        pass

    @abstractmethod
    def _fetch_anchor_id(self, from_hash: str, to_hash: str) -> int | None:
        """Fetch anchor ID from database."""
        pass

    @abstractmethod
    def batch_get_anchor_ids(self, keys: list, cur: sqlite3.Cursor) -> dict:
        """Batch fetch anchor IDs for keys."""
        pass

    @abstractmethod
    def set_anchor_id(self, key, anchor_id: int, cur: sqlite3.Cursor) -> None:
        """Set anchor_id for a key in the main table."""
        pass

    @abstractmethod
    def key_to_repr(self, key) -> str:
        """Convert a key to string representation for debugging."""
        pass

    @abstractmethod
    def collect_units(self) -> list[tuple]:
        """Collect all units as (key, counts) tuples for rebuild."""
        pass

    @abstractmethod
    def write_anchor_ids(self, membership: dict, cur: sqlite3.Cursor) -> None:
        """Batch write anchor IDs after rebuild."""
        pass

    @abstractmethod
    def _commit_outcomes(self, transitions: dict[tuple[str, str], list[float]], cur: sqlite3.Cursor) -> tuple[list, dict]:
        """Commit outcomes and return (keys, deltas) for anchor update."""
        pass

    @abstractmethod
    def _get_mode_specific_info(self) -> dict[str, Any]:
        """Return mode-specific info for get_info()."""
        pass

    # -------------------------------------------------------------------------
    # Anchor Queries (shared implementation)
    # -------------------------------------------------------------------------

    def get_anchor_id(self, from_hash: str, to_hash: str) -> int | None:
        """Get anchor ID for a move (cached)."""
        key = self._cache_key(from_hash, to_hash)
        if key in self._anchor_id_cache:
            return self._anchor_id_cache[key]

        aid = self._fetch_anchor_id(from_hash, to_hash)
        self._anchor_id_cache[key] = aid
        return aid

    def get_anchor_stats_by_id(self, anchor_id: int) -> Stats:
        """Get anchor stats by ID (cached)."""
        if anchor_id in self._anchor_stats_cache:
            return self._anchor_stats_cache[anchor_id]

        row = self.conn.execute(
            "SELECT wins, ties, losses FROM anchors WHERE anchor_id=?",
            (anchor_id,)
        ).fetchone()
        stats = Stats(*row) if row else Stats()
        self._anchor_stats_cache[anchor_id] = stats
        return stats

    def get_anchor_stats(self, from_hash: str, to_hash: str) -> Stats:
        """Get pooled statistics from the anchor cluster."""
        anchor_id = self.get_anchor_id(from_hash, to_hash)
        if anchor_id is not None:
            return self.get_anchor_stats_by_id(anchor_id)
        return self.get_move_stats(from_hash, to_hash)

    def get_effective_stats(self, from_hash: str, to_hash: str) -> Stats:
        """Get best available stats (anchor if compatible, else direct)."""
        direct = self.get_move_stats(from_hash, to_hash)
        anchor = self.get_anchor_stats(from_hash, to_hash)

        if anchor.total <= direct.total:
            return direct
        if direct.total > 0 and not compatible(direct.as_tuple(), anchor.as_tuple()):
            return direct
        return anchor

    def get_anchor_details(self) -> list[dict[str, Any]]:
        """Get detailed information about all anchors."""
        return self.anchors.get_details()

    def rebuild_anchors(self) -> int:
        """Full rebuild of anchor clustering."""
        if self.read_only:
            raise RuntimeError("Cannot rebuild anchors in read-only mode")

        units = self.collect_units()
        if not units:
            return 0

        self.conn.commit()
        cur = self.conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            num, membership = self.anchors.rebuild(units, self.key_to_repr, cur)
            self.write_anchor_ids(membership, cur)
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
        return num

    def consolidate_anchors(self) -> int:
        """Merge compatible anchors."""
        return self.anchors.consolidate()

    # -------------------------------------------------------------------------
    # Info
    # -------------------------------------------------------------------------

    def get_info(self) -> dict[str, Any]:
        """Get summary statistics."""
        anchors = self.conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
        return {
            "anchors": anchors,
            "concepts": len(self.concept_library.kept),
            **self._get_mode_specific_info(),
        }

    # -------------------------------------------------------------------------
    # Move Evaluation
    # -------------------------------------------------------------------------

    def evaluate_moves(self, game: GameBase, valid_moves: list[np.ndarray]) -> MoveEvaluation:
        """Evaluate all valid moves and group by anchor.

        Uses a single batch DB query per position instead of 3 queries per move.
        Also collects bell and concept scores so selection doesn't need to
        re-clone games and re-query.
        """
        current_board = game.get_state().board
        from_hash = hash_board(current_board)
        from_board_2d = current_board if current_board.ndim == 2 else current_board.reshape(1, -1)

        # Single batch query: fetch stats + anchor_id + bell for ALL moves from this position
        known_moves = self.batch_get_moves_from(from_hash) if hasattr(self, 'batch_get_moves_from') else {}

        anchors_with_moves: dict[int, list[tuple[np.ndarray, Stats]]] = defaultdict(list)
        anchor_stats: dict[int, Stats] = {}
        bell_scores: dict[tuple, float | None] = {}
        concept_scores: dict[tuple, float | None] = {}

        for move, to_hash, to_board in self._compute_move_hashes(game, valid_moves):
            mk = tuple(move)
            # invented-concept value of the resulting board (inert until the library is grown);
            # a discovered rule values EVERY board, including ones training never visited
            concept_scores[mk] = self.concept_library.value_for(
                to_board, _placed_token(from_board_2d, to_board))

            if to_hash in known_moves:
                direct_stats, anchor_id, bell = known_moves[to_hash]
                bell_scores[mk] = bell
                if direct_stats.total > 0:
                    aid = anchor_id if anchor_id is not None else UNEXPLORED_ANCHOR_ID
                    anchors_with_moves[aid].append((move, direct_stats))
                    if aid not in anchor_stats:
                        anchor_stats[aid] = self.get_anchor_stats_by_id(aid) if aid != UNEXPLORED_ANCHOR_ID else Stats()
                    continue

            bell_scores[mk] = None
            # Unseen transition — no stats yet; the concept score above still covers it
            anchors_with_moves[UNEXPLORED_ANCHOR_ID].append((move, Stats()))
            anchor_stats[UNEXPLORED_ANCHOR_ID] = Stats()

        return MoveEvaluation(
            anchors_with_moves=dict(anchors_with_moves),
            anchor_stats=anchor_stats,
            bell_scores=bell_scores,
            concept_scores=concept_scores,
        )

    def _compute_move_hashes(self, game: GameBase, valid_moves: list[np.ndarray]) -> list[tuple[np.ndarray, str, np.ndarray]]:
        """Generate (move, destination_hash, destination_board) triples.

        Moves come from valid_moves() so validation is skipped.
        """
        results = []
        for move in valid_moves:
            clone = game.deep_clone()
            try:
                clone.apply_move(move, validated=True)
                dest_board = clone.get_state().board
                results.append((move, hash_board(dest_board), dest_board))
            except (ValueError, IndexError):
                continue
        return results

    # -------------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------------

    def record_round(self, game_class: type, stacks: list[tuple]) -> tuple[int, int]:
        """
        Record outcomes from a batch of games — every move in a stack is
        credited with its mover's eventual outcome, flat. (A gamma-decay /
        max-ply credit knob existed here; it was never set off its neutral
        default anywhere, so it was deleted.)

        Stacks may be 2-tuples (moves, outcome) or 3-tuples
        (moves, outcome, all_outcomes) where all_outcomes is a dict mapping
        every player ID to their State for that game. The 3-tuple form
        enables cross-score recording for the alignment factor α.

        Returns:
            (transitions_written, transitions_swapped) — swap count is the
            core learning signal indicating how many beliefs changed.
        """
        if self.read_only:
            raise RuntimeError("Cannot record in read-only mode")

        from wise_explorer.games.game_state import GameState

        transitions: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        trajectory_keys: list[list[tuple[str, str]]] = []
        # Cross-score accumulator: (from_hash, to_hash, observer_role) -> [score_sum, count]
        cross_scores: dict[tuple[str, str, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
        # Board storage: hash -> (board_bytes, rows, cols) for concept invention
        boards_to_store: dict[str, tuple[bytes, int, int]] = {}

        for stack_entry in stacks:
            moves, outcome = stack_entry[0], stack_entry[1]
            all_outcomes = stack_entry[2] if len(stack_entry) > 2 else None

            outcome_idx = OUTCOME_INDEX.get(outcome, -1)
            if outcome_idx < 0:
                continue

            game = game_class()
            stack_keys: list[tuple[str, str]] = []
            for move, board, player in moves:
                from_hash = hash_board(board)
                game.set_state(GameState(board.copy(), player))
                game.apply_move(move, validated=True)
                dest_board = game.get_state().board
                to_hash = hash_board(dest_board)

                # Store both boards for concept invention (normalize to 2D)
                if from_hash not in boards_to_store:
                    fb = board if board.ndim == 2 else board.reshape(1, -1)
                    boards_to_store[from_hash] = (fb.tobytes(), fb.shape[0], fb.shape[1])
                if to_hash not in boards_to_store:
                    b = dest_board if dest_board.ndim == 2 else dest_board.reshape(1, -1)
                    boards_to_store[to_hash] = (b.tobytes(), b.shape[0], b.shape[1])

                stack_keys.append((from_hash, to_hash))
                transitions[(from_hash, to_hash)][outcome_idx] += 1.0

                # Accumulate cross-scores: each non-mover's outcome
                if all_outcomes is not None:
                    for obs_pid, obs_outcome in all_outcomes.items():
                        if obs_pid == player:
                            continue  # Skip mover's own outcome
                        obs_score = OUTCOME_SCORE.get(obs_outcome, 0.5)
                        cross_scores[(from_hash, to_hash, obs_pid)][0] += obs_score
                        cross_scores[(from_hash, to_hash, obs_pid)][1] += 1.0

            trajectory_keys.append(stack_keys)

        self._store_boards(boards_to_store)
        swaps = self._commit(transitions)
        self._record_cross_scores(cross_scores)
        self._propagate_bellman(trajectory_keys)

        return len(transitions), swaps

    def _store_boards(self, boards: dict[str, tuple[bytes, int, int]]) -> None:
        """Store board arrays for concept invention."""
        if not boards:
            return
        cur = self.conn.cursor()
        cur.executemany(
            "INSERT OR IGNORE INTO boards (board_hash, board_data, board_rows, board_cols) "
            "VALUES (?,?,?,?)",
            [(h, data, rows, cols) for h, (data, rows, cols) in boards.items()],
        )
        self.conn.commit()

    def _load_boards(self) -> dict[str, np.ndarray]:
        """Load all stored boards from the database."""
        rows = self.conn.execute(
            "SELECT board_hash, board_data, board_rows, board_cols FROM boards"
        ).fetchall()
        result = {}
        for h, data, nrows, ncols in rows:
            board = np.frombuffer(data, dtype=np.int8).reshape(nrows, ncols).copy()
            result[h] = board
        return result

    def _build_trans_scores(self, boards: dict[str, np.ndarray] | None = None) -> tuple[dict[str, np.ndarray], dict[tuple[str, str], tuple[Counts, float]]]:
        """Build the per-transition target the concept miner fits.

        The target is the minimax-propagated **Bellman** value of each transition
        — the raw win/tie/loss outcome with the prune phase's exploration noise
        filtered out by the backup — falling back to the raw outcome only for
        transitions the backup hasn't reached yet. Where Bellman is still flat
        (unconverged, as on large games) it carries no structure, so the miner's
        MDL stop finds nothing and abstains on its own; no separate "has it
        converged?" test is needed.

        Returns (boards, trans_scores) or (empty, empty) if insufficient data.
        """
        if boards is None:
            boards = self._load_boards()
        if not boards:
            return {}, {}

        try:
            rows = self.conn.execute(
                "SELECT from_hash, to_hash, wins, ties, losses, propagated_score "
                "FROM transitions"
            ).fetchall()
        except Exception:
            return {}, {}
        if not rows:
            return {}, {}

        keys: list[tuple[str, str]] = []
        counts_l: list[Counts] = []
        bell_l: list[float] = []
        for from_hash, to_hash, w, t, l, bell in rows:
            if from_hash not in boards or to_hash not in boards:
                continue
            s = Stats(w, t, l)
            if s.total <= 0:
                continue
            solo = s.utility if is_decisive(s) else s.mean_score
            keys.append((from_hash, to_hash))
            counts_l.append((w, t, l))
            bell_l.append(bell if bell is not None else solo)

        if not keys:
            return boards, {}

        trans_scores: dict[tuple[str, str], tuple[Counts, float]] = {
            key: (counts_l[i], bell_l[i]) for i, key in enumerate(keys)
        }
        return boards, trans_scores

    def grow_concepts(self, game=None) -> int:
        """Turn the value loop's wheel once: evidence → heal → distill → re-heal.

        ``solve_graph`` re-derives every value from raw counts (evidence re-anchors the
        loop each cycle), ``complete_values`` lets the library price the legal replies
        nobody has played so the backup stops trusting blind spots, and ``rebuild`` then
        fits discovery on the completed values — the system's best current belief — before
        a final heal with the rules just distilled. (Measured, seeded 8-pile Nim: fitting
        on evidence-only values instead collapses play 80→32/200 — with ~93% of positions
        never visited the un-healed backup is mostly noise and the refit shreds the
        transferred rules. Fitting on healed values holds ~99%. The guard against
        self-echo is the evidence re-anchor plus the MDL gate, not starving discovery of
        its own signal.)

        This is discovery's only venue — between calls the library is the last considered
        fit. Without a ``game`` the heals are skipped (values stay evidence-only)."""
        if self.read_only or getattr(self, "is_markov", False):
            return 0
        from wise_explorer import synthesis
        self.solve_graph()
        # one structural enumeration serves the whole boundary: the loop's beats never
        # add boards or transitions, so both healing passes share this reply graph and
        # every beat reuses its loaded boards
        graph = self.reply_graph(game) if game is not None else None
        boards = graph["boards"] if graph else None
        if game is not None and not self.concept_library.rules:
            # bootstrap: completion can't lend prices from an empty head. On the first
            # boundary (cold start, or a freshly seeded library whose rules are cleared)
            # read the notebook once to mint a provisional fit; the rebuild below then
            # re-distills properly from the completed values. (Measured: skipping this
            # leaves the first boundary's fit on raw evidence AND heals with it —
            # 87/200 on seeded 8-pile Nim vs ~199 once rules exist.)
            self.concept_library.rebuild(*synthesis._boards_values(self, boards))
        if game is not None:
            self.complete_values(game, graph)
        B, V, M = synthesis._boards_values(self, boards)
        kept = self.concept_library.rebuild(B, V, M)
        if game is not None:
            self.complete_values(game, graph)    # re-price with the rules just distilled
        return kept

    def _record_cross_scores(self, cross_scores: dict) -> None:
        """Hook for recording cross-scores. No-op for Markov memory."""
        pass

    def _propagate_bellman(self, trajectory_keys: list[list[tuple[str, str]]]) -> None:
        """Hook for Bellman propagation. No-op for Markov memory."""
        pass

    def solve_graph(self, epsilon: float = 1e-6, max_iters: int = 200) -> int:
        """Full value iteration on the game graph. No-op for Markov memory."""
        return 0

    def reply_graph(self, game):
        """Structural reply enumeration for the value loop. None for Markov memory."""
        return None

    def complete_values(self, game, graph=None) -> int:
        """Library-completed value pass. No-op for Markov memory."""
        return 0

    def _commit(self, transitions: dict[tuple[str, str], list[float]]) -> int:
        """
        Write transitions to database with incremental anchor updates.

        Returns:
            Number of transitions that swapped anchors (beliefs changed).
        """
        if not transitions:
            return 0

        cur = self.conn.cursor()
        keys, deltas = self._commit_outcomes(transitions, cur)

        # Handle initialization if needed
        if self.anchors.needs_initialization():
            units = self.collect_units()
            if units:
                _, membership = self.anchors.rebuild(units, self.key_to_repr, cur)
                self.write_anchor_ids(membership, cur)

        # Gather current stats for changed keys
        changed_stats: dict[Any, Counts] = {}
        for key in keys:
            stats = self.get_stats_by_key(key)
            if stats.total > 0:
                changed_stats[key] = stats.as_tuple()

        if changed_stats:
            existing_aids = self.batch_get_anchor_ids(list(changed_stats.keys()), cur)
            assignments, swaps = self.anchors.update(
                changed_stats, deltas, existing_aids,
                key_to_repr=self.key_to_repr, cur=cur)
            for key, aid in assignments.items():
                self.set_anchor_id(key, aid, cur)
        else:
            swaps = 0

        self.conn.commit()
        self._clear_caches()
        return swaps

    def _clear_caches(self) -> None:
        """Clear all caches."""
        self._anchor_stats_cache.clear()
        self._anchor_id_cache.clear()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        if self._closed:
            return
        self._closed = True

        self._clear_caches()
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
