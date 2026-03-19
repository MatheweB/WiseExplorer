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
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TYPE_CHECKING

import numpy as np

from wise_explorer.core.types import Stats, Counts, OUTCOME_INDEX, OUTCOME_SCORE, is_decisive
from wise_explorer.core.hashing import hash_board
from wise_explorer.core.bayes import compatible
from wise_explorer.memory.anchor_manager import AnchorManager
from wise_explorer.memory.predicates import PredicateLibrary, TORCH_AVAILABLE
from wise_explorer.memory.iti_miner import ITIMiner
from wise_explorer.memory.tree_miner import TreeMiner

if TYPE_CHECKING:
    from wise_explorer.agent.agent import State
    from wise_explorer.games.game_base import GameBase
UNEXPLORED_ANCHOR_ID = -999
PREDICATE_ANCHOR_ID = -998


class MoveEvaluation(NamedTuple):
    """Result of evaluate_moves: moves grouped by anchor with anchor stats.

    Also carries bell and predicate scores so selection doesn't need to
    re-clone games and re-query the DB for the same information.
    """
    anchors_with_moves: Dict[int, List[Tuple[np.ndarray, Stats]]]
    anchor_stats: Dict[int, Stats]
    bell_scores: Dict[tuple, Optional[float]]  # move_key -> propagated_score
    pred_scores: Dict[tuple, Optional[float]]  # move_key -> predicate mean_score


class GameMemory(ABC):
    """Abstract base for game memory implementations."""

    main_table: str  # Subclasses define: "transitions" or "states"
    is_markov: bool  # Subclasses define: False for Transition, True for Markov

    def __init__(self, db_path: str | Path, read_only: bool = False,
                 gamma: float = 1.0, max_ply: Optional[int] = None):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.read_only = read_only
        self.gamma = gamma
        self.max_ply = max_ply
        self._closed = False

        self._anchor_stats_cache: Dict[int, Stats] = {}
        self._anchor_id_cache: Dict[Any, Optional[int]] = {}

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA cache_size=-65536")
        self.conn.execute("PRAGMA synchronous=NORMAL")

        if not read_only:
            self.conn.executescript(self._schema())
            self.conn.commit()

        self.anchors = AnchorManager(self.conn, self.main_table, self.read_only)
        self.predicate_library = PredicateLibrary(self.conn, self.read_only)
        self._batch_miner = TreeMiner()   # batch CART for end-of-training
        self._iti_miner = ITIMiner()       # incremental for per-wave updates
        self._cached_trans_scores: Dict[Tuple[str, str], Tuple[Counts, float]] = {}
        self._cached_boards: Dict[str, np.ndarray] = {}  # board cache (append-only)
        self._last_wave_keys: List[Tuple[str, str]] = []  # transitions touched last wave

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
    def _fetch_anchor_id(self, from_hash: str, to_hash: str) -> Optional[int]:
        """Fetch anchor ID from database."""
        pass

    @abstractmethod
    def batch_get_anchor_ids(self, keys: List, cur: sqlite3.Cursor) -> Dict:
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
    def collect_units(self) -> List[Tuple]:
        """Collect all units as (key, counts) tuples for rebuild."""
        pass

    @abstractmethod
    def write_anchor_ids(self, membership: Dict, cur: sqlite3.Cursor) -> None:
        """Batch write anchor IDs after rebuild."""
        pass

    @abstractmethod
    def _commit_outcomes(self, transitions: Dict[Tuple[str, str], List[float]], cur: sqlite3.Cursor) -> Tuple[List, Dict]:
        """Commit outcomes and return (keys, deltas) for anchor update."""
        pass

    @abstractmethod
    def _aggregate_destination_scores(self) -> Dict[str, Counts]:
        """Aggregate scores per destination board hash for predicate mining."""
        pass

    def _get_destination_bellman_scores(self) -> Dict[str, float]:
        """Get average Bellman score per destination hash. Override in subclasses with Bellman."""
        return {}

    def _get_transition_from_hashes(self) -> Dict[str, str]:
        """Get the most common from_hash for each to_hash. Override in subclasses."""
        return {}

    @abstractmethod
    def _get_mode_specific_info(self) -> Dict[str, Any]:
        """Return mode-specific info for get_info()."""
        pass

    # -------------------------------------------------------------------------
    # Anchor Queries (shared implementation)
    # -------------------------------------------------------------------------

    def get_anchor_id(self, from_hash: str, to_hash: str) -> Optional[int]:
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

    def get_anchor_details(self) -> List[Dict[str, Any]]:
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

    def get_info(self) -> Dict[str, Any]:
        """Get summary statistics."""
        anchors = self.conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
        return {
            "anchors": anchors,
            "predicates": self.predicate_library.count,
            **self._get_mode_specific_info(),
        }

    # -------------------------------------------------------------------------
    # Move Evaluation
    # -------------------------------------------------------------------------

    def evaluate_moves(self, game: "GameBase", valid_moves: List[np.ndarray]) -> MoveEvaluation:
        """Evaluate all valid moves and group by anchor.

        Uses a single batch DB query per position instead of 3 queries per move.
        Also collects bell and predicate scores so selection doesn't need to
        re-clone games and re-query.
        """
        current_board = game.get_state().board
        from_hash = hash_board(current_board)
        from_board_2d = current_board if current_board.ndim == 2 else current_board.reshape(1, -1)

        # Single batch query: fetch stats + anchor_id + bell for ALL moves from this position
        known_moves = self.batch_get_moves_from(from_hash) if hasattr(self, 'batch_get_moves_from') else {}

        anchors_with_moves: Dict[int, List[Tuple[np.ndarray, Stats]]] = defaultdict(list)
        anchor_stats: Dict[int, Stats] = {}
        bell_scores: Dict[tuple, Optional[float]] = {}
        pred_scores: Dict[tuple, Optional[float]] = {}

        for move, to_hash, to_board in self._compute_move_hashes(game, valid_moves):
            mk = tuple(move)

            if to_hash in known_moves:
                direct_stats, anchor_id, bell = known_moves[to_hash]
                bell_scores[mk] = bell
                if direct_stats.total > 0:
                    aid = anchor_id if anchor_id is not None else UNEXPLORED_ANCHOR_ID
                    anchors_with_moves[aid].append((move, direct_stats))
                    if aid not in anchor_stats:
                        anchor_stats[aid] = self.get_anchor_stats_by_id(aid) if aid != UNEXPLORED_ANCHOR_ID else Stats()

                    # Also check predicate for the 4th signal
                    board_2d = to_board if to_board.ndim == 2 else to_board.reshape(1, -1)
                    ps = self.predicate_library.match(board_2d, from_board_2d)
                    if ps is not None:
                        a_rate = anchor_stats[aid].mean_score if aid in anchor_stats else 0.5
                        pred_scores[mk] = ps.utility if is_decisive(ps, a_rate) else ps.mean_score
                    else:
                        pred_scores[mk] = None
                    continue

            bell_scores[mk] = None

            # Unseen transition — check predicate library for a prior
            board_2d = to_board if to_board.ndim == 2 else to_board.reshape(1, -1)
            pred_stats = self.predicate_library.match(board_2d, from_board_2d)
            if pred_stats is not None:
                anchors_with_moves[PREDICATE_ANCHOR_ID].append((move, pred_stats))
                if PREDICATE_ANCHOR_ID not in anchor_stats:
                    anchor_stats[PREDICATE_ANCHOR_ID] = pred_stats
                pred_scores[mk] = pred_stats.utility if is_decisive(pred_stats) else pred_stats.mean_score
            else:
                anchors_with_moves[UNEXPLORED_ANCHOR_ID].append((move, Stats()))
                anchor_stats[UNEXPLORED_ANCHOR_ID] = Stats()
                pred_scores[mk] = None

        return MoveEvaluation(
            anchors_with_moves=dict(anchors_with_moves),
            anchor_stats=anchor_stats,
            bell_scores=bell_scores,
            pred_scores=pred_scores,
        )

    def _compute_move_hashes(self, game: "GameBase", valid_moves: List[np.ndarray]) -> List[Tuple[np.ndarray, str, np.ndarray]]:
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

    def record_round(self, game_class: type, stacks: List[Tuple]) -> Tuple[int, int]:
        """
        Record outcomes from a batch of games with reverse n-ply credit.

        Each move in a player's stack receives geometrically decaying credit
        based on its distance from the terminal state:

            weight = gamma ^ depth_from_end

        where depth_from_end = (stack_length - 1 - position). The last move
        always receives full credit (gamma^0 = 1). Earlier moves receive
        exponentially less, filtering noise from causally distant decisions.

        With gamma=1.0 (default), this reproduces the original flat credit.

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

        gamma = self.gamma
        max_ply = self.max_ply
        transitions: Dict[Tuple[str, str], List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        trajectory_keys: List[List[Tuple[str, str]]] = []
        # Cross-score accumulator: (from_hash, to_hash, observer_role) -> [score_sum, count]
        cross_scores: Dict[Tuple[str, str, int], List[float]] = defaultdict(lambda: [0.0, 0.0])
        # Board storage: hash -> (board_bytes, rows, cols) for predicate mining
        boards_to_store: Dict[str, Tuple[bytes, int, int]] = {}

        for stack_entry in stacks:
            moves, outcome = stack_entry[0], stack_entry[1]
            all_outcomes = stack_entry[2] if len(stack_entry) > 2 else None

            outcome_idx = OUTCOME_INDEX.get(outcome, -1)
            if outcome_idx < 0:
                continue

            k = len(moves)
            game = game_class()
            stack_keys: List[Tuple[str, str]] = []
            for i, (move, board, player) in enumerate(moves):
                depth_from_end = k - 1 - i

                from_hash = hash_board(board)
                game.set_state(GameState(board.copy(), player))
                game.apply_move(move, validated=True)
                dest_board = game.get_state().board
                to_hash = hash_board(dest_board)

                # Store both boards for predicate mining (normalize to 2D)
                if from_hash not in boards_to_store:
                    fb = board if board.ndim == 2 else board.reshape(1, -1)
                    boards_to_store[from_hash] = (fb.tobytes(), fb.shape[0], fb.shape[1])
                if to_hash not in boards_to_store:
                    b = dest_board if dest_board.ndim == 2 else dest_board.reshape(1, -1)
                    boards_to_store[to_hash] = (b.tobytes(), b.shape[0], b.shape[1])

                stack_keys.append((from_hash, to_hash))

                if max_ply is not None and depth_from_end >= max_ply:
                    continue  # Skip moves beyond ply cap for credit

                weight = gamma ** depth_from_end
                transitions[(from_hash, to_hash)][outcome_idx] += weight

                # Accumulate cross-scores: each non-mover's outcome
                if all_outcomes is not None:
                    for obs_pid, obs_outcome in all_outcomes.items():
                        if obs_pid == player:
                            continue  # Skip mover's own outcome
                        obs_score = OUTCOME_SCORE.get(obs_outcome, 0.5)
                        cross_scores[(from_hash, to_hash, obs_pid)][0] += weight * obs_score
                        cross_scores[(from_hash, to_hash, obs_pid)][1] += weight

            trajectory_keys.append(stack_keys)

        self._store_boards(boards_to_store)
        swaps = self._commit(transitions)
        self._record_cross_scores(cross_scores)
        self._propagate_bellman(trajectory_keys)

        # Track which transitions were touched for incremental mining
        self._last_wave_keys = list(transitions.keys())

        return len(transitions), swaps

    def _store_boards(self, boards: Dict[str, Tuple[bytes, int, int]]) -> None:
        """Store board arrays for predicate mining."""
        if not boards:
            return
        cur = self.conn.cursor()
        cur.executemany(
            "INSERT OR IGNORE INTO boards (board_hash, board_data, board_rows, board_cols) "
            "VALUES (?,?,?,?)",
            [(h, data, rows, cols) for h, (data, rows, cols) in boards.items()],
        )
        self.conn.commit()

    def _load_boards(self) -> Dict[str, np.ndarray]:
        """Load all stored boards from the database."""
        rows = self.conn.execute(
            "SELECT board_hash, board_data, board_rows, board_cols FROM boards"
        ).fetchall()
        result = {}
        for h, data, nrows, ncols in rows:
            board = np.frombuffer(data, dtype=np.int8).reshape(nrows, ncols).copy()
            result[h] = board
        return result

    def _load_new_boards(self) -> None:
        """Load only boards not already in cache (boards table is append-only)."""
        if not self._cached_boards:
            # First call: load everything
            self._cached_boards = self._load_boards()
            return
        # Only load new boards by checking which hashes from last wave are missing
        needed = set()
        for fh, th in self._last_wave_keys:
            if fh not in self._cached_boards:
                needed.add(fh)
            if th not in self._cached_boards:
                needed.add(th)
        if not needed:
            return
        for h in needed:
            row = self.conn.execute(
                "SELECT board_data, board_rows, board_cols FROM boards WHERE board_hash=?",
                (h,),
            ).fetchone()
            if row:
                data, nrows, ncols = row
                self._cached_boards[h] = np.frombuffer(
                    data, dtype=np.int8,
                ).reshape(nrows, ncols).copy()

    def _build_trans_scores(self) -> Tuple[Dict[str, np.ndarray], Dict[Tuple[str, str], Tuple[Counts, float]]]:
        """Load transitions and build per-transition scores with per-from-board signal selection.

        Returns (boards, trans_scores) or (empty, empty) if insufficient data.
        """
        boards = self._load_boards()
        if not boards:
            return {}, {}

        try:
            rows = self.conn.execute(
                "SELECT from_hash, to_hash, wins, ties, losses, "
                "propagated_score, anchor_id "
                "FROM transitions"
            ).fetchall()
        except Exception:
            return {}, {}

        if not rows:
            return {}, {}

        import numpy as _np

        # Load anchor stats for anchor_mean signal
        anchor_stats: Dict[int, float] = {}
        try:
            anchor_rows = self.conn.execute(
                "SELECT anchor_id, wins, ties, losses FROM anchors"
            ).fetchall()
            for aid, aw, at, al in anchor_rows:
                s = Stats(aw, at, al)
                anchor_stats[aid] = (s.utility if is_decisive(s) else s.mean_score) if s.total > 0 else 0.5
        except Exception:
            pass

        # Group transitions by from_hash with all four signals
        # (bell, anchor, solo, pred). Pred comes from the predicate library —
        # if predicates have learned structural patterns that discriminate
        # better than raw scores, they win the variance ranking and guide
        # mining toward better splits. Bell eventually overrides as it
        # converges, so any wrong predicates self-correct.
        from_groups: Dict[str, list] = {}
        for from_hash, to_hash, w, t, l, bell, anchor_id in rows:
            if to_hash not in boards or from_hash not in boards:
                continue
            s = Stats(w, t, l)
            if s.total <= 0:
                continue
            anchor_mean = anchor_stats.get(anchor_id, s.mean_score) if anchor_id is not None else s.mean_score
            solo_mean = s.utility if is_decisive(s, anchor_mean) else s.mean_score

            # Predicate score: structural prior from the predicate library
            pred_score = None
            if self.predicate_library.count > 0:
                to_2d = boards[to_hash] if boards[to_hash].ndim == 2 else boards[to_hash].reshape(1, -1)
                from_2d = boards[from_hash] if boards[from_hash].ndim == 2 else boards[from_hash].reshape(1, -1)
                pred_stats = self.predicate_library.match(to_2d, from_2d)
                if pred_stats is not None:
                    pred_score = pred_stats.utility if is_decisive(pred_stats, anchor_mean) else pred_stats.mean_score

            if from_hash not in from_groups:
                from_groups[from_hash] = []
            from_groups[from_hash].append((to_hash, (w, t, l), bell, anchor_mean, solo_mean, pred_score))

        # Per-from-board: rank all 4 signals by variance (matches selection)
        trans_scores: Dict[Tuple[str, str], Tuple[Counts, float]] = {}
        for from_hash, transitions in from_groups.items():
            bell_vals = [t[2] for t in transitions if t[2] is not None]
            anchor_vals = [t[3] for t in transitions]
            solo_vals = [t[4] for t in transitions]
            pred_vals = [t[5] for t in transitions if t[5] is not None]

            variances = {
                "bell": float(_np.var(bell_vals)) if len(bell_vals) >= 2 else 0.0,
                "anchor": float(_np.var(anchor_vals)) if len(anchor_vals) >= 2 else 0.0,
                "solo": float(_np.var(solo_vals)) if len(solo_vals) >= 2 else 0.0,
                "pred": float(_np.var(pred_vals)) if len(pred_vals) >= 2 else 0.0,
            }
            # Signals need sufficient coverage
            if len(bell_vals) <= len(transitions) * 0.5:
                variances["bell"] = 0.0
            if len(pred_vals) <= len(transitions) * 0.5:
                variances["pred"] = 0.0

            best_signal = max(variances, key=variances.get)

            for to_hash, counts, bell, anchor_mean, solo_mean, pred_score in transitions:
                if best_signal == "bell" and bell is not None:
                    score = bell
                elif best_signal == "anchor":
                    score = anchor_mean
                elif best_signal == "pred" and pred_score is not None:
                    score = pred_score
                else:
                    score = solo_mean
                trans_scores[(from_hash, to_hash)] = (counts, score)

        return boards, trans_scores

    def mine_predicates(self, incremental: bool = False) -> int:
        """Discover structural predicates from stored transitions.

        Args:
            incremental: If True, use ITI for fast per-wave update (~0.5ms).
                         If False, use batch CART for full rebuild (~8ms).

        Mines per-transition (from→to pairs), preserving implicit player
        identity. Signal selection (bell vs mean) is per-from-board,
        matching how move selection works.

        Returns:
            Number of predicates discovered.
        """
        if self.read_only:
            raise RuntimeError("Cannot mine predicates in read-only mode")

        if incremental:
            # Per-wave: only query the transitions touched this wave.
            # Update the caches incrementally, then pass to ITI
            # (ITI internally skips already-known transitions).
            if not self._last_wave_keys:
                return 0
            self._load_new_boards()
            self._update_trans_cache(self._cached_boards)
            if not self._cached_trans_scores:
                return 0
            predicates = self._iti_miner.mine(
                self._cached_boards, self._cached_trans_scores,
                prune=False, wave_keys=self._last_wave_keys,
            )
        else:
            # End-of-training: full rebuild for compact output.
            boards, trans_scores = self._build_trans_scores()
            if not trans_scores:
                return 0
            if self._iti_miner._root is not None:
                predicates = self._iti_miner.mine(boards, trans_scores, prune=True)
            else:
                predicates = self._batch_miner.mine(boards, trans_scores)
        self.predicate_library.save(predicates)
        return len(predicates)

    def _update_trans_cache(self, boards: Dict[str, np.ndarray]) -> None:
        """Incrementally update cached transition scores for touched keys only."""
        import numpy as _np

        if not self._last_wave_keys:
            return

        # Query only the touched transitions
        touched = set(self._last_wave_keys)
        placeholders = ",".join(["(?,?)"] * len(touched))
        params = [v for k in touched for v in k]

        try:
            rows = self.conn.execute(
                f"SELECT from_hash, to_hash, wins, ties, losses, propagated_score "
                f"FROM transitions WHERE (from_hash, to_hash) IN (VALUES {placeholders})",
                params,
            ).fetchall()
        except Exception:
            # Fallback: query all (some SQLite versions don't support VALUES)
            rows = self.conn.execute(
                "SELECT from_hash, to_hash, wins, ties, losses, propagated_score "
                "FROM transitions"
            ).fetchall()

        # Group touched transitions by from_hash for signal selection
        from_groups: Dict[str, list] = {}
        for from_hash, to_hash, w, t, l, bell in rows:
            if to_hash not in boards or from_hash not in boards:
                continue
            s = Stats(w, t, l)
            if s.total <= 0:
                continue
            if from_hash not in from_groups:
                from_groups[from_hash] = []
            from_groups[from_hash].append((to_hash, (w, t, l), bell, s.mean_score))

        for from_hash, transitions in from_groups.items():
            bell_vals = [t[2] for t in transitions if t[2] is not None]
            mean_vals = [t[3] for t in transitions]
            bell_var = float(_np.var(bell_vals)) if len(bell_vals) >= 2 else 0.0
            mean_var = float(_np.var(mean_vals)) if len(mean_vals) >= 2 else 0.0
            use_bell = bell_var > mean_var and len(bell_vals) > len(transitions) * 0.5

            for to_hash, counts, bell, mean in transitions:
                score = bell if (use_bell and bell is not None) else mean
                self._cached_trans_scores[(from_hash, to_hash)] = (counts, score)

    def _record_cross_scores(self, cross_scores: Dict) -> None:
        """Hook for recording cross-scores. No-op for Markov memory."""
        pass

    def _propagate_bellman(self, trajectory_keys: List[List[Tuple[str, str]]]) -> None:
        """Hook for Bellman propagation. No-op for Markov memory."""
        pass

    def _commit(self, transitions: Dict[Tuple[str, str], List[float]]) -> int:
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
        changed_stats: Dict[Any, Counts] = {}
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
