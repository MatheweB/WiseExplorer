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

from wise_explorer.core.types import Stats, Counts, OUTCOME_INDEX
from wise_explorer.core.hashing import hash_board
from wise_explorer.core.bayes import compatible
from wise_explorer.memory.anchor_manager import AnchorManager

if TYPE_CHECKING:
    from wise_explorer.agent.agent import State
    from wise_explorer.games.game_base import GameBase
UNEXPLORED_ANCHOR_ID = -999


class MoveEvaluation(NamedTuple):
    """Result of evaluate_moves: moves grouped by anchor with anchor stats."""
    anchors_with_moves: Dict[int, List[Tuple[np.ndarray, Stats]]]
    anchor_stats: Dict[int, Stats]


class GameMemory(ABC):
    """Abstract base for game memory implementations."""

    main_table: str  # Subclasses define: "transitions" or "states"
    is_markov: bool  # Subclasses define: False for Transition, True for Markov

    def __init__(self, db_path: str | Path, read_only: bool = False):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.read_only = read_only
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

        self._anchors = AnchorManager(self)

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
    def _get_stats_by_key(self, key) -> Stats:
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
    def _batch_get_anchor_ids(self, keys: List, cur: sqlite3.Cursor) -> Dict:
        """Batch fetch anchor IDs for keys."""
        pass

    @abstractmethod
    def _set_anchor_id(self, key, anchor_id: int, cur: sqlite3.Cursor) -> None:
        """Set anchor_id for a key in the main table."""
        pass

    @abstractmethod
    def _key_to_repr(self, key) -> str:
        """Convert a key to string representation for debugging."""
        pass

    @abstractmethod
    def _collect_units(self) -> List[Tuple]:
        """Collect all units as (key, counts) tuples for rebuild."""
        pass

    @abstractmethod
    def _write_anchor_ids(self, membership: Dict, cur: sqlite3.Cursor) -> None:
        """Batch write anchor IDs after rebuild."""
        pass

    @abstractmethod
    def _commit_outcomes(self, transitions: Dict[Tuple[str, str], List[int]], cur: sqlite3.Cursor) -> Tuple[List, Dict]:
        """Commit outcomes and return (keys, deltas) for anchor update."""
        pass

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
        return self._anchors.get_details()

    def rebuild_anchors(self) -> int:
        """Full rebuild of anchor clustering."""
        return self._anchors.rebuild()

    def consolidate_anchors(self) -> int:
        """Merge compatible anchors."""
        return self._anchors.consolidate()

    # -------------------------------------------------------------------------
    # Info
    # -------------------------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        """Get summary statistics."""
        anchors = self.conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
        return {"anchors": anchors, **self._get_mode_specific_info()}

    # -------------------------------------------------------------------------
    # Move Evaluation
    # -------------------------------------------------------------------------

    def evaluate_moves(self, game: "GameBase", valid_moves: List[np.ndarray]) -> MoveEvaluation:
        """Evaluate all valid moves and group by anchor."""
        from_hash = hash_board(game.get_state().board)

        anchors_with_moves: Dict[int, List[Tuple[np.ndarray, Stats]]] = defaultdict(list)
        anchor_stats: Dict[int, Stats] = {}

        for move, to_hash in self._compute_move_hashes(game, valid_moves):
            direct_stats = self.get_move_stats(from_hash, to_hash)

            if direct_stats.total > 0:
                anchor_id = self.get_anchor_id(from_hash, to_hash)
                aid = anchor_id if anchor_id is not None else UNEXPLORED_ANCHOR_ID
                anchors_with_moves[aid].append((move, direct_stats))

                if aid not in anchor_stats:
                    anchor_stats[aid] = self.get_anchor_stats_by_id(aid) if aid != UNEXPLORED_ANCHOR_ID else Stats()
            else:
                anchors_with_moves[UNEXPLORED_ANCHOR_ID].append((move, Stats()))
                anchor_stats[UNEXPLORED_ANCHOR_ID] = Stats()

        return MoveEvaluation(anchors_with_moves=dict(anchors_with_moves), anchor_stats=anchor_stats)

    def _compute_move_hashes(self, game: "GameBase", valid_moves: List[np.ndarray]) -> List[Tuple[np.ndarray, str]]:
        """Generate (move, destination_hash) pairs.

        Moves come from valid_moves() so validation is skipped.
        """
        results = []
        for move in valid_moves:
            clone = game.deep_clone()
            try:
                clone.apply_move(move, validated=True)
                results.append((move, hash_board(clone.get_state().board)))
            except (ValueError, IndexError):
                continue
        return results

    # -------------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------------

    def record_round(self, game_class: type, stacks: List[Tuple[List[Tuple[Any, np.ndarray, int]], "State"]]) -> Tuple[int, int]:
        """
        Record outcomes from a batch of games.
        
        Returns:
            (transitions_written, transitions_swapped) — swap count is the
            core learning signal indicating how many beliefs changed.
        """
        if self.read_only:
            raise RuntimeError("Cannot record in read-only mode")

        from wise_explorer.games.game_state import GameState

        transitions: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0, 0])

        for moves, outcome in stacks:
            outcome_idx = OUTCOME_INDEX.get(outcome, -1)
            if outcome_idx < 0:
                continue

            game = game_class()
            for move, board, player in moves:
                from_hash = hash_board(board)
                game.set_state(GameState(board.copy(), player))
                game.apply_move(move, validated=True)
                to_hash = hash_board(game.get_state().board)
                transitions[(from_hash, to_hash)][outcome_idx] += 1

        swaps = self._commit(transitions)
        return len(transitions), swaps

    def _commit(self, transitions: Dict[Tuple[str, str], List[int]]) -> int:
        """
        Write transitions to database with incremental anchor updates.
        
        Returns:
            Number of transitions that swapped anchors (beliefs changed).
        """
        if not transitions:
            return 0

        cur = self.conn.cursor()
        keys, deltas = self._commit_outcomes(transitions, cur)
        swaps = self._anchors.update(keys, deltas, cur)
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
