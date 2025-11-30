"""
Game memory manager with orbit-based state canonicalization.

Provides persistent storage and retrieval of game positions, moves, and statistics
using SQLite with automatic state canonicalization for position-independent learning.
"""

from __future__ import annotations

import json
import hashlib
import logging
from typing import Protocol
from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path

import numpy as np
from peewee import (
    Model, IntegerField, BlobField, TextField, SqliteDatabase, CompositeKey
)

from omnicron.serializers import serialize_array, deserialize_array ###
from agent.agent import State
from omnicron.state_canonicalizer import canonicalize_state, CanonicalResult ###
from omnicron.debug_viz import render_debug ###


# ============================================================================
# Configuration & Logging
# ============================================================================

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# ============================================================================
# Type Definitions
# ============================================================================

class GameState(Protocol):
    """Protocol for game states compatible with memory system."""
    board: np.ndarray
    current_player: int


@dataclass(frozen=True)
class MoveStatistics:
    """Statistics for a single move from a position."""
    
    wins: int
    ties: int
    neutral: int
    losses: int
    
    @property
    def total(self) -> int:
        return self.wins + self.ties + self.neutral + self.losses
    
    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total > 0 else 0.0
    
    @property
    def tie_rate(self) -> float:
        return self.ties / self.total if self.total > 0 else 0.0
    
    @property
    def loss_rate(self) -> float:
        return self.losses / self.total if self.total > 0 else 0.0
    
    @property
    def utility(self) -> float:
        """Expected utility: P(win) + 0.5 * P(tie)."""
        if self.total == 0:
            return 0.5
        return self.win_rate + 0.5 * self.tie_rate
    
    @property
    def certainty(self) -> float:
        """Confidence measure based on outcome distribution."""
        if self.total == 0:
            return 0.0
        neutral_rate = self.neutral / self.total
        return (self.win_rate + self.tie_rate + neutral_rate) - self.loss_rate
    
    @property
    def score(self) -> float:
        """Combined score: utility weighted by certainty."""
        return self.utility * self.certainty if self.total > 0 else 0.5
    
    def flip_perspective(self) -> MoveStatistics:
        """Return statistics from opponent's perspective."""
        return MoveStatistics(
            wins=self.losses,
            ties=self.ties,
            neutral=self.neutral,
            losses=self.wins,
        )


@dataclass(frozen=True)
class MoveEvaluation:
    """Complete evaluation of a move from a position."""
    
    move: np.ndarray
    move_hash: str
    stats: MoveStatistics


# ============================================================================
# LRU Cache
# ============================================================================

class LRUCache:
    """Simple LRU cache with O(1) operations."""
    
    def __init__(self, maxsize: int = 4096):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, CanonicalResult] = OrderedDict()
    
    def get(self, key: str) -> CanonicalResult | None:
        if key not in self._cache:
            return None
        value = self._cache.pop(key)
        self._cache[key] = value  # Move to end
        return value
    
    def put(self, key: str, value: CanonicalResult) -> None:
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
    
    def clear(self) -> None:
        self._cache.clear()


# ============================================================================
# Database Schema
# ============================================================================

db = SqliteDatabase(
    None,
    pragmas={
        'journal_mode': 'WAL',
        'cache_size': -1024 * 64,
        'synchronous': 'NORMAL',
        'mmap_size': 1024 * 1024 * 128,
        'temp_store': 'MEMORY',
    }
)


class BaseModel(Model):
    class Meta:
        database = db


class Position(BaseModel):
    """Canonical game positions."""
    
    board_hash = TextField(primary_key=True)
    game_id = TextField()
    board_bytes = BlobField()
    original_board_bytes = BlobField()
    meta_json = TextField()
    
    class Meta:
        table_name = "positions"
        indexes = ((("game_id", "board_hash"), False),)


class PlayStats(BaseModel):
    """Move statistics from specific positions."""
    
    game_id = TextField()
    board_hash = TextField()
    move_hash = TextField()
    snapshot_player = IntegerField()
    acting_player = IntegerField()
    
    win_count = IntegerField(default=0)
    tie_count = IntegerField(default=0)
    neutral_count = IntegerField(default=0)
    loss_count = IntegerField(default=0)
    
    class Meta:
        table_name = "play_stats"
        primary_key = CompositeKey(
            "game_id", "board_hash", "move_hash", "snapshot_player", "acting_player"
        )
        indexes = ((("game_id", "board_hash", "snapshot_player"), False),)


# ============================================================================
# Game Memory Manager
# ============================================================================
class GameMemory:
    """
    Persistent memory system for game learning with orbit-based canonicalization.

    Automatically canonicalizes game states and moves to enable learning from
    symmetric positions (rotations, reflections, etc.). Equivalent moves are
    consolidated into single statistics entries.

    Caching strategy:
      - LRU cache keyed by a stable state key (board bytes + player + shape)
      - Additional mapping self._canonical_by_hash: canonical.hash -> CanonicalResult
        for quick lookup of canonical mapping by hash (useful during inversion).
    """
    
    def __init__(
        self,
        db_path: str | Path = "memory.db",
        cache_size: int = 4096,
    ):
        self.db_path = Path(db_path)
        self._cache = LRUCache(maxsize=cache_size)
        # Additional mapping for direct canonical.hash -> CanonicalResult
        self._canonical_by_hash: dict[str, CanonicalResult] = {}
        
        db.init(str(self.db_path))
        db.connect(reuse_if_open=True)
        db.create_tables([Position, PlayStats])
        
        logger.info(f"GameMemory initialized: {self.db_path} (cache_size={cache_size})")
    
    # ========================================================================
    # Writing Game Results
    # ========================================================================
    
    def record_outcome(
        self,
        game_id: str,
        state: GameState,
        move: np.ndarray,
        acting_player: int,
        outcome: State,
    ) -> bool:
        """
        Record a move and its outcome from a game.

        canonicalize_state(state) -> CanonicalResult
        canonical.canonicalize_move(move) -> 'sig_<hex>:<index>' or str(move) for unknown
        store move_hash as that canonical signature in play_stats
        """
        if state is None:
            logger.warning("Skipping record: state is None")
            return False
        
        game_id = self._normalize_game_id(game_id)
        
        try:
            canonical = self._get_canonical_cached(state)
        except Exception as e:
            logger.error(f"Canonicalization failed: {e}", exc_info=True)
            return False
        
        if not self._store_position(game_id, canonical, state.board):
            return False
        
        try:
            canonical_move_sig = canonical.canonicalize_move(move)
            # canonical_move_sig is either 'sig_<hex>:<index>' for cell moves
            # or str(move) for abstract/unknown moves
            move_hash = self._hash_canonical_move(canonical_move_sig)
        except Exception as e:
            logger.error(f"Move canonicalization failed: {e}", exc_info=True)
            return False
        
        return self._update_statistics(
            game_id=game_id,
            board_hash=canonical.hash,
            move_hash=move_hash,
            snapshot_player=int(state.current_player),
            acting_player=int(acting_player),
            outcome=outcome,
        )
    
    def _store_position(
        self,
        game_id: str,
        canonical: CanonicalResult,
        original_board: np.ndarray,
    ) -> bool:
        """Store canonical position in database."""
        try:
            board_bytes, dtype, dtype_str, shape = serialize_array(original_board)
            meta = self._make_meta_json(dtype, dtype_str, shape)
            
            db.execute_sql(
                """
                INSERT OR IGNORE INTO positions 
                (board_hash, game_id, board_bytes, original_board_bytes, meta_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (canonical.hash, game_id, board_bytes, board_bytes, meta)
            )

            # Ensure canonical mapping is cached by hash
            self._canonical_by_hash[canonical.hash] = canonical
            return True
        except Exception as e:
            logger.error(f"Failed to store position: {e}", exc_info=True)
            return False
    
    def _update_statistics(
        self,
        game_id: str,
        board_hash: str,
        move_hash: str,
        snapshot_player: int,
        acting_player: int,
        outcome: State,
    ) -> bool:
        """Update play statistics for a move."""
        try:
            counts = [0, 0, 0, 0]  # [wins, ties, neutral, losses]
            outcome_map = {
                State.WIN: 0,
                State.TIE: 1,
                State.NEUTRAL: 2,
                State.LOSS: 3,
            }
            if outcome in outcome_map:
                counts[outcome_map[outcome]] = 1
            
            db.execute_sql(
                """
                INSERT INTO play_stats
                (game_id, board_hash, move_hash, snapshot_player, acting_player,
                 win_count, tie_count, neutral_count, loss_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id, board_hash, move_hash, snapshot_player, acting_player)
                DO UPDATE SET
                    win_count = win_count + excluded.win_count,
                    tie_count = tie_count + excluded.tie_count,
                    neutral_count = neutral_count + excluded.neutral_count,
                    loss_count = loss_count + excluded.loss_count
                """,
                (game_id, board_hash, move_hash, snapshot_player, acting_player, *counts)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update statistics: {e}", exc_info=True)
            return False
    
    # ========================================================================
    # Retrieving Best/Worst Moves
    # ========================================================================
    
    def get_best_move(
        self,
        game_id: str,
        state: GameState,
        debug: bool = False,
    ) -> np.ndarray | None:
        """Retrieve the best known move for a position."""
        return self._select_move(game_id, state, select_best=True, debug=debug)
    
    def get_worst_move(
        self,
        game_id: str,
        state: GameState,
        debug: bool = False,
    ) -> np.ndarray | None:
        """Retrieve the worst known move for a position."""
        return self._select_move(game_id, state, select_best=False, debug=debug)
    
    def _select_move(
        self,
        game_id: str,
        state: GameState,
        select_best: bool,
        debug: bool,
    ) -> np.ndarray | None:
        """Internal move selection logic."""
        game_id = self._normalize_game_id(game_id)
        
        try:
            canonical = self._get_canonical_cached(state)
        except Exception as e:
            logger.error(f"Canonicalization failed during select: {e}", exc_info=True)
            return None
        
        evaluations = self._fetch_move_evaluations(
            game_id=game_id,
            board_hash=canonical.hash,
            snapshot_player=int(state.current_player),
            canonical=canonical,
        )
        
        if not evaluations:
            return None
        
        evaluations.sort(key=lambda e: e.stats.score, reverse=select_best)
        best_eval = evaluations[0]
        
        if debug:
            self._visualize_moves(state.board, evaluations, best_eval.move_hash)
        
        return best_eval.move
    
    def _fetch_move_evaluations(
        self,
        game_id: str,
        board_hash: str,
        snapshot_player: int,
        canonical: CanonicalResult,
    ) -> list[MoveEvaluation]:
        """Fetch and evaluate all moves from a position."""
        cursor = db.execute_sql(
            """
            SELECT move_hash, acting_player, win_count, tie_count, neutral_count, loss_count
            FROM play_stats
            WHERE game_id = ? AND board_hash = ? AND snapshot_player = ?
            """,
            (game_id, board_hash, snapshot_player)
        )
        
        evaluations = []
        for row in cursor.fetchall():
            try:
                evaluation = self._make_evaluation(
                    move_hash=row[0],
                    acting_player=int(row[1]),
                    snapshot_player=snapshot_player,
                    stats_tuple=(int(row[2]), int(row[3]), int(row[4]), int(row[5])),
                    canonical=canonical,
                )
                if evaluation:
                    evaluations.append(evaluation)
            except Exception as e:
                logger.warning(f"Failed to evaluate move {row[0]}: {e}")
        
        return evaluations
    
    def _make_evaluation(
        self,
        move_hash: str,
        acting_player: int,
        snapshot_player: int,
        stats_tuple: tuple[int, int, int, int],
        canonical: CanonicalResult,
    ) -> MoveEvaluation | None:
        """Create MoveEvaluation from database row."""
        # Parse canonical move from hash
        canonical_move = self._parse_canonical_move(move_hash)
        if canonical_move is None:
            logger.debug(f"Could not parse move hash: {move_hash} (likely corrupted or very old format)")
            return None
        
        # Invert to original coordinates (in current state's orientation)
        try:
            original_move = canonical.invert_move(canonical_move)
            if original_move is None:
                # This is expected for abstract moves (pass, resign, etc.)
                # or raw hashes from unknown moves that can't be inverted
                logger.debug(f"Move {move_hash} has no coordinates (abstract move or unknown format)")
                return None
            if not isinstance(original_move, np.ndarray):
                original_move = np.asarray(original_move, dtype=int)
        except Exception as e:
            logger.warning(f"Failed to invert move {move_hash}: {e}")
            return None
        
        # Build statistics (flip perspective if opponent's move)
        stats = MoveStatistics(*stats_tuple)
        if snapshot_player != acting_player:
            stats = stats.flip_perspective()
        
        return MoveEvaluation(
            move=original_move,
            move_hash=move_hash,
            stats=stats,
        )
    
    # ========================================================================
    # Canonicalization with Caching
    # ========================================================================
    
    def _get_canonical_cached(self, state: GameState) -> CanonicalResult:
        """Get canonical representation with LRU caching.

        Caching behavior:
        - primary: LRU cache keyed by state-derived key (board bytes + player + shape)
        - secondary: map canonical.hash -> CanonicalResult to allow quick retrieval
                     of node mappings when only the hash is known.
        """
        cache_key = self._make_cache_key(state)
        
        cached = self._cache.get(cache_key)
        if cached is not None:
            # Ensure also present in canonical_by_hash
            self._canonical_by_hash[cached.hash] = cached
            return cached
        
        canonical = canonicalize_state(state)
        # Put into both caches
        self._cache.put(cache_key, canonical)
        self._canonical_by_hash[canonical.hash] = canonical
        return canonical
    
    def _get_canonical_by_hash(self, board_hash: str) -> CanonicalResult | None:
        """Retrieve CanonicalResult by canonical hash (if cached)."""
        return self._canonical_by_hash.get(board_hash)
    
    def _make_cache_key(self, state: GameState) -> str:
        """Create stable cache key for a game state."""
        try:
            board = np.asarray(state.board)
            board_bytes, *_ = serialize_array(board)
            player = int(getattr(state, "current_player", 0))
            shape = json.dumps(board.shape)
            
            hasher = hashlib.sha256()
            hasher.update(board_bytes)
            hasher.update(str(player).encode())
            hasher.update(shape.encode())
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to create cache key: {e}")
            return f"uncached_{id(state)}"
    
    # ========================================================================
    # Move Hash Management
    # ========================================================================
    
    def _hash_canonical_move(self, canonical_move: int | str | any) -> str:
        """
        Generate stable hash for a canonical move stored in DB.

        Handles:
        - 'sig_<hex>:<index>' -> stored as-is (preferred for cell moves)
        - 'sig_<hex>' -> stored as-is
        - integers -> legacy 'orbit_N' format
        - other strings -> hashed with sha256
        """
        # Signature strings from canonicalize_move (already stable)
        if isinstance(canonical_move, str):
            if canonical_move.startswith("sig_"):
                return canonical_move
            # Unknown move strings -> hash them for storage
            return hashlib.sha256(canonical_move.encode()).hexdigest()

        # Legacy integer fallback encoded as 'orbit_N'
        if isinstance(canonical_move, (int, np.integer)):
            return f"orbit_{int(canonical_move)}"

        # Fallback: hash the string representation
        return hashlib.sha256(str(canonical_move).encode()).hexdigest()
    
    def _parse_canonical_move(self, move_hash: str) -> str | int | None:
        """
        Parse canonical move from its hash.
        
        Returns:
        - 'sig_<hex>:<index>' or 'sig_<hex>' for signature-based moves
        - int for legacy 'orbit_N' format
        - move_hash itself for raw 64-char hex hashes (abstract moves)
        - None for unrecognized formats
        """
        if move_hash.startswith("sig_"):
            return move_hash
        
        if move_hash.startswith("orbit_"):
            try:
                return int(move_hash[6:])
            except ValueError:
                return None
        
        # Handle raw hashes (64 char hex) - these are from abstract/unknown moves
        # that were hashed by _hash_canonical_move
        if len(move_hash) == 64 and all(c in '0123456789abcdef' for c in move_hash):
            return move_hash  # Return as-is, invert_move will return None
        
        # Unrecognized format
        return None
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    @staticmethod
    def _normalize_game_id(game_id: str) -> str:
        """Normalize game ID to consistent format."""
        return game_id.lower().replace(" ", "_").replace("-", "_")
    
    @staticmethod
    def _make_meta_json(dtype, dtype_str, shape) -> str:
        """Serialize array metadata to JSON."""
        shape_val = list(shape) if isinstance(shape, tuple) else shape
        return json.dumps({
            "dtype": str(dtype),
            "dtype_str": str(dtype_str),
            "shape": shape_val,
        })
    
    def _visualize_moves(
        self,
        board: np.ndarray,
        evaluations: list[MoveEvaluation],
        selected_hash: str,
    ) -> None:
        """Visualize all candidate moves with statistics."""
        debug_data = []
        for eval in evaluations:
            debug_data.append({
                "move_array": eval.move.tolist(),
                "move_hash": eval.move_hash,
                "is_selected": eval.move_hash == selected_hash,
                "score": eval.stats.score,
                "utility": eval.stats.utility,
                "certainty": eval.stats.certainty,
                "total": eval.stats.total,
                "pW": eval.stats.win_rate,
                "pT": eval.stats.tie_rate,
                "pL": eval.stats.loss_rate,
            })
        
        render_debug(board, debug_data)
    
    # ========================================================================
    # Lifecycle
    # ========================================================================
    
    def close(self) -> None:
        """Close database connection and clear cache."""
        self._cache.clear()
        self._canonical_by_hash.clear()
        if not db.is_closed():
            db.close()
        logger.info("GameMemory closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()