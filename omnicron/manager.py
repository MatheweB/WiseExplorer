"""
Game memory manager with orbit-based state canonicalization.

Provides persistent storage and retrieval of game positions, moves, and statistics
using SQLite with automatic state canonicalization for position-independent learning.
"""

from __future__ import annotations

import json
import hashlib
import logging
from typing import Protocol, Any, List, Optional, Union
from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path

import numpy as np
from peewee import (
    Model, IntegerField, BlobField, TextField, SqliteDatabase, CompositeKey
)

from omnicron.serializers import serialize_array, deserialize_array
from agent.agent import State
from omnicron.state_canonicalizer import canonicalize_state, CanonicalResult
from omnicron.debug_viz import render_debug


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
    def neutral_rate(self) -> float:
        return self.neutral / self.total if self.total > 0 else 0.0
    
    @property
    def loss_rate(self) -> float:
        return self.losses / self.total if self.total > 0 else 0.0
    
    @property
    def certainty(self) -> float:
        N = self.total
        K = 4.0  # 4 categories
        if N <= 0.0:
            return 0.0
        return N / (N + K)

    @property
    def utility(self) -> float:
        return 3*self.win_rate + 2*self.tie_rate + 1*self.neutral_rate - 3*self.loss_rate
        

    @property
    def score(self) -> float:
        # deterministic score: certainty-weighted utility
        return float(self.certainty * self.utility)


@dataclass(frozen=True)
class MoveEvaluation:
    """Complete evaluation of a move from a position."""
    
    move: Any  # Can be np.ndarray, int, or object depending on game
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
    player_id = IntegerField()
    
    win_count = IntegerField(default=0)
    tie_count = IntegerField(default=0)
    neutral_count = IntegerField(default=0)
    loss_count = IntegerField(default=0)
    
    class Meta:
        table_name = "play_stats"
        primary_key = CompositeKey("game_id", "board_hash", "move_hash", "player_id")
        indexes = ((("game_id", "board_hash", "player_id"), False),)


# ============================================================================
# Game Memory Manager
# ============================================================================
class GameMemory:
    """
    Persistent memory system for game learning with orbit-based canonicalization.

    Automatically canonicalizes game states and moves to enable learning from
    symmetric positions (rotations, reflections, etc.). Equivalent moves are
    consolidated into single statistics entries.

    Stats are stored per (game_id, board_hash, move_hash, player_id).
    Each player learns from their own moves only.
    
    INVARIANT: snapshot_player == acting_player always (enforced at record time)
    """
    
    def __init__(
        self,
        db_path: str | Path = "memory.db",
        cache_size: int = 4096,
    ):
        # Default to storing in omnicron folder if relative path given
        db_path = Path(db_path)
        if not db_path.is_absolute() and db_path.parent == Path('.'):
            omnicron_dir = Path(__file__).parent
            db_path = omnicron_dir / db_path
        
        self.db_path = db_path
        self._cache = LRUCache(maxsize=cache_size)
        self._canonical_by_hash: dict[str, CanonicalResult] = {}
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
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
        move: Any,
        acting_player: int,
        outcome: State,
    ) -> bool:
        """
        Record a move and its outcome from a game.
        
        INVARIANT: state.current_player must equal acting_player.
        """
        if state is None:
            logger.warning("Skipping record: state is None")
            return False
        
        if state.current_player != acting_player:
            logger.error(
                f"Invariant violation: snapshot player {state.current_player} "
                f"!= acting player {acting_player}. Move not recorded."
            )
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
            # Handles generic move types (int, tuple, array, etc) via canonicalizer
            canonical_move_sig = canonical.canonicalize_move(move)
            
            # Robust hash generation for storage
            move_hash = self._hash_canonical_move(canonical_move_sig)
        except Exception as e:
            logger.error(f"Move canonicalization failed: {e}", exc_info=True)
            return False
        
        return self._update_statistics(
            game_id=game_id,
            board_hash=canonical.hash,
            move_hash=move_hash,
            player_id=int(acting_player),
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
        player_id: int,
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
                (game_id, board_hash, move_hash, player_id,
                 win_count, tie_count, neutral_count, loss_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id, board_hash, move_hash, player_id)
                DO UPDATE SET
                    win_count = win_count + excluded.win_count,
                    tie_count = tie_count + excluded.tie_count,
                    neutral_count = neutral_count + excluded.neutral_count,
                    loss_count = loss_count + excluded.loss_count
                """,
                (game_id, board_hash, move_hash, player_id, *counts)
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
    ) -> Any | None:
        """Retrieve the best known move for a position."""
        return self._select_move(game_id, state, select_best=True, debug=debug)
    
    def get_worst_move(
        self,
        game_id: str,
        state: GameState,
        debug: bool = False,
    ) -> Any | None:
        """Retrieve the worst known move for a position."""
        return self._select_move(game_id, state, select_best=False, debug=debug)
    
    def _select_move(
        self,
        game_id: str,
        state: GameState,
        select_best: bool,
        debug: bool,
    ) -> Any | None:
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
            player_id=int(state.current_player),
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
        player_id: int,
        canonical: CanonicalResult,
    ) -> list[MoveEvaluation]:
        """
        Fetch and evaluate all moves from a position.
        """
        cursor = db.execute_sql(
            """
            SELECT 
                move_hash,
                SUM(win_count) as total_wins,
                SUM(tie_count) as total_ties,
                SUM(neutral_count) as total_neutral,
                SUM(loss_count) as total_losses
            FROM play_stats
            WHERE game_id = ? AND board_hash = ? AND player_id = ?
            GROUP BY move_hash
            """,
            (game_id, board_hash, player_id)
        )
        
        evaluations = []
        for row in cursor.fetchall():
            try:
                # We fetch the canonical signature (move_hash) from DB
                # and expand it into real moves using the canonicalizer
                evaluations_for_sig = self._make_evaluations_for_signature(
                    move_hash=row[0],
                    stats_tuple=(int(row[1]), int(row[2]), int(row[3]), int(row[4])),
                    canonical=canonical,
                )
                evaluations.extend(evaluations_for_sig)
            except Exception as e:
                logger.warning(f"Failed to evaluate move {row[0]}: {e}")
        
        return evaluations
    
    def _make_evaluations_for_signature(
        self,
        move_hash: str,
        stats_tuple: tuple[int, int, int, int],
        canonical: CanonicalResult,
    ) -> list[MoveEvaluation]:
        """
        Create MoveEvaluation for ALL symmetric moves in this signature.
        
        Uses the `get_all_symmetric_moves` capability of the canonicalizer
        to transform the stored 'sig_X->sig_Y' back into real board moves.
        """
        stats = MoveStatistics(*stats_tuple)
        
        try:
            # We treat the hash as the signature directly if it matches the format
            signature = move_hash
            
            # The canonicalizer is responsible for handling 'sig_', 'orbit_', and '→' 
            # logic internally via get_all_symmetric_moves.
            # This makes the manager agnostic to how complex the move is (Chess vs TicTacToe).
            
            all_symmetric_moves = canonical.get_all_symmetric_moves(signature)
            
            if not all_symmetric_moves:
                # Fallback for legacy data or simple integer moves (e.g. TicTacToe raw indices)
                # If the canonicalizer didn't recognize it as a signature, try direct inversion
                try:
                    parsed = self._parse_canonical_move(signature)
                    if parsed is not None:
                         # Attempt to treat as direct index if supported
                         inverted = canonical.invert_move(parsed)
                         if inverted is not None:
                            all_symmetric_moves = [inverted]
                except Exception:
                    pass

            return [
                MoveEvaluation(
                    move=move,
                    move_hash=move_hash,
                    stats=stats,
                )
                for move in all_symmetric_moves
            ]
            
        except Exception as e:
            logger.warning(f"Failed to get symmetric moves for {move_hash}: {e}", exc_info=True)
            return []
    
    # ========================================================================
    # Canonicalization with Caching
    # ========================================================================
    
    def _get_canonical_cached(self, state: GameState) -> CanonicalResult:
        """Get canonical representation with LRU caching."""
        cache_key = self._make_cache_key(state)
        
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._canonical_by_hash[cached.hash] = cached
            return cached
        
        canonical = canonicalize_state(state)
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
    
    def _hash_canonical_move(self, canonical_move: int | str | Any) -> str:
        """
        Generate stable hash for a canonical move stored in DB.
        
        If it's a signature (sig_X or sig_X->sig_Y), we store it as is
        so we can reverse it later.
        """
        if isinstance(canonical_move, str):
            # We trust the signature string from the canonicalizer
            return canonical_move
            
        # Legacy integer fallback for simple games
        if isinstance(canonical_move, (int, np.integer)):
            return f"orbit_{int(canonical_move)}"

        # Fallback: hash the string representation (unrecoverable, usually bad)
        return hashlib.sha256(str(canonical_move).encode()).hexdigest()
    
    def _parse_canonical_move(self, move_hash: str) -> str | int | None:
        """
        Parse canonical move from its hash.
        """
        if move_hash.startswith("sig_"):
            return move_hash
        
        if move_hash.startswith("move_"):
            return move_hash
        
        if move_hash.startswith("orbit_"):
            try:
                return int(move_hash[6:])
            except ValueError:
                return None
        
        # Handle compound moves with →
        if "→" in move_hash:
            return move_hash
        
        # Raw hashes (64 char hex) - legacy
        if len(move_hash) == 64 and all(c in '0123456789abcdef' for c in move_hash):
            return move_hash
        
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
            # Handle numpy arrays for serialization
            if isinstance(eval.move, np.ndarray):
                move_display = eval.move.tolist()
            else:
                move_display = eval.move

            debug_data.append({
                "move_array": move_display,
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