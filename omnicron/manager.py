"""
Game memory manager with TRANSITION-based canonical learning.

FIXED & OPTIMIZED SCHEMA:
- Positions store BOARD ONLY (no current_player in hash)
- Transitions are (from_hash → to_hash) - acting_player is implicit
- PlayStats keyed by transition_hash only (player is implicit)

Key insight: In alternating-turn games, the acting_player is DERIVABLE
from the board state (whoever's turn it is). Including it in hashes
doubles the storage unnecessarily.

Key idea:
- Learn on canonical state transitions, not moves
- Moves are derived by simulating valid moves and matching the canonical next-state
- DB stores (game_id, position_hash) and (game_id, transition_hash -> from_hash, to_hash)
- PlayStats stores aggregated outcome counts keyed by (game_id, transition_hash)
"""

from __future__ import annotations

import json
import hashlib
import logging
from typing import Protocol, Any, List, Optional, Iterable
from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path

import numpy as np
from peewee import (
    Model,
    IntegerField,
    BlobField,
    TextField,
    SqliteDatabase,
    CompositeKey,
)

from omnicron.serializers import serialize_array
from agent.agent import State
from omnicron.state_canonicalizer import canonicalize_board, CanonicalResult
from omnicron.debug_viz import render_debug
from games.game_base import GameBase
from games.game_state import GameState


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MoveStatistics:
    wins: int
    ties: int
    neutral: int
    losses: int

    @property
    def total(self) -> int:
        return self.wins + self.ties + self.neutral + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total else 0.0

    @property
    def tie_rate(self) -> float:
        return self.ties / self.total if self.total else 0.0

    @property
    def neutral_rate(self) -> float:
        return self.neutral / self.total if self.total else 0.0

    @property
    def loss_rate(self) -> float:
        return self.losses / self.total if self.total else 0.0

    @property
    def certainty(self) -> float:
        N = self.total
        K = 4
        return N / (N + K) if N else 0.0

    @property
    def utility(self) -> float:
        return 2 * self.win_rate + 1 * self.tie_rate - 2 * self.loss_rate

    @property
    def score(self) -> float:
        return float(self.certainty * self.utility)


@dataclass(frozen=True)
class TransitionEvaluation:
    move: Any  # concrete move (derived at selection time)
    next_state: GameState  # concrete next state (after applying move)
    to_hash: str  # canonical hash for next state
    stats: MoveStatistics


# ---------------------------------------------------------------------------
# LRU Cache (canonicalization)
# ---------------------------------------------------------------------------
class LRUCache:
    def __init__(self, maxsize: int = 4096):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, CanonicalResult] = OrderedDict()

    def get(self, key: str) -> Optional[CanonicalResult]:
        if key not in self._cache:
            return None
        val = self._cache.pop(key)
        self._cache[key] = val
        return val

    def put(self, key: str, value: CanonicalResult) -> None:
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Database models (OPTIMIZED - removed redundant acting_player/player_id)
# ---------------------------------------------------------------------------
db = SqliteDatabase(None)


class BaseModel(Model):
    class Meta:
        database = db


class Position(BaseModel):
    """Stores BOARD configuration only (no current_player)."""

    game_id = TextField()
    position_hash = TextField()
    board_bytes = BlobField()
    meta_json = TextField()

    class Meta:
        table_name = "positions"
        primary_key = CompositeKey("game_id", "position_hash")


class TransitionRow(BaseModel):
    """
    Stores canonical mapping for a transition.

    OPTIMIZED: No acting_player field - it's implicit from from_hash.
    In any alternating-turn game, the board state determines whose turn it is.
    """

    game_id = TextField()
    transition_hash = TextField()
    from_hash = TextField()
    to_hash = TextField()

    class Meta:
        table_name = "transitions"
        primary_key = CompositeKey("game_id", "transition_hash")


class PlayStats(BaseModel):
    """
    Stores aggregated statistics per (game_id, transition_hash).

    OPTIMIZED: No player_id field - the acting player is implicit.
    The outcome (WIN/LOSS/TIE) is always from the perspective of
    whoever made the move (the acting player at from_hash).
    """

    game_id = TextField()
    transition_hash = TextField()

    win_count = IntegerField(default=0)
    tie_count = IntegerField(default=0)
    neutral_count = IntegerField(default=0)
    loss_count = IntegerField(default=0)

    class Meta:
        table_name = "play_stats"
        primary_key = CompositeKey("game_id", "transition_hash")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _transition_hash(from_hash: str, to_hash: str) -> str:
    """
    Compute transition hash from (from_board → to_board).

    OPTIMIZED: No acting_player - it's derivable from the board state.
    """
    h = hashlib.sha256()
    h.update(from_hash.encode("utf-8"))
    h.update(to_hash.encode("utf-8"))
    return h.hexdigest()


def _diff_states(before_board: np.ndarray, after_board: np.ndarray):
    """Return list of (coord_tuple, before_val, after_val) where values differ."""
    diffs = []
    for idx in np.ndindex(before_board.shape):
        if before_board[idx] != after_board[idx]:
            diffs.append((idx, before_board[idx], after_board[idx]))
    return diffs


# ---------------------------------------------------------------------------
# GameMemory
# ---------------------------------------------------------------------------
class GameMemory:
    def __init__(self, db_path: str | Path = "memory.db", cache_size: int = 4096):
        db_path = Path(db_path)
        if not db_path.is_absolute():
            db_path = Path(__file__).parent / db_path

        self._cache = LRUCache(maxsize=cache_size)

        # Initialize DB
        db.init(str(db_path))
        db.connect(reuse_if_open=True)
        db.create_tables([Position, TransitionRow, PlayStats])

        logger.info(f"GameMemory initialized: {db_path}")

    # -------------------------
    # Record a transition/outcome (SIMPLIFIED)
    # -------------------------
    def record_outcome(
        self,
        game: GameBase,
        move: Any,
        acting_player: int,  # Still passed for context, but not stored in hash
        outcome: State,
    ) -> bool:
        """
        Record that from `game.get_state()` applying `move` (by cloning & applying)
        resulted in an outcome (WIN/TIE/NEUTRAL/LOSS) for `acting_player`.

        OPTIMIZED:
        - Canonicalizes BOARD ONLY (no current_player in hash)
        - Transition hash is just (from_hash → to_hash)
        - acting_player is implicit (whoever's turn at from_hash)
        """
        try:
            game_id = game.game_id()
            before_state = game.get_state()

            # Canonicalize BOARD ONLY
            canon_before = self._get_canonical_board_cached(before_state.board)

            # Create a copy of the game and apply move
            game_next = game.deep_clone()
            game_next.apply_move(move)
            after_state = game_next.get_state()

            # Canonicalize BOARD ONLY
            canon_after = self._get_canonical_board_cached(after_state.board)

            # OPTIMIZED: No acting_player in transition hash
            t_hash = _transition_hash(canon_before.hash, canon_after.hash)

            # Persist positions (board only)
            before_bytes, *_ = serialize_array(before_state.board)
            Position.get_or_create(
                game_id=game_id,
                position_hash=canon_before.hash,
                defaults={
                    "board_bytes": before_bytes,
                    "meta_json": json.dumps({"shape": list(before_state.board.shape)}),
                },
            )

            after_bytes, *_ = serialize_array(after_state.board)
            Position.get_or_create(
                game_id=game_id,
                position_hash=canon_after.hash,
                defaults={
                    "board_bytes": after_bytes,
                    "meta_json": json.dumps({"shape": list(after_state.board.shape)}),
                },
            )

            # OPTIMIZED: No acting_player in transition row
            TransitionRow.get_or_create(
                game_id=game_id,
                transition_hash=t_hash,
                defaults={
                    "from_hash": canon_before.hash,
                    "to_hash": canon_after.hash,
                },
            )

            # OPTIMIZED: No player_id in stats
            stats_row, _ = PlayStats.get_or_create(
                game_id=game_id,
                transition_hash=t_hash,
            )

            if outcome == State.WIN:
                stats_row.win_count += 1
            elif outcome == State.TIE:
                stats_row.tie_count += 1
            elif outcome == State.NEUTRAL:
                stats_row.neutral_count += 1
            elif outcome == State.LOSS:
                stats_row.loss_count += 1

            stats_row.save()
            return True

        except Exception as e:
            logger.exception("Failed to record outcome: %s", e)
            return False

    # -------------------------
    # Selection helpers
    # -------------------------
    def get_best_move(self, game: GameBase, debug: bool = False) -> Optional[Any]:
        return self._select_move(game, best=True, debug=debug)

    def get_worst_move(self, game: GameBase, debug: bool = False) -> Optional[Any]:
        return self._select_move(game, best=False, debug=debug)

    def _select_move(self, game: GameBase, best: bool, debug: bool) -> Optional[Any]:
        """
        For the given game, enumerate game.valid_moves(), simulate each move,
        canonicalize the resulting next-state, and look up play stats for the
        corresponding transition_hash. Choose by score.

        OPTIMIZED: No player filtering needed - transitions are universal.
        """
        game_id = game.game_id()
        state = game.get_state()

        # Canonicalize BOARD ONLY
        canon_from = self._get_canonical_board_cached(state.board)

        # Find all transitions FROM this board
        transitions_from = list(
            TransitionRow.select().where(
                (TransitionRow.game_id == game_id)
                & (TransitionRow.from_hash == canon_from.hash)
            )
        )

        if not transitions_from:
            return None

        transition_hashes = {t.transition_hash for t in transitions_from}

        # Load PlayStats for these transitions (single query)
        stats_rows = list(
            PlayStats.select().where(
                (PlayStats.game_id == game_id)
                & (PlayStats.transition_hash.in_(transition_hashes))
            )
        )

        if not stats_rows:
            return None

        stats_by_hash = {r.transition_hash: r for r in stats_rows}

        # Iterate valid moves, simulate, compute transition_hash, collect evaluations
        evaluations: List[TransitionEvaluation] = []

        for move in game.valid_moves():
            game_next = game.deep_clone()
            try:
                game_next.apply_move(move)
            except Exception:
                # Illegal move or engine error: skip
                continue

            next_state = game_next.get_state()

            # Canonicalize BOARD ONLY
            canon_next = self._get_canonical_board_cached(next_state.board)

            # OPTIMIZED: No acting_player in transition hash
            t_hash = _transition_hash(canon_from.hash, canon_next.hash)

            stat_row = stats_by_hash.get(t_hash)
            if not stat_row:
                continue

            stats = MoveStatistics(
                wins=int(stat_row.win_count),
                ties=int(stat_row.tie_count),
                neutral=int(stat_row.neutral_count),
                losses=int(stat_row.loss_count),
            )

            evaluations.append(
                TransitionEvaluation(
                    move=move,
                    next_state=next_state,
                    to_hash=canon_next.hash,
                    stats=stats,
                )
            )

        if not evaluations:
            return None

        # Sort by score
        evaluations.sort(key=lambda e: e.stats.score, reverse=best)
        chosen = evaluations[0]

        # Debug visualization (diffs)
        if debug:
            debug_rows = []
            chosen_hash = chosen.to_hash
            for e in evaluations:
                diff = _diff_states(state.board, e.next_state.board)
                debug_rows.append(
                    {
                        "diff": diff,
                        "move": e.move,
                        "is_selected": e.to_hash == chosen_hash,
                        "score": e.stats.score,
                        "utility": e.stats.utility,
                        "certainty": e.stats.certainty,
                        "total": e.stats.total,
                        "pW": e.stats.win_rate,
                        "pT": e.stats.tie_rate,
                        "pN": e.stats.neutral_rate,
                        "pL": e.stats.loss_rate,
                    }
                )
            render_debug(state.board, debug_rows)

        return chosen.move

    # -------------------------
    # Canonicalization cache wrapper
    # -------------------------
    def _get_canonical_board_cached(self, board: np.ndarray) -> CanonicalResult:
        """
        Canonicalize BOARD ONLY (no current_player).
        Creates cache key from board bytes only.
        """
        board_bytes, *_ = serialize_array(board)
        key = hashlib.sha256(board_bytes).hexdigest()

        cached = self._cache.get(key)
        if cached:
            return cached

        canonical = canonicalize_board(board)
        self._cache.put(key, canonical)
        return canonical

    # -------------------------
    # Lifecycle
    # -------------------------
    def close(self) -> None:
        self._cache.clear()
        if not db.is_closed():
            db.close()
