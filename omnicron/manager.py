"""
Game memory manager with partitioned storage, atomic writes, and query caching.

Core Concept
------------
1. We take in an ndarray "state" and create a canonical hash of that state based on its colored graph representation.
2. We then hash "transitions" that happen from canonical state A to new state B (i.e. moves in a game), and store those
    in our table with an associated parent "state"
3. We collect statistics on each "state + transition" from historical simulation data, and use this accumulated data to inform
    the "score" and statistics.

The "score" parameter for each "state + transition" informs our Wise Explorer training algorithm on which moves to pick.
We maintain "parent" stats which are the cumulative wins/losses/ties/neutral outcomes for ALL transitions from a given parent state A.
    

Architecture
------------
- PARTITIONING: Each game type gets its own DB file (tic_tac_toe.db, minichess.db).
    This keeps indexes small and queries fast regardless of other games' data.

- ATOMIC WRITES: Entire rounds are written in a single transaction via record_round().
    No partial games are ever recorded.

- QUERY CACHING: Read-only workers cache DB results. Since workers never see
    updates during their lifetime, stale reads are impossible.

Usage
-----
    # Option 1: Auto-partitioned (recommended)
    memory = GameMemory.for_game(TicTacToe(), base_dir="data/memory")

    # Option 2: Explicit path
    memory = GameMemory("data/memory/tic_tac_toe.db")
"""

from __future__ import annotations

import math
import random
import hashlib
import logging
from typing import Any, List, Optional, Dict, Tuple
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

from agent.agent import State
from omnicron.state_canonicalizer import canonicalize_board, CanonicalResult
from omnicron.debug_viz import render_debug
from games.game_base import GameBase
from games.game_state import GameState


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database Setup
# ---------------------------------------------------------------------------

db = SqliteDatabase(
    None,
    pragmas={
        "journal_mode": "wal",
        "cache_size": -1024 * 64,
    },
)


class BaseModel(Model):
    class Meta:
        database = db


class Position(BaseModel):
    game_id = TextField()
    position_hash = TextField()
    board_bytes = BlobField()
    meta_json = TextField()

    class Meta:
        table_name = "positions"
        primary_key = CompositeKey("game_id", "position_hash")


class TransitionRow(BaseModel):
    game_id = TextField()
    transition_hash = TextField()
    from_hash = TextField(index=True)
    to_hash = TextField()

    class Meta:
        table_name = "transitions"
        primary_key = CompositeKey("game_id", "transition_hash")


class PlayStats(BaseModel):
    game_id = TextField()
    transition_hash = TextField()
    win_count = IntegerField(default=0)
    tie_count = IntegerField(default=0)
    neutral_count = IntegerField(default=0)
    loss_count = IntegerField(default=0)

    class Meta:
        table_name = "play_stats"
        primary_key = CompositeKey("game_id", "transition_hash")


class ParentNode(BaseModel):
    game_id = TextField()
    position_hash = TextField()
    player_to_move = IntegerField()
    win_count = IntegerField(default=0)
    tie_count = IntegerField(default=0)
    neutral_count = IntegerField(default=0)
    loss_count = IntegerField(default=0)

    class Meta:
        table_name = "parent_nodes"
        primary_key = CompositeKey("game_id", "position_hash", "player_to_move")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoveStatistics:
    """Statistics for a single transition (state -> next_state)."""

    wins: int
    ties: int
    neutral: int
    losses: int
    parent_wins: int
    parent_ties: int
    parent_neutral: int
    parent_losses: int

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
    def utility(self) -> float:
        """Expected value: (W - L) / n"""
        return (self.wins - self.losses) / self.total

    @property
    def score(self) -> float:
        '''
        The score we use to determine which move to pick.
            The idea is that we pick the move centered around the mean that we are the
            most confident about. Sparse data decreases our score, and lots of ties
            have a diluting effect on our score.
        '''
        n = self.total
        W, L = self.wins, self.losses
        mean = (W - L) / n

        # Standard error is undefined
        if n == 1:
            return (mean + 1.0) / 2.0

        se = math.sqrt(((W + L) - n * mean * mean) / (n * (n - 1)))

        # Measures how confident we are that the mean is reliable
        # (specifically in the worst case of 1 standard error unit below the mean)
        LCB = mean - se

        # Normalize LCB from [-1,1] to [0,1]
        return (LCB + 1.0) / 2.0

    @property
    def certainty(self) -> float:
        '''
        How far one standard error is away from 1.
            The closer SE is to 0, the more certain we are of our outcome.
        '''
        n = self.total
        W, L = self.wins, self.losses
        mean = (W - L) / n

        # Standard error is undefined
        if n == 1:
            return 1.0  # No variance in a single observation

        se = math.sqrt(((W + L) - n * mean * mean) / (n * (n - 1)))

        return 1.0 - se


@dataclass(frozen=True)
class TransitionEvaluation:
    """A candidate move with its resulting state and statistics."""
    move: Any
    next_state: GameState
    to_hash: str
    stats: MoveStatistics


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class LRUCache:
    """Simple LRU cache for canonical board results."""

    def __init__(self, maxsize: int = 4096):
        self.maxsize = maxsize
        self._cache: OrderedDict[Any, Any] = OrderedDict()

    def get(self, key: Any) -> Optional[Any]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: Any, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            self._cache[key] = value
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _transition_hash(from_hash: str, to_hash: str) -> str:
    return hashlib.sha256(f"{from_hash}{to_hash}".encode()).hexdigest()


def _diff_states(before: np.ndarray, after: np.ndarray) -> List:
    diffs = []
    for idx in np.ndindex(before.shape):
        if before[idx] != after[idx]:
            diffs.append((idx, before[idx], after[idx]))
    return diffs


# ---------------------------------------------------------------------------
# GameMemory
# ---------------------------------------------------------------------------


class GameMemory:
    """
    High-performance game transition storage.

    Features:
    - Atomic writes: record_round() writes entire rounds atomically
    - Query caching: Read-only instances cache results for fast repeated access
    - Partitioning: Use for_game() to auto-create per-game DB files
    """

    def __init__(
        self,
        db_path: str | Path = "memory.db",
        read_only: bool = False,
    ):
        db_path = Path(db_path)

        # Make absolute (relative to cwd, not this file)
        if not db_path.is_absolute():
            db_path = Path.cwd() / db_path

        # Ensure parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.read_only = read_only

        # Canonicalization cache (both readers and writers)
        self._canon_cache = LRUCache(maxsize=4096)

        # Query caches (readers only - workers cache aggressively)
        self._transition_cache: Dict[str, Dict[str, PlayStats]] = {}
        self._parent_cache: Dict[str, Optional[ParentNode]] = {}

        db.init(str(db_path))
        db.connect(reuse_if_open=True)

        if not read_only:
            db.create_tables([Position, TransitionRow, PlayStats, ParentNode])

        mode = "read-only" if read_only else "read/write"
        logger.info("GameMemory (%s): %s", mode, db_path)

    @classmethod
    def for_game(
        cls, game: GameBase, base_dir: str | Path = "data/memory", **kwargs
    ) -> "GameMemory":
        """
        Create a partitioned GameMemory for a specific game type.

        Each game type gets its own DB file, keeping indexes small
        and queries fast regardless of other games' data.

        Example:
            memory = GameMemory.for_game(TicTacToe(), base_dir="data/memory")
            # Creates: data/memory/tic_tac_toe.db
        """
        base_dir = Path(base_dir)
        game_id = game.game_id()
        db_path = base_dir / f"{game_id}.db"

        return cls(db_path, **kwargs)

    # -------------------------------------------------------------------------
    # Recording (with batching)
    # -------------------------------------------------------------------------

    def _write_transition(
        self,
        game: GameBase,
        move: Any,
        player_to_move: int,
        outcome: State,
    ) -> bool:
        """
        Internal: Write a single transition. Must be called within db.atomic().
        """
        try:
            game_id = game.game_id()
            before_state = game.get_state()

            canon_before = self._canonicalize(before_state.board)

            game_next = game.deep_clone()
            game_next.apply_move(move)
            after_state = game_next.get_state()
            canon_after = self._canonicalize(after_state.board)

            t_hash = _transition_hash(canon_before.hash, canon_after.hash)

            Position.get_or_create(
                game_id=game_id,
                position_hash=canon_before.hash,
                defaults={
                    "board_bytes": before_state.board.tobytes(),
                    "meta_json": "{}",
                },
            )
            Position.get_or_create(
                game_id=game_id,
                position_hash=canon_after.hash,
                defaults={
                    "board_bytes": after_state.board.tobytes(),
                    "meta_json": "{}",
                },
            )

            TransitionRow.get_or_create(
                game_id=game_id,
                transition_hash=t_hash,
                defaults={"from_hash": canon_before.hash, "to_hash": canon_after.hash},
            )

            edge, _ = PlayStats.get_or_create(
                game_id=game_id,
                transition_hash=t_hash,
            )

            parent, _ = ParentNode.get_or_create(
                game_id=game_id,
                position_hash=canon_before.hash,
                player_to_move=player_to_move,
            )

            if outcome == State.WIN:
                edge.win_count += 1
                parent.win_count += 1
            elif outcome == State.TIE:
                edge.tie_count += 1
                parent.tie_count += 1
            elif outcome == State.NEUTRAL:
                edge.neutral_count += 1
                parent.neutral_count += 1
            elif outcome == State.LOSS:
                edge.loss_count += 1
                parent.loss_count += 1

            edge.save()
            parent.save()
            return True

        except Exception:
            logger.exception("Failed to write transition")
            return False

    def record_round(
        self,
        game_class: type,
        stacks: List[Tuple[List[Tuple[Any, np.ndarray, int]], State]],
    ) -> int:
        """
        Record all game stacks from a round in a single atomic transaction.

        Args:
            game_class: The game class to reconstruct states
            stacks: List of (moves, outcome) where moves is [(move, board_before, player)]

        Returns:
            Number of transitions recorded
        """
        if self.read_only:
            raise RuntimeError("Cannot write to read-only GameMemory")

        count = 0
        with db.atomic():
            for moves, outcome in stacks:
                for move, board_before, player_who_moved in moves:
                    gs = game_class()
                    gs.set_state(GameState(board_before, player_who_moved))
                    if self._write_transition(gs, move, player_who_moved, outcome):
                        count += 1
        return count

    # -------------------------------------------------------------------------
    # Move Selection (with caching)
    # -------------------------------------------------------------------------

    def get_best_move(self, game: GameBase, debug: bool = False) -> Optional[Any]:
        result = self._select_move(game, best=True, debug=debug)
        return result[0] if result else None

    def get_worst_move(self, game: GameBase, debug: bool = False) -> Optional[Any]:
        result = self._select_move(game, best=False, debug=debug)
        return result[0] if result else None

    def get_best_move_with_score(
        self, game: GameBase, debug: bool = False
    ) -> Optional[Tuple[Any, float]]:
        result = self._select_move(game, best=True, debug=debug)
        return result

    def get_worst_move_with_score(
        self, game: GameBase, debug: bool = False
    ) -> Optional[Tuple[Any, float]]:
        result = self._select_move(game, best=False, debug=debug)
        return result

    def _select_move(
        self, game: GameBase, best: bool, debug: bool
    ) -> Optional[Tuple[Any, float]]:
        game_id = game.game_id()
        state = game.get_state()
        player_to_move = state.current_player
        canon_from = self._canonicalize(state.board)

        parent = self._get_parent_cached(game_id, canon_from.hash, player_to_move)
        pw, pt, pn, pl = (0, 0, 0, 0)
        if parent:
            pw = int(getattr(parent, "win_count"))
            pt = int(getattr(parent, "tie_count"))
            pn = int(getattr(parent, "neutral_count"))
            pl = int(getattr(parent, "loss_count"))

        stats_by_to_hash = self._get_transitions_cached(game_id, canon_from.hash)

        evaluations: List[TransitionEvaluation] = []

        for move in game.valid_moves():
            game_next = game.deep_clone()
            try:
                game_next.apply_move(move)
            except Exception:
                continue

            next_state = game_next.get_state()
            canon_next = self._canonicalize(next_state.board)

            edge = stats_by_to_hash.get(canon_next.hash)

            # Do not assume anything about a state we've never seen, only act on what we know
            if not edge:
                continue

            ew = int(getattr(edge, "win_count"))
            et = int(getattr(edge, "tie_count"))
            en = int(getattr(edge, "neutral_count"))
            el = int(getattr(edge, "loss_count"))

            stats = MoveStatistics(
                wins=ew,
                ties=et,
                neutral=en,
                losses=el,
                parent_wins=pw,
                parent_ties=pt,
                parent_neutral=pn,
                parent_losses=pl,
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

        evaluations.sort(key=lambda e: e.stats.score, reverse=best)
        potential_moves = [evaluations[0]]
        for move in evaluations[1:]:
            if move.stats.score == potential_moves[0].stats.score:
                potential_moves.append(move)

        # We want to choose a random move among identically-scored moves for diversity
        chosen = potential_moves[random.randint(0,len(potential_moves)-1)]

        if debug:
            self._render_debug(state, evaluations, chosen.to_hash)

        return (chosen.move, chosen.stats.score)

    def _get_transitions_cached(
        self, game_id: str, from_hash: str
    ) -> Dict[str, PlayStats]:
        """Get all transitions from a position, with caching for read-only instances."""
        cache_key = f"{game_id}:{from_hash}"

        if self.read_only and cache_key in self._transition_cache:
            return self._transition_cache[cache_key]

        query = (
            TransitionRow.select(TransitionRow, PlayStats)
            .join(
                PlayStats,
                on=(TransitionRow.transition_hash == PlayStats.transition_hash),
            )
            .where(
                (TransitionRow.game_id == game_id)
                & (TransitionRow.from_hash == from_hash)
            )
        )

        result = {row.to_hash: row.playstats for row in query}

        if self.read_only:
            self._transition_cache[cache_key] = result

        return result

    def _get_parent_cached(
        self, game_id: str, position_hash: str, player_to_move: int
    ) -> Optional[ParentNode]:
        """Get parent node stats, with caching for read-only instances."""
        cache_key = f"{game_id}:{position_hash}:{player_to_move}"

        if self.read_only and cache_key in self._parent_cache:
            return self._parent_cache[cache_key]

        parent = ParentNode.get_or_none(
            game_id=game_id,
            position_hash=position_hash,
            player_to_move=player_to_move,
        )

        if self.read_only:
            self._parent_cache[cache_key] = parent

        return parent

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _canonicalize(self, board: np.ndarray) -> CanonicalResult:
        key = board.tobytes()
        cached = self._canon_cache.get(key)
        if cached:
            return cached

        result = canonicalize_board(board)
        self._canon_cache.put(key, result)
        return result

    def _render_debug(
        self,
        state: GameState,
        evaluations: List[TransitionEvaluation],
        chosen_hash: str,
    ) -> None:
        debug_rows = []
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

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Clear caches and close database connection."""
        self._canon_cache.clear()
        self._transition_cache.clear()
        self._parent_cache.clear()

        if not db.is_closed():
            try:
                db.execute_sql("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            db.close()
