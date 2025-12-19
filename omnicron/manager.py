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

import hashlib
import logging
from typing import Any, List, Optional, Dict
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
    # Edge statistics (s, a)
    wins: int
    ties: int
    neutral: int
    losses: int

    # Parent node statistics (s, player_to_move)
    parent_wins: int
    parent_ties: int
    parent_neutral: int
    parent_losses: int

    # -------------------------
    # Edge totals
    # -------------------------
    @property
    def total(self) -> int:
        return self.wins + self.ties + self.neutral + self.losses

    # -------------------------
    # Parent totals
    # -------------------------
    @property
    def parent_total(self) -> int:
        return (
            self.parent_wins
            + self.parent_ties
            + self.parent_neutral
            + self.parent_losses
        )

    # -------------------------
    # Rates
    # -------------------------
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

    # -------------------------
    # Utility / value
    # -------------------------
    @property
    def certainty(self) -> float:
        return self.win_rate

    @property
    def utility(self) -> float:
        """
        Expected terminal value, conditioned on resolution.
        """
        effective = self.total - self.neutral
        if effective <= 0:
            return 0.0

        return (self.wins + 0.5 * self.ties - self.losses) / effective

    # -------------------------
    # Bayesian exploitation
    # -------------------------
    @property
    def bayes_exploit(self) -> float:
        # Can also try Beta(0.5, 0.5) Jeffreys prior - less biased than Beta(1,1) for extreme rates
        alpha = 1.0 + self.wins + 0.5 * self.ties
        beta = 1.0 + self.losses + 0.5 * self.ties
        return alpha / (alpha + beta)

    @property
    def score(self) -> float:
        return self.bayes_exploit


@dataclass(frozen=True)
class TransitionEvaluation:
    move: Any
    next_state: GameState
    to_hash: str
    stats: MoveStatistics


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------
class LRUCache:
    def __init__(self, maxsize: int = 4096):
        self.maxsize = maxsize
        self._cache: OrderedDict[Any, CanonicalResult] = OrderedDict()

    def get(self, key: Any) -> Optional[CanonicalResult]:
        if key not in self._cache:
            return None
        val = self._cache.pop(key)
        self._cache[key] = val
        return val

    def put(self, key: Any, value: CanonicalResult) -> None:
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
db = SqliteDatabase(None, pragmas={"journal_mode": "wal", "cache_size": -1024 * 64})


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
        primary_key = CompositeKey(
            "game_id",
            "position_hash",
            "player_to_move",
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _transition_hash(from_hash: str, to_hash: str) -> str:
    return hashlib.sha256(f"{from_hash}{to_hash}".encode("utf-8")).hexdigest()


def _diff_states(before_board: np.ndarray, after_board: np.ndarray):
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

        db.init(str(db_path))
        db.connect(reuse_if_open=True)
        db.create_tables(
            [
                Position,
                TransitionRow,
                PlayStats,
                ParentNode,
            ]
        )

        logger.info(f"GameMemory initialized: {db_path}")

    # ---------------------------------------------------------------------
    # Record outcome
    # ---------------------------------------------------------------------
    def record_outcome(
        self,
        game: GameBase,
        move: Any,
        acting_player: int,
        outcome: State,
    ) -> bool:
        try:
            game_id = game.game_id()
            before_state = game.get_state()
            player_to_move = before_state.current_player

            canon_before = self._get_canonical_board_cached(before_state.board)

            game_next = game.deep_clone()
            game_next.apply_move(move)
            after_state = game_next.get_state()
            canon_after = self._get_canonical_board_cached(after_state.board)

            t_hash = _transition_hash(canon_before.hash, canon_after.hash)

            with db.atomic():
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
                    defaults={
                        "from_hash": canon_before.hash,
                        "to_hash": canon_after.hash,
                    },
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
            logger.exception("Failed to record outcome")
            return False

    # ---------------------------------------------------------------------
    # Selection
    # ---------------------------------------------------------------------
    def get_best_move(self, game: GameBase, debug: bool = False) -> Optional[Any]:
        return self._select_move(game, best=True, debug=debug)

    def get_worst_move(self, game: GameBase, debug: bool = False) -> Optional[Any]:
        return self._select_move(game, best=False, debug=debug)

    def _select_move(self, game: GameBase, best: bool, debug: bool) -> Optional[Any]:
        game_id = game.game_id()
        state = game.get_state()
        player_to_move = state.current_player
        canon_from = self._get_canonical_board_cached(state.board)

        parent = ParentNode.get_or_none(
            game_id=game_id,
            position_hash=canon_from.hash,
            player_to_move=player_to_move,
        )

        pw = parent.win_count if parent else 0
        pt = parent.tie_count if parent else 0
        pn = parent.neutral_count if parent else 0
        pl = parent.loss_count if parent else 0

        query = (
            TransitionRow.select(TransitionRow, PlayStats)
            .join(
                PlayStats,
                on=(TransitionRow.transition_hash == PlayStats.transition_hash),
            )
            .where(
                (TransitionRow.game_id == game_id)
                & (TransitionRow.from_hash == canon_from.hash)
            )
        )

        stats_by_to_hash: Dict[str, PlayStats] = {
            row.to_hash: row.playstats for row in query
        }

        evaluations: List[TransitionEvaluation] = []

        for move in game.valid_moves():
            game_next = game.deep_clone()
            try:
                game_next.apply_move(move)
            except Exception:
                continue

            next_state = game_next.get_state()
            canon_next = self._get_canonical_board_cached(next_state.board)

            edge = stats_by_to_hash.get(canon_next.hash)
            if not edge:
                continue

            stats = MoveStatistics(
                wins=edge.win_count,
                ties=edge.tie_count,
                neutral=edge.neutral_count,
                losses=edge.loss_count,
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
        chosen = evaluations[0]

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

    # ---------------------------------------------------------------------
    # Canonicalization cache
    # ---------------------------------------------------------------------
    def _get_canonical_board_cached(self, board: np.ndarray) -> CanonicalResult:
        key = board.tobytes()
        cached = self._cache.get(key)
        if cached:
            return cached

        canonical = canonicalize_board(board)
        self._cache.put(key, canonical)
        return canonical

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------
    def close(self) -> None:
        self._cache.clear()
        if not db.is_closed():
            db.close()
