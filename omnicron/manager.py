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
import sqlite3
from typing import Any, List, Optional, Dict, Tuple, NamedTuple
from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path

import numpy as np

from agent.agent import State
from omnicron.state_canonicalizer import canonicalize_board, CanonicalResult
from omnicron.debug_viz import render_debug
from games.game_base import GameBase
from games.game_state import GameState


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    position_hash TEXT PRIMARY KEY,
    board_bytes BLOB,
    meta_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS transitions (
    transition_hash TEXT PRIMARY KEY,
    from_hash TEXT,
    to_hash TEXT
);
CREATE INDEX IF NOT EXISTS idx_from_hash ON transitions(from_hash);

CREATE TABLE IF NOT EXISTS stats (
    transition_hash TEXT PRIMARY KEY,
    wins INTEGER DEFAULT 0,
    ties INTEGER DEFAULT 0,
    neutral INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS parents (
    parent_key TEXT PRIMARY KEY,
    wins INTEGER DEFAULT 0,
    ties INTEGER DEFAULT 0,
    neutral INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0
);
"""


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


class Stats(NamedTuple):
    """Win/tie/neutral/loss counts."""

    wins: int
    ties: int
    neutral: int
    losses: int

    @property
    def total(self) -> int:
        return self.wins + self.ties + self.neutral + self.losses


class MoveStatistics(NamedTuple):
    """Statistics for a transition, including parent context."""

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
        return (self.wins - self.losses) / self.total if self.total else 0.0

    @property
    def score(self) -> float:
        """
        Lower confidence bound - balances mean and certainty.
        - We pick the move centered around the mean that we most confident about.
        - Sparse data decreases our score.
        - Ties dilute our score.
        """
        n = self.total
        if n == 0:
            return 0.5

        W, L = self.wins, self.losses
        mean = (W - L) / n

        if n == 1:
            return (mean + 1.0) / 2.0

        se = math.sqrt(((W + L) - n * mean * mean) / (n * (n - 1)))
        lcb = mean - se
        return (lcb + 1.0) / 2.0

    @property
    def certainty(self) -> float:
        """How confident we are (1 - standard error)."""
        n = self.total
        if n <= 1:
            return 1.0

        W, L = self.wins, self.losses
        mean = (W - L) / n
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
# Utilities
# ---------------------------------------------------------------------------

OUTCOME_INDEX = {State.WIN: 0, State.TIE: 1, State.NEUTRAL: 2, State.LOSS: 3}


def _transition_hash(from_hash: str, to_hash: str) -> str:
    return hashlib.sha256(f"{from_hash}{to_hash}".encode()).hexdigest()


def _parent_key(position_hash: str, player: int) -> str:
    return f"{position_hash}:{player}"


def _diff_states(before: np.ndarray, after: np.ndarray) -> List:
    return [
        (idx, before[idx], after[idx])
        for idx in np.ndindex(before.shape)
        if before[idx] != after[idx]
    ]


class LRUCache:
    """Simple LRU cache."""

    def __init__(self, maxsize: int = 4096):
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()

    def get(self, key):
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            self._cache[key] = value
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()


# ---------------------------------------------------------------------------
# Round Data Collector
# ---------------------------------------------------------------------------


class _RoundData:
    """Collects and deduplicates round data before writing."""

    def __init__(self):
        self.positions: Dict[str, bytes] = {}
        self.transitions: Dict[str, Tuple[str, str]] = {}
        self.stats: Dict[str, List[int]] = {}  # [W, T, N, L]
        self.parents: Dict[str, List[int]] = {}

    def add_position(self, pos_hash: str, board_bytes: bytes):
        if pos_hash not in self.positions:
            self.positions[pos_hash] = board_bytes

    def add_transition(self, t_hash: str, from_hash: str, to_hash: str):
        if t_hash not in self.transitions:
            self.transitions[t_hash] = (from_hash, to_hash)

    def record_outcome(self, t_hash: str, parent_key: str, outcome: State):
        idx = OUTCOME_INDEX[outcome]

        if t_hash not in self.stats:
            self.stats[t_hash] = [0, 0, 0, 0]
        self.stats[t_hash][idx] += 1

        if parent_key not in self.parents:
            self.parents[parent_key] = [0, 0, 0, 0]
        self.parents[parent_key][idx] += 1


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

    def __init__(self, db_path: str | Path = "memory.db", read_only: bool = False):
        db_path = Path(db_path)
        if not db_path.is_absolute():
            db_path = Path.cwd() / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.read_only = read_only
        self._canon_cache = LRUCache(maxsize=4096)
        self._stats_cache: Dict[str, Dict[str, Stats]] = {}
        self._parent_cache: Dict[str, Stats] = {}

        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA cache_size=-65536")

        if not read_only:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

        logger.info(
            "GameMemory (%s): %s", "read-only" if read_only else "read/write", db_path
        )

    @classmethod
    def for_game(
        cls, game: GameBase, base_dir: str | Path = "data/memory", **kwargs
    ) -> "GameMemory":
        """Create a partitioned GameMemory for a specific game type."""
        base_dir = Path(base_dir)
        return cls(base_dir / f"{game.game_id()}.db", **kwargs)

    # -------------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------------

    def record_round(
        self,
        game_class: type,
        stacks: List[Tuple[List[Tuple[Any, np.ndarray, int]], State]],
    ) -> int:
        """Record all game stacks atomically. Returns count of unique transitions."""
        if self.read_only:
            raise RuntimeError("Cannot write to read-only GameMemory")

        data = self._collect(game_class, stacks)
        if not data.transitions:
            return 0

        self._commit(data)
        return len(data.stats)

    def _collect(self, game_class: type, stacks) -> _RoundData:
        """Collect all transitions from a round."""
        data = _RoundData()

        for moves, outcome in stacks:
            for move, board_before, player in moves:
                canon_before = self._canonicalize(board_before)

                game = game_class()
                game.set_state(GameState(board_before.copy(), player))
                game.apply_move(move)
                board_after = game.get_state().board

                canon_after = self._canonicalize(board_after)
                t_hash = _transition_hash(canon_before.hash, canon_after.hash)
                p_key = _parent_key(canon_before.hash, player)

                data.add_position(canon_before.hash, board_before.tobytes())
                data.add_position(canon_after.hash, board_after.tobytes())
                data.add_transition(t_hash, canon_before.hash, canon_after.hash)
                data.record_outcome(t_hash, p_key, outcome)

        return data

    def _commit(self, data: _RoundData):
        """Write collected data atomically."""
        cur = self.conn.cursor()
        try:
            cur.execute("BEGIN")

            cur.executemany(
                "INSERT OR IGNORE INTO positions (position_hash, board_bytes) VALUES (?, ?)",
                data.positions.items(),
            )

            cur.executemany(
                "INSERT OR IGNORE INTO transitions (transition_hash, from_hash, to_hash) VALUES (?, ?, ?)",
                [(t, f, to) for t, (f, to) in data.transitions.items()],
            )

            cur.executemany(
                """
                INSERT INTO stats (transition_hash, wins, ties, neutral, losses) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(transition_hash) DO UPDATE SET
                    wins = wins + excluded.wins,
                    ties = ties + excluded.ties,
                    neutral = neutral + excluded.neutral,
                    losses = losses + excluded.losses
            """,
                [(t, *counts) for t, counts in data.stats.items()],
            )

            cur.executemany(
                """
                INSERT INTO parents (parent_key, wins, ties, neutral, losses) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(parent_key) DO UPDATE SET
                    wins = wins + excluded.wins,
                    ties = ties + excluded.ties,
                    neutral = neutral + excluded.neutral,
                    losses = losses + excluded.losses
            """,
                [(k, *counts) for k, counts in data.parents.items()],
            )

            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_stats(self, t_hash: str) -> Optional[Stats]:
        """Get stats for a transition."""
        row = self.conn.execute(
            "SELECT wins, ties, neutral, losses FROM stats WHERE transition_hash = ?",
            (t_hash,),
        ).fetchone()
        return Stats(*row) if row else None

    def get_parent_stats(self, position_hash: str, player: int) -> Stats:
        """Get stats for a parent position."""
        p_key = _parent_key(position_hash, player)

        if self.read_only and p_key in self._parent_cache:
            return self._parent_cache[p_key]

        row = self.conn.execute(
            "SELECT wins, ties, neutral, losses FROM parents WHERE parent_key = ?",
            (p_key,),
        ).fetchone()

        result = Stats(*row) if row else Stats(0, 0, 0, 0)

        if self.read_only:
            self._parent_cache[p_key] = result

        return result

    def get_transitions_from(self, from_hash: str) -> Dict[str, Stats]:
        """Get all transitions from a position as {to_hash: Stats}."""
        cache_key = from_hash

        if self.read_only and cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        rows = self.conn.execute(
            """
            SELECT t.to_hash, s.wins, s.ties, s.neutral, s.losses
            FROM transitions t
            JOIN stats s ON t.transition_hash = s.transition_hash
            WHERE t.from_hash = ?
        """,
            (from_hash,),
        ).fetchall()

        result = {row[0]: Stats(*row[1:]) for row in rows}

        if self.read_only:
            self._stats_cache[cache_key] = result

        return result

    # -------------------------------------------------------------------------
    # Move Selection
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
        return self._select_move(game, best=True, debug=debug)

    def get_worst_move_with_score(
        self, game: GameBase, debug: bool = False
    ) -> Optional[Tuple[Any, float]]:
        return self._select_move(game, best=False, debug=debug)

    def _select_move(
        self, game: GameBase, best: bool, debug: bool, deterministic: bool = False
    ) -> Optional[Tuple[Any, float]]:
        """Select a move based on stored statistics."""
        state = game.get_state()
        canon_from = self._canonicalize(state.board)

        parent = self.get_parent_stats(canon_from.hash, state.current_player)
        transitions = self.get_transitions_from(canon_from.hash)

        evaluations = []
        for move in game.valid_moves():
            game_next = game.deep_clone()
            try:
                game_next.apply_move(move)
            except Exception:
                continue

            next_state = game_next.get_state()
            to_hash = self._canonicalize(next_state.board).hash
            edge = transitions.get(to_hash)

            if not edge:
                continue

            evaluations.append(
                TransitionEvaluation(
                    move=move,
                    next_state=next_state,
                    to_hash=to_hash,
                    stats=MoveStatistics(
                        edge.wins,
                        edge.ties,
                        edge.neutral,
                        edge.losses,
                        parent.wins,
                        parent.ties,
                        parent.neutral,
                        parent.losses,
                    ),
                )
            )

        if not evaluations:
            return None

        evaluations.sort(key=lambda e: e.stats.score, reverse=best)

        if deterministic:
            chosen = evaluations[0]
        else:
            top_score = evaluations[0].stats.score
            ties = [e for e in evaluations if e.stats.score == top_score]
            chosen = random.choice(ties)

        if debug:
            self._render_debug(state, evaluations, chosen)

        return (chosen.move, chosen.stats.score)

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
        chosen: TransitionEvaluation,
    ):
        debug_rows = [
            {
                "diff": _diff_states(state.board, e.next_state.board),
                "move": e.move,
                "is_selected": np.array_equal(
                    e.next_state.board, chosen.next_state.board
                ),
                "is_equivalent": e.to_hash == chosen.to_hash
                and not np.array_equal(e.next_state.board, chosen.next_state.board),
                "score": e.stats.score,
                "utility": e.stats.utility,
                "certainty": e.stats.certainty,
                "total": e.stats.total,
                "pW": e.stats.win_rate,
                "pT": e.stats.tie_rate,
                "pN": e.stats.neutral_rate,
                "pL": e.stats.loss_rate,
            }
            for e in evaluations
        ]
        render_debug(state.board, debug_rows)

    def close(self):
        """Close database connection."""
        self._canon_cache.clear()
        self._stats_cache.clear()
        self._parent_cache.clear()

        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        self.conn.close()
