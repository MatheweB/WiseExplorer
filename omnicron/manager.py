"""
GameMemory: Pattern-based learning for game AI.

Architecture:
    transitions       - Raw (from_state, to_state) → outcome counts
    state_values      - Aggregated outcomes by state (Markov mode only)
    anchors           - Clusters of similar scoring units (pooled stats)
    scoring_anchors   - Maps scoring units to anchor clusters

Core Principle:
    The transition (from_hash, to_hash) encodes EVERYTHING:
    - The board before/after the move
    - Who moved (implicit in the state change)
    - What move was made (the difference between states)

    Outcomes are recorded per transition. A "win" means the game was won
    by whoever made that transition. No player numbers needed.

Markov vs Non-Markov:
    The system supports both Markov and non-Markov domains:

    MARKOV (markov=True, default):
        - State is a sufficient statistic (e.g., board position)
        - Different paths to same state have same continuation distribution
        - Score by destination state (aggregated across all predecessors)
        - Anchors cluster similar states
        - Examples: TTT, Chess with random play

    NON-MARKOV (markov=False):
        - History matters beyond current state (e.g., language)
        - Different paths to same state may have different continuations
        - Score by specific transition (preserves context)
        - Anchors cluster similar transitions
        - Examples: Language modeling, Poker, Chess with skilled players

The "scoring unit" abstraction:
    - Markov: scoring_key = to_hash (destination state)
    - Non-Markov: scoring_key = (from_hash, to_hash) (transition)

    All logic (Stats, anchors, scoring formula) operates uniformly on scoring keys.

Anchor System:
    Scoring units are clustered into anchors based on outcome distribution.
    Similar units share an anchor → pooled statistics for faster convergence.
    Merge threshold: BayesFactor > √(n_larger / n_smaller)
"""

from __future__ import annotations

import hashlib
import logging
import math
import sqlite3
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from agent.agent import State
from games.game_base import GameBase
from games.game_state import GameState

logger = logging.getLogger(__name__)

# Outcome index mapping
OUTCOME_INDEX = {State.WIN: 0, State.TIE: 1, State.LOSS: 2}

# Neutral score for unexplored moves
UNEXPLORED_ANCHOR_ID = -999


# ---------------------------------------------------------------------------
# Stats Type
# ---------------------------------------------------------------------------


class Stats(NamedTuple):
    """Outcome counts with derived scoring properties."""

    wins: int = 0
    ties: int = 0
    losses: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.ties + self.losses

    @property
    def distribution(self) -> Tuple[float, float, float]:
        t = self.total
        return (self.wins / t, self.ties / t, self.losses / t)

    @property
    def score(self) -> float:
        """
        Lower confidence bound on utility, normalized to [0, 1].
        
        Uses Bayesian approach with uniform Dirichlet prior (α=1).
        Utilities: Win=+1, Tie=0, Loss=-1
        """
        # Apply Bayesian pseudocounts (uniform Dirichlet prior)
        w, t, l = self.wins + 1, self.ties + 1, self.losses + 1
        n = w + t + l
        
        # Mean utility ∈ [-1, 1]
        mean = (w + 0.5 * t - 1.5 * l) / n
        
        # Variance via Var(X) = E[X²] - E[X]²
        # Where E[X²] = (wins × 1² + ties × 0.5² + losses × (-1.5)²) / n
        mean_of_squares = (w + 0.25 * t + 2.25 * l) / n
        variance = mean_of_squares - mean**2
        
        # Lower confidence bound (LCB = μ - σ/√n)
        std_error = math.sqrt(max(0, variance / n))
        lcb = mean - std_error
        
        # Map [-1.5, 1] → [0, 1]
        return (lcb + 1.5) / 2.5

    @property
    def utility(self) -> float:
        """Expected value in [-1, 1]."""
        return (self.wins - self.losses) / self.total if self.total else 0.0

    @property
    def certainty(self) -> float:
        """
        Confidence in the utility estimate.
        
        Returns 1 - std_error, where std_error ∈ [0, 1].
        Higher values = more certain about the utility estimate.
        """
        if self.total <= 1:
            return 0.0
        
        # Use same pseudocounts as score
        w, t, l = self.wins + 1, self.ties + 1, self.losses + 1
        n = w + t + l
        
        # Mean utility on [-1.5, 1]
        mean = (w + 0.5 * t - 1.5 * l) / n
        
        # Variance: E[X²] - E[X]²
        mean_of_squares = (w + 0.25 * t + 2.25 * l) / n
        variance = mean_of_squares - mean**2
        
        # Standard error
        std_error = math.sqrt(max(0, variance / n))
        
        # Convert to certainty: lower std_error = higher certainty
        # Max std_error ≈ 1.0 for utilities in [-1, 1]
        return max(0.0, 1.0 - std_error)


# ---------------------------------------------------------------------------
# Hashing & Statistics
# ---------------------------------------------------------------------------


def _hash(board: np.ndarray) -> str:
    """Hash board state to 16-char hex string."""
    data = repr(board.tolist()).encode() if board.dtype == object else board.tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


@lru_cache(maxsize=4096)
def _log_dm_marginal(counts: Tuple[int, ...]) -> float:
    """
    Log marginal likelihood under Dirichlet-Multinomial with α=1 (uniform prior).

    P(counts | α=1) = (K-1)! / (n+K-1)! × ∏ c_i!
    """
    k, n = len(counts), sum(counts)
    if n == 0:
        return 0.0
    result = math.lgamma(k) - math.lgamma(n + k)
    for c in counts:
        result += math.lgamma(c + 1)
    return result


def _log_bayes_factor(counts1: Tuple[int, ...], counts2: Tuple[int, ...]) -> float:
    """
    Log Bayes factor: P(data|same) / P(data|different).
    Positive = evidence for same distribution.
    """
    pooled = tuple(c1 + c2 for c1, c2 in zip(counts1, counts2))
    return _log_dm_marginal(pooled) - (
        _log_dm_marginal(counts1) + _log_dm_marginal(counts2)
    )


def _compatible(counts1: Tuple[int, ...], counts2: Tuple[int, ...]) -> bool:
    """
    Are these distributions statistically compatible? (Parameter-free)

    Uses Bayes factor with sqrt-scaled threshold:
        threshold = 0.5 * log(n_large / n_small)

    This is the "information asymmetry" dampened by sqrt, giving:
        - Equal sizes: threshold = 0 (easy merge if BF > 0)
        - 10:1 ratio: threshold = 1.15
        - 100:1 ratio: threshold = 2.3
        - 1000:1 ratio: threshold = 3.45

    Used consistently for:
        - Checking if transition should STAY in anchor
        - Checking if transition should JOIN anchor
        - Checking if anchors should MERGE
        - Deciding whether to use anchor vs direct stats
    """
    n1, n2 = sum(counts1), sum(counts2)
    if n1 == 0:
        return True
    if n2 == 0:
        return False
    log_bf = _log_bayes_factor(counts1, counts2)
    log_threshold = 0.5 * math.log(max(n1, n2) / min(n1, n2))
    return log_bf > log_threshold


def _similarity(counts1: Tuple[int, ...], counts2: Tuple[int, ...]) -> float:
    """Similarity via sigmoid of Bayes factor. Returns [0, 1]."""
    n1, n2 = sum(counts1), sum(counts2)
    if n1 == 0 or n2 == 0:
        return 0.0
    log_bf = max(-50, min(50, _log_bayes_factor(counts1, counts2)))
    return 1.0 / (1.0 + math.exp(-log_bf))


# ---------------------------------------------------------------------------
# Database Schema
# ---------------------------------------------------------------------------

SCHEMA = """
-- Transitions: raw (from_state, to_state) → outcomes
-- Always populated regardless of mode
CREATE TABLE IF NOT EXISTS transitions (
    from_hash TEXT, to_hash TEXT,
    wins INTEGER DEFAULT 0, ties INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    PRIMARY KEY (from_hash, to_hash)
);
CREATE INDEX IF NOT EXISTS idx_from ON transitions(from_hash);
CREATE INDEX IF NOT EXISTS idx_to ON transitions(to_hash);

-- State values: aggregated outcomes by destination state
-- Only populated in Markov mode (markov=True)
CREATE TABLE IF NOT EXISTS state_values (
    state_hash TEXT PRIMARY KEY,
    wins INTEGER DEFAULT 0,
    ties INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0
);

-- Anchors: clusters of similar scoring units (pooled stats)
-- In Markov mode: clusters states
-- In non-Markov mode: clusters transitions
CREATE TABLE IF NOT EXISTS anchors (
    anchor_id INTEGER PRIMARY KEY,
    repr_key TEXT,  -- state_hash (Markov) or "from|to" (non-Markov)
    wins INTEGER DEFAULT 0, ties INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0
);

-- Scoring anchors: maps scoring units to anchor clusters
-- scoring_key is state_hash (Markov) or "from|to" (non-Markov)
CREATE TABLE IF NOT EXISTS scoring_anchors (
    scoring_key TEXT PRIMARY KEY,
    anchor_id INTEGER
);

-- Metadata: stores mode and other settings
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


# ---------------------------------------------------------------------------
# GameMemory
# ---------------------------------------------------------------------------


class GameMemory:
    """
    Game transition storage with automatic pattern discovery.

    Records (state, move, outcome) and clusters similar patterns into anchors.

    Args:
        db_path: Path to SQLite database
        read_only: If True, don't modify database
        markov: If True (default), use state-based scoring (Markov assumption).
                If False, use transition-based scoring (non-Markov, preserves context).
    """

    def __init__(
        self,
        db_path: str | Path = "memory.db",
        read_only: bool = False,
        markov: bool = False,
    ):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.read_only = read_only
        self.markov = markov
        self._cache: Dict[str, Dict[str, Stats]] = {}
        self._anchors_dirty = True

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA cache_size=-65536")
        if not read_only:
            self.conn.executescript(SCHEMA)
            # Store mode in metadata
            self.conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES ('markov', ?)",
                ("true" if markov else "false",),
            )
            self.conn.commit()
        else:
            # Read mode from database if available
            row = self.conn.execute(
                "SELECT value FROM metadata WHERE key='markov'"
            ).fetchone()
            if row:
                self.markov = row[0] == "true"

    def _scoring_key(self, from_hash: str, to_hash: str) -> str:
        """Get the scoring key based on mode.

        Markov: key = to_hash (destination state)
        Non-Markov: key = "from|to" (full transition)
        """
        if self.markov:
            return to_hash
        else:
            return f"{from_hash}|{to_hash}"

    @classmethod
    def for_game(
        cls,
        game: GameBase,
        base_dir: str | Path = "data/memory",
        markov: bool = True,
        **kw,
    ) -> "GameMemory":
        """Create GameMemory for a specific game type."""
        game_id = getattr(game, "game_id", lambda: type(game).__name__.lower())()
        return cls(Path(base_dir) / f"{game_id}.db", markov=markov, **kw)

    # -------------------------------------------------------------------------
    # Move Evaluation Helpers
    # -------------------------------------------------------------------------

    def _iter_moves_with_hashes(
        self, game: GameBase, valid_moves: List[np.ndarray]
    ) -> List[Tuple[np.ndarray, str]]:
        """Apply each move to clone and return (move, to_hash) pairs."""
        results = []
        for move in valid_moves:
            clone = game.deep_clone()
            try:
                clone.apply_move(move)
                results.append((move, _hash(clone.get_state().board)))
            except (ValueError, IndexError):
                continue
        return results

    def _get_move_data(
        self, from_hash: str, to_hash: str, transitions: Dict[str, Stats]
    ) -> Optional[Dict]:
        """Get full stats for a move, or None if unexplored."""
        if to_hash not in transitions:
            return None

        direct = transitions[to_hash]
        if direct.total == 0:
            return None

        scoring_key = self._scoring_key(from_hash, to_hash)

        if self.markov:
            # Markov mode: use state-based stats
            unit_stats = self.get_state_stats(to_hash)
        else:
            # Non-Markov mode: use transition-specific stats
            unit_stats = self.get_stats(from_hash, to_hash)

        anchor_stats = self._get_anchor_stats(scoring_key)
        effective = self.get_effective_stats(scoring_key)

        return {
            "direct": direct,
            "unit": unit_stats,  # Renamed from "state" for generality
            "anchor": anchor_stats,
            "anchor_id": self._get_anchor_id(scoring_key),
            "effective": effective,
        }

    # -------------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------------

    def record_round(
        self,
        game_class: type,
        stacks: List[Tuple[List[Tuple[Any, np.ndarray, int]], State]],
    ) -> int:
        """
        Record game outcomes atomically. Returns number of unique transitions.

        stacks: List of (moves, outcome) where moves is [(move, board, player), ...]
        """
        if self.read_only:
            raise RuntimeError("Cannot write to read-only GameMemory")

        transitions: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0, 0])

        for moves, outcome in stacks:
            outcome_idx = OUTCOME_INDEX.get(outcome, -1)
            game = game_class()
            for move, board, player in moves:
                from_hash = _hash(board)
                game.set_state(GameState(board.copy(), player))
                game.apply_move(move)
                to_hash = _hash(game.get_state().board)
                transitions[(from_hash, to_hash)][outcome_idx] += 1

        self._commit(transitions)
        return len(transitions)

    def _commit(self, transitions: Dict[Tuple[str, str], List[int]]) -> None:
        """Write transitions and update anchors."""
        if not transitions:
            return

        cur = self.conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            # Always update transitions table
            cur.executemany(
                """INSERT INTO transitions VALUES (?,?,?,?,?)
                   ON CONFLICT DO UPDATE SET
                   wins=wins+excluded.wins, ties=ties+excluded.ties,
                   losses=losses+excluded.losses""",
                [(f, t, *c) for (f, t), c in transitions.items()],
            )

            if self.markov:
                # Markov mode: aggregate outcomes by destination state
                state_updates = defaultdict(lambda: [0, 0, 0])
                for (_, to_hash), (w, t, l) in transitions.items():
                    state_updates[to_hash][0] += w
                    state_updates[to_hash][1] += t
                    state_updates[to_hash][2] += l

                cur.executemany(
                    """INSERT INTO state_values (state_hash, wins, ties, losses)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(state_hash) DO UPDATE SET
                           wins = wins + excluded.wins,
                           ties = ties + excluded.ties,
                           losses = losses + excluded.losses""",
                    [(s, *counts) for s, counts in state_updates.items()],
                )
                scoring_keys = list(state_updates.keys())
            else:

                # Non-Markov mode: scoring keys are transitions
                scoring_keys = [f"{f}|{t}" for f, t in transitions.keys()]

            cur.execute("COMMIT")
            self._cache.clear()
            # Update anchors for modified scoring units
            self._update_anchors(scoring_keys, cur)
        except Exception:
            cur.execute("ROLLBACK")
            raise

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_stats(self, from_hash: str = None, to_hash: str = None) -> Stats:
        """Get transition stats, or database summary if no args."""
        if from_hash is None and to_hash is None:
            return self.get_info()
        if from_hash is None or to_hash is None:
            raise ValueError("Provide both from_hash and to_hash, or neither")
        row = self.conn.execute(
            "SELECT wins,ties,losses FROM transitions WHERE from_hash=? AND to_hash=?",
            (from_hash, to_hash),
        ).fetchone()
        return Stats(*row) if row else Stats()

    def get_effective_stats(self, scoring_key: str) -> Stats:
        """
        Get stats for a scoring unit, preferring anchor when it provides more data.

        Args:
            scoring_key: State hash (Markov) or "from|to" (non-Markov)

        Returns:
            Stats from direct observation or anchor, whichever is more informative.
        """
        direct = self._get_unit_stats(scoring_key)
        anchor = self._get_anchor_stats(scoring_key)

        if anchor.total <= direct.total:
            return direct
        if direct.total > 0 and not _compatible(tuple(direct), tuple(anchor)):
            return direct
        return anchor

    def _get_unit_stats(self, scoring_key: str) -> Stats:
        """Get stats for a scoring unit (state or transition)."""
        if self.markov:
            return self.get_state_stats(scoring_key)
        else:
            # Non-Markov: scoring_key is "from|to"
            parts = scoring_key.split("|", 1)
            if len(parts) == 2:
                return self.get_stats(parts[0], parts[1])
            return Stats()

    # -------------------------------------------------------------------------
    # State Values (for scoring)
    # -------------------------------------------------------------------------

    def get_state_stats(self, state_hash: str) -> Stats:
        """
        Get aggregated Stats for transitions ending at this state.

        Returns Stats with the combined (wins, ties, losses) from
        all transitions that led to this state. Returns empty Stats if
        no transitions to this state have been recorded.
        """
        row = self.conn.execute(
            "SELECT wins, ties, losses FROM state_values WHERE state_hash = ?",
            (state_hash,),
        ).fetchone()
        return Stats(*row) if row else Stats()

    def get_move_score(self, from_hash: str, to_hash: str) -> float:
        """
        Get score for a move (from_hash → to_hash).

        In Markov mode: Uses aggregated state stats (destination state).
        In non-Markov mode: Uses transition-specific stats (preserves context).

        Returns:
            Score in [0, 1] using Stats.score (LCB formula)
        """
        if self.markov:
            stats = self.get_state_stats(to_hash)
        else:
            stats = self.get_stats(from_hash, to_hash)

        return stats.score

    def get_state_value_stats(self) -> Dict[str, Any]:
        """Get statistics about state value aggregation."""
        total_states = self.conn.execute(
            "SELECT COUNT(DISTINCT to_hash) FROM transitions"
        ).fetchone()[0]

        tracked_states = self.conn.execute(
            "SELECT COUNT(*) FROM state_values WHERE (wins + ties + losses) > 0"
        ).fetchone()[0]

        avg_count = self.conn.execute(
            "SELECT AVG(wins + ties + losses) FROM state_values WHERE (wins + ties + losses) > 0"
        ).fetchone()[0]

        return {
            "total_states": total_states,
            "tracked_states": tracked_states,
            "avg_sample_count": avg_count or 0,
            "coverage": tracked_states / total_states if total_states > 0 else 0,
        }

    def get_transitions_from(self, from_hash: str) -> Dict[str, Stats]:
        """All transitions from a state (cached)."""
        if from_hash not in self._cache:
            rows = self.conn.execute(
                "SELECT to_hash,wins,ties,losses FROM transitions WHERE from_hash=?",
                (from_hash,),
            ).fetchall()
            self._cache[from_hash] = {r[0]: Stats(*r[1:]) for r in rows}
        return self._cache[from_hash]

    def _get_anchor_stats(self, scoring_key: str) -> Stats:
        """Get anchor stats for a scoring unit."""
        row = self.conn.execute(
            """SELECT a.wins, a.ties, a.losses 
               FROM scoring_anchors sa
               JOIN anchors a ON sa.anchor_id = a.anchor_id
               WHERE sa.scoring_key = ?""",
            (scoring_key,),
        ).fetchone()
        return Stats(*row) if row else self._get_unit_stats(scoring_key)

    def _get_anchor_id(self, scoring_key: str) -> Optional[int]:
        """Get anchor ID for a scoring unit."""
        row = self.conn.execute(
            "SELECT anchor_id FROM scoring_anchors WHERE scoring_key = ?",
            (scoring_key,),
        ).fetchone()
        return row[0] if row else None

    # -------------------------------------------------------------------------
    # Move Evaluation (for selection)
    # -------------------------------------------------------------------------

    def get_all_moves_with_scores(
        self, game: GameBase, valid_moves: List[np.ndarray]
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Get scores for all valid moves.

        Uses aggregated state stats for scoring (more data than transition-specific).
        Unrecorded moves get default from unexplored score (~0.74)
        """
        state = game.get_state()
        from_hash = _hash(state.board)

        results = []
        for move, to_hash in self._iter_moves_with_hashes(game, valid_moves):
            score = self.get_move_score(from_hash, to_hash)
            results.append((move, score))
        return results

    def evaluate_moves_for_selection(
        self, game: GameBase, valid_moves: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Evaluate all moves for selection.

        Returns:
            anchors_with_moves: {anchor_id: [(move, score), ...]}
            anchor_scores: {anchor_id: pooled_score}
        """
        state = game.get_state()
        from_hash = _hash(state.board)
        transitions = self.get_transitions_from(from_hash)

        anchors_with_moves: Dict[int, List[Tuple[np.ndarray, float]]] = defaultdict(list)
        anchor_scores: Dict[int, float] = {}

        for move, to_hash in self._iter_moves_with_hashes(game, valid_moves):
            data = self._get_move_data(from_hash, to_hash, transitions)

            if data and data["effective"].total > 0:
                # Known move
                aid = data["anchor_id"]
                move_score = self.get_move_score(from_hash, to_hash)
                anchors_with_moves[aid].append((move, move_score))
                anchor_scores[aid] = data["effective"].score
            else:
                # Unexplored move - use natural LCB from pseudocounts
                unexplored_stats = Stats(0, 0, 0)
                unexplored_score = unexplored_stats.score
                
                # Each unexplored move gets its own synthetic anchor
                # (or group them together, but use the natural score)
                anchors_with_moves[UNEXPLORED_ANCHOR_ID].append((move, unexplored_score))
                anchor_scores[UNEXPLORED_ANCHOR_ID] = unexplored_score

        return {
            "anchors_with_moves": dict(anchors_with_moves),
            "anchor_scores": anchor_scores,
        }

    # -------------------------------------------------------------------------
    # Anchor System
    # -------------------------------------------------------------------------

    def _ensure_anchors(self) -> None:
        """Lazy anchor initialization."""
        if not self._anchors_dirty:
            return
        if self.read_only:
            self._anchors_dirty = False
            return
        if self.conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0] == 0:
            has_data = False
            if self.markov:
                has_data = (
                    self.conn.execute("SELECT COUNT(*) FROM state_values").fetchone()[0]
                    > 0
                )
            else:
                has_data = (
                    self.conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
                    > 0
                )
            if has_data:
                self.rebuild_anchors()
        self._anchors_dirty = False

    def _update_anchors(self, scoring_keys: List[str], cur: sqlite3.Cursor) -> None:
        """Incremental anchor maintenance after recording."""
        if not scoring_keys:
            return

        # Load changed scoring units
        changed = {}
        for key in scoring_keys:
            stats = self._get_unit_stats(key)
            if stats.total > 0:
                counts = tuple(stats)
                changed[key] = {
                    "counts": counts,
                    "dist": tuple(x / stats.total for x in counts),
                }

        if not changed:
            return

        cur.execute("BEGIN IMMEDIATE")
        try:
            # Load existing anchors
            anchors = {}
            for aid, repr_key, w, t, l in cur.execute(
                "SELECT anchor_id, repr_key, wins, ties, losses FROM anchors"
            ):
                total = w + t + l
                if total > 0:
                    anchors[aid] = {
                        "counts": (w, t, l),
                        "dist": (w / total, t / total, l / total),
                        "repr": repr_key,
                    }
            max_id = max(anchors.keys(), default=-1)

            # Get existing memberships
            existing = {}
            placeholders = ",".join("?" for _ in changed)
            rows = cur.execute(
                f"SELECT scoring_key, anchor_id FROM scoring_anchors WHERE scoring_key IN ({placeholders})",
                list(changed.keys()),
            ).fetchall()
            existing = {r[0]: r[1] for r in rows}

            modified, moved_from = set(), set()

            for key, data in changed.items():
                old_aid = existing.get(key)

                # Check if we should STAY in current anchor
                stay_in_current = False
                if old_aid is not None and old_aid in anchors:
                    if _compatible(data["counts"], anchors[old_aid]["counts"]):
                        stay_in_current = True
                    else:
                        moved_from.add(old_aid)

                if stay_in_current:
                    modified.add(old_aid)
                    continue

                # Look for a new home
                best_aid = self._find_nearest(data["counts"], anchors)
                can_join = best_aid is not None and _compatible(
                    data["counts"], anchors[best_aid]["counts"]
                )

                if can_join:
                    new_aid = best_aid
                else:
                    # Create new anchor
                    max_id += 1
                    new_aid = max_id
                    cur.execute(
                        "INSERT INTO anchors VALUES (?, ?, ?, ?, ?)",
                        (new_aid, key, *data["counts"]),
                    )
                    anchors[new_aid] = {
                        "counts": data["counts"],
                        "dist": data["dist"],
                        "repr": key,
                    }

                # Update membership
                if old_aid is None:
                    cur.execute(
                        "INSERT INTO scoring_anchors VALUES (?, ?)", (key, new_aid)
                    )
                elif old_aid != new_aid:
                    cur.execute(
                        "UPDATE scoring_anchors SET anchor_id = ? WHERE scoring_key = ?",
                        (new_aid, key),
                    )

                modified.add(new_aid)

            # Recalculate affected anchors
            for aid in moved_from | modified:
                self._recalc_anchor(aid, anchors, cur)

            # Consolidate
            if moved_from | modified:
                self._consolidate(anchors, cur, moved_from | modified)

            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def _find_nearest(
        self, counts: Tuple[int, ...], anchors: Dict[int, dict]
    ) -> Optional[int]:
        """Find most similar anchor by Bayes factor."""
        if not anchors:
            return None
        return max(
            anchors.keys(), key=lambda aid: _similarity(counts, anchors[aid]["counts"])
        )

    def _recalc_anchor(
        self, aid: int, anchors: Dict[int, dict], cur: sqlite3.Cursor
    ) -> None:
        """Recalculate anchor stats from members."""
        if self.markov:
            # Join with state_values
            row = cur.execute(
                """SELECT COALESCE(SUM(sv.wins),0), COALESCE(SUM(sv.ties),0),
                          COALESCE(SUM(sv.losses),0), COUNT(*)
                   FROM scoring_anchors sa
                   JOIN state_values sv ON sa.scoring_key = sv.state_hash
                   WHERE sa.anchor_id = ?""",
                (aid,),
            ).fetchone()
        else:
            # Non-Markov: Parse "from|to" keys and join with transitions
            # This is more complex - need to handle the key format
            # For simplicity, sum from all members
            members = cur.execute(
                "SELECT scoring_key FROM scoring_anchors WHERE anchor_id = ?",
                (aid,),
            ).fetchall()

            w, t, l = 0, 0, 0
            for (key,) in members:
                parts = key.split("|", 1)
                if len(parts) == 2:
                    r = cur.execute(
                        "SELECT wins, ties, losses FROM transitions WHERE from_hash=? AND to_hash=?",
                        (parts[0], parts[1]),
                    ).fetchone()
                    if r:
                        w += r[0]
                        t += r[1]
                        l += r[2]
            row = (w, t, l, len(members))

        w, t, l, count = row
        total = w + t + l

        if count == 0 or total == 0:
            cur.execute("DELETE FROM anchors WHERE anchor_id = ?", (aid,))
            anchors.pop(aid, None)
            return

        cur.execute(
            "UPDATE anchors SET wins = ?, ties = ?, losses = ? WHERE anchor_id = ?",
            (w, t, l, aid),
        )
        anchors[aid] = {
            "counts": (w, t, l),
            "dist": (w / total, t / total, l / total),
            "repr": anchors.get(aid, {}).get("repr"),
        }

    def _consolidate(
        self, anchors: Dict[int, dict], cur: sqlite3.Cursor, modified_only: set
    ) -> None:
        """Merge compatible anchors (O(m*n) where m=modified)."""
        if len(anchors) < 2:
            return

        to_check = set(modified_only)
        merged = True

        while merged:
            merged = False
            for aid1 in list(to_check):
                if aid1 not in anchors:
                    to_check.discard(aid1)
                    continue

                for aid2 in list(anchors.keys()):
                    if aid2 == aid1 or aid2 not in anchors:
                        continue

                    a1, a2 = anchors[aid1], anchors[aid2]

                    if _compatible(a1["counts"], a2["counts"]):
                        n1, n2 = sum(a1["counts"]), sum(a2["counts"])
                        if n1 >= n2:
                            survivor, absorbed = aid1, aid2
                        else:
                            survivor, absorbed = aid2, aid1

                        cur.execute(
                            "UPDATE scoring_anchors SET anchor_id = ? WHERE anchor_id = ?",
                            (survivor, absorbed),
                        )
                        cur.execute(
                            "DELETE FROM anchors WHERE anchor_id = ?", (absorbed,)
                        )
                        del anchors[absorbed]
                        to_check.discard(absorbed)
                        self._recalc_anchor(survivor, anchors, cur)
                        to_check.add(survivor)
                        merged = True
                        break
                if merged:
                    break

    def rebuild_anchors(self) -> int:
        """Full anchor rebuild. Returns anchor count."""
        if self.read_only:
            raise RuntimeError("Read-only")

        # Get all scoring units with their stats
        units = []
        if self.markov:
            rows = self.conn.execute(
                """SELECT state_hash, wins, ties, losses
                   FROM state_values WHERE wins + ties + losses > 0"""
            ).fetchall()
            for key, w, ti, l in rows:
                total = w + ti + l
                dist = (w / total, ti / total, l / total)
                entropy = -sum(p * math.log(p + 1e-12) for p in dist)
                units.append(
                    {"key": key, "counts": (w, ti, l), "dist": dist, "entropy": entropy}
                )
        else:
            # Non-Markov: use transitions as scoring units
            rows = self.conn.execute(
                """SELECT from_hash, to_hash, wins, ties, losses
                   FROM transitions WHERE wins + ties + losses > 0"""
            ).fetchall()
            for fh, th, w, ti, l in rows:
                key = f"{fh}|{th}"
                total = w + ti + l
                dist = (w / total, ti / total, l / total)
                entropy = -sum(p * math.log(p + 1e-12) for p in dist)
                units.append(
                    {"key": key, "counts": (w, ti, l), "dist": dist, "entropy": entropy}
                )

        if not units:
            return 0

        # Sort by entropy (most decisive first)
        units.sort(key=lambda x: x["entropy"])

        # Cluster
        anchors, membership = [], {}
        for unit in units:
            best_idx = None
            if anchors:
                best_idx = max(
                    range(len(anchors)),
                    key=lambda i: _similarity(unit["counts"], anchors[i]["counts"]),
                )

            compatible = best_idx is not None and _compatible(
                unit["counts"], anchors[best_idx]["counts"]
            )

            if compatible:
                a = anchors[best_idx]
                a["counts"] = tuple(
                    a["counts"][i] + unit["counts"][i] for i in range(3)
                )
                total = sum(a["counts"])
                a["dist"] = tuple(c / total for c in a["counts"])
                membership[unit["key"]] = best_idx
            else:
                membership[unit["key"]] = len(anchors)
                anchors.append(
                    {
                        "counts": unit["counts"],
                        "dist": unit["dist"],
                        "repr": unit["key"],
                    }
                )

        # Persist
        cur = self.conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            cur.execute("DELETE FROM anchors")
            cur.execute("DELETE FROM scoring_anchors")
            for i, a in enumerate(anchors):
                cur.execute(
                    "INSERT INTO anchors VALUES (?, ?, ?, ?, ?)",
                    (i, a["repr"], *a["counts"]),
                )
            cur.executemany(
                "INSERT INTO scoring_anchors VALUES (?, ?)",
                [(k, v) for k, v in membership.items()],
            )
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

        return len(anchors)

    # -------------------------------------------------------------------------
    # Debug Visualization
    # -------------------------------------------------------------------------

    def debug_move_selection(
        self,
        game: GameBase,
        valid_moves: List[np.ndarray],
        chosen_move: Optional[np.ndarray] = None,
    ) -> None:
        """Render debug visualization for move selection."""
        try:
            from omnicron.debug_viz import render_debug
        except ImportError:
            try:
                from debug_viz import render_debug
            except ImportError:
                print("debug_viz not available")
                return

        state = game.get_state()
        from_hash = _hash(state.board)
        transitions = self.get_transitions_from(from_hash)

        # Find chosen move's hash
        chosen_to_hash = None
        if chosen_move is not None:
            clone = game.deep_clone()
            try:
                clone.apply_move(chosen_move)
                chosen_to_hash = _hash(clone.get_state().board)
            except (ValueError, IndexError):
                pass

        # Build debug rows
        debug_rows = []
        for move, to_hash in self._iter_moves_with_hashes(game, valid_moves):
            clone = game.deep_clone()
            clone.apply_move(move)
            next_state = clone.get_state()

            data = self._get_move_data(from_hash, to_hash, transitions)
            is_selected = to_hash == chosen_to_hash

            # Board diff
            diff = [
                (i, state.board[i], next_state.board[i])
                for i in np.ndindex(state.board.shape)
                if state.board[i] != next_state.board[i]
            ]

            if data and data["effective"].total > 0:
                eff, direct, anchor = data["effective"], data["direct"], data["anchor"]
                debug_rows.append(
                    {
                        "diff": diff,
                        "move": move,
                        "is_selected": is_selected,
                        "score": eff.score,
                        "utility": eff.utility,
                        "certainty": eff.certainty,
                        "total": eff.total,
                        "pW": eff.wins / eff.total,
                        "pT": eff.ties / eff.total,
                        "pL": eff.losses / eff.total,
                        "direct_total": direct.total,
                        "direct_W": direct.wins,
                        "direct_T": direct.ties,
                        "direct_L": direct.losses,
                        "anchor_id": data["anchor_id"],
                        "anchor_total": anchor.total,
                        "anchor_W": anchor.wins,
                        "anchor_L": anchor.losses,
                        "direct_score": direct.score,
                        "using_anchor": eff.total == anchor.total
                        and anchor.total > direct.total,
                    }
                )
            else:
                debug_rows.append(
                    {
                        "diff": diff,
                        "move": move,
                        "is_selected": is_selected,
                        "score": 0.0,
                        "utility": 0.0,
                        "certainty": 0.0,
                        "total": 0,
                        "pW": 0.0,
                        "pT": 0.0,
                        "pL": 0.0,
                        "direct_total": 0,
                        "direct_W": 0,
                        "direct_T": 0,
                        "direct_L": 0,
                        "anchor_id": None,
                        "anchor_total": 0,
                        "anchor_W": 0,
                        "anchor_L": 0,
                        "direct_score": 0.0,
                        "using_anchor": False,
                        "unexplored": True,
                    }
                )

        if debug_rows:
            render_debug(state.board, debug_rows)
        else:
            print("No candidates to display")

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        """Database statistics."""
        trans = self.conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
        anchors = self.conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
        samples = self.conn.execute(
            "SELECT COALESCE(SUM(wins+ties+losses),0) FROM transitions"
        ).fetchone()[0]
        from_states = self.conn.execute(
            "SELECT COUNT(DISTINCT from_hash) FROM transitions"
        ).fetchone()[0]
        to_states = self.conn.execute(
            "SELECT COUNT(DISTINCT to_hash) FROM transitions"
        ).fetchone()[0]
        return {
            "transitions": trans,
            "from_states": from_states,
            "to_states": to_states,
            "total_samples": samples,
            "anchors": anchors,
            "compression_ratio": to_states / anchors if anchors else 1.0,
        }

    def get_stats_summary(self) -> Dict[str, Any]:
        """Alias for get_info."""
        return self.get_info()

    def get_anchor_details(self) -> List[Dict[str, Any]]:
        """Detailed anchor information."""
        rows = self.conn.execute(
            """SELECT a.anchor_id, a.repr_key, a.wins, a.ties, a.losses,
                      COUNT(sa.scoring_key)
               FROM anchors a LEFT JOIN scoring_anchors sa ON a.anchor_id = sa.anchor_id
               GROUP BY a.anchor_id ORDER BY (a.wins + a.ties + a.losses) DESC"""
        ).fetchall()

        return [
            {
                "anchor_id": r[0],
                "repr_key": r[1],
                "wins": r[2],
                "ties": r[3],
                "losses": r[4],
                "total": (t := r[2] + r[3] + r[4]),
                "members": r[5],
                "distribution": (r[2] / t, r[3] / t, r[4] / t) if t else (0, 0, 0),
            }
            for r in rows
        ]

    def get_anchor_for_key(self, scoring_key: str) -> Optional[str]:
        """Get representative key for this scoring unit's anchor."""
        self._ensure_anchors()
        row = self.conn.execute(
            """SELECT a.repr_key FROM scoring_anchors sa
               JOIN anchors a ON sa.anchor_id = a.anchor_id
               WHERE sa.scoring_key = ?""",
            (scoring_key,),
        ).fetchone()
        return row[0] if row else None

    def get_anchor_stats_for_key(self, scoring_key: str) -> Stats:
        """Public API for anchor stats by scoring key."""
        self._ensure_anchors()
        return self._get_anchor_stats(scoring_key)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def close(self) -> None:
        self._cache.clear()
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError:
            pass
        self.conn.close()

    def __enter__(self) -> "GameMemory":
        return self

    def __exit__(self, *_) -> None:
        self.close()
