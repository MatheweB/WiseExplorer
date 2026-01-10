"""
GameMemory: Parameter-free behavioral pattern learning for game AI.

ONE RULE: Bayesian Compatibility Test with Evidence Scaling
    Merge if BF > √(n_larger / n_smaller)

Uses Dirichlet-Multinomial Bayes factor with uniform prior (α=1).
The threshold scales with sample size imbalance - sparse transitions need
proportionally stronger evidence to join large anchors. This prevents
premature grouping while allowing natural convergence as data accumulates.

Architecture:
    transitions: Raw (from_state, to_state) → outcome counts
    anchors: Emergent behavioral patterns (stats = sum of members)
    transition_anchors: Which anchor each transition belongs to
"""

from __future__ import annotations
import hashlib
import logging
import math
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from numpy.typing import NDArray
import numpy as np

from agent.agent import State
from games.game_base import GameBase
from games.game_state import GameState


logger = logging.getLogger(__name__)
OUTCOME_INDEX = {State.WIN: 0, State.TIE: 1, State.NEUTRAL: 2, State.LOSS: 3}

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class Stats(NamedTuple):
    """Outcome counts with derived properties."""

    wins: int = 0
    ties: int = 0
    neutral: int = 0
    losses: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.ties + self.neutral + self.losses

    @property
    def distribution(self) -> Tuple[float, float, float, float]:
        t = self.total
        return tuple(x / t for x in self)

    @property
    def score(self) -> float:
        """
        Lower confidence bound on utility, normalized to [0, 1].
        
        Uses Bayesian approach with uniform Dirichlet prior (α=1 pseudocounts).
        This prevents extreme scores from small samples while converging to
        empirical values as data accumulates. The α=1 prior is the same
        uninformative prior used in anchor compatibility testing.
        """
        # Add pseudocounts (uniform prior: 1 for each outcome)
        w_eff = self.wins + 1
        l_eff = self.losses + 1
        n_eff = self.total + 4
        
        mean = (w_eff - l_eff) / n_eff
        var = max(0, (w_eff + l_eff - n_eff * mean**2) / (n_eff - 1))
        
        return (mean - math.sqrt(var / n_eff) + 1) / 2

    @property
    def utility(self) -> float:
        """Expected value in [-1, 1]."""
        return (self.wins - self.losses) / self.total

    @property
    def certainty(self) -> float:
        """Confidence in the utility estimate."""
        n = self.total
        if n == 1:
            return 0.0
        mean = (self.wins - self.losses) / n
        var = max(0, (self.wins + self.losses - n * mean**2) / (n - 1))
        return max(0.0, 1.0 - math.sqrt(var / n))


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def _hash(board: np.ndarray) -> str:
    """Hash board state to 16-char hex string."""
    data = repr(board.tolist()).encode() if board.dtype == object else board.tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _log_dm_marginal(counts: Tuple[int, ...]) -> float:
    """
    Log marginal likelihood under Dirichlet-Multinomial with α=1 (uniform prior).

    P(counts | α=1) = Γ(K) / Γ(n+K) × ∏ Γ(c_i + 1) / Γ(1)
                    = (K-1)! / (n+K-1)! × ∏ c_i!

    This is the standard uninformative prior for multinomial data.
    """
    k = len(counts)
    n = sum(counts)
    if n == 0:
        return 0.0
    # With α=1: lgamma(1) = 0, so terms simplify
    result = math.lgamma(k) - math.lgamma(n + k)
    for c in counts:
        result += math.lgamma(c + 1)  # lgamma(c + 1) = log(c!)
    return result


def _log_bayes_factor(counts1: Tuple[int, ...], counts2: Tuple[int, ...]) -> float:
    """
    Log Bayes factor comparing:
      H₀: Both samples from same multinomial (pooled)
      H₁: Samples from independent multinomials

    Returns log(P(data|H₀) / P(data|H₁)).
    Positive = evidence for same distribution.
    """
    pooled = tuple(c1 + c2 for c1, c2 in zip(counts1, counts2))
    return _log_dm_marginal(pooled) - (
        _log_dm_marginal(counts1) + _log_dm_marginal(counts2)
    )


def _compatible(counts1: Tuple[int, ...], dist1: Tuple[float, ...],
                counts2: Tuple[int, ...], dist2: Tuple[float, ...]) -> bool:
    """
    Are these distributions statistically compatible?
    
    Uses Bayes factor with evidence threshold scaled by sample size imbalance.
    Requires BF > √(n_larger / n_smaller) to merge, ensuring sparse transitions
    need proportionally stronger evidence to join large anchors.
    """
    n1, n2 = sum(counts1), sum(counts2)
    if n1 == 0:
        return True  # Empty can merge with anything
    if n2 == 0:
        return False  # Can't merge with empty anchor
    
    log_bf = _log_bayes_factor(counts1, counts2)
    # Threshold: √(n_max/n_min) in log space (since log(√x) = 0.5*log(x))
    log_threshold = 0.5 * math.log(max(n1, n2) / min(n1, n2))
    return log_bf > log_threshold


def _similarity(counts1: Tuple[int, ...], counts2: Tuple[int, ...]) -> float:
    """
    Similarity score based on Bayes factor.
    Returns value in [0, 1] where higher = more similar.

    Maps log BF through sigmoid: 1 / (1 + exp(-logBF))
    """
    n1, n2 = sum(counts1), sum(counts2)
    if n1 == 0 or n2 == 0:
        return 0.0
    log_bf = _log_bayes_factor(counts1, counts2)
    # Clamp to avoid overflow in exp()
    log_bf = max(-50, min(50, log_bf))
    # Sigmoid maps (-∞, +∞) to (0, 1)
    return 1.0 / (1.0 + math.exp(-log_bf))


# Backward compatibility aliases
def _hellinger(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Hellinger distance (kept for backward compatibility)."""
    return math.sqrt(sum((math.sqrt(x) - math.sqrt(y)) ** 2 for x, y in zip(a, b)) / 2)


def _hellinger_similarity(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Hellinger similarity (kept for backward compatibility)."""
    return 1.0 - _hellinger(a, b)


_statistically_compatible = _compatible


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS transitions (
    from_hash TEXT, to_hash TEXT,
    wins INTEGER DEFAULT 0, ties INTEGER DEFAULT 0,
    neutral INTEGER DEFAULT 0, losses INTEGER DEFAULT 0,
    PRIMARY KEY (from_hash, to_hash)
);
CREATE INDEX IF NOT EXISTS idx_from ON transitions(from_hash);

CREATE TABLE IF NOT EXISTS anchors (
    anchor_id INTEGER PRIMARY KEY,
    repr_from TEXT, repr_to TEXT,
    wins INTEGER DEFAULT 0, ties INTEGER DEFAULT 0,
    neutral INTEGER DEFAULT 0, losses INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS transition_anchors (
    from_hash TEXT, to_hash TEXT, anchor_id INTEGER,
    PRIMARY KEY (from_hash, to_hash)
);
"""


# ---------------------------------------------------------------------------
# GameMemory
# ---------------------------------------------------------------------------


class GameMemory:
    """
    Game transition storage with automatic pattern discovery.

    Records (state, move, outcome) tuples and automatically clusters
    similar behavioral patterns into anchors. Uses pure statistical
    compatibility - no magic thresholds.
    """

    def __init__(self, db_path: str | Path = "memory.db", read_only: bool = False):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.read_only = read_only
        self._cache: Dict[str, Dict[str, Stats]] = {}
        self._anchors_dirty = True

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA cache_size=-65536")
        if not read_only:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    @classmethod
    def for_game(
        cls, game: GameBase, base_dir: str | Path = "data/memory", **kw
    ) -> "GameMemory":
        """Factory method to create GameMemory for a specific game type."""
        game_id = getattr(game, "game_id", lambda: type(game).__name__.lower())()
        return cls(Path(base_dir) / f"{game_id}.db", **kw)

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

        Args:
            game_class: Game class to instantiate for move application
            stacks: List of (moves, outcome) where moves is [(move, board, player), ...]
        """
        if self.read_only:
            raise RuntimeError("Cannot write to read-only GameMemory")

        transitions: Dict[Tuple[str, str], List[int]] = defaultdict(
            lambda: [0, 0, 0, 0]
        )
        for moves, outcome in stacks:
            outcome_idx = OUTCOME_INDEX.get(outcome, 2)
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
            cur.executemany(
                """
                INSERT INTO transitions VALUES (?,?,?,?,?,?)
                ON CONFLICT DO UPDATE SET
                    wins=wins+excluded.wins, ties=ties+excluded.ties,
                    neutral=neutral+excluded.neutral, losses=losses+excluded.losses
            """,
                [(f, t, *c) for (f, t), c in transitions.items()],
            )
            cur.execute("COMMIT")
            self._cache.clear()
            self._update_anchors(list(transitions.keys()), cur)
        except Exception:
            cur.execute("ROLLBACK")
            raise

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_state_stats(self, state_hash: str) -> Stats:
        """Aggregate stats for all transitions FROM a state."""
        row = self.conn.execute(
            """
            SELECT COALESCE(SUM(wins),0), COALESCE(SUM(ties),0),
                   COALESCE(SUM(neutral),0), COALESCE(SUM(losses),0)
            FROM transitions WHERE from_hash = ?
        """,
            (state_hash,),
        ).fetchone()
        return Stats(*row)

    def get_stats(self, from_hash: str = None, to_hash: str = None):
        """Get transition stats, or database summary if no args."""
        if from_hash is None and to_hash is None:
            # Return database summary (backwards compatibility)
            return self.get_info()
        if from_hash is None or to_hash is None:
            raise ValueError("Provide both from_hash and to_hash, or neither")
        row = self.conn.execute(
            "SELECT wins,ties,neutral,losses FROM transitions WHERE from_hash=? AND to_hash=?",
            (from_hash, to_hash),
        ).fetchone()
        return Stats(*row) if row else Stats()

    def _get_transition_stats(self, from_hash: str, to_hash: str) -> Stats:
        """Alias for get_stats with explicit arguments."""
        return self.get_stats(from_hash, to_hash)

    def get_effective_stats(self, from_hash: str, to_hash: str) -> Stats:
        """Get stats, using anchor when it provides more data."""
        direct = self.get_stats(from_hash, to_hash)
        anchor = self._get_anchor_stats(from_hash, to_hash)

        if anchor.total <= direct.total:
            return direct
        if direct.total > 0:
            # Check if direct is still compatible with anchor using Bayes factor
            if not _compatible(
                tuple(direct), direct.distribution, tuple(anchor), anchor.distribution
            ):
                return direct
        return anchor

    def get_all_moves_with_scores(
        self,
        game: "GameBase",
        valid_moves: np.ndarray,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Get direct scores for all valid moves (for exploration).
        
        Uses DIRECT stats only (not anchor-pooled) because exploration
        needs to know about THIS specific state, not pooled averages.
        
        Recorded moves get their direct score.
        Unrecorded moves get min(known scores) as prior, which creates:
          - Conservative exploitation (don't gamble on unknowns)
          - Aggressive prune exploration (unknowns might be bad!)
        Falls back to 0.5 if no moves have been recorded yet.
        """
        state = game.get_state()
        from_hash = _hash(state.board)
        
        # First pass: collect scores, mark unexplored
        moves_and_scores = []  # (move, score_or_None)
        known_scores = []
        
        for move in valid_moves:
            clone = game.deep_clone()
            try:
                clone.apply_move(move)
            except (ValueError, IndexError):
                continue
            
            to_hash = _hash(clone.get_state().board)
            direct = self.get_stats(from_hash, to_hash)
            
            if direct.total > 0:
                score = direct.score
                known_scores.append(score)
                moves_and_scores.append((move, score))
            else:
                moves_and_scores.append((move, None))  # Mark as unexplored
        
        # Determine prior for unexplored moves
        prior = min(known_scores) if known_scores else 0.5
        
        # Second pass: fill in unexplored with prior
        return [(move, score if score is not None else prior) 
                for move, score in moves_and_scores]

    def get_transitions_from(self, from_hash: str) -> Dict[str, Stats]:
        """All transitions from a state (cached)."""
        if from_hash not in self._cache:
            rows = self.conn.execute(
                "SELECT to_hash,wins,ties,neutral,losses FROM transitions WHERE from_hash=?",
                (from_hash,),
            ).fetchall()
            self._cache[from_hash] = {r[0]: Stats(*r[1:]) for r in rows}
        return self._cache[from_hash]

    def get_anchor(self, from_hash: str, to_hash: str) -> Optional[Tuple[str, str]]:
        """Get representative transition for this transition's anchor."""
        self._ensure_anchors()
        row = self.conn.execute(
            """
            SELECT a.repr_from, a.repr_to FROM transition_anchors ta
            JOIN anchors a ON ta.anchor_id = a.anchor_id
            WHERE ta.from_hash = ? AND ta.to_hash = ?
        """,
            (from_hash, to_hash),
        ).fetchone()
        return tuple(row) if row else None

    def get_anchor_stats(self, from_hash: str, to_hash: str) -> Stats:
        """Public API for anchor stats."""
        self._ensure_anchors()
        return self._get_anchor_stats(from_hash, to_hash)

    def _get_anchor_stats(self, from_hash: str, to_hash: str) -> Stats:
        """Get anchor stats for a transition."""
        row = self.conn.execute(
            """
            SELECT a.wins,a.ties,a.neutral,a.losses FROM transition_anchors ta
            JOIN anchors a ON ta.anchor_id=a.anchor_id
            WHERE ta.from_hash=? AND ta.to_hash=?
        """,
            (from_hash, to_hash),
        ).fetchone()
        return Stats(*row) if row else self.get_stats(from_hash, to_hash)

    def _ensure_anchors(self) -> None:
        """Ensure anchors exist. Lazy initialization on first read."""
        if not self._anchors_dirty:
            return
        if self.read_only:
            self._anchors_dirty = False
            return
        if self.conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0] == 0:
            if self.conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[0] > 0:
                self.rebuild_anchors()
        self._anchors_dirty = False

    # -------------------------------------------------------------------------
    # Anchor System
    # -------------------------------------------------------------------------

    def _update_anchors(self, keys: List[Tuple[str, str]], cur: sqlite3.Cursor) -> None:
        """Incremental anchor maintenance."""
        if not keys:
            return

        # Load transitions
        changed = {}
        for f, t in keys:
            row = self.conn.execute(
                "SELECT wins,ties,neutral,losses FROM transitions WHERE from_hash=? AND to_hash=?",
                (f, t),
            ).fetchone()
            if row and sum(row) > 0:
                total = sum(row)
                changed[(f, t)] = {
                    "counts": row,
                    "dist": tuple(x / total for x in row),
                }

        if not changed:
            return

        cur.execute("BEGIN IMMEDIATE")
        try:
            # Load anchors
            anchors = {}
            for aid, rf, rt, w, t, n, l in cur.execute(
                "SELECT anchor_id,repr_from,repr_to,wins,ties,neutral,losses FROM anchors"
            ):
                total = w + t + n + l
                if total > 0:
                    anchors[aid] = {
                        "counts": (w, t, n, l),
                        "dist": (w / total, t / total, n / total, l / total),
                        "repr": (rf, rt),
                    }

            max_id = max(anchors.keys(), default=-1)

            # Get existing assignments
            existing = {}
            if keys:
                rows = cur.execute(
                    f"SELECT from_hash,to_hash,anchor_id FROM transition_anchors "
                    f"WHERE (from_hash||'|'||to_hash) IN ({','.join('?' for _ in keys)})",
                    [f"{f}|{t}" for f, t in changed.keys()],
                ).fetchall()
                existing = {(r[0], r[1]): r[2] for r in rows}

            modified = set()
            moved_from = set()

            for key, data in changed.items():
                old_aid = existing.get(key)

                # Find best anchor
                best_aid = self._find_nearest(data["counts"], anchors)
                compatible = False
                if best_aid is not None:
                    a = anchors[best_aid]
                    compatible = _compatible(
                        data["counts"], data["dist"], a["counts"], a["dist"]
                    )

                if compatible:
                    new_aid = best_aid
                else:
                    max_id += 1
                    new_aid = max_id
                    cur.execute(
                        "INSERT INTO anchors VALUES (?,?,?,?,?,?,?)",
                        (new_aid, key[0], key[1], *data["counts"]),
                    )
                    anchors[new_aid] = {
                        "counts": data["counts"],
                        "dist": data["dist"],
                        "repr": key,
                    }

                # Update membership
                if old_aid is None:
                    cur.execute(
                        "INSERT INTO transition_anchors VALUES (?,?,?)",
                        (key[0], key[1], new_aid),
                    )
                elif old_aid != new_aid:
                    cur.execute(
                        "UPDATE transition_anchors SET anchor_id=? WHERE from_hash=? AND to_hash=?",
                        (new_aid, key[0], key[1]),
                    )
                    moved_from.add(old_aid)

                modified.add(new_aid)

            # Recalculate affected anchors
            for aid in moved_from | modified:
                self._recalc_anchor(aid, anchors, cur)

            # Consolidate
            self._consolidate(anchors, cur)

            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def _find_nearest(
        self, counts: Tuple[int, ...], anchors: Dict[int, dict]
    ) -> Optional[int]:
        """Find anchor with most similar distribution using Bayes factor."""
        if not anchors:
            return None
        best_aid, best_sim = None, -1
        for aid, a in anchors.items():
            sim = _similarity(counts, a["counts"])
            if sim > best_sim:
                best_aid, best_sim = aid, sim
        return best_aid

    def _recalc_anchor(
        self, aid: int, anchors: Dict[int, dict], cur: sqlite3.Cursor
    ) -> None:
        """Recalculate anchor stats from members. Delete if empty."""
        row = cur.execute(
            """
            SELECT COALESCE(SUM(t.wins),0), COALESCE(SUM(t.ties),0),
                   COALESCE(SUM(t.neutral),0), COALESCE(SUM(t.losses),0), COUNT(*)
            FROM transition_anchors ta
            JOIN transitions t ON ta.from_hash=t.from_hash AND ta.to_hash=t.to_hash
            WHERE ta.anchor_id=?
        """,
            (aid,),
        ).fetchone()

        w, t, n, l, count = row
        total = w + t + n + l

        if count == 0 or total == 0:
            cur.execute("DELETE FROM anchors WHERE anchor_id=?", (aid,))
            anchors.pop(aid, None)
            return

        cur.execute(
            "UPDATE anchors SET wins=?,ties=?,neutral=?,losses=? WHERE anchor_id=?",
            (w, t, n, l, aid),
        )
        anchors[aid] = {
            "counts": (w, t, n, l),
            "dist": (w / total, t / total, n / total, l / total),
            "repr": anchors.get(aid, {}).get("repr", (None, None)),
        }

    def _consolidate(self, anchors: Dict[int, dict], cur: sqlite3.Cursor) -> None:
        """Merge compatible anchors."""
        if len(anchors) < 2:
            return

        merged = True
        while merged:
            merged = False
            aids = list(anchors.keys())

            for i, aid1 in enumerate(aids):
                if aid1 not in anchors:
                    continue
                a1 = anchors[aid1]

                for aid2 in aids[i + 1 :]:
                    if aid2 not in anchors:
                        continue
                    a2 = anchors[aid2]

                    if _compatible(a1["counts"], a1["dist"], a2["counts"], a2["dist"]):
                        # Larger absorbs smaller
                        if sum(a1["counts"]) >= sum(a2["counts"]):
                            survivor, absorbed = aid1, aid2
                        else:
                            survivor, absorbed = aid2, aid1

                        cur.execute(
                            "UPDATE transition_anchors SET anchor_id=? WHERE anchor_id=?",
                            (survivor, absorbed),
                        )
                        cur.execute(
                            "DELETE FROM anchors WHERE anchor_id=?", (absorbed,)
                        )
                        del anchors[absorbed]
                        self._recalc_anchor(survivor, anchors, cur)
                        merged = True
                        break

                if merged:
                    break

    def rebuild_anchors(self) -> int:
        """Force full rebuild. Returns anchor count."""
        if self.read_only:
            raise RuntimeError("Read-only")

        rows = self.conn.execute(
            """
            SELECT from_hash,to_hash,wins,ties,neutral,losses
            FROM transitions WHERE wins+ties+neutral+losses > 0
        """
        ).fetchall()

        if not rows:
            return 0

        # Build transitions sorted by decisiveness
        transitions = []
        for f, t, w, ti, n, l in rows:
            total = w + ti + n + l
            dist = (w / total, ti / total, n / total, l / total)
            entropy = -sum(p * math.log(p + 1e-12) for p in dist)
            transitions.append(
                {
                    "key": (f, t),
                    "counts": (w, ti, n, l),
                    "dist": dist,
                    "entropy": entropy,
                }
            )
        transitions.sort(key=lambda x: x["entropy"])

        # Cluster
        anchors = []
        membership = {}
        for tr in transitions:
            best_idx = None
            best_sim = -1
            for i, a in enumerate(anchors):
                sim = _similarity(tr["counts"], a["counts"])
                if sim > best_sim:
                    best_idx, best_sim = i, sim

            compatible = False
            if best_idx is not None:
                a = anchors[best_idx]
                compatible = _compatible(
                    tr["counts"], tr["dist"], a["counts"], a["dist"]
                )

            if compatible:
                a = anchors[best_idx]
                a["counts"] = tuple(a["counts"][i] + tr["counts"][i] for i in range(4))
                total = sum(a["counts"])
                a["dist"] = tuple(c / total for c in a["counts"])
                membership[tr["key"]] = best_idx
            else:
                membership[tr["key"]] = len(anchors)
                anchors.append(
                    {
                        "counts": tr["counts"],
                        "dist": tr["dist"],
                        "repr": tr["key"],
                    }
                )

        # Persist
        cur = self.conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            cur.execute("DELETE FROM anchors")
            cur.execute("DELETE FROM transition_anchors")
            for i, a in enumerate(anchors):
                cur.execute(
                    "INSERT INTO anchors VALUES (?,?,?,?,?,?,?)",
                    (i, a["repr"][0], a["repr"][1], *a["counts"]),
                )
            cur.executemany(
                "INSERT INTO transition_anchors VALUES (?,?,?)",
                [(k[0], k[1], v) for k, v in membership.items()],
            )
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

        return len(anchors)

    def revalidate_anchors(self) -> Tuple[int, int]:
        """
        Fast revalidation of anchor memberships using sqrt threshold.
        O(n) - just checks each transition against its current anchor.
        Returns (kept, ejected) counts.
        """
        if self.read_only:
            raise RuntimeError("Read-only")

        cur = self.conn.cursor()
        cur.execute("BEGIN IMMEDIATE")

        try:
            # Get all transition-anchor pairs
            rows = cur.execute("""
                SELECT ta.from_hash, ta.to_hash, ta.anchor_id,
                       t.wins, t.ties, t.neutral, t.losses,
                       a.wins, a.ties, a.neutral, a.losses
                FROM transition_anchors ta
                JOIN transitions t ON ta.from_hash = t.from_hash AND ta.to_hash = t.to_hash
                JOIN anchors a ON ta.anchor_id = a.anchor_id
            """).fetchall()

            kept, ejected = 0, 0
            to_eject = []

            for from_h, to_h, aid, tw, tt, tn, tl, aw, at, an, al in rows:
                direct_counts = (tw, tt, tn, tl)
                anchor_counts = (aw, at, an, al)
                n_direct = sum(direct_counts)
                n_anchor = sum(anchor_counts)

                if n_direct == 0 or n_anchor == 0:
                    continue

                # Check sqrt threshold
                log_bf = _log_bayes_factor(direct_counts, anchor_counts)
                log_threshold = 0.5 * math.log(max(n_anchor, n_direct) / min(n_anchor, n_direct))

                if log_bf > log_threshold:
                    kept += 1
                else:
                    ejected += 1
                    to_eject.append((from_h, to_h, aid))

            # Eject incompatible transitions (they become their own anchors)
            max_id = cur.execute("SELECT MAX(anchor_id) FROM anchors").fetchone()[0] or 0

            for from_h, to_h, old_aid in to_eject:
                max_id += 1
                # Get transition stats
                row = cur.execute(
                    "SELECT wins,ties,neutral,losses FROM transitions WHERE from_hash=? AND to_hash=?",
                    (from_h, to_h)
                ).fetchone()
                # Create new single-member anchor
                cur.execute(
                    "INSERT INTO anchors VALUES (?,?,?,?,?,?,?)",
                    (max_id, from_h, to_h, *row)
                )
                # Update membership
                cur.execute(
                    "UPDATE transition_anchors SET anchor_id=? WHERE from_hash=? AND to_hash=?",
                    (max_id, from_h, to_h)
                )

            # Recalculate affected anchor stats
            affected = set(aid for _, _, aid in to_eject)
            for aid in affected:
                row = cur.execute("""
                    SELECT COALESCE(SUM(t.wins),0), COALESCE(SUM(t.ties),0),
                           COALESCE(SUM(t.neutral),0), COALESCE(SUM(t.losses),0), COUNT(*)
                    FROM transition_anchors ta
                    JOIN transitions t ON ta.from_hash=t.from_hash AND ta.to_hash=t.to_hash
                    WHERE ta.anchor_id=?
                """, (aid,)).fetchone()

                if row[4] == 0:  # No members left
                    cur.execute("DELETE FROM anchors WHERE anchor_id=?", (aid,))
                else:
                    cur.execute(
                        "UPDATE anchors SET wins=?,ties=?,neutral=?,losses=? WHERE anchor_id=?",
                        (row[0], row[1], row[2], row[3], aid)
                    )

            cur.execute("COMMIT")
            return kept, ejected

        except Exception:
            cur.execute("ROLLBACK")
            raise

    def consolidate_anchors(self) -> Tuple[int, int]:
        """
        Merge compatible anchors together. Call after revalidate_anchors().
        Returns (before_count, after_count).
        """
        if self.read_only:
            raise RuntimeError("Read-only")

        cur = self.conn.cursor()

        # Load all anchors
        rows = cur.execute(
            "SELECT anchor_id, repr_from, repr_to, wins, ties, neutral, losses FROM anchors"
        ).fetchall()

        before_count = len(rows)
        if before_count < 2:
            return before_count, before_count

        anchors = {}
        for aid, rf, rt, w, t, n, l in rows:
            total = w + t + n + l
            if total > 0:
                anchors[aid] = {
                    "counts": (w, t, n, l),
                    "dist": (w / total, t / total, n / total, l / total),
                    "repr": (rf, rt),
                }

        cur.execute("BEGIN IMMEDIATE")
        try:
            # Sort by total samples (largest first) for better merging
            sorted_aids = sorted(
                anchors.keys(), key=lambda a: sum(anchors[a]["counts"]), reverse=True
            )

            # Track which anchors have been absorbed
            absorbed = set()
            merges = []  # (absorbed_id, survivor_id)

            for i, aid1 in enumerate(sorted_aids):
                if aid1 in absorbed:
                    continue
                a1 = anchors[aid1]

                for aid2 in sorted_aids[i + 1 :]:
                    if aid2 in absorbed:
                        continue
                    a2 = anchors[aid2]

                    if _compatible(a1["counts"], a1["dist"], a2["counts"], a2["dist"]):
                        # aid1 (larger) absorbs aid2
                        absorbed.add(aid2)
                        merges.append((aid2, aid1))
                        # Update a1's counts for future comparisons
                        a1["counts"] = tuple(
                            a1["counts"][j] + a2["counts"][j] for j in range(4)
                        )
                        total = sum(a1["counts"])
                        a1["dist"] = tuple(c / total for c in a1["counts"])

            # Apply merges
            for absorbed_id, survivor_id in merges:
                cur.execute(
                    "UPDATE transition_anchors SET anchor_id=? WHERE anchor_id=?",
                    (survivor_id, absorbed_id),
                )
                cur.execute("DELETE FROM anchors WHERE anchor_id=?", (absorbed_id,))

            # Recalculate survivor stats
            survivors = set(sid for _, sid in merges)
            for aid in survivors:
                row = cur.execute(
                    """
                    SELECT COALESCE(SUM(t.wins),0), COALESCE(SUM(t.ties),0),
                           COALESCE(SUM(t.neutral),0), COALESCE(SUM(t.losses),0)
                    FROM transition_anchors ta
                    JOIN transitions t ON ta.from_hash=t.from_hash AND ta.to_hash=t.to_hash
                    WHERE ta.anchor_id=?
                """,
                    (aid,),
                ).fetchone()
                cur.execute(
                    "UPDATE anchors SET wins=?,ties=?,neutral=?,losses=? WHERE anchor_id=?",
                    (*row, aid),
                )

            cur.execute("COMMIT")

            after_count = cur.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
            return before_count, after_count

        except Exception:
            cur.execute("ROLLBACK")
            raise

    # -------------------------------------------------------------------------
    # Move Selection
    # -------------------------------------------------------------------------

    def get_best_move(
        self,
        game: GameBase,
        valid_moves: NDArray,
        deterministic: bool = False,
        debug: bool = False,
    ) -> Optional[NDArray]:
        """Get move with highest score."""
        result = self._select_move(
            game, valid_moves, best=True, deterministic=deterministic, debug=debug
        )
        return result[0] if result else None

    def get_worst_move(
        self,
        game: GameBase,
        valid_moves: NDArray,
        deterministic: bool = False,
        debug: bool = False,
    ) -> Optional[NDArray]:
        """Get move with lowest score."""
        result = self._select_move(
            game, valid_moves, best=False, deterministic=deterministic, debug=debug
        )
        return result[0] if result else None

    def get_best_move_with_score(
        self,
        game: GameBase,
        valid_moves: NDArray,
        deterministic: bool = False,
        debug: bool = False,
    ) -> Optional[Tuple[NDArray, float]]:
        """Get best move along with its score."""
        return self._select_move(
            game, valid_moves, best=True, deterministic=deterministic, debug=debug
        )

    def get_worst_move_with_score(
        self,
        game: GameBase,
        valid_moves: NDArray,
        deterministic: bool = False,
        debug: bool = False,
    ) -> Optional[Tuple[NDArray, float]]:
        """Get worst move along with its score."""
        return self._select_move(
            game, valid_moves, best=False, deterministic=deterministic, debug=debug
        )

    def get_all_possible_recorded_moves(
        self, game: GameBase, valid_moves: NDArray
    ) -> Optional[NDArray]:
        """
        Return all possible recorded moves from this game state
        """
        state = game.get_state()
        from_hash = _hash(state.board)
        transitions = self.get_transitions_from(from_hash)

        moves = []
        for move in valid_moves:
            clone = game.deep_clone()
            try:
                clone.apply_move(move)
            except (ValueError, IndexError):
                continue
            to_hash = _hash(clone.get_state().board)
            if to_hash in transitions:
                effective = self.get_effective_stats(from_hash, to_hash)
                if effective.total > 0:
                    moves.append(move)

        if not moves:
            return None

        return np.array(moves)

    def _select_move(
        self,
        game: GameBase,
        valid_moves: NDArray,
        best: bool,
        deterministic: bool = False,
        debug: bool = False,
    ) -> Optional[Tuple[NDArray, float]]:
        """
        Select move by score.

        Selection process:
          1. Pick the best ANCHOR (by pooled score)
          2. Within that anchor, select move weighted by direct score
             - deterministic=True: always pick highest direct score
             - deterministic=False: weighted random (higher direct = higher probability)
        """
        state = game.get_state()
        from_hash = _hash(state.board)
        state_stats = self.get_state_stats(from_hash)
        transitions = self.get_transitions_from(from_hash)

        candidates = []
        for move in valid_moves:
            clone = game.deep_clone()
            try:
                clone.apply_move(move)
            except (ValueError, IndexError):
                continue
            to_hash = _hash(clone.get_state().board)
            if to_hash in transitions:
                direct = self.get_stats(from_hash, to_hash)
                anchor = self._get_anchor_stats(from_hash, to_hash)
                anchor_id = self._get_anchor_id(from_hash, to_hash)
                effective = self.get_effective_stats(from_hash, to_hash)
                if effective.total > 0:
                    candidates.append(
                        {
                            "move": move,
                            "next_state": clone.get_state(),
                            "to_hash": to_hash,
                            "direct": direct,
                            "direct_score": direct.score,
                            "anchor": anchor,
                            "anchor_id": anchor_id,
                            "effective": effective,
                        }
                    )

        if not candidates:
            return None

        # Sort by effective (pooled) score to find best anchor
        candidates.sort(key=lambda x: x["effective"].score, reverse=best)

        # Get all moves in the top anchor (same effective score = same anchor)
        top_score = candidates[0]["effective"].score
        top_anchor_moves = [c for c in candidates if c["effective"].score == top_score]

        if deterministic:
            # Pick highest direct score within anchor
            top_anchor_moves.sort(key=lambda x: x["direct_score"], reverse=best)
            chosen = top_anchor_moves[0]
        else:
            # Weighted random by direct score within anchor
            chosen = self._weighted_choice(top_anchor_moves, best)

        if debug:
            self._render_debug(state, from_hash, candidates, chosen)

        return (chosen["move"], chosen["effective"].score)

    def _weighted_choice(self, moves: list, best: bool) -> dict:
        """
        Select from moves with probability proportional to direct score.

        For 'best' selection: higher direct score = higher probability
        For 'worst' selection: lower direct score = higher probability

        Uses softmax-style weighting to handle negative/zero scores gracefully.
        """
        if len(moves) == 1:
            return moves[0]

        scores = [m["direct_score"] for m in moves]

        # Normalize scores to [0, 1] range for weighting
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            # All equal - uniform random
            return random.choice(moves)

        # Normalize to [0, 1]
        normalized = [(s - min_s) / (max_s - min_s) for s in scores]

        # For 'worst' mode, invert the weights
        if not best:
            normalized = [1 - n for n in normalized]

        # Add small epsilon to ensure all moves have some chance
        epsilon = 0.1
        weights = [n + epsilon for n in normalized]

        # Weighted random selection
        total = sum(weights)
        r = random.random() * total
        cumulative = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return moves[i]

        return moves[-1]  # Fallback

    def _get_anchor_id(self, from_hash: str, to_hash: str) -> Optional[int]:
        """Get anchor ID for a transition."""
        row = self.conn.execute(
            "SELECT anchor_id FROM transition_anchors WHERE from_hash=? AND to_hash=?",
            (from_hash, to_hash),
        ).fetchone()
        return row[0] if row else None

    def _render_debug(
        self, state, from_hash: str, candidates: list, chosen: dict
    ) -> None:
        """Render debug visualization if available."""
        try:
            from omnicron.debug_viz import render_debug
        except ImportError:
            from debug_viz import render_debug

        def diff(before, after):
            return [
                (i, before[i], after[i])
                for i in np.ndindex(before.shape)
                if before[i] != after[i]
            ]

        debug_rows = []
        for c in candidates:
            eff = c["effective"]
            direct = c["direct"]
            anchor = c["anchor"]

            debug_rows.append(
                {
                    "diff": diff(state.board, c["next_state"].board),
                    "move": c["move"],
                    "is_selected": c["to_hash"] == chosen["to_hash"],
                    # Effective stats (what's actually used)
                    "score": eff.score,
                    "utility": eff.utility,
                    "certainty": eff.certainty,
                    "total": eff.total,
                    "pW": eff.wins / eff.total,
                    "pT": eff.ties / eff.total,
                    "pN": eff.neutral / eff.total,
                    "pL": eff.losses / eff.total,
                    # Direct transition stats
                    "direct_total": direct.total,
                    "direct_W": direct.wins,
                    "direct_T": direct.ties,
                    "direct_L": direct.losses,
                    # Anchor stats
                    "anchor_id": c["anchor_id"],
                    "anchor_total": anchor.total,
                    "anchor_W": anchor.wins,
                    "anchor_L": anchor.losses,
                    # Direct score (used for within-anchor weighting)
                    "direct_score": c["direct_score"],
                    # Which stats are being used
                    "using_anchor": eff.total == anchor.total
                    and anchor.total > direct.total,
                }
            )

        render_debug(state.board, debug_rows)

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def get_stats_summary(self) -> Dict[str, Any]:
        """Database statistics (alias for get_info)."""
        return self.get_info()

    def get_info(self) -> Dict[str, Any]:
        """Database statistics."""
        trans = self.conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
        anchors = self.conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
        samples = self.conn.execute(
            "SELECT COALESCE(SUM(wins+ties+neutral+losses),0) FROM transitions"
        ).fetchone()[0]
        states = self.conn.execute(
            "SELECT COUNT(DISTINCT from_hash) FROM transitions"
        ).fetchone()[0]
        return {
            "unique_states": states,
            "transitions": trans,
            "total_samples": samples,
            "anchors": anchors,
            "compression_ratio": trans / anchors if anchors else 1.0,
        }

    def get_anchor_details(self) -> List[Dict[str, Any]]:
        """Detailed anchor information for debugging."""
        rows = self.conn.execute(
            """
            SELECT a.anchor_id, a.repr_from, a.repr_to, a.wins, a.ties, a.neutral, a.losses,
                   COUNT(ta.from_hash)
            FROM anchors a LEFT JOIN transition_anchors ta ON a.anchor_id = ta.anchor_id
            GROUP BY a.anchor_id ORDER BY (a.wins+a.ties+a.neutral+a.losses) DESC
        """
        ).fetchall()

        return [
            {
                "anchor_id": r[0],
                "repr": (r[1], r[2]),
                "wins": r[3],
                "ties": r[4],
                "neutral": r[5],
                "losses": r[6],
                "total": (t := r[3] + r[4] + r[5] + r[6]),
                "members": r[7],
                "distribution": (r[3] / t, r[4] / t, r[5] / t, r[6] / t),
            }
            for r in rows
        ]

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
