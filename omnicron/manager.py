"""
GameMemory: Pattern-based learning for game AI.

Architecture:
    transitions       - Raw (from_state, to_state) → outcome counts
    anchors           - Emergent behavioral patterns (pooled stats)
    transition_anchors - Maps each transition to its anchor

Anchor Merging Rule:
    Merge if BayesFactor > √(n_larger / n_smaller)
    
    Uses Dirichlet-Multinomial with uniform prior (α=1). The sqrt threshold
    scales with sample imbalance - sparse transitions need stronger evidence
    to join large anchors.
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
OUTCOME_INDEX = {State.WIN: 0, State.TIE: 1, State.NEUTRAL: 2, State.LOSS: 3}

# Neutral score for unexplored moves
NEUTRAL_SCORE = 0.5
UNEXPLORED_ANCHOR_ID = -999


# ---------------------------------------------------------------------------
# Stats Type
# ---------------------------------------------------------------------------


class Stats(NamedTuple):
    """Outcome counts with derived scoring properties."""
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
        return tuple(x / t for x in self) if t > 0 else (0.25, 0.25, 0.25, 0.25)

    @property
    def score(self) -> float:
        """
        Lower confidence bound on utility, normalized to [0, 1].
        
        Uses Bayesian approach with uniform Dirichlet prior (α=1).
        """
        w, l, n = self.wins + 1, self.losses + 1, self.total + 4
        mean = (w - l) / n
        var = max(0, (w + l - n * mean**2) / (n - 1))
        return (mean - math.sqrt(var / n) + 1) / 2

    @property
    def utility(self) -> float:
        """Expected value in [-1, 1]."""
        return (self.wins - self.losses) / self.total if self.total else 0.0

    @property
    def certainty(self) -> float:
        """Confidence in the utility estimate."""
        n = self.total
        if n <= 1:
            return 0.0
        mean = (self.wins - self.losses) / n
        var = max(0, (self.wins + self.losses - n * mean**2) / (n - 1))
        return max(0.0, 1.0 - math.sqrt(var / n))


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
    return _log_dm_marginal(pooled) - (_log_dm_marginal(counts1) + _log_dm_marginal(counts2))


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
    
    Records (state, move, outcome) and clusters similar patterns into anchors.
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
    def for_game(cls, game: GameBase, base_dir: str | Path = "data/memory", **kw) -> "GameMemory":
        """Create GameMemory for a specific game type."""
        game_id = getattr(game, "game_id", lambda: type(game).__name__.lower())()
        return cls(Path(base_dir) / f"{game_id}.db", **kw)

    # -------------------------------------------------------------------------
    # Move Evaluation Helpers
    # -------------------------------------------------------------------------

    def _iter_moves_with_hashes(
        self, game: GameBase, valid_moves: np.ndarray
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
        """Get full stats for a transition, or None if unexplored."""
        if to_hash not in transitions:
            return None
        
        direct = self.get_stats(from_hash, to_hash)
        if direct.total == 0:
            return None
            
        return {
            "direct": direct,
            "anchor": self._get_anchor_stats(from_hash, to_hash),
            "anchor_id": self._get_anchor_id(from_hash, to_hash),
            "effective": self.get_effective_stats(from_hash, to_hash),
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

        transitions: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0, 0, 0])
        
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
                """INSERT INTO transitions VALUES (?,?,?,?,?,?)
                   ON CONFLICT DO UPDATE SET
                   wins=wins+excluded.wins, ties=ties+excluded.ties,
                   neutral=neutral+excluded.neutral, losses=losses+excluded.losses""",
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

    def get_stats(self, from_hash: str = None, to_hash: str = None) -> Stats:
        """Get transition stats, or database summary if no args."""
        if from_hash is None and to_hash is None:
            return self.get_info()
        if from_hash is None or to_hash is None:
            raise ValueError("Provide both from_hash and to_hash, or neither")
        row = self.conn.execute(
            "SELECT wins,ties,neutral,losses FROM transitions WHERE from_hash=? AND to_hash=?",
            (from_hash, to_hash),
        ).fetchone()
        return Stats(*row) if row else Stats()

    def get_state_stats(self, state_hash: str) -> Stats:
        """Aggregate stats for all transitions FROM a state."""
        row = self.conn.execute(
            """SELECT COALESCE(SUM(wins),0), COALESCE(SUM(ties),0),
                      COALESCE(SUM(neutral),0), COALESCE(SUM(losses),0)
               FROM transitions WHERE from_hash = ?""",
            (state_hash,),
        ).fetchone()
        return Stats(*row)

    def get_effective_stats(self, from_hash: str, to_hash: str) -> Stats:
        """Get stats, preferring anchor when it provides more data and is compatible."""
        direct = self.get_stats(from_hash, to_hash)
        anchor = self._get_anchor_stats(from_hash, to_hash)

        if anchor.total <= direct.total:
            return direct
        if direct.total > 0 and not _compatible(tuple(direct), tuple(anchor)):
            return direct
        return anchor

    def get_transitions_from(self, from_hash: str) -> Dict[str, Stats]:
        """All transitions from a state (cached)."""
        if from_hash not in self._cache:
            rows = self.conn.execute(
                "SELECT to_hash,wins,ties,neutral,losses FROM transitions WHERE from_hash=?",
                (from_hash,),
            ).fetchall()
            self._cache[from_hash] = {r[0]: Stats(*r[1:]) for r in rows}
        return self._cache[from_hash]

    def _get_anchor_stats(self, from_hash: str, to_hash: str) -> Stats:
        """Get anchor stats for a transition."""
        row = self.conn.execute(
            """SELECT a.wins,a.ties,a.neutral,a.losses FROM transition_anchors ta
               JOIN anchors a ON ta.anchor_id=a.anchor_id
               WHERE ta.from_hash=? AND ta.to_hash=?""",
            (from_hash, to_hash),
        ).fetchone()
        return Stats(*row) if row else self.get_stats(from_hash, to_hash)

    def _get_anchor_id(self, from_hash: str, to_hash: str) -> Optional[int]:
        """Get anchor ID for a transition."""
        row = self.conn.execute(
            "SELECT anchor_id FROM transition_anchors WHERE from_hash=? AND to_hash=?",
            (from_hash, to_hash),
        ).fetchone()
        return row[0] if row else None

    # -------------------------------------------------------------------------
    # Move Evaluation (for selection)
    # -------------------------------------------------------------------------

    def get_all_moves_with_scores(
        self, game: GameBase, valid_moves: np.ndarray
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Get direct scores for all valid moves.
        Recorded moves get their score, unrecorded get 0.5 (neutral).
        """
        state = game.get_state()
        from_hash = _hash(state.board)
        
        results = []
        for move, to_hash in self._iter_moves_with_hashes(game, valid_moves):
            direct = self.get_stats(from_hash, to_hash)
            score = direct.score if direct.total > 0 else NEUTRAL_SCORE
            results.append((move, score))
        return results

    def evaluate_moves_for_selection(
        self, game: GameBase, valid_moves: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate all moves for wise_explorer selection.
        
        Returns:
            anchors_with_moves: {anchor_id: [(move, direct_score), ...]}
            anchor_scores: {anchor_id: pooled_score}
        """
        state = game.get_state()
        from_hash = _hash(state.board)
        transitions = self.get_transitions_from(from_hash)

        anchors_with_moves: Dict[int, List[Tuple[np.ndarray, float]]] = defaultdict(list)
        anchor_scores: Dict[int, float] = {}
        unexplored: List[np.ndarray] = []

        for move, to_hash in self._iter_moves_with_hashes(game, valid_moves):
            data = self._get_move_data(from_hash, to_hash, transitions)
            
            if data and data["effective"].total > 0:
                aid = data["anchor_id"]
                anchors_with_moves[aid].append((move, data["direct"].score))
                anchor_scores[aid] = data["effective"].score
            else:
                unexplored.append(move)

        # Add unexplored as synthetic anchor
        if unexplored:
            anchor_scores[UNEXPLORED_ANCHOR_ID] = NEUTRAL_SCORE
            for move in unexplored:
                anchors_with_moves[UNEXPLORED_ANCHOR_ID].append((move, NEUTRAL_SCORE))

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
            if self.conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[0] > 0:
                self.rebuild_anchors()
        self._anchors_dirty = False

    def _update_anchors(self, keys: List[Tuple[str, str]], cur: sqlite3.Cursor) -> None:
        """Incremental anchor maintenance after recording."""
        if not keys:
            return

        # Load changed transitions
        changed = {}
        for f, t in keys:
            row = self.conn.execute(
                "SELECT wins,ties,neutral,losses FROM transitions WHERE from_hash=? AND to_hash=?",
                (f, t),
            ).fetchone()
            if row and sum(row) > 0:
                changed[(f, t)] = {"counts": row, "dist": tuple(x / sum(row) for x in row)}

        if not changed:
            return

        cur.execute("BEGIN IMMEDIATE")
        try:
            # Load existing anchors
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

            # Get existing memberships
            existing = {}
            if keys:
                rows = cur.execute(
                    f"""SELECT from_hash,to_hash,anchor_id FROM transition_anchors
                        WHERE (from_hash||'|'||to_hash) IN ({','.join('?' for _ in keys)})""",
                    [f"{f}|{t}" for f, t in changed.keys()],
                ).fetchall()
                existing = {(r[0], r[1]): r[2] for r in rows}

            modified, moved_from = set(), set()

            for key, data in changed.items():
                old_aid = existing.get(key)
                
                # Step 1: Check if we should STAY in current anchor
                stay_in_current = False
                if old_aid is not None and old_aid in anchors:
                    if _compatible(data["counts"], anchors[old_aid]["counts"]):
                        stay_in_current = True
                    else:
                        # We've diverged - will leave current anchor
                        moved_from.add(old_aid)
                
                if stay_in_current:
                    # Happy where we are - just update stats
                    modified.add(old_aid)
                    continue
                
                # Step 2: Look for a new home (stricter threshold)
                best_aid = self._find_nearest(data["counts"], anchors)
                can_join = False
                if best_aid is not None:
                    can_join = _compatible(data["counts"], anchors[best_aid]["counts"])
                
                if can_join:
                    new_aid = best_aid
                else:
                    # Create our own anchor
                    max_id += 1
                    new_aid = max_id
                    cur.execute(
                        "INSERT INTO anchors VALUES (?,?,?,?,?,?,?)",
                        (new_aid, key[0], key[1], *data["counts"]),
                    )
                    anchors[new_aid] = {"counts": data["counts"], "dist": data["dist"], "repr": key}

                # Update membership
                if old_aid is None:
                    cur.execute("INSERT INTO transition_anchors VALUES (?,?,?)", (key[0], key[1], new_aid))
                elif old_aid != new_aid:
                    cur.execute(
                        "UPDATE transition_anchors SET anchor_id=? WHERE from_hash=? AND to_hash=?",
                        (new_aid, key[0], key[1]),
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

    def _find_nearest(self, counts: Tuple[int, ...], anchors: Dict[int, dict]) -> Optional[int]:
        """Find most similar anchor by Bayes factor."""
        if not anchors:
            return None
        return max(anchors.keys(), key=lambda aid: _similarity(counts, anchors[aid]["counts"]))

    def _recalc_anchor(self, aid: int, anchors: Dict[int, dict], cur: sqlite3.Cursor) -> None:
        """Recalculate anchor stats from members."""
        row = cur.execute(
            """SELECT COALESCE(SUM(t.wins),0), COALESCE(SUM(t.ties),0),
                      COALESCE(SUM(t.neutral),0), COALESCE(SUM(t.losses),0), COUNT(*)
               FROM transition_anchors ta
               JOIN transitions t ON ta.from_hash=t.from_hash AND ta.to_hash=t.to_hash
               WHERE ta.anchor_id=?""",
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
                    
                    # _compatible handles size ratio automatically
                    if _compatible(a1["counts"], a2["counts"]):
                        # Larger absorbs smaller
                        n1, n2 = sum(a1["counts"]), sum(a2["counts"])
                        if n1 >= n2:
                            survivor, absorbed = aid1, aid2
                        else:
                            survivor, absorbed = aid2, aid1

                        cur.execute(
                            "UPDATE transition_anchors SET anchor_id=? WHERE anchor_id=?",
                            (survivor, absorbed),
                        )
                        cur.execute("DELETE FROM anchors WHERE anchor_id=?", (absorbed,))
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

        rows = self.conn.execute(
            """SELECT from_hash,to_hash,wins,ties,neutral,losses
               FROM transitions WHERE wins+ties+neutral+losses > 0"""
        ).fetchall()

        if not rows:
            return 0

        # Sort by entropy (most decisive first)
        transitions = []
        for f, t, w, ti, n, l in rows:
            total = w + ti + n + l
            dist = (w / total, ti / total, n / total, l / total)
            entropy = -sum(p * math.log(p + 1e-12) for p in dist)
            transitions.append({"key": (f, t), "counts": (w, ti, n, l), "dist": dist, "entropy": entropy})
        transitions.sort(key=lambda x: x["entropy"])

        # Cluster
        anchors, membership = [], {}
        for tr in transitions:
            best_idx = None
            if anchors:
                best_idx = max(range(len(anchors)), key=lambda i: _similarity(tr["counts"], anchors[i]["counts"]))

            compatible = best_idx is not None and _compatible(tr["counts"], anchors[best_idx]["counts"])

            if compatible:
                a = anchors[best_idx]
                a["counts"] = tuple(a["counts"][i] + tr["counts"][i] for i in range(4))
                total = sum(a["counts"])
                a["dist"] = tuple(c / total for c in a["counts"])
                membership[tr["key"]] = best_idx
            else:
                membership[tr["key"]] = len(anchors)
                anchors.append({"counts": tr["counts"], "dist": tr["dist"], "repr": tr["key"]})

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
            diff = [(i, state.board[i], next_state.board[i]) 
                    for i in np.ndindex(state.board.shape) 
                    if state.board[i] != next_state.board[i]]

            if data and data["effective"].total > 0:
                eff, direct, anchor = data["effective"], data["direct"], data["anchor"]
                debug_rows.append({
                    "diff": diff, "move": move, "is_selected": is_selected,
                    "score": eff.score, "utility": eff.utility, "certainty": eff.certainty,
                    "total": eff.total,
                    "pW": eff.wins / eff.total, "pT": eff.ties / eff.total,
                    "pN": eff.neutral / eff.total, "pL": eff.losses / eff.total,
                    "direct_total": direct.total, "direct_W": direct.wins,
                    "direct_T": direct.ties, "direct_L": direct.losses,
                    "anchor_id": data["anchor_id"], "anchor_total": anchor.total,
                    "anchor_W": anchor.wins, "anchor_L": anchor.losses,
                    "direct_score": direct.score,
                    "using_anchor": eff.total == anchor.total and anchor.total > direct.total,
                })
            else:
                debug_rows.append({
                    "diff": diff, "move": move, "is_selected": is_selected,
                    "score": 0.5, "utility": 0.0, "certainty": 0.0, "total": 0,
                    "pW": 0.0, "pT": 0.0, "pN": 0.0, "pL": 0.0,
                    "direct_total": 0, "direct_W": 0, "direct_T": 0, "direct_L": 0,
                    "anchor_id": None, "anchor_total": 0, "anchor_W": 0, "anchor_L": 0,
                    "direct_score": 0.5, "using_anchor": False, "unexplored": True,
                })

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
            "SELECT COALESCE(SUM(wins+ties+neutral+losses),0) FROM transitions"
        ).fetchone()[0]
        states = self.conn.execute("SELECT COUNT(DISTINCT from_hash) FROM transitions").fetchone()[0]
        return {
            "unique_states": states,
            "transitions": trans,
            "total_samples": samples,
            "anchors": anchors,
            "compression_ratio": trans / anchors if anchors else 1.0,
        }

    def get_stats_summary(self) -> Dict[str, Any]:
        """Alias for get_info."""
        return self.get_info()

    def get_anchor_details(self) -> List[Dict[str, Any]]:
        """Detailed anchor information."""
        rows = self.conn.execute(
            """SELECT a.anchor_id, a.repr_from, a.repr_to, a.wins, a.ties, a.neutral, a.losses,
                      COUNT(ta.from_hash)
               FROM anchors a LEFT JOIN transition_anchors ta ON a.anchor_id = ta.anchor_id
               GROUP BY a.anchor_id ORDER BY (a.wins+a.ties+a.neutral+a.losses) DESC"""
        ).fetchall()

        return [
            {
                "anchor_id": r[0], "repr": (r[1], r[2]),
                "wins": r[3], "ties": r[4], "neutral": r[5], "losses": r[6],
                "total": (t := r[3] + r[4] + r[5] + r[6]), "members": r[7],
                "distribution": (r[3] / t, r[4] / t, r[5] / t, r[6] / t) if t else (0, 0, 0, 0),
            }
            for r in rows
        ]

    def get_anchor(self, from_hash: str, to_hash: str) -> Optional[Tuple[str, str]]:
        """Get representative transition for this transition's anchor."""
        self._ensure_anchors()
        row = self.conn.execute(
            """SELECT a.repr_from, a.repr_to FROM transition_anchors ta
               JOIN anchors a ON ta.anchor_id = a.anchor_id
               WHERE ta.from_hash = ? AND ta.to_hash = ?""",
            (from_hash, to_hash),
        ).fetchone()
        return tuple(row) if row else None

    def get_anchor_stats(self, from_hash: str, to_hash: str) -> Stats:
        """Public API for anchor stats."""
        self._ensure_anchors()
        return self._get_anchor_stats(from_hash, to_hash)

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
