"""
GameMemory: Pattern-based learning for game AI.

Uses Bayes factor clustering (parameter-free) to group similar moves for competitive play.

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

    NON-MARKOV (markov=False):
        - History matters beyond current state (e.g., language)
        - Different paths to same state may have different continuations
        - Score by specific transition (preserves context)
        - Anchors cluster similar transitions

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
import random
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


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                    CONFIGURABLE OUTCOME WEIGHTS                             ║
# ║                                                                             ║
# ║  Modify these to experiment with different scoring strategies:              ║
# ║                                                                             ║
# ║  Loss-averse:            W=1.0, T=0.5, L=-1.5  → ties are half-wins         ║
# ║  Symmetric (current):    W=1.0, T=0.0, L=-1.0  → pure win/loss              ║
# ║  Win-focused:            W=1.0, T=0.0, L=-0.5  → losses hurt less           ║
# ║  Tie-positive:           W=1.0, T=0.8, L=-1.0  → ties nearly as good        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

W_WEIGHT = 1.0    # Utility for a WIN
T_WEIGHT = 0.5   # Utility for a TIE
L_WEIGHT = -1.0   # Utility for a LOSS

# ═══════════════════════════════════════════════════════════════════════════════

# Derived constants (calculated from weights above - don't modify directly)
_SCORE_MIN = min(W_WEIGHT, T_WEIGHT, L_WEIGHT)
_SCORE_MAX = max(W_WEIGHT, T_WEIGHT, L_WEIGHT)
_SCORE_RANGE = _SCORE_MAX - _SCORE_MIN

OUTCOME_INDEX = {State.WIN: 0, State.TIE: 1, State.LOSS: 2}
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
        if t == 0:
            return (0.0, 0.0, 0.0)
        return (self.wins / t, self.ties / t, self.losses / t)

    @property
    def mean_score(self) -> float:
        """Mean utility normalized to [0, 1]. Uses Bayesian pseudocounts (α=1)."""
        w, t, l = self.wins + 1, self.ties + 1, self.losses + 1
        n = w + t + l
        mean = (w * W_WEIGHT + t * T_WEIGHT + l * L_WEIGHT) / n
        return (mean - _SCORE_MIN) / _SCORE_RANGE

    @property
    def std_error(self) -> float:
        """Standard error (UNCAPPED). Returns inf for insufficient data."""
        if self.total <= 1:
            return float('inf')
        
        w, t, l = self.wins + 1, self.ties + 1, self.losses + 1
        n = w + t + l
        mean = (w * W_WEIGHT + t * T_WEIGHT + l * L_WEIGHT) / n
        mean_sq = (w * W_WEIGHT**2 + t * T_WEIGHT**2 + l * L_WEIGHT**2) / n
        variance = mean_sq - mean**2
        
        raw_se = math.sqrt(max(0, variance / n))
        return raw_se / _SCORE_RANGE

    def sample_score(self, method: str = 'gaussian') -> float:
        """ (Unused) Thompson sampling from posterior. Returns [0, 1]."""
        if method == 'gaussian':
            mean, se = self.mean_score, self.std_error
            if se == float('inf'):
                return random.random()
            return max(0.0, min(1.0, random.gauss(mean, se)))
        
        elif method == 'dirichlet':
            alpha = [self.wins + 1, self.ties + 1, self.losses + 1]
            probs = np.random.dirichlet(alpha)
            utility = probs[0] * W_WEIGHT + probs[1] * T_WEIGHT + probs[2] * L_WEIGHT
            return (utility - _SCORE_MIN) / _SCORE_RANGE
        
        raise ValueError(f"Unknown method: {method}")

    @property
    def optimistic_score(self) -> float:
        """ 
        UCB score normalized to [0, 1].

        * Note: Unused. We use mean_score() + std_error() to separate concerns. 
            This achieves the same but with normalization.
        """
        w, t, l = self.wins + 1, self.ties + 1, self.losses + 1
        n = w + t + l
        mean = (w * W_WEIGHT + t * T_WEIGHT + l * L_WEIGHT) / n
        mean_sq = (w * W_WEIGHT**2 + t * T_WEIGHT**2 + l * L_WEIGHT**2) / n
        variance = mean_sq - mean**2
        se = math.sqrt(max(0, variance / n))
        ucb = mean + se
        return (ucb - _SCORE_MIN) / _SCORE_RANGE

    @property
    def utility(self) -> float:
        """Raw expected value (not normalized)."""
        if self.total == 0:
            return 0.0
        return (self.wins * W_WEIGHT + self.ties * T_WEIGHT + self.losses * L_WEIGHT) / self.total

    @property
    def certainty(self) -> float:
        """Confidence in estimate, capped to [0, 1]."""
        if self.total <= 1:
            return 0.0
        se = self.std_error
        return 0.0 if se == float('inf') else max(0.0, min(1.0, 1.0 - se))


# ---------------------------------------------------------------------------
# Bayes Factor Statistics
# ---------------------------------------------------------------------------


def _hash(board: np.ndarray) -> str:
    data = repr(board.tolist()).encode() if board.dtype == object else board.tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


@lru_cache(maxsize=4096)
def _log_dm_marginal(counts: Tuple[int, ...]) -> float:
    """Log marginal likelihood under Dirichlet-Multinomial with α=1."""
    k, n = len(counts), sum(counts)
    if n == 0:
        return 0.0
    result = math.lgamma(k) - math.lgamma(n + k)
    for c in counts:
        result += math.lgamma(c + 1)
    return result


def _log_bayes_factor(counts1: Tuple[int, ...], counts2: Tuple[int, ...]) -> float:
    """Log Bayes factor: P(same) / P(different). Positive = same distribution."""
    pooled = tuple(c1 + c2 for c1, c2 in zip(counts1, counts2))
    return _log_dm_marginal(pooled) - (_log_dm_marginal(counts1) + _log_dm_marginal(counts2))


def _compatible(counts1: Tuple[int, ...], counts2: Tuple[int, ...]) -> bool:
    """Are these statistically indistinguishable? (PARAMETER-FREE)"""
    n1, n2 = sum(counts1), sum(counts2)
    if n1 == 0:
        return True
    if n2 == 0:
        return False
    log_bf = _log_bayes_factor(counts1, counts2)
    log_threshold = 0.5 * math.log(max(n1, n2) / min(n1, n2))
    return log_bf > log_threshold


def compatible(stats1: Stats, stats2: Stats) -> bool:
    """Public API: Are two Stats objects statistically indistinguishable?"""
    return _compatible(tuple(stats1), tuple(stats2))


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
    wins INTEGER DEFAULT 0, ties INTEGER DEFAULT 0, losses INTEGER DEFAULT 0,
    PRIMARY KEY (from_hash, to_hash)
);
CREATE INDEX IF NOT EXISTS idx_from ON transitions(from_hash);
CREATE INDEX IF NOT EXISTS idx_to ON transitions(to_hash);

CREATE TABLE IF NOT EXISTS state_values (
    state_hash TEXT PRIMARY KEY,
    wins INTEGER DEFAULT 0, ties INTEGER DEFAULT 0, losses INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS anchors (
    anchor_id INTEGER PRIMARY KEY, repr_key TEXT,
    wins INTEGER DEFAULT 0, ties INTEGER DEFAULT 0, losses INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS scoring_anchors (
    scoring_key TEXT PRIMARY KEY, anchor_id INTEGER
);

CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT);
"""


# ---------------------------------------------------------------------------
# GameMemory
# ---------------------------------------------------------------------------


class GameMemory:
    """Game transition storage with Bayes factor clustering."""

    def __init__(self, db_path: str | Path = "memory.db", read_only: bool = False, markov: bool = False):
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
            self.conn.execute("INSERT OR REPLACE INTO metadata VALUES ('markov', ?)", ("true" if markov else "false",))
            self.conn.commit()
        else:
            row = self.conn.execute("SELECT value FROM metadata WHERE key='markov'").fetchone()
            if row:
                self.markov = row[0] == "true"

    def _scoring_key(self, from_hash: str, to_hash: str) -> str:
        return to_hash if self.markov else f"{from_hash}|{to_hash}"

    @classmethod
    def for_game(cls, game: GameBase, base_dir: str | Path = "data/memory", markov: bool = False, **kw) -> "GameMemory":
        game_id = getattr(game, "game_id", lambda: type(game).__name__.lower())()
        return cls(Path(base_dir) / f"{game_id}.db", markov=markov, **kw)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _iter_moves_with_hashes(self, game: GameBase, valid_moves: List[np.ndarray]) -> List[Tuple[np.ndarray, str]]:
        results = []
        for move in valid_moves:
            clone = game.deep_clone()
            try:
                clone.apply_move(move)
                results.append((move, _hash(clone.get_state().board)))
            except (ValueError, IndexError):
                continue
        return results

    def _get_move_data(self, from_hash: str, to_hash: str, transitions: Dict[str, Stats]) -> Optional[Dict]:
        if to_hash not in transitions or transitions[to_hash].total == 0:
            return None
        scoring_key = self._scoring_key(from_hash, to_hash)
        return {
            "direct": transitions[to_hash],
            "unit": self.get_state_stats(to_hash) if self.markov else self.get_stats(from_hash, to_hash),
            "anchor": self._get_anchor_stats(scoring_key),
            "anchor_id": self._get_anchor_id(scoring_key),
            "effective": self.get_effective_stats(scoring_key),
        }

    # -------------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------------

    def record_round(self, game_class: type, stacks: List[Tuple[List[Tuple[Any, np.ndarray, int]], State]]) -> int:
        if self.read_only:
            raise RuntimeError("Read-only")
        transitions: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0, 0])
        for moves, outcome in stacks:
            outcome_idx = OUTCOME_INDEX.get(outcome, -1)
            game = game_class()
            for move, board, player in moves:
                from_hash = _hash(board)
                game.set_state(GameState(board.copy(), player))
                game.apply_move(move)
                transitions[(_hash(board), _hash(game.get_state().board))][outcome_idx] += 1
        self._commit(transitions)
        return len(transitions)

    def _commit(self, transitions: Dict[Tuple[str, str], List[int]]) -> None:
        if not transitions:
            return
        cur = self.conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            cur.executemany(
                """INSERT INTO transitions VALUES (?,?,?,?,?) ON CONFLICT DO UPDATE SET
                   wins=wins+excluded.wins, ties=ties+excluded.ties, losses=losses+excluded.losses""",
                [(f, t, *c) for (f, t), c in transitions.items()],
            )
            if self.markov:
                state_updates = defaultdict(lambda: [0, 0, 0])
                for (_, to_hash), (w, t, l) in transitions.items():
                    state_updates[to_hash][0] += w
                    state_updates[to_hash][1] += t
                    state_updates[to_hash][2] += l
                cur.executemany(
                    """INSERT INTO state_values VALUES (?,?,?,?) ON CONFLICT DO UPDATE SET
                       wins=wins+excluded.wins, ties=ties+excluded.ties, losses=losses+excluded.losses""",
                    [(s, *c) for s, c in state_updates.items()],
                )
                scoring_keys = list(state_updates.keys())
            else:
                scoring_keys = [f"{f}|{t}" for f, t in transitions.keys()]
            cur.execute("COMMIT")
            self._cache.clear()
            self._update_anchors(scoring_keys, cur)
        except Exception:
            cur.execute("ROLLBACK")
            raise

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_stats(self, from_hash: str = None, to_hash: str = None) -> Stats:
        if from_hash is None and to_hash is None:
            return self.get_info()
        if from_hash is None or to_hash is None:
            raise ValueError("Provide both or neither")
        row = self.conn.execute("SELECT wins,ties,losses FROM transitions WHERE from_hash=? AND to_hash=?", (from_hash, to_hash)).fetchone()
        return Stats(*row) if row else Stats()

    def get_effective_stats(self, scoring_key: str) -> Stats:
        direct = self._get_unit_stats(scoring_key)
        anchor = self._get_anchor_stats(scoring_key)
        if anchor.total <= direct.total:
            return direct
        if direct.total > 0 and not _compatible(tuple(direct), tuple(anchor)):
            return direct
        return anchor

    def _get_unit_stats(self, scoring_key: str) -> Stats:
        if self.markov:
            return self.get_state_stats(scoring_key)
        parts = scoring_key.split("|", 1)
        return self.get_stats(parts[0], parts[1]) if len(parts) == 2 else Stats()

    def get_state_stats(self, state_hash: str) -> Stats:
        row = self.conn.execute("SELECT wins,ties,losses FROM state_values WHERE state_hash=?", (state_hash,)).fetchone()
        return Stats(*row) if row else Stats()

    def get_transitions_from(self, from_hash: str) -> Dict[str, Stats]:
        if from_hash not in self._cache:
            rows = self.conn.execute("SELECT to_hash,wins,ties,losses FROM transitions WHERE from_hash=?", (from_hash,)).fetchall()
            self._cache[from_hash] = {r[0]: Stats(*r[1:]) for r in rows}
        return self._cache[from_hash]

    def _get_anchor_stats(self, scoring_key: str) -> Stats:
        row = self.conn.execute(
            "SELECT a.wins,a.ties,a.losses FROM scoring_anchors sa JOIN anchors a ON sa.anchor_id=a.anchor_id WHERE sa.scoring_key=?",
            (scoring_key,),
        ).fetchone()
        return Stats(*row) if row else self._get_unit_stats(scoring_key)

    def _get_anchor_id(self, scoring_key: str) -> Optional[int]:
        row = self.conn.execute("SELECT anchor_id FROM scoring_anchors WHERE scoring_key=?", (scoring_key,)).fetchone()
        return row[0] if row else None

    # -------------------------------------------------------------------------
    # Move Evaluation
    # -------------------------------------------------------------------------

    def evaluate_moves_for_selection(self, game: GameBase, valid_moves: List[np.ndarray]) -> Dict[str, Any]:
        state = game.get_state()
        from_hash = _hash(state.board)
        transitions = self.get_transitions_from(from_hash)

        anchors_with_moves: Dict[int, List[Tuple[np.ndarray, Stats]]] = defaultdict(list)
        anchor_stats: Dict[int, Stats] = {}

        for move, to_hash in self._iter_moves_with_hashes(game, valid_moves):
            data = self._get_move_data(from_hash, to_hash, transitions)
            if data and data["effective"].total > 0:
                aid = data["anchor_id"]
                stats = data["effective"]
                anchors_with_moves[aid].append((move, stats))
                anchor_stats[aid] = stats
            else:
                anchors_with_moves[UNEXPLORED_ANCHOR_ID].append((move, Stats()))
                anchor_stats[UNEXPLORED_ANCHOR_ID] = Stats()

        return {"anchors_with_moves": dict(anchors_with_moves), "anchor_stats": anchor_stats}

    # -------------------------------------------------------------------------
    # Anchor System
    # -------------------------------------------------------------------------

    def _ensure_anchors(self) -> None:
        if not self._anchors_dirty:
            return
        if self.read_only:
            self._anchors_dirty = False
            return
        if self.conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0] == 0:
            has_data = self.conn.execute(
                "SELECT COUNT(*) FROM state_values" if self.markov else "SELECT COUNT(*) FROM transitions"
            ).fetchone()[0] > 0
            if has_data:
                self.rebuild_anchors()
        self._anchors_dirty = False

    def _update_anchors(self, scoring_keys: List[str], cur: sqlite3.Cursor) -> None:
        if not scoring_keys:
            return
        changed = {k: {"counts": tuple(s := self._get_unit_stats(k)), "dist": s.distribution} for k in scoring_keys if (s := self._get_unit_stats(k)).total > 0}
        if not changed:
            return

        cur.execute("BEGIN IMMEDIATE")
        try:
            anchors = {}
            for aid, repr_key, w, t, l in cur.execute("SELECT anchor_id,repr_key,wins,ties,losses FROM anchors"):
                if (total := w + t + l) > 0:
                    anchors[aid] = {"counts": (w, t, l), "dist": (w/total, t/total, l/total), "repr": repr_key}
            max_id = max(anchors.keys(), default=-1)

            existing = dict(cur.execute(f"SELECT scoring_key,anchor_id FROM scoring_anchors WHERE scoring_key IN ({','.join('?' for _ in changed)})", list(changed.keys())).fetchall())
            modified, moved_from = set(), set()

            for key, data in changed.items():
                old_aid = existing.get(key)
                if old_aid is not None and old_aid in anchors and _compatible(data["counts"], anchors[old_aid]["counts"]):
                    modified.add(old_aid)
                    continue
                if old_aid is not None:
                    moved_from.add(old_aid)

                best_aid = max(anchors.keys(), key=lambda a: _similarity(data["counts"], anchors[a]["counts"])) if anchors else None
                if best_aid and _compatible(data["counts"], anchors[best_aid]["counts"]):
                    new_aid = best_aid
                else:
                    max_id += 1
                    new_aid = max_id
                    cur.execute("INSERT INTO anchors VALUES (?,?,?,?,?)", (new_aid, key, *data["counts"]))
                    anchors[new_aid] = {"counts": data["counts"], "dist": data["dist"], "repr": key}

                if old_aid is None:
                    cur.execute("INSERT INTO scoring_anchors VALUES (?,?)", (key, new_aid))
                elif old_aid != new_aid:
                    cur.execute("UPDATE scoring_anchors SET anchor_id=? WHERE scoring_key=?", (new_aid, key))
                modified.add(new_aid)

            for aid in moved_from | modified:
                self._recalc_anchor(aid, anchors, cur)
            if moved_from | modified:
                self._consolidate(anchors, cur, moved_from | modified)
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def _recalc_anchor(self, aid: int, anchors: Dict[int, dict], cur: sqlite3.Cursor) -> None:
        if self.markov:
            row = cur.execute(
                "SELECT COALESCE(SUM(sv.wins),0),COALESCE(SUM(sv.ties),0),COALESCE(SUM(sv.losses),0),COUNT(*) FROM scoring_anchors sa JOIN state_values sv ON sa.scoring_key=sv.state_hash WHERE sa.anchor_id=?",
                (aid,),
            ).fetchone()
        else:
            w, t, l = 0, 0, 0
            for (key,) in cur.execute("SELECT scoring_key FROM scoring_anchors WHERE anchor_id=?", (aid,)):
                parts = key.split("|", 1)
                if len(parts) == 2 and (r := cur.execute("SELECT wins,ties,losses FROM transitions WHERE from_hash=? AND to_hash=?", parts).fetchone()):
                    w, t, l = w + r[0], t + r[1], l + r[2]
            row = (w, t, l, cur.execute("SELECT COUNT(*) FROM scoring_anchors WHERE anchor_id=?", (aid,)).fetchone()[0])

        w, t, l, count = row
        if count == 0 or (total := w + t + l) == 0:
            cur.execute("DELETE FROM anchors WHERE anchor_id=?", (aid,))
            anchors.pop(aid, None)
        else:
            cur.execute("UPDATE anchors SET wins=?,ties=?,losses=? WHERE anchor_id=?", (w, t, l, aid))
            anchors[aid] = {"counts": (w, t, l), "dist": (w/total, t/total, l/total), "repr": anchors.get(aid, {}).get("repr")}

    def _consolidate(self, anchors: Dict[int, dict], cur: sqlite3.Cursor, modified: set) -> None:
        to_check = set(modified)
        while True:
            merged = False
            for aid1 in list(to_check):
                if aid1 not in anchors:
                    to_check.discard(aid1)
                    continue
                for aid2 in list(anchors.keys()):
                    if aid2 != aid1 and aid2 in anchors and _compatible(anchors[aid1]["counts"], anchors[aid2]["counts"]):
                        survivor, absorbed = (aid1, aid2) if sum(anchors[aid1]["counts"]) >= sum(anchors[aid2]["counts"]) else (aid2, aid1)
                        cur.execute("UPDATE scoring_anchors SET anchor_id=? WHERE anchor_id=?", (survivor, absorbed))
                        cur.execute("DELETE FROM anchors WHERE anchor_id=?", (absorbed,))
                        del anchors[absorbed]
                        to_check.discard(absorbed)
                        self._recalc_anchor(survivor, anchors, cur)
                        to_check.add(survivor)
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                break

    def rebuild_anchors(self) -> int:
        if self.read_only:
            raise RuntimeError("Read-only")
        units = []
        if self.markov:
            for key, w, t, l in self.conn.execute("SELECT state_hash,wins,ties,losses FROM state_values WHERE wins+ties+losses>0"):
                total = w + t + l
                dist = (w/total, t/total, l/total)
                units.append({"key": key, "counts": (w, t, l), "dist": dist, "entropy": -sum(p * math.log(p + 1e-12) for p in dist)})
        else:
            for fh, th, w, t, l in self.conn.execute("SELECT from_hash,to_hash,wins,ties,losses FROM transitions WHERE wins+ties+losses>0"):
                total = w + t + l
                dist = (w/total, t/total, l/total)
                units.append({"key": f"{fh}|{th}", "counts": (w, t, l), "dist": dist, "entropy": -sum(p * math.log(p + 1e-12) for p in dist)})
        if not units:
            return 0

        units.sort(key=lambda x: x["entropy"])
        anchors, membership = [], {}
        for unit in units:
            best_idx = max(range(len(anchors)), key=lambda i: _similarity(unit["counts"], anchors[i]["counts"])) if anchors else None
            if best_idx is not None and _compatible(unit["counts"], anchors[best_idx]["counts"]):
                a = anchors[best_idx]
                a["counts"] = tuple(a["counts"][i] + unit["counts"][i] for i in range(3))
                total = sum(a["counts"])
                a["dist"] = tuple(c/total for c in a["counts"])
                membership[unit["key"]] = best_idx
            else:
                membership[unit["key"]] = len(anchors)
                anchors.append({"counts": unit["counts"], "dist": unit["dist"], "repr": unit["key"]})

        cur = self.conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            cur.execute("DELETE FROM anchors")
            cur.execute("DELETE FROM scoring_anchors")
            for i, a in enumerate(anchors):
                cur.execute("INSERT INTO anchors VALUES (?,?,?,?,?)", (i, a["repr"], *a["counts"]))
            cur.executemany("INSERT INTO scoring_anchors VALUES (?,?)", list(membership.items()))
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
        return len(anchors)

    # -------------------------------------------------------------------------
    # Debug & Info
    # -------------------------------------------------------------------------

    def debug_move_selection(self, game: GameBase, valid_moves: List[np.ndarray], chosen_move: Optional[np.ndarray] = None) -> None:
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
        chosen_to_hash = None
        if chosen_move is not None:
            clone = game.deep_clone()
            try:
                clone.apply_move(chosen_move)
                chosen_to_hash = _hash(clone.get_state().board)
            except:
                pass

        debug_rows = []
        for move, to_hash in self._iter_moves_with_hashes(game, valid_moves):
            clone = game.deep_clone()
            clone.apply_move(move)
            data = self._get_move_data(from_hash, to_hash, transitions)
            is_selected = to_hash == chosen_to_hash
            diff = [(i, state.board[i], clone.get_state().board[i]) for i in np.ndindex(state.board.shape) if state.board[i] != clone.get_state().board[i]]

            if data and data["effective"].total > 0:
                eff = data["effective"]
                debug_rows.append({
                    "diff": diff, "move": move, "is_selected": is_selected,
                    "score": eff.mean_score, "utility": eff.utility, "certainty": eff.certainty,
                    "std_error": eff.std_error, "total": eff.total,
                    "pW": eff.wins/eff.total, "pT": eff.ties/eff.total, "pL": eff.losses/eff.total,
                    "direct_total": data["direct"].total, "direct_W": data["direct"].wins,
                    "direct_T": data["direct"].ties, "direct_L": data["direct"].losses,
                    "anchor_id": data["anchor_id"], "anchor_total": data["anchor"].total,
                    "anchor_W": data["anchor"].wins, "anchor_L": data["anchor"].losses,
                    "direct_score": data["direct"].mean_score,
                    "using_anchor": eff.total == data["anchor"].total and data["anchor"].total > data["direct"].total,
                })
            else:
                debug_rows.append({
                    "diff": diff, "move": move, "is_selected": is_selected,
                    "score": 0.0, "utility": 0.0, "certainty": 0.0, "std_error": float('inf'),
                    "total": 0, "pW": 0.0, "pT": 0.0, "pL": 0.0,
                    "direct_total": 0, "direct_W": 0, "direct_T": 0, "direct_L": 0,
                    "anchor_id": None, "anchor_total": 0, "anchor_W": 0, "anchor_L": 0,
                    "direct_score": 0.0, "using_anchor": False, "unexplored": True,
                })
        render_debug(state.board, debug_rows) if debug_rows else print("No candidates")

    def get_info(self) -> Dict[str, Any]:
        trans = self.conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
        anchors = self.conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
        samples = self.conn.execute("SELECT COALESCE(SUM(wins+ties+losses),0) FROM transitions").fetchone()[0]
        from_states = self.conn.execute("SELECT COUNT(DISTINCT from_hash) FROM transitions").fetchone()[0]
        to_states = self.conn.execute("SELECT COUNT(DISTINCT to_hash) FROM transitions").fetchone()[0]
        return {"transitions": trans, "from_states": from_states, "to_states": to_states,
                "total_samples": samples, "anchors": anchors, "compression_ratio": to_states/anchors if anchors else 1.0}

    def get_anchor_details(self) -> List[Dict[str, Any]]:
        return [{
            "anchor_id": r[0], "repr_key": r[1], "wins": r[2], "ties": r[3], "losses": r[4],
            "total": (t := r[2]+r[3]+r[4]), "members": r[5],
            "distribution": (r[2]/t, r[3]/t, r[4]/t) if t else (0,0,0),
        } for r in self.conn.execute(
            "SELECT a.anchor_id,a.repr_key,a.wins,a.ties,a.losses,COUNT(sa.scoring_key) FROM anchors a LEFT JOIN scoring_anchors sa ON a.anchor_id=sa.anchor_id GROUP BY a.anchor_id ORDER BY a.wins+a.ties+a.losses DESC"
        )]

    def close(self) -> None:
        self._cache.clear()
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except:
            pass
        self.conn.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
