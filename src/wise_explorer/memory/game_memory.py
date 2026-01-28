"""
GameMemory: Pattern-based learning for game AI.

Uses Bayes factor clustering to group similar moves for competitive play.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from wise_explorer.core.types import Stats, OUTCOME_INDEX, UNEXPLORED_ANCHOR_ID
from wise_explorer.core.hashing import hash_board
from wise_explorer.core.bayes import compatible
from wise_explorer.memory.schema import SCHEMA
from wise_explorer.memory.anchor_manager import AnchorManager

if TYPE_CHECKING:
    from wise_explorer.agent.agent import State
    from wise_explorer.games.game_base import GameBase

logger = logging.getLogger(__name__)


class GameMemory:
    """Game transition storage with Bayes factor clustering."""

    def __init__(self, db_path: str | Path = "memory.db", read_only: bool = False, markov: bool = False):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.read_only = read_only
        self.markov = markov
        
        # Caches
        self._transition_cache: Dict[str, Dict[str, Tuple[Optional[int], Stats]]] = {}
        self._anchor_stats_cache: Dict[int, Stats] = {}
        self._anchor_id_cache: Dict[str, Optional[int]] = {}

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA cache_size=-65536")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        
        if read_only:
            row = self.conn.execute("SELECT value FROM metadata WHERE key='markov'").fetchone()
            if row:
                self.markov = row[0] == "true"
        else:
            self.conn.executescript(SCHEMA)
            self.conn.execute("INSERT OR REPLACE INTO metadata VALUES ('markov', ?)", 
                            ("true" if markov else "false",))
            self.conn.commit()

        self._anchors = AnchorManager(self)

    # -------------------------------------------------------------------------
    # Basic Queries
    # -------------------------------------------------------------------------

    def _scoring_key(self, from_hash: str, to_hash: str) -> str:
        return to_hash if self.markov else f"{from_hash}|{to_hash}"

    def get_stats(self, from_hash: str, to_hash: str) -> Stats:
        """Get statistics for a specific transition."""
        row = self.conn.execute(
            "SELECT wins, ties, losses FROM transitions WHERE from_hash=? AND to_hash=?",
            (from_hash, to_hash)
        ).fetchone()
        return Stats(*row) if row else Stats()

    def get_state_stats(self, state_hash: str) -> Stats:
        """Get aggregated statistics for a state (Markov mode)."""
        row = self.conn.execute(
            "SELECT wins, ties, losses FROM state_values WHERE state_hash=?",
            (state_hash,)
        ).fetchone()
        return Stats(*row) if row else Stats()

    def get_unit_stats(self, scoring_key: str) -> Stats:
        """Get stats for a scoring unit (state or transition)."""
        if self.markov:
            return self.get_state_stats(scoring_key)
        parts = scoring_key.split("|", 1)
        return self.get_stats(parts[0], parts[1]) if len(parts) == 2 else Stats()

    def get_direct_stats(self, from_hash: str, to_hash: str) -> Stats:
        """Get direct stats for a move, respecting current mode."""
        if self.markov:
            return self.get_state_stats(to_hash)
        return self.get_stats(from_hash, to_hash)

    def get_transitions_from(self, from_hash: str) -> Dict[str, Stats]:
        """Get all transitions from a given state."""
        cached = self._get_transitions_with_anchors(from_hash)
        return {to_hash: stats for to_hash, (_, stats) in cached.items()}

    def _get_transitions_with_anchors(self, from_hash: str) -> Dict[str, Tuple[Optional[int], Stats]]:
        """
        Get all transitions from a position with anchor IDs.
        
        Note: This returns per-transition data from the transitions table.
        In Markov mode, use get_state_stats() and get_anchor_id() instead
        for proper state-based statistics.
        """
        if from_hash in self._transition_cache:
            return self._transition_cache[from_hash]
        
        rows = self.conn.execute(
            "SELECT to_hash, anchor_id, wins, ties, losses FROM transitions WHERE from_hash=?",
            (from_hash,)
        ).fetchall()
        
        result = {r[0]: (r[1], Stats(r[2], r[3], r[4])) for r in rows}
        self._transition_cache[from_hash] = result
        return result

    def get_info(self) -> Dict[str, Any]:
        """Get summary statistics about the memory database."""
        trans = self.conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
        anchors = self.conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
        samples = self.conn.execute("SELECT COALESCE(SUM(wins+ties+losses), 0) FROM transitions").fetchone()[0]
        from_states = self.conn.execute("SELECT COUNT(DISTINCT from_hash) FROM transitions").fetchone()[0]
        to_states = self.conn.execute("SELECT COUNT(DISTINCT to_hash) FROM transitions").fetchone()[0]
        
        info = {
            "transitions": trans,
            "from_states": from_states,
            "to_states": to_states,
            "total_samples": samples,
            "anchors": anchors,
            "compression_ratio": to_states / anchors if anchors else 1.0,
            "markov_mode": self.markov,
        }
        
        if self.markov:
            state_count = self.conn.execute("SELECT COUNT(*) FROM state_values").fetchone()[0]
            state_samples = self.conn.execute(
                "SELECT COALESCE(SUM(wins+ties+losses), 0) FROM state_values"
            ).fetchone()[0]
            info["unique_states"] = state_count
            info["state_samples"] = state_samples
        
        return info

    # -------------------------------------------------------------------------
    # Anchor-Aware Queries
    # -------------------------------------------------------------------------

    def get_anchor_stats(self, scoring_key: str) -> Stats:
        """Get pooled statistics from the anchor cluster."""
        anchor_id = self.get_anchor_id(scoring_key)
        return self.get_anchor_stats_by_id(anchor_id) if anchor_id is not None else self.get_unit_stats(scoring_key)

    def get_anchor_stats_by_id(self, anchor_id: int) -> Stats:
        """Get anchor stats by ID (cached)."""
        if anchor_id in self._anchor_stats_cache:
            return self._anchor_stats_cache[anchor_id]
        
        row = self.conn.execute(
            "SELECT wins, ties, losses FROM anchors WHERE anchor_id = ?",
            (anchor_id,)
        ).fetchone()
        stats = Stats(*row) if row else Stats()
        self._anchor_stats_cache[anchor_id] = stats
        return stats

    def get_anchor_id(self, scoring_key: str) -> Optional[int]:
        """Get the anchor ID for a scoring key (cached)."""
        if scoring_key in self._anchor_id_cache:
            return self._anchor_id_cache[scoring_key]
        
        row = self.conn.execute(
            "SELECT anchor_id FROM scoring_anchors WHERE scoring_key=?",
            (scoring_key,)
        ).fetchone()
        aid = row[0] if row else None
        self._anchor_id_cache[scoring_key] = aid
        return aid

    def get_effective_stats(self, scoring_key: str) -> Stats:
        """Get best available stats (anchor if compatible, else direct)."""
        direct = self.get_unit_stats(scoring_key)
        anchor = self.get_anchor_stats(scoring_key)
        
        if anchor.total <= direct.total:
            return direct
        if direct.total > 0 and not compatible(tuple(direct), tuple(anchor)):
            return direct
        return anchor

    def get_anchor_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about all anchors."""
        return self._anchors.get_details()

    def rebuild_anchors(self) -> int:
        """Full rebuild of anchor clustering."""
        return self._anchors.rebuild()

    def consolidate_anchors(self) -> int:
        """Merge compatible anchors. Call after training."""
        return self._anchors.consolidate()

    # -------------------------------------------------------------------------
    # Move Evaluation
    # -------------------------------------------------------------------------

    def evaluate_moves(self, game: "GameBase", valid_moves: List[np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate all valid moves and group by anchor.
        
        Returns dict with anchors_with_moves and anchor_stats.
        """
        from_hash = hash_board(game.get_state().board)

        anchors_with_moves: Dict[int, List[Tuple[np.ndarray, Stats]]] = defaultdict(list)
        anchor_stats: Dict[int, Stats] = {}

        for move, to_hash in self._compute_move_hashes(game, valid_moves):
            scoring_key = self._scoring_key(from_hash, to_hash)
            
            # Get stats appropriate for mode
            direct_stats = self.get_direct_stats(from_hash, to_hash)
            
            if direct_stats.total > 0:
                anchor_id = self.get_anchor_id(scoring_key)
                aid = anchor_id if anchor_id is not None else UNEXPLORED_ANCHOR_ID
                anchors_with_moves[aid].append((move, direct_stats))
                
                if aid not in anchor_stats:
                    anchor_stats[aid] = self.get_anchor_stats_by_id(aid) if aid != UNEXPLORED_ANCHOR_ID else Stats()
            else:
                anchors_with_moves[UNEXPLORED_ANCHOR_ID].append((move, Stats()))
                anchor_stats[UNEXPLORED_ANCHOR_ID] = Stats()

        return {"anchors_with_moves": dict(anchors_with_moves), "anchor_stats": anchor_stats}

    def _compute_move_hashes(self, game: "GameBase", valid_moves: List[np.ndarray]) -> List[Tuple[np.ndarray, str]]:
        """Generate (move, destination_hash) pairs for all valid moves."""
        results = []
        for move in valid_moves:
            clone = game.deep_clone()
            try:
                clone.apply_move(move)
                results.append((move, hash_board(clone.get_state().board)))
            except (ValueError, IndexError):
                continue
        return results

    # -------------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------------

    def record_round(self, game_class: type, stacks: List[Tuple[List[Tuple[Any, np.ndarray, int]], "State"]]) -> int:
        """Record outcomes from a batch of games."""
        if self.read_only:
            raise RuntimeError("Cannot record in read-only mode")
        
        from wise_explorer.games.game_state import GameState
        
        transitions: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0, 0])
        
        for moves, outcome in stacks:
            outcome_idx = OUTCOME_INDEX.get(outcome, -1)
            if outcome_idx < 0:
                continue
                
            game = game_class()
            for move, board, player in moves:
                from_hash = hash_board(board)
                game.set_state(GameState(board.copy(), player))
                game.apply_move(move)
                to_hash = hash_board(game.get_state().board)
                transitions[(from_hash, to_hash)][outcome_idx] += 1
        
        self._commit(transitions)
        return len(transitions)

    def _commit(self, transitions: Dict[Tuple[str, str], List[int]]) -> None:
        """Write transitions to database with incremental anchor updates."""
        if not transitions:
            return
        
        cur = self.conn.cursor()
        
        # Build insert data and track deltas
        deltas: Dict[str, Tuple[int, int, int]] = {}
        insert_data = []
        for (from_hash, to_hash), counts in transitions.items():
            scoring_key = self._scoring_key(from_hash, to_hash)
            insert_data.append((from_hash, to_hash, scoring_key, *counts))
            deltas[scoring_key] = tuple(counts)
        
        # Batch upsert transitions
        cur.executemany(
            """INSERT INTO transitions (from_hash, to_hash, scoring_key, wins, ties, losses)
               VALUES (?,?,?,?,?,?) 
               ON CONFLICT(from_hash, to_hash) DO UPDATE SET
               wins = wins + excluded.wins,
               ties = ties + excluded.ties,
               losses = losses + excluded.losses""",
            insert_data,
        )
        
        # Handle Markov mode
        if self.markov:
            state_updates: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
            for (_, to_hash), counts in transitions.items():
                for i in range(3):
                    state_updates[to_hash][i] += counts[i]
            
            cur.executemany(
                """INSERT INTO state_values (state_hash, wins, ties, losses)
                   VALUES (?,?,?,?) 
                   ON CONFLICT DO UPDATE SET
                   wins = wins + excluded.wins,
                   ties = ties + excluded.ties,
                   losses = losses + excluded.losses""",
                [(s, *c) for s, c in state_updates.items()],
            )
            scoring_keys = list(state_updates.keys())
            deltas = {k: tuple(v) for k, v in state_updates.items()}
        else:
            scoring_keys = list(deltas.keys())
        
        # Incremental anchor update
        self._anchors.update(scoring_keys, deltas, cur)
        
        self.conn.commit()
        self._clear_caches()

    def _clear_caches(self) -> None:
        """Clear all caches."""
        self._transition_cache.clear()
        self._anchor_stats_cache.clear()
        self._anchor_id_cache.clear()

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------

    def debug_move_selection(self, game: "GameBase", valid_moves: List[np.ndarray], chosen_move: Optional[np.ndarray] = None) -> None:
        """Display debug visualization for move selection."""
        try:
            from wise_explorer.debug.viz import render_debug
        except ImportError:
            print("debug.viz not available")
            return

        state = game.get_state()
        from_hash = hash_board(state.board)
        
        # Get chosen move's destination hash
        chosen_to_hash = None
        if chosen_move is not None:
            clone = game.deep_clone()
            try:
                clone.apply_move(chosen_move)
                chosen_to_hash = hash_board(clone.get_state().board)
            except:
                pass

        debug_rows = []
        for move, to_hash in self._compute_move_hashes(game, valid_moves):
            clone = game.deep_clone()
            clone.apply_move(move)
            
            diff = [
                (i, state.board[i], clone.get_state().board[i])
                for i in np.ndindex(state.board.shape)
                if state.board[i] != clone.get_state().board[i]
            ]

            scoring_key = self._scoring_key(from_hash, to_hash)
            
            # Get stats appropriate for mode
            direct = self.get_direct_stats(from_hash, to_hash)
            
            if direct.total > 0:
                anchor_id = self.get_anchor_id(scoring_key)
                anchor = self.get_anchor_stats(scoring_key)
                debug_rows.append({
                    "diff": diff,
                    "move": move,
                    "is_selected": to_hash == chosen_to_hash,
                    "direct_total": direct.total,
                    "direct_W": direct.wins, "direct_T": direct.ties, "direct_L": direct.losses,
                    "direct_score": direct.mean_score,
                    "anchor_id": anchor_id,
                    "anchor_total": anchor.total,
                    "anchor_W": anchor.wins, "anchor_T": anchor.ties, "anchor_L": anchor.losses,
                    "anchor_score": anchor.mean_score,
                })
            else:
                debug_rows.append({
                    "diff": diff,
                    "move": move,
                    "is_selected": to_hash == chosen_to_hash,
                    "direct_total": 0,
                    "direct_W": 0, "direct_T": 0, "direct_L": 0,
                    "direct_score": 0.0,
                    "anchor_id": None,
                    "anchor_total": 0,
                    "anchor_W": 0, "anchor_T": 0, "anchor_L": 0,
                    "anchor_score": 0.0,
                    "unexplored": True,
                })
        
        if debug_rows:
            render_debug(state.board, debug_rows)
        else:
            print("No candidates")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    @classmethod
    def for_game(cls, game: "GameBase", base_dir: str | Path = "data/memory", markov: bool = False, **kw) -> "GameMemory":
        """Create a GameMemory instance for a specific game."""
        game_id = getattr(game, "game_id", lambda: type(game).__name__.lower())()
        return cls(Path(base_dir) / f"{game_id}.db", markov=markov, **kw)

    def close(self) -> None:
        """Close the database connection."""
        self._clear_caches()
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except:
            pass
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()