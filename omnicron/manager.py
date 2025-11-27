# manager.py
import os
import json
import logging
import sqlite3
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from functools import lru_cache
from peewee import (
    Model, IntegerField, BlobField, TextField, SqliteDatabase, CompositeKey
)
from omnicron.serializers import serialize_array, deserialize_array, hash_array
from agent.agent import State
from games.game_state import GameState
from omnicron.debug_viz import render_debug

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
db = SqliteDatabase(None, pragmas={
    'journal_mode': 'WAL',
    'cache_size': -1024 * 64,
    'synchronous': 'NORMAL',
    'mmap_size': 1024 * 1024 * 128,
    'temp_store': 'MEMORY'
})

# -----------------------------------------------------------
# DATA STRUCTURES
# -----------------------------------------------------------
@dataclass
class StatBlock:
    """Helper to organize raw DB stats."""
    wins: int
    ties: int
    neutral: int
    losses: int
    total: int

    @property
    def probs(self):
        """Returns tuple (pW, pT, pN, pL)"""
        if self.total == 0: return (0.0, 0.0, 0.0, 0.0)
        return (self.wins/self.total, self.ties/self.total, self.neutral/self.total, self.losses/self.total)

    @property
    def certainty(self) -> float:
        if self.total == 0: return 0.0
        return max(self.wins, self.ties, self.neutral, self.losses) / self.total

    @property
    def utility(self) -> float:
        if self.total == 0: return 0.0
        return (self.wins + self.ties + self.neutral - self.losses) / self.total

    @classmethod
    def from_row(cls, row, snapshot_player: int, acting_player: int):
        w, t, n, l = row['win_count'], row['tie_count'], row['neutral_count'], row['loss_count']
        if snapshot_player != acting_player:
            w, l = l, w 
        return cls(w, t, n, l, w + t + n + l)

# -----------------------------------------------------------
# DATABASE MODELS
# -----------------------------------------------------------
class BaseModel(Model):
    class Meta:
        database = db

class Position(BaseModel):
    game_id = TextField()
    board_hash = IntegerField(primary_key=True)
    board_bytes = BlobField()
    meta_json = TextField()
    class Meta:
        table_name = "positions"

class Move(BaseModel):
    game_id = TextField()
    move_hash = IntegerField(primary_key=True)
    move_bytes = BlobField()
    meta_json = TextField()
    class Meta:
        table_name = "moves"

class PlayStats(BaseModel):
    game_id = TextField()
    board_hash = IntegerField()
    move_hash = IntegerField()
    snapshot_player = IntegerField()
    acting_player = IntegerField()
    win_count = IntegerField(default=0)
    tie_count = IntegerField(default=0)
    neutral_count = IntegerField(default=0)
    loss_count = IntegerField(default=0)
    class Meta:
        table_name = "play_stats"
        primary_key = CompositeKey("game_id", "board_hash", "move_hash", "snapshot_player", "acting_player")
        indexes = ((('game_id', 'board_hash', 'snapshot_player'), False),)

# -----------------------------------------------------------
# GAME MEMORY MANAGER
# -----------------------------------------------------------
class GameMemory:
    def __init__(self, base_dir="omnicron/game_memory/games", db_path="omnicron/game_memory/memory.db", flush_every=5000):
        self.base_dir = base_dir
        self.flush_every = int(flush_every)
        self.write_counter = 0
        os.makedirs(self.base_dir, exist_ok=True)
        db.init(db_path)
        db.connect(reuse_if_open=True)
        db.create_tables([Position, Move, PlayStats])
        self._transition_cache: Dict[Tuple[int, int], int] = {}

        # Optimized SQL queries
        self._sql = {
            'upsert_stats': """
                INSERT INTO play_stats (game_id, board_hash, move_hash, snapshot_player, acting_player, win_count, tie_count, neutral_count, loss_count) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id, board_hash, move_hash, snapshot_player, acting_player) 
                DO UPDATE SET win_count=win_count+excluded.win_count, tie_count=tie_count+excluded.tie_count, 
                              neutral_count=neutral_count+excluded.neutral_count, loss_count=loss_count+excluded.loss_count
            """,
            'get_moves': "SELECT move_hash, acting_player, win_count, tie_count, neutral_count, loss_count FROM play_stats WHERE game_id=? AND board_hash=? AND snapshot_player=?",
            'get_move_bytes': "SELECT move_bytes, meta_json FROM moves WHERE game_id=? AND move_hash=?",
            'insert_pos': "INSERT OR IGNORE INTO positions (game_id, board_hash, board_bytes, meta_json) VALUES (?, ?, ?, ?)",
            'insert_move': "INSERT OR IGNORE INTO moves (game_id, move_hash, move_bytes, meta_json) VALUES (?, ?, ?, ?)"
        }

    # -------------------------
    # Utilities (Cached)
    # -------------------------
    def _normalize_board(self, board) -> np.ndarray:
        if isinstance(board, np.ndarray) and board.dtype == np.int64: return board
        arr = np.array(board, copy=True)
        if arr.dtype == object: arr = np.where(arr == None, 0, arr)
        return arr.astype(np.int64)

    @lru_cache(maxsize=10000)
    def _decode_move(self, gid: str, move_hash: int) -> Optional[Tuple]:
        cursor = db.execute_sql(self._sql['get_move_bytes'], (gid, move_hash))
        row = cursor.fetchone()
        if not row: return None
        meta = json.loads(row[1])
        arr = deserialize_array(row[0], meta["dtype"], json.dumps(meta["shape"]))
        return tuple(arr.ravel())

    # -------------------------
    # Fast Write
    # -------------------------
    def write(self, game_id: str, snapshot_state: GameState, acting_player: int, outcome: State, move: np.ndarray):
        if snapshot_state is None: return
        gid = str(game_id).lower().replace(" ", "_")
        board_arr = self._normalize_board(snapshot_state.board)
        move_arr = np.array(move, copy=True)
        b_hash, m_hash = int(hash_array(board_arr)), int(hash_array(move_arr))
        s_player = int(snapshot_state.current_player)

        # Optimistic Writes
        b_bytes, b_dtype, _, b_shape = serialize_array(board_arr)
        db.execute_sql(self._sql['insert_pos'], (gid, b_hash, b_bytes, json.dumps({"dtype": b_dtype, "shape": json.loads(b_shape)})))
        m_bytes, m_dtype, _, m_shape = serialize_array(move_arr)
        db.execute_sql(self._sql['insert_move'], (gid, m_hash, m_bytes, json.dumps({"dtype": m_dtype, "shape": json.loads(m_shape)})))

        # Upsert Stats
        counts = [0, 0, 0, 0]
        if outcome == State.WIN: counts[0] = 1
        elif outcome == State.TIE: counts[1] = 1
        elif outcome == State.NEUTRAL: counts[2] = 1
        elif outcome == State.LOSS: counts[3] = 1
        db.execute_sql(self._sql['upsert_stats'], (gid, b_hash, m_hash, s_player, acting_player, *counts))
        self._check_flush(gid)

    def _check_flush(self, gid):
        self.write_counter += 1
        if self.write_counter % self.flush_every == 0:
            try:
                backup_path = os.path.join(self.base_dir, gid, "plays.sqlite")
                db.execute_sql(f"VACUUM main INTO '{backup_path}';")
            except Exception: pass

    # -------------------------
    # Logic: Evaluation
    # -------------------------
    def _evaluate_candidate(self, gid: str, board_arr: np.ndarray, row_dict: dict, snapshot_player: int) -> dict:
        """Converts DB row to rich dictionary matching original code's debug schema."""
        stats = StatBlock.from_row(row_dict, snapshot_player, row_dict['acting_player'])
        if stats.total == 0: return None
        pW, pT, pN, pL = stats.probs

        # Lookahead variables
        opp_util = None
        opp_cert = None
        opp_move_coords = None
        opp_probs = (None, None, None, None) # pW, pT, pN, pL for opponent

        # Calculate Transition
        move_hash = row_dict['move_hash']
        cache_key = (int(hash_array(board_arr)), move_hash)
        
        if cache_key not in self._transition_cache:
            move_tup = self._decode_move(gid, move_hash)
            if move_tup and len(move_tup) >= 2:
                next_board = board_arr.copy()
                try:
                    next_board[int(move_tup[0]), int(move_tup[1])] = snapshot_player
                    self._transition_cache[cache_key] = int(hash_array(next_board))
                except IndexError:
                    self._transition_cache[cache_key] = -1
            else:
                self._transition_cache[cache_key] = -1
        
        next_hash = self._transition_cache[cache_key]

        if next_hash != -1:
            # We get (StatBlock, move_hash) from the opponent helper
            opp_res = self._get_best_opponent_stats(gid, next_hash, snapshot_player)
            if opp_res:
                opp_stats, opp_m_hash = opp_res
                opp_util = -opp_stats.utility # Flip utility
                opp_cert = opp_stats.certainty
                opp_probs = opp_stats.probs
                
                # Decode opponent move for "Opp Reply" visualization
                opp_move_tup = self._decode_move(gid, opp_m_hash)
                opp_move_coords = list(opp_move_tup) if opp_move_tup else None

        adjusted_utility = stats.utility + (opp_util if opp_util is not None else 0)
        dangerous = (opp_util is not None and opp_util <= -0.9 and opp_cert >= 0.8)

        # Full Dictionary expected by debug_viz
        return {
            "move_hash": move_hash,
            "utility": stats.utility,
            "adjusted_utility": adjusted_utility,
            "certainty": stats.certainty,
            "total": stats.total,
            "pW": pW, "pT": pT, "pN": pN, "pL": pL,
            # Opponent Data
            "opponent_data_exists": opp_util is not None,
            "adjusted": opp_util is not None, # Triggers "Eye" icon
            "opponent_best_move": opp_move_coords,
            "opponent_pW": opp_probs[0],
            "opponent_pT": opp_probs[1],
            "opponent_pN": opp_probs[2],
            "opponent_pL": opp_probs[3],
            "opponent_util_for_me": opp_util,
            "opponent_cert": opp_cert,
            "dangerous": dangerous
        }

    def _get_best_opponent_stats(self, gid: str, board_hash: int, my_player_id: int) -> Optional[Tuple[StatBlock, int]]:
        """Finds best opponent move. Returns (StatBlock, move_hash)."""
        query = """
            SELECT win_count, tie_count, neutral_count, loss_count, snapshot_player, acting_player, move_hash 
            FROM play_stats 
            WHERE game_id=? AND board_hash=? AND snapshot_player != ?
        """
        cursor = db.execute_sql(query, (gid, board_hash, my_player_id))
        
        best_block = None
        best_m_hash = None
        best_util = -9999.0

        for row in cursor.fetchall():
            row_d = {'win_count': row[0], 'tie_count': row[1], 'neutral_count': row[2], 'loss_count': row[3]}
            sb = StatBlock.from_row(row_d, row[4], row[5])
            if sb.utility > best_util:
                best_util = sb.utility
                best_block = sb
                best_m_hash = row[6]
        
        if best_block:
            return (best_block, best_m_hash)
        return None

    # -------------------------
    # Public API
    # -------------------------
    def _select_move(self, game_id: str, game_state: GameState, pick_best: bool, debug_move: bool):
        gid = str(game_id).lower().replace(" ", "_")
        snap_player = int(game_state.current_player)
        board_arr = self._normalize_board(game_state.board)
        board_hash = int(hash_array(board_arr))

        cursor = db.execute_sql(self._sql['get_moves'], (gid, board_hash, snap_player))
        columns = [c[0] for c in cursor.description]
        rows = [dict(zip(columns, r)) for r in cursor.fetchall()]

        if not rows: return None

        evals = []
        for r in rows:
            res = self._evaluate_candidate(gid, board_arr, r, snap_player)
            if res: evals.append(res)

        if not evals: return None

        # Sort key logic
        key = lambda x: (x['adjusted_utility'], x['total'], x['certainty'])
        if not pick_best:
            # For worst: prefer LOW adjusted utility, but HIGH total/certainty (reliable failure)
            key = lambda x: (-x['adjusted_utility'], x['total'], x['certainty'])
            
        evals.sort(key=key, reverse=True)
        chosen = evals[0]

        if debug_move:
            self._debug_render(gid, board_arr, evals, chosen['move_hash'], pick_best)

        tup = self._decode_move(gid, chosen['move_hash'])
        return np.array(tup) if tup else None

    def _debug_render(self, gid, board, evals, chosen_hash, pick_best):
        """Passes full data context to visualizer."""
        debug_rows = []
        for e in evals:
            tup = self._decode_move(gid, e['move_hash'])
            # Merge the evaluation dict with the specific flags the UI needs
            row_data = {
                **e,
                "move_array": list(tup) if tup else None,
                "is_selected": e['move_hash'] == chosen_hash,
                "is_best": pick_best and (e['move_hash'] == chosen_hash),  # Triggers Best Badge
                "is_worst": (not pick_best) and (e['move_hash'] == chosen_hash), # Triggers Worst Badge
                # Opponent metrics mapping for display
                "opponent_best_util_for_them": (-e['opponent_util_for_me']) if e['opponent_util_for_me'] is not None else None,
                "opponent_best_util_for_me": e['opponent_util_for_me'],
                "opponent_best_cert": e['opponent_cert'],
            }
            debug_rows.append(row_data)
        render_debug(board, debug_rows)

    def get_best_move(self, game_id: str, game_state: GameState, debug_move=False):
        return self._select_move(game_id, game_state, True, debug_move)

    def get_worst_move(self, game_id: str, game_state: GameState, debug_move=False):
        return self._select_move(game_id, game_state, False, debug_move)