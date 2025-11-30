"""
Game memory manager with hybrid graph/tensor canonicalization (Option 3).
Fully compatible with refactored state_canonicalizer.py.
"""

import json
import logging
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass
from collections import OrderedDict

import numpy as np
from peewee import (
    Model, IntegerField, BlobField, TextField, SqliteDatabase, CompositeKey
)

# Assuming these are defined in your environment
from omnicron.serializers import serialize_array, deserialize_array, hash_array
from agent.agent import State
from omnicron.state_canonicalizer import (
    canonicalize_state,
    canonicalize_move,
    invert_canonical_move,
)
from omnicron.debug_viz import render_debug

# --- Logger Setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------
# CONFIG: small canonicalization LRU for repeated lookups
# -----------------------------------------------------------
class SimpleLRU:
    # ... (existing SimpleLRU class definition is unchanged) ...
    def __init__(self, maxsize=8192):
        self.maxsize = int(maxsize)
        self._cache = OrderedDict()

    def get(self, key):
        try:
            v = self._cache.pop(key)
            self._cache[key] = v
            return v
        except KeyError:
            return None

    def put(self, key, value):
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

# small global cache (configurable)
_CANON_CACHE = SimpleLRU(maxsize=4096)

# -----------------------------------------------------------
# DATABASE SETUP (unchanged schema)
# -----------------------------------------------------------
db = SqliteDatabase(None, pragmas={
    'journal_mode': 'WAL',
    'cache_size': -1024 * 64,
    'synchronous': 'NORMAL',
    'mmap_size': 1024 * 1024 * 128,
    'temp_store': 'MEMORY'
})

class BaseModel(Model):
    class Meta:
        database = db

class Position(BaseModel):
    board_hash = TextField(primary_key=True)
    game_id = TextField()
    board_bytes = BlobField()
    original_board_bytes = BlobField()
    meta_json = TextField()

    class Meta:
        table_name = "positions"
        indexes = (
            (("game_id", "board_hash"), False),
        )

class Move(BaseModel):
    move_hash = TextField(primary_key=True)
    game_id = TextField()
    move_bytes = BlobField()
    meta_json = TextField()

    class Meta:
        table_name = "moves"
        indexes = (
            (("game_id", "move_hash"), False),
        )

class PlayStats(BaseModel):
    game_id = TextField()
    board_hash = TextField()
    move_hash = TextField()
    snapshot_player = IntegerField()
    acting_player = IntegerField()

    win_count = IntegerField(default=0)
    tie_count = IntegerField(default=0)
    neutral_count = IntegerField(default=0)
    loss_count = IntegerField(default=0)

    class Meta:
        table_name = "play_stats"
        primary_key = CompositeKey(
            "game_id", "board_hash", "move_hash", "snapshot_player", "acting_player"
        )
        indexes = (
            (("game_id", "board_hash", "snapshot_player"), False),
        )

# -----------------------------------------------------------
# STATISTICS HELPER (unchanged)
# -----------------------------------------------------------
@dataclass
class StatBlock:
    wins: int
    ties: int
    neutral: int
    losses: int

    @property
    def total(self) -> int:
        return self.wins + self.ties + self.neutral + self.losses

    @property
    def probs(self) -> Tuple[float, float, float, float]:
        if self.total == 0:
            return (0.0, 0.0, 0.0, 0.0)
        t = self.total
        return (self.wins / t, self.ties / t, self.neutral / t, self.losses / t)

    @property
    def certainty(self) -> float:
        if self.total == 0:
            return 0.0
        pW, pT, pN, pL = self.probs
        return (pW + pT + pN) - pL

    @property
    def utility(self) -> float:
        if self.total == 0:
            return 0.5
        pW = self.probs[0]
        pT = self.probs[1]
        return pW + 0.5*pT

    @property
    def score(self) -> float:
        if self.total == 0:
            return 0.5
        U = self.utility
        C = self.certainty
        return U * C

    @classmethod
    def from_row(cls, row_dict: dict, snapshot_player: int, acting_player: int):
        w = int(row_dict.get('win_count', 0))
        t = int(row_dict.get('tie_count', 0))
        n = int(row_dict.get('neutral_count', 0))
        l = int(row_dict.get('loss_count', 0))
        if snapshot_player != acting_player:
            w, l = l, w
        return cls(wins=w, ties=t, neutral=n, losses=l)

# -----------------------------------------------------------
# GAME MEMORY MANAGER
# -----------------------------------------------------------
class GameMemory:
    def __init__(self, db_path: str = "memory.db", canon_cache_size: int = 4096):
        self.db_path = db_path
        self._canon_cache = _CANON_CACHE
        db.init(db_path)
        db.connect(reuse_if_open=True)
        db.create_tables([Position, Move, PlayStats])
        logger.info(f"GameMemory initialized: {db_path} (canon_cache_size={canon_cache_size})")

    def _normalize_game_id(self, game_id: str) -> str:
        return game_id.lower().replace(" ", "_").replace("-", "_")

    def _serialize_meta(self, dtype, dtype_str, shape) -> str:
        if isinstance(shape, str):
            try:
                shape_val = json.loads(shape)
            except Exception:
                shape_val = shape
        elif isinstance(shape, tuple):
            shape_val = list(shape)
        else:
            shape_val = shape
        return json.dumps({
            "dtype": str(dtype),
            "dtype_str": str(dtype_str),
            "shape": shape_val
        })

    # -------------------------
    # Canonicalization helper with LRU keyed by board bytes + player (when possible)
    # -------------------------
    def _canonicalize_cached(self, game_state: Any) -> Dict[str, Any]:
        """
        Try to cache canonicalize_state results when we can build a stable bytes key.
        The key now incorporates board bytes, current player, and board shape for stability.
        """
        # Fast path for board-like states
        try:
            if hasattr(game_state, "board"):
                board = np.asarray(game_state.board, dtype=object)
                b_bytes, b_dtype, _, b_shape = serialize_array(board)
                player = int(getattr(game_state, "current_player", 0))
                
                # --- FIX: Use the corrected triple-arg hash function for the cache key ---
                key = hashlib_sha256_bytes_triple_arg(b_bytes, player, json.dumps(b_shape))
                
                cached = self._canon_cache.get(key)
                if cached:
                    return cached
                # not cached -> compute and store
                res = canonicalize_state(game_state)
                # ensure required keys exist
                if not isinstance(res, dict) or 'hash' not in res or 'iso_map' not in res:
                    raise RuntimeError("canonicalize_state produced unexpected result")
                self._canon_cache.put(key, res)
                return res
        except Exception as e:
            # fall back to uncached call on any error
            logger.debug("canonicalize cache fast-path failed, falling back: %s", e)

        # fallback: no caching for general states
        res = canonicalize_state(game_state)
        if not isinstance(res, dict) or 'hash' not in res or 'iso_map' not in res:
            raise RuntimeError("canonicalize_state produced unexpected result")
        return res

    # -------------------------
    # Main write method
    # -------------------------
    def write(self,
              game_id: str,
              snapshot_state,
              acting_player: int,
              outcome: State,
              move: np.ndarray) -> None:
        if snapshot_state is None:
            logger.warning("Skipping write: snapshot_state is None")
            return

        gid = self._normalize_game_id(game_id)

        # Canonicalize (cached when possible)
        try:
            canonical = self._canonicalize_cached(snapshot_state)
        except Exception as exc:
            logger.exception("Canonicalization failed; skipping write. Error: %s", exc)
            return

        # canonical expected keys: 'hash' (str), 'iso_map' (dict)
        board_hash = str(canonical["hash"])
        iso_map = canonical["iso_map"]

        # Store the original board bytes for debug / visualization
        original_board = snapshot_state.board
        orig_bytes, orig_dtype, orig_dtype_str, orig_shape = serialize_array(original_board)
        orig_meta = self._serialize_meta(orig_dtype, orig_dtype_str, orig_shape)

        try:
            db.execute_sql(
                "INSERT OR IGNORE INTO positions (board_hash, game_id, board_bytes, original_board_bytes, meta_json) VALUES (?, ?, ?, ?, ?)",
                (board_hash, gid, orig_bytes, orig_bytes, orig_meta)
            )
        except Exception as exc:
            logger.exception("Failed to INSERT position: %s", exc)
            # proceed — if insert failed it's probably ok (duplicate or DB issue)

        # Canonicalize the move using iso_map
        try:
            canonical_move = canonicalize_move(move, iso_map)
        except Exception as exc:
            logger.exception("canonicalize_move failed; skipping write. Error: %s", exc)
            return

        if canonical_move is None:
            # treat as no-op: we can't canonicalize the move (store raw)
            canonical_move = np.asarray(move, dtype=object)

        try:
            move_hash = str(hash_array(canonical_move))
        except Exception:
            # ensure hashable representation (this uses the stable array hashing)
            move_hash = hashlib_sha256_bytes_for_array(canonical_move)

        # Serialize canonical move
        move_bytes, move_dtype, move_dtype_str, move_shape = serialize_array(canonical_move)
        move_meta = self._serialize_meta(move_dtype, move_dtype_str, move_shape)

        try:
            db.execute_sql(
                "INSERT OR IGNORE INTO moves (move_hash, game_id, move_bytes, meta_json) VALUES (?, ?, ?, ?)",
                (move_hash, gid, move_bytes, move_meta)
            )
        except Exception as exc:
            logger.exception("Failed to INSERT move: %s", exc)

        # Update stats (ensure ints)
        try:
            snapshot_player = int(snapshot_state.current_player)
        except Exception:
            snapshot_player = int(getattr(snapshot_state, "current_player", 0))
        try:
            acting_player = int(acting_player)
        except Exception:
            acting_player = int(acting_player) if isinstance(acting_player, int) else 0

        outcome_counts = [0, 0, 0, 0]
        if outcome == State.WIN:
            outcome_counts[0] = 1
        elif outcome == State.TIE:
            outcome_counts[1] = 1
        elif outcome == State.NEUTRAL:
            outcome_counts[2] = 1
        elif outcome == State.LOSS:
            outcome_counts[3] = 1

        try:
            db.execute_sql("""
                INSERT INTO play_stats (
                    game_id, board_hash, move_hash,
                    snapshot_player, acting_player,
                    win_count, tie_count, neutral_count, loss_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id, board_hash, move_hash, snapshot_player, acting_player)
                DO UPDATE SET
                    win_count = win_count + excluded.win_count,
                    tie_count = tie_count + excluded.tie_count,
                    neutral_count = neutral_count + excluded.neutral_count,
                    loss_count = loss_count + excluded.loss_count
            """, (gid, board_hash, move_hash, snapshot_player, acting_player, *outcome_counts))
        except Exception as exc:
            logger.exception("Failed to upsert play_stats: %s", exc)

    # -------------------------
    # Helpers
    # -------------------------
    def _get_original_board(self, game_id: str, board_hash: str) -> Optional[np.ndarray]:
        gid = self._normalize_game_id(game_id)
        cursor = db.execute_sql(
            "SELECT original_board_bytes, meta_json FROM positions WHERE game_id=? AND board_hash=?",
            (gid, board_hash)
        )
        row = cursor.fetchone()
        if not row:
            return None
        orig_bytes, meta_json = row
        meta = json.loads(meta_json)
        dtype_str = meta.get('dtype_str', meta.get('dtype'))
        shape_json = json.dumps(meta['shape'])
        return deserialize_array(orig_bytes, dtype_str, shape_json)

    def _decode_move(self, game_id: str, move_hash: str) -> Optional[np.ndarray]:
        gid = self._normalize_game_id(game_id)
        cursor = db.execute_sql(
            "SELECT move_bytes, meta_json FROM moves WHERE game_id=? AND move_hash=?",
            (gid, move_hash)
        )
        row = cursor.fetchone()
        if not row:
            return None
        move_bytes, meta_json = row
        meta = json.loads(meta_json)
        dtype_str = meta.get('dtype_str', meta.get('dtype'))
        shape_json = json.dumps(meta['shape'])
        return deserialize_array(move_bytes, dtype_str, shape_json)

    def _evaluate_move(self, game_id: str, game_state, row_dict: dict, iso_map: dict) -> dict:
        snapshot_player = int(game_state.current_player)
        stats = StatBlock.from_row(row_dict, snapshot_player, int(row_dict['acting_player']))
        pW, pT, pN, pL = stats.probs
        move_hash = row_dict['move_hash']
        canonical_move = self._decode_move(game_id, move_hash)

        move_coords = None

        if canonical_move is not None:
            try:
                # --- FIX: make sure we pass the inverse map (int -> original node) to invert_canonical_move
                # If the caller gave us iso_map (orig_node -> int), build the inverse here.
                if iso_map is None:
                    inverse_map = None
                else:
                    # If iso_map already appears to be inverse (int->orig), detect that:
                    # check a single key type to avoid inverting twice.
                    sample_key = next(iter(iso_map.keys()))
                    if isinstance(sample_key, int):
                        inverse_map = iso_map  # it is already inverse
                    else:
                        inverse_map = {v: k for k, v in iso_map.items()}

                original_move = invert_canonical_move(canonical_move, inverse_map)

                # original_move is either np.ndarray coords or an object/tuple
                if isinstance(original_move, np.ndarray):
                    move_coords = original_move.tolist()
                elif isinstance(original_move, (list, tuple)):
                    move_coords = list(original_move)
                else:
                    # Can be None or some abstract move token (e.g. "pass")
                    move_coords = original_move
                # If inversion produced None, log for debug
                if move_coords is None:
                    logger.debug("invert_canonical_move returned None for move_hash=%s canonical_move=%r",
                                 move_hash, canonical_move)
            except Exception:
                logger.exception("Failed to invert canonical move for move_hash=%s", move_hash)
                move_coords = None
        else:
            move_coords = None

        return {
            "move_hash": move_hash,
            "move_array": move_coords,
            "utility": stats.utility,
            "certainty": stats.certainty,
            "score": stats.score,
            "adjusted_score": stats.score,
            "total": stats.total,
            "pW": pW, "pT": pT, "pN": pN, "pL": pL,
            "opponent_data_exists": False,
            "adjusted": False,
            "opponent_best_move": None,
            "opponent_pW": None,
            "opponent_pT": None,
            "opponent_pN": None,
            "opponent_pL": None,
            "opponent_util": None,
            "opponent_cert": None,
            "opponent_score": None,
            "dangerous": False
        }


    def _select_move(self, game_id: str, game_state, pick_best: bool = True,
                     debug_move: bool = False) -> Optional[np.ndarray]:
        gid = self._normalize_game_id(game_id)
        snapshot_player = int(game_state.current_player)

        # canonicalize current state
        try:
            canonical = self._canonicalize_cached(game_state)
        except Exception as exc:
            logger.exception("Canonicalization failed during select: %s", exc)
            return None

        board_hash = str(canonical["hash"])
        iso_map = canonical["iso_map"]

        cursor = db.execute_sql("""
            SELECT move_hash, acting_player, win_count, tie_count, neutral_count, loss_count
            FROM play_stats
            WHERE game_id=? AND board_hash=? AND snapshot_player=?
        """, (gid, board_hash, snapshot_player))

        rows = cursor.fetchall()
        if not rows:
            return None

        evals = []
        for row in rows:
            row_dict = {
                'move_hash': row[0],
                'acting_player': int(row[1]),
                'win_count': int(row[2]),
                'tie_count': int(row[3]),
                'neutral_count': int(row[4]),
                'loss_count': int(row[5]),
            }
            # The _evaluate_move call uses the iso_map to invert the canonical move
            evals.append(self._evaluate_move(gid, game_state, row_dict, iso_map))

        if not evals:
            return None

        if pick_best:
            # key = lambda x: (x['score'], x['certainty'])
            key = lambda x: (x['score'])
        else:
            # key = lambda x: (-x['score'], x['certainty'])
            key = lambda x: (-x['score'])

        evals.sort(key=key, reverse=True)
        chosen = evals[0]
        chosen_hash = chosen['move_hash']

        if debug_move:
            debug_rows = []
            for e in evals:
                row_data = {
                    **e,
                    "is_selected": e['move_hash'] == chosen_hash,
                    "is_best": pick_best and (e['move_hash'] == chosen_hash),
                    "is_worst": (not pick_best) and (e['move_hash'] == chosen_hash),
                }
                debug_rows.append(row_data)
            # Assuming render_debug handles array or list inputs for moves
            render_debug(game_state.board, debug_rows)

        if chosen['move_array'] is not None:
            # Must return a numpy array of integers for the game engine
            # It's already been converted to a list/tuple in _evaluate_move
            return np.array(chosen['move_array'], dtype=int)
        return None

    def get_best_move(self, game_id: str, game_state, debug_move: bool = False) -> Optional[np.ndarray]:
        return self._select_move(game_id, game_state, pick_best=True, debug_move=debug_move)

    def get_worst_move(self, game_id: str, game_state, debug_move: bool = False) -> Optional[np.ndarray]:
        return self._select_move(game_id, game_state, pick_best=False, debug_move=debug_move)

    def close(self):
        if not db.is_closed():
            db.close()

# ---------------------------
# Small helpers (module-level)
# ---------------------------
import hashlib as _hashlib

# --- FIX: Renamed and clarified hash helpers ---

def hashlib_sha256_bytes_single_arg(b: bytes) -> str:
    """Helper for simple bytes hashing (e.g., in Move hashing fallback)"""
    return _hashlib.sha256(b).hexdigest()

def hashlib_sha256_bytes_for_array(arr: np.ndarray) -> str:
    """Hashes an array using serialization."""
    try:
        a = np.asarray(arr)
        b, *_ = serialize_array(a)
        # Use the single-arg version for simple content hash
        return hashlib_sha256_bytes_single_arg(b)
    except Exception:
        # worst-case fallback
        return hashlib_sha256_bytes_single_arg(repr(arr).encode())

def hashlib_sha256_bytes_triple_arg(b: bytes, player: int, shape_json: str) -> str:
    """Helper for state/cache key hashing (data, player, shape)"""
    m = _hashlib.sha256()
    m.update(b)
    m.update(str(int(player)).encode())
    m.update(shape_json.encode())
    return m.hexdigest()