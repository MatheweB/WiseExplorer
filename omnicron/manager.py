# manager.py
import os
import json
import logging
import numpy as np
from typing import List, Tuple, Optional
from peewee import (
    Model,
    IntegerField,
    BlobField,
    TextField,
    SqliteDatabase,
    CompositeKey,
    DoesNotExist,
)
from omnicron.serializers import serialize_array, deserialize_array, hash_array
from agent.agent import State
from games.game_state import GameState
from .debug_viz import render_debug
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)
db = SqliteDatabase(None)
# ===========================================================
# DATABASE MODELS
# ===========================================================
class Position(Model):
    """Stores unique board positions."""
    game_id = TextField()
    board_hash = IntegerField(primary_key=True)
    board_bytes = BlobField()
    meta_json = TextField()
    class Meta:
        database = db
        table_name = "positions"
class Move(Model):
    """Stores unique moves."""
    game_id = TextField()
    move_hash = IntegerField(primary_key=True)
    move_bytes = BlobField()
    meta_json = TextField()
    class Meta:
        database = db
        table_name = "moves"
class PlayStats(Model):
    """Stores outcome statistics for (board, move, player) combinations."""
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
        database = db
        table_name = "play_stats"
        primary_key = CompositeKey(
            "game_id", "board_hash", "move_hash", "snapshot_player", "acting_player"
        )
# ===========================================================
# GAME MEMORY
# ===========================================================
class GameMemory:
    """
    Manages game position learning with N-player minimax lookahead.
    Uses normalized board representations for consistent hashing.
    """
    def __init__(
        self,
        base_dir="omnicron/game_memory/games",
        db_path="omnicron/game_memory/memory.db",
        flush_every: int = 5000,
    ):
        self.base_dir = base_dir
        self.flush_every = int(flush_every)
        self.write_counter = 0
        os.makedirs(self.base_dir, exist_ok=True)
        # Initialize database
        db.init(db_path)
        db.connect(reuse_if_open=True)
        db.create_tables([Position, Move, PlayStats])
        # Performance optimizations
        try:
            db.execute_sql("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        # Create indexes for faster queries
        db.execute_sql(
            "CREATE INDEX IF NOT EXISTS idx_ps_gb ON play_stats(game_id, board_hash);"
        )
        db.execute_sql(
            "CREATE INDEX IF NOT EXISTS idx_ps_mv ON play_stats(game_id, move_hash);"
        )
        db.execute_sql(
            "CREATE INDEX IF NOT EXISTS idx_ps_snap ON play_stats(game_id, snapshot_player);"
        )
        db.execute_sql(
            "CREATE INDEX IF NOT EXISTS idx_ps_act ON play_stats(game_id, acting_player);"
        )
    # -------------------------------------------------------
    # Board Normalization
    # -------------------------------------------------------
    def _normalize_board(self, board) -> np.ndarray:
        """
        Normalize board representation for consistent hashing.
        Converts None to 0 and ensures int64 dtype.
        """
        arr = np.array(board, copy=True)
        # Convert object dtype (contains None) to numeric
        if arr.dtype == object:
            arr = np.where(arr == None, 0, arr)
            arr = arr.astype(np.int64)
        elif arr.dtype != np.int64:
            arr = arr.astype(np.int64)
        return arr
    # -------------------------------------------------------
    # Write Training Data
    # -------------------------------------------------------
    def write(
        self,
        game_id: str,
        snapshot_state: GameState,
        acting_player: int,
        outcome: State,
        move: np.ndarray,
    ):
        """Record the outcome of a move from a given board position."""
        if snapshot_state is None:
            raise ValueError("snapshot_state must not be None")
        gid = str(game_id).lower().replace(" ", "_")
        board_arr = self._normalize_board(snapshot_state.board)
        move_arr = np.array(move, copy=True)
        snapshot_player = int(snapshot_state.current_player)
        board_hash = int(hash_array(board_arr))
        move_hash = int(hash_array(move_arr))
        # Store unique board position
        try:
            Position.get(
                (Position.game_id == gid) & (Position.board_hash == board_hash)
            )
        except DoesNotExist:
            b_bytes, dtype, dtype2, shape_str = serialize_array(board_arr)
            meta = {"dtype": dtype, "dtype2": dtype2, "shape": json.loads(shape_str)}
            Position.create(
                game_id=gid,
                board_hash=board_hash,
                board_bytes=b_bytes,
                meta_json=json.dumps(meta),
            )
        # Store unique move
        try:
            Move.get((Move.game_id == gid) & (Move.move_hash == move_hash))
        except DoesNotExist:
            m_bytes, dtype, dtype2, shape_str = serialize_array(move_arr)
            meta = {"dtype": dtype, "dtype2": dtype2, "shape": json.loads(shape_str)}
            Move.create(
                game_id=gid,
                move_hash=move_hash,
                move_bytes=m_bytes,
                meta_json=json.dumps(meta),
            )
        # Update or create play statistics
        try:
            row = PlayStats.get(
                (PlayStats.game_id == gid)
                & (PlayStats.board_hash == board_hash)
                & (PlayStats.move_hash == move_hash)
                & (PlayStats.snapshot_player == snapshot_player)
                & (PlayStats.acting_player == acting_player)
            )
        except DoesNotExist:
            row = PlayStats.create(
                game_id=gid,
                board_hash=board_hash,
                move_hash=move_hash,
                snapshot_player=snapshot_player,
                acting_player=acting_player,
            )
        # Increment outcome counts
        if outcome == State.WIN:
            row.win_count += 1
        elif outcome == State.TIE:
            row.tie_count += 1
        elif outcome == State.NEUTRAL:
            row.neutral_count += 1
        elif outcome == State.LOSS:
            row.loss_count += 1
        row.save()
        # Periodic database backup
        self.write_counter += 1
        if self.write_counter % self.flush_every == 0:
            try:
                game_dir = os.path.join(self.base_dir, gid)
                os.makedirs(game_dir, exist_ok=True)
                backup_path = os.path.join(game_dir, "plays.sqlite")
                db.execute_sql(f"VACUUM main INTO '{backup_path}';")
            except Exception:
                pass
    # -------------------------------------------------------
    # Opponent Analysis
    # -------------------------------------------------------
    def _best_opponent_reply(
        self, gid: str, board: np.ndarray, opponents: List[int]
    ) -> Tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[List[int]],
    ]:
        """
        Find the best reply move among all opponents for a given board.
        Returns:
            (opp_pW, opp_pT, opp_pN, opp_pL, opponent_utility, certainty, move_coords)
            where all values are from opponent's perspective
        """
        board_normalized = self._normalize_board(board)
        board_hash = int(hash_array(board_normalized))
        best_util = -9999.0
        best_pW = None
        best_pT = None
        best_pN = None
        best_pL = None
        best_cert = None
        best_move_hash = None
        # Check each opponent's moves on this board
        for opp in opponents:
            rows = list(
                PlayStats.select().where(
                    (PlayStats.game_id == gid)
                    & (PlayStats.board_hash == board_hash)
                    & (PlayStats.snapshot_player == opp)
                )
            )
            if not rows:
                continue
            # Find this opponent's best move
            for row in rows:
                total = (
                    row.win_count + row.tie_count + row.neutral_count + row.loss_count
                )
                if total == 0:
                    continue
                pW = row.win_count / total
                pT = row.tie_count / total
                pN = row.neutral_count / total
                pL = row.loss_count / total
                # Flip to opponent's perspective if needed
                if int(row.acting_player) != opp:
                    pW, pL = pL, pW
                cert = max(pW, pT, pN, pL)
                util = pW + pT + pN - pL
                if util > best_util:
                    best_util = util
                    best_pW = pW
                    best_pT = pT
                    best_pN = pN
                    best_pL = pL
                    best_cert = cert
                    best_move_hash = int(row.move_hash)
        if best_move_hash is None:
            return None, None, None, None, None, None, None
        # Decode move coordinates
        try:
            mv = Move.get((Move.game_id == gid) & (Move.move_hash == best_move_hash))
            meta = json.loads(mv.meta_json)
            move_arr = deserialize_array(
                mv.move_bytes, meta["dtype"], json.dumps(meta["shape"])
            )
            move_coords = list(map(int, np.array(move_arr).ravel()))
        except Exception:
            move_coords = None
        return best_pW, best_pT, best_pN, best_pL, best_util, best_cert, move_coords
    # -------------------------------------------------------
    # Best Move Selection
    # -------------------------------------------------------
    def get_best_move(self, game_id: str, game_state: GameState, debug_move=False):
        """
        Select the best move for the current game state using risk-adjusted minimax.
        Returns:
            Best move as numpy array, or None if no data available
        """
        gid = str(game_id).lower().replace(" ", "_")
        snapshot_player = int(game_state.current_player)
        board_arr = self._normalize_board(game_state.board)
        board_hash = int(hash_array(board_arr))
        # Get candidate moves for current player
        rows = list(
            PlayStats.select().where(
                (PlayStats.game_id == gid)
                & (PlayStats.board_hash == board_hash)
                & (PlayStats.snapshot_player == snapshot_player)
            )
        )
        if not rows:
            return None
        # Evaluate each candidate move
        move_evaluations = []
        for row in rows:
            # Calculate base statistics from our perspective
            total = row.win_count + row.tie_count + row.neutral_count + row.loss_count
            if total == 0:
                pW = pT = pN = pL = 0.0
            else:
                pW = row.win_count / total
                pT = row.tie_count / total
                pN = row.neutral_count / total
                pL = row.loss_count / total
            # Flip to snapshot player's perspective if needed
            if int(row.acting_player) != snapshot_player:
                pW, pL = pL, pW
            certainty = max(pW, pT, pN, pL)
            utility = pW + pT + pN - pL
            # Simulate this move to get resulting board
            next_board = None
            try:
                mv = Move.get((Move.game_id == gid) & (Move.move_hash == row.move_hash))
                meta = json.loads(mv.meta_json)
                move_arr = deserialize_array(
                    mv.move_bytes, meta["dtype"], json.dumps(meta["shape"])
                )
                coords = np.array(move_arr).ravel()
                if coords.size >= 2:
                    next_board = np.array(board_arr, copy=True)
                    next_board[int(coords[0]), int(coords[1])] = snapshot_player
            except Exception:
                pass
            # Evaluate opponent's best response
            opp_pW = None
            opp_pT = None
            opp_pN = None
            opp_pL = None
            opp_util = None
            opp_cert = None
            opp_move = None
            if next_board is not None:
                next_board_norm = self._normalize_board(next_board)
                next_hash = int(hash_array(next_board_norm))
                # Find opponents who have data on this position
                opps_query = (
                    PlayStats.select(PlayStats.snapshot_player)
                    .where(
                        (PlayStats.game_id == gid) & (PlayStats.board_hash == next_hash)
                    )
                    .distinct()
                )
                opps_with_data = [
                    int(r.snapshot_player)
                    for r in opps_query
                    if int(r.snapshot_player) != snapshot_player
                ]
                if opps_with_data:
                    (
                        opp_pW,
                        opp_pT,
                        opp_pN,
                        opp_pL,
                        opp_util_for_them,
                        opp_cert,
                        opp_move,
                    ) = self._best_opponent_reply(gid, next_board_norm, opps_with_data)
                    if opp_util_for_them is not None:
                        opp_util = -opp_util_for_them  # Convert to our perspective
            # Risk adjustment: penalize moves where opponent has strong reply
            # Method 1: Expected damage calculation (current method)
            # This method uses the opponent's certainty to weight the opponent's utility.
            # It aims to account for the reliability of the opponent's best reply.
            # -----
            # adjusted_utility = utility
            # if opp_util is not None and opp_cert is not None:
            #     expected_damage = opp_util * opp_cert
            #     adjusted_utility = utility + expected_damage

            # Method 2: Direct addition of negative utility (alternative method)
            # This method directly adds the negative utility from the opponent's best reply.
            # It does not account for the certainty of the opponent's reply.
            # ------
            adjusted_utility = utility + (opp_util if opp_util is not None else 0)

            # Determine if this move is dangerous
            dangerous = False
            if opp_util is not None and opp_cert is not None:
                if (
                    -opp_util >= 0.9 and opp_cert >= 0.8
                ):  # Opponent has strong advantage
                    dangerous = True
            move_evaluations.append(
                {
                    "move_hash": row.move_hash,
                    "certainty": certainty,
                    "utility": utility,
                    "adjusted_utility": adjusted_utility,
                    "total": total,
                    "pW": pW,
                    "pT": pT,
                    "pN": pN,
                    "pL": pL,
                    "opponent_best_move": opp_move,
                    "opponent_pW": opp_pW,
                    "opponent_pT": opp_pT,
                    "opponent_pN": opp_pN,
                    "opponent_pL": opp_pL,
                    "opponent_util_for_them": (
                        -opp_util if opp_util is not None else None
                    ),
                    "opponent_util_for_me": opp_util,
                    "opponent_cert": opp_cert,
                    "opponent_data_exists": opp_util is not None,
                    "adjusted": adjusted_utility != utility,
                    "dangerous": dangerous,
                }
            )
        # Sort by adjusted utility, then total observations, then certainty
        move_evaluations.sort(
            key=lambda m: (m["adjusted_utility"], m["total"], m["certainty"]),
            reverse=True,
        )
        if not move_evaluations:
            return None
        best_move_hash = move_evaluations[0]["move_hash"]
        # Debug visualization
        if debug_move:
            debug_rows = []
            for eval_data in move_evaluations:
                try:
                    mv = Move.get(
                        (Move.game_id == gid)
                        & (Move.move_hash == eval_data["move_hash"])
                    )
                    meta = json.loads(mv.meta_json)
                    move_arr = deserialize_array(
                        mv.move_bytes, meta["dtype"], json.dumps(meta["shape"])
                    )
                    debug_rows.append(
                        {
                            "move_hash": eval_data["move_hash"],
                            "move_array": (
                                np.array(move_arr).tolist()
                                if not isinstance(move_arr, list)
                                else move_arr
                            ),
                            "certainty": float(eval_data["certainty"]),
                            "utility": float(eval_data["utility"]),
                            "adjusted_utility": float(eval_data["adjusted_utility"]),
                            "total": int(eval_data["total"]),
                            "pW": float(eval_data["pW"]),
                            "pT": float(eval_data["pT"]),
                            "pN": float(eval_data["pN"]),
                            "pL": float(eval_data["pL"]),
                            "opponent_data_exists": bool(
                                eval_data["opponent_data_exists"]
                            ),
                            "adjusted": bool(eval_data["adjusted"]),
                            "opponent_best_move": eval_data["opponent_best_move"],
                            "opponent_pW": (
                                float(eval_data["opponent_pW"])
                                if eval_data.get("opponent_pW") is not None
                                else None
                            ),
                            "opponent_pT": (
                                float(eval_data["opponent_pT"])
                                if eval_data.get("opponent_pT") is not None
                                else None
                            ),
                            "opponent_pN": (
                                float(eval_data["opponent_pN"])
                                if eval_data.get("opponent_pN") is not None
                                else None
                            ),
                            "opponent_pL": (
                                float(eval_data["opponent_pL"])
                                if eval_data.get("opponent_pL") is not None
                                else None
                            ),
                            "opponent_best_util_for_them": (
                                float(eval_data["opponent_util_for_them"])
                                if eval_data["opponent_util_for_them"] is not None
                                else None
                            ),
                            "opponent_best_util_for_me": (
                                float(eval_data["opponent_util_for_me"])
                                if eval_data["opponent_util_for_me"] is not None
                                else None
                            ),
                            "opponent_best_cert": (
                                float(eval_data["opponent_cert"])
                                if eval_data["opponent_cert"] is not None
                                else None
                            ),
                            "dangerous": bool(eval_data["dangerous"]),
                            "is_best": eval_data["move_hash"] == best_move_hash,
                            "sort_key": (
                                eval_data["adjusted_utility"],
                                eval_data["total"],
                                eval_data["certainty"],
                            ),
                        }
                    )
                except Exception:
                    continue
            render_debug(board_arr, debug_rows)
        # Return the best move
        try:
            mv = Move.get((Move.game_id == gid) & (Move.move_hash == best_move_hash))
            meta = json.loads(mv.meta_json)
            return deserialize_array(
                mv.move_bytes, meta["dtype"], json.dumps(meta["shape"])
            )
        except Exception:
            return None
