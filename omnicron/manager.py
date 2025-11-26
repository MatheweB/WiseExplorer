# manager.py
import os
import json
import logging
import numpy as np
from typing import cast, List, Tuple, Optional, Any
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
# NORMALIZED TABLES
# ===========================================================
class Position(Model):
    """One row per unique board."""
    game_id = TextField()
    board_hash = IntegerField(primary_key=True)
    board_bytes = BlobField()
    meta_json = TextField()

    class Meta:
        database = db
        table_name = "positions"


class Move(Model):
    """One row per unique move."""
    game_id = TextField()
    move_hash = IntegerField(primary_key=True)
    move_bytes = BlobField()
    meta_json = TextField()

    class Meta:
        database = db
        table_name = "moves"


class PlayStats(Model):
    """Outcome counts only. The table that grows."""
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
# GAME MEMORY CLASS (parameterless N-player minimax lookahead)
# ===========================================================
class GameMemory:
    """Relational + JSON metadata version."""

    def __init__(
        self,
        base_dir="omnicron/game_memory/games",
        db_path="omnicron/game_memory/memory.db",
        flush_every: int = 5000,
    ):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.flush_every = int(flush_every)
        self.write_counter = 0

        db.init(db_path)
        db.connect(reuse_if_open=True)
        db.create_tables([Position, Move, PlayStats])

        try:
            db.execute_sql("PRAGMA journal_mode=WAL;")
        except Exception:
            pass

        db.execute_sql("CREATE INDEX IF NOT EXISTS idx_ps_gb ON play_stats(game_id, board_hash);")
        db.execute_sql("CREATE INDEX IF NOT EXISTS idx_ps_mv ON play_stats(game_id, move_hash);")
        db.execute_sql("CREATE INDEX IF NOT EXISTS idx_ps_snap ON play_stats(game_id, snapshot_player);")
        db.execute_sql("CREATE INDEX IF NOT EXISTS idx_ps_act ON play_stats(game_id, acting_player);")

    # -------------------------------------------------------
    # Board Normalization Helper
    # -------------------------------------------------------
    def _normalize_board(self, board) -> np.ndarray:
        """
        Normalize board representation for consistent hashing.
        Converts None to 0 and ensures consistent dtype.
        """
        arr = np.array(board, copy=True)
        
        # If object dtype (contains None), convert to numeric
        if arr.dtype == object:
            # Replace None with 0
            arr = np.where(arr == None, 0, arr)
            # Convert to int64 for consistency
            arr = arr.astype(np.int64)
        
        # Ensure consistent dtype even if already numeric
        if arr.dtype != np.int64:
            arr = arr.astype(np.int64)
        
        return arr

    # -------------------------------------------------------
    # WRITE OBSERVATION (with board normalization)
    # -------------------------------------------------------
    def write(
        self,
        game_id: str,
        snapshot_state: GameState,
        acting_player: int,
        outcome: State,
        move: np.ndarray,
    ):
        if snapshot_state is None:
            raise ValueError("snapshot_state must not be None")

        gid = str(game_id).lower().replace(" ", "_")

        # Normalize board before hashing
        board_arr = self._normalize_board(snapshot_state.board)
        move_arr = np.array(move, copy=True)
        snapshot_player = int(snapshot_state.current_player)

        bh = int(hash_array(board_arr))
        mh = int(hash_array(move_arr))

        try:
            Position.get((Position.game_id == gid) & (Position.board_hash == bh))
        except DoesNotExist:
            b_bytes, dtype, dtype2, shape_str = serialize_array(board_arr)
            meta = {"dtype": dtype, "dtype2": dtype2, "shape": json.loads(shape_str)}
            Position.create(game_id=gid, board_hash=bh, board_bytes=b_bytes, meta_json=json.dumps(meta))

        try:
            Move.get((Move.game_id == gid) & (Move.move_hash == mh))
        except DoesNotExist:
            m_bytes, dtype, dtype2, shape_str = serialize_array(move_arr)
            meta = {"dtype": dtype, "dtype2": dtype2, "shape": json.loads(shape_str)}
            Move.create(game_id=gid, move_hash=mh, move_bytes=m_bytes, meta_json=json.dumps(meta))

        try:
            row = PlayStats.get(
                (PlayStats.game_id == gid)
                & (PlayStats.board_hash == bh)
                & (PlayStats.move_hash == mh)
                & (PlayStats.snapshot_player == snapshot_player)
                & (PlayStats.acting_player == acting_player)
            )
        except DoesNotExist:
            row = PlayStats.create(
                game_id=gid,
                board_hash=bh,
                move_hash=mh,
                snapshot_player=snapshot_player,
                acting_player=acting_player,
            )

        if outcome == State.WIN:
            row.win_count += 1
        elif outcome == State.TIE:
            row.tie_count += 1
        elif outcome == State.NEUTRAL:
            row.neutral_count += 1
        elif outcome == State.LOSS:
            row.loss_count += 1

        row.save()

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
    # scoring helper
    # -------------------------------------------------------
    def _score(self, row: PlayStats):
        win = cast(int, row.win_count)
        tie = cast(int, row.tie_count)
        neu = cast(int, row.neutral_count)
        loss = cast(int, row.loss_count)

        total = win + tie + neu + loss
        if total == 0:
            return -1.0, -1.0, 0

        pW = win / total
        pT = tie / total
        pN = neu / total
        pL = loss / total

        certainty = max(pW, pT, pN, pL)
        utility = pW * 1.0 + pT * 1.0 + pN * 1.0 - pL * 1.0

        return certainty, utility, total

    # -------------------------------------------------------
    # Helper: list players observed on a board
    # -------------------------------------------------------
    def _players_on_board(self, gid: str, bh: int) -> List[int]:
        """Return distinct snapshot_player values observed for this game_id & board_hash."""
        try:
            q = PlayStats.select(PlayStats.snapshot_player).where(
                (PlayStats.game_id == gid) & (PlayStats.board_hash == bh)
            ).distinct()
            players = [int(r.snapshot_player) for r in q]
            return players
        except Exception:
            # Fallback: empty
            return []

    # -------------------------------------------------------
    # Helper: compute best opponent utility across a set of opponents
    # -------------------------------------------------------
    def _best_opponent_reply_on_board(self, gid: str, board: np.ndarray, opponents: List[int]) -> Tuple[
        Optional[float], Optional[float], Optional[float], Optional[int], Optional[int], Optional[List[int]]
    ]:
        """
        For the given board where it's some opponent's turn, find the best reply among all opponents.

        Returns:
            (best_pW, best_util_for_them, best_cert, best_player, best_move_hash, best_move_array)

        Note:
            - best_util_for_them is the utility from the opponent's POV (Win=+1, Loss=-1)
            - When used in get_best_move we will compute best_util_for_me = -best_util_for_them
        """
        # Normalize board before hashing
        board_normalized = self._normalize_board(board)
        bh = int(hash_array(board_normalized))
        
        best_pW = -1.0
        best_util_for_them = -9999.0
        best_cert = -1.0
        best_player = None
        best_move_hash = None
        best_move_array = None
        any_data = False

        for opp in opponents:
            # load rows where it's opp's to-move on this exact board hash
            rows_next = list(
                PlayStats.select().where(
                    (PlayStats.game_id == gid) &
                    (PlayStats.board_hash == bh) &
                    (PlayStats.snapshot_player == opp)
                )
            )
            if not rows_next:
                continue

            any_data = True
            # evaluate this opponent's best reply (and capture which move_hash produced it)
            local_best_util = -9999.0
            local_best_pW = -1.0
            local_best_cert = -1.0
            local_best_mh = None

            for rr in rows_next:
                win = int(rr.win_count); tie = int(rr.tie_count)
                neu = int(rr.neutral_count); loss = int(rr.loss_count)
                total = win + tie + neu + loss
                if total == 0:
                    pW = pT = pN = pL = 0.0
                else:
                    pW = win / total; pT = tie / total; pN = neu / total; pL = loss / total

                # flip to opponent POV if acting_player != opp
                if int(rr.acting_player) != opp:
                    pW, pL = pL, pW

                cert = max(pW, pT, pN, pL)
                util_for_them = pW * 1.0 + pT * 1.0 + pN * 1.0 - pL * 1.0

                if util_for_them > local_best_util:
                    local_best_util = util_for_them
                    local_best_pW = pW
                    local_best_cert = cert
                    local_best_mh = int(rr.move_hash)

            # compare to global best across opponents (opponent that is worst for us)
            if local_best_util > best_util_for_them:
                best_util_for_them = local_best_util
                best_pW = local_best_pW
                best_cert = local_best_cert
                best_player = opp
                best_move_hash = local_best_mh

        if not any_data:
            return None, None, None, None, None, None

        # try to resolve best_move_hash -> coords (best_move_array)
        if best_move_hash is not None:
            try:
                mv = Move.get((Move.game_id == gid) & (Move.move_hash == best_move_hash))
                meta = json.loads(mv.meta_json)
                move_arr = deserialize_array(mv.move_bytes, meta["dtype"], json.dumps(meta["shape"]))
                best_move_array = list(map(int, np.array(move_arr).ravel())) if move_arr is not None else None
            except Exception:
                best_move_array = None

        return best_pW, best_util_for_them, best_cert, best_player, best_move_hash, best_move_array

    # -------------------------------------------------------
    # BEST MOVE (parameterless N-player minimax with risk-adjustment)
    # -------------------------------------------------------
    def get_best_move(self, game_id: str, game_state: GameState, debug_move=False):
        """
        Get the best move for the current game state.
        
        Args:
            game_id: Game identifier
            game_state: Current game state
            debug_move: If True, print debug visualization
        """
        gid = str(game_id).lower().replace(" ", "_")
        snapshot_player = int(game_state.current_player)
        
        # Normalize board before hashing
        board_arr = self._normalize_board(game_state.board)
        bh = int(hash_array(board_arr))

        # Gather all players in the game (opponents = all != snapshot_player)
        q = PlayStats.select(PlayStats.snapshot_player).where(
            PlayStats.game_id == gid
        ).distinct()
        players_in_game = [int(r.snapshot_player) for r in q]
        opponents = [p for p in players_in_game if p != snapshot_player]

        # Candidate moves for this snapshot player from training data
        rows = list(
            PlayStats.select().where(
                (PlayStats.game_id == gid) 
                & (PlayStats.board_hash == bh) 
                & (PlayStats.snapshot_player == snapshot_player)
            )
        )
        
        if not rows:
            # No data for this position yet - silently return None
            return None

        candidates = []
        for r in rows:
            win = int(r.win_count)
            tie = int(r.tie_count)
            neu = int(r.neutral_count)
            loss = int(r.loss_count)
            total = win + tie + neu + loss
            
            if total == 0:
                pW = pT = pN = pL = 0.0
            else:
                pW = win / total
                pT = tie / total
                pN = neu / total
                pL = loss / total
            
            if int(r.acting_player) != snapshot_player:
                # Flip outcomes to snapshot player's POV
                pW, pL = pL, pW
            
            cert = max(pW, pT, pN, pL)
            util = pW*1 + pT*1 + pN*1 - pL*1
            
            candidates.append((cert, util, total, r.move_hash, pW, pT, pN, pL))

        adjusted_entries: List[dict] = []
        
        for cert, util, total, mh, pW, pT, pN, pL in candidates:
            next_board = None
            move_coords = None
            
            try:
                mv = Move.get((Move.game_id == gid) & (Move.move_hash == mh))
                meta = json.loads(mv.meta_json)
                move_arr = deserialize_array(mv.move_bytes, meta["dtype"], json.dumps(meta["shape"]))
                arr = np.array(move_arr).ravel()
                
                if arr.size >= 2:
                    rpos, cpos = int(arr[0]), int(arr[1])
                    move_coords = (rpos, cpos)
                    next_board = np.array(board_arr, copy=True)
                    next_board[rpos, cpos] = snapshot_player
            except Exception:
                next_board = None

            # Initialize opponent reply variables
            opp_pW = None
            opp_for_them = None
            opp_for_me = None
            opp_cert = None
            opponent_data_exists = False
            opponent_best_move = None
            opponent_best_move_hash = None
            opponent_best_player = None

            # Evaluate opponent response after this move
            if next_board is not None:
                next_board_normalized = self._normalize_board(next_board)
                next_bh = int(hash_array(next_board_normalized))
                
                # Which players have entries on this next board?
                opps_next_q = PlayStats.select(PlayStats.snapshot_player).where(
                    (PlayStats.game_id == gid) 
                    & (PlayStats.board_hash == next_bh)
                ).distinct()
                opps_next = [
                    int(r.snapshot_player) 
                    for r in opps_next_q 
                    if int(r.snapshot_player) != snapshot_player
                ]
                
                if opps_next:
                    (
                        opp_pW, 
                        opp_util_for_them, 
                        opp_cert_val, 
                        opponent_best_player, 
                        opponent_best_move_hash, 
                        opponent_best_move
                    ) = self._best_opponent_reply_on_board(gid, next_board_normalized, opps_next)
                    
                    if opp_util_for_them is not None:
                        opp_for_them = float(opp_util_for_them)
                        opp_for_me = -float(opp_util_for_them)
                        opp_cert = float(opp_cert_val) if opp_cert_val is not None else None
                        opponent_data_exists = True

            # Risk-adjusted utility
            risk_adjusted_util = util
            adjusted = False
            
            if opp_for_me is not None and opp_cert is not None:
                alpha = 1.0
                # Weight opponent's reply by their certainty
                expected_reply_damage = opp_for_me * opp_cert
                risk_adjusted_util = util + alpha * expected_reply_damage
                adjusted = (risk_adjusted_util != util)

            # Forced-win detection (using properly scoped variables)
            def is_forced_win(pW_val: Optional[float], cert_val: Optional[float]) -> bool:
                if pW_val is None or cert_val is None:
                    return False
                return float(cert_val) >= 0.95 and float(pW_val) >= 0.95
            
            forced_after = is_forced_win(opp_pW, opp_cert)
            blocked = False
            created = forced_after

            # Mark as dangerous if opponent has a strong reply
            dangerous = False
            if opp_for_them is not None and opp_cert is not None:
                # Dangerous if opponent has high utility AND high certainty
                if opp_for_them >= 0.9 and opp_cert >= 0.8:
                    dangerous = True

            adjusted_entries.append({
                "cert": cert,
                "util": util,
                "adjusted_util": risk_adjusted_util,
                "total": total,
                "move_hash": mh,
                "pW": pW,
                "pT": pT,
                "pN": pN,
                "pL": pL,
                "opponent_data_exists": opponent_data_exists,
                "adjusted": adjusted,
                "blocked": blocked,
                "created_forced_win": created,
                "opponent_best_player": opponent_best_player,
                "opponent_best_move_hash": opponent_best_move_hash,
                "opponent_best_move": opponent_best_move,
                "opponent_best_util_for_them": opp_for_them,
                "opponent_best_util_for_me": opp_for_me,
                "opponent_best_cert": opp_cert,
                "dangerous": dangerous,
            })

        # Sort by adjusted utility → total → certainty
        adjusted_entries.sort(
            key=lambda v: (v["adjusted_util"], v["total"], v["cert"]), 
            reverse=True
        )

        if not adjusted_entries:
            return None

        best_entry = adjusted_entries[0]
        best_hash = best_entry["move_hash"]

        # Debug printing
        if debug_move:
            debug_rows = []
            for entry in adjusted_entries:
                try:
                    mv = Move.get((Move.game_id == gid) & (Move.move_hash == entry["move_hash"]))
                    meta = json.loads(mv.meta_json)
                    move_arr = deserialize_array(
                        mv.move_bytes, 
                        meta["dtype"], 
                        json.dumps(meta["shape"])
                    )
                except Exception:
                    continue

                debug_rows.append({
                    "move_hash": entry["move_hash"],
                    "move_array": np.array(move_arr).tolist() 
                        if not isinstance(move_arr, list) else move_arr,
                    "certainty": float(entry["cert"]),
                    "utility": float(entry["util"]),
                    "adjusted_utility": float(entry["adjusted_util"]),
                    "total": int(entry["total"]),
                    "pW": float(entry["pW"]),
                    "pT": float(entry["pT"]),
                    "pN": float(entry["pN"]),
                    "pL": float(entry["pL"]),
                    "blocked": bool(entry["blocked"]),
                    "created_forced_win": bool(entry["created_forced_win"]),
                    "opponent_data_exists": bool(entry["opponent_data_exists"]),
                    "adjusted": bool(entry["adjusted"]),
                    "opponent_best_move": entry.get("opponent_best_move"),
                    "opponent_best_move_hash": entry.get("opponent_best_move_hash"),
                    "opponent_best_player": entry.get("opponent_best_player"),
                    "opponent_best_util_for_them": 
                        float(entry["opponent_best_util_for_them"]) 
                        if entry.get("opponent_best_util_for_them") is not None else None,
                    "opponent_best_util_for_me": 
                        float(entry["opponent_best_util_for_me"]) 
                        if entry.get("opponent_best_util_for_me") is not None else None,
                    "opponent_best_cert": 
                        float(entry["opponent_best_cert"]) 
                        if entry.get("opponent_best_cert") is not None else None,
                    "dangerous": bool(entry.get("dangerous", False)),
                    "is_best": entry["move_hash"] == best_hash,
                    "sort_key": (entry["adjusted_util"], entry["total"], entry["cert"]),
                })

            render_debug(board_arr, debug_rows)

        # Return deserialized best move
        try:
            mv = Move.get((Move.game_id == gid) & (Move.move_hash == best_hash))
            meta = json.loads(mv.meta_json)
            return deserialize_array(
                mv.move_bytes, 
                meta["dtype"], 
                json.dumps(meta["shape"])
            )
        except Exception:
            return None