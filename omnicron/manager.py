# manager.py
import os
import json
import logging
import numpy as np
from typing import cast, List, Tuple, Optional
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
# NORMALIZED TABLES (unchanged)
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
    # WRITE OBSERVATION (unchanged)
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

        board_arr = np.array(snapshot_state.board, copy=True)
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
    # scoring helper (unchanged)
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
        utility = pW * 1.0 + pT * 1.0 - pL * 1.0

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
    def _best_opponent_reply_on_board(self, gid: str, board: np.ndarray, opponents: List[int]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[int]]:
        """
        For the given board where it's some opponent's turn, find the best reply among all opponents.
        Returns: (best_pW, best_util, best_cert, best_player) or (None, None, None, None) if no data.
        """
        bh = int(hash_array(board))
        best_pW = -1.0
        best_util = -9999.0
        best_cert = -1.0
        best_player = None
        any_data = False

        for opp in opponents:
            # load rows where it's opp's to-move
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
            # evaluate this opponent's best reply
            local_best_util = -9999.0
            local_best_pW = -1.0
            local_best_cert = -1.0

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
                util = pW * 1.0 + pT * 1.0 - pL * 1.0

                if util > local_best_util:
                    local_best_util = util
                    local_best_pW = pW
                    local_best_cert = cert

            # compare to global best across opponents
            if local_best_util > best_util:
                best_util = local_best_util
                best_pW = local_best_pW
                best_cert = local_best_cert
                best_player = opp

        if not any_data:
            return None, None, None, None

        return best_pW, best_util, best_cert, best_player
    # -------------------------------------------------------
    # BEST MOVE (parameterless N-player minimax with risk-adjustment)
    # -------------------------------------------------------
    def get_best_move(self, game_id: str, game_state: GameState, debug_move=False):
        gid = str(game_id).lower().replace(" ", "_")
        snapshot_player = int(game_state.current_player)
        board_arr = np.array(game_state.board, copy=True)
        bh = int(hash_array(board_arr))

        players_on_board = self._players_on_board(gid, bh)
        opponents = [p for p in players_on_board if int(p) != snapshot_player]

        # Opponent best before move
        if opponents:
            op_before_pW, op_before_util, op_before_cert, _ = self._best_opponent_reply_on_board(gid, board_arr, opponents)
        else:
            op_before_pW = op_before_util = op_before_cert = None

        rows = list(PlayStats.select().where(
            (PlayStats.game_id == gid) &
            (PlayStats.board_hash == bh) &
            (PlayStats.snapshot_player == snapshot_player)
        ))
        if not rows:
            return None

        candidates = []
        for r in rows:
            win, tie, neu, loss = int(r.win_count), int(r.tie_count), int(r.neutral_count), int(r.loss_count)
            total = win + tie + neu + loss
            pW = win / total if total else 0.0
            pT = tie / total if total else 0.0
            pN = neu / total if total else 0.0
            pL = loss / total if total else 0.0

            if int(r.acting_player) != snapshot_player:
                pW, pL = pL, pW

            cert = max(pW, pT, pN, pL)
            util = pW*1 + pT*1 + pN*1 - pL*1
            candidates.append((cert, util, total, r.move_hash, pW, pT, pN, pL))

        adjusted_entries = []
        for cert, util, total, mh, pW, pT, pN, pL in candidates:
            next_board = None
            try:
                mv = Move.get((Move.game_id == gid) & (Move.move_hash == mh))
                meta = json.loads(mv.meta_json)
                move_arr = deserialize_array(mv.move_bytes, meta["dtype"], json.dumps(meta["shape"]))
                arr = np.array(move_arr).ravel()
                if arr.size >= 2:
                    rpos, cpos = int(arr[0]), int(arr[1])
                    next_board = np.array(board_arr, copy=True)
                    next_board[rpos, cpos] = snapshot_player
            except Exception:
                next_board = None

            op_after_util = None
            op_after_pW = op_after_cert = None
            if next_board is not None and opponents:
                op_after_pW, op_after_util, op_after_cert, _ = self._best_opponent_reply_on_board(gid, next_board, opponents)

            # Risk-adjusted utility
            alpha = 1
            risk_adjusted_util = util - alpha * op_after_util if op_after_util is not None else util

            # Forced-win diagnostics
            def is_forced_win(pW_val, cert_val):
                if pW_val is None or cert_val is None:
                    return False
                return float(cert_val) == 1.0 and float(pW_val) > 0.5

            forced_before = is_forced_win(op_before_pW, op_before_cert)
            forced_after = is_forced_win(op_after_pW, op_after_cert)

            blocked = forced_before and not forced_after
            created = (not forced_before) and forced_after
            dangerous = created  # new flag: move allows opponent forced win

            adjusted_entries.append((cert, util, risk_adjusted_util, total, mh, pW, pT, pN, pL, blocked, created, dangerous))

        # Sort by adjusted utility, then total, then certainty
        adjusted_entries.sort(key=lambda v: (v[2], v[3], v[0]), reverse=True)
        if not adjusted_entries:
            return None

        best_cert, best_util_orig, best_util_adj, best_total, best_hash, *_rest = adjusted_entries[0]

        if debug_move:
            debug_rows = []
            for cert, util, adj_util, total, mh, pW, pT, pN, pL, blocked, created, dangerous in adjusted_entries:
                try:
                    mv = Move.get((Move.game_id == gid) & (Move.move_hash == mh))
                    meta = json.loads(mv.meta_json)
                    move_arr = deserialize_array(mv.move_bytes, meta["dtype"], json.dumps(meta["shape"]))
                except Exception:
                    continue
                debug_rows.append({
                    "move_hash": mh,
                    "move_array": np.array(move_arr).tolist() if not isinstance(move_arr, list) else move_arr,
                    "certainty": float(cert),
                    "utility": float(util),
                    "adjusted_utility": float(adj_util),
                    "total": int(total),
                    "pW": float(pW),
                    "pT": float(pT),
                    "pN": float(pN),
                    "pL": float(pL),
                    "blocked": bool(blocked),
                    "created_forced_win": bool(created),
                    "dangerous": bool(dangerous),
                    "is_best": (mh == best_hash),
                    "sort_key": (adj_util, total, cert),
                })

            from .debug_viz import render_debug
            render_debug(board_arr, debug_rows)

        try:
            mv = Move.get((Move.game_id == gid) & (Move.move_hash == best_hash))
            meta = json.loads(mv.meta_json)
            return deserialize_array(mv.move_bytes, meta["dtype"], json.dumps(meta["shape"]))
        except Exception:
            return None


