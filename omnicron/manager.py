# manager.py
import os
import json
import logging
import numpy as np
from typing import cast
from peewee import (
    Model, IntegerField, BlobField, TextField,
    SqliteDatabase, CompositeKey, DoesNotExist
)

from omnicron.serializers import serialize_array, deserialize_array, hash_array
from agent.agent import State
from games.game_state import GameState

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

db = SqliteDatabase(None)


# ===========================================================
#              TERMINAL HEATMAP VISUALIZATION
# ===========================================================

def _terminal_heatmap_tictactoe(board, debug_rows):
    """
    Renders a terminal-friendly tic-tac-toe heatmap overlay.
    Shows:
        - Board marks (X/O/·)
        - Certainty heat (ANSI colored)
        - Certainty + Utility values per cell
        - Move summary with BEST highlighted
    """

    import numpy as _np

    reset = "\033[0m"

    # box drawing
    TL, TM, TR = "┌", "┬", "┐"
    ML, MM, MR = "├", "┼", "┤"
    BL, BM, BR = "└", "┴", "┘"
    H, V = "─", "│"

    # prepare 3×3 matrices
    heat_cert = _np.full((3, 3), _np.nan)
    heat_util = _np.full((3, 3), _np.nan)

    # fill matrices
    for d in debug_rows:
        r, c = map(int, d["move_array"])
        heat_cert[r, c] = d["certainty"]
        heat_util[r, c] = d["utility"]

    def mark(v):
        return "X" if v == 1 else ("O" if v == -1 else "·")

    def color_for(x):
        if x is None or _np.isnan(x):
            return "\033[48;5;236m"  # dark gray
        v = max(0.0, min(1.0, float(x)))
        # gradient 22 (green) → 220 (yellow) → 196 (red)
        if v < 0.5:
            ratio = v / 0.5
            idx = int(22 + ratio * (220 - 22))
        else:
            ratio = (v - 0.5) / 0.5
            idx = int(220 + ratio * (196 - 220))
        return f"\033[48;5;{idx}m"

    # -------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------
    print("\n=== MOVE SUMMARY ===")
    sorted_rows = sorted(debug_rows, key=lambda d: d["sort_key"], reverse=True)
    for d in sorted_rows:
        star = "★ BEST" if d["is_best"] else "  "
        print(
            f"{star} move={d['move_array']}  "
            f"certainty={d['certainty']:.3f}  "
            f"utility={d['utility']:.3f}  total={d['total']}"
        )

    print("\n=== HEATMAP GRID ===\n")

    # -------------------------------------------------------
    # GRID RENDER
    # -------------------------------------------------------
    for r in range(3):
        if r == 0:
            print(TL + H*7 + TM + H*7 + TM + H*7 + TR)

        line = V
        for c in range(3):
            bg = color_for(heat_cert[r, c])

            cstr = (
                f" {mark(board[r,c])}"
                f" c:{heat_cert[r,c]:.2f}" if not _np.isnan(heat_cert[r,c]) else " c:-- "
            )
            ostr = (
                f" u:{heat_util[r,c]:.2f}" if not _np.isnan(heat_util[r,c]) else " u:-- "
            )
            text = (cstr + ostr)[:7].ljust(7)

            line += f"{bg}{text}{reset}{V}"

        print(line)

        if r < 2:
            print(ML + H*7 + MM + H*7 + MM + H*7 + MR)

    print(BL + H*7 + BM + H*7 + BM + H*7 + BR)
    print()


# ===========================================================
#                    NORMALIZED TABLES
# ===========================================================

class Position(Model):
    """One row per unique board."""
    game_id = TextField()
    board_hash = IntegerField(primary_key=True)

    board_bytes = BlobField()
    meta_json = TextField()  # {"dtype": "...", "shape": [...]}

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
            "game_id",
            "board_hash",
            "move_hash",
            "snapshot_player",
            "acting_player"
        )


# ===========================================================
#                     GAME MEMORY
# ===========================================================

class GameMemory:
    """Relational + JSON metadata version."""

    def __init__(self, base_dir="omnicron/game_memory/games",
                 db_path="omnicron/game_memory/memory.db",
                 flush_every: int = 5000):

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
    # WRITE OBSERVATION
    # -------------------------------------------------------

    def write(self,
              game_id: str,
              snapshot_state: GameState,
              acting_player: int,
              outcome: State,
              move: np.ndarray):

        if snapshot_state is None:
            raise ValueError("snapshot_state must not be None")

        gid = str(game_id).lower().replace(" ", "_")

        board_arr = np.array(snapshot_state.board, copy=True)
        move_arr = np.array(move, copy=True)
        snapshot_player = int(snapshot_state.current_player)

        bh = int(hash_array(board_arr))
        mh = int(hash_array(move_arr))

        # ---------------------------------------------------
        # Insert unique Position
        # ---------------------------------------------------
        try:
            Position.get((Position.game_id == gid) & (Position.board_hash == bh))
        except DoesNotExist:
            b_bytes, dtype, dtype2, shape_str = serialize_array(board_arr)
            meta = {"dtype": dtype, "dtype2": dtype2, "shape": json.loads(shape_str)}
            Position.create(
                game_id=gid,
                board_hash=bh,
                board_bytes=b_bytes,
                meta_json=json.dumps(meta),
            )

        # ---------------------------------------------------
        # Insert unique Move
        # ---------------------------------------------------
        try:
            Move.get((Move.game_id == gid) & (Move.move_hash == mh))
        except DoesNotExist:
            m_bytes, dtype, dtype2, shape_str = serialize_array(move_arr)
            meta = {"dtype": dtype, "dtype2": dtype2, "shape": json.loads(shape_str)}
            Move.create(
                game_id=gid,
                move_hash=mh,
                move_bytes=m_bytes,
                meta_json=json.dumps(meta),
            )

        # ---------------------------------------------------
        # Update stats
        # ---------------------------------------------------
        try:
            row = PlayStats.get(
                (PlayStats.game_id == gid) &
                (PlayStats.board_hash == bh) &
                (PlayStats.move_hash == mh) &
                (PlayStats.snapshot_player == snapshot_player) &
                (PlayStats.acting_player == acting_player)
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

        # DB snapshot
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
    # SCORING
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
        utility = pW * 1.0 + pT * 0.2 - pL * 1.0

        return certainty, utility, total

    # -------------------------------------------------------
    # BEST MOVE
    # -------------------------------------------------------

    def get_best_move(self, game_id: str, game_state: GameState, debug_move=False):

        gid = str(game_id).lower().replace(" ", "_")
        snapshot_player = int(game_state.current_player)

        board_arr = np.array(game_state.board, copy=True)
        bh = int(hash_array(board_arr))

        rows = list(
            PlayStats.select().where(
                (PlayStats.game_id == gid) &
                (PlayStats.board_hash == bh) &
                (PlayStats.snapshot_player == snapshot_player) &
                (PlayStats.acting_player == snapshot_player)
            )
        )
        if not rows:
            return None

        # ---------------------------------------------
        # Compute scoring
        # ---------------------------------------------
        scored = []
        for r in rows:
            cert, util, total = self._score(r)
            scored.append((cert, util, total, r.move_hash, r))

        # sort by certainty → utility → total
        scored.sort(key=lambda v: (v[0], v[1], v[2]), reverse=True)
        best_cert, best_util, best_total, best_hash, _ = scored[0]

        # ---------------------------------------------
        # Debug info aggregation if requested
        # ---------------------------------------------
        debug_rows = []
        if debug_move:
            for rank, (cert, util, total, mh, row) in enumerate(scored):
                mv = Move.get((Move.game_id == gid) & (Move.move_hash == mh))
                meta = json.loads(mv.meta_json)
                move_arr = deserialize_array(mv.move_bytes, meta["dtype"], json.dumps(meta["shape"]))

                debug_rows.append({
                    "move_hash": mh,
                    "move_array": move_arr,
                    "certainty": cert,
                    "utility": util,
                    "total": total,
                    "is_best": (mh == best_hash),
                    "sort_key": (cert, util, total),
                })

            # call visualization
            try:
                _terminal_heatmap_tictactoe(board_arr, debug_rows)
            except Exception as exc:
                logger.exception("Heatmap render failed: %s", exc)

        # ---------------------------------------------
        # Return best move
        # ---------------------------------------------
        try:
            mv = Move.get(
                (Move.game_id == gid) &
                (Move.move_hash == best_hash)
            )
            meta = json.loads(mv.meta_json)
            return deserialize_array(mv.move_bytes, meta["dtype"], json.dumps(meta["shape"]))
        except Exception:
            logger.exception("Failed to deserialize move for %s", best_hash)
            return None
