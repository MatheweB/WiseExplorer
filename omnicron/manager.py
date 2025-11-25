# manager.py
import os
import duckdb
import numpy as np

from serializers import serialize_array, deserialize_array, hash_array

# Classes
from agent.agent import State
from games.game_state import GameState

# outcome scoring
OUTCOME_SCORE = {2: 3, 1: 2, 0: 1, -1: 0}
CERTAINTY_WEIGHT = 100


class GameMemory:
    def __init__(
        self,
        base_dir="omnicron/game_memory/games",
        db_path="omnicron/game_memory/memory.db",
    ):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.con = duckdb.connect(db_path, read_only=False)
        self.known = set()

    # --------------------------
    # table creation
    # --------------------------
    def _ensure_table(self, game_id: str):
        table = f"plays_{game_id}"
        if table in self.known:
            return

        sql = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            board_hash BIGINT,
            move_hash BIGINT,
            board_bytes BLOB,
            move_bytes BLOB,
            board_dtype TEXT,
            board_dtype2 TEXT,
            board_shape TEXT,
            move_dtype TEXT,
            move_dtype2 TEXT,
            move_shape TEXT,
            outcome TINYINT,
            current_player TINYINT,     -- NEW COLUMN
            count INTEGER
        );
        """
        self.con.execute(sql)

        # If parquet exists, load it once
        parquet_path = f"{self.base_dir}/{game_id}/plays.parquet"
        if os.path.exists(parquet_path):
            self.con.execute(
                f"INSERT INTO {table} SELECT * FROM read_parquet('{parquet_path}')"
            )

        self.known.add(table)


    # --------------------------
    # PUBLIC API: WRITE
    # --------------------------
    def write(self, game_id: str, game_state: GameState, outcome: State, move: np.ndarray):
        """
        Stores (board, move, outcome) and increments count if duplicate exists.
        Very readable, simple, synchronous.
        """
        board = game_state.board
        current_player = game_state.current_player
        game_id = game_id.lower().replace(" ", "_")

        self._ensure_table(game_id)
        table = f"plays_{game_id}"

        # serialize
        bh = hash_array(board)
        mh = hash_array(move)
        (b_bytes, b_dt, b_dt2, b_shape) = serialize_array(board)
        (m_bytes, m_dt, m_dt2, m_shape) = serialize_array(move)

        # Try update count
        update = f"""
            UPDATE {table}
            SET count = count + 1
            WHERE board_hash={bh}
            AND move_hash={mh}
            AND outcome={outcome.value}
            AND current_player={current_player};
        """
        self.con.execute(update)

        # Check if row existed
        check = f"""
            SELECT 1 FROM {table}
            WHERE board_hash={bh} AND move_hash={mh} AND outcome={outcome.value}
            LIMIT 1;
        """
        exists = self.con.execute(check).fetchone()

        if not exists:
            insert = f"""
            INSERT INTO {table} (
                board_hash, move_hash,
                board_bytes, move_bytes,
                board_dtype, board_dtype2, board_shape,
                move_dtype, move_dtype2, move_shape,
                outcome, current_player, count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            self.con.execute(
                insert,
                (
                    bh,
                    mh,
                    b_bytes,
                    m_bytes,
                    b_dt,
                    b_dt2,
                    b_shape,
                    m_dt,
                    m_dt2,
                    m_shape,
                    outcome.value,
                    current_player,   # NEW
                    1,
                ),
            )

        # always persist simple parquet snapshot (small + simple)
        game_dir = f"{self.base_dir}/{game_id}"
        os.makedirs(game_dir, exist_ok=True)
        pq = f"{game_dir}/plays.parquet"
        self.con.execute(f"COPY (SELECT * FROM {table}) TO '{pq}' (FORMAT PARQUET)")

    # --------------------------
    # PUBLIC API: READ BEST MOVE
    # --------------------------
    def get_best_move(self, game_id: str, game_state: GameState) -> np.ndarray | None:
        """
        Returns best move for a given board, or None.
        Certainty scoring = count*W + outcome_score.
        """
        board = game_state.board
        current_player = game_state.current_player
        game_id = game_id.lower().replace(" ", "_")

        self._ensure_table(game_id)
        table = f"plays_{game_id}"

        bh = hash_array(board)

        sql = f"""
        SELECT
            move_bytes,
            move_dtype2,
            move_shape,
            (count * {CERTAINTY_WEIGHT} +
                CASE outcome
                    WHEN 2 THEN 3
                    WHEN 1 THEN 2
                    WHEN 0 THEN 1
                    WHEN -1 THEN 0
                END
            ) AS score
        FROM {table}
        WHERE board_hash = {bh}
        AND current_player = {current_player}   -- NEW FILTER
        ORDER BY score DESC
        LIMIT 1;
        """

        row = self.con.execute(sql).fetchone()
        if not row:
            return None

        move_bytes, move_dt2, move_shape = row[0], row[1], row[2]
        return deserialize_array(move_bytes, move_dt2, move_shape)
