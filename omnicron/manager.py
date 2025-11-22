# manager.py
import os
import duckdb
import numpy as np

from serializers import serialize_array, deserialize_array, hash_array

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
    def _ensure_table(self, game):
        table = f"plays_{game}"
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
            count INTEGER
        );
        """
        self.con.execute(sql)

        # If parquet exists, load it once
        parquet_path = f"{self.base_dir}/{game}/plays.parquet"
        if os.path.exists(parquet_path):
            self.con.execute(
                f"INSERT INTO {table} SELECT * FROM read_parquet('{parquet_path}')"
            )

        self.known.add(table)

    # --------------------------
    # PUBLIC API: WRITE
    # --------------------------
    def write(self, game, outcome, board: np.ndarray, move: np.ndarray):
        """
        Stores (board, move, outcome) and increments count if duplicate exists.
        Very readable, simple, synchronous.
        """
        game = game.lower().replace(" ", "_")
        self._ensure_table(game)
        table = f"plays_{game}"

        # serialize
        bh = hash_array(board)
        mh = hash_array(move)
        (b_bytes, b_dt, b_dt2, b_shape) = serialize_array(board)
        (m_bytes, m_dt, m_dt2, m_shape) = serialize_array(move)

        # Try update count
        update = f"""
            UPDATE {table}
            SET count = count + 1
            WHERE board_hash={bh} AND move_hash={mh} AND outcome={outcome};
        """
        self.con.execute(update)

        # Check if row existed
        check = f"""
            SELECT 1 FROM {table}
            WHERE board_hash={bh} AND move_hash={mh} AND outcome={outcome}
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
                outcome, count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
                    outcome,
                    1,
                ),
            )

        # always persist simple parquet snapshot (small + simple)
        game_dir = f"{self.base_dir}/{game}"
        os.makedirs(game_dir, exist_ok=True)
        pq = f"{game_dir}/plays.parquet"
        self.con.execute(f"COPY (SELECT * FROM {table}) TO '{pq}' (FORMAT PARQUET)")

    # --------------------------
    # PUBLIC API: READ BEST MOVE
    # --------------------------
    def get_best_move(self, game, board: np.ndarray):
        """
        Returns best move for a given board, or None.
        Certainty scoring = count*W + outcome_score.
        """
        game = game.lower().replace(" ", "_")
        self._ensure_table(game)
        table = f"plays_{game}"

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
        ORDER BY score DESC
        LIMIT 1;
        """

        row = self.con.execute(sql).fetchone()
        if not row:
            return None

        move_bytes, move_dt2, move_shape = row[0], row[1], row[2]
        return deserialize_array(move_bytes, move_dt2, move_shape)
