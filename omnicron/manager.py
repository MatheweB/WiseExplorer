import os
import duckdb
import numpy as np

from omnicron.serializers import serialize_array, deserialize_array, hash_array
from agent.agent import State
from games.game_state import GameState


class GameMemory:
    """
    Stores game states and moves in a DuckDB table per game.

    - Each (board, move, outcome, current_player) combination has a count.
    - Non-loss moves are always preferred over loss moves.
    """

    def __init__(
        self,
        base_dir="omnicron/game_memory/games",
        db_path="omnicron/game_memory/memory.db",
    ):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.con = duckdb.connect(db_path, read_only=False)
        self.known = set()

    # ------------------------------------------------------------
    # TABLE INITIALIZATION
    # ------------------------------------------------------------
    def _ensure_table(self, game_id: str):
        table = f"plays_{game_id}"
        if table in self.known:
            return

        self.con.execute(
            f"""
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
                current_player TINYINT,
                count INTEGER
            );
            """
        )

        # Load snapshot if exists
        parquet_path = f"{self.base_dir}/{game_id}/plays.parquet"
        if os.path.exists(parquet_path):
            self.con.execute(
                f"INSERT INTO {table} SELECT * FROM read_parquet('{parquet_path}')"
            )

        self.known.add(table)

    # ------------------------------------------------------------
    # WRITE OPERATION
    # ------------------------------------------------------------
    def write(
        self, game_id: str, game_state: GameState, outcome: State, move: np.ndarray
    ):
        game_id = game_id.lower().replace(" ", "_")
        table = f"plays_{game_id}"
        self._ensure_table(game_id)

        board, player = game_state.board, game_state.current_player

        bh = int(hash_array(board))
        mh = int(hash_array(move))

        b_bytes, b_dt, b_dt2, b_shape = serialize_array(board)
        m_bytes, m_dt, m_dt2, m_shape = serialize_array(move)

        params = (bh, mh, outcome.value, player)

        # Check for existing row
        row = self.con.execute(
            f"""
            SELECT count FROM {table}
            WHERE board_hash=? AND move_hash=? 
            AND outcome=? AND current_player=?
            LIMIT 1;
            """,
            params,
        ).fetchone()

        if row:
            # Increment count
            self.con.execute(
                f"""
                UPDATE {table}
                SET count = count + 1
                WHERE board_hash=? AND move_hash=?
                AND outcome=? AND current_player=?;
                """,
                params,
            )
        else:
            # Insert new row
            self.con.execute(
                f"""
                INSERT INTO {table} (
                    board_hash, move_hash,
                    board_bytes, move_bytes,
                    board_dtype, board_dtype2, board_shape,
                    move_dtype, move_dtype2, move_shape,
                    outcome, current_player, count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
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
                    player,
                    1,
                ),
            )

        # Save Parquet snapshot
        game_dir = f"{self.base_dir}/{game_id}"
        os.makedirs(game_dir, exist_ok=True)
        pq = f"{game_dir}/plays.parquet"
        self.con.execute(f"COPY (SELECT * FROM {table}) TO '{pq}' (FORMAT PARQUET)")

    # ------------------------------------------------------------
    # READ BEST MOVE
    # ------------------------------------------------------------
    def get_best_move(
        self, game_id: str, game_state: GameState, debug_move: bool = False
    ) -> np.ndarray | None:
        """
        Returns the best move for a given game state.
        If debug_move=True, prints detailed distributions of considered moves.
        """
        game_id = game_id.lower().replace(" ", "_")
        table = f"plays_{game_id}"
        self._ensure_table(game_id)

        bh = int(hash_array(game_state.board))
        player = game_state.current_player
        loss_val = State.LOSS.value

        # Query for non-loss moves first
        rows = self.con.execute(
            f"""
            SELECT move_bytes, move_dtype, move_shape, outcome, count
            FROM {table}
            WHERE board_hash=? AND current_player=? AND outcome != ?
            ORDER BY count DESC, outcome DESC;
            """,
            (bh, player, loss_val),
        ).fetchall()

        if debug_move:
            print(f"\n--- DEBUG: Non-loss moves for player {player} ---")
            if not rows:
                print("No non-loss moves found.")
            for i, (b, dt, shape, outcome, count) in enumerate(rows):
                move = deserialize_array(b, dt, shape)
                print(f"[{i}] Move: {move}, Outcome: {State(outcome).name}, Count: {count}")

        if rows:
            # Pick the first (highest count / best outcome)
            selected = rows[0]
            move = deserialize_array(selected[0], selected[1], selected[2])
            if debug_move:
                print(f"-> Selected move: {move} (Reason: highest count / non-loss)")
            return move

        # Fallback to loss-only moves
        rows = self.con.execute(
            f"""
            SELECT move_bytes, move_dtype, move_shape, outcome, count
            FROM {table}
            WHERE board_hash=? AND current_player=? AND outcome = ?
            ORDER BY count DESC;
            """,
            (bh, player, loss_val),
        ).fetchall()

        if debug_move:
            print(f"\n--- DEBUG: Loss moves for player {player} ---")
            if not rows:
                print("No moves found at all.")
            for i, (b, dt, shape, outcome, count) in enumerate(rows):
                move = deserialize_array(b, dt, shape)
                print(f"[{i}] Move: {move}, Outcome: {State(outcome).name}, Count: {count}")

        if rows:
            selected = rows[0]
            move = deserialize_array(selected[0], selected[1], selected[2])
            if debug_move:
                print(f"-> Selected move: {move} (Reason: fallback to loss moves)")
            return move

        return None
