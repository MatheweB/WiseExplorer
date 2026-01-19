# mini_chess.py
from copy import copy
from typing import List, Tuple
import numpy as np

# Functions
from games.game_rules import in_bounds

# Classes
from games.game_base import GameBase
from games.game_state import GameState
from agent.agent import State


class MiniChess(GameBase):
    """
    Mini Chess implementation on a 6x4 board.

    Starting position:
        Row 0 (Player 1): C1 K1 Q1 C1  (Castle, King, Queen, Castle)
        Row 1 (Player 1): P1 P1 P1 P1  (Pawns)
        Row 2-3: Empty
        Row 4 (Player 2): P2 P2 P2 P2  (Pawns)
        Row 5 (Player 2): C2 K2 Q2 C2  (Castle, King, Queen, Castle)

    Simplified rules:
        - Pawns move forward 1 square, capture diagonally forward
        - Castles (Rooks) move orthogonally any distance
        - King moves 1 square in any direction
        - Queen moves any distance in any direction
        - Win by capturing opponent's King
        - No castling, en passant, or pawn promotion
        - Draw after 100 moves

    ARCHITECTURE NOTE:
    -----------------
    This class exposes MOVES via valid_moves() and apply_move().
    Transitions are DERIVED externally by the memory system through:
        1. game.deep_clone()
        2. clone.apply_move(move)
        3. clone.get_state()

    Do NOT add valid_transitions() - it violates the architecture.
    """

    ROWS = 6
    COLS = 4

    def __init__(self):
        """Initialize a fresh game with starting position."""
        board = self._create_initial_board()
        self.state = GameState(board, current_player=1)
        self.winner = None
        self.move_count = 0
        self.max_moves = 100  # Draw after 100 moves

    def _create_initial_board(self) -> np.ndarray:
        """Create the starting board configuration."""
        board = np.full((self.ROWS, self.COLS), None, dtype=object)

        # Player 1 pieces (top)
        board[0] = ["C1", "K1", "Q1", "C1"]
        board[1] = ["P1", "P1", "P1", "P1"]

        # Player 2 pieces (bottom)
        board[4] = ["P2", "P2", "P2", "P2"]
        board[5] = ["C2", "K2", "Q2", "C2"]

        return board

    # --------------------------------------------------------------
    # Required interface methods
    # --------------------------------------------------------------
    def game_id(self) -> str:
        return "mini_chess"

    def num_players(self) -> int:
        return 2

    def clone(self) -> "MiniChess":
        return copy(self)

    def deep_clone(self) -> "MiniChess":
        clone = copy(self)
        clone.state = self.state.copy()
        return clone

    def get_state(self) -> GameState:
        return self.state

    def set_state(self, game_state: GameState) -> None:
        self.state = game_state

    def current_player(self) -> int:
        return self.state.current_player

    def valid_moves(self) -> List[np.ndarray]:
        """
        Return all legal MOVES as [(from_r, from_c, to_r, to_c)].

        Format: Each move is [from_row, from_col, to_row, to_col]

        To derive transitions from these moves:
            for move in game.valid_moves():
                clone = game.deep_clone()
                clone.apply_move(move)
                next_state = clone.get_state()
                # Now you have the transition: current_state → next_state
        """
        board = self.state.board
        player = self.current_player()
        moves = []

        for r in range(self.ROWS):
            for c in range(self.COLS):
                piece = board[r, c]
                if piece:
                    piece_player = self._get_piece_player(piece)
                    if piece_player == player:
                        piece_moves = self._get_piece_moves(r, c, piece)
                        moves.extend(piece_moves)

        if not moves:
            return np.array([]).reshape(0, 4)

        return np.array(moves, dtype=np.int32)

    # --------------------------------------------------------------
    # Core move logic
    # --------------------------------------------------------------
    def apply_move(self, move: np.ndarray) -> None:
        """
        Apply a move: [from_r, from_c, to_r, to_c].

        This is the game engine's responsibility.
        Memory systems call this via deep_clone() to derive transitions.
        """
        if len(move) != 4:
            raise ValueError(
                f"Invalid move format: {move}. Expected [from_r, from_c, to_r, to_c]"
            )

        from_r, from_c, to_r, to_c = map(int, move)

        # Validate bounds
        if not (
            in_bounds(self.state.board, from_r, from_c)
            and in_bounds(self.state.board, to_r, to_c)
        ):
            raise ValueError(f"Move {move} is out of bounds.")

        piece = self.state.board[from_r, from_c]
        target = self.state.board[to_r, to_c]

        # Validate piece exists
        if not piece:
            raise ValueError(
                f"No piece at ({from_r},{from_c}).\n"
                f"Board state:\n{self.state_string()}\n"
                f"Attempted move: {move}"
            )

        # Validate piece ownership
        piece_owner = self._get_piece_player(piece)
        current = self.current_player()

        if piece_owner != current:
            raise ValueError(
                f"Piece ownership error:\n"
                f"  Move: {move}\n"
                f"  Current player: {current}\n"
                f"  Piece at ({from_r},{from_c}): {piece}\n"
                f"  Piece owner: {piece_owner}\n"
                f"This usually means you're applying a move to the wrong game state."
            )

        # Validate move legality
        if not self._is_legal_move(from_r, from_c, to_r, to_c, piece):
            valid = self.valid_moves()
            raise ValueError(
                f"Illegal move: {move}\n"
                f"Piece {piece} at ({from_r},{from_c}) cannot move to ({to_r},{to_c})\n"
                f"Valid moves for this piece: {[m for m in valid if m[0] == from_r and m[1] == from_c]}"
            )

        # Check if capturing opponent's King
        if target and target[0] == "K" and self._get_piece_player(target) != current:
            self.winner = current

        # Execute move
        self.state.board[to_r, to_c] = piece
        self.state.board[from_r, from_c] = None

        self.move_count += 1
        self._increment_to_next_turn()

    def is_over(self) -> bool:
        """Game ends when a King is captured or move limit reached."""
        if self.winner is not None:
            return True
        if self.move_count >= self.max_moves:
            return True
        # Check if either King is missing
        return not self._king_exists(1) or not self._king_exists(2)

    def get_result(self, agent_id: int) -> State:
        """Return LOSS, NEUTRAL, TIE, or WIN."""
        if self.winner is not None:
            return State.WIN if self.winner == agent_id else State.LOSS

        if self.move_count >= self.max_moves:
            return State.TIE

        # Check King capture
        if not self._king_exists(agent_id):
            return State.LOSS

        opponent = 3 - agent_id
        if not self._king_exists(opponent):
            return State.WIN

        return State.NEUTRAL

    def state_string(self) -> str:
        """Pretty-print the board."""
        board = self.state.board

        lines = ["╭────┬────┬────┬────╮"]
        for i in range(self.ROWS):
            row_strs = []
            for j in range(self.COLS):
                piece = board[i, j]
                cell = piece if piece else "  "
                row_strs.append(f"{cell:^2}")
            lines.append("│ " + " │ ".join(row_strs) + " │")
            if i < self.ROWS - 1:
                lines.append("├────┼────┼────┼────┤")
        lines.append("╰────┴────┴────┴────╯")

        result = "\n".join(lines)
        result += f"\n\nCurrent player: {self.state.current_player}"
        result += f"\nMove count: {self.move_count}"
        return result

    # --------------------------------------------------------------
    # Internal helper methods
    # --------------------------------------------------------------
    def _get_piece_player(self, piece: str) -> int:
        """Extract player number from piece string (e.g., 'K1' → 1)."""
        return int(piece[1])

    def _king_exists(self, player: int) -> bool:
        """Check if the specified player's King is on the board."""
        king = f"K{player}"
        return bool(np.any(self.state.board == king))

    def _get_piece_moves(
        self, r: int, c: int, piece: str
    ) -> List[Tuple[int, int, int, int]]:
        """Get all legal moves for a piece at position (r, c)."""
        piece_type = piece[0]

        if piece_type == "P":
            return self._get_pawn_moves(r, c, piece)
        elif piece_type == "C":
            return self._get_castle_moves(r, c, piece)
        elif piece_type == "K":
            return self._get_king_moves(r, c, piece)
        elif piece_type == "Q":
            return self._get_queen_moves(r, c, piece)

        return []

    def _get_pawn_moves(
        self, r: int, c: int, piece: str
    ) -> List[Tuple[int, int, int, int]]:
        """
        Get pawn moves (forward 1, capture diagonally).
        Player 1 pawns move down (row increases), Player 2 pawns move up (row decreases).
        """
        player = self._get_piece_player(piece)
        # Player 1 (top) moves down (+1), Player 2 (bottom) moves up (-1)
        direction = 1 if player == 1 else -1
        moves = []

        # Move forward (only if empty)
        new_r = r + direction
        if in_bounds(self.state.board, new_r, c):
            if self.state.board[new_r, c] is None:
                moves.append((r, c, new_r, c))

            # Capture diagonally (only if enemy piece present)
            for dc in [-1, 1]:
                new_c = c + dc
                if in_bounds(self.state.board, new_r, new_c):
                    target = self.state.board[new_r, new_c]
                    if target and self._get_piece_player(target) != player:
                        moves.append((r, c, new_r, new_c))

        return moves

    def _get_castle_moves(
        self, r: int, c: int, piece: str
    ) -> List[Tuple[int, int, int, int]]:
        """Get castle (rook) moves (orthogonal lines)."""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        return self._get_line_moves(r, c, piece, directions)

    def _get_king_moves(
        self, r: int, c: int, piece: str
    ) -> List[Tuple[int, int, int, int]]:
        """Get king moves (1 square in any direction)."""
        player = self._get_piece_player(piece)
        moves = []

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_r, new_c = r + dr, c + dc
                if in_bounds(self.state.board, new_r, new_c):
                    target = self.state.board[new_r, new_c]
                    if target is None or self._get_piece_player(target) != player:
                        moves.append((r, c, new_r, new_c))

        return moves

    def _get_queen_moves(
        self, r: int, c: int, piece: str
    ) -> List[Tuple[int, int, int, int]]:
        """Get queen moves (all directions)."""
        directions = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),  # Orthogonal
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),  # Diagonal
        ]
        return self._get_line_moves(r, c, piece, directions)

    def _get_line_moves(
        self, r: int, c: int, piece: str, directions: List[Tuple[int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Get moves along specified directions until blocked."""
        player = self._get_piece_player(piece)
        moves = []

        for dr, dc in directions:
            new_r, new_c = r + dr, c + dc
            while in_bounds(self.state.board, new_r, new_c):
                target = self.state.board[new_r, new_c]

                if target is None:
                    # Empty square - can move here and continue
                    moves.append((r, c, new_r, new_c))
                    new_r += dr
                    new_c += dc
                elif self._get_piece_player(target) != player:
                    # Enemy piece - can capture, but stops here
                    moves.append((r, c, new_r, new_c))
                    break
                else:
                    # Own piece - blocked
                    break

        return moves

    def _is_legal_move(
        self, from_r: int, from_c: int, to_r: int, to_c: int, piece: str
    ) -> bool:
        """Check if a move is legal by comparing against valid moves."""
        legal_moves = self._get_piece_moves(from_r, from_c, piece)
        return any(m[2] == to_r and m[3] == to_c for m in legal_moves)

    def _increment_to_next_turn(self) -> None:
        """Switch to the next player."""
        self.state.current_player = 2 if self.current_player() == 1 else 1
