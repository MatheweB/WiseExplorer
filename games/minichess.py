# mini_chess.py
from copy import copy, deepcopy
import numpy as np
from typing import List, Tuple, Optional

# Functions
from games.game_rules import in_bounds, board_full

# Classes
from games.game_base import GameBase
from games.game_state import GameState
from agent.agent import State


class MiniChess(GameBase):
    """
    Mini Chess implementation on a 4x4 board.
    Starting position:
    Row 0 (Player 1): C1 K1 Q1 C1  (Castle, King, Queen, Castle)
    Row 1 (Player 1): P1 P1 P1 P1  (Pawns)
    Row 2: Empty
    Row 3: Empty
    Row 4 (Player 2): P2 P2 P2 P2  (Pawns)
    Row 5 (Player 2): C2 K2 Q2 C2  (Castle, King, Queen, Castle)
    
    Simplified rules:
    - Pawns move forward 1 square, capture diagonally forward
    - Castles (Rooks) move orthogonally any distance
    - King moves 1 square in any direction
    - Queen moves any distance in any direction
    - Win by capturing opponent's King
    - No castling, en passant, or pawn promotion
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
        board[0] = ['C1', 'K1', 'Q1', 'C1']
        board[1] = ['P1', 'P1', 'P1', 'P1']
        
        # Player 2 pieces (bottom)
        board[4] = ['P2', 'P2', 'P2', 'P2']
        board[5] = ['C2', 'K2', 'Q2', 'C2']
        
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
        return deepcopy(self)

    def get_state(self) -> GameState:
        return self.state

    def set_state(self, game_state: GameState) -> None:
        self.state = game_state

    def current_player(self) -> int:
        return self.state.current_player

    def valid_moves(self) -> np.ndarray:
        """Return all legal moves as [(from_r, from_c, to_r, to_c)]."""
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
        
        result = np.array(moves, dtype=np.int32)
        return result

    # --------------------------------------------------------------
    # Core move logic
    # --------------------------------------------------------------
    def apply_move(self, move: np.ndarray) -> None:
        """Apply a move: [from_r, from_c, to_r, to_c]."""
        if len(move) != 4:
            raise ValueError(f"Invalid move format: {move}")

        from_r, from_c, to_r, to_c = map(int, move)

        # Validate bounds
        if not (in_bounds(self.state.board, from_r, from_c) and 
                in_bounds(self.state.board, to_r, to_c)):
            raise ValueError(f"Move {move} is out of bounds.")

        piece = self.state.board[from_r, from_c]
        target = self.state.board[to_r, to_c]

        # Debug info
        if not piece:
            raise ValueError(f"No piece at {(from_r, from_c)}. Board state:\n{self.state_string()}\nAttempted move: {move}")

        # Validate piece ownership
        if self._get_piece_player(piece) != self.current_player():
            # This is the error - log everything
            print(f"\n=== PIECE OWNERSHIP ERROR ===")
            print(f"Attempted move: {move}")
            print(f"Current player: {self.current_player()}")
            print(f"Piece at ({from_r},{from_c}): {piece}")
            print(f"Piece belongs to player: {self._get_piece_player(piece)}")
            print(f"\nCurrent board state:")
            print(self.state_string())
            print(f"\nValid moves for current player {self.current_player()}:")
            valid = self.valid_moves()
            for i, m in enumerate(valid):
                if i < 5:  # Show first 5
                    print(f"  {m}")
            print(f"  ... ({len(valid)} total moves)")
            raise ValueError(f"Invalid piece at {(from_r, from_c)}: {piece} belongs to player {self._get_piece_player(piece)}, current player is {self.current_player()}")

        # Validate move legality
        if not self._is_legal_move(from_r, from_c, to_r, to_c, piece):
            raise ValueError(f"Illegal move: {move}")

        # Check if capturing opponent's King
        if target and target[0] == 'K' and self._get_piece_player(target) != self.current_player():
            self.winner = self.current_player()

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
        """Extract player number from piece string."""
        return int(piece[1])

    def _king_exists(self, player: int) -> bool:
        """Check if the specified player's King is on the board."""
        king = f'K{player}'
        return np.any(self.state.board == king)

    def _get_piece_moves(self, r: int, c: int, piece: str) -> List[Tuple[int, int, int, int]]:
        """Get all legal moves for a piece at position (r, c)."""
        piece_type = piece[0]
        moves = []

        if piece_type == 'P':
            moves = self._get_pawn_moves(r, c, piece)
        elif piece_type == 'C':
            moves = self._get_castle_moves(r, c, piece)
        elif piece_type == 'K':
            moves = self._get_king_moves(r, c, piece)
        elif piece_type == 'Q':
            moves = self._get_queen_moves(r, c, piece)

        return moves

    def _get_pawn_moves(self, r: int, c: int, piece: str) -> List[Tuple[int, int, int, int]]:
        """Get pawn moves (forward 1, capture diagonally).
        Player 1 pawns move down (row increases), Player 2 pawns move up (row decreases)."""
        player = self._get_piece_player(piece)
        # Player 1 (top) moves down (+1), Player 2 (bottom) moves up (-1)
        direction = 1 if player == 1 else -1
        moves = []

        # Move forward
        new_r = r + direction
        if in_bounds(self.state.board, new_r, c):
            if self.state.board[new_r, c] is None:
                moves.append((r, c, new_r, c))

            # Capture diagonally (only if there's an enemy piece)
            for dc in [-1, 1]:
                new_c = c + dc
                if in_bounds(self.state.board, new_r, new_c):
                    target = self.state.board[new_r, new_c]
                    if target and self._get_piece_player(target) != player:
                        moves.append((r, c, new_r, new_c))

        return moves

    def _get_castle_moves(self, r: int, c: int, piece: str) -> List[Tuple[int, int, int, int]]:
        """Get castle (rook) moves (orthogonal lines)."""
        return self._get_line_moves(r, c, piece, [(0, 1), (0, -1), (1, 0), (-1, 0)])

    def _get_king_moves(self, r: int, c: int, piece: str) -> List[Tuple[int, int, int, int]]:
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

    def _get_queen_moves(self, r: int, c: int, piece: str) -> List[Tuple[int, int, int, int]]:
        """Get queen moves (all directions)."""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        return self._get_line_moves(r, c, piece, directions)

    def _get_line_moves(self, r: int, c: int, piece: str, directions: List[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
        """Get moves along specified directions until blocked."""
        player = self._get_piece_player(piece)
        moves = []

        for dr, dc in directions:
            new_r, new_c = r + dr, c + dc
            while in_bounds(self.state.board, new_r, new_c):
                target = self.state.board[new_r, new_c]
                
                if target is None:
                    moves.append((r, c, new_r, new_c))
                    new_r += dr
                    new_c += dc
                elif self._get_piece_player(target) != player:
                    # Can capture enemy piece
                    moves.append((r, c, new_r, new_c))
                    break
                else:
                    # Blocked by own piece
                    break

        return moves

    def _is_legal_move(self, from_r: int, from_c: int, to_r: int, to_c: int, piece: str) -> bool:
        """Check if a move is legal."""
        legal_moves = self._get_piece_moves(from_r, from_c, piece)
        return any(m[2] == to_r and m[3] == to_c for m in legal_moves)

    def _increment_to_next_turn(self) -> None:
        """Switch to the next player."""
        self.state.current_player = 2 if self.current_player() == 1 else 1