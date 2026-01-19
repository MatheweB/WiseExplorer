# tic_tac_toe.py
from copy import copy
from typing import List
import numpy as np

# Functions
from games.game_rules import get_rows, get_cols, get_diagonals, board_full

# Classes
from games.game_base import GameBase
from games.game_state import GameState
from agent.agent import State


class TicTacToe(GameBase):
    """
    Fully implemented TicTacToe game.
    Designed for multi-agent simulations.

    ARCHITECTURE NOTE:
    -----------------
    This class exposes MOVES via valid_moves() and apply_move().
    Transitions are DERIVED externally by:
        1. Calling deep_clone()
        2. Calling apply_move() on the clone
        3. Extracting the new state via get_state()

    Do NOT add valid_transitions() here - it violates the architecture.
    """

    SIZE = 3  # 3x3 board

    def __init__(self):
        """Initialize a fresh game."""
        empty = np.full((self.SIZE, self.SIZE), None, dtype=object)
        self.state = GameState(empty, current_player=1)
        self.winner = None  # 1 or 2, or None

    # --------------------------------------------------------------
    # Required interface methods
    # --------------------------------------------------------------
    def game_id(self) -> str:
        return "tic_tac_toe"

    def num_players(self) -> int:
        return 2

    def clone(self) -> "TicTacToe":
        return copy(self)

    def deep_clone(self) -> "TicTacToe":
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
        Return all legal MOVES from the current state.

        Returns array of (row, col) positions.

        To derive transitions from these moves:
            for move in game.valid_moves():
                clone = game.deep_clone()
                clone.apply_move(move)
                next_state = clone.get_state()
                # Now you have the transition: current_state → next_state
        """
        board = self.state.board
        mask = np.array([[x is None for x in row] for row in board])
        return np.argwhere(mask)

    # --------------------------------------------------------------
    # Core move logic
    # --------------------------------------------------------------
    def apply_move(self, move: np.ndarray) -> None:
        """
        Apply a move for the current player and update state.

        This is the game engine's responsibility.
        Memory systems call this via deep_clone() to derive transitions.
        """

        if len(move) != 2:
            raise ValueError(f"Invalid move format: {move}")

        r, c = move

        # Validate bounds
        if not (0 <= r < self.SIZE and 0 <= c < self.SIZE):
            raise ValueError(f"Move {move} is out of bounds.")

        # Validate empty cell
        if self.state.board[r, c] is not None:
            raise ValueError(f"Cell {move} is already occupied.")

        player = self.state.current_player

        # Place the mark
        self.state.board[r, c] = player

        # Check winner with updated board
        if self._check_winner():
            self.winner = player

        # Advance turn
        self._increment_to_next_turn()

    def is_over(self) -> bool:
        """Game ends when someone wins or board is full."""
        return self.winner is not None or board_full(self.state.board)

    def get_result(self, agent_id: int) -> State:
        """
        Return LOSS, NEUTRAL, TIE, or WIN
        """
        if self.winner is not None:
            return State.WIN if self.winner == agent_id else State.LOSS

        if self.is_over():
            return State.TIE

        return State.NEUTRAL

    def state_string(self) -> str:
        """Pretty-print the board and current player."""
        symbols = {None: " ", 1: "X", 2: "O"}

        top = "╭───┬───┬───╮"
        mid = "├───┼───┼───┤"
        bot = "╰───┴───┴───╯"

        board = self.state.board
        lines = [top]
        for i, row in enumerate(board):
            line = "│ " + " │ ".join(symbols[x] for x in row) + " │"
            lines.append(line)
            if i < self.SIZE - 1:
                lines.append(mid)
        lines.append(bot)

        result = "\n".join(lines)
        result += f"\n\nCurrent player: {self.state.current_player}"
        return result

    # --------------------------------------------------------------
    # Internal helper logic
    # --------------------------------------------------------------
    def _check_winner(self) -> bool:
        """Check if any row, column, or diagonal has 3 identical non-empty marks."""
        board = self.state.board
        lines: List[np.ndarray] = []

        # Rows, columns, diagonals
        lines.extend(get_rows(board))
        lines.extend(get_cols(board))
        lines.extend(get_diagonals(board))

        # Check each line
        for line in lines:
            if line[0] is not None and np.all(line == line[0]):
                return True
        return False

    def _increment_to_next_turn(self) -> None:
        self.state.current_player = 2 if self.current_player() == 1 else 1
