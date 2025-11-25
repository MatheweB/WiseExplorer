# tic_tac_toe.py
# Libraries
from copy import copy
from copy import deepcopy
import numpy as np
from typing import Tuple

# Functions
from game_rules import get_rows, get_cols, get_diagonals, all_equal, board_full

# Classes
from game_base import GameBase
from game_state import GameState
from agent.agent import State


class TicTacToe(GameBase):
    """
    Fully implemented TicTacToe game.
    Designed for multi-agent simulations.
    """

    SIZE = 3  # 3x3 board

    def __init__(self):
        """Initialize a fresh game."""
        empty = np.full((self.SIZE, self.SIZE), None)
        self.state = GameState(empty, current_player=0)
        self.winner = None  # 0 or 1, or None

    # --------------------------------------------------------------
    # Required interface methods
    # --------------------------------------------------------------
    def game_id(self) -> str:
        """
        Returns the name of the game we're playing (e.g. tic_tac_toe)
        """
        return "tic_tac_toe"

    def clone(self) -> "TicTacToe":
        """Return a shallow copy of the game."""
        return copy(self)

    def deep_clone(self) -> "TicTacToe":
        """Return a deep copy of the game."""
        return deepcopy(self)

    def get_state(self) -> GameState:
        """
        Return the current state of the game.
        """
        return self.state

    def set_state(self, game_state: GameState) -> None:
        """
        Return the current state of the game.
        """
        self.state = game_state

    def current_player(self) -> int:
        """Return the index of the current player (0 or 1)."""
        return self.state.current_player

    def valid_moves(self) -> np.ndarray:
        """Return list of all (row, col) tuples that are empty."""
        board = self.state.board
        moves = np.argwhere(board is None)
        return moves

    def apply_move(self, move: Tuple[int, int]) -> None:
        """Apply (r, c) move for current player and update state."""
        r, c = move
        board = self.state.board
        # Place mark
        player = self.state.current_player
        board[r][c] = player
        # Check win
        if self._check_winner():
            self.winner = player
        # Change current_player variables
        self._increment_to_next_turn()

    def is_over(self) -> bool:
        """Game ends if someone won or board is full."""
        return self.winner is not None or board_full(self.state.board)

    def get_result(self, agent_id: int) -> State:
        """
        Return +1 for win, 0 for draw, -1 for loss.
        This is exactly the form agents expect.
        """
        if self.winner is None:
            if self.is_over():
                # tie
                return State.TIE
            # incomplete
            return State.NEUTRAL
        return State.WIN if self.winner == agent_id else State.LOSS

    def print_state(self) -> None:
        """
        Prints out the current state of the game in a visually appealing way
        """
        symbols = {0: " ", 1: "X", 2: "O"}

        top = "╭───┬───┬───╮"
        mid = "├───┼───┼───┤"
        bot = "╰───┴───┴───╯"

        board = self.state.board
        current_player = self.state.current_player
        lines = [top]
        for i, row in enumerate(board):
            line = "│ " + " │ ".join(symbols[x] for x in row) + " │"
            lines.append(line)
            if i < 2:
                lines.append(mid)
        lines.append(bot)

        result = "\n".join(lines)
        if current_player is not None:
            result += f"\n\nCurrent player: {current_player}"

        print(result)

    # --------------------------------------------------------------
    # Internal helper logic
    # --------------------------------------------------------------
    def _check_winner(self) -> bool:
        """Check if any row/col/diagonal has 3 in a row."""
        board = self.state.board
        lines = []
        lines.extend(get_rows(board))
        lines.extend(get_cols(board))
        lines.extend(get_diagonals(board))
        for line in lines:
            if all_equal(line):
                return True
        return False
    
    # Needed for apply_move to get to the next turn gracefully
    def _increment_to_next_turn(self) -> None:
        if self.current_player() == 1:
            self.get_state().current_player = 2
        else:
            self.get_state().current_player = 1
