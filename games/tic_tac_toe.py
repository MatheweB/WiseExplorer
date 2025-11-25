# tic_tac_toe.py
# Libraries
from copy import copy
from copy import deepcopy
import numpy as np
from typing import Tuple

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
    """

    SIZE = 3  # 3x3 board

    def __init__(self):
        """Initialize a fresh game."""
        empty = np.full((self.SIZE, self.SIZE), None)
        self.state = GameState(empty, current_player=1)
        self.winner = None  # 1 or 2, or None

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
        """Return the index of the current player (1 or 2)."""
        return self.state.current_player

    def valid_moves(self) -> np.ndarray:
        """Return list of all (row, col) tuples that are empty."""
        board = self.state.board
        moves = np.argwhere(board == np.array(None))
        return moves

    def apply_move(self, move: np.ndarray) -> None:
        """Apply (r, c) move for current player and update state."""
        r, c = move
        board = self.state.board
        # The player who just moved
        current_player_id = self.state.current_player

        # Place mark
        board[r, c] = current_player_id

        # Check win
        if self._check_winner():
            # The winner is the player who just moved
            self.winner = current_player_id

        # Change current_player variables
        self._increment_to_next_turn()

    def is_over(self) -> bool:
        """Game ends if someone won or board is full."""
        return self.winner is not None or board_full(self.state.board)

    def get_result(self, agent_id: int) -> State:
            """
            Return LOSS, NEUTRAL, TIE, or WIN
            """
            # 1. Someone won (WIN or LOSS)
            if self.winner is not None:
                return State.WIN if self.winner == agent_id else State.LOSS
            
            # 2. No one won (winner is None)
            else:
                # Check if the game is over (board is full)
                if self.is_over():
                    # Game is over, no winner -> TIE
                    return State.TIE
                else:
                    # Game is NOT over, no winner -> NEUTRAL (Incomplete)
                    return State.NEUTRAL

    def state_string(self) -> None:
        """
        Prints out the current state of the game in a visually appealing way
        """
        symbols = {None: " ", 1: "X", 2: "O"}

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

        return result

    # --------------------------------------------------------------
    # Internal helper logic
    # --------------------------------------------------------------
    def _check_winner(self) -> bool:
        """Check if any row/col/diagonal has 3 in a row."""
        board = self.state.board
        lines = []
        # Collect all lines (rows, cols, diagonals)
        lines.extend(get_rows(board))
        lines.extend(get_cols(board))
        lines.extend(get_diagonals(board))

        for line in lines:
            # 1. Check line[0] is not None (prevents winning with empty lines)
            # 2. Check if all elements in the line equal the first element
            if line[0] is not None and np.all(line == line[0]):
                return True
        return False
        
    # Needed for apply_move to get to the next turn gracefully
    def _increment_to_next_turn(self) -> None:
        if self.current_player() == 1:
            self.get_state().current_player = 2
        else:
            self.get_state().current_player = 1
