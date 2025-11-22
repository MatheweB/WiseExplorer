import numpy as np


class GameState:
    """
    Represents the full state of a board game at a given moment.
    This includes:
        - board (np.ndarray)
        - current_player (int)
    """

    def __init__(self, board: np.ndarray, current_player: int):
        # board: NumPy array containing game-specific symbols
        # current_player: int ID (usually 0 or 1)
        self.board = board
        self.current_player = current_player

    def clone(self) -> "GameState":
        """
        Return a deep copy of this state.
        Agents depend on this for simulation.
        """
        return GameState(board=self.board.copy(), current_player=self.current_player)
