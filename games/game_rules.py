import numpy as np
from typing import List


def in_bounds(board: np.ndarray, r: int, c: int) -> bool:
    """Return True if (r, c) is within the board dimensions."""
    return 0 <= r < board.shape[0] and 0 <= c < board.shape[1]


def all_equal(line: np.ndarray) -> bool:
    """Return True if all values in a line are equal and not empty."""
    if line.size == 0 or line[0] is None:
        return False
    return bool(np.all(line == line[0]))


def get_rows(board: np.ndarray) -> np.ndarray:
    """Return the rows of the board as a NumPy array."""
    return board


def get_cols(board: np.ndarray) -> np.ndarray:
    """Return the columns of the board as a NumPy array."""
    return board.T  # Transpose to get columns


def get_diagonals(board: np.ndarray) -> List[np.ndarray]:
    """Return both major diagonals from a square board as NumPy arrays."""
    return [
        board.diagonal(),            # Top-Left to Bottom-Right
        np.fliplr(board).diagonal()  # Top-Right to Bottom-Left
    ]


def board_full(board: np.ndarray) -> bool:
    """Return True if every space is filled."""
    return bool(np.all(board != None))
