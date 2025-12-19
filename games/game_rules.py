"""
game_rules.py

High-performance NumPy utilities for board games.
Designed for fast rollouts, large board sizes,
and vectorized win detection.

All operations avoid ambiguous None broadcasts and avoid dangerous
NumPy views (for dtypes=object boards).
"""

from __future__ import annotations
import numpy as np
from typing import List


# ---------------------------------------------------------------------
# Bounds checking
# ---------------------------------------------------------------------
def in_bounds(board: np.ndarray, r: int, c: int) -> bool:
    """Return True if (r, c) is inside the board."""
    rows, cols = board.shape
    return 0 <= r < rows and 0 <= c < cols


# ---------------------------------------------------------------------
# Equality checks
# ---------------------------------------------------------------------
def all_equal(line: np.ndarray) -> bool:
    """
    Return True if:
    - line is nonempty
    - first value is not None
    - all values equal the first

    Uses np.equal for speed but falls back to Python eq for object dtype.
    """
    if line.size == 0:
        return False

    first = line[0]
    if first is None:
        return False

    # Fast path: use NumPy if possible
    try:
        return bool(np.all(line == first))
    except Exception:
        # Slow path: fall back to Python equality
        return all(x == first for x in line)


# ---------------------------------------------------------------------
# Row/column extraction
# ---------------------------------------------------------------------
def get_rows(board: np.ndarray) -> List[np.ndarray]:
    """
    Fast row extraction.
    Uses slicing, but slices are copied to avoid shared memory issues.
    """
    # Copy each row slice; still faster than building manually.
    return [row.copy() for row in board]


def get_cols(board: np.ndarray) -> List[np.ndarray]:
    """
    Fast column extraction using slicing on board.T.
    board.T does create a view, but copying each column removes risk.
    """
    bt = board.T  # shape: (cols, rows)
    return [col.copy() for col in bt]


# ---------------------------------------------------------------------
# Diagonals (optimized)
# ---------------------------------------------------------------------
def get_diagonals(board: np.ndarray) -> List[np.ndarray]:
    """
    Optimized diagonal extraction using NumPy's diagonal efficiently.

    board.diagonal() is a *view* in most cases, so we explicitly copy it.
    """
    major = board.diagonal().copy()  # top-left → bottom-right
    minor = np.fliplr(board).diagonal().copy()  # top-right → bottom-left
    return [major, minor]


# ---------------------------------------------------------------------
# Board completion check
# ---------------------------------------------------------------------
def board_full(board: np.ndarray) -> bool:
    """
    Return True if the board has no None values.

    Optimized:
        - Flatten once
        - Use Python-level identity comparison
        - Much faster on dtype=object arrays than np.all(board != None)
    """
    # Python-level scan is fastest for None checks
    return all(x is not None for x in board.flat)
