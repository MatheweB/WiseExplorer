# test_game_rules.py
import pytest
import numpy as np

from games.game_rules import (
    in_bounds,
    all_equal,
    get_rows,
    get_cols,
    get_diagonals,
    board_full,
)


# ---------------------------------------------------------
# in_bounds
# ---------------------------------------------------------
def test_in_bounds():
    board = np.zeros((3, 3))

    assert in_bounds(board, 0, 0)
    assert in_bounds(board, 2, 2)

    assert not in_bounds(board, -1, 0)
    assert not in_bounds(board, 0, -1)
    assert not in_bounds(board, 3, 0)
    assert not in_bounds(board, 0, 3)


# ---------------------------------------------------------
# all_equal
# ---------------------------------------------------------
def test_all_equal_basic():
    assert all_equal(np.array([1, 1, 1], dtype=object)) is True
    assert all_equal(np.array(["X", "X", "X"], dtype=object)) is True
    assert all_equal(np.array([None, None, None], dtype=object)) is False
    assert all_equal(np.array([], dtype=object)) is False


def test_all_equal_mixed():
    assert all_equal(np.array([1, 2, 1], dtype=object)) is False
    assert all_equal(np.array(["X", "O", "X"], dtype=object)) is False


# ---------------------------------------------------------
# get_rows
# ---------------------------------------------------------
def test_get_rows():
    board = np.array([[1, 2], [3, 4]], dtype=object)
    rows = get_rows(board)

    assert isinstance(rows, list)
    assert len(rows) == 2
    assert np.array_equal(rows[0], np.array([1, 2], dtype=object))
    assert np.array_equal(rows[1], np.array([3, 4], dtype=object))


# ---------------------------------------------------------
# get_cols
# ---------------------------------------------------------
def test_get_cols():
    board = np.array([[1, 2], [3, 4]], dtype=object)
    cols = get_cols(board)

    assert isinstance(cols, list)
    assert len(cols) == 2
    assert np.array_equal(cols[0], np.array([1, 3], dtype=object))
    assert np.array_equal(cols[1], np.array([2, 4], dtype=object))


# ---------------------------------------------------------
# get_diagonals
# ---------------------------------------------------------
def test_get_diagonals():
    board = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=object
    )

    diags = get_diagonals(board)
    assert isinstance(diags, list)
    assert len(diags) == 2

    major, minor = diags

    assert np.array_equal(major, np.array([1, 5, 9], dtype=object))
    assert np.array_equal(minor, np.array([3, 5, 7], dtype=object))


# ---------------------------------------------------------
# board_full
# ---------------------------------------------------------
def test_board_full_true():
    board = np.array([
        ["X", "O"],
        ["O", "X"]
    ], dtype=object)
    assert board_full(board) is True


def test_board_full_false():
    board = np.array([
        ["X", None],
        ["O", "X"]
    ], dtype=object)
    assert board_full(board) is False