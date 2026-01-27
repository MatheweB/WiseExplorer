"""
Tests for wise_explorer.core.hashing

Tests board hashing utilities for game state identification.
"""

import numpy as np
import pytest

from wise_explorer.core.hashing import hash_board


class TestHashDeterminism:
    """Tests that hashing is deterministic."""

    def test_same_board_same_hash(self):
        """Identical boards produce identical hashes."""
        board = np.array([[1, 0, 2], [0, 1, 0], [2, 0, 1]], dtype=np.int8)
        assert hash_board(board) == hash_board(board.copy())

    def test_independent_creation_same_hash(self):
        """Independently created identical boards have same hash."""
        board1 = np.array([[1, 2], [3, 4]], dtype=np.int8)
        board2 = np.array([[1, 2], [3, 4]], dtype=np.int8)
        assert hash_board(board1) == hash_board(board2)


class TestHashUniqueness:
    """Tests that different boards produce different hashes."""

    def test_different_boards_different_hash(self):
        """Different boards produce different hashes."""
        board1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
        board2 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
        assert hash_board(board1) != hash_board(board2)

    def test_sign_matters(self):
        """Positive vs negative values produce different hashes."""
        board1 = np.array([[1, 2, 3]], dtype=np.int8)
        board2 = np.array([[-1, 2, 3]], dtype=np.int8)
        assert hash_board(board1) != hash_board(board2)


class TestHashFormat:
    """Tests for hash format."""

    def test_hash_is_hex_string(self):
        """Hash is a 16-character hex string."""
        board = np.zeros((3, 3), dtype=np.int8)
        h = hash_board(board)
        assert isinstance(h, str)
        assert len(h) == 16
        assert all(c in '0123456789abcdef' for c in h)


class TestArrayTypes:
    """Tests for different numpy array types."""

    def test_int8_array(self):
        """Works with int8 arrays (primary use case)."""
        board = np.array([[1, 0], [0, 2]], dtype=np.int8)
        assert len(hash_board(board)) == 16

    def test_int32_array(self):
        """Works with int32 arrays."""
        board = np.array([[1, 0], [0, 2]], dtype=np.int32)
        assert len(hash_board(board)) == 16

    def test_object_array(self):
        """Works with object arrays (fallback path)."""
        board = np.array([["a", "b"], ["c", "d"]], dtype=np.object_)
        assert len(hash_board(board)) == 16


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_array(self):
        """Works with empty array."""
        assert len(hash_board(np.array([], dtype=np.int8))) == 16

    def test_single_element(self):
        """Works with single element array."""
        assert len(hash_board(np.array([1], dtype=np.int8))) == 16

    def test_large_board(self):
        """Works with large board."""
        assert len(hash_board(np.zeros((100, 100), dtype=np.int8))) == 16

    def test_negative_values(self):
        """Handles negative values (used for player 2 pieces)."""
        board = np.array([[-1, -2, -3], [1, 2, 3]], dtype=np.int8)
        assert len(hash_board(board)) == 16


class TestCollisionResistance:
    """Basic collision resistance tests."""

    def test_many_random_boards_unique(self):
        """Random boards produce unique hashes."""
        np.random.seed(42)
        hashes = {
            hash_board(np.random.randint(-4, 5, size=(3, 3), dtype=np.int8))
            for _ in range(1000)
        }
        assert len(hashes) >= 990  # Allow for rare collisions

    def test_sequential_modifications_unique(self):
        """Sequential board modifications produce unique hashes."""
        board = np.zeros((3, 3), dtype=np.int8)
        hashes = []
        for i in range(9):
            board.flat[i] = 1
            hashes.append(hash_board(board.copy()))
        assert len(set(hashes)) == 9