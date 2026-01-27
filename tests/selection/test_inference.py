"""
Tests for wise_explorer.selection.inference

Tests deterministic move selection for competitive play.
"""

from typing import Dict, List, Tuple

import numpy as np
import pytest

from wise_explorer.core.types import Stats
from wise_explorer.selection.inference import best_anchor, best_move, random_move


class TestBestAnchor:
    """best_anchor function tests."""

    def test_picks_highest_score(self, anchor_stats_varied: Dict[int, Stats]):
        """Picks anchor with highest mean_score when pick_best=True."""
        result = best_anchor(anchor_stats_varied, pick_best=True)
        assert result == 0  # Highest win rate

    def test_picks_lowest_score(self, anchor_stats_varied: Dict[int, Stats]):
        """Picks anchor with lowest mean_score when pick_best=False."""
        result = best_anchor(anchor_stats_varied, pick_best=False)
        assert result == 2  # Highest loss rate

    def test_single_anchor(self):
        """Returns only anchor when just one exists."""
        stats = {42: Stats(10, 5, 5)}
        assert best_anchor(stats, pick_best=True) == 42


class TestBestMove:
    """best_move function tests."""

    def test_picks_highest_score(self, moves_with_stats: List[Tuple[np.ndarray, Stats]]):
        """Picks move with highest mean_score when pick_best=True."""
        result = best_move(moves_with_stats, pick_best=True)
        np.testing.assert_array_equal(result, np.array([0, 0]))

    def test_picks_lowest_score(self, moves_with_stats: List[Tuple[np.ndarray, Stats]]):
        """Picks move with lowest mean_score when pick_best=False."""
        result = best_move(moves_with_stats, pick_best=False)
        np.testing.assert_array_equal(result, np.array([1, 1]))

    def test_single_move(self):
        """Returns only move when just one exists."""
        moves = [(np.array([5, 5]), Stats(10, 5, 5))]
        result = best_move(moves, pick_best=True)
        np.testing.assert_array_equal(result, np.array([5, 5]))

    def test_returns_array(self, moves_with_stats: List[Tuple[np.ndarray, Stats]]):
        """Returns numpy array."""
        result = best_move(moves_with_stats, pick_best=True)
        assert isinstance(result, np.ndarray)


class TestRandomMove:
    """random_move function tests."""

    def test_returns_valid_move(self, moves_with_stats: List[Tuple[np.ndarray, Stats]]):
        """Returns a move from the list."""
        result = random_move(moves_with_stats)
        move_arrays = [m[0] for m in moves_with_stats]
        assert any(np.array_equal(result, m) for m in move_arrays)

    def test_single_move(self):
        """Returns only move when just one exists."""
        moves = [(np.array([7, 8]), Stats(10, 5, 5))]
        result = random_move(moves)
        np.testing.assert_array_equal(result, np.array([7, 8]))

    def test_selects_multiple_over_trials(self, moves_with_stats: List[Tuple[np.ndarray, Stats]]):
        """Random selection covers multiple options."""
        selections = {tuple(random_move(moves_with_stats)) for _ in range(100)}
        assert len(selections) > 1


class TestConsistency:
    """Consistency tests between functions."""

    def test_best_anchor_consistent_ordering(self):
        """best_anchor returns anchor with correct ordering."""
        stats = {0: Stats(100, 10, 10), 1: Stats(10, 10, 100)}
        
        best = best_anchor(stats, pick_best=True)
        worst = best_anchor(stats, pick_best=False)
        
        assert stats[best].mean_score >= stats[worst].mean_score

    def test_best_move_consistent_ordering(self):
        """best_move returns move with correct ordering."""
        moves = [
            (np.array([0, 0]), Stats(100, 10, 10)),
            (np.array([1, 1]), Stats(10, 10, 100)),
        ]
        
        best = best_move(moves, pick_best=True)
        worst = best_move(moves, pick_best=False)
        
        best_stats = next(s for m, s in moves if np.array_equal(m, best))
        worst_stats = next(s for m, s in moves if np.array_equal(m, worst))
        
        assert best_stats.mean_score >= worst_stats.mean_score