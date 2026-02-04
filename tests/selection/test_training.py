"""
Tests for wise_explorer.selection.training

Tests probabilistic move selection for training with exploration.
"""

import random
from typing import Dict, List, Tuple

import numpy as np
import pytest

from wise_explorer.core.types import Stats
from wise_explorer.selection.training import (
    _exploration_weight,
    select_anchor_deterministic,
    select_move_random,
)


class TestExplorationWeight:
    """_exploration_weight function tests."""

    def test_pick_best_vs_pick_prune(self):
        """pick_best affects weight direction."""
        winning = Stats(100, 0, 0)
        losing = Stats(0, 0, 100)
        
        # pick_best=True favors high scores
        assert _exploration_weight(winning, True) > _exploration_weight(losing, True)
        # pick_best=False favors low scores
        assert _exploration_weight(losing, False) > _exploration_weight(winning, False)


class TestSelectAnchor:
    """select_anchor function tests."""

    def test_returns_valid_id(self, anchor_stats_varied: Dict[int, Stats]):
        """Returns a valid anchor ID."""
        random.seed(42)
        result = select_anchor_deterministic(anchor_stats_varied, pick_best=True)
        assert result in anchor_stats_varied

    def test_single_anchor(self):
        """Returns only anchor when just one exists."""
        stats = {42: Stats(10, 5, 5)}
        assert select_anchor_deterministic(stats, pick_best=True) == 42

    def test_empty_raises(self):
        """Raises on empty anchor stats."""
        with pytest.raises(ValueError, match="No anchors"):
            select_anchor_deterministic({}, pick_best=True)

    def test_deterministic(self, anchor_stats_varied: Dict[int, Stats]):
        """Selection picks the best mean_score"""
        selected_anchor_id = select_anchor_deterministic(anchor_stats_varied, pick_best=True)
        max_scored_anchor = max(anchor_stats_varied.items(), key=lambda a: a[1].mean_score)[1]
        assert anchor_stats_varied[selected_anchor_id] == max_scored_anchor

    def test_favors_unexplored(self):
        """Unexplored anchors are favored over losing anchors."""
        stats = {
            0: Stats(0, 0, 5000),  # Very certain loss
            1: Stats(0, 0, 0),     # Unexplored
        }
        
        random.seed(42)
        unexplored_count = sum(
            1 for _ in range(100) if select_anchor_deterministic(stats, pick_best=True) == 1
        )
        assert unexplored_count > 80


class TestSelectMove:
    """select_move function tests."""

    def test_returns_valid_move(self, moves_with_stats: List[Tuple[np.ndarray, Stats]]):
        """Returns a valid move from the list."""
        random.seed(42)
        result = select_move_random(moves_with_stats, pick_best=True)
        
        move_arrays = [m[0] for m in moves_with_stats]
        assert any(np.array_equal(result, m) for m in move_arrays)

    def test_single_move(self):
        """Returns only move when just one exists."""
        moves = [(np.array([7, 8]), Stats(10, 5, 5))]
        result = select_move_random(moves, pick_best=True)
        np.testing.assert_array_equal(result, np.array([7, 8]))

    def test_returns_array(self, moves_with_stats: List[Tuple[np.ndarray, Stats]]):
        """Returns numpy array."""
        random.seed(42)
        result = select_move_random(moves_with_stats, pick_best=True)
        assert isinstance(result, np.ndarray)

class TestReproducibility:
    """Reproducibility with random seed tests."""
    
    def test_move_selection_reproducible(self, moves_with_stats: List[Tuple[np.ndarray, Stats]]):
        """Same seed produces same results."""
        random.seed(12345)
        result1 = [tuple(select_move_random(moves_with_stats, pick_best=True)) for _ in range(10)]
        
        random.seed(12345)
        result2 = [tuple(select_move_random(moves_with_stats, pick_best=True)) for _ in range(10)]
        
        assert result1 == result2