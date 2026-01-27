"""
Tests for wise_explorer.simulation.training

Tests training orchestration with prune/exploit phases.
"""

from unittest.mock import MagicMock

import pytest

from wise_explorer.simulation.training import run_training, run_training_interleaved


class TestRunTraining:
    """run_training function tests."""

    def test_zero_sims_returns_zero(self):
        """Zero simulations returns 0."""
        runner = MagicMock()
        result = run_training(runner, {1: [], 2: []}, MagicMock(), 0, 10)
        
        assert result == 0
        runner.run_batch.assert_not_called()

    def test_empty_swarms_returns_zero(self):
        """Empty swarms returns 0."""
        result = run_training(MagicMock(), {}, MagicMock(), 100, 10)
        assert result == 0

    def test_splits_prune_exploit(self):
        """Splits simulations between prune and exploit phases."""
        runner = MagicMock()
        runner.run_batch.return_value = 10
        swarms = {1: [MagicMock()], 2: [MagicMock()]}
        
        run_training(runner, swarms, MagicMock(), 100, 10)
        
        # Should have multiple calls (prune for each player + exploit)
        assert runner.run_batch.call_count >= 2

    def test_prunes_each_player(self):
        """Each player gets pruned."""
        runner = MagicMock()
        runner.run_batch.return_value = 10
        swarms = {1: [MagicMock()], 2: [MagicMock()]}
        
        run_training(runner, swarms, MagicMock(), 100, 10)
        
        prune_args = [c[1]['prune_players'] for c in runner.run_batch.call_args_list]
        assert {1} in prune_args
        assert {2} in prune_args
        assert set() in prune_args  # Exploit phase

    def test_returns_total_transitions(self):
        """Returns sum of all transitions."""
        runner = MagicMock()
        runner.run_batch.side_effect = [10, 20, 30]
        swarms = {1: [MagicMock()], 2: [MagicMock()]}
        
        result = run_training(runner, swarms, MagicMock(), 100, 10)
        assert result == 60

    def test_passes_turn_depth(self):
        """Passes turn_depth to runner."""
        runner = MagicMock()
        runner.run_batch.return_value = 0
        swarms = {1: [MagicMock()]}
        
        run_training(runner, swarms, MagicMock(), 10, turn_depth=42)
        
        for call_args in runner.run_batch.call_args_list:
            assert call_args[1]['max_turns'] == 42


class TestRunTrainingInterleaved:
    """run_training_interleaved function tests."""

    def test_zero_sims_returns_zero(self):
        """Zero simulations returns 0."""
        result = run_training_interleaved(MagicMock(), {1: [], 2: []}, MagicMock(), 0, 10)
        assert result == 0

    def test_alternates_phases(self):
        """Alternates prune/exploit phases."""
        runner = MagicMock()
        runner.run_batch.return_value = 5
        swarms = {1: [MagicMock()], 2: [MagicMock()]}
        
        run_training_interleaved(runner, swarms, MagicMock(), 150, 10, phase_size=50)
        
        prune_args = [c[1]['prune_players'] for c in runner.run_batch.call_args_list]
        
        # Pattern: {1}, {2}, {}, repeat
        assert prune_args[0] == {1}
        assert prune_args[1] == {2}
        assert prune_args[2] == set()

    def test_respects_phase_size(self):
        """Uses specified phase_size."""
        runner = MagicMock()
        runner.run_batch.return_value = 5
        swarms = {1: [MagicMock()]}
        
        run_training_interleaved(runner, swarms, MagicMock(), 100, 10, phase_size=25)
        
        assert runner.run_batch.call_count == 4  # 100 / 25 = 4 phases

    def test_returns_total_transitions(self):
        """Returns sum of all transitions."""
        runner = MagicMock()
        runner.run_batch.side_effect = [10, 20, 30, 40]
        swarms = {1: [MagicMock()], 2: [MagicMock()]}
        
        result = run_training_interleaved(runner, swarms, MagicMock(), 200, 10, phase_size=50)
        assert result == 100


class TestEdgeCases:
    """Edge case tests."""

    def test_single_player(self):
        """Works with single player."""
        runner = MagicMock()
        runner.run_batch.return_value = 10
        swarms = {1: [MagicMock()]}
        
        result = run_training(runner, swarms, MagicMock(), 50, 10)
        assert result > 0

    def test_odd_simulation_count(self):
        """Handles odd simulation counts."""
        runner = MagicMock()
        runner.run_batch.return_value = 5
        swarms = {1: [MagicMock()], 2: [MagicMock()]}
        
        run_training(runner, swarms, MagicMock(), 101, 10)
        
        total_sims = sum(c[1]['num_sims'] for c in runner.run_batch.call_args_list)
        assert total_sims == 101