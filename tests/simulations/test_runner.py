"""
Tests for wise_explorer.simulation.runner

Tests SimulationRunner for parallel game execution.
"""

from typing import Dict, List

import pytest

from wise_explorer.agent.agent import Agent
from wise_explorer.games.game_base import GameBase
from wise_explorer.memory.game_memory import GameMemory
from wise_explorer.simulation.runner import SimulationRunner, DEFAULT_WORKER_COUNT


class TestDefaultWorkerCount:
    """DEFAULT_WORKER_COUNT tests."""

    def test_positive(self):
        """At least 1 worker."""
        assert DEFAULT_WORKER_COUNT >= 1


class TestCreation:
    """SimulationRunner creation tests."""

    def test_creation(self, memory: GameMemory):
        """Runner can be created."""
        runner = SimulationRunner(memory, num_workers=1)
        assert runner.memory is memory
        assert runner.num_workers == 1
        assert runner._pool is None

    def test_default_workers(self, memory: GameMemory):
        """Uses DEFAULT_WORKER_COUNT when not specified."""
        runner = SimulationRunner(memory)
        assert runner.num_workers == DEFAULT_WORKER_COUNT


class TestContextManager:
    """Context manager tests."""

    def test_creates_pool(self, memory: GameMemory):
        """Context manager creates pool on entry."""
        runner = SimulationRunner(memory, num_workers=1)
        
        with runner:
            assert runner._pool is not None

    def test_handles_exception(self, memory: GameMemory):
        """Context manager handles exceptions."""
        runner = SimulationRunner(memory, num_workers=1)
        
        with pytest.raises(ValueError):
            with runner:
                raise ValueError("Test")


class TestRunBatchEdgeCases:
    """run_batch edge case tests."""

    def test_zero_sims(self, memory: GameMemory, agent_swarms: Dict[int, List[Agent]], two_player_game: GameBase):
        """Zero simulations returns 0."""
        runner = SimulationRunner(memory, num_workers=1)
        result = runner.run_batch(agent_swarms, two_player_game, 0, 10, set())
        assert result == 0

    def test_empty_swarms(self, memory: GameMemory, two_player_game: GameBase):
        """Empty swarms returns 0."""
        runner = SimulationRunner(memory, num_workers=1)
        result = runner.run_batch({}, two_player_game, 10, 10, set())
        assert result == 0


class TestMakeJobs:
    """_make_jobs tests."""

    def test_job_count(self, memory: GameMemory, agent_swarms: Dict[int, List[Agent]], two_player_game: GameBase):
        """Creates correct number of jobs."""
        runner = SimulationRunner(memory, num_workers=1)
        jobs = runner._make_jobs(agent_swarms, two_player_game, count=5, max_turns=10, prune_players=set())
        assert len(jobs) == 5

    def test_job_structure(self, memory: GameMemory, agent_swarms: Dict[int, List[Agent]], two_player_game: GameBase):
        """Jobs have correct structure."""
        runner = SimulationRunner(memory, num_workers=1)
        jobs = runner._make_jobs(agent_swarms, two_player_game, count=1, max_turns=15, prune_players={1})
        
        job = jobs[0]
        assert job.max_turns == 15
        assert job.prune_players == {1}

    def test_deep_clones_game(self, memory: GameMemory, agent_swarms: Dict[int, List[Agent]], two_player_game: GameBase):
        """Each job has independent game."""
        runner = SimulationRunner(memory, num_workers=1)
        jobs = runner._make_jobs(agent_swarms, two_player_game, count=2, max_turns=10, prune_players=set())
        
        assert jobs[0].game is not jobs[1].game
        assert jobs[0].game is not two_player_game


class TestShutdown:
    """Shutdown tests."""

    def test_shutdown_no_pool(self, memory: GameMemory):
        """Shutdown with no pool does nothing."""
        runner = SimulationRunner(memory, num_workers=1)
        runner.shutdown()  # Should not raise

    def test_shutdown_force(self, memory: GameMemory):
        """Force shutdown terminates pool."""
        runner = SimulationRunner(memory, num_workers=1)
        runner._ensure_pool()
        runner.shutdown(force=True)
        assert runner._pool is None


@pytest.mark.slow
class TestIntegration:
    """Integration tests (may be slow)."""

    def test_run_batch(self, memory: GameMemory, agent_swarms: Dict[int, List[Agent]], two_player_game: GameBase):
        """Run a small batch."""
        runner = SimulationRunner(memory, num_workers=1)
        
        with runner:
            result = runner.run_batch(agent_swarms, two_player_game, 2, 10, set())
        
        assert result >= 0