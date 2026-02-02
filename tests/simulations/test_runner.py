"""
Tests for wise_explorer.simulation.runner

Tests SimulationRunner for parallel game execution.
"""

import pytest

from wise_explorer.simulation.runner import (
    SimulationRunner,
    DEFAULT_WORKER_COUNT,
    _active_runners,
    _active_memories,
    register_memory,
)


@pytest.fixture(autouse=True)
def cleanup_globals():
    """Ensure global registries are clean before and after each test."""
    _active_runners.clear()
    _active_memories.clear()
    yield
    # Force shutdown any leaked runners
    for runner in _active_runners[:]:
        try:
            runner.shutdown(force=True)
        except Exception:
            pass
    _active_runners.clear()
    _active_memories.clear()


class TestDefaultWorkerCount:
    """DEFAULT_WORKER_COUNT tests."""

    def test_positive(self):
        """At least 1 worker."""
        assert DEFAULT_WORKER_COUNT >= 1


class TestGlobalRegistry:
    """Tests for cleanup infrastructure."""

    def test_runner_registers_on_init(self, transition_memory):
        """Runner registers itself and memory on creation."""
        assert len(_active_runners) == 0
        assert len(_active_memories) == 0

        runner = SimulationRunner(transition_memory, num_workers=1)

        assert runner in _active_runners
        assert transition_memory in _active_memories

        runner.shutdown()

    def test_runner_unregisters_on_shutdown(self, transition_memory):
        """Runner removes itself from registry on shutdown."""
        runner = SimulationRunner(transition_memory, num_workers=1)
        assert runner in _active_runners

        runner.shutdown()

        assert runner not in _active_runners

    def test_register_memory_idempotent(self, transition_memory):
        """Registering same memory twice doesn't duplicate."""
        register_memory(transition_memory)
        register_memory(transition_memory)

        assert _active_memories.count(transition_memory) == 1


# =============================================================================
# Transition Runner Tests
# =============================================================================

class TestTransitionCreation:
    """SimulationRunner creation with TransitionMemory."""

    def test_creation(self, transition_memory):
        """Runner can be created."""
        runner = SimulationRunner(transition_memory, num_workers=1)

        assert runner.memory is transition_memory
        assert runner.num_workers == 1
        assert runner._pool is None

        runner.shutdown()

    def test_default_workers(self, transition_memory):
        """Uses DEFAULT_WORKER_COUNT when not specified."""
        runner = SimulationRunner(transition_memory)

        assert runner.num_workers == DEFAULT_WORKER_COUNT

        runner.shutdown()


class TestTransitionContextManager:
    """Context manager tests with TransitionMemory."""

    def test_creates_pool(self, transition_memory):
        """Context manager creates pool on entry."""
        runner = SimulationRunner(transition_memory, num_workers=1)

        with runner:
            assert runner._pool is not None

        assert runner._pool is None  # Cleaned up on exit


class TestTransitionRunBatchEdgeCases:
    """run_batch edge cases with TransitionMemory."""

    def test_zero_sims(self, transition_memory, agent_swarms, two_player_game):
        """Zero simulations returns 0."""
        with SimulationRunner(transition_memory, num_workers=1) as runner:
            result = runner.run_batch(agent_swarms, two_player_game, 0, 10, set())

        assert result == 0

    def test_empty_swarms(self, transition_memory, two_player_game):
        """Empty swarms returns 0."""
        with SimulationRunner(transition_memory, num_workers=1) as runner:
            result = runner.run_batch({}, two_player_game, 10, 10, set())

        assert result == 0


class TestTransitionMakeJobs:
    """_make_jobs tests with TransitionMemory."""

    def test_job_count(self, transition_memory, agent_swarms, two_player_game):
        """Creates correct number of jobs."""
        runner = SimulationRunner(transition_memory, num_workers=1)

        jobs = runner._make_jobs(agent_swarms, two_player_game, count=5, max_turns=10, prune_players=set())

        assert len(jobs) == 5
        runner.shutdown()

    def test_job_structure(self, transition_memory, agent_swarms, two_player_game):
        """Jobs have correct structure."""
        runner = SimulationRunner(transition_memory, num_workers=1)

        jobs = runner._make_jobs(agent_swarms, two_player_game, count=1, max_turns=15, prune_players={1})

        job = jobs[0]
        assert job.max_turns == 15
        assert job.prune_players == {1}
        runner.shutdown()

    def test_deep_clones_game(self, transition_memory, agent_swarms, two_player_game):
        """Each job has independent game."""
        runner = SimulationRunner(transition_memory, num_workers=1)

        jobs = runner._make_jobs(agent_swarms, two_player_game, count=2, max_turns=10, prune_players=set())

        assert jobs[0].game is not jobs[1].game
        assert jobs[0].game is not two_player_game
        runner.shutdown()


class TestTransitionShutdown:
    """Shutdown tests with TransitionMemory."""

    def test_shutdown_no_pool(self, transition_memory):
        """Shutdown with no pool does nothing."""
        runner = SimulationRunner(transition_memory, num_workers=1)
        runner.shutdown()  # Should not raise

    def test_shutdown_idempotent(self, transition_memory):
        """Multiple shutdowns are safe."""
        runner = SimulationRunner(transition_memory, num_workers=1)

        runner.shutdown()
        runner.shutdown()  # Should not raise
        runner.shutdown(force=True)  # Should not raise

    def test_context_manager_cleans_up(self, transition_memory):
        """Context manager shuts down pool on exit."""
        runner = SimulationRunner(transition_memory, num_workers=1)

        with runner:
            assert runner._pool is not None

        assert runner._pool is None


@pytest.mark.slow
class TestTransitionIntegration:
    """Integration tests with TransitionMemory."""

    def test_run_batch(self, transition_memory, agent_swarms, two_player_game):
        """Run a small batch."""
        with SimulationRunner(transition_memory, num_workers=1) as runner:
            result = runner.run_batch(agent_swarms, two_player_game, 2, 10, set())

        assert result >= 0


# =============================================================================
# Markov Runner Tests
# =============================================================================

class TestMarkovCreation:
    """SimulationRunner creation with MarkovMemory."""

    def test_creation(self, markov_memory):
        """Runner can be created."""
        runner = SimulationRunner(markov_memory, num_workers=1)

        assert runner.memory is markov_memory
        assert runner.num_workers == 1
        assert runner._pool is None

        runner.shutdown()

    def test_default_workers(self, markov_memory):
        """Uses DEFAULT_WORKER_COUNT when not specified."""
        runner = SimulationRunner(markov_memory)

        assert runner.num_workers == DEFAULT_WORKER_COUNT

        runner.shutdown()


class TestMarkovContextManager:
    """Context manager tests with MarkovMemory."""

    def test_creates_pool(self, markov_memory):
        """Context manager creates pool on entry."""
        runner = SimulationRunner(markov_memory, num_workers=1)

        with runner:
            assert runner._pool is not None

        assert runner._pool is None


class TestMarkovRunBatchEdgeCases:
    """run_batch edge cases with MarkovMemory."""

    def test_zero_sims(self, markov_memory, agent_swarms, two_player_game):
        """Zero simulations returns 0."""
        with SimulationRunner(markov_memory, num_workers=1) as runner:
            result = runner.run_batch(agent_swarms, two_player_game, 0, 10, set())

        assert result == 0

    def test_empty_swarms(self, markov_memory, two_player_game):
        """Empty swarms returns 0."""
        with SimulationRunner(markov_memory, num_workers=1) as runner:
            result = runner.run_batch({}, two_player_game, 10, 10, set())

        assert result == 0


class TestMarkovMakeJobs:
    """_make_jobs tests with MarkovMemory."""

    def test_job_count(self, markov_memory, agent_swarms, two_player_game):
        """Creates correct number of jobs."""
        runner = SimulationRunner(markov_memory, num_workers=1)

        jobs = runner._make_jobs(agent_swarms, two_player_game, count=5, max_turns=10, prune_players=set())

        assert len(jobs) == 5
        runner.shutdown()

    def test_deep_clones_game(self, markov_memory, agent_swarms, two_player_game):
        """Each job has independent game."""
        runner = SimulationRunner(markov_memory, num_workers=1)

        jobs = runner._make_jobs(agent_swarms, two_player_game, count=2, max_turns=10, prune_players=set())

        assert jobs[0].game is not jobs[1].game
        runner.shutdown()


class TestMarkovShutdown:
    """Shutdown tests with MarkovMemory."""

    def test_shutdown_no_pool(self, markov_memory):
        """Shutdown with no pool does nothing."""
        runner = SimulationRunner(markov_memory, num_workers=1)
        runner.shutdown()

    def test_context_manager_cleans_up(self, markov_memory):
        """Context manager shuts down pool on exit."""
        runner = SimulationRunner(markov_memory, num_workers=1)

        with runner:
            assert runner._pool is not None

        assert runner._pool is None


@pytest.mark.slow
class TestMarkovIntegration:
    """Integration tests with MarkovMemory."""

    def test_run_batch(self, markov_memory, agent_swarms, two_player_game):
        """Run a small batch."""
        with SimulationRunner(markov_memory, num_workers=1) as runner:
            result = runner.run_batch(agent_swarms, two_player_game, 2, 10, set())

        assert result >= 0