"""
Tests for wise_explorer.simulation.runner

Tests SimulationRunner for parallel game execution.
"""
import pytest

from wise_explorer.simulation.runner import SimulationRunner, DEFAULT_WORKER_COUNT


class TestDefaultWorkerCount:
    """DEFAULT_WORKER_COUNT tests."""

    def test_positive(self):
        """At least 1 worker."""
        assert DEFAULT_WORKER_COUNT >= 1


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

    def test_default_workers(self, transition_memory):
        """Uses DEFAULT_WORKER_COUNT when not specified."""
        runner = SimulationRunner(transition_memory)
        assert runner.num_workers == DEFAULT_WORKER_COUNT


class TestTransitionContextManager:
    """Context manager tests with TransitionMemory."""

    def test_creates_pool(self, transition_memory):
        """Context manager creates pool on entry."""
        runner = SimulationRunner(transition_memory, num_workers=1)

        with runner:
            assert runner._pool is not None

    def test_handles_exception(self, transition_memory):
        """Context manager handles exceptions."""
        runner = SimulationRunner(transition_memory, num_workers=1)

        with pytest.raises(ValueError):
            with runner:
                raise ValueError("Test")


class TestTransitionRunBatchEdgeCases:
    """run_batch edge cases with TransitionMemory."""

    def test_zero_sims(self, transition_memory, agent_swarms, two_player_game):
        """Zero simulations returns 0."""
        runner = SimulationRunner(transition_memory, num_workers=1)
        result = runner.run_batch(agent_swarms, two_player_game, 0, 10, set())
        assert result == 0

    def test_empty_swarms(self, transition_memory, two_player_game):
        """Empty swarms returns 0."""
        runner = SimulationRunner(transition_memory, num_workers=1)
        result = runner.run_batch({}, two_player_game, 10, 10, set())
        assert result == 0


class TestTransitionMakeJobs:
    """_make_jobs tests with TransitionMemory."""

    def test_job_count(self, transition_memory, agent_swarms, two_player_game):
        """Creates correct number of jobs."""
        runner = SimulationRunner(transition_memory, num_workers=1)
        jobs = runner._make_jobs(agent_swarms, two_player_game, count=5, max_turns=10, prune_players=set())
        assert len(jobs) == 5

    def test_job_structure(self, transition_memory, agent_swarms, two_player_game):
        """Jobs have correct structure."""
        runner = SimulationRunner(transition_memory, num_workers=1)
        jobs = runner._make_jobs(agent_swarms, two_player_game, count=1, max_turns=15, prune_players={1})

        job = jobs[0]
        assert job.max_turns == 15
        assert job.prune_players == {1}

    def test_deep_clones_game(self, transition_memory, agent_swarms, two_player_game):
        """Each job has independent game."""
        runner = SimulationRunner(transition_memory, num_workers=1)
        jobs = runner._make_jobs(agent_swarms, two_player_game, count=2, max_turns=10, prune_players=set())

        assert jobs[0].game is not jobs[1].game
        assert jobs[0].game is not two_player_game


class TestTransitionShutdown:
    """Shutdown tests with TransitionMemory."""

    def test_shutdown_no_pool(self, transition_memory):
        """Shutdown with no pool does nothing."""
        runner = SimulationRunner(transition_memory, num_workers=1)
        runner.shutdown()

    def test_shutdown_force(self, transition_memory):
        """Force shutdown terminates pool."""
        runner = SimulationRunner(transition_memory, num_workers=1)
        runner._ensure_pool()
        runner.shutdown(force=True)
        assert runner._pool is None


@pytest.mark.slow
class TestTransitionIntegration:
    """Integration tests with TransitionMemory."""

    def test_run_batch(self, transition_memory, agent_swarms, two_player_game):
        """Run a small batch."""
        runner = SimulationRunner(transition_memory, num_workers=1)

        with runner:
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

    def test_default_workers(self, markov_memory):
        """Uses DEFAULT_WORKER_COUNT when not specified."""
        runner = SimulationRunner(markov_memory)
        assert runner.num_workers == DEFAULT_WORKER_COUNT


class TestMarkovContextManager:
    """Context manager tests with MarkovMemory."""

    def test_creates_pool(self, markov_memory):
        """Context manager creates pool on entry."""
        runner = SimulationRunner(markov_memory, num_workers=1)

        with runner:
            assert runner._pool is not None

    def test_handles_exception(self, markov_memory):
        """Context manager handles exceptions."""
        runner = SimulationRunner(markov_memory, num_workers=1)

        with pytest.raises(ValueError):
            with runner:
                raise ValueError("Test")


class TestMarkovRunBatchEdgeCases:
    """run_batch edge cases with MarkovMemory."""

    def test_zero_sims(self, markov_memory, agent_swarms, two_player_game):
        """Zero simulations returns 0."""
        runner = SimulationRunner(markov_memory, num_workers=1)
        result = runner.run_batch(agent_swarms, two_player_game, 0, 10, set())
        assert result == 0

    def test_empty_swarms(self, markov_memory, two_player_game):
        """Empty swarms returns 0."""
        runner = SimulationRunner(markov_memory, num_workers=1)
        result = runner.run_batch({}, two_player_game, 10, 10, set())
        assert result == 0


class TestMarkovMakeJobs:
    """_make_jobs tests with MarkovMemory."""

    def test_job_count(self, markov_memory, agent_swarms, two_player_game):
        """Creates correct number of jobs."""
        runner = SimulationRunner(markov_memory, num_workers=1)
        jobs = runner._make_jobs(agent_swarms, two_player_game, count=5, max_turns=10, prune_players=set())
        assert len(jobs) == 5

    def test_deep_clones_game(self, markov_memory, agent_swarms, two_player_game):
        """Each job has independent game."""
        runner = SimulationRunner(markov_memory, num_workers=1)
        jobs = runner._make_jobs(agent_swarms, two_player_game, count=2, max_turns=10, prune_players=set())

        assert jobs[0].game is not jobs[1].game


class TestMarkovShutdown:
    """Shutdown tests with MarkovMemory."""

    def test_shutdown_no_pool(self, markov_memory):
        """Shutdown with no pool does nothing."""
        runner = SimulationRunner(markov_memory, num_workers=1)
        runner.shutdown()

    def test_shutdown_force(self, markov_memory):
        """Force shutdown terminates pool."""
        runner = SimulationRunner(markov_memory, num_workers=1)
        runner._ensure_pool()
        runner.shutdown(force=True)
        assert runner._pool is None


@pytest.mark.slow
class TestMarkovIntegration:
    """Integration tests with MarkovMemory."""

    def test_run_batch(self, markov_memory, agent_swarms, two_player_game):
        """Run a small batch."""
        runner = SimulationRunner(markov_memory, num_workers=1)

        with runner:
            result = runner.run_batch(agent_swarms, two_player_game, 2, 10, set())

        assert result >= 0