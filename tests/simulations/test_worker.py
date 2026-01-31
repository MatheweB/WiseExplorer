"""
Tests for wise_explorer.simulation.worker

Tests worker process logic for parallel simulation.
"""

import numpy as np
import pytest

from wise_explorer.agent.agent import State
from wise_explorer.games.game_base import GameBase
from wise_explorer.memory import TransitionMemory, MarkovMemory, open_readonly
from wise_explorer.simulation.jobs import GameJob, JobResult
from wise_explorer.simulation import worker
from wise_explorer.simulation.worker import worker_init, run_game


# =============================================================================
# Worker Init Tests
# =============================================================================

class TestTransitionWorkerInit:
    """worker_init with TransitionMemory."""

    def test_creates_read_only_memory(self, temp_db_path):
        """worker_init creates read-only TransitionMemory."""
        mem = TransitionMemory(temp_db_path)
        mem.close()

        worker_init(str(temp_db_path), is_markov=False)

        assert worker._worker_memory is not None
        assert worker._worker_memory.read_only is True
        assert isinstance(worker._worker_memory, TransitionMemory)

        worker._worker_memory.close()
        worker._worker_memory = None


class TestMarkovWorkerInit:
    """worker_init with MarkovMemory."""

    def test_creates_read_only_memory(self, temp_db_path):
        """worker_init creates read-only MarkovMemory."""
        mem = MarkovMemory(temp_db_path)
        mem.close()

        worker_init(str(temp_db_path), is_markov=True)

        assert worker._worker_memory is not None
        assert worker._worker_memory.read_only is True
        assert isinstance(worker._worker_memory, MarkovMemory)

        worker._worker_memory.close()
        worker._worker_memory = None


# =============================================================================
# open_readonly Tests
# =============================================================================

class TestOpenReadonly:
    """open_readonly function tests."""

    def test_opens_transition_memory(self, temp_db_path):
        """Returns read-only TransitionMemory."""
        mem = TransitionMemory(temp_db_path)
        mem.close()

        mem_ro = open_readonly(temp_db_path, is_markov=False)
        assert isinstance(mem_ro, TransitionMemory)
        assert mem_ro.read_only is True
        mem_ro.close()

    def test_opens_markov_memory(self, temp_db_path):
        """Returns read-only MarkovMemory."""
        mem = MarkovMemory(temp_db_path)
        mem.close()

        mem_ro = open_readonly(temp_db_path, is_markov=True)
        assert isinstance(mem_ro, MarkovMemory)
        assert mem_ro.read_only is True
        mem_ro.close()


# =============================================================================
# Uninitialized Tests
# =============================================================================

class TestRunGameUninitialized:
    """run_game without initialization tests."""

    def test_raises_if_not_initialized(self, any_game: GameBase):
        """Raises RuntimeError if worker not initialized."""
        original = worker._worker_memory
        worker._worker_memory = None

        try:
            job = GameJob(any_game, {1: 0, 2: 1}, 10, set())
            with pytest.raises(RuntimeError, match="Worker not initialized"):
                run_game(job)
        finally:
            worker._worker_memory = original


# =============================================================================
# Worker Fixtures
# =============================================================================

@pytest.fixture
def transition_worker(temp_db_path):
    """Initialize worker with TransitionMemory."""
    mem = TransitionMemory(temp_db_path)
    mem.close()

    worker_init(str(temp_db_path), is_markov=False)
    yield

    if worker._worker_memory:
        worker._worker_memory.close()
        worker._worker_memory = None


@pytest.fixture
def markov_worker(temp_db_path):
    """Initialize worker with MarkovMemory."""
    mem = MarkovMemory(temp_db_path)
    mem.close()

    worker_init(str(temp_db_path), is_markov=True)
    yield

    if worker._worker_memory:
        worker._worker_memory.close()
        worker._worker_memory = None


# =============================================================================
# Transition Worker Game Tests
# =============================================================================

@pytest.mark.usefixtures("transition_worker")
class TestTransitionRunGame:
    """run_game with TransitionMemory worker."""

    def test_returns_job_result(self, two_player_game: GameBase):
        """Returns JobResult."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, set())
        result = run_game(job)
        assert isinstance(result, JobResult)

    def test_records_moves(self, two_player_game: GameBase):
        """Records moves for each player."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, set())
        result = run_game(job)

        assert 1 in result.moves
        assert 2 in result.moves

    def test_records_outcomes(self, two_player_game: GameBase):
        """Records outcomes for each player."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, set())
        result = run_game(job)

        assert 1 in result.outcomes
        assert 2 in result.outcomes
        assert isinstance(result.outcomes[1], State)

    def test_stores_game_class(self, two_player_game: GameBase):
        """Stores game class in result."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, set())
        result = run_game(job)

        assert result.game_class is type(two_player_game)

    def test_respects_max_turns(self, two_player_game: GameBase):
        """Stops at max_turns."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, max_turns=3, prune_players=set())
        result = run_game(job)

        total_moves = sum(len(moves) for moves in result.moves.values())
        assert total_moves <= 3

    def test_handles_prune_players(self, two_player_game: GameBase):
        """Handles prune players without error."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, prune_players={1})
        result = run_game(job)

        assert result is not None


# =============================================================================
# Markov Worker Game Tests
# =============================================================================

@pytest.mark.usefixtures("markov_worker")
class TestMarkovRunGame:
    """run_game with MarkovMemory worker."""

    def test_returns_job_result(self, two_player_game: GameBase):
        """Returns JobResult."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, set())
        result = run_game(job)
        assert isinstance(result, JobResult)

    def test_records_moves(self, two_player_game: GameBase):
        """Records moves for each player."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, set())
        result = run_game(job)

        assert 1 in result.moves
        assert 2 in result.moves

    def test_records_outcomes(self, two_player_game: GameBase):
        """Records outcomes for each player."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, set())
        result = run_game(job)

        assert 1 in result.outcomes
        assert 2 in result.outcomes
        assert isinstance(result.outcomes[1], State)

    def test_stores_game_class(self, two_player_game: GameBase):
        """Stores game class in result."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, set())
        result = run_game(job)

        assert result.game_class is type(two_player_game)

    def test_respects_max_turns(self, two_player_game: GameBase):
        """Stops at max_turns."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, max_turns=3, prune_players=set())
        result = run_game(job)

        total_moves = sum(len(moves) for moves in result.moves.values())
        assert total_moves <= 3


# =============================================================================
# MoveRecord Tests (mode-independent, uses transition worker)
# =============================================================================

@pytest.mark.usefixtures("transition_worker")
class TestMoveRecordContents:
    """MoveRecord contents tests."""

    def test_move_is_array(self, two_player_game: GameBase):
        """MoveRecords have move arrays."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, set())
        result = run_game(job)

        for moves in result.moves.values():
            for record in moves:
                assert isinstance(record.move, np.ndarray)

    def test_board_before_is_array(self, two_player_game: GameBase):
        """MoveRecords have board_before arrays."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, set())
        result = run_game(job)

        for moves in result.moves.values():
            for record in moves:
                assert isinstance(record.board_before, np.ndarray)

    def test_player_matches_key(self, two_player_game: GameBase):
        """MoveRecord player matches moves dict key."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, set())
        result = run_game(job)

        for pid, moves in result.moves.items():
            for record in moves:
                assert record.player == pid