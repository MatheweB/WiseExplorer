"""
Tests for wise_explorer.simulation.worker

Tests worker process logic for parallel simulation.
"""

import numpy as np
import pytest

from wise_explorer.agent.agent import State
from wise_explorer.games.game_base import GameBase
from wise_explorer.memory.game_memory import GameMemory
from wise_explorer.simulation.jobs import GameJob, JobResult
from wise_explorer.simulation import worker
from wise_explorer.simulation.worker import worker_init, run_game


class TestWorkerInit:
    """worker_init function tests."""

    def test_creates_read_only_memory(self, temp_db_path):
        """worker_init creates read-only GameMemory."""
        # Create database first
        mem = GameMemory(temp_db_path)
        mem.close()
        
        worker_init(str(temp_db_path))
        
        assert worker._worker_memory is not None
        assert worker._worker_memory.read_only is True
        
        worker._worker_memory.close()
        worker._worker_memory = None


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


@pytest.fixture
def initialized_worker(temp_db_path):
    """Initialize worker for testing."""
    mem = GameMemory(temp_db_path)
    mem.close()
    
    worker_init(str(temp_db_path))
    yield
    
    if worker._worker_memory:
        worker._worker_memory.close()
        worker._worker_memory = None


class TestRunGame:
    """run_game function tests."""

    def test_returns_job_result(self, initialized_worker, two_player_game: GameBase):
        """Returns JobResult."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, set())
        result = run_game(job)
        assert isinstance(result, JobResult)

    def test_records_moves(self, initialized_worker, two_player_game: GameBase):
        """Records moves for each player."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, set())
        result = run_game(job)
        
        assert 1 in result.moves
        assert 2 in result.moves

    def test_records_outcomes(self, initialized_worker, two_player_game: GameBase):
        """Records outcomes for each player."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, set())
        result = run_game(job)
        
        assert 1 in result.outcomes
        assert 2 in result.outcomes
        assert isinstance(result.outcomes[1], State)

    def test_stores_game_class(self, initialized_worker, two_player_game: GameBase):
        """Stores game class in result."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, set())
        result = run_game(job)
        
        assert result.game_class is type(two_player_game)

    def test_respects_max_turns(self, initialized_worker, two_player_game: GameBase):
        """Stops at max_turns."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, max_turns=3, prune_players=set())
        result = run_game(job)
        
        total_moves = sum(len(moves) for moves in result.moves.values())
        assert total_moves <= 3

    def test_handles_prune_players(self, initialized_worker, two_player_game: GameBase):
        """Handles prune players without error."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, prune_players={1})
        result = run_game(job)
        
        assert result is not None


class TestMoveRecordContents:
    """MoveRecord contents tests."""

    def test_move_is_array(self, initialized_worker, two_player_game: GameBase):
        """MoveRecords have move arrays."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, set())
        result = run_game(job)
        
        for moves in result.moves.values():
            for record in moves:
                assert isinstance(record.move, np.ndarray)

    def test_board_before_is_array(self, initialized_worker, two_player_game: GameBase):
        """MoveRecords have board_before arrays."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, set())
        result = run_game(job)
        
        for moves in result.moves.values():
            for record in moves:
                assert isinstance(record.board_before, np.ndarray)

    def test_player_matches_key(self, initialized_worker, two_player_game: GameBase):
        """MoveRecord player matches moves dict key."""
        job = GameJob(two_player_game.deep_clone(), {1: 0, 2: 1}, 20, set())
        result = run_game(job)
        
        for pid, moves in result.moves.items():
            for record in moves:
                assert record.player == pid