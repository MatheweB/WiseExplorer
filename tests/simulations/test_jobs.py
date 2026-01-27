"""
Tests for wise_explorer.simulation.jobs

Tests job data structures for parallel simulation.
"""

import numpy as np
import pytest

from wise_explorer.agent.agent import State
from wise_explorer.games.game_base import GameBase
from wise_explorer.simulation.jobs import MoveRecord, GameJob, JobResult


class TestMoveRecord:
    """MoveRecord dataclass tests."""

    def test_creation(self):
        """MoveRecord stores move, board, and player."""
        move = np.array([0, 0])
        board = np.zeros((3, 3), dtype=np.int8)
        
        record = MoveRecord(move=move, board_before=board, player=1)
        
        np.testing.assert_array_equal(record.move, move)
        np.testing.assert_array_equal(record.board_before, board)
        assert record.player == 1


class TestGameJob:
    """GameJob dataclass tests."""

    def test_creation(self, any_game: GameBase):
        """GameJob can be created with all fields."""
        job = GameJob(
            game=any_game,
            player_map={1: 0, 2: 1},
            max_turns=20,
            prune_players={1},
        )
        
        assert job.game is any_game
        assert job.player_map == {1: 0, 2: 1}
        assert job.max_turns == 20
        assert job.prune_players == {1}

    def test_frozen(self, any_game: GameBase):
        """GameJob is immutable."""
        job = GameJob(
            game=any_game,
            player_map={1: 0},
            max_turns=20,
            prune_players=set(),
        )
        
        with pytest.raises(AttributeError):
            job.max_turns = 100


class TestJobResult:
    """JobResult dataclass tests."""

    def test_creation(self, any_game: GameBase):
        """JobResult stores moves, outcomes, and game class."""
        moves = {
            1: [MoveRecord(np.array([0, 0]), np.zeros((3, 3), dtype=np.int8), 1)],
            2: [],
        }
        outcomes = {1: State.WIN, 2: State.LOSS}
        
        result = JobResult(
            moves=moves,
            outcomes=outcomes,
            player_map={1: 0, 2: 1},
            game_class=type(any_game),
        )
        
        assert len(result.moves[1]) == 1
        assert result.outcomes[1] == State.WIN
        assert result.game_class is type(any_game)

    def test_mutable(self, any_game: GameBase):
        """JobResult moves can be modified."""
        result = JobResult(
            moves={1: [], 2: []},
            outcomes={1: State.NEUTRAL, 2: State.NEUTRAL},
            player_map={1: 0, 2: 1},
            game_class=type(any_game),
        )
        
        result.moves[1].append(
            MoveRecord(np.array([1, 1]), np.zeros((3, 3), dtype=np.int8), 1)
        )
        assert len(result.moves[1]) == 1