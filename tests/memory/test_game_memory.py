"""
Tests for wise_explorer.memory

Tests TransitionMemory and MarkovMemory classes for storing game data.
"""

from pathlib import Path

import numpy as np
import pytest

from wise_explorer.agent.agent import State
from wise_explorer.core.types import Stats
from wise_explorer.games.game_base import GameBase
from wise_explorer.memory import TransitionMemory, MarkovMemory, GameMemory, for_game


# =============================================================================
# Shared Tests (both modes)
# =============================================================================

class TestCreation:
    """Memory creation tests."""

    def test_transition_creates_database(self, temp_db_path):
        """TransitionMemory creates database file."""
        mem = TransitionMemory(temp_db_path)
        assert temp_db_path.exists()
        mem.close()

    def test_markov_creates_database(self, temp_db_path):
        """MarkovMemory creates database file."""
        mem = MarkovMemory(temp_db_path)
        assert temp_db_path.exists()
        mem.close()

    def test_creates_parent_directory(self, temp_dir):
        """Creates parent directory if needed."""
        db_path = temp_dir / "subdir" / "memory.db"
        mem = TransitionMemory(db_path)
        assert db_path.parent.exists()
        mem.close()

    def test_transition_schema(self, transition_memory):
        """TransitionMemory creates transitions table."""
        result = transition_memory.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='transitions'"
        ).fetchone()
        assert result is not None

    def test_markov_schema(self, markov_memory):
        """MarkovMemory creates states table."""
        result = markov_memory.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='states'"
        ).fetchone()
        assert result is not None


class TestMarkovFlag:
    """Markov flag tests."""

    def test_transition_not_markov(self, transition_memory):
        """TransitionMemory has markov=False."""
        assert transition_memory.is_markov is False

    def test_markov_is_markov(self, markov_memory):
        """MarkovMemory has markov=True."""
        assert markov_memory.is_markov is True


class TestReadOnly:
    """Read-only mode tests for both types."""

    def test_transition_read_only(self, temp_db_path):
        """TransitionMemory read-only can read existing data."""
        mem = TransitionMemory(temp_db_path)
        mem.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, wins) VALUES ('a', 'b', 10)"
        )
        mem.conn.commit()
        mem.close()

        mem_ro = TransitionMemory(temp_db_path, read_only=True)
        result = mem_ro.conn.execute("SELECT wins FROM transitions WHERE from_hash='a'").fetchone()
        assert result[0] == 10
        mem_ro.close()

    def test_markov_read_only(self, temp_db_path):
        """MarkovMemory read-only can read existing data."""
        mem = MarkovMemory(temp_db_path)
        mem.conn.execute(
            "INSERT INTO states (state_hash, wins) VALUES ('abc', 10)"
        )
        mem.conn.commit()
        mem.close()

        mem_ro = MarkovMemory(temp_db_path, read_only=True)
        result = mem_ro.conn.execute("SELECT wins FROM states WHERE state_hash='abc'").fetchone()
        assert result[0] == 10
        mem_ro.close()

    def test_transition_record_read_only_raises(self, temp_db_path):
        """TransitionMemory record_round raises in read-only mode."""
        mem = TransitionMemory(temp_db_path)
        mem.close()

        mem_ro = TransitionMemory(temp_db_path, read_only=True)
        with pytest.raises(RuntimeError, match="read-only"):
            mem_ro.record_round(type(None), [])
        mem_ro.close()

    def test_markov_record_read_only_raises(self, temp_db_path):
        """MarkovMemory record_round raises in read-only mode."""
        mem = MarkovMemory(temp_db_path)
        mem.close()

        mem_ro = MarkovMemory(temp_db_path, read_only=True)
        with pytest.raises(RuntimeError, match="read-only"):
            mem_ro.record_round(type(None), [])
        mem_ro.close()


class TestContextManager:
    """Context manager tests."""

    def test_transition_context_manager(self, temp_db_path):
        """TransitionMemory can use as context manager."""
        with TransitionMemory(temp_db_path) as mem:
            mem.conn.execute(
                "INSERT INTO transitions (from_hash, to_hash, wins) VALUES ('a', 'b', 10)"
            )
            mem.conn.commit()

        with TransitionMemory(temp_db_path, read_only=True) as mem2:
            result = mem2.conn.execute("SELECT wins FROM transitions").fetchone()
            assert result[0] == 10

    def test_markov_context_manager(self, temp_db_path):
        """MarkovMemory can use as context manager."""
        with MarkovMemory(temp_db_path) as mem:
            mem.conn.execute(
                "INSERT INTO states (state_hash, wins) VALUES ('abc', 10)"
            )
            mem.conn.commit()

        with MarkovMemory(temp_db_path, read_only=True) as mem2:
            result = mem2.conn.execute("SELECT wins FROM states").fetchone()
            assert result[0] == 10


class TestForGame:
    """for_game function tests."""

    def test_transition_path(self, temp_dir, any_game):
        """Transition mode creates correct path."""
        mem = for_game(any_game, base_dir=temp_dir)
        assert mem.db_path.name == f"{any_game.game_id()}.db"
        assert isinstance(mem, TransitionMemory)
        mem.close()

    def test_markov_path(self, temp_dir, any_game):
        """Markov mode creates correct path."""
        mem = for_game(any_game, base_dir=temp_dir, markov=True)
        assert mem.db_path.name == f"{any_game.game_id()}_markov.db"
        assert isinstance(mem, MarkovMemory)
        mem.close()


class TestRecordRound:
    """record_round tests for both types."""

    def test_transition_records(self, transition_memory, any_game):
        """TransitionMemory stores transitions."""
        board_before = any_game.get_state().board.copy()
        move = any_game.valid_moves()[0]

        stacks = [([(move, board_before, 1)], State.WIN)]
        count_transitions, _count_swaps = transition_memory.record_round(type(any_game), stacks)

        assert count_transitions == 1
        assert transition_memory.get_info()["transitions"] == 1

    def test_markov_records(self, markov_memory, any_game):
        """MarkovMemory stores state values."""
        board_before = any_game.get_state().board.copy()
        move = any_game.valid_moves()[0]

        stacks = [([(move, board_before, 1)], State.WIN)]
        count_transitions, _count_swaps = markov_memory.record_round(type(any_game), stacks)

        assert count_transitions == 1
        assert markov_memory.get_info()["unique_states"] >= 1

    def test_transition_updates_existing(self, transition_memory, any_game):
        """TransitionMemory updates existing transitions."""
        board_before = any_game.get_state().board.copy()
        move = any_game.valid_moves()[0]

        stacks = [([(move, board_before, 1)], State.WIN)]
        transition_memory.record_round(type(any_game), stacks)
        transition_memory.record_round(type(any_game), stacks)

        assert transition_memory.get_info()["transitions"] == 1
        assert transition_memory.get_info()["total_samples"] == 2

    def test_markov_updates_existing(self, markov_memory, any_game):
        """MarkovMemory updates existing state values."""
        board_before = any_game.get_state().board.copy()
        move = any_game.valid_moves()[0]

        stacks = [([(move, board_before, 1)], State.WIN)]
        markov_memory.record_round(type(any_game), stacks)
        markov_memory.record_round(type(any_game), stacks)

        assert markov_memory.get_info()["total_samples"] == 2

    def test_skips_neutral(self, any_memory, any_game):
        """Both modes skip NEUTRAL outcomes."""
        board_before = any_game.get_state().board.copy()
        move = any_game.valid_moves()[0]

        stacks = [([(move, board_before, 1)], State.NEUTRAL)]
        count_states, _count_swaps = any_memory.record_round(type(any_game), stacks)

        assert count_states == 0


# =============================================================================
# Transition-Specific Tests
# =============================================================================

class TestTransitionStats:
    """TransitionMemory statistics tests."""

    def test_get_move_stats_empty(self, transition_memory):
        """Returns empty Stats for non-existent transition."""
        stats = transition_memory.get_move_stats("nonexistent", "also")
        assert stats == Stats(0, 0, 0)

    def test_get_move_stats_with_data(self, transition_memory):
        """Returns correct counts."""
        transition_memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, wins, ties, losses) "
            "VALUES ('a', 'b', 10, 5, 3)"
        )
        transition_memory.conn.commit()

        stats = transition_memory.get_move_stats("a", "b")
        assert (stats.wins, stats.ties, stats.losses) == (10, 5, 3)

    def test_get_transitions_from_empty(self, transition_memory):
        """Returns empty dict for unknown state."""
        assert transition_memory.get_transitions_from("unknown") == {}

    def test_get_transitions_from_with_data(self, transition_memory):
        """Returns all transitions from a state."""
        transition_memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, wins) VALUES ('a', 'b', 10)"
        )
        transition_memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, wins) VALUES ('a', 'c', 20)"
        )
        transition_memory.conn.commit()

        result = transition_memory.get_transitions_from("a")
        assert len(result) == 2
        assert result["b"].wins == 10
        assert result["c"].wins == 20


class TestTransitionInfo:
    """TransitionMemory get_info tests."""

    def test_empty(self, transition_memory):
        """Returns zeros for empty database."""
        info = transition_memory.get_info()
        assert info["mode"] == "transition"
        assert info["transitions"] == 0
        assert info["total_samples"] == 0
        assert info["anchors"] == 0

    def test_with_data(self, transition_memory):
        """Returns correct statistics."""
        transition_memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, wins, ties, losses) "
            "VALUES ('a', 'b', 10, 5, 3)"
        )
        transition_memory.conn.execute("INSERT INTO anchors VALUES (1, 'a→b', 10, 5, 3)")
        transition_memory.conn.commit()

        info = transition_memory.get_info()
        assert info["transitions"] == 1
        assert info["total_samples"] == 18
        assert info["anchors"] == 1


class TestTransitionAnchors:
    """TransitionMemory anchor stats tests."""

    def test_fallback_to_move_stats(self, transition_memory):
        """Falls back to move stats when no anchor."""
        transition_memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, wins, ties, losses) "
            "VALUES ('a', 'b', 10, 5, 3)"
        )
        transition_memory.conn.commit()

        stats = transition_memory.get_anchor_stats("a", "b")
        assert stats.wins == 10

    def test_uses_anchor_stats(self, transition_memory):
        """Uses anchor stats when available."""
        transition_memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, wins, anchor_id) "
            "VALUES ('a', 'b', 10, 1)"
        )
        transition_memory.conn.execute("INSERT INTO anchors VALUES (1, 'a→b', 100, 50, 30)")
        transition_memory.conn.commit()

        stats = transition_memory.get_anchor_stats("a", "b")
        assert stats.wins == 100


class TestTransitionCaching:
    """TransitionMemory cache tests."""

    def test_anchor_id_cache(self, transition_memory):
        """Anchor ID cache uses (from, to) tuple."""
        transition_memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, wins, anchor_id) VALUES ('a', 'b', 10, 1)"
        )
        transition_memory.conn.execute("INSERT INTO anchors VALUES (1, 'k', 10, 0, 0)")
        transition_memory.conn.commit()

        transition_memory.get_anchor_id("a", "b")
        assert ("a", "b") in transition_memory._anchor_id_cache

        result = transition_memory.get_anchor_id("a", "b")
        assert result == 1


# =============================================================================
# Markov-Specific Tests
# =============================================================================

class TestMarkovStats:
    """MarkovMemory statistics tests."""

    def test_get_state_stats_empty(self, markov_memory):
        """Returns empty Stats for non-existent state."""
        stats = markov_memory.get_state_stats("nonexistent")
        assert stats == Stats(0, 0, 0)

    def test_get_state_stats_with_data(self, markov_memory):
        """Returns correct counts."""
        markov_memory.conn.execute(
            "INSERT INTO states (state_hash, wins, ties, losses) VALUES ('abc', 10, 5, 3)"
        )
        markov_memory.conn.commit()

        stats = markov_memory.get_state_stats("abc")
        assert (stats.wins, stats.ties, stats.losses) == (10, 5, 3)

    def test_get_move_stats_ignores_from_hash(self, markov_memory):
        """get_move_stats only uses to_hash."""
        markov_memory.conn.execute(
            "INSERT INTO states (state_hash, wins, ties, losses) VALUES ('to_state', 10, 5, 3)"
        )
        markov_memory.conn.commit()

        stats = markov_memory.get_move_stats("any_from", "to_state")
        assert (stats.wins, stats.ties, stats.losses) == (10, 5, 3)


class TestMarkovInfo:
    """MarkovMemory get_info tests."""

    def test_empty(self, markov_memory):
        """Returns zeros for empty database."""
        info = markov_memory.get_info()
        assert info["mode"] == "markov"
        assert info["unique_states"] == 0
        assert info["total_samples"] == 0
        assert info["anchors"] == 0

    def test_with_data(self, markov_memory):
        """Returns correct statistics."""
        markov_memory.conn.execute(
            "INSERT INTO states (state_hash, wins, ties, losses) VALUES ('abc', 10, 5, 3)"
        )
        markov_memory.conn.execute("INSERT INTO anchors VALUES (1, 'abc', 10, 5, 3)")
        markov_memory.conn.commit()

        info = markov_memory.get_info()
        assert info["unique_states"] == 1
        assert info["total_samples"] == 18
        assert info["anchors"] == 1


class TestMarkovAnchors:
    """MarkovMemory anchor stats tests."""

    def test_fallback_to_state_stats(self, markov_memory):
        """Falls back to state stats when no anchor."""
        markov_memory.conn.execute(
            "INSERT INTO states (state_hash, wins, ties, losses) VALUES ('abc', 10, 5, 3)"
        )
        markov_memory.conn.commit()

        stats = markov_memory.get_anchor_stats("any_from", "abc")
        assert stats.wins == 10

    def test_uses_anchor_stats(self, markov_memory):
        """Uses anchor stats when available."""
        markov_memory.conn.execute(
            "INSERT INTO states (state_hash, wins, anchor_id) VALUES ('abc', 10, 1)"
        )
        markov_memory.conn.execute("INSERT INTO anchors VALUES (1, 'abc', 100, 50, 30)")
        markov_memory.conn.commit()

        stats = markov_memory.get_anchor_stats("any_from", "abc")
        assert stats.wins == 100


class TestMarkovCaching:
    """MarkovMemory cache tests."""

    def test_anchor_id_cache_uses_to_hash(self, markov_memory):
        """Anchor ID cache keys on to_hash only."""
        markov_memory.conn.execute(
            "INSERT INTO states (state_hash, wins, anchor_id) VALUES ('abc', 10, 1)"
        )
        markov_memory.conn.execute("INSERT INTO anchors VALUES (1, 'abc', 10, 0, 0)")
        markov_memory.conn.commit()

        markov_memory.get_anchor_id("any_from", "abc")
        assert "abc" in markov_memory._anchor_id_cache

        # Different from_hash, same to_hash — should hit cache
        result = markov_memory.get_anchor_id("different_from", "abc")
        assert result == 1