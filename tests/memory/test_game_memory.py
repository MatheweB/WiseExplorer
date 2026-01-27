"""
Tests for wise_explorer.memory.game_memory

Tests GameMemory class for storing game transitions.
"""

from pathlib import Path

import numpy as np
import pytest

from wise_explorer.agent.agent import State
from wise_explorer.core.types import Stats
from wise_explorer.games.game_base import GameBase
from wise_explorer.memory.game_memory import GameMemory


class TestCreation:
    """GameMemory creation tests."""

    def test_creates_database(self, temp_db_path):
        """Creates database file on disk."""
        mem = GameMemory(temp_db_path)
        assert temp_db_path.exists()
        mem.close()

    def test_creates_parent_directory(self, temp_dir):
        """Creates parent directory if needed."""
        db_path = temp_dir / "subdir" / "memory.db"
        mem = GameMemory(db_path)
        assert db_path.parent.exists()
        mem.close()

    def test_initializes_schema(self, memory):
        """Initializes database schema."""
        result = memory.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='transitions'"
        ).fetchone()
        assert result is not None


class TestReadOnly:
    """Read-only mode tests."""

    def test_read_only_can_read(self, temp_db_path):
        """Read-only mode can read existing data."""
        mem = GameMemory(temp_db_path)
        mem.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, scoring_key, wins) VALUES ('a', 'b', 'a|b', 10)"
        )
        mem.conn.commit()
        mem.close()
        
        mem_ro = GameMemory(temp_db_path, read_only=True)
        result = mem_ro.conn.execute("SELECT wins FROM transitions WHERE from_hash='a'").fetchone()
        assert result[0] == 10
        mem_ro.close()

    def test_record_in_read_only_raises(self, temp_db_path):
        """record_round raises in read-only mode."""
        mem = GameMemory(temp_db_path)
        mem.close()
        
        mem_ro = GameMemory(temp_db_path, read_only=True)
        with pytest.raises(RuntimeError, match="read-only"):
            mem_ro.record_round(type(None), [])
        mem_ro.close()


class TestScoringKey:
    """Scoring key tests."""

    def test_non_markov_key(self, memory):
        """Non-Markov mode uses from|to as scoring key."""
        assert memory._scoring_key("from", "to") == "from|to"

    def test_markov_key(self, temp_db_path):
        """Markov mode uses to_hash as scoring key."""
        mem = GameMemory(temp_db_path, markov=True)
        assert mem._scoring_key("from", "to") == "to"
        mem.close()


class TestGetStats:
    """Statistics retrieval tests."""

    def test_get_stats_empty(self, memory):
        """get_stats returns empty Stats for non-existent transition."""
        stats = memory.get_stats("nonexistent", "also_nonexistent")
        assert stats == Stats(0, 0, 0)

    def test_get_stats_with_data(self, memory):
        """get_stats returns correct counts."""
        memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, scoring_key, wins, ties, losses) "
            "VALUES ('a', 'b', 'a|b', 10, 5, 3)"
        )
        memory.conn.commit()
        
        stats = memory.get_stats("a", "b")
        assert (stats.wins, stats.ties, stats.losses) == (10, 5, 3)


class TestGetTransitionsFrom:
    """get_transitions_from tests."""

    def test_empty(self, memory):
        """Returns empty dict for unknown state."""
        assert memory.get_transitions_from("unknown") == {}

    def test_with_data(self, memory):
        """Returns all transitions from a state."""
        memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, scoring_key, wins) VALUES ('a', 'b', 'a|b', 10)"
        )
        memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, scoring_key, wins) VALUES ('a', 'c', 'a|c', 20)"
        )
        memory.conn.commit()
        
        result = memory.get_transitions_from("a")
        assert len(result) == 2
        assert result["b"].wins == 10
        assert result["c"].wins == 20


class TestGetInfo:
    """get_info tests."""

    def test_empty(self, memory):
        """Returns zeros for empty database."""
        info = memory.get_info()
        assert info["transitions"] == 0
        assert info["total_samples"] == 0
        assert info["anchors"] == 0

    def test_with_data(self, memory):
        """Returns correct statistics."""
        memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, scoring_key, wins, ties, losses) "
            "VALUES ('a', 'b', 'a|b', 10, 5, 3)"
        )
        memory.conn.execute("INSERT INTO anchors VALUES (1, 'a|b', 10, 5, 3)")
        memory.conn.commit()
        
        info = memory.get_info()
        assert info["transitions"] == 1
        assert info["total_samples"] == 18
        assert info["anchors"] == 1


class TestRecordRound:
    """record_round tests."""

    def test_records_transitions(self, memory, any_game):
        """record_round stores transitions."""
        game = any_game
        board_before = game.get_state().board.copy()
        move = game.valid_moves()[0]
        
        stacks = [([(move, board_before, 1)], State.WIN)]
        count = memory.record_round(type(game), stacks)
        
        assert count == 1
        assert memory.get_info()["transitions"] == 1

    def test_updates_existing(self, memory, any_game):
        """record_round updates existing transitions."""
        game = any_game
        board_before = game.get_state().board.copy()
        move = game.valid_moves()[0]
        
        stacks = [([(move, board_before, 1)], State.WIN)]
        memory.record_round(type(game), stacks)
        memory.record_round(type(game), stacks)
        
        assert memory.get_info()["transitions"] == 1
        assert memory.get_info()["total_samples"] == 2

    def test_skips_neutral(self, memory, any_game):
        """record_round skips NEUTRAL outcomes."""
        game = any_game
        board_before = game.get_state().board.copy()
        move = game.valid_moves()[0]
        
        stacks = [([(move, board_before, 1)], State.NEUTRAL)]
        count = memory.record_round(type(game), stacks)
        
        assert count == 0


class TestAnchorStats:
    """Anchor-aware statistics tests."""

    def test_fallback_to_unit_stats(self, memory):
        """Falls back to unit stats when no anchor."""
        memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, scoring_key, wins, ties, losses) "
            "VALUES ('a', 'b', 'a|b', 10, 5, 3)"
        )
        memory.conn.commit()
        
        stats = memory.get_anchor_stats("a|b")
        assert stats.wins == 10

    def test_uses_anchor_stats(self, memory):
        """Uses anchor stats when available."""
        memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, scoring_key, wins, anchor_id) "
            "VALUES ('a', 'b', 'a|b', 10, 1)"
        )
        memory.conn.execute("INSERT INTO anchors VALUES (1, 'a|b', 100, 50, 30)")
        memory.conn.execute("INSERT INTO scoring_anchors VALUES ('a|b', 1)")
        memory.conn.commit()
        
        stats = memory.get_anchor_stats("a|b")
        assert stats.wins == 100  # Anchor stats


class TestForGame:
    """for_game class method tests."""

    def test_creates_correct_path(self, temp_dir, any_game):
        """Creates database with game-based name."""
        mem = GameMemory.for_game(any_game, base_dir=temp_dir)
        
        expected_name = f"{any_game.game_id()}.db"
        assert mem.db_path.name == expected_name
        
        mem.close()


class TestContextManager:
    """Context manager tests."""

    def test_context_manager(self, temp_db_path):
        """Can use as context manager."""
        with GameMemory(temp_db_path) as mem:
            mem.conn.execute(
                "INSERT INTO transitions (from_hash, to_hash, scoring_key, wins) "
                "VALUES ('a', 'b', 'a|b', 10)"
            )
            mem.conn.commit()
        
        # Verify data persisted
        with GameMemory(temp_db_path, read_only=True) as mem2:
            result = mem2.conn.execute("SELECT wins FROM transitions").fetchone()
            assert result[0] == 10


class TestCaching:
    """Cache behavior tests."""

    def test_transition_cache(self, memory):
        """Transition cache works."""
        memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, scoring_key, wins) VALUES ('a', 'b', 'a|b', 10)"
        )
        memory.conn.commit()
        
        memory.get_transitions_from("a")  # Populate cache
        assert "a" in memory._transition_cache
        
        result = memory.get_transitions_from("a")  # Use cache
        assert result["b"].wins == 10