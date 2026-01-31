"""
Tests for wise_explorer.memory.anchor_manager

Tests anchor clustering system for grouping similar moves.
"""

import pytest

from wise_explorer.core.types import Stats
from wise_explorer.memory.anchor_manager import Anchor, AnchorManager, _sub_counts
from wise_explorer.memory import TransitionMemory, MarkovMemory


# =============================================================================
# Unit Tests (mode-independent)
# =============================================================================

class TestAnchor:
    """Anchor dataclass tests."""

    def test_creation(self):
        """Anchor can be created with counts and key."""
        anchor = Anchor((10, 5, 5), "test_key")
        assert anchor.counts == (10, 5, 5)
        assert anchor.repr_key == "test_key"

    def test_total(self):
        """Total sums counts."""
        assert Anchor((10, 20, 30), "k").total == 60
        assert Anchor((0, 0, 0), "k").total == 0

    def test_add(self):
        """Add increments counts."""
        anchor = Anchor((10, 5, 5), "k")
        anchor.add((5, 3, 2))
        assert anchor.counts == (15, 8, 7)

    def test_without(self):
        """Without returns subtracted counts without modifying original."""
        anchor = Anchor((10, 5, 5), "k")
        result = anchor.without((3, 2, 1))
        assert result == (7, 3, 4)
        assert anchor.counts == (10, 5, 5)


class TestSubCounts:
    """_sub_counts helper tests."""

    def test_basic(self):
        """Subtracts element-wise."""
        assert _sub_counts((10, 5, 5), (3, 2, 1)) == (7, 3, 4)

    def test_to_zero(self):
        """Can result in zeros."""
        assert _sub_counts((10, 5, 5), (10, 5, 5)) == (0, 0, 0)


# =============================================================================
# Transition Anchor Tests
# =============================================================================

class TestTransitionAnchorQueries:
    """AnchorManager query tests with TransitionMemory."""

    @pytest.fixture
    def memory_with_anchors(self, temp_db_path):
        """TransitionMemory with pre-populated anchors."""
        mem = TransitionMemory(temp_db_path)

        mem.conn.execute("INSERT INTO anchors VALUES (1, 'key1', 100, 50, 25)")
        mem.conn.execute("INSERT INTO anchors VALUES (2, 'key2', 10, 10, 10)")
        mem.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, wins, ties, losses, anchor_id) "
            "VALUES ('a', 'b', 50, 25, 12, 1)"
        )
        mem.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, wins, ties, losses, anchor_id) "
            "VALUES ('c', 'd', 10, 10, 10, 2)"
        )
        mem.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, wins, ties, losses, anchor_id) "
            "VALUES ('e', 'f', 50, 25, 13, 1)"
        )
        mem.conn.commit()

        yield mem
        mem.close()

    def test_get_anchor_id(self, memory_with_anchors):
        """get_anchor_id returns correct ID or None."""
        mem = memory_with_anchors
        assert mem.get_anchor_id("a", "b") == 1
        assert mem.get_anchor_id("c", "d") == 2
        assert mem.get_anchor_id("nonexistent", "also") is None

    def test_get_anchor_stats(self, memory_with_anchors):
        """get_anchor_stats returns pooled stats."""
        mem = memory_with_anchors
        stats = mem.get_anchor_stats("a", "b")
        assert stats.wins == 100
        assert stats.ties == 50
        assert stats.losses == 25

    def test_get_details(self, memory_with_anchors):
        """get_details returns anchor information."""
        details = memory_with_anchors._anchors.get_details()
        assert len(details) == 2

        a1 = next(d for d in details if d["anchor_id"] == 1)
        assert a1["wins"] == 100
        assert a1["members"] == 2


class TestTransitionAnchorRebuild:
    """AnchorManager rebuild tests with TransitionMemory."""

    def test_rebuild_empty_returns_zero(self, transition_memory):
        """Rebuild on empty database returns 0."""
        assert transition_memory._anchors.rebuild() == 0

    def test_rebuild_creates_anchors(self, transition_memory):
        """Rebuild creates anchors from existing data."""
        transition_memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, wins, ties, losses) "
            "VALUES ('a', 'b', 10, 5, 5)"
        )
        transition_memory.conn.commit()

        num = transition_memory._anchors.rebuild()
        assert num > 0

    def test_rebuild_read_only_raises(self, temp_db_path):
        """Rebuild raises in read-only mode."""
        mem = TransitionMemory(temp_db_path)
        mem.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, wins) VALUES ('a', 'b', 10)"
        )
        mem.conn.commit()
        mem.close()

        mem_ro = TransitionMemory(temp_db_path, read_only=True)
        with pytest.raises(RuntimeError, match="read-only"):
            mem_ro._anchors.rebuild()
        mem_ro.close()


class TestTransitionAnchorConsolidate:
    """AnchorManager consolidate tests with TransitionMemory."""

    def test_keeps_incompatible(self, transition_memory):
        """Consolidate does not merge incompatible anchors."""
        transition_memory.conn.execute("INSERT INTO anchors VALUES (1, 'k1', 100, 0, 0)")
        transition_memory.conn.execute("INSERT INTO anchors VALUES (2, 'k2', 0, 0, 100)")
        transition_memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, wins, anchor_id) VALUES ('a', 'b', 100, 1)"
        )
        transition_memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, losses, anchor_id) VALUES ('c', 'd', 100, 2)"
        )
        transition_memory.conn.commit()

        transition_memory._anchors.consolidate()

        count = transition_memory.conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
        assert count == 2

    def test_read_only_returns_zero(self, temp_db_path):
        """Consolidate returns 0 in read-only mode."""
        mem = TransitionMemory(temp_db_path)
        mem.conn.execute("INSERT INTO anchors VALUES (1, 'k1', 100, 50, 50)")
        mem.conn.commit()
        mem.close()

        mem_ro = TransitionMemory(temp_db_path, read_only=True)
        assert mem_ro._anchors.consolidate() == 0
        mem_ro.close()


class TestTransitionClustering:
    """Clustering behavior tests with TransitionMemory."""

    def test_identical_distributions_cluster(self, transition_memory):
        """Identical distributions cluster together."""
        for i in range(5):
            transition_memory.conn.execute(
                f"INSERT INTO transitions (from_hash, to_hash, wins, ties, losses) "
                f"VALUES ('a{i}', 'b{i}', 100, 50, 50)"
            )
        transition_memory.conn.commit()

        num = transition_memory._anchors.rebuild()
        assert num == 1

    def test_different_distributions_separate(self, transition_memory):
        """Very different distributions stay separate."""
        transition_memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, wins, ties, losses) "
            "VALUES ('a', 'b', 100, 0, 0)"
        )
        transition_memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, wins, ties, losses) "
            "VALUES ('c', 'd', 0, 0, 100)"
        )
        transition_memory.conn.commit()

        num = transition_memory._anchors.rebuild()
        assert num == 2


# =============================================================================
# Markov Anchor Tests
# =============================================================================

class TestMarkovAnchorQueries:
    """AnchorManager query tests with MarkovMemory."""

    @pytest.fixture
    def memory_with_anchors(self, temp_db_path):
        """MarkovMemory with pre-populated anchors."""
        mem = MarkovMemory(temp_db_path)

        mem.conn.execute("INSERT INTO anchors VALUES (1, 'state1', 100, 50, 25)")
        mem.conn.execute("INSERT INTO anchors VALUES (2, 'state2', 10, 10, 10)")
        mem.conn.execute(
            "INSERT INTO states (state_hash, wins, ties, losses, anchor_id) "
            "VALUES ('abc', 50, 25, 12, 1)"
        )
        mem.conn.execute(
            "INSERT INTO states (state_hash, wins, ties, losses, anchor_id) "
            "VALUES ('def', 10, 10, 10, 2)"
        )
        mem.conn.execute(
            "INSERT INTO states (state_hash, wins, ties, losses, anchor_id) "
            "VALUES ('ghi', 50, 25, 13, 1)"
        )
        mem.conn.commit()

        yield mem
        mem.close()

    def test_get_anchor_id(self, memory_with_anchors):
        """get_anchor_id returns correct ID using to_hash."""
        mem = memory_with_anchors
        assert mem.get_anchor_id("any_from", "abc") == 1
        assert mem.get_anchor_id("any_from", "def") == 2
        assert mem.get_anchor_id("any_from", "nonexistent") is None

    def test_get_anchor_stats(self, memory_with_anchors):
        """get_anchor_stats returns pooled stats."""
        mem = memory_with_anchors
        stats = mem.get_anchor_stats("any_from", "abc")
        assert stats.wins == 100
        assert stats.ties == 50
        assert stats.losses == 25

    def test_get_details(self, memory_with_anchors):
        """get_details returns anchor information."""
        details = memory_with_anchors._anchors.get_details()
        assert len(details) == 2

        a1 = next(d for d in details if d["anchor_id"] == 1)
        assert a1["wins"] == 100
        assert a1["members"] == 2


class TestMarkovAnchorRebuild:
    """AnchorManager rebuild tests with MarkovMemory."""

    def test_rebuild_empty_returns_zero(self, markov_memory):
        """Rebuild on empty database returns 0."""
        assert markov_memory._anchors.rebuild() == 0

    def test_rebuild_creates_anchors(self, markov_memory):
        """Rebuild creates anchors from existing data."""
        markov_memory.conn.execute(
            "INSERT INTO states (state_hash, wins, ties, losses) VALUES ('abc', 10, 5, 5)"
        )
        markov_memory.conn.commit()

        num = markov_memory._anchors.rebuild()
        assert num > 0

    def test_rebuild_read_only_raises(self, temp_db_path):
        """Rebuild raises in read-only mode."""
        mem = MarkovMemory(temp_db_path)
        mem.conn.execute(
            "INSERT INTO states (state_hash, wins) VALUES ('abc', 10)"
        )
        mem.conn.commit()
        mem.close()

        mem_ro = MarkovMemory(temp_db_path, read_only=True)
        with pytest.raises(RuntimeError, match="read-only"):
            mem_ro._anchors.rebuild()
        mem_ro.close()


class TestMarkovAnchorConsolidate:
    """AnchorManager consolidate tests with MarkovMemory."""

    def test_keeps_incompatible(self, markov_memory):
        """Consolidate does not merge incompatible anchors."""
        markov_memory.conn.execute("INSERT INTO anchors VALUES (1, 'k1', 100, 0, 0)")
        markov_memory.conn.execute("INSERT INTO anchors VALUES (2, 'k2', 0, 0, 100)")
        markov_memory.conn.execute(
            "INSERT INTO states (state_hash, wins, anchor_id) VALUES ('abc', 100, 1)"
        )
        markov_memory.conn.execute(
            "INSERT INTO states (state_hash, losses, anchor_id) VALUES ('def', 100, 2)"
        )
        markov_memory.conn.commit()

        markov_memory._anchors.consolidate()

        count = markov_memory.conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
        assert count == 2

    def test_read_only_returns_zero(self, temp_db_path):
        """Consolidate returns 0 in read-only mode."""
        mem = MarkovMemory(temp_db_path)
        mem.conn.execute("INSERT INTO anchors VALUES (1, 'k1', 100, 50, 50)")
        mem.conn.commit()
        mem.close()

        mem_ro = MarkovMemory(temp_db_path, read_only=True)
        assert mem_ro._anchors.consolidate() == 0
        mem_ro.close()


class TestMarkovClustering:
    """Clustering behavior tests with MarkovMemory."""

    def test_identical_distributions_cluster(self, markov_memory):
        """Identical distributions cluster together."""
        for i in range(5):
            markov_memory.conn.execute(
                f"INSERT INTO states (state_hash, wins, ties, losses) "
                f"VALUES ('state{i}', 100, 50, 50)"
            )
        markov_memory.conn.commit()

        num = markov_memory._anchors.rebuild()
        assert num == 1

    def test_different_distributions_separate(self, markov_memory):
        """Very different distributions stay separate."""
        markov_memory.conn.execute(
            "INSERT INTO states (state_hash, wins, ties, losses) VALUES ('abc', 100, 0, 0)"
        )
        markov_memory.conn.execute(
            "INSERT INTO states (state_hash, wins, ties, losses) VALUES ('def', 0, 0, 100)"
        )
        markov_memory.conn.commit()

        num = markov_memory._anchors.rebuild()
        assert num == 2