"""
Tests for wise_explorer.memory.anchor_manager

Tests anchor clustering system for grouping similar moves.
"""

import pytest

from wise_explorer.core.types import Stats
from wise_explorer.memory.anchor_manager import Anchor, AnchorManager, _sub_counts
from wise_explorer.memory.game_memory import GameMemory


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

    def test_distribution(self):
        """Distribution returns normalized probabilities."""
        assert Anchor((10, 10, 10), "k").distribution == pytest.approx((1/3, 1/3, 1/3))
        assert Anchor((0, 0, 0), "k").distribution == (0.0, 0.0, 0.0)

    def test_add(self):
        """Add increments counts."""
        anchor = Anchor((10, 5, 5), "k")
        anchor.add((5, 3, 2))
        assert anchor.counts == (15, 8, 7)

    def test_subtract(self):
        """Subtract decrements counts."""
        anchor = Anchor((10, 5, 5), "k")
        anchor.subtract((3, 2, 1))
        assert anchor.counts == (7, 3, 4)

    def test_without(self):
        """Without returns subtracted counts without modifying original."""
        anchor = Anchor((10, 5, 5), "k")
        result = anchor.without((3, 2, 1))
        assert result == (7, 3, 4)
        assert anchor.counts == (10, 5, 5)  # Unchanged


class TestSubCounts:
    """_sub_counts helper tests."""

    def test_basic(self):
        """Subtracts element-wise."""
        assert _sub_counts((10, 5, 5), (3, 2, 1)) == (7, 3, 4)

    def test_to_zero(self):
        """Can result in zeros."""
        assert _sub_counts((10, 5, 5), (10, 5, 5)) == (0, 0, 0)


class TestAnchorManagerQueries:
    """AnchorManager query tests."""

    @pytest.fixture
    def memory_with_anchors(self, temp_db_path):
        """Memory with pre-populated anchors."""
        mem = GameMemory(temp_db_path)
        
        mem.conn.execute("INSERT INTO anchors VALUES (1, 'key1', 100, 50, 25)")
        mem.conn.execute("INSERT INTO anchors VALUES (2, 'key2', 10, 10, 10)")
        mem.conn.execute("INSERT INTO scoring_anchors VALUES ('key1', 1)")
        mem.conn.execute("INSERT INTO scoring_anchors VALUES ('key2', 2)")
        mem.conn.execute("INSERT INTO scoring_anchors VALUES ('key3', 1)")
        mem.conn.commit()
        
        yield mem
        mem.close()

    def test_get_anchor_id(self, memory_with_anchors):
        """get_anchor_id returns correct ID or None."""
        manager = memory_with_anchors._anchors
        assert manager.get_anchor_id("key1") == 1
        assert manager.get_anchor_id("key2") == 2
        assert manager.get_anchor_id("nonexistent") is None

    def test_get_anchor_stats(self, memory_with_anchors):
        """get_anchor_stats returns pooled stats."""
        manager = memory_with_anchors._anchors
        stats = manager.get_anchor_stats("key1")
        assert stats.wins == 100
        assert stats.ties == 50
        assert stats.losses == 25

    def test_get_details(self, memory_with_anchors):
        """get_details returns anchor information."""
        details = memory_with_anchors._anchors.get_details()
        assert len(details) == 2
        
        a1 = next(d for d in details if d["anchor_id"] == 1)
        assert a1["wins"] == 100
        assert a1["members"] == 2  # key1 and key3


class TestAnchorManagerRebuild:
    """AnchorManager rebuild tests."""

    def test_rebuild_empty_returns_zero(self, memory):
        """Rebuild on empty database returns 0."""
        assert memory._anchors.rebuild() == 0

    def test_rebuild_creates_anchors(self, memory):
        """Rebuild creates anchors from existing data."""
        memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, scoring_key, wins, ties, losses) "
            "VALUES ('a', 'b', 'a|b', 10, 5, 5)"
        )
        memory.conn.commit()
        
        num = memory._anchors.rebuild()
        assert num > 0

    def test_rebuild_read_only_raises(self, temp_db_path):
        """Rebuild raises in read-only mode."""
        mem = GameMemory(temp_db_path)
        mem.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, scoring_key, wins) VALUES ('a', 'b', 'a|b', 10)"
        )
        mem.conn.commit()
        mem.close()
        
        mem_ro = GameMemory(temp_db_path, read_only=True)
        with pytest.raises(RuntimeError, match="read-only"):
            mem_ro._anchors.rebuild()
        mem_ro.close()


class TestAnchorManagerConsolidate:
    """AnchorManager consolidate tests."""

    def test_consolidate_keeps_incompatible(self, memory):
        """Consolidate does not merge incompatible anchors."""
        memory.conn.execute("INSERT INTO anchors VALUES (1, 'k1', 100, 0, 0)")
        memory.conn.execute("INSERT INTO anchors VALUES (2, 'k2', 0, 0, 100)")
        memory.conn.execute("INSERT INTO scoring_anchors VALUES ('k1', 1)")
        memory.conn.execute("INSERT INTO scoring_anchors VALUES ('k2', 2)")
        memory.conn.commit()
        
        memory._anchors.consolidate()
        
        count = memory.conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
        assert count == 2

    def test_consolidate_read_only_returns_zero(self, temp_db_path):
        """Consolidate returns 0 in read-only mode."""
        mem = GameMemory(temp_db_path)
        mem.conn.execute("INSERT INTO anchors VALUES (1, 'k1', 100, 50, 50)")
        mem.conn.commit()
        mem.close()
        
        mem_ro = GameMemory(temp_db_path, read_only=True)
        assert mem_ro._anchors.consolidate() == 0
        mem_ro.close()


class TestClustering:
    """Clustering behavior tests."""

    def test_identical_distributions_cluster(self, memory):
        """Identical distributions cluster together."""
        for i in range(5):
            memory.conn.execute(
                f"INSERT INTO transitions (from_hash, to_hash, scoring_key, wins, ties, losses) "
                f"VALUES ('a{i}', 'b{i}', 'a{i}|b{i}', 100, 50, 50)"
            )
        memory.conn.commit()
        
        num = memory._anchors.rebuild()
        assert num == 1  # All in one anchor

    def test_different_distributions_separate(self, memory):
        """Very different distributions stay separate."""
        memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, scoring_key, wins, ties, losses) "
            "VALUES ('a', 'b', 'a|b', 100, 0, 0)"
        )
        memory.conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, scoring_key, wins, ties, losses) "
            "VALUES ('c', 'd', 'c|d', 0, 0, 100)"
        )
        memory.conn.commit()
        
        num = memory._anchors.rebuild()
        assert num == 2  # Separate anchors