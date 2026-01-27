"""
Tests for wise_explorer.memory.schema

Tests database schema validity.
"""

import sqlite3
import tempfile

import pytest

from wise_explorer.memory.schema import SCHEMA


class TestSchemaValidity:
    """Schema SQL validity tests."""

    def test_schema_executes(self):
        """Schema executes without error."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA)
        conn.close()

    def test_schema_idempotent(self):
        """Schema can be executed multiple times."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA)
        conn.executescript(SCHEMA)  # Second time
        conn.close()


class TestRequiredTables:
    """Required table tests."""

    @pytest.fixture
    def conn(self):
        """In-memory database with schema."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA)
        yield conn
        conn.close()

    @pytest.mark.parametrize("table", [
        "transitions",
        "state_values", 
        "anchors",
        "scoring_anchors",
        "metadata",
    ])
    def test_table_exists(self, conn, table):
        """Required table exists."""
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,)
        ).fetchone()
        assert result is not None


class TestRequiredIndexes:
    """Required index tests."""

    @pytest.fixture
    def conn(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA)
        yield conn
        conn.close()

    @pytest.mark.parametrize("index", [
        "idx_scoring_key",
        "idx_from_hash",
        "idx_trans_anchor",
        "idx_sv_anchor",
        "idx_sa_anchor",
    ])
    def test_index_exists(self, conn, index):
        """Required index exists."""
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
            (index,)
        ).fetchone()
        assert result is not None


class TestTableStructure:
    """Table structure tests."""

    @pytest.fixture
    def conn(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA)
        yield conn
        conn.close()

    def test_transitions_primary_key(self, conn):
        """Transitions has primary key on (from_hash, to_hash)."""
        conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, scoring_key) VALUES ('a', 'b', 'a|b')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO transitions (from_hash, to_hash, scoring_key) VALUES ('a', 'b', 'dup')"
            )

    def test_transitions_default_counts(self, conn):
        """Transitions outcome columns default to 0."""
        conn.execute(
            "INSERT INTO transitions (from_hash, to_hash, scoring_key) VALUES ('a', 'b', 'a|b')"
        )
        result = conn.execute(
            "SELECT wins, ties, losses FROM transitions WHERE from_hash='a'"
        ).fetchone()
        assert result == (0, 0, 0)

    def test_metadata_key_value(self, conn):
        """Metadata stores key-value pairs."""
        conn.execute("INSERT INTO metadata VALUES ('key', 'value')")
        result = conn.execute("SELECT value FROM metadata WHERE key='key'").fetchone()
        assert result[0] == "value"