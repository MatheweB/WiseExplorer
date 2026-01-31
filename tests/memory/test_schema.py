"""
Tests for wise_explorer.memory.schema

Tests database schema validity.
"""

import sqlite3

import pytest

from wise_explorer.memory.schema import SCHEMA_TRANSITIONS, SCHEMA_MARKOV


class TestSchemaValidityTransitions:
    """Schema SQL validity tests."""

    def test_transition_schema_executes(self):
        """Schema executes without error."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA_TRANSITIONS)
        conn.close()

    def test_schema_idempotent(self):
        """Schema can be executed multiple times."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA_TRANSITIONS)
        conn.executescript(SCHEMA_TRANSITIONS)  # Second time
        conn.close()

class TestSchemaValidityMarkov:
    """Schema SQL validity tests."""

    def test_transition_schema_executes(self):
        """Schema executes without error."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA_MARKOV)
        conn.close()

    def test_schema_idempotent(self):
        """Schema can be executed multiple times."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA_MARKOV)
        conn.executescript(SCHEMA_MARKOV)  # Second time
        conn.close()


class TestRequiredTablesTransitions:
    """Required table tests."""

    @pytest.fixture
    def conn(self):
        """In-memory database with schema."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA_TRANSITIONS)
        yield conn
        conn.close()

    @pytest.mark.parametrize("table", [
        "transitions",
        "anchors",
        "metadata",
    ])
    def test_table_exists(self, conn, table):
        """Required table exists."""
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,)
        ).fetchone()
        assert result is not None
    
class TestRequiredTablesMarkov:
    """Required table tests."""

    @pytest.fixture
    def conn(self):
        """In-memory database with schema."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA_MARKOV)
        yield conn
        conn.close()

    @pytest.mark.parametrize("table", [
        "states", 
        "anchors",
        "metadata",
    ])
    def test_table_exists(self, conn, table):
        """Required table exists."""
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,)
        ).fetchone()
        assert result is not None


class TestRequiredIndexesTransitions:
    """Required index tests."""

    @pytest.fixture
    def conn(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA_TRANSITIONS)
        yield conn
        conn.close()

    @pytest.mark.parametrize("index", [
        "idx_from_hash",
        "idx_trans_anchor",
    ])
    def test_index_exists(self, conn, index):
        """Required index exists."""
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
            (index,)
        ).fetchone()
        assert result is not None

class TestRequiredIndexesMarkov:
    """Required index tests."""

    @pytest.fixture
    def conn(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA_MARKOV)
        yield conn
        conn.close()

    @pytest.mark.parametrize("index", [
        "idx_state_anchor",
    ])
    def test_index_exists(self, conn, index):
        """Required index exists."""
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
            (index,)
        ).fetchone()
        assert result is not None


class TestTableStructureTransitions:
    """Table structure tests."""

    @pytest.fixture
    def conn(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA_TRANSITIONS)
        yield conn
        conn.close()

    def test_transitions_default_counts(self, conn):
        """Transitions outcome columns default to 0."""
        conn.execute(
            "INSERT INTO transitions (from_hash, to_hash) VALUES ('a', 'b')"
        )
        result = conn.execute(
            "SELECT wins, ties, losses FROM transitions WHERE from_hash='a' AND to_hash='b'"
        ).fetchone()
        assert result == (0, 0, 0)

    def test_metadata_key_value(self, conn):
        """Metadata stores key-value pairs."""
        conn.execute("INSERT INTO metadata VALUES ('key', 'value')")
        result = conn.execute("SELECT value FROM metadata WHERE key='key'").fetchone()
        assert result[0] == "value"


class TestTableStructureMarkov:
    """Table structure tests."""

    @pytest.fixture
    def conn(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA_MARKOV)
        yield conn
        conn.close()

    def test_states_default_counts(self, conn):
        """Transitions outcome columns default to 0."""
        conn.execute(
            "INSERT INTO states (state_hash) VALUES ('a')"
        )
        result = conn.execute(
            "SELECT wins, ties, losses FROM states WHERE state_hash='a'"
        ).fetchone()
        assert result == (0, 0, 0)

    def test_metadata_key_value(self, conn):
        """Metadata stores key-value pairs."""
        conn.execute("INSERT INTO metadata VALUES ('key', 'value')")
        result = conn.execute("SELECT value FROM metadata WHERE key='key'").fetchone()
        assert result[0] == "value"