"""
Database schema for game memory.

Two modes, each with its own database file:

Non-Markov (transition-based): the unit of learning is a (from_hash, to_hash)
pair — path-dependent, V(s) = f(s_prev, s). Markov (state-based): the unit is
the destination state alone, V(s) = f(s).

`propagated_score` holds the value engine's output (Bellman solve + library
completion). The engine writes it and discovery fits it; move selection never
reads it. `certificates` holds game-proven board values
(docs/certified-forgetting.md).
"""

# Raw board arrays by hash — the concept library reads these to turn stored
# transitions back into (board, value) examples it can invent from.
_SCHEMA_BOARDS = """
CREATE TABLE IF NOT EXISTS boards (
    board_hash TEXT PRIMARY KEY,
    board_data BLOB NOT NULL,
    board_rows INTEGER NOT NULL,
    board_cols INTEGER NOT NULL,
    to_move INTEGER DEFAULT 0
);
"""

_SCHEMA_CERTIFICATES = """
CREATE TABLE IF NOT EXISTS certificates (
    board_hash TEXT PRIMARY KEY,
    value REAL NOT NULL
);
"""

SCHEMA_TRANSITIONS = """
CREATE TABLE IF NOT EXISTS transitions (
    from_hash TEXT NOT NULL,
    to_hash TEXT NOT NULL,
    wins REAL DEFAULT 0.0,
    ties REAL DEFAULT 0.0,
    losses REAL DEFAULT 0.0,
    propagated_score REAL DEFAULT NULL,
    PRIMARY KEY (from_hash, to_hash)
);
CREATE INDEX IF NOT EXISTS idx_from_hash ON transitions(from_hash);

CREATE TABLE IF NOT EXISTS cross_scores (
    from_hash TEXT NOT NULL,
    to_hash TEXT NOT NULL,
    observer_role INTEGER NOT NULL,
    score_sum REAL DEFAULT 0,
    score_count REAL DEFAULT 0,
    PRIMARY KEY (from_hash, to_hash, observer_role)
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
""" + _SCHEMA_BOARDS + _SCHEMA_CERTIFICATES

SCHEMA_MARKOV = """
CREATE TABLE IF NOT EXISTS states (
    state_hash TEXT PRIMARY KEY,
    wins REAL DEFAULT 0.0,
    ties REAL DEFAULT 0.0,
    losses REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
""" + _SCHEMA_BOARDS + _SCHEMA_CERTIFICATES
