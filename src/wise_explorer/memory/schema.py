"""
Database schema for game memory.

Two separate modes with no shared state:

Non-Markov Mode (transition-based):
    - Unit of learning: (from_hash, to_hash) pairs
    - Path-dependent: V(s) = f(s_prev, s)
    - Tables: transitions, anchors

Markov Mode (state-based):
    - Unit of learning: state_hash (destination only)
    - Assumes the Markov property: V(s) = f(s)
    - Tables: states, anchors

Each mode has its own database file for isolation.
They cluster fundamentally different units and converge to different equilibria.
"""

_SCHEMA_PREDICATES = """
CREATE TABLE IF NOT EXISTS boards (
    board_hash TEXT PRIMARY KEY,
    board_data BLOB NOT NULL,
    board_rows INTEGER NOT NULL,
    board_cols INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS predicates (
    predicate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    atoms_json TEXT NOT NULL,
    wins REAL DEFAULT 0.0,
    ties REAL DEFAULT 0.0,
    losses REAL DEFAULT 0.0,
    support INTEGER DEFAULT 0,
    score_variance REAL DEFAULT 1.0,
    mining_score REAL DEFAULT NULL
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
    anchor_id INTEGER,
    PRIMARY KEY (from_hash, to_hash)
);
CREATE INDEX IF NOT EXISTS idx_from_hash ON transitions(from_hash);
CREATE INDEX IF NOT EXISTS idx_trans_anchor ON transitions(anchor_id);

CREATE TABLE IF NOT EXISTS anchors (
    anchor_id INTEGER PRIMARY KEY,
    repr_key TEXT,
    wins REAL DEFAULT 0.0,
    ties REAL DEFAULT 0.0,
    losses REAL DEFAULT 0.0
);

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
""" + _SCHEMA_PREDICATES

SCHEMA_MARKOV = """
CREATE TABLE IF NOT EXISTS states (
    state_hash TEXT PRIMARY KEY,
    wins REAL DEFAULT 0.0,
    ties REAL DEFAULT 0.0,
    losses REAL DEFAULT 0.0,
    anchor_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_state_anchor ON states(anchor_id);

CREATE TABLE IF NOT EXISTS anchors (
    anchor_id INTEGER PRIMARY KEY,
    repr_key TEXT,
    wins REAL DEFAULT 0.0,
    ties REAL DEFAULT 0.0,
    losses REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
""" + _SCHEMA_PREDICATES