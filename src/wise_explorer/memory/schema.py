"""
Database schema for game memory.

Two separate modes with no shared state:

Non-Markov Mode (transition-based):
    - Unit of learning: (from_hash, to_hash) pairs
    - Captures path-dependent information
    - Tables: transitions, anchors

Markov Mode (state-based):
    - Unit of learning: state_hash (destination only)
    - Path-independent, position evaluation
    - Tables: states, anchors

Each mode has its own database file for isolation.
They cluster fundamentally different units and converge to different equilibria.
"""

SCHEMA_TRANSITIONS = """
CREATE TABLE IF NOT EXISTS transitions (
    from_hash TEXT NOT NULL,
    to_hash TEXT NOT NULL,
    wins INTEGER DEFAULT 0,
    ties INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    anchor_id INTEGER,
    PRIMARY KEY (from_hash, to_hash)
);
CREATE INDEX IF NOT EXISTS idx_from_hash ON transitions(from_hash);
CREATE INDEX IF NOT EXISTS idx_trans_anchor ON transitions(anchor_id);

CREATE TABLE IF NOT EXISTS anchors (
    anchor_id INTEGER PRIMARY KEY,
    repr_key TEXT,
    wins INTEGER DEFAULT 0,
    ties INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

SCHEMA_MARKOV = """
CREATE TABLE IF NOT EXISTS states (
    state_hash TEXT PRIMARY KEY,
    wins INTEGER DEFAULT 0,
    ties INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    anchor_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_state_anchor ON states(anchor_id);

CREATE TABLE IF NOT EXISTS anchors (
    anchor_id INTEGER PRIMARY KEY,
    repr_key TEXT,
    wins INTEGER DEFAULT 0,
    ties INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""