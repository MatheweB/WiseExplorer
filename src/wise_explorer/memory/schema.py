"""
Database schema for game memory.

Tables:
    transitions     - State-to-state transition outcomes (denormalized with anchor_id)
    state_values    - Aggregated state values (for Markov mode)
    anchors         - Cluster statistics
    scoring_anchors - Maps scoring_key → anchor_id
    metadata        - Key-value store for settings

Indexes:
    idx_scoring_key  - Fast lookup by scoring_key
    idx_from_hash    - Fast batch lookup for evaluate_moves
    idx_trans_anchor - Fast anchor membership queries
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS transitions (
    from_hash TEXT NOT NULL,
    to_hash TEXT NOT NULL,
    scoring_key TEXT NOT NULL,
    wins INTEGER DEFAULT 0,
    ties INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    anchor_id INTEGER,
    PRIMARY KEY (from_hash, to_hash)
);

CREATE INDEX IF NOT EXISTS idx_scoring_key ON transitions(scoring_key);
CREATE INDEX IF NOT EXISTS idx_from_hash ON transitions(from_hash);
CREATE INDEX IF NOT EXISTS idx_trans_anchor ON transitions(anchor_id);

CREATE TABLE IF NOT EXISTS state_values (
    state_hash TEXT PRIMARY KEY,
    wins INTEGER DEFAULT 0,
    ties INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    anchor_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_sv_anchor ON state_values(anchor_id);

CREATE TABLE IF NOT EXISTS anchors (
    anchor_id INTEGER PRIMARY KEY,
    repr_key TEXT,
    wins INTEGER DEFAULT 0,
    ties INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS scoring_anchors (
    scoring_key TEXT PRIMARY KEY,
    anchor_id INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sa_anchor ON scoring_anchors(anchor_id);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""