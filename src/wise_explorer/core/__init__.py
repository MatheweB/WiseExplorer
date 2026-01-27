"""
Core module - fundamental types, hashing, and statistics.

This module provides the building blocks used throughout the game AI system.
"""

from wise_explorer.core.types import (
    Stats,
    W_WEIGHT,
    T_WEIGHT,
    L_WEIGHT,
    SCORE_MIN,
    SCORE_MAX,
    SCORE_RANGE,
    OUTCOME_INDEX,
    UNEXPLORED_ANCHOR_ID,
)
from wise_explorer.core.hashing import hash_board
from wise_explorer.core.bayes import compatible, similarity

__all__ = [
    # Types
    "Stats",
    # Constants
    "W_WEIGHT",
    "T_WEIGHT",
    "L_WEIGHT",
    "SCORE_MIN",
    "SCORE_MAX",
    "SCORE_RANGE",
    "OUTCOME_INDEX",
    "UNEXPLORED_ANCHOR_ID",
    # Functions
    "hash_board",
    "compatible",
    "similarity",
]