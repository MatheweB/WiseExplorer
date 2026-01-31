"""
Game memory module - pattern-based learning for game AI.

Two separate implementations:
- TransitionMemory: path-dependent (from, to) learning
- MarkovMemory: path-independent state learning

Use `for_game()` factory or instantiate directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from wise_explorer.memory.game_memory import GameMemory
from wise_explorer.memory.transition_memory import TransitionMemory
from wise_explorer.memory.markov_memory import MarkovMemory

if TYPE_CHECKING:
    from wise_explorer.games.game_base import GameBase


def for_game(
    game: "GameBase",
    base_dir: str | Path = "data/memory",
    markov: bool = False,
    **kwargs
) -> GameMemory:
    """
    Create a memory instance for a specific game.
    
    Args:
        game: Game instance to create memory for
        base_dir: Base directory for database files
        markov: If True, use MarkovMemory; otherwise TransitionMemory
        **kwargs: Additional arguments (e.g., read_only=True)
        
    Returns:
        TransitionMemory or MarkovMemory instance
    """
    base_dir = Path(base_dir)
    game_id = game.game_id()
    
    if markov:
        return MarkovMemory(base_dir / f"{game_id}_markov.db", **kwargs)
    return TransitionMemory(base_dir / f"{game_id}.db", **kwargs)


def open_readonly(db_path: str | Path, is_markov: bool = False) -> GameMemory:
    """
    Open an existing database in read-only mode.
    
    Args:
        db_path: Path to the database file
        markov: If True, open as MarkovMemory; otherwise TransitionMemory
        
    Returns:
        Read-only memory instance
    """
    cls = MarkovMemory if is_markov else TransitionMemory
    return cls(db_path, read_only=True)


__all__ = [
    "GameMemory",
    "TransitionMemory",
    "MarkovMemory",
    "for_game",
    "open_readonly",
]