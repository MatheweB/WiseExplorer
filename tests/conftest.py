"""
Shared test fixtures for wise_explorer tests.

Design principles:
- Game-agnostic fixtures where possible
- Clean imports at module level
- Minimal, focused fixtures
"""

import tempfile
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np
import pytest

from wise_explorer.agent.agent import Agent, State
from wise_explorer.core.types import Stats
from wise_explorer.games.game_base import GameBase
from wise_explorer.games.game_state import GameState
from wise_explorer.memory.game_memory import GameMemory


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Temporary database file with cleanup."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    yield path
    for suffix in ["", "-wal", "-shm"]:
        p = Path(str(path) + suffix)
        if p.exists():
            p.unlink()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory with cleanup."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# =============================================================================
# Stats Fixtures
# =============================================================================

@pytest.fixture
def zero_stats() -> Stats:
    return Stats(0, 0, 0)


@pytest.fixture
def balanced_stats() -> Stats:
    return Stats(10, 10, 10)


@pytest.fixture
def winning_stats() -> Stats:
    return Stats(100, 10, 5)


@pytest.fixture
def losing_stats() -> Stats:
    return Stats(5, 10, 100)


# =============================================================================
# Game Fixtures (Game-Agnostic)
# =============================================================================

@pytest.fixture
def any_game() -> GameBase:
    """Any available game for generic testing."""
    from wise_explorer.utils.config import GAMES
    game_class = next(iter(GAMES.values()))
    return game_class()


@pytest.fixture
def two_player_game() -> GameBase:
    """A 2-player game for testing."""
    from wise_explorer.utils.config import GAMES
    for game_class in GAMES.values():
        game = game_class()
        if game.num_players() == 2:
            return game
    pytest.skip("No 2-player game available")


# =============================================================================
# Agent Fixtures
# =============================================================================

@pytest.fixture
def sample_agent() -> Agent:
    """Configured Agent instance."""
    agent = Agent()
    agent.player_id = 1
    agent.core_move = np.array([0, 0])
    agent.move = np.array([1, 1])
    agent.game_state = State.NEUTRAL
    agent.move_depth = 5
    return agent


@pytest.fixture
def agent_swarms(two_player_game: GameBase) -> Dict[int, List[Agent]]:
    """Agent swarms for a 2-player game."""
    swarms: Dict[int, List[Agent]] = {}
    for pid in range(1, two_player_game.num_players() + 1):
        swarms[pid] = [Agent() for _ in range(4)]
        for agent in swarms[pid]:
            agent.player_id = pid
    return swarms


# =============================================================================
# Memory Fixtures
# =============================================================================

@pytest.fixture
def memory(temp_db_path: Path) -> Generator[GameMemory, None, None]:
    """GameMemory instance with temporary database."""
    mem = GameMemory(temp_db_path)
    yield mem
    mem.close()


# =============================================================================
# Selection Fixtures
# =============================================================================

@pytest.fixture
def anchor_stats_varied() -> Dict[int, Stats]:
    """Anchor stats with varied win/tie/loss distributions."""
    return {
        0: Stats(100, 10, 10),
        1: Stats(10, 80, 10),
        2: Stats(10, 10, 100),
        3: Stats(1, 1, 1),
    }


@pytest.fixture
def moves_with_stats() -> List[Tuple[np.ndarray, Stats]]:
    """Move arrays paired with Stats."""
    return [
        (np.array([0, 0]), Stats(50, 10, 5)),
        (np.array([0, 1]), Stats(20, 20, 20)),
        (np.array([1, 1]), Stats(5, 10, 50)),
    ]