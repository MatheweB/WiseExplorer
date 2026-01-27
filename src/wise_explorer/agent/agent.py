"""
A class representing an agent in a simulation with various attributes and behaviors.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np


class State(Enum):
    WIN = auto()
    TIE = auto()
    LOSS = auto()
    NEUTRAL = auto()


@dataclass
class Agent:
    _core_move: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # The initial move that an agent will make in a simulation (based on the initial board state before simulating)
    _move: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # The move that the agent is going to make on a given turn in the simulation.
    _change: bool = (
        False  # A boolean indicating whether the agent should change its coreMove in the next simulation.
    )
    _game_state: State = (
        State.NEUTRAL
    )  # A state indicating whether the agent won/tied-or-neutral/lost the last game it played
    _move_depth: int = (
        0  # An integer indicating the depth (in turns) before the game outcome was determined.
    )
    _player_id: int = (
        0  # An integer indicating the role/ID of the player (e.g. 1 or 2, or any number of distinct "roles")
    )

    @property
    def move(self) -> np.ndarray:
        """Returns the current move of the agent."""
        return self._move

    @move.setter
    def move(self, value: np.ndarray):
        """Sets the current move of the agent."""
        self._move = value

    @property
    def core_move(self) -> np.ndarray:
        """Returns the core move of the agent."""
        return self._core_move

    @core_move.setter
    def core_move(self, value: np.ndarray):
        """Sets the core move of the agent."""
        self._core_move = value

    @property
    def change(self) -> bool:
        """Returns whether the agent should change its coreMove in the next simulation."""
        return self._change

    @change.setter
    def change(self, value: bool):
        """Sets whether the agent should change its coreMove in the next simulation."""
        self._change = value

    @property
    def game_state(self) -> State:
        """Returns whether the agent won the last simulation."""
        return self._game_state

    @game_state.setter
    def game_state(self, value: State) -> None:
        """Sets whether the agent won the last simulation."""
        self._game_state = value

    @property
    def move_depth(self) -> int:
        """Returns the depth (in turns) before the game outcome was determined."""
        return self._move_depth

    @move_depth.setter
    def move_depth(self, value: int):
        """Sets the depth (in turns) before the game outcome was determined."""
        self._move_depth = value

    @property
    def player_id(self) -> int:
        """Returns the player id (e.g. 1 or 2)."""
        return self._player_id

    @player_id.setter
    def player_id(self, value: int):
        """Sets the player id (e.g. 1 or 2)."""
        self._player_id = value
