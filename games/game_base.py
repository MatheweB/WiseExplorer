# game_base.py
from abc import ABC, abstractmethod

# Libraries
from numpy.typing import NDArray

# Classes
from agent.agent import State
from games.game_state import GameState


class GameBase(ABC):
    """
    Abstract base class for all board games in the system.
    Every game must implement these methods.
    Agents use only these methods — never the underlying implementation.
    """

    # ----------------------------------------------------------
    # Core required methods
    # ----------------------------------------------------------
    @abstractmethod
    def game_id(self) -> str:
        """
        Returns the id of the game we're playing (e.g. tic_tac_toe)
        """
        pass

    @abstractmethod
    def clone(self) -> "GameBase":
        """
        Return a deep copy of the entire game (GameBase + GameState).
        """
        pass

    @abstractmethod
    def deep_clone(self) -> "GameBase":
        """
        Return a shallow copy of the entire game (GameBase + GameState).
        Used heavily in agent lookahead/simulation.
        """
        pass

    @abstractmethod
    def get_state(self) -> "GameState":
        """
        Return the current state of the game.
        """
        pass

    @abstractmethod
    def set_state(self, game_state: GameState) -> None:
        """
        Sets the current state of the game.
        """
        pass

    @abstractmethod
    def current_player(self) -> int:
        """
        Return ID of the player to act: typically 0 or 1.
        """
        pass

    @abstractmethod
    def valid_moves(self) -> NDArray:
        """
        Return a list of all legal actions for the current state.
        Example output: [(row, col), (row, col), ...]
        """
        pass

    @abstractmethod
    def apply_move(self, move: NDArray) -> None:
        """
        Apply a move to the current game state.
        Mutates internal state.
        """
        pass

    @abstractmethod
    def is_over(self) -> bool:
        """
        Return True if game has ended (win, loss, or draw).
        """
        pass

    @abstractmethod
    def get_result(self, agent_id: int) -> State:
        """
        Return payoff for the agent:
            win
            tie
            neutral
            loss
        """
        pass

    @abstractmethod
    def print_state(self) -> None:
        """
        Prints out the current state of the game in a visually appealing way
        """
        pass
