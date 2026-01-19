# game_base.py
from abc import ABC, abstractmethod
from typing import List
from numpy.typing import NDArray

from agent.agent import State
from games.game_state import GameState


class GameBase(ABC):
    """
    Abstract base class for all board games.

    IMPORTANT ARCHITECTURE NOTE:
    -----------------------------
    • Games expose MOVES, not transitions.
    • Transitions are DERIVED by simulating moves.
    • Omnicron memory canonicalizes STATE → STATE transitions internally.

    Do NOT add valid_transitions() here.
    """

    # ----------------------------------------------------------
    # Identity / structure
    # ----------------------------------------------------------
    @abstractmethod
    def game_id(self) -> str:
        """Return a stable identifier (e.g. 'tic_tac_toe')."""
        pass

    @abstractmethod
    def num_players(self) -> int:
        """Return number of players in the game."""
        pass

    # ----------------------------------------------------------
    # Cloning
    # ----------------------------------------------------------
    @abstractmethod
    def clone(self) -> "GameBase":
        """Shallow copy (rarely used)."""
        pass

    @abstractmethod
    def deep_clone(self) -> "GameBase":
        """
        Deep copy of game + state.
        Used heavily for simulation and transition derivation.
        """
        pass

    # ----------------------------------------------------------
    # State access
    # ----------------------------------------------------------
    @abstractmethod
    def get_state(self) -> GameState:
        """Return the current game state."""
        pass

    @abstractmethod
    def set_state(self, game_state: GameState) -> None:
        """Replace the current game state."""
        pass

    @abstractmethod
    def current_player(self) -> int:
        """Return ID of player to act."""
        pass

    # ----------------------------------------------------------
    # Move interface (PRIMARY)
    # ----------------------------------------------------------
    @abstractmethod
    def valid_moves(self) -> List[NDArray]:
        """
        Return all legal moves from the current state.
        Example (TicTacToe): [(row, col), ...]
        """
        pass

    @abstractmethod
    def apply_move(self, move: NDArray) -> None:
        """
        Apply a move to the game.
        Mutates internal state.
        """
        pass

    # ----------------------------------------------------------
    # Termination / payoff
    # ----------------------------------------------------------
    @abstractmethod
    def is_over(self) -> bool:
        """Return True if the game has ended."""
        pass

    @abstractmethod
    def get_result(self, agent_id: int) -> State:
        """
        Return payoff for the agent:
            WIN / TIE / NEUTRAL / LOSS
        """
        pass

    # ----------------------------------------------------------
    # Debug / UI
    # ----------------------------------------------------------
    @abstractmethod
    def state_string(self) -> str:
        """Pretty string representation of the state."""
        pass
