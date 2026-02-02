"""
GameBase - abstract base class for all board games.
"""

from abc import ABC, abstractmethod
from typing import List

from numpy.typing import NDArray

from wise_explorer.agent.agent import State
from wise_explorer.games.game_state import GameState


class GameBase(ABC):
    """
    Abstract base class for all board games.

    IMPORTANT ARCHITECTURE NOTE:
    -----------------------------
    - Games expose MOVES, not transitions.
    - Transitions are DERIVED by simulating moves.
    - Memory system canonicalizes STATE → STATE transitions internally.

    Do NOT add valid_transitions() here.
    """

    @abstractmethod
    def game_id(self) -> str:
        """Return a stable identifier (e.g. 'tic_tac_toe')."""
        pass

    @abstractmethod
    def num_players(self) -> int:
        """Return number of players in the game."""
        pass

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

    @abstractmethod
    def valid_moves(self) -> List[NDArray]:
        """
        Return all legal moves from the current state.
        Example (TicTacToe): [(row, col), ...]
        """
        pass

    @abstractmethod
    def apply_move(self, move: NDArray, *, validated: bool = False) -> None:
        """
        Apply a move to the game. Mutates internal state.

        Args:
            move: The move to apply.
            validated:  If True, skip validation (caller guarantees
                        the move came from valid_moves()). Games may
                        ignore this hint if validation is already cheap.
        """
        pass

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

    @abstractmethod
    def get_cell_strings(self) -> dict[int, str]:
        """
        Return a dictionary of [int -> str] where each cell value maps to its display string
            (e.g. {0: " ", 1: "X", 2: "O"} for tic_tac_toe)
        """
        pass

    @abstractmethod
    def state_string(self) -> str:
        """Pretty string representation of the state."""
        pass
