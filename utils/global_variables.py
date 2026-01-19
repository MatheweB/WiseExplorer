import math
import numpy as np

from games.tic_tac_toe import TicTacToe
from games.minichess import MiniChess
from games.game_state import GameState
from simulation.simulation import DEFAULT_WORKER_COUNT

GAMES = {"tic_tac_toe": TicTacToe, "minichess": MiniChess}

# ----------------- TIC TAC TOE -----------------
tic_tac_toe_init_board = np.full((TicTacToe().SIZE, TicTacToe().SIZE), None)
tic_tac_toe_init_state = GameState(tic_tac_toe_init_board, current_player=1)

# ----------------- MINICHESS -----------------
minichess_init_board = MiniChess()._create_initial_board()  # pylint: disable=[W0212]
minichess_init_state = GameState(minichess_init_board, current_player=1)



INITIAL_STATES = {
    "tic_tac_toe": tic_tac_toe_init_state,
    "minichess": minichess_init_state,
}

def _round_epochs_for_clean_split(epochs: int, num_players: int) -> int:
    """Round epochs up to ensure clean division in training phases for prune-rounds."""
    divisor = num_players + 1  # 2-player → 3, 3-player → 4, etc.
    return math.ceil(epochs / divisor) * divisor


SELECTED_GAME = "tic_tac_toe"
EPOCHS_BEFORE_ROUNDING = 100
EPOCHS =_round_epochs_for_clean_split(EPOCHS_BEFORE_ROUNDING, GAMES[SELECTED_GAME]().num_players())
NUM_AGENTS = DEFAULT_WORKER_COUNT
SIMULATIONS = EPOCHS * NUM_AGENTS
TURN_DEPTH = 40
