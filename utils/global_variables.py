import numpy as np

from games.tic_tac_toe import TicTacToe
from games.game_state import GameState

GAMES = {"tic_tac_toe": TicTacToe}

# ----------------- TIC TAC TOE -----------------
tic_tac_toe_init_board = np.full((TicTacToe().SIZE, TicTacToe().SIZE), None)
# tic_tac_toe_init_board = np.array([
#     [None, 1, None],
#     [None, None, None],
#     [None, None, None]
# ])

tic_tac_toe_init_state = GameState(tic_tac_toe_init_board, current_player=1)
INITIAL_STATES = {"tic_tac_toe": tic_tac_toe_init_state}
# -----------------------------------------------

EPOCHS = 10
NUM_AGENTS = 10
SIMULATIONS = EPOCHS * NUM_AGENTS
TURN_DEPTH = 20
SELECTED_GAME = "tic_tac_toe"
