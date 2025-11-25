# Classes
from agent.agent import Agent
from games.game_base import GameBase
from games.game_base import GameState


# Functions
from simulation.simulation import start_simulations
from omnicron.manager import GameMemory

# Global variables
from utils.global_variables import TURN_DEPTH
from utils.global_variables import SIMULATIONS
from utils.global_variables import SELECTED_GAME, GAMES
from utils.global_variables import INITIAL_STATES

from typing import List


def create_agents(n: int, player_id: int) -> List[Agent]:
    agents = []
    for _ in range(n):
        agent = Agent()
        agent.player_id = player_id
        agents.append(agent)
    return agents


def _configure_initial_game() -> GameBase:
    game_class = GAMES[SELECTED_GAME]
    initial_game_state: GameState = INITIAL_STATES[SELECTED_GAME]
    initialized_game = game_class()

    # Sets the state of the game to whatever is set in global vars
    initialized_game.set_state(initial_game_state)
    return initialized_game


def main():
    # TODO(Generalize the player_id, but for now 2-player games are where we are)
    agents = create_agents(50, 1)
    anti_agents = create_agents(50, 2)
    game = _configure_initial_game()
    omnicron = GameMemory()
    start_simulations(agents, anti_agents, game, TURN_DEPTH, SIMULATIONS, omnicron)


if __name__ == "__main__":
    main()
