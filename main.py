# Classes
from agent.agent import Agent
from games.game_base import GameBase

# Functions
from simulation.simulation import start_simulations

# Global variables
from utils.global_variables import TURN_DEPTH
from utils.global_variables import SIMULATIONS
from utils.global_variables import SELECTED_GAME, GAMES

from typing import List


def create_agents(n: int) -> List[Agent]:
    return [Agent() for _ in range(n)]


def main():
    agents = create_agents(50)
    anti_agents = create_agents(50)
    game_class: type[GameBase] = GAMES[SELECTED_GAME]
    start_simulations(agents, anti_agents, game_class, TURN_DEPTH, SIMULATIONS)


if __name__ == "__main__":
    main()
