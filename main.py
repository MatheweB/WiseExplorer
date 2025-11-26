# main.py

from agent.agent import Agent
from games.game_base import GameBase, GameState

from simulation.simulation import start_simulations
from omnicron.manager import GameMemory

from utils.global_variables import (
    TURN_DEPTH,
    SIMULATIONS,
    SELECTED_GAME,
    GAMES,
    INITIAL_STATES,
)

from typing import Dict, List


# ------------------------------------------------------
# Create a swarm (many agents) per player
# ------------------------------------------------------
def create_agent_swarm(
    players: List[int], agents_per_player: int
) -> Dict[int, List[Agent]]:
    players_map: Dict[int, List[Agent]] = {}

    for pid in players:
        swarm = []
        for _ in range(agents_per_player):
            a = Agent()
            a.player_id = pid
            swarm.append(a)
        players_map[pid] = swarm

    return players_map


# ------------------------------------------------------
# Load game with the initial predefined state
# ------------------------------------------------------
def _configure_initial_game() -> GameBase:
    game_class = GAMES[SELECTED_GAME]
    initial_game_state: GameState = INITIAL_STATES[SELECTED_GAME]
    game = game_class()
    game.set_state(initial_game_state)
    return game


# ------------------------------------------------------
# Fully generalized main
# ------------------------------------------------------
def main():
    game = _configure_initial_game()

    # auto-detect number of players from the game engine
    num_players = game.num_players()
    players = list(range(1, num_players + 1))

    # number of agents PER player (tune this!)
    AGENTS_PER_PLAYER = 40

    # build a multi-agent swarm
    players_map = create_agent_swarm(players, AGENTS_PER_PLAYER)

    # memory manager
    omnicron = GameMemory()

    # start generalized multi-agent, multi-player simulation
    start_simulations(
        players_map=players_map,
        game=game,
        turn_depth=TURN_DEPTH,
        simulations=SIMULATIONS,
        omnicron=omnicron,
    )


if __name__ == "__main__":
    main()
