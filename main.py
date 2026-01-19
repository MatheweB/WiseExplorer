# main.py
from typing import Dict, List
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
    NUM_AGENTS,
    DEFAULT_WORKER_COUNT
)


# ------------------------------------------------------
# Create a swarm (many agents) per player
# ------------------------------------------------------
def create_agent_swarms(
    players: List[int], agents_per_player: int
) -> Dict[int, List[Agent]]:
    players_map: Dict[int, List[Agent]] = {}
    for pid in players:
        swarm = [Agent() for _ in range(agents_per_player)]
        for a in swarm:
            a.player_id = pid
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

    # Auto-detect number of players from the game engine
    num_players = game.num_players()
    players = list(range(1, num_players + 1))

    # Build a multi-agent swarm
    agent_swarms = create_agent_swarms(players, NUM_AGENTS)

    # Memory manager - partitioned by game type
    # Creates: data/memory/{game_id}.db (e.g., tic_tac_toe.db, minichess.db)
    memory = GameMemory.for_game(game, base_dir="data/memory", markov=True)

    # Start generalized multi-agent, multi-player simulation
    start_simulations(
        agent_swarms=agent_swarms,
        game=game,
        turn_depth=TURN_DEPTH,
        simulations=SIMULATIONS,
        memory=memory,
        num_workers=DEFAULT_WORKER_COUNT,
        training_enabled=True,
        human_players=[],
        debug_move_statistics=True
    )


if __name__ == "__main__":
    main()
