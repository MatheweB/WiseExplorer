# simulation.py
# --------------------------------------------------------------
# Fixed-agent-per-game N-player simulation with:
#   • agent swarms (any number of agents per player)
#   • random agent selection per player for each simulated game
#   • correct snapshot recording
#   • outcome storage via omnicron
#   • clean main loop (no debug logs)
#
# Stack item format:
#   (move: np.ndarray, snapshot_state: GameState, acting_player: int)
#
# Public API:
#   start_simulations(players_map, game, turn_depth, simulations, omnicron)
#
# players_map format:
#   {
#       1: [Agent, Agent, ...],
#       2: [Agent, Agent, ...],
#       ...
#   }

from typing import Dict, List, Tuple
import random
import numpy as np

from agent.agent import Agent, State
from games.game_base import GameBase
from games.game_state import GameState
from omnicron.manager import GameMemory
from wise_explorer import wise_explorer_algorithm

MoveStackItem = Tuple[np.ndarray, GameState, int]
_TERMINAL_STATES = {State.WIN, State.LOSS, State.TIE}


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------


def reset_agent(agent: Agent) -> None:
    """Reset the internal working state of a single agent."""
    agent.core_move = np.array([])
    agent.move = np.array([])
    agent.change = False
    agent.game_state = State.NEUTRAL
    agent.move_depth = 0


def _choose_agents_for_game(players_map: Dict[int, List[Agent]]) -> Dict[int, Agent]:
    """
    Randomly choose ONE agent per player for this entire simulation.
    """
    chosen = {}
    for pid, swarm in players_map.items():
        chosen[pid] = random.choice(swarm)
        reset_agent(chosen[pid])
    return chosen


def _apply_policy(
    agent: Agent, game: GameBase, omnicron: GameMemory, prune: bool
) -> None:
    """Fill agent.core_move via Wise Explorer."""
    wise_explorer_algorithm.update_agent(agent, game, omnicron, prune)


def _apply_move_and_record(
    agent: Agent, game: GameBase, acting_player: int, stack: List[MoveStackItem]
) -> State:
    """
    Snapshot BEFORE move, then apply.
    stack += (move_copy, snapshot_state, acting_player)
    """
    snapshot = game.get_state().clone()
    agent.move = agent.core_move

    move_copy = np.array(agent.move, copy=True)
    stack.append((move_copy, snapshot, acting_player))

    game.apply_move(move_copy)

    result = game.get_result(agent.player_id)
    agent.game_state = result
    return result


def _store_outcome_data(
    game_id: str,
    stacks: Dict[int, List[MoveStackItem]],
    results: Dict[int, State],
    omnicron: GameMemory,
) -> None:
    """Write all moves for all players to omnicron."""
    for pid, stack in stacks.items():
        outcome = results[pid]
        for move_arr, snapshot_state, acting_player in stack:
            omnicron.write(game_id, snapshot_state, acting_player, outcome, move_arr)


# --------------------------------------------------------------
# Single simulated game with fixed chosen agents
# --------------------------------------------------------------


def _simulate_game(
    players_map: Dict[int, List[Agent]],
    game: GameBase,
    turn_depth: int,
    omnicron: GameMemory,
    prune_stage: bool,
) -> None:
    """
    Run an entire simulation using a FIXED agent per player for the whole game.
    """
    num_players = game.num_players()

    # Choose one agent per player
    agents_for_game: Dict[int, Agent] = _choose_agents_for_game(players_map)

    # Per-player move stacks
    stacks = {p: [] for p in range(1, num_players + 1)}
    final_results = {p: State.NEUTRAL for p in range(1, num_players + 1)}

    depth = 0

    while depth < turn_depth:
        depth += 1

        cp = game.current_player()
        actor = agents_for_game[cp]

        _apply_policy(actor, game, omnicron, prune_stage)
        result = _apply_move_and_record(actor, game, cp, stacks[cp])

        # Terminal check
        if result in _TERMINAL_STATES or game.is_over():
            for pid in final_results:
                final_results[pid] = game.get_result(pid)
            break

    _store_outcome_data(game.game_id(), stacks, final_results, omnicron)


# --------------------------------------------------------------
# Batches
# --------------------------------------------------------------


def _run_simulation_batch(
    players_map: Dict[int, List[Agent]],
    game: GameBase,
    turn_depth: int,
    count: int,
    omnicron: GameMemory,
    prune_stage: bool,
) -> None:
    """Runs `count` simulations using fixed agent tuples."""
    for _ in range(count):
        cloned = game.deep_clone()
        _simulate_game(players_map, cloned, turn_depth, omnicron, prune_stage)


# --------------------------------------------------------------
# Public API
# --------------------------------------------------------------


def start_simulations(
    players_map: Dict[int, List[Agent]],
    game: GameBase,
    turn_depth: int,
    simulations: int,
    omnicron: GameMemory,
) -> None:
    """
    Full simulation driver:
    • prune pass
    • optimize pass
    • choose next real move
    • continue until terminal
    """
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if turn_depth <= 0:
        raise ValueError("turn_depth must be positive")

    while True:
        prune_count = simulations // 2
        optimize_count = simulations - prune_count

        _run_simulation_batch(
            players_map, game, turn_depth, prune_count, omnicron, prune_stage=True
        )
        _run_simulation_batch(
            players_map, game, turn_depth, optimize_count, omnicron, prune_stage=False
        )

        # Choose best next move for the actual game
        current_state = game.get_state().clone()
        best_move = omnicron.get_worst_move(
            game.game_id(), current_state, debug_move=True
        )

        if best_move is None:
            valid = game.valid_moves()
            if not valid:
                break
            best_move = random.choice(valid)

        game.apply_move(best_move)
        print(game.state_string())

        if game.is_over():
            break
