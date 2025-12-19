# simulation.py
# --------------------------------------------------------------
# N-player Monte Carlo simulation system (TRANSITION-BASED)
# --------------------------------------------------------------

from typing import Dict, List, Tuple
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent.agent import Agent, State
from games.game_base import GameBase
from games.game_state import GameState
from omnicron.manager import GameMemory
from wise_explorer import wise_explorer_algorithm

MoveStackItem = Tuple[np.ndarray, int, GameState]  # (move, acting_player, state_before)
_TERMINAL_STATES = {State.WIN, State.LOSS, State.TIE}

PARALLEL_WORKERS = 16


# ==============================================================
# AGENT STATE MANAGEMENT
# ==============================================================


def _reset_agent_preserve_memory(agent: Agent) -> None:
    agent.core_move = np.array([])
    agent.move = np.array([])
    agent.game_state = State.NEUTRAL
    agent.move_depth = 0


def _update_agent_memory(agent: Agent) -> None:
    agent.change = agent.game_state in {State.LOSS, State.TIE}


def _reset_all_agent_memories(players_map: Dict[int, List[Agent]]) -> None:
    for swarm in players_map.values():
        for agent in swarm:
            agent.change = False


# ==============================================================
# MATCHMAKING
# ==============================================================


def _generate_matchmade_groups(
    players_map: Dict[int, List[Agent]], player_ids: List[int]
) -> List[Dict[int, Agent]]:
    shuffled = {
        pid: random.sample(players_map[pid], len(players_map[pid]))
        for pid in player_ids
    }
    min_size = min(len(swarm) for swarm in shuffled.values())
    return [{pid: shuffled[pid][i] for pid in player_ids} for i in range(min_size)]


# ==============================================================
# GAME SIMULATION
# ==============================================================


def _apply_policy(
    agent: Agent, game: GameBase, omnicron: GameMemory, prune: bool
) -> None:
    wise_explorer_algorithm.update_agent(agent, game, omnicron, prune)


def _execute_move(
    agent: Agent,
    game: GameBase,
    acting_player: int,
    stack: List[MoveStackItem],
) -> State:
    """
    Execute move and save state BEFORE move for later recording.
    """
    agent.move = agent.core_move
    move_copy = np.array(agent.move, copy=True)

    # Save state BEFORE applying move
    state_before = game.get_state().clone()

    # Store: (move, acting_player, state_before)
    stack.append((move_copy, acting_player, state_before))

    # Now apply move
    game.apply_move(move_copy)

    result = game.get_result(agent.player_id)
    agent.game_state = result
    return result


def _simulate_single_game(
    chosen_agents: Dict[int, Agent],
    game: GameBase,
    turn_depth: int,
    omnicron: GameMemory,
    prune_stage: bool,
    num_players: int,
):
    for agent in chosen_agents.values():
        _reset_agent_preserve_memory(agent)

    stacks: Dict[int, List[MoveStackItem]] = {p: [] for p in range(1, num_players + 1)}
    final_results = {p: State.NEUTRAL for p in range(1, num_players + 1)}

    for _ in range(turn_depth):
        current_player = game.current_player()
        actor = chosen_agents[current_player]

        _apply_policy(actor, game, omnicron, prune_stage)
        result = _execute_move(actor, game, current_player, stacks[current_player])

        if result in _TERMINAL_STATES or game.is_over():
            for pid in final_results:
                final_results[pid] = game.get_result(pid)
            break

    for agent in chosen_agents.values():
        agent.game_state = final_results[agent.player_id]
        _update_agent_memory(agent)

    return stacks, final_results


# ==============================================================
# STORAGE (TRANSITION-BASED)
# ==============================================================


def _store_transitions(
    game: GameBase,
    stacks: Dict[int, List[MoveStackItem]],
    results: Dict[int, State],
    omnicron: GameMemory,
) -> None:
    """
    Store transitions using GameMemory.
    """
    for pid, stack in stacks.items():
        outcome = results[pid]

        # Each stack item now contains: (move, acting_player, state_before)
        for move_arr, acting_player, state_before in stack:
            # Reconstruct game at before-state
            game_before = game.__class__()  # Create fresh game instance
            game_before.set_state(state_before)

            # Record transition
            omnicron.record_outcome(
                game_before,
                move_arr,
                acting_player,
                outcome,
            )


# ==============================================================
# PARALLEL BATCH EXECUTION
# ==============================================================


def _run_simulation_batch(
    players_map: Dict[int, List[Agent]],
    game: GameBase,
    turn_depth: int,
    count: int,
    omnicron: GameMemory,
    prune_stage: bool,
    player_ids: List[int],
) -> None:
    num_players = len(player_ids)
    total_sims = 0

    while total_sims < count:
        groups = _generate_matchmade_groups(players_map, player_ids)
        tasks = []

        for group in groups:
            if total_sims >= count:
                break
            tasks.append(
                (
                    group,
                    game.deep_clone(),
                    turn_depth,
                    omnicron,
                    prune_stage,
                    num_players,
                )
            )
            total_sims += 1

        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = [executor.submit(_simulate_single_game, *task) for task in tasks]

            for future in as_completed(futures):
                stacks, results = future.result()
                _store_transitions(game, stacks, results, omnicron)

    _reset_all_agent_memories(players_map)


# ==============================================================
# PUBLIC API
# ==============================================================


def start_simulations(
    players_map: Dict[int, List[Agent]],
    game: GameBase,
    turn_depth: int,
    simulations: int,
    omnicron: GameMemory,
) -> None:
    if simulations <= 0 or turn_depth <= 0:
        raise ValueError("simulations and turn_depth must be positive")

    player_ids = sorted(players_map.keys())

    playSelf = True
    trainMode = True
    realPlayer = False

    while not game.is_over():
        prune_count = simulations // 2
        optimize_count = simulations - prune_count

        if not realPlayer and trainMode:
            _run_simulation_batch(
                players_map, game, turn_depth, prune_count, omnicron, True, player_ids
            )
            _run_simulation_batch(
                players_map,
                game,
                turn_depth,
                optimize_count,
                omnicron,
                False,
                player_ids,
            )

        if not realPlayer:
            best_move = omnicron.get_best_move(game, debug=True)
            print(best_move)

            if best_move is None:
                valid_moves = game.valid_moves()
                if len(valid_moves) == 0:
                    break
                best_move = random.choice(valid_moves)

            game.apply_move(best_move)
            if not playSelf:
                realPlayer = True
        else:
            row = int(input("row: "))
            col = int(input("col: "))
            game.apply_move(np.array([row, col]))
            realPlayer = False
