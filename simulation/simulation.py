# simulation.py
# --------------------------------------------------------------
# N-player Monte Carlo simulation system with:
#   • Agent swarms with matchmaking (shuffle + zip grouping)
#   • Epoch-based re-grouping for exploration diversity
#   • Agent memory persistence (change flag carries across games)
#   • Full player permutation symmetry (uncapped N! efficiency)
#   • Parallel simulation execution
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
from itertools import permutations
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent.agent import Agent, State
from games.game_base import GameBase
from games.game_state import GameState
from omnicron.manager import GameMemory
from wise_explorer import wise_explorer_algorithm

MoveStackItem = Tuple[np.ndarray, GameState, int]
_TERMINAL_STATES = {State.WIN, State.LOSS, State.TIE}

# Performance tuning
PARALLEL_WORKERS = 16  # Concurrent simulations


# ==============================================================
# AGENT STATE MANAGEMENT
# ==============================================================


def _reset_agent_preserve_memory(agent: Agent) -> None:
    """Reset agent's transient state while preserving memory."""
    agent.core_move = np.array([])
    agent.move = np.array([])
    agent.game_state = State.NEUTRAL
    agent.move_depth = 0


def _update_agent_memory(agent: Agent) -> None:
    """
    Update agent memory based on game outcome.
    
    WIN  → change = False (success)
    LOSS → change = True  (failure)
    TIE  → change = True  (suboptimal)
    """
    agent.change = agent.game_state in {State.LOSS, State.TIE}


def _reset_all_agent_memories(players_map: Dict[int, List[Agent]]) -> None:
    """Reset all agent memories after batch completion."""
    for swarm in players_map.values():
        for agent in swarm:
            agent.change = False


# ==============================================================
# MATCHMAKING
# ==============================================================


def _generate_matchmade_groups(
    players_map: Dict[int, List[Agent]], player_ids: List[int]
) -> List[Dict[int, Agent]]:
    """
    Generate matchmade groups via shuffle-and-zip.
    Returns K groups where K = min(swarm_size).
    """
    # Shuffle each swarm independently
    shuffled = {pid: random.sample(players_map[pid], len(players_map[pid])) for pid in player_ids}
    
    # Zip together (limited by smallest swarm)
    min_size = min(len(swarm) for swarm in shuffled.values())
    return [{pid: shuffled[pid][i] for pid in player_ids} for i in range(min_size)]


# ==============================================================
# GAME SIMULATION
# ==============================================================


def _apply_policy(agent: Agent, game: GameBase, omnicron: GameMemory, prune: bool) -> None:
    """Compute agent's next move using Wise Explorer."""
    wise_explorer_algorithm.update_agent(agent, game, omnicron, prune)


def _execute_move(
    agent: Agent, game: GameBase, acting_player: int, stack: List[MoveStackItem]
) -> State:
    """Execute move and record transition."""
    snapshot = game.get_state().clone()
    agent.move = agent.core_move
    move_copy = np.array(agent.move, copy=True)
    
    stack.append((move_copy, snapshot, acting_player))
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
) -> Tuple[str, Dict[int, List[MoveStackItem]], Dict[int, State]]:
    """
    Run complete game simulation.
    Returns game data for storage.
    """
    # Reset transient state, preserve memory
    for agent in chosen_agents.values():
        _reset_agent_preserve_memory(agent)
    
    # Pre-allocate structures
    stacks = {p: [] for p in range(1, num_players + 1)}
    final_results = {p: State.NEUTRAL for p in range(1, num_players + 1)}
    
    # Play to completion or depth limit
    for _ in range(turn_depth):
        current_player = game.current_player()
        actor = chosen_agents[current_player]
        
        _apply_policy(actor, game, omnicron, prune_stage)
        result = _execute_move(actor, game, current_player, stacks[current_player])
        
        if result in _TERMINAL_STATES or game.is_over():
            for pid in final_results:
                final_results[pid] = game.get_result(pid)
            break
    
    # Update agent memories
    for agent in chosen_agents.values():
        agent.game_state = final_results[agent.player_id]
        _update_agent_memory(agent)
    
    return game.game_id(), stacks, final_results


# ==============================================================
# PLAYER PERMUTATION SYMMETRY
# ==============================================================


# def _permute_board(board: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
#     """Remap player IDs on board using vectorized operations."""
#     result = np.zeros_like(board)
#     for old_pid, new_pid in mapping.items():
#         result[board == old_pid] = new_pid
#     result[board == 0] = 0  # Preserve empty squares
#     return result


# def _permute_game_state(state: GameState, mapping: Dict[int, int]) -> GameState:
#     """Create permuted state with remapped players."""
#     permuted = state.clone()
#     permuted.board = _permute_board(state.board, mapping)
#     permuted.current_player = mapping.get(state.current_player, state.current_player)
#     return permuted


def _store_with_permutations(
    game_id: str,
    stacks: Dict[int, List[MoveStackItem]],
    results: Dict[int, State],
    player_ids: List[int],
    omnicron: GameMemory,
) -> None:
    """
    Store original + all permutations.
    
    Efficiency: 1! = 1x, 2! = 2x, 3! = 6x, 4! = 24x, 5! = 120x
    """
    # num_players = len(player_ids)
    
    # Store original
    for pid, stack in stacks.items():
        outcome = results[pid]
        for move_arr, snapshot_state, acting_player in stack:
            omnicron.write(game_id, snapshot_state, acting_player, outcome, move_arr)
    
    # if num_players <= 1:
    #     return  # No permutations for 1-player
    
    # # Fast path for 2-player (most common case)
    # if num_players == 2:
    #     p1, p2 = player_ids
    #     mapping = {p1: p2, p2: p1}
    #     inv_results = {p2: results[p1], p1: results[p2]}
        
    #     for pid, stack in stacks.items():
    #         inv_pid = mapping[pid]
    #         inv_outcome = inv_results[inv_pid]
            
    #         for move_arr, snapshot_state, acting_player in stack:
    #             inv_state = _permute_game_state(snapshot_state, mapping)
    #             inv_acting = mapping[acting_player]
    #             omnicron.write(game_id, inv_state, inv_acting, inv_outcome, move_arr)
    #     return
    
    # # General N-player: all permutations (uncapped)
    # for perm in permutations(player_ids):
    #     if perm == tuple(player_ids):
    #         continue  # Skip identity
        
    #     mapping = dict(zip(player_ids, perm))
    #     perm_results = {mapping[pid]: results[pid] for pid in player_ids}
        
    #     for pid, stack in stacks.items():
    #         perm_pid = mapping[pid]
    #         perm_outcome = perm_results[perm_pid]
            
    #         for move_arr, snapshot_state, acting_player in stack:
    #             perm_state = _permute_game_state(snapshot_state, mapping)
    #             perm_acting = mapping[acting_player]
    #             omnicron.write(game_id, perm_state, perm_acting, perm_outcome, move_arr)


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
    """Execute batch with parallelization."""
    num_players = len(player_ids)
    total_sims = 0
    
    while total_sims < count:
        groups = _generate_matchmade_groups(players_map, player_ids)
        
        # Prepare parallel tasks
        tasks = []
        for group in groups:
            if total_sims >= count:
                break
            tasks.append((group, game.deep_clone(), turn_depth, omnicron, prune_stage, num_players))
            total_sims += 1
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = [executor.submit(_simulate_single_game, *task) for task in tasks]
            
            for future in as_completed(futures):
                game_id, stacks, results = future.result()
                _store_with_permutations(game_id, stacks, results, player_ids, omnicron)
    
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
    """
    Two-phase MCTS with parallel execution.
    
    Phase 1: Prune (explore bad paths)
    Phase 2: Optimize (refine good paths)
    """
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if turn_depth <= 0:
        raise ValueError("turn_depth must be positive")
    
    # Cache player IDs (computed once)
    player_ids = sorted(players_map.keys())
    
    while not game.is_over():
        # Two-phase simulation
        prune_count = simulations // 2
        optimize_count = simulations - prune_count
        
        _run_simulation_batch(
            players_map, game, turn_depth, prune_count, omnicron, True, player_ids
        )
        _run_simulation_batch(
            players_map, game, turn_depth, optimize_count, omnicron, False, player_ids
        )
        
        # Select and execute best move
        current_state = game.get_state().clone()
        best_move = omnicron.get_best_move(game.game_id(), current_state, debug_move=True)
        
        if best_move is None:
            valid_moves = game.valid_moves()
            if not valid_moves:
                break
            best_move = random.choice(valid_moves)
        
        game.apply_move(best_move)