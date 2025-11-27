# simulation.py
# --------------------------------------------------------------
# Fixed-agent-per-game N-player simulation with:
#   • agent swarms (any number of agents per player)
#   • matchmade groups (shuffle + zip, one sim per group)
#   • epoch-based re-grouping (1 epoch = K simulations, K = min swarm size)
#   • agent.change management based on game outcomes
#   • agent.change PRESERVED across games (learns from previous outcomes)
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


def reset_agent_preserve_change(agent: Agent) -> None:
    """
    Reset the internal working state of a single agent.
    PRESERVES agent.change so the agent remembers its last outcome.
    """
    # agent.change is PRESERVED (not reset) - this is the agent's memory
    agent.core_move = np.array([])
    agent.move = np.array([])
    agent.game_state = State.NEUTRAL
    agent.move_depth = 0


def _update_agent_change_from_outcome(agent: Agent) -> None:
    """
    Update agent.change based on their final game_state outcome.
    
    Rules:
    - LOSS → change = True
    - TIE → change = True
    - WIN → change = False
    - NEUTRAL → no change (shouldn't happen at game end)
    """
    if agent.game_state == State.LOSS:
        agent.change = True
    elif agent.game_state == State.TIE:
        agent.change = True
    elif agent.game_state == State.WIN:
        agent.change = False
    # NEUTRAL: leave as-is (shouldn't occur at end of game)


def _reset_all_agent_changes(players_map: Dict[int, List[Agent]]) -> None:
    """
    Reset agent.change to False for all agents in the players_map.
    Called at the end of each simulation batch.
    """
    for swarm in players_map.values():
        for agent in swarm:
            agent.change = False


def _generate_matchmade_groups(players_map: Dict[int, List[Agent]]) -> List[Dict[int, Agent]]:
    """
    Generate matchmade groups where each agent is matched with exactly one partner per player.
    Works for any N-player game (1, 2, 3, 4, 5+ players).
    
    For 1-player games: returns one group per agent in the swarm.
    For N-player games: shuffles each swarm and zips them together.
    
    Returns as many groups as the smallest swarm allows.
    """
    player_ids = sorted(players_map.keys())
    
    # Shuffle each player's agent list independently
    shuffled_swarms = {}
    for pid in player_ids:
        swarm = players_map[pid][:]  # Make a copy
        random.shuffle(swarm)
        shuffled_swarms[pid] = swarm
    
    # Find the minimum swarm size (this limits how many groups we can make)
    min_swarm_size = min(len(swarm) for swarm in shuffled_swarms.values())
    
    # Zip agents together to create groups
    groups = []
    for i in range(min_swarm_size):
        group = {pid: shuffled_swarms[pid][i] for pid in player_ids}
        groups.append(group)
    
    return groups


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


def _simulate_game_with_fixed_agents(
    chosen_agents: Dict[int, Agent],
    game: GameBase,
    turn_depth: int,
    omnicron: GameMemory,
    prune_stage: bool,
) -> None:
    """
    Run an entire simulation using pre-chosen agents (one per player).
    At the end, update each agent's change attribute based on outcome.
    
    agent.change is PRESERVED from previous games, allowing agents to
    use their last outcome to inform their strategy in the current game.
    """
    num_players = game.num_players()
    
    # Reset agents but PRESERVE agent.change from their previous game
    for agent in chosen_agents.values():
        reset_agent_preserve_change(agent)

    # Per-player move stacks
    stacks = {p: [] for p in range(1, num_players + 1)}
    final_results = {p: State.NEUTRAL for p in range(1, num_players + 1)}

    depth = 0

    while depth < turn_depth:
        depth += 1

        cp = game.current_player()
        actor = chosen_agents[cp]

        _apply_policy(actor, game, omnicron, prune_stage)
        result = _apply_move_and_record(actor, game, cp, stacks[cp])

        # Terminal check
        if result in _TERMINAL_STATES or game.is_over():
            for pid in final_results:
                final_results[pid] = game.get_result(pid)
            break

    _store_outcome_data(game.game_id(), stacks, final_results, omnicron)
    
    # Update each agent's change attribute based on their outcome
    # This will be used in their NEXT game
    for agent in chosen_agents.values():
        agent.game_state = final_results[agent.player_id]
        _update_agent_change_from_outcome(agent)


# --------------------------------------------------------------
# Batches with epoch-based matchmaking
# --------------------------------------------------------------


def _run_simulation_batch(
    players_map: Dict[int, List[Agent]],
    game: GameBase,
    turn_depth: int,
    count: int,
    omnicron: GameMemory,
    prune_stage: bool,
) -> None:
    """
    Runs `count` simulations using epoch-based matchmade groups.
    
    Each epoch:
        1. Generate K matchmade groups (K = min swarm size)
        2. Run ONE simulation per group
        3. Re-shuffle and repeat
    
    Stops after exactly `count` total simulations.
    At the end, resets all agent.change back to False.
    """
    total_sims_done = 0
    
    while total_sims_done < count:
        # Generate fresh matchmade groups for this epoch
        groups = _generate_matchmade_groups(players_map)
        
        # Run one simulation per group (or until we hit count)
        for group in groups:
            if total_sims_done >= count:
                break
            
            cloned = game.deep_clone()
            _simulate_game_with_fixed_agents(group, cloned, turn_depth, omnicron, prune_stage)
            total_sims_done += 1
    
    # Reset all agent.change back to False after batch completes
    _reset_all_agent_changes(players_map)


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
        best_move = omnicron.get_best_move(
            game.game_id(), current_state, debug_move=True
        )

        if best_move is None:
            valid = game.valid_moves()
            if not valid:
                break
            best_move = random.choice(valid)

        game.apply_move(best_move)
        
        if game.is_over():
            break