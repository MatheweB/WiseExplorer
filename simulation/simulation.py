# Libraries
import random
import numpy as np
from typing import Dict, Tuple, List, Optional
from omnicron.manager import GameMemory

# Classes
from agent.agent import Agent
from agent.agent import State
from games.game_base import GameBase
from games.game_state import GameState

# Function
from wise_explorer import wise_explorer_algorithm


def randomly_assign_agent_pairs(
    agents: List[Agent], anti_agents: List[Agent], n: Optional[int] = None
) -> List[Tuple[int, int]]:
    """
    Randomly pair indices from two agent lists.
    Parameters
    ----------
    agents : Sequence
        First list of agents.
    anti_agents : Sequence
        Second list of agents.
    n : int, optional
        Maximum number of pairs to return (default: as many as possible).
    Returns
    -------
    list[tuple[int, int]]
        List of `(i, j)` index pairs, shuffled randomly.
    """
    idx1 = list(range(len(agents)))
    idx2 = list(range(len(anti_agents)))
    random.shuffle(idx1)
    random.shuffle(idx2)
    max_pairs = min(len(idx1), len(idx2))
    if n is not None:
        max_pairs = min(max_pairs, n)
    return list(zip(idx1[:max_pairs], idx2[:max_pairs]))


# --- Helper that determines *how* to modify a pair of agents after a move has been made ---
# The tuple is: (opponent_state, player_change, opponent_change)
_TERMINAL_MAP: Dict[State, Tuple[State, bool, bool]] = {
    State.WIN: (State.LOSS, False, True),
    State.TIE: (State.TIE, True, True),
    State.LOSS: (State.WIN, True, False),
}


def _simulate_game(
    agent: Agent,
    anti_agent: Agent,
    game: GameBase,
    turn_depth: int,
    omnicron: GameMemory,
) -> None:
    """
    Play a deterministic two-player game for *turn_depth* full cycles
    (agent + anti_agent per cycle).
    Parameters
    ----------
    agent, anti_agent : Agent
        The two participants.  The simulator will set their `move`,
        `game_state`, `change` and `move_depth` attributes in-place.
    turn_depth : int
        Number of turns (each turn consists of *agent* followed by
        *anti_agent*).  Must be > 0.
    game
        A :class:`GameBase` that implements a ``get_result(move)`` method returning
        a :class:`State` for the current board position.
    Notes
    -----
    The simulator terminates early if any player wins or the game ends
    in a tie.  Otherwise, after *turn_depth* turns all agents are marked
    as `NEUTRAL` and `change = True`.
    """
    if turn_depth <= 0:
        raise ValueError("turn_depth must be a positive integer")

    def _set_terminal(
        player: Agent,
        opponent: Agent,
        player_state: State,
    ) -> None:
        """
        Update *player* and *opponent* after a terminal result.
        """
        opp_state, p_chg, o_chg = _TERMINAL_MAP[player_state]
        player.game_state = player_state
        opponent.game_state = opp_state
        player.change = p_chg
        opponent.change = o_chg

    def _add_to_stack(
        move_state_depth_stack: List[Tuple[np.ndarray, GameState, int]],
        initial_game_state: GameBase,
        player: Agent,
        depth: int,
    ) -> None:
        move_state_depth_stack.append(
            (player.move.copy(), initial_game_state.get_state().clone(), depth)
        )

    # ------------------------------------------------------------------
    # Local helper that applies a single move and updates the agents.
    # ------------------------------------------------------------------
    def _apply_move(
        player: Agent,
        opponent: Agent,
        depth: int,
        player_move_state_depth_stack: List[Tuple[np.ndarray, GameState, int]],
    ) -> State:
        """
        Apply the move that *player* decided on, update both agents, and
        return ``True`` if the game reached a terminal state (WIN, TIE, LOSS).
        """
        # Commit the move
        player.move = player.core_move
        initial_game_state = game.deep_clone()
        # Ask the engine what the result is
        game.apply_move(player.move)
        player.game_state = game.get_result(player.player_id)
        # Record depth for this move (both agents see the same depth)
        player.move_depth = opponent.move_depth = depth
        # Clone the game state to push on to data stack
        # Resolve terminal states (if any)
        if player.game_state in _TERMINAL_MAP:
            _set_terminal(player, opponent, player.game_state)
            _add_to_stack(
                player_move_state_depth_stack, initial_game_state, agent, depth
            )
            return player.game_state  # game finished
        # Otherwise we are still playing
        player.game_state = opponent.game_state = State.NEUTRAL
        player.change = opponent.change = True
        _add_to_stack(player_move_state_depth_stack, initial_game_state, agent, depth)
        return State.NEUTRAL  # game undecided

    # ------------------------------------------------------------------
    # Main simulation loop – one full cycle = two calls to _apply_move.
    # ------------------------------------------------------------------
    depth = 0
    outcome = State.NEUTRAL
    agent_move_state_depth_stack: List[Tuple[np.ndarray, GameState, int]] = []
    anti_agent_move_state_depth_stack: List[Tuple[np.ndarray, GameState, int]] = []
    while depth <= turn_depth:
        # Agent's turn
        depth += 1
        outcome = _apply_move(agent, anti_agent, depth, agent_move_state_depth_stack)
        if outcome in _TERMINAL_MAP:
            break
        # Anti-agent's turn
        depth += 1
        outcome = _apply_move(
            anti_agent, agent, depth, anti_agent_move_state_depth_stack
        )
        if outcome in _TERMINAL_MAP:
            break

    if outcome is None:
        raise ValueError("outcome cannot be None")

    _store_outcome_data(
        game.game_id(),
        agent.game_state,
        agent_move_state_depth_stack,
        anti_agent.game_state,
        anti_agent_move_state_depth_stack,
        omnicron,
    )


def _start_simulation(
    agents: List[Agent],
    anti_agents: List[Agent],
    initial_game: GameBase,
    turn_depth: int,
    omnicron: GameMemory,
) -> GameBase:
    pair_indices = randomly_assign_agent_pairs(agents, anti_agents)
    for agent, anti_agent in ((agents[i1], anti_agents[i2]) for i1, i2 in pair_indices):
        _simulate_game(
            agent, anti_agent, initial_game.deep_clone(), turn_depth, omnicron
        )
    return initial_game


# TODO(Store/record data here) data-structure = [Move :-> (BoardState, Outcome)]
# Omnicron
def _store_outcome_data(
    game_id: str,
    agent_outcome: State,
    agent_move_state_depth_stack: List[Tuple[np.ndarray, GameState, int]],
    anti_agent_outcome: State,
    anti_agent_move_state_depth_stack: List[Tuple[np.ndarray, GameState, int]],
    omnicron: GameMemory,
):
    for agent_move, game_state, _depth in agent_move_state_depth_stack:
        omnicron.write(game_id, game_state, agent_outcome, agent_move)

    for anti_agent_move, game_state, _depth in anti_agent_move_state_depth_stack:
        omnicron.write(game_id, game_state, anti_agent_outcome, anti_agent_move)


def _apply_wise_explorer(
    agents: List[Agent],
    anti_agents: List[Agent],
    game: GameBase,
    omnicron: GameMemory,
    is_prune_stage: bool,
) -> None:
    wise_explorer_algorithm.update_agents(
        agents, anti_agents, game, omnicron, is_prune_stage
    )


def start_simulations(
    agents: List[Agent],
    anti_agents: List[Agent],
    game: GameBase,
    turn_depth: int,
    simulations: int,
    omnicron: GameMemory,
):
    done = False
    while not done:
        # Prune stage
        for _ in range(simulations // 2):
            game_from_simulation = _start_simulation(
                agents, anti_agents, game.deep_clone(), turn_depth, omnicron
            )
            # After all agents have played their games, we run wise-explorer to learn before the next simulation run
            _apply_wise_explorer(
                agents, anti_agents, game_from_simulation, omnicron, is_prune_stage=True
            )
        # Optimize stage
        for _ in range(simulations // 2):
            game_from_simulation = _start_simulation(
                agents, anti_agents, game.deep_clone(), turn_depth, omnicron
            )
            # After all agents have played their games, we run wise-explorer to learn before the next simulation run
            _apply_wise_explorer(
                agents, anti_agents, game.deep_clone(), omnicron, is_prune_stage=False
            )

        # We update the game with the best move given the initial state here, and then move on to the next "step", until we're finished!
        best_move = omnicron.get_best_move(game.game_id(), game.get_state().clone())
        if best_move is not None:
            game.apply_move(best_move)
        else:
            game.apply_move(game.valid_moves())

        if game.get_result(game.current_player()) in _TERMINAL_MAP:
            break

    # We pick the best move for the board here, apply it, and then move on to the next "step!"
