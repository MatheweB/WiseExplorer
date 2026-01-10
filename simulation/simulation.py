"""
N-player Monte Carlo simulation with multiprocessing.

Overview
--------
Runs parallel game simulations to train agents via wise_explorer.

Key concepts:
- SWARMS: Each player has a "swarm" of agents that act as state holders.
    They don't play directly—temporary agents copy their state, play a game,
    and sync results back.

- JOBS: Self-contained units of work (simulation parameters) distributable to any worker.

- ROUNDS: All games in a round complete before writing to DB. This ensures
    no partial games are recorded and learning is consistent.

- PHASES: Training alternates between:
    * Prune phase: ALL agents play worst moves (worst vs worst) - find bad lines
    * Exploit phase: ALL agents play best moves (best vs best) - reinforce good lines

Recommended usage with partitioned memory:

    from games.tic_tac_toe import TicTacToe

    game = TicTacToe()
    memory = GameMemory.for_game(game, base_dir="data/memory")
    swarms = {1: [Agent() for _ in range(20)], 2: [Agent() for _ in range(20)]}

    start_simulations(swarms, game, turn_depth=20, simulations=200, memory=memory)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import random
from multiprocessing.pool import Pool
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from agent.agent import Agent, State
from games.game_base import GameBase
from omnicron.manager import GameMemory
from wise_explorer import wise_explorer_algorithm


logger = logging.getLogger(__name__)

DEFAULT_WORKER_COUNT = max(1, mp.cpu_count() - 1)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class MoveRecord:
    """A single move with its context."""

    move: np.ndarray
    board_before: np.ndarray
    player: int


@dataclass(frozen=True)
class GameJob:
    """
    A completely self-contained job description.
    Can be pickled and sent to any worker.
    """

    game: GameBase
    player_map: Dict[int, int]  # player_id -> swarm_index
    max_turns: int
    is_prune: bool  # Global phase: True = worst vs worst, False = best vs best


@dataclass
class JobResult:
    """The result payload returned by a worker."""

    moves: Dict[int, List[MoveRecord]]  # player_id -> move history
    outcomes: Dict[int, State]  # player_id -> final state
    player_map: Dict[int, int]  # Echo back for swarm sync
    game_class: type


# ---------------------------------------------------------------------------
# Worker Process
# ---------------------------------------------------------------------------

_worker_memory: Optional[GameMemory] = None


def _worker_init(db_path: str) -> None:
    """Initialize DB connection for this worker process."""
    global _worker_memory
    _worker_memory = GameMemory(db_path, read_only=True)


def run_game(job: GameJob) -> JobResult:
    """
    Execute a single game simulation.
    This is the distributable unit of work.
    """
    if _worker_memory is None:
        raise RuntimeError("Worker not initialized")

    game = job.game
    players = sorted(job.player_map.keys())

    # Create local agents
    agents = {pid: _make_agent(pid) for pid in players}

    # Play the game
    moves: Dict[int, List[MoveRecord]] = {pid: [] for pid in players}

    for _ in range(job.max_turns):
        if game.is_over():
            break

        pid = game.current_player()
        agent = agents[pid]

        # Get move from wise_explorer (global phase applies to all)
        wise_explorer_algorithm.update_agent(agent, game, _worker_memory, job.is_prune)

        # Record and apply
        board_before = game.get_state().board.copy()
        move = np.array(agent.core_move, copy=True)

        game.apply_move(move)
        moves[pid].append(MoveRecord(move, board_before, pid))

    return JobResult(
        moves=moves,
        outcomes={pid: game.get_result(pid) for pid in players},
        player_map=job.player_map,
        game_class=type(game),
    )


def _make_agent(player_id: int) -> Agent:
    """Create an agent configured for simulation."""
    agent = Agent()
    agent.player_id = player_id
    return agent


# ---------------------------------------------------------------------------
# Simulation Runner
# ---------------------------------------------------------------------------


class SimulationRunner:
    """Manages the pool of workers and the job pipeline."""

    def __init__(self, memory: GameMemory, num_workers: int = DEFAULT_WORKER_COUNT):
        self.memory = memory
        self.num_workers = num_workers
        self._pool: Optional[Pool] = None
        self._round_id = 0

    def __enter__(self):
        self._ensure_pool()
        return self

    def __exit__(self, *_):
        self.shutdown()

    def _ensure_pool(self) -> Pool:
        """Lazy pool initialization."""
        if self._pool is None:
            self._pool = mp.Pool(
                processes=self.num_workers,
                initializer=_worker_init,
                initargs=(str(self.memory.db_path),),
            )
        return self._pool

    def shutdown(self, force: bool = False):
        """
        Clean up the worker pool.

        Args:
            force: If True, terminate immediately (for Ctrl+C handling).
        """
        if self._pool:
            if force:
                self._pool.terminate()
            else:
                self._pool.close()
            self._pool.join()
            self._pool = None

    def run_batch(
        self,
        swarms: Dict[int, List[Agent]],
        game: GameBase,
        num_sims: int,
        max_turns: int,
        is_prune: bool,
    ) -> int:
        """
        Runs a batch of simulations with the given phase.

        Args:
            swarms: Player ID -> list of agents
            game: Game to simulate from
            num_sims: Number of simulations to run
            max_turns: Maximum turns per game
            is_prune: True = worst vs worst, False = best vs best

        Returns the total number of move transitions recorded.
        """
        if num_sims <= 0 or not swarms:
            return 0

        swarm_size = min(len(s) for s in swarms.values())
        if swarm_size == 0:
            return 0

        pool = self._ensure_pool()
        remaining = num_sims
        total_transitions = 0

        while remaining > 0:
            self._round_id += 1
            batch_size = min(swarm_size, remaining)

            # 1. Create and run jobs
            jobs = self._make_jobs(swarms, game, batch_size, max_turns, is_prune)
            results = pool.map(run_game, jobs)

            # 2. Commit results atomically
            total_transitions += self._commit(results, skip_neutral=True)

            remaining -= batch_size

        return total_transitions

    def _make_jobs(
        self,
        swarms: Dict[int, List[Agent]],
        game: GameBase,
        count: int,
        max_turns: int,
        is_prune: bool,
    ) -> List[GameJob]:
        """Generate randomized match-ups."""
        players = sorted(swarms.keys())

        # Shuffle indices for random pairings
        indices = {
            pid: random.sample(range(len(swarms[pid])), count) for pid in players
        }

        return [
            GameJob(
                game=game.deep_clone(),
                player_map={pid: indices[pid][i] for pid in players},
                max_turns=max_turns,
                is_prune=is_prune,
            )
            for i in range(count)
        ]

    def _commit(self, results: List[JobResult], skip_neutral: bool = True) -> int:
        """
        Write all game stacks from a round in one atomic transaction.

        Args:
            results: Job results from workers
            skip_neutral: If True, don't record games that ended in NEUTRAL
        """
        if not results:
            return 0

        game_class = results[0].game_class

        # Build stacks: List[(moves, outcome)]
        stacks = []
        for result in results:
            for pid, move_list in result.moves.items():
                outcome = result.outcomes[pid]
                if skip_neutral and outcome == State.NEUTRAL:
                    continue

                stacks.append(
                    (
                        [(m.move, m.board_before, m.player) for m in move_list],
                        outcome,
                    )
                )

        if not stacks:
            return 0

        logger.debug("Committing round %d: %d stacks", self._round_id, len(stacks))
        return self.memory.record_round(game_class, stacks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start_simulations(
    agent_swarms: Dict[int, List[Agent]],
    game: GameBase,
    turn_depth: int,
    simulations: int,
    memory: GameMemory,
    num_workers: int = DEFAULT_WORKER_COUNT,
    training_enabled: bool = True,
    human_players: Optional[List[int]] = None,
    debug_move_statistics: bool = True,
) -> None:
    """
    Main entry point. Orchestrates Training and Play cycle.

    Training runs in two symmetric phases:
      1. Prune phase (worst vs worst) - find and record bad lines
      2. Exploit phase (best vs best) - reinforce good lines

    Args:
        agent_swarms: Maps player_id -> list of agents
        game: The game instance to play
        turn_depth: Max turns per simulated game
        simulations: Number of simulations per training step
        memory: Database for storing/retrieving learned moves
        num_workers: Number of parallel worker processes
        training_enabled: If False, skip training and just play from existing memory
        human_players: List of player IDs controlled by human input
        debug_move_statistics: If True, print move analysis during AI turns
    """
    human_set = set(human_players) if human_players else set()
    runner = SimulationRunner(memory, num_workers)

    print(
        f"Starting sim with {num_workers} workers. Training: {'ON' if training_enabled else 'OFF'}"
    )
    print(game.state_string())

    try:
        with runner:
            while not game.is_over():
                current = game.current_player()
                is_human = current in human_set

                # Get and apply move
                if is_human:
                    move = _human_turn(game)
                    print(f"\nYou played: {_format_move(move)}")
                else:
                    if training_enabled:
                        _train(runner, agent_swarms, game, simulations, turn_depth)
                    move = _ai_turn(
                        game, memory,
                        deterministic_top_moves=True,
                        debug_move_statistics=debug_move_statistics
                    )
                    if move is None:
                        break
                    print(f"\nAI (Player {current}) played: {_format_move(move)}")

                print(game.state_string())

            _print_result()

    except KeyboardInterrupt:
        print("\nInterrupted - shutting down...")
        runner.shutdown(force=True)
    except Exception:
        logger.exception("Fatal error in simulation loop")
        raise
    finally:
        memory.close()


def _train(
    runner: SimulationRunner,
    swarms: Dict[int, List[Agent]],
    game: GameBase,
    simulations: int,
    turn_depth: int,
):
    """
    Run training simulations in two phases:
      1. Prune phase (worst vs worst) - first half
      2. Exploit phase (best vs best) - second half
    """
    prune_count = simulations // 2
    exploit_count = simulations - prune_count

    # Phase 1: Prune (worst vs worst)
    runner.run_batch(swarms, game, prune_count, max_turns=turn_depth, is_prune=True)

    # Phase 2: Exploit (best vs best)
    runner.run_batch(swarms, game, exploit_count, max_turns=turn_depth, is_prune=False)


def _ai_turn(
    game: GameBase, memory: GameMemory, deterministic_top_moves: bool, debug_move_statistics: bool
) -> Optional[np.ndarray]:
    """AI selects and applies move. Returns the move made, or None if no valid moves."""
    valid_moves = game.valid_moves()
    move = memory.get_best_move(
        game, valid_moves,
        deterministic=deterministic_top_moves,
        debug=debug_move_statistics
    )

    if move is None:
        if len(valid_moves) == 0:
            return None
        move = random.choice(valid_moves)

    game.apply_move(move)
    return move


def _format_move(move: np.ndarray) -> str:
    """Format move for display."""
    return ",".join(map(str, move))


def _human_turn(game: GameBase) -> np.ndarray:
    """Get and apply human move. Returns the move made."""
    valid = game.valid_moves()
    if len(valid) > 0:
        example = valid[0]
        print(f"\nYour turn (Player {game.current_player()})")
        print(
            f"Format: {len(example)} values comma-separated (e.g., {','.join(map(str, example))})"
        )

    while True:
        try:
            raw = input("Move: ").strip()
            values = [int(x.strip()) for x in raw.split(",")]
            move = np.array(values)
            game.apply_move(move)
            return move
        except ValueError as e:
            print(f"Invalid input: {e}")
        except Exception as e:
            print(f"Illegal move: {e}")


def _print_result():
    """Print game over banner."""
    print("\n" + "=" * 40)
    print("GAME OVER")
    print("=" * 40)
