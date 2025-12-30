"""
N-player Monte Carlo simulation with multiprocessing.

Overview
--------
Runs parallel game simulations to train agents via wise_explorer.

Key concepts:
- SWARMS: Each player has a "swarm" of agents that act as state holders.
    They don't play directly—temporary agents copy their state, play a game,
    and sync results back. This enables parallel sims while preserving memory
    across rounds (e.g., "I lost last time, explore more").

- JOBS: Self-contained units of work (simulation parameters) distributable to any worker.

- ROUNDS: All games in a round complete before writing to DB. This ensures
    no partial games are recorded and learning is consistent.

Recommended usage with partitioned memory:

    from games.tic_tac_toe import TicTacToe

    game = TicTacToe()
    memory = GameMemory.for_game(game, base_dir="data/memory")  # Creates tic_tac_toe.db
    swarms = {1: [Agent() for _ in range(20)], 2: [Agent() for _ in range(20)]}

    start_simulations(swarms, game, turn_depth=20, simulations=200, memory=memory)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from multiprocessing.pool import Pool
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from agent.agent import Agent, State
from games.game_base import GameBase
from omnicron.manager import GameMemory
from wise_explorer import wise_explorer_algorithm


logger = logging.getLogger(__name__)

DEFAULT_WORKER_COUNT = max(1, mp.cpu_count() - 1)
TERMINAL_STATES = frozenset({State.WIN, State.LOSS, State.TIE})


# ---------------------------------------------------------------------------
# Data Structures (Picklable)
# ---------------------------------------------------------------------------

@dataclass
class MoveRecord:
    """Lightweight struct to hold move data for storage."""
    move: np.ndarray
    board_before: np.ndarray
    player_who_moved: int


@dataclass(frozen=True)
class GameJob:
    """
    A completely self-contained job description.
    Can be pickled and sent to any worker.
    """
    game: GameBase
    player_map: Dict[int, int]  # player_id -> swarm_index
    exploration_settings: Dict[int, bool]  # player_id -> should explore?
    max_turns: int
    is_prune_phase: bool


@dataclass
class JobResult:
    """The result payload returned by a worker."""
    move_history: Dict[int, List[MoveRecord]]
    outcomes: Dict[int, State]
    player_map: Dict[int, int]  # Returned to map results back to swarms
    game_class: type


# ---------------------------------------------------------------------------
# Worker Process Logic
# ---------------------------------------------------------------------------

_worker_memory: Optional[GameMemory] = None


def _worker_init(db_path: str) -> None:
    """Initializes the persistent DB connection for the worker process."""
    global _worker_memory
    _worker_memory = GameMemory(db_path, read_only=True)
    # Note: read-only WAL connections don't create write locks,
    # so no explicit cleanup needed


def play_simulation_job(job: GameJob) -> JobResult:
    """
    Executes a single game simulation.
    This is the distributable unit of work.
    """
    if _worker_memory is None:
        raise RuntimeError("Worker not initialized with DB connection.")

    game = job.game
    player_ids = sorted(job.player_map.keys())

    # Setup local agents for this simulation
    agents: Dict[int, Agent] = {}
    for pid in player_ids:
        agent = Agent()
        agent.player_id = pid
        agent.change = job.exploration_settings.get(pid, False)
        agents[pid] = agent

    move_history: Dict[int, List[MoveRecord]] = {pid: [] for pid in player_ids}

    # Run game loop
    for _ in range(job.max_turns):
        if game.is_over():
            break

        current_pid = game.current_player()
        agent = agents[current_pid]

        # Decision making via wise_explorer
        wise_explorer_algorithm.update_agent(
            agent, game, _worker_memory, job.is_prune_phase
        )

        # Snapshot state BEFORE move
        board_snapshot = game.get_state().board.copy()
        move = np.array(agent.core_move, copy=True)

        # Apply move
        game.apply_move(move)
        agent.move = move

        # Log move
        move_history[current_pid].append(
            MoveRecord(
                move=move,
                board_before=board_snapshot,
                player_who_moved=current_pid,
            )
        )

        # Correct terminal detection
        if game.is_over():
            break

    # Finalize outcomes
    outcomes = {pid: game.get_result(pid) for pid in player_ids}

    return JobResult(
        move_history=move_history,
        outcomes=outcomes,
        player_map=job.player_map,
        game_class=type(game),
    )


# ---------------------------------------------------------------------------
# Simulation Orchestrator (Main Process)
# ---------------------------------------------------------------------------

class SimulationRunner:
    """Manages the pool of workers and the job pipeline."""

    def __init__(self, memory: GameMemory, num_workers: int = DEFAULT_WORKER_COUNT):
        self.memory = memory
        self.num_workers = num_workers
        self._pool: Optional[Pool] = None
        self._round_id = 0

    def __enter__(self):
        self._start_pool()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def _start_pool(self) -> Pool:
        """Lazy init - returns the pool, creating if needed."""
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
        base_game: GameBase,
        total_sims: int,
        is_prune: bool,
        max_turns: int,
    ) -> int:
        """
        Runs a batch of simulations.

        All games in a round complete before any writes occur.
        Returns the total number of move transitions recorded.
        """
        if total_sims <= 0 or not swarms:
            return 0

        swarm_size = min(len(s) for s in swarms.values())
        if swarm_size == 0:
            return 0

        pool = self._start_pool()
        sims_remaining = total_sims
        total_transitions = 0

        while sims_remaining > 0:
            self._round_id += 1
            round_size = min(swarm_size, sims_remaining)

            # 1. Create jobs with random pairings
            jobs = self._create_jobs(
                swarms, base_game, round_size, is_prune, max_turns
            )

            # 2. Execute all games in parallel
            results = pool.map(play_simulation_job, jobs)

            # Debug invariants
            assert len(results) == len(jobs), "Lost simulation results"

            # 3. Write all stacks atomically
            total_transitions += self._commit_round(results, record_neutral_results=False)

            # 4. Update swarm states
            for result in results:
                self._sync_swarm_state(swarms, result)

            sims_remaining -= round_size

        return total_transitions

    def _commit_round(self, results: List[JobResult], record_neutral_results: bool = False) -> int:
        """
        Write all game stacks from a round in one atomic transaction.
        We can choose whether to record the results of games that end in "neutral" (default = False)

        """
        game_classes = {r.game_class for r in results}
        assert len(game_classes) == 1, f"Mixed game classes in round (not allowed): {game_classes}"
        game_class = game_classes.pop()

        all_stacks = []

        for result in results:
            for pid, moves in result.move_history.items():
                outcome = result.outcomes[pid]
                if not record_neutral_results and outcome == State.NEUTRAL:
                    continue
                move_tuples = [
                    (rec.move, rec.board_before, rec.player_who_moved)
                    for rec in moves
                ]
                all_stacks.append((move_tuples, outcome))

        if not all_stacks:
            return 0

        logger.debug(
            "Committing round %d with %d stacks",
            self._round_id,
            len(all_stacks),
        )

        return self.memory.record_round(game_class, all_stacks)

    def _create_jobs(
        self,
        swarms: Dict[int, List[Agent]],
        game: GameBase,
        count: int,
        is_prune: bool,
        max_turns: int,
    ) -> List[GameJob]:
        """Generates a list of randomized match-ups."""
        jobs = []
        player_ids = sorted(swarms.keys())

        shuffled_indices = {
            pid: random.sample(range(len(swarms[pid])), count)
            for pid in player_ids
        }

        for i in range(count):
            p_map = {pid: shuffled_indices[pid][i] for pid in player_ids}
            exp_settings = {
                pid: swarms[pid][idx].change for pid, idx in p_map.items()
            }

            jobs.append(
                GameJob(
                    game=game.deep_clone(),
                    player_map=p_map,
                    exploration_settings=exp_settings,
                    max_turns=max_turns,
                    is_prune_phase=is_prune,
                )
            )
        return jobs

    def _sync_swarm_state(self, swarms: Dict[int, List[Agent]], result: JobResult):
        """Updates agent exploration flags based on results."""
        for pid, swarm_idx in result.player_map.items():
            outcome = result.outcomes[pid]
            swarms[pid][swarm_idx].change = outcome in {State.LOSS, State.TIE}


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
) -> None:
    """
    Main entry point. Orchestrates the Training (Prune/Optimize) and Play cycle.

    Args:
        agent_swarms: Maps player_id -> list of agents (state holders)
        game: The game instance to play
        turn_depth: Max turns per simulated game
        simulations: Number of simulations per training step
        memory: Database for storing/retrieving learned moves
        num_workers: Number of parallel worker processes
        training_enabled: If False, skip training and just play from existing memory
    """
    play_against_self = False  # Set to False for human vs AI
    human_turn = True
    training_enabled = False

    runner = SimulationRunner(memory, num_workers)
    print(
        f"Starting engine with {num_workers} workers. "
        f"Training: {'ON' if training_enabled else 'OFF'}"
    )

    try:
        with runner:
            while not game.is_over():
                # --- AI Training Step ---
                if not human_turn and training_enabled:
                    count_prune = simulations // 2
                    runner.run_batch(
                        agent_swarms,
                        game,
                        count_prune,
                        is_prune=True,
                        max_turns=turn_depth,
                    )

                    runner.run_batch(
                        agent_swarms,
                        game,
                        simulations - count_prune,
                        is_prune=False,
                        max_turns=turn_depth,
                    )

                # --- Move Selection ---
                if not human_turn:
                    best_move = memory.get_best_move(game, debug=True)

                    if best_move is None:
                        valid = game.valid_moves()
                        if len(valid) == 0:
                            break
                        best_move = random.choice(valid)
                        print(f"AI selected random move: {best_move}")
                        game.apply_move(best_move)
                        print(game.state_string())
                    else:
                        print(f"AI selected best move: {best_move}")
                        game.apply_move(best_move)

                    if not play_against_self:
                        human_turn = True
                else:
                    print(f"Current Player {game.current_player()} (Human)")
                    try:
                        row = int(input("row: "))
                        col = int(input("col: "))
                        game.apply_move(np.array([row, col]))
                        human_turn = False
                    except ValueError:
                        print("Invalid input, try again.")
                        continue

                print(f"Game State Updated. Next Player: {game.current_player()}")

    except KeyboardInterrupt:
        print("\nInterrupted - shutting down workers...")
        runner.shutdown(force=True)
    except Exception:
        logger.exception("Fatal error in simulation loop")
        raise
    finally:
        memory.close()
