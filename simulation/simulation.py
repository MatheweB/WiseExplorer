"""
Parallel Monte Carlo game simulation for training.

Usage:
    game = TicTacToe()
    memory = GameMemory.for_game(game, base_dir="data/memory")
    swarms = {1: [Agent() for _ in range(20)], 2: [Agent() for _ in range(20)]}
    start_simulations(swarms, game, turn_depth=20, simulations=200, memory=memory)

Training alternates between two phases:
    - Prune: All agents play worst moves (find bad lines)
    - Exploit: All agents play best moves (reinforce good lines)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from agent.agent import Agent, State
from games.game_base import GameBase
from omnicron.manager import GameMemory
from wise_explorer.wise_explorer_algorithm import select_move, select_move_for_training

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
    """Self-contained job for a worker process."""
    game: GameBase
    player_map: Dict[int, int]  # player_id -> swarm_index
    max_turns: int
    is_prune: bool


@dataclass
class JobResult:
    """Result from a completed game simulation."""
    moves: Dict[int, List[MoveRecord]]
    outcomes: Dict[int, State]
    player_map: Dict[int, int]
    game_class: type


# ---------------------------------------------------------------------------
# Worker Process
# ---------------------------------------------------------------------------

_worker_memory: Optional[GameMemory] = None


def _worker_init(db_path: str) -> None:
    """Initialize DB connection for worker process."""
    global _worker_memory
    _worker_memory = GameMemory(db_path, read_only=True)


def run_game(job: GameJob) -> JobResult:
    """Execute a single game simulation (runs in worker process)."""
    if _worker_memory is None:
        raise RuntimeError("Worker not initialized")

    game = job.game
    players = sorted(job.player_map.keys())
    moves: Dict[int, List[MoveRecord]] = {pid: [] for pid in players}

    for _ in range(job.max_turns):
        if game.is_over():
            break

        pid = game.current_player()
        board_before = game.get_state().board.copy()
        
        move = select_move_for_training(game, _worker_memory, job.is_prune)
        move = np.array(move, copy=True)
        
        game.apply_move(move)
        moves[pid].append(MoveRecord(move, board_before, pid))

    return JobResult(
        moves=moves,
        outcomes={pid: game.get_result(pid) for pid in players},
        player_map=job.player_map,
        game_class=type(game),
    )


# ---------------------------------------------------------------------------
# Simulation Runner
# ---------------------------------------------------------------------------


class SimulationRunner:
    """Manages parallel simulation workers."""

    def __init__(self, memory: GameMemory, num_workers: int = DEFAULT_WORKER_COUNT):
        self.memory = memory
        self.num_workers = num_workers
        self._pool: Optional[mp.Pool] = None
        self._round_id = 0

    def __enter__(self):
        self._ensure_pool()
        return self

    def __exit__(self, *_):
        self.shutdown()

    def _ensure_pool(self) -> mp.Pool:
        if self._pool is None:
            self._pool = mp.Pool(
                processes=self.num_workers,
                initializer=_worker_init,
                initargs=(str(self.memory.db_path),),
            )
        return self._pool

    def shutdown(self, force: bool = False):
        """Clean up worker pool. Use force=True for interrupt handling."""
        if self._pool:
            self._pool.terminate() if force else self._pool.close()
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
        Run a batch of simulations.
        
        Returns total transitions recorded.
        """
        if num_sims <= 0 or not swarms:
            return 0

        swarm_size = min(len(s) for s in swarms.values())
        if swarm_size == 0:
            return 0

        pool = self._ensure_pool()
        total_transitions = 0
        remaining = num_sims

        while remaining > 0:
            self._round_id += 1
            batch_size = min(swarm_size, remaining)
            
            jobs = self._make_jobs(swarms, game, batch_size, max_turns, is_prune)
            results = pool.map(run_game, jobs)
            total_transitions += self._commit(results)
            
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
        indices = {pid: random.sample(range(len(swarms[pid])), count) for pid in players}

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
        """Write game results to memory. Returns transitions recorded."""
        if not results:
            return 0

        stacks = []
        for result in results:
            for pid, move_list in result.moves.items():
                outcome = result.outcomes[pid]
                if skip_neutral and outcome == State.NEUTRAL:
                    continue
                stacks.append((
                    [(m.move, m.board_before, m.player) for m in move_list],
                    outcome,
                ))

        if not stacks:
            return 0

        logger.debug("Committing round %d: %d stacks", self._round_id, len(stacks))
        return self.memory.record_round(results[0].game_class, stacks)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train(
    runner: SimulationRunner,
    swarms: Dict[int, List[Agent]],
    game: GameBase,
    simulations: int,
    turn_depth: int,
) -> None:
    """Run training: half prune (worst vs worst), half exploit (best vs best)."""
    half = simulations // 2
    runner.run_batch(swarms, game, half, max_turns=turn_depth, is_prune=True)
    runner.run_batch(swarms, game, simulations - half, max_turns=turn_depth, is_prune=False)


# ---------------------------------------------------------------------------
# Turn Handlers
# ---------------------------------------------------------------------------


def _ai_turn(game: GameBase, memory: GameMemory, debug: bool = False) -> Optional[np.ndarray]:
    """AI selects and applies move. Returns move or None if no valid moves."""
    if not game.valid_moves().size:
        return None
    
    move = select_move(game, memory, debug=debug)
    game.apply_move(move)
    return move


def _human_turn(game: GameBase) -> np.ndarray:
    """Prompt human for move, apply it, return move."""
    valid = game.valid_moves()
    if len(valid) > 0:
        example = valid[0]
        print(f"\nYour turn (Player {game.current_player()})")
        print(f"Format: {len(example)} comma-separated values (e.g., {','.join(map(str, example))})")

    while True:
        try:
            raw = input("Move: ").strip()
            move = np.array([int(x.strip()) for x in raw.split(",")])
            game.apply_move(move)
            return move
        except ValueError as e:
            print(f"Invalid input: {e}")
        except Exception as e:
            print(f"Illegal move: {e}")


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
    Main entry point: orchestrate training and play.
    
    Parameters
    ----------
    agent_swarms : Dict[int, List[Agent]]
        Player ID -> list of agents for that player.
    game : GameBase
        The game instance to play.
    turn_depth : int
        Max turns per simulated game.
    simulations : int
        Simulations per training step (half prune, half exploit).
    memory : GameMemory
        Database for storing/retrieving learned moves.
    num_workers : int
        Parallel worker processes.
    training_enabled : bool
        If False, skip training and play from existing memory.
    human_players : List[int], optional
        Player IDs controlled by human input.
    debug_move_statistics : bool
        If True, show debug visualization during AI turns.
    """
    human_set = set(human_players or [])
    runner = SimulationRunner(memory, num_workers)

    print(f"Starting sim with {num_workers} workers. Training: {'ON' if training_enabled else 'OFF'}")
    print(game.state_string())

    try:
        with runner:
            while not game.is_over():
                current = game.current_player()

                if current in human_set:
                    move = _human_turn(game)
                    print(f"\nYou played: {','.join(map(str, move))}")
                else:
                    if training_enabled:
                        _train(runner, agent_swarms, game, simulations, turn_depth)
                    move = _ai_turn(game, memory, debug=debug_move_statistics)
                    if move is None:
                        break
                    print(f"\nAI (Player {current}) played: {','.join(map(str, move))}")

                print(game.state_string())

            print("\n" + "=" * 40)
            print("GAME OVER")
            print("=" * 40)
            
            # Final stats
            info = memory.get_info()
            print(f"Final: {info['transitions']} transitions, {info['anchors']} anchors, "
                  f"{info['total_samples']} samples")

    except KeyboardInterrupt:
        print("\nInterrupted - shutting down...")
        runner.shutdown(force=True)
    except Exception:
        logger.exception("Fatal error in simulation loop")
        raise
    finally:
        memory.close()
