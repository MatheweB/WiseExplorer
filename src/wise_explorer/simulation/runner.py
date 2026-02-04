"""
Parallel simulation runner with synchronized wave-based learning.

Each wave of games completes and writes before the next wave starts,
ensuring fresh data propagates between waves.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import random
import signal
import sys
from multiprocessing.pool import Pool
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from wise_explorer.agent.agent import State
from wise_explorer.simulation.jobs import GameJob, JobResult
from wise_explorer.simulation.worker import worker_init, run_game

if TYPE_CHECKING:
    from wise_explorer.agent.agent import Agent
    from wise_explorer.games.game_base import GameBase
    from wise_explorer.memory.game_memory import GameMemory

logger = logging.getLogger(__name__)

DEFAULT_WORKER_COUNT = max(1, mp.cpu_count() - 1)

# ---------------------------------------------------------------------------
# Process cleanup
# ---------------------------------------------------------------------------

_active_runners: List["SimulationRunner"] = []
_active_memories: List["GameMemory"] = []


def _shutdown_all():
    for runner in _active_runners[:]:
        runner.shutdown(force=True)
    for memory in _active_memories[:]:
        try:
            memory.close()
        except Exception:
            pass
    _active_memories.clear()


def register_memory(memory: "GameMemory") -> None:
    """Register a memory instance for cleanup on exit/interrupt."""
    if memory not in _active_memories:
        _active_memories.append(memory)


def _worker_init_wrapper(db_path: str, is_markov: bool):
    """Workers ignore SIGINT — only main process handles Ctrl+C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    worker_init(db_path, is_markov)


def _on_signal(signum, frame):
    _shutdown_all()
    if signum == signal.SIGTSTP:
        sys.exit(1)  # Clean exit — resuming paused IPC often deadlocks
    elif signum == signal.SIGINT:
        raise KeyboardInterrupt


if mp.current_process().name == 'MainProcess':
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    if hasattr(signal, 'SIGTSTP'):
        signal.signal(signal.SIGTSTP, _on_signal)
    atexit.register(_shutdown_all)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class SimulationRunner:
    """
    Manages parallel game simulations with synchronized wave-based learning.
    
    Games run in waves of num_workers. Each wave completes and writes to DB
    before the next wave starts, ensuring subsequent games see updated statistics.
    """

    def __init__(self, memory: "GameMemory", num_workers: int = DEFAULT_WORKER_COUNT):
        self.memory = memory
        self.num_workers = num_workers
        self._pool: Optional[Pool] = None

        _active_runners.append(self)
        register_memory(memory)

    def __enter__(self):
        self._ensure_pool()
        return self

    def __exit__(self, exc_type, *_):
        self.shutdown(force=exc_type is not None)

    def _ensure_pool(self) -> Pool:
        if self._pool is None:
            self._pool = Pool(
                processes=self.num_workers,
                initializer=_worker_init_wrapper,
                initargs=(str(self.memory.db_path), self.memory.is_markov),
            )
        return self._pool

    def shutdown(self, force: bool = False) -> None:
        if self in _active_runners:
            _active_runners.remove(self)

        if self._pool is None:
            return

        pool, self._pool = self._pool, None
        pool.terminate() if force else pool.close()
        pool.join()

    def run_batch(
        self,
        swarms: Dict[int, List["Agent"]],
        game: "GameBase",
        num_sims: int,
        max_turns: int,
        prune_players: Set[int],
    ) -> int:
        """
        Run simulations in synchronized waves.
        
        Each wave of num_workers games completes fully, writes to DB, and
        consolidates anchors before the next wave starts. This ensures
        subsequent games see updated statistics.
        """
        if num_sims <= 0 or not swarms:
            return 0

        pool = self._ensure_pool()
        all_jobs = self._make_jobs(swarms, game, num_sims, max_turns, prune_players)
        
        total_transitions = 0
        job_idx = 0

        try:
            while job_idx < len(all_jobs):
                # Get next wave of jobs (one per worker)
                wave_jobs = all_jobs[job_idx : job_idx + self.num_workers]
                
                # Run wave in parallel, block until all complete
                wave_results = pool.map(run_game, wave_jobs)
                
                # Write results
                transitions, _swaps = self._commit(wave_results)
                total_transitions += transitions
                
                # Consolidate anchors
                self.memory.consolidate_anchors()
                
                job_idx += len(wave_jobs)

        except KeyboardInterrupt:
            logger.info("Interrupted — partial results already committed")
            raise

        return total_transitions

    def _make_jobs(
        self,
        swarms: Dict[int, List["Agent"]],
        game: "GameBase",
        count: int,
        max_turns: int,
        prune_players: Set[int],
    ) -> List[GameJob]:
        players = sorted(swarms.keys())
        indices = {
            pid: [random.randrange(len(swarms[pid])) for _ in range(count)]
            for pid in players
        }

        return [
            GameJob(
                game=game.deep_clone(),
                player_map={pid: indices[pid][i] for pid in players},
                max_turns=max_turns,
                prune_players=prune_players,
            )
            for i in range(count)
        ]

    def _commit(self, results: List[JobResult]) -> Tuple[int, int]:
        """
        Commit game results to memory.
        
        Returns:
            (transitions_written, transitions_swapped)
        """
        if not results:
            return 0, 0

        stacks = []
        for result in results:
            for pid, moves in result.moves.items():
                outcome = result.outcomes[pid]
                if outcome == State.NEUTRAL:
                    continue
                stacks.append((
                    [(m.move, m.board_before, m.player) for m in moves],
                    outcome,
                ))

        if not stacks:
            return 0, 0
        
        return self.memory.record_round(results[0].game_class, stacks)