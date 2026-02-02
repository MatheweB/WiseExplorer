"""
Parallel simulation runner with robust process cleanup.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import random
import signal
import sys
from multiprocessing.pool import Pool
from typing import Dict, List, Optional, Set, TYPE_CHECKING

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
    """Manages parallel game simulations."""

    def __init__(self, memory: "GameMemory", num_workers: int = DEFAULT_WORKER_COUNT):
        self.memory = memory
        self.num_workers = num_workers
        
        self._pool: Optional[Pool] = None
        self._round_id = 0
        self._batch_size = num_workers * 2
        self._games_since_merge = 0

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
        """Run simulations and stream results to memory."""
        if num_sims <= 0 or not swarms:
            return 0

        pool = self._ensure_pool()
        jobs = self._make_jobs(swarms, game, num_sims, max_turns, prune_players)
        chunksize = max(1, len(jobs) // (self.num_workers * 4))

        total = 0
        pending: List[JobResult] = []

        try:
            for result in pool.imap_unordered(run_game, jobs, chunksize=chunksize):
                pending.append(result)

                if len(pending) >= self._batch_size:
                    total += self._flush(pending, prune_players)
                    pending = []

        except KeyboardInterrupt:
            logger.info("Interrupted — committing partial results")
            if pending:
                total += self._flush(pending, prune_players)
            raise

        if pending:
            total += self._flush(pending, prune_players)

        return total

    def _flush(self, results: List[JobResult], prune_players: Set[int]) -> int:
        """Commit results and adapt batch size."""
        self._round_id += 1
        transitions = self._commit(results, prune_players)
        merges = self.memory.consolidate_anchors()
        self._adapt_batch_size(merges, len(results))
        return transitions

    def _adapt_batch_size(self, merges: int, games: int) -> None:
        self._games_since_merge += games

        if merges > 0:
            self._games_since_merge = 0
            self._batch_size = max(self.num_workers, self._batch_size // 2)
        elif self._games_since_merge > self._batch_size * 3:
            self._batch_size = min(200, self._batch_size + self.num_workers)

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

    def _commit(self, results: List[JobResult], prune_players: Set[int]) -> int:
        if not results:
            return 0

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

        return self.memory.record_round(results[0].game_class, stacks) if stacks else 0