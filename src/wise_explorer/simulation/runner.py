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
from typing import TYPE_CHECKING

from wise_explorer.agent.agent import State
from wise_explorer.simulation.jobs import GameJob, JobResult
from wise_explorer.simulation.worker import worker_init, run_game

if TYPE_CHECKING:
    from wise_explorer.agent.agent import Agent
    from wise_explorer.games.game_base import GameBase
    from wise_explorer.memory.game_memory import GameMemory
from wise_explorer import synthesis

logger = logging.getLogger(__name__)

DEFAULT_WORKER_COUNT = max(1, mp.cpu_count() - 1)

# Games committed (and proven against) together as one wave — the learning granularity,
# independent of num_workers. Bigger waves reweight coverage toward the high mode and cut
# prove/commit overhead; smaller ones keep stats fresher. 50 is the chosen default.
DEFAULT_WAVE_SIZE = 50

# ---------------------------------------------------------------------------
# Process cleanup
# ---------------------------------------------------------------------------

_active_runners: list[SimulationRunner] = []
_active_memories: list[GameMemory] = []


def _shutdown_all():
    for runner in _active_runners[:]:
        runner.shutdown(force=True)
    for memory in _active_memories[:]:
        try:
            memory.close()
        except Exception:
            pass
    _active_memories.clear()


def register_memory(memory: GameMemory) -> None:
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
    
    Games run in waves of wave_size (default: DEFAULT_WAVE_SIZE — independent of num_workers).
    Each wave completes and writes to DB before the next starts, so subsequent games see updated
    statistics; the pool spreads a wave across num_workers — parallelism only, not learning.
    """

    def __init__(self, memory: GameMemory, num_workers: int = DEFAULT_WORKER_COUNT,
                 wave_size: int | None = None, seed: int | None = None):
        self.memory = memory
        self.num_workers = num_workers
        # A wave is the unit of learning — games committed together before the next ones see
        # updated stats and the proof frontier advances — independent of num_workers (which is
        # parallelism only; the pool spreads a wave across the workers it has).
        self.wave_size = max(1, wave_size if wave_size is not None else DEFAULT_WAVE_SIZE)
        # With a base seed the run is reproducible regardless of worker scheduling: job assignment
        # draws from a dedicated RNG and each game carries its own seed, so a game's play is fixed
        # no matter which worker runs it. seed=None keeps the old nondeterministic behaviour.
        self.base_seed = seed
        self._rng = random.Random(seed) if seed is not None else random
        self._job_counter = 0
        self._pool: Pool | None = None
        # The value loop runs two tiers (see run_batch). The cheap march — prove + forget —
        # runs EVERY wave, so the proof frontier keeps advancing. The expensive synthesis search
        # runs on a geometric schedule (the games since the last search double), O(log games)
        # times spread across the run; its own MDL gate decides what actually pays.
        self._since = 0                       # games since the last search
        self._wait = synthesis.MIN_BOARDS     # games to accumulate before the next one
        _active_runners.append(self)
        register_memory(memory)

    def __enter__(self):
        if self.num_workers > 1:
            self._ensure_pool()
        else:
            # Single worker: initialize worker memory in-process (skip pool)
            worker_init(str(self.memory.db_path), self.memory.is_markov)
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
            self.memory.pool = self._pool       # lent for boundary work (reply_graph)
        return self._pool

    def join_wheel(self) -> None:
        """No-op — value-loop cycles now run synchronously in :meth:`run_batch`.
        Kept for call-site compatibility."""

    def shutdown(self, force: bool = False) -> None:
        if self in _active_runners:
            _active_runners.remove(self)

        if self._pool is None:
            return

        pool, self._pool = self._pool, None
        self.memory.pool = None
        pool.terminate() if force else pool.close()
        pool.join()

    def run_batch(
        self,
        swarms: dict[int, list[Agent]],
        game: GameBase,
        num_sims: int,
        max_turns: int,
        prune_players: set[int],
    ) -> int:
        """
        Run simulations in synchronized waves.
        
        Each wave of num_workers games completes fully, writes to DB, and
        commits before the next wave starts. This ensures
        subsequent games see updated statistics.
        """
        if num_sims <= 0 or not swarms:
            return 0

        if self.num_workers > 1:
            pool = self._ensure_pool()
        all_jobs = self._make_jobs(swarms, game, num_sims, max_turns, prune_players)
        
        total_transitions = 0
        job_idx = 0

        # Skip pool overhead for single worker — run in-process
        single_worker = self.num_workers == 1

        try:
            while job_idx < len(all_jobs):
                # Get next wave of jobs (spread across however many workers exist)
                wave_jobs = all_jobs[job_idx : job_idx + self.wave_size]

                # Run wave: in-process for single worker, pool for multi
                if single_worker:
                    wave_results = [run_game(job) for job in wave_jobs]
                else:
                    wave_results = pool.map(run_game, wave_jobs)
                
                # Write results
                transitions, _swaps = self._commit(wave_results)
                total_transitions += transitions

                # Value loop (docs/value-loop.md), run while the pool is idle. Every wave: the
                # cheap march — prove what now chains to terminals, forget the rows it reproduces.
                # The expensive search runs on a geometric schedule — every time the games since the
                # last search double — so it fires O(log games) times spread across the run, never
                # in a burst. Its own MDL gate decides what actually pays; stragglers are caught at
                # the next doubling and the closing fit. No threshold, no cadence to persist.
                self.memory.prove_and_forget(game)
                self._since += len(wave_jobs)
                if self._since >= self._wait:
                    self._since = 0
                    self._wait *= 2
                    self.memory.grow_concepts(game=game)

                job_idx += len(wave_jobs)

        except KeyboardInterrupt:
            logger.info("Interrupted — partial results already committed")
            raise

        return total_transitions

    def _make_jobs(
        self,
        swarms: dict[int, list[Agent]],
        game: GameBase,
        count: int,
        max_turns: int,
        prune_players: set[int],
    ) -> list[GameJob]:
        players = sorted(swarms.keys())
        indices = {
            pid: [self._rng.randrange(len(swarms[pid])) for _ in range(count)]
            for pid in players
        }

        jobs = []
        for i in range(count):
            seed = None if self.base_seed is None else self.base_seed + self._job_counter
            self._job_counter += 1
            jobs.append(GameJob(
                game=game.deep_clone(),
                player_map={pid: indices[pid][i] for pid in players},
                max_turns=max_turns,
                prune_players=prune_players,
                seed=seed,
            ))
        return jobs

    def _commit(self, results: list[JobResult]) -> tuple[int, int]:
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
                    result.outcomes,
                ))

        if not stacks:
            return 0, 0
        
        return self.memory.record_round(results[0].game_class, stacks)