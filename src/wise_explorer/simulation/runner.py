"""
Simulation runner for parallel game execution.

Manages a pool of worker processes and coordinates game simulations.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import random
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


class SimulationRunner:
    """Manages parallel simulation workers."""

    def __init__(
        self,
        memory: "GameMemory",
        num_workers: int = DEFAULT_WORKER_COUNT
    ):
        self.memory = memory
        self.num_workers = num_workers
        self._pool: Optional[Pool] = None
        self._round_id = 0
        
        # Adaptive batch sizing based on consolidate results
        self._batch_size = num_workers * 2
        self._games_since_merge = 0

    def __enter__(self):
        self._ensure_pool()
        return self

    def __exit__(self, *_):
        self.shutdown()

    def _ensure_pool(self) -> Pool:
        if self._pool is None:
            self._pool = Pool(
                processes=self.num_workers,
                initializer=worker_init,
                initargs=(str(self.memory.db_path), self.memory.is_markov),
            )
        return self._pool

    def shutdown(self, force: bool = False) -> None:
        if self._pool:
            if force:
                self._pool.terminate()
            else:
                self._pool.close()
            self._pool.join()
            self._pool = None

    def run_batch(
        self,
        swarms: Dict[int, List["Agent"]],
        game: "GameBase",
        num_sims: int,
        max_turns: int,
        prune_players: Set[int],
    ) -> int:
        """Run a batch of simulations with streaming commits."""
        if num_sims <= 0 or not swarms:
            return 0

        swarm_size = min(len(s) for s in swarms.values())
        if swarm_size == 0:
            return 0

        pool = self._ensure_pool()
        total_transitions = 0

        jobs = self._make_jobs(swarms, game, num_sims, max_turns, prune_players)
        pending_results: List[JobResult] = []
        
        for result in pool.imap_unordered(run_game, jobs, chunksize=max(1, len(jobs) // (self.num_workers * 4))):
            pending_results.append(result)
            
            if len(pending_results) >= self._batch_size:
                self._round_id += 1
                total_transitions += self._commit(pending_results, prune_players)
                
                merged = self.memory.consolidate_anchors()
                self._adapt_batch_size(merged, len(pending_results))
                
                pending_results = []
        
        # Commit remaining
        if pending_results:
            self._round_id += 1
            total_transitions += self._commit(pending_results, prune_players)
            self.memory.consolidate_anchors()

        return total_transitions

    def _adapt_batch_size(self, merges: int, games: int) -> None:
        """
        Adjust batch size based on whether consolidate found work.
        
        If merges > 0: anchors are actively changing, check often
        If merges == 0: stable, batch more before checking again
        """
        self._games_since_merge += games
        
        if merges > 0:
            # Active merging - reset counter, keep batches small
            self._games_since_merge = 0
            self._batch_size = max(self.num_workers, self._batch_size // 2)
        elif self._games_since_merge > self._batch_size * 3:
            # No merges for a while - increase batch size
            self._batch_size = min(200, self._batch_size + self.num_workers)

    def _make_jobs(
        self,
        swarms: Dict[int, List["Agent"]],
        game: "GameBase",
        count: int,
        max_turns: int,
        prune_players: Set[int],
    ) -> List[GameJob]:
        """Generate randomized match-ups."""
        players = sorted(swarms.keys())
        
        # Pre-generate all indices at once
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

    def _commit(
        self,
        results: List[JobResult],
        prune_players: Set[int],
        skip_neutral: bool = True
    ) -> int:
        """Write game results to memory."""
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

        return self.memory.record_round(results[0].game_class, stacks)