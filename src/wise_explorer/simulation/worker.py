"""
Worker process logic for parallel simulation.

Each worker maintains its own read-only connection to the game memory
database. Workers receive GameJob objects and return JobResult objects.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from wise_explorer.memory import open_readonly

from wise_explorer.simulation.jobs import GameJob, JobResult, MoveRecord
from wise_explorer.memory.game_memory import GameMemory


# Global worker state (initialized per process)
_worker_memory: Optional["GameMemory"] = None


def worker_init(db_path: str, is_markov: bool = False) -> None:
    """Initialize read-only database connection for worker process."""
    global _worker_memory
    _worker_memory = open_readonly(db_path, is_markov)


def run_game(job: GameJob) -> JobResult:
    """Execute a single game simulation."""
    if _worker_memory is None:
        raise RuntimeError("Worker not initialized")

    from wise_explorer.selection import select_move_for_training

    game = job.game
    players = sorted(job.player_map.keys())
    moves: Dict[int, List[MoveRecord]] = {pid: [] for pid in players}

    for _ in range(job.max_turns):
        if game.is_over():
            break

        pid = game.current_player()
        board_before = game.get_state().board.copy()
        is_prune = pid in job.prune_players

        move = select_move_for_training(game, _worker_memory, is_prune)
        
        game.apply_move(move)
        moves[pid].append(MoveRecord(move, board_before, pid))

    return JobResult(
        moves=moves,
        outcomes={pid: game.get_result(pid) for pid in players},
        player_map=job.player_map,
        game_class=type(game),
    )
