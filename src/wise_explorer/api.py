"""
Public API: train a model, then play it.

    from wise_explorer import train, play
    from wise_explorer.memory import for_game
    from wise_explorer.games import TicTacToe

    game, memory = TicTacToe(), for_game(TicTacToe())
    train(memory, game, games=2000)        # cumulative self-play
    play(memory, game, human_players=[1])  # play against it
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from wise_explorer.memory import GameMemory
from wise_explorer.selection import select_move
from wise_explorer.simulation import SimulationRunner, DEFAULT_WORKER_COUNT, run_training
from wise_explorer.utils.config import default_turn_depth
from wise_explorer.utils.factory import create_agent_swarms

if TYPE_CHECKING:
    from wise_explorer.games.game_base import GameBase

# Self-play samples per player per simulated game — an internal sampler width,
# not a user knob.
_SWARM = 4


def train(
    memory: GameMemory,
    game: GameBase,
    games: int,
    *,
    workers: int = DEFAULT_WORKER_COUNT,
    turn_depth: int | None = None,
    progress: Callable[[int, int, GameMemory], None] | None = None,
) -> None:
    """Run `games` self-play games from `game`'s state into `memory`, cumulatively.

    Training is split into chunks so the value loop's doubling cadence runs
    throughout and `progress(done, total, memory)` can report live; one closing
    cycle (solve → complete → fit → prove → forget) fits the final library.
    """
    if games <= 0:
        return
    td = turn_depth or default_turn_depth(game.game_id())
    swarms = create_agent_swarms(list(range(1, game.num_players() + 1)), _SWARM)
    step = max(1, games // 20)
    done = 0
    with SimulationRunner(memory, workers) as runner:
        while done < games:
            batch = min(step, games - done)
            run_training(runner, swarms, game, simulations=batch,
                         turn_depth=td, final_cycle=False)
            done += batch
            if progress:
                progress(done, games, memory)
        runner.join_wheel()
        memory.grow_concepts(game=game)
    if progress:
        progress(games, games, memory)


def play(
    memory: GameMemory,
    game: GameBase,
    *,
    human_players: list[int] | None = None,
    ponder: int = 0,
    explain: bool = False,
    verbose: bool = False,
    workers: int = 1,
) -> None:
    """Play `game` to completion. Human seats prompt for input; the rest are
    played by the model. With `ponder > 0`, the model runs that many self-play
    games from the *current position* before each of its moves — local learning
    that sharpens play as the game goes. `explain` shows each AI move's evidence
    ladder; `verbose` dumps every candidate.
    """
    human = set(human_players or [])
    td = default_turn_depth(game.game_id())
    swarms = create_agent_swarms(list(range(1, game.num_players() + 1)), _SWARM)
    runner = SimulationRunner(memory, workers) if ponder else None

    if (explain or verbose) and memory.concept_library.kept:
        print("Rules the AI has discovered:")
        print(memory.concept_library.summary())
        print()
    print(game.state_string())

    try:
        if runner:
            runner.__enter__()
        while not game.is_over():
            current = game.current_player()
            if current in human:
                move = _human_turn(game)
                print(f"\nYou played: {','.join(map(str, move))}")
            else:
                if runner:
                    run_training(runner, swarms, game, simulations=ponder, turn_depth=td)
                move = _ai_move(game, memory, explain=explain, verbose=verbose)
                if move is None:
                    break
                print(f"\nAI (Player {current}) played: {','.join(map(str, move))}")
            print(game.state_string())
        _print_result(game, memory)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        if runner:
            runner.shutdown(force=True)
    finally:
        if runner:
            runner.shutdown()
        memory.close()


# ---------------------------------------------------------------------------
# Turn helpers
# ---------------------------------------------------------------------------

def _ai_move(game, memory, *, explain: bool, verbose: bool):
    if len(game.valid_moves()) == 0:
        return None
    move = select_move(game, memory, debug=verbose)
    if explain:
        from wise_explorer.debug.viz import explain_move
        explain_move(memory, game, move)
    game.apply_move(move)
    return move


def _human_turn(game):
    valid = game.valid_moves()
    if len(valid) > 0:
        example = valid[0]
        print(f"\nYour turn (Player {game.current_player()})")
        print(f"Format: {len(example)} comma-separated values "
              f"(e.g., {','.join(map(str, example))})")
    while True:
        try:
            raw = input("Move: ").strip()
            move = np.array([int(x.strip()) for x in raw.split(",")])
            game.apply_move(move)
            return move
        except (EOFError, KeyboardInterrupt):
            raise KeyboardInterrupt
        except ValueError as e:
            print(f"Invalid input: {e}")
        except Exception as e:
            print(f"Illegal move: {e}")


def _print_result(game, memory):
    print("\n" + "=" * 40 + "\nGAME OVER\n" + "=" * 40)
    info = memory.get_info()
    if memory.is_markov:
        print(f"Memory: {info['unique_states']} states · {info['total_samples']:.0f} samples")
    else:
        proven = len(memory.certified_values)
        print(f"Memory: {info['transitions']} transitions · {info['concepts']} "
              f"concepts · {proven} proven · {info['total_samples']:.0f} samples")


# ---------------------------------------------------------------------------
# Legacy wrapper (prefer train()/play())
# ---------------------------------------------------------------------------

def start_simulations(agent_swarms, game, turn_depth, simulations, memory,
                      num_workers=DEFAULT_WORKER_COUNT, training_enabled=True,
                      human_players=None, debug_move_statistics=False):
    """Deprecated: kept for older callers. Equivalent to ``play`` with
    ``ponder=simulations`` (or ``ponder=0`` when training is disabled)."""
    play(memory, game, human_players=human_players or [],
         ponder=simulations if training_enabled else 0,
         verbose=debug_move_statistics, workers=num_workers)


__all__ = [
    "train",
    "play",
    "start_simulations",
    "GameMemory",
    "select_move",
    "SimulationRunner",
    "DEFAULT_WORKER_COUNT",
]
