"""
Public API for game AI training and play.

Usage:
    from game_ai import start_simulations, GameMemory
    
    game = TicTacToe()
    memory = GameMemory.for_game(game, base_dir="data/memory")
    swarms = {1: [Agent() for _ in range(20)], 2: [Agent() for _ in range(20)]}
    start_simulations(swarms, game, turn_depth=20, simulations=200, memory=memory)
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from wise_explorer.memory import GameMemory
from wise_explorer.selection import select_move, select_move_for_training
from wise_explorer.simulation import SimulationRunner, DEFAULT_WORKER_COUNT, run_training

if TYPE_CHECKING:
    from wise_explorer.agent.agent import Agent
    from wise_explorer.games.game_base import GameBase


def _ai_turn(
    game: "GameBase",
    memory: GameMemory,
    debug: bool = False
) -> Optional[np.ndarray]:
    """AI selects and applies move. Returns move or None if no valid moves."""
    if len(game.valid_moves()) == 0:
        return None

    move = select_move(game, memory, random_in_anchor=False, debug=debug)
    game.apply_move(move)
    return move


def _human_turn(game: "GameBase") -> np.ndarray:
    """Prompt human for move, apply it, return move."""
    valid = game.valid_moves()
    if len(valid) > 0:
        example = valid[0]
        print(f"\nYour turn (Player {game.current_player()})")
        print(
            f"Format: {len(example)} comma-separated values "
            f"(e.g., {','.join(map(str, example))})"
        )

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


def start_simulations(
    agent_swarms: Dict[int, List["Agent"]],
    game: "GameBase",
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

    print(
        f"Starting {game.game_id()} sim with {num_workers} workers. "
        f"Training: {'ON' if training_enabled else 'OFF'}"
    )
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
                        run_training(
                            runner, agent_swarms, game, simulations, turn_depth
                        )
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
            print(
                f"Final: {info['transitions']} transitions, "
                f"{info['anchors']} anchors, "
                f"{info['total_samples']} samples"
            )

    except KeyboardInterrupt:
        print("\nInterrupted - shutting down...")
        runner.shutdown(force=True)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Fatal error in simulation loop")
        raise
    finally:
        memory.close()


__all__ = [
    "start_simulations",
    "GameMemory",
    "select_move",
    "select_move_for_training",
    "SimulationRunner",
    "DEFAULT_WORKER_COUNT",
]