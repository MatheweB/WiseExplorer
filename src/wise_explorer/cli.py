"""
Command-line interface for game AI training and play.
"""

import argparse

from wise_explorer.api import start_simulations
from wise_explorer.memory import GameMemory
from wise_explorer.utils.config import Config, GAMES, MEMORY_DIR
from wise_explorer.utils.factory import create_game, create_agent_swarms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and play games with pattern-based AI"
    )
    parser.add_argument(
        "--game", "-g",
        choices=list(GAMES.keys()),
        default="tic_tac_toe",
        help="Game to play (default: tic_tac_toe)",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Training epochs (default: 100)",
    )
    parser.add_argument(
        "--turn-depth", "-t",
        type=int,
        default=40,
        help="Max turns per simulation (default: 40)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1)",
    )
    parser.add_argument(
        "--no-training",
        action="store_true",
        help="Play without training (use existing memory)",
    )
    parser.add_argument(
        "--ai-only",
        action="store_true",
        help="AI plays both sides (no human player)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Build configuration
    config_kwargs = {
        "game_name": args.game,
        "epochs": args.epochs,
        "turn_depth": args.turn_depth,
    }
    if args.workers:
        config_kwargs["num_workers"] = args.workers
    
    config = Config(**config_kwargs)
    
    # Set up game
    game = create_game(config.game_name)
    players = list(range(1, game.num_players() + 1))
    agent_swarms = create_agent_swarms(players, config.num_agents)
    
    # Set up memory
    memory = GameMemory.for_game(game, base_dir=MEMORY_DIR)
    
    # Determine human players
    human_players = [] if args.ai_only else [1]
    
    # Run
    start_simulations(
        agent_swarms=agent_swarms,
        game=game,
        turn_depth=config.turn_depth,
        simulations=config.simulations,
        memory=memory,
        num_workers=config.num_workers,
        training_enabled=not args.no_training,
        human_players=human_players,
        debug_move_statistics=True,
    )


if __name__ == "__main__":
    main()