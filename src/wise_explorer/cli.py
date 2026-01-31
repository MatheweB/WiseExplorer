"""
Command-line interface for game AI training and play.
"""

import argparse

from wise_explorer.api import start_simulations
import wise_explorer.memory as Memory
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
        "--self-play",
        action="store_true",
        help="AI plays for all players (no human players)",
    )
    parser.add_argument(
        "--players", "-p",
        type=str,
        default=None,
        help="Comma-separated list of human player numbers (e.g., '1,2'). Overrides --self-play.",
    )
    parser.add_argument(
        "--markov",
        action="store_true",
        help="Uses Markov states in favor of transitions",
    )
    return parser.parse_args()

def parse_human_players(players_str: str | None, num_players: int, game_id: str, self_play: bool) -> list[int]:
    """Parse and validate the human players argument."""
    if self_play and players_str is None:
        return []
    
    if players_str is None:
        return [1]  # Default: player 1 is human
    
    # Parse comma-separated values
    try:
        human_players = [int(p.strip()) for p in players_str.split(",") if p.strip()]
    except ValueError as e:
        raise ValueError(
            f"Invalid --players format: '{players_str}'. "
            "Expected comma-separated integers (e.g., '1,2')."
        ) from e
    
    # Validate player numbers
    invalid = [p for p in human_players if p < 1 or p > num_players]
    if invalid:
        raise ValueError(
            f"Invalid player number(s): {invalid}. {game_id} only supports players 1-{num_players}."
        )
    
    return sorted(set(human_players))

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
    memory = Memory.for_game(game, base_dir=MEMORY_DIR, markov=args.markov)
    
    # Determine human players
    human_players = parse_human_players(args.players, game.num_players(), game.game_id(), args.self_play)

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