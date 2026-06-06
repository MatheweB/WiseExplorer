"""
Command-line interface for game AI training, play, and inspection.

    wise-explorer                       # train + play (default)
    wise-explorer inspect -g nim        # show the rules it has learned (no training)
"""

import argparse
import sys
from pathlib import Path

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
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Reverse n-ply decay rate (default: 1.0 = flat credit). Lower values discount early moves more.",
    )
    parser.add_argument(
        "--max-ply",
        type=int,
        default=None,
        help="Max plies from end to credit (default: None = all moves).",
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

def run_inspect(argv: list[str]) -> None:
    """`wise-explorer inspect` — render the rules a game has already learned.

    Reads the saved predicate library straight from the same DB the training
    loop writes to (no retraining). Use --fresh N for a self-contained demo on
    a throwaway database.
    """
    from wise_explorer.inspection import render_predicates

    p = argparse.ArgumentParser(
        prog="wise-explorer inspect",
        description="Show the human-readable rules a game has learned.",
    )
    p.add_argument("--game", "-g", choices=list(GAMES.keys()), default="tic_tac_toe",
                   help="Game whose learned rules to inspect (default: tic_tac_toe)")
    p.add_argument("--top", type=int, default=None,
                   help="Show only the N most decisive rules (half wins, half losses)")
    p.add_argument("--wins-only", action="store_true", help="Show only winning rules")
    p.add_argument("--losses-only", action="store_true", help="Show only losing rules")
    p.add_argument("--saved", action="store_true",
                   help="Render the agent's saved compact library instead of re-mining the stored transitions")
    p.add_argument("--fresh", type=int, default=None, metavar="N",
                   help="Train N self-play games into a throwaway DB first (quick demo)")
    p.add_argument("--markov", action="store_true", help="Inspect the Markov-mode DB")
    a = p.parse_args(argv)

    game = create_game(a.game)
    render_kw = dict(
        game_id=a.game, top_n=a.top, wins_only=a.wins_only,
        losses_only=a.losses_only, remine=not a.saved,
    )

    if a.fresh:
        mem = _train_throwaway(game, a.fresh, a.markov)
        render_predicates(mem, db_label="fresh demo", **render_kw)
        mem.close()
        return

    db_path = Path(MEMORY_DIR) / f"{game.game_id()}{'_markov' if a.markov else ''}.db"
    if not db_path.exists():
        print(f"No trained model for '{a.game}' at {db_path}.")
        print(f"  Train one:     wise-explorer -g {a.game} -e 2000")
        print(f"  Or quick demo: wise-explorer inspect -g {a.game} --fresh 4000")
        return

    mem = Memory.for_game(game, base_dir=MEMORY_DIR, markov=a.markov, read_only=True)
    render_predicates(mem, db_label=db_path.name, **render_kw)
    mem.close()


def _train_throwaway(game, sims: int, markov: bool):
    """Train `sims` self-play games into a temp DB; return the open memory."""
    import tempfile
    from wise_explorer.simulation.runner import SimulationRunner
    from wise_explorer.simulation.training import run_training

    base = Path(tempfile.gettempdir()) / "we_inspect"
    for suffix in ("", "-wal", "-shm"):
        f = base / f"{game.game_id()}{'_markov' if markov else ''}.db{suffix}"
        if f.exists():
            f.unlink()
    mem = Memory.for_game(game, base_dir=str(base), markov=markov)
    swarms = create_agent_swarms(list(range(1, game.num_players() + 1)), 4)
    td = 25 if game.game_id() == "mini_chess" else 60
    print(f"Training {game.game_id()} × {sims} self-play games…")
    runner = SimulationRunner(mem, num_workers=1)
    with runner:
        run_training(runner, swarms, game, simulations=sims, turn_depth=td)
    return mem


def main() -> None:
    # Hybrid CLI: a bare `wise-explorer` still trains + plays; `inspect` is a verb.
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        run_inspect(sys.argv[2:])
        return

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
    memory = Memory.for_game(
        game, base_dir=MEMORY_DIR, markov=args.markov,
        gamma=args.gamma, max_ply=args.max_ply,
    )
    
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