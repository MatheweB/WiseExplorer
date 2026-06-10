"""
Command-line interface for game AI training, play, and concept invention.

    wise-explorer                       # train + play (default)
    wise-explorer invent -g nim         # show the concepts it invented (no training)
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
        description="Train and play games with pattern-based AI",
        epilog=("other commands:\n"
                "  wise-explorer invent    show the concepts a trained model discovered\n"
                "  wise-explorer transfer  discover on 4-pile Nim, play a bigger Nim zero-shot"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--game", "-g",
        choices=list(GAMES.keys()),
        default="tic_tac_toe",
        help="Game to play (default: tic_tac_toe)",
    )
    parser.add_argument(
        "--size", "-n",
        type=int,
        default=None,
        help="Board size, for games that support it (nim: piles; default 4)",
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

def run_invent(argv: list[str]) -> None:
    """`wise-explorer invent` — show the concepts a trained model discovered.

    By default prints the persisted library — exactly what move selection uses —
    instantly. --remine re-runs discovery over the stored games and prints the
    full bits ledger; --fresh N trains a quick throwaway demo first.
    See docs/concept-invention.md.
    """
    from wise_explorer import synthesis

    p = argparse.ArgumentParser(
        prog="wise-explorer invent",
        description="Show the human-readable concepts a game discovered (e.g. the nim-sum).",
    )
    p.add_argument("--game", "-g", choices=list(GAMES.keys()), default="nim",
                   help="Game whose concepts to show (default: nim)")
    p.add_argument("--size", type=int, default=None,
                   help="Board size, for games that support it (nim: piles; default 4)")
    p.add_argument("--fresh", type=int, default=None, metavar="N",
                   help="Train N self-play games into a throwaway DB first (quick demo)")
    p.add_argument("--remine", action="store_true",
                   help="Re-run discovery over the stored games and print the full bits ledger (slower)")
    p.add_argument("--rounds", type=int, default=4, help="Max reuse rounds when mining (default: 4)")
    a = p.parse_args(argv)

    game = create_game(a.game, size=a.size)
    label = game.game_id()

    if a.fresh:
        mem = _train_throwaway(game, a.fresh, markov=False)
        print(synthesis.render(synthesis.invent(mem, max_rounds=a.rounds), label=label))
        mem.close()
        return

    db_path = Path(MEMORY_DIR) / f"{label}.db"
    if not db_path.exists():
        print(f"No trained model for '{label}' at {db_path}.")
        print(f"  Train one:     wise-explorer -g {a.game} -e 2000")
        print(f"  Or quick demo: wise-explorer invent -g {a.game} --fresh 10000")
        return

    mem = Memory.for_game(game, base_dir=MEMORY_DIR, read_only=True)
    if a.remine:
        print(f"Re-discovering from the games stored in {db_path.name} — this can take a while…")
        print(synthesis.render(synthesis.invent(mem, max_rounds=a.rounds), label=label))
    else:
        print(f"{label} — the library the agent plays with ({db_path.name}):")
        print(mem.concept_library.summary())
        print()
        print("(--remine re-runs discovery and shows the full bits ledger)")
    mem.close()


def run_transfer(argv: list[str]) -> None:
    """`wise-explorer transfer` — discover the nim-sum on 4-pile Nim, then play a
    much bigger Nim perfectly with zero training on it. --full adds the controls
    (from-scratch never finds the rule; seeded retraining holds 400/400 because
    the value loop keeps the rule in charge)."""
    import random
    import sqlite3
    import tempfile

    import numpy as np

    from wise_explorer.games.game_state import GameState
    from wise_explorer.games.nim import Nim
    from wise_explorer.memory.concept_library import ConceptLibrary
    from wise_explorer.selection import select_move

    p = argparse.ArgumentParser(
        prog="wise-explorer transfer",
        description="Discover the nim-sum on 4-pile Nim, then play N-pile Nim zero-shot.",
    )
    p.add_argument("--piles", type=int, default=8,
                   help="Target Nim size for the zero-shot act (default: 8)")
    p.add_argument("--full", action="store_true",
                   help="Also run the honest controls (from-scratch + seeded retraining, slower)")
    a = p.parse_args(argv)
    n = max(a.piles, 2)

    def nim_sum(h):
        return int(np.bitwise_xor.reduce(h.astype(np.int64)))

    def space(k):
        out = 1
        for i in range(1, k + 1):
            out *= (i + 1)
        return out

    def fresh_mem(piles, tag):
        base = Path(tempfile.gettempdir()) / "we_transfer_demo" / tag
        if base.exists():
            for f in base.glob("*.db*"):
                f.unlink()
        return Memory.for_game(Nim(n=piles), base_dir=str(base))

    def train(mem, piles, sims):
        from wise_explorer.simulation.runner import SimulationRunner
        from wise_explorer.simulation.training import run_training
        swarms = create_agent_swarms([1, 2], 4)
        with SimulationRunner(mem, num_workers=1) as r:
            run_training(r, swarms, Nim(n=piles), simulations=sims, turn_depth=60)

    def optimal_sampled(pick, piles, samples=400, seed=7):
        rng = random.Random(seed)
        tried = opt = 0
        while tried < samples:
            h = np.array([rng.randint(0, i + 1) for i in range(piles)], dtype=np.int8)
            if h.sum() == 0 or nim_sum(h) == 0:
                continue
            tried += 1
            mv = pick(h)
            nh = h.copy(); nh[int(mv[0])] -= int(mv[1])
            opt += nim_sum(nh) == 0
        return opt, tried

    def by_memory(mem, piles):
        def pick(h):
            g = Nim(n=piles)
            g.set_state(GameState(h.copy(), current_player=1))
            return select_move(g, mem)
        return pick

    print("ACT 1 — discover the rule on 4-pile Nim (120 positions, 2,000 games)")
    mem4 = fresh_mem(4, "n4")
    train(mem4, 4, 2000)
    o, w = optimal_sampled(by_memory(mem4, 4), 4, samples=96)
    print("  " + mem4.concept_library.summary().replace("\n", "\n  "))
    print(f"  optimal play: {o}/{w} sampled winning positions\n")
    n4_db = str(mem4.db_path)
    mem4.close()

    print(f"ACT 2 — zero-shot: the same library plays {n}-pile Nim "
          f"({space(n):,} positions, no training on it)")
    lib = ConceptLibrary(sqlite3.connect(n4_db), read_only=True)

    def by_rule(h):
        best, bestv = None, -1.0
        g = Nim(n=n)
        g.set_state(GameState(h.copy(), current_player=1))
        for mv in g.valid_moves():
            nh = h.copy(); nh[int(mv[0])] -= int(mv[1])
            v = lib.value_for(nh.astype(np.int64))
            if v is not None and v > bestv:
                best, bestv = mv, v
        return best

    o, w = optimal_sampled(by_rule, n)
    print(f"  optimal play: {o}/{w} — the rule is a program, so it values boards"
          " it has never seen\n")

    if not a.full:
        print("Done. --full adds the honest controls (from-scratch fails; retraining degrades).")
        return

    print(f"ACT 3a — control: train {n}-pile Nim from scratch (3,000 games)")
    memc = fresh_mem(n, "n_scratch")
    train(memc, n, 3000)
    o, w = optimal_sampled(by_memory(memc, n), n)
    found = any("⊕" in str(c) for c in memc.concept_library.kept)
    print(f"  optimal {o}/{w} — {'found' if found else 'never found'} the rule "
          f"(saw {memc.conn.execute('SELECT COUNT(*) FROM boards').fetchone()[0]:,}"
          f" of {space(n):,} positions)\n")
    memc.close()

    print(f"ACT 3b — seeded: start {n}-pile training FROM the 4-pile library")
    mem8 = fresh_mem(n, "n_seeded")
    mem8.concept_library.seed_from(sqlite3.connect(n4_db))
    train(mem8, n, 3000)
    o, w = optimal_sampled(by_memory(mem8, n), n)
    survives = any(str(c) == "fold(⊕, board, cell) = 0" for c in mem8.concept_library.kept)
    print(f"  optimal {o}/{w} — the nim-sum {'survives' if survives else 'was LOST'}. With"
          " ~93% of positions never")
    print("  visited, training would normally bury the rule under coverage-biased Bellman"
          " backups; the value loop")
    print("  instead uses the rule to price the replies training never played, healing"
          " that signal (docs/value-loop.md).")
    mem8.close()


def _train_throwaway(game, sims: int, markov: bool):
    """Train `sims` self-play games into a temp DB; return the open memory."""
    import tempfile
    from wise_explorer.simulation.runner import SimulationRunner
    from wise_explorer.simulation.training import run_training

    base = Path(tempfile.gettempdir()) / "we_invent"
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
    # Hybrid CLI: a bare `wise-explorer` trains + plays; `invent`/`transfer` are verbs.
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        print("`inspect` (the old predicate miner) is gone — its replacement is `invent`,")
        print("which shows the concepts and rules the engine discovered during training:")
        print("  wise-explorer invent -g nim")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "invent":
        run_invent(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "transfer":
        run_transfer(sys.argv[2:])
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
    game = create_game(config.game_name, size=args.size)
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