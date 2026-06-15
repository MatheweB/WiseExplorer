"""
Command-line interface.

    wise-explorer play   [-g GAME]   play against the AI (default)
    wise-explorer train  [-g GAME]   run self-play training
    wise-explorer eval   [-g GAME]   score the trained model vs a perfect oracle
    wise-explorer invent [-g GAME]   show the concepts it discovered
    wise-explorer transfer           discover on small Nim, play big Nim zero-shot
"""

import argparse
import sys
import time
from pathlib import Path

from wise_explorer.api import train, play
import wise_explorer.memory as Memory
from wise_explorer.utils.config import GAMES, MEMORY_DIR, default_ponder
from wise_explorer.utils.factory import create_game

GAME_CHOICES = list(GAMES.keys())


def parse_human_players(players_str: str | None, num_players: int, game_id: str) -> list[int]:
    """Parse the --players argument; default is player 1 alone."""
    if players_str is None:
        return [1]
    try:
        humans = [int(p.strip()) for p in players_str.split(",") if p.strip()]
    except ValueError as e:
        raise ValueError(f"Invalid --players '{players_str}': expected e.g. '1,2'.") from e
    invalid = [p for p in humans if p < 1 or p > num_players]
    if invalid:
        raise ValueError(f"Invalid player(s) {invalid}: {game_id} has players 1-{num_players}.")
    return sorted(set(humans))


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def run_train(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="wise-explorer train",
                                description="Run self-play training (cumulative).")
    p.add_argument("--game", "-g", choices=GAME_CHOICES, default="tic_tac_toe")
    p.add_argument("--size", "-n", type=int, default=None,
                   help="Board size where supported (Nim: piles; default 4)")
    p.add_argument("--games", type=int, default=2000,
                   help="Self-play games to run (default: 2000)")
    p.add_argument("--workers", "-w", type=int, default=None,
                   help="Worker processes (default: CPU count − 1)")
    p.add_argument("--markov", action="store_true",
                   help="Path-independent (state) memory instead of transitions")
    a = p.parse_args(argv)

    game = create_game(a.game, size=a.size)
    memory = Memory.for_game(game, base_dir=MEMORY_DIR, markov=a.markov)
    print(f"Training {game.game_id()} — {a.games:,} self-play games (cumulative).")

    t0 = time.time()

    def report(done, total, mem):
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        info = mem.get_info()
        rows = info.get("transitions", info.get("unique_states", 0))
        proven = 0 if mem.is_markov else len(mem.certified_values)
        sys.stdout.write(
            f"\r  {done:,}/{total:,} games · {rate:,.0f}/s · "
            f"{info['concepts']} concepts · {proven:,} proven · {rows:,} rows    ")
        sys.stdout.flush()

    kwargs = {"progress": report}
    if a.workers:
        kwargs["workers"] = a.workers
    train(memory, game, a.games, **kwargs)
    print("\n")
    summary = memory.concept_library.summary()
    print(summary if summary else "(no concepts discovered yet — try more games)")
    _print_optimal_rate(memory, game)
    memory.close()


def _print_optimal_rate(memory, game) -> None:
    """Print the optimal-move rate against the game's oracle, if one exists."""
    if memory.is_markov:
        return
    from wise_explorer.benchmark import optimal_rate
    result = optimal_rate(memory, game)
    if result is None:
        return
    opt, total, desc = result
    pct = 100 * opt / total if total else 0
    print(f"\nOptimal play: {opt}/{total} ({pct:.1f}%) — {desc} (vs oracle).")


# ---------------------------------------------------------------------------
# play
# ---------------------------------------------------------------------------

def run_play(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="wise-explorer play",
                                description="Play against the AI (uses trained rules; "
                                            "--learn to learn while playing).")
    p.add_argument("--game", "-g", choices=GAME_CHOICES, default="tic_tac_toe")
    p.add_argument("--size", "-n", type=int, default=None,
                   help="Board size where supported (Nim: piles; default 4)")
    p.add_argument("--players", "-p", default=None,
                   help="Human seats, comma-separated (default: 1)")
    p.add_argument("--watch", action="store_true",
                   help="AI plays every seat — watch it play itself")
    p.add_argument("--learn", action="store_true",
                   help="Learn while playing: self-play from the current position "
                        "before each AI move (off by default)")
    p.add_argument("--ponder", type=int, default=None,
                   help="Self-play games per AI move (implies --learn); 0 = frozen")
    p.add_argument("--explain", action="store_true",
                   help="Show why each AI move was chosen")
    p.add_argument("--verbose", action="store_true",
                   help="Dump every candidate move's values")
    p.add_argument("--workers", "-w", type=int, default=1)
    p.add_argument("--markov", action="store_true")
    a = p.parse_args(argv)

    game = create_game(a.game, size=a.size)
    memory = Memory.for_game(game, base_dir=MEMORY_DIR, markov=a.markov)
    humans = [] if a.watch else parse_human_players(
        a.players, game.num_players(), game.game_id())
    # frozen by default; --ponder N or --learn opts into live learning
    if a.ponder is not None:
        ponder = a.ponder
    elif a.learn:
        ponder = default_ponder(game.game_id())
    else:
        ponder = 0

    info = memory.get_info()
    if info.get("transitions", info.get("unique_states", 0)) == 0:
        print(f"(No training yet for {game.game_id()}. "
              f"Run `wise-explorer train -g {a.game}` for a strong opponent.)\n")

    play(memory, game, human_players=humans, ponder=ponder,
         explain=a.explain, verbose=a.verbose, workers=a.workers)


def run_eval(argv: list[str]) -> None:
    """Score a trained model's play against a perfect oracle (Nim, Tic-Tac-Toe)."""
    from wise_explorer.benchmark import optimal_rate

    p = argparse.ArgumentParser(
        prog="wise-explorer eval",
        description="Optimal-move rate of a trained model vs a perfect solver.")
    p.add_argument("--game", "-g", choices=GAME_CHOICES, default="nim")
    p.add_argument("--size", "-n", type=int, default=None,
                   help="Board size where supported (Nim: piles; default 4)")
    a = p.parse_args(argv)

    game = create_game(a.game, size=a.size)
    db_path = Path(MEMORY_DIR) / f"{game.game_id()}.db"
    if not db_path.exists():
        print(f"No trained model for '{game.game_id()}'. Train one:")
        print(f"  wise-explorer train -g {a.game} --games 2000")
        return
    memory = Memory.for_game(game, base_dir=MEMORY_DIR, read_only=True)
    result = optimal_rate(memory, game)
    if result is None:
        print(f"No oracle available for '{game.game_id()}'.")
    else:
        opt, total, desc = result
        pct = 100 * opt / total if total else 0
        print(f"{game.game_id()}: {opt}/{total} ({pct:.1f}%) optimal — {desc} (vs oracle).")
    memory.close()


# ---------------------------------------------------------------------------
# invent
# ---------------------------------------------------------------------------

def run_invent(argv: list[str]) -> None:
    """Show the concepts a trained model discovered (e.g. the nim-sum)."""
    from wise_explorer import synthesis

    p = argparse.ArgumentParser(
        prog="wise-explorer invent",
        description="Show the human-readable concepts a game discovered.")
    p.add_argument("--game", "-g", choices=GAME_CHOICES, default="nim")
    p.add_argument("--size", "-n", type=int, default=None,
                   help="Board size where supported (Nim: piles; default 4)")
    p.add_argument("--ledger", action="store_true",
                   help="Re-run discovery and print the full bits-saved ledger (slower)")
    p.add_argument("--expand", action="store_true",
                   help="Print every formula fully spelled out")
    a = p.parse_args(argv)

    game = create_game(a.game, size=a.size)
    label = game.game_id()
    db_path = Path(MEMORY_DIR) / f"{label}.db"
    if not db_path.exists():
        print(f"No trained model for '{label}'. Train one:")
        print(f"  wise-explorer train -g {a.game} --games 2000")
        return

    mem = Memory.for_game(game, base_dir=MEMORY_DIR, read_only=True)
    if a.ledger:
        print(f"Re-discovering from the games stored in {db_path.name} — this can take a while…")
        print(synthesis.render(synthesis.invent(mem, max_rounds=32), label=label,
                               expand=a.expand))
    else:
        print(f"{label} — the library the agent plays with ({db_path.name}):")
        print(mem.concept_library.summary(expand=a.expand))
        print("\n(--ledger re-runs discovery and shows the full bits ledger)")
    mem.close()


# ---------------------------------------------------------------------------
# transfer
# ---------------------------------------------------------------------------

def run_transfer(argv: list[str]) -> None:
    """Discover the nim-sum on 4-pile Nim, then play a bigger Nim zero-shot."""
    import random
    import sqlite3
    import tempfile

    import numpy as np

    from wise_explorer.api import train as _train
    from wise_explorer.games.game_state import GameState
    from wise_explorer.games.nim import Nim
    from wise_explorer.memory.concept_library import ConceptLibrary
    from wise_explorer.selection import select_move

    p = argparse.ArgumentParser(
        prog="wise-explorer transfer",
        description="Discover the nim-sum on 4-pile Nim, then play N-pile Nim zero-shot.")
    p.add_argument("--piles", type=int, default=8,
                   help="Target Nim size for the zero-shot act (default: 8)")
    p.add_argument("--full", action="store_true",
                   help="Also run the controls (from-scratch + seeded retraining, slower)")
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

    def optimal_sampled(pick, piles, samples=400, seed=7):
        rng = random.Random(seed)
        tried = opt = 0
        while tried < samples:
            h = np.array([rng.randint(0, i + 1) for i in range(piles)], dtype=np.int8)
            if h.sum() == 0 or nim_sum(h) == 0:
                continue
            tried += 1
            mv = pick(h)
            nh = h.copy()
            nh[int(mv[0])] -= int(mv[1])
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
    _train(mem4, Nim(n=4), 2000, workers=1)
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
            nh = h.copy()
            nh[int(mv[0])] -= int(mv[1])
            v = lib.value_for(nh.astype(np.int64))
            if v is not None and v > bestv:
                best, bestv = mv, v
        return best

    o, w = optimal_sampled(by_rule, n)
    print(f"  optimal play: {o}/{w} — the rule is a program, so it values boards"
          " it has never seen\n")

    if not a.full:
        print("Done. --full adds the controls (from-scratch fails; retraining holds).")
        return

    print(f"ACT 3a — control: train {n}-pile Nim from scratch (3,000 games)")
    memc = fresh_mem(n, "n_scratch")
    _train(memc, Nim(n=n), 3000, workers=1)
    o, w = optimal_sampled(by_memory(memc, n), n)
    found = any("⊕" in str(c) for c in memc.concept_library.kept)
    print(f"  optimal {o}/{w} — {'found' if found else 'never found'} the rule "
          f"(saw {memc.conn.execute('SELECT COUNT(*) FROM boards').fetchone()[0]:,}"
          f" of {space(n):,} positions)\n")
    memc.close()

    print(f"ACT 3b — seeded: start {n}-pile training FROM the 4-pile library")
    mem8 = fresh_mem(n, "n_seeded")
    mem8.concept_library.seed_from(sqlite3.connect(n4_db))
    _train(mem8, Nim(n=n), 3000, workers=1)
    o, w = optimal_sampled(by_memory(mem8, n), n)
    survives = any(str(c) == "fold(⊕, board, cell) = 0" for c in mem8.concept_library.kept)
    print(f"  optimal {o}/{w} — the nim-sum {'survives' if survives else 'was LOST'}. "
          "The value loop prices the replies training never played, so retraining at "
          "scale keeps the rule (docs/value-loop.md).")
    mem8.close()


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

_HELP = """wise-explorer — zero-knowledge self-play that learns readable rules

  wise-explorer play   [-g GAME]   play against the AI (default; uses trained rules)
  wise-explorer train  [-g GAME]   run self-play training
  wise-explorer eval   [-g GAME]   score the trained model vs a perfect oracle
  wise-explorer invent [-g GAME]   show the concepts it discovered
  wise-explorer transfer           discover on 4-pile Nim, play big Nim zero-shot

Games: tic_tac_toe (default), nim, minichess.  Add -h to any command for options.
"""


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in ("-h", "--help", "help"):
        print(_HELP)
        return
    if argv and not argv[0].startswith("-"):
        verb, rest = argv[0], argv[1:]
    else:
        verb, rest = "play", argv

    commands = {"play": run_play, "train": run_train, "eval": run_eval,
                "invent": run_invent, "transfer": run_transfer}
    if verb not in commands:
        print(f"Unknown command '{verb}'.\n")
        print(_HELP)
        return
    commands[verb](rest)


if __name__ == "__main__":
    main()
