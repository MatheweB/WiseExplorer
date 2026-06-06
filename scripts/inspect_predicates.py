"""
Thin wrapper around `wise-explorer inspect` (kept for backward compatibility).

The real logic now lives in the package (`wise_explorer.inspection`) so the CLI
and this script share one code path. Prefer the CLI:

    wise-explorer inspect -g nim            # rules already learned (no training)
    wise-explorer inspect -g nim --fresh N  # train N games into a throwaway DB

This script maps its old positional form onto that command:

    python scripts/inspect_predicates.py nim            # inspect saved data/memory/nim.db
    python scripts/inspect_predicates.py nim 10000      # train 10000 games, then inspect
    python scripts/inspect_predicates.py nim 10000 --top 8 --wins-only
"""

import argparse
import sys
from pathlib import Path

# Make the local package importable even without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wise_explorer.cli import run_inspect  # noqa: E402

# Old short names → game registry ids.
_ALIAS = {
    "ttt": "tic_tac_toe", "chess": "minichess", "nim": "nim",
    "tic_tac_toe": "tic_tac_toe", "minichess": "minichess",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect the rules a game has learned.")
    p.add_argument("source", nargs="?", default="nim",
                   help="game: nim | ttt | chess (default: nim)")
    p.add_argument("sims", nargs="?", type=int, default=None,
                   help="if given, train this many self-play games first (throwaway DB)")
    p.add_argument("--top", type=int, default=None)
    p.add_argument("--wins-only", action="store_true")
    p.add_argument("--losses-only", action="store_true")
    p.add_argument("--saved", action="store_true",
                   help="render the agent's saved compact library instead of re-mining")
    a = p.parse_args()

    argv = ["-g", _ALIAS.get(a.source, a.source)]
    if a.sims:
        argv += ["--fresh", str(a.sims)]
    if a.top:
        argv += ["--top", str(a.top)]
    if a.wins_only:
        argv.append("--wins-only")
    if a.losses_only:
        argv.append("--losses-only")
    if a.saved:
        argv.append("--saved")

    run_inspect(argv)


if __name__ == "__main__":
    main()
