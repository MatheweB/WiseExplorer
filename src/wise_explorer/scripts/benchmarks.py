#!/usr/bin/env python3
"""
Simulation Performance Profiler
===============================

Location: scripts/benchmark.py

This script profiles the wise_explorer simulation system to identify
performance bottlenecks. It measures time spent in game operations,
database queries, move selection, and full simulation batches.

USAGE
-----
    python scripts/benchmark.py [game] [num_sims] [options]

ARGUMENTS
---------
    game        Game to profile: "tic_tac_toe" or "minichess" (default: tic_tac_toe)
    num_sims    Number of simulations to run (default: 50)

OPTIONS
-------
    --full      Run comprehensive component-by-component breakdown
                (slower but more detailed)
    --markov    Use MarkovMemory instead of TransitionMemory

EXAMPLES
--------
    # Quick profile of TicTacToe with 50 simulations
    python scripts/benchmark.py

    # Profile TicTacToe with 100 simulations  
    python scripts/benchmark.py tic_tac_toe 100

    # Profile MiniChess with 50 simulations
    python scripts/benchmark.py minichess 50

    # Profile using Markov mode
    python scripts/benchmark.py tic_tac_toe 50 --markov

    # Full breakdown of all components (slower)
    python scripts/benchmark.py minichess 100 --full

OUTPUT
------
The profiler runs several tests and prints timing breakdowns:

1. GAME OPERATIONS PROFILE
   - deep_clone: Time to copy game state
   - valid_moves: Time to generate legal moves
   - apply_move: Time to execute a move
   - hash_board: Time to hash board state
   
   → If slow: Optimize game implementation (use int8 boards, __slots__)

2. SINGLE GAME PROFILE  
   - Per-turn breakdown showing where time goes during gameplay
   - Shows: valid_moves → evaluate_moves → select_move → apply_move
   
   → If evaluate_moves dominates: That's the bottleneck (usually is)

3. SIMULATION BATCH PROFILE (cProfile)
   - Full call tree with cumulative and self time
   - TOP N BY CUMULATIVE: Functions that take most total time (including subcalls)
   - TOP N BY SELF TIME: Functions that do the most work themselves
   
   → Look for unexpected hotspots in the call tree

4. STABILITY CHECK
   - Runs multiple passes to measure variance
   - High variance may indicate I/O contention or GC pressure

INTERPRETING RESULTS
--------------------
Common bottlenecks and fixes:

| Symptom                          | Likely Cause              | Fix                           |
|----------------------------------|---------------------------|-------------------------------|
| deep_clone > 0.1ms               | Object dtype arrays       | Use int8 boards               |
| hash_board > 0.05ms              | repr() on object arrays   | Use tobytes() for int arrays  |
| get_move_stats > 0.1ms           | Missing DB index          | Add index, enable caching     |
| evaluate_moves > 5ms             | Too many clones per call  | Batch operations, cache more  |
| High variance between runs       | DB write contention       | Batch commits, use WAL mode   |

PROGRAMMATIC USAGE
------------------
You can also import and use the profiler functions directly:

    from wise_explorer.debug.profiler import (
        profile_simulations,      # cProfile of full batch
        profile_single_game,      # Per-turn breakdown
        profile_evaluate_moves,   # Move evaluation bottleneck
        profile_game_ops,         # Game operation overhead
        profile_database,         # DB query performance
        full_profile,             # All of the above
        quick_diagnostic,         # Fast overview with projections
        compare_runs,             # Measure variance
        timed,                    # Context manager for custom timing
        print_timing_summary,     # Print accumulated timings
    )

    # Example: Profile just evaluate_moves
    profile_evaluate_moves(memory, game, num_calls=100)

    # Example: Custom timing
    with timed("my_operation"):
        do_something()
    print_timing_summary()

SEE ALSO
--------
    wise_explorer/debug/profiler.py  - Core profiling utilities
    wise_explorer/debug/viz.py       - Debug visualization

NOTE
----
    Do NOT name this file "profile.py" - it will shadow Python's
    built-in profile module and break cProfile imports.
"""

import sys
from pathlib import Path

from wise_explorer.utils.config import GAMES, MEMORY_DIR
from wise_explorer.utils.factory import create_game, create_agent_swarms
from wise_explorer.memory import for_game, TransitionMemory, MarkovMemory
from wise_explorer.simulation.runner import SimulationRunner
from wise_explorer.debug.profiler import (
    profile_simulations,
    profile_game_ops,
    profile_single_game,
    full_profile,
    compare_runs,
)
    

# Add src to path if running as script
src_path = Path(__file__).resolve().parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def main():
    # Handle --help
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)
    
    # Parse args
    game_name = "tic_tac_toe"
    num_sims = 50
    do_full = False
    use_markov = False
    
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    
    for flag in flags:
        if flag == "--full":
            do_full = True
        elif flag == "--markov":
            use_markov = True
        elif flag not in ("-h", "--help"):
            print(f"Unknown flag: {flag}")
            print("Use --help for usage information")
            sys.exit(1)
    
    for arg in args:
        if arg in GAMES:
            game_name = arg
        elif arg.isdigit():
            num_sims = int(arg)
        else:
            print(f"Unknown argument: {arg}")
            print(f"Available games: {', '.join(GAMES.keys())}")
            print("Use --help for usage information")
            sys.exit(1)
    
    if game_name not in GAMES:
        print(f"Unknown game: {game_name}")
        print(f"Available: {', '.join(GAMES.keys())}")
        sys.exit(1)
    
    mode_label = "markov" if use_markov else "transition"

    print(f"\n{'='*80}")
    print(f"Profiling: {game_name} ({mode_label} mode)")
    print(f"Simulations: {num_sims}")
    print(f"{'='*80}")
    
    # Setup
    game = create_game(game_name)
    players = list(range(1, game.num_players() + 1))
    swarms = create_agent_swarms(players, agents_per_player=4)
    
    memory = for_game(game, base_dir=MEMORY_DIR, markov=use_markov)
    
    try:
        db_display = memory.db_path.relative_to(Path.cwd())
    except ValueError:
        db_display = memory.db_path
    print(f"\nDB: {db_display}")
    print(f"Mode: {mode_label}")
    info = memory.get_info()
    print(f"Anchors: {info['anchors']:,}")
    print(f"Samples: {info['total_samples']:,}")
    if not use_markov:
        print(f"Transitions: {info['transitions']:,}")
    else:
        print(f"States: {info['unique_states']:,}")
    
    runner = SimulationRunner(memory, num_workers=4)
    
    with runner:
        if do_full:
            # Full component-by-component breakdown
            full_profile(runner, swarms, game, memory, num_sims=num_sims)
        else:
            # Quick overview
            print("\n[1/4] Game Operations...")
            profile_game_ops(game, num_iters=500)
            
            print("\n[2/4] Single Game...")
            profile_single_game(game.deep_clone(), memory, max_turns=30)
            
            print("\n[3/4] Simulation Batch...")
            profile_simulations(
                runner, swarms, game,
                num_sims=num_sims,
                max_turns=30,
                top_n=25
            )
            
            print("\n[4/4] Stability Check...")
            compare_runs(runner, swarms, game, num_sims=20, num_runs=3)
    
    memory.close()
    print("\nDone!")


if __name__ == "__main__":
    main()