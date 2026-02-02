"""
Profiling utility for simulation performance analysis.

Location: src/wise_explorer/debug/profiler.py

Usage:
    from wise_explorer.debug.profiler import (
        profile_simulations,
        quick_diagnostic,
        profile_single_game,
        profile_evaluate_moves,
        profile_game_ops,
        timed,
        print_timing_summary,
    )

    # Full simulation profiling
    profile_simulations(runner, swarms, game, num_sims=100)

    # Quick diagnostic
    quick_diagnostic(runner, swarms, game, num_sims=50)

    # Single game breakdown  
    profile_single_game(game.deep_clone(), memory)

    # Time specific blocks
    with timed("my_operation"):
        do_something()
    print_timing_summary()
"""

import cProfile
import os
import pstats
import io
import re
import time
import functools
from contextlib import contextmanager
from typing import Optional, Callable, Dict, Set, Any
from dataclasses import dataclass

from wise_explorer.core.hashing import hash_board
from wise_explorer.selection import select_move_for_training


# ---------------------------------------------------------------------------
# Path Helpers
# ---------------------------------------------------------------------------

def _find_project_root() -> Optional[str]:
    """Find the project root by walking up from wise_explorer's location."""
    import wise_explorer
    pkg_dir = os.path.dirname(os.path.abspath(wise_explorer.__file__))
    # pkg_dir is .../src/wise_explorer — go up to project root
    # Walk up until we leave 'src' or hit a pyproject.toml/setup.py
    candidate = pkg_dir
    for _ in range(5):
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
        if os.path.exists(os.path.join(candidate, "pyproject.toml")) or \
           os.path.exists(os.path.join(candidate, "setup.py")):
            return candidate + os.sep
    # Fallback: two levels up from wise_explorer package
    return os.path.dirname(os.path.dirname(pkg_dir)) + os.sep


_project_root: Optional[str] = None


def _get_project_root() -> str:
    """Cached project root lookup."""
    global _project_root
    if _project_root is None:
        _project_root = _find_project_root() or ""
    return _project_root


def _relativize_profile_output(text: str) -> str:
    """
    Replace absolute project paths with relative ones in cProfile output.
    
    Project files:  /Users/x/project/src/wise_explorer/core/foo.py → src/wise_explorer/core/foo.py
    External files:  /Users/x/.../site-packages/numpy/core/foo.py  → numpy/core/foo.py
    Stdlib files:    /usr/lib/python3.12/foo.py                    → foo.py
    """
    root = _get_project_root()

    # Strip project root prefix (keeps src/wise_explorer/... structure)
    if root:
        text = text.replace(root, "")

    # Collapse site-packages paths to just package-relative
    text = re.sub(r'[^\s(]*/site-packages/', '', text)

    # Collapse stdlib paths to just filename
    text = re.sub(r'/[^\s(]*/lib/python[\d.]*/(?!site-packages)', '', text)

    return text


# ---------------------------------------------------------------------------
# Timing Context Manager & Decorator
# ---------------------------------------------------------------------------

@dataclass
class TimingStats:
    """Accumulated timing statistics."""
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.call_count if self.call_count else 0.0

    def record(self, elapsed: float) -> None:
        self.total_time += elapsed
        self.call_count += 1
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)


# Global timing registry
_timing_registry: Dict[str, TimingStats] = {}


@contextmanager
def timed(name: str, print_immediate: bool = False):
    """
    Context manager to time a block of code.
    
    Usage:
        with timed("evaluate_moves"):
            result = memory.evaluate_moves(game, moves)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if name not in _timing_registry:
            _timing_registry[name] = TimingStats()
        _timing_registry[name].record(elapsed)
        
        if print_immediate:
            print(f"[{name}] {elapsed*1000:.2f}ms")


def timed_func(name: Optional[str] = None):
    """
    Decorator to time function calls.
    
    Usage:
        @timed_func("select_move")
        def select_move(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        label = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timed(label):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def print_timing_summary():
    """Print summary of all timed operations."""
    if not _timing_registry:
        print("No timing data collected.")
        return

    print("\n" + "=" * 85)
    print(f"{'TIMING SUMMARY':^85}")
    print("=" * 85)
    print(f"{'Operation':<30} {'Calls':>8} {'Total':>10} {'Avg':>10} {'Min':>8} {'Max':>8} {'%':>6}")
    print("-" * 85)

    sorted_items = sorted(
        _timing_registry.items(),
        key=lambda x: x[1].total_time,
        reverse=True
    )

    total_tracked = sum(s.total_time for _, s in sorted_items)

    for name, stats in sorted_items:
        pct = (stats.total_time / total_tracked * 100) if total_tracked else 0
        min_ms = stats.min_time * 1000 if stats.min_time != float('inf') else 0
        print(
            f"{name:<30} "
            f"{stats.call_count:>8} "
            f"{stats.total_time*1000:>8.1f}ms "
            f"{stats.avg_time*1000:>8.3f}ms "
            f"{min_ms:>7.3f}ms "
            f"{stats.max_time*1000:>7.3f}ms "
            f"{pct:>5.1f}%"
        )

    print("=" * 85)


def clear_timing_stats():
    """Clear all accumulated timing data."""
    _timing_registry.clear()


def get_timing_stats() -> Dict[str, TimingStats]:
    """Get a copy of the timing registry."""
    return _timing_registry.copy()


# ---------------------------------------------------------------------------
# cProfile-based Profiling
# ---------------------------------------------------------------------------

def profile_simulations(
    runner,
    swarms: Dict,
    game,
    num_sims: int = 50,
    max_turns: int = 20,
    top_n: int = 40,
    sort_by: str = "cumulative",
):
    """
    Profile a batch of simulations with cProfile.
    
    Shows which functions consume the most time.
    """
    pr = cProfile.Profile()
    
    print(f"\n{'='*80}")
    print(f"PROFILING {num_sims} SIMULATIONS")
    print(f"{'='*80}\n")
    
    pr.enable()
    start = time.perf_counter()
    
    runner.run_batch(
        swarms=swarms,
        game=game,
        num_sims=num_sims,
        max_turns=max_turns,
        prune_players=set(),
    )
    
    elapsed = time.perf_counter() - start
    pr.disable()
    
    print(f"\nTotal wall time: {elapsed:.3f}s ({num_sims/elapsed:.1f} sims/sec)\n")
    
    _print_profile_view(pr, sort_by, top_n, f"TOP {top_n} BY {sort_by.upper()}")
    
    if sort_by != "tottime":
        _print_profile_view(pr, "tottime", 20, "TOP 20 BY SELF TIME")
    
    return pr


def _print_profile_view(pr: cProfile.Profile, sort_by: str, top_n: int, title: str):
    """Print a single view of profile data."""
    stream = io.StringIO()
    stats = pstats.Stats(pr, stream=stream)
    stats.sort_stats(sort_by)
    
    print(f"\n{'-'*80}")
    print(f"{title:^80}")
    print(f"{'-'*80}")
    
    stats.print_stats(top_n)
    print(_relativize_profile_output(stream.getvalue()))


def profile_function(func: Callable, *args, top_n: int = 30, **kwargs):
    """Profile a single function call with cProfile."""
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    
    stream = io.StringIO()
    stats = pstats.Stats(pr, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(top_n)
    print(_relativize_profile_output(stream.getvalue()))
    
    return result


# ---------------------------------------------------------------------------
# Quick Diagnostics
# ---------------------------------------------------------------------------

def quick_diagnostic(
    runner,
    swarms: Dict,
    game,
    num_sims: int = 20,
    max_turns: int = 20,
):
    """Run a quick diagnostic with high-level timing."""
    print("\n" + "=" * 80)
    print(f"{'QUICK DIAGNOSTIC':^80}")
    print("=" * 80)
    
    clear_timing_stats()
    
    start = time.perf_counter()
    
    with timed("run_batch_total"):
        runner.run_batch(
            swarms=swarms,
            game=game,
            num_sims=num_sims,
            max_turns=max_turns,
            prune_players=set(),
        )
    
    total = time.perf_counter() - start
    
    print(f"\nRan {num_sims} simulations in {total:.3f}s")
    print(f"Throughput: {num_sims/total:.1f} sims/sec")
    print(f"Avg per sim: {total/num_sims*1000:.1f}ms")
    
    print_timing_summary()
    
    print(f"\nProjected times:")
    for target in [100, 500, 1000, 5000]:
        projected = (total / num_sims) * target
        print(f"  {target:>5} sims: {projected:>6.1f}s ({target/projected:.0f} sims/sec)")


def compare_runs(
    runner,
    swarms: Dict,
    game,
    num_sims: int = 50,
    max_turns: int = 20,
    num_runs: int = 3,
):
    """Run multiple passes to measure variance."""
    print("\n" + "=" * 80)
    print(f"{'COMPARISON: ' + str(num_runs) + ' RUNS':^80}")
    print("=" * 80)
    
    times = []
    
    for i in range(num_runs):
        start = time.perf_counter()
        runner.run_batch(
            swarms=swarms,
            game=game,
            num_sims=num_sims,
            max_turns=max_turns,
            prune_players=set(),
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s ({num_sims/elapsed:.1f} sims/sec)")
    
    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    
    print(f"\n  Average: {avg:.3f}s ({num_sims/avg:.1f} sims/sec)")
    print(f"  Min:     {min_t:.3f}s")
    print(f"  Max:     {max_t:.3f}s")
    print(f"  Spread:  {max_t - min_t:.3f}s ({(max_t-min_t)/avg*100:.1f}%)")


# ---------------------------------------------------------------------------
# Component-Level Profiling
# ---------------------------------------------------------------------------

def profile_game_ops(game, num_iters: int = 1000):
    """
    Profile game operations in isolation.
    
    Tests: deep_clone, valid_moves, apply_move, hash_board
    """
    
    print("\n" + "=" * 80)
    print(f"{'GAME OPERATIONS PROFILE':^80}")
    print("=" * 80)
    print(f"Running {num_iters} iterations...\n")
    
    clear_timing_stats()
    
    moves = game.valid_moves()
    
    for _ in range(num_iters):
        with timed("deep_clone"):
            clone = game.deep_clone()
        
        with timed("valid_moves"):
            _ = clone.valid_moves()
        
        with timed("get_state"):
            state = clone.get_state()
        
        with timed("hash_board"):
            _ = hash_board(state.board)
        
        with timed("board.copy"):
            _ = state.board.copy()
        
        if len(moves) > 0:
            with timed("apply_move"):
                test = game.deep_clone()
                test.apply_move(moves[0])
    
    print_timing_summary()


def profile_evaluate_moves(memory, game, num_calls: int = 100):
    """
    Profile evaluate_moves in isolation.
    
    This is often the main bottleneck.
    """
    print("\n" + "=" * 80)
    print(f"{'EVALUATE_MOVES PROFILE':^80}")
    print("=" * 80)
    print(f"Running {num_calls} calls...\n")
    
    clear_timing_stats()
    
    valid_moves = game.valid_moves()
    num_moves = len(valid_moves)
    
    start = time.perf_counter()
    
    for _ in range(num_calls):
        with timed("evaluate_moves"):
            memory.evaluate_moves(game, valid_moves)
    
    elapsed = time.perf_counter() - start
    
    print(f"{num_calls} calls in {elapsed*1000:.1f}ms")
    print(f"Avg per call: {elapsed/num_calls*1000:.3f}ms")
    print(f"Moves evaluated per call: {num_moves}")
    print(f"Avg per move: {elapsed/num_calls/num_moves*1000:.3f}ms")
    
    print_timing_summary()
    
    # cProfile breakdown
    print("\n" + "-" * 80)
    print("cProfile breakdown (top 25):")
    print("-" * 80)
    
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(num_calls):
        memory.evaluate_moves(game, valid_moves)
    pr.disable()
    
    stream = io.StringIO()
    stats = pstats.Stats(pr, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(25)
    print(_relativize_profile_output(stream.getvalue()))


def profile_single_game(game, memory, max_turns: int = 50):
    """
    Profile a single game with per-operation breakdown.
    
    Shows exactly where time goes during gameplay.
    """
    
    print("\n" + "=" * 80)
    print(f"{'SINGLE GAME PROFILE':^80}")
    print("=" * 80)
    
    clear_timing_stats()
    
    test_game = game.deep_clone()
    moves_made = 0
    
    start = time.perf_counter()
    
    for _ in range(max_turns):
        if test_game.is_over():
            break
        
        pid = test_game.current_player()
        
        with timed("1.valid_moves"):
            valid = test_game.valid_moves()
        
        with timed("2.evaluate_moves"):
            _ = memory.evaluate_moves(test_game, valid)
        
        with timed("3.select_move"):
            move = select_move_for_training(test_game, memory, is_prune=False)
        
        with timed("4.apply_move"):
            test_game.apply_move(move)
        
        moves_made += 1
    
    elapsed = time.perf_counter() - start
    
    print(f"\nGame completed: {moves_made} moves in {elapsed*1000:.1f}ms")
    print(f"Avg per move: {elapsed/moves_made*1000:.2f}ms")
    
    print_timing_summary()


def profile_database(memory, num_queries: int = 500):
    """
    Profile database query performance.
    
    Works with both TransitionMemory and MarkovMemory.
    """
    import random
    from wise_explorer.memory.transition_memory import TransitionMemory

    print("\n" + "=" * 80)
    print(f"{'DATABASE PROFILE':^80}")
    print("=" * 80)

    is_transition = isinstance(memory, TransitionMemory)
    mode_label = "transition" if is_transition else "markov"
    print(f"Mode: {mode_label}")

    # Get sample keys
    if is_transition:
        rows = memory.conn.execute(
            "SELECT from_hash, to_hash FROM transitions LIMIT 100"
        ).fetchall()
    else:
        rows = memory.conn.execute(
            "SELECT state_hash FROM states LIMIT 100"
        ).fetchall()

    if not rows:
        print("No data in database to profile.")
        return

    print(f"Running {num_queries} queries...\n")

    clear_timing_stats()

    for _ in range(num_queries):
        # Clear caches to force real queries
        memory._anchor_id_cache.clear()
        memory._anchor_stats_cache.clear()

        if is_transition:
            from_h, to_h = random.choice(rows)

            with timed("get_move_stats"):
                _ = memory.get_move_stats(from_h, to_h)

            with timed("get_anchor_id"):
                _ = memory.get_anchor_id(from_h, to_h)

            with timed("get_anchor_stats"):
                _ = memory.get_anchor_stats(from_h, to_h)

            with timed("get_transitions_from"):
                _ = memory.get_transitions_from(from_h)
        else:
            state_hash = random.choice(rows)[0]

            with timed("get_state_stats"):
                _ = memory.get_state_stats(state_hash)

            with timed("get_anchor_id"):
                _ = memory.get_anchor_id("_", state_hash)

            with timed("get_anchor_stats"):
                _ = memory.get_anchor_stats("_", state_hash)

    print_timing_summary()


# ---------------------------------------------------------------------------
# Full System Profile
# ---------------------------------------------------------------------------

def full_profile(runner, swarms, game, memory, num_sims: int = 50):
    """Run comprehensive profiling of all components."""
    print("\n" + "=" * 80)
    print(f"{'FULL SYSTEM PROFILE':^80}")
    print("=" * 80)
    
    # 1. Game operations
    print("\n[1/5] Game Operations...")
    profile_game_ops(game, num_iters=500)
    
    # 2. Database
    print("\n[2/5] Database Queries...")
    profile_database(memory, num_queries=500)
    
    # 3. evaluate_moves
    print("\n[3/5] evaluate_moves...")
    profile_evaluate_moves(memory, game, num_calls=50)
    
    # 4. Single game
    print("\n[4/5] Single Game...")
    profile_single_game(game.deep_clone(), memory, max_turns=50)
    
    # 5. Full simulation batch
    print("\n[5/5] Full Simulations...")
    profile_simulations(runner, swarms, game, num_sims=num_sims, top_n=30)
    
    print("\n" + "=" * 80)
    print("FULL PROFILE COMPLETE")
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(__doc__)