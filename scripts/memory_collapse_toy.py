"""Toy lab for the certify -> collapse -> self-heal memory cycle, on tiny Nim.

The question under test: can the rules (structural model) replace stored
transitions — and when a "definitive" rule turns out wrong AFTER its supporting
transitions were deleted, does the system notice and repair itself?

Four acts:
  1 BASELINE   train, inspect rules / exact play / row count
  2 CERTIFY    rollout-certify confident boards (game-as-kernel, non-circular);
               negative control: an inverted library should certify ~nothing
  3 COLLAPSE   delete transitions the certified rules explain (keep exceptions)
  4 THE FEAR   corrupt the library post-collapse, keep training, watch the
               self-heal: surprisal rows return, refit repairs, play recovers

Usage:
    python scripts/memory_collapse_toy.py                 # all four acts
    python scripts/memory_collapse_toy.py --piles 3 --sims 600 --chunks 4
"""

from __future__ import annotations

import argparse
import itertools
import random
import tempfile
from pathlib import Path

import numpy as np

import wise_explorer.memory as Memory
from wise_explorer.agent.agent import State
from wise_explorer.games.game_base import GameState
from wise_explorer.games.nim import Nim
from wise_explorer.selection import select_move
from wise_explorer.utils.factory import create_agent_swarms

BASE = Path(tempfile.gettempdir()) / "we_collapse_toy"


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def fresh_mem(piles: int):
    if BASE.exists():
        for f in BASE.glob("*.db*"):
            f.unlink()
    return Memory.for_game(Nim(n=piles), base_dir=str(BASE))


def train(mem, piles: int, sims: int) -> None:
    from wise_explorer.simulation.runner import SimulationRunner
    from wise_explorer.simulation.training import run_training
    swarms = create_agent_swarms([1, 2], 4)
    with SimulationRunner(mem, num_workers=1) as r:
        run_training(r, swarms, Nim(n=piles), simulations=sims, turn_depth=40)
    mem.concept_library._load()


def all_boards(piles: int):
    """Every board of tiny Nim: pile i holds 0..i+1 objects."""
    for tup in itertools.product(*(range(i + 2) for i in range(piles))):
        yield np.array(tup, dtype=np.int8)


def nim_sum(b) -> int:
    return int(np.bitwise_xor.reduce(b.astype(np.int64)))


def optimal_all(mem, piles: int) -> tuple[int, int]:
    """EXACT optimal-move rate: every winning position, no sampling."""
    opt = tot = 0
    for b in all_boards(piles):
        if b.sum() == 0 or nim_sum(b) == 0:
            continue
        tot += 1
        g = Nim(n=piles)
        g.set_state(GameState(b.copy(), current_player=1))
        mv = select_move(g, mem)
        nb = b.copy()
        nb[int(mv[0])] -= int(mv[1])
        opt += nim_sum(nb) == 0
    return opt, tot


def rows(mem) -> int:
    return mem.conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]


def corrupt(lib) -> None:
    """Invert every rule in place (and persist): the theory is now wrong."""
    for r in lib.rules:
        r.avg = 1.0 - r.avg
        r.verdict = {"WIN": "LOSS", "LOSS": "WIN"}.get(r.verdict, r.verdict)
    lib.save()


def lib_says(lib, piles: int) -> str:
    win = np.zeros(piles, dtype=np.int64); win[0] = win[1] = 1     # xor == 0
    loss = np.zeros(piles, dtype=np.int64); loss[0] = 1            # xor != 0
    vw, vl = lib.value_for(win), lib.value_for(loss)
    if vw is None or vl is None:
        return "no opinion"
    return f"{'HEALTHY' if vw > vl else 'WRONG'} (win-board {vw:.2f} / loss-board {vl:.2f})"


# ---------------------------------------------------------------------------
# Act 2 — certification by rollout (the game is the kernel)
# ---------------------------------------------------------------------------

def rule_move(g, lib, rng):
    """The strategy a library implies: create the board it prices highest."""
    best, bestv = [], -1.0
    h = g.get_state().board
    for mv in g.valid_moves():
        nb = h.copy()
        nb[int(mv[0])] -= int(mv[1])
        v = lib.value_for(nb.astype(np.int64))
        v = -0.5 if v is None else v
        if v > bestv + 1e-9:
            best, bestv = [mv], v
        elif abs(v - bestv) <= 1e-9:
            best.append(mv)
    return best[rng.randrange(len(best))]


def _key(b) -> bytes:
    """Board identity independent of dtype/shape quirks."""
    return np.asarray(b).ravel().astype(np.int64).tobytes()


def playout_path(lib, board, piles: int, rng):
    """One bilateral rule-guided game from `board` (creator = seat 2, opponent
    seat 1 moves first). Returns every board on the path tagged with the seat
    that created it, plus each seat's final result."""
    g = Nim(n=piles)
    g.set_state(GameState(board.copy(), current_player=1))
    path = [(board.copy(), 2)]
    while not g.is_over():
        seat = g.current_player()
        g.apply_move(np.asarray(rule_move(g, lib, rng)))
        path.append((g.get_state().board.copy(), seat))
    return path, {p: g.get_result(p) for p in (1, 2)}


def certify_paths(mem, piles: int, k: int, margin: float, seed: int = 3,
                  quiet: bool = False) -> tuple[set[str], int]:
    """Path-credited certification: every playout tests EVERY confident claim on
    its path (each board's price vs its own creator-seat's final result), not
    just the root's. Roots are always the least-confirmed claim. Returns
    (certified hashes, playouts used)."""
    lib = mem.concept_library
    rng = random.Random(seed)
    certified: set[str] = set()
    claims: dict[bytes, tuple[str, np.ndarray, float]] = {}   # key -> (hash, board, L)
    terminal = 0
    for h, b in mem._load_boards().items():
        b = np.asarray(b).ravel()
        v = lib.value_for(b.astype(np.int64))
        if v is None:
            continue
        tv = terminal_value(piles, b)
        if tv is not None:                              # grade 1: direct verdict
            if abs(v - tv) <= 0.1:
                certified.add(h)
                terminal += 1
            continue
        if abs(v - 0.5) >= margin:
            claims[_key(b)] = (h, b, v)
    confirms = {key: 0 for key in claims}
    refuted: set[bytes] = set()
    playouts = 0
    while True:
        pending = [key for key in claims if key not in refuted and confirms[key] < k]
        if not pending:
            break
        root_key = min(pending, key=lambda key: confirms[key])
        path, results = playout_path(lib, claims[root_key][1], piles, rng)
        playouts += 1
        for pb, seat in path:
            pk = _key(pb)
            if pk not in claims or pk in refuted:
                continue
            won = results[seat] == State.WIN
            if (claims[pk][2] >= 0.5) == won:
                confirms[pk] += 1
            else:
                refuted.add(pk)
    certified |= {claims[key][0] for key, c in confirms.items()
                  if c >= k and key not in refuted}
    if not quiet:
        print(f"    path-credited: {len(certified) - terminal} interior + "
              f"{terminal} terminal certified, {len(refuted)} refuted — "
              f"in {playouts} playouts (per-board method: ~{k * len(claims)})")
    return certified, playouts


def rollout_confirms(lib, board, piles: int, rng) -> bool:
    """Bilateral rule-guided play from `board` (its creator's opponent moves
    first). Returns True iff the outcome matches the library's prediction."""
    pred_creator_wins = lib.value_for(board.astype(np.int64)) >= 0.5
    g = Nim(n=piles)
    g.set_state(GameState(board.copy(), current_player=1))   # seat 1 = opponent
    while not g.is_over():
        g.apply_move(np.asarray(rule_move(g, lib, rng)))
    opponent_lost = g.get_result(1) == State.LOSS
    return opponent_lost == pred_creator_wins


def terminal_value(piles: int, board) -> float | None:
    """The game's own verdict for the player who landed on `board`, or None if
    the board isn't terminal. No statistics — the game is the oracle."""
    g = Nim(n=piles)
    g.set_state(GameState(board.copy(), current_player=1))
    if not g.is_over():
        return None
    vals = {State.WIN: 1.0, State.TIE: 0.5, State.LOSS: 0.0}
    # the mover who LANDED here gets the best seat's outcome (they just moved)
    return max(vals[g.get_result(p)] for p in (1, 2))


def certify(mem, piles: int, k: int, margin: float, seed: int = 3,
            quiet: bool = False) -> set[str]:
    """Certify stored boards, by certificate grade:

    - terminal boards: direct verdict check — the library's price must match
      the game's own outcome (exact, free);
    - interior boards: the library must price confidently AND the prediction
      must survive k adversarial rollouts (the game as kernel).
    """
    lib = mem.concept_library
    rng = random.Random(seed)
    certified: set[str] = set()
    confident = terminal = 0
    for h, b in mem._load_boards().items():
        b = np.asarray(b).ravel()
        v = lib.value_for(b.astype(np.int64))
        if v is None:
            continue                                    # library has no opinion
        tv = terminal_value(piles, b)
        if tv is not None:                              # grade 1: direct verdict
            if abs(v - tv) <= 0.1:
                certified.add(h)
                terminal += 1
            continue
        if abs(v - 0.5) < margin:
            continue                                    # library not confident
        confident += 1
        if all(rollout_confirms(lib, b, piles, rng) for _ in range(k)):
            certified.add(h)
    if not quiet:
        print(f"    confident boards: {confident} · rollout-certified: "
              f"{len(certified) - terminal} · terminal-verdict-certified: {terminal}")
    return certified


# ---------------------------------------------------------------------------
# Act 3 — collapse: forget what the theory explains, keep the exceptions
# ---------------------------------------------------------------------------

def collapse(mem, certified: set[str], eps: float) -> tuple[int, int]:
    """Delete transitions landing on certified boards whose COMPLETED value the
    rule reproduces. The completed value (propagated_score) is what the rule
    claims to predict — raw tallies are mixed-quality play averages and never
    match a sharp prediction. Rows the rule can't reproduce are exceptions:
    kept, they are the bits the theory cannot explain."""
    lib = mem.concept_library
    boards = mem._load_boards()
    deleted = kept = 0
    cur = mem.conn.cursor()
    for to_hash in certified:
        b = np.asarray(boards[to_hash]).ravel().astype(np.int64)
        L = lib.value_for(b)
        for f, ps in mem.conn.execute(
                "SELECT from_hash, propagated_score FROM transitions "
                "WHERE to_hash=?", (to_hash,)).fetchall():
            if ps is not None and abs(ps - L) <= eps:   # theory reproduces it
                cur.execute("DELETE FROM transitions WHERE from_hash=? AND to_hash=?",
                            (f, to_hash))
                deleted += 1
            else:                                       # surprisal (or frontier): keep
                kept += 1
    mem.conn.commit()
    return deleted, kept


def surprisal_rows(mem, eps: float) -> int:
    """How many stored rows does the CURRENT library fail to reproduce?"""
    lib = mem.concept_library
    boards = mem._load_boards()
    count = 0
    for to, ps in mem.conn.execute(
            "SELECT to_hash, propagated_score FROM transitions "
            "WHERE propagated_score IS NOT NULL").fetchall():
        if to not in boards:
            continue
        b = np.asarray(boards[to]).ravel().astype(np.int64)
        L = lib.value_for(b)
        if L is not None and abs(ps - L) > eps:
            count += 1
    return count


# ---------------------------------------------------------------------------
# The four acts
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piles", type=int, default=3)
    ap.add_argument("--sims", type=int, default=600, help="Act-1 training games")
    ap.add_argument("--k", type=int, default=3, help="rollouts per certificate")
    ap.add_argument("--margin", type=float, default=0.3, help="library confidence bar")
    ap.add_argument("--eps", type=float, default=0.25, help="evidence-agrees-with-rule band")
    ap.add_argument("--chunks", type=int, default=4, help="Act-4 recovery chunks")
    ap.add_argument("--chunk-games", type=int, default=150)
    a = ap.parse_args()
    n = a.piles
    random.seed(7); np.random.seed(7)

    print(f"══ ACT 1 · BASELINE — {n}-pile Nim, {a.sims} games ══")
    mem = fresh_mem(n)
    train(mem, n, a.sims)
    o, t = optimal_all(mem, n)
    print(f"    play {o}/{t} optimal (exact, all winning positions) · "
          f"{rows(mem)} transition rows · library {lib_says(mem.concept_library, n)}")
    print("    " + mem.concept_library.summary().replace("\n", "\n    "))

    print(f"\n══ ACT 2 · CERTIFY by rollout (k={a.k}, margin={a.margin}) ══")
    certified = certify(mem, n, a.k, a.margin)
    cert_p, _ = certify_paths(mem, n, a.k, a.margin)
    if cert_p != certified:
        print(f"    NOTE: path-credited set differs from per-board "
              f"({len(cert_p)} vs {len(certified)})")
    certified = cert_p
    print("    negative control — invert the library, certify again:")
    corrupt(mem.concept_library)
    bogus, _ = certify_paths(mem, n, a.k, a.margin)
    corrupt(mem.concept_library)                        # invert back = restore
    print(f"    inverted library certifies {len(bogus)} boards "
          f"(self-agreement can't fool a rollout)")

    print(f"\n══ ACT 3 · COLLAPSE — forget what the theory explains (eps={a.eps}) ══")
    before = rows(mem)
    deleted, exceptions = collapse(mem, certified, a.eps)
    o3, _ = optimal_all(mem, n)
    print(f"    rows {before} → {rows(mem)}  (deleted {deleted}, "
          f"kept {exceptions} exceptions)")
    print(f"    play after collapse: {o3}/{t} — the rules carry the deleted region")

    print(f"\n══ ACT 4 · THE FEAR — corrupt the theory AFTER the data is gone ══")
    corrupt(mem.concept_library)
    o4, _ = optimal_all(mem, n)
    print(f"    library {lib_says(mem.concept_library, n)} · play {o4}/{t}")
    for c in range(1, a.chunks + 1):
        train(mem, n, a.chunk_games)
        o, _ = optimal_all(mem, n)
        pre = rows(mem)
        # the loop's steady state: re-certify and re-collapse every cycle —
        # a wrong library fails its rollouts and licenses no deletion
        cert, n_play = certify_paths(mem, n, a.k, a.margin, quiet=True)
        collapse(mem, cert, a.eps)
        print(f"    +{c * a.chunk_games} games: play {o}/{t} · "
              f"rows {pre} → {rows(mem)} after re-collapse "
              f"(certified {len(cert)} in {n_play} playouts) · "
              f"surprisal {surprisal_rows(mem, a.eps)} · "
              f"library {lib_says(mem.concept_library, n)}")
    print("\n    verdict: recovered" if o == o3 else
          "\n    verdict: NOT yet recovered — inspect above")
    mem.close()


if __name__ == "__main__":
    main()
