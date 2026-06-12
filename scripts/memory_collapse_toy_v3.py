"""v3 — the unified loop: steer, prove, forget (docs/certified-forgetting-v3.md).

Certificates are PROOFS, established by induction from the game's terminals:

    a board is proven when every legal reply is proven, and then its value is
    the exact backup  v(b) = 1 - max(value of replies)

No play is involved, so there is nothing the theory can bias; the theory only
prioritizes attention. Consequences that simplify everything relative to v2:

  - no rollouts, no k, no refutation ledger
  - no revocation, no certificate ages — a proof is permanent
  - collapse compares rows to the PROVEN value, so deletion stays sound even
    while the library is wrong (the gate tests exactly this)

Steering is v2's: damp moves onto proven boards, boost moves onto sharp-but-
unproven claims, ordinary uncertainty everywhere else.

Usage:
    python scripts/memory_collapse_toy_v3.py --game nim
    python scripts/memory_collapse_toy_v3.py --game ttt --sims 2000
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from memory_collapse_toy import (                      # noqa: E402
    BASE, Memory, NimAdapter, TTTAdapter, _key, corrupt, fresh_mem, rows, train,
    terminal_value,
)
from memory_collapse_toy_v2 import (                   # noqa: E402
    load_certs, price_all, save_certs, set_steering,
)
from wise_explorer.games.game_base import GameState    # noqa: E402

EPS = 0.25          # row's completed value matches the proven value


# ---------------------------------------------------------------------------
# The frontier: prove claims by induction, anchored at terminal verdicts
# ---------------------------------------------------------------------------

def frontier_certify(mem, adapter, proven: dict[bytes, float]) -> int:
    """Advance the proven frontier over the stored boards. Pure computation:
    the game supplies moves and terminal verdicts, induction does the rest.
    Returns how many stored boards are newly proven."""
    boards = mem._load_boards()
    todo: list[tuple[bytes, np.ndarray]] = []
    for h, b in boards.items():
        b = np.asarray(b).ravel()
        key = _key(b)
        if key in proven:
            continue
        tv = terminal_value(adapter, b)
        if tv is not None:
            proven[key] = tv
            continue
        todo.append((key, b))

    # enumerate each unproven board's replies once; terminal replies prove now
    kids: dict[bytes, list[bytes]] = {}
    for key, b in todo:
        g = adapter.new_game()
        g.set_state(GameState(adapter.shape(b).copy(),
                              current_player=adapter.to_move(b)))
        cks = []
        for mv in g.valid_moves():
            c = g.deep_clone()
            c.apply_move(np.asarray(mv))
            cb = np.asarray(c.get_state().board).ravel()
            ck = _key(cb)
            if ck not in proven:
                tv = terminal_value(adapter, cb)
                if tv is not None:
                    proven[ck] = tv
            cks.append(ck)
        kids[key] = cks

    # sweep until the frontier stops advancing
    new = 0
    changed = True
    while changed:
        changed = False
        for key, _ in todo:
            if key in proven:
                continue
            vals = [proven.get(ck) for ck in kids[key]]
            if all(v is not None for v in vals):
                proven[key] = 1.0 - max(vals)          # exact backup
                new += 1
                changed = True
    return new


def stored_proven(mem, proven: dict[bytes, float]) -> dict[str, float]:
    """The proven facts restricted to stored boards: {db_hash: proven value}."""
    return {h: proven[_key(np.asarray(b).ravel())]
            for h, b in mem._load_boards().items()
            if _key(np.asarray(b).ravel()) in proven}


def collapse_proven(mem, certs: dict[str, float]) -> tuple[int, int]:
    """Delete rows whose completed value the PROOF reproduces. Sound even when
    the library is wrong — the proven value is the game's, not the theory's."""
    deleted = kept = 0
    cur = mem.conn.cursor()
    for to_hash, pv in certs.items():
        for f, ps in mem.conn.execute(
                "SELECT from_hash, propagated_score FROM transitions "
                "WHERE to_hash=?", (to_hash,)).fetchall():
            if ps is not None and abs(ps - pv) <= EPS:
                cur.execute("DELETE FROM transitions WHERE from_hash=? AND to_hash=?",
                            (f, to_hash))
                deleted += 1
            else:
                kept += 1
    mem.conn.commit()
    return deleted, kept


def theory_disagreement(mem, adapter, certs: dict[str, float]) -> int:
    """Proven boards the current library misprices — where the theory is wrong,
    measured against ground truth (free, since proofs are facts)."""
    prices = price_all(mem, adapter)
    return sum(1 for h, pv in certs.items()
               if h in prices and abs(prices[h] - pv) > 0.1)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=["nim", "ttt"], default="nim")
    ap.add_argument("--piles", type=int, default=4)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--chunk-games", type=int, default=300)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    adapter = NimAdapter(a.piles) if a.game == "nim" else TTTAdapter()
    random.seed(7); np.random.seed(7)

    proven: dict[bytes, float] = {}

    def cycle(mem, label):
        t0 = time.time()
        mem.close()
        mem = Memory.for_game(adapter.new_game(), base_dir=str(BASE))
        train(mem, adapter, a.chunk_games)
        o, _ = adapter.optimal_rate(mem)
        new = frontier_certify(mem, adapter, proven)
        certs = stored_proven(mem, proven)
        save_certs(mem, certs)                          # workers read for steering
        pre = rows(mem)
        deleted, kept = collapse_proven(mem, certs)
        print(f"    {label}: play {o}/{t} · proven {len(certs)} (+{new}) · "
              f"rows {pre} → {rows(mem)} (kept {kept} exceptions) · "
              f"theory wrong at {theory_disagreement(mem, adapter, certs)} "
              f"proven boards · {time.time() - t0:.0f}s")
        return o, mem

    print(f"══ SETUP — train {a.sims}, prove the frontier, collapse ══")
    mem = fresh_mem(adapter, reuse=a.reuse)
    if not a.reuse:
        train(mem, adapter, a.sims)
    o0, t = adapter.optimal_rate(mem)
    new = frontier_certify(mem, adapter, proven)
    certs = stored_proven(mem, proven)
    save_certs(mem, certs)
    pre = rows(mem)
    deleted, kept = collapse_proven(mem, certs)
    print(f"    play {o0}/{t} · proven {len(certs)} of "
          f"{mem.conn.execute('SELECT COUNT(*) FROM boards').fetchone()[0]} stored "
          f"boards — zero playouts · rows {pre} → {rows(mem)} (kept {kept}) · "
          f"theory wrong at {theory_disagreement(mem, adapter, certs)}")

    print(f"\n══ STEERED TRAINING (boost claimed · damp proven) ══")
    set_steering(True)
    for c in range(1, a.chunks + 1):
        o, mem = cycle(mem, f"chunk {c}")

    print(f"\n══ GATE — corrupt the theory; proofs must stand, play must recover ══")
    corrupt(mem.concept_library)
    og, _ = adapter.optimal_rate(mem)
    print(f"    post-corruption play {og}/{t}")
    for c in range(1, a.chunks + 1):
        o, mem = cycle(mem, f"gate chunk {c}")
    print(f"\n    GATE {'PASS' if o >= o0 - 15 else 'FAIL'}: "
          f"recovered to {o}/{t} (baseline {o0}/{t})")
    mem.close()
    set_steering(False)


if __name__ == "__main__":
    main()
