"""v2 — steering-only: point exploration at what needs testing. No deletion.

The simplification (vs memory_collapse_toy.py's budget mode): certificates steer
training but never delete rows and never need watchdog games.

  - PROVEN  (certified)              -> damp x0.05: the game already confirmed it
  - CLAIMED (sharp price, untested)  -> boost x2.0: evidence actively wanted here
  - GUESSED (soft / silent)          -> x1.0: ordinary uncertainty-driven play

Because nothing is deleted, a wrong certificate costs only misdirected games —
the evidence channels stay whole and the ordinary refit self-corrects. That lets
revocation drop from game-playing watchdogs to ONE batched re-pricing pass per
cycle: a certificate dies the moment the current theory stops backing its claim.
Refutations are durable too: a failed claim is not retried until its price moves.

Usage:
    python scripts/memory_collapse_toy_v2.py                  # ttt, A/B + gate
    python scripts/memory_collapse_toy_v2.py --game nim
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from memory_collapse_toy import (                      # noqa: E402
    BASE, Memory, NimAdapter, TTTAdapter, certify_paths, corrupt, fresh_mem,
    rows, train,
)
import wise_explorer.selection as selection_mod        # noqa: E402

PRICE_EPS = 0.1     # a certificate/refutation survives while the price moves less


def set_steering(on: bool) -> None:
    os.environ["WISE_CERT_AWARE"] = "2" if on else "0"
    selection_mod.CERT_AWARE = 2 if on else 0


def price_all(mem, adapter) -> dict[str, float]:
    """One batched pricing pass over every stored board (the cheap revocation)."""
    lib = mem.concept_library
    if not lib.rules:
        return {}
    hs, bs, ms = [], [], []
    for h, b in mem._load_boards().items():
        b = np.asarray(b).ravel().astype(np.int64)
        m = adapter.m_of(b)
        if adapter.needs_m and m is None:
            continue
        hs.append(h); bs.append(b); ms.append(0 if m is None else m)
    if not hs:
        return {}
    V = lib.values_for(np.stack(bs), None if not adapter.needs_m
                       else np.array(ms, dtype=np.int64))
    return {h: float(v) for h, v in zip(hs, V) if not np.isnan(v)}


def save_certs(mem, certs: dict[str, float]) -> None:
    mem.conn.execute("DROP TABLE IF EXISTS certificates")
    mem.conn.execute(
        "CREATE TABLE certificates (board_hash TEXT PRIMARY KEY, price REAL)")
    mem.conn.executemany("INSERT INTO certificates VALUES (?,?)", certs.items())
    mem.conn.commit()
    mem._certified_cache = None


def load_certs(mem) -> dict[str, float]:
    try:
        return dict(mem.conn.execute(
            "SELECT board_hash, price FROM certificates"))
    except Exception:
        return {}


def boards_count(mem) -> int:
    return mem.conn.execute("SELECT COUNT(*) FROM boards").fetchone()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=["nim", "ttt"], default="ttt")
    ap.add_argument("--piles", type=int, default=4)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--chunk-games", type=int, default=300)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    adapter = NimAdapter(a.piles) if a.game == "nim" else TTTAdapter()
    random.seed(7); np.random.seed(7)

    print(f"══ SETUP — train {a.sims}, certify once, snapshot (no collapse) ══")
    mem = fresh_mem(adapter, reuse=a.reuse)
    if not a.reuse:
        train(mem, adapter, a.sims)
    certified, _, _ = certify_paths(mem, adapter, a.k, a.margin)
    prices = price_all(mem, adapter)
    save_certs(mem, {h: prices[h] for h in certified if h in prices})
    o0, t = adapter.optimal_rate(mem)
    print(f"    snapshot: play {o0}/{t} · rows {rows(mem)} (kept — steering only) "
          f"· certificates {len(certified)}")
    db = Path(mem.db_path)
    snap = db.with_suffix(".snap")
    if snap.exists():
        snap.unlink()
    mem.conn.execute(f"VACUUM INTO '{snap}'")
    mem.close()

    def restore():
        for suf in ("-wal", "-shm"):
            p = Path(str(db) + suf)
            if p.exists():
                p.unlink()
        shutil.copy(snap, db)
        return Memory.for_game(adapter.new_game(), base_dir=str(BASE))

    failed: dict[str, float] = {}                       # durable refutations

    def chunk_cycle(mem, label):
        t0 = time.time()
        mem.close()
        mem = Memory.for_game(adapter.new_game(), base_dir=str(BASE))
        b_before = boards_count(mem)
        train(mem, adapter, a.chunk_games)
        o, _ = adapter.optimal_rate(mem)
        prices = price_all(mem, adapter)
        certs = load_certs(mem)
        # revoke by price-consistency: the theory no longer backs the claim
        live = {h: p for h, p in certs.items()
                if abs(prices.get(h, 9.0) - p) <= PRICE_EPS}
        revoked = len(certs) - len(live)
        # durable refutations: retry only claims whose price has moved
        still_failed = {h for h, p in failed.items()
                        if abs(prices.get(h, 9.0) - p) <= PRICE_EPS}
        new, n_play, refuted = certify_paths(
            mem, adapter, a.k, a.margin, quiet=True,
            skip=set(live) | still_failed)
        for h in refuted:
            failed[h] = prices.get(h, 0.5)
        for h in new:
            failed.pop(h, None)
            if h in prices:
                live[h] = prices[h]
        save_certs(mem, live)
        print(f"    {label}: play {o}/{t} · certs {len(live)} "
              f"(revoked {revoked}, +{len(new)} new in {n_play} playouts, "
              f"{len(still_failed)} known-failed skipped) · "
              f"boards +{boards_count(mem) - b_before} · rows {rows(mem)} · "
              f"{time.time() - t0:.0f}s")
        return o, mem

    for arm in ("off", "on"):
        print(f"\n══ ARM: steering {arm.upper()} ══")
        set_steering(arm == "on")
        failed.clear()
        random.seed(13); np.random.seed(13)
        mem = restore()
        for c in range(1, a.chunks + 1):
            o, mem = chunk_cycle(mem, f"chunk {c}")
        if arm == "on":
            print(f"\n══ GATE — corrupt the theory; steering active, nothing "
                  f"deleted ══")
            corrupt(mem.concept_library)
            og, _ = adapter.optimal_rate(mem)
            print(f"    post-corruption play {og}/{t}")
            for c in range(1, a.chunks + 1):
                o, mem = chunk_cycle(mem, f"gate chunk {c}")
            verdict = "PASS" if o >= o0 - 15 else "FAIL"
            print(f"\n    GATE {verdict}: recovered to {o}/{t} "
                  f"(pre-corruption baseline {o0}/{t})")
        mem.close()
    set_steering(False)


if __name__ == "__main__":
    main()
