"""Proof/verification of the two claims.

CLAIM 1 (value backup): maxn (per-player value vectors, each mover maximizes its
own component) is the correct n-player / any-turn-order value; the scalar negamax
backup V = 1 - max(child) equals maxn's value-to-just-moved EXACTLY iff the game
is 2-player, zero-sum, and strictly alternating — and diverges otherwise.
  -> tested over thousands of random game trees (varying #players, turn order
     incl. non-monotone, and zero-sum vs not). Assert exactness in the 2p-zs-alt
     class; measure divergence outside it.

CLAIM 2 (mover recording): the shipped fix stores each board's mover at play time,
and reply_graph reads it back correctly. Verified on REAL trained games (TTT,
minichess) against an independent rule-based oracle: stored to_move == the unique
seat whose legal moves actually produce a recorded child.
"""
import random
import tempfile
from pathlib import Path
import numpy as np

import wise_explorer.memory as Memory
from wise_explorer.api import train
from wise_explorer.utils.factory import create_game
from wise_explorer.core.hashing import hash_board
from wise_explorer.games.game_state import GameState


# ----------------------------------------------------------------------------
# CLAIM 1 — random game trees
# ----------------------------------------------------------------------------
def gen(N, depth, alt, zerosum, rng, tm):
    """tm = this node's to_move (used when alt). Returns a node dict."""
    if depth == 0 or rng.random() < 0.35:
        if zerosum:                       # one winner gets 1 (sums to 1)
            v = [0.0] * N
            v[rng.randrange(N)] = 1.0
        else:                             # arbitrary payoffs (not complementary)
            v = [round(rng.random(), 3) for _ in range(N)]
        return {"leaf": True, "payoff": v}
    mover = tm if alt else rng.randint(1, N)
    nkids = rng.randint(1, 3)
    nxt = (mover % N) + 1                  # cyclic next, used only when alt
    kids = [gen(N, depth - 1, alt, zerosum, rng, nxt) for _ in range(nkids)]
    return {"leaf": False, "tm": mover, "kids": kids}


def maxn(node):
    """Per-player optimal-value vector (backward induction)."""
    if node["leaf"]:
        return node["payoff"]
    M = node["tm"]
    best = None
    for k in node["kids"]:
        v = maxn(k)
        if best is None or v[M - 1] > best[M - 1] + 1e-12:    # first-child tie-break
            best = v
    return best


def negamax(node, just_moved):
    """Scalar value to whoever just moved in, via V = 1 - max(child)."""
    if node["leaf"]:
        return node["payoff"][just_moved - 1]
    M = node["tm"]
    return 1.0 - max(negamax(k, M) for k in node["kids"])


def compare(node, just_moved, st):
    """Walk the tree; for every non-root node compare negamax vs maxn's value to
    the just-moved player. Also check the mover identity to_move(parent)=just_moved(child)."""
    if just_moved is not None:
        ng = negamax(node, just_moved)
        mx = maxn(node)[just_moved - 1]
        st["n"] += 1
        if abs(ng - mx) > 1e-9:
            st["div"] += 1
    if not node["leaf"]:
        for k in node["kids"]:
            compare(k, node["tm"], st)     # parent's to_move IS the child's just-moved


def claim1():
    print("=== CLAIM 1: maxn vs negamax over random trees ===")
    rng = random.Random(7)
    cats = [
        ("2p  zero-sum  ALTERNATING ", 2, True,  True),
        ("2p  zero-sum  non-monotone", 2, False, True),
        ("2p  NON-zero-sum           ", 2, False, False),
        ("3p  one-winner             ", 3, False, True),
        ("4p  one-winner  alternating", 4, True,  True),
    ]
    ok = True
    for name, N, alt, zs in cats:
        st = {"n": 0, "div": 0}
        for _ in range(3000):
            root = gen(N, rng.randint(2, 6), alt, zs, rng, 1)
            compare(root, None, st)
        pct = 100.0 * st["div"] / max(st["n"], 1)
        exact = st["div"] == 0
        tag = "negamax EXACT" if exact else f"negamax DIVERGES on {pct:.1f}% of nodes"
        print(f"  {name}: {tag}   ({st['n']} nodes)")
        # the theory: exact iff 2-player, zero-sum, alternating
        should_be_exact = (N == 2 and zs and alt)
        if exact != should_be_exact:
            ok = False
            print(f"     !!! THEORY VIOLATED (expected {'exact' if should_be_exact else 'divergence'})")
    print(f"  --> CLAIM 1 {'HOLDS' if ok else 'FAILED'}: negamax exact iff 2p+zero-sum+alternating; "
          f"maxn is the general value.\n")
    return ok


# ----------------------------------------------------------------------------
# CLAIM 2 — stored mover on real trained games vs rule-based oracle
# ----------------------------------------------------------------------------
def claim2(game_id, size, games, sample):
    g = create_game(game_id, size=size)
    m = Memory.for_game(g, base_dir=Path(tempfile.mkdtemp()))
    train(m, g, games, workers=6)
    shape = g.get_state().board.shape
    boards = m._load_boards()
    out = {}
    for f, t in m.conn.execute("SELECT from_hash, to_hash FROM transitions"):
        out.setdefault(f, t)
    rows = m.conn.execute("SELECT board_hash, to_move FROM boards WHERE to_move>0").fetchall()
    rng = random.Random(0)
    rows = rng.sample(rows, min(sample, len(rows)))
    checked = mism = 0
    for h, tm in rows:
        C = out.get(h)
        if C is None or h not in boards:
            continue
        producers = []                     # which seat's legal moves actually produce C
        for p in range(1, g.num_players() + 1):
            gg = g.deep_clone()
            gg.set_state(GameState(boards[h].reshape(shape).astype(np.int8).copy(), current_player=p))
            if any(_apply(gg, mv) == C for mv in gg.valid_moves()):
                producers.append(p)
        truth = producers[0] if len(producers) == 1 else 0
        if truth:
            checked += 1
            mism += (truth != tm)
    m.close()
    status = "HOLDS" if mism == 0 else f"FAILED ({mism} mismatches)"
    print(f"  {game_id:10s}: stored to_move == true mover on {checked-mism}/{checked} boards  -> {status}")
    return mism == 0


def _apply(g, mv):
    c = g.deep_clone()
    c.apply_move(mv, validated=True)
    return hash_board(c.get_state().board)


def main():
    c1 = claim1()
    print("=== CLAIM 2: stored mover vs rule-based oracle on real games ===")
    c2a = claim2("tic_tac_toe", None, 1500, 400)
    c2b = claim2("minichess", None, 1000, 400)
    print(f"\n==== {'ALL PROOFS HOLD' if (c1 and c2a and c2b) else 'SOMETHING FAILED'} ====")


if __name__ == "__main__":
    main()
