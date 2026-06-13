"""Toy lab for the certify -> collapse -> self-heal memory cycle.

The question under test: can the rules (structural model) replace stored
transitions — and when a "definitive" rule turns out wrong AFTER its supporting
transitions were deleted, does the system notice and repair itself?

Four acts:
  1 BASELINE   train, inspect rules / play quality / row count
  2 CERTIFY    path-credited rollout certification (game-as-kernel);
               negative control: an inverted library should certify ~nothing
  3 COLLAPSE   delete transitions the certified rules explain (keep exceptions)
  4 THE FEAR   corrupt the library post-collapse, keep training, watch the
               self-heal: surprisal rows return, refit repairs, play recovers

Games: nim (complete theory — the table should empty) and ttt (partial theory —
the table should empty only where the theory holds).

Usage:
    python scripts/memory_collapse_toy.py                      # nim, all acts
    python scripts/memory_collapse_toy.py --game ttt --sims 3000
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

import wise_explorer.memory as Memory
from wise_explorer.agent.agent import State
from wise_explorer.games.game_base import GameState
from wise_explorer.games.nim import Nim
from wise_explorer.games.tic_tac_toe import TicTacToe
from wise_explorer.selection import select_move
from wise_explorer.utils.factory import create_agent_swarms

BASE = Path(tempfile.gettempdir()) / "we_collapse_toy"


# ---------------------------------------------------------------------------
# Game adapters
# ---------------------------------------------------------------------------

class NimAdapter:
    """Complete-theory case: the nim-sum explains the whole game."""
    name = "nim"
    default_sims = 2000
    needs_m = False                                     # cell-only concepts

    def __init__(self, piles: int = 4):
        self.piles = piles

    def new_game(self):
        return Nim(n=self.piles)

    def shape(self, board):
        return np.asarray(board).ravel().astype(np.int8)

    def m_of(self, board):
        return None                                     # cell-only concepts

    def to_move(self, board):
        return 1                                        # Nim moves are seat-free

    def probe(self, lib) -> str:
        win = np.zeros(self.piles, dtype=np.int64); win[0] = win[1] = 1
        loss = np.zeros(self.piles, dtype=np.int64); loss[0] = 1
        vw, vl = lib.value_for(win), lib.value_for(loss)
        if vw is None or vl is None:
            return "no opinion"
        return f"{'HEALTHY' if vw > vl else 'WRONG'} ({vw:.2f}/{vl:.2f})"

    def _check(self, mem, b) -> bool:
        g = self.new_game()
        g.set_state(GameState(b.copy(), current_player=1))
        mv = select_move(g, mem)
        nb = b.copy()
        nb[int(mv[0])] -= int(mv[1])
        return _nim_sum(nb) == 0

    def optimal_rate(self, mem) -> tuple[int, int]:
        """Exact over every winning position when the space is small; a seeded
        200-position sample on bigger boards."""
        space = 1
        for i in range(self.piles):
            space *= i + 2
        if space <= 1000:
            opt = tot = 0
            for tup in itertools.product(*(range(i + 2) for i in range(self.piles))):
                b = np.array(tup, dtype=np.int8)
                if b.sum() == 0 or _nim_sum(b) == 0:
                    continue
                tot += 1
                opt += self._check(mem, b)
            return opt, tot
        rng = random.Random(7)
        opt = tried = 0
        while tried < 200:
            b = np.array([rng.randint(0, i + 1) for i in range(self.piles)],
                         dtype=np.int8)
            if b.sum() == 0 or _nim_sum(b) == 0:
                continue
            tried += 1
            opt += self._check(mem, b)
        return opt, tried


def _nim_sum(b) -> int:
    return int(np.bitwise_xor.reduce(b.astype(np.int64)))


_TTT_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8),
              (0, 4, 8), (2, 4, 6)]


def _ttt_winner(flat) -> int:
    for a, b, c in _TTT_LINES:
        if flat[a] != 0 and flat[a] == flat[b] == flat[c]:
            return int(flat[a])
    return 0


@lru_cache(maxsize=None)
def _ttt_minimax(flat: tuple, player: int) -> int:
    """Value for the player to move: 1 win, 0 draw, -1 loss."""
    if _ttt_winner(flat):
        return -1                                       # previous mover won
    if 0 not in flat:
        return 0
    best = -2
    for i in range(9):
        if flat[i] == 0:
            best = max(best, -_ttt_minimax(flat[:i] + (player,) + flat[i + 1:],
                                           3 - player))
            if best == 1:
                break
    return best


def _ttt_optimal_moves(flat: tuple, player: int) -> set[int]:
    vals = {i: -_ttt_minimax(flat[:i] + (player,) + flat[i + 1:], 3 - player)
            for i in range(9) if flat[i] == 0}
    best = max(vals.values())
    return {i for i, v in vals.items() if v == best}


class TTTAdapter:
    """Partial-theory case: lines/threats explain some of the game, not all."""
    name = "ttt"
    default_sims = 3000
    needs_m = True                                      # threat concepts read m

    def __init__(self, eval_n: int = 300, seed: int = 5):
        rng = random.Random(seed)
        pool: set[tuple[tuple, int]] = set()
        while len(pool) < eval_n * 6:
            flat, player = (0,) * 9, 1
            while _ttt_winner(flat) == 0 and 0 in flat:
                cells = [i for i in range(9) if flat[i] == 0]
                if len(_ttt_optimal_moves(flat, player)) < len(cells):
                    pool.add((flat, player))
                i = rng.choice(cells)
                flat = flat[:i] + (player,) + flat[i + 1:]
                player = 3 - player
        self.eval_positions = rng.sample(sorted(pool), eval_n)

    def new_game(self):
        return TicTacToe()

    def shape(self, board):
        return np.asarray(board).reshape(3, 3).astype(np.int8)

    def m_of(self, board):
        """The just-played token, recovered from parity (X always moves first)."""
        f = np.asarray(board).ravel()
        x, o = int((f == 1).sum()), int((f == 2).sum())
        if x + o == 0:
            return None                                 # nobody has moved
        return 1 if x > o else 2

    def to_move(self, board):
        m = self.m_of(board)
        return 1 if m is None else 3 - m

    def probe(self, lib) -> str:
        return f"{len(lib.rules)} rules"

    def optimal_rate(self, mem) -> tuple[int, int]:
        opt = 0
        for flat, player in self.eval_positions:
            g = self.new_game()
            g.set_state(GameState(np.array(flat, dtype=np.int8).reshape(3, 3),
                                  current_player=player))
            mv = select_move(g, mem)
            opt += int(mv[0]) * 3 + int(mv[1]) in _ttt_optimal_moves(flat, player)
        return opt, len(self.eval_positions)


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def fresh_mem(adapter, reuse: bool = False):
    if BASE.exists() and not reuse:
        for f in BASE.glob("*.db*"):
            f.unlink()
    return Memory.for_game(adapter.new_game(), base_dir=str(BASE))


def train(mem, adapter, sims: int) -> None:
    from wise_explorer.simulation.runner import SimulationRunner
    from wise_explorer.simulation.training import run_training
    # one priming write so the -wal/-shm files exist for read-only workers
    # (a closing wheel process may have checkpointed them away)
    v = mem.conn.execute("PRAGMA user_version").fetchone()[0]
    mem.conn.execute(f"PRAGMA user_version={(v + 1) % 1000000}")
    mem.conn.commit()
    swarms = create_agent_swarms([1, 2], 4)
    with SimulationRunner(mem, num_workers=1) as r:
        run_training(r, swarms, adapter.new_game(), simulations=sims, turn_depth=40)
    mem.concept_library._load()
    # single-worker mode leaves a module-global read-only connection open;
    # close it so later snapshot/restore cycles don't trip over its stale locks
    import wise_explorer.simulation.worker as _w
    if _w._worker_memory is not None:
        _w._worker_memory.close()
        _w._worker_memory = None


def rows(mem) -> int:
    return mem.conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]


def corrupt(lib) -> None:
    for r in lib.rules:
        r.avg = 1.0 - r.avg
        r.verdict = {"WIN": "LOSS", "LOSS": "WIN"}.get(r.verdict, r.verdict)
    lib.save()


def _key(b) -> bytes:
    return np.asarray(b).ravel().astype(np.int64).tobytes()


def _price(lib, adapter, board):
    m = adapter.m_of(board)
    if adapter.needs_m and m is None:
        return None             # no just-played token (e.g. the empty board)
    return lib.value_for(np.asarray(board).ravel().astype(np.int64), m)


def terminal_value(adapter, board) -> float | None:
    """The game's own verdict for whoever landed on `board` (None: not terminal)."""
    g = adapter.new_game()
    g.set_state(GameState(adapter.shape(board).copy(),
                          current_player=adapter.to_move(board)))
    if not g.is_over():
        return None
    vals = {State.WIN: 1.0, State.TIE: 0.5, State.LOSS: 0.0}
    return max(vals[g.get_result(p)] for p in (1, 2))


# ---------------------------------------------------------------------------
# Act 2 — path-credited certification (the game is the kernel)
# ---------------------------------------------------------------------------

def rule_move(g, lib, adapter, rng):
    """The strategy the library implies: create the board it prices highest."""
    m = g.current_player()
    best, bestv = [], -1.0
    for mv in g.valid_moves():
        c = g.deep_clone()
        c.apply_move(np.asarray(mv))
        v = lib.value_for(c.get_state().board.ravel().astype(np.int64), m)
        v = -0.5 if v is None else v
        if v > bestv + 1e-9:
            best, bestv = [mv], v
        elif abs(v - bestv) <= 1e-9:
            best.append(mv)
    return best[rng.randrange(len(best))]


def playout_path(adapter, lib, board, rng):
    """One bilateral rule-guided game from `board`. Returns every board on the
    path tagged with the seat that created it, plus per-seat results."""
    g = adapter.new_game()
    start = adapter.to_move(board)
    g.set_state(GameState(adapter.shape(board).copy(), current_player=start))
    path = [(adapter.shape(board).copy(), 3 - start)]   # `board`'s creator
    while not g.is_over():
        seat = g.current_player()
        g.apply_move(np.asarray(rule_move(g, lib, adapter, rng)))
        path.append((g.get_state().board.copy(), seat))
    return path, {p: g.get_result(p) for p in (1, 2)}


def certify_paths(mem, adapter, k: int, margin: float, seed: int = 3,
                  quiet: bool = False, skip: set[str] | None = None) -> tuple[set[str], int]:
    """Path-credited certification: every playout tests EVERY confident claim on
    its path (each board's price vs its own creator-seat's final result).
    Roots are always the least-confirmed claim. ``skip`` holds already-certified
    boards (durable certificates): they are not re-tested — only the watchdog
    can revoke them — so per-cycle cost covers just the uncertified delta."""
    lib = mem.concept_library
    rng = random.Random(seed)
    certified: set[str] = set()
    claims: dict[bytes, tuple[str, np.ndarray, float]] = {}
    terminal = no_opinion = unsure = 0
    for h, b in mem._load_boards().items():
        if skip and h in skip:
            continue
        b = np.asarray(b).ravel()
        v = _price(lib, adapter, b)
        if v is None:
            no_opinion += 1
            continue
        tv = terminal_value(adapter, b)
        if tv is not None:                              # grade 1: direct verdict
            if abs(v - tv) <= 0.1:
                certified.add(h)
                terminal += 1
            continue
        if abs(v - 0.5) < margin:
            unsure += 1
            continue                                    # claim too soft to test
        claims[_key(b)] = (h, b, v)
    confirms = {key: 0 for key in claims}
    refuted: set[bytes] = set()
    playouts = 0
    while True:
        pending = [key for key in claims if key not in refuted and confirms[key] < k]
        if not pending:
            break
        root = min(pending, key=lambda key: confirms[key])
        path, results = playout_path(adapter, lib, claims[root][1], rng)
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
    refuted_hashes = {claims[key][0] for key in refuted}
    if not quiet:
        print(f"    boards: {len(certified) - terminal} interior + {terminal} "
              f"terminal certified · {len(refuted)} refuted · {unsure} too-soft "
              f"· {no_opinion} no-opinion — {playouts} playouts")
    return certified, playouts, refuted_hashes


# ---------------------------------------------------------------------------
# Act 3 — collapse: forget what the theory explains, keep the exceptions
# ---------------------------------------------------------------------------

def collapse(mem, adapter, certified: set[str], eps: float) -> tuple[int, int]:
    """Delete transitions landing on certified boards whose COMPLETED value the
    rule reproduces (raw tallies are play-quality averages — wrong quantity).
    Rows the rule can't reproduce are exceptions: kept."""
    lib = mem.concept_library
    boards = mem._load_boards()
    deleted = kept = 0
    cur = mem.conn.cursor()
    for to_hash in certified:
        L = _price(lib, adapter, np.asarray(boards[to_hash]).ravel())
        for f, ps in mem.conn.execute(
                "SELECT from_hash, propagated_score FROM transitions "
                "WHERE to_hash=?", (to_hash,)).fetchall():
            if ps is not None and abs(ps - L) <= eps:
                cur.execute("DELETE FROM transitions WHERE from_hash=? AND to_hash=?",
                            (f, to_hash))
                deleted += 1
            else:
                kept += 1
    mem.conn.commit()
    return deleted, kept


def surprisal_rows(mem, adapter, eps: float) -> int:
    """Stored rows the CURRENT library fails to reproduce."""
    lib = mem.concept_library
    boards = mem._load_boards()
    count = 0
    for to, ps in mem.conn.execute(
            "SELECT to_hash, propagated_score FROM transitions "
            "WHERE propagated_score IS NOT NULL").fetchall():
        if to not in boards:
            continue
        L = _price(lib, adapter, np.asarray(boards[to]).ravel())
        if L is not None and abs(ps - L) > eps:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Harvest mode — amortized verification from training-style games
# ---------------------------------------------------------------------------

def harvest(mem, adapter, games: int, k: int, margin: float, seed: int = 9,
            quiet: bool = False) -> set[str]:
    """Play training-style games (uncertainty-driven selection, untouched) and
    measure what certification evidence falls out for free.

    Per claim on a game's path:
      - CONFIRM iff the suffix is bilaterally rule-consistent (both seats kept
        choosing argmax-L moves) — such a suffix IS a rollout sample.
      - REFUTE (sound) iff the seat the claim says wins stayed rule-consistent
        and still lost — the claimed strategy failed; opponent quality is moot.

    Returns the refuted boards' hashes (the watchdog's verdicts).
    """
    from wise_explorer.selection import select_move_for_training
    lib = mem.concept_library
    rng = random.Random(seed)
    claims: dict[bytes, float] = {}
    hash_of: dict[bytes, str] = {}
    for h, b in mem._load_boards().items():
        b = np.asarray(b).ravel()
        v = _price(lib, adapter, b)
        if v is None or terminal_value(adapter, b) is not None:
            continue
        if abs(v - 0.5) >= margin:
            claims[_key(b)] = v
            hash_of[_key(b)] = h
    confirms: dict[bytes, int] = {}
    refutes: dict[bytes, int] = {}
    depth_of_confirm: list[int] = []                    # plies from terminal
    for n in range(games):
        g = adapter.new_game()
        prune_seat = rng.choice([0, 1, 2])              # 0 = exploit game
        path = []                                       # (key, seat, move_consistent)
        while not g.is_over():
            seat = g.current_player()
            mv = select_move_for_training(g, mem, is_prune=(seat == prune_seat))
            best, chosen = -1.0, None
            for c_mv in g.valid_moves():
                c = g.deep_clone()
                c.apply_move(np.asarray(c_mv))
                v = lib.value_for(c.get_state().board.ravel().astype(np.int64), seat)
                v = -0.5 if v is None else v
                best = max(best, v)
                if np.array_equal(np.asarray(c_mv), np.asarray(mv)):
                    chosen = v
            g.apply_move(np.asarray(mv))
            consistent = chosen is not None and chosen >= best - 1e-9
            path.append((_key(g.get_state().board), seat, consistent))
        results = {p: g.get_result(p) for p in (1, 2)}
        for i, (key, seat, _) in enumerate(path):
            if key not in claims:
                continue
            L = claims[key]
            winner_seat = seat if L >= 0.5 else 3 - seat    # whom the claim backs
            suffix = path[i + 1:]
            won = results[seat] == State.WIN
            if all(c for _, _, c in suffix):                # bilateral-consistent
                if (L >= 0.5) == won:
                    confirms[key] = confirms.get(key, 0) + 1
                    depth_of_confirm.append(len(suffix))
                else:
                    refutes[key] = refutes.get(key, 0) + 1
            elif (all(c for _, s, c in suffix if s == winner_seat)
                  and results[winner_seat] == State.LOSS):  # sound refutation
                refutes[key] = refutes.get(key, 0) + 1
    certified = {key for key, c in confirms.items() if c >= k and key not in refutes}
    if not quiet:
        print(f"    {games} training-style games over {len(claims)} sharp claims:")
        print(f"      confirmations: {sum(confirms.values())} credits on "
              f"{len(confirms)} distinct claims · k={k}-certified for free: {len(certified)}")
        print(f"      sound refutations: {sum(refutes.values())} on {len(refutes)} claims")
        if depth_of_confirm:
            d = np.array(depth_of_confirm)
            print(f"      confirmed-suffix length (plies to terminal): "
                  f"median {int(np.median(d))} · p90 {int(np.percentile(d, 90))} "
                  f"— the frontier creeps back from the endgame")
        print(f"      dedicated-playout equivalent for the same coverage: "
              f"~{k * len(claims)} playouts; harvested from games already being played")
    return {hash_of[key] for key in refutes}


# ---------------------------------------------------------------------------
# Budget mode — certificate-aware exploration (docs/certificate-aware-exploration.md)
# ---------------------------------------------------------------------------

def set_cert_aware(on: bool) -> None:
    import wise_explorer.selection as selection_mod
    os.environ["WISE_CERT_AWARE"] = "1" if on else "0"
    selection_mod.CERT_AWARE = on


def save_certs(mem, certified: set[str]) -> None:
    mem.conn.execute(
        "CREATE TABLE IF NOT EXISTS certificates (board_hash TEXT PRIMARY KEY)")
    mem.conn.execute("DELETE FROM certificates")
    mem.conn.executemany("INSERT INTO certificates VALUES (?)",
                         [(h,) for h in certified])
    mem.conn.commit()
    mem._certified_cache = None                         # parent re-reads


def load_certs(mem) -> set[str]:
    try:
        return {r[0] for r in mem.conn.execute("SELECT board_hash FROM certificates")}
    except Exception:
        return set()


def run_budget(adapter, a) -> None:
    """A/B the certificate-aware exploration budget from identical snapshots,
    then run the make-or-break gate: corrupt a certified region while
    exploration avoids it — the watchdog must catch and revoke."""
    import shutil

    print(f"══ SETUP — train, certify, collapse, snapshot ══")
    mem = fresh_mem(adapter, reuse=a.reuse)
    if not a.reuse:
        train(mem, adapter, a.sims or adapter.default_sims)
    certified, _, _ = certify_paths(mem, adapter, a.k, a.margin)
    save_certs(mem, certified)
    collapse(mem, adapter, certified, a.eps)
    o, t = adapter.optimal_rate(mem)
    print(f"    snapshot state: play {o}/{t} · rows {rows(mem)} · "
          f"certificates {len(certified)}")
    db = Path(mem.db_path)
    snap = db.with_suffix(".snap")
    if snap.exists():
        snap.unlink()
    # VACUUM INTO writes a consistent single-file copy — a plain file copy of a
    # WAL database tears off the pages still living in the -wal
    mem.conn.execute(f"VACUUM INTO '{snap}'")
    mem.close()

    def restore():
        for suf in ("-wal", "-shm"):
            p = Path(str(db) + suf)
            if p.exists():
                p.unlink()
        shutil.copy(snap, db)
        return Memory.for_game(adapter.new_game(), base_dir=str(BASE))

    def chunk_cycle(mem, label):
        # fresh connection per chunk: closing the only handle checkpoints the
        # WAL cleanly, mimicking the CLI's process-per-run pattern
        mem.close()
        mem = Memory.for_game(adapter.new_game(), base_dir=str(BASE))
        train(mem, adapter, a.chunk_games)
        o, _ = adapter.optimal_rate(mem)
        pre = rows(mem)
        revoked = harvest(mem, adapter, 150, a.k, a.margin, quiet=True)
        certs = load_certs(mem) - revoked
        new, n_play, _ = certify_paths(mem, adapter, a.k, a.margin,
                                    quiet=True, skip=certs)
        certs |= new
        save_certs(mem, certs)
        collapse(mem, adapter, certs, a.eps)
        print(f"    {label}: play {o}/{t} · rows {pre} → {rows(mem)} · "
              f"certs {len(certs)} (revoked {len(revoked)}, +{len(new)} new "
              f"in {n_play} playouts) · library {adapter.probe(mem.concept_library)}")
        return o, mem

    for arm in ("off", "on"):
        print(f"\n══ ARM: certificate-aware exploration {arm.upper()} ══")
        set_cert_aware(arm == "on")
        random.seed(13); np.random.seed(13)
        mem = restore()
        for c in range(1, a.chunks + 1):
            _, mem = chunk_cycle(mem, f"chunk {c}")
        if arm == "on":
            print(f"\n══ GATE — corrupt the theory; exploration is avoiding "
                  f"certified territory ══")
            corrupt(mem.concept_library)
            o4, _ = adapter.optimal_rate(mem)
            print(f"    library {adapter.probe(mem.concept_library)} · play {o4}/{t}")
            for c in range(1, a.chunks + 1):
                o, mem = chunk_cycle(mem, f"gate chunk {c}")
            print(f"\n    GATE {'PASS' if o >= t * 0.95 else 'FAIL'}: "
                  f"recovery under avoidance reached {o}/{t}")
        mem.close()
    set_cert_aware(False)


# ---------------------------------------------------------------------------
# The four acts
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=["nim", "ttt"], default="nim")
    ap.add_argument("--piles", type=int, default=4)
    ap.add_argument("--sims", type=int, default=None, help="Act-1 training games")
    ap.add_argument("--k", type=int, default=3, help="confirms per certificate")
    ap.add_argument("--margin", type=float, default=0.3, help="claim sharpness bar")
    ap.add_argument("--eps", type=float, default=0.25, help="rule-reproduces-value band")
    ap.add_argument("--chunks", type=int, default=4, help="Act-4 recovery chunks")
    ap.add_argument("--chunk-games", type=int, default=400)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse the existing trained DB; skip Act-1 training")
    ap.add_argument("--harvest", type=int, default=0,
                    help="harvest-yield mode: play N training-style games and "
                         "count free certification evidence (requires --reuse)")
    ap.add_argument("--budget", action="store_true",
                    help="certificate-aware exploration A/B + the corruption gate")
    a = ap.parse_args()
    adapter = NimAdapter(a.piles) if a.game == "nim" else TTTAdapter()
    sims = a.sims or adapter.default_sims
    random.seed(7); np.random.seed(7)

    if a.budget:
        run_budget(adapter, a)
        return

    if a.harvest:
        mem = fresh_mem(adapter, reuse=True)
        print(f"══ HARVEST — amortized verification, {a.game} ══")
        harvest(mem, adapter, a.harvest, a.k, a.margin)
        mem.close()
        return

    print(f"══ ACT 1 · BASELINE — {a.game}, "
          f"{'reusing trained DB' if a.reuse else f'{sims} games'} ══")
    mem = fresh_mem(adapter, reuse=a.reuse)
    if not a.reuse:
        train(mem, adapter, sims)
    o, t = adapter.optimal_rate(mem)
    base_rows = rows(mem)
    print(f"    play {o}/{t} optimal · {base_rows} transition rows · "
          f"library {adapter.probe(mem.concept_library)}")
    print("    " + mem.concept_library.summary().replace("\n", "\n    "))

    print(f"\n══ ACT 2 · CERTIFY by path-credited rollout (k={a.k}, margin={a.margin}) ══")
    certified, _, _ = certify_paths(mem, adapter, a.k, a.margin)
    print("    negative control — invert the library, certify again:")
    corrupt(mem.concept_library)
    certify_paths(mem, adapter, a.k, a.margin)
    corrupt(mem.concept_library)                        # invert back = restore

    print(f"\n══ ACT 3 · COLLAPSE — forget what the theory explains (eps={a.eps}) ══")
    deleted, exceptions = collapse(mem, adapter, certified, a.eps)
    o3, _ = adapter.optimal_rate(mem)
    print(f"    rows {base_rows} → {rows(mem)}  (deleted {deleted}, "
          f"kept {exceptions} exceptions on certified boards)")
    print(f"    play after collapse: {o3}/{t}")

    print(f"\n══ ACT 4 · THE FEAR — corrupt the theory AFTER the data is gone ══")
    corrupt(mem.concept_library)
    o4, _ = adapter.optimal_rate(mem)
    print(f"    library {adapter.probe(mem.concept_library)} · play {o4}/{t}")
    for c in range(1, a.chunks + 1):
        train(mem, adapter, a.chunk_games)
        o, _ = adapter.optimal_rate(mem)
        pre = rows(mem)
        cert, n_play, _ = certify_paths(mem, adapter, a.k, a.margin, quiet=True)
        collapse(mem, adapter, cert, a.eps)
        print(f"    +{c * a.chunk_games} games: play {o}/{t} · "
              f"rows {pre} → {rows(mem)} after re-collapse "
              f"(certified {len(cert)} in {n_play} playouts) · "
              f"surprisal {surprisal_rows(mem, adapter, a.eps)} · "
              f"library {adapter.probe(mem.concept_library)}")
    print(f"\n    verdict: {'recovered' if o >= o3 else 'NOT yet recovered — inspect above'}")
    mem.close()


if __name__ == "__main__":
    main()
