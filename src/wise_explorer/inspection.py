"""
Render discovered predicates as a human-readable report.

This is the shared engine behind both `wise-explorer inspect` and the
`scripts/inspect_predicates.py` wrapper. The two public entry points are:

    render_predicates(memory, ...)   # full visual report (boards, rules, verdicts)
    summarize_predicates(memory, ...)  # one-line "what it learned" summary

Both read the predicate library already persisted in `memory` — no retraining.
Pass remine=True to recompute the finer (unpruned) tree instead.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from wise_explorer.core.types import Stats
from wise_explorer.memory.predicates import (
    AtomClause, Eq, Neq, Literal, BoardAt, MakeSq, FromBoardAt, Predicate,
    AggAtom, NegAggAtom,
)

try:
    from wise_explorer.games.minichess import MiniChess, KING
    HAS_CHESS = True
except ImportError:
    HAS_CHESS = False

try:
    from wise_explorer.games.minichess import MiniChess, KING
    HAS_CHESS = True
except ImportError:
    HAS_CHESS = False


# ═══════════════════════════════════════════════════════════════════════════
# Box drawing — one width, one pad function, guaranteed alignment
# ═══════════════════════════════════════════════════════════════════════════

W = 66  # inner content width (between the ║ walls)


def row(text=""):
    """Pad text to exactly W and wrap in box walls."""
    # Truncate if too long
    if len(text) > W:
        text = text[:W - 1] + "…"
    return f"  ║ {text:<{W}} ║"


def top(title=""):
    if title:
        t = f" {title} "
        side = (W - len(t)) // 2
        return f"  ╔{'═' * side}{t}{'═' * (W - side - len(t) + 2)}╗"
    return f"  ╔{'═' * (W + 2)}╗"


def mid():
    return f"  ╠{'═' * (W + 2)}╣"


def bot(subtitle=""):
    if subtitle:
        t = f" {subtitle} "
        side = (W - len(t)) // 2
        return f"  ╚{'═' * side}{t}{'═' * (W - side - len(t) + 2)}╝"
    return f"  ╚{'═' * (W + 2)}╝"


def hdr(title=""):
    """Simple header box."""
    t = f" {title} "
    side = (W - len(t)) // 2
    return f"  ┌{'─' * side}{t}{'─' * (W - side - len(t) + 2)}┐"


def hdr_row(text=""):
    return f"  │ {text:<{W}} │"


def hdr_bot():
    return f"  └{'─' * (W + 2)}┘"


# ═══════════════════════════════════════════════════════════════════════════
# Game configs
# ═══════════════════════════════════════════════════════════════════════════

PIECE_SYM = {
    "ttt":   {0: ".", 1: "X", 2: "O"},
    "nim":   {i: str(i) for i in range(10)},
    "chess": {
        0: ".", 1: "P1", -1: "p2", 2: "C1", -2: "c2",
        3: "K1", -3: "k2", 4: "Q1", -4: "q2",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Constraint extraction
# ═══════════════════════════════════════════════════════════════════════════

def _get_cell(expr):
    if isinstance(expr, (BoardAt, FromBoardAt)):
        sq = expr.square
        if isinstance(sq, MakeSq) and isinstance(sq.rank, Literal) and isinstance(sq.file, Literal):
            return ("from" if isinstance(expr, FromBoardAt) else "board",
                    sq.rank.value, sq.file.value)
    return None


def extract(pred):
    beq, bneq, feq, fneq, cross = {}, {}, {}, {}, []
    for clause in pred.conjunction.clauses:
        atom = clause.atom if isinstance(clause, AtomClause) else clause
        if not isinstance(atom, (Eq, Neq)):
            continue
        lc, rc = _get_cell(atom.left), _get_cell(atom.right)
        rl = atom.right.value if isinstance(atom.right, Literal) else None
        if isinstance(atom, Eq):
            if lc and rl is not None:
                (beq if lc[0] == "board" else feq)[(lc[1], lc[2])] = rl
            elif lc and rc:
                cross.append((lc, rc))
        elif isinstance(atom, Neq):
            if lc and rl is not None:
                (bneq if lc[0] == "board" else fneq)[(lc[1], lc[2])] = rl
    bl, fl = {}, {}
    for i, (left, right) in enumerate(cross):
        ch = chr(ord("A") + i) if i < 26 else "?"
        {"board": bl, "from": fl}[left[0]][(left[1], left[2])] = ch
        {"board": bl, "from": fl}[right[0]][(right[1], right[2])] = ch
    return beq, bneq, feq, fneq, cross, bl, fl


# ═══════════════════════════════════════════════════════════════════════════
# Cell rendering
# ═══════════════════════════════════════════════════════════════════════════

def cell(r, c, eq, neq, labels, sym):
    if (r, c) in eq:
        return sym.get(eq[(r, c)], str(eq[(r, c)]))
    if (r, c) in neq:
        v = neq[(r, c)]
        return "#" if v == 0 else f"!{sym.get(v, str(v))}"
    if labels and (r, c) in labels:
        return labels[(r, c)]
    return "."


# ═══════════════════════════════════════════════════════════════════════════
# Board grid rendering
# ═══════════════════════════════════════════════════════════════════════════

def grid_lines(eq, neq, rows, cols, labels, sym, gt):
    """Return list of strings for a board grid (no box walls)."""
    cw = max(3, max((len(str(v)) for v in sym.values()), default=1) + 1)
    lines = []
    # Column headers
    if gt == "nim":
        hdrs = [f"H{c}".center(cw) for c in range(cols)]
    elif gt == "chess":
        hdrs = [f"{'abcdefgh'[c]}".center(cw) for c in range(cols)]
    else:
        hdrs = [str(c).center(cw) for c in range(cols)]

    pfx_w = 3 if gt == "chess" else 0
    pfx = " " * pfx_w
    lines.append(pfx + " ".join(hdrs))
    lines.append(pfx + "┌" + "┬".join("─" * cw for _ in range(cols)) + "┐")
    for r in range(rows):
        rank = f"{rows - r:>2} " if gt == "chess" else ""
        cells = [cell(r, c, eq, neq, labels, sym).center(cw) for c in range(cols)]
        lines.append(rank + "│" + "│".join(cells) + "│")
    lines.append(pfx + "└" + "┴".join("─" * cw for _ in range(cols)) + "┘")
    return lines


# ═══════════════════════════════════════════════════════════════════════════
# Natural language
# ═══════════════════════════════════════════════════════════════════════════

def _pos(r, c, gt, rows):
    if gt == "nim":   return f"H{c}"
    if gt == "chess": return f"{'abcdefgh'[c]}{rows - r}"
    return {(0,0):"TL",(0,1):"TC",(0,2):"TR",(1,0):"ML",(1,1):"MID",
            (1,2):"MR",(2,0):"BL",(2,1):"BC",(2,2):"BR"}.get((r,c), f"[{r},{c}]")


def _agg_phrase(atom, gt):
    """Friendly text for an aggregate atom (sum/max/min/count/xor over a group).

    The natural-language renderer above only covers single-cell Eq/Neq atoms;
    aggregate atoms (which carry the most powerful patterns, e.g. Nim's nim-sum)
    would otherwise be dropped, making a rule look like a meaningless catch-all.
    """
    neg = isinstance(atom, NegAggAtom)
    desc = atom.descriptor
    kind, group = desc[0], desc[1]

    # Nim's nim-sum (XOR of all heaps) is the whole theory of the game — name it.
    if gt == "nim" and group == "all" and kind == "agg_xor_eq":
        return f"nim-sum {'≠' if neg else '='} {desc[3]}"

    # Otherwise reuse AggAtom's own repr (e.g. "sum(all)>2"), with a readable
    # group name and an explicit negation when needed.
    grp = ("all heaps" if gt == "nim" else "the board") if group == "all" else group
    base = repr(AggAtom(descriptor=desc)).replace(f"({group})", f"({grp})")
    return f"NOT {base}" if neg else base


def nat_lang(pred, rows, cols, gt):
    beq, bneq, feq, fneq, cross, _, _ = extract(pred)
    sym = PIECE_SYM.get(gt, {})
    vn = lambda v: sym.get(v, str(v))
    pn = lambda r, c: _pos(r, c, gt, rows)
    parts = []
    # Aggregate atoms first — they carry the headline pattern (e.g. the nim-sum).
    for clause in pred.conjunction.clauses:
        atom = clause.atom if isinstance(clause, AtomClause) else clause
        if isinstance(atom, (AggAtom, NegAggAtom)):
            parts.append(_agg_phrase(atom, gt))
    for (r,c), v in sorted(feq.items()):  parts.append(f"{pn(r,c)} was {vn(v)}")
    for (r,c), v in sorted(fneq.items()): parts.append(f"{pn(r,c)} was {'occupied' if v==0 else f'not {vn(v)}'}")
    for (r,c), v in sorted(beq.items()):  parts.append(f"{pn(r,c)} is {vn(v)}")
    for (r,c), v in sorted(bneq.items()): parts.append(f"{pn(r,c)} is {'occupied' if v==0 else f'not {vn(v)}'}")
    for left, right in cross:
        s1 = ("was:" if left[0]=="from" else "") + pn(left[1], left[2])
        s2 = ("was:" if right[0]=="from" else "") + pn(right[1], right[2])
        parts.append(f"{s1} == {s2}")
    return " AND ".join(parts) if parts else "(catch-all)"


# ═══════════════════════════════════════════════════════════════════════════
# Transition-aware matching
# ═══════════════════════════════════════════════════════════════════════════

def _match_pred(pred, boards, trans_scores):
    """Match predicate against transitions (from, to pairs).

    Returns list of (to_board, from_board) pairs that match.
    Uses bindings with _from for cross-board atoms.
    """
    matched = []
    seen = set()
    for (fh, th) in trans_scores:
        if th in seen:
            continue
        if fh not in boards or th not in boards:
            continue
        to_b = boards[th]
        from_b = boards[fh]
        to_2d = to_b if to_b.ndim == 2 else to_b.reshape(1, -1)
        bindings = {"_from": from_b if from_b.ndim == 2 else from_b.reshape(1, -1)}
        if pred.conjunction.matches(to_2d, bindings):
            matched.append((to_b, from_b))
            seen.add(th)
    return matched


# ═══════════════════════════════════════════════════════════════════════════
# TRUE score
# ═══════════════════════════════════════════════════════════════════════════

def true_nim(matched):
    m = [(tuple(to_b.ravel()), int(np.bitwise_xor.reduce(to_b.ravel())))
         for to_b, _ in matched]
    if not m: return None, "no matches"
    nw = sum(1 for _, ns in m if ns == 0)
    if nw == len(m): return 1.0, f"all {len(m)} ns=0 -> WIN"
    if nw == 0:      return 0.0, f"all {len(m)} ns!=0 -> LOSS"
    return None, f"{nw}/{len(m)} ns=0"

def true_ttt(matched):
    LINES = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    m = []
    for to_b, _ in matched:
        f = to_b.ravel(); w = 0
        for ln in LINES:
            if f[ln[0]] != 0 and f[ln[0]] == f[ln[1]] == f[ln[2]]: w = int(f[ln[0]]); break
        m.append((w, not np.any(f == 0)))
    if not m: return None, "no matches"
    nw = sum(1 for w, _ in m if w != 0)
    nd = sum(1 for w, full in m if w == 0 and full)
    if nw == len(m): return 1.0, f"all {len(m)} have winner"
    if nd == len(m): return 0.5, f"all {len(m)} drawn"
    return None, f"{nw} won, {nd} drawn / {len(m)}"

def true_chess(matched):
    if not HAS_CHESS: return None, "N/A"
    m = [(bool(np.any(to_b == KING)), bool(np.any(to_b == -KING)))
         for to_b, _ in matched]
    if not m: return None, "no matches"
    k1d = sum(1 for k1, _ in m if not k1)
    k2d = sum(1 for _, k2 in m if not k2)
    n = len(m)
    if k1d == n: return 1.0, f"all {n}: K1 captured -> P2 wins"
    if k2d == n: return 1.0, f"all {n}: k2 captured -> P1 wins"
    if k1d+k2d > 0: return None, f"{k1d} K1-dead, {k2d} k2-dead / {n}"
    return None, f"all {n}: kings alive"

TRUE_FN = {"nim": true_nim, "ttt": true_ttt, "chess": true_chess}


# ═══════════════════════════════════════════════════════════════════════════
# Score helpers
# ═══════════════════════════════════════════════════════════════════════════

def tag(s):
    if s > 0.65: return "WIN  ++"
    if s > 0.55: return "GOOD + "
    if s > 0.51: return "SLGT + "
    if s < 0.35: return "LOSS --"
    if s < 0.45: return "BAD  - "
    if s < 0.49: return "SLGT - "
    return "NEUTRAL"

def conf(support, variance):
    s = min(support / 200, 1.0)
    v = max(1.0 - variance * 20, 0.0)
    n = int((s * 0.5 + v * 0.5) * 10)
    return "\u2588" * n + "\u2591" * (10 - n)


# ═══════════════════════════════════════════════════════════════════════════
# Example board rendering
# ═══════════════════════════════════════════════════════════════════════════

def _render_board_compact(board, sym, gt, rows, cols):
    """Render a full board state as compact lines."""
    if gt == "nim":
        heaps = board.ravel()
        return ["[" + ", ".join(str(int(h)) for h in heaps) + "]"]

    lines = []
    cw = max(2, max((len(str(v)) for v in sym.values()), default=1))
    for r in range(rows):
        cells = []
        for c in range(cols):
            v = int(board[r, c]) if board.ndim == 2 else int(board[r * cols + c])
            s = sym.get(v, str(v))
            cells.append(s.center(cw))
        lines.append("│" + "│".join(cells) + "│")
    return lines


def _collect_examples(pred, boards, sym, gt, rows, cols, max_examples=3):
    """Collect a few example board renderings that match this predicate."""
    examples = []
    count = 0
    for h, board in boards.items():
        if count >= max_examples:
            break
        if pred.matches(board):
            lines = _render_board_compact(board, sym, gt, rows, cols)
            examples.append(lines)
            count += 1
    return examples


def _show_matching_boards(matched, sym, gt, rows, cols):
    """Unified display: summary line + up to 4 example boards with verdicts."""
    MAX_SHOW = 4

    # Build entries with game-specific verdict
    entries = []
    for to_b, from_b in matched:
        compact = _render_board_compact(to_b, sym, gt, rows, cols)
        if gt == "nim":
            heaps = to_b.ravel()
            ns = int(np.bitwise_xor.reduce(heaps))
            verdict = "WIN " if ns == 0 else "LOSS"
            detail = f"ns={ns}"
        elif gt == "chess" and HAS_CHESS:
            k1 = bool(np.any(to_b == KING))
            k2 = bool(np.any(to_b == -KING))
            if not k1:     verdict, detail = "WIN ", "K1 captured"
            elif not k2:   verdict, detail = "WIN ", "k2 captured"
            else:          verdict, detail = "    ", "ongoing"
        elif gt == "ttt":
            LINES = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
            flat = to_b.ravel()
            w = 0
            for ln in LINES:
                if flat[ln[0]] != 0 and flat[ln[0]] == flat[ln[1]] == flat[ln[2]]:
                    w = int(flat[ln[0]]); break
            if w != 0:       verdict, detail = "WIN ", f"P{w} wins"
            elif not np.any(flat == 0): verdict, detail = "DRAW", "full board"
            else:            verdict, detail = "    ", "ongoing"
        else:
            verdict, detail = "    ", ""
        entries.append((compact, verdict, detail))

    total = len(entries)
    if total == 0:
        print(row(f"  Matching boards: 0"))
        return

    # Summary counts
    n_win = sum(1 for _, v, _ in entries if v.strip() == "WIN")
    n_loss = sum(1 for _, v, _ in entries if v.strip() == "LOSS")
    n_draw = sum(1 for _, v, _ in entries if v.strip() == "DRAW")
    n_other = total - n_win - n_loss - n_draw

    parts = [f"{total} boards"]
    if n_win:  parts.append(f"{n_win} WIN")
    if n_loss: parts.append(f"{n_loss} LOSS")
    if n_draw: parts.append(f"{n_draw} DRAW")
    if n_other and (n_win or n_loss or n_draw): parts.append(f"{n_other} other")
    print(row(f"  Matching: {', '.join(parts)}"))

    # Show up to MAX_SHOW examples
    shown = entries[:MAX_SHOW]
    if shown:
        print(row())
    for compact, verdict, detail in shown:
        label = f"[{verdict}]" if verdict.strip() else ""
        if gt == "nim":
            # Single-line for nim
            print(row(f"    {compact[0]:<24s} {detail:<10s} {label}"))
        else:
            # Multi-line grid for 2D games
            for j, line in enumerate(compact):
                if j == 0:
                    print(row(f"    {line}   {label} {detail}"))
                else:
                    print(row(f"    {line}"))
            print(row())

    if total > MAX_SHOW:
        print(row(f"    ... +{total - MAX_SHOW} more"))


# ═══════════════════════════════════════════════════════════════════════════
# Predicate display
# ═══════════════════════════════════════════════════════════════════════════

def show_pred(pred, idx, boards, rows, cols, sym, gt, trans_scores):
    beq, bneq, feq, fneq, cross, bl, fl = extract(pred)
    has_from = feq or fneq or fl
    ms = pred.mining_score
    cs = Stats(*pred.counts)
    w, t, l = pred.counts

    # Match against transitions (handles cross-board atoms correctly)
    matched = _match_pred(pred, boards, trans_scores)
    tfn = TRUE_FN.get(gt)
    tv, td = tfn(matched) if tfn else (None, "N/A")
    tg = tag(ms)

    # ── Header ──
    print(top(f"#{idx}  {tg}  |  score: {ms:.3f}"))
    print(row())

    # ── Scores ──
    print(row(f"  Mining:     {ms:.3f}      (tree-split signal)"))
    print(row(f"  Counts:     {cs.mean_score:.3f}      ({w:.0f}W / {t:.0f}T / {l:.0f}L = {w+t+l:.0f} games)"))
    if tv is not None:
        print(row(f"  TRUE:       {tv:.3f}  <-- {td}"))
    else:
        print(row(f"  TRUE:       ???       ({td})"))
    print(row(f"  Confidence: {conf(pred.support, pred.variance)}   n={pred.support}  var={pred.variance:.4f}"))
    print(row())
    print(mid())
    print(row())

    # ── Board visual ──
    if gt == "nim":
        _show_nim_board(feq, fneq, fl, beq, bneq, bl, has_from, cols, sym)
    else:
        _show_grid_board(feq, fneq, fl, beq, bneq, bl, has_from, rows, cols, sym, gt)

    print(row())
    print(mid())

    # ── Rule ──
    rule = nat_lang(pred, rows, cols, gt)
    # Word-wrap on AND
    max_r = W - 10
    if len(rule) <= max_r:
        print(row(f"  Rule: {rule}"))
    else:
        parts = rule.split(" AND ")
        line = ""
        first = True
        for part in parts:
            candidate = (line + " AND " + part) if line else part
            if len(candidate) > max_r and line:
                pfx = "  Rule: " if first else "        "
                first = False
                print(row(f"{pfx}{line}"))
                line = "AND " + part
            else:
                line = candidate
        pfx = "  Rule: " if first else "        "
        print(row(f"{pfx}{line}"))

    # ── Matching boards (combined analysis + examples) ──
    print(row())
    _show_matching_boards(matched, sym, gt, rows, cols)

    print(row())
    print(bot(f"support: {pred.support}  |  var: {pred.variance:.4f}"))
    print()


def _show_nim_board(feq, fneq, fl, beq, bneq, bl, has_from, cols, sym):
    def _heap_row(eq, neq, labels):
        return "  ".join(f"H{c}={cell(0,c,eq,neq,labels,sym):>2s}" for c in range(cols))
    if has_from:
        print(row(f"  BEFORE:  {_heap_row(feq, fneq, fl)}"))
        print(row(f"      |"))
        print(row(f"      v"))
        print(row(f"  AFTER:   {_heap_row(beq, bneq, bl)}"))
    else:
        print(row(f"  HEAPS:   {_heap_row(beq, bneq, bl)}"))


def _show_grid_board(feq, fneq, fl, beq, bneq, bl, has_from, rows, cols, sym, gt):
    after = grid_lines(beq, bneq, rows, cols, bl, sym, gt)
    if has_from:
        before = grid_lines(feq, fneq, rows, cols, fl, sym, gt)
        gw = max(len(s) for s in before)
        print(row(f"  {'BEFORE':^{gw}}       {'AFTER':^{gw}}"))
        for b, a in zip(before, after):
            print(row(f"  {b}    -> {a}"))
    else:
        for line in after:
            print(row(f"  {line}"))


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def show_summary(predicates, gt, boards, trans_scores):
    nw = sum(1 for p in predicates if p.mining_score > 0.55)
    nl = sum(1 for p in predicates if p.mining_score < 0.45)
    nm = len(predicates) - nw - nl
    tfn = TRUE_FN.get(gt)
    np_ = 0
    if tfn:
        for p in predicates:
            matched = _match_pred(p, boards, trans_scores)
            tv, _ = tfn(matched)
            if tv is not None:
                np_ += 1

    print(hdr("SUMMARY"))
    print(hdr_row(f"Total predicates:   {len(predicates)}"))
    print(hdr_row(f"Winning (>0.55):    {nw}"))
    print(hdr_row(f"Losing  (<0.45):    {nl}"))
    print(hdr_row(f"Neutral:            {nm}"))
    if tfn:
        print(hdr_row(f"Provably correct:   {np_}"))
    print(hdr_bot())
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Game type detection
# ═══════════════════════════════════════════════════════════════════════════

# Map a game's registry id to the inspector's internal game-type label.
_GAME_ID_TO_GT = {"nim": "nim", "tic_tac_toe": "ttt", "minichess": "chess"}

# Module-level alias so functions taking a `top_n` argument can still reach the
# box-drawing `top()` without it being shadowed by the parameter name.
_box_top = top


def detect_game(mem, game_id: Optional[str] = None):
    """Infer (game_type, boards, rows, cols) from a memory's stored boards.

    A game_id hint (the registry name) wins over shape-based guessing when given.
    """
    boards = mem._load_boards()
    hint = _GAME_ID_TO_GT.get(game_id or "")
    if not boards:
        if hint == "nim":
            return "nim", boards, 1, 4
        return hint or "ttt", boards, 3, 3
    s = next(iter(boards.values()))
    if s.ndim == 1:
        return hint or "nim", boards, 1, len(s)
    r, c = s.shape
    if hint:
        return hint, boards, r, c
    if r == 1:
        return "nim", boards, 1, c
    if r == 3 and c == 3:
        return "ttt", boards, r, c
    return "chess", boards, r, c


# ═══════════════════════════════════════════════════════════════════════════
# Public entry points
# ═══════════════════════════════════════════════════════════════════════════

def _score(p) -> float:
    """Best available score for a predicate (mining signal, else raw counts)."""
    return p.mean_score


def _dedup_sort(preds):
    seen, uniq = set(), []
    for p in preds:
        k = str(p.conjunction)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    uniq.sort(key=_score)
    return uniq


def render_predicates(
    memory,
    *,
    game_id: Optional[str] = None,
    top_n: Optional[int] = None,
    wins_only: bool = False,
    losses_only: bool = False,
    remine: bool = True,
    db_label: Optional[str] = None,
) -> int:
    """Print the full predicate report for an OPEN memory.

    By default (remine=True) the rule tree is recomputed from the stored
    transitions — fast, no self-play, and the cleanest view of the patterns the
    data supports. Pass remine=False to instead render the persisted predicate
    library exactly (the compact set the agent plays with). Returns the number
    of predicates shown.
    """
    gt, boards, rows, cols = detect_game(memory, game_id)
    sym = PIECE_SYM.get(gt, {0: "."})
    _, trans_scores = memory._build_trans_scores()

    if remine or memory.predicate_library.count == 0:
        # Batch CART for one-shot inspection: deterministic and globally optimal
        # at each split (the incremental ITI miner used during training trades
        # that off for speed and can pick sub-optimal splits).
        from wise_explorer.memory.tree_miner import TreeMiner
        preds = TreeMiner().mine(boards, trans_scores) if trans_scores else []
        source = (f"mined from {len(trans_scores)} stored transitions (no retraining)"
                  if remine else "saved library empty — mined from transitions")
    else:
        preds = memory.predicate_library.predicates
        source = "saved predicate library (the agent's compact set)"

    unique = _dedup_sort(preds)
    if wins_only:
        unique = [p for p in unique if _score(p) > 0.55]
    elif losses_only:
        unique = [p for p in unique if _score(p) < 0.45]
    if top_n:
        n = top_n // 2
        unique = [p for p in unique if _score(p) < 0.5][:n] + \
                 [p for p in unique if _score(p) >= 0.5][-n:]

    print()
    print(_box_top(f"PREDICATE INSPECTOR - {gt.upper()}"))
    print(row(f"{len(trans_scores)} transitions  |  {len(boards)} boards  |  {rows}x{cols} board"))
    print(row(f"source: {source}"))
    if db_label:
        print(row(f"DB: {db_label}"))
    print(bot())
    print()

    show_summary(unique, gt, boards, trans_scores)
    for i, pred in enumerate(unique, 1):
        show_pred(pred, i, boards, rows, cols, sym, gt, trans_scores)
    return len(unique)


def summarize_predicates(memory, game_id: Optional[str] = None) -> str:
    """One-line summary of what was learned, for printing after training.

    Returns "" when nothing has been mined yet.
    """
    preds = memory.predicate_library.predicates
    if not preds:
        return ""
    gt, boards, rows, cols = detect_game(memory, game_id)
    wins = [p for p in preds if _score(p) > 0.5]
    pick = (max(wins, key=lambda p: (_score(p), p.support)) if wins
            else max(preds, key=lambda p: p.support))
    rule = nat_lang(pick, rows, cols, gt)
    w, t, l = pick.counts
    verdict = "WIN" if _score(pick) > 0.5 else "LOSS"
    hint = "wise-explorer inspect" + (f" -g {game_id}" if game_id else "")
    return (f"Learned {len(preds)} rule(s) · top: ({rule}) → {verdict}  "
            f"[{w:.0f}W/{l:.0f}L]\n  → see all:  {hint}")
