"""
Terminal visualizer for move analysis — game-agnostic.

Supports move formats:
    - Single-step: [fr,fc,tr,tc] (or general: [a1,a2,...,b1,b2,...] where halves are origin/dest)
    - Multi-step: [[fr,fc,tr,tc], [fr2,fc2,tr2,tc2]]
    - Flattened multi-dim: [1,2,3,4,2,3] -> origin=(1,2,3), dest=(4,2,3)

Diff entries are expected as (pos, before, after).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# ANSI Colors & Styling
# ═══════════════════════════════════════════════════════════════════════════════

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

FG = {
    "green": "\033[38;5;28m",
    "red": "\033[38;5;124m",
    "yellow": "\033[38;5;142m",
    "orange": "\033[38;5;166m",
    "gray": "\033[38;5;245m",
    "white": "\033[38;5;255m",
    "black": "\033[38;5;233m",
    "cyan": "\033[38;5;37m",
}

BG = {
    "selected": "\033[48;5;22m",  # Dark green
    "capture": "\033[48;5;52m",  # Dark red
}


def bg_gray(level: int) -> str:
    """Return a gray background code for level in [0..23]."""
    level = max(0, min(23, level))
    return f"\033[48;5;{232 + level}m"


def score_color(score: float) -> str:
    if score >= 0.65:
        return FG["green"]
    if score >= 0.45:
        return FG["yellow"]
    if score >= 0.35:
        return FG["orange"]
    return FG["red"]


# ═══════════════════════════════════════════════════════════════════════════════
# Text utilities
# ═══════════════════════════════════════════════════════════════════════════════

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def visible_len(text: str) -> int:
    return len(strip_ansi(text))


def pad(text: str, width: int, align: str = "left") -> str:
    gap = max(0, width - visible_len(text))
    if align == "right":
        return " " * gap + text
    if align == "center":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


# ═══════════════════════════════════════════════════════════════════════════════
# Simple helpers for diffs / pieces / positions
# ═══════════════════════════════════════════════════════════════════════════════

EmptyVals = (None, 0)


def is_empty(val: Any) -> bool:
    return val in EmptyVals


def normalize_pos(pos: Any) -> Tuple[int, ...]:
    """
    Normalize a position into a coordinate tuple.

    Accepts:
      - tuple/list/np array of ints -> tuple
      - single int -> (int,)
    """
    if isinstance(pos, (tuple, list, np.ndarray)):
        return tuple(int(x) for x in pos)
    return (int(pos),)


def fmt_piece(val: Any, cell_strings: Optional[Dict[int, str]] = None) -> str:
    """Format a board value using the cell_strings dict. Returns '' for empty/None."""
    if is_empty(val):
        return ""
    if cell_strings and val in cell_strings:
        return str(cell_strings[val])
    return str(val)


# ═══════════════════════════════════════════════════════════════════════════════
# Diff analysis
# ═══════════════════════════════════════════════════════════════════════════════

Change = Dict[str, Any]  # {"before": v, "after": v, "type": "arr"|"dep"|"cap"}


def analyze_diff(diff: List[Sequence]) -> Dict[Tuple[int, ...], Change]:
    """
    Normalize and classify a list of diffs.

    Input diff entries: (pos, before, after)
      - pos may be scalar or sequence, will be normalized to a tuple
    Returns mapping: normalized_pos -> {"before": before, "after": after, "type": ...}

    Types:
      - "arr": empty -> piece (arrival)
      - "dep": piece -> empty (departure)
      - "cap": piece -> different piece (capture / replacement)
    No-op entries where before == after are ignored.
    """
    out: Dict[Tuple[int, ...], Change] = {}
    for entry in diff:
        if not entry or len(entry) < 3:
            continue
        raw_pos, before, after = entry[0], entry[1], entry[2]
        pos = normalize_pos(raw_pos)
        if before == after:
            # nothing changed here — skip
            continue

        empty_before = is_empty(before)
        empty_after = is_empty(after)

        if empty_before and not empty_after:
            ctype = "arr"
        elif not empty_before and empty_after:
            ctype = "dep"
        else:
            ctype = "cap"

        out[pos] = {"before": before, "after": after, "type": ctype}
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Move normalization & string helpers (game-agnostic)
# ═══════════════════════════════════════════════════════════════════════════════


def normalize_move_to_steps(move: Any) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    Normalize various move shapes into a list of (origin_tuple, dest_tuple) steps.

    Supported shapes:
      - A nested list of steps: [[fr,fc,tr,tc], [fr2,fc2,tr2,tc2], ...]
        Each step is split in half (origin | dest).
      - A flat iterable: [a1,a2,...,b1,b2,...] → split into two equal halves: origin=(a1,...), dest=(b1,...)
      - A short placement: [r,c] -> treated as single step with origin=(), dest=(r,c)
    Returns: list of (origin_tuple, dest_tuple)
    """
    steps: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []

    # If already a list of steps (first element iterable and not a scalar)
    if isinstance(move, list) and move and hasattr(move[0], "__iter__") and not isinstance(move[0], (str, bytes)):
        for m in move:
            arr = np.asarray(m).flatten()
            if arr.size == 0:
                continue
            if arr.size == 2:
                # placement: no origin
                origin = tuple()
                dest = (int(arr[0]), int(arr[1]))
            elif arr.size >= 2 and arr.size % 2 == 0:
                half = arr.size // 2
                origin = tuple(int(x) for x in arr[:half])
                dest = tuple(int(x) for x in arr[half:])
            else:
                # odd-length; treat first element(s) as origin except last as dest (fallback)
                origin = tuple(int(x) for x in arr[:-1])
                dest = (int(arr[-1]),)
            steps.append((origin, dest))
        return steps

    # Else treat move as a single flat iterable (list/tuple/ndarray)
    arr = np.asarray(move).flatten()
    if arr.size == 0:
        return steps
    if arr.size == 2:
        # placement
        steps.append((tuple(), (int(arr[0]), int(arr[1]))))
        return steps
    if arr.size >= 2 and arr.size % 2 == 0:
        half = arr.size // 2
        origin = tuple(int(x) for x in arr[:half])
        dest = tuple(int(x) for x in arr[half:])
        steps.append((origin, dest))
        return steps

    # odd-length fallback: treat first n-1 as origin, last as dest
    origin = tuple(int(x) for x in arr[:-1])
    dest = (int(arr[-1]),)
    steps.append((origin, dest))
    return steps


def _is_capture_at(diff_map: Dict[Tuple[int, ...], Change], dest: Tuple[int, ...]) -> Optional[Any]:
    """Return captured piece (before) at dest if dest is a capture, else None."""
    ch = diff_map.get(dest)
    if ch and ch["type"] == "cap":
        return ch["before"]
    return None


def _coords_to_str(coords: Tuple[int, ...]) -> str:
    """Return comma-joined coords or '' for empty origin."""
    if not coords:
        return ""
    return ",".join(str(int(x)) for x in coords)


def move_str_from_array(move: Any, diff: List[Sequence], cell_strings: Dict[int, str]) -> str:
    """
    Build a concise human move string from an array (game-agnostic).
    Uses normalize_move_to_steps to handle all supported shapes.
    """
    steps = normalize_move_to_steps(move)
    if not steps:
        return "?"

    diff_map = analyze_diff(diff or [])

    parts: List[str] = []
    for origin, dest in steps:
        orig_s = _coords_to_str(origin) if origin else "place"
        dest_s = _coords_to_str(dest)
        victim = _is_capture_at(diff_map, dest)
        if victim not in EmptyVals:
            parts.append(f"{orig_s}→{dest_s}×{fmt_piece(victim, cell_strings)}")
        else:
            parts.append(f"{orig_s}→{dest_s}")
    return " → ".join(parts)


def move_str_from_diff(diff: List[Sequence], cell_strings: Dict[int, str]) -> str:
    """
    Build a move string from a diff only (when move array isn't available).
    Uses counts & classifications to create a reasonable string (unchanged behavior).
    """
    if not diff:
        return "?"
    changes = analyze_diff(diff)
    deps = [(p, c["before"]) for p, c in changes.items() if c["type"] == "dep"]
    arrs = [(p, c["after"]) for p, c in changes.items() if c["type"] == "arr"]
    caps = [(p, c["before"], c["after"]) for p, c in changes.items() if c["type"] == "cap"]

    if len(arrs) == 1 and not deps and not caps:
        return pos_str(arrs[0][0])
    if len(deps) == 1 and len(arrs) == 1 and not caps:
        return f"{pos_str(deps[0][0])}→{pos_str(arrs[0][0])}"
    if len(deps) == 1 and not arrs and len(caps) == 1:
        victim = caps[0][1]
        return f"{pos_str(deps[0][0])}→{pos_str(caps[0][0])}×{fmt_piece(victim, cell_strings)}"

    if arrs and deps:
        num_caps = max(0, len(deps) - 1 + len(caps))
        suffix = f" ×{num_caps}" if num_caps else ""
        return f"{pos_str(deps[0][0])}→{pos_str(arrs[0][0])}{suffix}"

    parts = []
    if deps:
        parts.append(f"{len(deps)}↑")
    if arrs:
        parts.append(f"{len(arrs)}↓")
    if caps:
        parts.append(f"{len(caps)}×")
    return f"[{''.join(parts)}]" if parts else "?"


def pos_str(pos: Sequence) -> str:
    """Format position as 'r,c' (pos may be tuple or scalar)."""
    p = normalize_pos(pos)
    return ",".join(str(int(x)) for x in p)


def get_move_str(row: Dict, cell_strings: Dict[int, str]) -> str:
    move = row.get("move")
    diff = row.get("diff", [])
    if move is not None:
        return move_str_from_array(move, diff, cell_strings)
    return move_str_from_diff(diff, cell_strings)


def get_move_desc(row: Dict, cell_strings: Dict[int, str]) -> str:
    """
    Human-readable move description for header. Prefer move array if available,
    otherwise fall back to a diff-based description.
    """
    move = row.get("move")
    diff = row.get("diff", [])
    changes = analyze_diff(diff)

    captured = next((c["before"] for c in changes.values() if c["type"] == "cap"), None)

    if move is not None:
        steps = normalize_move_to_steps(move)
        if len(steps) > 1:
            first_origin, _ = steps[0]
            _, last_dest = steps[-1]
            return f"({_coords_to_str(first_origin) or 'place'}) → ({_coords_to_str(last_dest)}) ({len(steps)} steps)"
        # single step
        origin, dest = steps[0]
        if not origin:
            return f"place at ({_coords_to_str(dest)})"
        desc = f"({_coords_to_str(origin)}) → ({_coords_to_str(dest)})"
        if captured is not None:
            desc += f" ×{fmt_piece(captured, cell_strings)}"
        return desc

    # diff-based fallback
    deps = [p for p, c in changes.items() if c["type"] == "dep"]
    arrs = [p for p, c in changes.items() if c["type"] == "arr"]
    caps = [p for p, c in changes.items() if c["type"] == "cap"]

    if len(arrs) == 1 and not deps and not caps:
        return f"place at {pos_str(arrs[0])}"
    if len(deps) == 1 and len(arrs) == 1 and not caps:
        return f"{pos_str(deps[0])} → {pos_str(arrs[0])}"
    if len(deps) == 1 and not arrs and len(caps) == 1:
        victim = changes[caps[0]]["before"]
        return f"{pos_str(deps[0])} → {pos_str(caps[0])} ×{fmt_piece(victim, cell_strings)}"

    parts = []
    if deps:
        parts.append(f"{len(deps)} leave")
    if arrs:
        parts.append(f"{len(arrs)} arrive")
    if caps:
        parts.append(f"{len(caps)} capture")
    return ", ".join(parts) if parts else "?"


# ═══════════════════════════════════════════════════════════════════════════════
# Table rendering
# ═══════════════════════════════════════════════════════════════════════════════

H, V = "─", "│"
CORNERS = {"tl": "┌", "tr": "┐", "bl": "└", "br": "┘"}
TEES = {"lt": "├", "rt": "┤", "tt": "┬", "bt": "┴", "x": "┼"}
TABLE_W = 70


def hline(left: str, right: str, style: str = "") -> str:
    line = f"{left}{H * TABLE_W}{right}"
    return f"{style}{line}{RESET}" if style else line


def trow(content: str, style: str = "") -> str:
    padding = " " * max(0, TABLE_W - visible_len(content))
    if style:
        return f"{style}{V}{RESET}{content}{padding}{style}{V}{RESET}"
    return f"{V}{content}{padding}{V}"


def wlt_bar(win_pct: float, loss_pct: float, width: int = 10) -> str:
    w = round(win_pct / 100 * width)
    l = round(loss_pct / 100 * width)
    t = width - w - l
    return f"{FG['green']}{'█' * w}{FG['red']}{'█' * l}{FG['yellow']}{'█' * t}{RESET}"


def render_table(rows: List[Dict], rank: int, cell_strings: Dict[int, str]) -> List[str]:
    """Render one anchor group as a table (compact, readable)."""
    first = rows[0]
    aid = first.get("anchor_id")
    total = first.get("anchor_total", 0)
    score = first.get("anchor_score", 0)
    wp = (first.get("anchor_W", 0) / total * 100) if total else 0
    lp = (first.get("anchor_L", 0) / total * 100) if total else 0
    tp = (first.get("anchor_T", 0) / total * 100) if total else 0

    best = rank == 0
    style = f"{BOLD}{FG['green']}" if best else ""

    lines: List[str] = []
    lines.append(hline(CORNERS["tl"], CORNERS["tr"], style))
    marker = f"{FG['green']}★ BEST{RESET}" if best else f"{FG['gray']}#{rank + 1}{RESET}"
    equiv = f" ({len(rows)} equiv)" if len(rows) > 1 else ""
    left = f" {marker}  Anchor {aid}{equiv}"
    right = f"Score: {score_color(score)}{BOLD}{score:.3f}{RESET} "
    lines.append(trow(f"{left}{' ' * (TABLE_W - visible_len(left) - visible_len(right))}{right}", style))

    stats = (
        f" n={total:<5}  "
        f"{FG['green']}W:{wp:5.1f}%{RESET}  "
        f"{FG['red']}L:{lp:5.1f}%{RESET}  "
        f"{FG['yellow']}T:{tp:5.1f}%{RESET}  "
        f"{wlt_bar(wp, lp)}"
    )
    lines.append(trow(stats, style))
    lines.append(hline(TEES["lt"], TEES["rt"], style))

    hdr = (
        f"{DIM} {pad('Move', 16)}{V}"
        f"{pad('W%', 6, 'right')}{pad('L%', 7, 'right')}{pad('T%', 7, 'right')}"
        f"{pad('(n)', 8, 'right')}{V}"
        f"{pad('Solo', 7, 'right')}{V}{pad('Pool', 7, 'right')}{V}     {RESET}"
    )
    lines.append(trow(hdr, style))
    lines.append(hline(TEES["lt"], TEES["rt"], style))

    sorted_rows = sorted(rows, key=lambda r: r.get("direct_score", 0), reverse=True)
    for i, row in enumerate(sorted_rows):
        mv = get_move_str(row, cell_strings)
        n = row.get("direct_total", 0)
        ds, ps = row.get("direct_score", 0), row.get("anchor_score", 0)
        w = (row.get("direct_W", 0) / n * 100) if n else 0
        l = (row.get("direct_L", 0) / n * 100) if n else 0
        t = (row.get("direct_T", 0) / n * 100) if n else 0

        rlbl = f"{FG['yellow']}1st{RESET}" if i == 0 else f"{FG['gray']}#{i+1}{RESET}"
        sel = f"{FG['green']}◀{RESET}" if row.get("is_selected") else " "

        # Build colored & padded pieces safely
        w_colored = FG["green"] + f"{w:.1f}" + RESET
        l_colored = FG["red"] + f"{l:.1f}" + RESET
        t_colored = FG["yellow"] + f"{t:.1f}" + RESET
        n_colored = DIM + f"{n}" + RESET
        ds_colored = score_color(ds) + f"{ds:.3f}" + RESET
        ps_colored = score_color(ps) + f"{ps:.3f}" + RESET

        data = (
            f" {pad(mv, 16)}{V}"
            f"{pad(w_colored, 6, 'right')}"
            f"{pad(l_colored, 7, 'right')}"
            f"{pad(t_colored, 7, 'right')}"
            f"{pad(n_colored, 8, 'right')}{V}"
            f"{pad(ds_colored, 7, 'right')}{V}"
            f"{pad(ps_colored, 7, 'right')}{V}"
            f"{rlbl} {sel}"
        )
        lines.append(trow(data, style))

    lines.append(hline(CORNERS["bl"], CORNERS["br"], style))
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# Board rendering
# ═══════════════════════════════════════════════════════════════════════════════

def render_board(board: np.ndarray, rows: List[Dict], cell_strings: Dict[int, str]) -> List[str]:
    """Render the game board with move visualization (selected move + candidates)."""
    arr = np.atleast_2d(np.array(board))
    if arr.ndim == 1:
        # attempt to reshape into square if possible; otherwise leave as 2D row
        side = int(np.sqrt(arr.size))
        if side * side == arr.size:
            arr = arr.reshape((side, side))

    n_rows, n_cols = arr.shape
    cw = 7  # cell width

    # find selected changes and candidate destination scores
    sel_changes: Optional[Dict[Tuple[int, ...], Change]] = None
    sel_row: Optional[Dict] = None
    candidates: Dict[Tuple[int, ...], float] = {}

    for row in rows:
        dmap = analyze_diff(row.get("diff", []))
        sc = row.get("anchor_score", row.get("direct_score", 0))
        if row.get("is_selected"):
            sel_changes, sel_row = dmap, row
        else:
            for pos, info in dmap.items():
                if info["type"] in ("arr", "cap"):
                    # keep highest-scoring candidate for each dest
                    candidates[pos] = max(sc, candidates.get(pos, -float("inf")))

    def _cell(r: int, c: int) -> str:
        pos = (r, c)
        val = arr[r, c]

        # Selected move visualization
        if sel_changes and pos in sel_changes:
            info = sel_changes[pos]
            if info["type"] == "dep":
                txt = f"○{fmt_piece(info['before'], cell_strings)}"
                return f"{bg_gray(8)}{FG['cyan']}{pad(txt, cw, 'center')}{RESET}"
            if info["type"] == "arr":
                txt = f"●{fmt_piece(info['after'], cell_strings)}"
                return f"{BG['selected']}{FG['white']}{BOLD}{pad(txt, cw, 'center')}{RESET}"
            if info["type"] == "cap":
                txt = f"{fmt_piece(info['after'], cell_strings)}×{fmt_piece(info['before'], cell_strings)}"
                return f"{BG['capture']}{FG['white']}{BOLD}{pad(txt, cw, 'center')}{RESET}"

        # Candidate destination shading (shows score)
        if pos in candidates:
            sc = candidates[pos]
            lv = int(4 + sc * 15)
            lv = max(0, min(23, lv))
            fg = FG["white"] if lv < 12 else FG["black"]
            return f"{bg_gray(lv)}{fg}{pad(f'{sc:.3f}', cw, 'center')}{RESET}"

        # Occupied cell
        if not is_empty(val):
            return f"{bg_gray(4)}{FG['white']}{pad(fmt_piece(int(val), cell_strings), cw, 'center')}{RESET}"

        # Empty cell
        return f"{bg_gray(2)}{FG['gray']}{pad('·', cw, 'center')}{RESET}"

    # Build the ASCII board
    lines: List[str] = ["", f"  {BOLD}Board{RESET}", f"  {'─' * 5}"]
    if sel_row:
        lines.append(f"  {FG['cyan']}Selected:{RESET} {get_move_desc(sel_row, cell_strings)}")
    lines.append("")

    row_sep = "─" * cw
    top = f"  ┌{('┬'.join([row_sep] * n_cols))}┐"
    mid = f"  ├{('┼'.join([row_sep] * n_cols))}┤"
    bot = f"  └{('┴'.join([row_sep] * n_cols))}┘"

    lines.append(top)
    for r in range(n_rows):
        lines.append(f"  │{'│'.join(_cell(r, c) for c in range(n_cols))}│")
        if r < n_rows - 1:
            lines.append(mid)
    lines.append(bot)
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry: render_debug
# ═══════════════════════════════════════════════════════════════════════════════

def render_debug(
    board: np.ndarray,
    debug_rows: List[Dict[str, Any]],
    cell_strings: Dict[int, str],
    show_board: bool = True,
) -> str:
    """
    Render the full move analysis output (tables + optional board).
    - board: current board ndarray
    - debug_rows: list of move dicts (keys: diff, move, is_selected, anchor_id/score/etc)
    - cell_strings: mapping from cell values to display strings
    """
    if not debug_rows:
        print("No moves to analyze")
        return ""

    # Group rows by anchor id
    groups: Dict[Any, List[Dict]] = defaultdict(list)
    for row in debug_rows:
        groups[row.get("anchor_id")].append(row)

    # Sort anchor groups by anchor_score (best first)
    sorted_groups = sorted(groups.values(), key=lambda g: g[0].get("anchor_score", 0), reverse=True)

    out_lines: List[str] = ["", f"  {BOLD}MOVE ANALYSIS{RESET}", f"  {'─' * 13}"]
    for rank, group in enumerate(sorted_groups):
        out_lines.append("")
        out_lines.extend(render_table(group, rank, cell_strings))

    out_lines.append("")
    out_lines.append(f"  {DIM}Solo = this move │ Pool = anchor cluster │ ◀ = selected{RESET}")

    if show_board:
        out_lines.extend(render_board(board, debug_rows, cell_strings))

    out_lines.append("")
    output = "\n".join(out_lines)
    print(output)
    return output