"""
Terminal debug visualizer for move analysis.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re

import numpy as np


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

GREEN = "\033[38;5;28m"
RED = "\033[38;5;124m"
YELLOW = "\033[38;5;142m"
ORANGE = "\033[38;5;166m"
GRAY = "\033[38;5;245m"
WHITE = "\033[38;5;255m"
BLACK = "\033[38;5;233m"

BG_SELECT = "\033[48;5;22m"


def bg_gray(level: int) -> str:
    return f"\033[48;5;{232 + max(0, min(23, level))}m"


def score_color(score: float) -> str:
    if score >= 0.65:
        return GREEN
    if score >= 0.45:
        return YELLOW
    if score >= 0.35:
        return ORANGE
    return RED


def strip_ansi(s: str) -> str:
    """Remove ANSI escape codes to get visible text."""
    return re.sub(r'\033\[[0-9;]*m', '', s)


def visible_len(s: str) -> int:
    """Get visible length of string (excluding ANSI codes)."""
    return len(strip_ansi(s))


# ---------------------------------------------------------------------------
# Box Drawing  
# ---------------------------------------------------------------------------

H, V = "─", "│"
TL, TR, BL, BR = "┌", "┐", "└", "┘"
LT, RT, TT, BT, X = "├", "┤", "┬", "┴", "┼"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class MoveInfo:
    move_str: str
    is_selected: bool
    w: int
    t: int
    l: int
    n: int
    direct_score: float
    anchor_score: float

    @property
    def w_pct(self) -> float:
        return self.w / self.n * 100 if self.n else 0

    @property
    def l_pct(self) -> float:
        return self.l / self.n * 100 if self.n else 0

    @property
    def t_pct(self) -> float:
        return self.t / self.n * 100 if self.n else 0


@dataclass
class AnchorInfo:
    anchor_id: Optional[int]
    w: int
    t: int
    l: int
    n: int
    score: float
    moves: List[MoveInfo]

    @property
    def w_pct(self) -> float:
        return self.w / self.n * 100 if self.n else 0

    @property
    def l_pct(self) -> float:
        return self.l / self.n * 100 if self.n else 0

    @property
    def t_pct(self) -> float:
        return self.t / self.n * 100 if self.n else 0


def parse_move(diff: List) -> str:
    """Extract move string from board diff.
    
    For TTT: Single cell changes, returns "r,c"
    For Chess: Two cells change (from→to), returns "r,c→r,c"
    For Captures: from_cell loses piece, to_cell gains piece (had enemy before)
    """
    if not diff:
        return "?"
    
    from_cell = None
    to_cell = None
    captured = None
    
    for idx, before, after in diff:
        # Piece left this cell (becomes empty)
        if before not in (None, 0) and after in (None, 0):
            from_cell = idx
        
        # Piece arrived at this cell (was empty OR had enemy = capture)
        if after not in (None, 0) and before != after:
            to_cell = idx
            if before not in (None, 0):
                # Capture! Record what was taken
                captured = before
    
    # Chess move with capture
    if from_cell is not None and to_cell is not None and captured is not None:
        return f"{from_cell[0]},{from_cell[1]}→{to_cell[0]},{to_cell[1]}x"
    
    # Chess move (no capture)
    if from_cell is not None and to_cell is not None:
        return f"{from_cell[0]},{from_cell[1]}→{to_cell[0]},{to_cell[1]}"
    
    # TTT or simple placement
    if to_cell is not None:
        return f"{to_cell[0]},{to_cell[1]}"
    
    # Fallback
    if diff:
        return f"{diff[0][0][0]},{diff[0][0][1]}"
    return "?"


def parse_rows(rows: List[Dict]) -> AnchorInfo:
    moves = []
    for r in rows:
        moves.append(MoveInfo(
            move_str=parse_move(r.get("diff", [])),
            is_selected=r.get("is_selected", False),
            w=r.get("direct_W", 0),
            t=r.get("direct_T", 0),
            l=r.get("direct_L", 0),
            n=r.get("direct_total", 0),
            direct_score=r.get("direct_score", 0.0),
            anchor_score=r.get("anchor_score", 0.0),
        ))
    moves.sort(key=lambda m: m.direct_score, reverse=True)
    f = rows[0]
    return AnchorInfo(
        anchor_id=f.get("anchor_id"),
        w=f.get("anchor_W", 0),
        t=f.get("anchor_T", 0),
        l=f.get("anchor_L", 0),
        n=f.get("anchor_total", 0),
        score=f.get("anchor_score", 0.0),
        moves=moves,
    )


# ---------------------------------------------------------------------------
# Table Rendering
# ---------------------------------------------------------------------------

WIDTH = 70  # Inner content width (between │ and │)


def hline(left: str, right: str, style: str = "") -> str:
    """Horizontal border line."""
    line = f"{left}{H * WIDTH}{right}"
    return f"{style}{line}{RESET}" if style else line


def make_row(content: str, style: str = "") -> str:
    """Create a row, padding content to WIDTH."""
    pad = WIDTH - visible_len(content)
    inner = content + " " * max(0, pad)
    if style:
        return f"{style}{V}{RESET}{inner}{style}{V}{RESET}"
    return f"{V}{inner}{V}"


def bar(w_pct: float, l_pct: float, t_pct: float, size: int = 10) -> str:
    """Colored WLT distribution bar."""
    w = round(w_pct / 100 * size)
    l = round(l_pct / 100 * size)
    t = size - w - l
    return f"{GREEN}{'█'*w}{RED}{'█'*l}{YELLOW}{'█'*t}{RESET}"


def render_anchor(a: AnchorInfo, rank: int) -> List[str]:
    """Render one anchor group."""
    out = []
    best = rank == 0
    n = len(a.moves)
    
    style = f"{BOLD}{GREEN}" if best else ""

    out.append(hline(TL, TR, style))

    # === Title row ===
    if best:
        marker = f"{GREEN}* BEST{RESET}"
    else:
        marker = f"{GRAY}#{rank+1}{RESET}"
    
    eq = f" ({n} equiv)" if n > 1 else ""
    left = f" {marker}  Anchor {a.anchor_id}{eq}"
    
    sc = f"{score_color(a.score)}{BOLD}{a.score:.3f}{RESET}"
    right = f"Score: {sc} "
    
    # Calculate gap using visible lengths
    gap = WIDTH - visible_len(left) - visible_len(right)
    title = f"{left}{' ' * gap}{right}"
    out.append(make_row(title, style))

    # === Stats row ===
    stats = (
        f" n={a.n:<5}  "
        f"{GREEN}W:{a.w_pct:5.1f}%{RESET}  "
        f"{RED}L:{a.l_pct:5.1f}%{RESET}  "
        f"{YELLOW}T:{a.t_pct:5.1f}%{RESET}  "
        f"{bar(a.w_pct, a.l_pct, a.t_pct)}"
    )
    out.append(make_row(stats, style))

    out.append(hline(LT, RT, style))

    # === Header row ===
    hdr = f"{DIM} {'Move':<9} {V} {'W%':>5}  {'L%':>5}  {'T%':>5}  {'(n)':>4} {V} {'Solo':>5} {V} {'Pool':>5} {V}    {RESET}"
    out.append(make_row(hdr, style))

    out.append(hline(LT, RT, style))

    # === Data rows ===
    for i, m in enumerate(a.moves):
        rk = f"{YELLOW}1st{RESET}" if i == 0 else f"{GRAY}#{i+1:<2}{RESET}"
        sel = f"{GREEN}◀{RESET}" if m.is_selected else " "

        row = (
            f" {m.move_str:<9} {V} "
            f"{GREEN}{m.w_pct:5.1f}{RESET}  "
            f"{RED}{m.l_pct:5.1f}{RESET}  "
            f"{YELLOW}{m.t_pct:5.1f}{RESET}  "
            f"{DIM}{m.n:4}{RESET} {V} "
            f"{score_color(m.direct_score)}{m.direct_score:5.3f}{RESET} {V} "
            f"{score_color(m.anchor_score)}{m.anchor_score:5.3f}{RESET} {V} "
            f"{rk} {sel}"
        )
        out.append(make_row(row, style))

    out.append(hline(BL, BR, style))

    return out


def _piece_symbol(v, is_chess: bool = False) -> str:
    """Convert piece value to display symbol.
    
    Args:
        v: Piece value from board
        is_chess: If True, use chess piece encoding
    """
    if isinstance(v, (int, np.integer)):
        if not is_chess:
            # TTT: 1=X, 2=O
            if v == 1:
                return "X"
            if v == 2:
                return "O"
        # Chess int8: positive=P1, negative=P2
        # 1=Pawn, 2=Castle, 3=King, 4=Queen
        piece_names = {1: "P", 2: "C", 3: "K", 4: "Q"}
        player = "1" if v > 0 else "2"
        piece_type = abs(v)
        if piece_type in piece_names:
            return f"{piece_names[piece_type]}{player}"
        return str(v)
    # Legacy object arrays (string pieces like "K1")
    return str(v)[:2]


def render_board(board: np.ndarray, rows: List[Dict]) -> List[str]:
    """Render board with scores overlaid."""
    out = []
    arr = np.array(board)
    if arr.ndim == 1:
        s = int(np.sqrt(arr.size))
        arr = arr.reshape((s, s))

    h, w = arr.shape
    cw = 7
    
    # Detect chess by board dimensions or presence of negative values
    is_chess = (h != w) or (h > 3) or np.any(arr < 0)

    scores = {}
    sel = None
    for r in rows:
        for idx, bef, aft in r.get("diff", []):
            if (bef is None or bef == 0) and aft not in (None, 0):
                scores[tuple(idx)] = r.get("anchor_score", r.get("direct_score", 0))
                if r.get("is_selected"):
                    sel = tuple(idx)

    out.append("")
    out.append(f"  {BOLD}Board{RESET}")
    out.append("")
    out.append("  " + TL + (H * cw + TT) * (w - 1) + H * cw + TR)

    for r in range(h):
        cells = []
        for c in range(w):
            pos = (r, c)
            v = arr[r, c]

            if pos == sel:
                cells.append(f"{BG_SELECT}{WHITE}{BOLD}{'SEL':^{cw}}{RESET}")
            elif pos in scores:
                sc = scores[pos]
                lv = int(4 + sc * 15)
                fg = WHITE if lv < 12 else BLACK
                cells.append(f"{bg_gray(lv)}{fg}{sc:^{cw}.3f}{RESET}")
            elif v != 0 and v is not None:
                sym = _piece_symbol(v, is_chess)
                cells.append(f"{bg_gray(4)}{WHITE}{sym:^{cw}}{RESET}")
            else:
                cells.append(f"{bg_gray(2)}{GRAY}{'·':^{cw}}{RESET}")

        out.append("  " + V + V.join(cells) + V)
        if r < h - 1:
            out.append("  " + LT + (H * cw + X) * (w - 1) + H * cw + RT)

    out.append("  " + BL + (H * cw + BT) * (w - 1) + H * cw + BR)
    return out


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def render_debug(
    board: np.ndarray,
    debug_rows: List[Dict[str, Any]],
    show_board: bool = True,
) -> str:
    """Render move analysis grouped by anchor."""
    if not debug_rows:
        print("No moves to analyze")
        return ""

    lines = ["", f"  {BOLD}MOVE ANALYSIS{RESET}", f"  {H * 13}"]

    groups: Dict[Any, List[Dict]] = defaultdict(list)
    for r in debug_rows:
        groups[r.get("anchor_id")].append(r)

    anchors = [parse_rows(g) for g in groups.values()]
    anchors.sort(key=lambda x: x.score, reverse=True)

    for i, a in enumerate(anchors):
        lines.append("")
        lines.extend(render_anchor(a, i))

    lines.append("")
    lines.append(f"  {DIM}Solo = move only │ Pool = anchor cluster │ ◀ = selected{RESET}")

    if show_board:
        lines.extend(render_board(board, debug_rows))

    lines.append("")

    output = "\n".join(lines)
    print(output)
    return output