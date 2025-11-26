"""
Terminal debug visualizer for game move statistics (parameterless).

Primary metric: certainty (max of Win/Tie/Loss probability)
Secondary metric: utility (Win=+1, Tie=0, Loss=-1)
"""

from typing import List, Dict, Any, Optional
import numpy as np

# ANSI codes / small palette
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
WHITE_FG = "\033[38;5;255m"
GRAY_FG = "\033[38;5;244m"

BEST_BG = "\033[48;5;21m"
BEST_FG = "\033[38;5;255m"

# -------------------------
# Helpers
# -------------------------
def _bg_color_for(val: Optional[float]) -> str:
    """Return ANSI background color for a value normalized 0..1. NaN -> dark gray."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "\033[48;5;236m"
    v = float(max(0.0, min(1.0, val)))
    if v < 0.5:
        ratio = v / 0.5
        idx = int(22 + ratio * (220 - 22))
    else:
        ratio = (v - 0.5) / 0.5
        idx = int(220 + ratio * (196 - 220))
    return f"\033[48;5;{idx}m"

def _fg_for(val: Optional[float]) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return GRAY_FG
    return WHITE_FG

def _barstr(value: Optional[float], width: int) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return " " * width
    v = float(max(0.0, min(1.0, value)))
    full = int(round(v * width))
    return "█" * full + " " * (width - full)

# -------------------------
# Main renderer
# -------------------------
def render_debug(board: np.ndarray,
                 debug_rows: List[Dict[str, Any]],
                 *,
                 primary_metric: str = "certainty",
                 secondary_metric: str = "utility",
                 show_bars: bool = True,
                 cell_width: int = 14,
                 return_str: bool = False) -> Optional[str]:

    board_arr = np.array(board)
    N = board_arr.shape[0]
    primary_mat = np.full((N, N), np.nan, dtype=float)
    secondary_mat = np.full((N, N), np.nan, dtype=float)
    is_best_mat = np.zeros((N, N), dtype=bool)
    danger_mat = np.zeros((N, N), dtype=bool)

    for d in debug_rows:
        try:
            r, c = int(d["move_array"][0]), int(d["move_array"][1])
        except Exception:
            continue
        if not (0 <= r < N and 0 <= c < N):
            continue
        primary_mat[r, c] = float(d.get(primary_metric, np.nan))
        secondary_mat[r, c] = float(d.get(secondary_metric, np.nan))
        is_best_mat[r, c] = bool(d.get("is_best", False))
        danger_mat[r, c] = bool(d.get("dangerous", False))

    def mark_char(v):
        try:
            if int(v) == 1: return "X"
            if int(v) == -1: return "O"
            return "·"
        except Exception:
            return "·"

    out_lines: List[str] = []
    out_lines.append("")
    out_lines.append(f"{BOLD}Move Analysis Summary{RESET}")
    out_lines.append(f"{DIM}{'=' * 95}{RESET}")
    header = f" {'Move':<10s} │ {'pW':<5s} {'pT':<5s} {'pN':<5s} {'pL':<5s} │ {'Cert':<6s} {'Util':<6s} {'Adj':<6s} {'Tot':<6s} │ Note"
    out_lines.append(header)
    out_lines.append(f"{DIM}{'-' * 95}{RESET}")

    sorted_rows = sorted(debug_rows, key=lambda d: d.get("sort_key", (0, 0, 0)), reverse=True)
    for d in sorted_rows:
        mv = d.get("move_array")
        mv_s = f"[{mv[0]}, {mv[1]}]" if mv else "[]"
        pW = d.get("pW", 0.0); pT = d.get("pT",0.0); pN = d.get("pN",0.0); pL = d.get("pL",0.0)
        cert = d.get("certainty",0.0); util = d.get("utility",0.0); adj = d.get("adjusted_utility",0.0)
        tot = d.get("total",0)
        best = "★ BEST" if d.get("is_best") else ""
        danger = "⚠" if d.get("dangerous") else ""
        out_lines.append(f" {mv_s:<10s} │ {pW:0.3f} {pT:0.3f} {pN:0.3f} {pL:0.3f} │ {cert:0.3f} {util:0.3f} {adj:0.3f} {tot:6d} │ {best} {danger}")

    out_lines.append("")
    out_lines.append(f"{BOLD}Heatmap Visualizer{RESET}\n")

    TL, TM, TR = "┌", "┬", "┐"
    ML, MM, MR = "├", "┼", "┤"
    BL, BM, BR = "└", "┴", "┘"
    H, V = "─", "│"

    def make_sep(left, mid, right):
        seg = H * cell_width
        return left + mid.join([seg]*N) + right

    out_lines.append(make_sep(TL, TM, TR))
    bar_width = max(1, cell_width - 4)

    for r in range(N):
        top_line_cells, mid_line_cells, bot_line_cells = [], [], []

        for c in range(N):
            p_val = primary_mat[r,c]; s_val = secondary_mat[r,c]
            m = mark_char(board_arr[r,c])
            is_best = is_best_mat[r,c]
            is_danger = danger_mat[r,c]

            inner_top = f" {m} ".center(cell_width)
            inner_mid = f"{p_val:.2f} {s_val:.2f}".center(cell_width) if not np.isnan(p_val) else "--".center(cell_width)
            inner_bot = (" " + _barstr(float(p_val), bar_width) + " ").ljust(cell_width) if show_bars and not np.isnan(p_val) else " "*cell_width

            if is_best and cell_width >=3:
                inner_top = "▌"+inner_top[1:-1]+"▐"
                inner_mid = "▌"+inner_mid[1:-1]+"▐"
                inner_bot = "▌"+inner_bot[1:-1]+"▐"

            top_line_cells.append(inner_top)
            mid_line_cells.append(inner_mid)
            bot_line_cells.append(inner_bot)

        def colorize(c_idx, line_cells):
            p_val = primary_mat[r, c_idx]
            if danger_mat[r,c_idx]:
                bg, fg = "\033[48;5;196m", WHITE_FG  # red for dangerous
            elif is_best_mat[r,c_idx]:
                bg, fg = BEST_BG, BEST_FG
            else:
                bg, fg = _bg_color_for(p_val), _fg_for(p_val)
            return f"{bg}{fg}{line_cells[c_idx]}{RESET}"

        top_join = V + V.join([colorize(ci, top_line_cells) for ci in range(N)]) + V
        mid_join = V + V.join([colorize(ci, mid_line_cells) for ci in range(N)]) + V
        bot_join = V + V.join([colorize(ci, bot_line_cells) for ci in range(N)]) + V

        out_lines.extend([top_join, mid_join, bot_join])
        out_lines.append(make_sep(ML, MM, MR) if r<N-1 else make_sep(BL, BM, BR))

    rendered = "\n".join(out_lines)
    if return_str: return rendered
    print(rendered)
