"""
Terminal debug visualizer for game move statistics.
Updated to support string-based piece names (e.g., 'Q1', 'P2') for MiniChess.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

# ===========================================================
# ANSI COLOR CODES
# ===========================================================
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
WHITE_FG = "\033[38;5;255m"
GRAY_FG = "\033[38;5;244m"
BEST_BG = "\033[48;5;21m" 
BEST_FG = "\033[38;5;255m"

# ===========================================================
# LAYOUT CONFIGURATION
# ===========================================================
W_MOVE = 12
W_STAT = 6
W_METRIC = 8
W_TOTAL = 6

def _bg_color_for_value(val: float) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "\033[48;5;236m"
    v = float(max(0.0, min(1.0, val)))
    if v < 0.33:
        ratio = v / 0.33
        idx = int(22 + ratio * (220 - 22))
    elif v < 0.67:
        ratio = (v - 0.33) / 0.34
        idx = int(220 + ratio * (208 - 220))
    else:
        ratio = (v - 0.67) / 0.33
        idx = int(208 + ratio * (196 - 208))
    return f"\033[48;5;{idx}m"

def _bar_string(value: float, width: int) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return " " * width
    v = float(max(0.0, min(1.0, value)))
    filled = int(round(v * width))
    return "█" * filled + " " * (width - filled)

def _player_marker(piece_val) -> str:
    """
    Improved marker logic:
    - Supports MiniChess strings: 'K1', 'Q2', etc.
    - Supports numeric players: 1, -1, 0.
    """
    if piece_val is None or piece_val == 0:
        return "·"
    
    # If it's a string (like 'Q1' or 'P2'), return it as is
    if isinstance(piece_val, str):
        return piece_val
        
    # Handle numeric fallbacks
    try:
        val = int(piece_val)
        if val == 1: return "X"
        if val == 2 or val == -1: return "O"
    except (ValueError, TypeError):
        pass
        
    return str(piece_val)

def render_debug(
    board: np.ndarray,
    debug_rows: List[Dict[str, Any]],
    *,
    primary_metric: str = "score",
    secondary_metric: str = "certainty",
    show_bars: bool = True,
    cell_width: int = 14,
    return_str: bool = False,
) -> str:
    board_arr = np.array(board)
    
    # Auto-reshape 1D to square if needed
    if board_arr.ndim == 1:
        side = int(np.sqrt(board_arr.size))
        board_arr = board_arr.reshape((side, side)) if side*side == board_arr.size else board_arr.reshape((1, -1))
    
    ROWS, COLS = board_arr.shape

    # Prepare data matrices
    primary_mat = np.full((ROWS, COLS), np.nan)
    secondary_mat = np.full((ROWS, COLS), np.nan)
    is_selected_mat = np.zeros((ROWS, COLS), dtype=bool)

    for d in debug_rows:
        mv = d.get("move_array")
        if mv is None: continue
        
        # In Chess [from_r, from_c, to_r, to_c], 'move_array' usually refers to 'to' coords 
        # or the signature location. Here we assume index 2,3 for destination if it's length 4.
        arr_mv = np.atleast_1d(mv)
        if len(arr_mv) == 4:
            r, c = int(arr_mv[2]), int(arr_mv[3])
        elif len(arr_mv) == 2:
            r, c = int(arr_mv[0]), int(arr_mv[1])
        else:
            idx = int(arr_mv[0])
            r, c = divmod(idx, COLS)

        if 0 <= r < ROWS and 0 <= c < COLS:
            primary_mat[r, c] = float(d.get(primary_metric, np.nan))
            secondary_mat[r, c] = float(d.get(secondary_metric, np.nan))
            is_selected_mat[r, c] = bool(d.get("is_selected", False))

    output = ["", f"{BOLD}╔" + "═"*78 + "╗", f"║{'MOVE ANALYSIS SUMMARY':^78}║", f"╚" + "═"*78 + "╝", ""]
    
    # Table Header
    header = f"{'Move':^{W_MOVE}} │ {'Rates (W/T/N/L)':^{W_STAT*4+3}} │ {'Metrics':^{W_METRIC*3+W_TOTAL+2}}"
    output.append(header)
    output.append(f"{DIM}{'─' * len(header)}{RESET}")

    # Table Data
    sorted_rows = sorted(debug_rows, key=lambda x: x.get("score", 0), reverse=True)
    for d in sorted_rows:
        mv = d.get("move_array")
        mv_str = f"[{','.join(map(str, mv))}]" if hasattr(mv, '__iter__') else str(mv)
        row_str = (f"{mv_str:>{W_MOVE}} │ "
                   f"{d.get('pW',0):.2f} {d.get('pT',0):.2f} {d.get('pN',0):.2f} {d.get('pL',0):.2f} │ "
                   f"C:{d.get('certainty',0):.2f} U:{d.get('utility',0):+.2f} S:{d.get('score',0):.2f} "
                   f"({d.get('total',0)})")
        if d.get("is_selected"): row_str += f" {BOLD}→ SELECTED{RESET}"
        output.append(row_str)

    # Heatmap
    output.append("\n" + f"{BOLD}╔" + "═"*78 + "╗")
    output.append(f"║{'HEATMAP VISUALIZER (Destination Squares)':^78}║")
    output.append(f"╚" + "═"*78 + "╝\n")

    TL, TM, TR = "┌", "┬", "┐"
    ML, MM, MR = "├", "┼", "┤"
    BL, BM, BR = "└", "┴", "┘"
    H, V = "─", "│"

    def make_sep(l, m, r):
        return l + m.join([H * cell_width] * COLS) + r

    output.append(make_sep(TL, TM, TR))
    bar_w = max(1, cell_width - 4)

    for r in range(ROWS):
        lines = [[], [], []]
        for c in range(COLS):
            p_val = primary_mat[r, c]
            s_val = secondary_mat[r, c]
            sel = is_selected_mat[r, c]
            
            marker = _player_marker(board_arr[r, c])
            p_s = f"{p_val:.2f}" if not np.isnan(p_val) else "--"
            s_s = f"{s_val:.2f}" if not np.isnan(s_val) else "--"
            
            # Formatting text to fit cell_width
            t_txt = f" {marker} ".center(cell_width)
            m_txt = f"{p_s} {s_s}".center(cell_width)
            b_txt = (" " + _bar_string(p_val, bar_w) + " ").ljust(cell_width) if show_bars else " "*cell_width

            if sel:
                t_txt, m_txt, b_txt = "▌"+t_txt[1:-1]+"▐", "▌"+m_txt[1:-1]+"▐", "▌"+b_txt[1:-1]+"▐"

            bg = BEST_BG if sel else _bg_color_for_value(p_val)
            fg = BEST_FG if sel else WHITE_FG
            
            for i, txt in enumerate([t_txt, m_txt, b_txt]):
                lines[i].append(f"{bg}{fg}{txt}{RESET}")

        for line in lines:
            output.append(V + V.join(line) + V)
        
        if r < ROWS - 1: output.append(make_sep(ML, MM, MR))
        else: output.append(make_sep(BL, BM, BR))

    final_str = "\n".join(output)
    if not return_str: print(final_str)
    return final_str