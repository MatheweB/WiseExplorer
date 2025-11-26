"""
Terminal debug visualizer for game move statistics (parameterless).

Primary metric: certainty (max of Win/Tie/Neutral/Loss probability)
Secondary metric: utility (Win=+1, Tie=+1, Neutral=+1, Loss=-1) for parameterless setup.

This renderer now:
 - Shows opponent best reply (coords + util + cert) per candidate
 - Marks 'dangerous' moves (strict threshold: opponent_best_util >= 1.0) in red
 - Prints a wider Move Analysis Summary with Opp info
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

# --- Column Width Definitions (Used for alignment) ---
WIDTH_MOVE = 10
WIDTH_STAT = 5
WIDTH_CERT_UTIL = 6
WIDTH_TOTAL = 6
WIDTH_OPP_MOVE = 14
WIDTH_OPP_UTIL = 6
WIDTH_OPP_CERT = 6
WIDTH_NOTES_MIN = 20 # Arbitrary large value for the right-most, non-fixed column

# Total calculated width for the main summary table
TOTAL_WIDTH = (
    1  # Initial space
    + WIDTH_MOVE
    + 3  # Separator ' │ '
    + (WIDTH_STAT * 4) + 3 # pW+pT+pN+pL + 3 spaces
    + 3  # Separator ' │ '
    + (WIDTH_CERT_UTIL * 3) + 3 # Cert/Util/Adj + 3 spaces
    + WIDTH_TOTAL
    + 3  # Separator ' │ '
    + WIDTH_OPP_MOVE + 1 + WIDTH_OPP_UTIL + 1 + WIDTH_OPP_CERT # OPP_REPLY/OPP_U/OPP_C + 2 spaces
    + 3  # Separator ' │ '
    + WIDTH_NOTES_MIN # Note column
)

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
    if board_arr.ndim != 2 or board_arr.shape[0] != board_arr.shape[1]:
        raise ValueError("board must be a square 2D array")
    N = board_arr.shape[0]

    primary_mat = np.full((N, N), np.nan, dtype=float)
    secondary_mat = np.full((N, N), np.nan, dtype=float)
    is_best_mat = np.zeros((N, N), dtype=bool)
    danger_mat = np.zeros((N, N), dtype=bool)

    # Map move -> debug entry for easier lookups (optional)
    entries_by_cell: Dict[tuple, Dict[str, Any]] = {}

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
        entries_by_cell[(r, c)] = d

    def mark_char(v):
        try:
            if int(v) == 1:
                return "X"
            if int(v) == -1:
                return "O"
            return "·"
        except Exception:
            return "·"

    out_lines: List[str] = []
    out_lines.append("")
    out_lines.append(f"{BOLD}Move Analysis Summary{RESET}")
    # Use TOTAL_WIDTH for separator alignment
    out_lines.append(f"{DIM}{'=' * (TOTAL_WIDTH - 2)}{RESET}") 
    
    # ------------------------------------------------------------------
    # Header construction using defined widths for guaranteed alignment
    # ------------------------------------------------------------------
    header = (
        f" {'Move':<{WIDTH_MOVE}s} │ "
        f"{'pW':<{WIDTH_STAT}s} {'pT':<{WIDTH_STAT}s} {'pN':<{WIDTH_STAT}s} {'pL':<{WIDTH_STAT}s} │ "
        f"{'Cert':<{WIDTH_CERT_UTIL}s} {'Util':<{WIDTH_CERT_UTIL}s} {'Adj':<{WIDTH_CERT_UTIL}s} {'Tot':<{WIDTH_TOTAL}s} │ "
        f"{'OPP_REPLY':<{WIDTH_OPP_MOVE}s} {'OPP_U':<{WIDTH_OPP_UTIL}s} {'OPP_C':<{WIDTH_OPP_CERT}s} │ Note"
    )
    out_lines.append(header)
    out_lines.append(f"{DIM}{'-' * (TOTAL_WIDTH - 2)}{RESET}") # Use TOTAL_WIDTH for separator alignment

    sorted_rows = sorted(debug_rows, key=lambda d: d.get("sort_key", (0, 0, 0)), reverse=True)
    
    # ------------------------------------------------------------------
    # Row printing with robust padding for optional fields
    # ------------------------------------------------------------------
    for d in sorted_rows:
        # Move
        mv = d.get("move_array")
        mv_s = f"[{mv[0]}, {mv[1]}]" if mv is not None else "[]"
        
        # Player Stats
        pW = d.get("pW", 0.0)
        pT = d.get("pT", 0.0)
        pN = d.get("pN", 0.0)
        pL = d.get("pL", 0.0)
        cert = d.get("certainty", 0.0)
        util = d.get("utility", 0.0)
        adj = d.get("adjusted_utility", 0.0)
        tot = d.get("total", 0)

        # Opponent Move
        opp_move = d.get("opponent_best_move")
        opp_move_s = f"[{opp_move[0]}, {opp_move[1]}]" if opp_move else "--"
        
        # Opponent Stats (using the correct key and robust padding)
        opp_u = d.get("opponent_best_util_for_them") # Fixed key lookup
        opp_c = d.get("opponent_best_cert")
        
        # Create padded string representations for the optional fields
        # If None, use a string of spaces equal to the WIDTH_OPP_UTIL/CERT to maintain alignment
        opp_u_str = f"{opp_u:0.3f}" if opp_u is not None else " " * WIDTH_OPP_UTIL
        opp_c_str = f"{opp_c:0.3f}" if opp_c is not None else " " * WIDTH_OPP_CERT
        
        # Flags/Notes (Final Fix: Use join to eliminate spurious spaces)
        opp_flag = "✅" if d.get("opponent_data_exists") else ""
        adj_flag = "✅" if d.get("adjusted") else ""
        danger_mark = "🔥" if d.get("dangerous") else ""
        best = "★ BEST" if d.get("is_best") else ""

        # Join only non-empty items to create a perfectly spaced notes string
        notes_list = [opp_flag, adj_flag, danger_mark, best]
        notes = " ".join(item for item in notes_list if item)
        
        out_lines.append(
            f" {mv_s:<{WIDTH_MOVE}s} │ "
            f"{pW:0.3f} {pT:0.3f} {pN:0.3f} {pL:0.3f} │ "
            f"{cert:0.3f} {util:0.3f} {adj:0.3f} {tot:{WIDTH_TOTAL}d} │ "
            f"{opp_move_s:<{WIDTH_OPP_MOVE}s} {opp_u_str:<{WIDTH_OPP_UTIL}s} {opp_c_str:<{WIDTH_OPP_CERT}s} │ {notes}"
        )

    out_lines.append("")
    out_lines.append(f"{BOLD}Heatmap Visualizer{RESET}")
    out_lines.append("")

    TL, TM, TR = "┌", "┬", "┐"
    ML, MM, MR = "├", "┼", "┤"
    BL, BM, BR = "└", "┴", "┘"
    H, V = "─", "│"

    def make_sep(left, mid, right):
        seg = H * cell_width
        return left + mid.join([seg] * N) + right

    out_lines.append(make_sep(TL, TM, TR))
    bar_width = max(1, cell_width - 4)

    for r in range(N):
        top_line_cells = []
        mid_line_cells = []
        bot_line_cells = []

        for c in range(N):
            p_val = primary_mat[r, c]
            s_val = secondary_mat[r, c]
            is_best = bool(is_best_mat[r, c])
            is_danger = bool(danger_mat[r, c])
            m = mark_char(board_arr[r, c])

            # Top: mark
            inner_top = f" {m} ".center(cell_width)
            # Mid: primary + secondary
            p_num = f"{p_val:.2f}" if not np.isnan(p_val) else "--"
            s_num = f"{s_val:.2f}" if not np.isnan(s_val) else "--"
            inner_mid = f"{p_num} {s_num}".center(cell_width)
            # Bottom: bar
            if show_bars and not np.isnan(p_val):
                bar = _barstr(float(p_val), bar_width)
                inner_bot = (" " + bar + " ").ljust(cell_width)
            else:
                inner_bot = " " * cell_width

            if is_best and cell_width >= 3:
                inner_top = "▌" + inner_top[1:-1] + "▐"
                inner_mid = "▌" + inner_mid[1:-1] + "▐"
                inner_bot = "▌" + inner_bot[1:-1] + "▐"

            top_line_cells.append(inner_top)
            mid_line_cells.append(inner_mid)
            bot_line_cells.append(inner_bot)

        # Colorize and join
        def colorize(c_idx, line_cells):
            p_val = primary_mat[r, c_idx]
            if danger_mat[r, c_idx]:
                # red background for dangerous
                bg, fg = "\033[48;5;196m", WHITE_FG
            elif is_best_mat[r, c_idx]:
                bg, fg = BEST_BG, BEST_FG
            else:
                bg, fg = _bg_color_for(p_val), _fg_for(p_val)
            return f"{bg}{fg}{line_cells[c_idx]}{RESET}"

        top_join = V + V.join([colorize(ci, top_line_cells) for ci in range(N)]) + V
        mid_join = V + V.join([colorize(ci, mid_line_cells) for ci in range(N)]) + V
        bot_join = V + V.join([colorize(ci, bot_line_cells) for ci in range(N)]) + V

        out_lines.append(top_join)
        out_lines.append(mid_join)
        out_lines.append(bot_join)

        # Row separator
        out_lines.append(make_sep(ML, MM, MR) if r < N - 1 else make_sep(BL, BM, BR))

    rendered = "\n".join(out_lines)
    if return_str:
        return rendered
    print(rendered)