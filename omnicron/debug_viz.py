"""
Terminal debug visualizer for game TRANSITION statistics.

ARCHITECTURE NOTE:
-----------------
This visualizer is TRANSITION-FIRST:
  - Primary input: transition diffs (list of (coord, before, after))
  - Legacy fallback: move_array (deprecated, for backward compatibility)

Transitions represent STATE → STATE changes.
Moves are a game-engine detail used to derive transitions.

Supports:
- Transition-based diffs (PREFERRED)
- Legacy move_array (fallback, deprecated)
- String-based pieces (MiniChess, etc.)
"""

from typing import List, Dict, Any
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
    if piece_val is None or piece_val == 0:
        return "·"

    if isinstance(piece_val, str):
        return piece_val

    try:
        val = int(piece_val)
        if val == 1:
            return "X"
        if val == 2 or val == -1:
            return "O"
    except (ValueError, TypeError):
        pass

    return str(piece_val)


def _format_move(move) -> str:
    """
    Format a move for display.

    Supports:
    - NumPy arrays: [r, c] or [from_r, from_c, to_r, to_c]
    - Tuples/Lists: (r, c)
    - Integers: flat index
    - Strings: pass through
    - None: "?"

    Examples:
        [1, 2] → "(1,2)"
        [0, 1, 2, 3] → "(0,1)→(2,3)"
        42 → "#42"
    """
    if move is None:
        return "?"

    # Convert NumPy array to list
    if isinstance(move, np.ndarray):
        move = move.tolist()

    # List or tuple
    if isinstance(move, (list, tuple)):
        if len(move) == 2:
            return f"({move[0]},{move[1]})"  # Single position: (1,2)
        elif len(move) == 4:
            return f"({move[0]},{move[1]})→({move[2]},{move[3]})"  # From→To
        elif len(move) == 1:
            return f"#{move[0]}"  # Single value in array
        else:
            # Fallback for unusual lengths
            return str(move)

    # Integer (flat index)
    if isinstance(move, (int, np.integer)):
        return f"#{int(move)}"

    # String or other
    return str(move)


# ===========================================================
# Destination extraction from TRANSITION DIFF (primary method)
# ===========================================================


def _dest_from_diff(diff, cols: int):
    """
    Extract destination square from transition diff.

    Args:
        diff: List[(idx, before, after)] where idx is (row, col)

    Returns:
        (row, col) of destination square, or None

    The destination is defined as the square where 'after' is non-empty.
    For multi-square moves (e.g., chess castling), this returns the first
    destination found, which is sufficient for visualization purposes.
    """
    for idx, before, after in diff:
        if after is not None and after != 0:
            r, c = idx
            return r, c
    return None


# ===========================================================
# LEGACY: destination extraction from move_array (deprecated)
# ===========================================================


def _dest_from_move_array(move_array, cols: int):
    """
    DEPRECATED: Extract destination from legacy move_array format.

    This function exists only for backward compatibility.
    New code should pass transition diffs instead.

    Supported move_array formats:
      - [from_r, from_c, to_r, to_c]  → returns (to_r, to_c)
      - [to_r, to_c]                  → returns (to_r, to_c)
      - [flat_index]                  → returns divmod(flat_index, cols)
    """
    mv = np.atleast_1d(move_array)
    if len(mv) == 4:
        return int(mv[2]), int(mv[3])
    elif len(mv) == 2:
        return int(mv[0]), int(mv[1])
    elif len(mv) == 1:
        return divmod(int(mv[0]), cols)
    return None


# ===========================================================
# Main render function
# ===========================================================


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
    """
    Render transition statistics as a terminal heatmap.

    Args:
        board: Current game board (np.ndarray)
        debug_rows: List of transition statistics, where each dict contains:
          - diff: List[(idx, before, after)]  [PREFERRED]
          - move_array: np.ndarray            [DEPRECATED FALLBACK]
          - score, utility, certainty, etc.   [statistics]
          - is_selected: bool                 [highlight flag]

        primary_metric: Metric to use for heatmap colors (default: "score")
        secondary_metric: Second metric to display (default: "certainty")
        show_bars: Whether to show visual bars in cells
        cell_width: Width of each cell in characters
        return_str: If True, return string instead of printing

    Returns:
        Formatted string (always), and prints unless return_str=True

    ARCHITECTURE NOTE:
    -----------------
    This function is TRANSITION-FIRST. Each debug_row represents:
      STATE_A → STATE_B

    The diff shows what changed between states.
    The move_array is a legacy fallback and should not be used in new code.
    """
    board_arr = np.array(board)

    if board_arr.ndim == 1:
        side = int(np.sqrt(board_arr.size))
        board_arr = (
            board_arr.reshape((side, side))
            if side * side == board_arr.size
            else board_arr.reshape((1, -1))
        )

    ROWS, COLS = board_arr.shape

    primary_mat = np.full((ROWS, COLS), np.nan)
    secondary_mat = np.full((ROWS, COLS), np.nan)
    is_selected_mat = np.zeros((ROWS, COLS), dtype=bool)

    # =======================================================
    # Populate matrices (TRANSITION-FIRST)
    # =======================================================

    for d in debug_rows:
        r = c = None

        # PRIMARY: transition diff
        if "diff" in d and d["diff"]:
            dest = _dest_from_diff(d["diff"], COLS)
            if dest:
                r, c = dest

        # FALLBACK: legacy move_array (deprecated)
        elif "move_array" in d:
            dest = _dest_from_move_array(d["move_array"], COLS)
            if dest:
                r, c = dest

        if r is None or c is None:
            continue

        if 0 <= r < ROWS and 0 <= c < COLS:
            primary_mat[r, c] = float(d.get(primary_metric, np.nan))
            secondary_mat[r, c] = float(d.get(secondary_metric, np.nan))
            is_selected_mat[r, c] = bool(d.get("is_selected", False))

    # =======================================================
    # Output assembly
    # =======================================================

    output = [
        "",
        f"{BOLD}╔" + "═" * 78 + "╗",
        f"║{'TRANSITION ANALYSIS SUMMARY':^78}║",
        f"╚" + "═" * 78 + "╝",
        "",
    ]

    header = f"{'Transition':^{W_MOVE}} │ {'Rates (W/T/N/L)':^{W_STAT*4+3}} │ {'Metrics':^{W_METRIC*3+W_TOTAL+2}}"
    output.append(header)
    output.append(f"{DIM}{'─' * len(header)}{RESET}")

    sorted_rows = sorted(debug_rows, key=lambda x: x.get("score", 0), reverse=True)

    for d in sorted_rows:
        # Format the move (or use delta symbol if no move available)
        label = _format_move(d.get("move", None)) if "move" in d else "Δ"

        row_str = (
            f"{label:>{W_MOVE}} │ "
            f"{d.get('pW',0):.2f} {d.get('pT',0):.2f} {d.get('pN',0):.2f} {d.get('pL',0):.2f} │ "
            f"C:{d.get('certainty',0):.2f} "
            f"U:{d.get('utility',0):+.2f} "
            f"S:{d.get('score',0):.2f} "
            f"({d.get('total',0)})"
        )
        if d.get("is_selected"):
            row_str += f" {BOLD}→ SELECTED{RESET}"
        output.append(row_str)

    # =======================================================
    # Heatmap
    # =======================================================

    output.append("\n" + f"{BOLD}╔" + "═" * 78 + "╗")
    output.append(f"║{'HEATMAP VISUALIZER (Destination Squares)':^78}║")
    output.append(f"╚" + "═" * 78 + "╝\n")

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

            t_txt = f" {marker} ".center(cell_width)
            m_txt = f"{p_s} {s_s}".center(cell_width)
            b_txt = (
                (" " + _bar_string(p_val, bar_w) + " ").ljust(cell_width)
                if show_bars
                else " " * cell_width
            )

            if sel:
                t_txt, m_txt, b_txt = (
                    "▌" + t_txt[1:-1] + "▐",
                    "▌" + m_txt[1:-1] + "▐",
                    "▌" + b_txt[1:-1] + "▐",
                )

            bg = BEST_BG if sel else _bg_color_for_value(p_val)
            fg = BEST_FG if sel else WHITE_FG

            for i, txt in enumerate([t_txt, m_txt, b_txt]):
                lines[i].append(f"{bg}{fg}{txt}{RESET}")

        for line in lines:
            output.append(V + V.join(line) + V)

        if r < ROWS - 1:
            output.append(make_sep(ML, MM, MR))
        else:
            output.append(make_sep(BL, BM, BR))

    final_str = "\n".join(output)
    if not return_str:
        print(final_str)
    return final_str
