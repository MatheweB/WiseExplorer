"""
Terminal debug visualizer for game move statistics.

Displays:
1. Move Analysis Summary - table of all candidate moves with stats
2. Heatmap Visualizer - visual board representation with move quality
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

BEST_BG = "\033[48;5;21m"  # Blue background for best move
BEST_FG = "\033[38;5;255m"


# ===========================================================
# LAYOUT CONFIGURATION
# ===========================================================
# Column widths for the summary table
W_MOVE = 8
W_STAT = 5
W_METRIC = 6
W_TOTAL = 5
W_OPP_MOVE = 8
W_OPP_STAT = 5
W_OPP_METRIC = 6


# ===========================================================
# UTILITY FUNCTIONS
# ===========================================================
def _bg_color_for_value(val: float) -> str:
    """Return ANSI background color for a value in [0, 1]."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "\033[48;5;236m"  # Dark gray for missing data

    v = float(max(0.0, min(1.0, val)))

    # Gradient: dark (bad) -> yellow (neutral) -> red (good)
    if v < 0.5:
        ratio = v / 0.5
        idx = int(22 + ratio * (220 - 22))  # Dark gray to yellow
    else:
        ratio = (v - 0.5) / 0.5
        idx = int(220 + ratio * (196 - 220))  # Yellow to red

    return f"\033[48;5;{idx}m"


def _fg_for_value(val: float) -> str:
    """Return ANSI foreground color for a value."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return GRAY_FG
    return WHITE_FG


def _bar_string(value: float, width: int) -> str:
    """Create a visual bar representation of a value in [0, 1]."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return " " * width

    v = float(max(0.0, min(1.0, value)))
    filled = int(round(v * width))
    return "█" * filled + " " * (width - filled)


def _format_optional(val: float, fmt: str = "0.2f") -> str:
    """Format an optional float value, returning '--' if None."""
    if val is None:
        return "--"
    return f"{val:{fmt}}"


def _player_marker(player_value) -> str:
    """Convert board cell value to display character."""
    try:
        if int(player_value) == 1:
            return "X"
        elif int(player_value) == -1 or int(player_value) == 2:
            return "O"
        return "·"
    except Exception:
        return "·"


# ===========================================================
# MAIN RENDERER
# ===========================================================
def render_debug(
    board: np.ndarray,
    debug_rows: List[Dict[str, Any]],
    *,
    primary_metric: str = "certainty",
    secondary_metric: str = "utility",
    show_bars: bool = True,
    cell_width: int = 14,
    return_str: bool = False,
) -> str:
    """
    Render debug visualization for move analysis.

    Args:
        board: Current board state as 2D numpy array
        debug_rows: List of move evaluation dictionaries
        primary_metric: Primary metric for heatmap coloring
        secondary_metric: Secondary metric to display
        show_bars: Whether to show visual bars in heatmap
        cell_width: Width of each cell in heatmap
        return_str: If True, return string instead of printing

    Returns:
        Rendered string if return_str=True, otherwise None
    """
    board_arr = np.array(board)

    if board_arr.ndim != 2 or board_arr.shape[0] != board_arr.shape[1]:
        raise ValueError("board must be a square 2D array")

    N = board_arr.shape[0]

    # Build matrices for heatmap
    primary_mat = np.full((N, N), np.nan, dtype=float)
    secondary_mat = np.full((N, N), np.nan, dtype=float)
    is_best_mat = np.zeros((N, N), dtype=bool)
    is_danger_mat = np.zeros((N, N), dtype=bool)

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
        is_danger_mat[r, c] = bool(d.get("dangerous", False))

    output = []
    output.append("")
    output.append(
        f"{BOLD}╔═══════════════════════════════════════════════════════════════════╗{RESET}"
    )
    output.append(
        f"{BOLD}║                      MOVE ANALYSIS SUMMARY                        ║{RESET}"
    )
    output.append(
        f"{BOLD}╚═══════════════════════════════════════════════════════════════════╝{RESET}"
    )
    output.append("")

    # ========================================
    # SUMMARY TABLE
    # ========================================

    # Header
    header_parts = [
        f"{'Move':^{W_MOVE}}",
        "│",
        f"{'Our Stats':^{W_STAT*4+3}}",
        "│",
        f"{'Our Metrics':^{W_METRIC*3+W_TOTAL+3}}",
        "│",
        f"{'Opp Reply':^{W_OPP_MOVE}}",
        f"{'Opp Stats':^{W_OPP_STAT*4+3}}",
        f"{'Opp Metrics':^{W_OPP_METRIC*2+1}}",
        "│",
        "Notes",
    ]
    output.append(" ".join(header_parts))

    # Sub-header
    subheader_parts = [
        f"{' ':^{W_MOVE}}",
        "│",
        f"{'pW':<{W_STAT}}",
        f"{'pT':<{W_STAT}}",
        f"{'pN':<{W_STAT}}",
        f"{'pL':<{W_STAT}}",
        "│",
        f"{'Cert':<{W_METRIC}}",
        f"{'Util':<{W_METRIC}}",
        f"{'Adj':<{W_METRIC}}",
        f"{'N':<{W_TOTAL}}",
        "│",
        f"{'Move':<{W_OPP_MOVE}}",
        f"{'pW':<{W_OPP_STAT}}",
        f"{'pT':<{W_OPP_STAT}}",
        f"{'pN':<{W_OPP_STAT}}",
        f"{'pL':<{W_OPP_STAT}}",
        f"{'Util':<{W_OPP_METRIC}}",
        f"{'Cert':<{W_OPP_METRIC}}",
        "│",
    ]
    output.append(" ".join(subheader_parts))

    sep_length = sum(
        [
            W_MOVE,
            1,
            W_STAT * 4 + 3,
            1,
            W_METRIC * 3 + W_TOTAL + 3,
            1,
            W_OPP_MOVE,
            W_OPP_STAT * 4 + 3,
            W_OPP_METRIC * 2 + 1,
            1,
            20,
        ]
    )
    output.append(f"{DIM}{'─' * sep_length}{RESET}")

    # Sort moves by adjusted utility
    sorted_rows = sorted(
        debug_rows, key=lambda d: d.get("sort_key", (0, 0, 0)), reverse=True
    )

    # Data rows
    for d in sorted_rows:
        # Our move
        mv = d.get("move_array")
        mv_str = f"[{mv[0]},{mv[1]}]" if mv else "[]"

        # Our stats
        pW = d.get("pW", 0.0)
        pT = d.get("pT", 0.0)
        pN = d.get("pN", 0.0)
        pL = d.get("pL", 0.0)

        # Our metrics
        cert = d.get("certainty", 0.0)
        util = d.get("utility", 0.0)
        adj = d.get("adjusted_utility", 0.0)
        total = d.get("total", 0)

        # Opponent reply
        opp_move = d.get("opponent_best_move")
        opp_move_str = f"[{opp_move[0]},{opp_move[1]}]" if opp_move else "--"

        # Opponent stats
        opp_pW = d.get("opponent_pW")
        opp_pT = d.get("opponent_pT")
        opp_pN = d.get("opponent_pN")
        opp_pL = d.get("opponent_pL")

        # Opponent metrics
        opp_util = d.get("opponent_best_util_for_them")
        opp_cert = d.get("opponent_best_cert")

        # Flags
        is_best = d.get("is_best", False)
        is_adjusted = d.get("adjusted", False)
        is_danger = d.get("dangerous", False)
        has_opp_data = d.get("opponent_data_exists", False)

        # Build notes
        notes = []
        if is_best:
            notes.append("★ BEST")
        if is_danger:
            notes.append("🔥")
        if is_adjusted:
            notes.append("↓")
        if has_opp_data:
            notes.append("👁")

        # Format row
        row_parts = [
            f"{mv_str:>{W_MOVE}}",
            "│",
            f"{pW:.3f}",
            f"{pT:.3f}",
            f"{pN:.3f}",
            f"{pL:.3f}",
            "│",
            f"{cert:.3f}",
            f"{util:+.2f}",
            f"{adj:+.2f}",
            f"{total:{W_TOTAL}d}",
            "│",
            f"{opp_move_str:>{W_OPP_MOVE}}",
            f"{_format_optional(opp_pW, '.3f'):>{W_OPP_STAT}}",
            f"{_format_optional(opp_pT, '.3f'):>{W_OPP_STAT}}",
            f"{_format_optional(opp_pN, '.3f'):>{W_OPP_STAT}}",
            f"{_format_optional(opp_pL, '.3f'):>{W_OPP_STAT}}",
            f"{_format_optional(opp_util, '+.2f'):>{W_OPP_METRIC}}",
            f"{_format_optional(opp_cert, '.3f'):>{W_OPP_METRIC}}",
            "│",
            " ".join(notes),
        ]

        output.append(" ".join(row_parts))

    output.append("")
    output.append(
        f"{DIM}Legend: ★=Best Move, 🔥=Dangerous, ↓=Risk-Adjusted, 👁=Opponent Data{RESET}"
    )
    output.append("")

    # ========================================
    # HEATMAP VISUALIZER
    # ========================================

    output.append(
        f"{BOLD}╔═══════════════════════════════════════════════════════════════════╗{RESET}"
    )
    output.append(
        f"{BOLD}║                       HEATMAP VISUALIZER                          ║{RESET}"
    )
    output.append(
        f"{BOLD}╚═══════════════════════════════════════════════════════════════════╝{RESET}"
    )
    output.append("")

    # Box drawing characters
    TL, TM, TR = "┌", "┬", "┐"
    ML, MM, MR = "├", "┼", "┤"
    BL, BM, BR = "└", "┴", "┘"
    H, V = "─", "│"

    def make_separator(left, mid, right):
        seg = H * cell_width
        return left + mid.join([seg] * N) + right

    output.append(make_separator(TL, TM, TR))

    bar_width = max(1, cell_width - 4)

    for r in range(N):
        top_cells = []
        mid_cells = []
        bot_cells = []

        for c in range(N):
            primary_val = primary_mat[r, c]
            secondary_val = secondary_mat[r, c]
            is_best = bool(is_best_mat[r, c])
            is_danger = bool(is_danger_mat[r, c])

            marker = _player_marker(board_arr[r, c])

            # Top: player marker
            cell_top = f" {marker} ".center(cell_width)

            # Middle: primary and secondary values
            p_str = f"{primary_val:.2f}" if not np.isnan(primary_val) else "--"
            s_str = f"{secondary_val:.2f}" if not np.isnan(secondary_val) else "--"
            cell_mid = f"{p_str} {s_str}".center(cell_width)

            # Bottom: visual bar
            if show_bars and not np.isnan(primary_val):
                bar = _bar_string(float(primary_val), bar_width)
                cell_bot = (" " + bar + " ").ljust(cell_width)
            else:
                cell_bot = " " * cell_width

            # Add best move markers
            if is_best and cell_width >= 3:
                cell_top = "▌" + cell_top[1:-1] + "▐"
                cell_mid = "▌" + cell_mid[1:-1] + "▐"
                cell_bot = "▌" + cell_bot[1:-1] + "▐"

            top_cells.append(cell_top)
            mid_cells.append(cell_mid)
            bot_cells.append(cell_bot)

        # Apply colors and join cells
        def colorize_row(cells):
            colored = []
            for c_idx, cell_text in enumerate(cells):
                primary_val = primary_mat[r, c_idx]

                if is_danger_mat[r, c_idx]:
                    bg = "\033[48;5;196m"  # Red for dangerous
                    fg = WHITE_FG
                elif is_best_mat[r, c_idx]:
                    bg = BEST_BG
                    fg = BEST_FG
                else:
                    bg = _bg_color_for_value(primary_val)
                    fg = _fg_for_value(primary_val)

                colored.append(f"{bg}{fg}{cell_text}{RESET}")
            return V + V.join(colored) + V

        output.append(colorize_row(top_cells))
        output.append(colorize_row(mid_cells))
        output.append(colorize_row(bot_cells))

        # Row separator
        if r < N - 1:
            output.append(make_separator(ML, MM, MR))
        else:
            output.append(make_separator(BL, BM, BR))

    output.append("")

    rendered = "\n".join(output)

    if return_str:
        return rendered

    print(rendered)
    return rendered
