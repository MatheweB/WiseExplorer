"""
Terminal debug visualizer for game TRANSITION statistics.

ARCHITECTURE NOTE:
-----------------
This visualizer is TRANSITION-FIRST:
  - Primary input: transition diffs (list of (coord, before, after))
  - Legacy fallback: move_array (deprecated, for backward compatibility)

Transitions represent STATE → STATE changes.
Moves are a game-engine detail used to derive transitions.

VISUALIZATION MODES:
-------------------
- PLACEMENT games (tic-tac-toe): Destination-based heatmap
- MOVEMENT games (chess): Source-based heatmap with arrows to destinations

Supports:
- Transition-based diffs (PREFERRED)
- Legacy move_array (fallback, deprecated)
- String-based pieces (MiniChess, etc.)
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# ===========================================================
# ANSI COLOR CODES
# ===========================================================
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def _fg_brightness(level: int) -> str:
    """Return ANSI foreground color for grayscale level 0-23."""
    return f"\033[38;5;{232 + level}m"


def _bg_brightness(level: int) -> str:
    """Return ANSI background color for grayscale level 0-23."""
    return f"\033[48;5;{232 + level}m"


# Fixed colors
WHITE_FG = "\033[38;5;255m"
BLACK_FG = "\033[38;5;232m"
GRAY_FG = "\033[38;5;244m"
SOURCE_BG = "\033[48;5;24m"  # Blue for source
SOURCE_FG = "\033[38;5;255m"
DEST_BG = "\033[48;5;22m"  # Green for destination
DEST_FG = "\033[38;5;255m"

# ===========================================================
# LAYOUT CONFIGURATION
# ===========================================================


def _brightness_for_score(score: float) -> int:
    """
    Convert score [0, 1] to grayscale brightness level [0, 23].
    Higher score = brighter, Lower score = darker.
    """
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return 6
    s = float(max(0.0, min(1.0, score)))
    return int(2 + s * 21)


def _bar_string(value: float, width: int) -> str:
    """Create a simple bar using block characters."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "·" * width
    v = float(max(0.0, min(1.0, value)))
    filled = int(round(v * width))
    return "█" * filled + "·" * (width - filled)


def _player_marker(piece_val) -> str:
    """Convert piece value to display character."""
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
    """Format a move for display."""
    if move is None:
        return "?"
    if isinstance(move, np.ndarray):
        move = move.tolist()
    if isinstance(move, (list, tuple)):
        if len(move) == 2:
            return f"({move[0]},{move[1]})"
        elif len(move) == 4:
            return f"({move[0]},{move[1]})→({move[2]},{move[3]})"
        elif len(move) == 1:
            return f"#{move[0]}"
        else:
            return str(move)
    if isinstance(move, (int, np.integer)):
        return f"#{int(move)}"
    return str(move)


def _format_transition_short(diff: List) -> str:
    """Format transition using delta notation: Δ(r,c) before→after."""
    if not diff:
        return "no change"

    parts = []
    for idx, before, after in diff:
        before_str = _player_marker(before)
        after_str = _player_marker(after)
        r, c = idx
        parts.append(f"Δ({r},{c}) {before_str}→{after_str}")

    return " | ".join(parts)


# ===========================================================
# Move type detection and coordinate extraction
# ===========================================================


def _extract_source_dest(
    diff: List,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Extract source and destination coordinates from a diff.

    Returns:
        (source, dest) where each is (row, col) or None

    For placement games: source=None, dest=where piece appeared
    For movement games: source=where piece left, dest=where piece arrived
    """
    source = None
    dest = None

    for idx, before, after in diff:
        r, c = idx
        before_empty = before is None or before == 0 or before == "·"
        after_empty = after is None or after == 0 or after == "·"

        if not before_empty and after_empty:
            # Piece left this square
            source = (r, c)
        elif after_empty is False:  # Piece arrived (including captures)
            dest = (r, c)

    return source, dest


def _is_movement_game(debug_rows: List[Dict[str, Any]]) -> bool:
    """
    Detect if this is a movement game (chess) vs placement game (tic-tac-toe).

    Movement games have diffs with both a source (piece leaves) and dest (piece arrives).
    Placement games only have a dest (piece appears on empty square).
    """
    for d in debug_rows:
        diff = d.get("diff", [])
        if len(diff) >= 2:  # Multiple squares changed = movement
            source, _dest = _extract_source_dest(diff)
            if source is not None:
                return True
    return False


# ===========================================================
# Main render function
# ===========================================================


def render_debug(
    board: np.ndarray,
    debug_rows: List[Dict[str, Any]],
    *,
    show_bars: bool = True,
    cell_width: int = 14,
    return_str: bool = False,
) -> str:
    """
    Render transition statistics as a terminal visualization.

    Args:
        board: Current game board (np.ndarray)
        debug_rows: List of transition statistics, each containing:
          - diff: List[(idx, before, after)]
          - move: The move representation
          - score, utility, certainty, etc. (statistics)
          - pW, pT, pN, pL (win/tie/neutral/loss rates)
          - total: sample count
          - is_selected: bool (highlight flag)
        show_bars: Whether to show visual bars in cells
        cell_width: Width of each cell in characters
        return_str: If True, return string instead of printing

    Returns:
        Formatted string (always), prints unless return_str=True
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
    is_movement = _is_movement_game(debug_rows)

    # =======================================================
    # Output assembly - Move table
    # =======================================================

    output = [
        "",
        f"{BOLD}╔" + "═" * 78 + "╗",
        f"║{'TRANSITION ANALYSIS SUMMARY':^78}║",
        "╚" + "═" * 78 + f"╝{RESET}",
        "",
    ]

    sorted_rows = sorted(
        debug_rows, key=lambda x: x.get("score", float("-inf")), reverse=True
    )

    # Build row data first to calculate column widths
    table_data = []
    for d in sorted_rows:
        diff = d.get("diff", [])

        # Build transition parts separately for alignment
        trans_parts = []
        for idx, before, after in diff:
            before_str = _player_marker(before)
            after_str = _player_marker(after)
            r, c = idx
            trans_parts.append(f"Δ({r},{c}) {before_str}→{after_str}")

        rates_str = f"{d.get('pW',0):.2f} {d.get('pT',0):.2f} {d.get('pN',0):.2f} {d.get('pL',0):.2f}"
        metrics_str = f"C:{d.get('certainty',0):.2f} U:{d.get('utility',0):+.2f} S:{d.get('score',0):.2f} ({d.get('total',0)})"

        table_data.append(
            {
                "trans_parts": trans_parts,
                "rates": rates_str,
                "metrics": metrics_str,
                "is_selected": d.get("is_selected", False),
                "score": d.get("score", float("-inf")),
            }
        )

    # Calculate max width for each transition part position
    max_parts = max((len(r["trans_parts"]) for r in table_data), default=0)
    part_widths = []
    for i in range(max_parts):
        w = max(
            (
                len(r["trans_parts"][i]) if i < len(r["trans_parts"]) else 0
                for r in table_data
            ),
            default=0,
        )
        part_widths.append(w)

    # Build aligned transition strings
    for r in table_data:
        aligned_parts = []
        for i, w in enumerate(part_widths):
            if i < len(r["trans_parts"]):
                aligned_parts.append(f"{r['trans_parts'][i]:<{w}}")
            else:
                aligned_parts.append(" " * w)
        r["transition"] = " | ".join(aligned_parts)

    # Calculate column widths
    w_trans = max(
        len("Transition"), max((len(r["transition"]) for r in table_data), default=0)
    )
    w_rates = max(
        len("Rates (W/T/N/L)"), max((len(r["rates"]) for r in table_data), default=0)
    )
    w_metrics = max(
        len("Metrics"), max((len(r["metrics"]) for r in table_data), default=0)
    )

    # Build header
    header = f"{'Transition':<{w_trans}} │ {'Rates (W/T/N/L)':<{w_rates}} │ {'Metrics':<{w_metrics}}"
    output.append(header)
    output.append(f"{DIM}{'─' * len(header)}{RESET}")

    # Build rows
    for r in table_data:
        score = r["score"]
        brightness = _brightness_for_score(score) if score > float("-inf") else 6
        fg = _fg_brightness(brightness)

        row_str = (
            f"{r['transition']:<{w_trans}} │ "
            f"{r['rates']:<{w_rates}} │ "
            f"{fg}{r['metrics']:<{w_metrics}}{RESET}"
        )

        if r["is_selected"]:
            row_str = f"{BOLD}{row_str} ◀ SELECTED{RESET}"

        output.append(row_str)

    # =======================================================
    # Board visualization
    # =======================================================

    output.append("")
    output.append(f"{BOLD}╔" + "═" * 78 + "╗")
    output.append(f"║{'GAME STATE WITH SELECTED TRANSITION':^78}║")
    output.append("╚" + "═" * 78 + f"╝{RESET}")
    output.append("")

    # Find selected move's source and destination
    selected_source = None
    selected_dest = None
    selected_diff = None

    for d in debug_rows:
        if d.get("is_selected"):
            selected_diff = d.get("diff", [])
            selected_source, selected_dest = _extract_source_dest(selected_diff)
            break

    # For placement games, build score matrix by destination
    score_by_dest = {}
    if not is_movement:
        for d in debug_rows:
            diff = d.get("diff", [])
            _, dest = _extract_source_dest(diff)
            if dest:
                score_by_dest[dest] = d.get("score", np.nan)

    # Build the grid
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
            piece = _player_marker(board_arr[r, c])
            is_source = (r, c) == selected_source
            is_dest = (r, c) == selected_dest

            if is_movement:
                # Movement game: show current piece, highlight selected move
                if is_source:
                    # Show what piece is moving and where
                    dest_str = (
                        f"→({selected_dest[0]},{selected_dest[1]})"
                        if selected_dest
                        else ""
                    )
                    t_txt = f" {piece}{dest_str} ".center(cell_width)
                    m_txt = "SOURCE".center(cell_width)
                    b_txt = " " * cell_width
                elif is_dest:
                    # Show what's arriving
                    arriving = "?"
                    for idx, before, after in selected_diff or []:
                        if idx == (r, c):
                            arriving = _player_marker(after)
                            break
                    t_txt = f" {piece}→{arriving} ".center(cell_width)
                    m_txt = "DEST".center(cell_width)
                    b_txt = " " * cell_width
                else:
                    # Regular square
                    t_txt = f" {piece} ".center(cell_width)
                    m_txt = f"({r},{c})".center(cell_width)
                    b_txt = " " * cell_width

                # Colors for movement game
                if is_source:
                    bg, fg = SOURCE_BG, SOURCE_FG
                elif is_dest:
                    bg, fg = DEST_BG, DEST_FG
                else:
                    bg = _bg_brightness(4)
                    fg = WHITE_FG

            else:
                # Placement game: heatmap by destination score
                score = score_by_dest.get((r, c), np.nan)
                score_str = f"{score:.2f}" if not np.isnan(score) else "---"

                t_txt = f" {piece} ".center(cell_width)
                m_txt = f"{score_str}".center(cell_width)
                b_txt = (
                    (" " + _bar_string(score, bar_w) + " ").ljust(cell_width)
                    if show_bars and not np.isnan(score)
                    else " " * cell_width
                )

                if is_dest:  # Selected in placement game
                    bg, fg = DEST_BG, DEST_FG
                elif np.isnan(score):
                    bg = _bg_brightness(3)
                    fg = _fg_brightness(10)
                else:
                    brightness = _brightness_for_score(score)
                    bg = _bg_brightness(brightness)
                    fg = BLACK_FG if brightness > 12 else WHITE_FG

            # Selection borders
            if is_source or is_dest:
                t_txt = "▌" + t_txt[1:-1] + "▐"
                m_txt = "▌" + m_txt[1:-1] + "▐"
                b_txt = "▌" + b_txt[1:-1] + "▐"

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
