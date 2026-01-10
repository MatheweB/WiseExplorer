"""
Terminal debug visualizer for game transition analysis.

Shows:
- Moves grouped by anchor (equivalent moves)
- Win/Loss/Tie distribution with visual bars
- Direct stats vs pooled stats
- Within-anchor preference ranking
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import math
import numpy as np

# ===========================================================
# ANSI COLOR CODES
# ===========================================================
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

WHITE = "\033[38;5;255m"
BLACK = "\033[38;5;232m"
GRAY = "\033[38;5;245m"
GREEN = "\033[38;5;82m"
RED = "\033[38;5;196m"
CYAN = "\033[38;5;87m"
YELLOW = "\033[38;5;220m"
ORANGE = "\033[38;5;208m"
BLUE = "\033[38;5;75m"

BG_SOURCE = "\033[48;5;24m"
BG_DEST = "\033[48;5;22m"

# Box drawing characters
BOX_H = "─"
BOX_V = "│"
BOX_TL = "┌"
BOX_TR = "┐"
BOX_BL = "└"
BOX_BR = "┘"
BOX_LT = "├"
BOX_RT = "┤"
BOX_TT = "┬"
BOX_BT = "┴"
BOX_X = "┼"

# Block characters for bars
BLOCK_FULL = "█"
BLOCK_LIGHT = "░"


def _fg_brightness(level: int) -> str:
    return f"\033[38;5;{232 + level}m"


def _bg_brightness(level: int) -> str:
    return f"\033[48;5;{232 + level}m"


def _score_color(score: float) -> str:
    if score >= 0.65: return GREEN
    if score >= 0.45: return YELLOW
    if score >= 0.35: return ORANGE
    return RED


def _compute_score(wins: int, losses: int, total: int) -> float:
    """Bayesian score with uniform Dirichlet prior (α=1 pseudocounts)."""
    w_eff = wins + 1
    l_eff = losses + 1
    n_eff = total + 4
    mean = (w_eff - l_eff) / n_eff
    var = max(0, (w_eff + l_eff - n_eff * mean**2) / (n_eff - 1))
    return (mean - math.sqrt(var / n_eff) + 1) / 2


def _player_marker(piece_val) -> str:
    if piece_val is None or piece_val == 0:
        return "·"
    if isinstance(piece_val, str):
        return piece_val
    try:
        val = int(piece_val)
        return "X" if val == 1 else "O" if val in (2, -1) else str(piece_val)
    except (ValueError, TypeError):
        return str(piece_val)


def _extract_source_dest(diff: List) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    source, dest = None, None
    for idx, before, after in diff:
        r, c = idx
        before_empty = before is None or before == 0 or before == "." or before == "·"
        after_empty = after is None or after == 0 or after == "." or after == "·"
        if not before_empty and after_empty:
            source = (r, c)
        elif not after_empty:
            dest = (r, c)
    return source, dest


def _is_movement_game(debug_rows: List[Dict]) -> bool:
    for d in debug_rows:
        diff = d.get("diff", [])
        if len(diff) >= 2:
            source, _ = _extract_source_dest(diff)
            if source is not None:
                return True
    return False


def _format_move(diff: List, is_movement: bool) -> str:
    source, dest = _extract_source_dest(diff)
    if is_movement and source and dest:
        return f"{source[0]},{source[1]}→{dest[0]},{dest[1]}"
    elif dest:
        return f"{dest[0]},{dest[1]}"
    elif diff:
        r, c = diff[0][0]
        return f"{r},{c}"
    return "?"


def _wlt_bar(w_pct: float, l_pct: float, t_pct: float, width: int = 12) -> str:
    """Create a colored W/L/T distribution bar."""
    w_chars = int(round(w_pct / 100 * width))
    l_chars = int(round(l_pct / 100 * width))
    t_chars = width - w_chars - l_chars
    
    bar = f"{GREEN}{BLOCK_FULL * w_chars}{RESET}"
    bar += f"{RED}{BLOCK_FULL * l_chars}{RESET}"
    bar += f"{YELLOW}{BLOCK_FULL * t_chars}{RESET}"
    return bar


def _horizontal_line(width: int, left: str, mid: str, right: str) -> str:
    """Create a horizontal line with box characters."""
    return left + BOX_H * (width - 2) + right


# ===========================================================
# Main render function  
# ===========================================================

def render_debug(
    board: np.ndarray,
    debug_rows: List[Dict[str, Any]],
    *,
    show_board: bool = True,
    return_str: bool = False,
) -> str:
    """Render transition analysis grouped by anchor."""
    
    lines = []
    is_movement = _is_movement_game(debug_rows)
    
    W = 90
    move_w = 12 if is_movement else 6
    
    # Title
    lines.append("")
    lines.append(f"{BOLD}{_horizontal_line(W, BOX_TL, BOX_H, BOX_TR)}{RESET}")
    title = "MOVE ANALYSIS"
    padding = (W - 2 - len(title)) // 2
    lines.append(f"{BOLD}{BOX_V}{' ' * padding}{title}{' ' * (W - 2 - padding - len(title))}{BOX_V}{RESET}")
    lines.append(f"{BOLD}{_horizontal_line(W, BOX_BL, BOX_H, BOX_BR)}{RESET}")
    
    # Group by anchor
    by_anchor = defaultdict(list)
    for d in debug_rows:
        by_anchor[d.get("anchor_id", -1)].append(d)
    
    sorted_anchors = sorted(by_anchor.keys(), 
                           key=lambda a: max(d.get("score", 0) for d in by_anchor[a]), 
                           reverse=True)
    
    # Render each anchor group
    for rank, aid in enumerate(sorted_anchors):
        moves = sorted(by_anchor[aid], 
                      key=lambda x: x.get("direct_score", x.get("score", 0)), 
                      reverse=True)
        
        # Use effective stats
        eff_total = moves[0].get("total", 0)
        eff_pW = moves[0].get("pW", 0)
        eff_pL = moves[0].get("pL", 0)
        eff_pT = moves[0].get("pT", 0)
        anchor_score = moves[0].get("score", 0)
        
        is_best = (rank == 0)
        n_moves = len(moves)
        
        lines.append("")
        
        # Anchor header
        border_color = f"{BOLD}{GREEN}" if is_best else ""
        reset_border = RESET if is_best else ""
        
        lines.append(f"{border_color}{_horizontal_line(W, BOX_TL, BOX_H, BOX_TR)}{reset_border}")
        
        # Title line
        if is_best:
            marker = f"{GREEN}★ BEST{RESET}"
            marker_len = 6
        else:
            marker = f"{GRAY}#{rank+1}{RESET}"
            marker_len = len(f"#{rank+1}")
        
        equiv_str = f" ({n_moves} equiv)" if n_moves > 1 else ""
        score_col = _score_color(anchor_score)
        
        title_text = f"Anchor {aid}{equiv_str}"
        score_text = f"Score: {anchor_score:.3f}"
        
        # Fixed positioning
        left_content = f" {marker}  {title_text}"
        left_visible_len = 1 + marker_len + 2 + len(title_text)
        right_content = f"{score_col}{BOLD}{score_text}{RESET} "
        right_visible_len = len(score_text) + 1
        
        inner_padding = W - 2 - left_visible_len - right_visible_len
        lines.append(f"{border_color}{BOX_V}{reset_border}{left_content}{' ' * inner_padding}{right_content}{border_color}{BOX_V}{reset_border}")
        
        # Stats line with W/L/T distribution
        w_pct = eff_pW * 100
        l_pct = eff_pL * 100
        t_pct = eff_pT * 100
        
        stats_text = f" n={eff_total:<6}"
        stats_text += f"  {GREEN}W:{w_pct:4.1f}%{RESET}"
        stats_text += f"  {RED}L:{l_pct:4.1f}%{RESET}"
        stats_text += f"  {YELLOW}T:{t_pct:4.1f}%{RESET}"
        stats_text += f"  {_wlt_bar(w_pct, l_pct, t_pct, 16)}"
        
        stats_visible_len = len(f" n={eff_total:<6}  W:{w_pct:4.1f}%  L:{l_pct:4.1f}%  T:{t_pct:4.1f}%  ") + 16
        stats_padding = W - 2 - stats_visible_len
        
        lines.append(f"{border_color}{BOX_V}{reset_border}{stats_text}{' ' * stats_padding}{border_color}{BOX_V}{reset_border}")
        lines.append(f"{border_color}{BOX_LT}{BOX_H * (W-2)}{BOX_RT}{reset_border}")
        
        # Column headers
        hdr = f"{DIM} {'Move':<{move_w}} {BOX_V}    W%     L%     T%    (n) {BOX_V}  Solo  {BOX_V} Pooled {BOX_V}       {RESET}"
        lines.append(f"{border_color}{BOX_V}{reset_border}{hdr}{border_color}{BOX_V}{reset_border}")
        lines.append(f"{border_color}{BOX_LT}{BOX_H * (W-2)}{BOX_RT}{reset_border}")
        
        # Each move
        for i, m in enumerate(moves):
            diff = m.get("diff", [])
            move_str = _format_move(diff, is_movement)[:move_w].ljust(move_w)
            
            direct_total = m.get("direct_total", 0)
            direct_W = m.get("direct_W", 0)
            direct_L = m.get("direct_L", 0)
            direct_T = m.get("direct_T", 0)
            
            if direct_total > 0:
                d_w_pct = direct_W / direct_total * 100
                d_l_pct = direct_L / direct_total * 100
                d_t_pct = direct_T / direct_total * 100
            else:
                d_w_pct = d_l_pct = d_t_pct = 0
            
            solo_score = _compute_score(direct_W, direct_L, direct_total)
            pooled_score = m.get("score", 0)
            
            solo_col = _score_color(solo_score)
            pooled_col = _score_color(pooled_score)
            
            is_selected = m.get("is_selected", False)
            
            # Rank indicator with fixed width
            if i == 0:
                rank_str = f"{YELLOW}1st{RESET}"
            else:
                rank_str = f"{GRAY}#{i+1:<2}{RESET}"
            
            # Selection marker with fixed width
            sel_str = f"{GREEN}◀{RESET}" if is_selected else " "
            
            # Build row with fixed widths
            row = f" {move_str} {BOX_V} "
            row += f"{GREEN}{d_w_pct:5.1f}{RESET}  "
            row += f"{RED}{d_l_pct:5.1f}{RESET}  "
            row += f"{YELLOW}{d_t_pct:5.1f}{RESET}  "
            row += f"{DIM}{direct_total:>5}{RESET} "
            row += f"{BOX_V} {solo_col}{solo_score:.3f}{RESET} "
            row += f"{BOX_V} {pooled_col}{BOLD}{pooled_score:.3f}{RESET}  "
            row += f"{BOX_V} {rank_str} {sel_str} "
            
            lines.append(f"{border_color}{BOX_V}{reset_border}{row}{border_color}{BOX_V}{reset_border}")
        
        lines.append(f"{border_color}{_horizontal_line(W, BOX_BL, BOX_H, BOX_BR)}{reset_border}")
    
    # Legend
    lines.append("")
    lines.append(f"{DIM}  Solo = direct data only │ Pooled = anchor stats │ ◀ = selected move{RESET}")
    
    # Board
    if show_board:
        board_arr = np.array(board)
        if board_arr.ndim == 1:
            side = int(np.sqrt(board_arr.size))
            board_arr = board_arr.reshape((side, side)) if side * side == board_arr.size else board_arr.reshape((1, -1))
        
        ROWS, COLS = board_arr.shape
        
        selected_source, selected_dest = None, None
        for d in debug_rows:
            if d.get("is_selected"):
                selected_source, selected_dest = _extract_source_dest(d.get("diff", []))
                break
        
        info_by_dest = {}
        for d in debug_rows:
            _, dest = _extract_source_dest(d.get("diff", []))
            if dest:
                info_by_dest[dest] = {
                    'score': d.get("score", float('nan')),
                    'anchor_id': d.get("anchor_id"),
                }
        
        lines.append("")
        lines.append(f"{BOLD}  Board State{RESET}")
        
        cell_w = 10
        
        # Top border
        lines.append("    " + BOX_TL + (BOX_H * cell_w + BOX_TT) * (COLS - 1) + BOX_H * cell_w + BOX_TR)
        
        for r in range(ROWS):
            cells = []
            for c in range(COLS):
                piece = _player_marker(board_arr[r, c])
                info = info_by_dest.get((r, c))
                is_source = (r, c) == selected_source
                is_dest = (r, c) == selected_dest
                
                if is_source:
                    label = f"{piece} SRC"
                    cell = f"{BG_SOURCE}{WHITE}{label:^{cell_w}}{RESET}"
                elif is_dest:
                    aid = info['anchor_id'] if info else '?'
                    label = f"[{aid}] DST"
                    cell = f"{BG_DEST}{WHITE}{label:^{cell_w}}{RESET}"
                elif info:
                    score = info['score']
                    label = f"{score:.3f}"
                    brightness = int(4 + score * 18) if not np.isnan(score) else 6
                    bg = _bg_brightness(brightness)
                    fg = BLACK if brightness > 14 else WHITE
                    cell = f"{bg}{fg}{label:^{cell_w}}{RESET}"
                else:
                    cell = f"{_bg_brightness(3)}{GRAY}{piece:^{cell_w}}{RESET}"
                cells.append(cell)
            
            lines.append("    " + BOX_V + (BOX_V).join(cells) + BOX_V)
            
            if r < ROWS - 1:
                lines.append("    " + BOX_LT + (BOX_H * cell_w + BOX_X) * (COLS - 1) + BOX_H * cell_w + BOX_RT)
        
        # Bottom border
        lines.append("    " + BOX_BL + (BOX_H * cell_w + BOX_BT) * (COLS - 1) + BOX_H * cell_w + BOX_BR)
        
        if is_movement and selected_source and selected_dest:
            lines.append(f"    {DIM}Selected: {selected_source} → {selected_dest}{RESET}")
    
    lines.append("")
    
    result = "\n".join(lines)
    if not return_str:
        print(result)
    return result
