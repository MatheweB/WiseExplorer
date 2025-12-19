"""
State canonicalization with graph-based symmetry handling.

FIXED VERSION v3 - CERTIFICATE-BASED APPROACH:
- Uses pynauty.certificate() which is GUARANTEED identical for isomorphic graphs
- No need to interpret canon_label semantics
- Much simpler and more reliable

The key insight: pynauty.certificate() returns a canonical byte representation
of the graph structure. Isomorphic graphs get identical certificates.
"""

from __future__ import annotations

import hashlib
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import pynauty  # type: ignore


# ============================================================================
# Canonical Result
# ============================================================================


@dataclass(frozen=True)
class CanonicalResult:
    hash: str
    labeling: tuple[int, ...]  # Still available if needed
    inv_labeling: tuple[int, ...]  # Still available if needed
    shape: tuple[int, ...]


# ============================================================================
# Public API
# ============================================================================


def canonicalize_state(state) -> CanonicalResult:
    """
    Canonicalize a game state using graph isomorphism.
    INCLUDES current_player in the hash.
    """
    board = np.asarray(state.board, dtype=object)
    return _canonicalize_impl(board, include_player=state.current_player)


def canonicalize_board(board: np.ndarray) -> CanonicalResult:
    """
    Canonicalize a board configuration ONLY (ignore current_player).
    """
    board = np.asarray(board, dtype=object)
    return _canonicalize_impl(board, include_player=None)


def _canonicalize_impl(
    board: np.ndarray, include_player: Optional[int]
) -> CanonicalResult:
    """Internal implementation for canonicalization."""
    shape = board.shape
    graph, n, coord_list = _board_to_graph(board, include_player)

    # Get certificate - guaranteed identical for isomorphic graphs
    cert = pynauty.certificate(graph)

    # Hash the certificate (include player if specified)
    if include_player is not None:
        hash_input = (cert, include_player)
    else:
        hash_input = cert

    state_hash = hashlib.sha256(repr(hash_input).encode()).hexdigest()

    # Also get labeling for potential future use (e.g., move transformation)
    labeling = _get_canonical_labeling(graph, n)
    inv_labeling = _invert(labeling)

    return CanonicalResult(
        hash=state_hash,
        labeling=labeling,
        inv_labeling=tuple(inv_labeling),
        shape=shape,
    )


# ============================================================================
# Position Type Classification (for grid symmetry)
# ============================================================================


def _get_position_type(coord: Tuple[int, ...], shape: Tuple[int, ...]) -> Tuple:
    """
    Classify a coordinate by its structural position type.

    For a 2D grid, this creates equivalence classes:
    - Corners: distance pattern from edges is (0,0) in sorted form
    - Edges: distance pattern is (0, k) for some k > 0
    - Interior: distance pattern is (k1, k2) for k1, k2 > 0

    The key: we compute distance to nearest edge for each axis,
    then SORT these distances. This makes the classification
    invariant under the grid's symmetry group (rotations + reflections).
    """
    distances = []
    for c, s in zip(coord, shape):
        dist = min(c, s - 1 - c)
        distances.append(dist)
    return tuple(sorted(distances))


# ============================================================================
# Graph Construction
# ============================================================================


def _board_to_graph(board: np.ndarray, player: Optional[int]):
    """
    Convert a game board into a colored graph for canonical labeling.

    Node colors encode BOTH:
    1. The cell value (empty, X, O, etc.)
    2. The position TYPE (corner, edge, center, etc.)

    This ensures pynauty only finds symmetries that map corners to corners,
    edges to edges, etc. - exactly the D4 symmetry group for a square grid.
    """
    shape = board.shape
    coords = list(np.ndindex(shape))
    n = len(coords)

    index = {coord: i for i, coord in enumerate(coords)}
    adjacency = {i: [] for i in range(n)}
    color_classes: dict[tuple, list[int]] = {}

    for coord in coords:
        i = index[coord]
        cell = board[coord]

        # Normalize cell value
        if cell is None:
            cell_label = ("EMPTY",)
        elif isinstance(cell, (bool, int, np.integer)):
            cell_label = ("INT", int(cell))
        elif isinstance(cell, (float, np.floating)):
            cell_label = ("FLOAT", float(cell))
        else:
            cell_label = ("OBJ", repr(cell))

        # Include position type in color
        pos_type = _get_position_type(coord, shape)

        # Build color key (player not needed in color - it's added to hash separately)
        color_key = (cell_label, pos_type)
        color_classes.setdefault(color_key, []).append(i)

        # Build adjacency (grid neighbors)
        for axis in range(board.ndim):
            for delta in [-1, 1]:
                nbr = list(coord)
                nbr[axis] += delta
                if 0 <= nbr[axis] < shape[axis]:
                    j = index[tuple(nbr)]
                    if j not in adjacency[i]:
                        adjacency[i].append(j)

    # Sort color classes for determinism
    sorted_keys = sorted(
        color_classes.keys(), key=lambda k: (repr(k), min(color_classes[k]))
    )
    vertex_coloring = [set(color_classes[k]) for k in sorted_keys]

    graph = pynauty.Graph(
        number_of_vertices=n,
        adjacency_dict=adjacency,
        directed=False,
        vertex_coloring=vertex_coloring,
    )

    return graph, n, coords


# ============================================================================
# Canonical Labeling (for move transformation, not hashing)
# ============================================================================


def _get_canonical_labeling(graph, n: int) -> tuple[int, ...]:
    """Get canonical labeling from pynauty (for move transformation use)."""
    result = pynauty.canon_label(graph)
    labeling = _extract_labeling(result, n)
    return tuple(int(x) for x in labeling)


def _extract_labeling(result, n: int) -> List[int]:
    """Extract canonical labeling from pynauty result."""
    if isinstance(result, (list, tuple)) and len(result) == n:
        if set(result) == set(range(n)):
            return list(result)
    if isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, (list, tuple)) and len(item) == n:
                if set(item) == set(range(n)):
                    return list(item)
    raise RuntimeError(f"Unable to extract canonical labeling from: {result}")


def _invert(labeling):
    """Invert a permutation."""
    inv = [0] * len(labeling)
    for i, c in enumerate(labeling):
        inv[c] = i
    return inv
