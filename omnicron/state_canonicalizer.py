"""
Production-grade graph canonicalization for game states.
Uses orbit-based canonicalization to detect automorphism symmetries.

Public API:
  - canonicalize_state(state) -> CanonicalResult
  - CanonicalResult.canonicalize_move(move) -> canonical signature string
  - CanonicalResult.invert_move(canonical_sig) -> original move
  - CanonicalResult.get_all_symmetric_moves(canonical_sig) -> list of equivalent moves
"""
from __future__ import annotations
import hashlib
import logging
from typing import Any, Protocol, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import product
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism

# Try to import pynauty for speed, fall back to networkx if missing
try:
    import pynauty  # type: ignore
    NAUTY_AVAILABLE = True
except ImportError:
    NAUTY_AVAILABLE = False

logger = logging.getLogger(__name__)

# ===========================
# Core Types
# ===========================

class GameState(Protocol):
    """Expected interface for game states passed to this module."""
    board: np.ndarray | Any
    current_player: int


@dataclass(frozen=True)
class Move:
    """
    Normalized internal representation of a move.
    Always stores moves as immutable, hashable objects.
    """
    coords: tuple[int, ...]  # Numeric coordinates (e.g., (row, col))
    node_id: tuple[str, Any] | None  # Original NodeID if symbolic (e.g., ("cell", "A"))
    
    def __hash__(self):
        # Hash only by coords for consistent lookup across different node metadata
        return hash(self.coords)
    
    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        # Compare only by coords
        return self.coords == other.coords
    
    def to_output(self) -> tuple[int, ...]:
        """Return the canonical output format (always coords tuple)."""
        return self.coords


@dataclass
class CanonicalResult:
    """
    Result of state canonicalization with orbit detection.
    Contains all mappings necessary to translate between Real Moves and Canonical Signatures.
    """
    hash: str
    graph: nx.Graph
    canonical_graph: nx.Graph
    
    # Core mappings (Internal use)
    _node_to_canonical: dict[tuple, int]  # NodeID -> canonical int
    _canonical_to_node: dict[int, tuple]  # canonical int -> NodeID
    _orbit_representatives: dict[int, int]  # canonical int -> orbit representative
    _orbit_id_by_node: dict[tuple, int]  # NodeID -> orbit ID
    
    # Signature mappings (The key to move translation)
    _signature_to_moves: dict[str, list[Move]]  # signature -> all equivalent moves
    _move_to_signature: dict[Move, str]  # move -> its signature
    
    # Debug info
    original_board: Optional[np.ndarray] = None
    original_shape: Optional[tuple] = None
    
    # ===========================
    # Public API
    # ===========================
    
    def canonicalize_move(self, move: Any, move_parts: int = None) -> str:
        """
        Convert any move format to a canonical signature.
        
        Game-agnostic: automatically handles single-part moves (tic-tac-toe) 
        and multi-part moves (chess: source→destination).
        
        Args:
            move: Can be:
                - tuple of ints: (row, col) or (from_r, from_c, to_r, to_c)
                - numpy array: np.array([row, col]) or np.array([from_r, from_c, to_r, to_c])
                - NodeID: ("cell", coords)
            move_parts: Number of coordinate pairs in the move (auto-detected if None)
        
        Returns:
            Canonical signature string:
            - Single-part: "sig_<hash>:<index>" or "move_<coords>"
            - Multi-part: "sig_<hash1>:<idx1>→sig_<hash2>:<idx2>"
        """
        move_array = np.asarray(move).flatten()
        
        # Auto-detect move structure if not explicit
        if move_parts is None:
            if len(move_array) == 2:
                move_parts = 1 # Simple 2D position
            elif len(move_array) == 4:
                move_parts = 2 # Two 2D positions (chess move)
            elif len(move_array) > 0 and len(move_array) % 2 == 0:
                move_parts = len(move_array) // 2
            else:
                move_parts = 1 # Fallback
        
        # Handle single-part moves (Base Case)
        if move_parts == 1:
            normalized = self._normalize_input_move(move_array)
            
            # Check if this move corresponds to a known node in the graph
            if normalized in self._move_to_signature:
                sig = self._move_to_signature[normalized]
                
                # We need to distinguish between identical orbits (collisions).
                # Example: Two empty corners have same orbit, but are distinct moves.
                # We append the index of this specific move within that orbit.
                moves_in_orbit = self._signature_to_moves[sig]
                try:
                    idx = moves_in_orbit.index(normalized)
                    return f"{sig}:{idx}"
                except ValueError:
                    return sig
            
            # Not in graph - return raw coordinate format (fallback)
            coords_str = "_".join(str(c) for c in normalized.coords)
            return f"move_{coords_str}"
        
        # Handle multi-part moves (Recursive Case)
        else:
            signatures = []
            coords_per_part = len(move_array) // move_parts
            
            for i in range(move_parts):
                part = move_array[i*coords_per_part : (i+1)*coords_per_part]
                # Recursively canonicalize each part
                part_sig = self.canonicalize_move(part, move_parts=1)
                signatures.append(part_sig)
            
            # Join with arrow to create compound signature
            return "→".join(signatures)
    
    def invert_move(self, canonical_sig: str | int) -> Optional[np.ndarray]:
        """
        Convert canonical signature back to a representative original move.
        Useful for getting *one* valid move to play.
        """
        # Legacy integer support
        if isinstance(canonical_sig, (int, np.integer)):
            return self._invert_legacy_int(canonical_sig)
        
        if not isinstance(canonical_sig, str):
            return None
        
        # Handle compound moves (e.g., chess castling or moves)
        if "→" in canonical_sig:
            parts = canonical_sig.split("→")
            inverted_parts = []
            for part in parts:
                part_result = self.invert_move(part)
                if part_result is None:
                    return None
                inverted_parts.append(part_result)
            return np.concatenate(inverted_parts)
        
        # Handle raw coordinate fallback
        if canonical_sig.startswith("move_"):
            try:
                coords_str = canonical_sig[5:]
                coords = tuple(int(c) for c in coords_str.split("_"))
                return np.array(coords, dtype=int)
            except (ValueError, IndexError):
                return None

        # Handle orbit signatures "sig_<hash>:<index>"
        if canonical_sig.startswith("sig_"):
            return self._invert_signature(canonical_sig)
            
        return None

    def get_all_symmetric_moves(self, canonical_sig: str) -> list[np.ndarray]:
        """
        Get ALL moves that are symmetric/equivalent to this signature.
        Used for learning: one observed move updates stats for all symmetric moves.
        """
        # Handle compound moves (recurse and combine)
        if "→" in canonical_sig:
            parts = canonical_sig.split("→")
            
            # Get symmetric options for each part
            # part_options is a list of lists: [ [options_for_part1], [options_for_part2] ]
            part_options = []
            for part in parts:
                base_sig = part.split(":", 1)[0]
                # If it's a raw move (move_X_Y), it has no symmetry other than itself
                if part.startswith("move_"):
                    inverted = self.invert_move(part)
                    if inverted is not None:
                        part_options.append([inverted])
                    else:
                        return []
                else:
                    # It is an orbit signature, fetch all nodes in that orbit
                    moves = self._signature_to_moves.get(base_sig, [])
                    part_options.append([np.array(m.to_output(), dtype=int) for m in moves])
            
            if not all(part_options):
                return []

            # Cartesian product: generate all valid combinations
            # e.g., if Corner->Center, generate [CornerA->Center, CornerB->Center, ...]
            result = []
            for combo in product(*part_options):
                full_move = np.concatenate(combo)
                result.append(full_move)
            
            return result
        
        # Single part move
        base_sig = canonical_sig.split(":", 1)[0]
        
        if base_sig.startswith("move_"):
             # No symmetry known
             inverted = self.invert_move(canonical_sig)
             return [inverted] if inverted is not None else []
             
        # Orbit symmetry
        moves = self._signature_to_moves.get(base_sig, [])
        return [np.array(m.to_output(), dtype=int) for m in moves]

    # Alias to satisfy Protocol in manager.py if needed
    def get_real_moves_for_signature(self, signature: str) -> list[np.ndarray]:
        return self.get_all_symmetric_moves(signature)

    # ===========================
    # Internal Helpers
    # ===========================
    
    def _invert_signature(self, canonical_sig: str) -> Optional[np.ndarray]:
        """Invert a sig_<hash>:<index> format signature using pre-computed map."""
        parts = canonical_sig.split(":", 1)
        base_sig = parts[0]
        
        moves = self._signature_to_moves.get(base_sig)
        if not moves:
            return None
            
        # If index is present, try to grab that specific move
        if len(parts) == 2:
            try:
                idx = int(parts[1])
                if 0 <= idx < len(moves):
                    return np.array(moves[idx].to_output(), dtype=int)
            except ValueError:
                pass
        
        # Default to first available move in that orbit
        return np.array(moves[0].to_output(), dtype=int)
    
    def _invert_legacy_int(self, canon_id: int) -> Optional[np.ndarray]:
        """Invert legacy integer canonical ID."""
        node = self._canonical_to_node.get(canon_id)
        if node:
            move = self._extract_move_from_node_obj(node)
            return np.array(move.to_output(), dtype=int)
        return None

    def _normalize_input_move(self, move_array: np.ndarray) -> Move:
        """Convert raw array to hashable Move object."""
        if move_array.ndim == 0:
            coords = (int(move_array.item()),)
        else:
            coords = tuple(int(x) for x in move_array.flatten())
        return Move(coords=coords, node_id=None)

    def _extract_move_from_node_obj(self, node: Any) -> Move:
        """Helper to create a Move object from a graph node ID."""
        # This mirrors logic in _extract_move_from_node but for retrieval
        if isinstance(node, tuple) and len(node) >= 2 and node[0] == "cell":
            try:
                coords = tuple(int(x) for x in np.atleast_1d(node[1]))
                return Move(coords=coords, node_id=node)
            except (TypeError, ValueError):
                pass
        
        fake_coord = (hash(str(node)) % 100000,)
        return Move(coords=fake_coord, node_id=node)


# ===========================
# Graph Construction
# ===========================

def state_to_graph(state: GameState) -> nx.Graph:
    """Convert game state to labeled graph."""
    if hasattr(state, "to_graph") and callable(state.to_graph):
        graph = state.to_graph()
        _ensure_labels(graph)
        return graph
    
    if hasattr(state, "board"):
        return _board_to_graph(state)
    
    return _object_to_graph(state)


def _ensure_labels(graph: nx.Graph) -> None:
    """Ensure all nodes have labels for isomorphism checks."""
    for n in graph.nodes():
        if 'label' not in graph.nodes[n]:
            value = graph.nodes[n].get('value', '')
            graph.nodes[n]['label'] = _stable_repr(value)


def _board_to_graph(state: GameState) -> nx.Graph:
    """Convert board-based state to graph (Grid Topology)."""
    graph = nx.Graph()
    board = np.asarray(state.board)
    
    # Add nodes for each cell
    it = np.nditer(board, flags=['multi_index', 'refs_ok'])
    for value in it:
        idx = it.multi_index
        node_id = ("cell", idx)
        
        # Value acts as the "Color" for isomorphism
        # If two cells have different values (X vs O), they cannot map to each other
        label = _stable_repr(value.item() if hasattr(value, 'item') else value)
        
        graph.add_node(
            node_id,
            label=label,
            coords=idx,
            value=value
        )
    
    # Add edges (Grid Adjacency)
    # This defines the geometry of the board
    for idx in np.ndindex(board.shape):
        u = ("cell", idx)
        for axis in range(len(board.shape)):
            neighbor = list(idx)
            neighbor[axis] += 1
            if neighbor[axis] < board.shape[axis]:
                v = ("cell", tuple(neighbor))
                graph.add_edge(u, v)
    
    return graph


def _object_to_graph(state: Any) -> nx.Graph:
    """Convert arbitrary object tree to graph."""
    graph = nx.Graph()
    root = ("root", 0)
    graph.add_node(root, label="Root")
    
    try:
        attributes = vars(state)
    except Exception:
        attributes = {"value": state}
    
    for key, value in attributes.items():
        attr_node = ("attr", key)
        val_node = ("val", key)
        graph.add_node(attr_node, label=f"Attr:{key}")
        graph.add_node(val_node, label=_stable_repr(value))
        graph.add_edge(root, attr_node)
        graph.add_edge(attr_node, val_node)
    
    return graph


# ===========================
# Orbit Detection
# ===========================

def _compute_orbits(graph: nx.Graph) -> dict:
    """
    Compute automorphism orbits.
    Nodes in the same orbit are topologically indistinguishable.
    """
    if NAUTY_AVAILABLE:
        try:
            return _compute_orbits_nauty(graph)
        except Exception as e:
            logger.warning(f"Nauty failed, falling back to NetworkX: {e}")
            
    return _compute_orbits_networkx(graph)


def _compute_orbits_networkx(graph: nx.Graph) -> dict:
    """Compute orbits using VF2 algorithm (NetworkX). Slow for large graphs."""
    nodes = list(graph.nodes())
    
    def node_match(n1, n2):
        return (n1.get('label', '') == n2.get('label', ''))
    
    # Isomorphism Matcher
    gm = isomorphism.GraphMatcher(graph, graph, node_match=node_match)
    
    # Find generators for the automorphism group
    # Note: iterating all isomorphisms is factorial time; 
    # checking generators is faster but complex to implement manually.
    # We use a simplified approximation here by checking connectivity of isomorphic mappings.
    
    # Optimization: Only map nodes with same label and degree
    orbit_id = {n: i for i, n in enumerate(nodes)}
    
    # For small graphs, full enumeration is okay. 
    # For larger graphs, we might need a limit.
    limit = 100 
    count = 0
    
    for mapping in gm.isomorphisms_iter():
        count += 1
        if count > limit: break
        
        # Union-Find style merge
        for u, v in mapping.items():
            root_u = orbit_id[u]
            root_v = orbit_id[v]
            if root_u != root_v:
                new_root = min(root_u, root_v)
                # Remap all nodes pointing to the old roots
                for n, r in orbit_id.items():
                    if r == root_u or r == root_v:
                        orbit_id[n] = new_root
                        
    return orbit_id


def _compute_orbits_nauty(graph: nx.Graph) -> dict:
    """Compute orbits using Nauty (Fast C implementation)."""
    nodes = list(graph.nodes())
    n = len(nodes)
    
    # Nauty requires integer coloring
    label_to_int = {}
    next_color = 0
    colors = []
    
    for node in nodes:
        label = graph.nodes[node].get('label', '')
        if label not in label_to_int:
            label_to_int[label] = next_color
            next_color += 1
        colors.append(label_to_int[label])
    
    # Create adjacency dict
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    adjacency = {i: [] for i in range(n)}
    for u, v in graph.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        adjacency[i].append(j)
        adjacency[j].append(i)
    
    # Group by color
    color_classes = {}
    for i, c in enumerate(colors):
        color_classes.setdefault(c, []).append(i)
    vertex_coloring = [sorted(color_classes[c]) for c in sorted(color_classes)]
    
    # Run Nauty
    g = pynauty.Graph(n, adjacency_dict=adjacency, vertex_coloring=vertex_coloring)
    auto_group = pynauty.autgrp(g)
    
    # auto_group[3] contains the orbits array
    raw_orbits = auto_group[3]
    
    return {nodes[i]: raw_orbits[i] for i in range(n)}


def _node_key(node, graph: nx.Graph):
    """Deterministic key for node sorting."""
    label = graph.nodes[node].get('label', '')
    degree = graph.degree[node]
    # Include neighbor labels for deeper stability
    neighbor_labels = sorted(graph.nodes[n].get('label', '') for n in graph.neighbors(node))
    return (label, degree, tuple(neighbor_labels))


# ===========================
# Canonicalization Logic
# ===========================

def canonicalize_state(state: GameState) -> CanonicalResult:
    """
    Main entry point.
    1. Converts State -> Graph
    2. Computes Orbits (Symmetries)
    3. Generates Canonical Hash
    4. Builds Mappings for Move Translations
    """
    graph = state_to_graph(state)
    
    original_board = None
    original_shape = None
    if hasattr(state, "board"):
        original_board = np.asarray(state.board)
        original_shape = original_board.shape
    
    # 1. Compute Orbits
    orbit_id = _compute_orbits(graph)
    nodes = list(graph.nodes())
    
    # 2. Sort nodes to create a canonical ordering
    # Primary key: Orbit ID (topological equivalence)
    # Secondary key: Local properties (stability)
    def sort_key(n):
        return (orbit_id[n], *_node_key(n, graph))
    
    sorted_nodes = sorted(nodes, key=sort_key)
    
    node_to_canonical = {node: idx for idx, node in enumerate(sorted_nodes)}
    canonical_to_node = {idx: node for idx, node in enumerate(sorted_nodes)}
    
    # Determine representatives for each orbit (min canonical ID in the orbit)
    orbit_to_min = {}
    for node, canon_id in node_to_canonical.items():
        oid = orbit_id[node]
        if oid not in orbit_to_min:
            orbit_to_min[oid] = canon_id
        else:
            orbit_to_min[oid] = min(orbit_to_min[oid], canon_id)
            
    orbit_representatives = {
        canon_id: orbit_to_min[orbit_id[node]]
        for node, canon_id in node_to_canonical.items()
    }
    
    # 3. Build Move Signatures
    signature_to_moves = defaultdict(list)
    move_to_signature = {}
    orbit_id_by_node = {}
    
    for node in nodes:
        oid = orbit_id[node]
        orbit_id_by_node[node] = oid
        
        # Generate a stable signature for this orbit
        # We use properties of the node + its orbit ID
        label = graph.nodes[node].get('label', '')
        degree = graph.degree[node]
        
        # Note: We include Orbit ID in signature. 
        # This ensures that structurally different nodes never share a signature.
        sig_text = f"O:{oid}|L:{label}|D:{degree}"
        sig_hash = hashlib.sha256(sig_text.encode()).hexdigest()
        signature = f"sig_{sig_hash}"
        
        # Extract Move object
        move = _extract_move_from_node(node, graph)
        
        signature_to_moves[signature].append(move)
        move_to_signature[move] = signature
    
    # 4. Generate Graph Hash
    # We hash the canonical structure (edges remapped to canonical IDs)
    canonical_edges = sorted([
        tuple(sorted((node_to_canonical[u], node_to_canonical[v])))
        for u, v in graph.edges()
    ])
    
    # The hash must also include the node labels in canonical order
    node_labels = [graph.nodes[canonical_to_node[i]].get('label', '') 
                   for i in range(len(nodes))]
    
    full_structure = {
        "nodes": node_labels,
        "edges": canonical_edges,
        "orbits": [orbit_id[canonical_to_node[i]] for i in range(len(nodes))]
    }
    
    graph_hash = hashlib.sha256(str(full_structure).encode()).hexdigest()
    
    # 5. Build Canonical Graph (for visualization/debug)
    canonical_graph = nx.Graph()
    for i in range(len(nodes)):
        canonical_graph.add_node(i, label=node_labels[i])
    for u, v in canonical_edges:
        canonical_graph.add_edge(u, v)
    
    return CanonicalResult(
        hash=graph_hash,
        graph=graph,
        canonical_graph=canonical_graph,
        _node_to_canonical=node_to_canonical,
        _canonical_to_node=canonical_to_node,
        _orbit_representatives=orbit_representatives,
        _orbit_id_by_node=orbit_id_by_node,
        _signature_to_moves=dict(signature_to_moves),
        _move_to_signature=move_to_signature,
        original_board=original_board,
        original_shape=original_shape,
    )


def _extract_move_from_node(node, graph: nx.Graph) -> Move:
    """Extract a Move object from a graph node."""
    # Priority 1: Explicit coords attribute
    coords_attr = graph.nodes[node].get('coords')
    if coords_attr is not None:
        try:
            coords = tuple(int(x) for x in np.atleast_1d(coords_attr))
            return Move(coords=coords, node_id=node)
        except (TypeError, ValueError):
            pass
    
    # Priority 2: Node ID pattern ("cell", (r, c))
    if isinstance(node, tuple) and len(node) >= 2:
        node_type, node_data = node[0], node[1]
        
        if node_type == "cell":
            if isinstance(node_data, (tuple, list, np.ndarray)):
                try:
                    coords = tuple(int(x) for x in np.atleast_1d(node_data))
                    return Move(coords=coords, node_id=node)
                except (TypeError, ValueError):
                    pass
    
    # Priority 3: Hash fallback (abstract nodes)
    fake_coord = (hash(str(node)) % 100000,)
    return Move(coords=fake_coord, node_id=node)


# ===========================
# Utilities
# ===========================

def _stable_repr(obj: Any) -> str:
    """Deterministic string representation without memory addresses."""
    if isinstance(obj, (int, float, bool, str, bytes)):
        return repr(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_stable_repr(x) for x in obj) + "]"
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: str(kv[0]))
        return "{" + ",".join(f"{_stable_repr(k)}:{_stable_repr(v)}" for k, v in items) + "}"
    if isinstance(obj, np.ndarray):
        return f"array({obj.tolist()})"
    if hasattr(obj, "item"): # Numpy scalars
        try:
            return str(obj.item())
        except Exception:
            return repr(obj)
    
    # Generic Object handling
    try:
        if hasattr(obj, "__dict__"):
            attrs = sorted(vars(obj).items())
            return obj.__class__.__name__ + "(" + ",".join(
                f"{k}:{_stable_repr(v)}" for k, v in attrs
            ) + ")"
    except Exception:
        pass
        
    return str(obj)