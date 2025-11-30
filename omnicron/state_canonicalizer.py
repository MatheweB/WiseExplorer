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
from typing import Any, Protocol, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism

try:
    import pynauty  # type: ignore
    NAUTY_AVAILABLE = True
except Exception:
    NAUTY_AVAILABLE = False

# ===========================
# Core Types
# ===========================

class GameState(Protocol):
    """Expected interface for game states."""
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
        # Hash only by coords for consistent lookup
        return hash(self.coords)
    
    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        # Compare only by coords - node_id is just metadata
        return self.coords == other.coords
    
    def to_output(self) -> tuple[int, ...]:
        """Return the canonical output format (always coords tuple)."""
        return self.coords


@dataclass(frozen=True)
class CanonicalResult:
    """Result of state canonicalization with orbit detection."""
    hash: str
    graph: nx.Graph
    canonical_graph: nx.Graph
    
    # Core mappings
    _node_to_canonical: dict[tuple, int]  # NodeID -> canonical int
    _canonical_to_node: dict[int, tuple]  # canonical int -> NodeID
    _orbit_representatives: dict[int, int]  # canonical int -> orbit representative
    _orbit_id_by_node: dict[tuple, int]  # NodeID -> orbit ID (NEW - needed for reconstruction)
    
    # Signature mappings (the key to move translation)
    _signature_to_moves: dict[str, list[Move]]  # signature -> all equivalent moves
    _move_to_signature: dict[Move, str]  # move -> its signature
    
    # Debug info
    original_board: Optional[np.ndarray] = None
    original_shape: Optional[tuple] = None
    
    # ===========================
    # Public API
    # ===========================
    
    def canonicalize_move(self, move: Any) -> str:
        """
        Convert any move format to a canonical signature.
        
        Args:
            move: Can be:
                - tuple of ints: (row, col) or (x, y, z)
                - numpy array: np.array([row, col])
                - NodeID: ("cell", coords) or ("cell", "A")
        
        Returns:
            Canonical signature string like "sig_<hash>:<index>"
            where index distinguishes between symmetric equivalent moves.
            
            For moves not in the graph, returns "move_<coords>" format
            that can be inverted back to the original coordinates.
        """
        normalized = self._normalize_input_move(move)
        
        # Try to find this move in our mappings
        if normalized in self._move_to_signature:
            sig = self._move_to_signature[normalized]
            moves = self._signature_to_moves[sig]
            
            # Find the index of this specific move variant
            try:
                idx = moves.index(normalized)
                return f"{sig}:{idx}"
            except ValueError:
                # Shouldn't happen, but fallback to base signature
                return sig
        
        # Move not recognized in the graph
        # Return a parseable format: "move_<coords>" so we can invert it later
        coords_str = "_".join(str(c) for c in normalized.coords)
        return f"move_{coords_str}"
    
    def invert_move(self, canonical_sig: str | int) -> Optional[tuple[int, ...] | np.ndarray]:
        """
        Convert canonical signature back to original move coordinates.
        
        Args:
            canonical_sig: Signature like "sig_<hash>:<index>" or "sig_<hash>"
                          or "move_<coords>" for non-graph moves
                          or legacy integer canonical ID
                          or raw hash (legacy - returns None)
        
        Returns:
            Numpy array of coordinates for compatibility with manager.py,
            or None if not found or if it's an abstract move
        """
        # Handle legacy integer format (orbit_N or raw int)
        if isinstance(canonical_sig, (int, np.integer)):
            return self._invert_legacy_int(canonical_sig)
        
        if not isinstance(canonical_sig, str):
            return None
        
        # Handle "orbit_N" format (legacy)
        if canonical_sig.startswith("orbit_"):
            try:
                canon_id = int(canonical_sig.split("_", 1)[1])
                return self._invert_legacy_int(canon_id)
            except (ValueError, IndexError):
                return None
        
        # Handle "move_<coords>" format (non-graph moves)
        if canonical_sig.startswith("move_"):
            try:
                coords_str = canonical_sig[5:]  # Remove "move_" prefix
                coords = tuple(int(c) for c in coords_str.split("_"))
                return np.array(coords, dtype=int)
            except (ValueError, IndexError):
                return None
        
        # Handle signature format "sig_<hash>" or "sig_<hash>:<index>"
        if canonical_sig.startswith("sig_"):
            return self._invert_signature(canonical_sig)
        
        # Raw hash (64 char hex) - these are abstract/unknown moves from old format
        # They can't be inverted to coordinates
        if len(canonical_sig) == 64 and all(c in '0123456789abcdef' for c in canonical_sig):
            return None
        
        # Unknown format
        return None
    
    def _invert_signature(self, canonical_sig: str) -> Optional[np.ndarray]:
        """Invert a sig_<hash>:<index> format signature."""
        # Parse signature
        parts = canonical_sig.split(":", 1)
        base_sig = parts[0]
        
        # ALWAYS use node mapping reconstruction since _signature_to_moves
        # may be from a different CanonicalResult instance
        # This is more reliable than trusting the Move objects
        result = self._invert_from_node_mapping(base_sig, parts)
        
        if result is not None:
            return result
        
        # Final fallback: try the pre-built mapping if it exists
        # (this works if we're using the same CanonicalResult instance)
        if base_sig in self._signature_to_moves:
            moves = self._signature_to_moves[base_sig]
            if moves:
                target_move = None
                if len(parts) == 2:
                    try:
                        idx = int(parts[1])
                        if 0 <= idx < len(moves):
                            target_move = moves[idx]
                    except ValueError:
                        pass
                
                if target_move is None:
                    target_move = moves[0]
                
                return np.array(target_move.to_output(), dtype=int)
        
        return None
    
    def _invert_from_node_mapping(self, base_sig: str, parts: list[str]) -> Optional[np.ndarray]:
        """
        Fallback: try to find a node that would produce this signature.
        This handles cases where we have the same state but different CanonicalResult instances.
        """
        # Safety check: ensure graph has nodes
        if not self.graph or self.graph.number_of_nodes() == 0:
            return None
        
        # We need to reconstruct which nodes would have this signature
        # by recomputing signatures for all nodes
        sig_to_coords = defaultdict(list)
        
        # Rebuild signature mapping by recomputing from the graph
        for node in self.graph.nodes():
            label = self.graph.nodes[node].get('label', '')
            degree = self.graph.degree[node]
            value = self.graph.nodes[node].get('value', '')
            neighbor_labels = sorted(self.graph.nodes[n].get('label', '') for n in self.graph.neighbors(node))
            
            # Get orbit ID for this node
            canon_id = self._node_to_canonical.get(node)
            if canon_id is None:
                continue
            oid = self._orbit_id_by_node.get(node, 0)
            
            sig_text = f"O:{oid}|L:{label}|D:{degree}|V:{value}|N:{tuple(neighbor_labels)}"
            sig_hash = hashlib.sha256(sig_text.encode()).hexdigest()
            signature = f"sig_{sig_hash}"
            
            # Collect ALL nodes with coordinates
            if isinstance(node, tuple) and len(node) >= 2 and node[0] == "cell":
                coords_data = node[1]
                if isinstance(coords_data, (tuple, list)):
                    try:
                        coords = tuple(int(x) for x in coords_data)
                        sig_to_coords[signature].append(coords)
                    except (TypeError, ValueError):
                        continue
        
        # Now check if we found matching nodes
        matching_coords = sig_to_coords.get(base_sig, [])
        if not matching_coords:
            return None
        
        # Sort for deterministic ordering
        matching_coords.sort()
        
        # If index specified, try to use it
        if len(parts) == 2:
            try:
                idx = int(parts[1])
                if 0 <= idx < len(matching_coords):
                    return np.array(matching_coords[idx], dtype=int)
            except ValueError:
                pass
        
        # Return first match as fallback
        return np.array(matching_coords[0], dtype=int)
    
    def _invert_legacy_int(self, canon_id: int) -> Optional[np.ndarray]:
        """Invert legacy integer canonical ID to coordinates."""
        node = self._canonical_to_node.get(canon_id)
        if node is None:
            return None
        
        # Extract coordinates from node
        if isinstance(node, tuple) and len(node) >= 2 and node[0] == "cell":
            coords = node[1]
            if isinstance(coords, (tuple, list)):
                try:
                    return np.array(tuple(int(x) for x in coords), dtype=int)
                except (TypeError, ValueError):
                    pass
        
        return None
    
    def get_all_symmetric_moves(self, canonical_sig: str) -> list[np.ndarray]:
        """
        Get all moves that are symmetric/equivalent to this canonical signature.
        
        Args:
            canonical_sig: Signature (with or without index)
        
        Returns:
            List of all equivalent move coordinates as numpy arrays
        """
        base_sig = canonical_sig.split(":", 1)[0]
        moves = self._signature_to_moves.get(base_sig, [])
        return [np.array(m.to_output(), dtype=int) for m in moves]
    
    # ===========================
    # Internal helpers
    # ===========================
    
    def _normalize_input_move(self, move: Any) -> Move:
        """
        Convert any input move format to our internal Move representation.
        """
        # Case 1: numpy array -> extract coordinates
        if isinstance(move, np.ndarray):
            if move.ndim == 0:
                # Scalar array
                coords = (int(move.item()),)
            else:
                coords = tuple(int(x) for x in move.flatten())
            return Move(coords=coords, node_id=None)
        
        # Case 2: tuple/list of integers -> coordinates
        if isinstance(move, (list, tuple)):
            # Check if it's a NodeID like ("cell", coords)
            if len(move) >= 2 and isinstance(move[0], str):
                node_type, node_data = move[0], move[1]
                
                if node_type == "cell":
                    # Extract coordinates from cell
                    if isinstance(node_data, (np.ndarray, list, tuple)):
                        try:
                            coords = tuple(int(x) for x in np.atleast_1d(node_data))
                            return Move(coords=coords, node_id=move)
                        except (TypeError, ValueError):
                            pass
                    
                    # Symbolic cell like ("cell", "A")
                    # Use hash as fake coordinate
                    fake_coord = (hash(str(move)) % 100000,)
                    return Move(coords=fake_coord, node_id=move)
                
                # Other node types
                fake_coord = (hash(str(move)) % 100000,)
                return Move(coords=fake_coord, node_id=move)
            
            # Just a tuple of numbers
            try:
                coords = tuple(int(x) for x in move)
                return Move(coords=coords, node_id=None)
            except (TypeError, ValueError):
                # Not numeric, treat as symbolic
                fake_coord = (hash(str(move)) % 100000,)
                return Move(coords=fake_coord, node_id=move)
        
        # Case 3: anything else -> create fake coordinate from hash
        fake_coord = (hash(str(move)) % 100000,)
        return Move(coords=fake_coord, node_id=move)


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
    """Ensure all nodes have labels."""
    for n in graph.nodes():
        if 'label' not in graph.nodes[n]:
            value = graph.nodes[n].get('value', '')
            graph.nodes[n]['label'] = _stable_repr(value)


def _board_to_graph(state: GameState) -> nx.Graph:
    """Convert board-based state to graph."""
    graph = nx.Graph()
    board = np.asarray(state.board)
    
    # Add nodes for each cell
    for idx in np.ndindex(board.shape):
        node_id = ("cell", idx)
        value = board[idx].item() if hasattr(board[idx], 'item') else board[idx]
        graph.add_node(
            node_id,
            label=_stable_repr(value),
            coords=idx,
            value=value
        )
    
    # Add edges (adjacency)
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
    """Convert arbitrary object to graph representation."""
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

def _compute_orbits(graph: nx.Graph, use_nauty: bool = NAUTY_AVAILABLE) -> dict:
    """
    Compute automorphism orbits using nauty or NetworkX.
    Returns dict mapping each node to its orbit ID.
    """
    if use_nauty and NAUTY_AVAILABLE:
        return _compute_orbits_nauty(graph)
    return _compute_orbits_networkx(graph)


def _compute_orbits_networkx(graph: nx.Graph) -> dict:
    """Compute orbits using NetworkX automorphism detection."""
    nodes = list(graph.nodes())
    
    def node_match(n1, n2):
        return (n1.get('label', '') == n2.get('label', '') and 
                n1.get('value', '') == n2.get('value', ''))
    
    gm = isomorphism.GraphMatcher(graph, graph, node_match=node_match)
    automorphisms = list(gm.isomorphisms_iter())
    
    # Union-find to group nodes in same orbit
    parent = {node: node for node in nodes}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # Deterministic choice of root
            if _node_key(ra, graph) <= _node_key(rb, graph):
                parent[rb] = ra
            else:
                parent[ra] = rb
    
    # Apply all automorphisms
    for auto in automorphisms:
        for node, mapped in auto.items():
            union(node, mapped)
    
    # Assign orbit IDs
    orbit_map = {}
    orbit_id = {}
    next_id = 0
    
    for node in nodes:
        root = find(node)
        if root not in orbit_map:
            orbit_map[root] = next_id
            next_id += 1
        orbit_id[node] = orbit_map[root]
    
    return orbit_id


def _compute_orbits_nauty(graph: nx.Graph) -> dict:
    """Compute orbits using pynauty."""
    nodes = list(graph.nodes())
    n = len(nodes)
    
    # Color nodes by label
    label_to_color = {}
    colors = []
    for node in nodes:
        label = graph.nodes[node].get('label', '')
        if label not in label_to_color:
            label_to_color[label] = len(label_to_color)
        colors.append(label_to_color[label])
    
    # Build adjacency
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
    
    # Run nauty
    g = pynauty.Graph(n, adjacency_dict=adjacency, vertex_coloring=vertex_coloring)
    auto_group = pynauty.autgrp(g)
    orbits = auto_group[3]
    
    return {nodes[i]: orbits[i] for i in range(n)}


def _node_key(node, graph: nx.Graph):
    """Deterministic key for node comparison."""
    label = graph.nodes[node].get('label', '')
    degree = graph.degree[node]
    neighbor_labels = sorted(graph.nodes[n].get('label', '') for n in graph.neighbors(node))
    return (label, degree, tuple(neighbor_labels))


# ===========================
# Canonicalization
# ===========================

def canonicalize_state(state: GameState) -> CanonicalResult:
    """
    Main entry point: canonicalize a game state.
    
    Returns CanonicalResult with all mappings needed for move translation.
    """
    graph = state_to_graph(state)
    
    # Store original board info
    original_board = None
    original_shape = None
    if hasattr(state, "board"):
        original_board = np.asarray(state.board)
        original_shape = original_board.shape
    
    # Compute orbits
    orbit_id = _compute_orbits(graph)
    nodes = list(graph.nodes())
    
    # Sort nodes deterministically within orbits
    def sort_key(n):
        return (orbit_id[n], *_node_key(n, graph))
    
    sorted_nodes = sorted(nodes, key=sort_key)
    
    # Create canonical mappings
    node_to_canonical = {node: idx for idx, node in enumerate(sorted_nodes)}
    canonical_to_node = {idx: node for idx, node in enumerate(sorted_nodes)}
    
    # Find orbit representatives (min canonical ID per orbit)
    orbit_to_min = {}
    for node, canon_id in node_to_canonical.items():
        oid = orbit_id[node]
        orbit_to_min[oid] = min(orbit_to_min.get(oid, canon_id), canon_id)
    
    orbit_representatives = {
        canon_id: orbit_to_min[orbit_id[node]]
        for node, canon_id in node_to_canonical.items()
    }
    
    # Create signatures and move mappings
    signature_to_moves = {}
    move_to_signature = {}
    
    for node, canon_id in node_to_canonical.items():
        # Build signature from orbit + local structure
        label = graph.nodes[node].get('label', '')
        degree = graph.degree[node]
        value = graph.nodes[node].get('value', '')
        neighbor_labels = sorted(graph.nodes[n].get('label', '') for n in graph.neighbors(node))
        oid = orbit_id[node]
        
        sig_text = f"O:{oid}|L:{label}|D:{degree}|V:{value}|N:{tuple(neighbor_labels)}"
        sig_hash = hashlib.sha256(sig_text.encode()).hexdigest()
        signature = f"sig_{sig_hash}"
        
        # Extract move from node
        move = _extract_move_from_node(node, graph)
        
        # Store bidirectional mapping
        signature_to_moves.setdefault(signature, []).append(move)
        move_to_signature[move] = signature
    
    # Build canonical graph for debugging
    canonical_graph = nx.Graph()
    for node, canon_id in node_to_canonical.items():
        label = graph.nodes[node].get('label', '')
        canonical_graph.add_node(canon_id, label=label)
    for u, v in graph.edges():
        canonical_graph.add_edge(node_to_canonical[u], node_to_canonical[v])
    
    # Compute final hash
    canonical_edges = sorted([
        (min(node_to_canonical[u], node_to_canonical[v]),
         max(node_to_canonical[u], node_to_canonical[v]))
        for u, v in graph.edges()
    ])
    orbit_structure = "_".join(str(orbit_representatives[i]) for i in range(len(nodes)))
    signature = f"N:{len(nodes)}|O:{orbit_structure}|E:{canonical_edges}"
    graph_hash = hashlib.sha256(signature.encode()).hexdigest()
    
    return CanonicalResult(
        hash=graph_hash,
        graph=graph,
        canonical_graph=canonical_graph,
        _node_to_canonical=node_to_canonical,
        _canonical_to_node=canonical_to_node,
        _orbit_representatives=orbit_representatives,
        _orbit_id_by_node=orbit_id,  # Store the orbit IDs
        _signature_to_moves=signature_to_moves,
        _move_to_signature=move_to_signature,
        original_board=original_board,
        original_shape=original_shape,
    )


def _extract_move_from_node(node, graph: nx.Graph) -> Move:
    """
    Extract a Move object from a graph node.
    Handles both coordinate-based and symbolic nodes.
    """
    # Try to get coordinates from node attributes
    coords_attr = graph.nodes[node].get('coords')
    if coords_attr is not None:
        try:
            coords = tuple(int(x) for x in np.atleast_1d(coords_attr))
            return Move(coords=coords, node_id=node)
        except (TypeError, ValueError):
            pass
    
    # Try to extract from NodeID structure
    if isinstance(node, tuple) and len(node) >= 2:
        node_type, node_data = node[0], node[1]
        
        if node_type == "cell":
            # Cell with coordinates
            if isinstance(node_data, (tuple, list, np.ndarray)):
                try:
                    coords = tuple(int(x) for x in np.atleast_1d(node_data))
                    return Move(coords=coords, node_id=node)
                except (TypeError, ValueError):
                    pass
            
            # Symbolic cell
            fake_coord = (hash(str(node)) % 100000,)
            return Move(coords=fake_coord, node_id=node)
    
    # Fallback: use hash as coordinate
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
    if hasattr(obj, "item"):
        try:
            return str(obj.item())
        except Exception:
            return repr(obj)
    try:
        attrs = sorted(vars(obj).items())
        return obj.__class__.__name__ + "(" + ",".join(
            f"{k}:{_stable_repr(v)}" for k, v in attrs
        ) + ")"
    except Exception:
        return f"<{obj.__class__.__name__}>"