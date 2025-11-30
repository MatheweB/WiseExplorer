"""
Graph-automorphism canonicalizer (General Game Playing).

Refactored for robustness:
1. Auto-bridges the gap between spatial moves (coords) and graph nodes (tagged tuples).
2. Uses Weisfeiler-Lehman (WL) for stable canonical labeling.
3. Removes dependency on memory-address based repr() for consistent hashing across runs.
"""

from __future__ import annotations
import hashlib
import json
import functools
from typing import Any, Dict, Tuple, Optional, List, Union
import numpy as np
import networkx as nx

# ----------------------------
# 1. Deterministic Helpers
# ----------------------------

def _stable_repr(obj: Any) -> str:
    """
    Returns a string representation that is stable across process runs.
    Avoids 'object at 0x...' which breaks hashing.
    """
    if isinstance(obj, (int, float, bool, str, bytes)):
        return repr(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_stable_repr(x) for x in obj) + "]"
    if isinstance(obj, dict):
        # Sort keys for stability
        items = sorted(obj.items(), key=lambda x: str(x[0]))
        return "{" + ",".join(f"{_stable_repr(k)}:{_stable_repr(v)}" for k, v in items) + "}"
    if isinstance(obj, np.ndarray):
        return f"array({obj.tolist()})"
    if hasattr(obj, "item"):  # numpy scalars
        return str(obj.item())
    
    # Fallback for custom objects: use class name + dict, ignore memory address
    try:
        attrs = sorted(vars(obj).items())
        attr_str = ",".join(f"{k}:{_stable_repr(v)}" for k, v in attrs)
        return f"{obj.__class__.__name__}({attr_str})"
    except Exception:
        # Absolute fallback: just the type
        return f"<{obj.__class__.__name__}>"

def _get_node_label(G: nx.Graph, n: Any) -> str:
    return str(G.nodes[n].get('label', ''))

# ----------------------------
# 2. Graph Construction
# ----------------------------

def state_to_graph(game_state: Any) -> nx.Graph:
    """
    Converts a game state into a Labeled Graph.
    Standardizes grid-based games to use node IDs: ('cell', (r, c, ...))
    """
    # 1. Trust the game if it implements to_graph
    if hasattr(game_state, "to_graph") and callable(game_state.to_graph):
        G = game_state.to_graph()
        # Ensure labels exist
        for n in G.nodes():
            if 'label' not in G.nodes[n]:
                val = G.nodes[n].get('value', None)
                G.nodes[n]['label'] = _stable_repr(val)
        return G

    G = nx.Graph()

    # 2. Grid/Board Heuristic (Most board games)
    if hasattr(game_state, "board"):
        board = np.asarray(game_state.board)
        # Handle scalar boards or complex objects
        it = np.nditer(board, flags=['multi_index', 'refs_ok'], op_flags=['readonly'])
        
        # Add Nodes
        for v in it:
            coords = it.multi_index
            # Standardize node ID for grids
            nid = ("cell", coords) 
            # Label is the content of the cell (e.g., 'WhitePawn', '0', '1')
            G.add_node(nid, label=_stable_repr(v.item()), coords=coords)

        # Add Edges (Grid Adjacency)
        shape = board.shape
        ndim = len(shape)
        for idx in np.ndindex(shape):
            current_node = ("cell", idx)
            for axis in range(ndim):
                neighbor_idx = list(idx)
                neighbor_idx[axis] += 1
                if neighbor_idx[axis] < shape[axis]:
                    neighbor_node = ("cell", tuple(neighbor_idx))
                    G.add_edge(current_node, neighbor_node)
        
        # Add Metadata (Player turn)
        if hasattr(game_state, "current_player"):
            p_node = ("meta", "player")
            G.add_node(p_node, label=f"Player:{game_state.current_player}")
            # Connect player to all cells (global context) or just root
            # Connecting to all cells ensures player ID affects the whole graph structure
            # but connecting to a single 'root' is usually cleaner for automorphisms.
            root = ("root", 0)
            G.add_node(root, label="Root")
            G.add_edge(root, p_node)
            # Connect root to cells to unify graph
            for idx in np.ndindex(shape):
                G.add_edge(root, ("cell", idx))
        return G

    # 3. Object Fallback (Generic Python Objects)
    root = ("root", 0)
    G.add_node(root, label="Root")
    
    try:
        attributes = vars(game_state)
    except Exception:
        attributes = {"value": game_state}

    for k, v in attributes.items():
        attr_node = ("attr", k)
        G.add_node(attr_node, label=f"Attr:{k}")
        G.add_edge(root, attr_node)
        
        val_node = ("val", k)
        G.add_node(val_node, label=_stable_repr(v))
        G.add_edge(attr_node, val_node)
        
    return G

# ----------------------------
# 3. Weisfeiler-Lehman Isomorphism
# ----------------------------

def _wl_canonical_labeling(G: nx.Graph, rounds: int = None) -> Tuple[str, Dict[Any, int]]:
    """
    Standard 1-D Weisfeiler-Lehman color refinement.
    Returns: (graph_hash, mapping[original_node -> canonical_int_id])
    """
    nodes = sorted(list(G.nodes()), key=lambda n: _stable_repr(n))
    
    # Initial coloring based on labels
    colors = {n: hashlib.sha256(_get_node_label(G, n).encode()).hexdigest() for n in nodes}
    
    # Iterate refinement
    # For most game graphs, diameter is small, 5-10 rounds usually suffice.
    if rounds is None:
        rounds = max(3, len(nodes))

    for _ in range(rounds):
        new_colors = {}
        color_counts = {}
        
        for n in nodes:
            # Gather neighbor colors
            neighbor_colors = sorted([colors[nbr] for nbr in G.adj[n]])
            # Form signature: (my_color, [sorted_neighbor_colors])
            signature = colors[n] + "".join(neighbor_colors)
            # Hash signature
            new_hash = hashlib.sha256(signature.encode()).hexdigest()
            new_colors[n] = new_hash
        
        # Check for convergence (if partition hasn't changed, we are done)
        # Optimization: Just check if set of values changed size, 
        # but pure string compare is safer for debugging.
        if new_colors == colors:
            break
        colors = new_colors

    # Generate Canonical Indexing
    # Sort nodes by their final WL color, then by their stable repr (to break ties deterministically)
    sorted_nodes = sorted(nodes, key=lambda n: (colors[n], _stable_repr(n)))
    
    # Map original node -> 0..N-1
    iso_map = {node: i for i, node in enumerate(sorted_nodes)}
    
    # Create final graph hash
    # Hash of (Number of nodes + Sorted Color String + Sorted Edge List in canonical indices)
    canon_edges = []
    for u, v in G.edges():
        idx_u, idx_v = iso_map[u], iso_map[v]
        if idx_u > idx_v: 
            idx_u, idx_v = idx_v, idx_u
        canon_edges.append((idx_u, idx_v))
    canon_edges.sort()
    
    final_sig = f"N:{len(nodes)}|C:{''.join(colors[n] for n in sorted_nodes)}|E:{str(canon_edges)}"
    graph_hash = hashlib.sha256(final_sig.encode()).hexdigest()
    
    return graph_hash, iso_map

# ----------------------------
# 4. Public API
# ----------------------------

def canonicalize_state(game_state: Any) -> Dict[str, Any]:
    """
    Primary Entry Point.
    """
    G = state_to_graph(game_state)
    
    # Perform Canonicalization (WL Algorithm)
    canon_hash, iso_map = _wl_canonical_labeling(G)
    
    # Create Inverse Map
    inverse_iso_map = {v: k for k, v in iso_map.items()}
    
    # Construct Canonical Graph (for visualization/debugging)
    # Nodes are integers 0..N-1
    canon_G = nx.Graph()
    for orig_node, idx in iso_map.items():
        canon_G.add_node(idx, label=_get_node_label(G, orig_node))
    for u, v in G.edges():
        canon_G.add_edge(iso_map[u], iso_map[v])

    return {
        "hash": canon_hash,
        "iso_map": iso_map,
        "inverse_iso_map": inverse_iso_map,
        "canonical_graph": canon_G,
        "graph": G
    }

def canonicalize_move(move: Any, iso_map: Dict[Any, int]) -> Union[int, Any]:
    """
    Transforms a move (coordinates or index) into a canonical integer ID.
    
    CRITICAL LOGIC:
    1. If 'move' is a tuple/array (coords), look for a node in the graph that *contains* these coords.
    2. If 'move' is directly a node ID, look it up.
    """
    # 1. Normalize Move to something hashable (tuple) if it's a numpy array
    move_struct = move
    if isinstance(move, np.ndarray):
        if move.ndim == 0:
            move_struct = move.item()
        else:
            move_struct = tuple(move.tolist())
    
    # 2. Direct Lookup (Rare in grid games, common in abstract graphs)
    if move_struct in iso_map:
        return iso_map[move_struct]

    # 3. Coordinate Lookup (The Bridge)
    # Scan iso_map keys. If the key is ('cell', (r,c)) and move is (r,c), match them.
    # This is O(N), but N (nodes) is usually small (<1000) for board games.
    # We can optimize this by building a secondary lookup if needed.
    for node_key, canon_id in iso_map.items():
        # Heuristic: Check if node_key looks like ("cell", coords)
        if isinstance(node_key, tuple) and len(node_key) >= 2:
            if node_key[1] == move_struct:
                return canon_id
            
            # Sometimes move is [x, y] but node is ("cell", (x, y))
            # Just comparing node_key[1] covers it if move_struct is tuple
    
    # 4. If canonicalization fails (move is not referencing a board position), 
    # return original move. It implies the move is abstract (e.g. "pass").
    return move

def invert_canonical_move(canonical_move: Any, inverse_iso_map: Dict[int, Any]) -> Any:
    """
    Transforms a canonical integer ID back into the original move coordinates.
    """
    # Handle numpy wrapper
    if hasattr(canonical_move, "item"):
        idx = canonical_move.item()
    else:
        idx = canonical_move
        
    if not isinstance(idx, int):
        # Was not canonicalized (e.g. "pass")
        return canonical_move
        
    orig_node = inverse_iso_map.get(idx)
    
    if orig_node is None:
        return None
        
    # Unpack Logic: If the node is ("cell", (0,0)), return array([0,0])
    if isinstance(orig_node, tuple) and len(orig_node) >= 2 and orig_node[0] == "cell":
        return np.array(orig_node[1])
        
    # Fallback: return the object itself
    return orig_node