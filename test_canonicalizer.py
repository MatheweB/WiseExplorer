"""
Test script to verify orbit-based canonicalization is working correctly.
Tests if symmetric moves on an empty Tic-Tac-Toe board share the same signature.

Run from project root: python -m pytest test_symmetry.py -v -s
Or just: python test_symmetry.py
"""

import sys
import os
import numpy as np
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omnicron.state_canonicalizer import canonicalize_state

# Simple Tic-Tac-Toe state for testing
@dataclass
class TicTacToeState:
    board: np.ndarray
    current_player: int = 0
    
    def clone(self):
        return TicTacToeState(board=self.board.copy(), current_player=self.current_player)


def test_empty_board_symmetry():
    """Test that symmetric positions on empty board have same signature."""
    print("=" * 80)
    print("TEST 1: Empty Board Symmetry")
    print("=" * 80)
    
    # Create empty 3x3 board
    state = TicTacToeState(board=np.zeros((3, 3), dtype=int))
    
    # Canonicalize
    canonical = canonicalize_state(state)
    
    print(f"\nBoard hash: {canonical.hash}")
    print(f"Graph has {canonical.graph.number_of_nodes()} nodes")
    print(f"Signatures: {len(canonical._signature_to_moves)} unique signatures")
    
    # Test all 9 positions
    positions = {
        "corners": [(0, 0), (0, 2), (2, 0), (2, 2)],
        "edges": [(0, 1), (1, 0), (1, 2), (2, 1)],
        "center": [(1, 1)]
    }
    
    results = {}
    
    print("\n" + "-" * 80)
    print("Position Analysis:")
    print("-" * 80)
    
    for group_name, coords_list in positions.items():
        print(f"\n{group_name.upper()}:")
        group_sigs = []
        
        for coords in coords_list:
            move = np.array(coords)
            sig = canonical.canonicalize_move(move)
            base_sig = sig.split(":")[0]  # Remove index
            group_sigs.append(base_sig)
            
            # Try to invert
            inverted = canonical.invert_move(sig)
            
            print(f"  {coords} → {sig}")
            print(f"    Base: {base_sig[:20]}...")
            print(f"    Inverts to: {inverted}")
            
            results[coords] = sig
        
        # Check if all in group have same base signature
        unique_sigs = set(group_sigs)
        if len(unique_sigs) == 1:
            print(f"  ✓ All {group_name} share the SAME signature")
        else:
            print(f"  ✗ {group_name} have DIFFERENT signatures: {len(unique_sigs)} unique")
            print(f"    Signatures: {unique_sigs}")
    
    return results


def test_with_x_in_corner():
    """Test with X in corner - should break some symmetries."""
    print("\n\n" + "=" * 80)
    print("TEST 2: Board with X in (0,0)")
    print("=" * 80)
    
    # Board with X in top-left corner
    board = np.zeros((3, 3), dtype=int)
    board[0, 0] = 1  # X = 1
    state = TicTacToeState(board=board, current_player=1)
    
    print("\nBoard:")
    print(board)
    
    canonical = canonicalize_state(state)
    print(f"\nBoard hash: {canonical.hash}")
    print(f"Signatures: {len(canonical._signature_to_moves)} unique signatures")
    
    # Test remaining corners
    print("\n" + "-" * 80)
    print("Remaining corners:")
    print("-" * 80)
    
    corners = [(0, 2), (2, 0), (2, 2)]
    corner_sigs = []
    
    for coords in corners:
        move = np.array(coords)
        sig = canonical.canonicalize_move(move)
        base_sig = sig.split(":")[0]
        corner_sigs.append(base_sig)
        
        inverted = canonical.invert_move(sig)
        print(f"  {coords} → {sig[:30]}... → inverts to {inverted}")
    
    unique_sigs = set(corner_sigs)
    print(f"\nUnique signatures among remaining corners: {len(unique_sigs)}")
    if len(unique_sigs) == 1:
        print("  ✓ Still symmetric (all share same signature)")
    else:
        print(f"  → Symmetry broken into {len(unique_sigs)} groups")


def test_orbit_details():
    """Detailed orbit analysis."""
    print("\n\n" + "=" * 80)
    print("TEST 3: Detailed Orbit Analysis")
    print("=" * 80)
    
    state = TicTacToeState(board=np.zeros((3, 3), dtype=int))
    canonical = canonicalize_state(state)
    
    print(f"\nOrbit representatives: {len(set(canonical._orbit_representatives.values()))} unique orbits")
    
    # Group nodes by orbit
    from collections import defaultdict
    orbits = defaultdict(list)
    
    for node, canon_id in canonical._node_to_canonical.items():
        orbit_rep = canonical._orbit_representatives[canon_id]
        if isinstance(node, tuple) and len(node) >= 2 and node[0] == "cell":
            coords = node[1]
            orbits[orbit_rep].append(coords)
    
    print("\nOrbit groups:")
    for orbit_id, coords_list in sorted(orbits.items()):
        print(f"  Orbit {orbit_id}: {coords_list}")
    
    # Check signature mapping
    print("\n" + "-" * 80)
    print("Signature → Coordinates mapping:")
    print("-" * 80)
    
    for sig, moves in sorted(canonical._signature_to_moves.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        coords = [m.coords for m in moves]
        print(f"  {sig[:20]}... → {len(moves)} moves: {coords}")


def test_record_and_retrieve():
    """Test full record/retrieve cycle."""
    print("\n\n" + "=" * 80)
    print("TEST 4: Record and Retrieve Simulation")
    print("=" * 80)
    
    try:
        from omnicron.manager import GameMemory, db
        from agent.agent import State
    except ImportError as e:
        print(f"  ⚠ Skipping test (imports not available): {e}")
        return
    
    import tempfile
    
    # Create temp database
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        db_path = f.name
    
    try:
        memory = GameMemory(db_path=db_path)
        
        # Empty board
        state = TicTacToeState(board=np.zeros((3, 3), dtype=int))
        
        print("\nRecording outcomes for corners:")
        # Record wins for each corner
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        for coords in corners:
            move = np.array(coords)
            success = memory.record_outcome(
                game_id="tictactoe",
                state=state,
                move=move,
                acting_player=0,
                outcome=State.WIN
            )
            print(f"  {coords}: {'✓' if success else '✗'}")
        
        # Now retrieve best move
        print("\nRetrieving best move:")
        best_move = memory.get_best_move("tictactoe", state, debug=False)
        print(f"  Best move: {best_move}")
        
        # Check all corner stats
        print("\nChecking stats for all corners:")
        canonical = memory._get_canonical_cached(state)
        
        for coords in corners:
            move = np.array(coords)
            sig = canonical.canonicalize_move(move)
            move_hash = memory._hash_canonical_move(sig)
            
            cursor = db.execute_sql(
                "SELECT win_count, loss_count FROM play_stats WHERE move_hash = ?",
                (move_hash,)
            )
            result = cursor.fetchone()
            if result:
                print(f"  {coords} ({move_hash[:20]}...): wins={result[0]}, losses={result[1]}")
            else:
                print(f"  {coords} ({move_hash[:20]}...): NOT FOUND")
        
        memory.close()
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ORBIT-BASED CANONICALIZATION TEST SUITE")
    print("=" * 80)
    
    # Run tests
    test_empty_board_symmetry()
    test_with_x_in_corner()
    test_orbit_details()
    test_record_and_retrieve()
    
    print("\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)
    print("\nExpected Results:")
    print("  • Empty board: 4 corners should share signature")
    print("  • Empty board: 4 edges should share signature")
    print("  • Empty board: 1 center (unique)")
    print("  • With X: Some symmetries broken")
    print("  • Record/Retrieve: All corners should have wins=1 under SAME move_hash")
    print()
"""
Test script to verify orbit-based canonicalization is working correctly.
Tests if symmetric moves on an empty Tic-Tac-Toe board share the same signature.
"""

import numpy as np
from dataclasses import dataclass
from omnicron.state_canonicalizer import canonicalize_state

# Simple Tic-Tac-Toe state for testing
@dataclass
class TicTacToeState:
    board: np.ndarray
    current_player: int = 0
    
    def clone(self):
        return TicTacToeState(board=self.board.copy(), current_player=self.current_player)


def test_empty_board_symmetry():
    """Test that symmetric positions on empty board have same signature."""
    print("=" * 80)
    print("TEST 1: Empty Board Symmetry")
    print("=" * 80)
    
    # Create empty 3x3 board
    state = TicTacToeState(board=np.zeros((3, 3), dtype=int))
    
    # Canonicalize
    canonical = canonicalize_state(state)
    
    print(f"\nBoard hash: {canonical.hash}")
    print(f"Graph has {canonical.graph.number_of_nodes()} nodes")
    print(f"Signatures: {len(canonical._signature_to_moves)} unique signatures")
    
    # Test all 9 positions
    positions = {
        "corners": [(0, 0), (0, 2), (2, 0), (2, 2)],
        "edges": [(0, 1), (1, 0), (1, 2), (2, 1)],
        "center": [(1, 1)]
    }
    
    results = {}
    
    print("\n" + "-" * 80)
    print("Position Analysis:")
    print("-" * 80)
    
    for group_name, coords_list in positions.items():
        print(f"\n{group_name.upper()}:")
        group_sigs = []
        
        for coords in coords_list:
            move = np.array(coords)
            sig = canonical.canonicalize_move(move)
            base_sig = sig.split(":")[0]  # Remove index
            group_sigs.append(base_sig)
            
            # Try to invert
            inverted = canonical.invert_move(sig)
            
            print(f"  {coords} → {sig}")
            print(f"    Base: {base_sig[:20]}...")
            print(f"    Inverts to: {inverted}")
            
            results[coords] = sig
        
        # Check if all in group have same base signature
        unique_sigs = set(group_sigs)
        if len(unique_sigs) == 1:
            print(f"  ✓ All {group_name} share the SAME signature")
        else:
            print(f"  ✗ {group_name} have DIFFERENT signatures: {len(unique_sigs)} unique")
            print(f"    Signatures: {unique_sigs}")
    
    return results


def test_with_x_in_corner():
    """Test with X in corner - should break some symmetries."""
    print("\n\n" + "=" * 80)
    print("TEST 2: Board with X in (0,0)")
    print("=" * 80)
    
    # Board with X in top-left corner
    board = np.zeros((3, 3), dtype=int)
    board[0, 0] = 1  # X = 1
    state = TicTacToeState(board=board, current_player=1)
    
    print("\nBoard:")
    print(board)
    
    canonical = canonicalize_state(state)
    print(f"\nBoard hash: {canonical.hash}")
    print(f"Signatures: {len(canonical._signature_to_moves)} unique signatures")
    
    # Test remaining corners
    print("\n" + "-" * 80)
    print("Remaining corners:")
    print("-" * 80)
    
    corners = [(0, 2), (2, 0), (2, 2)]
    corner_sigs = []
    
    for coords in corners:
        move = np.array(coords)
        sig = canonical.canonicalize_move(move)
        base_sig = sig.split(":")[0]
        corner_sigs.append(base_sig)
        
        inverted = canonical.invert_move(sig)
        print(f"  {coords} → {sig[:30]}... → inverts to {inverted}")
    
    unique_sigs = set(corner_sigs)
    print(f"\nUnique signatures among remaining corners: {len(unique_sigs)}")
    if len(unique_sigs) == 1:
        print("  ✓ Still symmetric (all share same signature)")
    else:
        print(f"  → Symmetry broken into {len(unique_sigs)} groups")


def test_orbit_details():
    """Detailed orbit analysis."""
    print("\n\n" + "=" * 80)
    print("TEST 3: Detailed Orbit Analysis")
    print("=" * 80)
    
    state = TicTacToeState(board=np.zeros((3, 3), dtype=int))
    canonical = canonicalize_state(state)
    
    print(f"\nOrbit representatives: {len(set(canonical._orbit_representatives.values()))} unique orbits")
    
    # Group nodes by orbit
    from collections import defaultdict
    orbits = defaultdict(list)
    
    for node, canon_id in canonical._node_to_canonical.items():
        orbit_rep = canonical._orbit_representatives[canon_id]
        if isinstance(node, tuple) and len(node) >= 2 and node[0] == "cell":
            coords = node[1]
            orbits[orbit_rep].append(coords)
    
    print("\nOrbit groups:")
    for orbit_id, coords_list in sorted(orbits.items()):
        print(f"  Orbit {orbit_id}: {coords_list}")
    
    # Check signature mapping
    print("\n" + "-" * 80)
    print("Signature → Coordinates mapping:")
    print("-" * 80)
    
    for sig, moves in sorted(canonical._signature_to_moves.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        coords = [m.coords for m in moves]
        print(f"  {sig[:20]}... → {len(moves)} moves: {coords}")


def test_record_and_retrieve():
    """Test full record/retrieve cycle."""
    print("\n\n" + "=" * 80)
    print("TEST 4: Record and Retrieve Simulation")
    print("=" * 80)
    
    from omnicron.manager import GameMemory
    from agent.agent import State
    import tempfile
    import os
    
    # Create temp database
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        db_path = f.name
    
    try:
        memory = GameMemory(db_path=db_path)
        
        # Empty board
        state = TicTacToeState(board=np.zeros((3, 3), dtype=int))
        
        print("\nRecording outcomes for corners:")
        # Record wins for each corner
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        for coords in corners:
            move = np.array(coords)
            success = memory.record_outcome(
                game_id="tictactoe",
                state=state,
                move=move,
                acting_player=0,
                outcome=State.WIN
            )
            print(f"  {coords}: {'✓' if success else '✗'}")
        
        # Now retrieve best move
        print("\nRetrieving best move:")
        best_move = memory.get_best_move("tictactoe", state, debug=False)
        print(f"  Best move: {best_move}")
        
        # Check all corner stats
        print("\nChecking stats for all corners:")
        canonical = memory._get_canonical_cached(state)
        
        from omnicron.manager import db
        for coords in corners:
            move = np.array(coords)
            sig = canonical.canonicalize_move(move)
            move_hash = memory._hash_canonical_move(sig)
            
            cursor = db.execute_sql(
                "SELECT win_count, loss_count FROM play_stats WHERE move_hash = ?",
                (move_hash,)
            )
            result = cursor.fetchone()
            if result:
                print(f"  {coords} ({move_hash[:20]}...): wins={result[0]}, losses={result[1]}")
            else:
                print(f"  {coords} ({move_hash[:20]}...): NOT FOUND")
        
        memory.close()
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ORBIT-BASED CANONICALIZATION TEST SUITE")
    print("=" * 80)
    
    # Run tests
    test_empty_board_symmetry()
    test_with_x_in_corner()
    test_orbit_details()
    test_record_and_retrieve()
    
    print("\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)
    print("\nExpected Results:")
    print("  • Empty board: 4 corners should share signature")
    print("  • Empty board: 4 edges should share signature")
    print("  • Empty board: 1 center (unique)")
    print("  • With X: Some symmetries broken")
    print("  • Record/Retrieve: All corners should have wins=1 under SAME move_hash")
    print()