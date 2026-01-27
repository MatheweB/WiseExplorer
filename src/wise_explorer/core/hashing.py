"""
Board hashing utilities - optimized for int8 arrays.
"""

import hashlib
import numpy as np


def hash_board(board: np.ndarray) -> str:
    """
    Fast hash for board state.
    
    Optimized for contiguous int8/int32 arrays (direct tobytes).
    Falls back to repr for object arrays.
    """
    if board.dtype == np.object_:
        # Legacy object arrays - slower path
        data = repr(board.tolist()).encode()
    else:
        # Fast path for numeric arrays
        data = board.tobytes()
    
    return hashlib.sha256(data).hexdigest()[:16]