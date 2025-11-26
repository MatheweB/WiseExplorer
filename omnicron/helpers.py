import numpy as np
import hashlib

def hash_array(arr: np.ndarray) -> int:
    """Consistent hash for numpy array."""
    return int(hashlib.sha256(arr.tobytes()).hexdigest(), 16) % (2 ** 31)

def serialize_array(arr: np.ndarray):
    """Serialize numpy array to bytes and metadata."""
    return arr.tobytes(), str(arr.dtype), arr.shape

def deserialize_array(b: bytes, dtype: str, shape):
    """Deserialize bytes to numpy array."""
    return np.frombuffer(b, dtype=dtype).reshape(shape)
