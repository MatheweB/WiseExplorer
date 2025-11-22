# serializers.py
import hashlib
import json
import numpy as np


def serialize_array(arr: np.ndarray):
    """Convert numpy array → (bytes, dtype_str, dtype_str2, json_shape)."""
    arr = np.asarray(arr)
    return (arr.tobytes(), str(arr.dtype), arr.dtype.str, json.dumps(arr.shape))


def deserialize_array(b: bytes, dtype_str2: str, shape_json: str):
    """Convert BLOB + dtype info → numpy array."""
    dt = np.dtype(dtype_str2)
    shape = tuple(json.loads(shape_json))
    arr = np.frombuffer(b, dtype=dt)
    return arr.reshape(shape)


def hash_array(arr: np.ndarray) -> int:
    """Stable 64-bit hash of board or move."""
    h = hashlib.blake2b(arr.tobytes(), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=True)
