"""Shared utilities for the planning helpers."""
import sys

import numpy as np


def get_true_size(obj, seen=None):
    """Recursively find the true memory footprint of an object, in bytes,
    correctly accounting for the buffers of NumPy arrays.

    `sys.getsizeof` on an ndarray already includes its data buffer when the
    array owns it, and only the object header for a view (whose buffer belongs
    to its base) -- so it is already correct and we must not add `obj.nbytes`
    on top, which would double-count the buffer of every base array. For
    containers and objects with a __dict__ we recurse, tracking ids in `seen`
    so shared references are counted once.
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, np.ndarray):
        pass  # getsizeof already counts the buffer (or just the header, for a view)
    elif hasattr(obj, "__dict__"):
        size += get_true_size(obj.__dict__, seen)
    elif isinstance(obj, dict):
        size += sum(get_true_size(v, seen) + get_true_size(k, seen)
                    for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(get_true_size(item, seen) for item in obj)

    return size
