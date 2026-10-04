from __future__ import annotations
from numba import njit
import numpy as np
@njit(cache=True)
def _rank_window_component_sizes(
    indices: np.ndarray,
    ordered_rows: np.ndarray,
    width: int,
    edge_rank: int,
) -> np.ndarray:
    n = indices.shape[0]
    windows = (ordered_rows.size + width - 1) // width
    window_id = np.full(n, -1, dtype=np.int32)
    parent = np.full(n, -1, dtype=np.int32)
    for position in range(ordered_rows.size):
        row = ordered_rows[position]
        window_id[row] = position // width
        parent[row] = row

    for position in range(ordered_rows.size):
        row = ordered_rows[position]
        row_window = window_id[row]
        for edge in range(edge_rank):
            neighbor = indices[row, edge]
            if window_id[neighbor] != row_window:
                continue
            left = row
            while parent[left] != left:
                parent[left] = parent[parent[left]]
                left = parent[left]
            right = neighbor
            while parent[right] != right:
                parent[right] = parent[parent[right]]
                right = parent[right]
            if left != right:
                parent[right] = left

    counts = np.zeros(n, dtype=np.int32)
    largest = np.zeros(windows, dtype=np.int32)
    for position in range(ordered_rows.size):
        row = ordered_rows[position]
        root = row
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        counts[root] += 1
        window = window_id[row]
        if counts[root] > largest[window]:
            largest[window] = counts[root]
    return largest
