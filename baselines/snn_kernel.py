from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=False)
def shared_counts_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n, k = b.shape
    out = np.empty((n, k), dtype=np.int16)
    for i in range(n):
        ai = a[i]
        for r in range(k):
            j = b[i, r]
            aj = a[j]
            p = 0
            q = 0
            count = 0
            while p < k and q < k:
                if ai[p] == aj[q]:
                    count += 1
                    p += 1
                    q += 1
                elif ai[p] < aj[q]:
                    p += 1
                else:
                    q += 1
            out[i, r] = count
    return out


def warm() -> None:
    dummy = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32)
    shared_counts_numba(dummy, dummy)
