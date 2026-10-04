from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class KNNGraph:
    """Dense row-wise kNN graph, excluding self-neighbours.

    Rows are expected to be sorted by nondecreasing distance. The graph builder
    guarantees this invariant. External graphs may be checked once with
    :meth:`validate`; clustering hot paths do not repeatedly sort or validate.
    """

    indices: np.ndarray
    distances: np.ndarray
    sorted_unique: bool = True

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices)
        distances = np.asarray(self.distances)
        if indices.ndim != 2 or distances.ndim != 2:
            raise ValueError("indices and distances must be 2-D")
        if indices.shape != distances.shape:
            raise ValueError("indices and distances must have identical shape")
        object.__setattr__(self, "indices", indices.astype(np.int32, copy=False))
        object.__setattr__(self, "distances", distances.astype(np.float32, copy=False))

    @property
    def n_samples(self) -> int:
        return int(self.indices.shape[0])

    @property
    def k(self) -> int:
        return int(self.indices.shape[1])

    def validate(self, *, deep: bool = False) -> "KNNGraph":
        n = self.n_samples
        if self.k < 1:
            raise ValueError("kNN graph must contain at least one neighbour")
        if np.any(self.indices < 0) or np.any(self.indices >= n):
            raise ValueError("kNN graph contains out-of-range indices")
        if not np.all(np.isfinite(self.distances)) or np.any(self.distances < 0):
            raise ValueError("kNN graph distances must be finite and nonnegative")
        if np.any(np.diff(self.distances, axis=1) < -1e-7):
            raise ValueError("kNN rows must be sorted by distance")
        rows = np.arange(n, dtype=np.int32)[:, None]
        if np.any(self.indices == rows):
            raise ValueError("kNN graph must exclude self-neighbours")
        if deep:
            # Deep validation is deliberately opt-in; it is O(n k log k).
            sorted_idx = np.sort(self.indices, axis=1)
            if np.any(np.diff(sorted_idx, axis=1) == 0):
                raise ValueError("each row must contain distinct neighbours")
        return self
