from .neighbors import HNSWConfig, build_knn_graph
from .optimization import (
    GammaMDLConfig,
    OptimizationStrataSCAN,
    OptimizationStrictCoreConfig,
    OptimizationStrictCoreResult,
)
from .types import KNNGraph

__all__ = [
    "GammaMDLConfig",
    "HNSWConfig",
    "KNNGraph",
    "OptimizationStrataSCAN",
    "OptimizationStrictCoreConfig",
    "OptimizationStrictCoreResult",
    "StrataSCAN",
    "build_knn_graph",
]

# The short public name follows the current released estimator. The explicit
# class name remains available for precise configuration and provenance.
StrataSCAN = OptimizationStrataSCAN

__version__ = "0.2.4"
