from .multiscale import MultiscaleConfig
from .neighbors import HNSWConfig, build_knn_graph
from .optimization import (
    GammaMDLConfig,
    OptimizationStrataSCAN,
    OptimizationStrictCoreConfig,
    OptimizationStrictCoreResult,
)
from .predictive import PredictiveMultiscaleConfig, PredictiveMultiscaleStrataSCAN
from .strict_core import (
    GammaStrictCoreConfig,
    GammaStrictCoreResult,
    gamma_strict_core_from_graph,
)
from .types import KNNGraph
from .uniform_tail import UniformTailConfig

__all__ = [
    "GammaStrictCoreConfig",
    "GammaStrictCoreResult",
    "GammaMDLConfig",
    "HNSWConfig",
    "KNNGraph",
    "MultiscaleConfig",
    "OptimizationStrataSCAN",
    "OptimizationStrictCoreConfig",
    "OptimizationStrictCoreResult",
    "PredictiveMultiscaleConfig",
    "PredictiveMultiscaleStrataSCAN",
    "StrataSCAN",
    "UniformTailConfig",
    "build_knn_graph",
    "gamma_strict_core_from_graph",
]

# The short public name follows the current released estimator. The explicit
# class name remains available for precise configuration and provenance.
StrataSCAN = OptimizationStrataSCAN

__version__ = "0.2.0"
