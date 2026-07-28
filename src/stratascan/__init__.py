from .multiscale import MultiscaleConfig
from .neighbors import HNSWConfig, build_knn_graph
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
    "HNSWConfig",
    "KNNGraph",
    "MultiscaleConfig",
    "PredictiveMultiscaleConfig",
    "PredictiveMultiscaleStrataSCAN",
    "StrataSCAN",
    "UniformTailConfig",
    "build_knn_graph",
    "gamma_strict_core_from_graph",
]

# The short public name follows the current released estimator.  The explicit
# class name remains available for code that wants the release method recorded
# directly in its configuration or provenance.
StrataSCAN = PredictiveMultiscaleStrataSCAN

__version__ = "0.1.2"
