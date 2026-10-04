"""Controlled reference implementations used by the benchmark suite."""

from baselines.algorithms import dispatch_baseline, warm_numba_kernels

__all__ = ["dispatch_baseline", "warm_numba_kernels"]
