"""
Kernels package
---------------

Domain-organised re-exports of Numba-accelerated kernels.
This module exports all _calculate_*_numba functions from the domain modules.
"""

from .trend_numba import *  # noqa: F401,F403
from .momentum_numba import *  # noqa: F401,F403
from .volatility_numba import *  # noqa: F401,F403
from .volume_numba import *  # noqa: F401,F403

__all__ = [
    name
    for name in globals().keys()
    if name.startswith("_calculate_") and name.endswith("_numba")
]
