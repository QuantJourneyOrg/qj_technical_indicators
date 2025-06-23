"""
QuantJourney Technical-Indicators
=================================
Convenience import layer for the **quantjourney_ti** package. It re-exports:

• `TechnicalIndicators`  - High-level class offering 60+ indicator methods.
• `timer`                - Decorator for benchmarking function execution time.
• `numba_fallback`       - Decorator for Numba fallback to pandas/numpy.
• Selected helper functions from `_utils` for validation, analysis, and plotting.

Most projects need only::

    import quantjourney_ti as ti
    df['sma'] = ti.sma(df['close'], 20)

Under the hood, `sma`, `ema`, etc., are thin wrappers around a shared
`TechnicalIndicators` singleton to avoid re-compiling Numba kernels.

Example usage
-------------
Flat helpers (quick one-liners)::

    import quantjourney_ti as ti
    df["sma"] = ti.sma(df["close"], 20)

Full flexibility via the class::

    from quantjourney_ti import TechnicalIndicators
    ti = TechnicalIndicators()
    df["atr"] = ti.ATR(ohlc_df, 14)

Power-user shortcut (shared singleton)::

    import quantjourney_ti.indicators as ind
    ind._TI_INSTANCE.ATR(ohlc_df, 14)  # Same object, no extra compile

Notes:
- Compatible with pandas 2.x (uses `pandas.api.types.is_any_real_numeric_dtype`).
- Requires `numpy`, `pandas`, and `numba`. `matplotlib` is optional for plotting.
- Logging uses thread-safe `QueueHandler`. Ensure a queue processing thread is active.

Author: Jakub Polec <jakub@quantjourney.pro>
License: MIT
"""

from __future__ import annotations

from .indicators import TechnicalIndicators
from ._decorators import timer, numba_fallback
from ._utils import (
    validate_data,
    validate_and_get_prices,
    validate_window,
    detect_divergence,
    detect_crossovers,
    plot_indicators,
    optimize_memory,
)

# Initialize shared singleton
_TI_INSTANCE = TechnicalIndicators()

# Public exports
__all__ = [
    "TechnicalIndicators",
    "timer",
    "numba_fallback",
    "validate_data",
    "validate_and_get_prices",
    "validate_window",
    "detect_divergence",
    "detect_crossovers",
    "plot_indicators",
    "optimize_memory",
    "__author__",
    "__email__",
    "__url__",
    "__version__",
]

# Metadata
__author__ = "Jakub Polec"
__email__ = "jakub@quantjourney.pro"
__url__ = "https://quantjourney.substack.com"
__version__ = "0.2.0"  # Consider using setuptools_scm for dynamic versioning

# Dynamic wrappers for TechnicalIndicators methods
for _name in dir(TechnicalIndicators):
    if _name.startswith("_") or not callable(getattr(TechnicalIndicators, _name)):
        continue
    globals()[_name.lower()] = getattr(_TI_INSTANCE, _name)
    __all__.append(_name.lower())

# Dynamic wrappers for utility functions
# Use globals() to access already imported functions
_util_functions = [
    "validate_data",
    "validate_and_get_prices",
    "validate_window",
    "detect_divergence",
    "detect_crossovers",
    "plot_indicators",
    "optimize_memory",
]
for _name in _util_functions:
    globals()[_name] = globals()[_name]
    __all__.append(_name)