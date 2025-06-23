"""
QuantJourney Technical-Indicators - Decorators
==============================================

Light-weight decorator utilities used throughout the QuantJourney
Technical-Indicators codebase. They are kept in a dedicated module to avoid
importing heavy numerical packages (NumPy, Numba, Pandas) just for helper
decorators.

Currently provided:
- timer
    Measures a function's wall-clock run-time and emits a log entry at the INFO
    level. Useful for benchmarking indicator kernels or spotting slow data pipelines.
- numba_fallback
    Handles Numba errors by falling back to a specified function (e.g., Pandas
    implementation), logging the fallback event as JSON via IndicatorCalculationError.
    Ensures robustness in Numba-optimized indicator methods.

Example:
    from quantjourney_ti.decorators import timer, numba_fallback
    import logging
    logging.basicConfig(level=logging.INFO)

    @timer
    def slow_add(a, b):
        import time; time.sleep(0.5); return a + b

    slow_add(2, 3)
    # INFO:quantjourney_ti.decorators:Finished slow_add in 0.5001 seconds
    # 5

    def pandas_fallback(self, data):
        return data.mean()

    @numba_fallback(pandas_fallback)
    def numba_mean(self, data):
        raise ValueError("Numba failed")

    class Example:
        def mean(self, data): return numba_mean(self, data)

    Example().mean([1, 2, 3])
    # WARNING:quantjourney_ti.decorators:{"type": "IndicatorCalculationError", ...}
    # 2.0

Author: Jakub Polec <jakub@quantjourney.pro>
License: MIT
"""

import os
import time
import logging
from functools import wraps
from typing import Callable, TypeVar

from ._errors import IndicatorCalculationError

T = TypeVar("T")

logger = logging.getLogger(__name__)
log_level = os.getenv("QUANTJOURNEY_LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def timer(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to measure and log the execution time of a function.
    Logs the function name and execution time in seconds at the INFO level.

    Args:
        func: Function to decorate.

    Returns:
        Callable: Wrapped function that logs execution time and returns original result.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.info(f"Finished {func.__name__} in {run_time:.4f} seconds")
        return result
    return wrapper


def numba_fallback(fallback_fn: Callable[..., T]) -> Callable[..., Callable[..., T]]:
    """
    Decorator to handle Numba errors with a fallback function.
    Attempts the Numba-optimized calculation, falling back to the provided function
    if a Numba-related error occurs. Logs the error as JSON using IndicatorCalculationError
    at the WARNING level for fallbacks or ERROR level for re-raised exceptions.

    Args:
        fallback_fn: Function to call if Numba fails, with the same signature as the decorated method.

    Returns:
        Callable: Decorated method that attempts Numba calculation and falls back if needed.
    """
    def decorator(method: Callable[..., T]) -> Callable[..., T]:
        @wraps(method)
        def wrapper(*args, **kwargs) -> T:
            try:
                return method(*args, **kwargs)
            except (ValueError, TypeError) as e:
                error = IndicatorCalculationError(
                    indicator=method.__name__,
                    message=str(e),
                    context={"args": str(args[1:]), "kwargs": kwargs}
                )
                logger.warning(error.to_json())
                return fallback_fn(*args, **kwargs)
            except Exception as e:
                error = IndicatorCalculationError(
                    indicator=method.__name__,
                    message=str(e),
                    context={"args": str(args[1:]), "kwargs": kwargs}
                )
                logger.error(error.to_json())
                raise
        return wrapper
    return decorator
