#!/usr/bin/env python3
"""
Decorator utilities for the Technical Indicators Library.

Provides timing functionality to measure and log execution times of functions.
"""

import time
import logging

# Configure logging if not already set up by the main module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def timer(func):
    """Decorator to measure and log the execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.info(f"Finished {func.__name__} in {run_time:.4f} seconds")
        return result
    return wrapper