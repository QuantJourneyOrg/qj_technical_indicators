#!/usr/bin/env python3
"""
Technical Indicators Demo Script
================================

This script demonstrates the usage of the Technical Indicators Library by fetching AAPL data from
yfinance and calculating various technical indicators with timing measurements. It serves as an
example of how to use the library, which is available at:
https://github.com/QuantJourneyOrg/qj_technical_indicators

License: MIT License - see LICENSE.md for details.

For questions or feedback, contact Jakub at jakub@quantjourney.pro.

Last Updated: April 09, 2025
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf
import logging
from technical_indicators import TechnicalIndicators  # Import from the separate file

# Set up generic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Timing decorator
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

# Main function to try all indicators
@timer
def try_all_indicators():
    """Fetch AAPL data and calculate all technical indicators with timing."""
    logger.info("Fetching AAPL data from yfinance...")
    data = yf.download('AAPL', start='2010-01-01', end='2025-01-01', progress=False)
    
    # Log raw columns for debugging
    logger.info(f"Raw DataFrame columns: {data.columns}")
    
    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        logger.warning("Multi-level columns detected, extracting AAPL data")
        data = data.xs('AAPL', level=1, axis=1)  # Extract AAPL data
    
    # Rename columns
    data = data.rename(columns={
        'Adj Close': 'adj_close',
        'adj close': 'adj_close',
        'Close': 'close',
        'close': 'close',
        'Open': 'open',
        'open': 'open',
        'High': 'high',
        'high': 'high',
        'Low': 'low',
        'low': 'low',
        'Volume': 'volume',
        'volume': 'volume'
    })
    
    # Verify columns
    logger.info(f"DataFrame columns after renaming: {data.columns.tolist()}")
    if 'close' not in data.columns:
        logger.error("No 'close' column found in DataFrame")
        raise ValueError("Required column 'close' not found")
    
    # Inspect data['close']
    logger.info(f"Type of data['close']: {type(data['close'])}")
    logger.info(f"First few values of data['close']: {data['close'].head()}")
    
    ti = TechnicalIndicators()
    indicators = [
        ('SMA', lambda: ti.SMA(data['close'], period=20)),
        ('EMA', lambda: ti.EMA(data['close'], period=20)),
        # Add more indicators as needed
    ]
    
    results = {}
    for name, func in indicators:
        logger.info(f"Calculating {name}...")
        result = func()
        results[name] = result
        logger.info(f"{name} sample:\n{result.tail(5)}\n")
    
    return results

if __name__ == "__main__":
    results = try_all_indicators()