"""
QuantJourney Technical-Indicators Example Script
=============================================

This script demonstrates the usage of the Technical Indicators Library by fetching AAPL data from
yfinance and calculating various technical indicators with timing measurements. It serves as an
example of how to use the library, which is available at:
https://github.com/QuantJourneyOrg/qj_technical_indicators

License: MIT License - see LICENSE.md for details.
"""

import logging
import os
import numpy as np
import pandas as pd
import yfinance as yf

from quantjourney_ti import TechnicalIndicators, validate_data
from quantjourney_ti._decorators import timer
from quantjourney_ti._utils import plot_indicators
# ------------------------------------------------------------------------------
# Logging setup (clean and conflict-free)
# ------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []  # Clear old handlers

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ------------------------------------------------------------------------------
# Data fetching
# ------------------------------------------------------------------------------

@timer
def fetch_data(ticker="AAPL", start="2024-01-01", end="2025-02-01"):
    logger.info(f"Fetching {ticker} data from yfinance...")
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    except Exception as e:
        logger.error(f"Failed to fetch data: {str(e)}")
        raise

    if df.empty:
        logger.error("No data returned from yfinance")
        raise ValueError("Empty DataFrame")

    print("=== RAW df.columns ===")
    print(df.columns)
    print("=== RAW df.head() ===")
    print(df.head())

    # Flatten yfinance MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0).str.lower()
    else:
        df.columns = df.columns.str.lower()

    df = df.rename(columns={"adj close": "adj_close"})

    # Pre-clean volume for MFI
    if "volume" in df.columns:
        df["volume"] = df["volume"].replace(0, np.nan).ffill()

    logger.info(f"DataFrame columns: {df.columns.tolist()}")

    required_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        raise ValueError(f"Missing columns: {missing}")

    try:
        validate_data(df, required_columns)
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        raise

    return df

# ------------------------------------------------------------------------------
# Indicator calculation
# ------------------------------------------------------------------------------

@timer
def calculate_indicators(df):
    ti = TechnicalIndicators()
    indicators = [
        ("SMA", lambda: ti.SMA(df["close"], period=14)),
        ("EMA", lambda: ti.EMA(df["close"], period=14)),
        ("RSI", lambda: ti.RSI(df["close"], period=14)),
        ("WILLR", lambda: ti.WILLR(df[["high", "low", "close"]], period=14)),
        ("MFI", lambda: ti.MFI(df[["high", "low", "close", "volume"]], period=14)),
        ("ElderRay", lambda: ti.ELDER_RAY(df[["high", "low", "close"]], period=14)),
    ]

    results = {}
    for name, func in indicators:
        logger.info(f"Calculating {name}...")
        try:
            result = func()
            results[name] = result
            logger.info(f"{name} sample (last 5):\n{result.tail(5)}\n")
        except Exception as e:
            logger.error(f"Failed to calculate {name}: {str(e)}")
    return results

# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------

@timer
def plot_results(df, indicators):
    logger.info("Plotting indicators...")
    try:
        plot_indicators(df, indicators, price_col="close")
    except Exception as e:
        logger.error(f"Plotting failed: {str(e)}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    try:
        df = fetch_data()
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame tail: \n{df.tail(15)}")
        indicators = calculate_indicators(df)
        plot_results(df, indicators)

        logger.info("Demo completed successfully")
        for name, result in indicators.items():
            try:
                if isinstance(result, pd.Series):
                    last_val = result.dropna()
                    if not last_val.empty:
                        logger.info(f"Last {name}: {last_val.iloc[-1]:.2f}")
                    else:
                        logger.warning(f"{name} has only NaN values.")
                elif isinstance(result, pd.DataFrame):
                    last_row = result.dropna()
                    if not last_row.empty:
                        logger.info(f"Last {name}:\n{last_row.iloc[-1]}")
                    else:
                        logger.warning(f"{name} DataFrame is empty after dropping NaNs.")
            except Exception as e:
                logger.error(f"Error reporting {name}: {e}")
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
