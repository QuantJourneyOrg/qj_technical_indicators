"""
Technical Indicators Demo Script
================================

This script demonstrates the usage of the Technical Indicators Library by fetching AAPL data from
yfinance and calculating 20 top-used technical indicators, saving individual plots for each.
It serves as an example of how to use the library, which is available at:
https://github.com/QuantJourneyOrg/qj_technical_indicators

License: MIT License - see LICENSE.md for details.

For questions or feedback, contact Jakub at jakub@quantjourney.pro.

Last Updated: June 23, 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
from quantjourney_ti import TechnicalIndicators
from quantjourney_ti._utils import plot_indicators
import os

def fetch_data(ticker="AAPL", start="2024-01-01", end="2025-02-01"):
    """Fetch data from yfinance."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError("Empty DataFrame")
    # Flatten yfinance MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0).str.lower().str.replace(' ', '_')
    df.index.name = df.index.name.lower() if df.index.name else 'date'
    df["volume"] = df["volume"].replace(0, np.nan).ffill()  # Fix FutureWarning
    return df

def calculate_indicators(df):
    """Calculate 20 top-used technical indicators."""
    ti = TechnicalIndicators()
    indicators = [
        ("SMA", lambda: ti.SMA(df["close"], period=14)),
        ("EMA", lambda: ti.EMA(df["close"], period=14)),
        ("RSI", lambda: ti.RSI(df["close"], period=14)),
        ("WILLR", lambda: ti.WILLR(df[["high", "low", "close"]], period=14)),
        ("MFI", lambda: ti.MFI(df[["high", "low", "close", "volume"]], period=14)),
        ("MomentumIndex", lambda: ti.MOMENTUM_INDEX(df["close"], period=14)["MomentumIndex"]),
        ("RVI", lambda: ti.RVI(df[["open", "high", "low", "close"]], period=14)),
        ("AD", lambda: ti.AD(df[["high", "low", "close", "volume"]])),
        ("ADOSC", lambda: ti.ADOSC(df[["high", "low", "close", "volume"]], fast_period=3, slow_period=10)),
        ("ElderRay_Bull", lambda: ti.ELDER_RAY(df[["high", "low", "close"]], period=14)["BullPower"]),
        ("MACD", lambda: ti.MACD(df["close"], fast_period=12, slow_period=26, signal_period=9)["MACD"]),
        ("BB_Middle", lambda: ti.BB(df["close"], period=20, num_std=2.0)["BB_Middle"]),
        ("ATR", lambda: ti.ATR(df[["high", "low", "close"]], period=14)),
        ("STOCH_K", lambda: ti.STOCH(df[["high", "low", "close"]], k_period=14, d_period=3)["K"]),
        ("CCI", lambda: ti.CCI(df[["high", "low", "close"]], period=20)),
        ("ROC", lambda: ti.ROC(df["close"], period=12)),
        ("OBV", lambda: ti.OBV(df[["close", "volume"]])),
        ("VWAP", lambda: ti.VWAP(df[["high", "low", "close", "volume"]], period=14)),
        ("DEMA", lambda: ti.DEMA(df["close"], period=14)),
        ("KAMA", lambda: ti.KAMA(df["close"], er_period=10)),
    ]
    results = {}
    for name, func in indicators:
        print(f"Calculating {name}...")
        try:
            result = func()
            results[name] = result
            print(f"{name} sample (last 5):\n{result.tail(5)}\n")
        except Exception as e:
            print(f"Failed to calculate {name}: {str(e)}")
    return results

def plot_results(df, indicators):
    """Save individual plots for each indicator."""
    print("Saving indicator plots...")
    try:
        os.makedirs("indicator_plots", exist_ok=True)
        for name, result in indicators.items():
            if isinstance(result, pd.Series):
                plot_indicators_dict = {name: result}
            elif isinstance(result, pd.DataFrame):
                plot_indicators_dict = {name: result.iloc[:, 0]}
            else:
                print(f"Skipping {name}: not a Series or DataFrame")
                continue
            plot_indicators(
                df,
                plot_indicators_dict,
                title=f"{name} Indicator",
                price_col="close",
                save_path=f"indicator_plots/{name}_plot.png"
            )
            print(f"Saved plot for {name} to indicator_plots/{name}_plot.png")
    except Exception as e:
        print(f"Plotting failed: {str(e)}")

def main():
    """Main function to run the demo."""
    try:
        df = fetch_data()
        indicators = calculate_indicators(df)
        plot_results(df, indicators)
        print("Demo completed successfully")
        for name, result in indicators.items():
            try:
                if isinstance(result, pd.Series):
                    last_val = result.dropna()
                    if not last_val.empty:
                        print(f"Last {name}: {last_val.iloc[-1]:.2f}")
                    else:
                        print(f"{name} has only NaN values.")
                elif isinstance(result, pd.DataFrame):
                    last_row = result.dropna()
                    if not last_row.empty:
                        print(f"Last {name}:\n{last_row.iloc[-1]}")
                    else:
                        print(f"{name} DataFrame is empty after dropping NaNs.")
            except Exception as e:
                print(f"Error reporting {name}: {e}")
    except Exception as e:
        print(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()