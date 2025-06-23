# QuantJourney Technical Indicators

**A high-performance Python library for calculating technical indicators, optimized with Numba for speed and designed for financial data analysis. This project is part of the Quantitative Infrastructure initiative by [QuantJourney](https://quantjourney.substack.com), providing robust tools for traders and researchers.**

**License**: MIT License - see [LICENSE.md](LICENSE.md) for details.  
**Author**: Jakub Polec ([jakub@quantjourney.pro](mailto:jakub@quantjourney.pro))  
**Repository**: [github.com/QuantJourneyOrg/qj_technical_indicators](https://github.com/QuantJourneyOrg/qj_technical_indicators)

## Overview

The QuantJourney Technical Indicators library offers a comprehensive set of technical indicators for analyzing financial time series data. Key features include:
- **Numba-Optimized Calculations**: Fast, JIT-compiled functions for performance-critical computations.
- **Flexible API**: Supports both standalone functions and a `TechnicalIndicators` class for object-oriented usage.
- **Robust Error Handling**: Validates inputs and handles edge cases like NaNs and empty data.
- **Visualization**: Generates individual plots for indicators, saved as PNG files in an `indicator_plots` directory.
- **Integration**: Works seamlessly with `pandas` DataFrames and `yfinance` for data fetching.

The library is ideal for backtesting trading strategies, real-time analysis, and research, with a focus on simplicity and extensibility.

## Project Structure

The repository is organized as follows:

```
quantjourney_ti/
├── __init__.py               # Package initialization and imports
├── _decorators.py            # Decorators for timing and fallback mechanisms
├── _errors.py                # Custom error classes for input validation
├── _indicator_kernels.py     # Numba-optimized functions for indicator calculations
├── _legacy_/                 # Legacy code (not actively maintained)
├── _utils.py                 # Utility functions for validation, plotting, and memory optimization
├── indicators.py             # Main API class (TechnicalIndicators) with public methods
├── docs/                     # Documentation
│   ├── INDICATORS.md         # Explanation of each indicator
├── examples/                 # Example scripts demonstrating usage
│   ├── example_basic.py      # Basic indicator calculations
│   ├── example_indicators.py # Advanced usage with multiple indicators and plotting
├── tests/                    # Unit and integration tests
│   ├── __init__.py
│   ├── _yf.py                # yfinance test utilities
│   ├── test_all_indicators.py # Tests for all indicators
│   ├── test_basic.py         # Basic functionality tests
│   ├── test_decorators.py    # Decorator tests
│   ├── test_demo.py          # Demo script tests
│   ├── test_indicators.py    # Individual indicator tests
│   ├── test_integration_yf.py # Integration tests with yfinance
│   ├── test_utils.py         # Utility function tests
├── quantjourney_ti.egg-info/ # Package metadata (generated, typically in .gitignore)
├── README.md                 # Project documentation (this file)
├── LICENSE.md                # License details
├── setup.py                  # Package installation configuration
```

**Note**: The `quantjourney_ti.egg-info` directory is generated during package installation (e.g., `pip install -e .`). It can be safely removed if not using editable mode, and should be included in `.gitignore` to avoid version control.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/QuantJourneyOrg/qj_technical_indicators.git
   cd qj_technical_indicators
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

**Requirements**:
- Python 3.8+
- `pandas`, `numpy`, `yfinance`, `numba`, `matplotlib`

## Usage

The library provides a `TechnicalIndicators` class for calculating indicators and saving plots. The `examples/run_indicators.py` script fetches PL data, calculates 20 popular indicators, and saves individual plots to an `indicator_plots` directory. Example:

```python
from quantjourney_ti import TechnicalIndicators
from quantjourney_ti._utils import plot_indicators
import pandas as pd
import yfinance as yf
import os

# Fetch data
df = yf.download("PL", start="2024-01-01", end="2025-02-01")
df.columns = df.columns.str.lower().str.replace(' ', '_')
df["volume"] = df["volume"].replace(0, np.nan).ffill()

# Calculate indicators
ti = TechnicalIndicators()
indicators = [
    ("SMA", lambda: ti.SMA(df["close"], period=14)),
    ("EMA", lambda: ti.EMA(df["close"], period=14)),
    # ... 18 more indicators
]
results = {name: func() for name, func in indicators}

# Save plots
os.makedirs("indicator_plots", exist_ok=True)
for name, result in results.items():
    plot_indicators_dict = {name: result if isinstance(result, pd.Series) else result.iloc[:, 0]}
    plot_indicators(df, plot_indicators_dict, title=f"{name} Indicator", save_path=f"indicator_plots/{name}_plot.png")
```

Run the example:

```bash
python examples/run_indicators.py
```

**Output**:
- Console: Last 5 and final values for each indicator.
- Files: PNG plots in `indicator_plots/` (e.g., `SMA_plot.png`).

## 📊 Example Plot
![Technical Indicator Example](docs/technical_indicator.png)

## Supported Indicators

The library supports 39 indicators (54 series):
- **Single-Series Indicators** (21):
  - SMA (Simple Moving Average)
  - EMA (Exponential Moving Average)
  - RSI (Relative Strength Index)
  - ATR (Average True Range)
  - MFI (Money Flow Index)
  - TRIX
  - CCI (Commodity Channel Index)
  - ROC (Rate of Change)
  - WILLR (Williams %R)
  - DEMA (Double Exponential Moving Average)
  - KAMA (Kaufman Adaptive Moving Average)
  - AO (Awesome Oscillator)
  - ULTIMATE_OSCILLATOR
  - CMO (Chande Momentum Oscillator)
  - DPO (Detrended Price Oscillator)
  - MASS_INDEX
  - VWAP (Volume Weighted Average Price)
  - AD (Accumulation/Distribution Line)
  - HULL_MA (Hull Moving Average)
  - OBV (On-Balance Volume)
  - RVI (Relative Vigor Index)
- **Multi-Series Indicators** (18):
  - MACD (MACD, Signal, Histogram)
  - BB (Bollinger Bands: BB_Upper, BB_Middle, BB_Lower)
  - STOCH (Stochastic Oscillator: K, D)
  - ADX (Average Directional Index: ADX, +DI, -DI)
  - ICHIMOKU (Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, Chikou Span)
  - KELTNER (Keltner Channels: KC_Upper, KC_Middle, KC_Lower)
  - DONCHIAN (Donchian Channels: DC_Upper, DC_Middle, DC_Lower)
  - AROON (AROON_UP, AROON_DOWN, AROON_OSC)
  - VOLUME_INDICATORS (Volume_SMA, Force_Index, VPT)
  - PIVOT_POINTS (PP, R1, R2, S1, S2)
  - RAINBOW (9 SMAs for periods 2-10)
  - BETA
  - DI (Directional Indicator: +DI, -DI)
  - ADOSC (Chaikin A/D Oscillator)
  - HEIKEN_ASHI (HA_Open, HA_High, HA_Low, HA_Close)
  - BENFORD_LAW (Observed, Expected)
  - MOMENTUM_INDEX (MomentumIndex, NegativeIndex)
  - ELDER_RAY (BullPower, BearPower)

See `indicators.py` for the full list and parameters.

## Development

To contribute:
1. Fork the repository and create a branch.
2. Add new indicators in `_indicator_kernels.py` with Numba optimization.
3. Define public methods in `indicators.py`.
4. Update tests in `tests/`.
5. Submit a pull request.

**Testing**:
```bash
pytest tests/
```

**Cleaning**:
Remove generated files:
```bash
rm -rf quantjourney_ti.egg-info dist build
```

## Future Work

- Add more indicators (e.g., PPO, Ichimoku Cloud).
- Enhance plotting with customizable layouts.
- Optimize Numba functions for additional edge cases.
- Support real-time data feeds.

## Contact

For issues or feedback, contact Jakub Polec at [jakub@quantjourney.pro](mailto:jakub@quantjourney.pro) or open an issue on GitHub.