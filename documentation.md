# Documentation: qj_technical_indicators Library

## Overview

This document provides documentation for the `qj_technical_indicators` Python library, a custom implementation for calculating various financial technical indicators.

**Key Features:**

* **Standalone:** Developed initially for the QuantJourney project, it's now a standalone library with no dependencies on external C libraries like TA-Lib.
* **Performance:** Leverages NumPy for array operations, Pandas for data handling, and Numba (`@njit`) for significant performance optimization of core calculations.
* **Comprehensive:** Offers a wide range of common and less common technical indicators.
* **Utilities:** Includes helper functions for data validation, plotting (requires `matplotlib`), divergence detection, and crossover detection.
* **Open Source:** Licensed under the MIT License, encouraging use and contributions.

**Dependencies:**

* `numpy`
* `pandas`
* `numba`
* `decorator` (used for the `@timer` decorator in test code)
* `logging` (standard Python library)
* *Optional:* `matplotlib` (for plotting functionality)
* *Optional (for test code):* `yfinance` (used to download sample data in the test section)

**Contact:** Jakub Polec (jakub@quantjourney.pro)
**GitHub:** [https://github.com/QuantJourneyOrg/qj_technical_indicators](https://github.com/QuantJourneyOrg/qj_technical_indicators)

*(Note: The library code includes a test suite section using `yfinance` and an `Enum`. This documentation focuses on the `TechnicalIndicators` class itself, which is the core library functionality.)*

---

## Available Indicators

The library provides the following technical indicators, grouped by category:

### Trend Indicators

* **SMA (Simple Moving Average):** Calculates the average price over a specified period.
    * [Investopedia Link](https://www.investopedia.com/terms/s/sma.asp)
* **EMA (Exponential Moving Average):** A moving average that gives more weight to recent prices.
    * [Investopedia Link](https://www.investopedia.com/terms/e/ema.asp)
* **DEMA (Double Exponential Moving Average):** A faster-reacting moving average that aims to reduce lag.
    * [Investopedia Link](https://www.investopedia.com/terms/d/double-exponential-moving-average.asp)
* **KAMA (Kaufman Adaptive Moving Average):** A moving average that adjusts its speed based on market volatility.
    * [Investopedia Link](https://www.investopedia.com/terms/k/kaufmanns-adaptive-moving-average-kama.asp)
* **ALMA (Arnaud Legoux Moving Average):** A moving average designed to be responsive and smooth, reducing lag by applying Gaussian-weighted offsets.
    * [Fidelity Link](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/alma)
* **ICHIMOKU (Ichimoku Cloud):** A comprehensive indicator defining support/resistance, momentum, and trend direction using multiple lines (Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span).
    * [Investopedia Link](https://www.investopedia.com/terms/i/ichimoku-cloud.asp)
* **SUPERTREND:** A trend-following indicator plotted on the price chart, indicating the current trend direction and providing potential entry/exit signals.
    * [Investopedia Link](https://www.investopedia.com/terms/s/supertrend.asp)
* **LINEAR_REGRESSION_CHANNEL:** Plots a linear regression trendline over a specified period, along with upper and lower channel lines based on standard deviations from the trendline.
    * [TradingView Link](https://www.tradingview.com/support/solutions/43000502340-linear-regression-channels/)
* **HULL_MA (Hull Moving Average):** An extremely responsive and smooth moving average developed by Alan Hull.
    * [Investopedia Link](https://www.investopedia.com/terms/h/hullmovingaverage.asp)
* **QSTICK:** Measures the average difference between the opening and closing prices over a specified period, often used as a short-term momentum indicator.
    * [StockCharts Link](https://school.stockcharts.com/doku.php?id=technical_indicators:qstick)
* **RAINBOW:** Calculates and returns multiple Simple Moving Averages (SMAs) with different periods, often visualized together to form a "rainbow" pattern.
    * (Similar concept to Guppy Multiple Moving Averages: [Investopedia Link](https://www.investopedia.com/terms/g/guppy-multiple-moving-average.asp))

### Momentum Indicators

* **RSI (Relative Strength Index):** A momentum oscillator measuring the speed and change of price movements, ranging from 0 to 100.
    * [Investopedia Link](https://www.investopedia.com/terms/r/rsi.asp)
* **MACD (Moving Average Convergence Divergence):** Shows the relationship between two EMAs of prices. Includes the MACD line, Signal line (EMA of MACD), and Histogram (MACD - Signal).
    * [Investopedia Link](https://www.investopedia.com/terms/m/macd.asp)
* **STOCH (Stochastic Oscillator):** Compares a closing price to its price range over a period, indicating momentum and trend reversals. Returns %K and %D lines.
    * [Investopedia Link](https://www.investopedia.com/terms/s/stochasticoscillator.asp)
* **CCI (Commodity Channel Index):** An oscillator measuring the current price level relative to an average price level over a given period.
    * [Investopedia Link](https://www.investopedia.com/terms/c/commoditychannelindex.asp)
* **WILLR (Williams %R):** A momentum indicator that is the inverse of the Stochastic Oscillator, measuring overbought/oversold levels.
    * [Investopedia Link](https://www.investopedia.com/terms/w/williamsr.asp)
* **ROC (Rate of Change):** Measures the percentage change in price between the current price and the price N periods ago.
    * [Investopedia Link](https://www.investopedia.com/terms/r/rateofchange.asp)
* **TRIX:** A momentum oscillator showing the percentage rate of change of a triple exponentially smoothed moving average.
    * [Investopedia Link](https://www.investopedia.com/terms/t/trix.asp)
* **ULTIMATE_OSCILLATOR:** A momentum oscillator combining price action across three different timeframes into one value.
    * [Investopedia Link](https://www.investopedia.com/terms/u/ultimateoscillator.asp)
* **CMO (Chande Momentum Oscillator):** Measures momentum using the sum of recent gains and losses, scaled between -100 and +100.
    * [Investopedia Link](https://www.investopedia.com/terms/c/chandemomentumoscillator.asp)
* **DPO (Detrended Price Oscillator):** Attempts to remove the longer-term trend from prices to identify cycles. It compares a past price to a recent SMA.
    * [Investopedia Link](https://www.investopedia.com/terms/d/detrendedpriceoscillator.asp)
* **KDJ:** An oscillator derived from the Stochastic Oscillator, commonly used in Asian markets. Includes %K, %D, and %J lines to signal overbought/oversold conditions.
    * [Investopedia Link](https://www.investopedia.com/terms/k/kdj-indicator.asp)
* **AO (Awesome Oscillator):** Calculates the difference between a 34-period and 5-period simple moving average of the median price (High+Low)/2.
    * [Investopedia Link](https://www.investopedia.com/terms/a/awesomeoscillator.asp)
* **MA_MOMENTUM (Moving Average Momentum):** Calculates the momentum (Rate of Change) of a Simple Moving Average.
* **DISPARITY:** Measures the percentage difference between the current closing price and its Simple Moving Average.
    * [TradingView Link](https://www.tradingview.com/scripts/disparityindex/)
* **COPPOCK (Coppock Curve):** A long-term momentum indicator calculated as a weighted moving average of the sum of two rates of change.
    * [Investopedia Link](https://www.investopedia.com/terms/c/coppockcurve.asp)
* **RVI (Relative Vigor Index):** An oscillator based on the idea that prices tend to close higher than they open in uptrends and lower than they open in downtrends.
    * [Investopedia Link](https://www.investopedia.com/terms/r/relative_vigor_index.asp)
* **PGO (Pretty Good Oscillator):** Measures the distance of the current close from its N-period low, expressed as a percentage of the N-period range (-50 to +50).
    * [Markplex Link](https://markplex.com/free-tutorials/tradestation-easylanguage-tutorial-17-pretty-good-oscillator-pgo/)
* **PSL (Psychological Line):** An oscillator measuring the percentage of days within the lookback period that the price closed higher than the previous day (0 to 100).
    * [FXSSI Link](https://fxssi.com/psychological-line-indicator)
* **MOMENTUM_INDEX:** Calculates separate indices for positive and negative price momentum over a period, typically normalized to sum to 100.

### Volume Indicators

* **AD (Accumulation/Distribution Line):** Relates price changes and volume, rising when closes are in the upper part of the daily range with high volume, falling otherwise.
    * [Investopedia Link](https://www.investopedia.com/terms/a/accumulationdistribution.asp)
* **ADOSC (Chaikin A/D Oscillator):** Measures the momentum of the Accumulation/Distribution Line using the MACD formula (difference between fast and slow EMAs of the A/D Line).
    * [Investopedia Link](https://www.investopedia.com/terms/c/chaikinoscillator.asp)
* **MFI (Money Flow Index):** A volume-weighted RSI that measures buying and selling pressure based on price and volume (0 to 100).
    * [Investopedia Link](https://www.investopedia.com/terms/m/mfi.asp)
* **OBV (On-Balance Volume):** A running total of volume, adding volume on up days and subtracting it on down days, used to confirm price trends.
    * [Investopedia Link](https://www.investopedia.com/terms/o/onbalancevolume.asp)
* **PVO (Percentage Volume Oscillator):** Applies the MACD concept to volume, showing the relationship between a fast and slow EMA of volume. Returns PVO, Signal, and Histogram.
    * [StockCharts Link](https://school.stockcharts.com/doku.php?id=technical_indicators:percentage_volume_oscillator_pvo)
* **VWAP (Volume Weighted Average Price):** The average price weighted by volume over a specified period.
    * [Investopedia Link](https://www.investopedia.com/terms/v/vwap.asp)
* **VOLUME_INDICATORS:** A composite function calculating:
    * `Volume_SMA`: Simple Moving Average of volume.
    * `Force_Index`: Links price change and volume (`(Current Close - Prior Close) * Volume`). [Investopedia Link](https://www.investopedia.com/terms/f/forceindex.asp)
    * `VPT (Volume Price Trend)`: A running cumulative volume adjusted by the percentage change in price. [StockCharts Link](https://school.stockcharts.com/doku.php?id=technical_indicators:volume_price_trend_vpt)

### Volatility Indicators

* **BB (Bollinger Bands):** Bands plotted N standard deviations above and below a simple moving average, indicating volatility. Returns Upper, Middle (SMA), and Lower bands.
    * [Investopedia Link](https://www.investopedia.com/terms/b/bollingerbands.asp)
* **ATR (Average True Range):** Measures market volatility by decomposing the entire range of an asset price for that period.
    * [Investopedia Link](https://www.investopedia.com/terms/a/atr.asp)
* **HISTORICAL_VOLATILITY:** Measures the degree of price variation over a period, typically calculated as the annualized standard deviation of log returns.
    * [Investopedia Link](https://www.investopedia.com/terms/h/historicalvolatility.asp)
* **CHAIKIN_VOLATILITY:** Measures volatility by comparing the spread between high and low prices, based on the rate of change of an EMA of the high-low range.
    * [StockCharts Link](https://school.stockcharts.com/doku.php?id=technical_indicators:chaikin_volatility)
* **KELTNER (Keltner Channels):** Volatility bands plotted above and below an EMA, where the width is determined by the Average True Range (ATR). Returns Upper, Middle (EMA), and Lower bands.
    * [Investopedia Link](https://www.investopedia.com/terms/k/keltnerchannel.asp)
* **DONCHIAN (Donchian Channels):** Plots the highest high and lowest low over a specified period, forming volatility bands. Returns Upper, Middle, and Lower bands.
    * [Investopedia Link](https://www.investopedia.com/terms/d/donchianchannels.asp)
* **CHOPPINESS (Choppiness Index):** An indicator designed to determine if the market is trending (low values) or consolidating (high values), ranging from 0 to 100.
    * [Investopedia Link](https://www.investopedia.com/terms/c/choppinessindex.asp)

### Directional Indicators

* **ADX (Average Directional Index):** Measures trend strength (not direction). Often used with +DI and -DI. Returns ADX, +DI, and -DI lines.
    * [Investopedia Link](https://www.investopedia.com/terms/a/adx.asp)
* **DI (Directional Indicator):** Calculates the Positive Directional Indicator (+DI) and Negative Directional Indicator (-DI), components of the ADX system.
    * [Investopedia Link](https://www.investopedia.com/terms/p/positivedirectionalindicator.asp)
* **AROON:** Identifies trend direction and strength. Includes Aroon Up (time since highest high) and Aroon Down (time since lowest low). Returns Aroon Up, Aroon Down, and Aroon Oscillator (Up - Down).
    * [Investopedia Link](https://www.investopedia.com/terms/a/aroon.asp)

### Risk / Relationship Indicators

* **BETA:** Measures the volatility of an asset compared to the overall market (benchmark). A beta > 1 indicates higher volatility than the market, < 1 lower volatility.
    * [Investopedia Link](https://www.investopedia.com/terms/b/beta.asp)

### Other / Utility Indicators

* **MASS_INDEX:** Suggests trend reversals by measuring the narrowing and widening of the range between high and low prices.
    * [Investopedia Link](https://www.investopedia.com/terms/m/massindex.asp)
* **HEIKEN_ASHI:** A candlestick charting technique that averages price movements to create a smoother chart, aiding trend identification. Returns HA Open, High, Low, Close.
    * [Investopedia Link](https://www.investopedia.com/terms/h/heikinashi.asp)
* **BENFORD_LAW:** Calculates the observed frequency distribution of the first digits of price data and compares it to the expected distribution according to Benford's Law. Used potentially for anomaly detection.
    * [Wikipedia Link](https://en.wikipedia.org/wiki/Benford%27s_law)
* **PIVOT_POINTS:** Calculates standard pivot points (PP) and associated support (S1, S2) and resistance (R1, R2) levels based on the previous period's high, low, and close.
    * [Investopedia Link](https://www.investopedia.com/terms/p/pivotpoint.asp)
* **ELDER_RAY:** Measures buying and selling pressure by comparing the high and low prices to an EMA. Includes Bull Power (High - EMA) and Bear Power (Low - EMA).
    * [StockCharts Link](https://school.stockcharts.com/doku.php?id=technical_indicators:elder_ray_index)
* **TYPICAL_PRICE:** Calculates the average of the high, low, and close prices for each period `((H+L+C)/3)`.
    * [Investopedia Link](https://www.investopedia.com/terms/t/typicalprice.asp)
* **WEIGHTED_CLOSE:** Calculates a weighted average price giving more weight to the closing price `((H+L+C+C)/4)`.
    * [Investopedia Link](https://www.investopedia.com/terms/w/weightedclose.asp)
* **FRACTAL:** Identifies potential turning points based on local highs (bullish fractal) or lows (bearish fractal) compared to surrounding bars. Returns binary signals (1 if fractal detected, 0 otherwise).
    * [Investopedia Link](https://www.investopedia.com/terms/f/fractal.asp)

---

## Technical Documentation: How to Use

### 1. Installation

Currently, the library is provided as a single Python file (`.py`). To use it, save the file in your project directory and import the `TechnicalIndicators` class:

```python
from your_saved_filename import TechnicalIndicators # Replace 'your_saved_filename'