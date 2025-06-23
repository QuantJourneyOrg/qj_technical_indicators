# QuantJourney Technical Indicators Library Documentation

**Version**: 1.0  
**Author**: Jakub Polec <jakub@quantjourney.pro>  
**License**: MIT  
**Date**: June 22, 2025  


This document provides a comprehensive guide to the `TechnicalIndicators` class in the `QuantJourney Technical-Indicators` Python library, a high-performance tool for calculating financial technical indicators. Designed for quantitative traders, data scientists, and developers, the library leverages Numba for optimized computations and supports a wide range of indicators for market analysis, trading strategy development, and risk management.

The library is part of the **Quantitative Infrastructure** initiative, released under the MIT License. It separates concerns into numerical kernels (`quantjourney_ti._indicator_kernels`), data validation, and plotting utilities (`quantjourney_ti.utils`) for modularity and maintainability, while preserving backward compatibility with earlier implementations.

---

## Overview

The `TechnicalIndicators` class offers an extensive suite of technical indicators, from basic moving averages to advanced momentum, volatility, and volume-based measures. Key features include:

- **High Performance**: Numba-optimized kernels ensure fast computation, critical for large datasets or real-time applications.
- **Flexible Input**: Supports pandas `Series` and `DataFrame` inputs with robust validation.
- **Parallel Processing**: Enables concurrent calculation of multiple indicators via `calculate_multiple_indicators`.
- **Singleton Access**: A pre-initialized `_TI_INSTANCE` provides rapid access to compiled kernels.
- **Extensibility**: Modular design allows users to add custom indicators or extend functionality.

The library is ideal for backtesting trading strategies, generating trading signals, conducting market analysis, or building quantitative models. It assumes familiarity with financial markets, pandas for data handling, and basic Python programming.

---

## Installation and Setup

Install the library via pip:

```bash
pip install quantjourney_ti
```

Import the library in Python:

```python
import quantjourney_ti as ti
from quantjourney_ti import TechnicalIndicators
```

Input data should be a pandas `DataFrame` with columns such as `'open'`, `'high'`, `'low'`, `'close'`, and `'volume'`, or a `Series` for single-column indicators. Example data:

```python
import pandas as pd
data = pd.DataFrame({
    'open': [99, 100, 101, 100, 98],
    'high': [102, 103, 104, 101, 100],
    'low': [99, 100, 101, 99, 98],
    'close': [100, 101, 102, 100, 99],
    'volume': [1000, 1200, 1100, 900, 1000]
}, index=pd.date_range('2025-06-01', periods=5))
```

This sample data will be used in examples throughout the document. Ensure data is clean (no non-numeric values) and has sufficient length for the chosen periods.

---

## Class Initialization

### `__init__`

- **Purpose**: Initializes the `TechnicalIndicators` class, pre-compiling Numba-optimized kernels for low-latency calculations.
- **Input Parameters**:
  - None
- **Output**:
  - None (creates an instance with pre-compiled kernels)
- **Description**: 
  Creates a dummy array (`np.ones(10, dtype=np.float64)`) and calls all Numba-optimized kernel functions (e.g., `_calculate_sma_numba`, `_calculate_ema_numba`) to compile them at initialization. This avoids just-in-time compilation delays during actual calculations, improving performance for real-time or large-scale applications.
- **Use Case**:
  Instantiate the class for single or batch indicator calculations:
  ```python
  ti = TechnicalIndicators()
  ```
  Access the singleton instance for rapid calculations:
  ```python
  import quantjourney_ti.indicators as ind
  result = ind._TI_INSTANCE.SMA(data['close'], period=3)
  ```
- **Performance Notes**:
  - Compilation occurs once per instance, reducing latency for subsequent calls.
  - Memory usage is minimal due to the small dummy array.
- **Edge Cases**:
  - No input validation is required, as the constructor accepts no parameters.
  - Ensure sufficient memory for Numba compilation on resource-constrained systems.

---

## Indicator Functions

Each indicator function validates input data, performs calculations using a Numba-optimized kernel (unless noted), and returns a pandas `Series` or `DataFrame` with aligned indices. Indicators requiring multiple columns (e.g., `ATR`, `MACD`) expect a `DataFrame` with specific columns, while single-series indicators (e.g., `SMA`, `RSI`) accept a `Series` or `DataFrame`. Initial values are typically `NaN` for the first `period-1` entries due to insufficient data.

### SMA (Simple Moving Average)

- **Purpose**: Computes the Simple Moving Average, a trend-following indicator that smooths price data.
- **Mathematical Formula**:
  \[
  SMA_t = \frac{1}{n} \sum_{i=t-n+1}^{t} P_i
  \]
  where \( P_i \) is the price at time \( i \), and \( n \) is the period.
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series (typically `'close'`) or DataFrame with a `'close'` column.
  - `period`: `int` (default=20) - Number of periods (must be positive and ≤ data length).
- **Output**:
  - `pd.Series` - SMA values, indexed like the input, named `SMA_{period}`.
- **Description**:
  Calculates the arithmetic mean of prices over a sliding window. Validates input with `_validate_and_get_prices`, converts to a NumPy array, and applies `_calculate_sma_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Trend Identification**: Price above SMA indicates an uptrend.
  - **Crossover Strategies**: Use fast (e.g., 20-period) and slow (e.g., 50-period) SMAs for buy/sell signals.
  Example:
  ```python
  df = data.copy()
  df['SMA_3'] = ti.SMA(data['close'], period=3)
  print(df[['close', 'SMA_3']].round(2))
  # Output:
  #              close  SMA_3
  # 2025-06-01    100    NaN
  # 2025-06-02    101    NaN
  # 2025-06-03    102  101.0
  # 2025-06-04    100  101.0
  # 2025-06-05     99  100.3
  ```
- **Performance Notes**:
  - Time complexity: O(n), where n is data length.
  - Numba reduces overhead compared to pandas `rolling`.
  - Memory: O(n) for output.
- **Edge Cases**:
  - `period` > data length raises `ValueError`.
  - Missing values propagate as `NaN`.
  - `period=1` returns the input prices.

### EMA (Exponential Moving Average)

- **Purpose**: Computes the Exponential Moving Average, weighting recent prices more heavily for reduced lag.
- **Mathematical Formula**:
  \[
  EMA_t = \alpha \cdot P_t + (1 - \alpha) \cdot EMA_{t-1}, \quad \alpha = \frac{2}{n+1}
  \]
  where \( P_t \) is the price, \( n \) is the period, and the initial EMA is the SMA over the first `period` observations.
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series or DataFrame with `'close'`.
  - `period`: `int` (default=20) - Period (positive).
- **Output**:
  - `pd.Series` - EMA values, named `EMA_{period}`.
- **Description**:
  Applies exponential weighting, validated with `_validate_and_get_prices`, and computed via `_calculate_ema_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Trend Following**: Used in MACD or EMA crossover strategies.
  - **Mean Reversion**: Identify price deviations from EMA.
  Example:
  ```python
  df['EMA_3'] = ti.EMA(data['close'], period=3)
  print(df[['close', 'EMA_3']].round(2))
  # Output:
  #              close  EMA_3
  # 2025-06-01    100    NaN
  # 2025-06-02    101    NaN
  # 2025-06-03    102  101.5
  # 2025-06-04    100  100.8
  # 2025-06-05     99   99.9
  ```
- **Performance Notes**:
  - Time complexity: O(n).
  - Numba outperforms pandas `ewm`.
  - Memory: O(n).
- **Edge Cases**:
  - Small `period` (e.g., 1) makes EMA highly volatile.
  - Missing values propagate `NaN`.

### RSI (Relative Strength Index)

- **Purpose**: Measures price momentum to identify overbought/oversold conditions.
- **Mathematical Formula**:
  \[
  RSI = 100 - \frac{100}{1 + RS}, \quad RS = \frac{\text{Average Gain}}{\text{Average Loss}}
  \]
  where Average Gain/Loss are EMAs over `period` observations.
- **Input Parameters**:
  - `data`: `pd.Series` - Price series (typically `'close'`).
  - `period`: `int` (default=14) - Lookback period (positive).
- **Output**:
  - `pd.Series` - RSI values (0-100), named `RSI_{period}`.
- **Description**:
  Computes relative strength of gains vs. losses, using `_calculate_rsi_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Overbought/Oversold**: RSI > 70 (overbought), < 30 (oversold).
  - **Divergence**: Spot price-RSI divergences for reversals.
  Example:
  ```python
  df['RSI_3'] = ti.RSI(data['close'], period=3)
  print(df[['close', 'RSI_3']].round(2))
  # Output (illustrative, requires more data for stability):
  #              close  RSI_3
  # 2025-06-01    100    NaN
  # 2025-06-02    101    NaN
  # 2025-06-03    102    NaN
  # 2025-06-04    100  33.33
  # 2025-06-05     99   0.00
  ```
- **Performance Notes**:
  - Time complexity: O(n).
  - Numba minimizes overhead.
  - Memory: O(n).
- **Edge Cases**:
  - Zero losses cause RS to be undefined (RSI set to 100).
  - Short periods produce volatile RSI.

### MACD (Moving Average Convergence Divergence)

- **Purpose**: Identifies trend changes using the difference between fast and slow EMAs.
- **Mathematical Formula**:
  \[
  MACD = EMA_{\text{fast}}(P) - EMA_{\text{slow}}(P)
  \]
  \[
  Signal = EMA_{\text{signal}}(MACD), \quad Histogram = MACD - Signal
  \]
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series or DataFrame with `'close'`.
  - `fast_period`: `int` (default=12) - Fast EMA period.
  - `slow_period`: `int` (default=26) - Slow EMA period.
  - `signal_period`: `int` (default=9) - Signal line EMA period.
- **Output**:
  - `pd.DataFrame` - Columns: `'MACD'`, `'Signal'`, `'Histogram'`.
- **Description**:
  Computes MACD line, signal line, and histogram using `_calculate_macd_numba`. First `slow_period-1` values are `NaN`.
- **Use Case**:
  - **Trend Signals**: Buy when MACD crosses above Signal; sell when below.
  - **Momentum**: Histogram divergence indicates momentum shifts.
  Example:
  ```python
  df[['MACD', 'Signal', 'Histogram']] = ti.MACD(data['close'], 3, 5, 2)
  print(df[['close', 'MACD', 'Signal']].round(2))
  # Output (illustrative):
  #              close  MACD  Signal
  # 2025-06-01    100   NaN     NaN
  # 2025-06-02    101   NaN     NaN
  # 2025-06-03    102   NaN     NaN
  # 2025-06-04    100  -0.5    -0.2
  # 2025-06-05     99  -0.8    -0.5
  ```
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for three output series.
- **Edge Cases**:
  - `fast_period` > `slow_period` may invert signals.
  - Small periods increase noise.

### BB (Bollinger Bands)

- **Purpose**: Measures volatility with a middle SMA and upper/lower bands based on standard deviation.
- **Mathematical Formula**:
  \[
  Middle = SMA_n(P), \quad Upper = Middle + k \cdot \sigma_n, \quad Lower = Middle - k \cdot \sigma_n
  \]
  where \( \sigma_n \) is the standard deviation over `period`, and \( k \) is `num_std`.
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series or DataFrame with `'close'`.
  - `period`: `int` (default=20) - SMA and standard deviation period.
  - `num_std`: `float` (default=2.0) - Standard deviations for bands.
- **Output**:
  - `pd.DataFrame` - Columns: `'BB_Upper'`, `'BB_Middle'`, `'BB_Lower'`.
- **Description**:
  Computes bands using `_calculate_bollinger_bands_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Breakouts**: Price crossing bands signals potential breakouts.
  - **Volatility**: Wider bands indicate higher volatility.
  Example:
  ```python
  df[['BB_Upper', 'BB_Middle', 'BB_Lower']] = ti.BB(data['close'], period=3, num_std=1.0)
  print(df[['close', 'BB_Middle']].round(2))
  # Output:
  #              close  BB_Middle
  # 2025-06-01    100        NaN
  # 2025-06-02    101        NaN
  # 2025-06-03    102      101.0
  # 2025-06-04    100      101.0
  # 2025-06-05     99      100.3
  ```
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for three series.
- **Edge Cases**:
  - Low volatility may collapse bands.
  - Missing data propagates `NaN`.

### ATR (Average True Range)

- **Purpose**: Measures volatility based on the true range of price movements.
- **Mathematical Formula**:
  \[
  TR = \max(High - Low, |High - Close_{t-1}|, |Low - Close_{t-1}|)
  \]
  \[
  ATR = EMA_n(TR)
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`.
  - `period`: `int` (default=14) - EMA period.
- **Output**:
  - `pd.Series` - ATR values, named `ATR_{period}`.
- **Description**:
  Computes true range and smooths it with an EMA using `_calculate_atr_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Stop-Loss Placement**: Use ATR multiples for dynamic stops.
  - **Volatility Filter**: High ATR indicates volatile markets.
  Example:
  ```python
  df['ATR_3'] = ti.ATR(data, period=3)
  print(df[['close', 'ATR_3']].round(2))
  # Output (illustrative):
  #              close  ATR_3
  # 2025-06-01    100    NaN
  # 2025-06-02    101    NaN
  # 2025-06-03    102    3.0
  # 2025-06-04    100    2.5
  # 2025-06-05     99    2.0
  ```
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Missing columns raise `ValueError`.
  - Low price movement may result in near-zero ATR.

### STOCH (Stochastic Oscillator)

- **Purpose**: Compares closing price to the price range to identify overbought/oversold conditions.
- **Mathematical Formula**:
  \[
  \%K = 100 \cdot \frac{Close - Low_n}{High_n - Low_n}, \quad \%D = SMA_m(\%K)
  \]
  where \( High_n \) and \( Low_n \) are the highest high and lowest low over `k_period`, and \( m \) is `d_period`.
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`.
  - `k_period`: `int` (default=14) - Lookback period for %K.
  - `d_period`: `int` (default=3) - SMA period for %D.
- **Output**:
  - `pd.DataFrame` - Columns: `'K'`, `'D'`.
- **Description**:
  Computes %K and %D using `_calculate_stochastic_numba`. First `k_period-1` %K and `k_period+d_period-2` %D values are `NaN`.
- **Use Case**:
  - **Overbought/Oversold**: %K > 80 (overbought), < 20 (oversold).
  - **Crossover Signals**: %K crossing %D generates buy/sell signals.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for two series.
- **Edge Cases**:
  - Flat price ranges cause division issues (handled in kernel).
  - Missing data propagates `NaN`.

### ADX (Average Directional Index)

- **Purpose**: Measures trend strength with ADX, +DI, and -DI.
- **Mathematical Formula**:
  \[
  +DM = High_t - High_{t-1}, \quad -DM = Low_{t-1} - Low_t
  \]
  \[
  +DI = 100 \cdot \frac{EMA_n(+DM)}{ATR}, \quad -DI = 100 \cdot \frac{EMA_n(-DM)}{ATR}
  \]
  \[
  ADX = 100 \cdot EMA_n\left(\frac{|+DI - -DI|}{|+DI + -DI|}\right)
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`.
  - `period`: `int` (default=14) - EMA period.
- **Output**:
  - `pd.DataFrame` - Columns: `'ADX'`, `'+DI'`, `'-DI'`.
- **Description**:
  Uses `_calculate_adx_numba`. First `2*period-1` ADX and `period` DI values are `NaN`.
- **Use Case**:
  - **Trend Strength**: ADX > 25 indicates a strong trend.
  - **Direction**: +DI > -DI suggests bullish trend.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for three series.
- **Edge Cases**:
  - Low volatility may produce unstable ADX.

### ICHIMOKU (Ichimoku Cloud)

- **Purpose**: Provides a comprehensive view of trend, momentum, and support/resistance.
- **Mathematical Formula**:
  \[
  Tenkan = \frac{High_n + Low_n}{2}, \quad Kijun = \frac{High_m + Low_m}{2}
  \]
  \[
  Senkou A = \frac{Tenkan + Kijun}{2}, \quad Senkou B = \frac{High_k + Low_k}{2}
  \]
  \[
  Chikou = Close_{t-displacement}
  \]
  where \( n \), \( m \), \( k \) are `tenkan_period`, `kijun_period`, `senkou_span_b_period`.
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`.
  - `tenkan_period`: `int` (default=9) - Tenkan-sen period.
  - `kijun_period`: `int` (default=26) - Kijun-sen period.
  - `senkou_span_b_period`: `int` (default=52) - Senkou Span B period.
  - `displacement`: `int` (default=26) - Forward/backward shift.
- **Output**:
  - `pd.DataFrame` - Columns: `'Tenkan-sen'`, `'Kijun-sen'`, `'Senkou Span A'`, `'Senkou Span B'`, `'Chikou Span'`.
- **Description**:
  Computes components using `_calculate_ichimoku_numba`. Senkou Spans are shifted forward, Chikou backward.
- **Use Case**:
  - **Trend Analysis**: Price above cloud indicates uptrend.
  - **Support/Resistance**: Cloud acts as dynamic levels.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for five series.
- **Edge Cases**:
  - Large `displacement` may truncate usable data.

### KELTNER (Keltner Channels)

- **Purpose**: Volatility-based bands using EMA and ATR.
- **Mathematical Formula**:
  \[
  Middle = EMA_n(P), \quad Upper = Middle + k \cdot ATR_m, \quad Lower = Middle - k \cdot ATR_m
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`.
  - `ema_period`: `int` (default=20) - EMA period.
  - `atr_period`: `int` (default=10) - ATR period.
  - `multiplier`: `float` (default=2.0) - ATR multiplier.
- **Output**:
  - `pd.DataFrame` - Columns: `'KC_Upper'`, `'KC_Middle'`, `'KC_Lower'`.
- **Description**:
  Uses `_calculate_keltner_channels_numba`. First `ema_period-1` values are `NaN`.
- **Use Case**:
  - **Breakouts**: Price crossing channels signals trend changes.
  - **Volatility**: Wider channels indicate higher volatility.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for three series.
- **Edge Cases**:
  - Small `atr_period` increases noise.

### MFI (Money Flow Index)

- **Purpose**: Volume-weighted momentum indicator for overbought/oversold conditions.
- **Mathematical Formula**:
  \[
  MFI = 100 - \frac{100}{1 + \frac{\text{Positive MF}}{\text{Negative MF}}}
  \]
  where Money Flow = Typical Price × Volume, and Typical Price = (High + Low + Close)/3.
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`, `'volume'`.
  - `period`: `int` (default=14) - Lookback period.
- **Output**:
  - `pd.Series` - MFI values (0-100), named `MFI_{period}`.
- **Description**:
  Uses `_calculate_mfi_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Overbought/Oversold**: MFI > 80 (overbought), < 20 (oversold).
  - **Divergence**: Spot price-MFI divergences.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Zero negative MF causes MFI = 100.

### TRIX

- **Purpose**: Triple-smoothed momentum oscillator for trend analysis.
- **Mathematical Formula**:
  \[
  TRIX = 100 \cdot \frac{EMA_n^3(P) - EMA_{n-1}^3(P)}{EMA_{n-1}^3(P)}
  \]
  where \( EMA_n^3 \) is the triple EMA.
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series or DataFrame with `'close'`.
  - `period`: `int` (default=15) - EMA period.
- **Output**:
  - `pd.Series` - TRIX values, named `TRIX_{period}`.
- **Description**:
  Uses `_calculate_trix_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Trend Reversals**: Zero-line crossovers signal trend changes.
  - **Momentum**: Rising TRIX indicates increasing momentum.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Small periods increase noise.

### CCI (Commodity Channel Index)

- **Purpose**: Momentum oscillator for overbought/oversold conditions.
- **Mathematical Formula**:
  \[
  CCI = \frac{Typical Price - SMA_n(Typical Price)}{k \cdot Mean Deviation}
  \]
  where Typical Price = (High + Low + Close)/3, and \( k \) is `constant`.
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`.
  - `period`: `int` (default=20) - Lookback period.
  - `constant`: `float` (default=0.015) - Scaling factor.
- **Output**:
  - `pd.Series` - CCI values, named `CCI_{period}`.
- **Description**:
  Uses `_calculate_cci_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Overbought/Oversold**: CCI > +100 (overbought), < -100 (oversold).
  - **Divergence**: Spot price-CCI divergences.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Low volatility may produce flat CCI.

### ROC (Rate of Change)

- **Purpose**: Measures percentage price change over a period.
- **Mathematical Formula**:
  \[
  ROC = 100 \cdot \frac{P_t - P_{t-n}}{P_{t-n}}
  \]
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series or DataFrame with `'close'`.
  - `period`: `int` (default=12) - Lookback period.
- **Output**:
  - `pd.Series` - ROC values, named `ROC_{period}`.
- **Description**:
  Uses `_calculate_roc_numba`. First `period` values are `NaN`.
- **Use Case**:
  - **Momentum**: Positive ROC indicates upward momentum.
  - **Trend Confirmation**: Use with other indicators.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Zero prices cause division errors (handled in kernel).

### WILLR (Williams %R)

- **Purpose**: Momentum indicator for overbought/oversold conditions.
- **Mathematical Formula**:
  \[
  \%R = 100 \cdot \frac{High_n - Close}{High_n - Low_n}
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`.
  - `period`: `int` (default=14) - Lookback period.
- **Output**:
  - `pd.Series` - %R values (-100 to 0), named `WILLR_{period}`.
- **Description**:
  Uses `_calculate_willr_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Overbought/Oversold**: %R > -20 (overbought), < -80 (oversold).
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Flat ranges cause division issues.

### DEMA (Double Exponential Moving Average)

- **Purpose**: Reduces lag compared to EMA using double smoothing.
- **Mathematical Formula**:
  \[
  DEMA = 2 \cdot EMA_n(P) - EMA_n(EMA_n(P))
  \]
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series or DataFrame with `'close'`.
  - `period`: `int` (default=20) - EMA period.
- **Output**:
  - `pd.Series` - DEMA values, named `DEMA_{period}`.
- **Description**:
  Uses `_calculate_dema_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Trend Following**: Faster response than EMA.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Small periods increase volatility.

### KAMA (Kaufman Adaptive Moving Average)

- **Purpose**: Adapts smoothing based on market volatility.
- **Mathematical Formula**:
  \[
  ER = \frac{|P_t - P_{t-n}|}{\sum_{i=t-n+1}^{t} |P_i - P_{i-1}|}
  \]
  \[
  SC = \left[ER \cdot \left(\frac{2}{f+1} - \frac{2}{s+1}\right) + \frac{2}{s+1}\right]^2
  \]
  \[
  KAMA_t = KAMA_{t-1} + SC \cdot (P_t - KAMA_{t-1})
  \]
  where \( f \) is `fast_period`, \( s \) is `slow_period`.
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series or DataFrame with `'close'`.
  - `er_period`: `int` (default=10) - Efficiency ratio period.
  - `fast_period`: `int` (default=2) - Fast EMA period.
  - `slow_period`: `int` (default=30) - Slow EMA period.
- **Output**:
  - `pd.Series` - KAMA values, named `KAMA_{er_period}`.
- **Description**:
  Uses `_calculate_kama_numba`. First `er_period-1` values are `NaN`.
- **Use Case**:
  - **Adaptive Trends**: Responsive in trends, smooth in ranges.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Low volatility may stall KAMA.

### DONCHIAN (Donchian Channels)

- **Purpose**: Identifies breakout levels based on price extremes.
- **Mathematical Formula**:
  \[
  Upper = \max(High_n), \quad Lower = \min(Low_n), \quad Middle = \frac{Upper + Lower}{2}
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`.
  - `period`: `int` (default=20) - Lookback period.
- **Output**:
  - `pd.DataFrame` - Columns: `'DC_Upper'`, `'DC_Middle'`, `'DC_Lower'`.
- **Description**:
  Uses `_calculate_donchian_channels_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Breakouts**: Price crossing channels signals trades.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for three series.
- **Edge Cases**:
  - Narrow ranges produce tight channels.

### AROON

- **Purpose**: Measures time since highest high/lowest low for trend strength.
- **Mathematical Formula**:
  \[
  Aroon Up = 100 \cdot \frac{n - \text{Periods since High}_n}{n}
  \]
  \[
  Aroon Down = 100 \cdot \frac{n - \text{Periods since Low}_n}{n}
  \]
  \[
  Aroon Oscillator = Aroon Up - Aroon Down
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`.
  - `period`: `int` (default=25) - Lookback period.
- **Output**:
  - `pd.DataFrame` - Columns: `'AROON_UP'`, `'AROON_DOWN'`, `'AROON_OSC'`.
- **Description**:
  Uses `_calculate_aroon_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Trend Strength**: High Aroon Up indicates strong uptrend.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for three series.
- **Edge Cases**:
  - Flat markets produce neutral values.

### AO (Awesome Oscillator)

- **Purpose**: Compares short- and long-term price momentum.
- **Mathematical Formula**:
  \[
  AO = SMA_m(\text{Median Price}) - SMA_n(\text{Median Price})
  \]
  where Median Price = (High + Low)/2.
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`.
  - `short_period`: `int` (default=5) - Short SMA period.
  - `long_period`: `int` (default=34) - Long SMA period.
- **Output**:
  - `pd.Series` - AO values, named `'AO'`.
- **Description**:
  Uses `_calculate_awesome_oscillator_numba`. First `long_period-1` values are `NaN`.
- **Use Case**:
  - **Momentum**: Positive AO indicates bullish momentum.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Small periods increase noise.

### ULTIMATE_OSCILLATOR

- **Purpose**: Combines momentum across three timeframes.
- **Mathematical Formula**:
  \[
  BP = Close - \min(Low, Close_{t-1}), \quad TR = \max(High, Close_{t-1}) - \min(Low, Close_{t-1})
  \]
  \[
  UO = 100 \cdot \frac{w_1 \cdot \sum BP_n + w_2 \cdot \sum BP_m + w_3 \cdot \sum BP_k}{w_1 \cdot \sum TR_n + w_2 \cdot \sum TR_m + w_3 \cdot \sum TR_k}
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`.
  - `period1`: `int` (default=7) - First period.
  - `period2`: `int` (default=14) - Second period.
  - `period3`: `int` (default=28) - Third period.
  - `weight1`: `float` (default=4.0) - First weight.
  - `weight2`: `float` (default=2.0) - Second weight.
  - `weight3`: `float` (default=1.0) - Third weight.
- **Output**:
  - `pd.Series` - UO values (0-100), named `'UO'`.
- **Description**:
  Uses `_calculate_ultimate_oscillator_numba`. First `max(period1, period2, period3)-1` values are `NaN`.
- **Use Case**:
  - **Overbought/Oversold**: UO > 70 (overbought), < 30 (oversold).
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Zero TR causes division issues.

### CMO (Chande Momentum Oscillator)

- **Purpose**: Measures momentum based on gains and losses.
- **Mathematical Formula**:
  \[
  CMO = 100 \cdot \frac{\sum_n (P_t - P_{t-1})^+ - \sum_n (P_t - P_{t-1})^-}{\sum_n |P_t - P_{t-1}|}
  \]
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series or DataFrame with `'close'`.
  - `period`: `int` (default=14) - Lookback period.
- **Output**:
  - `pd.Series` - CMO values (-100 to +100), named `CMO_{period}`.
- **Description**:
  Uses `_calculate_chande_momentum_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Momentum**: Positive CMO indicates bullish momentum.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Zero price changes cause division issues.

### DPO (Detrended Price Oscillator)

- **Purpose**: Removes trend to highlight price cycles.
- **Mathematical Formula**:
  \[
  DPO = P_t - SMA_n\left(P_{t-\lfloor n/2 \rfloor}\right)
  \]
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series or DataFrame with `'close'`.
  - `period`: `int` (default=20) - Lookback period.
- **Output**:
  - `pd.Series` - DPO values, named `DPO_{period}`.
- **Description**:
  Uses `_calculate_dpo_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Cycle Analysis**: Identify overbought/oversold within cycles.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Small periods reduce detrending effect.

### MASS_INDEX

- **Purpose**: Identifies reversals based on range expansion.
- **Mathematical Formula**:
  \[
  MI = \sum_{i=1}^m \frac{EMA_n(High - Low)}{EMA_n(EMA_n(High - Low))}
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`.
  - `ema_period`: `int` (default=9) - EMA period.
  - `sum_period`: `int` (default=25) - Summation period.
- **Output**:
  - `pd.Series` - MI values, named `MI_{ema_period}_{sum_period}`.
- **Description**:
  Uses `_calculate_mass_index_numba`. First `sum_period-1` values are `NaN`.
- **Use Case**:
  - **Reversals**: MI > 27 suggests potential reversals.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Low volatility produces low MI.

### VWAP (Volume Weighted Average Price)

- **Purpose**: Computes a volume-weighted price benchmark.
- **Mathematical Formula**:
  \[
  VWAP = \frac{\sum_n (Typical Price \cdot Volume)}{\sum_n Volume}
  \]
  where Typical Price = (High + Low + Close)/3.
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`, `'volume'`.
  - `period`: `int` (default=14) - Lookback period.
- **Output**:
  - `pd.Series` - VWAP values, named `VWAP_{period}`.
- **Description**:
  Uses `_calculate_vwap_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Trading Benchmark**: Compare price to VWAP for fair value.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Zero volume causes division issues.

### SUPERTREND

- **Purpose**: Trend-following indicator combining price and volatility.
- **Mathematical Formula**:
  \[
  Upper Band = \frac{High + Low}{2} + k \cdot ATR_n
  \]
  \[
  Lower Band = \frac{High + Low}{2} - k \cdot ATR_n
  \]
  Supertrend switches based on price crossing bands.
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`.
  - `period`: `int` (default=10) - ATR period.
  - `multiplier`: `float` (default=3.0) - ATR multiplier.
- **Output**:
  - `pd.DataFrame` - Columns: `'Supertrend'`, `'Direction'` (1 for uptrend, -1 for downtrend).
- **Description**:
  Uses `_calculate_supertrend_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Trend Following**: Trade in direction of Supertrend.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for two series.
- **Edge Cases**:
  - Choppy markets cause frequent switches.

### PVO (Percentage Volume Oscillator)

- **Purpose**: Measures volume momentum.
- **Mathematical Formula**:
  \[
  PVO = 100 \cdot \frac{EMA_m(Volume) - EMA_n(Volume)}{EMA_n(Volume)}
  \]
  \[
  Signal = EMA_k(PVO)
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Column: `'volume'`.
  - `short_period`: `int` (default=12) - Short EMA period.
  - `long_period`: `int` (default=26) - Long EMA period.
- **Output**:
  - `pd.DataFrame` - Columns: `'PVO'`, `'Signal'`, `'Histogram'`.
- **Description**:
  Uses `_calculate_pvo_numba`. First `long_period-1` values are `NaN`.
- **Use Case**:
  - **Volume Trends**: PVO crossovers signal volume shifts.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for three series.
- **Edge Cases**:
  - Low volume produces flat PVO.

### HISTORICAL_VOLATILITY

- **Purpose**: Measures annualized price volatility.
- **Mathematical Formula**:
  \[
  HV = \sqrt{\frac{252}{n} \sum_{i=t-n+1}^{t} (\ln(P_i / P_{i-1}))^2}
  \]
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series or DataFrame with `'close'`.
  - `period`: `int` (default=20) - Lookback period.
  - `trading_days`: `int` (default=252) - Annual trading days.
- **Output**:
  - `pd.Series` - HV values, named `HV_{period}`.
- **Description**:
  Uses `_calculate_historical_volatility_numba`. First `period` values are `NaN`.
- **Use Case**:
  - **Risk Management**: Use HV for position sizing.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Zero returns cause issues (handled in kernel).

### CHAIKIN_VOLATILITY

- **Purpose**: Measures rate of change in price range.
- **Mathematical Formula**:
  \[
  CV = 100 \cdot \frac{EMA_n(High - Low) - EMA_n(High - Low)_{t-m}}{EMA_n(High - Low)_{t-m}}
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`.
  - `ema_period`: `int` (default=10) - EMA period.
  - `roc_period`: `int` (default=10) - ROC period.
- **Output**:
  - `pd.Series` - CV values, named `CV_{ema_period}_{roc_period}`.
- **Description**:
  Uses `_calculate_chaikin_volatility_numba`. First `ema_period+roc_period-2` values are `NaN`.
- **Use Case**:
  - **Volatility Spikes**: High CV indicates increasing volatility.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Low ranges produce flat CV.

### LINEAR_REGRESSION_CHANNEL

- **Purpose**: Fits a linear trend with volatility bands.
- **Mathematical Formula**:
  \[
  Middle = \text{Linear Regression}(P, n)
  \]
  \[
  Upper = Middle + k \cdot \sigma, \quad Lower = Middle - k \cdot \sigma
  \]
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series or DataFrame with `'close'`.
  - `period`: `int` (default=20) - Regression period.
  - `deviations`: `float` (default=2.0) - Standard deviation multiplier.
- **Output**:
  - `pd.DataFrame` - Columns: `'LRC_Upper'`, `'LRC_Middle'`, `'LRC_Lower'`.
- **Description**:
  Uses `_calculate_linear_regression_channel_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Trend Analysis**: Middle line indicates trend direction.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for three series.
- **Edge Cases**:
  - Non-linear trends reduce accuracy.

### AD (Accumulation/Distribution Line)

- **Purpose**: Measures volume-based buying/selling pressure.
- **Mathematical Formula**:
  \[
  AD = AD_{t-1} + \left(\frac{Close - Low - (High - Close)}{High - Low} \cdot Volume\right)
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`, `'volume'`.
- **Output**:
  - `pd.Series` - AD values, named `'AD'`.
- **Description**:
  Uses `_calculate_ad_numba`. Cumulative sum starts at zero.
- **Use Case**:
  - **Divergence**: AD diverging from price signals reversals.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Zero range causes division issues.

### ALMA (Arnaud Legoux Moving Average)

- **Purpose**: Low-lag moving average with Gaussian weighting.
- **Mathematical Formula**:
  \[
  ALMA = \sum_{i=0}^{n-1} w_i \cdot P_{t-i}, \quad w_i = e^{-\frac{(i - \text{offset} \cdot n)^2}{2 \cdot \sigma^2}}
  \]
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series or DataFrame with `'close'`.
  - `period`: `int` (default=9) - Window size.
  - `sigma`: `float` (default=6.0) - Gaussian smoothness.
  - `offset`: `float` (default=0.85) - Weight shift.
- **Output**:
  - `pd.Series` - ALMA values, named `ALMA_{period}`.
- **Description**:
  Uses `_calculate_alma_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Trend Following**: Low-lag alternative to EMA.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Small `sigma` increases noise.

### KDJ

- **Purpose**: Extends Stochastic Oscillator with a J line.
- **Mathematical Formula**:
  \[
  K = \%K, \quad D = SMA_m(K), \quad J = 3 \cdot K - 2 \cdot D
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`.
  - `k_period`: `int` (default=9) - %K period.
  - `d_period`: `int` (default=3) - %D period.
- **Output**:
  - `pd.DataFrame` - Columns: `'K'`, `'D'`, `'J'`.
- **Description**:
  Uses `_calculate_kdj_numba`. First `k_period-1` K and `k_period+d_period-2` D/J values are `NaN`.
- **Use Case**:
  - **Overbought/Oversold**: J > 80 (overbought), < 20 (oversold).
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for three series.
- **Edge Cases**:
  - Flat ranges cause volatile J.

### HEIKEN_ASHI

- **Purpose**: Smoothed candlesticks to reduce noise.
- **Mathematical Formula**:
  \[
  HA Close = \frac{Open + High + Low + Close}{4}
  \]
  \[
  HA Open = \frac{HA Open_{t-1} + HA Close_{t-1}}{2}
  \]
  \[
  HA High = \max(High, HA Open, HA Close), \quad HA Low = \min(Low, HA Open, HA Close)
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'open'`, `'high'`, `'low'`, `'close'`.
- **Output**:
  - `pd.DataFrame` - Columns: `'HA_Open'`, `'HA_High'`, `'HA_Low'`, `'HA_Close'`.
- **Description**:
  Uses `_calculate_heiken_ashi_numba`. No `NaN` values unless input is missing.
- **Use Case**:
  - **Trend Identification**: Smoother candles highlight trends.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for four series.
- **Edge Cases**:
  - Missing candles disrupt continuity.

### BENFORD_LAW

- **Purpose**: Analyzes first-digit distribution per Benford’s Law.
- **Mathematical Formula**:
  \[
  P(d) = \log_{10}\left(1 + \frac{1}{d}\right), \quad d = 1, \ldots, 9
  \]
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series or DataFrame with `'close'`.
- **Output**:
  - `pd.DataFrame` - Columns: `'Observed'`, `'Expected'`, indexed by digits 1-9.
- **Description**:
  Computes observed vs. expected first-digit frequencies. No Numba kernel; uses pandas.
- **Use Case**:
  - **Anomaly Detection**: Deviations suggest data irregularities.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(1) for small output.
- **Edge Cases**:
  - Small datasets produce unreliable distributions.

### OBV (On-Balance Volume)

- **Purpose**: Measures volume-based buying/selling pressure.
- **Mathematical Formula**:
  \[
  OBV_t = OBV_{t-1} + \begin{cases} 
  Volume_t & \text{if } Close_t > Close_{t-1} \\
  -Volume_t & \text{if } Close_t < Close_{t-1} \\
  0 & \text{otherwise}
  \end{cases}
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'close'`, `'volume'`.
- **Output**:
  - `pd.Series` - OBV values, named `'OBV'`.
- **Description**:
  Pure Python implementation. Cumulative sum starts at zero.
- **Use Case**:
  - **Trend Confirmation**: Rising OBV confirms uptrends.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Zero volume produces flat OBV.

### BETA

- **Purpose**: Measures asset volatility relative to a market index.
- **Mathematical Formula**:
  \[
  \beta = \frac{\text{Cov}(R_a, R_m)}{\text{Var}(R_m)}
  \]
  where \( R_a \), \( R_m \) are asset and market returns.
- **Input Parameters**:
  - `data`: `pd.Series` - Asset price or returns.
  - `market_data`: `pd.Series` - Market price or returns.
  - `period`: `int` (default=252) - Lookback period.
- **Output**:
  - `pd.Series` - Beta values, named `BETA_{period}`.
- **Description**:
  Uses `_calculate_beta_numba`. First `period` values are `NaN`.
- **Use Case**:
  - **Risk Assessment**: Beta > 1 indicates higher volatility.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Zero market variance causes division issues.

### DI (Directional Indicator)

- **Purpose**: Measures bullish/bearish directional movement.
- **Mathematical Formula**:
  \[
  +DI = 100 \cdot \frac{EMA_n(+DM)}{ATR}, \quad -DI = 100 \cdot \frac{EMA_n(-DM)}{ATR}
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`.
  - `period`: `int` (default=14) - EMA period.
- **Output**:
  - `pd.DataFrame` - Columns: `'+DI'`, `'-DI'`.
- **Description**:
  Uses `_calculate_di_numba`. First `period` values are `NaN`.
- **Use Case**:
  - **Trend Direction**: +DI > -DI indicates bullish trend.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for two series.
- **Edge Cases**:
  - Low volatility produces flat DI.

### ADOSC (Chaikin A/D Oscillator)

- **Purpose**: Volume-based momentum indicator.
- **Mathematical Formula**:
  \[
  ADOSC = EMA_m(AD) - EMA_n(AD)
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`, `'volume'`.
  - `fast_period`: `int` (default=3) - Fast EMA period.
  - `slow_period`: `int` (default=10) - Slow EMA period.
- **Output**:
  - `pd.Series` - ADOSC values, named `ADOSC_{fast_period}_{slow_period}`.
- **Description**:
  Uses `_calculate_adosc_numba`. First `slow_period-1` values are `NaN`.
- **Use Case**:
  - **Momentum**: Positive ADOSC indicates buying pressure.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Low volume flattens ADOSC.

### VOLUME_INDICATORS

- **Purpose**: Computes volume SMA, Force Index, and Volume Price Trend.
- **Mathematical Formula**:
  \[
  Volume SMA = SMA_n(Volume)
  \]
  \[
  Force Index = (Close_t - Close_{t-1}) \cdot Volume
  \]
  \[
  VPT = VPT_{t-1} + Volume \cdot \frac{Close_t - Close_{t-1}}{Close_{t-1}}
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'close'`, `'volume'`.
  - `period`: `int` (default=20) - SMA period.
- **Output**:
  - `pd.DataFrame` - Columns: `'Volume_SMA'`, `'Force_Index'`, `'VPT'`.
- **Description**:
  Uses `_calculate_volume_indicators_numba`. First `period-1` Volume SMA and first Force Index values are `NaN`.
- **Use Case**:
  - **Volume Analysis**: Force Index confirms price moves.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for three series.
- **Edge Cases**:
  - Zero volume flattens VPT.

### HULL_MA (Hull Moving Average)

- **Purpose**: Low-lag moving average using weighted averages.
- **Mathematical Formula**:
  \[
  HMA = WMA_m(2 \cdot WMA_{\lfloor n/2 \rfloor}(P) - WMA_n(P))
  \]
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series or DataFrame with `'close'`.
  - `period`: `int` (default=10) - WMA period.
- **Output**:
  - `pd.Series` - HMA values, named `HULL_{period}`.
- **Description**:
  Uses `_calculate_hull_ma_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Trend Following**: Faster than EMA.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Small periods increase noise.

### PIVOT_POINTS

- **Purpose**: Calculates support/resistance levels.
- **Mathematical Formula**:
  \[
  PP = \frac{High + Low + Close}{3}
  \]
  \[
  R1 = 2 \cdot PP - Low, \quad S1 = 2 \cdot PP - High
  \]
  \[
  R2 = PP + (High - Low), \quad S2 = PP - (High - Low)
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`.
- **Output**:
  - `pd.DataFrame` - Columns: `'PP'`, `'R1'`, `'R2'`, `'S1'`, `'S2'`.
- **Description**:
  Uses `_calculate_pivot_points_numba`. No `NaN` unless input is missing.
- **Use Case**:
  - **Reversal Levels**: Trade at support/resistance.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for five series.
- **Edge Cases**:
  - Flat prices produce tight levels.

### ELDER_RAY

- **Purpose**: Measures bullish/bearish power relative to EMA.
- **Mathematical Formula**:
  \[
  Bull Power = High - EMA_n(Close), \quad Bear Power = Low - EMA_n(Close)
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`.
  - `period`: `int` (default=13) - EMA period.
- **Output**:
  - `pd.DataFrame` - Columns: `'Bull_Power'`, `'Bear_Power'`.
- **Description**:
  Uses `_calculate_elder_ray_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Trend Strength**: Positive Bull Power in uptrends.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for two series.
- **Edge Cases**:
  - Low volatility flattens powers.

### CHOPPINESS

- **Purpose**: Measures market trendiness vs. consolidation.
- **Mathematical Formula**:
  \[
  CI = 100 \cdot \frac{\log_{10}\left(\sum_n (High - Low)\right)}{\log_{10}\left(\max(High_n) - \min(Low_n)\right)}
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`.
  - `period`: `int` (default=14) - Lookback period.
- **Output**:
  - `pd.Series` - CI values (0-100), named `CI_{period}`.
- **Description**:
  Uses `_calculate_choppiness_index_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Market State**: CI > 61.8 (choppy), < 38.2 (trending).
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Flat ranges produce high CI.

### DISPARITY

- **Purpose**: Measures price deviation from SMA.
- **Mathematical Formula**:
  \[
  DI = 100 \cdot \frac{P_t - SMA_n(P)}{SMA_n(P)}
  \]
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series or DataFrame with `'close'`.
  - `period`: `int` (default=14) - SMA period.
- **Output**:
  - `pd.Series` - DI values, named `DI_{period}`.
- **Description**:
  Uses `_calculate_disparity_index_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Mean Reversion**: High DI suggests reversals.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Zero SMA causes division issues.

### COPPOCK

- **Purpose**: Long-term momentum for major bottoms.
- **Mathematical Formula**:
  \[
  Coppock = WMA_m(ROC_n(P) + ROC_k(P))
  \]
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series or DataFrame with `'close'`.
  - `roc1_period`: `int` (default=14) - First ROC period.
  - `roc2_period`: `int` (default=11) - Second ROC period.
  - `wma_period`: `int` (default=10) - WMA period.
- **Output**:
  - `pd.Series` - Coppock values, named `'COPPOCK'`.
- **Description**:
  Uses `_calculate_coppock_curve_numba`. First `max(roc1_period, roc2_period)+wma_period-1` values are `NaN`.
- **Use Case**:
  - **Buy Signals**: Upward zero crossings.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Long periods delay signals.

### RVI (Relative Vigor Index)

- **Purpose**: Measures trend strength based on price movement.
- **Mathematical Formula**:
  \[
  RVI = \frac{\sum_n (Close - Open)}{\sum_n (High - Low)}
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'open'`, `'high'`, `'low'`, `'close'`.
  - `period`: `int` (default=10) - Lookback period.
- **Output**:
  - `pd.Series` - RVI values, named `RVI_{period}`.
- **Description**:
  Uses `_calculate_rvi_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Trend Strength**: High RVI indicates bullish trends.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Flat ranges produce neutral RVI.

### PGO (Pretty Good Oscillator)

- **Purpose**: Combines price and trend for momentum.
- **Mathematical Formula**:
  \[
  PGO = \frac{Close - SMA_n(Close)}{\text{Mean Deviation}_n}
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`.
  - `period`: `int` (default=21) - Lookback period.
- **Output**:
  - `pd.Series` - PGO values, named `PGO_{period}`.
- **Description**:
  Uses `_calculate_pgo_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Momentum**: Positive PGO indicates bullish momentum.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Low volatility flattens PGO.

### PSL (Psychological Line)

- **Purpose**: Measures market sentiment based on price direction.
- **Mathematical Formula**:
  \[
  PSL = 100 \cdot \frac{\text{Number of Up Days}_n}{n}
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Column: `'close'`.
  - `period`: `int` (default=12) - Lookback period.
- **Output**:
  - `pd.Series` - PSL values (0-100), named `PSL_{period}`.
- **Description**:
  Uses `_calculate_psl_numba`. First `period-1` values are `NaN`.

- **Use Case**:
  - **Sentiment Analysis**: High PSL (>70) indicates bullish sentiment, low PSL (<30) suggests bearish sentiment.
- **Performance Notes**:
  - Time complexity: O(n), where n is the data length.
  - Memory: O(n) for the output series.
- **Edge Cases**:
  - Flat price movements (no up/down days) result in neutral PSL values.
  - Short periods increase volatility in PSL readings.

### RAINBOW (Rainbow Oscillator)

- **Purpose**: Computes multiple SMAs with different periods to analyze trend alignment across timeframes.
- **Mathematical Formula**:
  \[
  SMA_{p,i} = \frac{1}{p_i} \sum_{j=t-p_i+1}^{t} P_j, \quad \text{for each period } p_i \in \text{periods}
  \]
  where \( P_j \) is the price at time \( j \), and \( p_i \) is the i-th period in the input list.
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series or DataFrame with a `'close'` column.
  - `periods`: `List[int]` (default=[2, 3, 4, 5, 6, 7, 8, 9, 10]) - List of SMA periods.
- **Output**:
  - `pd.DataFrame` - Columns named `SMA_{period}` for each period in `periods`, indexed like the input data.
- **Description**:
  Calculates SMAs for each specified period using `_calculate_rainbow_numba`. For each SMA, the first `period-1` values are `NaN`. The function validates input with `_validate_and_get_prices` and converts data to a NumPy array for efficient computation.
- **Use Case**:
  - **Trend Alignment**: When shorter SMAs are above longer SMAs, it indicates a bullish trend; reverse for bearish.
  - **Divergence Analysis**: Spread between SMAs can indicate trend strength or consolidation.
  Example:
  ```python
  df = data.copy()
  rainbow = ti.RAINBOW(data['close'], periods=[2, 3])
  df[['SMA_2', 'SMA_3']] = rainbow[['SMA_2', 'SMA_3']]
  print(df[['close', 'SMA_2', 'SMA_3']].round(2))
  # Output:
  #              close  SMA_2  SMA_3
  # 2025-06-01    100    NaN    NaN
  # 2025-06-02    101  100.5    NaN
  # 2025-06-03    102  101.5  101.0
  # 2025-06-04    100  101.0  101.0
  # 2025-06-05     99  100.5  100.3
  ```
- **Performance Notes**:
  - Time complexity: O(n * k), where n is data length and k is the number of periods.
  - Memory: O(n * k) for k output series.
  - Numba optimization reduces overhead for multiple SMAs.
- **Edge Cases**:
  - Large `periods` lists increase computation time and memory usage.
  - Periods exceeding data length raise a `ValueError`.
  - Missing values propagate as `NaN`.

### MOMENTUM_INDEX

- **Purpose**: Separates positive and negative price movements to quantify momentum.
- **Mathematical Formula**:
  \[
  Positive Index = \sum_{i=t-n+1}^{t} \max(P_i - P_{i-1}, 0)
  \]
  \[
  Negative Index = \sum_{i=t-n+1}^{t} \max(P_{i-1} - P_i, 0)
  \]
  where \( P_i \) is the price at time \( i \), and \( n \) is the period.
- **Input Parameters**:
  - `data`: `pd.DataFrame` - DataFrame with a `'close'` column.
  - `period`: `int` (default=10) - Lookback period.
- **Output**:
  - `pd.DataFrame` - Columns: `'Positive_Index'`, `'Negative_Index'`, indexed like the input data.
- **Description**:
  Computes cumulative positive and negative price changes over the period using `_calculate_momentum_index_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Momentum Analysis**: Higher Positive Index relative to Negative Index signals bullish momentum.
  - **Divergence**: Divergences between indices and price indicate potential reversals.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for two output series.
- **Edge Cases**:
  - Flat markets produce zero or low index values.
  - Missing data propagates `NaN`.

### MA_MOMENTUM (Moving Average Momentum)

- **Purpose**: Measures the momentum of a Simple Moving Average.
- **Mathematical Formula**:
  \[
  MA Momentum = SMA_n(P)_t - SMA_n(P)_{t-m}
  \]
  where \( n \) is `ma_period`, and \( m \) is `momentum_period`.
- **Input Parameters**:
  - `data`: `Union[pd.Series, pd.DataFrame]` - Price series or DataFrame with a `'close'` column.
  - `ma_period`: `int` (default=10) - SMA period.
  - `momentum_period`: `int` (default=10) - Lookback period for momentum.
- **Output**:
  - `pd.Series` - MA Momentum values, named `MA_Momentum_{ma_period}_{momentum_period}`.
- **Description**:
  Computes the change in SMA over the momentum period using `_calculate_ma_momentum_numba`. First `ma_period+momentum_period-1` values are `NaN`.
- **Use Case**:
  - **Trend Momentum**: Positive values indicate increasing trend strength.
  - **Reversal Signals**: Zero crossings suggest trend changes.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Large `momentum_period` delays signals.
  - Missing data propagates `NaN`.

### QSTICK

- **Purpose**: Measures trend strength based on the difference between open and close prices.
- **Mathematical Formula**:
  \[
  QStick = SMA_n(Close - Open)
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'open'`, `'close'`.
  - `period`: `int` (default=10) - SMA period.
- **Output**:
  - `pd.Series` - QStick values, named `QStick_{period}`.
- **Description**:
  Computes the SMA of the difference between close and open prices using `_calculate_qstick_numba`. First `period-1` values are `NaN`.
- **Use Case**:
  - **Trend Direction**: Positive QStick indicates bullish trend, negative indicates bearish.
  - **Reversal Signals**: Zero crossings suggest trend reversals.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Small open-close differences produce flat QStick.
  - Missing data propagates `NaN`.

### TYPICAL_PRICE

- **Purpose**: Computes the average of high, low, and close prices as a reference price.
- **Mathematical Formula**:
  \[
  Typical Price = \frac{High + Low + Close}{3}
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`.
- **Output**:
  - `pd.Series` - Typical Price values, named `'Typical_Price'`.
- **Description**:
  Calculates the average price using `_calculate_typical_price_numba`. No `NaN` values unless input is missing.
- **Use Case**:
  - **Reference Price**: Used in indicators like MFI or VWAP.
  - **Simplified Price**: Represents a balanced price point.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Missing columns raise `ValueError`.
  - Extreme price spreads may skew results.

### WEIGHTED_CLOSE

- **Purpose**: Computes a weighted average price, emphasizing the closing price.
- **Mathematical Formula**:
  \[
  Weighted Close = \frac{High + Low + 2 \cdot Close}{4}
  \]
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`, `'close'`.
- **Output**:
  - `pd.Series` - Weighted Close values, named `'Weighted_Close'`.
- **Description**:
  Calculates a weighted price using `_calculate_weighted_close_numba`. No `NaN` values unless input is missing.
- **Use Case**:
  - **Smoothed Price**: Used as input for other indicators.
  - **Trend Analysis**: Emphasizes closing price for trend signals.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n).
- **Edge Cases**:
  - Missing columns raise `ValueError`.

### FRACTAL

- **Purpose**: Identifies potential reversal points using fractal patterns.
- **Mathematical Formula**:
  \[
  Bullish Fractal: Low_t < \min(Low_{t-1}, Low_{t+1}) \text{ and } Low_t < \min(Low_{t-2}, Low_{t+2})
  \]
  \[
  Bearish Fractal: High_t > \max(High_{t-1}, High_{t+1}) \text{ and } High_t > \max(High_{t-2}, High_{t+2})
  \]
  for a 5-period fractal (`period=5`).
- **Input Parameters**:
  - `data`: `pd.DataFrame` - Columns: `'high'`, `'low'`.
  - `period`: `int` (default=5) - Lookback period (must be odd).
- **Output**:
  - `pd.DataFrame` - Columns: `'Bullish_Fractal'`, `'Bearish_Fractal'`, with binary values (1 for fractal, 0 otherwise).
- **Description**:
  Identifies fractal patterns using `_calculate_fractal_numba`. Requires at least `period` data points.
- **Use Case**:
  - **Support/Resistance**: Fractals mark potential reversal levels.
  - **Breakout Trading**: Trade on breaks of fractal levels.
- **Performance Notes**:
  - Time complexity: O(n).
  - Memory: O(n) for two series.
- **Edge Cases**:
  - Even `period` values are invalid (must be odd).
  - Insufficient data raises errors.

