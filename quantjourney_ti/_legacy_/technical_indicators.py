#!/usr/bin/env python3
"""
Technical Indicators Library
============================

This module provides a custom implementation of various technical indicators for financial analysis,
originally developed as part of the QuantJourney project (quantjourney.substack.com) by Jakub.
It is now a standalone, open-source library under the MIT License, free for anyone to use, fork, and
contribute to. The implementation avoids dependencies like TA-Lib, relying on NumPy, Pandas, and Numba
for high-performance calculations.

Features:
- Wide range of technical indicators (e.g., SMA, EMA, RSI, MACD, etc.)
- Optimized with Numba for speed
- No external library dependencies beyond NumPy, Pandas, and yfinance
- Includes utility methods for validation and plotting

License: MIT License - see LICENSE.md for details.

Contributions are welcome! Submit pull requests or report issues on the GitHub repository:
https://github.com/QuantJourneyOrg/qj_technical_indicators

For questions or feedback, contact Jakub Polec at jakub@quantjourney.pro.

Last Updated: April 09, 2025
"""

import numpy as np
import pandas as pd
from numba import njit, prange
from typing import Tuple, Union, Dict, Optional, List
from enum import Enum
import logging
from .decorators import timer

# Set up generic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# TechnicalIndicators class ------------------------------------------------------
class TechnicalIndicators:
    """
    High-performance technical indicators implementation with Numba optimization.

    Provides methods for:
    - Supportive utilities (data validation, price extraction, plotting)
    - Private calculation methods (optimized with Numba)
    - Public indicator methods (e.g., SMA, EMA, RSI)
    """

    def __init__(self):
        # Warm-up Numba functions
        dummy_array = np.ones(10, dtype=np.float64)
        self._calculate_sma_numba(dummy_array, 5)
        self._calculate_ema_numba(dummy_array, 5)
        self._calculate_rsi_numba(dummy_array, 5)
        self._calculate_macd_numba(dummy_array, 5, 5, 5)
        self._calculate_bollinger_bands_numba(dummy_array, 5, 5)
        self._calculate_cci_numba(dummy_array, dummy_array, dummy_array, 5)
        self._calculate_roc_numba(dummy_array, 5)
        self._calculate_willr_numba(dummy_array, dummy_array, dummy_array, 5)

    # Supportive methods -------------------------------------------------------

    @staticmethod
    def _validate_data(
        data: Union[pd.DataFrame, pd.Series], required_columns: list = None
    ) -> bool:
        """Validate input data and required columns"""
        if required_columns:
            if not isinstance(data, pd.DataFrame):
                raise ValueError(
                    f"Data must be a DataFrame with columns: {required_columns}"
                )
            missing = [col for col in required_columns if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

        if len(data) < 2:
            raise ValueError("Data must contain at least 2 rows")

        if data.isnull().any().any():
            logger.warning(
                "Data contains NaN values, they will be handled in calculations"
            )

        return True

    @staticmethod
    def _validate_and_get_prices(
        data: Union[pd.Series, pd.DataFrame], price_col: str = "adj_close"
    ) -> pd.Series:
        """Validate and extract price data from input"""
        if isinstance(data, pd.Series):
            return data
        elif isinstance(data, pd.DataFrame):
            if price_col in data.columns:
                return data[price_col]
            elif "close" in data.columns:
                logger.warning(f"'{price_col}' not found, using 'close' instead.")
                return data["close"]
            else:
                raise ValueError(
                    f"Neither '{price_col}' nor 'close' found in DataFrame"
                )
        else:
            raise ValueError("Input must be a pandas Series or DataFrame")

    @staticmethod
    def detect_divergence(
        price: pd.Series, indicator: pd.Series, window: int = 20
    ) -> pd.DataFrame:
        """Detect bullish and bearish divergences between price and indicator"""
        # Validate inputs
        if len(price) != len(indicator):
            raise ValueError("Price and indicator series must have the same length")

        divergence = pd.DataFrame(index=price.index)
        divergence["bullish"] = 0
        divergence["bearish"] = 0

        for i in range(window, len(price)):
            # Get window of data
            price_window = price[i - window : i]
            indicator_window = indicator[i - window : i]

            # Find local extrema
            price_min = price_window.min()
            price_max = price_window.max()
            ind_min = indicator_window.min()
            ind_max = indicator_window.max()

            # Detect divergences
            if price[i] < price_min and indicator[i] > ind_min:
                divergence.loc[price.index[i], "bullish"] = 1

            if price[i] > price_max and indicator[i] < ind_max:
                divergence.loc[price.index[i], "bearish"] = 1

        return divergence

    @staticmethod
    def detect_crossovers(series1: pd.Series, series2: pd.Series) -> pd.DataFrame:
        """Detect crossovers between two series"""
        if len(series1) != len(series2):
            raise ValueError("Both series must have the same length")

        crossovers = pd.DataFrame(index=series1.index)
        crossovers["bullish"] = 0
        crossovers["bearish"] = 0

        # Previous state
        prev_diff = series1 - series2

        # Current state
        curr_diff = prev_diff.shift(-1)

        # Detect crossovers
        crossovers["bullish"] = ((prev_diff < 0) & (curr_diff > 0)).astype(int)
        crossovers["bearish"] = ((prev_diff > 0) & (curr_diff < 0)).astype(int)

        return crossovers

    def plot_indicators(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        title: str = "Technical Analysis",
        figsize: Tuple[int, int] = (15, 10),
        overlay: bool = False,
    ):
        """Plot price data with multiple indicators"""
        try:
            import matplotlib.pyplot as plt

            if overlay:
                fig, ax = plt.subplots(figsize=figsize)

                # Plot price
                ax.plot(data.index, data["close"], label="Close Price", color="black")

                # Plot all indicators on same axis
                for name, indicator in indicators.items():
                    ax.plot(data.index, indicator, label=name, alpha=0.7)

                ax.set_title(title)
                ax.set_ylabel("Value")
                ax.grid(True)
                ax.legend()

            else:
                fig, (ax1, ax2) = plt.subplots(
                    2, 1, figsize=figsize, height_ratios=[2, 1], sharex=True
                )

                # Plot price
                ax1.plot(data.index, data["close"], label="Close Price", color="black")

                # Plot overlay indicators
                for name, indicator in indicators.items():
                    if "overlay" in name.lower():
                        ax1.plot(data.index, indicator, label=name, alpha=0.7)
                    else:
                        ax2.plot(data.index, indicator, label=name, alpha=0.7)

                ax1.set_title(title)
                ax1.set_ylabel("Price")
                ax1.grid(True)
                ax1.legend()

                ax2.set_ylabel("Indicator Values")
                ax2.grid(True)
                ax2.legend()

            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("Matplotlib is required for plotting")

    @staticmethod
    def validate_window(data_length: int, window: int, min_window: int = 2) -> bool:
        """Validate window size for calculations"""
        if window < min_window:
            raise ValueError(f"Window size must be at least {min_window}")

        if window >= data_length:
            raise ValueError(
                f"Window size ({window}) must be less than data length ({data_length})"
            )

        return True

    def _optimize_memory(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]

        for col in data.select_dtypes(include=numerics).columns:
            col_type = data[col].dtype
            if col_type in ["int64", "float64"]:
                c_min = data[col].min()
                c_max = data[col].max()

                if col_type == "int64":
                    if (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        data[col] = data[col].astype(np.int32)
                else:
                    if (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        data[col] = data[col].astype(np.float32)

        return data

    # Indicator calculations (private) ---------------------------------------------------

    @staticmethod
    @njit(parallel=False, fastmath=True)
    def _calculate_sma_numba(prices: np.ndarray, window: int) -> np.ndarray:
        """Optimized Simple Moving Average calculation"""
        sma = np.full_like(prices, np.nan, dtype=np.float64)  # Initialize all to NaN
        n = len(prices)

        if window > n:
            return sma  # Return all NaNs if window is larger than data length

        running_sum = 0.0
        # Calculate initial running sum for the first 'window' elements
        for i in range(window):
            running_sum += prices[i]
        sma[window - 1] = running_sum / window

        # Slide the window and update the running sum
        for i in prange(window, n):
            running_sum += prices[i] - prices[i - window]
            sma[i] = running_sum / window

        return sma  # Return the NumPy array

    @staticmethod
    @njit(parallel=False, fastmath=True)
    def _calculate_ema_numba(prices: np.ndarray, window: int) -> np.ndarray:
        """Optimized Exponential Moving Average calculation"""
        ema = np.zeros_like(prices)
        multiplier = 2.0 / (window + 1)

        # Initialize first value
        ema[window - 1] = np.mean(prices[:window])

        # Calculate EMA using the multiplier
        for i in prange(window, len(prices)):
            ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]

        # Set initial values to NaN
        ema[: window - 1] = np.nan
        return ema

    @staticmethod
    @njit(parallel=False, fastmath=True)
    def _calculate_rsi_numba(prices: np.ndarray, window: int) -> np.ndarray:
        """Optimized RSI calculation"""
        rsi = np.full_like(prices, np.nan)
        gains = np.zeros_like(prices)
        losses = np.zeros_like(prices)

        # Calculate price changes
        changes = np.zeros_like(prices)
        changes[1:] = prices[1:] - prices[:-1]

        # Separate gains and losses
        gains[changes > 0] = changes[changes > 0]
        losses[changes < 0] = -changes[changes < 0]

        # Calculate average gains and losses
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)

        # First average
        avg_gain[window] = np.mean(gains[1 : window + 1])
        avg_loss[window] = np.mean(losses[1 : window + 1])

        # Calculate subsequent averages
        for i in prange(window + 1, len(prices)):
            avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gains[i]) / window
            avg_loss[i] = (avg_loss[i - 1] * (window - 1) + losses[i]) / window

        # Calculate RSI
        rs = avg_gain[window:] / np.maximum(
            avg_loss[window:], 1e-10
        )  # Avoid division by zero
        rsi[window:] = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    @njit(parallel=False, fastmath=True)
    def _calculate_macd_numba(
        prices: np.ndarray, fast_period: int, slow_period: int, signal_period: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optimized MACD calculation"""
        macd_line = np.zeros_like(prices)
        signal_line = np.zeros_like(prices)
        histogram = np.zeros_like(prices)

        # Calculate multipliers
        fast_multiplier = 2.0 / (fast_period + 1)
        slow_multiplier = 2.0 / (slow_period + 1)
        signal_multiplier = 2.0 / (signal_period + 1)

        # Initialize EMA values
        fast_ema = prices[0]
        slow_ema = prices[0]
        signal_ema = 0.0

        for i in prange(len(prices)):
            fast_ema = (prices[i] - fast_ema) * fast_multiplier + fast_ema
            slow_ema = (prices[i] - slow_ema) * slow_multiplier + slow_ema
            macd_line[i] = fast_ema - slow_ema

            if i == slow_period - 1:
                signal_ema = macd_line[i]
            elif i >= slow_period:
                signal_ema = (
                    macd_line[i] - signal_ema
                ) * signal_multiplier + signal_ema
                signal_line[i] = signal_ema
                histogram[i] = macd_line[i] - signal_ema
            else:
                signal_line[i] = np.nan
                histogram[i] = np.nan

        # Handle initial values
        for i in prange(slow_period - 1):
            macd_line[i] = np.nan
            signal_line[i] = np.nan
            histogram[i] = np.nan

        return macd_line, signal_line, histogram

    @staticmethod
    @njit(parallel=False, fastmath=True)
    def _calculate_bollinger_bands_numba(
        prices: np.ndarray, window: int, num_std: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optimized Bollinger Bands calculation"""
        middle_band = np.full_like(prices, np.nan, dtype=np.float64)
        upper_band = np.full_like(prices, np.nan, dtype=np.float64)
        lower_band = np.full_like(prices, np.nan, dtype=np.float64)

        for i in prange(window - 1, len(prices)):
            sum_prices = 0.0
            sum_sq = 0.0
            for j in range(i - window + 1, i + 1):
                sum_prices += prices[j]
                sum_sq += prices[j] * prices[j]
            mean = sum_prices / window
            variance = (sum_sq / window) - (mean * mean)
            std = np.sqrt(variance) if variance > 0 else 0.0
            middle_band[i] = mean
            upper_band[i] = mean + (std * num_std)
            lower_band[i] = mean - (std * num_std)

        return upper_band, middle_band, lower_band

    @staticmethod
    @njit(parallel=False, fastmath=True)
    def _calculate_atr_numba(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int
    ) -> np.ndarray:
        """Optimized ATR calculation"""
        tr = np.zeros(len(high), dtype=np.float64)
        atr = np.full(len(high), np.nan, dtype=np.float64)

        for i in prange(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Calculate initial ATR
        if window <= len(tr):
            initial_sum = 0.0
            for i in range(1, window + 1):
                initial_sum += tr[i]
            atr[window] = initial_sum / window

            # Calculate subsequent ATR values
            for i in prange(window + 1, len(high)):
                atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window

        return atr

    @staticmethod
    @njit
    def _calculate_stochastic_numba(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int,
        d_period: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized Stochastic Oscillator calculation"""
        k_line = np.zeros(len(close))
        d_line = np.zeros(len(close))

        for i in range(k_period - 1, len(close)):
            window_high = np.max(high[i - k_period + 1 : i + 1])
            window_low = np.min(low[i - k_period + 1 : i + 1])
            k_line[i] = 100 * (close[i] - window_low) / (window_high - window_low)

        # Calculate D line (SMA of K line)
        for i in range(k_period + d_period - 2, len(close)):
            d_line[i] = np.mean(k_line[i - d_period + 1 : i + 1])

        return k_line, d_line

    @staticmethod
    @njit
    def _calculate_adx_numba(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optimized ADX calculation"""
        tr = np.zeros(len(high))
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))

        # Calculate True Range and Directional Movement
        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0

            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0

        # Calculate smoothed values
        smoothed_tr = np.zeros(len(high))
        smoothed_plus_dm = np.zeros(len(high))
        smoothed_minus_dm = np.zeros(len(high))

        # Initialize first values
        smoothed_tr[window] = np.sum(tr[1 : window + 1])
        smoothed_plus_dm[window] = np.sum(plus_dm[1 : window + 1])
        smoothed_minus_dm[window] = np.sum(minus_dm[1 : window + 1])

        # Calculate smoothed values
        for i in range(window + 1, len(high)):
            smoothed_tr[i] = smoothed_tr[i - 1] - (smoothed_tr[i - 1] / window) + tr[i]
            smoothed_plus_dm[i] = (
                smoothed_plus_dm[i - 1]
                - (smoothed_plus_dm[i - 1] / window)
                + plus_dm[i]
            )
            smoothed_minus_dm[i] = (
                smoothed_minus_dm[i - 1]
                - (smoothed_minus_dm[i - 1] / window)
                + minus_dm[i]
            )

        # Calculate DI+ and DI-
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr

        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = np.zeros(len(high))

        # Calculate initial ADX
        adx[2 * window - 1] = np.mean(dx[window : 2 * window])

        # Calculate remaining ADX values
        for i in range(2 * window, len(high)):
            adx[i] = (adx[i - 1] * (window - 1) + dx[i]) / window

        return adx, plus_di, minus_di

    @staticmethod
    @njit
    def _calculate_ichimoku_numba(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        tenkan_period: int,
        kijun_period: int,
        senkou_span_b_period: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Optimized Ichimoku Cloud calculation"""
        tenkan_sen = np.zeros(len(close))
        kijun_sen = np.zeros(len(close))
        senkou_span_a = np.zeros(len(close))
        senkou_span_b = np.zeros(len(close))

        # Calculate Tenkan-sen and Kijun-sen
        for i in range(tenkan_period - 1, len(close)):
            tenkan_high = np.max(high[i - tenkan_period + 1 : i + 1])
            tenkan_low = np.min(low[i - tenkan_period + 1 : i + 1])
            tenkan_sen[i] = (tenkan_high + tenkan_low) / 2

        for i in range(kijun_period - 1, len(close)):
            kijun_high = np.max(high[i - kijun_period + 1 : i + 1])
            kijun_low = np.min(low[i - kijun_period + 1 : i + 1])
            kijun_sen[i] = (kijun_high + kijun_low) / 2

        # Calculate Senkou Span A and B
        for i in range(kijun_period - 1, len(close)):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2

        for i in range(senkou_span_b_period - 1, len(close)):
            senkou_high = np.max(high[i - senkou_span_b_period + 1 : i + 1])
            senkou_low = np.min(low[i - senkou_span_b_period + 1 : i + 1])
            senkou_span_b[i] = (senkou_high + senkou_low) / 2

        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b

    @staticmethod
    @njit
    def _calculate_keltner_channels_numba(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        ema_period: int,
        atr_period: int,
        multiplier: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optimized Keltner Channels calculation"""
        # Calculate EMA of close prices
        ema = np.zeros_like(close)
        multiplier_ema = 2.0 / (ema_period + 1)
        ema[0] = close[0]

        for i in range(1, len(close)):
            ema[i] = (close[i] - ema[i - 1]) * multiplier_ema + ema[i - 1]

        # Calculate ATR
        tr = np.zeros_like(close)
        atr = np.zeros_like(close)

        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        # Calculate ATR with smoothing
        atr[atr_period] = np.mean(tr[1 : atr_period + 1])
        for i in range(atr_period + 1, len(close)):
            atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period

        # Calculate bands
        upper = ema + (multiplier * atr)
        lower = ema - (multiplier * atr)

        return upper, ema, lower

    @staticmethod
    @njit
    def _calculate_mfi_numba(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        period: int,
    ) -> np.ndarray:
        """Optimized Money Flow Index calculation"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        positive_flow = np.zeros_like(close)
        negative_flow = np.zeros_like(close)

        # Calculate positive and negative money flow
        for i in range(1, len(close)):
            if typical_price[i] > typical_price[i - 1]:
                positive_flow[i] = money_flow[i]
                negative_flow[i] = 0
            elif typical_price[i] < typical_price[i - 1]:
                positive_flow[i] = 0
                negative_flow[i] = money_flow[i]
            else:
                positive_flow[i] = 0
                negative_flow[i] = 0

        # Calculate MFI
        mfi = np.zeros_like(close)
        for i in range(period, len(close)):
            positive_sum = np.sum(positive_flow[i - period + 1 : i + 1])
            negative_sum = np.sum(negative_flow[i - period + 1 : i + 1])

            if negative_sum == 0:
                mfi[i] = 100
            else:
                money_ratio = positive_sum / negative_sum
                mfi[i] = 100 - (100 / (1 + money_ratio))

        return mfi

    @staticmethod
    @njit
    def _calculate_trix_numba(close: np.ndarray, period: int) -> np.ndarray:
        """Optimized TRIX calculation"""
        ema1 = np.zeros_like(close)
        ema2 = np.zeros_like(close)
        ema3 = np.zeros_like(close)
        trix = np.zeros_like(close)

        # First EMA
        multiplier = 2.0 / (period + 1)
        ema1[0] = close[0]
        for i in range(1, len(close)):
            ema1[i] = (close[i] - ema1[i - 1]) * multiplier + ema1[i - 1]

        # Second EMA
        ema2[0] = ema1[0]
        for i in range(1, len(close)):
            ema2[i] = (ema1[i] - ema2[i - 1]) * multiplier + ema2[i - 1]

        # Third EMA
        ema3[0] = ema2[0]
        for i in range(1, len(close)):
            ema3[i] = (ema2[i] - ema3[i - 1]) * multiplier + ema3[i - 1]

        # Calculate TRIX
        for i in range(1, len(close)):
            trix[i] = ((ema3[i] - ema3[i - 1]) / ema3[i - 1]) * 100

        return trix

    @staticmethod
    @njit(parallel=False, fastmath=True)
    def _calculate_cci_numba(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int,
        constant: float = 0.015,
    ) -> np.ndarray:
        """Optimized Commodity Channel Index calculation"""
        tp = (high + low + close) / 3.0
        cci = np.full(len(close), np.nan, dtype=np.float64)

        for i in prange(period - 1, len(close)):
            sum_tp = 0.0
            for j in range(i - period + 1, i + 1):
                sum_tp += tp[j]
            mean_tp = sum_tp / period

            sum_dev = 0.0
            for j in range(i - period + 1, i + 1):
                sum_dev += abs(tp[j] - mean_tp)
            mean_deviation = sum_dev / period

            if mean_deviation != 0:
                cci[i] = (tp[i] - mean_tp) / (constant * mean_deviation)
            else:
                cci[i] = 0.0  # Avoid division by zero

        return cci

    @staticmethod
    @njit(parallel=False, fastmath=True)
    def _calculate_roc_numba(close: np.ndarray, period: int) -> np.ndarray:
        """Optimized Rate of Change calculation"""
        roc = np.full(len(close), np.nan, dtype=np.float64)

        for i in prange(period, len(close)):
            if close[i - period] != 0:
                roc[i] = ((close[i] - close[i - period]) / close[i - period]) * 100.0
            else:
                roc[i] = 0.0  # Avoid division by zero

        return roc

    @staticmethod
    @njit(parallel=False, fastmath=True)
    def _calculate_willr_numba(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        """Optimized Williams %R calculation"""
        willr = np.full(len(close), np.nan, dtype=np.float64)

        for i in prange(period - 1, len(close)):
            highest_high = high[i - period + 1]
            lowest_low = low[i - period + 1]

            for j in range(i - period + 1, i + 1):
                if high[j] > highest_high:
                    highest_high = high[j]
                if low[j] < lowest_low:
                    lowest_low = low[j]

            if highest_high == lowest_low:
                willr[i] = -50.0  # Neutral value if no range
            else:
                willr[i] = (
                    (highest_high - close[i]) / (highest_high - lowest_low)
                ) * -100.0

        return willr

    @staticmethod
    @njit
    def _calculate_dema_numba(close: np.ndarray, period: int) -> np.ndarray:
        """Optimized Double Exponential Moving Average calculation"""
        alpha = 2.0 / (period + 1)

        # First EMA calculation
        ema1 = np.zeros_like(close)
        ema1[0] = close[0]
        for i in range(1, len(close)):
            ema1[i] = (close[i] - ema1[i - 1]) * alpha + ema1[i - 1]

        # Second EMA calculation
        ema2 = np.zeros_like(close)
        ema2[0] = ema1[0]
        for i in range(1, len(close)):
            ema2[i] = (ema1[i] - ema2[i - 1]) * alpha + ema2[i - 1]

        # DEMA = 2 * EMA1 - EMA2
        dema = 2 * ema1 - ema2
        return dema

    @staticmethod
    @njit
    def _calculate_kama_numba(
        close: np.ndarray,
        er_period: int = 10,
        fast_period: int = 2,
        slow_period: int = 30,
    ) -> np.ndarray:
        """Optimized Kaufman Adaptive Moving Average calculation"""
        er = np.zeros_like(close)  # Efficiency Ratio
        kama = np.zeros_like(close)

        # Calculate direction and volatility
        direction = np.abs(close[er_period:] - close[:-er_period])
        volatility = np.zeros_like(close)

        for i in range(er_period, len(close)):
            volatility[i] = np.sum(
                np.abs(close[i - er_period + 1 : i + 1] - close[i - er_period : i])
            )

        # Calculate Efficiency Ratio
        er[er_period:] = np.where(
            volatility[er_period:] != 0, direction / volatility[er_period:], 0
        )

        # Calculate smoothing constant
        fast_alpha = 2.0 / (fast_period + 1)
        slow_alpha = 2.0 / (slow_period + 1)
        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2

        # Initialize KAMA
        kama[er_period] = close[er_period]

        # Calculate KAMA
        for i in range(er_period + 1, len(close)):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])

        return kama

    @staticmethod
    @njit
    def _calculate_aroon_numba(
        high: np.ndarray, low: np.ndarray, period: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized Aroon Indicator calculation"""
        aroon_up = np.zeros_like(high)
        aroon_down = np.zeros_like(low)

        for i in range(period, len(high)):
            # Find the number of periods since the highest high
            high_window = high[i - period + 1 : i + 1]
            high_idx = np.argmax(high_window)
            aroon_up[i] = ((period - (period - high_idx - 1)) / period) * 100

            # Find the number of periods since the lowest low
            low_window = low[i - period + 1 : i + 1]
            low_idx = np.argmin(low_window)
            aroon_down[i] = ((period - (period - low_idx - 1)) / period) * 100

        return aroon_up, aroon_down

    @staticmethod
    @njit
    def _calculate_awesome_oscillator_numba(
        high: np.ndarray, low: np.ndarray, short_period: int = 5, long_period: int = 34
    ) -> np.ndarray:
        """Optimized Awesome Oscillator calculation"""
        ao = np.zeros_like(high)
        median_price = (high + low) / 2

        # Calculate simple moving averages
        for i in range(long_period - 1, len(high)):
            short_sma = np.mean(median_price[i - short_period + 1 : i + 1])
            long_sma = np.mean(median_price[i - long_period + 1 : i + 1])
            ao[i] = short_sma - long_sma

        return ao

    @staticmethod
    @njit
    def _calculate_ultimate_oscillator_numba(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28,
        weight1: float = 4.0,
        weight2: float = 2.0,
        weight3: float = 1.0,
    ) -> np.ndarray:
        """Optimized Ultimate Oscillator calculation"""
        uo = np.zeros_like(close)
        buying_pressure = np.zeros_like(close)
        true_range = np.zeros_like(close)

        # Calculate Buying Pressure and True Range
        for i in range(1, len(close)):
            buying_pressure[i] = close[i] - min(low[i], close[i - 1])
            true_range[i] = max(high[i], close[i - 1]) - min(low[i], close[i - 1])

        # Calculate averages for different periods
        for i in range(max(period1, period2, period3), len(close)):
            avg7 = np.sum(buying_pressure[i - period1 + 1 : i + 1]) / np.sum(
                true_range[i - period1 + 1 : i + 1]
            )
            avg14 = np.sum(buying_pressure[i - period2 + 1 : i + 1]) / np.sum(
                true_range[i - period2 + 1 : i + 1]
            )
            avg28 = np.sum(buying_pressure[i - period3 + 1 : i + 1]) / np.sum(
                true_range[i - period3 + 1 : i + 1]
            )

            uo[i] = (
                100
                * (weight1 * avg7 + weight2 * avg14 + weight3 * avg28)
                / (weight1 + weight2 + weight3)
            )

        return uo

    @staticmethod
    @njit
    def _calculate_dpo_numba(close: np.ndarray, period: int) -> np.ndarray:
        """Optimized Detrended Price Oscillator calculation"""
        dpo = np.zeros_like(close)
        shift = period // 2 + 1

        # Calculate centered SMA
        for i in range(period - 1, len(close)):
            sma = np.mean(close[i - period + 1 : i + 1])
            if i - shift >= 0:
                dpo[i] = close[i - shift] - sma

        return dpo

    @staticmethod
    @njit
    def _calculate_mass_index_numba(
        high: np.ndarray, low: np.ndarray, ema_period: int = 9, sum_period: int = 25
    ) -> np.ndarray:
        """Optimized Mass Index calculation"""
        mi = np.zeros_like(high)
        ema1 = np.zeros_like(high)
        ema2 = np.zeros_like(high)
        diff = high - low

        # Calculate first EMA
        alpha1 = 2.0 / (ema_period + 1)
        ema1[0] = diff[0]
        for i in range(1, len(high)):
            ema1[i] = (diff[i] - ema1[i - 1]) * alpha1 + ema1[i - 1]

        # Calculate second EMA
        ema2[0] = ema1[0]
        for i in range(1, len(high)):
            ema2[i] = (ema1[i] - ema2[i - 1]) * alpha1 + ema2[i - 1]

        # Calculate ratio and sum
        ema_ratio = ema1 / ema2
        for i in range(sum_period - 1, len(high)):
            mi[i] = np.sum(ema_ratio[i - sum_period + 1 : i + 1])

        return mi

    @staticmethod
    @njit
    def _calculate_vwap_numba(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        period: int,
    ) -> np.ndarray:
        """Optimized Volume Weighted Average Price calculation"""
        typical_price = (high + low + close) / 3
        vwap = np.zeros_like(close)

        for i in range(period - 1, len(close)):
            price_vol_sum = 0
            vol_sum = 0
            for j in range(i - period + 1, i + 1):
                price_vol_sum += typical_price[j] * volume[j]
                vol_sum += volume[j]
            if vol_sum != 0:
                vwap[i] = price_vol_sum / vol_sum

        return vwap

    @staticmethod
    @njit
    def _calculate_supertrend_numba(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized Supertrend calculation"""
        # Calculate ATR
        tr = np.zeros_like(close)
        atr = np.zeros_like(close)
        supertrend = np.zeros_like(close)
        direction = np.zeros_like(close)  # 1 for uptrend, -1 for downtrend

        # Calculate True Range
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        # Calculate ATR
        atr[period] = np.mean(tr[1 : period + 1])
        for i in range(period + 1, len(close)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        # Calculate Supertrend
        upperband = (high + low) / 2 + multiplier * atr
        lowerband = (high + low) / 2 - multiplier * atr

        # Initialize Supertrend
        supertrend[period] = (upperband[period] + lowerband[period]) / 2
        direction[period] = 1 if close[period] > supertrend[period] else -1

        # Calculate Supertrend values
        for i in range(period + 1, len(close)):
            if close[i - 1] <= supertrend[i - 1]:
                supertrend[i] = max(upperband[i], supertrend[i - 1])
            else:
                supertrend[i] = min(lowerband[i], supertrend[i - 1])

            direction[i] = 1 if close[i] > supertrend[i] else -1

        return supertrend, direction

    @staticmethod
    @njit
    def _calculate_pvo_numba(
        volume: np.ndarray, short_period: int = 12, long_period: int = 26
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized Percentage Volume Oscillator calculation"""
        # Calculate EMAs
        short_ema = np.zeros_like(volume)
        long_ema = np.zeros_like(volume)

        # Calculate multipliers
        short_multiplier = 2.0 / (short_period + 1)
        long_multiplier = 2.0 / (long_period + 1)

        # Initialize EMAs
        short_ema[0] = volume[0]
        long_ema[0] = volume[0]

        # Calculate EMAs
        for i in range(1, len(volume)):
            short_ema[i] = (
                volume[i] - short_ema[i - 1]
            ) * short_multiplier + short_ema[i - 1]
            long_ema[i] = (volume[i] - long_ema[i - 1]) * long_multiplier + long_ema[
                i - 1
            ]

        # Calculate PVO
        pvo = 100 * (short_ema - long_ema) / long_ema

        # Calculate signal line (9-period EMA of PVO)
        signal = np.zeros_like(pvo)
        signal_multiplier = 2.0 / (9 + 1)
        signal[0] = pvo[0]

        for i in range(1, len(pvo)):
            signal[i] = (pvo[i] - signal[i - 1]) * signal_multiplier + signal[i - 1]

        return pvo, signal

    @staticmethod
    @njit
    def _calculate_historical_volatility_numba(
        close: np.ndarray, period: int = 20, trading_days: int = 252
    ) -> np.ndarray:
        """Optimized Historical Volatility calculation"""
        returns = np.zeros(len(close) - 1)
        for i in range(1, len(close)):
            returns[i - 1] = np.log(close[i] / close[i - 1])

        hv = np.zeros(len(close))
        for i in range(period, len(close)):
            std = np.std(returns[i - period : i])
            hv[i] = std * np.sqrt(trading_days)

        return hv * 100  # Convert to percentage

    @staticmethod
    @njit
    def _calculate_chaikin_volatility_numba(
        high: np.ndarray, low: np.ndarray, ema_period: int = 10, roc_period: int = 10
    ) -> np.ndarray:
        """Optimized Chaikin Volatility calculation"""
        # Calculate High-Low difference
        hl_range = high - low

        # Calculate EMA of range
        ema = np.zeros_like(hl_range)
        multiplier = 2.0 / (ema_period + 1)

        ema[0] = hl_range[0]
        for i in range(1, len(hl_range)):
            ema[i] = (hl_range[i] - ema[i - 1]) * multiplier + ema[i - 1]

        # Calculate ROC of EMA
        cv = np.zeros_like(hl_range)
        for i in range(roc_period, len(hl_range)):
            cv[i] = ((ema[i] - ema[i - roc_period]) / ema[i - roc_period]) * 100

        return cv

    @staticmethod
    @njit
    def _calculate_linear_regression_channel_numba(
        close: np.ndarray, period: int, deviations: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optimized Linear Regression Channel calculation"""
        middle = np.zeros_like(close)
        upper = np.zeros_like(close)
        lower = np.zeros_like(close)

        for i in range(period - 1, len(close)):
            y = close[i - period + 1 : i + 1]
            x = np.arange(period)

            # Calculate means
            x_mean = np.mean(x)
            y_mean = np.mean(y)

            # Calculate slope and intercept
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)

            if denominator != 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean

                # Calculate predicted value
                predict = slope * (period - 1) + intercept
                middle[i] = predict

                # Calculate standard error
                y_pred = x * slope + intercept
                std_error = np.sqrt(np.sum((y - y_pred) ** 2) / (period - 2))

                # Calculate channels
                upper[i] = predict + deviations * std_error
                lower[i] = predict - deviations * std_error

        return upper, middle, lower

    @staticmethod
    @njit
    def _calculate_ad_numba(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
    ) -> np.ndarray:
        """Optimized Accumulation/Distribution Line calculation"""
        ad = np.zeros_like(close)

        for i in range(len(close)):
            if high[i] != low[i]:
                clv = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            else:
                clv = 0

            money_flow_vol = clv * volume[i]
            ad[i] = money_flow_vol if i == 0 else ad[i - 1] + money_flow_vol

        return ad

    @staticmethod
    @njit
    def _calculate_alma_numba(
        close: np.ndarray, window: int, sigma: float = 6.0, offset: float = 0.85
    ) -> np.ndarray:
        """Optimized Arnaud Legoux Moving Average calculation"""
        alma = np.zeros_like(close)
        m = offset * (window - 1)
        s = window / sigma

        weights = np.zeros(window)
        for i in range(window):
            weights[i] = np.exp(-((i - m) ** 2) / (2 * s * s))

        weights = weights / np.sum(weights)

        for i in range(window - 1, len(close)):
            weighted_sum = 0
            for j in range(window):
                weighted_sum += close[i - window + 1 + j] * weights[j]
            alma[i] = weighted_sum

        return alma

    @staticmethod
    @njit
    def _calculate_kdj_numba(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int = 9,
        d_period: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optimized KDJ indicator calculation"""
        rsv = np.zeros_like(close)
        k = np.zeros_like(close)
        d = np.zeros_like(close)
        j = np.zeros_like(close)

        for i in range(k_period - 1, len(close)):
            high_window = high[i - k_period + 1 : i + 1]
            low_window = low[i - k_period + 1 : i + 1]

            highest_high = np.max(high_window)
            lowest_low = np.min(low_window)

            if highest_high != lowest_low:
                rsv[i] = (close[i] - lowest_low) / (highest_high - lowest_low) * 100
            else:
                rsv[i] = 50

        # Initialize K
        k[k_period - 1] = rsv[k_period - 1]

        # Calculate K, D, J
        for i in range(k_period, len(close)):
            k[i] = (2 / 3) * k[i - 1] + (1 / 3) * rsv[i]

        # Calculate D (3-period SMA of K)
        for i in range(k_period + d_period - 1, len(close)):
            d[i] = np.mean(k[i - d_period + 1 : i + 1])

        # Calculate J
        j = 3 * k - 2 * d

        return k, d, j

    @staticmethod
    @njit
    def _calculate_heiken_ashi_numba(
        open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Optimized Heiken Ashi calculation"""
        ha_close = (open_ + high + low + close) / 4
        ha_open = np.zeros_like(close)
        ha_high = np.zeros_like(close)
        ha_low = np.zeros_like(close)

        # Initialize first values
        ha_open[0] = (open_[0] + close[0]) / 2

        # Calculate values
        for i in range(1, len(close)):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
            ha_high[i] = max(high[i], ha_open[i], ha_close[i])
            ha_low[i] = min(low[i], ha_open[i], ha_close[i])

        return ha_open, ha_high, ha_low, ha_close

    @staticmethod
    @njit
    def _calculate_donchian_channels_numba(
        high: np.ndarray, low: np.ndarray, period: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optimized Donchian Channels calculation"""
        upper = np.zeros_like(high)
        lower = np.zeros_like(low)
        middle = np.zeros_like(high)

        for i in range(period - 1, len(high)):
            upper[i] = np.max(high[i - period + 1 : i + 1])
            lower[i] = np.min(low[i - period + 1 : i + 1])
            middle[i] = (upper[i] + lower[i]) / 2

        return upper, middle, lower

    @staticmethod
    @njit
    def _calculate_ad_numba(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
    ) -> np.ndarray:
        """Optimized Accumulation/Distribution Line calculation"""
        ad = np.zeros_like(close)

        for i in range(len(close)):
            if high[i] != low[i]:
                clv = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            else:
                clv = 0

            money_flow_vol = clv * volume[i]
            ad[i] = money_flow_vol if i == 0 else ad[i - 1] + money_flow_vol

        return ad

    @staticmethod
    @njit
    def _calculate_beta_numba(
        returns: np.ndarray, market_returns: np.ndarray, window: int
    ) -> np.ndarray:
        """Optimized Beta calculation"""
        beta = np.zeros_like(returns)

        for i in range(window, len(returns)):
            ret_window = returns[i - window : i]
            mkt_window = market_returns[i - window : i]

            covar = np.mean(
                (ret_window - np.mean(ret_window)) * (mkt_window - np.mean(mkt_window))
            )
            market_var = np.var(mkt_window)

            if market_var != 0:
                beta[i] = covar / market_var

        return beta

    @staticmethod
    @njit
    def _calculate_di_numba(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized Directional Indicator calculation"""
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)

        # Calculate True Range and Directional Movement
        for i in range(1, len(high)):
            high_diff = high[i] - high[i - 1]
            low_diff = low[i - 1] - low[i]

            # Calculate +DM and -DM
            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            else:
                plus_dm[i] = 0

            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff
            else:
                minus_dm[i] = 0

            # Calculate True Range
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Smooth the values
        smooth_plus_dm = np.zeros_like(plus_dm)
        smooth_minus_dm = np.zeros_like(minus_dm)
        smooth_tr = np.zeros_like(tr)

        # Initialize
        smooth_plus_dm[period] = np.sum(plus_dm[1 : period + 1])
        smooth_minus_dm[period] = np.sum(minus_dm[1 : period + 1])
        smooth_tr[period] = np.sum(tr[1 : period + 1])

        # Calculate smoothed values
        for i in range(period + 1, len(high)):
            smooth_plus_dm[i] = (
                smooth_plus_dm[i - 1] - (smooth_plus_dm[i - 1] / period) + plus_dm[i]
            )
            smooth_minus_dm[i] = (
                smooth_minus_dm[i - 1] - (smooth_minus_dm[i - 1] / period) + minus_dm[i]
            )
            smooth_tr[i] = smooth_tr[i - 1] - (smooth_tr[i - 1] / period) + tr[i]

        # Calculate +DI and -DI
        plus_di = 100 * smooth_plus_dm / smooth_tr
        minus_di = 100 * smooth_minus_dm / smooth_tr

        return plus_di, minus_di

    @staticmethod
    @njit
    def _calculate_adosc_numba(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        fast_period: int = 3,
        slow_period: int = 10,
    ) -> np.ndarray:
        """Optimized Chaikin A/D Oscillator calculation"""
        # Calculate A/D Line first
        ad = np.zeros_like(close)
        for i in range(len(close)):
            if high[i] != low[i]:
                clv = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            else:
                clv = 0
            money_flow_vol = clv * volume[i]
            ad[i] = money_flow_vol if i == 0 else ad[i - 1] + money_flow_vol

        # Calculate EMAs of A/D Line
        fast_ema = np.zeros_like(ad)
        slow_ema = np.zeros_like(ad)

        # Initialize
        fast_ema[0] = ad[0]
        slow_ema[0] = ad[0]

        # Calculate multipliers
        fast_mult = 2.0 / (fast_period + 1)
        slow_mult = 2.0 / (slow_period + 1)

        # Calculate EMAs
        for i in range(1, len(ad)):
            fast_ema[i] = (ad[i] - fast_ema[i - 1]) * fast_mult + fast_ema[i - 1]
            slow_ema[i] = (ad[i] - slow_ema[i - 1]) * slow_mult + slow_ema[i - 1]

        # Calculate Oscillator
        return fast_ema - slow_ema

    @staticmethod
    @njit
    def _calculate_chande_momentum_numba(close: np.ndarray, period: int) -> np.ndarray:
        """Optimized Chande Momentum Oscillator calculation"""
        cmo = np.zeros_like(close)

        for i in range(period, len(close)):
            gains = 0
            losses = 0

            for j in range(i - period + 1, i + 1):
                change = close[j] - close[j - 1]
                if change > 0:
                    gains += change
                else:
                    losses += abs(change)

            if (gains + losses) != 0:
                cmo[i] = 100 * (gains - losses) / (gains + losses)

        return cmo

    @staticmethod
    @njit
    def _calculate_volume_indicators_numba(
        close: np.ndarray, volume: np.ndarray, period: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optimized Volume Indicators calculation"""
        # Volume SMA
        vol_sma = np.zeros_like(volume, dtype=np.float64)
        # Volume Force Index
        force_index = np.zeros_like(volume, dtype=np.float64)
        # Volume Price Trend
        vpt = np.zeros_like(volume, dtype=np.float64)

        # Calculate indicators
        for i in range(period, len(volume)):
            # Volume SMA
            vol_sma[i] = np.mean(volume[i - period + 1 : i + 1])

            # Force Index
            if i > 0:
                force_index[i] = volume[i] * (close[i] - close[i - 1])

            # Volume Price Trend
            if i > 0:
                price_change = (close[i] - close[i - 1]) / close[i - 1]
                vpt[i] = vpt[i - 1] + volume[i] * price_change

        return vol_sma, force_index, vpt

    @staticmethod
    @njit
    def _calculate_hull_ma_numba(close: np.ndarray, period: int) -> np.ndarray:
        """Optimized Hull Moving Average calculation"""
        hull = np.zeros_like(close)
        # Calculate WMA with period/2
        half_period = period // 2
        wma1 = np.zeros_like(close)
        wma2 = np.zeros_like(close)

        # Calculate first WMA (period/2)
        for i in range(half_period - 1, len(close)):
            weights_sum = 0
            data_sum = 0
            for j in range(half_period):
                weight = half_period - j
                weights_sum += weight
                data_sum += close[i - j] * weight
            wma1[i] = data_sum / weights_sum

        # Calculate second WMA (period)
        for i in range(period - 1, len(close)):
            weights_sum = 0
            data_sum = 0
            for j in range(period):
                weight = period - j
                weights_sum += weight
                data_sum += close[i - j] * weight
            wma2[i] = data_sum / weights_sum

        # Calculate 2 * WMA(n/2) - WMA(n)
        raw = 2 * wma1 - wma2

        # Calculate final Hull MA using sqrt(n)
        sqrt_period = int(np.sqrt(period))
        for i in range(sqrt_period - 1, len(close)):
            weights_sum = 0
            data_sum = 0
            for j in range(sqrt_period):
                weight = sqrt_period - j
                weights_sum += weight
                data_sum += raw[i - j] * weight
            hull[i] = data_sum / weights_sum

        return hull

    @staticmethod
    @njit
    def _calculate_pivot_points_numba(
        high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Optimized Pivot Points calculation"""
        pp = np.zeros_like(close)
        r1 = np.zeros_like(close)
        r2 = np.zeros_like(close)
        s1 = np.zeros_like(close)
        s2 = np.zeros_like(close)

        for i in range(1, len(close)):
            pp[i] = (high[i - 1] + low[i - 1] + close[i - 1]) / 3
            r1[i] = 2 * pp[i] - low[i - 1]
            r2[i] = pp[i] + (high[i - 1] - low[i - 1])
            s1[i] = 2 * pp[i] - high[i - 1]
            s2[i] = pp[i] - (high[i - 1] - low[i - 1])

        return pp, r1, r2, s1, s2

    @staticmethod
    @njit
    def _calculate_elder_ray_numba(
        close: np.ndarray, period: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized Elder Ray Index calculation"""
        bull_power = np.zeros_like(close)
        bear_power = np.zeros_like(close)
        ema = np.zeros_like(close)

        # Calculate EMA
        multiplier = 2.0 / (period + 1)
        ema[0] = close[0]
        for i in range(1, len(close)):
            ema[i] = (close[i] - ema[i - 1]) * multiplier + ema[i - 1]

        # Calculate Bull and Bear Power
        for i in range(len(close)):
            bull_power[i] = high[i] - ema[i]
            bear_power[i] = low[i] - ema[i]

        return bull_power, bear_power

    @staticmethod
    @njit
    def _calculate_choppiness_index_numba(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        """Optimized Choppiness Index calculation"""
        ci = np.zeros_like(close)
        atr_sum = np.zeros_like(close)
        tr = np.zeros_like(close)

        # Calculate True Range
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        # Calculate ATR sum
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(tr[i - period + 1 : i + 1])

        # Calculate highest high and lowest low
        for i in range(period, len(close)):
            highest_high = np.max(high[i - period + 1 : i + 1])
            lowest_low = np.min(low[i - period + 1 : i + 1])

            if (highest_high - lowest_low) != 0:
                ci[i] = (
                    100
                    * np.log10(atr_sum[i] / (highest_high - lowest_low))
                    / np.log10(period)
                )

        return ci

    @staticmethod
    @njit
    def _calculate_disparity_index_numba(close: np.ndarray, period: int) -> np.ndarray:
        """Optimized Disparity Index calculation"""
        disparity = np.zeros_like(close)
        sma = np.zeros_like(close)

        # Calculate SMA
        for i in range(period - 1, len(close)):
            sma[i] = np.mean(close[i - period + 1 : i + 1])
            if sma[i] != 0:
                disparity[i] = (close[i] / sma[i] - 1) * 100

        return disparity

    @staticmethod
    @njit
    def _calculate_coppock_curve_numba(
        close: np.ndarray,
        roc1_period: int = 14,
        roc2_period: int = 11,
        wma_period: int = 10,
    ) -> np.ndarray:
        """Optimized Coppock Curve calculation"""
        roc1 = np.zeros_like(close)
        roc2 = np.zeros_like(close)
        coppock = np.zeros_like(close)

        # Calculate ROCs
        for i in range(max(roc1_period, roc2_period), len(close)):
            if close[i - roc1_period] != 0:
                roc1[i] = (
                    (close[i] - close[i - roc1_period]) / close[i - roc1_period] * 100
                )
            if close[i - roc2_period] != 0:
                roc2[i] = (
                    (close[i] - close[i - roc2_period]) / close[i - roc2_period] * 100
                )

        # Calculate Coppock using WMA of ROC sum
        roc_sum = roc1 + roc2
        for i in range(wma_period - 1, len(close)):
            weights_sum = 0
            data_sum = 0
            for j in range(wma_period):
                weight = wma_period - j
                weights_sum += weight
                data_sum += roc_sum[i - j] * weight
            coppock[i] = data_sum / weights_sum

        return coppock

    @staticmethod
    @njit
    def _calculate_rvi_numba(
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int,
    ) -> np.ndarray:
        """Optimized Relative Vigor Index calculation"""
        rvi = np.zeros_like(close)

        # Calculate numerator and denominator components
        num = np.zeros_like(close)
        den = np.zeros_like(close)

        for i in range(3, len(close)):
            # Numerator = (Close - Open) + 2*(Previous Close - Previous Open) + 2*(2nd Previous...) + (3rd Previous...)
            num[i] = (
                (close[i] - open_[i])
                + 2 * (close[i - 1] - open_[i - 1])
                + 2 * (close[i - 2] - open_[i - 2])
                + (close[i - 3] - open_[i - 3])
            ) / 6

            # Denominator = (High - Low) + 2*(Previous High - Previous Low) + 2*(2nd Previous...) + (3rd Previous...)
            den[i] = (
                (high[i] - low[i])
                + 2 * (high[i - 1] - low[i - 1])
                + 2 * (high[i - 2] - low[i - 2])
                + (high[i - 3] - low[i - 3])
            ) / 6

        # Calculate RVI
        for i in range(period + 3, len(close)):
            num_sum = 0.0
            den_sum = 0.0
            for j in range(period):
                num_sum += num[i - j]
                den_sum += den[i - j]

            if den_sum != 0:
                rvi[i] = num_sum / den_sum

        return rvi

    @staticmethod
    @njit
    def _calculate_pgo_numba(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 21
    ) -> np.ndarray:
        """Optimized Pretty Good Oscillator calculation"""
        pgo = np.zeros_like(close)

        for i in range(period, len(close)):
            highest_high = np.max(high[i - period + 1 : i + 1])
            lowest_low = np.min(low[i - period + 1 : i + 1])

            if highest_high - lowest_low != 0:
                pgo[i] = (
                    100 * (close[i] - lowest_low) / (highest_high - lowest_low) - 50
                )

        return pgo

    @staticmethod
    @njit
    def _calculate_psl_numba(close: np.ndarray, period: int = 12) -> np.ndarray:
        """Optimized Psychological Line calculation"""
        psl = np.zeros_like(close)

        for i in range(period, len(close)):
            up_count = 0
            for j in range(i - period + 1, i + 1):
                if close[j] > close[j - 1]:
                    up_count += 1
            psl[i] = (up_count / period) * 100

        return psl

    @staticmethod
    @njit
    def _calculate_rainbow_numba(
        close: np.ndarray, periods: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """Optimized Rainbow Oscillator calculation"""
        n_periods = len(periods)
        results = tuple(np.zeros_like(close) for _ in range(n_periods))

        for p_idx, period in enumerate(periods):
            # Calculate SMA for each period
            sma = np.zeros_like(close)
            for i in range(period - 1, len(close)):
                sma[i] = np.mean(close[i - period + 1 : i + 1])
            results[p_idx][:] = sma

        return results

    @staticmethod
    @njit
    def _calculate_momentum_index_numba(
        close: np.ndarray, period: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized Momentum Index calculation"""
        positive_sum = np.zeros_like(close)
        negative_sum = np.zeros_like(close)
        momentum_index = np.zeros_like(close)

        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                positive_sum[i] = positive_sum[i - 1] + (close[i] - close[i - 1])
                negative_sum[i] = negative_sum[i - 1]
            else:
                negative_sum[i] = negative_sum[i - 1] + (close[i - 1] - close[i])
                positive_sum[i] = positive_sum[i - 1]

        # Calculate Momentum Index
        for i in range(period, len(close)):
            pos_period = positive_sum[i] - positive_sum[i - period]
            neg_period = negative_sum[i] - negative_sum[i - period]
            total = pos_period + neg_period

            if total > 0:
                momentum_index[i] = 100 * (pos_period / total)

        return momentum_index, 100 - momentum_index

    @staticmethod
    @njit
    def _calculate_ma_momentum_numba(
        close: np.ndarray, ma_period: int = 10, momentum_period: int = 10
    ) -> np.ndarray:
        """Optimized Moving Average Momentum calculation"""
        # Calculate MA
        ma = np.zeros_like(close)
        for i in range(ma_period - 1, len(close)):
            ma[i] = np.mean(close[i - ma_period + 1 : i + 1])

        # Calculate momentum of MA
        momentum = np.zeros_like(close)
        for i in range(ma_period + momentum_period, len(close)):
            momentum[i] = (
                100 * (ma[i] - ma[i - momentum_period]) / ma[i - momentum_period]
            )

        return momentum

    @staticmethod
    @njit
    def _calculate_qstick_numba(
        close: np.ndarray, open_: np.ndarray, period: int = 10
    ) -> np.ndarray:
        """Optimized QStick calculation"""
        qstick = np.zeros_like(close)

        # Calculate difference between close and open
        diff = close - open_

        # Calculate moving average of differences
        for i in range(period - 1, len(close)):
            qstick[i] = np.mean(diff[i - period + 1 : i + 1])

        return qstick

    @staticmethod
    @njit
    def _calculate_typical_price_numba(
        high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """Optimized Typical Price calculation"""
        return (high + low + close) / 3

    @staticmethod
    @njit
    def _calculate_weighted_close_numba(
        high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """Optimized Weighted Close calculation"""
        return (high + low + close + close) / 4

    @staticmethod
    @njit
    def _calculate_fractal_numba(
        high: np.ndarray, low: np.ndarray, period: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized Fractal Indicator calculation"""
        bullish = np.zeros_like(high)
        bearish = np.zeros_like(low)
        half = period // 2

        for i in range(half, len(high) - half):
            # Bullish Fractal
            is_bullish = True
            peak = high[i]
            for j in range(1, half + 1):
                if high[i - j] >= peak or high[i + j] >= peak:
                    is_bullish = False
                    break
            if is_bullish:
                bullish[i] = 1

            # Bearish Fractal
            is_bearish = True
            trough = low[i]
            for j in range(1, half + 1):
                if low[i - j] <= trough or low[i + j] <= trough:
                    is_bearish = False
                    break
            if is_bearish:
                bearish[i] = 1

        return bullish, bearish

    # Indicator calculations (public) ---------------------------------------------------

    def SMA(self, data: Union[pd.Series, pd.DataFrame], period: int = 20) -> pd.Series:
        """Calculate Simple Moving Average"""
        prices = self._validate_and_get_prices(data)
        prices_np = np.ascontiguousarray(prices.values, dtype=np.float64)
        sma = self._calculate_sma_numba(prices_np, period)
        return pd.Series(sma, index=prices.index, name=f"SMA_{period}")

    def EMA(self, data: Union[pd.Series, pd.DataFrame], period: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average"""
        prices = self._validate_and_get_prices(data)
        prices_np = np.ascontiguousarray(prices.values, dtype=np.float64)
        ema = self._calculate_ema_numba(prices_np, period)
        return pd.Series(ema, index=prices.index, name=f"EMA_{period}")

    def RSI(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        prices = self._validate_and_get_prices(data)
        prices_np = np.ascontiguousarray(prices.values, dtype=np.float64)
        rsi = self._calculate_rsi_numba(prices_np, period)
        return pd.Series(rsi, index=data.index, name=f"RSI_{period}")

    def MACD(
        self,
        data: Union[pd.Series, pd.DataFrame],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> pd.DataFrame:
        """Calculate MACD, Signal Line, and Histogram"""
        prices = self._validate_and_get_prices(data)
        prices_np = np.ascontiguousarray(prices.values, dtype=np.float64)
        macd_line, signal_line, histogram = self._calculate_macd_numba(
            prices_np, fast_period, slow_period, signal_period
        )

        # Convert to pandas Series/DataFrame
        macd_series = pd.Series(macd_line, index=prices.index, name="MACD")
        signal_series = pd.Series(signal_line, index=prices.index, name="Signal")
        histogram_series = pd.Series(histogram, index=prices.index, name="Histogram")

        return pd.DataFrame(
            {
                "MACD": macd_series,
                "Signal": signal_series,
                "Histogram": histogram_series,
            }
        )

    def BB(
        self,
        data: Union[pd.Series, pd.DataFrame],
        period: int = 20,
        num_std: float = 2.0,
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        prices = self._validate_and_get_prices(data)
        prices_np = np.ascontiguousarray(prices.values, dtype=np.float64)
        upper, middle, lower = self._calculate_bollinger_bands_numba(
            prices_np, period, num_std
        )
        # Convert to pandas Series/DataFrame
        upper_series = pd.Series(upper, index=prices.index, name="BB_Upper")
        middle_series = pd.Series(middle, index=prices.index, name="BB_Middle")
        lower_series = pd.Series(lower, index=prices.index, name="BB_Lower")

        return pd.DataFrame(
            {
                "BB_Upper": upper_series,
                "BB_Middle": middle_series,
                "BB_Lower": lower_series,
            }
        )

    def ATR(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        self._validate_data(data, ["high", "low", "close"])
        high_np = np.ascontiguousarray(data["high"].values, dtype=np.float64)
        low_np = np.ascontiguousarray(data["low"].values, dtype=np.float64)
        close_np = np.ascontiguousarray(data["close"].values, dtype=np.float64)
        atr = self._calculate_atr_numba(high_np, low_np, close_np, period)
        return pd.Series(atr, index=data.index, name=f"ATR_{period}")

    def STOCH(
        self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        self._validate_data(data, ["high", "low", "close"])
        k_line, d_line = self._calculate_stochastic_numba(
            data["high"].values,
            data["low"].values,
            data["close"].values,
            k_period,
            d_period,
        )
        # Set the first `k_period-1` values to NaN
        k_line[: k_period - 1] = np.nan
        d_line[: k_period + d_period - 2] = np.nan
        return pd.DataFrame({"K": k_line, "D": d_line}, index=data.index)

    def ADX(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average Directional Index"""
        self._validate_data(data, ["high", "low", "close"])
        adx, plus_di, minus_di = self._calculate_adx_numba(
            data["high"].values, data["low"].values, data["close"].values, period
        )
        # Set the first `2 * period - 1` values to NaN for ADX
        adx[: 2 * period - 1] = np.nan
        # Set the first `period` values to NaN for +DI and -DI
        plus_di[:period] = np.nan
        minus_di[:period] = np.nan
        return pd.DataFrame(
            {"ADX": adx, "+DI": plus_di, "-DI": minus_di}, index=data.index
        )

    def ICHIMOKU(
        self,
        data: pd.DataFrame,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_span_b_period: int = 52,
        displacement: int = 26,
    ) -> pd.DataFrame:
        """Calculate Ichimoku Cloud components"""
        self._validate_data(data, ["high", "low", "close"])

        tenkan, kijun, senkou_a, senkou_b = self._calculate_ichimoku_numba(
            data["high"].values,
            data["low"].values,
            data["close"].values,
            tenkan_period,
            kijun_period,
            senkou_span_b_period,
        )

        # Set NaNs for initial periods
        tenkan[: tenkan_period - 1] = np.nan
        kijun[: kijun_period - 1] = np.nan
        senkou_a[: kijun_period - 1] = np.nan
        senkou_b[: senkou_span_b_period - 1] = np.nan

        # Shift Senkou Spans forward
        senkou_a = pd.Series(senkou_a, index=data.index).shift(displacement)
        senkou_b = pd.Series(senkou_b, index=data.index).shift(displacement)

        # Shift Chikou Span backward and set NaNs
        chikou_span = data["close"].shift(-displacement)
        chikou_span.iloc[-displacement:] = np.nan  # Set trailing NaNs

        # Create DataFrame with results
        result = pd.DataFrame(
            {
                "Tenkan-sen": tenkan,
                "Kijun-sen": kijun,
                "Senkou Span A": senkou_a,
                "Senkou Span B": senkou_b,
                "Chikou Span": chikou_span,
            },
            index=data.index,
        )

        return result

    def KELTNER(
        self,
        data: pd.DataFrame,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
    ) -> pd.DataFrame:
        """Calculate Keltner Channels"""
        self._validate_data(data, ["high", "low", "close"])

        upper, middle, lower = self._calculate_keltner_channels_numba(
            data["high"].values,
            data["low"].values,
            data["close"].values,
            ema_period,
            atr_period,
            multiplier,
        )
        # Set the first `ema_period-1` and `atr_period-1` values to NaN
        upper[: ema_period - 1] = np.nan
        middle[: ema_period - 1] = np.nan
        lower[: ema_period - 1] = np.nan
        return pd.DataFrame(
            {"KC_Upper": upper, "KC_Middle": middle, "KC_Lower": lower},
            index=data.index,
        )

    def MFI(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        self._validate_data(data, ["high", "low", "close", "volume"])

        mfi = self._calculate_mfi_numba(
            data["high"].values,
            data["low"].values,
            data["close"].values,
            data["volume"].values,
            period,
        )
        # Set the first `period-1` values to NaN
        mfi[: period - 1] = np.nan
        return pd.Series(mfi, index=data.index, name=f"MFI_{period}")

    def TRIX(self, data: Union[pd.Series, pd.DataFrame], period: int = 15) -> pd.Series:
        """Calculate TRIX indicator"""
        prices = self._validate_and_get_prices(data)

        trix = self._calculate_trix_numba(prices.values, period)
        # Set the first `period-1` values to NaN
        trix[: period - 1] = np.nan
        return pd.Series(trix, index=prices.index, name=f"TRIX_{period}")

    def CCI(
        self, data: pd.DataFrame, period: int = 20, constant: float = 0.015
    ) -> pd.Series:
        """Calculate Commodity Channel Index"""
        self._validate_data(data, ["high", "low", "close"])

        high_np = np.ascontiguousarray(data["high"].values, dtype=np.float64)
        low_np = np.ascontiguousarray(data["low"].values, dtype=np.float64)
        close_np = np.ascontiguousarray(data["close"].values, dtype=np.float64)

        cci = self._calculate_cci_numba(
            high_np,
            low_np,
            close_np,
            period,
            constant,
        )
        # Set the first `period-1` values to NaN
        cci[: period - 1] = np.nan
        return pd.Series(cci, index=data.index, name=f"CCI_{period}")

    def ROC(self, data: Union[pd.Series, pd.DataFrame], period: int = 12) -> pd.Series:
        """Calculate Rate of Change"""
        prices = self._validate_and_get_prices(data)

        roc = self._calculate_roc_numba(prices.values, period)
        # Set the first `period` values to NaN (since ROC starts at index `period`)
        roc[:period] = np.nan
        return pd.Series(roc, index=prices.index, name=f"ROC_{period}")

    def WILLR(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        self._validate_data(data, ["high", "low", "close"])

        high_np = np.ascontiguousarray(data["high"].values, dtype=np.float64)
        low_np = np.ascontiguousarray(data["low"].values, dtype=np.float64)
        close_np = np.ascontiguousarray(data["close"].values, dtype=np.float64)

        willr = self._calculate_willr_numba(high_np, low_np, close_np, period)
        # Set the first `period-1` values to NaN
        willr[: period - 1] = np.nan
        return pd.Series(willr, index=data.index, name=f"WILLR_{period}")

    def DEMA(self, data: Union[pd.Series, pd.DataFrame], period: int = 20) -> pd.Series:
        """Calculate Double Exponential Moving Average"""
        prices = self._validate_and_get_prices(data)

        dema = self._calculate_dema_numba(prices.values, period)
        # Set the first `period-1` values to NaN
        dema[: period - 1] = np.nan
        return pd.Series(dema, index=prices.index, name=f"DEMA_{period}")

    def KAMA(
        self,
        data: Union[pd.Series, pd.DataFrame],
        er_period: int = 10,
        fast_period: int = 2,
        slow_period: int = 30,
    ) -> pd.Series:
        """Calculate Kaufman Adaptive Moving Average"""
        prices = self._validate_and_get_prices(data)

        kama = self._calculate_kama_numba(
            prices.values, er_period, fast_period, slow_period
        )
        # Set the first `er_period-1` values to NaN
        kama[: er_period - 1] = np.nan
        return pd.Series(kama, index=prices.index, name=f"KAMA_{er_period}")

    def DONCHIAN(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Donchian Channels"""
        self._validate_data(data, ["high", "low"])

        upper, middle, lower = self._calculate_donchian_channels_numba(
            data["high"].values, data["low"].values, period
        )
        # Set the first `period - 1` values to NaN
        upper[: period - 1] = np.nan
        middle[: period - 1] = np.nan
        lower[: period - 1] = np.nan

        return pd.DataFrame(
            {"DC_Upper": upper, "DC_Middle": middle, "DC_Lower": lower},
            index=data.index,
        )

    def AROON(self, data: pd.DataFrame, period: int = 25) -> pd.DataFrame:
        """Calculate Aroon Indicator"""
        self._validate_data(data, ["high", "low"])

        aroon_up, aroon_down = self._calculate_aroon_numba(
            data["high"].values, data["low"].values, period
        )
        # Set the first `period - 1` values to NaN
        aroon_up[: period - 1] = np.nan
        aroon_down[: period - 1] = np.nan
        aroon_osc = aroon_up - aroon_down

        return pd.DataFrame(
            {"AROON_UP": aroon_up, "AROON_DOWN": aroon_down, "AROON_OSC": aroon_osc},
            index=data.index,
        )

    def AO(
        self, data: pd.DataFrame, short_period: int = 5, long_period: int = 34
    ) -> pd.Series:
        """Calculate Awesome Oscillator"""
        self._validate_data(data, ["high", "low"])

        ao = self._calculate_awesome_oscillator_numba(
            data["high"].values, data["low"].values, short_period, long_period
        )
        # Set the first `long_period - 1` values to NaN
        ao[: long_period - 1] = np.nan

        return pd.Series(ao, index=data.index, name="AO")

    def ULTIMATE_OSCILLATOR(
        self,
        data: pd.DataFrame,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28,
        weight1: float = 4.0,
        weight2: float = 2.0,
        weight3: float = 1.0,
    ) -> pd.Series:
        """Calculate Ultimate Oscillator"""
        self._validate_data(data, ["high", "low", "close"])

        uo = self._calculate_ultimate_oscillator_numba(
            data["high"].values,
            data["low"].values,
            data["close"].values,
            period1,
            period2,
            period3,
            weight1,
            weight2,
            weight3,
        )
        # Set the first `max(period1, period2, period3)-1` values to NaN
        uo[: max(period1, period2, period3) - 1] = np.nan
        return pd.Series(uo, index=data.index, name="UO")

    def CMO(self, data: Union[pd.Series, pd.DataFrame], period: int = 14) -> pd.Series:
        """Calculate Chande Momentum Oscillator"""
        prices = self._validate_and_get_prices(data)

        cmo = self._calculate_chande_momentum_numba(prices.values, period)
        cmo[: period - 1] = np.nan
        return pd.Series(cmo, index=prices.index, name=f"CMO_{period}")

    def DPO(self, data: Union[pd.Series, pd.DataFrame], period: int = 20) -> pd.Series:
        """Calculate Detrended Price Oscillator"""
        prices = self._validate_and_get_prices(data)

        dpo = self._calculate_dpo_numba(prices.values, period)
        # Set the first `period-1` values to NaN
        dpo[: period - 1] = np.nan
        return pd.Series(dpo, index=prices.index, name=f"DPO_{period}")

    def MASS_INDEX(
        self, data: pd.DataFrame, ema_period: int = 9, sum_period: int = 25
    ) -> pd.Series:
        """Calculate Mass Index"""
        self._validate_data(data, ["high", "low"])

        mi = self._calculate_mass_index_numba(
            data["high"].values, data["low"].values, ema_period, sum_period
        )
        # Set the first `sum_period - 1` values to NaN
        mi[: sum_period - 1] = np.nan

        return pd.Series(mi, index=data.index, name=f"MI_{ema_period}_{sum_period}")

    def VWAP(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        self._validate_data(data, ["high", "low", "close", "volume"])

        vwap = self._calculate_vwap_numba(
            data["high"].values,
            data["low"].values,
            data["close"].values,
            data["volume"].values,
            period,
        )
        # Set the first `period - 1` values to NaN
        vwap[: period - 1] = np.nan
        return pd.Series(vwap, index=data.index, name=f"VWAP_{period}")

    def SUPERTREND(
        self, data: pd.DataFrame, period: int = 10, multiplier: float = 3.0
    ) -> pd.DataFrame:
        """Calculate Supertrend indicator"""
        self._validate_data(data, ["high", "low", "close"])

        supertrend, direction = self._calculate_supertrend_numba(
            data["high"].values,
            data["low"].values,
            data["close"].values,
            period,
            multiplier,
        )
        # Set NaN values for the first `period - 1`
        supertrend[: period - 1] = np.nan
        direction[: period - 1] = np.nan
        return pd.DataFrame(
            {"Supertrend": supertrend, "Direction": direction}, index=data.index
        )

    def PVO(
        self, data: pd.DataFrame, short_period: int = 12, long_period: int = 26
    ) -> pd.DataFrame:
        """Calculate Percentage Volume Oscillator"""
        self._validate_data(data, ["volume"])

        pvo, signal = self._calculate_pvo_numba(
            data["volume"].values, short_period, long_period
        )

        # Set initial NaN values
        pvo[: long_period - 1] = np.nan
        signal[: long_period - 1] = np.nan
        histogram = pvo - signal

        return pd.DataFrame(
            {"PVO": pvo, "Signal": signal, "Histogram": histogram}, index=data.index
        )

    def HISTORICAL_VOLATILITY(
        self,
        data: Union[pd.Series, pd.DataFrame],
        period: int = 20,
        trading_days: int = 252,
    ) -> pd.Series:
        """Calculate Historical Volatility"""
        prices = self._validate_and_get_prices(data)

        hv = self._calculate_historical_volatility_numba(
            prices.values, period, trading_days
        )
        # Set the first `period` values to NaN
        hv[:period] = np.nan
        return pd.Series(hv, index=prices.index, name=f"HV_{period}")

    def CHAIKIN_VOLATILITY(
        self, data: pd.DataFrame, ema_period: int = 10, roc_period: int = 10
    ) -> pd.Series:
        """Calculate Chaikin Volatility"""
        self._validate_data(data, ["high", "low"])

        cv = self._calculate_chaikin_volatility_numba(
            data["high"].values, data["low"].values, ema_period, roc_period
        )
        # Set the first `ema_period + roc_period - 2` values to NaN
        cv[: ema_period + roc_period - 2] = np.nan
        return pd.Series(cv, index=data.index, name=f"CV_{ema_period}_{roc_period}")

    def LINEAR_REGRESSION_CHANNEL(
        self,
        data: Union[pd.Series, pd.DataFrame],
        period: int = 20,
        deviations: float = 2.0,
    ) -> pd.DataFrame:
        """Calculate Linear Regression Channel"""
        prices = self._validate_and_get_prices(data)

        upper, middle, lower = self._calculate_linear_regression_channel_numba(
            prices.values, period, deviations
        )
        # Set NaNs for the first `period - 1` values
        upper[: period - 1] = np.nan
        middle[: period - 1] = np.nan
        lower[: period - 1] = np.nan
        return pd.DataFrame(
            {"LRC_Upper": upper, "LRC_Middle": middle, "LRC_Lower": lower},
            index=prices.index,
        )

    def AD(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        self._validate_data(data, ["high", "low", "close", "volume"])

        ad = self._calculate_ad_numba(
            data["high"].values,
            data["low"].values,
            data["close"].values,
            data["volume"].values,
        )

        return pd.Series(ad, index=data.index, name="AD")

    def ALMA(
        self,
        data: Union[pd.Series, pd.DataFrame],
        period: int = 9,
        sigma: float = 6.0,
        offset: float = 0.85,
    ) -> pd.Series:
        """Calculate Arnaud Legoux Moving Average"""
        prices = self._validate_and_get_prices(data)

        alma = self._calculate_alma_numba(prices.values, period, sigma, offset)
        # Set the first `period - 1` values to NaN
        alma[: period - 1] = np.nan
        return pd.Series(alma, index=prices.index, name=f"ALMA_{period}")

    def KDJ(
        self, data: pd.DataFrame, k_period: int = 9, d_period: int = 3
    ) -> pd.DataFrame:
        """Calculate KDJ indicator"""
        self._validate_data(data, ["high", "low", "close"])

        k, d, j = self._calculate_kdj_numba(
            data["high"].values,
            data["low"].values,
            data["close"].values,
            k_period,
            d_period,
        )
        # Set NaN values for the first `k_period - 1`
        k[: k_period - 1] = np.nan
        d[: k_period + d_period - 2] = np.nan
        j[: k_period + d_period - 2] = np.nan
        return pd.DataFrame({"K": k, "D": d, "J": j}, index=data.index)

    def HEIKEN_ASHI(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Heiken Ashi candles"""
        self._validate_data(data, ["open", "high", "low", "close"])

        ha_open, ha_high, ha_low, ha_close = self._calculate_heiken_ashi_numba(
            data["open"].values,
            data["high"].values,
            data["low"].values,
            data["close"].values,
        )

        return pd.DataFrame(
            {
                "HA_Open": ha_open,
                "HA_High": ha_high,
                "HA_Low": ha_low,
                "HA_Close": ha_close,
            },
            index=data.index,
        )

    def BENFORD_LAW(self, data: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """Calculate Benford's Law distribution"""
        prices = self._validate_and_get_prices(data)

        def first_digit(x):
            return int(str(abs(x)).strip("0.")[0])

        first_digits = prices.apply(first_digit)
        observed_freq = first_digits.value_counts(normalize=True).sort_index()

        # Expected Benford's Law frequencies
        expected_freq = pd.Series(
            [np.log10(1 + 1 / d) for d in range(1, 10)], index=range(1, 10)
        )

        return pd.DataFrame({"Observed": observed_freq, "Expected": expected_freq})

    def OBV(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        self._validate_data(data, ["close", "volume"])

        obv = np.zeros(len(data))
        for i in range(1, len(data)):
            if data["close"].iloc[i] > data["close"].iloc[i - 1]:
                obv[i] = obv[i - 1] + data["volume"].iloc[i]
            elif data["close"].iloc[i] < data["close"].iloc[i - 1]:
                obv[i] = obv[i - 1] - data["volume"].iloc[i]
            else:
                obv[i] = obv[i - 1]

        return pd.Series(obv, index=data.index, name="OBV")

    def BETA(
        self, data: pd.Series, market_data: pd.Series, period: int = 252
    ) -> pd.Series:
        """Calculate Rolling Beta"""
        # Calculate returns if prices are provided
        if data.pct_change().std() < 0.1:  # Assume it's price data if volatility is low
            returns = data.pct_change().fillna(0)
            market_returns = market_data.pct_change().fillna(0)
        else:
            returns = data
            market_returns = market_data

        beta = self._calculate_beta_numba(returns.values, market_returns.values, period)

        # Set the first `period - 1` values to NaN
        beta[:period] = np.nan

        return pd.Series(beta, index=data.index, name=f"BETA_{period}")

    def DI(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Directional Indicator (+DI and -DI)"""
        self._validate_data(data, ["high", "low", "close"])

        plus_di, minus_di = self._calculate_di_numba(
            data["high"].values, data["low"].values, data["close"].values, period
        )

        return pd.DataFrame({"+DI": plus_di, "-DI": minus_di}, index=data.index)

    def ADOSC(
        self, data: pd.DataFrame, fast_period: int = 3, slow_period: int = 10
    ) -> pd.Series:
        """Calculate Chaikin A/D Oscillator"""
        self._validate_data(data, ["high", "low", "close", "volume"])

        adosc = self._calculate_adosc_numba(
            data["high"].values,
            data["low"].values,
            data["close"].values,
            data["volume"].values,
            fast_period,
            slow_period,
        )
        # Set the first `slow_period - 1` values to NaN
        adosc[: slow_period - 1] = np.nan
        return pd.Series(
            adosc, index=data.index, name=f"ADOSC_{fast_period}_{slow_period}"
        )

    def VOLUME_INDICATORS(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate various volume-based indicators"""
        self._validate_data(data, ["close", "volume"])

        vol_sma, force_index, vpt = self._calculate_volume_indicators_numba(
            data["close"].values, data["volume"].values, period
        )

        # Set initial NaN values
        vol_sma[: period - 1] = np.nan
        force_index[0] = np.nan  # Set the first value to NaN
        # VPT can start from zero or NaN; we'll keep it as zero

        return pd.DataFrame(
            {"Volume_SMA": vol_sma, "Force_Index": force_index, "VPT": vpt},
            index=data.index,
        )

    def HULL_MA(
        self, data: Union[pd.Series, pd.DataFrame], period: int = 10
    ) -> pd.Series:
        """Calculate Hull Moving Average"""
        prices = self._validate_and_get_prices(data)
        hull = self._calculate_hull_ma_numba(prices.values, period)
        return pd.Series(hull, index=prices.index, name=f"HULL_{period}")

    def PIVOT_POINTS(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Classic Pivot Points"""
        self._validate_data(data, ["high", "low", "close"])
        pp, r1, r2, s1, s2 = self._calculate_pivot_points_numba(
            data["high"].values, data["low"].values, data["close"].values
        )
        return pd.DataFrame(
            {"PP": pp, "R1": r1, "R2": r2, "S1": s1, "S2": s2}, index=data.index
        )

    def ELDER_RAY(self, data: pd.DataFrame, period: int = 13) -> pd.DataFrame:
        """Calculate Elder Ray Index"""
        self._validate_data(data, ["high", "low", "close"])
        bull_power, bear_power = self._calculate_elder_ray_numba(
            data["close"].values, period
        )
        return pd.DataFrame(
            {"Bull_Power": bull_power, "Bear_Power": bear_power}, index=data.index
        )

    def CHOPPINESS(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Choppiness Index"""
        self._validate_data(data, ["high", "low", "close"])
        ci = self._calculate_choppiness_index_numba(
            data["high"].values, data["low"].values, data["close"].values, period
        )
        return pd.Series(ci, index=data.index, name=f"CI_{period}")

    def DISPARITY(
        self, data: Union[pd.Series, pd.DataFrame], period: int = 14
    ) -> pd.Series:
        """Calculate Disparity Index"""
        prices = self._validate_and_get_prices(data)
        disparity = self._calculate_disparity_index_numba(prices.values, period)
        return pd.Series(disparity, index=prices.index, name=f"DI_{period}")

    def COPPOCK(
        self,
        data: Union[pd.Series, pd.DataFrame],
        roc1_period: int = 14,
        roc2_period: int = 11,
        wma_period: int = 10,
    ) -> pd.Series:
        """Calculate Coppock Curve"""
        prices = self._validate_and_get_prices(data)
        coppock = self._calculate_coppock_curve_numba(
            prices.values, roc1_period, roc2_period, wma_period
        )
        return pd.Series(coppock, index=prices.index, name="COPPOCK")

    def RVI(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate Relative Vigor Index"""
        self._validate_data(data, ["open", "high", "low", "close"])

        rvi = self._calculate_rvi_numba(
            data["open"].values,
            data["high"].values,
            data["low"].values,
            data["close"].values,
            period,
        )
        return pd.Series(rvi, index=data.index, name=f"RVI_{period}")

    def PGO(self, data: pd.DataFrame, period: int = 21) -> pd.Series:
        """Calculate Pretty Good Oscillator"""
        self._validate_data(data, ["high", "low", "close"])

        pgo = self._calculate_pgo_numba(
            data["high"].values, data["low"].values, data["close"].values, period
        )
        return pd.Series(pgo, index=data.index, name=f"PGO_{period}")

    def PSL(self, data: pd.DataFrame, period: int = 12) -> pd.Series:
        """Calculate Psychological Line"""
        self._validate_data(data, ["close"])

        psl = self._calculate_psl_numba(data["close"].values, period)
        return pd.Series(psl, index=data.index, name=f"PSL_{period}")

    def RAINBOW(
        self,
        data: Union[pd.Series, pd.DataFrame],
        periods: List[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10],
    ) -> pd.DataFrame:
        """Calculate Rainbow Oscillator"""
        prices = self._validate_and_get_prices(data)

        sma_lines = self._calculate_rainbow_numba(
            prices.values, np.array(periods, dtype=np.int32)
        )

        return pd.DataFrame(
            {f"SMA_{period}": sma for period, sma in zip(periods, sma_lines)},
            index=prices.index,
        )

    def MOMENTUM_INDEX(self, data: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """Calculate Momentum Index"""
        self._validate_data(data, ["close"])

        pos_idx, neg_idx = self._calculate_momentum_index_numba(
            data["close"].values, period
        )
        return pd.DataFrame(
            {"Positive_Index": pos_idx, "Negative_Index": neg_idx}, index=data.index
        )

    def MA_MOMENTUM(
        self,
        data: Union[pd.Series, pd.DataFrame],
        ma_period: int = 10,
        momentum_period: int = 10,
    ) -> pd.Series:
        """Calculate Moving Average Momentum"""
        prices = self._validate_and_get_prices(data)

        ma_momentum = self._calculate_ma_momentum_numba(
            prices.values, ma_period, momentum_period
        )
        return pd.Series(
            ma_momentum,
            index=prices.index,
            name=f"MA_Momentum_{ma_period}_{momentum_period}",
        )

    def QSTICK(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate QStick"""
        self._validate_data(data, ["open", "close"])

        qstick = self._calculate_qstick_numba(
            data["close"].values, data["open"].values, period
        )
        return pd.Series(qstick, index=data.index, name=f"QStick_{period}")

    def TYPICAL_PRICE(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Typical Price"""
        self._validate_data(data, ["high", "low", "close"])

        tp = self._calculate_typical_price_numba(
            data["high"].values, data["low"].values, data["close"].values
        )
        return pd.Series(tp, index=data.index, name="Typical_Price")

    def WEIGHTED_CLOSE(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Weighted Close Price"""
        self._validate_data(data, ["high", "low", "close"])

        wc = self._calculate_weighted_close_numba(
            data["high"].values, data["low"].values, data["close"].values
        )
        return pd.Series(wc, index=data.index, name="Weighted_Close")

    def FRACTAL(self, data: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """Calculate Fractal Indicator"""
        self._validate_data(data, ["high", "low"])

        bullish, bearish = self._calculate_fractal_numba(
            data["high"].values, data["low"].values, period
        )
        return pd.DataFrame(
            {"Bullish_Fractal": bullish, "Bearish_Fractal": bearish}, index=data.index
        )

    # Multiple Indicators -----------------------------------------------------

    # Add to TechnicalIndicators class
    def calculate_multiple_indicators(
        self,
        data: pd.DataFrame,
        indicators: List[Dict],
        symbol: str = None,
        n_jobs: int = -1,
        use_cache: bool = True,
    ) -> Dict[str, pd.Series]:
        """
        Calculate multiple indicators in parallel

        Example usage:
        indicators = [
                {'name': 'SMA', 'params': {'period': 20}},
                {'name': 'RSI', 'params': {'period': 14}},
        ]
        """
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing
        import time

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        results = {}
        calculation_times = {}

        def calculate_single(indicator):
            name = indicator["name"]
            params = indicator.get("params", {})
            columns = indicator.get("columns", None)

            start_time = time.time()
            try:
                # Get the indicator calculation method
                func = getattr(self, name)

                # Prepare data based on columns specification
                if columns:
                    indicator_data = data[columns]
                else:
                    indicator_data = data

                # Calculate indicator
                result = func(indicator_data, **params)

                calculation_time = time.time() - start_time
                return name, result, calculation_time

            except Exception as e:
                logger.error(f"Error calculating {name} for {symbol}: {str(e)}")
                return name, None, time.time() - start_time

        # Execute calculations in parallel
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(calculate_single, ind) for ind in indicators]

            for future in futures:
                name, result, calc_time = future.result()
                if result is not None:
                    results[name] = result
                    calculation_times[name] = calc_time

        # Log performance metrics
        logger.info(f"Indicator calculation times for {symbol}: {calculation_times}")
        # print(calculation_times)
        # print(results)
        return results


class UnitTests(Enum):
    # Trend Indicators
    SMA = 1
    EMA = 2
    DEMA = 3
    KAMA = 4
    ALMA = 5
    ICHIMOKU = 6
    SUPERTREND = 7
    LINEAR_REGRESSION = 8

    # Momentum Indicators
    RSI = 9
    MACD = 10
    STOCH = 11
    CCI = 12
    WILLR = 13
    ROC = 14
    TRIX = 15
    ULTIMATE_OSCILLATOR = 16
    CMO = 17
    DPO = 18
    KDJ = 19

    # Volume Indicators
    AD = 20
    ADOSC = 21
    MFI = 22
    OBV = 23
    PVO = 24
    VWAP = 25
    VOLUME_INDICATORS = 26

    # Volatility Indicators
    BB = 27
    ATR = 28
    HISTORICAL_VOLATILITY = 29
    CHAIKIN_VOLATILITY = 30
    KELTNER = 31
    DONCHIAN = 32

    # Directional Indicators
    ADX = 33
    DI = 34
    AROON = 35
    AO = 36
    BETA = 37

    # Pattern Indicators
    MASS_INDEX = 38
    HEIKEN_ASHI = 39
    BENFORD_LAW = 40

    # Multiple indicators
    PARALLEL_INDICATORS = 41


# Define the test configurations separately for clarity
def get_test_configs(ti, data, market_data):
    return {
        # Trend Indicators
        "SMA": {
            "func": ti.SMA,
            "args": (data["close"],),
            "kwargs": {"period": 20},
            "validation": {"check_nan_head": 19},
        },
        "EMA": {
            "func": ti.EMA,
            "args": (data["close"],),
            "kwargs": {"period": 20},
            "validation": {"check_nan_head": 19},
        },
        "DEMA": {
            "func": ti.DEMA,
            "args": (data["close"],),
            "kwargs": {"period": 20},
            "validation": {"check_nan_head": 19},
        },
        "KAMA": {
            "func": ti.KAMA,
            "args": (data["close"],),
            "kwargs": {"er_period": 10, "fast_period": 2, "slow_period": 30},
            "validation": {"check_nan_head": 9},
        },
        "ALMA": {
            "func": ti.ALMA,
            "args": (data["close"],),
            "kwargs": {"period": 9, "sigma": 6.0, "offset": 0.85},
            "validation": {"check_nan_head": 8},
        },
        "ICHIMOKU": {
            "func": ti.ICHIMOKU,
            "args": (data,),
            "kwargs": {},
            "validation": {"check_nan_tail": {"Chikou Span": 26}},  # displacement = 26
        },
        "SUPERTREND": {
            "func": ti.SUPERTREND,
            "args": (data,),
            "kwargs": {"period": 10, "multiplier": 3.0},
            "validation": {"check_nan_head": 9},
        },
        "LINEAR_REGRESSION": {
            "func": ti.LINEAR_REGRESSION_CHANNEL,
            "args": (data["close"],),
            "kwargs": {"period": 20, "deviations": 2.0},
            "validation": {"check_nan_head": 19},
        },
        # Momentum Indicators
        "RSI": {
            "func": ti.RSI,
            "args": (data["close"],),
            "kwargs": {"period": 14},
            "validation": {"min_value": 0, "max_value": 100, "check_nan_head": 13},
        },
        "MACD": {
            "func": ti.MACD,
            "args": (data["close"],),
            "kwargs": {},
            "validation": {"check_nan_head": 25},
        },
        "STOCH": {
            "func": ti.STOCH,
            "args": (data,),
            "kwargs": {"k_period": 14, "d_period": 3},
            "validation": {"min_value": 0, "max_value": 100, "check_nan_head": 13},
        },
        "CCI": {
            "func": ti.CCI,
            "args": (data,),
            "kwargs": {"period": 20},
            "validation": {"check_nan_head": 19},
        },
        "WILLR": {
            "func": ti.WILLR,
            "args": (data,),
            "kwargs": {"period": 14},
            "validation": {"min_value": -100, "max_value": 0, "check_nan_head": 13},
        },
        "ROC": {
            "func": ti.ROC,
            "args": (data["close"],),
            "kwargs": {"period": 12},
            "validation": {"check_nan_head": 11},
        },
        "TRIX": {
            "func": ti.TRIX,
            "args": (data["close"],),
            "kwargs": {"period": 15},
            "validation": {"check_nan_head": 14},
        },
        "ULTIMATE_OSCILLATOR": {
            "func": ti.ULTIMATE_OSCILLATOR,
            "args": (data,),
            "kwargs": {},
            "validation": {"min_value": 0, "max_value": 100, "check_nan_head": 27},
        },
        "CMO": {
            "func": ti.CMO,
            "args": (data["close"],),
            "kwargs": {"period": 14},
            "validation": {"min_value": -100, "max_value": 100, "check_nan_head": 13},
        },
        "DPO": {
            "func": ti.DPO,
            "args": (data["close"],),
            "kwargs": {"period": 20},
            "validation": {"check_nan_head": 19},
        },
        "KDJ": {
            "func": ti.KDJ,
            "args": (data,),
            "kwargs": {"k_period": 9, "d_period": 3},
            "validation": {
                "min_value": 0,
                "max_value": 100,
                "check_nan_head": {"K": 8, "D": 10},
                "columns": ["K", "D"],  # Specify columns to validate
            },
        },
        # Volume Indicators
        "AD": {
            "func": ti.AD,
            "args": (data,),
            "kwargs": {},
            "validation": {"check_nan_head": 0},
        },
        "ADOSC": {
            "func": ti.ADOSC,
            "args": (data,),
            "kwargs": {"fast_period": 3, "slow_period": 10},
            "validation": {"check_nan_head": 9},  # slow_period - 1 = 9
        },
        "MFI": {
            "func": ti.MFI,
            "args": (data,),
            "kwargs": {"period": 14},
            "validation": {"min_value": 0, "max_value": 100, "check_nan_head": 13},
        },
        "PVO": {
            "func": ti.PVO,
            "args": (data,),
            "kwargs": {},
            "validation": {"check_nan_head": 25},
        },
        "VWAP": {
            "func": ti.VWAP,
            "args": (data,),
            "kwargs": {"period": 14},
            "validation": {"check_nan_head": 13},  # period - 1 = 13
        },
        "VOLUME_INDICATORS": {
            "func": ti.VOLUME_INDICATORS,
            "args": (data,),
            "kwargs": {"period": 20},
            "validation": {
                "check_nan_head": {"Volume_SMA": 19, "Force_Index": 1, "VPT": 0}
            },
        },
        # Volatility Indicators
        "BB": {
            "func": ti.BB,
            "args": (data["close"],),
            "kwargs": {"period": 20, "num_std": 2.0},
            "validation": {"check_nan_head": 19},
        },
        "ATR": {
            "func": ti.ATR,
            "args": (data,),
            "kwargs": {"period": 14},
            "validation": {"min_value": 0, "check_nan_head": 13},
        },
        "HISTORICAL_VOLATILITY": {
            "func": ti.HISTORICAL_VOLATILITY,
            "args": (data["close"],),
            "kwargs": {"period": 20, "trading_days": 252},
            "validation": {"min_value": 0, "check_nan_head": 19},  # period - 1
        },
        "CHAIKIN_VOLATILITY": {
            "func": ti.CHAIKIN_VOLATILITY,
            "args": (data,),
            "kwargs": {"ema_period": 10, "roc_period": 10},
            "validation": {"check_nan_head": 18},  # (10 + 10 - 2) = 18
        },
        "KELTNER": {
            "func": ti.KELTNER,
            "args": (data,),
            "kwargs": {},
            "validation": {"check_nan_head": 19},
        },
        "DONCHIAN": {
            "func": ti.DONCHIAN,
            "args": (data,),
            "kwargs": {"period": 20},
            "validation": {"check_nan_head": 19},  # period - 1
        },
        "ADX": {
            "func": ti.ADX,
            "args": (data,),
            "kwargs": {"period": 14},
            "validation": {
                "min_value": 0,
                "max_value": 100,
                "check_nan_head": {"ADX": 27, "+DI": 14, "-DI": 14},
                "columns": ["ADX", "+DI", "-DI"],  # Specify columns explicitly
            },
        },
        "DI": {
            "func": ti.DI,
            "args": (data,),
            "kwargs": {"period": 14},  # Corrected from 'window' to 'period'
            "validation": {"min_value": 0, "max_value": 100, "check_nan_head": 13},
        },
        "AROON": {
            "func": ti.AROON,
            "args": (data,),
            "kwargs": {"period": 25},
            "validation": {
                "min_value": 0,
                "max_value": 100,
                "check_nan_head": 24,
                "columns": ["AROON_UP", "AROON_DOWN"],  # Only validate these columns
            },
        },
        "AO": {
            "func": ti.AO,
            "args": (data,),
            "kwargs": {},
            "validation": {"check_nan_head": 33},  # long_period - 1
        },
        "BETA": {
            "func": ti.BETA,
            "args": (data["close"], market_data),
            "kwargs": {"period": 252},  # Corrected from 'window' to 'period'
            "validation": {"check_nan_head": 251},
        },
        # Pattern Indicators
        "MASS_INDEX": {
            "func": ti.MASS_INDEX,
            "args": (data,),
            "kwargs": {},
            "validation": {"check_nan_head": 24},  # sum_period - 1
        },
        "HEIKEN_ASHI": {
            "func": ti.HEIKEN_ASHI,
            "args": (data,),
            "kwargs": {},
            "validation": {"check_nan_head": 0},
        },
        "BENFORD_LAW": {
            "func": ti.BENFORD_LAW,
            "args": (data["close"],),
            "kwargs": {},
            "validation": {},
        },
        "OBV": {
            "func": ti.OBV,  # Assuming you have an OBV function
            "args": (data,),
            "kwargs": {},
            "validation": {"check_nan_head": 0},
        },
        "PARALLEL_INDICATORS": {
            "func": ti.calculate_multiple_indicators,
            "args": (data,),
            "kwargs": {
                "indicators": [
                    {"name": "SMA", "params": {"period": 20}},
                    {"name": "RSI", "params": {"period": 14}},
                    {"name": "MACD", "params": {}},
                    {"name": "BB", "params": {"period": 20, "num_std": 2.0}},
                    {"name": "EMA", "params": {"period": 30}},
                ],
                "n_jobs": 2,  # Test with specific number of threads
            },
            "validation": {
                "expected_indicators": ["SMA", "RSI", "MACD", "BB", "EMA"],
                "check_nan_head": {
                    "SMA": 19,
                    "RSI": 13,
                    "MACD": 25,
                    "BB_Upper": 19,
                    "EMA": 29,
                },
            },
        },
    }


# Download the data once and reuse it for all tests
def download_data():
    logger.info("Downloading data...")
    data = yf.download("AAPL", start="2000-01-01", end="2024-05-31", progress=True)
    market_data = yf.download(
        "^GSPC", start="2000-01-01", end="2023-05-31", progress=True
    )["Adj Close"]

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs("AAPL", axis=1, level=1)

    # Rename columns for consistency
    data = data.rename(
        columns={
            "Adj Close": "adj_close",
            "Close": "close",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Volume": "volume",
        }
    )

    logger.info("Data downloaded successfully.")

    return data, market_data


@timer
def validate_parallel_indicators(result, validation):
    """Validate results from parallel indicator calculation"""
    if not isinstance(result, dict):
        logger.info("Result should be a dictionary")
        return False

    # Check if all expected indicators are present
    expected_indicators = validation.get("expected_indicators", [])
    for indicator in expected_indicators:
        if indicator not in result and not any(
            indicator in key for key in result.keys()
        ):
            logger.info(f"Missing indicator: {indicator}")
            return False

    # Check NaN values at the head for each indicator
    nan_checks = validation.get("check_nan_head", {})
    for indicator, expected_nans in nan_checks.items():
        # Handle MACD special case as it returns a DataFrame
        if indicator == "MACD":
            if not isinstance(result["MACD"], pd.DataFrame):
                logger.info("MACD result should be a DataFrame")
                return False
            num_nans = result["MACD"]["MACD"].head(expected_nans).isna().sum()
        # Handle BB special case as it returns multiple columns
        elif indicator.startswith("BB_"):
            if "BB" not in result:
                logger.info("BB result not found")
                return False
            num_nans = result["BB"][indicator].head(expected_nans).isna().sum()
        else:
            # Handle regular indicators that return Series
            series_key = next(
                (k for k in result.keys() if k.startswith(indicator)), None
            )
            if series_key is None:
                logger.info(f"Indicator {indicator} not found in results")
                return False
            num_nans = result[series_key].head(expected_nans).isna().sum()

        if num_nans < expected_nans:
            logger.info(
                f"Validation failed for {indicator}: Expected {expected_nans} NaN values but found {num_nans}"
            )
            return False

    return True


# Run individual tests
def run_unit_test(unit_test: UnitTests, test_configs):
    test_name = unit_test.name
    print(f"\n-- Running test for: {test_name} --")

    if test_name not in test_configs:
        logger.error(f"No test configuration found for {test_name}. Skipping...")
        return None

    config = test_configs[test_name]

    try:
        # Run the test
        result = config["func"](*config["args"], **config["kwargs"])

        # Special handling for parallel indicators test
        if test_name == "PARALLEL_INDICATORS":
            if validate_parallel_indicators(result, config["validation"]):
                logger.info(f"\nTest {test_name}: PASSED")
                print(result)
                return result
            else:
                logger.info(f"\nTest {test_name}: FAILED")
                return None

        # Validation checks
        if "validation" in config:
            validation = config["validation"]
            check_nan_head = validation.get("check_nan_head", None)
            check_nan_tail = validation.get("check_nan_tail", None)
            min_value = validation.get("min_value", None)
            max_value = validation.get("max_value", None)
            columns_to_validate = validation.get(
                "columns", result.columns if isinstance(result, pd.DataFrame) else None
            )

            # Perform validation for NaN values at the head
            if check_nan_head is not None:
                if isinstance(result, pd.DataFrame):
                    if isinstance(check_nan_head, dict):
                        # Check each specified column for NaN values
                        for column, expected_nans in check_nan_head.items():
                            num_nans = result[column].head(expected_nans).isna().sum()
                            if num_nans < expected_nans:
                                logger.info(
                                    f"Validation failed for {column}: Expected {expected_nans} NaN values but found {num_nans}."
                                )
                                return None
                    else:
                        # Check each column for NaN values
                        for column in result.columns:
                            num_nans = result[column].head(check_nan_head).isna().sum()
                            if num_nans < check_nan_head:
                                logger.info(
                                    f"Validation failed for {column}: Expected {check_nan_head} NaN values but found {num_nans}."
                                )
                                return None
                else:
                    # If it's a Series, check directly
                    num_nans = result.head(check_nan_head).isna().sum()
                    if num_nans < check_nan_head:
                        logger.info(
                            f"Validation failed: Expected {check_nan_head} NaN values but found {num_nans}."
                        )
                        return None

            # Perform validation for NaN values at the tail
            if check_nan_tail is not None:
                if isinstance(result, pd.DataFrame):
                    if isinstance(check_nan_tail, dict):
                        # Check each specified column for NaN values at the tail
                        for column, expected_nans in check_nan_tail.items():
                            num_nans = result[column].tail(expected_nans).isna().sum()
                            if num_nans < expected_nans:
                                logger.info(
                                    f"Validation failed for {column}: Expected {expected_nans} NaN values at the tail but found {num_nans}."
                                )
                                return None
                    else:
                        # Check each column for NaN values at the tail
                        for column in result.columns:
                            num_nans = result[column].tail(check_nan_tail).isna().sum()
                            if num_nans < check_nan_tail:
                                logger.info(
                                    f"Validation failed for {column}: Expected {check_nan_tail} NaN values at the tail but found {num_nans}."
                                )
                                return None
                else:
                    # If it's a Series, check directly
                    num_nans = result.tail(check_nan_tail).isna().sum()
                    if num_nans < check_nan_tail:
                        logger.info(
                            f"Validation failed: Expected {check_nan_tail} NaN values at the tail but found {num_nans}."
                        )
                        return None

            # Perform min/max validation
            if min_value is not None:
                if isinstance(result, pd.DataFrame):
                    for column in columns_to_validate:
                        min_result_value = result[column].min()
                        if min_result_value < min_value:
                            logger.info(
                                f"Validation failed: Some values in {column} are below {min_value}."
                            )
                            return None
                else:
                    if result.min() < min_value:
                        logger.info(
                            f"Validation failed: Some values are below {min_value}."
                        )
                        return None
            if max_value is not None:
                if isinstance(result, pd.DataFrame):
                    for column in columns_to_validate:
                        max_result_value = result[column].max()
                        if max_result_value > max_value:
                            logger.info(
                                f"Validation failed: Some values in {column} are above {max_value}."
                            )
                            return None
                else:
                    if result.max() > max_value:
                        logger.info(
                            f"Validation failed: Some values are above {max_value}."
                        )
                        return None

        logger.info(f"\nTest {test_name}: PASSED")
        # logger.info(f"Sample calculated values:\n{result.tail(5)}")

        return result

    except Exception as e:
        logger.info(f"\nTest {test_name}: ERROR")
        logger.info(f"FAILED: Error message: {str(e)}")
        return None


# Run all tests
def run_all_tests(data, market_data):
    ti = TechnicalIndicators()  # Assuming this class is defined elsewhere
    test_configs = get_test_configs(ti, data, market_data)

    # Run specific parallel processing test
    result = run_unit_test(UnitTests.PARALLEL_INDICATORS, test_configs)
    exit()

    results = {}
    for test in UnitTests:
        result = run_unit_test(test, test_configs)
        results[test.name] = result
    return results


if __name__ == "__main__":
    # Download data once
    data, market_data = download_data()

    # Run all tests
    results = run_all_tests(data, market_data)


from technical_indicators import TechnicalIndicators

# Fetch data
data = yf.download("AAPL", start="2020-01-01", end="2025-01-01")
ti = TechnicalIndicators()

# Calculate SMA and RSI
sma = ti.SMA(data["Close"], period=20)
rsi = ti.RSI(data["Close"], period=14)

# View results
print("SMA:\n", sma.tail())
print("RSI:\n", rsi.tail())
