"""
QuantJourney Technical-Indicators - Numba Kernels
=================================================
This module contains the low-level, **Numba-compiled** functions that perform
the heavy numerical work for every indicator exposed in
`quantjourney_ti.indicators`. Each function follows the naming
convention::

    _calculate_<indicator>_numba(...)

and is decorated with ``@njit(parallel=False, fastmath=True)`` so that it is
JIT-compiled as soon as the module is imported. Keeping the kernels in a
separate file allows the high-level API (`TechnicalIndicators` class) to stay
small and readable while maximising performance.

When adding a new indicator:
1. Implement the pure-Python algorithm here.
2. Prepend the ``@njit(parallel=False, fastmath=True)`` decorator.
3. Re-export the function from `quantjourney_ti.indicators` by ensuring the
   dynamic patching block in that file picks it up (it already scans for
   names that start with ``_calculate_`` and end with ``_numba``).

All functions handle edge cases (e.g., NaNs, empty arrays, invalid inputs) to
ensure robustness in production environments, minimizing the need for fallbacks.

Author: Jakub Polec  <jakub@quantjourney.pro>
License: MIT
"""

import numpy as np
from numba import njit
from typing import Tuple, Union, Optional


@njit(parallel=False, fastmath=True)
def _calculate_ad_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    """Optimized Accumulation/Distribution Line calculation."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or len(volume) != n:
        return np.empty(0, dtype=np.float64)  # Fix Numba type inference
    ad = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]) or np.isnan(volume[i]):
            ad[i] = ad[i - 1] if i > 0 else 0.0
            continue
        if high[i] == low[i]:
            money_flow_vol = 0.0
        else:
            clv = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            money_flow_vol = clv * volume[i]
        ad[i] = money_flow_vol if i == 0 else ad[i - 1] + money_flow_vol
    return ad


@njit(parallel=False, fastmath=True)
def _calculate_adosc_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    fast_period: int = 3,
    slow_period: int = 10,
) -> np.ndarray:
    """Optimized Chaikin A/D Oscillator calculation."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or len(volume) != n or fast_period <= 0 or slow_period <= 0:
        return np.empty(0, dtype=np.float64)  # Fix Numba type inference
    ad = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]) or np.isnan(volume[i]):
            ad[i] = ad[i - 1] if i > 0 else 0.0
            continue
        if high[i] == low[i]:
            money_flow_vol = 0.0
        else:
            clv = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            money_flow_vol = clv * volume[i]
        ad[i] = money_flow_vol if i == 0 else ad[i - 1] + money_flow_vol
    fast_ema = np.zeros(n, dtype=np.float64)
    slow_ema = np.zeros(n, dtype=np.float64)
    if n > 0:
        fast_ema[0] = ad[0]
        slow_ema[0] = ad[0]
    fast_mult = 2.0 / (fast_period + 1)
    slow_mult = 2.0 / (slow_period + 1)
    for i in range(1, n):
        fast_ema[i] = (ad[i] - fast_ema[i - 1]) * fast_mult + fast_ema[i - 1]
        slow_ema[i] = (ad[i] - slow_ema[i - 1]) * slow_mult + slow_ema[i - 1]
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        result[i] = fast_ema[i] - slow_ema[i]
    return result


@njit(parallel=False, fastmath=True)
def _calculate_adx_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized ADX calculation."""
    n = len(high)
    if n == 0 or len(low) != n or len(close) != n or window <= 0:
        return np.array([]), np.array([]), np.array([])
    tr = np.full(n, np.nan, dtype=np.float64)
    plus_dm = np.full(n, np.nan, dtype=np.float64)
    minus_dm = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i - 1]):
            continue
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    smoothed_tr = np.full(n, np.nan, dtype=np.float64)
    smoothed_plus_dm = np.full(n, np.nan, dtype=np.float64)
    smoothed_minus_dm = np.full(n, np.nan, dtype=np.float64)
    if window < n:
        smoothed_tr[window] = np.nansum(tr[1 : window + 1])
        smoothed_plus_dm[window] = np.nansum(plus_dm[1 : window + 1])
        smoothed_minus_dm[window] = np.nansum(minus_dm[1 : window + 1])
        for i in range(window + 1, n):
            if not np.isnan(tr[i]):
                smoothed_tr[i] = smoothed_tr[i - 1] - (smoothed_tr[i - 1] / window) + tr[i]
                smoothed_plus_dm[i] = (
                    smoothed_plus_dm[i - 1] - (smoothed_plus_dm[i - 1] / window) + plus_dm[i]
                )
                smoothed_minus_dm[i] = (
                    smoothed_minus_dm[i - 1] - (smoothed_minus_dm[i - 1] / window) + minus_dm[i]
                )
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    for i in range(window, n):
        if smoothed_tr[i] != 0 and not np.isnan(smoothed_tr[i]):
            plus_di[i] = 100 * smoothed_plus_dm[i] / smoothed_tr[i]
            minus_di[i] = 100 * smoothed_minus_dm[i] / smoothed_tr[i]
    dx = np.full(n, np.nan, dtype=np.float64)
    for i in range(window, n):
        if plus_di[i] + minus_di[i] != 0 and not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    adx = np.full(n, np.nan, dtype=np.float64)
    if 2 * window - 1 < n:
        adx[2 * window - 1] = np.nanmean(dx[window : 2 * window])
        for i in range(2 * window, n):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i - 1] * (window - 1) + dx[i]) / window
    return adx, plus_di, minus_di


@njit(parallel=False, fastmath=True)
def _calculate_alma_numba(
    close: np.ndarray, window: int, sigma: float = 6.0, offset: float = 0.85
) -> np.ndarray:
    """Optimized Arnaud Legoux Moving Average calculation."""
    n = len(close)
    if n == 0 or window <= 0 or window > n:
        return np.full(n, np.nan, dtype=np.float64)
    alma = np.full(n, np.nan, dtype=np.float64)
    m = offset * (window - 1)
    s = window / sigma
    weights = np.zeros(window, dtype=np.float64)
    weight_sum = 0.0
    for i in range(window):
        weights[i] = np.exp(-((i - m) ** 2) / (2 * s * s))
        weight_sum += weights[i]
    if weight_sum == 0:
        return alma
    for i in range(window):
        weights[i] /= weight_sum
    for i in range(window - 1, n):
        if np.any(np.isnan(close[i - window + 1 : i + 1])):
            continue
        weighted_sum = 0.0
        for j in range(window):
            weighted_sum += close[i - window + 1 + j] * weights[j]
        alma[i] = weighted_sum
    return alma


@njit(parallel=False, fastmath=True)
def _calculate_aroon_numba(
    high: np.ndarray, low: np.ndarray, period: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized Aroon Indicator calculation."""
    n = len(high)
    if n == 0 or len(low) != n or period <= 0 or period > n:
        return np.array([]), np.array([])
    aroon_up = np.full(n, np.nan, dtype=np.float64)
    aroon_down = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if np.any(np.isnan(high[i - period + 1 : i + 1])) or np.any(np.isnan(low[i - period + 1 : i + 1])):
            continue
        high_window = high[i - period + 1 : i + 1]
        low_window = low[i - period + 1 : i + 1]
        high_idx = np.argmax(high_window)
        low_idx = np.argmin(low_window)
        aroon_up[i] = ((period - (period - high_idx - 1)) / period) * 100
        aroon_down[i] = ((period - (period - low_idx - 1)) / period) * 100
    return aroon_up, aroon_down


@njit(parallel=False, fastmath=True)
def _calculate_atr_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int
) -> np.ndarray:
    """Optimized ATR calculation."""
    n = len(high)
    if n == 0 or len(low) != n or len(close) != n or window <= 0 or window > n:
        return np.full(n, np.nan, dtype=np.float64)
    tr = np.full(n, np.nan, dtype=np.float64)
    atr = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i - 1]):
            continue
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    if window < n:
        initial_sum = 0.0
        count = 0
        for i in range(1, window + 1):
            if not np.isnan(tr[i]):
                initial_sum += tr[i]
                count += 1
        if count > 0:
            atr[window] = initial_sum / count
        for i in range(window + 1, n):
            if not np.isnan(tr[i]) and not np.isnan(atr[i - 1]):
                atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
    return atr


@njit(parallel=False, fastmath=True)
def _calculate_awesome_oscillator_numba(
    high: np.ndarray, low: np.ndarray, short_period: int = 5, long_period: int = 34
) -> np.ndarray:
    """Optimized Awesome Oscillator calculation."""
    n = len(high)
    if n == 0 or len(low) != n or short_period <= 0 or long_period <= 0 or long_period > n:
        return np.full(n, np.nan, dtype=np.float64)
    ao = np.full(n, np.nan, dtype=np.float64)
    median_price = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(high[i]) and not np.isnan(low[i]):
            median_price[i] = (high[i] + low[i]) / 2
    for i in range(long_period - 1, n):
        if np.any(np.isnan(median_price[i - short_period + 1 : i + 1])) or np.any(np.isnan(median_price[i - long_period + 1 : i + 1])):
            continue
        short_sum = 0.0
        long_sum = 0.0
        short_count = 0
        long_count = 0
        for j in range(i - short_period + 1, i + 1):
            short_sum += median_price[j]
            short_count += 1
        for j in range(i - long_period + 1, i + 1):
            long_sum += median_price[j]
            long_count += 1
        if short_count > 0 and long_count > 0:
            ao[i] = (short_sum / short_count) - (long_sum / long_count)
    return ao

@njit(parallel=False, fastmath=True)
def _calculate_obv_numba(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    n = len(close)
    if n == 0 or len(volume) != n:
        return np.full(n, np.nan, dtype=np.float64)
    obv = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if np.isnan(close[i]) or np.isnan(close[i - 1]) or np.isnan(volume[i]):
            obv[i] = obv[i - 1]
        elif close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    return obv

@njit(parallel=False, fastmath=True)
def _calculate_beta_numba(
    returns: np.ndarray, market_returns: np.ndarray, window: int
) -> np.ndarray:
    """Optimized Beta calculation."""
    n = len(returns)
    if n == 0 or len(market_returns) != n or window <= 0 or window > n:
        return np.full(n, np.nan, dtype=np.float64)
    beta = np.full(n, np.nan, dtype=np.float64)
    for i in range(window, n):
        if np.any(np.isnan(returns[i - window : i])) or np.any(np.isnan(market_returns[i - window : i])):
            continue
        ret_window = returns[i - window : i]
        mkt_window = market_returns[i - window : i]
        ret_mean = 0.0
        mkt_mean = 0.0
        count = 0
        for j in range(window):
            ret_mean += ret_window[j]
            mkt_mean += mkt_window[j]
            count += 1
        ret_mean /= count
        mkt_mean /= count
        covar = 0.0
        mkt_var = 0.0
        for j in range(window):
            covar += (ret_window[j] - ret_mean) * (mkt_window[j] - mkt_mean)
            mkt_var += (mkt_window[j] - mkt_mean) ** 2
        covar /= count
        mkt_var /= count
        if mkt_var != 0:
            beta[i] = covar / mkt_var
    return beta


@njit(parallel=False, fastmath=True)
def _calculate_bollinger_bands_numba(
    prices: np.ndarray, window: int, num_std: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized Bollinger Bands calculation."""
    n = len(prices)
    if n == 0 or window <= 0 or window > n:
        empty = np.full(n, np.nan, dtype=np.float64)
        return empty, empty, empty
    middle_band = np.full(n, np.nan, dtype=np.float64)
    upper_band = np.full(n, np.nan, dtype=np.float64)
    lower_band = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        if np.any(np.isnan(prices[i - window + 1 : i + 1])):
            continue
        sum_prices = 0.0
        sum_sq = 0.0
        count = 0
        for j in range(i - window + 1, i + 1):
            sum_prices += prices[j]
            sum_sq += prices[j] * prices[j]
            count += 1
        mean = sum_prices / count
        variance = (sum_sq / count) - (mean * mean)
        std = np.sqrt(variance) if variance > 0 else 0.0
        middle_band[i] = mean
        upper_band[i] = mean + (std * num_std)
        lower_band[i] = mean - (std * num_std)
    return upper_band, middle_band, lower_band


@njit(parallel=False, fastmath=True)
def _calculate_cci_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    constant: float = 0.015,
) -> np.ndarray:
    """Optimized Commodity Channel Index calculation."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    tp = np.full(n, np.nan, dtype=np.float64)
    cci = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(high[i]) and not np.isnan(low[i]) and not np.isnan(close[i]):
            tp[i] = (high[i] + low[i] + close[i]) / 3.0
    for i in range(period - 1, n):
        if np.any(np.isnan(tp[i - period + 1 : i + 1])):
            continue
        sum_tp = 0.0
        count = 0
        for j in range(i - period + 1, i + 1):
            sum_tp += tp[j]
            count += 1
        mean_tp = sum_tp / count
        sum_dev = 0.0
        for j in range(i - period + 1, i + 1):
            sum_dev += abs(tp[j] - mean_tp)
        mean_deviation = sum_dev / count
        if mean_deviation != 0 and not np.isnan(tp[i]):
            cci[i] = (tp[i] - mean_tp) / (constant * mean_deviation)
    return cci


@njit(parallel=False, fastmath=True)
def _calculate_chaikin_volatility_numba(
    high: np.ndarray, low: np.ndarray, ema_period: int = 10, roc_period: int = 10
) -> np.ndarray:
    """Optimized Chaikin Volatility calculation."""
    n = len(high)
    if n == 0 or len(low) != n or ema_period <= 0 or roc_period <= 0 or ema_period + roc_period > n:
        return np.full(n, np.nan, dtype=np.float64)
    hl_range = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(high[i]) and not np.isnan(low[i]):
            hl_range[i] = high[i] - low[i]
    ema = np.full(n, np.nan, dtype=np.float64)
    multiplier = 2.0 / (ema_period + 1)
    if n > 0 and not np.isnan(hl_range[0]):
        ema[0] = hl_range[0]
    for i in range(1, n):
        if not np.isnan(hl_range[i]) and not np.isnan(ema[i - 1]):
            ema[i] = (hl_range[i] - ema[i - 1]) * multiplier + ema[i - 1]
    cv = np.full(n, np.nan, dtype=np.float64)
    for i in range(roc_period, n):
        if not np.isnan(ema[i]) and not np.isnan(ema[i - roc_period]) and ema[i - roc_period] != 0:
            cv[i] = ((ema[i] - ema[i - roc_period]) / ema[i - roc_period]) * 100
    return cv


@njit(parallel=False, fastmath=True)
def _calculate_chande_momentum_numba(close: np.ndarray, period: int) -> np.ndarray:
    """Optimized Chande Momentum Oscillator calculation."""
    n = len(close)
    if n == 0 or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    cmo = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if np.any(np.isnan(close[i - period : i + 1])):
            continue
        gains = 0.0
        losses = 0.0
        for j in range(i - period + 1, i + 1):
            change = close[j] - close[j - 1]
            if change > 0:
                gains += change
            else:
                losses += abs(change)
        if gains + losses != 0:
            cmo[i] = 100 * (gains - losses) / (gains + losses)
    return cmo


@njit(parallel=False, fastmath=True)
def _calculate_choppiness_index_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """Optimized Choppiness Index calculation."""
    n = len(high)
    if n == 0 or len(low) != n or len(close) != n or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    ci = np.full(n, np.nan, dtype=np.float64)
    tr = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i - 1]):
            continue
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    atr_sum = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if np.any(np.isnan(tr[i - period + 1 : i + 1])) or np.any(np.isnan(high[i - period + 1 : i + 1])) or np.any(np.isnan(low[i - period + 1 : i + 1])):
            continue
        sum_tr = 0.0
        count = 0
        for j in range(i - period + 1, i + 1):
            sum_tr += tr[j]
            count += 1
        atr_sum[i] = sum_tr / count
        highest_high = np.max(high[i - period + 1 : i + 1])
        lowest_low = np.min(low[i - period + 1 : i + 1])
        if highest_high != lowest_low:
            ci[i] = (
                100
                * np.log10(atr_sum[i] / (highest_high - lowest_low))
                / np.log10(period)
            )
    return ci


@njit(parallel=False, fastmath=True)
def _calculate_coppock_curve_numba(
    close: np.ndarray,
    roc1_period: int = 14,
    roc2_period: int = 11,
    wma_period: int = 10,
) -> np.ndarray:
    """Optimized Coppock Curve calculation."""
    n = len(close)
    if n == 0 or roc1_period <= 0 or roc2_period <= 0 or wma_period <= 0 or max(roc1_period, roc2_period) + wma_period > n:
        return np.full(n, np.nan, dtype=np.float64)
    roc1 = np.full(n, np.nan, dtype=np.float64)
    roc2 = np.full(n, np.nan, dtype=np.float64)
    max_period = max(roc1_period, roc2_period)
    for i in range(max_period, n):
        if np.any(np.isnan(close[i - max_period : i + 1])):
            continue
        if close[i - roc1_period] != 0:
            roc1[i] = (close[i] - close[i - roc1_period]) / close[i - roc1_period] * 100
        if close[i - roc2_period] != 0:
            roc2[i] = (close[i] - close[i - roc2_period]) / close[i - roc2_period] * 100
    roc_sum = np.full(n, np.nan, dtype=np.float64)
    for i in range(max_period, n):
        if not np.isnan(roc1[i]) and not np.isnan(roc2[i]):
            roc_sum[i] = roc1[i] + roc2[i]
    coppock = np.full(n, np.nan, dtype=np.float64)
    for i in range(max_period + wma_period - 1, n):
        if np.any(np.isnan(roc_sum[i - wma_period + 1 : i + 1])):
            continue
        weights_sum = 0.0
        data_sum = 0.0
        count = 0
        for j in range(wma_period):
            weight = wma_period - j
            weights_sum += weight
            data_sum += roc_sum[i - j] * weight
            count += 1
        if weights_sum != 0:
            coppock[i] = data_sum / weights_sum
    return coppock


@njit(parallel=False, fastmath=True)
def _calculate_dema_numba(close: np.ndarray, period: int) -> np.ndarray:
    """Optimized Double Exponential Moving Average calculation."""
    n = len(close)
    if n == 0 or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    alpha = 2.0 / (period + 1)
    ema1 = np.full(n, np.nan, dtype=np.float64)
    ema2 = np.full(n, np.nan, dtype=np.float64)
    if not np.isnan(close[0]):
        ema1[0] = close[0]
        ema2[0] = close[0]
    for i in range(1, n):
        if not np.isnan(close[i]) and not np.isnan(ema1[i - 1]):
            ema1[i] = (close[i] - ema1[i - 1]) * alpha + ema1[i - 1]
        if not np.isnan(ema1[i]) and not np.isnan(ema2[i - 1]):
            ema2[i] = (ema1[i] - ema2[i - 1]) * alpha + ema2[i - 1]
    dema = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(ema1[i]) and not np.isnan(ema2[i]):
            dema[i] = 2 * ema1[i] - ema2[i]
    return dema


@njit(parallel=False, fastmath=True)
def _calculate_di_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized Directional Indicator calculation."""
    n = len(high)
    if n == 0 or len(low) != n or len(close) != n or period <= 0 or period > n:
        return np.array([]), np.array([])
    plus_dm = np.full(n, np.nan, dtype=np.float64)
    minus_dm = np.full(n, np.nan, dtype=np.float64)
    tr = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(high[i - 1]) or np.isnan(low[i - 1]) or np.isnan(close[i - 1]):
            continue
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    smooth_plus_dm = np.full(n, np.nan, dtype=np.float64)
    smooth_minus_dm = np.full(n, np.nan, dtype=np.float64)
    smooth_tr = np.full(n, np.nan, dtype=np.float64)
    if period < n:
        smooth_plus_dm[period] = np.nansum(plus_dm[1 : period + 1])
        smooth_minus_dm[period] = np.nansum(minus_dm[1 : period + 1])
        smooth_tr[period] = np.nansum(tr[1 : period + 1])
        for i in range(period + 1, n):
            if not np.isnan(plus_dm[i]) and not np.isnan(smooth_plus_dm[i - 1]):
                smooth_plus_dm[i] = (
                    smooth_plus_dm[i - 1] - (smooth_plus_dm[i - 1] / period) + plus_dm[i]
                )
            if not np.isnan(minus_dm[i]) and not np.isnan(smooth_minus_dm[i - 1]):
                smooth_minus_dm[i] = (
                    smooth_minus_dm[i - 1] - (smooth_minus_dm[i - 1] / period) + minus_dm[i]
                )
            if not np.isnan(tr[i]) and not np.isnan(smooth_tr[i - 1]):
                smooth_tr[i] = smooth_tr[i - 1] - (smooth_tr[i - 1] / period) + tr[i]
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if smooth_tr[i] != 0 and not np.isnan(smooth_tr[i]):
            plus_di[i] = 100 * smooth_plus_dm[i] / smooth_tr[i]
            minus_di[i] = 100 * smooth_minus_dm[i] / smooth_tr[i]
    return plus_di, minus_di


@njit(parallel=False, fastmath=True)
def _calculate_disparity_index_numba(close: np.ndarray, period: int) -> np.ndarray:
    """Optimized Disparity Index calculation."""
    n = len(close)
    if n == 0 or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    disparity = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if np.any(np.isnan(close[i - period + 1 : i + 1])):
            continue
        sum_close = 0.0
        count = 0
        for j in range(i - period + 1, i + 1):
            sum_close += close[j]
            count += 1
        sma = sum_close / count
        if sma != 0 and not np.isnan(close[i]):
            disparity[i] = (close[i] / sma - 1) * 100
    return disparity


@njit(parallel=False, fastmath=True)
def _calculate_donchian_channels_numba(
    high: np.ndarray, low: np.ndarray, period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized Donchian Channels calculation."""
    n = len(high)
    if n == 0 or len(low) != n or period <= 0 or period > n:
        empty = np.full(n, np.nan, dtype=np.float64)
        return empty, empty, empty
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if np.any(np.isnan(high[i - period + 1 : i + 1])) or np.any(np.isnan(low[i - period + 1 : i + 1])):
            continue
        upper[i] = np.max(high[i - period + 1 : i + 1])
        lower[i] = np.min(low[i - period + 1 : i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    return upper, middle, lower


@njit(parallel=False, fastmath=True)
def _calculate_dpo_numba(close: np.ndarray, period: int) -> np.ndarray:
    """Optimized Detrended Price Oscillator calculation."""
    n = len(close)
    if n == 0 or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    dpo = np.full(n, np.nan, dtype=np.float64)
    shift = period // 2 + 1
    for i in range(period - 1, n):
        if np.any(np.isnan(close[i - period + 1 : i + 1])):
            continue
        sum_close = 0.0
        count = 0
        for j in range(i - period + 1, i + 1):
            sum_close += close[j]
            count += 1
        sma = sum_close / count
        if i - shift >= 0 and not np.isnan(close[i - shift]):
            dpo[i] = close[i - shift] - sma
    return dpo


@njit(parallel=False, fastmath=True)
def _calculate_elder_ray_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized Elder Ray Index calculation."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)
    bull_power = np.full(n, np.nan, dtype=np.float64)
    bear_power = np.full(n, np.nan, dtype=np.float64)
    ema = np.full(n, np.nan, dtype=np.float64)
    multiplier = 2.0 / (period + 1)
    if not np.isnan(close[0]):
        ema[0] = close[0]
    for i in range(1, n):
        if not np.isnan(close[i]) and not np.isnan(ema[i - 1]):
            ema[i] = (close[i] - ema[i - 1]) * multiplier + ema[i - 1]
    for i in range(n):
        if not np.isnan(high[i]) and not np.isnan(low[i]) and not np.isnan(ema[i]):
            bull_power[i] = high[i] - ema[i]
            bear_power[i] = low[i] - ema[i]
    return bull_power, bear_power


@njit(parallel=False, fastmath=True)
def _calculate_ema_numba(prices: np.ndarray, window: int) -> np.ndarray:
    """Optimized Exponential Moving Average calculation."""
    n = len(prices)
    if n == 0 or window <= 0 or window > n:
        return np.full(n, np.nan, dtype=np.float64)
    ema = np.full(n, np.nan, dtype=np.float64)
    multiplier = 2.0 / (window + 1)
    sum_prices = 0.0
    count = 0
    for i in range(window):
        if not np.isnan(prices[i]):
            sum_prices += prices[i]
            count += 1
    if count > 0:
        ema[window - 1] = sum_prices / count
    for i in range(window, n):
        if not np.isnan(prices[i]) and not np.isnan(ema[i - 1]):
            ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]
    return ema


@njit(parallel=False, fastmath=True)
def _calculate_fractal_numba(
    high: np.ndarray, low: np.ndarray, period: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized Fractal Indicator calculation."""
    n = len(high)
    if n == 0 or len(low) != n or period <= 0 or period > n or period % 2 == 0:
        return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)
    bullish = np.full(n, np.nan, dtype=np.float64)
    bearish = np.full(n, np.nan, dtype=np.float64)
    half = period // 2
    for i in range(half, n - half):
        if np.any(np.isnan(high[i - half : i + half + 1])) or np.any(np.isnan(low[i - half : i + half + 1])):
            continue
        is_bullish = True
        peak = high[i]
        for j in range(1, half + 1):
            if high[i - j] >= peak or high[i + j] >= peak:
                is_bullish = False
                break
        if is_bullish:
            bullish[i] = 1
        is_bearish = True
        trough = low[i]
        for j in range(1, half + 1):
            if low[i - j] <= trough or low[i + j] <= trough:
                is_bearish = False
                break
        if is_bearish:
            bearish[i] = 1
    return bullish, bearish


@njit(parallel=False, fastmath=True)
def _calculate_heiken_ashi_numba(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Optimized Heiken Ashi calculation."""
    n = len(close)
    if n == 0 or len(open_) != n or len(high) != n or len(low) != n:
        empty = np.full(n, np.nan, dtype=np.float64)
        return empty, empty, empty, empty
    ha_close = np.full(n, np.nan, dtype=np.float64)
    ha_open = np.full(n, np.nan, dtype=np.float64)
    ha_high = np.full(n, np.nan, dtype=np.float64)
    ha_low = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(open_[i]) and not np.isnan(high[i]) and not np.isnan(low[i]) and not np.isnan(close[i]):
            ha_close[i] = (open_[i] + high[i] + low[i] + close[i]) / 4
    if n > 0 and not np.isnan(open_[0]) and not np.isnan(close[0]):
        ha_open[0] = (open_[0] + close[0]) / 2
    for i in range(1, n):
        if not np.isnan(ha_open[i - 1]) and not np.isnan(ha_close[i - 1]):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
        if not np.isnan(high[i]) and not np.isnan(ha_open[i]) and not np.isnan(ha_close[i]):
            ha_high[i] = max(high[i], ha_open[i], ha_close[i])
            ha_low[i] = min(low[i], ha_open[i], ha_close[i])
    return ha_open, ha_high, ha_low, ha_close


@njit(parallel=False, fastmath=True)
def _calculate_historical_volatility_numba(
    close: np.ndarray, period: int = 20, trading_days: int = 252
) -> np.ndarray:
    """Optimized Historical Volatility calculation."""
    n = len(close)
    if n == 0 or period <= 0 or period >= n or trading_days <= 0:
        return np.full(n, np.nan, dtype=np.float64)
    hv = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if np.any(np.isnan(close[i - period : i + 1])):
            continue
        returns = np.zeros(period, dtype=np.float64)
        count = 0
        for j in range(1, period + 1):
            if close[i - period + j] != 0 and not np.isnan(close[i - period + j]) and not np.isnan(close[i - period + j - 1]):
                returns[j - 1] = np.log(close[i - period + j] / close[i - period + j - 1])
                count += 1
        if count > 0:
            mean_ret = 0.0
            for j in range(count):
                mean_ret += returns[j]
            mean_ret /= count
            variance = 0.0
            for j in range(count):
                variance += (returns[j] - mean_ret) ** 2
            variance /= count
            std = np.sqrt(variance) if variance > 0 else 0.0
            hv[i] = std * np.sqrt(trading_days) * 100
    return hv


@njit(parallel=False, fastmath=True)
def _calculate_hull_ma_numba(close: np.ndarray, period: int) -> np.ndarray:
    """Optimized Hull Moving Average calculation."""
    n = len(close)
    if n == 0 or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    hull = np.full(n, np.nan, dtype=np.float64)
    half_period = period // 2
    wma1 = np.full(n, np.nan, dtype=np.float64)
    wma2 = np.full(n, np.nan, dtype=np.float64)
    for i in range(half_period - 1, n):
        if np.any(np.isnan(close[i - half_period + 1 : i + 1])):
            continue
        weights_sum = 0.0
        data_sum = 0.0
        count = 0
        for j in range(half_period):
            weight = half_period - j
            weights_sum += weight
            data_sum += close[i - j] * weight
            count += 1
        if weights_sum != 0:
            wma1[i] = data_sum / weights_sum
    for i in range(period - 1, n):
        if np.any(np.isnan(close[i - period + 1 : i + 1])):
            continue
        weights_sum = 0.0
        data_sum = 0.0
        count = 0
        for j in range(period):
            weight = period - j
            weights_sum += weight
            data_sum += close[i - j] * weight
            count += 1
        if weights_sum != 0:
            wma2[i] = data_sum / weights_sum
    raw = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma1[i]) and not np.isnan(wma2[i]):
            raw[i] = 2 * wma1[i] - wma2[i]
    sqrt_period = int(np.sqrt(period))
    for i in range(sqrt_period - 1, n):
        if np.any(np.isnan(raw[i - sqrt_period + 1 : i + 1])):
            continue
        weights_sum = 0.0
        data_sum = 0.0
        count = 0
        for j in range(sqrt_period):
            weight = sqrt_period - j
            weights_sum += weight
            data_sum += raw[i - j] * weight
            count += 1
        if weights_sum != 0:
            hull[i] = data_sum / weights_sum
    return hull


@njit(parallel=False, fastmath=True)
def _calculate_ichimoku_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tenkan_period: int,
    kijun_period: int,
    senkou_span_b_period: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Optimized Ichimoku Cloud calculation."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or tenkan_period <= 0 or kijun_period <= 0 or senkou_span_b_period <= 0:
        empty = np.full(n, np.nan, dtype=np.float64)
        return empty, empty, empty, empty
    tenkan_sen = np.full(n, np.nan, dtype=np.float64)
    kijun_sen = np.full(n, np.nan, dtype=np.float64)
    senkou_span_a = np.full(n, np.nan, dtype=np.float64)
    senkou_span_b = np.full(n, np.nan, dtype=np.float64)
    for i in range(tenkan_period - 1, n):
        if np.any(np.isnan(high[i - tenkan_period + 1 : i + 1])) or np.any(np.isnan(low[i - tenkan_period + 1 : i + 1])):
            continue
        tenkan_high = np.max(high[i - tenkan_period + 1 : i + 1])
        tenkan_low = np.min(low[i - tenkan_period + 1 : i + 1])
        tenkan_sen[i] = (tenkan_high + tenkan_low) / 2
    for i in range(kijun_period - 1, n):
        if np.any(np.isnan(high[i - kijun_period + 1 : i + 1])) or np.any(np.isnan(low[i - kijun_period + 1 : i + 1])):
            continue
        kijun_high = np.max(high[i - kijun_period + 1 : i + 1])
        kijun_low = np.min(low[i - kijun_period + 1 : i + 1])
        kijun_sen[i] = (kijun_high + kijun_low) / 2
    for i in range(kijun_period - 1, n):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    for i in range(senkou_span_b_period - 1, n):
        if np.any(np.isnan(high[i - senkou_span_b_period + 1 : i + 1])) or np.any(np.isnan(low[i - senkou_span_b_period + 1 : i + 1])):
            continue
        senkou_high = np.max(high[i - senkou_span_b_period + 1 : i + 1])
        senkou_low = np.min(low[i - senkou_span_b_period + 1 : i + 1])
        senkou_span_b[i] = (senkou_high + senkou_low) / 2
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b


@njit(parallel=False, fastmath=True)
def _calculate_kama_numba(
    close: np.ndarray,
    er_period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
) -> np.ndarray:
    """Optimized Kaufman Adaptive Moving Average calculation."""
    n = len(close)
    if n == 0 or er_period <= 0 or fast_period <= 0 or slow_period <= 0 or er_period > n:
        return np.full(n, np.nan, dtype=np.float64)
    er = np.full(n, np.nan, dtype=np.float64)
    kama = np.full(n, np.nan, dtype=np.float64)
    direction = np.full(n - er_period, np.nan, dtype=np.float64)
    volatility = np.full(n, np.nan, dtype=np.float64)
    for i in range(er_period, n):
        if np.any(np.isnan(close[i - er_period : i + 1])):
            continue
        direction[i - er_period] = abs(close[i] - close[i - er_period])
        vol_sum = 0.0
        count = 0
        for j in range(i - er_period, i):
            if not np.isnan(close[j + 1]) and not np.isnan(close[j]):
                vol_sum += abs(close[j + 1] - close[j])
                count += 1
        volatility[i] = vol_sum / count if count > 0 else 0.0
        er[i] = direction[i - er_period] / volatility[i] if volatility[i] != 0 else 0.0
        if np.isnan(er[i]) or er[i] < 0 or er[i] > 1:
            er[i] = 0.0
    fast_alpha = 2.0 / (fast_period + 1)
    slow_alpha = 2.0 / (slow_period + 1)
    sc = np.full(n, np.nan, dtype=np.float64)
    for i in range(er_period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_alpha - slow_alpha) + slow_alpha) ** 2
            if sc[i] > 1.0 or sc[i] < 0.0:  # Clip extreme SC values
                sc[i] = 0.1
    if er_period < n and not np.isnan(close[er_period]):
        kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if not np.isnan(close[i]) and not np.isnan(kama[i - 1]) and not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    return kama


@njit(parallel=False, fastmath=True)
def _calculate_kdj_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 9,
    d_period: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized KDJ indicator calculation."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or k_period <= 0 or d_period <= 0 or k_period + d_period > n:
        empty = np.full(n, np.nan, dtype=np.float64)
        return empty, empty, empty
    rsv = np.full(n, np.nan, dtype=np.float64)
    k = np.full(n, np.nan, dtype=np.float64)
    d = np.full(n, np.nan, dtype=np.float64)
    j = np.full(n, np.nan, dtype=np.float64)
    for i in range(k_period - 1, n):
        if np.any(np.isnan(high[i - k_period + 1 : i + 1])) or np.any(np.isnan(low[i - k_period + 1 : i + 1])) or np.isnan(close[i]):
            continue
        high_window = high[i - k_period + 1 : i + 1]
        low_window = low[i - k_period + 1 : i + 1]
        highest_high = np.max(high_window)
        lowest_low = np.min(low_window)
        if highest_high != lowest_low:
            rsv[i] = (close[i] - lowest_low) / (highest_high - lowest_low) * 100
        else:
            rsv[i] = 50
    if k_period - 1 < n and not np.isnan(rsv[k_period - 1]):
        k[k_period - 1] = rsv[k_period - 1]
    for i in range(k_period, n):
        if not np.isnan(k[i - 1]) and not np.isnan(rsv[i]):
            k[i] = (2 / 3) * k[i - 1] + (1 / 3) * rsv[i]
    for i in range(k_period + d_period - 1, n):
        if np.any(np.isnan(k[i - d_period + 1 : i + 1])):
            continue
        sum_k = 0.0
        count = 0
        for j in range(i - d_period + 1, i + 1):
            sum_k += k[j]
            count += 1
        if count > 0:
            d[i] = sum_k / count
    for i in range(n):
        if not np.isnan(k[i]) and not np.isnan(d[i]):
            j[i] = 3 * k[i] - 2 * d[i]
    return k, d, j


@njit(parallel=False, fastmath=True)
def _calculate_keltner_channels_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ema_period: int,
    atr_period: int,
    multiplier: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized Keltner Channels calculation."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or ema_period <= 0 or atr_period <= 0 or max(ema_period, atr_period) > n:
        empty = np.full(n, np.nan, dtype=np.float64)
        return empty, empty, empty
    ema = np.full(n, np.nan, dtype=np.float64)
    multiplier_ema = 2.0 / (ema_period + 1)
    if n > 0 and not np.isnan(close[0]):
        ema[0] = close[0]
    for i in range(1, n):
        if not np.isnan(close[i]) and not np.isnan(ema[i - 1]):
            ema[i] = (close[i] - ema[i - 1]) * multiplier_ema + ema[i - 1]
    tr = np.full(n, np.nan, dtype=np.float64)
    atr = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i - 1]):
            continue
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    if atr_period < n:
        sum_tr = 0.0
        count = 0
        for i in range(1, atr_period + 1):
            if not np.isnan(tr[i]):
                sum_tr += tr[i]
                count += 1
        if count > 0:
            atr[atr_period] = sum_tr / count
        for i in range(atr_period + 1, n):
            if not np.isnan(tr[i]) and not np.isnan(atr[i - 1]):
                atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    for i in range(max(ema_period, atr_period) - 1, n):
        if not np.isnan(ema[i]) and not np.isnan(atr[i]):
            upper[i] = ema[i] + (multiplier * atr[i])
            lower[i] = ema[i] - (multiplier * atr[i])
    return upper, ema, lower


@njit(parallel=False, fastmath=True)
def _calculate_linear_regression_channel_numba(
    close: np.ndarray, period: int, deviations: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized Linear Regression Channel calculation."""
    n = len(close)
    if n == 0 or period <= 2 or period > n:
        empty = np.full(n, np.nan, dtype=np.float64)
        return empty, empty, empty
    middle = np.full(n, np.nan, dtype=np.float64)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if np.any(np.isnan(close[i - period + 1 : i + 1])):
            continue
        y = close[i - period + 1 : i + 1]
        x = np.arange(period, dtype=np.float64)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = 0.0
        denominator = 0.0
        for j in range(period):
            dx = x[j] - x_mean
            numerator += dx * (y[j] - y_mean)
            denominator += dx * dx
        if denominator != 0:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            predict = slope * (period - 1) + intercept
            middle[i] = predict
            y_pred = np.zeros(period, dtype=np.float64)
            for j in range(period):
                y_pred[j] = x[j] * slope + intercept
            std_error = 0.0
            for j in range(period):
                std_error += (y[j] - y_pred[j]) ** 2
            std_error = np.sqrt(std_error / (period - 2)) if period > 2 else 0.0
            upper[i] = predict + deviations * std_error
            lower[i] = predict - deviations * std_error
    return upper, middle, lower


@njit(parallel=False, fastmath=True)
def _calculate_ma_momentum_numba(
    close: np.ndarray, ma_period: int = 10, momentum_period: int = 10
) -> np.ndarray:
    """Optimized Moving Average Momentum calculation."""
    n = len(close)
    if n == 0 or ma_period <= 0 or momentum_period <= 0 or ma_period + momentum_period > n:
        return np.full(n, np.nan, dtype=np.float64)
    ma = np.full(n, np.nan, dtype=np.float64)
    for i in range(ma_period - 1, n):
        if np.any(np.isnan(close[i - ma_period + 1 : i + 1])):
            continue
        sum_close = 0.0
        count = 0
        for j in range(i - ma_period + 1, i + 1):
            sum_close += close[j]
            count += 1
        ma[i] = sum_close / count
    momentum = np.full(n, np.nan, dtype=np.float64)
    for i in range(ma_period + momentum_period - 1, n):
        if not np.isnan(ma[i]) and not np.isnan(ma[i - momentum_period]) and ma[i - momentum_period] != 0:
            momentum[i] = 100 * (ma[i] - ma[i - momentum_period]) / ma[i - momentum_period]
    return momentum


@njit(parallel=False, fastmath=True)
def _calculate_macd_numba(
    prices: np.ndarray, fast_period: int, slow_period: int, signal_period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized MACD calculation."""
    n = len(prices)
    if n == 0 or fast_period <= 0 or slow_period <= 0 or signal_period <= 0 or max(fast_period, slow_period) > n:
        empty = np.full(n, np.nan, dtype=np.float64)
        return empty, empty, empty
    macd_line = np.full(n, np.nan, dtype=np.float64)
    signal_line = np.full(n, np.nan, dtype=np.float64)
    histogram = np.full(n, np.nan, dtype=np.float64)
    fast_multiplier = 2.0 / (fast_period + 1)
    slow_multiplier = 2.0 / (slow_period + 1)
    signal_multiplier = 2.0 / (signal_period + 1)
    fast_ema = prices[0] if n > 0 and not np.isnan(prices[0]) else np.nan
    slow_ema = prices[0] if n > 0 and not np.isnan(prices[0]) else np.nan
    for i in range(1, n):
        if not np.isnan(prices[i]) and not np.isnan(fast_ema) and not np.isnan(slow_ema):
            fast_ema = (prices[i] - fast_ema) * fast_multiplier + fast_ema
            slow_ema = (prices[i] - slow_ema) * slow_multiplier + slow_ema
            macd_line[i] = fast_ema - slow_ema
    signal_ema = macd_line[slow_period - 1] if slow_period - 1 < n and not np.isnan(macd_line[slow_period - 1]) else np.nan
    for i in range(slow_period, n):
        if not np.isnan(macd_line[i]) and not np.isnan(signal_ema):
            signal_ema = (macd_line[i] - signal_ema) * signal_multiplier + signal_ema
            signal_line[i] = signal_ema
            histogram[i] = macd_line[i] - signal_ema
    return macd_line, signal_line, histogram


@njit(parallel=False, fastmath=True)
def _calculate_mass_index_numba(
    high: np.ndarray, low: np.ndarray, ema_period: int = 9, sum_period: int = 25
) -> np.ndarray:
    """Optimized Mass Index calculation."""
    n = len(high)
    if n == 0 or len(low) != n or ema_period <= 0 or sum_period <= 0 or ema_period + sum_period > n:
        return np.full(n, np.nan, dtype=np.float64)
    mi = np.full(n, np.nan, dtype=np.float64)
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(high[i]) and not np.isnan(low[i]):
            diff[i] = high[i] - low[i]
    ema1 = np.full(n, np.nan, dtype=np.float64)
    ema2 = np.full(n, np.nan, dtype=np.float64)
    alpha1 = 2.0 / (ema_period + 1)
    if n > 0 and not np.isnan(diff[0]):
        ema1[0] = diff[0]
        ema2[0] = diff[0]
    for i in range(1, n):
        if not np.isnan(diff[i]) and not np.isnan(ema1[i - 1]):
            ema1[i] = (diff[i] - ema1[i - 1]) * alpha1 + ema1[i - 1]
        if not np.isnan(ema1[i]) and not np.isnan(ema2[i - 1]):
            ema2[i] = (ema1[i] - ema2[i - 1]) * alpha1 + ema2[i - 1]
    ema_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(ema1[i]) and ema2[i] != 0:
            ema_ratio[i] = ema1[i] / ema2[i]
    for i in range(sum_period - 1, n):
        if np.any(np.isnan(ema_ratio[i - sum_period + 1 : i + 1])):
            continue
        sum_ratio = 0.0
        count = 0
        for j in range(i - sum_period + 1, i + 1):
            sum_ratio += ema_ratio[j]
            count += 1
        if count > 0:
            mi[i] = sum_ratio
    return mi


@njit(parallel=False, fastmath=True)
def _calculate_mfi_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int,
) -> np.ndarray:
    """Optimized Money Flow Index calculation."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or len(volume) != n or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    typical_price = np.full(n, np.nan, dtype=np.float64)
    money_flow = np.full(n, np.nan, dtype=np.float64)
    positive_flow = np.zeros(n, dtype=np.float64)  # Initialize to 0
    negative_flow = np.zeros(n, dtype=np.float64)  # Initialize to 0
    for i in range(n):
        if not np.isnan(high[i]) and not np.isnan(low[i]) and not np.isnan(close[i]):
            typical_price[i] = (high[i] + low[i] + close[i]) / 3
    for i in range(n):
        if not np.isnan(typical_price[i]) and not np.isnan(volume[i]):
            money_flow[i] = typical_price[i] * volume[i]
    for i in range(1, n):
        if not np.isnan(typical_price[i]) and not np.isnan(typical_price[i - 1]) and not np.isnan(money_flow[i]):
            if typical_price[i] > typical_price[i - 1]:
                positive_flow[i] = money_flow[i]
            elif typical_price[i] < typical_price[i - 1]:
                negative_flow[i] = money_flow[i]
    mfi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        positive_sum = 0.0
        negative_sum = 0.0
        for j in range(i - period + 1, i + 1):
            positive_sum += positive_flow[j]
            negative_sum += negative_flow[j]
        if positive_sum == 0 and negative_sum == 0:
            mfi[i] = 50.0  # Neutral MFI when no price movement
        elif negative_sum == 0:
            mfi[i] = 100.0
        else:
            money_ratio = positive_sum / negative_sum
            mfi[i] = 100 - (100 / (1 + money_ratio))
    return mfi


@njit(parallel=False, fastmath=True)
def _calculate_momentum_index_numba(
    close: np.ndarray, period: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized Momentum Index calculation."""
    n = len(close)
    if n == 0 or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)
    positive_sum = np.zeros(n, dtype=np.float64)  # Initialize to 0
    negative_sum = np.zeros(n, dtype=np.float64)  # Initialize to 0
    momentum_index = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - 1]):
            change = close[i] - close[i - 1]
            if change > 0:
                positive_sum[i] = positive_sum[i - 1] + change
                negative_sum[i] = negative_sum[i - 1]
            elif change < 0:
                negative_sum[i] = negative_sum[i - 1] + abs(change)
                positive_sum[i] = positive_sum[i - 1]
            else:
                positive_sum[i] = positive_sum[i - 1]
                negative_sum[i] = negative_sum[i - 1]
    for i in range(period, n):
        pos_period = positive_sum[i] - positive_sum[i - period]
        neg_period = negative_sum[i] - negative_sum[i - period]
        total = pos_period + neg_period
        if total > 0:
            momentum_index[i] = 100 * (pos_period / total)
        elif total == 0:
            momentum_index[i] = 50.0  # Neutral when no price movement
    negative_index = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(momentum_index[i]):
            negative_index[i] = 100 - momentum_index[i]
    return momentum_index, negative_index


@njit(parallel=False, fastmath=True)
def _calculate_pgo_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 21
) -> np.ndarray:
    """Optimized Pretty Good Oscillator calculation."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    pgo = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if np.any(np.isnan(high[i - period + 1 : i + 1])) or np.any(np.isnan(low[i - period + 1 : i + 1])) or np.isnan(close[i]):
            continue
        highest_high = np.max(high[i - period + 1 : i + 1])
        lowest_low = np.min(low[i - period + 1 : i + 1])
        if highest_high != lowest_low:
            pgo[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low) - 50
    return pgo


@njit(parallel=False, fastmath=True)
def _calculate_pivot_points_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Optimized Pivot Points calculation."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n:
        empty = np.full(n, np.nan, dtype=np.float64)
        return empty, empty, empty, empty, empty
    pp = np.full(n, np.nan, dtype=np.float64)
    r1 = np.full(n, np.nan, dtype=np.float64)
    r2 = np.full(n, np.nan, dtype=np.float64)
    s1 = np.full(n, np.nan, dtype=np.float64)
    s2 = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if np.isnan(high[i - 1]) or np.isnan(low[i - 1]) or np.isnan(close[i - 1]):
            continue
        pp[i] = (high[i - 1] + low[i - 1] + close[i - 1]) / 3
        r1[i] = 2 * pp[i] - low[i - 1]
        r2[i] = pp[i] + (high[i - 1] - low[i - 1])
        s1[i] = 2 * pp[i] - high[i - 1]
        s2[i] = pp[i] - (high[i - 1] - low[i - 1])
    return pp, r1, r2, s1, s2


@njit(parallel=False, fastmath=True)
def _calculate_psl_numba(close: np.ndarray, period: int = 12) -> np.ndarray:
    """Optimized Psychological Line calculation."""
    n = len(close)
    if n == 0 or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    psl = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if np.any(np.isnan(close[i - period : i + 1])):
            continue
        up_count = 0
        for j in range(i - period + 1, i + 1):
            if close[j] > close[j - 1]:
                up_count += 1
        psl[i] = (up_count / period) * 100
    return psl


@njit(parallel=False, fastmath=True)
def _calculate_pvo_numba(
    volume: np.ndarray, short_period: int = 12, long_period: int = 26
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized Percentage Volume Oscillator calculation."""
    n = len(volume)
    if n == 0 or short_period <= 0 or long_period <= 0 or max(short_period, long_period) > n:
        empty = np.full(n, np.nan, dtype=np.float64)
        return empty, empty
    short_ema = np.full(n, np.nan, dtype=np.float64)
    long_ema = np.full(n, np.nan, dtype=np.float64)
    pvo = np.full(n, np.nan, dtype=np.float64)
    signal = np.full(n, np.nan, dtype=np.float64)
    if n > 0 and not np.isnan(volume[0]):
        short_ema[0] = volume[0]
        long_ema[0] = volume[0]
    short_mult = 2.0 / (short_period + 1)
    long_mult = 2.0 / (long_period + 1)
    signal_mult = 2.0 / (9 + 1)  # Signal line uses 9-period EMA
    for i in range(1, n):
        if not np.isnan(volume[i]):
            short_ema[i] = (volume[i] - short_ema[i - 1]) * short_mult + short_ema[i - 1]
            long_ema[i] = (volume[i] - long_ema[i - 1]) * long_mult + long_ema[i - 1]
        if not np.isnan(short_ema[i]) and not np.isnan(long_ema[i]) and long_ema[i] != 0:
            pvo[i] = ((short_ema[i] - long_ema[i]) / long_ema[i]) * 100
    signal_ema = pvo[long_period - 1] if long_period - 1 < n and not np.isnan(pvo[long_period - 1]) else np.nan
    for i in range(long_period, n):
        if not np.isnan(pvo[i]) and not np.isnan(signal_ema):
            signal_ema = (pvo[i] - signal_ema) * signal_mult + signal_ema
            signal[i] = signal_ema
    return pvo, signal


@njit(parallel=False, fastmath=True)
def _calculate_qstick_numba(
    close: np.ndarray, open_: np.ndarray, period: int
) -> np.ndarray:
    """Optimized QStick calculation."""
    n = len(close)
    if n == 0 or len(open_) != n or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    qstick = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if np.any(np.isnan(close[i - period + 1 : i + 1])) or np.any(np.isnan(open_[i - period + 1 : i + 1])):
            continue
        sum_diff = 0.0
        count = 0
        for j in range(i - period + 1, i + 1):
            sum_diff += close[j] - open_[j]
            count += 1
        if count > 0:
            qstick[i] = sum_diff / count
    return qstick


@njit(parallel=False, fastmath=True)
def _calculate_rainbow_numba(
    prices: np.ndarray, periods: np.ndarray
) -> np.ndarray:
    """Optimized Rainbow Oscillator calculation."""
    n = len(prices)
    if n == 0 or len(periods) == 0 or np.any(periods <= 0) or np.any(periods > n):
        return np.zeros((0, n), dtype=np.float64)
    n_periods = len(periods)
    sma_lines = np.full((n_periods, n), np.nan, dtype=np.float64)
    for p_idx in range(n_periods):
        period = periods[p_idx]
        for i in range(period - 1, n):
            if np.any(np.isnan(prices[i - period + 1 : i + 1])):
                continue
            sum_prices = 0.0
            count = 0
            for j in range(i - period + 1, i + 1):
                sum_prices += prices[j]
                count += 1
            if count > 0:
                sma_lines[p_idx, i] = sum_prices / count
    return sma_lines


@njit(parallel=False, fastmath=True)
def _calculate_roc_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """Optimized Rate of Change calculation."""
    n = len(prices)
    if n == 0 or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(prices[i]) and not np.isnan(prices[i - period]) and prices[i - period] != 0:
            roc[i] = ((prices[i] - prices[i - period]) / prices[i - period]) * 100
    return roc


@njit(parallel=False, fastmath=True)
def _calculate_rsi_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """Optimized Relative Strength Index calculation."""
    n = len(prices)
    if n == 0 or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    gains = np.zeros(n, dtype=np.float64)  # Initialize to 0
    losses = np.zeros(n, dtype=np.float64)  # Initialize to 0
    for i in range(1, n):
        if not np.isnan(prices[i]) and not np.isnan(prices[i - 1]):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains[i] = change
            elif change < 0:
                losses[i] = abs(change)
    avg_gain = np.full(n, np.nan, dtype=np.float64)
    avg_loss = np.full(n, np.nan, dtype=np.float64)
    if period < n:
        sum_gain = 0.0
        sum_loss = 0.0
        count = 0
        for i in range(1, period + 1):
            sum_gain += gains[i]
            sum_loss += losses[i]
            count += 1
        if count > 0:
            avg_gain[period] = sum_gain / count
            avg_loss[period] = sum_loss / count
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(avg_gain[i]) and not np.isnan(avg_loss[i]):
            if avg_gain[i] == 0 and avg_loss[i] == 0:
                rsi[i] = 50.0  # Neutral RSI when no price movement
            elif avg_loss[i] == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))
    return rsi


@njit(parallel=False, fastmath=True)
def _calculate_rvi_numba(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """Optimized Relative Vigor Index calculation."""
    n = len(close)
    if n == 0 or len(open_) != n or len(high) != n or len(low) != n or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    rvi = np.full(n, np.nan, dtype=np.float64)
    num = np.zeros(n, dtype=np.float64)  # Initialize to 0
    denom = np.zeros(n, dtype=np.float64)  # Initialize to 0
    for i in range(n):
        if not np.isnan(open_[i]) and not np.isnan(high[i]) and not np.isnan(low[i]) and not np.isnan(close[i]):
            num[i] = (close[i] - open_[i]) / 2 + (close[i - 1] - open_[i - 1]) / 2 if i > 0 else 0
            denom[i] = (high[i] - low[i]) / 2 + (high[i - 1] - low[i - 1]) / 2 if i > 0 else 0
    rvi_num = np.full(n, np.nan, dtype=np.float64)
    rvi_denom = np.full(n, np.nan, dtype=np.float64)
    if period < n:
        sum_num = 0.0
        sum_denom = 0.0
        count = 0
        for i in range(1, period + 1):
            sum_num += num[i]
            sum_denom += denom[i]
            count += 1
        if count > 0:
            rvi_num[period] = sum_num / count
            rvi_denom[period] = sum_denom / count if sum_denom != 0 else np.nan
        for i in range(period + 1, n):
            rvi_num[i] = (rvi_num[i - 1] * (period - 1) + num[i]) / period
            rvi_denom[i] = (rvi_denom[i - 1] * (period - 1) + denom[i]) / period
    for i in range(period, n):
        if not np.isnan(rvi_num[i]) and not np.isnan(rvi_denom[i]) and rvi_denom[i] != 0:
            rvi[i] = rvi_num[i] / rvi_denom[i]
        elif rvi_num[i] == 0 and rvi_denom[i] == 0:
            rvi[i] = 0.0  # Neutral when no movement
    return rvi


@njit(parallel=False, fastmath=True)
def _calculate_sma_numba(prices: np.ndarray, window: int) -> np.ndarray:
    """Optimized Simple Moving Average calculation."""
    n = len(prices)
    if n == 0 or window <= 0 or window > n:
        return np.full(n, np.nan, dtype=np.float64)
    sma = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        if np.any(np.isnan(prices[i - window + 1 : i + 1])):
            continue
        sum_prices = 0.0
        count = 0
        for j in range(i - window + 1, i + 1):
            sum_prices += prices[j]
            count += 1
        sma[i] = sum_prices / count
    return sma


@njit(parallel=False, fastmath=True)
def _calculate_stochastic_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int, d_period: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized Stochastic Oscillator calculation."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or k_period <= 0 or d_period <= 0 or k_period + d_period > n:
        return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)
    k = np.full(n, np.nan, dtype=np.float64)
    d = np.full(n, np.nan, dtype=np.float64)
    for i in range(k_period - 1, n):
        if np.any(np.isnan(high[i - k_period + 1 : i + 1])) or np.any(np.isnan(low[i - k_period + 1 : i + 1])) or np.isnan(close[i]):
            continue
        high_window = high[i - k_period + 1 : i + 1]
        low_window = low[i - k_period + 1 : i + 1]
        highest_high = np.max(high_window)
        lowest_low = np.min(low_window)
        if highest_high != lowest_low:
            k[i] = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
    for i in range(k_period + d_period - 2, n):
        if np.any(np.isnan(k[i - d_period + 1 : i + 1])):
            continue
        sum_k = 0.0
        count = 0
        for j in range(i - d_period + 1, i + 1):
            sum_k += k[j]
            count += 1
        if count > 0:
            d[i] = sum_k / count
    return k, d


@njit(parallel=False, fastmath=True)
def _calculate_supertrend_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    multiplier: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized Supertrend calculation."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or period <= 0 or period > n:
        empty = np.full(n, np.nan, dtype=np.float64)
        return empty, empty
    tr = np.full(n, np.nan, dtype=np.float64)
    atr = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i - 1]):
            continue
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    if period < n:
        sum_tr = 0.0
        count = 0
        for i in range(1, period + 1):
            if not np.isnan(tr[i]):
                sum_tr += tr[i]
                count += 1
        if count > 0:
            atr[period] = sum_tr / count
        for i in range(period + 1, n):
            if not np.isnan(tr[i]) and not np.isnan(atr[i - 1]):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    upper_band = np.full(n, np.nan, dtype=np.float64)
    lower_band = np.full(n, np.nan, dtype=np.float64)
    supertrend = np.full(n, np.nan, dtype=np.float64)
    direction = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if np.any(np.isnan(high[i - period + 1 : i + 1])) or np.any(np.isnan(low[i - period + 1 : i + 1])) or np.isnan(close[i]) or np.isnan(atr[i]):
            continue
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + (multiplier * atr[i])
        lower_band[i] = mid - (multiplier * atr[i])
        if i == period:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            prev_close = close[i - 1]
            prev_supertrend = supertrend[i - 1]
            prev_direction = direction[i - 1]
            if not np.isnan(prev_close) and not np.isnan(prev_supertrend) and not np.isnan(prev_direction):
                if prev_direction == 1:
                    upper_band[i] = min(upper_band[i], upper_band[i - 1]) if not np.isnan(upper_band[i - 1]) else upper_band[i]
                    if prev_close > prev_supertrend:
                        supertrend[i] = lower_band[i]
                        direction[i] = 1
                    else:
                        supertrend[i] = upper_band[i]
                        direction[i] = -1
                else:
                    lower_band[i] = max(lower_band[i], lower_band[i - 1]) if not np.isnan(lower_band[i - 1]) else lower_band[i]
                    if prev_close < prev_supertrend:
                        supertrend[i] = upper_band[i]
                        direction[i] = -1
                    else:
                        supertrend[i] = lower_band[i]
                        direction[i] = 1
    return supertrend, direction


@njit(parallel=False, fastmath=True)
def _calculate_trix_numba(close: np.ndarray, period: int) -> np.ndarray:
    """Optimized TRIX calculation."""
    n = len(close)
    if n == 0 or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    ema1 = np.full(n, np.nan, dtype=np.float64)
    ema2 = np.full(n, np.nan, dtype=np.float64)
    ema3 = np.full(n, np.nan, dtype=np.float64)
    trix = np.full(n, np.nan, dtype=np.float64)
    alpha = 2.0 / (period + 1)
    if n > 0 and not np.isnan(close[0]):
        ema1[0] = close[0]
    for i in range(1, n):
        if not np.isnan(close[i]) and not np.isnan(ema1[i - 1]):
            ema1[i] = (close[i] - ema1[i - 1]) * alpha + ema1[i - 1]
    if not np.isnan(ema1[period - 1]):
        ema2[period - 1] = ema1[period - 1]
    for i in range(period, n):
        if not np.isnan(ema1[i]) and not np.isnan(ema2[i - 1]):
            ema2[i] = (ema1[i] - ema2[i - 1]) * alpha + ema2[i - 1]
    if not np.isnan(ema2[period - 1]):
        ema3[period - 1] = ema2[period - 1]
    for i in range(period, n):
        if not np.isnan(ema2[i]) and not np.isnan(ema3[i - 1]):
            ema3[i] = (ema2[i] - ema3[i - 1]) * alpha + ema3[i - 1]
    for i in range(1, n):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i - 1]) and ema3[i - 1] != 0:
            trix[i] = ((ema3[i] - ema3[i - 1]) / ema3[i - 1]) * 100
    return trix


@njit(parallel=False, fastmath=True)
def _calculate_typical_price_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Optimized Typical Price calculation."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n:
        return np.full(n, np.nan, dtype=np.float64)
    tp = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(high[i]) and not np.isnan(low[i]) and not np.isnan(close[i]):
            tp[i] = (high[i] + low[i] + close[i]) / 3
    return tp


@njit(parallel=False, fastmath=True)
def _calculate_ultimate_oscillator_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period1: int,
    period2: int,
    period3: int,
    weight1: float,
    weight2: float,
    weight3: float,
) -> np.ndarray:
    """Optimized Ultimate Oscillator calculation."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or period1 <= 0 or period2 <= 0 or period3 <= 0 or max(period1, period2, period3) > n:
        return np.full(n, np.nan, dtype=np.float64)
    bp = np.full(n, np.nan, dtype=np.float64)
    tr = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]) or np.isnan(close[i - 1]):
            continue
        bp[i] = close[i] - min(low[i], close[i - 1])
        tr[i] = max(high[i], close[i - 1]) - min(low[i], close[i - 1])
    avg_bp1 = np.full(n, np.nan, dtype=np.float64)
    avg_tr1 = np.full(n, np.nan, dtype=np.float64)
    avg_bp2 = np.full(n, np.nan, dtype=np.float64)
    avg_tr2 = np.full(n, np.nan, dtype=np.float64)
    avg_bp3 = np.full(n, np.nan, dtype=np.float64)
    avg_tr3 = np.full(n, np.nan, dtype=np.float64)
    for i in range(period1, n):
        if np.any(np.isnan(bp[i - period1 + 1 : i + 1])) or np.any(np.isnan(tr[i - period1 + 1 : i + 1])):
            continue
        sum_bp = 0.0
        sum_tr = 0.0
        count = 0
        for j in range(i - period1 + 1, i + 1):
            sum_bp += bp[j]
            sum_tr += tr[j]
            count += 1
        if count > 0:
            avg_bp1[i] = sum_bp
            avg_tr1[i] = sum_tr
    for i in range(period2, n):
        if np.any(np.isnan(bp[i - period2 + 1 : i + 1])) or np.any(np.isnan(tr[i - period2 + 1 : i + 1])):
            continue
        sum_bp = 0.0
        sum_tr = 0.0
        count = 0
        for j in range(i - period2 + 1, i + 1):
            sum_bp += bp[j]
            sum_tr += tr[j]
            count += 1
        if count > 0:
            avg_bp2[i] = sum_bp
            avg_tr2[i] = sum_tr
    for i in range(period3, n):
        if np.any(np.isnan(bp[i - period3 + 1 : i + 1])) or np.any(np.isnan(tr[i - period3 + 1 : i + 1])):
            continue
        sum_bp = 0.0
        sum_tr = 0.0
        count = 0
        for j in range(i - period3 + 1, i + 1):
            sum_bp += bp[j]
            sum_tr += tr[j]
            count += 1
        if count > 0:
            avg_bp3[i] = sum_bp
            avg_tr3[i] = sum_tr
    uo = np.full(n, np.nan, dtype=np.float64)
    max_period = max(period1, period2, period3)
    for i in range(max_period, n):
        if (
            not np.isnan(avg_bp1[i])
            and not np.isnan(avg_tr1[i])
            and not np.isnan(avg_bp2[i])
            and not np.isnan(avg_tr2[i])
            and not np.isnan(avg_bp3[i])
            and not np.isnan(avg_tr3[i])
            and (avg_tr1[i] + avg_tr2[i] + avg_tr3[i]) != 0
        ):
            uo[i] = (
                100
                * (
                    (weight1 * avg_bp1[i] / avg_tr1[i])
                    + (weight2 * avg_bp2[i] / avg_tr2[i])
                    + (weight3 * avg_bp3[i] / avg_tr3[i])
                )
                / (weight1 + weight2 + weight3)
            )
    return uo


@njit(parallel=False, fastmath=True)
def _calculate_volume_indicators_numba(
    close: np.ndarray, volume: np.ndarray, period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized Volume Indicators calculation."""
    n = len(close)
    if n == 0 or len(volume) != n or period <= 0 or period > n:
        empty = np.full(n, np.nan, dtype=np.float64)
        return empty, empty, empty
    vol_sma = np.full(n, np.nan, dtype=np.float64)
    force_index = np.full(n, np.nan, dtype=np.float64)
    vpt = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if np.any(np.isnan(volume[i - period + 1 : i + 1])):
            continue
        sum_vol = 0.0
        count = 0
        for j in range(i - period + 1, i + 1):
            sum_vol += volume[j]
            count += 1
        if count > 0:
            vol_sma[i] = sum_vol / count
    for i in range(1, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - 1]) and not np.isnan(volume[i]):
            force_index[i] = (close[i] - close[i - 1]) * volume[i]
    for i in range(1, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - 1]) and not np.isnan(volume[i]) and close[i - 1] != 0:
            vpt[i] = (vpt[i - 1] if i > 1 and not np.isnan(vpt[i - 1]) else 0) + volume[i] * (close[i] - close[i - 1]) / close[i - 1]
    return vol_sma, force_index, vpt


@njit(parallel=False, fastmath=True)
def _calculate_vwap_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int
) -> np.ndarray:
    """Optimized Volume Weighted Average Price calculation."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or len(volume) != n or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    vwap = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if np.any(np.isnan(high[i - period + 1 : i + 1])) or np.any(np.isnan(low[i - period + 1 : i + 1])) or np.any(np.isnan(close[i - period + 1 : i + 1])) or np.any(np.isnan(volume[i - period + 1 : i + 1])):
            continue
        sum_price_vol = 0.0
        sum_vol = 0.0
        count = 0
        for j in range(i - period + 1, i + 1):
            typical_price = (high[j] + low[j] + close[j]) / 3
            sum_price_vol += typical_price * volume[j]
            sum_vol += volume[j]
            count += 1
        if sum_vol != 0:
            vwap[i] = sum_price_vol / sum_vol
    return vwap


@njit(parallel=False, fastmath=True)
def _calculate_weighted_close_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Optimized Weighted Close Price calculation."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n:
        return np.full(n, np.nan, dtype=np.float64)
    wc = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(high[i]) and not np.isnan(low[i]) and not np.isnan(close[i]):
            wc[i] = (high[i] + low[i] + close[i] * 2) / 4
    return wc


@njit(parallel=False, fastmath=True)
def _calculate_willr_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """Optimized Williams %R calculation."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    willr = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if np.any(np.isnan(high[i - period + 1 : i + 1])) or np.any(np.isnan(low[i - period + 1 : i + 1])) or np.isnan(close[i]):
            continue
        highest_high = np.max(high[i - period + 1 : i + 1])
        lowest_low = np.min(low[i - period + 1 : i + 1])
        if highest_high != lowest_low:
            willr[i] = ((highest_high - close[i]) / (highest_high - lowest_low)) * -100
    return willr