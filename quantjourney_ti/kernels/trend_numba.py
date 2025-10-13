"""
Trend kernels
-------------

Numba-accelerated implementations for trend-oriented indicators.
Functions return float64 arrays aligned to the input length and emit NaNs
for warm-up regions and invalid slices. All functions are pure and
side-effect free.
"""
import numpy as np
from typing import Tuple
from .._indicator_kernels import njit as njit  # cached njit


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
def _calculate_dema_numba(close: np.ndarray, period: int) -> np.ndarray:
    """Double EMA: DEMA = 2*EMA1 - EMA2.

    Args:
        close: price array
        period: EMA window
    Returns:
        Float64 array of DEMA values with NaNs for warm-up.
    """
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
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(ema1[i]) and not np.isnan(ema2[i]):
            out[i] = 2 * ema1[i] - ema2[i]
    return out


@njit(parallel=False, fastmath=True)
def _calculate_hull_ma_numba(close: np.ndarray, period: int) -> np.ndarray:
    """Hull Moving Average (HMA) using WMA cascade.

    Returns a smoother, faster-reacting moving average.
    """
    n = len(close)
    if n == 0 or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    hma = np.full(n, np.nan, dtype=np.float64)
    # WMA helper
    def wma(arr: np.ndarray, p: int) -> np.ndarray:
        out = np.full(n, np.nan, dtype=np.float64)
        if p <= 0 or p > n:
            return out
        weight_sum = p * (p + 1) / 2.0
        for i in range(p - 1, n):
            s = 0.0
            w = 1
            valid = True
            for j in range(i - p + 1, i + 1):
                if np.isnan(arr[j]):
                    valid = False
                    break
                s += arr[j] * w
                w += 1
            if valid:
                out[i] = s / weight_sum
        return out
    half = period // 2
    sqrt_p = int(np.sqrt(period)) if period > 0 else 0
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    hma = wma(diff, sqrt_p)
    return hma


@njit(parallel=False, fastmath=True)
def _calculate_alma_numba(
    close: np.ndarray, window: int, sigma: float = 6.0, offset: float = 0.85
) -> np.ndarray:
    """Arnaud Legoux Moving Average (ALMA) with Gaussian weights."""
    n = len(close)
    if n == 0 or window <= 0 or window > n:
        return np.full(n, np.nan, dtype=np.float64)
    alma = np.full(n, np.nan, dtype=np.float64)
    m = offset * (window - 1)
    s = window / sigma
    weights = np.zeros(window, dtype=np.float64)
    wsum = 0.0
    for i in range(window):
        weights[i] = np.exp(-((i - m) ** 2) / (2 * s * s))
        wsum += weights[i]
    if wsum == 0:
        return alma
    for i in range(window):
        weights[i] /= wsum
    for i in range(window - 1, n):
        if np.any(np.isnan(close[i - window + 1 : i + 1])):
            continue
        acc = 0.0
        for j in range(window):
            acc += close[i - window + 1 + j] * weights[j]
        alma[i] = acc
    return alma
@njit(parallel=False, fastmath=True)
def _calculate_kama_numba(
    close: np.ndarray, er_period: int = 10, fast_period: int = 2, slow_period: int = 30
) -> np.ndarray:
    n = len(close)
    if n == 0 or er_period <= 0 or fast_period <= 0 or slow_period <= 0 or er_period > n:
        return np.full(n, np.nan, dtype=np.float64)
    er = np.zeros(n, dtype=np.float64)
    kama = np.full(n, np.nan, dtype=np.float64)
    # Direction and volatility
    for i in range(er_period, n):
        if np.isnan(close[i]) or np.isnan(close[i - er_period]):
            continue
        direction = abs(close[i] - close[i - er_period])
        vol = 0.0
        valid = True
        for j in range(i - er_period + 1, i + 1):
            if np.isnan(close[j]) or np.isnan(close[j - 1]):
                valid = False
                break
            vol += abs(close[j] - close[j - 1])
        if not valid or vol == 0:
            er[i] = 0.0
        else:
            er[i] = direction / vol
    fast_alpha = 2.0 / (fast_period + 1)
    slow_alpha = 2.0 / (slow_period + 1)
    # Initialize KAMA
    if not np.isnan(close[er_period]):
        kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_alpha - slow_alpha) + slow_alpha)
        sc = sc * sc
        if not np.isnan(kama[i - 1]) and not np.isnan(close[i]):
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    return kama


# Ichimoku Cloud
@njit(parallel=False, fastmath=True)
def _calculate_ichimoku_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tenkan_period: int,
    kijun_period: int,
    senkou_span_b_period: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    if (
        n == 0
        or len(high) != n
        or len(low) != n
        or tenkan_period <= 0
        or kijun_period <= 0
        or senkou_span_b_period <= 0
    ):
        empty = np.full(n, np.nan, dtype=np.float64)
        return empty, empty, empty, empty
    tenkan_sen = np.full(n, np.nan, dtype=np.float64)
    kijun_sen = np.full(n, np.nan, dtype=np.float64)
    senkou_span_a = np.full(n, np.nan, dtype=np.float64)
    senkou_span_b = np.full(n, np.nan, dtype=np.float64)
    # Tenkan
    for i in range(tenkan_period - 1, n):
        if (
            np.any(np.isnan(high[i - tenkan_period + 1 : i + 1]))
            or np.any(np.isnan(low[i - tenkan_period + 1 : i + 1]))
        ):
            continue
        tenkan_high = np.max(high[i - tenkan_period + 1 : i + 1])
        tenkan_low = np.min(low[i - tenkan_period + 1 : i + 1])
        tenkan_sen[i] = (tenkan_high + tenkan_low) / 2
    # Kijun
    for i in range(kijun_period - 1, n):
        if (
            np.any(np.isnan(high[i - kijun_period + 1 : i + 1]))
            or np.any(np.isnan(low[i - kijun_period + 1 : i + 1]))
        ):
            continue
        kijun_high = np.max(high[i - kijun_period + 1 : i + 1])
        kijun_low = np.min(low[i - kijun_period + 1 : i + 1])
        kijun_sen[i] = (kijun_high + kijun_low) / 2
    # Senkou A and B
    for i in range(kijun_period - 1, n):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    for i in range(senkou_span_b_period - 1, n):
        if (
            np.any(np.isnan(high[i - senkou_span_b_period + 1 : i + 1]))
            or np.any(np.isnan(low[i - senkou_span_b_period + 1 : i + 1]))
        ):
            continue
        senkou_high = np.max(high[i - senkou_span_b_period + 1 : i + 1])
        senkou_low = np.min(low[i - senkou_span_b_period + 1 : i + 1])
        senkou_span_b[i] = (senkou_high + senkou_low) / 2
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b


@njit(parallel=False, fastmath=True)
def _calculate_heiken_ashi_numba(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Heiken Ashi candle components."""
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
            ha_close[i] = (open_[i] + high[i] + low[i] + close[i]) / 4.0
    if not np.isnan(open_[0]) and not np.isnan(close[0]):
        ha_open[0] = (open_[0] + close[0]) / 2.0
    for i in range(1, n):
        if not np.isnan(ha_open[i - 1]) and not np.isnan(ha_close[i - 1]):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        if not np.isnan(high[i]) and not np.isnan(ha_open[i]) and not np.isnan(ha_close[i]):
            ha_high[i] = max(high[i], ha_open[i], ha_close[i])
        if not np.isnan(low[i]) and not np.isnan(ha_open[i]) and not np.isnan(ha_close[i]):
            ha_low[i] = min(low[i], ha_open[i], ha_close[i])
    return ha_open, ha_high, ha_low, ha_close


@njit(parallel=False, fastmath=True)
def _calculate_pivot_points_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Classic pivot points: PP, R1, R2, S1, S2."""
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
        pp[i] = (high[i - 1] + low[i - 1] + close[i - 1]) / 3.0
        r1[i] = 2 * pp[i] - low[i - 1]
        r2[i] = pp[i] + (high[i - 1] - low[i - 1])
        s1[i] = 2 * pp[i] - high[i - 1]
        s2[i] = pp[i] - (high[i - 1] - low[i - 1])
    return pp, r1, r2, s1, s2


@njit(parallel=False, fastmath=True)
def _calculate_rainbow_numba(prices: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Rainbow MA: stack multiple SMA lines for provided periods.

    Returns a 2D array shape (len(periods), len(prices)).
    """
    n = len(prices)
    k = len(periods)
    out = np.full((k, n), np.nan, dtype=np.float64)
    for p_idx in range(k):
        p = int(periods[p_idx])
        if p <= 0 or p > n:
            continue
        for i in range(p - 1, n):
            if np.any(np.isnan(prices[i - p + 1 : i + 1])):
                continue
            s = 0.0
            cnt = 0
            for j in range(i - p + 1, i + 1):
                s += prices[j]
                cnt += 1
            out[p_idx, i] = s / cnt
    return out
# Re-export additional trend kernels from the monolith until fully migrated
from .._indicator_kernels import (  # noqa: E402
    _calculate_linear_regression_channel_numba,
    _calculate_rainbow_numba,
)

__all__ = [
    "_calculate_sma_numba",
    "_calculate_ema_numba",
    "_calculate_dema_numba",
    "_calculate_kama_numba",
    "_calculate_ichimoku_numba",
    "_calculate_heiken_ashi_numba",
    "_calculate_pivot_points_numba",
    "_calculate_rainbow_numba",
    "_calculate_hull_ma_numba",
    "_calculate_alma_numba",
    "_calculate_linear_regression_channel_numba",
    "_calculate_rainbow_numba",
]
