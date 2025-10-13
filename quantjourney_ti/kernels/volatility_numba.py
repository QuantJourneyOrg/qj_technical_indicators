"""
Volatility kernels
"""
import numpy as np
from typing import Tuple
from .._indicator_kernels import njit as njit


@njit(parallel=False, fastmath=True)
def _calculate_atr_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int
) -> np.ndarray:
    n = len(high)
    if n == 0 or len(low) != n or len(close) != n or window <= 0 or window > n:
        return np.full(n, np.nan, dtype=np.float64)
    tr = np.full(n, np.nan, dtype=np.float64)
    atr = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i - 1]):
            continue
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
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
def _calculate_bollinger_bands_numba(
    prices: np.ndarray, window: int, num_std: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
def _calculate_keltner_channels_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ema_period: int,
    atr_period: int,
    multiplier: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
def _calculate_donchian_channels_numba(
    high: np.ndarray, low: np.ndarray, period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
def _calculate_supertrend_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, multiplier: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Supertrend line and direction (+1 uptrend, -1 downtrend)."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or period <= 0 or period > n:
        empty = np.full(n, np.nan, dtype=np.float64)
        return empty, empty
    tr = np.full(n, np.nan, dtype=np.float64)
    atr = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i - 1]):
            continue
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if period < n:
        s = 0.0
        cnt = 0
        for i in range(1, period + 1):
            if not np.isnan(tr[i]):
                s += tr[i]
                cnt += 1
        if cnt > 0:
            atr[period] = s / cnt
        for i in range(period + 1, n):
            if not np.isnan(tr[i]) and not np.isnan(atr[i - 1]):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    st = np.full(n, np.nan, dtype=np.float64)
    direction = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]) or np.isnan(atr[i]):
            continue
        mid = (high[i] + low[i]) / 2.0
        upper[i] = mid + multiplier * atr[i]
        lower[i] = mid - multiplier * atr[i]
        if i == period:
            st[i] = lower[i]
            direction[i] = 1
        else:
            if direction[i - 1] == 1:
                if not np.isnan(upper[i - 1]):
                    upper[i] = min(upper[i], upper[i - 1])
                if close[i] > st[i - 1]:
                    st[i] = lower[i]
                    direction[i] = 1
                else:
                    st[i] = upper[i]
                    direction[i] = -1
            else:
                if not np.isnan(lower[i - 1]):
                    lower[i] = max(lower[i], lower[i - 1])
                if close[i] < st[i - 1]:
                    st[i] = upper[i]
                    direction[i] = -1
                else:
                    st[i] = lower[i]
                    direction[i] = 1
    return st, direction
# Re-export remaining volatility kernels until migrated
@njit(parallel=False, fastmath=True)
def _calculate_chaikin_volatility_numba(
    high: np.ndarray, low: np.ndarray, ema_period: int = 10, roc_period: int = 10
) -> np.ndarray:
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
def _calculate_historical_volatility_numba(
    close: np.ndarray, period: int = 20, trading_days: int = 252
) -> np.ndarray:
    n = len(close)
    if n <= 1 or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    returns = np.full(n - 1, np.nan, dtype=np.float64)
    for i in range(1, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - 1]) and close[i - 1] != 0:
            returns[i - 1] = np.log(close[i] / close[i - 1])
    hv = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if np.any(np.isnan(returns[i - period : i])):
            continue
        std = np.std(returns[i - period : i])
        hv[i] = std * np.sqrt(trading_days) * 100
    return hv


@njit(parallel=False, fastmath=True)
def _calculate_linear_regression_channel_numba(
    close: np.ndarray, period: int, deviations: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    if n == 0 or period <= 2 or period > n:
        empty = np.full(n, np.nan, dtype=np.float64)
        return empty, empty, empty
    middle = np.full(n, np.nan, dtype=np.float64)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        y = close[i - period + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue
        x = np.arange(period, dtype=np.float64)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        num = 0.0
        den = 0.0
        for j in range(period):
            num += (x[j] - x_mean) * (y[j] - y_mean)
            den += (x[j] - x_mean) * (x[j] - x_mean)
        if den == 0:
            continue
        slope = num / den
        intercept = y_mean - slope * x_mean
        predict = slope * (period - 1) + intercept
        middle[i] = predict
        # compute std error
        std_err_sum = 0.0
        for j in range(period):
            y_pred = slope * x[j] + intercept
            std_err_sum += (y[j] - y_pred) * (y[j] - y_pred)
        std_err = np.sqrt(std_err_sum / (period - 2)) if period > 2 else 0.0
        upper[i] = predict + deviations * std_err
        lower[i] = predict - deviations * std_err
    return upper, middle, lower


@njit(parallel=False, fastmath=True)
def _calculate_choppiness_index_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    n = len(high)
    if n == 0 or len(low) != n or len(close) != n or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    tr = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i - 1]):
            continue
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    ci = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if (
            np.any(np.isnan(tr[i - period + 1 : i + 1]))
            or np.any(np.isnan(high[i - period + 1 : i + 1]))
            or np.any(np.isnan(low[i - period + 1 : i + 1]))
        ):
            continue
        sum_tr = 0.0
        for j in range(i - period + 1, i + 1):
            sum_tr += tr[j]
        highest_high = np.max(high[i - period + 1 : i + 1])
        lowest_low = np.min(low[i - period + 1 : i + 1])
        if highest_high != lowest_low and sum_tr > 0:
            ci[i] = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(period)
    return ci


@njit(parallel=False, fastmath=True)
def _calculate_mass_index_numba(
    high: np.ndarray, low: np.ndarray, ema_period: int = 9, sum_period: int = 25
) -> np.ndarray:
    """Mass Index based on EMA(high-low) and EMA(EMA(high-low))."""
    n = len(high)
    if n == 0 or len(low) != n or ema_period <= 0 or sum_period <= 0 or ema_period + sum_period > n:
        return np.full(n, np.nan, dtype=np.float64)
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(high[i]) and not np.isnan(low[i]):
            diff[i] = high[i] - low[i]
    ema1 = np.full(n, np.nan, dtype=np.float64)
    ema2 = np.full(n, np.nan, dtype=np.float64)
    a = 2.0 / (ema_period + 1)
    if n > 0 and not np.isnan(diff[0]):
        ema1[0] = diff[0]
        ema2[0] = diff[0]
    for i in range(1, n):
        if not np.isnan(diff[i]) and not np.isnan(ema1[i - 1]):
            ema1[i] = (diff[i] - ema1[i - 1]) * a + ema1[i - 1]
        if not np.isnan(ema1[i]) and not np.isnan(ema2[i - 1]):
            ema2[i] = (ema1[i] - ema2[i - 1]) * a + ema2[i - 1]
    ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(ema1[i]) and not np.isnan(ema2[i]) and ema2[i] != 0:
            ratio[i] = ema1[i] / ema2[i]
    mi = np.full(n, np.nan, dtype=np.float64)
    for i in range(sum_period - 1, n):
        if np.any(np.isnan(ratio[i - sum_period + 1 : i + 1])):
            continue
        s = 0.0
        for j in range(i - sum_period + 1, i + 1):
            s += ratio[j]
        mi[i] = s
    return mi

__all__ = [
    "_calculate_atr_numba",
    "_calculate_bollinger_bands_numba",
    "_calculate_keltner_channels_numba",
    "_calculate_donchian_channels_numba",
    "_calculate_supertrend_numba",
    "_calculate_chaikin_volatility_numba",
    "_calculate_historical_volatility_numba",
    "_calculate_linear_regression_channel_numba",
    "_calculate_choppiness_index_numba",
    "_calculate_mass_index_numba",
]
