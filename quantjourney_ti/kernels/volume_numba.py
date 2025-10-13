"""
Volume kernels
"""
import numpy as np
from typing import Tuple
from .._indicator_kernels import njit as njit


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
def _calculate_ad_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or len(volume) != n:
        return np.empty(0, dtype=np.float64)
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
def _calculate_mfi_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int,
) -> np.ndarray:
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or len(volume) != n or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    typical_price = np.full(n, np.nan, dtype=np.float64)
    money_flow = np.full(n, np.nan, dtype=np.float64)
    positive_flow = np.zeros(n, dtype=np.float64)
    negative_flow = np.zeros(n, dtype=np.float64)
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
            mfi[i] = 50.0
        elif negative_sum == 0:
            mfi[i] = 100.0
        else:
            money_ratio = positive_sum / negative_sum
            mfi[i] = 100 - (100 / (1 + money_ratio))
    return mfi


@njit(parallel=False, fastmath=True)
def _calculate_vwap_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int
) -> np.ndarray:
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or len(volume) != n or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    vwap = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if (
            np.any(np.isnan(high[i - period + 1 : i + 1]))
            or np.any(np.isnan(low[i - period + 1 : i + 1]))
            or np.any(np.isnan(close[i - period + 1 : i + 1]))
            or np.any(np.isnan(volume[i - period + 1 : i + 1]))
        ):
            continue
        sum_price_vol = 0.0
        sum_vol = 0.0
        count = 0
        for j in range(i - period + 1, i + 1):
            sum_price_vol += ((high[j] + low[j] + close[j]) / 3.0) * volume[j]
            sum_vol += volume[j]
            count += 1
        if sum_vol != 0:
            vwap[i] = sum_price_vol / sum_vol
    return vwap


@njit(parallel=False, fastmath=True)
def _calculate_typical_price_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(high[i]) and not np.isnan(low[i]) and not np.isnan(close[i]):
            out[i] = (high[i] + low[i] + close[i]) / 3.0
    return out


@njit(parallel=False, fastmath=True)
def _calculate_weighted_close_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(high[i]) and not np.isnan(low[i]) and not np.isnan(close[i]):
            out[i] = (high[i] + low[i] + 2 * close[i]) / 4.0
    return out


@njit(parallel=False, fastmath=True)
def _calculate_volume_indicators_numba(
    close: np.ndarray, volume: np.ndarray, period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

__all__ = [
    "_calculate_obv_numba",
    "_calculate_ad_numba",
    "_calculate_mfi_numba",
    "_calculate_vwap_numba",
    "_calculate_volume_indicators_numba",
    "_calculate_typical_price_numba",
    "_calculate_weighted_close_numba",
]


@njit(parallel=False, fastmath=True)
def _calculate_adosc_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, fast_period: int = 3, slow_period: int = 10
) -> np.ndarray:
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or len(volume) != n or fast_period <= 0 or slow_period <= 0:
        return np.full(n, np.nan, dtype=np.float64)
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
    fast_ema = np.full(n, np.nan, dtype=np.float64)
    slow_ema = np.full(n, np.nan, dtype=np.float64)
    if n > 0:
        fast_ema[0] = ad[0]
        slow_ema[0] = ad[0]
    fm = 2.0 / (fast_period + 1)
    sm = 2.0 / (slow_period + 1)
    for i in range(1, n):
        if not np.isnan(ad[i]) and not np.isnan(fast_ema[i - 1]):
            fast_ema[i] = (ad[i] - fast_ema[i - 1]) * fm + fast_ema[i - 1]
        if not np.isnan(ad[i]) and not np.isnan(slow_ema[i - 1]):
            slow_ema[i] = (ad[i] - slow_ema[i - 1]) * sm + slow_ema[i - 1]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(fast_ema[i]) and not np.isnan(slow_ema[i]):
            out[i] = fast_ema[i] - slow_ema[i]
    return out


@njit(parallel=False, fastmath=True)
def _calculate_pvo_numba(volume: np.ndarray, short_period: int = 12, long_period: int = 26) -> Tuple[np.ndarray, np.ndarray]:
    n = len(volume)
    if n == 0 or short_period <= 0 or long_period <= 0 or max(short_period, long_period) > n:
        return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)
    se = np.full(n, np.nan, dtype=np.float64)
    le = np.full(n, np.nan, dtype=np.float64)
    sm = 2.0 / (short_period + 1)
    lm = 2.0 / (long_period + 1)
    if n > 0 and not np.isnan(volume[0]):
        se[0] = volume[0]
        le[0] = volume[0]
    for i in range(1, n):
        if not np.isnan(volume[i]) and not np.isnan(se[i - 1]):
            se[i] = (volume[i] - se[i - 1]) * sm + se[i - 1]
        if not np.isnan(volume[i]) and not np.isnan(le[i - 1]):
            le[i] = (volume[i] - le[i - 1]) * lm + le[i - 1]
    pvo = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(se[i]) and not np.isnan(le[i]) and le[i] != 0:
            pvo[i] = 100 * (se[i] - le[i]) / le[i]
    # Signal EMA (9)
    signal = np.full(n, np.nan, dtype=np.float64)
    sigm = 2.0 / (9 + 1)
    if n > 0 and not np.isnan(pvo[0]):
        signal[0] = pvo[0]
    for i in range(1, n):
        if not np.isnan(pvo[i]) and not np.isnan(signal[i - 1]):
            signal[i] = (pvo[i] - signal[i - 1]) * sigm + signal[i - 1]
    return pvo, signal


__all__ += ["_calculate_adosc_numba", "_calculate_pvo_numba"]
