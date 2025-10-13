"""
Momentum kernels
"""
import numpy as np
from typing import Tuple
from .._indicator_kernels import njit as njit


@njit(parallel=False, fastmath=True)
def _calculate_rsi_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """Optimized Relative Strength Index calculation."""
    n = len(prices)
    if n == 0 or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
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
                rsi[i] = 50.0
            elif avg_loss[i] == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))
    return rsi


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


# Re-export remaining momentum kernels until migrated
from .._indicator_kernels import (  # noqa: E402
    _calculate_momentum_index_numba,  # will override next
    _calculate_ma_momentum_numba,     # will override next
)

__all__ = [
    "_calculate_rsi_numba",
    "_calculate_macd_numba",
    "_calculate_momentum_index_numba",
    "_calculate_ma_momentum_numba",
]


@njit(parallel=False, fastmath=True)
def _calculate_roc_numba(prices: np.ndarray, period: int) -> np.ndarray:
    n = len(prices)
    if n == 0 or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(prices[i]) and not np.isnan(prices[i - period]) and prices[i - period] != 0:
            roc[i] = ((prices[i] - prices[i - period]) / prices[i - period]) * 100
    return roc


@njit(parallel=False, fastmath=True)
def _calculate_dpo_numba(prices: np.ndarray, period: int) -> np.ndarray:
    n = len(prices)
    if n == 0 or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    dpo = np.full(n, np.nan, dtype=np.float64)
    shift = period // 2 + 1
    for i in range(period - 1, n):
        if np.any(np.isnan(prices[i - period + 1 : i + 1])):
            continue
        sum_close = 0.0
        count = 0
        for j in range(i - period + 1, i + 1):
            sum_close += prices[j]
            count += 1
        sma = sum_close / count
        if i - shift >= 0 and not np.isnan(prices[i - shift]):
            dpo[i] = prices[i - shift] - sma
    return dpo


__all__ += ["_calculate_roc_numba", "_calculate_dpo_numba"]


@njit(parallel=False, fastmath=True)
def _calculate_stochastic_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int, d_period: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or k_period <= 0 or d_period <= 0 or k_period + d_period > n:
        return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)
    k = np.full(n, np.nan, dtype=np.float64)
    d = np.full(n, np.nan, dtype=np.float64)
    for i in range(k_period - 1, n):
        if (
            np.any(np.isnan(high[i - k_period + 1 : i + 1]))
            or np.any(np.isnan(low[i - k_period + 1 : i + 1]))
            or np.isnan(close[i])
        ):
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
def _calculate_kdj_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 9, d_period: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or k_period <= 0 or d_period <= 0 or k_period + d_period > n:
        return (
            np.full(n, np.nan, dtype=np.float64),
            np.full(n, np.nan, dtype=np.float64),
            np.full(n, np.nan, dtype=np.float64),
        )
    rsv = np.full(n, np.nan, dtype=np.float64)
    k = np.full(n, np.nan, dtype=np.float64)
    d = np.full(n, np.nan, dtype=np.float64)
    j = np.full(n, np.nan, dtype=np.float64)
    for i in range(k_period - 1, n):
        if (
            np.any(np.isnan(high[i - k_period + 1 : i + 1]))
            or np.any(np.isnan(low[i - k_period + 1 : i + 1]))
            or np.isnan(close[i])
        ):
            continue
        highest_high = np.max(high[i - k_period + 1 : i + 1])
        lowest_low = np.min(low[i - k_period + 1 : i + 1])
        if highest_high != lowest_low:
            rsv[i] = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
    if not np.isnan(rsv[k_period - 1]):
        k[k_period - 1] = rsv[k_period - 1]
    for i in range(k_period, n):
        if not np.isnan(k[i - 1]) and not np.isnan(rsv[i]):
            k[i] = (2 / 3) * k[i - 1] + (1 / 3) * rsv[i]
    for i in range(k_period + d_period - 1, n):
        if np.any(np.isnan(k[i - d_period + 1 : i + 1])):
            continue
        d[i] = np.mean(k[i - d_period + 1 : i + 1])
    for i in range(n):
        if not np.isnan(k[i]) and not np.isnan(d[i]):
            j[i] = 3 * k[i] - 2 * d[i]
    return k, d, j


@njit(parallel=False, fastmath=True)
def _calculate_elder_ray_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> Tuple[np.ndarray, np.ndarray]:
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
    n = len(close)
    if (
        n == 0
        or len(high) != n
        or len(low) != n
        or period1 <= 0
        or period2 <= 0
        or period3 <= 0
        or max(period1, period2, period3) > n
    ):
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


__all__ += [
    "_calculate_stochastic_numba",
    "_calculate_kdj_numba",
    "_calculate_elder_ray_numba",
    "_calculate_ultimate_oscillator_numba",
]


@njit(parallel=False, fastmath=True)
def _calculate_trix_numba(close: np.ndarray, period: int) -> np.ndarray:
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
        ema2[0] = close[0]
        ema3[0] = close[0]
    for i in range(1, n):
        if not np.isnan(close[i]) and not np.isnan(ema1[i - 1]):
            ema1[i] = ema1[i - 1] + alpha * (close[i] - ema1[i - 1])
        if not np.isnan(ema1[i]) and not np.isnan(ema2[i - 1]):
            ema2[i] = ema2[i - 1] + alpha * (ema1[i] - ema2[i - 1])
        if not np.isnan(ema2[i]) and not np.isnan(ema3[i - 1]):
            ema3[i] = ema3[i - 1] + alpha * (ema2[i] - ema3[i - 1])
    for i in range(1, n):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i - 1]) and ema3[i - 1] != 0:
            trix[i] = ((ema3[i] - ema3[i - 1]) / ema3[i - 1]) * 100
    return trix


@njit(parallel=False, fastmath=True)
def _calculate_chande_momentum_numba(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    if n == 0 or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    cmo = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        gains = 0.0
        losses = 0.0
        valid = True
        for j in range(i - period + 1, i + 1):
            if np.isnan(close[j]) or np.isnan(close[j - 1]):
                valid = False
                break
            change = close[j] - close[j - 1]
            if change > 0:
                gains += change
            else:
                losses += abs(change)
        if valid and (gains + losses) != 0:
            cmo[i] = 100 * (gains - losses) / (gains + losses)
    return cmo


@njit(parallel=False, fastmath=True)
def _calculate_momentum_index_numba(close: np.ndarray, period: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    n = len(close)
    if n == 0 or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)
    pos_acc = np.zeros(n, dtype=np.float64)
    neg_acc = np.zeros(n, dtype=np.float64)
    mi = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - 1]):
            diff = close[i] - close[i - 1]
            if diff > 0:
                pos_acc[i] = pos_acc[i - 1] + diff
                neg_acc[i] = neg_acc[i - 1]
            elif diff < 0:
                neg_acc[i] = neg_acc[i - 1] + abs(diff)
                pos_acc[i] = pos_acc[i - 1]
            else:
                pos_acc[i] = pos_acc[i - 1]
                neg_acc[i] = neg_acc[i - 1]
    for i in range(period, n):
        pos_p = pos_acc[i] - pos_acc[i - period]
        neg_p = neg_acc[i] - neg_acc[i - period]
        tot = pos_p + neg_p
        if tot > 0:
            mi[i] = 100 * (pos_p / tot)
        elif tot == 0:
            mi[i] = 50.0
    mi_neg = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(mi[i]):
            mi_neg[i] = 100 - mi[i]
    return mi, mi_neg


@njit(parallel=False, fastmath=True)
def _calculate_ma_momentum_numba(prices: np.ndarray, ma_period: int, momentum_period: int) -> np.ndarray:
    n = len(prices)
    if n == 0 or ma_period <= 0 or momentum_period <= 0 or ma_period + momentum_period > n:
        return np.full(n, np.nan, dtype=np.float64)
    ma = np.full(n, np.nan, dtype=np.float64)
    for i in range(ma_period - 1, n):
        if np.any(np.isnan(prices[i - ma_period + 1 : i + 1])):
            continue
        acc = 0.0
        cnt = 0
        for j in range(i - ma_period + 1, i + 1):
            acc += prices[j]
            cnt += 1
        ma[i] = acc / cnt
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(ma_period + momentum_period - 1, n):
        if not np.isnan(ma[i]) and not np.isnan(ma[i - momentum_period]) and ma[i - momentum_period] != 0:
            out[i] = 100 * (ma[i] - ma[i - momentum_period]) / ma[i - momentum_period]
    return out


@njit(parallel=False, fastmath=True)
def _calculate_rvi_numba(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    if n == 0 or len(open_) != n or len(high) != n or len(low) != n or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    rvi = np.full(n, np.nan, dtype=np.float64)
    num = np.zeros(n, dtype=np.float64)
    den = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if not np.isnan(open_[i]) and not np.isnan(high[i]) and not np.isnan(low[i]) and not np.isnan(close[i]):
            num[i] = (close[i] - open_[i]) / 2 + (close[i - 1] - open_[i - 1]) / 2 if i > 0 else 0
            den[i] = (high[i] - low[i]) / 2 + (high[i - 1] - low[i - 1]) / 2 if i > 0 else 0
    avg_num = np.full(n, np.nan, dtype=np.float64)
    avg_den = np.full(n, np.nan, dtype=np.float64)
    if period < n:
        s_num = 0.0
        s_den = 0.0
        cnt = 0
        for i in range(1, period + 1):
            s_num += num[i]
            s_den += den[i]
            cnt += 1
        if cnt > 0:
            avg_num[period] = s_num / cnt
            avg_den[period] = s_den / cnt
        for i in range(period + 1, n):
            avg_num[i] = (avg_num[i - 1] * (period - 1) + num[i]) / period
            avg_den[i] = (avg_den[i - 1] * (period - 1) + den[i]) / period
    for i in range(period, n):
        if not np.isnan(avg_den[i]) and avg_den[i] != 0 and not np.isnan(avg_num[i]):
            rvi[i] = avg_num[i] / avg_den[i]
        elif avg_den[i] == 0 and avg_num[i] == 0:
            rvi[i] = 0.0
    return rvi


__all__ += [
    "_calculate_trix_numba",
    "_calculate_chande_momentum_numba",
    "_calculate_momentum_index_numba",
    "_calculate_ma_momentum_numba",
    "_calculate_rvi_numba",
]


@njit(parallel=False, fastmath=True)
def _calculate_adx_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average Directional Index (ADX), plus +DI and -DI."""
    n = len(high)
    if n == 0 or len(low) != n or len(close) != n or period <= 0 or period > n:
        return (
            np.full(n, np.nan, dtype=np.float64),
            np.full(n, np.nan, dtype=np.float64),
            np.full(n, np.nan, dtype=np.float64),
        )
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
    smooth_tr = np.full(n, np.nan, dtype=np.float64)
    smooth_plus = np.full(n, np.nan, dtype=np.float64)
    smooth_minus = np.full(n, np.nan, dtype=np.float64)
    if period < n:
        smooth_tr[period] = np.nansum(tr[1 : period + 1])
        smooth_plus[period] = np.nansum(plus_dm[1 : period + 1])
        smooth_minus[period] = np.nansum(minus_dm[1 : period + 1])
        for i in range(period + 1, n):
            if not np.isnan(tr[i]):
                smooth_tr[i] = smooth_tr[i - 1] - (smooth_tr[i - 1] / period) + tr[i]
                smooth_plus[i] = smooth_plus[i - 1] - (smooth_plus[i - 1] / period) + plus_dm[i]
                smooth_minus[i] = smooth_minus[i - 1] - (smooth_minus[i - 1] / period) + minus_dm[i]
    pdi = np.full(n, np.nan, dtype=np.float64)
    mdi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if smooth_tr[i] != 0 and not np.isnan(smooth_tr[i]):
            pdi[i] = 100 * smooth_plus[i] / smooth_tr[i]
            mdi[i] = 100 * smooth_minus[i] / smooth_tr[i]
    dx = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(pdi[i]) and not np.isnan(mdi[i]) and (pdi[i] + mdi[i]) != 0:
            dx[i] = 100 * abs(pdi[i] - mdi[i]) / (pdi[i] + mdi[i])
    adx = np.full(n, np.nan, dtype=np.float64)
    if 2 * period - 1 < n:
        # initial average of DX
        s = 0.0
        cnt = 0
        for i in range(period, 2 * period):
            if not np.isnan(dx[i]):
                s += dx[i]
                cnt += 1
        if cnt > 0:
            adx[2 * period - 1] = s / cnt
        for i in range(2 * period, n):
            if not np.isnan(dx[i]) and not np.isnan(adx[i - 1]):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx, pdi, mdi


@njit(parallel=False, fastmath=True)
def _calculate_cci_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, constant: float = 0.015
) -> np.ndarray:
    """Commodity Channel Index (CCI)."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    tp = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(high[i]) and not np.isnan(low[i]) and not np.isnan(close[i]):
            tp[i] = (high[i] + low[i] + close[i]) / 3.0
    cci = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if np.any(np.isnan(tp[i - period + 1 : i + 1])):
            continue
        mean_tp = 0.0
        cnt = 0
        for j in range(i - period + 1, i + 1):
            mean_tp += tp[j]
            cnt += 1
        mean_tp /= cnt
        sum_dev = 0.0
        for j in range(i - period + 1, i + 1):
            sum_dev += abs(tp[j] - mean_tp)
        mean_dev = sum_dev / cnt
        if mean_dev != 0 and not np.isnan(tp[i]):
            cci[i] = (tp[i] - mean_tp) / (constant * mean_dev)
    return cci


@njit(parallel=False, fastmath=True)
def _calculate_willr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Williams %R."""
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    willr = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        hh = high[i - period + 1]
        ll = low[i - period + 1]
        for j in range(i - period + 1, i + 1):
            if high[j] > hh:
                hh = high[j]
            if low[j] < ll:
                ll = low[j]
        if hh != ll and not np.isnan(close[i]):
            willr[i] = (hh - close[i]) / (hh - ll) * -100.0
    return willr


@njit(parallel=False, fastmath=True)
def _calculate_aroon_numba(high: np.ndarray, low: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(high)
    if n == 0 or len(low) != n or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)
    up = np.full(n, np.nan, dtype=np.float64)
    down = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        window_high = high[i - period + 1 : i + 1]
        window_low = low[i - period + 1 : i + 1]
        if np.any(np.isnan(window_high)) or np.any(np.isnan(window_low)):
            continue
        high_idx = 0
        low_idx = 0
        hh = window_high[0]
        ll = window_low[0]
        for j in range(period):
            if window_high[j] >= hh:
                hh = window_high[j]
                high_idx = j
            if window_low[j] <= ll:
                ll = window_low[j]
                low_idx = j
        up[i] = ((period - (period - high_idx - 1)) / period) * 100
        down[i] = ((period - (period - low_idx - 1)) / period) * 100
    return up, down


@njit(parallel=False, fastmath=True)
def _calculate_awesome_oscillator_numba(high: np.ndarray, low: np.ndarray, short_period: int = 5, long_period: int = 34) -> np.ndarray:
    n = len(high)
    if n == 0 or len(low) != n or short_period <= 0 or long_period <= 0 or long_period > n:
        return np.full(n, np.nan, dtype=np.float64)
    ao = np.full(n, np.nan, dtype=np.float64)
    median = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(high[i]) and not np.isnan(low[i]):
            median[i] = (high[i] + low[i]) / 2
    def sma(m: np.ndarray, p: int, i: int) -> float:
        tot = 0.0
        cnt = 0
        for j in range(i - p + 1, i + 1):
            tot += m[j]
            cnt += 1
        return tot / cnt if cnt > 0 else np.nan
    for i in range(long_period - 1, n):
        if np.any(np.isnan(median[i - long_period + 1 : i + 1])) or np.any(np.isnan(median[i - short_period + 1 : i + 1])):
            continue
        ao[i] = sma(median, short_period, i) - sma(median, long_period, i)
    return ao


@njit(parallel=False, fastmath=True)
def _calculate_beta_numba(returns: np.ndarray, market_returns: np.ndarray, period: int) -> np.ndarray:
    n = len(returns)
    if n == 0 or len(market_returns) != n or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64)
    beta = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        ret_win = returns[i - period : i]
        mkt_win = market_returns[i - period : i]
        if np.any(np.isnan(ret_win)) or np.any(np.isnan(mkt_win)):
            continue
        ret_mean = 0.0
        mkt_mean = 0.0
        for j in range(period):
            ret_mean += ret_win[j]
            mkt_mean += mkt_win[j]
        ret_mean /= period
        mkt_mean /= period
        cov = 0.0
        var = 0.0
        for j in range(period):
            cov += (ret_win[j] - ret_mean) * (mkt_win[j] - mkt_mean)
            var += (mkt_win[j] - mkt_mean) * (mkt_win[j] - mkt_mean)
        cov /= period
        var /= period
        if var != 0:
            beta[i] = cov / var
    return beta


@njit(parallel=False, fastmath=True)
def _calculate_di_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(high)
    if n == 0 or len(low) != n or len(close) != n or period <= 0 or period > n:
        return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)
    plus_dm = np.full(n, np.nan, dtype=np.float64)
    minus_dm = np.full(n, np.nan, dtype=np.float64)
    tr = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i - 1]):
            continue
        hd = high[i] - high[i - 1]
        ld = low[i - 1] - low[i]
        if hd > ld and hd > 0:
            plus_dm[i] = hd
        if ld > hd and ld > 0:
            minus_dm[i] = ld
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    smooth_plus = np.full(n, np.nan, dtype=np.float64)
    smooth_minus = np.full(n, np.nan, dtype=np.float64)
    smooth_tr = np.full(n, np.nan, dtype=np.float64)
    if period < n:
        smooth_plus[period] = np.nansum(plus_dm[1 : period + 1])
        smooth_minus[period] = np.nansum(minus_dm[1 : period + 1])
        smooth_tr[period] = np.nansum(tr[1 : period + 1])
        for i in range(period + 1, n):
            if not np.isnan(plus_dm[i]):
                smooth_plus[i] = smooth_plus[i - 1] - (smooth_plus[i - 1] / period) + plus_dm[i]
            if not np.isnan(minus_dm[i]):
                smooth_minus[i] = smooth_minus[i - 1] - (smooth_minus[i - 1] / period) + minus_dm[i]
            if not np.isnan(tr[i]):
                smooth_tr[i] = smooth_tr[i - 1] - (smooth_tr[i - 1] / period) + tr[i]
    pdi = np.full(n, np.nan, dtype=np.float64)
    mdi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if smooth_tr[i] != 0 and not np.isnan(smooth_tr[i]):
            pdi[i] = 100 * smooth_plus[i] / smooth_tr[i]
            mdi[i] = 100 * smooth_minus[i] / smooth_tr[i]
    return pdi, mdi


__all__ += [
    "_calculate_adx_numba",
    "_calculate_cci_numba",
    "_calculate_willr_numba",
    "_calculate_aroon_numba",
    "_calculate_awesome_oscillator_numba",
    "_calculate_beta_numba",
    "_calculate_di_numba",
]
