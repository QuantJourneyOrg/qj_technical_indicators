import numpy as np
import pandas as pd

from quantjourney_ti import TechnicalIndicators, StreamingIndicators


def _series(n=200, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    s = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100, index=idx)
    return s


def test_streaming_sma_matches_batch_last_value():
    s = _series()
    ti = TechnicalIndicators()
    si = StreamingIndicators(max_buffer_size=1000)
    symbol = "SYM"

    for ts, val in s.items():
        si.update_tick(symbol, ts, close=float(val))

    # batch SMA
    batch = ti.SMA(s, 20)
    # streaming SMA last value
    last = si.states[symbol].last_values.get("sma_20", np.nan)

    assert np.isfinite(last)
    assert abs(last - batch.iloc[-1]) < 1e-9


def test_streaming_ema_matches_batch_last_value():
    s = _series()
    ti = TechnicalIndicators()
    si = StreamingIndicators(max_buffer_size=1000)
    symbol = "SYM"

    for ts, val in s.items():
        si.update_tick(symbol, ts, close=float(val))

    batch = ti.EMA(s, 20)
    last = si.states[symbol].last_values.get("ema_20", np.nan)
    assert np.isfinite(last)
    assert abs(last - batch.iloc[-1]) < 1e-6

