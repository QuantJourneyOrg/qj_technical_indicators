import numpy as np
import pandas as pd

from quantjourney_ti import TechnicalIndicators


def _ohlc(n=250, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = np.cumsum(rng.normal(0, 1, n)) + 100
    high = close + np.abs(rng.normal(0.5, 0.2, n))
    low = close - np.abs(rng.normal(0.5, 0.2, n))
    open_ = close + rng.normal(0, 0.1, n)
    vol = np.abs(rng.normal(1e6, 2e5, n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_keltner_upper_lower_positions():
    df = _ohlc()
    ti = TechnicalIndicators()
    kc = ti.KELTNER(df[["high", "low", "close"]], 20, 10, 2.0)
    up = kc.iloc[:, 0]
    mid = kc.iloc[:, 1]
    lo = kc.iloc[:, 2]
    mask = ~(up.isna() | mid.isna() | lo.isna())
    assert (up[mask] >= mid[mask]).all()
    assert (mid[mask] >= lo[mask]).all()


def test_donchian_upper_lower_positions():
    df = _ohlc()
    ti = TechnicalIndicators()
    dc = ti.DONCHIAN(df[["high", "low"]], 20)
    up = dc.iloc[:, 0]
    mid = dc.iloc[:, 1]
    lo = dc.iloc[:, 2]
    mask = ~(up.isna() | mid.isna() | lo.isna())
    assert (up[mask] >= mid[mask]).all()
    assert (mid[mask] >= lo[mask]).all()

