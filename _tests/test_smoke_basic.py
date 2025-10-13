import numpy as np
import pandas as pd

from quantjourney_ti import TechnicalIndicators


def _make_series(n=100, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    base = np.cumsum(rng.normal(0, 1, n)) + 100
    return pd.Series(base, index=idx, name="close")


def test_basic_sma_rsi_shapes():
    s = _make_series()
    ti = TechnicalIndicators()

    sma = ti.SMA(s, 20)
    rsi = ti.RSI(s, 14)

    assert len(sma) == len(s)
    assert len(rsi) == len(s)
    assert sma.head(19).isna().all()
    assert rsi.isna().sum() >= 10  # warmup head should have NaNs

