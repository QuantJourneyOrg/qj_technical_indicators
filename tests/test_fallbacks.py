import pandas as pd
import numpy as np

from quantjourney_ti import TechnicalIndicators


def test_ema_fallback_on_kernel_error(monkeypatch):
    # Prepare data
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    s = pd.Series(np.linspace(100, 110, 50), index=idx, name="close")
    ti = TechnicalIndicators()

    # Monkeypatch the kernel to raise ValueError to trigger fallback
    def boom(*args, **kwargs):
        raise ValueError("numba failed")

    monkeypatch.setattr(TechnicalIndicators, "_calculate_ema_numba", staticmethod(boom))

    # Call EMA — decorator should fall back to pandas ewm
    out = ti.EMA(s, 10)
    ref = s.ewm(span=10, adjust=False).mean().rename("EMA_10")

    # Compare last values (allow tiny numeric differences)
    assert abs(out.iloc[-1] - ref.iloc[-1]) < 1e-9

