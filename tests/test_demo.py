import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from quantjourney_ti import TechnicalIndicators, willr, sma

@pytest.fixture
def mock_yfinance_data():
    return pd.DataFrame({
        "open": [100, 101, 102, 103, 104],
        "high": [101, 102, 103, 104, 105],
        "low": [99, 100, 101, 102, 103],
        "close": [100, 101, 102, 103, 104],
        "adj_close": [100, 101, 102, 103, 104],
        "volume": [1000, 1100, 1200, 1300, 1400]
    }, index=pd.date_range("2025-01-01", periods=5))

def test_demo_willr(monkeypatch, mock_yfinance_data):
    def mock_download(*args, **kwargs):
        return mock_yfinance_data
    monkeypatch.setattr("yfinance.download", mock_download)
    ti = TechnicalIndicators()
    result = ti.WILLR(mock_yfinance_data[["high", "low", "close"]], period=3)
    expected = pd.Series([np.nan, np.nan, -25.0, -25.0, -25.0], index=mock_yfinance_data.index, name="WILLR_3")
    pd.testing.assert_series_equal(result, expected, rtol=1e-4, atol=1e-4)

def test_demo_sma(monkeypatch, mock_yfinance_data):
    def mock_download(*args, **kwargs):
        return mock_yfinance_data
    monkeypatch.setattr("yfinance.download", mock_download)
    result = sma(mock_yfinance_data["close"], period=3)  # Changed window to period
    expected = pd.Series([np.nan, np.nan, 101.0, 102.0, 103.0], index=mock_yfinance_data.index, name="SMA_3")
    pd.testing.assert_series_equal(result, expected, rtol=1e-4, atol=1e-4)