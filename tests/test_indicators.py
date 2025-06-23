import pytest
import pandas as pd
import numpy as np

from quantjourney_ti import TechnicalIndicators
from quantjourney_ti._errors import IndicatorCalculationError, InvalidInputError

@pytest.fixture
def ti():
    return TechnicalIndicators()

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "open": [100, 101, 102, 103, 104],
        "high": [101, 102, 103, 104, 105],
        "low": [99, 100, 101, 102, 103],
        "close": [100, 101, 102, 103, 104],
        "volume": [1000, 1100, 1200, 1300, 1400]
    }, index=pd.date_range("2025-01-01", periods=5))

def test_willr(ti, sample_data):
    result = ti.WILLR(sample_data, period=3)
    expected = pd.Series([np.nan, np.nan, -25.0, -25.0, -25.0], index=sample_data.index, name="WILLR_3")
    pd.testing.assert_series_equal(result, expected, rtol=1e-4, atol=1e-4)

def test_mfi(ti, sample_data):
    result = ti.MFI(sample_data, period=3)
    # Add expected values based on manual calculation
    assert result.name == "MFI_3"
    assert len(result) == len(sample_data)

def test_invalid_index(ti, sample_data):
    data_invalid = sample_data.copy()
    data_invalid.index = ["a", "b", "c", "d", "e"]
    with pytest.raises(InvalidInputError, match="Index must be datetime or numeric"):
        ti.SMA(data_invalid["close"], period=3)

def test_sma(ti, sample_data):
    result = ti.SMA(sample_data["close"], period=3)
    expected = pd.Series([np.nan, np.nan, 101.0, 102.0, 103.0], index=sample_data.index, name="SMA_3")
    pd.testing.assert_series_equal(result, expected, rtol=1e-4, atol=1e-4)

def test_sma_empty(ti):
    empty_series = pd.Series([], index=pd.DatetimeIndex([]))
    with pytest.raises(InvalidInputError, match="Input data is empty"):
        ti.SMA(empty_series, period=3)

def test_willr_empty(ti):
    empty_df = pd.DataFrame({"high": [], "low": [], "close": []}, index=pd.DatetimeIndex([]))
    with pytest.raises(InvalidInputError, match="Input data is empty"):
        ti.WILLR(empty_df, period=3)

def test_willr_non_numeric(ti, sample_data):
    invalid_data = sample_data.copy()
    invalid_data["close"] = ["a", "b", "c", "d", "e"]
    with pytest.raises(InvalidInputError, match="numeric data"):
        ti.WILLR(invalid_data, period=3)