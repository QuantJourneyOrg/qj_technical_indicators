import pytest
import pandas as pd
import numpy as np
from quantjourney_ti._utils import (
    validate_data, validate_and_get_prices, validate_window,
    detect_divergence, detect_crossovers, optimize_memory
)
from quantjourney_ti._errors import InvalidInputError

@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "close": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103]
        },
        index=pd.date_range("2025-01-01", periods=5)
    )

def test_validate_data(sample_data):
    assert validate_data(sample_data, ["close", "high", "low"]) is True
    with pytest.raises(InvalidInputError, match="Missing required columns"):
        validate_data(sample_data, ["open"])
    with pytest.raises(InvalidInputError, match="Index must be datetime or numeric"):
        invalid_data = sample_data.copy()
        invalid_data.index = ["a", "b", "c", "d", "e"]
        validate_data(invalid_data)

def test_validate_and_get_prices(sample_data):
    prices = validate_and_get_prices(sample_data)
    assert prices.name == "close"
    assert len(prices) == 5
    with pytest.raises(InvalidInputError, match="numeric data"):
        invalid_data = sample_data.copy()
        invalid_data["close"] = ["a", "b", "c", "d", "e"]
        validate_and_get_prices(invalid_data)

def test_validate_window():
    assert validate_window(data_length=10, window=5) is True
    with pytest.raises(InvalidInputError, match="Window size must be at least"):
        validate_window(data_length=10, window=1)
    with pytest.raises(InvalidInputError, match="Window size.*must be less"):
        validate_window(data_length=5, window=5)

def test_optimize_memory():
    df = pd.DataFrame({"int_col": [1, 2, 3], "float_col": [1.0, 2.0, 3.0]})
    optimized = optimize_memory(df)
    assert optimized["int_col"].dtype == np.int32
    assert optimized["float_col"].dtype == np.float32