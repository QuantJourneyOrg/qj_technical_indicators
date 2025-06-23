"""Integration test fetching real data via yfinance.

Skipped automatically if network/yfinance unavailable.
"""

import pytest
import pandas as pd

try:
    import quantjourney_ti as qti
    from tests._yf import get_ohlcv
except Exception as exc:  # pragma: no cover
    pytest.skip(f"quantjourney_ti import failed: {exc}", allow_module_level=True)

TICKER = "AAPL"


@pytest.fixture(scope="module")
def price_df():
    try:
        df = get_ohlcv(TICKER, period="3mo", interval="1d").rename(columns=str.lower)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"yfinance unavailable or network error: {exc}")
    return df


def test_flat_helper(price_df):
    close = price_df["close"]
    sma = qti.sma(close, 20)
    assert isinstance(sma, pd.Series)
    assert len(sma) == len(close)


def test_class_instance(price_df):
    ti = qti.TechnicalIndicators()
    rsi = ti.RSI(price_df["close"], 14)
    assert isinstance(rsi, pd.Series)
    assert rsi.isna().sum() >= 14  # initial NaNs


def test_singleton_shortcut(price_df):
    import quantjourney_ti.indicators as ind  # noqa

    atr_series = ind._TI_INSTANCE.ATR(price_df, 14)  # uses high/low/close
    assert isinstance(atr_series, pd.Series)
    assert len(atr_series) == len(price_df)
