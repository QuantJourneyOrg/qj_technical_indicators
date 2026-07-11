import pandas as pd

from quantjourney_ti import TechnicalIndicators, clear_indicator_cache


def test_cache_uses_complete_series_content():
    clear_indicator_cache()
    original = pd.Series(range(30), dtype=float)
    changed = original.copy()
    changed.iloc[15] = 9999.0

    indicators = TechnicalIndicators()
    first = indicators.SMA(original, 3)
    second = indicators.SMA(changed, 3)

    assert not first.equals(second)
    assert second.iloc[15] == changed.rolling(3).mean().iloc[15]


def test_cached_result_is_isolated_from_caller_mutation():
    clear_indicator_cache()
    prices = pd.Series(range(30), dtype=float)
    indicators = TechnicalIndicators()

    first = indicators.SMA(prices, 3)
    expected = first.iloc[15]
    first.iloc[15] = -999.0

    second = indicators.SMA(prices, 3)
    assert second.iloc[15] == expected
