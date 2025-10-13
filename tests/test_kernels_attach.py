from quantjourney_ti import TechnicalIndicators


def test_kernels_attached_to_class():
    # The class should have static methods attached for kernels
    assert hasattr(TechnicalIndicators, "_calculate_sma_numba")
    assert hasattr(TechnicalIndicators, "_calculate_macd_numba")
    assert hasattr(TechnicalIndicators, "_calculate_atr_numba")

    ti = TechnicalIndicators()
    # Methods should be callable via instance too
    assert callable(ti._calculate_sma_numba)
    assert callable(ti._calculate_macd_numba)
    assert callable(ti._calculate_atr_numba)

