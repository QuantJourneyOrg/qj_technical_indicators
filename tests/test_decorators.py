import pytest
import logging
from quantjourney_ti._decorators import timer, numba_fallback
from quantjourney_ti._errors import IndicatorCalculationError

def test_timer(caplog):
    caplog.set_level(logging.INFO)
    @timer
    def slow_func():
        import time
        time.sleep(0.1)
        return 42
    assert slow_func() == 42
    assert "Finished slow_func" in caplog.text

def test_numba_fallback(caplog):
    caplog.set_level(logging.WARNING)
    def pandas_fallback(self, data):
        return sum(data)
    @numba_fallback(pandas_fallback)
    def numba_sum(self, data):
        raise ValueError("Numba failed")
    class TestClass:
        def sum(self, data):
            return numba_sum(self, data)
    result = TestClass().sum([1, 2, 3])
    assert result == 6
    assert "Numba failed" in caplog.text
    assert "IndicatorCalculationError" in caplog.text